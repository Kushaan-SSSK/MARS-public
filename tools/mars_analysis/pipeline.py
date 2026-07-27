from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import html
import json
from pathlib import Path

import pandas as pd

from .accusleepy_adapter import score_recording_to_mars_bundle
from .config import AnalysisConfig, RecordingSpec
from .edf_stream import envelope_for_signal, read_edf_header
from .edf_scoring import SCORING_SCHEMA_VERSION, score_edf_recording_to_mars_bundle
from .hashing import config_hash, file_fingerprint
from .labels import (
    LabelInventoryRow,
    discover_label_files,
    inventory_label_files,
    load_label_frame,
    write_mars_standard_predictions,
)
from .metrics import choose_baselines, choose_state_baselines, sanitize_name
from .paper_figures import write_paper_psd_figure, write_paper_spectrogram_figure, write_paper_trace_figure
from .plots import write_state_pie, write_trace_envelope
from .signal_analysis import (
    compute_hourly_streaming_psd,
    compute_spectrogram_tiles,
    compute_streaming_psd,
    write_epoch_inspector_plot,
    write_spectrogram_hour_index,
    write_spectrogram_preview,
    write_psd_plot,
)


class AnalysisRunner:
    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.output_dir = config.dataset_output_dir

    @classmethod
    def from_config_file(cls, path: str | Path) -> "AnalysisRunner":
        return cls(AnalysisConfig.from_json(path))

    def run_inventory(self) -> list[LabelInventoryRow]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        files = discover_label_files([*self.config.label_roots, self.output_dir])
        rows = inventory_label_files(files)
        allowed = set(self.config.label_sources)
        rows = [row for row in rows if row.label_source in allowed or row.label_source == "unknown"]
        self._write_inventory(rows)
        self._write_manifest("inventory", rows)
        return rows

    def standardize_existing_accusleepy_labels(
        self,
        rows: list[LabelInventoryRow],
    ) -> list[LabelInventoryRow]:
        generated_paths: list[Path] = []
        for row in rows:
            if row.status != "ok":
                continue
            if row.label_source != "offline_accusleepy" or row.format == "mars_standard":
                continue
            output_path = (
                self.output_dir
                / row.recording_id
                / "offline_accusleepy_standardized"
                / "predictions_standardized.csv"
            )
            if self.config.cache_policy.skip_existing and output_path.exists():
                generated_paths.append(output_path)
                continue
            frame = load_label_frame(
                row.path,
                recording_id=row.recording_id,
                default_epoch_length=row.epoch_length_sec or 2.5,
            )
            write_mars_standard_predictions(frame, output_path)
            self._write_mars_summary(row, output_path, len(frame))
            self._write_empty_run_sidecars(output_path.parent)
            generated_paths.append(output_path)

        if not generated_paths:
            return rows
        generated_rows = inventory_label_files(generated_paths)
        combined = _dedupe_rows(rows + generated_rows)
        self._write_inventory(combined)
        return combined

    def generate_missing_accusleepy_labels(
        self,
        rows: list[LabelInventoryRow],
    ) -> list[LabelInventoryRow]:
        existing = {
            row.recording_id
            for row in rows
            if row.status == "ok" and row.label_source == "offline_accusleepy"
        }
        generated_paths: list[Path] = []
        for spec in self.config.recordings:
            output_dir = (
                self.output_dir
                / spec.recording_id
                / "scored"
            )
            if spec.recording_id in existing and _is_current_scored_bundle(output_dir):
                continue
            predictions_path = output_dir / "predictions_standardized.csv"
            if (
                self.config.cache_policy.skip_existing
                and predictions_path.exists()
                and _is_current_scored_bundle(output_dir)
            ):
                generated_paths.append(predictions_path)
                continue
            if (
                spec.accusleepy_recording_file
                and spec.accusleepy_model_file
                and spec.accusleepy_calibration_file
                and spec.sampling_rate
            ):
                try:
                    generated_paths.append(
                        score_recording_to_mars_bundle(
                            spec,
                            output_dir,
                            model_alias=self.config.default_model_alias,
                        )
                    )
                except Exception as exc:  # noqa: BLE001
                    self._write_scoring_error(
                        spec,
                        output_dir,
                        status="error",
                        message=str(exc),
                    )
            elif spec.accusleepy_recording_file or spec.accusleepy_model_file or spec.accusleepy_calibration_file:
                missing = [
                    name
                    for name, value in [
                        ("accusleepy_recording_file", spec.accusleepy_recording_file),
                        ("accusleepy_model_file", spec.accusleepy_model_file),
                        ("accusleepy_calibration_file", spec.accusleepy_calibration_file),
                        ("sampling_rate", spec.sampling_rate),
                    ]
                    if value in (None, "")
                ]
                self._write_scoring_error(
                    spec,
                    output_dir,
                    status="needs_calibration" if "accusleepy_calibration_file" in missing else "missing_scoring_input",
                    message=f"AccuSleePy/parquet scoring input is incomplete: {', '.join(missing)}",
                )
            elif spec.edf_path:
                result = score_edf_recording_to_mars_bundle(spec, self.config, output_dir)
                if result is not None:
                    generated_paths.append(result)
        if not generated_paths:
            return rows
        combined = _dedupe_rows(rows + inventory_label_files(generated_paths))
        self._write_inventory(combined)
        return combined

    def run_analysis(self) -> dict[str, Path]:
        rows = self.run_inventory()
        rows = self.generate_missing_accusleepy_labels(rows)
        rows = self.standardize_existing_accusleepy_labels(rows)
        state_files = self._write_state_outputs(rows)
        group_files = self._write_group_analysis_outputs()
        sanity_file = self._write_prediction_sanity(rows)
        qc_file = self._write_edf_qc_and_traces()
        classification_status = self._write_classification_status()
        self._write_manifest("analyze", rows)
        return {
            "label_inventory": self.output_dir / "label_inventory.csv",
            "qc_summary": qc_file,
            "classification_status": classification_status,
            "prediction_sanity": sanity_file,
            **state_files,
            **group_files,
        }

    def _write_pdf_compatible_outputs(self, rows: list[LabelInventoryRow]) -> dict[str, Path]:
        if not self.config.pdf_benchmark_ground_truth:
            return {}
        summary, differences, confusion, alignment, aggregate = compare_pdf_compatible_heldout(
            rows,
            self.config.pdf_benchmark_ground_truth,
        )
        write_pdf_compatible_outputs(
            self.output_dir,
            summary,
            differences,
            confusion,
            alignment,
            aggregate,
            expected_summary_path=self.config.pdf_benchmark_expected_summary,
        )
        return {
            "pdf_compat_per_recording_summary": self.output_dir / "pdf_compat_per_recording_summary.csv",
            "pdf_compat_aggregate_summary": self.output_dir / "pdf_compat_aggregate_summary.json",
            "pdf_compat_alignment_summary": self.output_dir / "pdf_compat_alignment_summary.csv",
        }

    def _write_inventory(self, rows: list[LabelInventoryRow]) -> None:
        frame = pd.DataFrame([row.to_dict() for row in rows])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        frame.to_csv(self.output_dir / "label_inventory.csv", index=False)

    def _rows_for_config(self, rows: list[LabelInventoryRow]) -> list[LabelInventoryRow]:
        configured = {spec.recording_id for spec in self.config.recordings}
        if not configured:
            return rows
        return [row for row in rows if row.recording_id in configured]

    def _write_state_outputs(self, rows: list[LabelInventoryRow]) -> dict[str, Path]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        configured = {spec.recording_id for spec in self.config.recordings}
        baselines = {
            recording_id: row
            for recording_id, row in choose_state_baselines(rows).items()
            if not configured or recording_id in configured
        }
        state_rows = []
        hourly_rows = []
        for recording_id, row in baselines.items():
            frame = load_label_frame(row.path, recording_id=recording_id, default_epoch_length=row.epoch_length_sec or 2.5)
            counts = frame["PredLabel"].value_counts(dropna=False).rename_axis("PredLabel").reset_index(name="EpochCount")
            counts = _with_canonical_state_rows(counts)
            total = counts["EpochCount"].sum()
            counts["RecordingID"] = recording_id
            counts["Percent"] = counts["EpochCount"] / total if total else 0
            counts = _sort_state_rows(counts)
            state_rows.extend(counts.to_dict("records"))
            hourly = compute_hourly_bins(frame, self.config.visualizer.hour_bin_size)
            hourly_rows.extend(hourly.to_dict("records"))
            plot_dir = self.output_dir / recording_id / "plots"
            write_state_pie(
                counts,
                plot_dir / "state_percentages_pie.html",
                title=f"{recording_id} state percentages",
            )
        state_frame = pd.DataFrame(state_rows)
        hourly_frame = pd.DataFrame(hourly_rows)
        if state_frame.empty:
            state_frame = pd.DataFrame(columns=["PredLabel", "EpochCount", "RecordingID", "Percent"])
        if hourly_frame.empty:
            hourly_frame = pd.DataFrame(
                columns=["RecordingID", "HourBin", "PredLabel", "EpochCount", "Percent", "BinStartSeconds", "BinSizeHours"]
            )
        state_path = self.output_dir / "state_percentages.csv"
        hourly_path = self.output_dir / "hourly_bins.csv"
        state_frame.to_csv(state_path, index=False)
        hourly_frame.to_csv(hourly_path, index=False)
        validation_path, validation_summary_path = self._write_state_percentage_validation(state_frame)
        if not state_frame.empty:
            dataset_counts = (
                state_frame.groupby("PredLabel", dropna=False)["EpochCount"]
                .sum()
                .rename_axis("PredLabel")
                .reset_index()
            )
            write_state_pie(
                dataset_counts,
                self.output_dir / "plots" / "dataset_state_percentages_pie.html",
                title=f"{self.config.dataset_id} state percentages",
            )
        return {
            "state_percentages": state_path,
            "hourly_bins": hourly_path,
            "state_percentage_validation": validation_path,
            "state_percentage_validation_summary": validation_summary_path,
        }

    def _write_state_percentage_validation(self, state_frame: pd.DataFrame) -> tuple[Path, Path]:
        validation_rows = []
        state_lookup = {}
        if not state_frame.empty:
            for row in state_frame.to_dict("records"):
                key = (str(row.get("RecordingID", "")), str(row.get("PredLabel", "")))
                state_lookup[key] = row

        for spec in self.config.recordings:
            predictions_path = self.output_dir / spec.recording_id / "scored" / "predictions_standardized.csv"
            if not predictions_path.exists():
                for label in CANONICAL_STATE_ORDER:
                    actual = state_lookup.get((spec.recording_id, label), {})
                    validation_rows.append(
                        _state_validation_row(
                            spec.recording_id,
                            label,
                            expected_count=0,
                            actual_count=actual.get("EpochCount"),
                            expected_percent=0.0,
                            actual_percent=actual.get("Percent"),
                            status="missing_prediction_file",
                        )
                    )
                continue

            predictions = pd.read_csv(predictions_path, usecols=["PredLabel"])
            expected_counts = predictions["PredLabel"].astype(str).value_counts().to_dict()
            total = int(sum(expected_counts.values()))
            for label in CANONICAL_STATE_ORDER:
                expected_count = int(expected_counts.get(label, 0))
                expected_percent = expected_count / total if total else 0.0
                actual = state_lookup.get((spec.recording_id, label))
                if actual is None:
                    validation_rows.append(
                        _state_validation_row(
                            spec.recording_id,
                            label,
                            expected_count=expected_count,
                            actual_count=None,
                            expected_percent=expected_percent,
                            actual_percent=None,
                            status="missing_state_percentage_row",
                        )
                    )
                    continue
                actual_count = int(pd.to_numeric(pd.Series([actual.get("EpochCount")]), errors="coerce").fillna(-1).iloc[0])
                actual_percent = float(pd.to_numeric(pd.Series([actual.get("Percent")]), errors="coerce").fillna(float("nan")).iloc[0])
                count_diff = actual_count - expected_count
                percent_diff = actual_percent - expected_percent
                status = "ok" if count_diff == 0 and abs(percent_diff) <= 1e-12 else "mismatch"
                validation_rows.append(
                    _state_validation_row(
                        spec.recording_id,
                        label,
                        expected_count=expected_count,
                        actual_count=actual_count,
                        expected_percent=expected_percent,
                        actual_percent=actual_percent,
                        status=status,
                    )
                )

        validation = pd.DataFrame(validation_rows)
        validation_path = self.output_dir / "state_percentage_validation.csv"
        validation.to_csv(validation_path, index=False)

        summary = pd.DataFrame(
            [
                {
                    "ConfiguredRecordings": len(self.config.recordings),
                    "ValidatedStates": int((validation["Status"] == "ok").sum()) if not validation.empty else 0,
                    "FailedStates": int((validation["Status"] != "ok").sum()) if not validation.empty else 0,
                    "MissingPredictionFiles": int((validation["Status"] == "missing_prediction_file").sum() / len(CANONICAL_STATE_ORDER)) if not validation.empty else 0,
                    "Status": "ok" if not validation.empty and (validation["Status"] == "ok").all() else "failed",
                }
            ]
        )
        summary_path = self.output_dir / "state_percentage_validation_summary.csv"
        summary.to_csv(summary_path, index=False)
        return validation_path, summary_path

    def _write_group_analysis_outputs(self) -> dict[str, Path]:
        output_plot_dir = self.output_dir / "plots"
        output_plot_dir.mkdir(parents=True, exist_ok=True)
        hourly_path = self.output_dir / "hourly_bins.csv"
        state_path = self.output_dir / "state_percentages.csv"
        hourly_output = output_plot_dir / "group_hourly_state_composition.html"
        animal_output = output_plot_dir / "group_animal_state_summary.html"
        if not hourly_path.exists():
            hourly_output.write_text(
                f"<html><body><h1>Group Analysis</h1><p>Missing {hourly_path}</p></body></html>",
                encoding="utf-8",
            )
            animal_output.write_text(
                f"<html><body><h1>Animal State Summary</h1><p>Missing {hourly_path}</p></body></html>",
                encoding="utf-8",
            )
            return {"group_hourly_state_composition": hourly_output, "group_animal_state_summary": animal_output}
        try:
            hourly = pd.read_csv(hourly_path)
        except pd.errors.EmptyDataError:
            hourly = pd.DataFrame(
                columns=["RecordingID", "HourBin", "PredLabel", "EpochCount", "Percent", "BinStartSeconds", "BinSizeHours"]
            )
        meta = pd.DataFrame(
            [
                {
                    "RecordingID": spec.recording_id,
                    "AnimalID": spec.animal_id,
                    "Condition": spec.condition,
                    "Dose": spec.dose,
                    "Session": spec.session,
                }
                for spec in self.config.recordings
            ]
        )
        if hourly.empty:
            hourly_output.write_text("<html><body><h1>Group Analysis</h1><p>No hourly bins found.</p></body></html>", encoding="utf-8")
            animal_output.write_text("<html><body><h1>Animal State Summary</h1><p>No hourly bins found.</p></body></html>", encoding="utf-8")
            return {"group_hourly_state_composition": hourly_output, "group_animal_state_summary": animal_output}
        merged = hourly.merge(meta, on="RecordingID", how="left")
        grouped = (
            merged.groupby(["Condition", "Dose", "Session", "HourBin", "PredLabel"], dropna=False)["EpochCount"]
            .sum()
            .rename("EpochCount")
            .reset_index()
        )
        totals = grouped.groupby(["Condition", "Dose", "Session", "HourBin"])["EpochCount"].transform("sum")
        grouped["Percent"] = grouped["EpochCount"] / totals
        grouped["Group"] = (
            grouped["Condition"].fillna("").astype(str)
            + " / "
            + grouped["Dose"].fillna("").astype(str)
            + " / "
            + grouped["Session"].fillna("").astype(str)
        )
        animal = (
            merged.groupby(["AnimalID", "Condition", "Dose", "PredLabel"], dropna=False)["EpochCount"]
            .sum()
            .rename("EpochCount")
            .reset_index()
        )
        animal_totals = animal.groupby(["AnimalID", "Condition", "Dose"])["EpochCount"].transform("sum")
        animal["Percent"] = animal["EpochCount"] / animal_totals
        hourly_output.write_text(
            _html_page(
                "Group Analysis",
                [
                    "<p class='note'>Loaded lightweight group dashboard from hourly_bins.csv.</p>",
                    "<h2>Condition/Dose/Session by Hour</h2>",
                    _stacked_bar_svg(grouped, group_col="Group", x_col="HourBin"),
                    _html_table(grouped.head(500)),
                ],
            ),
            encoding="utf-8",
        )
        animal["Group"] = animal["AnimalID"].fillna("").astype(str) + " / " + animal["Condition"].fillna("").astype(str) + " / " + animal["Dose"].fillna("").astype(str)
        animal_output.write_text(
            _html_page(
                "Animal State Summary",
                [
                    "<p class='note'>Loaded lightweight animal-level dashboard from state bins.</p>",
                    "<h2>Animal-Level State Percentages</h2>",
                    _stacked_bar_svg(animal, group_col="Group", x_col=None),
                    _html_table(animal.head(500)),
                ],
            ),
            encoding="utf-8",
        )
        if state_path.exists():
            # Keep the dataset-level CSV as the source of truth; the two HTML files are display artifacts.
            pass
        return {"group_hourly_state_composition": hourly_output, "group_animal_state_summary": animal_output}

    def _write_prediction_sanity(self, rows: list[LabelInventoryRow]) -> Path:
        configured = {spec.recording_id for spec in self.config.recordings}
        baselines = {
            recording_id: row
            for recording_id, row in choose_state_baselines(rows).items()
            if not configured or recording_id in configured
        }
        sanity_rows = []
        for recording_id, row in baselines.items():
            try:
                frame = load_label_frame(row.path, recording_id=recording_id, default_epoch_length=row.epoch_length_sec or 2.5)
                labels = frame["PredLabel"].astype(str)
                counts = labels.value_counts(dropna=False)
                total = int(counts.sum())
                top_label = str(counts.index[0]) if total else ""
                top_fraction = float(counts.iloc[0] / total) if total else 0.0
                unique_labels = int(counts.size)
                confidence = pd.to_numeric(frame.get("Confidence", pd.Series(dtype=float)), errors="coerce")
                confidence_std = float(confidence.std(skipna=True)) if not confidence.dropna().empty else None
                confidence_unique = int(confidence.round(6).nunique(dropna=True)) if not confidence.dropna().empty else 0
                duration_seconds = float(total * (row.epoch_length_sec or 0.0))
                reasons = []
                if total == 0:
                    reasons.append("no_predictions")
                if top_fraction > 0.95:
                    reasons.append("dominant_state_gt_95pct")
                if duration_seconds >= 3600 and {"NREM", "REM"}.isdisjoint(set(labels.unique())):
                    reasons.append("missing_nrem_rem_long_recording")
                if confidence_unique <= 1 and total > 10:
                    reasons.append("constant_confidence")
                status = "fail" if reasons else "pass"
                sanity_rows.append(
                    {
                        "RecordingID": recording_id,
                        "Status": status,
                        "Reasons": ";".join(reasons),
                        "LabelSource": row.label_source,
                        "Model": row.model,
                        "EpochCount": total,
                        "UniqueLabels": unique_labels,
                        "TopLabel": top_label,
                        "TopFraction": top_fraction,
                        "ConfidenceStd": confidence_std,
                        "ConfidenceUniqueRounded": confidence_unique,
                        "Predictions": str(row.path),
                    }
                )
            except Exception as exc:  # noqa: BLE001
                sanity_rows.append(
                    {
                        "RecordingID": recording_id,
                        "Status": "error",
                        "Reasons": str(exc),
                        "LabelSource": row.label_source,
                        "Model": row.model,
                        "EpochCount": 0,
                        "UniqueLabels": 0,
                        "TopLabel": "",
                        "TopFraction": 0.0,
                        "ConfidenceStd": None,
                        "ConfidenceUniqueRounded": 0,
                        "Predictions": str(row.path),
                    }
                )
        path = self.output_dir / "prediction_sanity.csv"
        pd.DataFrame(
            sanity_rows,
            columns=[
                "RecordingID", "Status", "Reasons", "LabelSource", "Model", "EpochCount",
                "UniqueLabels", "TopLabel", "TopFraction", "ConfidenceStd",
                "ConfidenceUniqueRounded", "Predictions",
            ],
        ).to_csv(path, index=False)
        return path

    def _write_edf_qc_and_traces(self) -> Path:
        rows = []
        all_psd = []
        for spec in self.config.recordings:
            if not spec.edf_path:
                continue
            path = Path(spec.edf_path)
            if not path.exists():
                rows.append(_qc_row(spec, status="missing_edf", message=str(path)))
                continue
            try:
                header = read_edf_header(path)
                eeg_signal = _first_available_signal(header, spec.channel_map.eeg, self.config.visualizer.eeg_channel, 0)
                emg_signal = _first_available_signal(header, spec.channel_map.emg, self.config.visualizer.emg_channel, min(1, header.signal_count - 1))
                recording_dir = self.output_dir / spec.recording_id
                trace_dir = recording_dir / "trace_envelopes"
                plot_dir = recording_dir / "plots"
                spectrogram_dir = recording_dir / "spectrogram_tiles"
                trace_dir.mkdir(parents=True, exist_ok=True)
                plot_dir.mkdir(parents=True, exist_ok=True)
                for signal in [eeg_signal, emg_signal]:
                    env_manifest = trace_dir / f"{sanitize_name(str(signal))}_envelope.manifest.json"
                    envelope_path = trace_dir / f"{sanitize_name(str(signal))}_envelope.parquet"
                    if not self._skip_edf_artifact(path, env_manifest, [envelope_path]):
                        env = envelope_for_signal(
                            header,
                            signal,
                            bin_seconds=self.config.visualizer.trace_bin_seconds,
                        )
                        env.to_parquet(envelope_path, index=False)
                        write_trace_envelope(
                            env,
                            plot_dir / f"{sanitize_name(str(signal))}_trace_envelope.html",
                            title=f"{spec.recording_id} {signal} envelope",
                        )
                        self._write_artifact_manifest(path, env_manifest)
                    elif envelope_path.exists():
                        write_trace_envelope(
                            pd.read_parquet(envelope_path),
                            plot_dir / f"{sanitize_name(str(signal))}_trace_envelope.html",
                            title=f"{spec.recording_id} {signal} envelope",
                        )
                    psd_path = recording_dir / f"{sanitize_name(str(signal))}_power_spectra.parquet"
                    psd_manifest = recording_dir / f"{sanitize_name(str(signal))}_power_spectra.manifest.json"
                    if not self._skip_edf_artifact(path, psd_manifest, [psd_path]):
                        psd = compute_streaming_psd(
                            header,
                            signal,
                            chunk_seconds=self.config.visualizer.psd_chunk_seconds,
                            max_hz=max(60.0, self.config.visualizer.spectrogram_max_hz),
                        )
                        psd["RecordingID"] = spec.recording_id
                        psd.to_parquet(psd_path, index=False)
                        self._write_artifact_manifest(path, psd_manifest)
                    if psd_path.exists():
                        all_psd.append(pd.read_parquet(psd_path))
                    hourly_psd_path = recording_dir / f"{sanitize_name(str(signal))}_hourly_power_spectra.parquet"
                    hourly_psd_manifest = recording_dir / f"{sanitize_name(str(signal))}_hourly_power_spectra.manifest.json"
                    if not self._skip_edf_artifact(path, hourly_psd_manifest, [hourly_psd_path]):
                        hourly_psd = compute_hourly_streaming_psd(
                            header,
                            signal,
                            hour_bin_size=self.config.visualizer.hour_bin_size,
                            chunk_seconds=self.config.visualizer.psd_chunk_seconds,
                            max_hz=max(60.0, self.config.visualizer.spectrogram_max_hz),
                        )
                        hourly_psd["RecordingID"] = spec.recording_id
                        hourly_psd.to_parquet(hourly_psd_path, index=False)
                        write_psd_plot(hourly_psd, plot_dir / f"{sanitize_name(str(signal))}_hourly_power_spectra.html")
                        self._write_artifact_manifest(path, hourly_psd_manifest)
                    elif hourly_psd_path.exists():
                        write_psd_plot(
                            pd.read_parquet(hourly_psd_path),
                            plot_dir / f"{sanitize_name(str(signal))}_hourly_power_spectra.html",
                        )
                    if signal == eeg_signal:
                        spec_manifest = spectrogram_dir / f"{sanitize_name(str(signal))}_spectrogram.manifest.json"
                        expected_index = spectrogram_dir / f"{header.labels[header.signal_index(signal)]}_spectrogram_index.csv"
                        if not self._skip_edf_artifact(path, spec_manifest, [expected_index]):
                            spectrogram_index = compute_spectrogram_tiles(
                                header,
                                signal,
                                output_dir=spectrogram_dir,
                                chunk_seconds=self.config.visualizer.spectrogram_chunk_seconds,
                                max_hz=self.config.visualizer.spectrogram_max_hz,
                            )
                            write_spectrogram_hour_index(
                                spectrogram_index,
                                spectrogram_dir / f"{header.labels[header.signal_index(signal)]}_hourly_index.html",
                                hour_bin_size=self.config.visualizer.hour_bin_size,
                            )
                            self._write_artifact_manifest(path, spec_manifest)
                        elif expected_index.exists():
                            spectrogram_index = pd.read_csv(expected_index)
                            write_spectrogram_preview(
                                spectrogram_index,
                                spectrogram_dir / f"{header.labels[header.signal_index(signal)]}_spectrogram.html",
                            )
                            write_spectrogram_hour_index(
                                spectrogram_index,
                                spectrogram_dir / f"{header.labels[header.signal_index(signal)]}_hourly_index.html",
                                hour_bin_size=self.config.visualizer.hour_bin_size,
                            )
                epoch_path = recording_dir / "epoch_inspector.html"
                epoch_manifest = recording_dir / "epoch_inspector.manifest.json"
                if not self._skip_edf_artifact(path, epoch_manifest, [epoch_path]):
                    write_epoch_inspector_plot(
                        header,
                        eeg_signal,
                        emg_signal,
                        epoch_path,
                        epoch_index=1,
                        epoch_length_sec=spec.epoch_length_sec,
                    )
                    self._write_artifact_manifest(path, epoch_manifest)
                self._write_paper_figures(
                    spec,
                    header,
                    eeg_signal,
                    emg_signal,
                    recording_dir,
                    epoch_length_sec=spec.epoch_length_sec,
                )
                rows.append(
                    _qc_row(
                        spec,
                        status="ok",
                        duration_seconds=header.duration_seconds,
                        signal_count=header.signal_count,
                        signals=";".join(header.labels),
                    )
                )
            except Exception as exc:  # noqa: BLE001
                rows.append(_qc_row(spec, status="error", message=str(exc)))
        qc = pd.DataFrame(rows)
        qc_path = self.output_dir / "qc_summary.csv"
        qc.to_csv(qc_path, index=False)
        if all_psd:
            psd_all = pd.concat(all_psd, ignore_index=True)
            psd_all.to_parquet(self.output_dir / "power_spectra.parquet", index=False)
            write_psd_plot(psd_all, self.output_dir / "power_spectra.html")
        return qc_path

    def _write_paper_figures(
        self,
        spec: RecordingSpec,
        header,
        eeg_signal,
        emg_signal,
        recording_dir: Path,
        *,
        epoch_length_sec: float,
    ) -> None:
        figure_dir = recording_dir / "paper_figures"
        figure_dir.mkdir(parents=True, exist_ok=True)
        predictions_path = recording_dir / "scored" / "predictions_standardized.csv"
        eeg_env_path = recording_dir / "trace_envelopes" / f"{sanitize_name(str(eeg_signal))}_envelope.parquet"
        emg_env_path = recording_dir / "trace_envelopes" / f"{sanitize_name(str(emg_signal))}_envelope.parquet"
        if predictions_path.exists() and eeg_env_path.exists() and emg_env_path.exists():
            predictions = pd.read_csv(predictions_path)
            epoch_lengths = pd.to_numeric(predictions.get("EpochLengthSeconds", pd.Series(dtype=float)), errors="coerce").dropna()
            effective_epoch = float(epoch_lengths.mode().iloc[0]) if not epoch_lengths.empty else epoch_length_sec
            write_paper_trace_figure(
                recording_id=spec.recording_id,
                header=header,
                eeg_signal=eeg_signal,
                emg_signal=emg_signal,
                eeg_envelope=pd.read_parquet(eeg_env_path),
                emg_envelope=pd.read_parquet(emg_env_path),
                predictions=predictions,
                output_path=figure_dir / "paper_trace_overview.html",
                epoch_length_sec=effective_epoch,
                zoom_epochs=self.config.visualizer.zoom_epochs or None,
            )
        spectrogram_index_path = recording_dir / "spectrogram_tiles" / f"{header.labels[header.signal_index(eeg_signal)]}_spectrogram_index.csv"
        psd_frames = []
        for signal in [eeg_signal, emg_signal]:
            psd_path = recording_dir / f"{sanitize_name(str(signal))}_power_spectra.parquet"
            if psd_path.exists():
                psd_frames.append(pd.read_parquet(psd_path))
        if spectrogram_index_path.exists() or psd_frames:
            spectrogram_index = pd.read_csv(spectrogram_index_path) if spectrogram_index_path.exists() else pd.DataFrame()
            write_paper_spectrogram_figure(
                recording_id=spec.recording_id,
                spectrogram_index=spectrogram_index,
                output_path=figure_dir / "paper_spectrogram.html",
            )
        if psd_frames:
            predictions = pd.read_csv(predictions_path) if predictions_path.exists() else None
            epoch_lengths = (
                pd.to_numeric(predictions.get("EpochLengthSeconds", pd.Series(dtype=float)), errors="coerce").dropna()
                if predictions is not None
                else pd.Series(dtype=float)
            )
            effective_epoch = float(epoch_lengths.mode().iloc[0]) if not epoch_lengths.empty else epoch_length_sec
            write_paper_psd_figure(
                recording_id=spec.recording_id,
                psd_frames=psd_frames,
                output_path=figure_dir / "paper_state_power_spectra.html",
                header=header,
                eeg_signal=eeg_signal,
                emg_signal=emg_signal,
                predictions=predictions,
                epoch_length_sec=effective_epoch,
                max_faint_curves_per_state=max(0, int(self.config.visualizer.psd_max_faint_curves_per_state)),
                show_sem_band=bool(self.config.visualizer.psd_show_sem_band),
            )

    def _write_classification_status(self) -> Path:
        rows = []
        scored_dirs = sorted({path.parent for path in self.output_dir.glob("*/scored/*")})
        for scored_dir in scored_dirs:
            manifest_path = scored_dir / "classification_manifest.json"
            summary_path = scored_dir / "summary.json"
            predictions_path = scored_dir / "predictions_standardized.csv"
            recording_id = scored_dir.parents[0].name
            if summary_path.exists() and predictions_path.exists():
                try:
                    summary = json.loads(summary_path.read_text(encoding="utf-8"))
                    rows.append(
                        {
                            "RecordingID": recording_id,
                            "Status": "ok",
                            "Message": "",
                            "ModelAlias": summary.get("ModelAlias", ""),
                            "ModelFile": summary.get("ModelFile", ""),
                            "CalibrationFile": summary.get("CalibrationFile", ""),
                            "UnitScaleApplied": json.dumps(summary.get("UnitScaleApplied", {}), sort_keys=True),
                            "ParityStatus": (summary.get("Staging") or {}).get("parity", {}).get("status", ""),
                            "Predictions": str(predictions_path),
                        }
                    )
                    continue
                except (OSError, json.JSONDecodeError) as exc:
                    rows.append(
                        {
                            "RecordingID": recording_id,
                            "Status": "summary_error",
                            "Message": str(exc),
                            "ModelAlias": "",
                            "Predictions": str(predictions_path),
                        }
                    )
                    continue
            if not manifest_path.exists():
                continue
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                rows.append(
                    {
                        "RecordingID": recording_id,
                        "Status": "manifest_error",
                        "Message": str(exc),
                        "ModelAlias": "",
                        "Predictions": "",
                    }
                )
                continue
            rows.append(
                {
                    "RecordingID": recording_id,
                    "Status": manifest.get("status", ""),
                    "Message": manifest.get("message", ""),
                    "ModelAlias": manifest.get("model_alias", ""),
                    "ModelFile": manifest.get("model_file", ""),
                    "CalibrationFile": manifest.get("calibration_file", ""),
                    "UnitScaleApplied": json.dumps(manifest.get("unit_scale_applied", {}), sort_keys=True),
                    "ParityStatus": (manifest.get("staging") or {}).get("parity", {}).get("status", ""),
                    "Predictions": manifest.get("predictions_standardized", ""),
                }
            )
        path = self.output_dir / "classification_status.csv"
        pd.DataFrame(rows).to_csv(path, index=False)
        for spec in self.config.recordings:
            self._write_classification_dashboard(spec)
        return path

    def _write_classification_dashboard(self, spec: RecordingSpec) -> Path:
        recording_dir = self.output_dir / spec.recording_id
        scored_dir = recording_dir / "scored"
        output_path = recording_dir / "classification_dashboard.html"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path = scored_dir / "summary.json"
        manifest_path = scored_dir / "classification_manifest.json"
        state_path = self.output_dir / "state_percentages.csv"
        sanity_path = self.output_dir / "prediction_sanity.csv"
        pieces: list[str] = []
        cards: list[tuple[str, str]] = []
        if summary_path.exists():
            try:
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                cards.extend(
                    [
                        ("Model", str(summary.get("ModelAlias", ""))),
                        ("Epochs", str(summary.get("AggregateScoredEpochs", ""))),
                        ("Epoch Length", str(summary.get("EpochLengthSeconds", ""))),
                        ("Calibration", Path(str(summary.get("CalibrationFile", ""))).name),
                    ]
                )
                detail = pd.DataFrame(
                    [
                        {
                            "Model": summary.get("ModelAlias", ""),
                            "CalibrationFile": summary.get("CalibrationFile", ""),
                            "InputUnit": summary.get("InputUnit", ""),
                            "UnitScaleApplied": json.dumps(summary.get("UnitScaleApplied", {}), sort_keys=True),
                            "OutputDir": summary.get("OutputDir", ""),
                        }
                    ]
                )
                pieces.append("<h2>Run Details</h2>" + _html_table(detail))
            except Exception as exc:  # noqa: BLE001
                pieces.append(f"<p class='note'>Could not read summary.json: {html.escape(str(exc))}</p>")
        elif manifest_path.exists():
            pieces.append("<h2>Classification Manifest</h2><pre>" + html.escape(manifest_path.read_text(encoding="utf-8", errors="replace")) + "</pre>")
        else:
            pieces.append(f"<p class='note'>Run analysis to create {html.escape(str(summary_path))}.</p>")
        if cards:
            pieces.insert(0, _html_cards(cards))
        if state_path.exists():
            try:
                state = pd.read_csv(state_path)
                state = state[state["RecordingID"].astype(str) == spec.recording_id]
                if not state.empty:
                    state = _sort_state_rows(_with_canonical_state_rows(state[["PredLabel", "EpochCount", "Percent"]]))
                    pieces.append("<h2>Prediction Distribution</h2>" + _state_bar_svg(state) + _html_table(state))
            except Exception as exc:  # noqa: BLE001
                pieces.append(f"<p class='note'>Could not read state_percentages.csv: {html.escape(str(exc))}</p>")
        if sanity_path.exists():
            try:
                sanity = pd.read_csv(sanity_path)
                match = sanity[sanity["RecordingID"].astype(str) == spec.recording_id]
                if not match.empty:
                    status = str(match.iloc[0].get("Status", ""))
                    pieces.append(f"<h2>Prediction Sanity {'Passed' if status.lower() == 'pass' else 'Failed'}</h2>" + _html_table(match))
            except Exception as exc:  # noqa: BLE001
                pieces.append(f"<p class='note'>Could not read prediction_sanity.csv: {html.escape(str(exc))}</p>")
        output_path.write_text(_html_page(f"{spec.recording_id} Classification", pieces), encoding="utf-8")
        return output_path

    def _write_scoring_error(
        self,
        spec: RecordingSpec,
        output_dir: Path,
        *,
        status: str,
        message: str,
    ) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "classification_manifest.json").write_text(
            json.dumps(
                {
                    "schema_version": 2,
                    "created_utc": datetime.now(timezone.utc).isoformat(),
                    "status": status,
                    "message": message,
                    "model_alias": self.config.default_model_alias,
                    "model_file": spec.accusleepy_model_file,
                    "calibration_file": spec.accusleepy_calibration_file,
                    "predictions_standardized": None,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    def _write_comparison_plots(
        self,
        summary: pd.DataFrame,
        differences: pd.DataFrame,
        matrices: dict[str, pd.DataFrame],
        coverage: pd.DataFrame | None = None,
    ) -> None:
        plot_dir = self.output_dir / "plots"
        plot_dir.mkdir(parents=True, exist_ok=True)
        for pattern in [
            "confusion_*.html",
            "label_disagreement_timeline.html",
            "label_comparison_summary.html",
        ]:
            for old_plot in plot_dir.glob(pattern):
                old_plot.unlink()
        write_disagreement_timeline(
            differences,
            plot_dir / "label_disagreement_timeline.html",
            title="All label disagreements vs offline AccuSleePy",
        )
        if summary.empty and differences.empty:
            (plot_dir / "label_disagreement_timeline.html").write_text(
                "<html><body style='font-family: Inter, Segoe UI, Arial; padding: 28px;'>"
                "<h1>Label Disagreements</h1>"
                "<p>No comparison label sources were available for this run.</p>"
                "</body></html>",
                encoding="utf-8",
            )
        write_label_comparison_dashboard(
            summary,
            differences,
            plot_dir / "label_comparison_summary.html",
            coverage=coverage,
            title="Label comparison vs offline AccuSleePy",
        )
        for key, matrix in list(matrices.items())[:25]:
            write_confusion_heatmap(
                matrix,
                plot_dir / f"confusion_{sanitize_name(key)}.html",
                title=key,
            )

    def _write_mars_reference_plots(
        self,
        summary: pd.DataFrame,
        differences: pd.DataFrame,
        confusion: pd.DataFrame,
        alignment: pd.DataFrame,
    ) -> None:
        plot_dir = self.output_dir / "plots"
        plot_dir.mkdir(parents=True, exist_ok=True)
        for pattern in [
            "mars_vs_reference_dashboard.html",
            "label_comparison_summary.html",
            "label_disagreement_timeline.html",
        ]:
            for old_plot in plot_dir.glob(pattern):
                old_plot.unlink()
        write_mars_vs_reference_dashboard(
            summary,
            differences,
            confusion,
            alignment,
            plot_dir / "mars_vs_reference_dashboard.html",
        )

    def _write_manifest(self, stage: str, rows: list[LabelInventoryRow]) -> None:
        fingerprints = [row.file_sha256 for row in rows if row.file_sha256]
        values = [
            stage,
            self.config.dataset_id,
            json.dumps(self.config.to_dict(), sort_keys=True),
            *fingerprints,
        ]
        manifest = {
            "schema_version": 1,
            "stage": stage,
            "dataset_id": self.config.dataset_id,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "config_hash": config_hash(values),
            "config": self.config.to_dict(),
            "label_file_count": len(rows),
            "label_files": [row.to_dict() for row in rows],
        }
        (self.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    def _write_mars_summary(
        self,
        source_row: LabelInventoryRow,
        predictions_path: Path,
        epoch_count: int,
    ) -> None:
        out_dir = predictions_path.parent
        summary = {
            "RunName": f"{source_row.recording_id}_offline_accusleepy_standardized",
            "CreatedUTC": datetime.now(timezone.utc).isoformat(),
            "ScoringApplication": "MARS Analysis Offline AccuSleePy Adapter",
            "BridgeMode": "offline-accusleepy-normalized",
            "ModelProfile": source_row.model,
            "ModelAlias": source_row.model,
            "EpochLengthSeconds": source_row.epoch_length_sec,
            "OutputDir": str(out_dir),
            "AggregateScoredEpochs": epoch_count,
            "Outputs": {
                "Predictions": str(predictions_path),
                "Timing": str(out_dir / "epoch_timing.csv"),
                "StimDecisions": str(out_dir / "stim_decisions.csv"),
                "Summary": str(out_dir / "summary.json"),
            },
            "SourceLabelFile": source_row.path,
            "SourceFingerprint": file_fingerprint(source_row.path, content_hash=False).stable_value(),
        }
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    def _write_empty_run_sidecars(self, out_dir: Path) -> None:
        pd.DataFrame(columns=["EpochIndex", "EpochStartSeconds", "InferenceMs", "MissedDeadline"]).to_csv(
            out_dir / "epoch_timing.csv",
            index=False,
        )
        pd.DataFrame(columns=["EpochIndex", "Decision", "Reason"]).to_csv(
            out_dir / "stim_decisions.csv",
            index=False,
        )

    def _artifact_fingerprint(self, edf_path: Path) -> dict[str, object]:
        return {
            "schema_version": 2,
            "edf": file_fingerprint(edf_path, content_hash=False).stable_value(),
            "signal_unit": "uV",
            "visualizer": asdict(self.config.visualizer),
        }

    def _skip_edf_artifact(
        self,
        edf_path: Path,
        manifest_path: Path,
        expected_outputs: list[Path],
    ) -> bool:
        if not self.config.cache_policy.skip_existing:
            return False
        if not manifest_path.exists() or not all(path.exists() for path in expected_outputs):
            return False
        try:
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        return existing.get("fingerprint") == self._artifact_fingerprint(edf_path)

    def _write_artifact_manifest(self, edf_path: Path, manifest_path: Path) -> None:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            json.dumps(
                {
                    "schema_version": 2,
                    "created_utc": datetime.now(timezone.utc).isoformat(),
                    "fingerprint": self._artifact_fingerprint(edf_path),
                },
                indent=2,
            ),
            encoding="utf-8",
        )


CANONICAL_STATE_ORDER = ["NREM", "REM", "Wake"]
STATE_COLORS = {"Wake": "#d62728", "NREM": "#1f4fbf", "REM": "#2ca02c", "DISCO-T": "#6a3d9a", "Unknown": "#7f7f7f"}


def _with_canonical_state_rows(frame: pd.DataFrame) -> pd.DataFrame:
    rows = frame.copy()
    if "PredLabel" not in rows.columns:
        rows["PredLabel"] = []
    if "EpochCount" not in rows.columns:
        rows["EpochCount"] = []
    existing = set(rows["PredLabel"].astype(str))
    missing = [{"PredLabel": label, "EpochCount": 0} for label in CANONICAL_STATE_ORDER if label not in existing]
    if missing:
        rows = pd.concat([rows, pd.DataFrame(missing)], ignore_index=True)
    if "Percent" not in rows.columns:
        rows["Percent"] = 0.0
    rows["EpochCount"] = pd.to_numeric(rows["EpochCount"], errors="coerce").fillna(0).astype(int)
    rows["Percent"] = pd.to_numeric(rows["Percent"], errors="coerce").fillna(0.0)
    return rows


def _sort_state_rows(frame: pd.DataFrame) -> pd.DataFrame:
    rows = frame.copy()
    order = {label: index for index, label in enumerate(CANONICAL_STATE_ORDER)}
    rows["_StateOrder"] = rows["PredLabel"].astype(str).map(lambda label: order.get(label, len(order) + 1))
    return rows.sort_values(["_StateOrder", "PredLabel"]).drop(columns=["_StateOrder"]).reset_index(drop=True)


def _state_validation_row(
    recording_id: str,
    pred_label: str,
    *,
    expected_count: int,
    actual_count: object,
    expected_percent: float,
    actual_percent: object,
    status: str,
) -> dict[str, object]:
    actual_count_value = None if actual_count is None or pd.isna(actual_count) else int(actual_count)
    actual_percent_value = None if actual_percent is None or pd.isna(actual_percent) else float(actual_percent)
    return {
        "RecordingID": recording_id,
        "PredLabel": pred_label,
        "ExpectedEpochCount": expected_count,
        "ActualEpochCount": actual_count_value,
        "ExpectedPercent": expected_percent,
        "ActualPercent": actual_percent_value,
        "CountDifference": None if actual_count_value is None else actual_count_value - expected_count,
        "PercentDifference": None if actual_percent_value is None else actual_percent_value - expected_percent,
        "Status": status,
    }


def _html_page(title: str, pieces: list[str]) -> str:
    return (
        "<html><head><style>"
        "body{margin:0;background:#f7f8fb;color:#17213a;font-family:Inter,Segoe UI,Arial,sans-serif;}"
        "main{padding:26px 30px 42px;}"
        "h1{font-size:28px;line-height:1.2;margin:0 0 20px;font-weight:750;}"
        "h2{font-size:18px;line-height:1.25;margin:26px 0 10px;font-weight:700;color:#24314f;}"
        ".note{color:#5b667a;font-size:14px;}"
        ".cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px;margin:0 0 20px;}"
        ".card{background:white;border:1px solid #dfe5ee;border-radius:8px;padding:13px 15px;box-shadow:0 1px 2px rgba(20,30,50,.04);}"
        ".card .label{font-size:12px;text-transform:uppercase;letter-spacing:.04em;color:#637083;margin-bottom:6px;}"
        ".card .value{font-size:18px;font-weight:700;color:#17213a;overflow-wrap:anywhere;}"
        "table{border-collapse:collapse;width:100%;background:white;border:1px solid #dfe5ee;border-radius:8px;overflow:hidden;font-size:13px;}"
        "th{background:#eef3f8;color:#17213a;text-align:left;font-weight:700;padding:9px 10px;border-bottom:1px solid #dfe5ee;position:sticky;top:0;}"
        "td{padding:8px 10px;border-bottom:1px solid #edf1f5;vertical-align:top;}"
        "tr:nth-child(even) td{background:#fafbfd;}"
        "svg{max-width:100%;height:auto;background:white;border:1px solid #dfe5ee;border-radius:8px;}"
        "</style></head><body><main>"
        f"<h1>{html.escape(title)}</h1>"
        + "\n".join(pieces)
        + "</main></body></html>"
    )


def _html_table(frame: pd.DataFrame) -> str:
    out = frame.copy()
    for col in out.columns:
        if pd.api.types.is_float_dtype(out[col]):
            out[col] = out[col].map(lambda value: "" if pd.isna(value) else f"{value:.4g}")
    return out.to_html(index=False, escape=True)


def _html_cards(items: list[tuple[str, str]]) -> str:
    return "<div class='cards'>" + "".join(
        "<div class='card'>"
        f"<div class='label'>{html.escape(label)}</div>"
        f"<div class='value'>{html.escape(value)}</div>"
        "</div>"
        for label, value in items
    ) + "</div>"


def _state_bar_svg(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "<p class='note'>No state rows available.</p>"
    rows = _sort_state_rows(_with_canonical_state_rows(frame))
    rows["Percent"] = pd.to_numeric(rows["Percent"], errors="coerce").fillna(0)
    x = 20.0
    small_labels = 0
    parts = ['<svg viewBox="0 0 720 120" role="img" aria-label="Prediction distribution">']
    for _, row in rows.iterrows():
        percent = max(0.0, min(1.0, float(row["Percent"])))
        width = percent * 560.0
        label = str(row["PredLabel"])
        color = STATE_COLORS.get(label, "#7f7f7f")
        if width > 0:
            parts.append(f'<rect x="{x:.2f}" y="36" width="{width:.2f}" height="28" fill="{color}"/>')
        if width > 42:
            parts.append(f'<text x="{x + 6:.2f}" y="55" font-size="12" fill="white">{html.escape(label)}</text>')
        elif width > 0:
            center = x + width / 2.0
            label_y = 18 + small_labels * 14
            parts.append(f'<line x1="{center:.2f}" y1="34" x2="{center:.2f}" y2="{label_y + 3}" stroke="{color}" stroke-width="1"/>')
            parts.append(
                f'<text x="{center:.2f}" y="{label_y}" font-size="11" fill="{color}" '
                f'text-anchor="middle" font-weight="700">{html.escape(label)}</text>'
            )
            small_labels += 1
        x += width
    parts.append('<text x="20" y="92" font-size="12" fill="#59657a">State percentage by scored epochs</text></svg>')
    return "".join(parts)


def _stacked_bar_svg(frame: pd.DataFrame, *, group_col: str, x_col: str | None) -> str:
    if frame.empty:
        return "<p class='note'>No rows available.</p>"
    rows = frame.copy()
    rows["Percent"] = pd.to_numeric(rows["Percent"], errors="coerce").fillna(0)
    key_cols = [group_col] + ([x_col] if x_col else [])
    rows["_Key"] = rows[key_cols].astype(str).agg(" / ".join, axis=1)
    keys = rows["_Key"].drop_duplicates().head(40).tolist()
    height = 42 + 28 * len(keys)
    parts = [f'<svg viewBox="0 0 980 {height}" role="img" aria-label="State composition">']
    parts.append('<text x="18" y="24" font-size="14" font-weight="700" fill="#17213a">State composition</text>')
    y = 42
    for key in keys:
        subset = rows[rows["_Key"] == key]
        parts.append(f'<text x="18" y="{y + 14}" font-size="11" fill="#17213a">{html.escape(key[:48])}</text>')
        x = 280.0
        for _, row in subset.iterrows():
            width = max(0.0, min(1.0, float(row["Percent"]))) * 650.0
            color = STATE_COLORS.get(str(row["PredLabel"]), "#7f7f7f")
            parts.append(f'<rect x="{x:.2f}" y="{y}" width="{width:.2f}" height="18" fill="{color}"/>')
            x += width
        y += 28
    parts.append("</svg>")
    return "".join(parts)


def compute_hourly_bins(frame: pd.DataFrame, hour_bin_size: float) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    out = frame.copy()
    bin_seconds = max(float(hour_bin_size) * 3600.0, 1.0)
    out["HourBin"] = (out["EpochStartSeconds"] // bin_seconds).astype(int)
    grouped = (
        out.groupby(["RecordingID", "HourBin", "PredLabel"], dropna=False)
        .size()
        .rename("EpochCount")
        .reset_index()
    )
    totals = grouped.groupby(["RecordingID", "HourBin"])["EpochCount"].transform("sum")
    grouped["Percent"] = grouped["EpochCount"] / totals
    grouped["BinStartSeconds"] = grouped["HourBin"] * bin_seconds
    grouped["BinSizeHours"] = hour_bin_size
    return grouped


def _dedupe_rows(rows: list[LabelInventoryRow]) -> list[LabelInventoryRow]:
    deduped: dict[tuple[str, str, str, str], LabelInventoryRow] = {}
    for row in rows:
        key = (row.recording_id, row.label_source, row.model, row.path)
        deduped[key] = row
    return list(deduped.values())


def _is_current_scored_bundle(output_dir: Path) -> bool:
    manifest_path = output_dir / "classification_manifest.json"
    summary_path = output_dir / "summary.json"
    predictions_path = output_dir / "predictions_standardized.csv"
    if not manifest_path.exists() or not summary_path.exists() or not predictions_path.exists():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        manifest.get("schema_version") == SCORING_SCHEMA_VERSION
        and manifest.get("status") == "ok"
        and summary.get("InputUnit") == "uV"
        and bool(summary.get("UnitScaleApplied"))
    )


def _first_available_signal(header, *candidates):
    for candidate in candidates:
        if candidate is None:
            continue
        try:
            header.signal_index(candidate)
            return candidate
        except (KeyError, IndexError):
            continue
    return 0


def _qc_row(spec: RecordingSpec, **extra) -> dict[str, object]:
    row = {
        "RecordingID": spec.recording_id,
        "EDFFile": spec.edf_path,
        "AnimalID": spec.animal_id,
        "Condition": spec.condition,
        "Dose": spec.dose,
        "Session": spec.session,
    }
    row.update(extra)
    return row


