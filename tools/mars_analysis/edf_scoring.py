from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import importlib.util

import numpy as np
import pandas as pd

from .accusleepy_adapter import score_recording_to_mars_bundle
from .config import AnalysisConfig, RecordingSpec
from .edf_stream import (
    EDFHeader,
    digital_to_microvolts,
    iter_signal_digital_chunks,
    physical_dimension,
    read_edf_header,
    unit_scale_to_microvolts,
)
from .hashing import config_hash, file_fingerprint
from .model_registry import ResolvedModel, resolve_model


SCORING_SCHEMA_VERSION = 2


def score_edf_recording_to_mars_bundle(
    recording: RecordingSpec,
    config: AnalysisConfig,
    output_dir: str | Path,
) -> Path | None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model = resolve_model(config.default_model_alias, recording=recording)
    manifest_path = output_dir / "classification_manifest.json"
    predictions_path = output_dir / "predictions_standardized.csv"
    sidecars = [output_dir / "epoch_timing.csv", output_dir / "stim_decisions.csv", output_dir / "summary.json"]
    fingerprint = _classification_fingerprint(recording, config, model)

    if (
        config.cache_policy.skip_existing
        and manifest_path.exists()
        and predictions_path.exists()
        and all(path.exists() for path in sidecars)
    ):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if (
                manifest.get("schema_version") == SCORING_SCHEMA_VERSION
                and manifest.get("fingerprint") == fingerprint
                and manifest.get("status") == "ok"
            ):
                return predictions_path
        except (OSError, json.JSONDecodeError):
            pass

    if not recording.edf_path or not Path(recording.edf_path).exists():
        _write_status(manifest_path, "missing_edf", fingerprint, model, message=str(recording.edf_path))
        return None
    if model.calibration_file is None or not model.calibration_file.exists():
        _write_status(manifest_path, "needs_calibration", fingerprint, model, message=f"No calibration for {model.alias}")
        return None
    if not model.model_file.exists():
        _write_status(manifest_path, "missing_model", fingerprint, model, message=str(model.model_file))
        return None
    if importlib.util.find_spec("accusleepy") is None:
        _write_status(manifest_path, "dependency_missing", fingerprint, model, message="AccuSleePy is not importable in this Python environment.")
        return None

    try:
        header = read_edf_header(recording.edf_path)
        eeg_signal = _first_available_signal(header, recording.channel_map.eeg, config.visualizer.eeg_channel, 0)
        emg_signal = _first_available_signal(header, recording.channel_map.emg, config.visualizer.emg_channel, min(1, header.signal_count - 1))
        staging_manifest = prepare_edf_for_accusleepy(
            header,
            eeg_signal,
            emg_signal,
            output_dir / "staging",
            recording_id=recording.recording_id,
            fingerprint=fingerprint,
            skip_existing=config.cache_policy.skip_existing,
        )
        staging = json.loads(staging_manifest.read_text(encoding="utf-8"))
        score_spec = replace(
            recording,
            accusleepy_recording_file=str(staging_manifest),
            accusleepy_model_file=str(model.model_file),
            accusleepy_calibration_file=str(model.calibration_file),
            sampling_rate=float(staging["sampling_rate_hz"]),
            feature_window_sec=model.feature_window_sec,
            feature_mode=model.feature_mode,
            epoch_length_sec=model.epoch_length_sec,
        )
        result = score_recording_to_mars_bundle(score_spec, output_dir, model_alias=model.alias)
        _augment_summary_with_staging(output_dir / "summary.json", staging)
        _write_status(
            manifest_path,
            "ok",
            fingerprint,
            model,
            predictions_path=result,
            staging=staging,
        )
        return result
    except Exception as exc:  # noqa: BLE001
        _write_status(manifest_path, "error", fingerprint, model, message=str(exc))
        return None


def prepare_edf_for_accusleepy(
    header: EDFHeader,
    eeg_signal: int | str,
    emg_signal: int | str,
    output_dir: str | Path,
    *,
    recording_id: str,
    fingerprint: dict[str, object],
    skip_existing: bool = True,
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    eeg_index = header.signal_index(eeg_signal)
    emg_index = header.signal_index(emg_signal)
    eeg_rate = header.sample_rate(eeg_index)
    emg_rate = header.sample_rate(emg_index)
    if abs(eeg_rate - emg_rate) > 1e-6:
        raise ValueError(f"EEG and EMG sample rates differ: {eeg_rate} vs {emg_rate}")
    manifest_path = output_dir / "accusleepy_recording_manifest.json"
    eeg_path = output_dir / "eeg.npy"
    emg_path = output_dir / "emg.npy"
    if skip_existing and manifest_path.exists() and eeg_path.exists() and emg_path.exists():
        try:
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
            if existing.get("schema_version") == SCORING_SCHEMA_VERSION and existing.get("fingerprint") == fingerprint:
                return manifest_path
        except (OSError, json.JSONDecodeError):
            pass

    eeg_stats = _write_signal_npy(header, eeg_index, eeg_path)
    emg_stats = _write_signal_npy(header, emg_index, emg_path)
    parity = _write_parity_report(header, recording_id, eeg_path, emg_path, output_dir)
    manifest = {
        "schema_version": SCORING_SCHEMA_VERSION,
        "recording_id": recording_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_edf": header.path,
        "eeg_signal": header.labels[eeg_index],
        "emg_signal": header.labels[emg_index],
        "eeg_source_unit": physical_dimension(header, eeg_index),
        "emg_source_unit": physical_dimension(header, emg_index),
        "eeg_unit_scale_applied": unit_scale_to_microvolts(header, eeg_index),
        "emg_unit_scale_applied": unit_scale_to_microvolts(header, emg_index),
        "output_unit": "uV",
        "signal_stats": {
            "eeg": eeg_stats,
            "emg": emg_stats,
        },
        "parity": parity,
        "sampling_rate_hz": eeg_rate,
        "sample_count": int(min(_sample_count(header, eeg_index), _sample_count(header, emg_index))),
        "eeg_npy": str(eeg_path),
        "emg_npy": str(emg_path),
        "fingerprint": fingerprint,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path


def _write_signal_npy(header: EDFHeader, signal_index: int, path: Path) -> dict[str, object]:
    total_samples = _sample_count(header, signal_index)
    arr = np.lib.format.open_memmap(path, mode="w+", dtype="float32", shape=(total_samples,))
    offset = 0
    sampled: list[np.ndarray] = []
    sampled_count = 0
    sample_limit = 1_000_000
    for digital in iter_signal_digital_chunks(header, signal_index, chunk_records=64):
        values = digital_to_microvolts(header, signal_index, digital).astype("float32", copy=False)
        arr[offset : offset + len(values)] = values
        offset += len(values)
        if sampled_count < sample_limit:
            take = min(sample_limit - sampled_count, len(values))
            sampled.append(np.asarray(values[:take], dtype="float32"))
            sampled_count += take
    arr.flush()
    sample = np.concatenate(sampled) if sampled else np.array([], dtype="float32")
    return _array_stats(sample) | {
        "source_unit": physical_dimension(header, signal_index),
        "unit_scale_applied": unit_scale_to_microvolts(header, signal_index),
        "output_unit": "uV",
        "sample_count": int(total_samples),
        "sampled_count": int(len(sample)),
    }


def _sample_count(header: EDFHeader, signal_index: int) -> int:
    return int(header.data_records * header.samples_per_record[signal_index])


def _classification_fingerprint(recording: RecordingSpec, config: AnalysisConfig, model: ResolvedModel) -> dict[str, object]:
    unit_values: list[str] = []
    try:
        if recording.edf_path and Path(recording.edf_path).exists():
            header = read_edf_header(recording.edf_path)
            eeg_signal = _first_available_signal(header, recording.channel_map.eeg, config.visualizer.eeg_channel, 0)
            emg_signal = _first_available_signal(header, recording.channel_map.emg, config.visualizer.emg_channel, min(1, header.signal_count - 1))
            for signal in [eeg_signal, emg_signal]:
                unit_values.extend(
                    [
                        str(header.labels[header.signal_index(signal)]),
                        str(physical_dimension(header, signal)),
                        str(unit_scale_to_microvolts(header, signal)),
                    ]
                )
    except Exception:  # noqa: BLE001
        unit_values.append("unit-unavailable")
    values = [
        str(SCORING_SCHEMA_VERSION),
        recording.recording_id,
        str(recording.channel_map.eeg),
        str(recording.channel_map.emg),
        str(model.alias),
        str(model.epoch_length_sec),
        str(model.input_windows),
        str(model.feature_window_sec),
        str(model.feature_mode),
        file_fingerprint(recording.edf_path or "", content_hash=False).stable_value(),
        file_fingerprint(model.model_file, content_hash=True).stable_value(),
        file_fingerprint(model.calibration_file or "", content_hash=True).stable_value(),
        *unit_values,
    ]
    return {
        "hash": config_hash(values),
        "edf": values[9],
        "model": values[10],
        "calibration": values[11],
        "model_alias": model.alias,
        "unit_inputs": unit_values,
        "visual_channels": {
            "eeg": recording.channel_map.eeg or config.visualizer.eeg_channel,
            "emg": recording.channel_map.emg or config.visualizer.emg_channel,
        },
    }


def _write_status(
    manifest_path: Path,
    status: str,
    fingerprint: dict[str, object],
    model: ResolvedModel,
    *,
    message: str = "",
    predictions_path: Path | None = None,
    staging: dict[str, object] | None = None,
) -> None:
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": SCORING_SCHEMA_VERSION,
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "status": status,
                "message": message,
                "fingerprint": fingerprint,
                "model_alias": model.alias,
                "model_file": str(model.model_file),
                "calibration_file": str(model.calibration_file) if model.calibration_file else None,
                "predictions_standardized": str(predictions_path) if predictions_path else None,
                "unit_scale_applied": _staging_scale_summary(staging),
                "staging": staging,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _first_available_signal(header: EDFHeader, *candidates):
    for candidate in candidates:
        if candidate is None:
            continue
        try:
            header.signal_index(candidate)
            return candidate
        except (KeyError, IndexError):
            continue
    return 0


def _array_stats(values: np.ndarray) -> dict[str, float | int | None]:
    if values.size == 0:
        return {
            "min": None,
            "max": None,
            "mean": None,
            "std": None,
            "p01": None,
            "p50": None,
            "p99": None,
        }
    return {
        "min": float(np.nanmin(values)),
        "max": float(np.nanmax(values)),
        "mean": float(np.nanmean(values)),
        "std": float(np.nanstd(values)),
        "p01": float(np.nanpercentile(values, 1)),
        "p50": float(np.nanpercentile(values, 50)),
        "p99": float(np.nanpercentile(values, 99)),
    }


def _write_parity_report(
    header: EDFHeader,
    recording_id: str,
    eeg_path: Path,
    emg_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    parquet_path = _matching_accusleepy_parquet(header, recording_id)
    parity_path = output_dir / "input_parity.csv"
    if parquet_path is None or not parquet_path.exists():
        frame = pd.DataFrame(
            [
                {
                    "RecordingID": recording_id,
                    "Status": "missing_reference_parquet",
                    "ReferencePath": str(parquet_path) if parquet_path else "",
                }
            ]
        )
        frame.to_csv(parity_path, index=False)
        return {"status": "missing_reference_parquet", "path": str(parity_path)}
    try:
        reference, reference_length = _read_parquet_sample(parquet_path, columns=["eeg", "emg"], limit=1_000_000)
        rows = []
        for label, npy_path, ref_col in [
            ("eeg", eeg_path, "eeg"),
            ("emg", emg_path, "emg"),
        ]:
            staged = np.load(npy_path, mmap_mode="r")
            count = min(len(staged), len(reference), 1_000_000)
            staged_sample = np.asarray(staged[:count], dtype="float64")
            ref_sample = reference[ref_col].to_numpy(dtype="float64", copy=False)[:count]
            corr = float(np.corrcoef(staged_sample, ref_sample)[0, 1]) if count > 2 else np.nan
            rows.append(
                {
                    "RecordingID": recording_id,
                    "Signal": label,
                    "Status": "ok",
                    "ReferencePath": str(parquet_path),
                    "ComparedSamples": int(count),
                    "StagedLength": int(len(staged)),
                    "ReferenceLength": int(reference_length),
                    "StagedStd": float(np.nanstd(staged_sample)),
                    "ReferenceStd": float(np.nanstd(ref_sample)),
                    "StdRatio": float(np.nanstd(staged_sample) / np.nanstd(ref_sample)) if np.nanstd(ref_sample) else np.nan,
                    "Correlation": corr,
                    "MeanAbsDiff": float(np.nanmean(np.abs(staged_sample - ref_sample))),
                }
            )
        frame = pd.DataFrame(rows)
        frame.to_csv(parity_path, index=False)
        min_corr = float(np.nanmin(frame["Correlation"])) if not frame.empty else np.nan
        max_diff = float(np.nanmax(frame["MeanAbsDiff"])) if not frame.empty else np.nan
        return {
            "status": "ok",
            "path": str(parity_path),
            "reference_path": str(parquet_path),
            "min_correlation": min_corr,
            "max_mean_abs_diff": max_diff,
        }
    except Exception as exc:  # noqa: BLE001
        pd.DataFrame(
            [
                {
                    "RecordingID": recording_id,
                    "Status": "error",
                    "ReferencePath": str(parquet_path),
                    "Message": str(exc),
                }
            ]
        ).to_csv(parity_path, index=False)
        return {"status": "error", "path": str(parity_path), "message": str(exc)}


def _matching_accusleepy_parquet(header: EDFHeader, recording_id: str) -> Path | None:
    edf_path = Path(header.path)
    candidates = [
        edf_path.parent.parent / "accusleepy_parquet" / f"{recording_id}.parquet",
        edf_path.parent / "accusleepy_parquet" / f"{recording_id}.parquet",
    ]
    if os.environ.get("MARS_ACCUSLEEPY_PARQUET_ROOT"):
        candidates.append(Path(os.environ["MARS_ACCUSLEEPY_PARQUET_ROOT"]) / f"{recording_id}.parquet")
    return next((path for path in candidates if path.exists()), candidates[0])


def _read_parquet_sample(path: Path, *, columns: list[str], limit: int) -> tuple[pd.DataFrame, int]:
    try:
        import pyarrow.parquet as pq

        parquet = pq.ParquetFile(path)
        batches = parquet.iter_batches(batch_size=limit, columns=columns)
        batch = next(batches)
        return batch.to_pandas(), parquet.metadata.num_rows
    except Exception:  # noqa: BLE001
        frame = pd.read_parquet(path, columns=columns)
        return frame.head(limit), len(frame)


def _augment_summary_with_staging(summary_path: Path, staging: dict[str, object]) -> None:
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    summary["InputUnit"] = "uV"
    summary["UnitScaleApplied"] = _staging_scale_summary(staging)
    summary["SignalStats"] = staging.get("signal_stats", {})
    summary["InputParity"] = staging.get("parity", {})
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def _staging_scale_summary(staging: dict[str, object] | None) -> dict[str, object]:
    if not staging:
        return {}
    return {
        "eeg_source_unit": staging.get("eeg_source_unit"),
        "emg_source_unit": staging.get("emg_source_unit"),
        "eeg_unit_scale_applied": staging.get("eeg_unit_scale_applied"),
        "emg_unit_scale_applied": staging.get("emg_unit_scale_applied"),
        "output_unit": staging.get("output_unit"),
    }
