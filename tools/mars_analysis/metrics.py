from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .labels import LabelInventoryRow, load_label_frame, normalize_label


SCORED_STATES = ["Wake", "NREM", "REM"]
STRICT_EXCLUDED = {"TransitionalOrUnclassified", "Unknown", ""}
EXTRA_PRED_CLASSES = ["Artifact", "Other", "Missing"]
MAX_DETAILED_DIFFERENCES_PER_RECORDING = 5000


@dataclass(slots=True)
class ComparisonResult:
    recording_id: str
    baseline_source: str
    compared_source: str
    baseline_model: str
    compared_model: str
    baseline_path: str
    compared_path: str
    epoch_count_baseline: int
    epoch_count_compared: int
    aligned_epoch_count: int
    strict_epoch_count: int
    accuracy: float | None
    balanced_accuracy: float | None
    macro_f1: float | None
    weighted_f1: float | None
    kappa: float | None
    disagreement_count: int
    disagreement_pct: float | None
    strict_accuracy: float | None
    strict_balanced_accuracy: float | None
    strict_macro_f1: float | None
    status: str
    message: str = ""

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(slots=True)
class IntervalPrediction:
    start: float
    end: float
    label: str
    epoch_index: object = None
    confidence: object = None


def choose_baselines(rows: list[LabelInventoryRow]) -> dict[str, LabelInventoryRow]:
    baselines: dict[str, LabelInventoryRow] = {}
    for row in sorted(rows, key=_baseline_priority):
        if row.status != "ok" or row.recording_id in baselines:
            continue
        if row.label_source == "offline_accusleepy":
            baselines[row.recording_id] = row
    return baselines


def choose_state_baselines(rows: list[LabelInventoryRow]) -> dict[str, LabelInventoryRow]:
    """Pick one label frame per recording for state-percentage reporting.

    Unlike :func:`choose_baselines`, which only accepts offline AccuSleePy
    reference labels, this falls back to the model's own scored predictions
    (``real_time_mars``) and then any usable label. That keeps the vigilance
    state outputs populated when a user simply scores EDFs with the bundled
    model and has no external reference labels configured.
    """
    baselines: dict[str, LabelInventoryRow] = {}
    for row in sorted(rows, key=_state_baseline_priority):
        if row.status != "ok" or row.recording_id in baselines:
            continue
        baselines[row.recording_id] = row
    return baselines


def _state_baseline_priority(row: LabelInventoryRow) -> tuple[int, int, str]:
    source_order = {"offline_accusleepy": 0, "real_time_mars": 1, "user_labels": 2}
    source_rank = source_order.get(row.label_source, 5)
    format_rank = 0 if row.format == "mars_standard" else 1
    return (source_rank, format_rank, row.path)


def compare_against_accusleepy(
    rows: list[LabelInventoryRow],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]:
    rows = choose_representatives(rows)
    baselines = choose_baselines(rows)
    summaries: list[dict[str, object]] = []
    differences: list[pd.DataFrame] = []
    confusion_matrices: dict[str, pd.DataFrame] = {}

    for row in rows:
        if row.status != "ok":
            continue
        baseline = baselines.get(row.recording_id)
        if baseline is None:
            continue
        if row.path == baseline.path and row.model == baseline.model:
            continue
        if row.label_source == baseline.label_source:
            continue
        metadata_alignment_error = validate_inventory_alignment(baseline, row)
        if metadata_alignment_error:
            summaries.append(_comparison_dict(invalid_alignment_result(baseline, row, metadata_alignment_error), baseline, row))
            continue
        result, diff_frame, confusion = compare_pair(baseline, row)
        summaries.append(_comparison_dict(result, baseline, row))
        if not diff_frame.empty:
            differences.append(diff_frame)
        key = f"{row.recording_id}__{sanitize_name(row.label_source)}__{sanitize_name(row.model)}"
        confusion_matrices[key] = confusion

    summary_frame = pd.DataFrame(summaries)
    diff_all = pd.concat(differences, ignore_index=True) if differences else pd.DataFrame()
    return summary_frame, diff_all, confusion_matrices


def compare_mars_vs_reference(
    rows: list[LabelInventoryRow],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    reference_rows = _choose_reference_rows(rows)
    rows = choose_representatives(rows)
    baselines = choose_baselines(rows)
    summaries: list[dict[str, object]] = []
    differences: list[pd.DataFrame] = []
    confusion_frames: list[pd.DataFrame] = []
    alignment_rows: list[dict[str, object]] = []

    for recording_id, baseline in baselines.items():
        compared = reference_rows.get(recording_id)
        if compared is None:
            alignment_rows.append(_missing_reference_alignment_row(baseline))
            summaries.append(_missing_reference_summary_row(baseline))
            continue
        try:
            base = load_label_frame(
                baseline.path,
                recording_id=recording_id,
                default_epoch_length=baseline.epoch_length_sec or 2.5,
            )
            other = load_label_frame(
                compared.path,
                recording_id=recording_id,
                default_epoch_length=compared.epoch_length_sec or 2.0,
            )
            aligned, alignment = align_reference_to_mars_time_grid(base, other)
            alignment.update(
                {
                    "RecordingID": recording_id,
                    "MarsPath": baseline.path,
                    "ReferencePath": compared.path,
                    "MarsModel": baseline.model,
                    "MarsEpochLengthSeconds": _single_epoch_length(base),
                    "ReferenceEpochLengthSeconds": _single_epoch_length(other),
                }
            )
            if aligned.empty:
                alignment["Status"] = "no_time_overlap"
                alignment_rows.append(alignment)
                summaries.append(_no_overlap_summary_row(baseline, compared, len(base), len(other)))
                continue

            alignment_rows.append(alignment)
            result, diff, confusion = _mars_reference_result_frames(baseline, compared, aligned, len(base), len(other))
            summaries.append(result)
            if not diff.empty:
                differences.append(diff)
            confusion_frames.append(confusion)
        except Exception as exc:  # noqa: BLE001 - keep all recordings reportable
            alignment_rows.append(
                {
                    "RecordingID": recording_id,
                    "MarsEpochs": baseline.epoch_count,
                    "ReferenceEpochs": compared.epoch_count,
                    "AlignedEpochs": 0,
                    "MissingEpochs": baseline.epoch_count,
                    "CoveragePercent": 0.0,
                    "StrictValidEpochs": 0,
                    "Status": "corrupt_label_file",
                    "AlignmentMethod": "time_center",
                    "Message": str(exc),
                    "MarsPath": baseline.path,
                    "ReferencePath": compared.path,
                    "MarsModel": baseline.model,
                    "MarsEpochLengthSeconds": baseline.epoch_length_sec,
                    "ReferenceEpochLengthSeconds": compared.epoch_length_sec,
                }
            )
            summaries.append(_error_summary_row(baseline, compared, str(exc)))

    summary = pd.DataFrame(summaries)
    diff_all = pd.concat(differences, ignore_index=True) if differences else pd.DataFrame()
    confusion_all = (
        pd.concat(confusion_frames, ignore_index=True)
        .groupby(["ReferenceLabel", "MarsLabel"], as_index=False)["EpochCount"]
        .sum()
        if confusion_frames
        else pd.DataFrame(columns=["ReferenceLabel", "MarsLabel", "EpochCount"])
    )
    alignment_summary = pd.DataFrame(alignment_rows)
    return summary, diff_all, confusion_all, alignment_summary


def compare_pdf_compatible_heldout(
    rows: list[LabelInventoryRow],
    ground_truth_path: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Reproduce the legacy PDF benchmark orientation.

    The documentation benchmark iterated strict reference/reference epochs and
    chose the prediction interval with the largest temporal overlap. This is
    intentionally separate from the full-dataset concordance path, which maps
    reference labels onto the MARS epoch grid.
    """
    rows = choose_representatives(rows)
    baselines = choose_baselines(rows)
    truth = _load_pdf_benchmark_ground_truth(ground_truth_path)
    if baselines:
        truth = truth[truth["RecordingID"].astype(str).isin(set(baselines))]

    summaries: list[dict[str, object]] = []
    difference_frames: list[pd.DataFrame] = []
    alignment_rows: list[dict[str, object]] = []
    aggregate_pairs: list[pd.DataFrame] = []

    for recording_id, truth_group in truth.groupby("RecordingID", sort=True):
        baseline = baselines.get(str(recording_id))
        if baseline is None:
            alignment_rows.append(
                {
                    "RecordingID": recording_id,
                    "TruthEpochs": len(truth_group),
                    "PredictionEpochs": 0,
                    "AlignedTruthEpochs": 0,
                    "MissingPredictionEpochs": len(truth_group),
                    "CoveragePercent": 0.0,
                    "Status": "missing_mars_predictions",
                    "AlignmentMethod": "truth_epoch_max_prediction_overlap",
                    "MarsPath": "",
                    "GroundTruthPath": str(ground_truth_path),
                    "MarsModel": "",
                }
            )
            continue
        try:
            predictions = load_label_frame(
                baseline.path,
                recording_id=str(recording_id),
                default_epoch_length=baseline.epoch_length_sec or 2.5,
            )
            aligned = _align_truth_epochs_to_prediction_overlap(truth_group, predictions)
            metrics = _legacy_classification_metrics(aligned["TruthLabel"], aligned["MarsLabel"])
            mismatch = aligned[aligned["TruthLabel"] != aligned["MarsLabel"]].copy()
            summaries.append(
                {
                    "recording_id": recording_id,
                    "mars_model": baseline.model,
                    "mars_path": baseline.path,
                    "ground_truth_path": str(ground_truth_path),
                    "truth_epoch_count": len(truth_group),
                    "mars_epoch_count": len(predictions),
                    "aligned_epoch_count": len(aligned),
                    "missing_prediction_count": int((aligned["MarsLabel"] == "Missing").sum()),
                    "coverage_percent": float((aligned["MarsLabel"] != "Missing").mean()) if len(aligned) else 0.0,
                    "accuracy": metrics["Accuracy"],
                    "balanced_accuracy": metrics["BalancedAccuracy"],
                    "macro_f1": metrics["MacroF1"],
                    "weighted_f1": metrics["WeightedF1"],
                    "kappa": metrics["CohenKappa"],
                    "disagreement_count": len(mismatch),
                    "disagreement_pct": float(len(mismatch) / len(aligned)) if len(aligned) else None,
                    "status": "ok",
                    "message": "",
                }
            )
            alignment_rows.append(
                {
                    "RecordingID": recording_id,
                    "TruthEpochs": len(truth_group),
                    "PredictionEpochs": len(predictions),
                    "AlignedTruthEpochs": len(aligned),
                    "MissingPredictionEpochs": int((aligned["MarsLabel"] == "Missing").sum()),
                    "CoveragePercent": float((aligned["MarsLabel"] != "Missing").mean()) if len(aligned) else 0.0,
                    "Status": "ok",
                    "AlignmentMethod": "truth_epoch_max_prediction_overlap",
                    "MarsPath": baseline.path,
                    "GroundTruthPath": str(ground_truth_path),
                    "MarsModel": baseline.model,
                }
            )
            if not mismatch.empty:
                difference_frames.append(
                    pd.DataFrame(
                        {
                            "RecordingID": mismatch["RecordingID"],
                            "TruthEpochIndex": mismatch["TruthEpochIndex"],
                            "TruthEpochStartSeconds": mismatch["TruthEpochStartSeconds"],
                            "TruthLabel": mismatch["TruthLabel"],
                            "MarsEpochIndex": mismatch["MarsEpochIndex"],
                            "MarsEpochStartSeconds": mismatch["MarsEpochStartSeconds"],
                            "MarsLabel": mismatch["MarsLabel"],
                            "MarsConfidence": mismatch["MarsConfidence"],
                            "MarsPath": baseline.path,
                            "GroundTruthPath": str(ground_truth_path),
                        }
                    )
                )
            aggregate_pairs.append(aligned[["TruthLabel", "MarsLabel"]])
        except Exception as exc:  # noqa: BLE001 - report per-recording failures
            summaries.append(
                {
                    "recording_id": recording_id,
                    "mars_model": baseline.model,
                    "mars_path": baseline.path,
                    "ground_truth_path": str(ground_truth_path),
                    "truth_epoch_count": len(truth_group),
                    "mars_epoch_count": baseline.epoch_count,
                    "aligned_epoch_count": 0,
                    "missing_prediction_count": len(truth_group),
                    "coverage_percent": 0.0,
                    "accuracy": None,
                    "balanced_accuracy": None,
                    "macro_f1": None,
                    "weighted_f1": None,
                    "kappa": None,
                    "disagreement_count": 0,
                    "disagreement_pct": None,
                    "status": "error",
                    "message": str(exc),
                }
            )

    summary = pd.DataFrame(summaries)
    differences = pd.concat(difference_frames, ignore_index=True) if difference_frames else pd.DataFrame()
    alignment = pd.DataFrame(alignment_rows)
    aggregate = pd.concat(aggregate_pairs, ignore_index=True) if aggregate_pairs else pd.DataFrame(columns=["TruthLabel", "MarsLabel"])
    confusion = _legacy_confusion_frame(aggregate["TruthLabel"], aggregate["MarsLabel"])
    aggregate_summary = _legacy_classification_metrics(aggregate["TruthLabel"], aggregate["MarsLabel"])
    aggregate_summary.update(
        {
            "Model": _first_non_empty(summary.get("mars_model", pd.Series(dtype=object))),
            "PredictionRecordings": int((summary.get("status", pd.Series(dtype=object)) == "ok").sum()) if not summary.empty else 0,
            "GroundTruthRecordings": int(truth["RecordingID"].nunique()),
            "PredictionFile": ";".join(sorted(set(summary.get("mars_path", pd.Series(dtype=object)).dropna().astype(str)))) if not summary.empty else "",
            "GroundTruthFile": str(ground_truth_path),
        }
    )
    return summary, differences, confusion, alignment, aggregate_summary


def _choose_reference_rows(rows: list[LabelInventoryRow]) -> dict[str, LabelInventoryRow]:
    chosen: dict[str, LabelInventoryRow] = {}
    for row in sorted(rows, key=_reference_priority):
        if row.status == "ok" and row.label_source == "reference_reference":
            chosen.setdefault(row.recording_id, row)
    return chosen


def _reference_priority(row: LabelInventoryRow) -> tuple[int, int, str]:
    text = row.path.lower()
    aligned_rank = 0 if (
        row.format == "accusleepy_brain_state"
        or "reference_aligned_labels" in text
        or "reference_accusleepy_calibration_labels" in text
    ) else 1
    source_rank = 0 if row.format == "reference_reference_epoch_labels" else 1
    full_export_rank = 0 if "reference_reference_epoch_labels_full" in text else 1
    count_rank = 0 if row.epoch_count else 9
    return (aligned_rank, source_rank, full_export_rank, count_rank, row.path)


def align_reference_to_mars_time_grid(
    mars: pd.DataFrame,
    reference: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, object]]:
    mars = mars.copy().reset_index(drop=True)
    reference = reference.copy().reset_index(drop=True)
    for frame in (mars, reference):
        frame["EpochStartSeconds"] = pd.to_numeric(frame["EpochStartSeconds"], errors="coerce")
        frame["EpochLengthSeconds"] = pd.to_numeric(frame["EpochLengthSeconds"], errors="coerce")
        frame["EpochIndex"] = pd.to_numeric(frame["EpochIndex"], errors="coerce")
    mars = mars.dropna(subset=["EpochStartSeconds", "EpochLengthSeconds"])
    reference = reference.dropna(subset=["EpochStartSeconds", "EpochLengthSeconds"]).sort_values("EpochStartSeconds").reset_index(drop=True)
    if mars.empty or reference.empty:
        return pd.DataFrame(), {
            "MarsEpochs": len(mars),
            "ReferenceEpochs": len(reference),
            "AlignedEpochs": 0,
            "MissingEpochs": len(mars),
            "CoveragePercent": 0.0,
            "StrictValidEpochs": 0,
            "Status": "no_time_overlap",
            "AlignmentMethod": "time_center",
            "Message": "No usable epoch timing rows",
        }

    mat_starts = reference["EpochStartSeconds"].to_numpy(dtype=float)
    mat_lengths = reference["EpochLengthSeconds"].to_numpy(dtype=float)
    mat_ends = mat_starts + mat_lengths
    mat_centers = mat_starts + mat_lengths / 2.0
    mars_starts = mars["EpochStartSeconds"].to_numpy(dtype=float)
    mars_lengths = mars["EpochLengthSeconds"].to_numpy(dtype=float)
    mars_centers = mars_starts + mars_lengths / 2.0
    tolerance = max(float(np.nanmedian(mars_lengths)), float(np.nanmedian(mat_lengths))) / 2.0 + 1e-6

    interval_indices = np.searchsorted(mat_starts, mars_centers, side="right") - 1
    in_bounds = (interval_indices >= 0) & (interval_indices < len(reference))
    in_interval = np.zeros(len(mars_centers), dtype=bool)
    bounded_positions = np.where(in_bounds)[0]
    in_interval[bounded_positions] = (
        (mat_starts[interval_indices[bounded_positions]] <= mars_centers[bounded_positions])
        & (mars_centers[bounded_positions] < mat_ends[interval_indices[bounded_positions]])
    )

    matched_indices = np.full(len(mars_centers), -1, dtype=np.int64)
    methods = np.full(len(mars_centers), "missing", dtype=object)
    matched_indices[in_interval] = interval_indices[in_interval]
    methods[in_interval] = "interval_contains_center"

    missing_positions = np.where(~in_interval)[0]
    if len(missing_positions):
        insertion = np.searchsorted(mat_centers, mars_centers[missing_positions], side="left")
        right = np.clip(insertion, 0, len(mat_centers) - 1)
        left = np.clip(insertion - 1, 0, len(mat_centers) - 1)
        right_dist = np.abs(mat_centers[right] - mars_centers[missing_positions])
        left_dist = np.abs(mat_centers[left] - mars_centers[missing_positions])
        nearest = np.where(left_dist <= right_dist, left, right)
        nearest_dist = np.minimum(left_dist, right_dist)
        nearest_ok = nearest_dist <= tolerance
        ok_positions = missing_positions[nearest_ok]
        matched_indices[ok_positions] = nearest[nearest_ok]
        methods[ok_positions] = "nearest_center"

    valid_positions = np.where(matched_indices >= 0)[0]
    if len(valid_positions) == 0:
        return pd.DataFrame(), {
            "MarsEpochs": len(mars),
            "ReferenceEpochs": len(reference),
            "AlignedEpochs": 0,
            "MissingEpochs": len(mars),
            "CoveragePercent": 0.0,
            "StrictValidEpochs": 0,
            "Status": "no_time_overlap",
            "AlignmentMethod": "time_center",
            "Message": "No reference epoch interval overlaps MARS epoch centers",
        }

    mars_aligned = mars.iloc[valid_positions].reset_index(drop=True)
    reference_aligned = reference.iloc[matched_indices[valid_positions]].reset_index(drop=True)
    aligned = pd.DataFrame(
        {
            "RecordingID": mars_aligned["RecordingID"].astype(str),
            "MarsEpochIndex": mars_aligned["EpochIndex"],
            "MarsEpochStartSeconds": mars_aligned["EpochStartSeconds"],
            "MarsEpochLengthSeconds": mars_aligned["EpochLengthSeconds"],
            "MarsRawCode": mars_aligned["RawCode"],
            "MarsLabel": mars_aligned["PredLabel"].astype(str),
            "MarsConfidence": mars_aligned["Confidence"],
            "ReferenceEpochIndex": reference_aligned["EpochIndex"],
            "ReferenceEpochStartSeconds": reference_aligned["EpochStartSeconds"],
            "ReferenceEpochLengthSeconds": reference_aligned["EpochLengthSeconds"],
            "ReferenceRawCode": reference_aligned["RawCode"],
            "ReferenceLabel": reference_aligned["PredLabel"].astype(str),
            "AlignmentMethod": methods[valid_positions].tolist(),
        }
    )
    strict = aligned[~aligned["ReferenceLabel"].isin(STRICT_EXCLUDED)]
    method_counts = aligned["AlignmentMethod"].value_counts().to_dict()
    return aligned, {
        "MarsEpochs": len(mars),
        "ReferenceEpochs": len(reference),
        "AlignedEpochs": len(aligned),
        "MissingEpochs": max(0, len(mars) - len(aligned)),
        "CoveragePercent": float(len(aligned) / len(mars)) if len(mars) else 0.0,
        "StrictValidEpochs": len(strict),
        "Status": "ok",
        "AlignmentMethod": ";".join(f"{key}:{value}" for key, value in sorted(method_counts.items())),
        "Message": "",
    }


def _mars_reference_result_frames(
    baseline: LabelInventoryRow,
    compared: LabelInventoryRow,
    aligned: pd.DataFrame,
    baseline_count: int,
    compared_count: int,
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    metrics_all = classification_metrics(aligned["ReferenceLabel"], aligned["MarsLabel"])
    strict = aligned[~aligned["ReferenceLabel"].isin(STRICT_EXCLUDED)]
    metrics_strict = classification_metrics(strict["ReferenceLabel"], strict["MarsLabel"])
    mismatch = aligned[aligned["ReferenceLabel"] != aligned["MarsLabel"]].copy()
    mismatch_count = len(mismatch)
    exported_mismatch = mismatch.head(MAX_DETAILED_DIFFERENCES_PER_RECORDING)
    diff = pd.DataFrame(
        {
            "RecordingID": exported_mismatch["RecordingID"],
            "MarsEpochIndex": exported_mismatch["MarsEpochIndex"],
            "MarsEpochStartSeconds": exported_mismatch["MarsEpochStartSeconds"],
            "MarsLabel": exported_mismatch["MarsLabel"],
            "MarsConfidence": exported_mismatch["MarsConfidence"],
            "ReferenceEpochIndex": exported_mismatch["ReferenceEpochIndex"],
            "ReferenceEpochStartSeconds": exported_mismatch["ReferenceEpochStartSeconds"],
            "ReferenceLabel": exported_mismatch["ReferenceLabel"],
            "ReferenceRawCode": exported_mismatch["ReferenceRawCode"],
            "AlignmentMethod": exported_mismatch["AlignmentMethod"],
            "MarsPath": baseline.path,
            "ReferencePath": compared.path,
            "DifferenceExportTruncated": mismatch_count > MAX_DETAILED_DIFFERENCES_PER_RECORDING,
        }
    )
    confusion = (
        aligned.groupby(["ReferenceLabel", "MarsLabel"], dropna=False)
        .size()
        .rename("EpochCount")
        .reset_index()
    )
    return (
        {
            "recording_id": baseline.recording_id,
            "mars_source": baseline.label_source,
            "reference_source": compared.label_source,
            "mars_model": baseline.model,
            "reference_model": compared.model,
            "mars_path": baseline.path,
            "reference_path": compared.path,
            "mars_epoch_count": baseline_count,
            "reference_epoch_count": compared_count,
            "aligned_epoch_count": len(aligned),
            "strict_valid_epoch_count": len(strict),
            "coverage_percent": float(len(aligned) / baseline_count) if baseline_count else 0.0,
            "accuracy": metrics_all["accuracy"],
            "balanced_accuracy": metrics_all["balanced_accuracy"],
            "macro_f1": metrics_all["macro_f1"],
            "weighted_f1": metrics_all["weighted_f1"],
            "kappa": metrics_all["kappa"],
            "strict_accuracy": metrics_strict["accuracy"],
            "strict_balanced_accuracy": metrics_strict["balanced_accuracy"],
            "strict_macro_f1": metrics_strict["macro_f1"],
            "strict_weighted_f1": metrics_strict["weighted_f1"],
            "strict_kappa": metrics_strict["kappa"],
            "disagreement_count": mismatch_count,
            "disagreement_pct": float(mismatch_count / len(aligned)) if len(aligned) else None,
            "disagreement_exported_rows": len(diff),
            "disagreement_export_limit_per_recording": MAX_DETAILED_DIFFERENCES_PER_RECORDING,
            "status": "ok",
            "message": "",
            "mars_mapping": baseline.label_mapping,
            "reference_mapping": compared.label_mapping,
        },
        diff,
        confusion,
    )


def _load_pdf_benchmark_ground_truth(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"RecordingID", "EpochStartSeconds", "EpochLengthSeconds"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"PDF benchmark ground truth is missing columns: {sorted(missing)}")
    label_column = "BenchmarkStateLabel" if "BenchmarkStateLabel" in frame.columns else "ReferenceStateLabel"
    if label_column not in frame.columns:
        raise ValueError("PDF benchmark ground truth needs BenchmarkStateLabel or ReferenceStateLabel")
    out = pd.DataFrame(
        {
            "RecordingID": frame["RecordingID"].astype(str),
            "TruthEpochIndex": pd.to_numeric(frame["EpochIndex"], errors="coerce") if "EpochIndex" in frame.columns else range(1, len(frame) + 1),
            "TruthEpochStartSeconds": pd.to_numeric(frame["EpochStartSeconds"], errors="coerce"),
            "TruthEpochLengthSeconds": pd.to_numeric(frame["EpochLengthSeconds"], errors="coerce"),
            "TruthLabel": frame[label_column].map(lambda value: normalize_label(value, label_source="reference_reference")),
        }
    )
    out = out.dropna(subset=["TruthEpochStartSeconds", "TruthEpochLengthSeconds"])
    return out[out["TruthLabel"].isin(SCORED_STATES)].copy()


def _prediction_intervals(predictions: pd.DataFrame) -> list[IntervalPrediction]:
    frame = predictions.copy()
    frame["EpochStartSeconds"] = pd.to_numeric(frame["EpochStartSeconds"], errors="coerce")
    frame["EpochLengthSeconds"] = pd.to_numeric(frame["EpochLengthSeconds"], errors="coerce")
    frame = frame.dropna(subset=["EpochStartSeconds", "EpochLengthSeconds"]).sort_values("EpochStartSeconds")
    intervals = []
    for row in frame.itertuples(index=False):
        start = float(getattr(row, "EpochStartSeconds"))
        length = float(getattr(row, "EpochLengthSeconds"))
        intervals.append(
            IntervalPrediction(
                start=start,
                end=start + length,
                label=normalize_label(getattr(row, "PredLabel", "Missing")),
                epoch_index=getattr(row, "EpochIndex", None),
                confidence=getattr(row, "Confidence", None),
            )
        )
    return intervals


def _best_prediction_for_truth_epoch(
    intervals: list[IntervalPrediction],
    pointer: int,
    start: float,
    end: float,
) -> tuple[IntervalPrediction, int]:
    while pointer < len(intervals) and intervals[pointer].end <= start:
        pointer += 1
    best = IntervalPrediction(start=np.nan, end=np.nan, label="Missing")
    best_overlap = 0.0
    idx = pointer
    while idx < len(intervals) and intervals[idx].start < end:
        overlap = min(end, intervals[idx].end) - max(start, intervals[idx].start)
        if overlap > best_overlap:
            best_overlap = overlap
            best = intervals[idx]
        idx += 1
    return best, pointer


def _align_truth_epochs_to_prediction_overlap(truth: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    truth = truth.sort_values("TruthEpochStartSeconds").reset_index(drop=True)
    intervals = _prediction_intervals(predictions)
    rows: list[dict[str, object]] = []
    pointer = 0
    for row in truth.itertuples(index=False):
        start = float(getattr(row, "TruthEpochStartSeconds"))
        end = start + float(getattr(row, "TruthEpochLengthSeconds"))
        prediction, pointer = _best_prediction_for_truth_epoch(intervals, pointer, start, end)
        rows.append(
            {
                "RecordingID": getattr(row, "RecordingID"),
                "TruthEpochIndex": getattr(row, "TruthEpochIndex"),
                "TruthEpochStartSeconds": start,
                "TruthEpochLengthSeconds": getattr(row, "TruthEpochLengthSeconds"),
                "TruthLabel": getattr(row, "TruthLabel"),
                "MarsEpochIndex": prediction.epoch_index,
                "MarsEpochStartSeconds": prediction.start,
                "MarsEpochLengthSeconds": prediction.end - prediction.start if pd.notna(prediction.start) else np.nan,
                "MarsLabel": prediction.label,
                "MarsConfidence": prediction.confidence,
            }
        )
    return pd.DataFrame(rows)


def _legacy_per_class_metrics(confusion: dict[str, pd.Series]) -> list[dict[str, object]]:
    rows = []
    for label in SCORED_STATES:
        tp = int(confusion[label].get(label, 0))
        fp = int(sum(confusion[truth].get(label, 0) for truth in SCORED_STATES if truth != label))
        fn = int(sum(confusion[label].get(pred, 0) for pred in SCORED_STATES + EXTRA_PRED_CLASSES if pred != label))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        support = int(sum(confusion[label].get(pred, 0) for pred in SCORED_STATES + EXTRA_PRED_CLASSES))
        rows.append({"Class": label, "Precision": precision, "Recall": recall, "F1": f1, "Support": support})
    return rows


def _legacy_confusion_counts(y_true: pd.Series, y_pred: pd.Series) -> dict[str, pd.Series]:
    frame = pd.DataFrame({"true": y_true.astype(str), "pred": y_pred.astype(str)})
    return {
        label: frame[frame["true"] == label]["pred"].value_counts()
        for label in SCORED_STATES
    }


def _legacy_classification_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float | int | None]:
    frame = pd.DataFrame({"true": y_true, "pred": y_pred}).dropna()
    frame = frame[frame["true"].isin(SCORED_STATES)]
    total = len(frame)
    if total == 0:
        return {
            "TotalAlignedEpochs": 0,
            "Accuracy": None,
            "BalancedAccuracy": None,
            "MacroF1": None,
            "WeightedF1": None,
            "CohenKappa": None,
        }
    counts = _legacy_confusion_counts(frame["true"], frame["pred"])
    per_class = _legacy_per_class_metrics(counts)
    correct = sum(int(counts[label].get(label, 0)) for label in SCORED_STATES)
    truth_marginals = {label: int(counts[label].sum()) for label in SCORED_STATES}
    pred_marginals = {label: int(sum(counts[truth].get(label, 0) for truth in SCORED_STATES)) for label in SCORED_STATES}
    expected = sum(truth_marginals[label] * pred_marginals[label] for label in SCORED_STATES) / (total * total)
    observed = correct / total
    return {
        "TotalAlignedEpochs": total,
        "Accuracy": observed,
        "BalancedAccuracy": sum(float(row["Recall"]) for row in per_class) / len(per_class),
        "MacroF1": sum(float(row["F1"]) for row in per_class) / len(per_class),
        "WeightedF1": sum(float(row["F1"]) * int(row["Support"]) for row in per_class) / total,
        "CohenKappa": (observed - expected) / (1.0 - expected) if expected != 1.0 else (1.0 if observed == 1.0 else None),
    }


def _legacy_confusion_frame(y_true: pd.Series, y_pred: pd.Series) -> pd.DataFrame:
    counts = _legacy_confusion_counts(y_true, y_pred)
    rows = []
    for truth in SCORED_STATES:
        row = {"TruthLabel": truth}
        for pred in SCORED_STATES + EXTRA_PRED_CLASSES:
            row[pred] = int(counts[truth].get(pred, 0))
        rows.append(row)
    return pd.DataFrame(rows)


def _first_non_empty(values: pd.Series) -> str:
    for value in values.dropna().astype(str):
        if value:
            return value
    return ""


def choose_representatives(rows: list[LabelInventoryRow]) -> list[LabelInventoryRow]:
    """Keep the inventory complete but compare one best file per recording/source/model."""
    grouped: dict[tuple[str, str, str], LabelInventoryRow] = {}
    for row in sorted(rows, key=_representative_priority):
        if row.status != "ok":
            continue
        key = (row.recording_id, row.label_source, row.model)
        grouped.setdefault(key, row)
    return list(grouped.values())


def compare_pair(
    baseline: LabelInventoryRow,
    compared: LabelInventoryRow,
) -> tuple[ComparisonResult, pd.DataFrame, pd.DataFrame]:
    try:
        base = load_label_frame(
            baseline.path,
            recording_id=baseline.recording_id,
            default_epoch_length=baseline.epoch_length_sec or 2.5,
        )
        other = load_label_frame(
            compared.path,
            recording_id=compared.recording_id,
            default_epoch_length=compared.epoch_length_sec or baseline.epoch_length_sec or 2.5,
        )
        alignment_error = validate_alignment(baseline, compared, base, other)
        if alignment_error:
            result = ComparisonResult(
                recording_id=baseline.recording_id,
                baseline_source=baseline.label_source,
                compared_source=compared.label_source,
                baseline_model=baseline.model,
                compared_model=compared.model,
                baseline_path=baseline.path,
                compared_path=compared.path,
                epoch_count_baseline=len(base),
                epoch_count_compared=len(other),
                aligned_epoch_count=0,
                strict_epoch_count=0,
                accuracy=None,
                balanced_accuracy=None,
                macro_f1=None,
                weighted_f1=None,
                kappa=None,
                disagreement_count=0,
                disagreement_pct=None,
                strict_accuracy=None,
                strict_balanced_accuracy=None,
                strict_macro_f1=None,
                status="invalid_alignment",
                message=alignment_error,
            )
            return result, pd.DataFrame(), pd.DataFrame()
        merged = base.merge(
            other,
            on=["RecordingID", "EpochIndex"],
            suffixes=("_baseline", "_compared"),
            how="inner",
        )
        if merged.empty:
            raise ValueError("No aligned epochs after merging by RecordingID and EpochIndex")

        metrics_all = classification_metrics(
            merged["PredLabel_baseline"],
            merged["PredLabel_compared"],
        )
        strict_mask = ~merged["PredLabel_baseline"].isin(STRICT_EXCLUDED)
        strict = merged[strict_mask]
        metrics_strict = classification_metrics(
            strict["PredLabel_baseline"],
            strict["PredLabel_compared"],
        )
        confusion = confusion_matrix(
            merged["PredLabel_baseline"],
            merged["PredLabel_compared"],
        )
        diff = build_difference_frame(baseline, compared, merged)
        result = ComparisonResult(
            recording_id=baseline.recording_id,
            baseline_source=baseline.label_source,
            compared_source=compared.label_source,
            baseline_model=baseline.model,
            compared_model=compared.model,
            baseline_path=baseline.path,
            compared_path=compared.path,
            epoch_count_baseline=len(base),
            epoch_count_compared=len(other),
            aligned_epoch_count=len(merged),
            strict_epoch_count=len(strict),
            accuracy=metrics_all["accuracy"],
            balanced_accuracy=metrics_all["balanced_accuracy"],
            macro_f1=metrics_all["macro_f1"],
            weighted_f1=metrics_all["weighted_f1"],
            kappa=metrics_all["kappa"],
            disagreement_count=int((merged["PredLabel_baseline"] != merged["PredLabel_compared"]).sum()),
            disagreement_pct=float((merged["PredLabel_baseline"] != merged["PredLabel_compared"]).mean()),
            strict_accuracy=metrics_strict["accuracy"],
            strict_balanced_accuracy=metrics_strict["balanced_accuracy"],
            strict_macro_f1=metrics_strict["macro_f1"],
            status="ok",
        )
        return result, diff, confusion
    except Exception as exc:  # noqa: BLE001 - comparison should keep going
        result = ComparisonResult(
            recording_id=baseline.recording_id,
            baseline_source=baseline.label_source,
            compared_source=compared.label_source,
            baseline_model=baseline.model,
            compared_model=compared.model,
            baseline_path=baseline.path,
            compared_path=compared.path,
            epoch_count_baseline=baseline.epoch_count,
            epoch_count_compared=compared.epoch_count,
            aligned_epoch_count=0,
            strict_epoch_count=0,
            accuracy=None,
            balanced_accuracy=None,
            macro_f1=None,
            weighted_f1=None,
            kappa=None,
            disagreement_count=0,
            disagreement_pct=None,
            strict_accuracy=None,
            strict_balanced_accuracy=None,
            strict_macro_f1=None,
            status="error",
            message=str(exc),
        )
        return result, pd.DataFrame(), pd.DataFrame()


def validate_inventory_alignment(
    baseline: LabelInventoryRow,
    compared: LabelInventoryRow,
) -> str | None:
    if baseline.epoch_count != compared.epoch_count:
        return (
            f"Epoch count mismatch: baseline={baseline.epoch_count} "
            f"compared={compared.epoch_count}"
        )
    if baseline.epoch_length_sec is None or compared.epoch_length_sec is None:
        return "Missing epoch length"
    if abs(baseline.epoch_length_sec - compared.epoch_length_sec) > 1e-6:
        return (
            f"Epoch length mismatch: baseline={baseline.epoch_length_sec} "
            f"compared={compared.epoch_length_sec}"
        )
    if (
        baseline.first_epoch_start is not None
        and compared.first_epoch_start is not None
        and abs(baseline.first_epoch_start - compared.first_epoch_start)
        > max(1e-6, baseline.epoch_length_sec * 0.01)
    ):
        return (
            f"First epoch mismatch: baseline={baseline.first_epoch_start} "
            f"compared={compared.first_epoch_start}"
        )
    return None


def invalid_alignment_result(
    baseline: LabelInventoryRow,
    compared: LabelInventoryRow,
    message: str,
) -> ComparisonResult:
    return ComparisonResult(
        recording_id=baseline.recording_id,
        baseline_source=baseline.label_source,
        compared_source=compared.label_source,
        baseline_model=baseline.model,
        compared_model=compared.model,
        baseline_path=baseline.path,
        compared_path=compared.path,
        epoch_count_baseline=baseline.epoch_count,
        epoch_count_compared=compared.epoch_count,
        aligned_epoch_count=0,
        strict_epoch_count=0,
        accuracy=None,
        balanced_accuracy=None,
        macro_f1=None,
        weighted_f1=None,
        kappa=None,
        disagreement_count=0,
        disagreement_pct=None,
        strict_accuracy=None,
        strict_balanced_accuracy=None,
        strict_macro_f1=None,
        status="invalid_alignment",
        message=message,
    )


def validate_alignment(
    baseline: LabelInventoryRow,
    compared: LabelInventoryRow,
    base: pd.DataFrame,
    other: pd.DataFrame,
) -> str | None:
    if len(base) != len(other):
        return f"Epoch count mismatch: baseline={len(base)} compared={len(other)}"
    base_epoch = _single_epoch_length(base)
    other_epoch = _single_epoch_length(other)
    if base_epoch is None or other_epoch is None:
        return "Missing epoch length"
    if abs(base_epoch - other_epoch) > 1e-6:
        return f"Epoch length mismatch: baseline={base_epoch} compared={other_epoch}"
    base_starts = pd.to_numeric(base["EpochStartSeconds"], errors="coerce").reset_index(drop=True)
    other_starts = pd.to_numeric(other["EpochStartSeconds"], errors="coerce").reset_index(drop=True)
    if len(base_starts) != len(other_starts):
        return "Epoch start count mismatch"
    max_delta = (base_starts - other_starts).abs().max()
    if pd.notna(max_delta) and max_delta > max(1e-6, base_epoch * 0.01):
        return f"Epoch start mismatch: max_delta={max_delta}"
    return None


def validate_predictions(predictions_path: str | Path, labels_path: str | Path, output_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Validate standardized predictions against user-provided epoch labels."""
    predictions = load_label_frame(predictions_path)
    labels = load_label_frame(labels_path)
    merged = predictions.merge(
        labels[["RecordingID", "EpochStartSeconds", "PredLabel"]].rename(columns={"PredLabel": "TruthLabel"}),
        on=["RecordingID", "EpochStartSeconds"], how="inner",
    )
    merged = merged[merged["TruthLabel"].isin(["Wake", "NREM", "REM"]) & merged["PredLabel"].isin(["Wake", "NREM", "REM"])].copy()
    values = classification_metrics(merged["TruthLabel"], merged["PredLabel"])
    summary = pd.DataFrame([{"AlignedEpochs": len(merged), **values}])
    confusion = pd.crosstab(merged["TruthLabel"], merged["PredLabel"], dropna=False)
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output / "validation_summary.csv", index=False)
    confusion.to_csv(output / "validation_confusion.csv")
    return summary, confusion


def _single_epoch_length(frame: pd.DataFrame) -> float | None:
    values = pd.to_numeric(frame["EpochLengthSeconds"], errors="coerce").dropna()
    if values.empty:
        return None
    return float(values.round(6).mode().iloc[0])


def classification_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float | None]:
    frame = pd.DataFrame({"true": y_true, "pred": y_pred}).dropna()
    if frame.empty:
        return {
            "accuracy": None,
            "balanced_accuracy": None,
            "macro_f1": None,
            "weighted_f1": None,
            "kappa": None,
        }
    labels = sorted(set(SCORED_STATES) | set(frame["true"].astype(str)) | set(frame["pred"].astype(str)))
    accuracy = float((frame["true"] == frame["pred"]).mean())
    per_label = per_label_metrics(frame["true"], frame["pred"], labels)
    recalls = [m["recall"] for m in per_label.values() if m["support"] > 0 and m["recall"] is not None]
    f1s = [m["f1"] for m in per_label.values() if m["support"] > 0 and m["f1"] is not None]
    total = sum(m["support"] for m in per_label.values())
    weighted = (
        sum((m["f1"] or 0.0) * m["support"] for m in per_label.values()) / total
        if total
        else None
    )
    return {
        "accuracy": accuracy,
        "balanced_accuracy": _mean(recalls),
        "macro_f1": _mean(f1s),
        "weighted_f1": weighted,
        "kappa": cohen_kappa(frame["true"], frame["pred"], labels),
    }


def per_label_metrics(
    y_true: pd.Series,
    y_pred: pd.Series,
    labels: list[str],
) -> dict[str, dict[str, float | int | None]]:
    metrics: dict[str, dict[str, float | int | None]] = {}
    for label in labels:
        true_pos = int(((y_true == label) & (y_pred == label)).sum())
        false_pos = int(((y_true != label) & (y_pred == label)).sum())
        false_neg = int(((y_true == label) & (y_pred != label)).sum())
        support = int((y_true == label).sum())
        precision = true_pos / (true_pos + false_pos) if true_pos + false_pos else None
        recall = true_pos / (true_pos + false_neg) if true_pos + false_neg else None
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision is not None and recall is not None and precision + recall
            else None
        )
        metrics[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
    return metrics


def cohen_kappa(y_true: pd.Series, y_pred: pd.Series, labels: list[str]) -> float | None:
    n = len(y_true)
    if n == 0:
        return None
    observed = float((y_true == y_pred).mean())
    expected = 0.0
    for label in labels:
        expected += float((y_true == label).mean()) * float((y_pred == label).mean())
    if expected == 1.0:
        return 1.0 if observed == 1.0 else None
    return (observed - expected) / (1.0 - expected)


def confusion_matrix(y_true: pd.Series, y_pred: pd.Series) -> pd.DataFrame:
    labels = sorted(set(SCORED_STATES) | set(y_true.astype(str)) | set(y_pred.astype(str)))
    matrix = pd.DataFrame(0, index=labels, columns=labels, dtype=int)
    for true, pred in zip(y_true.astype(str), y_pred.astype(str), strict=False):
        matrix.loc[true, pred] += 1
    matrix.index.name = "Baseline"
    matrix.columns.name = "Compared"
    return matrix


def build_difference_frame(
    baseline: LabelInventoryRow,
    compared: LabelInventoryRow,
    merged: pd.DataFrame,
) -> pd.DataFrame:
    merged = merged.copy()
    baseline_labels = merged["PredLabel_baseline"].reset_index(drop=True)
    near_transition = baseline_labels.ne(baseline_labels.shift(1)) | baseline_labels.ne(
        baseline_labels.shift(-1)
    )
    merged["NearBaselineTransition"] = near_transition.to_numpy()
    mismatch = merged[merged["PredLabel_baseline"] != merged["PredLabel_compared"]].copy()
    if mismatch.empty:
        return mismatch
    out = pd.DataFrame(
        {
            "RecordingID": mismatch["RecordingID"],
            "EpochIndex": mismatch["EpochIndex"],
            "EpochStartSeconds": mismatch["EpochStartSeconds_baseline"],
            "BaselineSource": baseline.label_source,
            "ComparedSource": compared.label_source,
            "BaselineModel": baseline.model,
            "ComparedModel": compared.model,
            "AccuSleePyLabel": mismatch["PredLabel_baseline"],
            "ComparedLabel": mismatch["PredLabel_compared"],
            "AccuSleePyConfidence": mismatch["Confidence_baseline"],
            "ComparedConfidence": mismatch["Confidence_compared"],
            "NearBaselineTransition": mismatch["NearBaselineTransition"],
            "BaselineMapping": baseline.label_mapping,
            "ComparedMapping": compared.label_mapping,
            "BaselinePath": baseline.path,
            "ComparedPath": compared.path,
        }
    )
    return out


def _missing_reference_alignment_row(baseline: LabelInventoryRow) -> dict[str, object]:
    return {
        "RecordingID": baseline.recording_id,
        "MarsEpochs": baseline.epoch_count,
        "ReferenceEpochs": 0,
        "AlignedEpochs": 0,
        "MissingEpochs": baseline.epoch_count,
        "CoveragePercent": 0.0,
        "StrictValidEpochs": 0,
        "Status": "missing_reference_source",
        "AlignmentMethod": "",
        "Message": "No reference source label file was found for this recording",
        "MarsPath": baseline.path,
        "ReferencePath": "",
        "MarsModel": baseline.model,
        "MarsEpochLengthSeconds": baseline.epoch_length_sec,
        "ReferenceEpochLengthSeconds": None,
    }


def _missing_reference_summary_row(baseline: LabelInventoryRow) -> dict[str, object]:
    return {
        "recording_id": baseline.recording_id,
        "mars_source": baseline.label_source,
        "reference_source": "reference_reference",
        "mars_model": baseline.model,
        "reference_model": "reference",
        "mars_path": baseline.path,
        "reference_path": "",
        "mars_epoch_count": baseline.epoch_count,
        "reference_epoch_count": 0,
        "aligned_epoch_count": 0,
        "strict_valid_epoch_count": 0,
        "coverage_percent": 0.0,
        "accuracy": None,
        "balanced_accuracy": None,
        "macro_f1": None,
        "weighted_f1": None,
        "kappa": None,
        "strict_accuracy": None,
        "strict_balanced_accuracy": None,
        "strict_macro_f1": None,
        "strict_weighted_f1": None,
        "strict_kappa": None,
        "disagreement_count": 0,
        "disagreement_pct": None,
        "status": "missing_reference_source",
        "message": "No reference source label file was found for this recording",
        "mars_mapping": baseline.label_mapping,
        "reference_mapping": label_mapping_name_for_reference(),
    }


def _no_overlap_summary_row(
    baseline: LabelInventoryRow,
    compared: LabelInventoryRow,
    baseline_count: int,
    compared_count: int,
) -> dict[str, object]:
    row = _missing_reference_summary_row(baseline)
    row.update(
        {
            "reference_path": compared.path,
            "reference_epoch_count": compared_count,
            "mars_epoch_count": baseline_count,
            "status": "no_time_overlap",
            "message": "reference labels and MARS epochs had no overlapping time centers",
            "reference_mapping": compared.label_mapping,
        }
    )
    return row


def _error_summary_row(baseline: LabelInventoryRow, compared: LabelInventoryRow, message: str) -> dict[str, object]:
    row = _missing_reference_summary_row(baseline)
    row.update(
        {
            "reference_path": compared.path,
            "reference_epoch_count": compared.epoch_count,
            "status": "corrupt_label_file",
            "message": message,
            "reference_mapping": compared.label_mapping,
        }
    )
    return row


def label_mapping_name_for_reference() -> str:
    return "reference_brain_state_v1_wake_rem_nrem"


def write_mars_vs_reference_outputs(
    output_dir: str | Path,
    summary: pd.DataFrame,
    differences: pd.DataFrame,
    confusion: pd.DataFrame,
    alignment: pd.DataFrame,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_dir / "mars_vs_reference_summary.csv", index=False)
    differences.to_csv(output_dir / "mars_vs_reference_differences.csv", index=False)
    confusion.to_csv(output_dir / "mars_vs_reference_confusion_matrix.csv", index=False)
    alignment.to_csv(output_dir / "reference_alignment_summary.csv", index=False)


def write_pdf_compatible_outputs(
    output_dir: str | Path,
    summary: pd.DataFrame,
    differences: pd.DataFrame,
    confusion: pd.DataFrame,
    alignment: pd.DataFrame,
    aggregate_summary: dict[str, object],
    expected_summary_path: str | Path | None = None,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_dir / "pdf_compat_per_recording_summary.csv", index=False)
    differences.to_csv(output_dir / "pdf_compat_differences.csv", index=False)
    confusion.to_csv(output_dir / "pdf_compat_confusion_matrix.csv", index=False)
    alignment.to_csv(output_dir / "pdf_compat_alignment_summary.csv", index=False)
    expected = _load_expected_summary(expected_summary_path) if expected_summary_path else {}
    if expected:
        aggregate_summary = dict(aggregate_summary)
        for key in ["TotalAlignedEpochs", "Accuracy", "BalancedAccuracy", "MacroF1", "WeightedF1", "CohenKappa"]:
            expected_value = expected.get(key)
            actual_value = aggregate_summary.get(key)
            aggregate_summary[f"Expected{key}"] = expected_value
            if isinstance(actual_value, (int, float)) and isinstance(expected_value, (int, float)):
                aggregate_summary[f"Delta{key}"] = actual_value - expected_value
    (output_dir / "pdf_compat_aggregate_summary.json").write_text(
        json_dumps(aggregate_summary),
        encoding="utf-8",
    )
    pd.DataFrame([aggregate_summary]).to_csv(output_dir / "pdf_compat_aggregate_summary.csv", index=False)


def _load_expected_summary(path: str | Path | None) -> dict[str, object]:
    if not path:
        return {}
    path = Path(path)
    if not path.exists():
        return {}
    if path.suffix.lower() == ".json":
        import json

        data = json.loads(path.read_text(encoding="utf-8-sig"))
        return data if isinstance(data, dict) else {}
    frame = pd.read_csv(path)
    if frame.empty:
        return {}
    return frame.iloc[0].to_dict()


def json_dumps(data: dict[str, object]) -> str:
    import json

    return json.dumps(data, indent=2, sort_keys=True) + "\n"


def write_comparison_outputs(
    output_dir: str | Path,
    summary: pd.DataFrame,
    differences: pd.DataFrame,
    matrices: dict[str, pd.DataFrame],
    coverage: pd.DataFrame | None = None,
) -> None:
    output_dir = Path(output_dir)
    matrix_dir = output_dir / "confusion_matrices"
    matrix_dir.mkdir(parents=True, exist_ok=True)
    for old_matrix in matrix_dir.glob("*.csv"):
        old_matrix.unlink()
    summary.to_csv(output_dir / "label_comparison_summary.csv", index=False)
    differences.to_csv(output_dir / "label_differences.csv", index=False)
    (coverage if coverage is not None else pd.DataFrame()).to_csv(
        output_dir / "comparison_coverage.csv",
        index=False,
    )
    for key, matrix in matrices.items():
        matrix.to_csv(matrix_dir / f"{sanitize_name(key)}.csv")


def sanitize_name(value: str) -> str:
    return re_sub_invalid(str(value))


def re_sub_invalid(value: str) -> str:
    out = []
    for char in value:
        out.append(char if char.isalnum() or char in "._-" else "_")
    return "".join(out).strip("_") or "unnamed"


def compute_comparison_coverage(
    rows: list[LabelInventoryRow],
    summary: pd.DataFrame,
) -> pd.DataFrame:
    baselines = choose_baselines(rows)
    baseline_total = int(sum(row.epoch_count for row in baselines.values()))
    baseline_recordings = len(baselines)
    out = [
        {
            "ComparedSource": "offline_accusleepy",
            "ComparedModel": "baseline",
            "LabelMapping": "accusleepy_v1_rem_wake_nrem",
            "BaselineRecordings": baseline_recordings,
            "ComparedRecordings": baseline_recordings,
            "BaselineEpochs": baseline_total,
            "ComparedEpochs": baseline_total,
            "AlignedEpochs": baseline_total,
            "MissingEpochs": 0,
            "CoveragePercent": 1.0,
            "ValidPairs": baseline_recordings,
            "InvalidPairs": 0,
            "DisagreementEpochs": 0,
            "Status": "baseline",
        }
    ]
    if summary.empty:
        return pd.DataFrame(out)
    frame = summary.copy()
    for column in [
        "epoch_count_compared",
        "aligned_epoch_count",
        "disagreement_count",
    ]:
        frame[column] = pd.to_numeric(frame.get(column, 0), errors="coerce").fillna(0)
    for (source, model), group in frame.groupby(["compared_source", "compared_model"], dropna=False):
        aligned = int(group["aligned_epoch_count"].sum())
        invalid_pairs = int((group["status"].astype(str) != "ok").sum())
        valid_pairs = int((group["status"].astype(str) == "ok").sum())
        coverage = float(aligned / baseline_total) if baseline_total else 0.0
        if coverage >= 0.999 and invalid_pairs == 0:
            status = "full_dataset"
        elif aligned > 0:
            status = "partial_reference"
        elif invalid_pairs:
            status = "invalid_alignment"
        else:
            status = "no_aligned_epochs"
        out.append(
            {
                "ComparedSource": source,
                "ComparedModel": model,
                "LabelMapping": _mapping_for_group(group),
                "BaselineRecordings": baseline_recordings,
                "ComparedRecordings": int(group["recording_id"].nunique()),
                "BaselineEpochs": baseline_total,
                "ComparedEpochs": int(group["epoch_count_compared"].sum()),
                "AlignedEpochs": aligned,
                "MissingEpochs": max(0, baseline_total - aligned),
                "CoveragePercent": coverage,
                "ValidPairs": valid_pairs,
                "InvalidPairs": invalid_pairs,
                "DisagreementEpochs": int(group["disagreement_count"].sum()),
                "Status": status,
            }
        )
    return pd.DataFrame(out)


def _comparison_dict(result: ComparisonResult, baseline: LabelInventoryRow, compared: LabelInventoryRow) -> dict[str, object]:
    data = result.to_dict()
    data["baseline_mapping"] = baseline.label_mapping
    data["compared_mapping"] = compared.label_mapping
    return data


def _mapping_for_group(group: pd.DataFrame) -> str:
    if "compared_mapping" not in group.columns:
        return ""
    values = [str(value) for value in group["compared_mapping"].dropna().unique() if str(value)]
    return ";".join(sorted(values))


def _baseline_priority(row: LabelInventoryRow) -> tuple[int, int, str]:
    format_rank = 0 if row.format == "mars_standard" else 1
    source_rank = 0 if row.label_source == "offline_accusleepy" else 9
    return (source_rank, format_rank, row.path)


def _representative_priority(row: LabelInventoryRow) -> tuple[int, int, int, str]:
    text = row.path.lower()
    format_rank = 0 if row.format == "mars_standard" else 1
    curated_rank = 0
    if "standardized_predictions" in text:
        curated_rank = 0
    elif "mars_visualizer" in text:
        curated_rank = 1
    elif "03_model_outputs" in text:
        curated_rank = 2
    elif "output (" in text:
        curated_rank = 3
    elif "cpu benchmarks" in text or "archive" in text:
        curated_rank = 8
    else:
        curated_rank = 5
    return (format_rank, curated_rank, -row.epoch_count, row.path)


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None

