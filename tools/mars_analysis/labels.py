from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
from functools import lru_cache
import re
from pathlib import Path
from typing import Iterable

import pandas as pd

from .hashing import sha256_file


STANDARD_COLUMNS = [
    "RecordingID",
    "Model",
    "EDFFile",
    "EpochIndex",
    "EpochStartSeconds",
    "EpochLengthSeconds",
    "RawCode",
    "PredLabel",
    "Confidence",
]

DIGIT_TO_LABEL = {
    -1: "Unknown",
    0: "TransitionalOrUnclassified",
    1: "REM",
    2: "Wake",
    3: "NREM",
}
reference_DIGIT_TO_LABEL = {
    -1: "Unknown",
    0: "TransitionalOrUnclassified",
    1: "Wake",
    2: "REM",
    3: "NREM",
}
LABEL_TO_DIGIT = {label: digit for digit, label in DIGIT_TO_LABEL.items()}


@dataclass(slots=True)
class LabelInventoryRow:
    recording_id: str
    label_source: str
    model: str
    path: str
    format: str
    epoch_count: int
    epoch_length_sec: float | None
    first_epoch_start: float | None
    last_epoch_start: float | None
    file_sha256: str
    status: str
    message: str = ""
    label_mapping: str = ""

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def discover_label_files(search_roots: Iterable[str | Path]) -> list[Path]:
    matches: list[Path] = []
    suffixes = ("predictions_standardized.csv",)
    for root in search_roots:
        root = Path(root)
        if not root.exists():
            continue
        for path in _walk_files(root):
            name = path.name.lower()
            if (
                name.endswith(suffixes)
                or "prediction" in name and name.endswith(".csv")
                or "label" in name and name.endswith(".csv")
            ):
                matches.append(path)
    return sorted(set(matches), key=lambda p: str(p).lower())


def _walk_files(root: Path) -> Iterable[Path]:
    stack = [root]
    while stack:
        current = stack.pop()
        try:
            children = list(current.iterdir())
        except OSError:
            continue
        for child in children:
            if child.is_dir():
                stack.append(child)
            elif child.is_file():
                yield child


def inventory_label_files(files: Iterable[str | Path]) -> list[LabelInventoryRow]:
    rows: list[LabelInventoryRow] = []
    for file_path in files:
        path = Path(file_path)
        try:
            rows.extend(_inventory_one_file(path))
        except Exception as exc:  # noqa: BLE001 - keep inventory robust
            rows.append(
                LabelInventoryRow(
                    recording_id=infer_recording_id(path),
                    label_source=infer_label_source(path),
                    model=infer_model(path),
                    path=str(path),
                    format="unknown",
                    epoch_count=0,
                    epoch_length_sec=None,
                    first_epoch_start=None,
                    last_epoch_start=None,
                    file_sha256="",
                    status="error",
                    message=str(exc),
                    label_mapping=label_mapping_name(infer_label_source(path)),
                )
            )
    return rows


def _inventory_one_file(path: Path) -> list[LabelInventoryRow]:
    header = read_header(path)
    digest = sha256_file(path)
    if set(STANDARD_COLUMNS).issubset(header):
        frame = pd.read_csv(path, usecols=STANDARD_COLUMNS)
        rows = []
        for recording_id, group in frame.groupby("RecordingID", dropna=False):
            epoch_lengths = pd.to_numeric(group["EpochLengthSeconds"], errors="coerce")
            starts = pd.to_numeric(group["EpochStartSeconds"], errors="coerce")
            model = str(group["Model"].dropna().iloc[0]) if group["Model"].notna().any() else infer_model(path)
            rows.append(
                LabelInventoryRow(
                    recording_id=str(recording_id),
                    label_source=infer_label_source(path, model=model),
                    model=model,
                    path=str(path),
                    format="mars_standard",
                    epoch_count=len(group),
                    epoch_length_sec=_mode_float(epoch_lengths),
                    first_epoch_start=_safe_min(starts),
                    last_epoch_start=_safe_max(starts),
                    file_sha256=digest,
                    status="ok",
                    label_mapping=label_mapping_name(infer_label_source(path, model=model)),
                )
            )
        return rows
    if "brain_state" in header:
        frame = pd.read_csv(path)
        label_source = infer_label_source(path)
        model = "reference" if label_source == "reference_reference" else infer_model(path)
        starts = (
            pd.to_numeric(frame["epoch_start_seconds"], errors="coerce")
            if "epoch_start_seconds" in frame.columns
            else pd.Series(range(len(frame)), dtype="float64") * infer_epoch_length(path)
        )
        return [
                LabelInventoryRow(
                    recording_id=infer_recording_id(path),
                    label_source=label_source,
                    model=model,
                    path=str(path),
                    format="accusleepy_brain_state",
                    epoch_count=len(frame),
                    epoch_length_sec=infer_epoch_length(path, starts),
                    first_epoch_start=_safe_min(starts),
                    last_epoch_start=_safe_max(starts),
                    file_sha256=digest,
                    status="ok",
                    label_mapping="accusleepy_v1_rem_wake_nrem",
                )
            ]
    if {"RecordingID", "EpochStartSeconds", "EpochLengthSeconds"}.issubset(header) and (
        "ReferenceStateCode" in header or "ReferenceStateLabel" in header
    ):
        frame = pd.read_csv(path)
        rows = []
        for recording_id, group in frame.groupby("RecordingID", dropna=False):
            starts = pd.to_numeric(group["EpochStartSeconds"], errors="coerce")
            epoch_lengths = pd.to_numeric(group["EpochLengthSeconds"], errors="coerce")
            rows.append(
                LabelInventoryRow(
                    recording_id=str(recording_id),
                    label_source="reference_reference",
                    model="reference",
                    path=str(path),
                    format="reference_reference_epoch_labels",
                    epoch_count=len(group),
                    epoch_length_sec=_mode_float(epoch_lengths),
                    first_epoch_start=_safe_min(starts),
                    last_epoch_start=_safe_max(starts),
                    file_sha256=digest,
                    status="ok",
                    label_mapping=label_mapping_name("reference_reference"),
                )
            )
        return rows
    raise ValueError(f"Unsupported label CSV columns: {', '.join(header)}")


def read_header(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        return next(reader)


def load_label_frame(
    path: str | Path,
    *,
    recording_id: str | None = None,
    default_epoch_length: float = 2.5,
) -> pd.DataFrame:
    path = Path(path)
    header = read_header(path)
    if set(STANDARD_COLUMNS).issubset(header):
        cached = _read_csv_cached(str(path), path.stat().st_mtime_ns)
        if recording_id is not None:
            frame = _filter_recording(cached, recording_id).copy()
        else:
            frame = cached.copy()
        return normalize_standard_frame(frame)
    if "brain_state" in header:
        frame = _read_csv_cached(str(path), path.stat().st_mtime_ns).copy()
        label_source = infer_label_source(path)
        return standardize_brain_state_frame(
            frame,
            recording_id=recording_id or infer_recording_id(path),
            model="reference" if label_source == "reference_reference" else infer_model(path),
            source_path=path,
            default_epoch_length=default_epoch_length,
        )
    if {"RecordingID", "EpochStartSeconds", "EpochLengthSeconds"}.issubset(header) and (
        "ReferenceStateCode" in header or "ReferenceStateLabel" in header
    ):
        cached = _read_csv_cached(str(path), path.stat().st_mtime_ns)
        if recording_id is not None:
            frame = _filter_recording(cached, recording_id).copy()
        else:
            frame = cached.copy()
        return standardize_reference_reference_frame(frame, source_path=path)
    raise ValueError(f"Unsupported label CSV: {path}")


@lru_cache(maxsize=128)
def _read_csv_cached(path: str, _mtime_ns: int) -> pd.DataFrame:
    return pd.read_csv(path)


def _filter_recording(frame: pd.DataFrame, recording_id: str) -> pd.DataFrame:
    recording_id = str(recording_id)
    mask = frame["RecordingID"] == recording_id
    if mask.any():
        return frame[mask]
    return frame[frame["RecordingID"].astype(str) == recording_id]


def normalize_standard_frame(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    for column in STANDARD_COLUMNS:
        if column not in frame.columns:
            frame[column] = None
    frame["PredLabel"] = frame["PredLabel"].map(normalize_label)
    frame["EpochIndex"] = pd.to_numeric(frame["EpochIndex"], errors="coerce").astype("Int64")
    frame["EpochStartSeconds"] = pd.to_numeric(frame["EpochStartSeconds"], errors="coerce")
    frame["EpochLengthSeconds"] = pd.to_numeric(frame["EpochLengthSeconds"], errors="coerce")
    frame["Confidence"] = pd.to_numeric(frame["Confidence"], errors="coerce")
    return frame[STANDARD_COLUMNS]


def standardize_brain_state_frame(
    frame: pd.DataFrame,
    *,
    recording_id: str,
    model: str,
    source_path: str | Path,
    default_epoch_length: float,
) -> pd.DataFrame:
    label_source = infer_label_source(source_path, model=model)
    # ``brain_state`` files use AccuSleePy's numeric convention even when the
    # labels were derived from reference source truth: 1=REM, 2=Wake, 3=NREM.
    # Raw reference epoch exports are handled separately by
    # standardize_reference_reference_frame and keep reference's own code mapping.
    labels = frame["brain_state"].map(lambda value: normalize_label(value, label_source="offline_accusleepy"))
    raw_codes = pd.to_numeric(frame["brain_state"], errors="coerce")
    if "epoch_start_seconds" in frame.columns:
        starts = pd.to_numeric(frame["epoch_start_seconds"], errors="coerce")
        epoch_length = infer_epoch_length(Path(source_path), starts, default_epoch_length)
    else:
        epoch_length = default_epoch_length
        starts = pd.Series(range(len(frame)), dtype="float64") * epoch_length
    confidence = (
        pd.to_numeric(frame["confidence_score"], errors="coerce")
        if "confidence_score" in frame.columns
        else pd.Series([None] * len(frame), dtype="float64")
    )
    return pd.DataFrame(
        {
            "RecordingID": recording_id,
            "Model": model,
            "EDFFile": "",
            "EpochIndex": range(1, len(frame) + 1),
            "EpochStartSeconds": starts,
            "EpochLengthSeconds": epoch_length,
            "RawCode": raw_codes,
            "PredLabel": labels,
            "Confidence": confidence,
        },
        columns=STANDARD_COLUMNS,
    )


def standardize_reference_reference_frame(frame: pd.DataFrame, *, source_path: str | Path) -> pd.DataFrame:
    frame = frame.copy()
    raw_codes = (
        pd.to_numeric(frame["ReferenceStateCode"], errors="coerce")
        if "ReferenceStateCode" in frame.columns
        else pd.Series([None] * len(frame), dtype="float64")
    )
    if "ReferenceStateLabel" in frame.columns:
        labels = frame["ReferenceStateLabel"].map(lambda value: normalize_label(value, label_source="reference_reference"))
    else:
        labels = raw_codes.map(lambda value: normalize_label(value, label_source="reference_reference"))
    return pd.DataFrame(
        {
            "RecordingID": frame["RecordingID"].astype(str),
            "Model": "reference",
            "EDFFile": frame["EDFFile"].astype(str) if "EDFFile" in frame.columns else "",
            "EpochIndex": pd.to_numeric(frame["EpochIndex"], errors="coerce") if "EpochIndex" in frame.columns else range(1, len(frame) + 1),
            "EpochStartSeconds": pd.to_numeric(frame["EpochStartSeconds"], errors="coerce"),
            "EpochLengthSeconds": pd.to_numeric(frame["EpochLengthSeconds"], errors="coerce"),
            "RawCode": raw_codes,
            "PredLabel": labels,
            "Confidence": pd.Series([None] * len(frame), dtype="float64"),
        },
        columns=STANDARD_COLUMNS,
    )


def write_mars_standard_predictions(frame: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    normalize_standard_frame(frame).to_csv(path, index=False)


def infer_recording_id(path: str | Path) -> str:
    stem = Path(path).stem
    suffixes = [
        "_reference_accusleepy_calibration_labels",
        "_accusleepy_labels",
        "_reference_aligned_labels",
        "_legacy_reference_labels",
        "_predictions_standardized",
        "_predictions",
        "_labels",
    ]
    for suffix in suffixes:
        if stem.lower().endswith(suffix.lower()):
            stem = stem[: -len(suffix)]
            break
    stem = re.sub(r"^(accusleepy|rest|intellisleep)_", "", stem, flags=re.I)
    return stem


def infer_label_source(path: str | Path, *, model: str | None = None) -> str:
    text = str(path).lower()
    model_text = (model or "").lower()
    if "reference" in text and "calibration_labels" in text:
        return "reference_reference"
    if "calibration_labels" in text:
        return "offline_accusleepy_calibration"
    if "offline_accusleepy" in text or "accusleepy_scored" in text or "accusleepy_standardized" in text:
        return "offline_accusleepy"
    if "reference" in text:
        return "reference_reference"
    if "rest" in text or "rest" in model_text:
        return "rest"
    if "intellisleep" in text or "intellisleep" in model_text:
        return "intellisleep"
    if "mars_openephys" in text or "mars_open_ephys" in text or "e2.0w" in model_text or "e2.5w" in model_text:
        return "real_time_mars"
    if "accusleepy" in text or "accusleepy" in model_text:
        return "offline_accusleepy"
    return "unknown"


def infer_model(path: str | Path) -> str:
    text = str(path).lower()
    if "rest" in text:
        return "REST"
    if "intellisleep" in text:
        return "IntelliSleepScorer"
    if "e2p0w3" in text or "e2.0w3" in text:
        return "E2.0W3"
    if "e2p5w9" in text or "e2.5w9" in text:
        return "E2.5W9"
    if "accusleepy" in text:
        return "AccuSleePy"
    return Path(path).stem


def infer_epoch_length(
    path: str | Path,
    starts: pd.Series | None = None,
    default: float = 2.5,
) -> float:
    text = str(path).lower()
    if "rest" in text:
        return 4.0
    if starts is not None:
        diffs = pd.to_numeric(starts, errors="coerce").diff().dropna()
        diffs = diffs[diffs > 0]
        if not diffs.empty:
            return float(diffs.round(6).mode().iloc[0])
    return default


def normalize_label(value: object, *, label_source: str | None = None) -> str:
    if pd.isna(value):
        return "Unknown"
    if isinstance(value, str):
        stripped = value.strip()
        if stripped == "":
            return "Unknown"
        lower = stripped.lower()
        aliases = {
            "wake": "Wake",
            "w": "Wake",
            "nrem": "NREM",
            "non-rem": "NREM",
            "nonrem": "NREM",
            "rem": "REM",
            "transitionalorunclassified": "TransitionalOrUnclassified",
            "transitional": "TransitionalOrUnclassified",
            "unclassified": "TransitionalOrUnclassified",
            "unknown": "Unknown",
        }
        if lower in aliases:
            return aliases[lower]
        try:
            return digit_to_label_map(label_source).get(int(float(stripped)), stripped)
        except ValueError:
            return stripped
    try:
        return digit_to_label_map(label_source).get(int(value), str(value))
    except (TypeError, ValueError):
        return str(value)


def digit_to_label_map(label_source: str | None = None) -> dict[int, str]:
    if str(label_source or "").lower() == "reference_reference":
        return reference_DIGIT_TO_LABEL
    return DIGIT_TO_LABEL


def label_mapping_name(label_source: str | None = None) -> str:
    return "reference_brain_state_v1_wake_rem_nrem" if str(label_source or "").lower() == "reference_reference" else "accusleepy_v1_rem_wake_nrem"


def _mode_float(series: pd.Series) -> float | None:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return None
    return float(values.round(6).mode().iloc[0])


def _safe_min(series: pd.Series) -> float | None:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return None if values.empty else float(values.min())


def _safe_max(series: pd.Series) -> float | None:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return None if values.empty else float(values.max())

