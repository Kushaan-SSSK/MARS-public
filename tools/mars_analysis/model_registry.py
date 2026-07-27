from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

from .config import RecordingSpec


REPO_ROOT = Path(__file__).resolve().parents[2]
MARS_RESOURCES = REPO_ROOT / "plugin" / "MARSSleepScorer" / "Resources"
if not MARS_RESOURCES.exists():
    MARS_RESOURCES = REPO_ROOT / "MARSSleepScorer" / "Resources"
MODEL_REGISTRY_PATH = MARS_RESOURCES / "model_registry.json"
RUNTIME_CALIBRATIONS = REPO_ROOT.parent / "runtime_calibrations"
DEFAULT_CALIBRATIONS = MARS_RESOURCES / "calibration" / "default"


@dataclass(frozen=True, slots=True)
class ResolvedModel:
    alias: str
    model_file: Path
    calibration_file: Path | None
    epoch_length_sec: float
    input_windows: int
    feature_window_sec: float | None
    feature_mode: str
    registry_entry: dict[str, object]

    @property
    def scoreable(self) -> bool:
        return self.model_file.exists() and self.calibration_file is not None and self.calibration_file.exists()


def load_model_registry(path: str | Path = MODEL_REGISTRY_PATH) -> dict[str, object]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def resolve_model(
    alias: str,
    *,
    recording: RecordingSpec | None = None,
    registry_path: str | Path = MODEL_REGISTRY_PATH,
) -> ResolvedModel:
    registry_path = Path(registry_path)
    registry = load_model_registry(registry_path)
    aliases = registry.get("model_aliases", {})
    if alias not in aliases:
        available = ", ".join(sorted(aliases))
        raise KeyError(f"Unknown MARS model alias {alias!r}; available: {available}")
    entry = dict(aliases[alias])
    resource_root = registry_path.parent.parent
    model_file = (
        Path(recording.accusleepy_model_file)
        if recording and recording.accusleepy_model_file
        else _resolve_resource_path(resource_root, entry.get("pytorch_path"))
    )
    calibration_file = resolve_calibration(alias, recording=recording)
    return ResolvedModel(
        alias=alias,
        model_file=model_file,
        calibration_file=calibration_file,
        epoch_length_sec=float(entry.get("epoch_length_sec") or 2.5),
        input_windows=int(entry.get("input_windows") or 1),
        feature_window_sec=_optional_float(entry.get("feature_window_sec")),
        feature_mode=str(entry.get("feature_mode") or "default"),
        registry_entry=entry,
    )


def resolve_calibration(alias: str, *, recording: RecordingSpec | None = None) -> Path | None:
    if recording and recording.accusleepy_calibration_file:
        explicit = Path(recording.accusleepy_calibration_file)
        if explicit.exists():
            return explicit

    animal_id = (recording.animal_id if recording else None) or ""
    recording_id = (recording.recording_id if recording else None) or ""
    search_terms = [term.lower() for term in [recording_id, animal_id] if term]
    candidates: list[Path] = []
    for root in [RUNTIME_CALIBRATIONS, DEFAULT_CALIBRATIONS]:
        if root.exists():
            candidates.extend(sorted(root.rglob(f"*{alias}*calibration*.csv")))

    for path in candidates:
        text = str(path).lower()
        if search_terms and any(term in text for term in search_terms):
            return path

    default_name = f"MARS_default_{alias}_runtime_calibration.csv"
    default_path = DEFAULT_CALIBRATIONS / default_name
    if default_path.exists():
        return default_path

    return candidates[0] if candidates else None


def _resolve_resource_path(resource_root: Path, raw_path: object) -> Path:
    path = Path(str(raw_path or ""))
    if path.is_absolute():
        return path
    return resource_root / path


def _optional_float(value: object) -> float | None:
    if value in (None, ""):
        return None
    return float(value)
