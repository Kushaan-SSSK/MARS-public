from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import os
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = Path(os.environ.get("MARS_PROJECT_ROOT", REPO_ROOT))
DEFAULT_USER_ROOT = Path.home() / "Documents" / "MARS"
DEFAULT_DATASET_ROOT = DEFAULT_USER_ROOT / "Datasets"
DEFAULT_OUTPUT_ROOT = DEFAULT_USER_ROOT / "AnalysisOutputs"


@dataclass(slots=True)
class ChannelMap:
    eeg: str | int | None = None
    emg: str | int | None = None
    eeg_2: str | int | None = None


@dataclass(slots=True)
class RecordingSpec:
    recording_id: str
    edf_path: str | None = None
    metadata_json: str | None = None
    animal_id: str | None = None
    condition: str | None = None
    dose: str | None = None
    session: str | None = None
    epoch_length_sec: float = 2.5
    start_zt: str | None = None
    channel_map: ChannelMap = field(default_factory=ChannelMap)
    accusleepy_recording_file: str | None = None
    accusleepy_model_file: str | None = None
    accusleepy_calibration_file: str | None = None
    sampling_rate: float | None = None
    feature_window_sec: float | None = None
    feature_mode: str = "default"


@dataclass(slots=True)
class VisualizerConfig:
    eeg_channel: str | int | None = None
    emg_channel: str | int | None = None
    spectrogram_min_hz: float = 0.0
    spectrogram_max_hz: float = 20.0
    hour_bin_size: float = 1.0
    trace_bin_seconds: float = 5.0
    spectrogram_chunk_seconds: float = 1800.0
    psd_chunk_seconds: float = 1800.0
    zoom_epochs: list[int] = field(default_factory=list)
    psd_max_faint_curves_per_state: int = 2
    psd_show_sem_band: bool = True
    dashboard_refresh_seconds: int = 10
    state_colors: dict[str, str] = field(
        default_factory=lambda: {"Wake": "#d62728", "NREM": "#1f77b4", "REM": "#2ca02c", "Unknown": "#7f7f7f"}
    )


@dataclass(slots=True)
class CachePolicy:
    skip_existing: bool = True
    hash_large_files_by_stat: bool = True


@dataclass(slots=True)
class AnalysisConfig:
    dataset_id: str = "current_dataset"
    dataset_root: str = str(DEFAULT_DATASET_ROOT)
    output_root: str = str(DEFAULT_OUTPUT_ROOT)
    default_model_alias: str = "E2.5W9"
    recordings: list[RecordingSpec] = field(default_factory=list)
    label_roots: list[str] = field(default_factory=list)
    label_sources: list[str] = field(default_factory=lambda: ["offline_accusleepy", "real_time_mars", "user_labels", "unknown"])
    visualizer: VisualizerConfig = field(default_factory=VisualizerConfig)
    cache_policy: CachePolicy = field(default_factory=CachePolicy)

    @classmethod
    def from_json(cls, path: str | Path) -> "AnalysisConfig":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8-sig")))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AnalysisConfig":
        data = _expand_config_paths(dict(data))
        recordings = []
        for raw in data.get("recordings", []):
            item = dict(raw)
            item["channel_map"] = ChannelMap(**item.get("channel_map", {}))
            recordings.append(RecordingSpec(**item))
        allowed = set(cls.__dataclass_fields__)
        clean = {key: value for key, value in data.items() if key in allowed}
        clean["recordings"] = recordings
        if isinstance(clean.get("visualizer"), dict):
            visual_allowed = set(VisualizerConfig.__dataclass_fields__)
            clean["visualizer"] = VisualizerConfig(**{key: value for key, value in clean["visualizer"].items() if key in visual_allowed})
        if isinstance(clean.get("cache_policy"), dict):
            clean["cache_policy"] = CachePolicy(**clean["cache_policy"])
        return cls(**clean)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8")

    @property
    def dataset_output_dir(self) -> Path:
        return Path(self.output_root) / self.dataset_id


def default_config() -> AnalysisConfig:
    return AnalysisConfig()


def expand_path_text(value: str | None) -> str | None:
    if value in (None, ""):
        return value
    return os.path.expanduser(os.path.expandvars(str(value).replace("{repo}", str(REPO_ROOT))))


def _expand_config_paths(data: dict[str, Any]) -> dict[str, Any]:
    for key in ["dataset_root", "output_root"]:
        if isinstance(data.get(key), str):
            data[key] = expand_path_text(data[key])
    if isinstance(data.get("label_roots"), list):
        data["label_roots"] = [expand_path_text(value) if isinstance(value, str) else value for value in data["label_roots"]]
    for item in data.get("recordings", []):
        if not isinstance(item, dict):
            continue
        for key in ["edf_path", "metadata_json", "accusleepy_recording_file", "accusleepy_model_file", "accusleepy_calibration_file"]:
            if isinstance(item.get(key), str):
                item[key] = expand_path_text(item[key])
    return data
