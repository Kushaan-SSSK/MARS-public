from __future__ import annotations

from pathlib import Path

from .config import AnalysisConfig, ChannelMap, RecordingSpec
from .edf_stream import read_edf_header


def _first_matching(labels: list[str], token: str) -> str | None:
    token = token.casefold()
    return next((label for label in labels if token in label.casefold()), None)


def build_config_from_edf_tree(
    edf_root: str | Path,
    *,
    output_path: str | Path,
    output_root: str | Path | None = None,
    limit: int | None = None,
    dataset_id: str = "edf_local",
) -> AnalysisConfig:
    paths = sorted(Path(edf_root).rglob("*.edf"))
    if limit is not None:
        paths = paths[:limit]
    records = []
    for path in paths:
        header = read_edf_header(path)
        labels = list(header.labels)
        eeg_labels = [label for label in labels if "eeg" in label.casefold()]
        eeg = eeg_labels[0] if eeg_labels else (labels[0] if labels else None)
        eeg_2 = eeg_labels[1] if len(eeg_labels) > 1 else None
        emg = _first_matching(labels, "emg")
        if emg is None:
            emg = next((label for label in labels if label != eeg and label != eeg_2), None)
        records.append(
            RecordingSpec(
                recording_id=path.stem,
                edf_path=str(path),
                channel_map=ChannelMap(eeg=eeg, eeg_2=eeg_2, emg=emg),
            )
        )
    cfg = AnalysisConfig(
        dataset_id=dataset_id,
        dataset_root=str(Path(edf_root)),
        output_root=str(Path(output_root)) if output_root else AnalysisConfig().output_root,
        recordings=records,
    )
    cfg.save_json(output_path)
    return cfg
