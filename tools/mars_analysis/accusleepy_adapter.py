from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import RecordingSpec
from .labels import write_mars_standard_predictions


def score_recording_to_mars_bundle(
    recording: RecordingSpec,
    output_dir: str | Path,
    *,
    model_alias: str,
) -> Path:
    """Run AccuSleePy scoring and write a MARS-compatible run folder.

    This adapter expects AccuSleePy-ready CSV/parquet input, not raw EDF. EDF to
    AccuSleePy parquet conversion remains an upstream data-preparation step.
    """
    missing = [
        name
        for name, value in [
            ("accusleepy_recording_file", recording.accusleepy_recording_file),
            ("accusleepy_model_file", recording.accusleepy_model_file),
            ("accusleepy_calibration_file", recording.accusleepy_calibration_file),
            ("sampling_rate", recording.sampling_rate),
        ]
        if value in (None, "")
    ]
    if missing:
        raise ValueError(
            f"Cannot score {recording.recording_id}; missing {', '.join(missing)}"
        )

    try:
        from accusleepy.brain_state_set import BrainState, BrainStateSet
        from accusleepy.classification import score_recording
        import accusleepy.constants as c
        from accusleepy.fileio import load_calibration_file, load_config, load_recording
        from accusleepy.models import load_model
        from accusleepy.signal_processing import resample_and_standardize
    except ImportError as exc:
        raise RuntimeError(
            "AccuSleePy is required to generate missing labels. Run this command "
            "from an environment where the AccuSleePy package is importable."
        ) from exc

    model, epoch_length, epochs_per_img, _model_type, brain_states = load_model(
        recording.accusleepy_model_file
    )
    brain_state_set = BrainStateSet(
        [BrainState(**state) for state in brain_states],
        c.UNDEFINED_LABEL,
    )
    config = load_config()
    eeg, emg = _load_recording_input(recording.accusleepy_recording_file, load_recording)
    eeg, emg, sampling_rate = resample_and_standardize(
        eeg=eeg,
        emg=emg,
        sampling_rate=float(recording.sampling_rate),
        epoch_length=float(epoch_length),
    )
    mixture_means, mixture_sds = load_calibration_file(recording.accusleepy_calibration_file)
    labels, confidence = score_recording(
        model=model,
        eeg=eeg,
        emg=emg,
        mixture_means=mixture_means,
        mixture_sds=mixture_sds,
        sampling_rate=float(sampling_rate),
        epoch_length=float(epoch_length),
        epochs_per_img=int(epochs_per_img),
        brain_state_set=brain_state_set,
        emg_filter=config.emg_filter,
    )

    starts = pd.Series(range(len(labels)), dtype="float64") * float(epoch_length)
    frame = pd.DataFrame(
        {
            "RecordingID": recording.recording_id,
            "Model": model_alias,
            "EDFFile": recording.edf_path or "",
            "EpochIndex": range(1, len(labels) + 1),
            "EpochStartSeconds": starts,
            "EpochLengthSeconds": float(epoch_length),
            "RawCode": labels,
            "PredLabel": labels,
            "Confidence": confidence,
        }
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "predictions_standardized.csv"
    write_mars_standard_predictions(frame, predictions_path)
    pd.DataFrame(columns=["EpochIndex", "EpochStartSeconds", "InferenceMs", "MissedDeadline"]).to_csv(
        output_dir / "epoch_timing.csv",
        index=False,
    )
    pd.DataFrame(columns=["EpochIndex", "Decision", "Reason"]).to_csv(
        output_dir / "stim_decisions.csv",
        index=False,
    )
    summary = {
        "RunName": f"{recording.recording_id}_{model_alias}_offline_accusleepy",
        "CreatedUTC": datetime.now(timezone.utc).isoformat(),
        "ScoringApplication": "MARS Analysis Offline AccuSleePy Adapter",
        "BridgeMode": "offline-accusleepy-scoring",
        "ModelFile": recording.accusleepy_model_file,
        "ModelProfile": model_alias,
        "ModelAlias": model_alias,
        "EpochLengthSeconds": float(epoch_length),
        "EpochsPerImage": int(epochs_per_img),
        "FeatureWindowSeconds": recording.feature_window_sec,
        "FeatureMode": recording.feature_mode,
        "ProcessingSampleRate": float(recording.sampling_rate),
        "StandardizedSampleRate": float(sampling_rate),
        "CalibrationFile": recording.accusleepy_calibration_file,
        "OutputDir": str(output_dir),
        "AggregateScoredEpochs": len(labels),
        "Outputs": {
            "Predictions": str(predictions_path),
            "Timing": str(output_dir / "epoch_timing.csv"),
            "StimDecisions": str(output_dir / "stim_decisions.csv"),
            "Summary": str(output_dir / "summary.json"),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_dir / "classification_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "created_utc": summary["CreatedUTC"],
                "status": "ok",
                "message": "",
                "model_alias": model_alias,
                "model_file": recording.accusleepy_model_file,
                "calibration_file": recording.accusleepy_calibration_file,
                "predictions_standardized": str(predictions_path),
                "summary": str(output_dir / "summary.json"),
                "epoch_length_seconds": float(epoch_length),
                "epochs_per_image": int(epochs_per_img),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return predictions_path


def _load_recording_input(recording_file: str | Path, accusleepy_load_recording):
    path = Path(recording_file)
    if path.name == "accusleepy_recording_manifest.json":
        manifest = json.loads(path.read_text(encoding="utf-8"))
        eeg = np.load(manifest["eeg_npy"], mmap_mode="r")
        emg = np.load(manifest["emg_npy"], mmap_mode="r")
        sample_count = min(len(eeg), len(emg), int(manifest.get("sample_count") or min(len(eeg), len(emg))))
        return eeg[:sample_count], emg[:sample_count]
    return accusleepy_load_recording(path)
