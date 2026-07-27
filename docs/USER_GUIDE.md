# User Guide

## Offline EDF Reviewer

Dataset Status is a dataset-level health summary. `Configured recordings`
shows how many entries are in the analysis config, `Scored recordings` shows
how many completed successfully, and Classification Status groups recordings
by result such as `ok`. It does not repeat every recording as a separate row.

The Recordings list in the left sidebar contains all configured recordings.
Select one to view its Recording Overview, Classification, Trace QC,
Spectrogram, Power Spectra, and Epoch Inspector results.

Before using Group Analysis, populate `animal_id`, `condition`, `dose`, and
`session` for every recording in the analysis config. The EDF tree config
builder detects signal channels but does not infer this experimental metadata
from paths or filenames. Blank metadata causes recordings to be pooled into an
unnamed ` /  / ` group; it does not indicate a scoring failure.

## Single-Animal Scoring

Scoring-only chain:

```text
Acquisition Board -> MARS Sleep Scorer -> Splitter -> LFP Viewer
```

Use scoring-only mode first. Select the EEG channel, EMG channel, model, input scale, and calibration. Leave `Output enabled` off until scoring and logging are verified. In scoring-only mode, MARS writes predictions and timing files but does not send hardware stimulation pulses.

Hardware TTL chain:

```text
Acquisition Board -> MARS Sleep Scorer -> Splitter -> LFP Viewer
                                             -> TTL Display Panel
                                             -> Acq Board Output
```

Turn `Output enabled` on only after the manual TTL route works.

## Multi-Animal Scoring

Chain:

```text
Acquisition Board -> MARS Multi Sleep Scorer -> Splitter -> LFP Viewer
```

The multi scorer supports 20 slots. Each slot has an enable toggle, animal ID, EEG channel, EMG channel, model, TTL line, status, counters, and run folder. Slots A1-A8 default to physical TTL lines 1-8. A9-A20 default to line 0 and score/log without hardware output.

## Run Files

Each animal run folder writes:

```text
predictions_standardized.csv
epoch_timing.csv
stim_decisions.csv
summary.json
```

Multi sessions also write:

```text
multi_session_summary.json
```

## State Labels

MARS writes standardized sleep labels:

- `Wake`
- `NREM`
- `REM`

TTL stimulation policy is based on qualifying NREM epochs, confidence threshold, required consecutive epochs, cooldown, and hardware enable state.
