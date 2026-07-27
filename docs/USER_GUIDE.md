# User Guide

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
