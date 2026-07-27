# Hardware TTL

Hardware stimulation is guarded by explicit GUI controls and acquisition-state checks.

## Single Scorer

Use:

```text
Acquisition Board -> MARS Sleep Scorer -> Splitter -> TTL Display Panel / Acq Board Output
```

`MARS Sleep Scorer` emits TTL events. `Acq Board Output` routes the selected TTL line to Acquisition Board hardware.

## Multi Scorer

Use:

```text
Acquisition Board -> MARS Multi Sleep Scorer
```

`MARS Multi Sleep Scorer` sends `ACQBOARD TRIGGER <line> <duration_ms>` directly. This avoids needing one `Acq Board Output` node per animal.

## Physical Lines

The validated Acquisition Board path supports physical TTL lines 1-8.

- A1-A8 default to lines 1-8.
- A9-A20 default to line 0.
- Line 0 means score and log only.
- Duplicate physical TTL lines are rejected when output is enabled.

## Bench Test

1. Verify manual output with `TTL Toggle Panel -> Acq Board Output`.
2. Start acquisition.
3. Press the relevant MARS `Test TTL` button.
4. Confirm LED/scope output and console message.
5. Confirm `stim_decisions.csv` records the line, pulse width, and hardware result.

Use a visible LED only for bench validation. Live stimulation wiring should remain disconnected until scoring-only mode, logging, and TTL routing pass for that session.
