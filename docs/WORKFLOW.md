# MARS workflow

## Install

MARS targets Open Ephys GUI API10 on Windows. Keep `MARSSleepScorer.dll` beside the `MARSSleepScorer` runtime folder. Close Open Ephys, run `scripts\\windows\\Install-MARSPlugin.ps1`, restart Open Ephys, and confirm the MARS processors are visible.

## Configure and dry-run

For one subject use `Acquisition Board -> MARS Sleep Scorer -> Splitter -> LFP Viewer`. Select EEG and EMG channels and a model. Keep `Output enabled` off. Run at least 30 epochs and confirm valid predictions, timing rows, a complete run bundle, and zero missed deadlines.

For multiple subjects use `Acquisition Board -> MARS Multi Sleep Scorer -> Splitter -> LFP Viewer`. Verify each enabled slot’s identifier and channel pair. Physical TTL output is supported only for configured lines 1-8.

## TTL safety

Keep animal-facing stimulation disconnected during setup. First verify `TTL Toggle Panel -> Acq Board Output / TTL Display Panel`. Then test MARS output only to a display, LED, or scope. Verify the selected line, pulse width, and `stim_decisions.csv`. Enable animal-facing output only after the current dry-run and bench checks pass.

## Offline analysis

The offline workflow starts from EDF files. It does not require an Open Ephys acquisition session.

1. Install the analysis environment from the repository root:

   ```powershell
   uv sync --group analysis
   ```

2. Create an EDF configuration:

   ```powershell
   uv run python -m tools.mars_analysis build-edf-tree-config --edf-root "C:\path\to\edfs" --output-root "C:\path\to\mars_outputs" --output analysis_config.json
   ```

3. Open `analysis_config.json` and confirm every recording's EEG and EMG channel mapping. Do not score a file until those fields match its EDF signal labels.

4. Run offline scoring and create review artifacts:

   ```powershell
   uv run python -m tools.mars_analysis analyze analysis_config.json
   ```

5. Open the reviewer with the same configuration:

   ```powershell
   uv run python -m tools.mars_analysis gui analysis_config.json
   ```

The reviewer contains Dataset Status, Recording Overview, Classification, Trace QC, Spectrogram, Power Spectra, Epoch Inspector, Group Analysis, and Activity tabs. It reuses existing predictions when present and produces EDF-derived review artifacts in the configured output directory.

`inventory` is optional: it only lists discovered prediction/label files and does not run scoring. `analyze` is the usual command for a new dataset.

## Local validation

To evaluate your own independently labeled data, provide timestamped prediction and label CSVs with `RecordingID`, `EpochStartSeconds`, `EpochLengthSeconds`, and a state-label column. Run `uv run python -m tools.mars_analysis validate --predictions predictions_standardized.csv --labels labels.csv --output-dir validation`. The result is local to your dataset and is not a claim about performance on other recordings.
