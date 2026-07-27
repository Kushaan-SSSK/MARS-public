# MARS

<p align="center"><img src="assets/mars-logo.png" alt="MARS mouse sleep-scoring logo" width="260"></p>

Mouse Automated Real-Time Scoring for Open Ephys.

MARS provides native Open Ephys processors for real-time EEG/EMG state scoring, safe TTL routing, portable run folders, and an offline EDF analysis utility.

## Windows prerequisites

The offline workflow requires Windows PowerShell, an internet connection for
the first environment setup, and your own EDF recordings. Raw EDF data is not
included in this repository.

Install `uv` with Windows Package Manager:

```powershell
winget install --id=astral-sh.uv -e
```

Open a new PowerShell window, verify the installation, and change to the cloned
repository root:

```powershell
uv --version
cd "C:\path\to\MARS-public"
```

MARS supports Python 3.11 through 3.13. `uv` automatically downloads a
compatible Python version when one is not already available, so a separate
Python installation is normally unnecessary.

## Choose a workflow

### Real-time scoring in Open Ephys

1. Close Open Ephys.
2. Run `powershell -ExecutionPolicy Bypass -File .\scripts\windows\Install-MARSPlugin.ps1`.
3. Restart Open Ephys and add `MARS Sleep Scorer` or `MARS Multi Sleep Scorer`.
4. Select the model and EEG/EMG channels.
5. Start with `Output enabled` off. Confirm the run folder before any bench TTL test.

### Offline EDF scoring and review

1. Install the analysis environment: `uv sync --group analysis`.
2. Build a config from a folder of EDF files:

   ```powershell
   uv run python -m tools.mars_analysis build-edf-tree-config --edf-root "C:\path\to\edfs" --output-root "C:\path\to\mars_outputs" --output analysis_config.json
   ```

3. Open the config and confirm every recording's EEG and EMG columns. If you
   plan to use Group Analysis, also fill in `animal_id`, `condition`, `dose`,
   and `session` for each recording. The config builder does not infer those
   experimental fields from folder or file names.

4. Score and generate review artifacts:

   ```powershell
   uv run python -m tools.mars_analysis analyze analysis_config.json
   ```

   A first analysis of long recordings can take several minutes and may remain
   quiet while processing. The command prints the generated artifact paths when
   it finishes. Later runs reuse existing predictions when the configuration
   and source files have not changed.

5. Open the EDF reviewer:

   ```powershell
   uv run python -m tools.mars_analysis gui analysis_config.json
   ```

Dataset Status summarizes completion and status counts for the whole dataset.
Use the Recordings list and the Recording Overview or Classification tabs to
inspect individual recordings. Group Analysis is meaningful only when the
grouping metadata above is populated. The reviewer also provides EEG/EMG trace
QC, spectrograms, power spectra, and epoch inspection. See
[docs/WORKFLOW.md](docs/WORKFLOW.md) for the complete safety, real-time,
offline, and validation workflow.

## Benchmark snapshot

On the documented held-out benchmark (24 recordings; 1,005,992 scored epochs), the default offline model achieved 0.969 accuracy, 0.961 balanced accuracy, 0.914 macro F1, 0.971 weighted F1, and 0.945 Cohen's kappa. The primary real-time model (E2.0W3) achieved 95.65% accuracy with a 0.788 ms p95 inference time and no missed deadlines in its reported timing evaluation. These are dataset-specific research benchmarks, not a guarantee of performance on new recordings. See [docs/BENCHMARKS.md](docs/BENCHMARKS.md) for the full evaluation context and all reported results.

## Included components

- `plugin/`: Open Ephys DLL, runtime libraries, models, and default calibration assets.
- `scripts/windows/`: installer and offline analysis launcher.
- `tools/mars_analysis/`: EDF scoring, QC, plots, group summaries, and local user-label validation.

Each run writes `predictions_standardized.csv`, `epoch_timing.csv`, `stim_decisions.csv`, and `summary.json`.
