# MARS Analysis Tools

This package contains the offline MARS/AccuSleePy analysis pipeline and the
PySide6 desktop GUI.

For researcher setup and usage, start here:

```text
docs\WORKFLOW.md
```

## Quick Launch

From the repository root:

```text
scripts\windows\Launch-MARS-Analysis.bat
```

Or pass a config explicitly:

```text
scripts\windows\Launch-MARS-Analysis.bat tools\mars_analysis\analysis_config.example.json
```

With `uv`:

```powershell
uv sync --group analysis
uv run python -m tools.mars_analysis gui tools\mars_analysis\analysis_config.example.json
```

## Commands

```powershell
uv run python -m tools.mars_analysis init-config --output tools\mars_analysis\analysis_config.local.json
uv run python -m tools.mars_analysis build-edf-tree-config --edf-root "C:\path\to\edfs" --output tools\mars_analysis\analysis_config.local.json --dataset-id my_dataset
uv run python -m tools.mars_analysis inventory tools\mars_analysis\analysis_config.local.json
uv run python -m tools.mars_analysis analyze tools\mars_analysis\analysis_config.local.json
uv run python -m tools.mars_analysis gui tools\mars_analysis\analysis_config.local.json
```

## Output Contract

Each scored recording writes the MARS bundle:

- `predictions_standardized.csv`
- `epoch_timing.csv`
- `stim_decisions.csv`
- `summary.json`

Analysis outputs include:

- `classification_status.csv`
- `state_percentages.csv`
- `state_percentage_validation.csv`
- `state_percentage_validation_summary.csv`
- `hourly_bins.csv`
- `label_inventory.csv`
- `plots\*.html`

## Notes

- The public default model is `E2.5W9`, the trained offline AccuSleePy/MARS
  profile with 2.5 second epochs and 9 input windows.
- EDF reading and plotting are chunked so full raw recordings are not loaded
  into memory.
- The example config is portable and intentionally contains no recordings.
  Create `analysis_config.local.json` for your dataset.
