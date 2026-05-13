# AQ Spatial Reconstruction

`aq-spatial-reconstruction` is a research scaffold for reconstructing or forecasting one air-quality station from neighboring stations using a tabular preprocessing pipeline and configurable regressors.

## What It Does

- loads one CSV, a folder of CSVs, or an explicit station-to-file mapping
- resolves station ids from columns, file names, folder names, or config mappings
- computes correlation analysis and configurable target-column imputation
- supports configurable feature normalization types (`none`, `zscore`, `minmax`, `robust`, `maxabs`)
- pivots long-form station data into a station matrix
- builds supervised forecasting rows with neighbor features, optional target history, and optional timestamp features (month/day/day_of_week/hour)
- trains a configurable regressor
- evaluates on configurable testing periods or explicit time-series CV folds
- writes artifacts and validation outputs; plotting CLI generates consolidated prediction graph outputs

## Project Layout

```text
aq-spatial-reconstruction/
  src/
  configs/
  docs/
  tests/
  README.md
  pyproject.toml
  setup.py
```

The implementation package lives under [`src`](src).

## Quick Start

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
aq-spatial-reconstruction-train --config configs/example/example.yaml
aq-spatial-reconstruction-eval --config configs/example/example.yaml
aq-spatial-reconstruction-predict --config configs/example/example.yaml --input 12.5,15.1,17.2
aq-spatial-reconstruction-plot --config configs/example/example.yaml --open

# Optional: append a readable tag into generated artifact/output folder names
aq-spatial-reconstruction-train --config configs/example/example.yaml --comment baseline_march
```

Missing-value handling is currently drop-only: rows containing missing values in
`data.impute_columns` are removed before station-matrix generation.

For a detailed validation-tuning walkthrough (candidate generation, trial scoring, final-fit behavior, and where effective tuned params are stored), see [Model Validation Guide](docs/validation.md).

## Interactive Prediction vs Actual Graphs

Use the plotting CLI to generate a zoomable HTML chart from one or many evaluation runs.

For a detailed explanation of how plotting works, what files it reads, and how run resolution behaves, see [Plotting Deep Dive](docs/plotting.md).

Default usage (recommended):

```bash
aq-spatial-reconstruction-plot --config-dir configs/delhi
```

With only `--config-dir`, the command automatically:

- loads all non-base YAML configs in that directory recursively
- resolves the latest run per config
- writes outputs under a timestamped subfolder in `configs/delhi/outputs/plots/` (for example `20260402_141530_0`)
- if training run metadata has a run comment (or `--comment` is passed to plot), the folder name includes it (for example `baseline_march_20260402_141530_0`)
- writes interactive HTML as `prediction_vs_actual_interactive_<timestamp>.html`
- writes static prediction PNG/CSV files into the same folder (use `--no-static` to disable)

Examples:

```bash
# Use latest run referenced by the config marker
aq-spatial-reconstruction-plot --config configs/example/example.yaml --open

# Compare multiple algorithms/configs in one interactive chart
aq-spatial-reconstruction-plot ^
  --config configs/delhi/subconfigs/extra_trees/delhi_extra_trees_124.yaml ^
  --config configs/delhi/subconfigs/bilstm/pm25/delhi_bilstm_pm25_124.yaml ^
  --open

# Or provide run directories directly and control output path
aq-spatial-reconstruction-plot ^
  --run-dir configs/delhi/artifacts/delhi_extra_trees_124_20260320_153000_0 ^
  --run-dir configs/delhi/artifacts/delhi_bilstm_pm25_124_20260320_160500_0 ^
  --output outputs/prediction_vs_actual.html ^
  --output-dir outputs/plots
```

Chart features:

- mouse-wheel zoom and pan
- range slider for quick navigation through long time series
- algorithm dropdown filter
- in-chart legend controls (hide, right, bottom)
- separate traces for actual and prediction per station/target

Legend behavior:

- default: hidden (`--legend hidden`)
- alternatives: `--legend right`, `--legend bottom`, `--legend top`

## TensorFlow/LSTM Python Version

TensorFlow/LSTM support currently requires a Python 3.12 virtual environment in this project.

- Core non-LSTM workflows continue to work on newer Python versions supported by package metadata.
- TensorFlow/LSTM is currently not supported on Python 3.14 in this repo setup.
- This may change in the future as TensorFlow publishes compatible wheels.

Recommended commands for LSTM/TensorFlow workflows on Windows:

```powershell
py -3.12 -m venv .venv312
.\.venv312\Scripts\Activate.ps1
pip install -e ".[lstm]"
```

To run a Bi-LSTM variant, keep the same regressor class and set:

```yaml
model:
  algorithm: src.lstm_regressor.LocalLSTMRegressor
  params:
    bidirectional: true
```

Delhi non-base experiment profiles are organized under `configs/delhi/subconfigs/`.

Delhi profiles are target-specialized per station:

- Bi-LSTM PM2.5-only: `configs/delhi/subconfigs/bilstm/pm25/delhi_bilstm_pm25_<station>.yaml`
- Bi-LSTM NO2-only: `configs/delhi/subconfigs/bilstm/no2/delhi_bilstm_no2_<station>.yaml`
- Extra Trees PM2.5-only: `configs/delhi/subconfigs/extra_trees/pm25/delhi_extra_trees_pm25_<station>.yaml`
- Extra Trees NO2-only: `configs/delhi/subconfigs/extra_trees/no2/delhi_extra_trees_no2_<station>.yaml`

Or use the included bootstrap script:

```powershell
.\scripts\bootstrap-venv312.ps1 -Activate
```

The script creates `.venv312` with Python 3.12 if missing, then activates it.

### VS Code Auto-Activation (PowerShell)

This workspace now includes a PowerShell terminal profile that runs the bootstrap script automatically at terminal start.

- File: `.vscode/settings.json`
- Script: `scripts/bootstrap-venv312.ps1`

Result: opening a new integrated PowerShell terminal in this workspace auto-creates/activates `.venv312`.

## Auto-Update Docs Anchors On Commit

Line-number source links in docs can be refreshed automatically on every commit.

One-time setup:

```powershell
.\scripts\install-git-hooks.ps1
```

What this enables:

- `.githooks/pre-commit` runs `scripts/update-doc-anchors.py`
- Then it re-stages:
  - `docs/training-process.md`
  - `docs/evaluation.md`

Manual run (optional):

```powershell
python scripts/update-doc-anchors.py
```

## Run Multiple Configs In Sequence

Training and evaluation both support repeating `--config` so you can process several experiment profiles in one command.

You can also factor shared settings into a base profile and let each experiment override only differences:

```yaml
# configs/delhi/delhi_extra_trees.yaml
extends: delhi_base.yaml
model:
  algorithm: extra_trees
```

Train multiple profiles:

```bash
aq-spatial-reconstruction-train ^
  --config configs/example/example.yaml ^
  --config configs/delhi/delhi.yaml
```

Or point to a config directory and let the CLI expand all profile YAML files:

```bash
aq-spatial-reconstruction-train --config-dir configs/delhi
```

When `--config-dir` is used, files named like `base.yaml` or `*_base.yaml` are skipped automatically.

Evaluate multiple profiles:

```bash
aq-spatial-reconstruction-eval ^
  --config configs/example/example.yaml ^
  --config configs/delhi/delhi.yaml
```

Directory expansion works for evaluation too:

```bash
aq-spatial-reconstruction-eval --config-dir configs/delhi
```

## Data License

The Delhi air-quality CSVs in this repository use Central Pollution Control Board
(CPCB) / Ministry of Environment, Forest and Climate Change air-quality data.
The corresponding OGD India air-quality catalog lists CPCB as contributor and is
released under the National Data Sharing and Accessibility Policy (NDSAP). OGD
India states that content on data.gov.in is licensed under the Government Open
Data License - India (GODL).

Attribution: Central Pollution Control Board, Ministry of Environment, Forest and
Climate Change, Real time Air Quality Index, Open Government Data Platform India.
Published under the Government Open Data License - India:
https://up.data.gov.in/godl. 
Source catalog:
https://airquality.cpcb.gov.in/ccr/#/caaqm-dashboard-all/caaqm-landing/caaqm-data-repository 

When evaluating multiple configs together, stdout includes a `comparison` section that summarizes metrics across runs and ranks rows by RMSE when available.
The same multi-config evaluation run also writes a comparison CSV and returns its path as `comparison.csv` in stdout.
If `--config-dir` is used and `--comparison-csv` is omitted, the CSV is written to `<config-dir>/outputs/` automatically.

Comparison CSV behavior:

- standard single-target runs: one row per config
- multi-target runs: still one row per config, using the aggregate metrics from `metrics.overall`
- runs with `eval.separate_validation_period_results: true` and multiple `data.validation_periods`: one row per config per validation period
- per-period comparison rows include `validation_period_name`, `validation_period_start`, `validation_period_end`, and `validation_period`
- evaluation also emits baseline experiment rows:
  - `Baseline: Average of Other Stations`
  - `Baseline: Replace with Station <station_id>` for each configured input station

Baseline behavior details:

- Baselines are computed during evaluation from neighbor lag-0 feature columns in the supervised test rows.
- They are included in JSON output as `baseline_results` for standard runs.
- When per-period evaluation is enabled, they are included as `baseline_period_results` and expanded in comparison CSV rows.
- Baseline rows use the same metrics contract (`MSE`, `RMSE`, `MAE`, `R2`, `IA`) as model rows.

To control the file location, pass `--comparison-csv`:

```bash
aq-spatial-reconstruction-eval --config-dir configs/delhi --comparison-csv outputs/delhi_metrics_comparison.csv
```

If each config should evaluate a different saved artifact run, pass aligned `--run-dir` values in the same order:

```bash
aq-spatial-reconstruction-eval ^
  --config configs/example/example.yaml ^
  --run-dir configs/example/artifacts/20260322_120000_0 ^
  --config configs/delhi/delhi.yaml ^
  --run-dir configs/delhi/artifacts/20260322_130000_0
```

`artifacts.eval_run_dir` does not disable timestamped training folders. Training still writes new config-scoped timestamped run directories (for example, `example_20260326_101500_0`). `eval_run_dir` is only a default pointer used by evaluation when `--run-dir` is not passed.

## Documentation

- [Configuration Guide](docs/configuration.md)
- [Imputation Deep Dive](docs/imputation.md)
- [Evaluation Deep Dive](docs/evaluation.md)
- [Plotting Deep Dive](docs/plotting.md)
- [Training and Evaluation Workflow](docs/workflow.md)
- [Training Process Internals](docs/training-process.md)
- [Debugging and Development](docs/debugging.md)
- [Testing Guide](docs/testing.md)

## Notes

- `missing_pct_of_all_rows` means the percentage of rows in that column that are missing, out of all rows in the analyzed table.
- `present_pct_of_all_rows` is the complementary percentage of non-missing rows.
- The example config now includes concrete values for the new period-based settings, and leaves a commented example for time-series cross-validation so the default smoke-test behavior stays simple.
