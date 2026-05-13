# Training and Evaluation Workflow

## Profiles

Each folder under `configs/` is a self-contained experiment profile:

- one YAML config
- one `data/` directory
- one `artifacts/` directory

This keeps experiments isolated by city, station group, period, or model variant.

## Training

Training does the following:

1. Load the configured raw files.
2. Restrict raw data to the configured training slice.
3. Run correlation analysis and imputation on that training slice only.
4. Build the station matrix and supervised dataset.
5. Normalize features using train-only statistics when enabled.
6. Fit the configured model.
7. Save artifacts and optional training data dumps.

Because missing-value handling happens after the data slice is chosen, validation periods are not used to shape training preprocessing artifacts.

## Evaluation

Evaluation does the following:

1. Resolve the saved run from `--run-dir`, `artifacts.eval_run_dir`, `latest_run_<config-name>.txt`, or `latest_run.txt`.
2. Load only the years touched by the configured validation periods.
3. Run evaluation-side preprocessing on that evaluation slice.
4. Build validation rows from `validation_periods` and optional `test_time_steps`.
5. Score the trained model and write metrics, CSV outputs, and optional plots.

For a full step-by-step evaluation walkthrough, flowchart, and function links, see [Evaluation Deep Dive](evaluation.md).

For how the interactive plot command reads evaluation artifacts and builds HTML output, see [Plotting Deep Dive](plotting.md).

Using `artifacts.eval_run_dir` here does not change training output layout. Training still writes timestamped folders; evaluation simply reads from the configured run by default when `--run-dir` is omitted.

## Multi-Config Runs

Both training and evaluation accept repeated `--config` arguments.

They also accept repeated `--config-dir` arguments. Each directory is expanded to YAML files (`*.yaml`, `*.yml`) in sorted order.
Base configs named `base.yaml` or matching `*_base.yaml` are automatically excluded.

Examples:

```bash
aq-spatial-reconstruction-train --config configs/example/example.yaml --config configs/delhi/delhi.yaml
aq-spatial-reconstruction-eval --config configs/example/example.yaml --config configs/delhi/delhi.yaml
aq-spatial-reconstruction-train --config-dir configs/delhi
aq-spatial-reconstruction-eval --config-dir configs/delhi
```

For multi-config evaluation, you can also provide aligned `--run-dir` arguments if each config should evaluate a different saved run.

When multiple configs are evaluated in one command, the CLI writes a comparison CSV across runs and includes its path in the JSON payload as `comparison.csv`. If you used `--config-dir`, the default path is `<config-dir>/outputs/evaluation_comparison_<timestamp>.csv`. Use `--comparison-csv` to choose a different output file path.

Example:

```bash
aq-spatial-reconstruction-eval \
  --config configs/example/example.yaml \
  --run-dir configs/example/artifacts/20260322_120000_0 \
  --config configs/delhi/delhi.yaml \
  --run-dir configs/delhi/artifacts/20260322_130000_0
```

The ordering matters:

- the first `--run-dir` is paired with the first `--config`
- the second `--run-dir` is paired with the second `--config`

If you do not pass `--run-dir`, evaluation falls back to `artifacts.eval_run_dir`, then the config-scoped marker `latest_run_<config-name>.txt`, then `latest_run.txt` for each config.

The plotting command uses the same run-resolution approach when you pass configs instead of explicit run directories.

## Time-Series Cross Validation

If `eval.cross_validation.folds` is populated, evaluation writes fold-specific outputs under:

`validation/cross_validation/`

Each fold contains:

- `validation.csv`
- `metrics.json`

And the shared CV folder contains:

- `metrics_summary.csv`
- `summary.json`
