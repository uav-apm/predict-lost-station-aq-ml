# Configuration Guide

The main example config is [`../configs/example/example.yaml`](../configs/example/example.yaml).

## Config Inheritance

Configs can inherit from a base YAML file via `extends`.

Example:

```yaml
extends: delhi_base.yaml

model:
  algorithm: extra_trees
```

Rules:

- `extends` can be relative to the current config file or an absolute path.
- Nested dictionaries are deep-merged.
- Scalars and lists are replaced by the child config value.
- The `extends` key itself is not kept in the resolved runtime config snapshot.

This pattern works well for experiment families (for example one shared city baseline plus per-model overrides).

## Data Section

Key fields:

- `folder`, `csv`, or `station_files`: where raw data comes from
- `datetime_col`, `sort_col`: structural time columns
- `target_station_id`, `input_station_ids`: modeling roles
- `target_pollutant`: the station value used to build the station matrix; accepts a string (single-output) or list of strings (multi-output)
- `missing_value_strategy`: retained for compatibility, but current preprocessing always drops rows with missing values in `impute_columns`
- `normalization`: feature normalization type (`none`, `zscore`, `minmax`, `robust`, `maxabs`)


Backward compatibility:

- legacy `normalize: true/false` is still accepted and mapped to `zscore`/`none`

For multi-output training, set both `data.target_pollutant` and `model.target` to the same pollutant list.

## Period-Based Splits

The config now supports inclusive period ranges for training, validation, and testing.

Accepted forms:

- `"2025"`: whole year
- `"2025-03"`: whole month
- `"2025-03-22"`: one day
- `{start: "2025-03-01", end: "2025-03-31"}`: explicit range

Primary fields:

- `data.train_periods`
- `data.validation_periods`
- `data.testing_periods`
- `data.test_time_steps`

Legacy `data.test_days` is still supported, but `testing_periods` is clearer and more flexible for held-out evaluation.

If you configure multiple validation periods and want separate outputs per period from the same evaluation run,
set:

```yaml
eval:
  separate_validation_period_results: true
```

This keeps the normal combined `validation/metrics.json` output and also writes per-period files under
`validation/by_period/`.

If you want only per-period outputs (and no combined `validation.csv`/`metrics.json`), set:

```yaml
eval:
  separate_validation_period_results: true
  only_separate_validation_period_results: true
```

`only_separate_validation_period_results` requires at least two configured testing windows (`data.testing_periods` or legacy `data.validation_periods`).

## Missing Values and Statistics

The current pipeline does not fill missing values during preprocessing. It drops
rows where any configured `data.impute_columns` value is missing:

```yaml
data:
  missing_value_strategy: drop
```

That mode removes rows that contain missing values in `data.impute_columns` before station-matrix generation.

Column statistics now expose:

- `row_count`
- `present_count`
- `missing_count`
- `present_pct_of_all_rows`
- `missing_pct_of_all_rows`

`missing_pct_of_all_rows` means:
the share of rows in the analyzed dataset where that column is missing.

## Artifact Control

- `artifacts.dir`: where config-scoped timestamped runs are stored (`<config-name>_<YYYYmmdd_HHMMSS>_<n>`)
- `artifacts.dump_training_data`: writes pre-fit data dumps
- `artifacts.save_training_normalization`: when true, writes `norm_stats.json` during training so eval/predict can reuse training-set normalization stats
- `artifacts.eval_run_dir`: lets evaluation point to a specific artifact run without passing `--run-dir`
- training CLI supports `--comment`; when provided, training creates a comment-scoped parent folder under `artifacts.dir` and stores the timestamped run inside it
- training CLI supports `--save-normalization` / `--no-save-normalization`; default is to save normalization stats
- evaluation CLI supports `--use-training-normalization` / `--no-use-training-normalization`; default is to reuse saved training normalization stats
- plotting CLI supports `--comment`; when omitted, it reuses a single shared `run_comment` from selected run metadata when available

Important:

- training still creates normal timestamped run folders under `artifacts.dir` when no comment is provided
- with a comment, the layout becomes `artifacts.dir/<comment>/<config-name>_<YYYYmmdd_HHMMSS>_<n>`
- `artifacts.eval_run_dir` does not replace or suppress those folders
- it only changes which existing run evaluation reads by default
- training/eval default to using saved training normalization when feature normalization is enabled; disabling it is an explicit opt-out for ablation/debug scenarios

Evaluation run resolution priority is:

1. `--run-dir`
2. `artifacts.eval_run_dir`
3. `latest_run_<config-name>.txt` (derived from the config filename stem)
4. `latest_run.txt`

Each training run updates both markers:

- `latest_run.txt` (global marker within the artifact directory)
- `latest_run_<config-name>.txt` (config-scoped marker)

## Comparing Config Variants

When `aq-spatial-reconstruction-eval` is called with multiple `--config` values, the command output now includes:

- `runs`: per-config evaluation payloads
- `comparison`: compact cross-config metrics rows with optional RMSE ranking (`rank_by_rmse`)

Comparison row rules:

- single-target runs: one row per config
- multi-target runs: one row per config using `metrics.overall`
- runs with separate per-period outputs enabled and multiple validation periods: one row per config per validation period

Per-period comparison rows include:

- `validation_period_name`
- `validation_period_start`
- `validation_period_end`
- `validation_period`

For multi-output runs, comparison keeps pollutant-specific metric columns (no cross-pollutant averaging).
For cross-validation runs, comparison uses fold-averaged metrics.

When multi-config evaluation writes a comparison CSV, it also writes station-level chart images in the same output folder:

- one image per target station
- one subplot per metric
- period values on the x-axis as grouped bars by algorithm

Metrics comparison charts are configurable from YAML:

```yaml
plot:
  title_templates:
    prediction_static: "Prediction vs Actual ({target})\n{algorithm} | Station {station_id}"
    scatter_static: "Actual vs Predicted - {target}\n{algorithm} | Station {station_id}"
  metrics_graph:
    metrics: [RMSE, MAE, IA]
    group_output_by: pollutant
    comparison_mode: validation_periods
```

Static plot image titles are also configurable through `plot.title_templates`.
Each title template can use the same placeholder set: `{target}`, `{algorithm}`, `{station_id}`, `{pollutant}`, `{period_name}`, `{period_start}`, `{period_end}`, and `{run_dir}`.

Supported template keys:

- `prediction_static`
- `scatter_static`
- `baseline_scatter_static`
- `metrics_by_period`
- `metrics_by_station`

Common placeholders:

- `{target}`
- `{algorithm}`
- `{station_id}`
- `{pollutant}`
- `{period_name}`
- `{period_start}`
- `{period_end}`
- `{run_dir}`

`plot.metrics_graph` fields:

- `metrics`: subset of `RMSE`, `MAE`, `MSE`, `IA`, `R2` to include in generated charts
- `group_output_by`: `station` (legacy folder structure) or `pollutant`
- `comparison_mode`:
  - `validation_periods`: chart x-axis uses validation periods (existing behavior)
  - `stations`: chart x-axis uses station ids to compare stations within each validation period

## Model Validation

Validation tuning is documented in detail in [Model Validation Guide](validation.md).

Quick summary:

- Configure `model.validation.algorithm` as `none`, `random_search`, or `grid_search`.
- Candidate params come from `model.validation.param_grid` and are overlaid on top of `model.params`.
- Tuning uses dedicated rows selected by `data.validation_periods`.
- Best params are used for final fit on the full training set.
- Held-out evaluation should use `data.testing_periods`.
- The original config snapshot remains unchanged in `config_snapshot.yaml`.
- Effective tuned params and trial results are stored in `run_metadata.json` (`model_params`, `model_validation`).
- Multi-config training output contains a `runs` list. Training no longer writes a cross-run optimization comparison CSV.

### Algorithm Display Name For Graphs

Use `model.algorithm_name` to control how the algorithm is labeled in comparison and interactive graphs.

Example:

```yaml
model:
  algorithm: src.lstm_regressor.LocalLSTMRegressor
  algorithm_name: BiLSTM (2-layer)
```

## LSTM Model Parameters

When using `LocalLSTMRegressor` (or other neural network models), the `model.params` section controls training behavior.

**Architecture:**

- `timesteps`: Number of time steps per sample (default: 1 for tabular data)
- `lstm_units`: List of LSTM layer sizes, e.g., `[64, 32]` for two layers with 64 and 32 units
- `bidirectional`: Whether to use bidirectional LSTMs (reads sequence both forward and backward)
- `dense_units`: Number of units in the final dense layer before output (0 to skip)
- `dropout`: Dropout rate for regularization (typical range: 0.0 to 0.5)

**Training:**

- `learning_rate`: Adam optimizer learning rate (typical: 0.001 to 0.01)
- `batch_size`: Training batch size (typical: 32 or 64)
- `epochs`: Maximum number of training epochs
- `validation_split`: Fraction of training data reserved for validation (0.1 = 10%)
- `patience`: Early stopping patience, i.e. how many consecutive epochs are allowed without validation loss improvement before training stops automatically. Best weights are restored from the epoch with the lowest validation loss. Set to 0 to disable early stopping.
- `random_state`: Random seed for reproducibility. Keep this as a scalar default such as `42`; put seed values to compare during validation in `model.validation.param_grid.random_state`.
- `fallback_on_missing_backend`: When true, falls back to `MLPRegressor` if TensorFlow is unavailable; when false, raises an error

**Example:**

```yaml
model:
  algorithm: src.lstm_regressor.LocalLSTMRegressor
  params:
    timesteps: 1
    lstm_units: [64, 32]
    bidirectional: true
    dropout: 0.1
    dense_units: 32
    learning_rate: 0.001
    batch_size: 64
    epochs: 80
    validation_split: 0.1
    patience: 10
    random_state: 42
    fallback_on_missing_backend: false
  validation:
    param_grid:
      random_state: [42, 50, 100]
```

## Time-Series Cross Validation

Use `eval.cross_validation.folds` with explicit train and validation periods per fold.

Example:

```yaml
eval:
  cross_validation:
    folds:
      - name: fold_1
        train_periods:
          - start: "2024-01-01"
            end: "2024-06-30"
        validation_periods:
          - start: "2025-01-15"
            end: "2025-01-15"
```

When folds are present, evaluation runs cross-validation output instead of a single validation window.
