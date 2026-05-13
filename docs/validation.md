# Model Validation Guide

This document explains how `model.validation` runs during training, how candidates are selected, and where the selected settings are stored.

## Where Validation Runs

Validation tuning runs during training in `src/cli.py`, inside `_optimize_model_params(...)`, before the final model fit.

High-level flow:

1. Load config and preprocess data.
2. Build supervised training rows from `data.train_periods` (or year filters when configured).
3. Build supervised validation rows from `data.validation_periods`.
4. Run candidate search if `model.validation.algorithm` is enabled.
5. Fit final pipeline on the full training rows with best parameters.
6. Save artifacts and metadata.

## Configuration Fields

`model.validation` supports:

- `algorithm`: `none` | `random_search` | `grid_search`
- `metric`: `RMSE` | `R2`
- `max_trials`: used only for `random_search`
- `random_state`: RNG seed for `random_search` sampling
- `param_grid`: parameter candidates

Validation rules:

- `param_grid` must be a non-empty mapping when validation tuning is enabled.
- `data.validation_periods` must produce at least one supervised validation row.
- Put seed values that should be compared during validation under `model.validation.param_grid.random_state`.
- `model.params.random_state` should usually stay as a scalar fallback/default. A list there is still accepted for compatibility and is expanded into validation candidates when `param_grid.random_state` is not set.

Example:

```yaml
model:
  algorithm: src.lstm_regressor.LocalLSTMRegressor
  params:
    timesteps: 1
    lstm_units: [64, 32]
    bidirectional: true
    batch_size: 32
    epochs: 80
    patience: 10
    random_state: 42
    validation_split: 0.1
    verbose: 0
  validation:
    algorithm: random_search
    metric: RMSE
    max_trials: 12
    random_state: 42
    param_grid:
      random_state: [42, 50, 100]
      lstm_units:
        - [64, 32]
        - [32]
      dropout: [0.1, 0.2]
      batch_size: [32, 64]
      learning_rate: [0.001, 0.0005]
```

## Runtime Behavior

1. Build candidate parameter sets from `param_grid`.
2. For each candidate, merge with `model.params`.
3. Fit candidate on training rows.
4. Score on dedicated validation rows.
5. Select best candidate using metric direction:
   - Minimize: `RMSE`
   - Maximize: `R2`
6. Refit final model on all training rows with selected params.

## Artifacts and Metadata

The resolved config is saved unchanged in:

- `<run_dir>/config_snapshot.yaml`

The effective selected params are saved in:

- `<run_dir>/run_metadata.json`

Important fields in `run_metadata.json`:

- `model_params`: final fit params
- `model_validation`: validation summary (`algorithm`, `metric`, `best_score`, `best_params`, `trial_count`, `trials`)

For backward compatibility, `model_optimization` is also written with the same payload.

## Three-Period Workflow

Use explicit periods for temporal isolation:

- `data.train_periods`: used to fit models
- `data.validation_periods`: used for hyperparameter tuning
- `data.testing_periods`: used by evaluation CLI for held-out testing

This avoids reusing evaluation windows for tuning and helps reduce overfitting.

## Related Docs

- [Configuration Guide](configuration.md)
- [Training Process](training-process.md)
- [Evaluation Deep Dive](evaluation.md)
