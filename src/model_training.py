"""Model training, hyperparameter optimisation, and cross-validation orchestration.

This module implements the *train* and *cross-validation* workflows.  It
depends on ``config_loader``, ``artifact_io``, ``data_pipeline``, and the
``pipeline`` package but must not import from ``model_evaluation`` or
``plot_writer``.

Responsibilities
----------------
- ``_train_one`` — end-to-end single-config training run.
- ``_run_cross_validation`` — fold-based cross-validation.
- Hyperparameter optimisation (``_optimize_model_params``, ``_param_candidates``).
- Model-config construction (``_build_model_config``).
- Training-artefact helpers: ``_dump_training_inputs``.
"""

from __future__ import annotations

import shutil
from datetime import datetime
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .artifact_io import (
    _artifact_base_dir,
    _comment_artifact_dir,
    _config_latest_marker_name,
    _load_chunked_joblib,
    _normalization_artifact_path,
    _resolve_algorithm_display_name,
    _save_chunked_joblib,
    _save_json,
    _timestamped_run_dir,
)
from .config_loader import RUN_LOGGER, _load_cfg, _report_note, _report_phase
from .data import apply_norm_stats, normalize_train_test
from .data_pipeline import (
    _normalization_type,
    _periods_to_serializable,
    _positive_lag_count,
    _prepare_fold_training_inputs,
    _prepare_fold_validation_inputs,
    _prepare_training_data,
    _prepare_validation_data_for_tuning,
    _should_save_training_normalization,
)
from .forecasting import normalize_periods
from .pipeline import StationForecastConfig, StationForecastPipeline


# ---------------------------------------------------------------------------
# Model config helpers
# ---------------------------------------------------------------------------


def _build_model_config(cfgd: dict) -> StationForecastConfig:
    """Construct a :class:`~src.pipeline.StationForecastConfig` from the config dict.

    Args:
        cfgd: Fully-resolved configuration dictionary.

    Returns:
        Populated :class:`StationForecastConfig` instance.
    """
    return StationForecastConfig(
        target=cfgd["model"]["target"],
        datetime_col=cfgd["model"]["datetime_col"],
        algorithm=cfgd["model"].get("algorithm", "random_forest"),
        params=_normalize_model_params(cfgd["model"].get("params")),
    )


def _normalize_model_params(params: dict | None) -> dict | None:
    """Return estimator params safe to pass to a model constructor.

    ``model.params.random_state`` may be configured as a list to request a seed
    sweep during validation tuning.  Estimators still require one concrete seed
    for construction, so the first configured seed is used wherever a scalar
    parameter set is needed before/without tuning.
    """
    if not params:
        return params

    normalized = dict(params)
    random_state = normalized.get("random_state")
    if isinstance(random_state, list):
        if not random_state:
            raise ValueError("model.params.random_state must include at least one value when provided as a list.")
        normalized["random_state"] = _coerce_random_state_seed(
            random_state,
            config_key="model.params.random_state",
        )
    return normalized


def _validation_param_grid_with_random_state(
    param_grid: dict,
    model_params: dict | None,
) -> dict:
    """Include ``model.params.random_state`` list values in validation candidates."""
    resolved_grid = dict(param_grid)
    random_state_values = (model_params or {}).get("random_state")
    if isinstance(random_state_values, list) and "random_state" not in resolved_grid:
        if not random_state_values:
            raise ValueError("model.params.random_state must include at least one value when provided as a list.")
        resolved_grid["random_state"] = [int(value) for value in random_state_values]
    return resolved_grid


def _coerce_random_state_seed(raw, *, config_key: str) -> int:
    """Return one integer seed from a scalar or non-empty list config value."""
    if isinstance(raw, list):
        if not raw:
            raise ValueError(f"{config_key} must include at least one value when provided as a list.")
        raw = raw[0]
    return int(raw)


def _coerce_optimization_algorithm(raw: str | None) -> str:
    """Normalise a raw optimisation algorithm string to a canonical value.

    Args:
        raw: Raw string from ``model.validation.algorithm`` config.

    Returns:
        One of ``"none"``, ``"random_search"``, ``"grid_search"``.

    Raises:
        ValueError: When *raw* maps to an unsupported algorithm.
    """
    aliases = {
        "": "none",
        "off": "none",
        "disabled": "none",
        "random": "random_search",
        "randomsearch": "random_search",
        "grid": "grid_search",
        "gridsearch": "grid_search",
    }
    value = str(raw or "none").strip().lower()
    value = aliases.get(value, value)
    supported = {"none", "random_search", "grid_search"}
    if value not in supported:
        raise ValueError(
            "model.validation.algorithm must be one of: none, random_search, grid_search"
        )
    return value


def _resolve_optimization_metric(raw: str | None) -> str:
    """Normalise a raw metric name used during hyperparameter search.

    Args:
        raw: Raw string from ``model.validation.metric`` config.

    Returns:
        Canonical metric name: one of ``"RMSE"``, ``"R2"``.

    Raises:
        ValueError: When *raw* cannot be mapped to a supported metric.
    """
    aliases = {
        "rmse": "RMSE",
        "r2": "R2",
    }
    metric = aliases.get(str(raw or "RMSE").strip().lower())
    if metric is None:
        raise ValueError(
            "model.validation.metric must be one of: RMSE, R2"
        )
    return metric


def _model_validation_cfg(cfgd: dict) -> dict:
    """Return the model-validation (HPO) sub-config, supporting legacy keys.

    Looks for ``model.validation`` first, then falls back to the deprecated
    ``model.optimization`` key.

    Args:
        cfgd: Fully-resolved configuration dictionary.

    Returns:
        Validation config dict, or an empty dict when absent.
    """
    model_cfg = cfgd.get("model", {}) or {}
    validation_cfg = model_cfg.get("validation")
    if isinstance(validation_cfg, dict):
        return validation_cfg
    legacy_cfg = model_cfg.get("optimization")
    if isinstance(legacy_cfg, dict):
        return legacy_cfg
    return {}


def _metric_direction(metric_name: str) -> str:
    """Return ``"max"`` for metrics that should be maximised, ``"min"`` otherwise."""
    return "max" if metric_name == "R2" else "min"


def _candidate_metric_from_result(metrics_payload: dict, metric_name: str) -> float:
    """Extract a scalar metric value from an evaluation result payload.

    Handles both flat payloads (``{metric: value}``) and multi-target payloads
    (``{by_target: {name: {metric: value}, …}}``).

    Args:
        metrics_payload: Raw metrics dict from
            :meth:`~src.pipeline.StationForecastPipeline.evaluate`.
        metric_name: Canonical metric name to extract.

    Returns:
        Scalar float value for the requested metric.

    Raises:
        ValueError: When the metric is absent from the payload.
    """
    if metric_name in metrics_payload and isinstance(
        metrics_payload[metric_name], (int, float)
    ):
        return float(metrics_payload[metric_name])

    by_target = metrics_payload.get("by_target")
    if isinstance(by_target, dict) and by_target:
        values = []
        for target_metrics in by_target.values():
            if isinstance(target_metrics, dict) and metric_name in target_metrics:
                values.append(float(target_metrics[metric_name]))
        if values:
            return float(sum(values) / len(values))

    raise ValueError(f"Metric {metric_name} was not present in evaluation payload.")


def _param_candidates(
    param_grid: dict,
    algorithm: str,
    max_trials: int,
    random_state: int,
) -> list[dict]:
    """Generate candidate hyperparameter combinations from a grid specification.

    Args:
        param_grid: Mapping of parameter name to list of candidate values.
        algorithm: ``"grid_search"`` returns all combinations; ``"random_search"``
            samples up to *max_trials* combinations.
        max_trials: Upper bound on the number of candidates for random search.
        random_state: Seed for the random number generator.

    Returns:
        List of candidate parameter dicts.

    Raises:
        ValueError: When any entry in *param_grid* is empty or the grid
            produces no combinations.
    """
    items = []
    for key, values in param_grid.items():
        value_list = values if isinstance(values, list) else [values]
        if not value_list:
            raise ValueError(
                f"model.validation.param_grid.{key} must include at least one value."
            )
        items.append((key, value_list))

    keys = [key for key, _ in items]
    combinations = [
        dict(zip(keys, combo))
        for combo in product(*[vals for _, vals in items])
    ]
    if not combinations:
        raise ValueError("model.validation.param_grid produced no candidate combinations.")

    if algorithm == "random_search":
        count = min(max(1, int(max_trials)), len(combinations))
        rng = np.random.default_rng(int(random_state))
        indices = sorted(
            rng.choice(len(combinations), size=count, replace=False).tolist()
        )
        return [combinations[idx] for idx in indices]
    return combinations


def _optimize_model_params(
    cfgd: dict,
    train_X: pd.DataFrame,
    train_y,
    *,
    validation_X: pd.DataFrame | None = None,
    validation_y=None,
) -> dict | None:
    """Run hyperparameter optimisation and return the best parameter set.

    Skips optimisation and returns ``None`` when ``model.validation.algorithm``
    is ``"none"`` or absent.

    Args:
        cfgd: Resolved configuration dictionary.
        train_X: Normalised training feature matrix.
        train_y: Training target series or array.
        validation_X: Normalised validation feature matrix.  Required when
            optimisation is enabled.
        validation_y: Validation target series or array.  Required when
            optimisation is enabled.

    Returns:
        Dict with keys ``algorithm``, ``metric``, ``direction``, ``trial_count``,
        ``best_score``, ``best_params``, ``trials``, etc., or ``None`` when
        optimisation is disabled.

    Raises:
        ValueError: When optimisation is enabled but validation data or
            ``param_grid`` are missing/empty.
        RuntimeError: When optimisation completes without producing a best
            parameter set.
    """
    validation_cfg = _model_validation_cfg(cfgd)
    algorithm = _coerce_optimization_algorithm(validation_cfg.get("algorithm"))
    if algorithm == "none":
        return None

    raw_model_params = dict(cfgd.get("model", {}).get("params") or {})
    param_grid = validation_cfg.get("param_grid")
    if not isinstance(param_grid, dict) or not param_grid:
        raise ValueError(
            "model.validation.param_grid must be a non-empty mapping when validation "
            "tuning is enabled."
        )
    param_grid = _validation_param_grid_with_random_state(param_grid, raw_model_params)

    metric_name = _resolve_optimization_metric(validation_cfg.get("metric"))
    direction = _metric_direction(metric_name)
    max_trials = int(validation_cfg.get("max_trials", 20))
    random_state = _coerce_random_state_seed(
        validation_cfg.get("random_state", 42),
        config_key="model.validation.random_state",
    )

    if validation_X is None or validation_y is None:
        raise ValueError(
            "Validation tuning requires dedicated validation rows. "
            "Configure data.validation_periods and run training with model.validation enabled."
        )
    if len(validation_X) == 0:
        raise ValueError(
            "data.validation_periods produced no supervised validation rows for model.validation."
        )

    base_params = _normalize_model_params(raw_model_params) or {}
    candidates = _param_candidates(param_grid, algorithm, max_trials, random_state)
    total = len(candidates)

    best_score: float | None = None
    best_params: dict | None = None
    trials: list[dict] = []
    model_target = cfgd["model"]["target"]
    model_datetime = cfgd["model"]["datetime_col"]
    model_algorithm = cfgd["model"].get("algorithm", "random_forest")

    for idx, candidate in enumerate(candidates, start=1):
        merged_params = {**base_params, **candidate}
        trial_pipeline = StationForecastPipeline(
            StationForecastConfig(
                target=model_target,
                datetime_col=model_datetime,
                algorithm=model_algorithm,
                params=merged_params,
            )
        )
        trial_pipeline.fit(train_X, train_y)
        trial_result = trial_pipeline.evaluate(validation_X, validation_y)
        score = _candidate_metric_from_result(trial_result["metrics"], metric_name)

        is_better = (
            best_score is None
            or (direction == "min" and score < best_score)
            or (direction == "max" and score > best_score)
        )
        if is_better:
            best_score = score
            best_params = merged_params

        trials.append(
            {
                "trial": idx,
                "total_trials": total,
                "score": float(score),
                "params": candidate,
            }
        )

    if best_params is None:
        raise RuntimeError("Optimization failed to produce a best parameter set.")

    return {
        "algorithm": algorithm,
        "metric": metric_name,
        "direction": direction,
        "validation_rows": int(len(validation_X)),
        "validation_source": "data.validation_periods",
        "trial_count": total,
        "best_score": float(best_score),  # type: ignore[arg-type]
        "best_params": best_params,
        "trials": trials,
    }


# ---------------------------------------------------------------------------
# Training artefact helpers
# ---------------------------------------------------------------------------


def _dump_training_inputs(run_dir: Path, cfgd: dict, prepared: dict) -> None:
    """Persist pre-fit training data for offline inspection.

    Writes the supervised feature matrix, targets, station matrix, and a
    summary JSON into a ``training_dump/`` sub-directory under *run_dir*.

    Args:
        run_dir: Active run directory.
        cfgd: Resolved configuration dictionary.
        prepared: Return value of
            :func:`~src.data_pipeline._prepare_training_data`.
    """
    train_debug_dir = run_dir / "training_dump"
    train_debug_dir.mkdir(parents=True, exist_ok=True)

    prepared["train_rows"].to_csv(
        train_debug_dir / "training_supervised_rows.csv", index=False
    )
    prepared["train_X_raw"].to_csv(
        train_debug_dir / "training_features_raw.csv", index=False
    )
    prepared["train_X"].to_csv(
        train_debug_dir / "training_features_model_input.csv", index=False
    )
    train_target = prepared["train_y"]
    if isinstance(train_target, pd.Series):
        train_target.rename("target_value").to_csv(
            train_debug_dir / "training_target.csv", index=False
        )
    else:
        train_target.to_csv(train_debug_dir / "training_target.csv", index=False)
    prepared["station_matrix"].reset_index().to_csv(
        train_debug_dir / "training_station_matrix.csv", index=False
    )

    _save_json(
        train_debug_dir / "training_summary.json",
        {
            "row_count": len(prepared["train_rows"]),
            "feature_count": len(prepared["feature_names"]),
            "feature_names": prepared["feature_names"],
            "target_columns": prepared["target_names"],
            "target_station_id": cfgd["data"]["target_station_id"],
            "input_station_ids": cfgd["data"]["input_station_ids"],
            "target_pollutant": cfgd["data"]["target_pollutant"],
            "horizon": int(cfgd["eval"].get("horizon", 1)),
            "normalization": _normalization_type(cfgd["data"]),
            "algorithm": cfgd["model"]["algorithm"],
            "model_params": cfgd["model"].get("params") or {},
        },
    )


def _suppress_pipeline_verbose(pipeline: StationForecastPipeline) -> None:
    """Disable per-tree progress messages on the pipeline's inner estimator.

    sklearn tree ensembles propagate ``self.verbose`` to ``joblib.Parallel``
    during both ``fit()`` and ``predict()``.  Setting ``verbose=0`` prevents
    ``[Parallel(…)] Done N out of N`` lines during evaluation.

    Args:
        pipeline: Fitted (or unfitted) pipeline whose inner model verbose flag
            should be silenced.
    """
    model = getattr(pipeline, "model", None)
    if model is not None and hasattr(model, "verbose"):
        model.verbose = 0


# ---------------------------------------------------------------------------
# Top-level training entry points
# ---------------------------------------------------------------------------


def _train_one(
    config_path: str,
    run_comment: str | None = None,
    save_normalization: bool | None = None,
) -> dict:
    """Execute a complete training run for a single config file.

    Phases
    ------
    1. Load configuration and dataset.
    2. Preprocess raw data (imputation / drop).
    3. Build correlation analysis and heatmap.
    4. Apply missing-value strategy.
    5. Create run directory; optionally dump training inputs; fit pipeline.
    6. Save trained artefacts.
    7. Write run metadata.
    8. Summarise.

    Args:
        config_path: Path to the YAML configuration file.
        run_comment: Optional free-text label added to the run directory name.
        save_normalization: Override for ``artifacts.save_training_normalization``.

    Returns:
        Dict with keys ``config``, ``run_dir``, ``log_file``, ``algorithm``,
        ``algorithm_name``, ``model_params``, ``model_validation``,
        ``model_optimization``, ``feature_names``,
        ``training_imputation_plot``, ``training_imputation_plots``,
        ``training_loss_plot``, ``normalization_stats_file``,
        ``saved_training_normalization``, ``run_comment``.
    """
    total_phases = 8
    RUN_LOGGER.start(config_path)
    _report_note(f"Starting training run for config: {Path(config_path).name}")
    _report_phase("Loading configuration and input dataset", 1, total_phases)
    cfgd, base = _load_cfg(config_path)
    algorithm_name = _resolve_algorithm_display_name(cfgd)

    _report_phase("Preprocessing raw data", 2, total_phases)
    prepared = _prepare_training_data(cfgd, base)

    _report_phase("Building correlation analysis and heatmap", 3, total_phases)
    if not prepared["cache_paths"]["heatmap"].exists():
        _report_note(
            f"Correlation heatmap still missing at {prepared['cache_paths']['heatmap']}"
        )

    _report_phase("Applying configurable missing-value strategy", 4, total_phases)
    _report_note(
        "Dropped rows with missing values in configured columns: "
        + str(prepared["imputation_details"].get("dropped_rows", 0))
    )

    _report_phase(
        "Creating run directory, optional training dumps, and training regression pipeline",
        5,
        total_phases,
    )
    artifacts_base = _artifact_base_dir(cfgd, base)
    comment_artifact_dir = _comment_artifact_dir(artifacts_base, run_comment)
    marker_name = _config_latest_marker_name(config_path)
    run_dir = _timestamped_run_dir(
        comment_artifact_dir,
        latest_markers=[marker_name],
        run_label=Path(config_path).stem,
        marker_dirs=(
            [artifacts_base, comment_artifact_dir]
            if comment_artifact_dir != artifacts_base
            else [artifacts_base]
        ),
    )
    train_log_file = run_dir / "train.log"
    RUN_LOGGER.set_log_file(train_log_file)
    _report_note(f"Persisting training logs to {train_log_file}")
    validation_dir = run_dir / "validation"
    validation_dir.mkdir(parents=True, exist_ok=True)

    if bool(cfgd["artifacts"].get("dump_training_data", False)):
        _report_note(f"Dumping pre-fit training data to {run_dir / 'training_dump'}")
        _dump_training_inputs(run_dir, cfgd, prepared)

    validation_tuning_data = _prepare_validation_data_for_tuning(
        cfgd, base, prepared["norm_stats"]
    )
    optimization_result = _optimize_model_params(
        cfgd,
        prepared["train_X"],
        prepared["train_y"],
        validation_X=(validation_tuning_data or {}).get("validation_X"),
        validation_y=(validation_tuning_data or {}).get("validation_y"),
    )
    model_config = _build_model_config(cfgd)
    if optimization_result is not None:
        _report_note(
            "Model validation tuning completed with "
            f"{optimization_result['algorithm']} ({optimization_result['trial_count']} trials); "
            f"best {optimization_result['metric']}={optimization_result['best_score']:.6f}"
        )
        model_config.params = dict(optimization_result["best_params"])

    model_params = dict(getattr(model_config, "params", None) or {})
    pipeline = StationForecastPipeline(model_config)
    pipeline.fit(prepared["train_X"], prepared["train_y"])

    _report_phase("Saving trained artifacts", 6, total_phases)
    _save_chunked_joblib(pipeline, run_dir / "pipeline.pkl")
    with (run_dir / "config_snapshot.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(cfgd, handle, sort_keys=False)

    save_training_normalization = _should_save_training_normalization(
        cfgd.get("artifacts"),
        save_normalization,
    )
    norm_stats_file = _normalization_artifact_path(run_dir) if save_training_normalization else None
    if save_training_normalization:
        _save_json(norm_stats_file, prepared["norm_stats"])

    from .data import column_statistics_table  # local to avoid top-level cycle

    _save_json(
        run_dir / "column_statistics.json",
        {
            "before": prepared["column_stats_before"],
            "after": prepared["column_stats_after"],
        },
    )
    _save_json(run_dir / "correlation_analysis.json", prepared["correlation_analysis"])
    if prepared["cache_paths"]["heatmap"].exists():
        _report_note(
            f"Copying correlation heatmap into run artifacts: {run_dir / 'correlation_heatmap.png'}"
        )
        shutil.copy2(prepared["cache_paths"]["heatmap"], run_dir / "correlation_heatmap.png")
    else:
        _report_note(
            f"Expected correlation heatmap was not found at {prepared['cache_paths']['heatmap']}"
        )

    shutil.copy2(prepared["cache_paths"]["dataset"], run_dir / "imputed_dataset.csv")
    shutil.copy2(prepared["cache_paths"]["details"], run_dir / "imputation_details.json")
    shutil.copy2(prepared["cache_paths"]["stats"], run_dir / "imputation_stats.json")
    shutil.copy2(prepared["cache_paths"]["stats_csv"], run_dir / "imputation_stats.csv")
    shutil.copy2(
        prepared["cache_paths"]["column_stats_before_csv"],
        run_dir / "column_statistics_before.csv",
    )
    shutil.copy2(
        prepared["cache_paths"]["column_stats_after_csv"],
        run_dir / "column_statistics_after.csv",
    )

    imputation_performed = bool(prepared["imputation_details"].get("columns"))
    if imputation_performed:
        _save_json(
            run_dir / "imputation_usage.json",
            {
                "used_cache": prepared["used_cache"],
                "cache_dir": str(prepared["cache_dir"]),
            },
        )
    if _normalization_type(cfgd["data"]) != "none":
        prepared["train_X"].to_csv(run_dir / "normalized_dataset.csv", index=False)

    _report_phase("Writing run metadata", 7, total_phases)
    missing_strategy = "drop"
    _save_json(
        run_dir / "run_metadata.json",
        {
            "created_at": datetime.now().isoformat(),
            "config_path": str(Path(config_path).resolve()),
            "log_file": str(train_log_file),
            "data_source": prepared["data_source"],
            "run_dir": str(run_dir),
            "algorithm": cfgd["model"]["algorithm"],
            "algorithm_name": algorithm_name,
            "model_params": model_params,
            "model_validation": optimization_result,
            "model_optimization": optimization_result,
            "feature_names": pipeline.feature_names_,
            "target_names": prepared["target_names"],
            "target_station_id": cfgd["data"]["target_station_id"],
            "input_station_ids": cfgd["data"]["input_station_ids"],
            "station_history_lags": _positive_lag_count(
                cfgd["data"], "station_history_lags", 1
            ),
            "train_rows": len(prepared["train_rows"]),
            "test_rows": 0,
            "imputation_cache_dir": str(prepared["cache_dir"]),
            "used_imputation_cache": prepared["used_cache"],
            "normalization_stats_file": str(norm_stats_file) if norm_stats_file else None,
            "saved_training_normalization": save_training_normalization,
            "run_group_dir": (
                str(comment_artifact_dir)
                if comment_artifact_dir != artifacts_base
                else None
            ),
            "run_comment": (
                str(run_comment).strip()
                if run_comment and str(run_comment).strip()
                else None
            ),
            "missing_value_strategy": missing_strategy,
        },
    )

    _report_phase("Summarizing training outputs", 8, total_phases)
    _report_note(f"Training complete. Results written under {run_dir}")
    return {
        "config": str(Path(config_path).resolve()),
        "run_dir": str(run_dir),
        "log_file": str(train_log_file),
        "algorithm": cfgd["model"]["algorithm"],
        "algorithm_name": algorithm_name,
        "model_params": model_params,
        "model_validation": optimization_result,
        "model_optimization": optimization_result,
        "feature_names": pipeline.feature_names_,
        "normalization_stats_file": str(norm_stats_file) if norm_stats_file else None,
        "saved_training_normalization": save_training_normalization,
        "run_comment": (
            str(run_comment).strip()
            if run_comment and str(run_comment).strip()
            else None
        ),
    }


def _run_cross_validation(cfgd: dict, base: Path, run_dir: Path) -> dict:
    """Execute a fold-based cross-validation run and persist per-fold metrics.

    Args:
        cfgd: Resolved configuration dictionary.
        base: Config file directory.
        run_dir: Active run directory (created by the calling ``_eval_one``
            function).

    Returns:
        Dict with keys ``run_dir``, ``cross_validation`` (containing
        ``folds`` and ``metrics_summary``).

    Raises:
        ValueError: When ``eval.cross_validation.folds`` is empty or any fold
            is missing ``train_periods`` / ``validation_periods``.
    """
    cv_cfg = cfgd.get("eval", {}).get("cross_validation") or {}
    folds = cv_cfg.get("folds") or []
    if not folds:
        raise ValueError("eval.cross_validation.folds must define at least one fold.")

    validation_dir = run_dir / "validation"
    cv_dir = validation_dir / "cross_validation"
    cv_dir.mkdir(parents=True, exist_ok=True)

    fold_outputs: list[dict] = []
    for index, fold in enumerate(folds, start=1):
        fold_name = str(fold.get("name") or f"fold_{index}")
        train_periods = normalize_periods(fold.get("train_periods"))
        validation_periods = normalize_periods(fold.get("validation_periods"))
        if not train_periods or not validation_periods:
            raise ValueError(
                "Each cross-validation fold must provide train_periods and validation_periods."
            )

        _report_note(f"Running cross-validation {fold_name}")
        train_prepared = _prepare_fold_training_inputs(
            cfgd, base, train_periods, validation_periods, fold_name
        )
        pipeline = StationForecastPipeline(_build_model_config(cfgd))
        _suppress_pipeline_verbose(pipeline)
        pipeline.fit(train_prepared["train_X"], train_prepared["train_y"])

        eval_prepared = _prepare_fold_validation_inputs(
            cfgd,
            base,
            train_prepared["norm_stats"],
            validation_periods,
            fold_name,
        )
        result = pipeline.evaluate(
            eval_prepared["test_X"],
            eval_prepared["test_y"],
        )
        fold_dir = cv_dir / fold_name
        fold_dir.mkdir(parents=True, exist_ok=True)

        validation_df = pd.DataFrame(
            {"prediction_time": eval_prepared["test_rows"]["prediction_time"].values}
        )
        target_names = eval_prepared["target_names"]
        if len(target_names) == 1:
            validation_df["actual"] = result["actual"]
            validation_df["prediction"] = result["prediction"]
        else:
            for target_index, target_name in enumerate(target_names):
                validation_df[f"actual__{target_name}"] = result["actual"][:, target_index]
                validation_df[f"prediction__{target_name}"] = result["prediction"][:, target_index]

        validation_df.to_csv(fold_dir / "validation.csv", index=False)
        _save_json(fold_dir / "metrics.json", result["metrics"])

        fold_outputs.append(
            {
                "name": fold_name,
                "train_periods": _periods_to_serializable(train_periods),
                "validation_periods": _periods_to_serializable(validation_periods),
                "metrics": result["metrics"],
                "validation": str(fold_dir / "validation.csv"),
            }
        )

    metrics_df = pd.DataFrame(
        [{"fold": item["name"], **item["metrics"]} for item in fold_outputs]
    )
    metrics_df.to_csv(cv_dir / "metrics_summary.csv", index=False)
    summary_payload = {
        "folds": fold_outputs,
        "metrics_summary": str(cv_dir / "metrics_summary.csv"),
    }
    _save_json(cv_dir / "summary.json", summary_payload)
    return {
        "run_dir": str(run_dir),
        "cross_validation": summary_payload,
    }


