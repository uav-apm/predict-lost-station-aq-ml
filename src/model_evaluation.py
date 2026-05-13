"""Model evaluation, baseline comparison, and cross-config metric aggregation.

This module implements the *eval* workflow and all associated artefact helpers.
It depends on ``config_loader``, ``artifact_io``, ``data_pipeline``, and
``model_training`` but must not import from ``plot_writer``.

Responsibilities
----------------
- ``_eval_one`` — end-to-end single-config evaluation run.
- ``_evaluate_run`` — evaluation of a loaded pipeline against test data.
- Baseline metric helpers: ``_baseline_metrics``, ``_baseline_prediction_dataframe``,
  ``_compute_station_replacement_baselines``, ``_compute_station_replacement_predictions``.
- Prediction / detail dataframe builders: ``_prediction_dataframe``, ``_details_dataframe``.
- Plot-output helpers called during evaluation: ``_write_prediction_plots``,
  ``_write_imputation_plots``.
- Cross-config comparison helpers: ``_normalize_metrics_for_comparison``,
  ``_write_eval_comparison_csv``, ``_build_eval_comparison``.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .artifact_io import (
    _artifact_base_dir,
    _config_latest_marker_name,
    _ensure_config_alignment,
    _load_chunked_joblib,
    _load_saved_normalization_stats,
    _normalization_artifact_path,
    _resolve_algorithm_display_name,
    _resolve_run_dir,
    _save_json,
)
from .config_loader import RUN_LOGGER, _load_cfg, _report_note, _report_phase, _sanitize_marker_suffix
from .data import (
    apply_norm_stats,
    column_statistics_table,
    imputation_stats_table,
)
from .data_pipeline import (
    _normalization_type,
    _period_plot_suffix,
    _periods_to_serializable,
    _prepare_evaluation_data,
    _prepare_training_data,
    _should_use_training_normalization,
    _validation_period_columns,
)
from .forecasting import resolve_testing_periods, select_test_rows
from .metrics import r2, rmse
from .model_training import (
    _build_model_config,
    _run_cross_validation,
    _suppress_pipeline_verbose,
)
from .pipeline import StationForecastPipeline


# ---------------------------------------------------------------------------
# Prediction / detail dataframe builders
# ---------------------------------------------------------------------------


def _details_dataframe(
    test_rows: pd.DataFrame,
    preds,
    actual,
    target_names: list[str],
) -> pd.DataFrame:
    """Build a verbose per-row evaluation artefact with predictions and features.

    The output includes ``prediction_time``, ``actual`` / ``prediction`` (single
    target) or ``actual__<name>`` / ``prediction__<name>`` (multi-target), and
    all input feature columns prefixed with ``input__``.

    Args:
        test_rows: Supervised test rows dataframe (from the data pipeline).
        preds: Model predictions array.
        actual: Ground-truth target array.
        target_names: Ordered list of target column names.

    Returns:
        Flat dataframe suitable for CSV export.
    """
    out = pd.DataFrame({"prediction_time": test_rows["prediction_time"].values})
    if len(target_names) == 1:
        out["actual"] = actual
        out["prediction"] = preds
    else:
        for index, target_name in enumerate(target_names):
            out[f"actual__{target_name}"] = actual[:, index]
            out[f"prediction__{target_name}"] = preds[:, index]

    target_columns = set(target_names)
    for col in test_rows.columns:
        if col in {"feature_time", "prediction_time"} or col in target_columns:
            continue
        out[f"input__{col}"] = test_rows[col].values
    return out


def _prediction_dataframe(
    prediction_times,
    actual,
    prediction,
    target_names: list[str],
) -> pd.DataFrame:
    """Build a standardised prediction dataframe for single or multi-target outputs.

    Args:
        prediction_times: Array-like of prediction timestamps.
        actual: Ground-truth target array.
        prediction: Predicted values array.
        target_names: Ordered list of target column names.

    Returns:
        Dataframe with ``prediction_time`` plus ``actual`` / ``prediction``
        columns (single-target) or ``actual__<name>`` / ``prediction__<name>``
        columns (multi-target).
    """
    out = pd.DataFrame({"prediction_time": prediction_times})
    if len(target_names) == 1:
        out["actual"] = actual
        out["prediction"] = prediction
    else:
        for index, target_name in enumerate(target_names):
            out[f"actual__{target_name}"] = actual[:, index]
            out[f"prediction__{target_name}"] = prediction[:, index]
    return out


# ---------------------------------------------------------------------------
# Baseline metrics helpers
# ---------------------------------------------------------------------------


def _baseline_metrics(actual, prediction, target_names: list[str]) -> dict:
    """Compute a metric payload matching :meth:`StationForecastPipeline.evaluate`.

    Args:
        actual: Ground-truth target array.
        prediction: Predicted values array.
        target_names: Ordered list of target column names.

    Returns:
        Flat ``{RMSE, R2}`` dict for single-target outputs, or
        ``{by_target: {name: {…}}}`` for multi-target outputs.

    Raises:
        ValueError: When multi-target inputs are not 2-D.
    """
    actual_arr = np.asarray(actual, dtype=float)
    pred_arr = np.asarray(prediction, dtype=float)

    if len(target_names) <= 1:
        actual_series = actual_arr.reshape(-1)
        pred_series = pred_arr.reshape(-1)
        return {
            "RMSE": rmse(actual_series, pred_series),
            "R2": r2(actual_series, pred_series),
        }

    if actual_arr.ndim != 2 or pred_arr.ndim != 2:
        raise ValueError(
            "Multi-target baseline evaluation expects 2D actual/prediction arrays."
        )

    metrics_by_target: dict[str, dict[str, float]] = {}
    for index, target_name in enumerate(target_names):
        actual_target = actual_arr[:, index]
        pred_target = pred_arr[:, index]
        metrics_by_target[target_name] = {
            "RMSE": rmse(actual_target, pred_target),
            "R2": r2(actual_target, pred_target),
        }
    return {"by_target": metrics_by_target}


def _target_to_pollutant_name(target_name: str) -> str | None:
    """Extract the pollutant name from a multi-target column identifier.

    Args:
        target_name: Column name such as ``"target_value__PM2.5"``.

    Returns:
        Pollutant substring (e.g. ``"PM2.5"``), or ``None`` when the column
        does not follow the ``target_value__<name>`` convention.
    """
    if target_name.startswith("target_value__"):
        return target_name.split("__", 1)[1]
    return None


def _neighbor_feature_name(
    target_name: str,
    station_id: str,
    is_multi_target: bool,
) -> str:
    """Construct the neighbour feature column name for a station/target pair.

    Args:
        target_name: Target column name.
        station_id: Neighbour station identifier.
        is_multi_target: ``True`` when the model has multiple target columns.

    Returns:
        Feature column name string.

    Raises:
        ValueError: When *is_multi_target* is ``True`` but *target_name* does
            not match the expected naming convention.
    """
    if not is_multi_target:
        return f"neighbor__{station_id}"
    pollutant_name = _target_to_pollutant_name(target_name)
    if not pollutant_name:
        raise ValueError(
            f"Unexpected multi-target name format for baseline generation: {target_name}"
        )
    return f"neighbor__{pollutant_name}__{station_id}"


def _compute_station_replacement_predictions(
    cfgd: dict,
    test_rows: pd.DataFrame,
    actual,
    target_names: list[str],
) -> list[dict]:
    """Build baseline prediction arrays for mean-of-others and single-station replacement.

    Constructs one baseline per configured input station (``data.input_station_ids``)
    plus one "average of all other stations" baseline.

    Args:
        cfgd: Resolved configuration dictionary.
        test_rows: Supervised test rows dataframe.
        actual: Ground-truth target array.
        target_names: Ordered list of target column names.

    Returns:
        List of dicts, each with keys ``id``, ``name``, ``algorithm_name``,
        ``prediction`` (numpy array).
    """
    input_station_ids = [
        str(station_id)
        for station_id in (cfgd.get("data", {}).get("input_station_ids") or [])
    ]
    if not input_station_ids or len(target_names) == 0:
        return []

    actual_arr = np.asarray(actual, dtype=float)
    if len(target_names) == 1:
        actual_arr = actual_arr.reshape(-1)
    elif actual_arr.ndim == 1:
        actual_arr = actual_arr.reshape(-1, len(target_names))

    is_multi_target = len(target_names) > 1

    baseline_specs = [
        {
            "id": "baseline_average_of_other_stations",
            "name": "Baseline: Average of Other Stations",
            "station_id": None,
        }
    ]
    baseline_specs.extend(
        {
            "id": f"baseline_replace_with_station_{_sanitize_marker_suffix(station_id)}",
            "name": f"Baseline: Replace with Station {station_id}",
            "station_id": station_id,
        }
        for station_id in input_station_ids
    )

    results: list[dict] = []
    for spec in baseline_specs:
        if len(target_names) == 1:
            target_name = target_names[0]
            if spec["station_id"] is None:
                candidate_cols = [
                    _neighbor_feature_name(target_name, sid, is_multi_target)
                    for sid in input_station_ids
                ]
                available_cols = [c for c in candidate_cols if c in test_rows.columns]
                if not available_cols:
                    continue
                prediction = test_rows[available_cols].mean(axis=1).to_numpy(dtype=float)
            else:
                col = _neighbor_feature_name(
                    target_name, str(spec["station_id"]), is_multi_target
                )
                if col not in test_rows.columns:
                    continue
                prediction = test_rows[col].to_numpy(dtype=float)
        else:
            prediction = np.zeros_like(actual_arr, dtype=float)
            skip_baseline = False
            for target_index, target_name in enumerate(target_names):
                if spec["station_id"] is None:
                    candidate_cols = [
                        _neighbor_feature_name(target_name, sid, is_multi_target)
                        for sid in input_station_ids
                    ]
                    available_cols = [c for c in candidate_cols if c in test_rows.columns]
                    if not available_cols:
                        skip_baseline = True
                        break
                    prediction[:, target_index] = (
                        test_rows[available_cols].mean(axis=1).to_numpy(dtype=float)
                    )
                else:
                    col = _neighbor_feature_name(
                        target_name, str(spec["station_id"]), is_multi_target
                    )
                    if col not in test_rows.columns:
                        skip_baseline = True
                        break
                    prediction[:, target_index] = test_rows[col].to_numpy(dtype=float)
            if skip_baseline:
                continue

        results.append(
            {
                "id": spec["id"],
                "name": spec["name"],
                "algorithm_name": spec["name"],
                "prediction": prediction,
            }
        )

    return results


def _compute_station_replacement_baselines(
    cfgd: dict,
    test_rows: pd.DataFrame,
    actual,
    target_names: list[str],
) -> list[dict]:
    """Build baseline metrics for mean-of-others and per-station replacement.

    Args:
        cfgd: Resolved configuration dictionary.
        test_rows: Supervised test rows dataframe.
        actual: Ground-truth target array.
        target_names: Ordered list of target column names.

    Returns:
        List of dicts with keys ``id``, ``name``, ``algorithm_name``, ``metrics``.
    """
    baseline_predictions = _compute_station_replacement_predictions(
        cfgd, test_rows, actual, target_names
    )
    actual_arr = np.asarray(actual, dtype=float)
    if len(target_names) == 1:
        actual_arr = actual_arr.reshape(-1)
    elif actual_arr.ndim == 1:
        actual_arr = actual_arr.reshape(-1, len(target_names))

    return [
        {
            "id": baseline["id"],
            "name": baseline["name"],
            "algorithm_name": baseline["algorithm_name"],
            "metrics": _baseline_metrics(actual_arr, baseline["prediction"], target_names),
        }
        for baseline in baseline_predictions
    ]


def _baseline_prediction_dataframe(
    prediction_times,
    actual,
    baseline_predictions: list[dict],
    target_names: list[str],
) -> pd.DataFrame:
    """Build a tabular baseline-prediction artefact consumable by plotting.

    Args:
        prediction_times: Array-like of prediction timestamps.
        actual: Ground-truth target array.
        baseline_predictions: List of baseline dicts as returned by
            :func:`_compute_station_replacement_predictions`.
        target_names: Ordered list of target column names.

    Returns:
        Dataframe with ``prediction_time``, ``actual`` / ``actual__<name>``, and
        ``prediction__<baseline_id>`` / ``prediction__<baseline_id>__<name>``
        columns.
    """
    out = pd.DataFrame({"prediction_time": prediction_times})
    actual_arr = np.asarray(actual, dtype=float)

    if len(target_names) == 1:
        out["actual"] = actual_arr.reshape(-1)
        for baseline in baseline_predictions:
            baseline_id = str(baseline["id"])
            prediction = np.asarray(baseline["prediction"], dtype=float).reshape(-1)
            out[f"prediction__{baseline_id}"] = prediction
        return out

    if actual_arr.ndim == 1:
        actual_arr = actual_arr.reshape(-1, len(target_names))
    for target_index, target_name in enumerate(target_names):
        out[f"actual__{target_name}"] = actual_arr[:, target_index]

    for baseline in baseline_predictions:
        baseline_id = str(baseline["id"])
        prediction = np.asarray(baseline["prediction"], dtype=float)
        if prediction.ndim == 1:
            prediction = prediction.reshape(-1, len(target_names))
        for target_index, target_name in enumerate(target_names):
            out[f"prediction__{baseline_id}__{target_name}"] = prediction[:, target_index]

    return out


# ---------------------------------------------------------------------------
# Eval run orchestration
# ---------------------------------------------------------------------------


def _resolve_eval_run_dir_arg(cfgd: dict, run_dir_arg: str | None) -> str | None:
    """Return the effective eval run-directory argument, respecting config overrides."""
    return run_dir_arg or cfgd.get("artifacts", {}).get("eval_run_dir")


def _evaluate_run(
    cfgd: dict,
    base: Path,
    run_dir: Path,
    *,
    use_training_normalization: bool | None = None,
) -> dict:
    """Evaluate a trained pipeline against the configured test window.

    Loads the serialised pipeline from *run_dir*, runs predictions on prepared
    evaluation data, computes metrics, writes validation artefacts, and
    optionally generates per-period breakdowns.

    Args:
        cfgd: Resolved configuration dictionary.
        base: Config file directory.
        run_dir: Run directory containing a ``pipeline.pkl`` artefact.
        use_training_normalization: Override for ``eval.use_training_normalization``.

    Returns:
        Dict with keys ``run_dir``, ``metrics``, ``validation``,
        ``training_predictions``, ``details``, ``normalization_source``,
        ``normalization_stats_file``, ``test_imputation``, ``baseline_results``,
        and optionally ``period_results`` / ``period_results_summary`` /
        ``baseline_period_results``.

    Raises:
        ValueError: When ``eval.only_separate_validation_period_results`` is
            configured but fewer than two testing periods are available.
    """
    total_phases = 6
    normalization_method = _normalization_type(cfgd["data"])
    use_saved_normalization = _should_use_training_normalization(
        cfgd.get("eval"),
        use_training_normalization,
    )
    norm_stats = None
    norm_stats_path = None
    if use_saved_normalization:
        norm_stats = _load_saved_normalization_stats(
            run_dir, normalization_method=normalization_method
        )
        candidate_path = _normalization_artifact_path(run_dir)
        if candidate_path.exists():
            norm_stats_path = candidate_path

    prepared = _prepare_evaluation_data(cfgd, base, norm_stats)
    _report_phase("Building correlation analysis for evaluation years", 3, total_phases)
    _report_phase(
        "Applying configurable missing-value strategy to evaluation years",
        4,
        total_phases,
    )
    _report_note(
        "Evaluation-side dropped rows with missing values in configured columns: "
        + str(prepared["imputation_details"].get("dropped_rows", 0))
    )

    pipeline = _load_chunked_joblib(run_dir / "pipeline.pkl")
    _suppress_pipeline_verbose(pipeline)

    _report_phase("Running evaluation on configured test window", 5, total_phases)
    result = pipeline.evaluate(
        prepared["test_X"],
        prepared["test_y"],
    )
    baseline_predictions = _compute_station_replacement_predictions(
        cfgd,
        prepared["test_rows"],
        result["actual"],
        prepared["target_names"],
    )
    baseline_results = [
        {
            "id": baseline["id"],
            "name": baseline["name"],
            "algorithm_name": baseline["algorithm_name"],
            "metrics": _baseline_metrics(
                result["actual"], baseline["prediction"], prepared["target_names"]
            ),
        }
        for baseline in baseline_predictions
    ]

    _report_note("Preparing training-window predictions for scatter plot artifacts.")
    training_prepared = _prepare_training_data(cfgd, base)
    training_X = apply_norm_stats(training_prepared["train_X_raw"], prepared["norm_stats"])
    training_prediction = pipeline.predict(training_X)
    training_actual = (
        training_prepared["train_y"].to_numpy()
        if isinstance(training_prepared["train_y"], (pd.Series, pd.DataFrame))
        else training_prepared["train_y"]
    )

    _report_phase("Writing validation files and metrics", 6, total_phases)
    validation_dir = run_dir / "validation"
    validation_dir.mkdir(parents=True, exist_ok=True)
    eval_cfg = cfgd.get("eval", {})
    validation_periods = resolve_testing_periods(cfgd.get("data", {}))
    separate_period_results = bool(eval_cfg.get("separate_validation_period_results", False))
    only_separate_period_results = bool(
        eval_cfg.get("only_separate_validation_period_results", False)
    )
    if only_separate_period_results:
        separate_period_results = True
    separate_mode_active = separate_period_results and len(validation_periods) >= 1
    if only_separate_period_results and not separate_mode_active:
        raise ValueError(
            "eval.only_separate_validation_period_results requires at least two "
            "configured testing_periods (or legacy validation_periods)."
        )

    target_names = prepared["target_names"]
    validation_df = _prediction_dataframe(
        prepared["test_rows"]["prediction_time"].values,
        result["actual"],
        result["prediction"],
        target_names,
    )
    validation_df.to_csv(validation_dir / "validation.csv", index=False)

    training_prediction_path = validation_dir / "training_predictions.csv"
    training_prediction_df = _prediction_dataframe(
        training_prepared["train_rows"]["prediction_time"].values,
        training_actual,
        training_prediction,
        target_names,
    )
    training_prediction_df.to_csv(training_prediction_path, index=False)

    _save_json(validation_dir / "metrics.json", result["metrics"])
    _save_json(validation_dir / "baseline_results.json", baseline_results)
    if baseline_predictions:
        baseline_prediction_df = _baseline_prediction_dataframe(
            prepared["test_rows"]["prediction_time"].values,
            result["actual"],
            baseline_predictions,
            target_names,
        )
        baseline_prediction_df.to_csv(
            validation_dir / "baseline_predictions.csv", index=False
        )

    details_file = None
    if bool(cfgd["eval"].get("store_details", True)):
        details_file = validation_dir / "validation_details.csv"
        _details_dataframe(
            prepared["test_rows"],
            result["prediction"],
            result["actual"],
            target_names,
        ).to_csv(details_file, index=False)

    test_imputation_details = validation_dir / "test_imputation_details.json"
    test_imputation_stats = validation_dir / "test_imputation_stats.json"
    test_imputation_stats_csv = validation_dir / "test_imputation_stats.csv"
    column_statistics_before_csv = validation_dir / "test_column_statistics_before.csv"
    column_statistics_after_csv = validation_dir / "test_column_statistics_after.csv"

    _save_json(test_imputation_details, prepared["imputation_details"])
    _save_json(
        test_imputation_stats,
        {"before": prepared["stats_before"], "after": prepared["stats_after"]},
    )
    imputation_stats_table(
        prepared["stats_before"], prepared["stats_after"], prepared["imputation_details"]
    ).to_csv(test_imputation_stats_csv, index=False)
    column_statistics_table(prepared["column_stats_before"], "before").to_csv(
        column_statistics_before_csv, index=False
    )
    column_statistics_table(prepared["column_stats_after"], "after").to_csv(
        column_statistics_after_csv, index=False
    )

    output = {
        "run_dir": str(run_dir),
        "metrics": result["metrics"],
        "validation": str(validation_dir / "validation.csv"),
        "training_predictions": str(training_prediction_path),
        "details": str(details_file) if details_file else None,
        "normalization_source": prepared["normalization_source"],
        "normalization_stats_file": str(norm_stats_path) if norm_stats_path else None,
        "test_imputation": {
            "performed": bool(prepared["imputation_details"].get("columns")),
            "details_file": str(test_imputation_details),
            "stats_file": str(test_imputation_stats),
            "stats_csv": str(test_imputation_stats_csv),
            "cache_dir": str(prepared["cache_dir"]),
        },
        "baseline_results": baseline_results,
    }

    if separate_mode_active:
        _report_note("Generating separate evaluation outputs per validation period.")
        by_period_dir = validation_dir / "by_period"
        by_period_dir.mkdir(parents=True, exist_ok=True)

        period_outputs: list[dict] = []
        summary_rows: list[dict] = []
        baseline_period_outputs_by_id: dict[str, dict] = {}

        for index, period in enumerate(validation_periods, start=1):
            period_data_cfg = dict(cfgd["data"])
            period_data_cfg["testing_periods"] = _periods_to_serializable([period])
            period_data_cfg.pop("validation_periods", None)
            period_mask = select_test_rows(prepared["supervised_df"], period_data_cfg)
            period_rows = prepared["supervised_df"].loc[period_mask].reset_index(drop=True)

            period_suffix = _period_plot_suffix(period, index)
            period_name = f"period_{index:02d}"
            period_dir = by_period_dir / f"{period_name}_{period_suffix}"
            period_dir.mkdir(parents=True, exist_ok=True)

            period_payload: dict = {
                "name": period_name,
                "period": _periods_to_serializable([period])[0],
                "row_count": int(len(period_rows)),
                "validation": str(period_dir / "validation.csv"),
                "metrics_file": str(period_dir / "metrics.json"),
                "details": (
                    str(period_dir / "validation_details.csv")
                    if bool(eval_cfg.get("store_details", True))
                    else None
                ),
            }

            if len(period_rows) == 0:
                period_payload["skipped"] = True
                period_payload["reason"] = "No rows matched this validation period."
                period_outputs.append(period_payload)
                continue

            period_X_raw = period_rows[prepared["feature_names"]].copy()
            period_X = apply_norm_stats(period_X_raw, norm_stats)
            if len(target_names) == 1:
                period_y = period_rows[target_names[0]].copy()
            else:
                period_y = period_rows[target_names].copy()

            period_result = pipeline.evaluate(
                period_X,
                period_y,
            )

            period_validation_df = pd.DataFrame(
                {"prediction_time": period_rows["prediction_time"].values}
            )
            if len(target_names) == 1:
                period_validation_df["actual"] = period_result["actual"]
                period_validation_df["prediction"] = period_result["prediction"]
            else:
                for target_index, target_name in enumerate(target_names):
                    period_validation_df[f"actual__{target_name}"] = (
                        period_result["actual"][:, target_index]
                    )
                    period_validation_df[f"prediction__{target_name}"] = (
                        period_result["prediction"][:, target_index]
                    )

            period_validation_df.to_csv(period_dir / "validation.csv", index=False)
            _save_json(period_dir / "metrics.json", period_result["metrics"])

            period_details_file = None
            if bool(eval_cfg.get("store_details", True)):
                period_details_file = period_dir / "validation_details.csv"
                _details_dataframe(
                    period_rows,
                    period_result["prediction"],
                    period_result["actual"],
                    target_names,
                ).to_csv(period_details_file, index=False)

            period_payload.update(
                {
                    "metrics": period_result["metrics"],
                    "details": str(period_details_file) if period_details_file else None,
                }
            )
            period_outputs.append(period_payload)

            metrics_for_summary = (
                _normalize_metrics_for_comparison({"metrics": period_result["metrics"]}) or {}
            )
            target_station_id = cfgd["data"].get("target_station_id", "unknown")
            summary_rows.append(
                {
                    "target_station_id": str(target_station_id),
                    **_validation_period_columns(
                        period_payload["period"], period_name
                    ),
                    "row_count": int(len(period_rows)),
                    **metrics_for_summary,
                }
            )

            baseline_period_results = _compute_station_replacement_baselines(
                cfgd,
                period_rows,
                period_result["actual"],
                target_names,
            )
            for baseline in baseline_period_results:
                entry = baseline_period_outputs_by_id.setdefault(
                    baseline["id"],
                    {
                        "id": baseline["id"],
                        "name": baseline["name"],
                        "algorithm_name": baseline["algorithm_name"],
                        "period_results": [],
                    },
                )
                entry["period_results"].append(
                    {
                        "name": period_name,
                        "period": _periods_to_serializable([period])[0],
                        "row_count": int(len(period_rows)),
                        "metrics": baseline["metrics"],
                    }
                )

        summary_csv = None
        if summary_rows:
            summary_csv = by_period_dir / "metrics_summary.csv"
            pd.DataFrame(summary_rows).to_csv(summary_csv, index=False)

        output["period_results"] = period_outputs
        output["period_results_summary"] = str(summary_csv) if summary_csv else None
        baseline_period_results_payload = list(baseline_period_outputs_by_id.values())
        _save_json(
            by_period_dir / "baseline_period_results.json",
            baseline_period_results_payload,
        )
        output["baseline_period_results"] = baseline_period_results_payload

    if only_separate_period_results:
        for path in [
            validation_dir / "validation.csv",
            validation_dir / "metrics.json",
            validation_dir / "validation_details.csv",
            validation_dir / "test_imputation_details.json",
            validation_dir / "test_imputation_stats.json",
            validation_dir / "test_imputation_stats.csv",
            validation_dir / "test_column_statistics_before.csv",
            validation_dir / "test_column_statistics_after.csv",
        ]:
            path.unlink(missing_ok=True)

        output["combined_result_disabled"] = True
        output["metrics"] = None
        output["validation"] = None
        output["details"] = None
        output["test_imputation"] = None

    if only_separate_period_results:
        _report_note(
            "Evaluation complete. Combined outputs disabled; period outputs written under "
            + str(validation_dir / "by_period")
        )
    else:
        _report_note(
            f"Evaluation complete. Metrics written to {validation_dir / 'metrics.json'}"
        )
    return output


def _eval_one(
    config_path: str,
    run_dir_arg: str | None = None,
    use_training_normalization: bool | None = None,
) -> dict:
    """Execute a complete evaluation run for a single config file.

    Phases
    ------
    1. Load configuration and resolve run directory.
    2. Preprocess evaluation-year data (imputation / drop).
    3. Build correlation analysis for evaluation years.
    4. Apply missing-value strategy to evaluation years.
    5. Run evaluation on configured test window.
    6. Write validation files and metrics.

    Dispatches to :func:`_run_cross_validation` when
    ``eval.cross_validation.folds`` is configured.

    Args:
        config_path: Path to the YAML configuration file.
        run_dir_arg: Optional explicit run directory or latest-marker alias.
        use_training_normalization: Override for
            ``eval.use_training_normalization``.

    Returns:
        Dict merging :func:`_evaluate_run` (or
        :func:`_run_cross_validation`) output with ``config``,
        ``target_station_id``, ``algorithm``, ``algorithm_name``,
        ``log_file``.
    """
    total_phases = 6
    RUN_LOGGER.start(config_path)
    _report_note(f"Starting evaluation run for config: {Path(config_path).name}")
    _report_phase(
        "Loading configuration and resolving run directory", 1, total_phases
    )
    cfgd, base = _load_cfg(config_path)
    algorithm_name = _resolve_algorithm_display_name(cfgd)
    artifacts_base = _artifact_base_dir(cfgd, base)
    resolved_run_arg = _resolve_eval_run_dir_arg(cfgd, run_dir_arg)
    marker_name = _config_latest_marker_name(config_path)
    run_dir = _resolve_run_dir(
        artifacts_base,
        resolved_run_arg,
        latest_markers=[marker_name],
        include_default_latest_marker=False,
    )
    eval_log_file = (
        run_dir
        / "validation"
        / f"eval_{_sanitize_marker_suffix(Path(config_path).stem)}.log"
    )
    RUN_LOGGER.set_log_file(eval_log_file)
    _report_note(f"Persisting evaluation logs to {eval_log_file}")
    run_cfg = _load_run_config_from_dir(run_dir)
    _ensure_config_alignment(cfgd, run_cfg)

    if cfgd.get("eval", {}).get("cross_validation", {}).get("folds"):
        return {
            "config": str(Path(config_path).resolve()),
            "target_station_id": cfgd.get("data", {}).get("target_station_id"),
            "algorithm": cfgd.get("model", {}).get("algorithm"),
            "algorithm_name": algorithm_name,
            "log_file": str(eval_log_file),
            **_run_cross_validation(cfgd, base, run_dir),
        }

    _report_phase("Preprocessing evaluation-year data", 2, total_phases)
    output = _evaluate_run(
        cfgd,
        base,
        run_dir,
        use_training_normalization=use_training_normalization,
    )
    return {
        "config": str(Path(config_path).resolve()),
        "target_station_id": cfgd.get("data", {}).get("target_station_id"),
        "algorithm": cfgd.get("model", {}).get("algorithm"),
        "algorithm_name": algorithm_name,
        "log_file": str(eval_log_file),
        **output,
    }


def _load_run_config_from_dir(run_dir: Path) -> dict:
    """Load the config snapshot saved during training.

    Thin shim that avoids importing ``_load_run_config`` from ``artifact_io``
    inside :func:`_eval_one` while keeping artifact-reading logic centralised.

    Args:
        run_dir: Run directory containing ``run_metadata.json``.

    Returns:
        Metadata dict, or an empty dict when the file is absent.
    """
    from .artifact_io import _load_run_config  # local to avoid top-level cycle

    return _load_run_config(run_dir)


# ---------------------------------------------------------------------------
# Cross-config metric comparison helpers
# ---------------------------------------------------------------------------


def _normalize_metrics_for_comparison(result: dict) -> dict | None:
    """Extract a flat, comparable metrics payload from one evaluation result.

    Handles flat metrics dicts, multi-target ``by_target`` payloads, and
    cross-validation results (returning fold averages).

    Args:
        result: Evaluation result dict (must contain a ``"metrics"`` or
            ``"cross_validation"`` key).

    Returns:
        Flat ``{metric_name: float}`` dict, or ``None`` when no numeric
        metrics can be extracted.
    """

    def _is_numeric(value) -> bool:
        return isinstance(value, (int, float)) and not isinstance(value, bool)

    def _flatten_metrics(metrics_payload: dict, prefix: str = "") -> dict[str, float]:
        flattened: dict[str, float] = {}
        for key, value in metrics_payload.items():
            key_name = str(key)
            flat_key = f"{prefix}.{key_name}" if prefix else key_name
            if isinstance(value, dict):
                flattened.update(_flatten_metrics(value, flat_key))
            elif _is_numeric(value):
                flattened[flat_key] = float(value)
        return flattened

    metrics = result.get("metrics")
    if isinstance(metrics, dict):
        flattened = _flatten_metrics(metrics)
        return flattened or None

    cv_payload = result.get("cross_validation")
    if isinstance(cv_payload, dict):
        folds = cv_payload.get("folds") or []
        if folds:
            fold_metrics = [
                _normalize_metrics_for_comparison({"metrics": item.get("metrics")})
                for item in folds
            ]
            fold_metrics = [
                m for m in fold_metrics if isinstance(m, dict)
            ]
            if fold_metrics:
                metric_keys: set[str] = set()
                for m in fold_metrics:
                    metric_keys.update(m.keys())

                averaged: dict[str, float] = {}
                for metric_key in sorted(metric_keys):
                    values = [
                        float(m[metric_key])
                        for m in fold_metrics
                        if metric_key in m
                    ]
                    if values:
                        averaged[metric_key] = float(sum(values) / len(values))

                if averaged:
                    averaged["cv_fold_count"] = float(len(fold_metrics))
                    return averaged

    return None


def _write_eval_comparison_csv(
    comparison: dict,
    output_path_arg: str | None,
    config_dirs: list[str] | None = None,
) -> str:
    """Persist cross-config evaluation comparison rows as a CSV file.

    Args:
        comparison: Dict produced by :func:`_build_eval_comparison`.
        output_path_arg: Explicit output path, or ``None`` to auto-generate.
        config_dirs: List of config directories used to derive an automatic
            output location.

    Returns:
        Resolved absolute path string of the written CSV.

    Raises:
        ValueError: When *comparison* contains no rows.
    """
    rows = comparison.get("rows") or []
    if not rows:
        raise ValueError("Comparison rows are required to write a CSV file.")

    if output_path_arg:
        output_path = Path(output_path_arg)
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if config_dirs:
            base_dir = Path(config_dirs[0])
            output_path = base_dir / "outputs" / f"evaluation_comparison_{stamp}.csv"
        else:
            output_path = Path.cwd() / f"evaluation_comparison_{stamp}.csv"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)
    return str(output_path.resolve())


def _build_eval_comparison(results: list[dict]) -> dict | None:
    """Build a compact cross-config comparison table for multi-config evaluation.

    Includes model predictions, baselines, per-period breakdowns, and overall
    metrics in a unified flat-row format.

    Args:
        results: List of evaluation result dicts as returned by
            :func:`_eval_one`.

    Returns:
        Dict with keys ``rows`` and ``ranked_by_rmse``, or ``None`` when all
        results produce no comparable metrics.
    """

    def _comparison_sort_key(row: dict) -> tuple:
        return (
            str(row.get("target_station_id") or ""),
            str(row.get("validation_period_start") or ""),
            str(row.get("validation_period_end") or ""),
            str(row.get("validation_period_name") or ""),
            str(row.get("algorithm") or ""),
            str(row.get("config") or ""),
        )

    rows: list[dict] = []
    for result in results:
        config_value = result.get("config")
        config_label = (
            Path(config_value).name
            if isinstance(config_value, str) and config_value
            else config_value
        )
        base_row = {
            "config": config_label,
            "run_dir": result.get("run_dir"),
            "target_station_id": result.get("target_station_id"),
            "algorithm": result.get("algorithm_name") or result.get("algorithm"),
            "best_params": result.get("best_params"),
        }

        period_results = result.get("period_results")
        if isinstance(period_results, list) and period_results:
            for period_result in period_results:
                metrics = _normalize_metrics_for_comparison(period_result)
                if not metrics:
                    continue
                row = {
                    **base_row,
                    **_validation_period_columns(
                        period_result.get("period"), period_result.get("name")
                    ),
                    **metrics,
                }
                if "row_count" in period_result:
                    row["row_count"] = period_result["row_count"]
                rows.append(row)

            baseline_period_results = result.get("baseline_period_results")
            if isinstance(baseline_period_results, list):
                for baseline in baseline_period_results:
                    baseline_algorithm = (
                        baseline.get("algorithm_name")
                        or baseline.get("name")
                        or baseline.get("id")
                    )
                    for period_result in baseline.get("period_results") or []:
                        metrics = _normalize_metrics_for_comparison(period_result)
                        if not metrics:
                            continue
                        row = {
                            **base_row,
                            "algorithm": baseline_algorithm,
                            "best_params": None,
                            **_validation_period_columns(
                                period_result.get("period"), period_result.get("name")
                            ),
                            **metrics,
                        }
                        if "row_count" in period_result:
                            row["row_count"] = period_result["row_count"]
                        rows.append(row)

            overall_metrics = _normalize_metrics_for_comparison(result)
            if overall_metrics:
                rows.append(
                    {
                        **base_row,
                        "validation_period_name": "overall",
                        "validation_period_start": "",
                        "validation_period_end": "",
                        "validation_period": "overall",
                        **overall_metrics,
                    }
                )

            baseline_results = result.get("baseline_results")
            if isinstance(baseline_results, list):
                for baseline in baseline_results:
                    baseline_metrics = _normalize_metrics_for_comparison(baseline)
                    if not baseline_metrics:
                        continue
                    baseline_algorithm = (
                        baseline.get("algorithm_name")
                        or baseline.get("name")
                        or baseline.get("id")
                    )
                    rows.append(
                        {
                            **base_row,
                            "algorithm": baseline_algorithm,
                            "best_params": None,
                            "validation_period_name": "overall",
                            "validation_period_start": "",
                            "validation_period_end": "",
                            "validation_period": "overall",
                            **baseline_metrics,
                        }
                    )
            continue

        metrics = _normalize_metrics_for_comparison(result)
        if metrics:
            rows.append({**base_row, **metrics})

        baseline_results = result.get("baseline_results")
        if isinstance(baseline_results, list):
            for baseline in baseline_results:
                baseline_metrics = _normalize_metrics_for_comparison(baseline)
                if not baseline_metrics:
                    continue
                baseline_algorithm = (
                    baseline.get("algorithm_name")
                    or baseline.get("name")
                    or baseline.get("id")
                )
                rows.append(
                    {
                        **base_row,
                        "algorithm": baseline_algorithm,
                        "best_params": None,
                        **baseline_metrics,
                    }
                )

    if not rows:
        return None

    if all("RMSE" in row for row in rows):
        ranked = sorted(rows, key=lambda item: float(item["RMSE"]))
        for idx, row in enumerate(ranked, start=1):
            row["rank_by_rmse"] = idx

    rows.sort(key=_comparison_sort_key)
    return {"rows": rows, "ranked_by_rmse": all("rank_by_rmse" in row for row in rows)}
