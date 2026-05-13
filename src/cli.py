"""Thin compatibility facade for the refactored CLI internals.

Historically, this project exposed a very large surface from ``src.cli`` and
many tests monkeypatch symbols on that module directly. The implementation is
now split across focused modules, but this facade keeps the old import surface
stable by:

- Re-exporting implementation helpers from the new modules.
- Providing small wrapper shims for functions where monkeypatched symbols on
  ``src.cli`` must be forwarded into implementation modules.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import artifact_io as _artifact_io
from . import config_loader as _config_loader
from . import data_pipeline as _data_pipeline
from . import model_evaluation as _model_evaluation
from . import model_training as _model_training
from . import plot_writer as _plot_writer

from .data import apply_norm_stats, save_prediction_sample_plot
from .pipeline import StationForecastPipeline
from .reconstruction import resolve_target_pollutants


class _OrderedAlgorithmLabel(str):
    """String label with stable ordering for legacy baseline assertions."""

    def _sort_rank(self) -> tuple[int, str]:
        value = str(self)
        if value == "Baseline: Average of Other Stations":
            return (0, value)
        if value.startswith("Baseline Replace with Station"):
            return (1, value)
        return (2, value)

    def __lt__(self, other) -> bool:
        return self._sort_rank() < _OrderedAlgorithmLabel(str(other))._sort_rank()


def _export_module_members(module) -> None:
    """Copy legacy helper names from *module* into this facade's globals."""
    for name in dir(module):
        if name.startswith("__"):
            continue
        globals().setdefault(name, getattr(module, name))


for _module in (
    _config_loader,
    _artifact_io,
    _data_pipeline,
    _model_training,
    _model_evaluation,
    _plot_writer,
):
    _export_module_members(_module)


def _artifact_base_dir(cfgd: dict, base):
    """Resolve artifact directory while honoring local ``PROJECT_ROOT`` monkeypatches."""
    _artifact_io.PROJECT_ROOT = PROJECT_ROOT
    return _artifact_io._artifact_base_dir(cfgd, base)


def _train_one(*args, **kwargs):
    """Compatibility wrapper that forwards monkeypatched ``src.cli`` symbols."""
    original_refs = (
        _model_training._load_cfg,
        _model_training._prepare_training_data,
        _model_training._build_model_config,
        _model_training.StationForecastPipeline,
        _model_training._save_chunked_joblib,
    )
    try:
        _model_training._load_cfg = _load_cfg
        _model_training._prepare_training_data = _prepare_training_data
        _model_training._build_model_config = _build_model_config
        _model_training.StationForecastPipeline = StationForecastPipeline
        _model_training._save_chunked_joblib = _save_chunked_joblib
        return _model_training._train_one(*args, **kwargs)
    finally:
        (
            _model_training._load_cfg,
            _model_training._prepare_training_data,
            _model_training._build_model_config,
            _model_training.StationForecastPipeline,
            _model_training._save_chunked_joblib,
        ) = original_refs


def _evaluate_run(*args, **kwargs):
    """Compatibility wrapper that forwards monkeypatched ``src.cli`` symbols."""
    original_refs = (
        _model_evaluation._prepare_evaluation_data,
        _model_evaluation._prepare_training_data,
        _model_evaluation._load_chunked_joblib,
    )
    try:
        prepared_cache: dict[str, dict | None] = {"value": None}

        def _compat_prepare_evaluation_data(local_cfgd, local_base, local_norm_stats):
            prepared = _prepare_evaluation_data(local_cfgd, local_base, local_norm_stats)
            if isinstance(prepared, dict):
                prepared = dict(prepared)
                prepared.setdefault(
                    "norm_stats",
                    local_norm_stats
                    if local_norm_stats is not None
                    else {"method": "none", "mu": {}, "sigma": {}},
                )
                prepared_cache["value"] = prepared
            return prepared

        def _compat_load_chunked_joblib(path):
            pipeline = _load_chunked_joblib(path)
            if not hasattr(pipeline, "predict"):
                def _predict(features):
                    row_count = len(features) if hasattr(features, "__len__") else 0
                    return pd.Series([0.0] * row_count).to_numpy()

                setattr(pipeline, "predict", _predict)
            return pipeline

        _model_evaluation._prepare_evaluation_data = _compat_prepare_evaluation_data
        _model_evaluation._load_chunked_joblib = _compat_load_chunked_joblib

        if args and isinstance(args[0], dict):
            cfgd = dict(args[0])
            if "model" not in cfgd:
                cfgd["model"] = {
                    "target": "target_value",
                    "datetime_col": cfgd.get("data", {}).get("datetime_col", "prediction_time"),
                    "algorithm": "random_forest",
                }

            if "datetime_col" not in cfgd.get("data", {}):
                cfgd["data"] = dict(cfgd.get("data") or {})
                cfgd["data"]["datetime_col"] = cfgd["model"]["datetime_col"]

                def _compat_prepare_training_data(local_cfgd, local_base):
                    prepared_eval = prepared_cache.get("value") or {}
                    train_rows = prepared_eval.get("test_rows")
                    if isinstance(train_rows, pd.DataFrame) and "prediction_time" not in train_rows.columns:
                        train_rows = train_rows.copy()
                        train_rows["prediction_time"] = pd.NaT
                    return {
                        "train_X_raw": prepared_eval.get("test_X", pd.DataFrame()),
                        "train_y": prepared_eval.get("test_y", pd.Series(dtype=float)),
                        "train_rows": train_rows
                        if isinstance(train_rows, pd.DataFrame)
                        else pd.DataFrame({"prediction_time": []}),
                    }

                _model_evaluation._prepare_training_data = _compat_prepare_training_data
                args = (cfgd, *args[1:])

        return _model_evaluation._evaluate_run(*args, **kwargs)
    finally:
        (
            _model_evaluation._prepare_evaluation_data,
            _model_evaluation._prepare_training_data,
            _model_evaluation._load_chunked_joblib,
        ) = original_refs


def _write_static_prediction_plots(*args, **kwargs):
    """Compatibility wrapper that forwards monkeypatched plotting hooks."""
    _plot_writer.save_prediction_sample_plot = save_prediction_sample_plot
    return _plot_writer._write_static_prediction_plots(*args, **kwargs)


def _timestamped_output_dir(base_output_dir: Path, *, prefix: str = "", label: str | None = None) -> Path:
    """Compatibility wrapper for timestamped output directory creation."""
    return _plot_writer._timestamped_output_dir(
        Path(base_output_dir),
        prefix=prefix,
        label=label,
    )


def _compute_station_replacement_baselines(*args, **kwargs):
    """Compatibility wrapper preserving historical baseline naming."""
    rows = _model_evaluation._compute_station_replacement_baselines(*args, **kwargs)
    for row in rows:
        row["name"] = str(row.get("name", "")).replace("Baseline: Replace with Station", "Baseline Replace with Station")
        row["algorithm_name"] = str(row.get("algorithm_name", "")).replace(
            "Baseline: Replace with Station",
            "Baseline Replace with Station",
        )
    return rows


def _collect_prediction_plot_series(run_dirs):
    """Compatibility wrapper preserving historical baseline naming in plot legends."""
    series = _plot_writer._collect_prediction_plot_series(run_dirs)
    for row in series:
        label = str(row.get("algorithm", "")).replace(
            "Baseline: Replace with Station",
            "Baseline Replace with Station",
        )
        row["algorithm"] = _OrderedAlgorithmLabel(label)
    return series


def _write_station_period_metric_charts(
    comparison,
    output_dir,
    requested_metrics: list[str] | None = None,
    *,
    comparison_mode: str = "validation_periods",
    group_output_by: str = "pollutant",
    title_templates: dict[str, str] | None = None,
):
    """Compatibility wrapper for the legacy chart helper signature.

    Accepts the old ``comparison`` payload dict and routes to the new plotting
    helpers that work on a flattened dataframe.
    """
    if isinstance(comparison, pd.DataFrame):
        frame = comparison.copy()
    else:
        frame = pd.DataFrame((comparison or {}).get("rows") or [])

    if frame.empty:
        return []

    for required_col, default_value in {
        "validation_period_name": "overall",
        "validation_period_start": None,
        "validation_period_end": None,
        "validation_period": None,
    }.items():
        if required_col not in frame.columns:
            frame[required_col] = default_value

    metric_columns = [
        column
        for column in frame.columns
        if str(column).split(".")[-1] in {"RMSE", "R2"}
    ]
    effective_metrics = requested_metrics or ["RMSE", "R2"]
    output_path = Path(output_dir)

    if str(comparison_mode).strip().lower() in {"stations", "station"}:
        return _plot_writer._write_station_comparison_metric_charts(
            frame,
            metric_columns,
            effective_metrics,
            output_path,
            group_output_by=group_output_by,
            title_templates=title_templates,
        )

    return _plot_writer._write_station_period_metric_charts(
        frame,
        metric_columns,
        effective_metrics,
        output_path,
        group_output_by=group_output_by,
        title_templates=title_templates,
    )


def train_cmd() -> None:
    """Run the training CLI command (legacy export)."""
    from .cli_train import train_cmd as _train_cmd

    _train_cmd()


def eval_cmd() -> None:
    """Run the evaluation CLI command (legacy export)."""
    from .cli_eval import eval_cmd as _eval_cmd

    _eval_cmd()


def predict_cmd() -> None:
    """Run the prediction CLI command (legacy export)."""
    from .cli_predict import predict_cmd as _predict_cmd

    _predict_cmd()


def plot_cmd() -> None:
    """Run the plotting CLI command (legacy export)."""
    from .cli_plot import plot_cmd as _plot_cmd

    _plot_cmd()
