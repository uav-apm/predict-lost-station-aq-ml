"""Plot generation utilities for metric charts and prediction visualisations.

This module is responsible for every file-writing operation that produces
PNG or HTML artefacts during the ``aq-spatial-reconstruction-plot`` command.
It is intentionally free of training/evaluation logic so it can be used
independently from a set of already-completed run directories.

Responsibilities:
    - Static per-station prediction-vs-actual PNG exports.
    - Static actual-vs-prediction scatter PNG exports (including baselines).
    - Metric comparison bar charts grouped by validation period or station.
    - Interactive Plotly HTML combining all series into one chart.
    - Resolving CLI plot arguments to concrete run directories.
    - Reading and re-hydrating evaluation payloads from run directories.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .artifact_io import (
    _artifact_base_dir,
    _config_latest_marker_name,
    _resolve_run_dir,
    _resolve_algorithm_display_name_from_metadata,
)
from .config_loader import (
    _load_cfg,
    _report_note,
    _sanitize_marker_suffix,
)
from .data import save_prediction_sample_plot
from .model_evaluation import _resolve_eval_run_dir_arg


# ---------------------------------------------------------------------------
# Short label helpers
# ---------------------------------------------------------------------------


def _short_pollutant_name(label: str, metadata: dict | None = None) -> str:
    """Return a compact, filesystem-safe pollutant token from a column label.

    Strips the ``target_value__`` prefix, drops parenthesised unit suffixes,
    then sanitises the result so it can be used in directory and file names.
    
    When label is "target" (single-target case), infers pollutant from metadata.

    Args:
        label: A raw pollutant label such as ``"target_value__PM2.5 (µg/m³)"``.
        metadata: Optional metadata dict to infer pollutant when label is "target".

    Returns:
        A short sanitised token like ``"PM2_5"`` suitable for file names.
    """
    name = str(label or "")
    
    # Handle single-target "target" case - infer from metadata
    if name == "target" and metadata:
        target_pollutant = " ".join(
            str(metadata.get(key) or "")
            for key in ("target_pollutant", "algorithm_name", "algorithm", "config_path")
        ).lower()
        if "pm25" in target_pollutant or "pm2.5" in target_pollutant:
            return "PM2_5"
        elif "no2" in target_pollutant:
            return "NO2"
    prefix = "target_value__"
    if name.startswith(prefix):
        name = name[len(prefix):]
    paren_idx = name.find("(")
    if paren_idx > 0:
        name = name[:paren_idx].strip()
    return _sanitize_marker_suffix(name) or "pollutant"


# ---------------------------------------------------------------------------
# Plot title rendering
# ---------------------------------------------------------------------------


class _SafeTemplateDict(dict):
    """A dict subclass that returns the placeholder text for missing keys.

    This prevents :meth:`str.format_map` from raising :class:`KeyError` when
    a user-configured title template references a placeholder that was not
    supplied at render time.
    """

    def __missing__(self, key: str) -> str:  # noqa: D105
        return "{" + str(key) + "}"


def _render_plot_title(template: str, **context) -> str:
    """Render a user-configured plot title template safely.

    Unknown ``{placeholder}`` references are left unchanged rather than
    raising a :class:`KeyError`.

    Args:
        template: A Python format string, e.g.
            ``"Station {station_id} — {pollutant}"``.
        **context: Key-value pairs to substitute into *template*.

    Returns:
        The rendered title string.
    """
    return str(template).format_map(
        _SafeTemplateDict({k: "" if v is None else v for k, v in context.items()})
    )


# ---------------------------------------------------------------------------
# Metric chart helpers
# ---------------------------------------------------------------------------


def _write_station_comparison_metric_charts(
    frame: pd.DataFrame,
    metric_columns: list[str],
    requested_metrics: list[str],
    output_dir: Path,
    *,
    group_output_by: str,
    title_templates: dict[str, str] | None = None,
) -> list[str]:
    """Write pollutant charts comparing stations across each validation period.

    For every (pollutant, validation-period) combination found in *frame* a
    grouped bar chart is written showing how each algorithm performs across all
    stations.  Best and worst values per station are colour-coded (green / red).
    Companion CSV files are written alongside each PNG.

    Args:
        frame: Flattened evaluation comparison dataframe with columns
            ``target_station_id``, ``algorithm``, ``validation_period_*``, and
            one column per metric.
        metric_columns: The metric column names present in *frame*.
        requested_metrics: The short metric names (e.g. ``["RMSE", "MAE"]``)
            to include in the charts.
        output_dir: Root directory for chart outputs.
        group_output_by: Either ``"station"`` (default) or ``"pollutant"``.
            Controls whether output files are nested under station directories
            or pollutant directories.
        title_templates: Optional mapping of chart-type key → format string
            used to override default figure titles.

    Returns:
        Absolute paths of all PNG files written.
    """
    requested_metric_set = set(requested_metrics)

    def _period_rank(name: str) -> int:
        match = re.search(r"(\d+)", str(name or ""))
        return int(match.group(1)) if match else 10**9

    def _metric_group(metric_name: str) -> tuple[str, str]:
        prefix = "by_target."
        if metric_name.startswith(prefix):
            remainder = metric_name[len(prefix):]
            parts = remainder.rsplit(".", 1)
            if len(parts) == 2:
                return parts[0], parts[1]
        return "overall", metric_name

    frame = frame.copy()
    frame["_period_key"] = list(
        zip(
            frame["validation_period_start"].astype(str),
            frame["validation_period_end"].astype(str),
            frame["validation_period_name"].astype(str),
        )
    )
    frame["_period_rank"] = frame["validation_period_name"].map(_period_rank)
    frame = frame.sort_values([
        "_period_rank",
        "validation_period_start",
        "validation_period_end",
        "validation_period_name",
        "target_station_id",
        "algorithm",
    ])

    all_period_keys = list(dict.fromkeys(frame["_period_key"].tolist()))
    period_keys = [
        key
        for key in all_period_keys
        if str(key[2] or "").strip().lower() != "overall"
        and not (str(key[0] or "").strip() == "" and str(key[1] or "").strip() == "")
    ]
    if not period_keys:
        # Fallback for plot-time comparisons built from runs that only expose overall metrics.
        period_keys = [
            key
            for key in all_period_keys
            if str(key[2] or "").strip().lower() == "overall"
            or (str(key[0] or "").strip() == "" and str(key[1] or "").strip() == "")
        ]
    if not period_keys:
        return []

    def _is_baseline_algo(name: str) -> bool:
        return str(name).lower().startswith("baseline")

    _raw_algorithms = frame["algorithm"].astype(str).unique().tolist()
    _baseline_algos = sorted(a for a in _raw_algorithms if _is_baseline_algo(a))
    _model_algos = sorted(a for a in _raw_algorithms if not _is_baseline_algo(a))
    algorithms = _baseline_algos + _model_algos
    if not algorithms:
        return []

    metrics_by_group: dict[str, list[tuple[str, str]]] = {}
    for metric_name in metric_columns:
        if frame[metric_name].dropna().empty:
            continue
        pollutant_name, short_metric = _metric_group(metric_name)
        if short_metric not in requested_metric_set:
            continue
        metrics_by_group.setdefault(pollutant_name, []).append((metric_name, short_metric))

    for pollutant_name in list(metrics_by_group.keys()):
        ordered = sorted(
            metrics_by_group[pollutant_name],
            key=lambda item: requested_metrics.index(item[1]) if item[1] in requested_metric_set else 10**9,
        )
        metrics_by_group[pollutant_name] = ordered

    by_target_groups = [name for name in metrics_by_group.keys() if name != "overall"]
    if by_target_groups:
        metrics_by_group = {name: metrics_by_group[name] for name in by_target_groups}

    written: list[str] = []
    title_templates = title_templates or {}
    _lower_better_metrics = {"RMSE", "MAE", "MSE"}

    for pollutant_name, grouped_metrics in metrics_by_group.items():
        if not grouped_metrics:
            continue
        pollutant_token = _short_pollutant_name(pollutant_name)

        for period_index, period_key in enumerate(period_keys, start=1):
            period_start, period_end, period_name = period_key
            period_df = frame.loc[frame["_period_key"] == period_key].copy()
            if period_df.empty:
                continue

            station_labels = sorted(period_df["target_station_id"].astype(str).unique().tolist())
            if not station_labels:
                continue
            x_positions = list(range(len(station_labels)))

            pollutant_algorithms = [
                a
                for a in algorithms
                if any(
                    not period_df.loc[period_df["algorithm"].astype(str) == a, mn].dropna().empty
                    for mn, _ in grouped_metrics
                )
            ]
            if not pollutant_algorithms:
                continue

            _pollutant_baselines = [a for a in pollutant_algorithms if _is_baseline_algo(a)]
            _pollutant_models = [a for a in pollutant_algorithms if not _is_baseline_algo(a)]
            _pollutant_group_gap = 1.0 if _pollutant_baselines and _pollutant_models else 0.0
            _pollutant_algo_slots = [
                float(i) + (_pollutant_group_gap if not _is_baseline_algo(a) and _pollutant_baselines else 0.0)
                for i, a in enumerate(pollutant_algorithms)
            ]
            _pollutant_algo_center = (_pollutant_algo_slots[0] + _pollutant_algo_slots[-1]) / 2
            width = 0.65 / max(1, len(pollutant_algorithms) + _pollutant_group_gap)

            n_metrics = len(grouped_metrics)
            n_cols = 2
            n_rows = (n_metrics + n_cols - 1) // n_cols
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(max(10, n_cols * 6), max(4, n_rows * 3.8)))
            axes_list = list(axes.flatten()) if hasattr(axes, "flatten") else [axes]

            for idx, (metric_name, short_metric_name) in enumerate(grouped_metrics):
                ax = axes_list[idx]
                _is_lower = short_metric_name in _lower_better_metrics

                _best_per_station: dict[str, float] = {}
                _worst_per_station: dict[str, float] = {}
                for station_label in station_labels:
                    _vals: list[float] = []
                    for _algo in pollutant_algorithms:
                        _mask = (
                            (period_df["algorithm"].astype(str) == _algo)
                            & (period_df["target_station_id"].astype(str) == station_label)
                        )
                        _subset = period_df.loc[_mask, metric_name].dropna()
                        if not _subset.empty:
                            _vals.append(float(_subset.iloc[0]))
                    if _vals:
                        _best_per_station[station_label] = min(_vals) if _is_lower else max(_vals)
                        _worst_per_station[station_label] = max(_vals) if _is_lower else min(_vals)

                for algo_index, algorithm_name in enumerate(pollutant_algorithms):
                    values: list[float] = []
                    for station_label in station_labels:
                        mask = (
                            (period_df["algorithm"].astype(str) == algorithm_name)
                            & (period_df["target_station_id"].astype(str) == station_label)
                        )
                        subset = period_df.loc[mask, metric_name].dropna()
                        values.append(float(subset.iloc[0]) if not subset.empty else float("nan"))

                    if all(np.isnan(value) for value in values):
                        continue

                    offsets = [
                        x + (_pollutant_algo_slots[algo_index] - _pollutant_algo_center) * width
                        for x in x_positions
                    ]
                    bars = ax.bar(offsets, values, width=width, label=algorithm_name)
                    labels = ["" if np.isnan(value) else f"{value:.3f}" for value in values]
                    bar_texts = ax.bar_label(bars, labels=labels, padding=2, fontsize=7, rotation=90)
                    for station_label, txt, val in zip(station_labels, bar_texts, values):
                        if np.isnan(val) or not txt.get_text():
                            continue
                        if station_label in _best_per_station and abs(val - _best_per_station[station_label]) < 1e-9:
                            txt.set_color("green")
                            txt.set_fontweight("bold")
                        elif station_label in _worst_per_station and abs(val - _worst_per_station[station_label]) < 1e-9:
                            txt.set_color("red")
                            txt.set_fontweight("bold")

                _dir_indicator = "↓ lower is better" if _is_lower else "↑ higher is better"
                _title_color = "#b84800" if _is_lower else "#006400"
                ax.set_title(
                    f"{short_metric_name}  {_dir_indicator}",
                    fontsize=9,
                    color=_title_color,
                    fontweight="bold",
                )
                ax.set_xticks(x_positions)
                ax.set_xticklabels(station_labels, rotation=20, ha="right")
                ax.set_xlabel("Station")
                ax.grid(axis="y", alpha=0.25)
                ax.margins(y=0.18)
                # Correct axis margins based on metric type
                if short_metric_name in ("RMSE", "MAE", "MSE"):
                    # For error metrics: get current ylim and add margin
                    current_ymin, current_ymax = ax.get_ylim()
                    margin = 10 if short_metric_name == "RMSE" else 5
                    ax.set_ylim(current_ymin, current_ymax + margin)
                elif short_metric_name == "R2":
                    # For R2: add 1 to max, subtract 1 from min
                    current_ymin, current_ymax = ax.get_ylim()
                    ax.set_ylim(current_ymin - 1, min(current_ymax + 2, 2.0))
                else:
                    # For other metrics, use standard margin
                    ax.margins(y=0.18)

            for extra_ax in axes_list[n_metrics:]:
                extra_ax.axis("off")

            handles: list = []
            labels: list = []
            for ax in axes_list[:n_metrics]:
                for handle, label in zip(*ax.get_legend_handles_labels()):
                    if label not in labels:
                        handles.append(handle)
                        labels.append(label)

            has_figure_legend = False
            if len(axes_list) > n_metrics:
                legend_ax = axes_list[n_metrics]
                legend_ax.axis("off")
                if handles:
                    legend_ax.legend(handles, labels, loc="lower left", fontsize="small", frameon=True)
            elif handles:
                fig.legend(handles, labels, loc="lower center", ncol=min(4, len(labels)), fontsize="small", frameon=True)
                has_figure_legend = True

            period_label = str(period_name or f"period_{period_index}")
            show_period_in_title = len(period_keys) > 1
            title_period_label = period_label if show_period_in_title else ""
            # Handle NaN/None values in period start/end for title
            period_start_str = str(period_start)[:10] if pd.notna(period_start) else "?"
            period_end_str = str(period_end)[:10] if pd.notna(period_end) else "?"
            default_title_template = (
                "Metrics by Station - {pollutant} - {period_name}\n{period_start} to {period_end}"
                if show_period_in_title
                else "Metrics by Station - {pollutant}\n{period_start} to {period_end}"
            )
            fig.suptitle(
                _render_plot_title(
                    title_templates.get(
                        "metrics_by_station",
                        default_title_template,
                    ),
                    pollutant=pollutant_name,
                    period_name=title_period_label,
                    period_start=period_start_str,
                    period_end=period_end_str,
                    mode="stations",
                ),
                y=0.97,  # Adjusted from 0.99 to accommodate bar labels
            )
            fig.tight_layout(rect=[0, 0.08 if has_figure_legend else 0.02, 1, 0.92])  # Reduced top from 0.95

            period_token = _sanitize_marker_suffix(period_label)[:24] or f"period_{period_index}"
            # Station-comparison metrics are pollutant-level artefacts.
            chart_dir = output_dir / f"pollutant_{pollutant_token}"
            chart_dir.mkdir(parents=True, exist_ok=True)

            out_path = chart_dir / f"metrics_compare_stations_{pollutant_token}_{period_token}.png"
            fig.savefig(out_path, dpi=150)
            plt.close(fig)
            written.append(str(out_path.resolve()))

            csv_rows: list[dict] = []
            for algorithm_name in pollutant_algorithms:
                for station_label in station_labels:
                    row: dict = {
                        "algorithm": algorithm_name,
                        "station": station_label,
                        "period": period_label,
                        "period_start": str(period_start),
                        "period_end": str(period_end),
                    }
                    for metric_name, short_metric_name in grouped_metrics:
                        mask = (
                            (period_df["algorithm"].astype(str) == algorithm_name)
                            & (period_df["target_station_id"].astype(str) == station_label)
                        )
                        subset = period_df.loc[mask, metric_name].dropna()
                        row[short_metric_name] = float(subset.iloc[0]) if not subset.empty else None
                    csv_rows.append(row)

            csv_path = chart_dir / f"metrics_compare_stations_{pollutant_token}_{period_token}.csv"
            pd.DataFrame(csv_rows).to_csv(csv_path, index=False)

    return written


def _write_station_period_metric_charts(
    frame: pd.DataFrame,
    metric_columns: list[str],
    requested_metrics: list[str],
    output_dir: Path,
    *,
    group_output_by: str,
    title_templates: dict[str, str] | None = None,
) -> list[str]:
    """Write per-station metric charts showing all validation periods side by side.

    For each station a grouped bar chart is written with one subplot per
    requested metric.  Each bar group represents a validation period and bars
    within the group represent different algorithms.  Companion CSV and plain-
    text conclusions files are written alongside each PNG.

    Args:
        frame: Flattened evaluation comparison dataframe (same schema as used
            by :func:`_write_station_comparison_metric_charts`).
        metric_columns: Metric column names present in *frame*.
        requested_metrics: Short metric names to include.
        output_dir: Root directory for chart outputs.
        group_output_by: ``"station"`` or ``"pollutant"``.
        title_templates: Optional title format string overrides.

    Returns:
        Absolute paths of all PNG files written.
    """
    written: list[str] = []
    title_templates = title_templates or {}

    def _period_rank(name: str) -> int:
        match = re.search(r"(\d+)", str(name or ""))
        return int(match.group(1)) if match else 10**9

    def _metric_group(metric_name: str) -> tuple[str, str]:
        prefix = "by_target."
        if metric_name.startswith(prefix):
            remainder = metric_name[len(prefix):]
            parts = remainder.rsplit(".", 1)
            if len(parts) == 2:
                return parts[0], parts[1]
        return "overall", metric_name

    grouped = frame.groupby("target_station_id", dropna=False)
    for station_id, station_rows in grouped:
        station_df = station_rows.copy()
        station_label = str(station_id) if station_id is not None else "unknown"
        station_output_dir = output_dir / f"station_{_sanitize_marker_suffix(station_label)}"
        station_output_dir.mkdir(parents=True, exist_ok=True)

        station_df["_period_key"] = list(
            zip(
                station_df["validation_period_start"].astype(str),
                station_df["validation_period_end"].astype(str),
                station_df["validation_period_name"].astype(str),
            )
        )
        station_df["_period_rank"] = station_df["validation_period_name"].map(_period_rank)
        station_df = station_df.sort_values([
            "_period_rank",
            "validation_period_start",
            "validation_period_end",
            "validation_period_name",
            "algorithm",
        ])

        all_period_keys = list(dict.fromkeys(station_df["_period_key"].tolist()))
        # Omit synthetic overall rows from period charts; keep only explicit period windows.
        period_keys = [
            key
            for key in all_period_keys
            if str(key[2] or "").strip().lower() != "overall"
            and not (
                str(key[0] or "").strip() == ""
                and str(key[1] or "").strip() == ""
            )
        ]
        if not period_keys:
            # Fallback for plot-time comparisons built from runs that only expose overall metrics.
            period_keys = [
                key
                for key in all_period_keys
                if str(key[2] or "").strip().lower() == "overall"
                or (
                    str(key[0] or "").strip() == ""
                    and str(key[1] or "").strip() == ""
                )
            ]

        def _is_baseline_algo(name: str) -> bool:
            return str(name).lower().startswith("baseline")

        _raw_algorithms = station_df["algorithm"].astype(str).unique().tolist()
        _baseline_algos = sorted(a for a in _raw_algorithms if _is_baseline_algo(a))
        _model_algos = sorted(a for a in _raw_algorithms if not _is_baseline_algo(a))
        algorithms = _baseline_algos + _model_algos
        if not period_keys or not algorithms:
            continue

        metrics_by_group: dict[str, list[tuple[str, str]]] = {}
        for metric_name in metric_columns:
            if station_df[metric_name].dropna().empty:
                continue
            pollutant_name, short_metric = _metric_group(metric_name)
            if short_metric not in set(requested_metrics):
                continue
            metrics_by_group.setdefault(pollutant_name, []).append((metric_name, short_metric))

        for pollutant_name in list(metrics_by_group.keys()):
            ordered = sorted(
                metrics_by_group[pollutant_name],
                key=lambda item: requested_metrics.index(item[1]) if item[1] in set(requested_metrics) else 10**9,
            )
            metrics_by_group[pollutant_name] = ordered

        by_target_groups = [name for name in metrics_by_group.keys() if name != "overall"]
        if by_target_groups:
            metrics_by_group = {name: metrics_by_group[name] for name in by_target_groups}

        if len(period_keys) == 1 and str(period_keys[0][2] or "").strip().lower() == "overall":
            period_labels = ["overall"]
        else:
            period_labels = [f"period {idx + 1}" for idx, _ in enumerate(period_keys)]

        x_positions = list(range(len(period_keys)))
        # Insert a one-bar-wide gap between the baseline group and the model group.
        _group_gap = 1.0 if _baseline_algos and _model_algos else 0.0
        _algo_slots = [
            float(i) + (_group_gap if not _is_baseline_algo(a) and _baseline_algos else 0.0)
            for i, a in enumerate(algorithms)
        ]
        _algo_center = (_algo_slots[0] + _algo_slots[-1]) / 2
        width = 0.65 / max(1, len(algorithms) + _group_gap)

        _lower_better_metrics = {"RMSE", "MAE", "MSE"}
        _higher_better_metrics = {"IA", "R2"}

        for pollutant_name, grouped_metrics in metrics_by_group.items():
            if not grouped_metrics:
                continue

            # Algorithms with at least one real value for this pollutant's metrics.
            pollutant_algorithms = [
                a for a in algorithms
                if any(
                    not station_df.loc[station_df["algorithm"].astype(str) == a, mn].dropna().empty
                    for mn, _ in grouped_metrics
                )
            ]

            # Best-params lookup: algorithm display name → formatted param string (models only).
            _algo_params_str: dict[str, str] = {}
            if "best_params" in station_df.columns:
                for _algo in pollutant_algorithms:
                    if _is_baseline_algo(_algo):
                        continue
                    _algo_rows = station_df[station_df["algorithm"].astype(str) == _algo]
                    for _, _r in _algo_rows.iterrows():
                        _p = _r.get("best_params")
                        if isinstance(_p, dict) and _p:
                            _algo_params_str[_algo] = ", ".join(
                                f"{k}=[{','.join(str(x) for x in v)}]"
                                if isinstance(v, list)
                                else f"{k}={v}"
                                for k, v in _p.items()
                            )
                            break

            n_metrics = len(grouped_metrics)
            n_cols = 2
            n_rows = (n_metrics + n_cols - 1) // n_cols
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(max(10, n_cols * 6), max(4, n_rows * 3.8)))
            if hasattr(axes, "flatten"):
                axes_list = list(axes.flatten())
            else:
                axes_list = [axes]

            for idx, (metric_name, short_metric_name) in enumerate(grouped_metrics):
                ax = axes_list[idx]
                _is_lower = short_metric_name in _lower_better_metrics

                # Pre-compute best and worst value per period index for label colouring.
                _best_per_period: dict[int, float] = {}
                _worst_per_period: dict[int, float] = {}
                for _pidx, _pkey in enumerate(period_keys):
                    _pvals = []
                    for _a in algorithms:
                        _m = (station_df["algorithm"].astype(str) == _a) & (station_df["_period_key"] == _pkey)
                        _sub = station_df.loc[_m, metric_name].dropna()
                        if not _sub.empty:
                            _pvals.append(float(_sub.iloc[0]))
                    if _pvals:
                        _best_per_period[_pidx] = min(_pvals) if _is_lower else max(_pvals)
                        _worst_per_period[_pidx] = max(_pvals) if _is_lower else min(_pvals)

                for algo_index, algorithm_name in enumerate(algorithms):
                    values = []
                    for period_key in period_keys:
                        mask = (
                            (station_df["algorithm"].astype(str) == algorithm_name)
                            & (station_df["_period_key"] == period_key)
                        )
                        subset = station_df.loc[mask, metric_name].dropna()
                        values.append(float(subset.iloc[0]) if not subset.empty else float("nan"))

                    if all(np.isnan(value) for value in values):
                        continue

                    offsets = [x + (_algo_slots[algo_index] - _algo_center) * width for x in x_positions]
                    bars = ax.bar(offsets, values, width=width, label=algorithm_name)
                    labels = ["" if np.isnan(value) else f"{value:.3f}" for value in values]
                    bar_texts = ax.bar_label(bars, labels=labels, padding=2, fontsize=7, rotation=90)
                    for _pidx, (txt, val) in enumerate(zip(bar_texts, values)):
                        if np.isnan(val) or not txt.get_text():
                            continue
                        if _pidx in _best_per_period and abs(val - _best_per_period[_pidx]) < 1e-9:
                            txt.set_color("green")
                            txt.set_fontweight("bold")
                        elif _pidx in _worst_per_period and abs(val - _worst_per_period[_pidx]) < 1e-9:
                            txt.set_color("red")
                            txt.set_fontweight("bold")

                _dir_indicator = "↓ lower is better" if _is_lower else "↑ higher is better"
                _title_color = "#b84800" if _is_lower else "#006400"
                ax.set_title(f"{short_metric_name}  {_dir_indicator}", fontsize=9, color=_title_color, fontweight="bold")
                ax.set_xticks(x_positions)
                ax.set_xticklabels(period_labels, rotation=20, ha="right")
                ax.grid(axis="y", alpha=0.25)
                ax.margins(y=0.25)  # Increased from 0.18 to provide more space for bar labels

            for extra_ax in axes_list[n_metrics:]:
                extra_ax.axis("off")

            if len(axes_list) > n_metrics:
                legend_ax = axes_list[n_metrics]
                legend_ax.axis("off")
                period_lines = []
                for idx, key in enumerate(period_keys):
                    name = str(key[2] or "").strip().lower()
                    if name == "overall":
                        period_lines.append("overall")
                        continue
                    period_lines.append(f"period {idx + 1}: {str(key[0])[:10]} to {str(key[1])[:10]}")
                legend_ax.text(
                    0.0,
                    1.0,
                    "Validation Periods\n" + "\n".join(period_lines),
                    transform=legend_ax.transAxes,
                    va="top",
                    ha="left",
                    fontsize=8,
                )
                handles: list = []
                labels: list = []
                
                # Collect handles/labels and reorganize: baselines first, then models
                all_handles_labels: dict[str, tuple] = {}
                for ax in axes_list[:n_metrics]:
                    for h, l in zip(*ax.get_legend_handles_labels()):
                        if l not in all_handles_labels:
                            all_handles_labels[l] = h
                
                # Separate baselines and models
                baseline_labels = [l for l in all_handles_labels.keys() if str(l).lower().startswith("baseline")]
                model_labels = [l for l in all_handles_labels.keys() if not str(l).lower().startswith("baseline")]
                
                # Add baselines first, then models
                for label in sorted(baseline_labels) + sorted(model_labels):
                    handles.append(all_handles_labels[label])
                    labels.append(label)
                
                if handles:
                    legend_ax.legend(handles, labels, loc="lower left", fontsize="small", frameon=True, ncol=1)

            pollutant_label = pollutant_name if pollutant_name != "overall" else "overall"

            fig.suptitle(
                _render_plot_title(
                    title_templates.get(
                        "metrics_by_period",
                        "Metrics by Period - Station {station_id} - {pollutant}",
                    ),
                    station_id=station_label,
                    pollutant=pollutant_label,
                    mode="validation_periods",
                ),
                y=0.97,  # Adjusted from 0.99 to accommodate bar labels
            )
            fig.tight_layout(rect=[0, 0.02, 1, 0.92])  # Reduced top from 0.95 to provide more space for bar labels

            station_token = _sanitize_marker_suffix(station_label)[:8] or "station"
            pollutant_token = _short_pollutant_name(pollutant_label)
            if group_output_by == "pollutant":
                pollutant_output_dir = output_dir / f"pollutant_{pollutant_token}"
                pollutant_output_dir.mkdir(parents=True, exist_ok=True)
                out_path = pollutant_output_dir / f"metrics_{pollutant_token}_{station_token}.png"
            else:
                out_path = station_output_dir / f"metrics_{station_token}_{pollutant_token}.png"
            fig.savefig(out_path, dpi=150)
            plt.close(fig)
            written.append(str(out_path.resolve()))

            # --- CSV: one row per (algorithm, period) with metric values ---
            csv_rows = []
            for period_key in period_keys:
                p_start, p_end, p_name = period_key[0], period_key[1], period_key[2]
                for algorithm_name in pollutant_algorithms:
                    row: dict = {
                        "algorithm": algorithm_name,
                        "period": p_name,
                        "period_start": str(p_start),
                        "period_end": str(p_end),
                    }
                    for metric_name, short_metric_name in grouped_metrics:
                        mask = (
                            (station_df["algorithm"].astype(str) == algorithm_name)
                            & (station_df["_period_key"] == period_key)
                        )
                        subset = station_df.loc[mask, metric_name].dropna()
                        row[short_metric_name] = float(subset.iloc[0]) if not subset.empty else None
                    row["best_params"] = _algo_params_str.get(algorithm_name, "")
                    csv_rows.append(row)
            csv_path = out_path.with_suffix(".csv")
            pd.DataFrame(csv_rows).to_csv(csv_path, index=False)

            # --- Conclusions text file ---
            _lower_better = _lower_better_metrics
            _higher_better = _higher_better_metrics
            _metric_desc = {
                "RMSE": "Root Mean Squared Error — penalises large errors heavily; lower is better.",
                "MAE": "Mean Absolute Error — average absolute deviation from the true value; lower is better.",
                "MSE": "Mean Squared Error — like RMSE but not square-rooted, so larger errors dominate even more; lower is better.",
                "IA": "Index of Agreement — 0 to 1 scale of how well predictions track observed variability; higher is better (1 = perfect).",
                "R2": "Coefficient of Determination (R-squared) — fraction of variance explained by the model; higher is better (1 = perfect, negative = worse than predicting the mean).",
            }
            txt_lines: list[str] = [
                f"Conclusions — Station {station_label} — {pollutant_label}",
                "=" * 70,
                "",
                "Metric Definitions",
                "-" * 40,
            ]
            for _, short_m in grouped_metrics:
                if short_m in _metric_desc:
                    txt_lines.append(f"  {short_m}: {_metric_desc[short_m]}")
            txt_lines.append("")

            _period_wins: dict[str, dict[str, int]] = {}
            for pidx, period_key in enumerate(period_keys):
                p_start, p_end, p_name = period_key[0], period_key[1], period_key[2]
                p_start_str = str(p_start)[:10] if p_start else "?"
                p_end_str = str(p_end)[:10] if p_end else "?"
                txt_lines.append(f"Period {pidx + 1}: {p_start_str} to {p_end_str}")
                txt_lines.append("-" * 40)
                for metric_name, short_m in grouped_metrics:
                    algo_vals: dict[str, float] = {}
                    for algorithm_name in pollutant_algorithms:
                        mask = (
                            (station_df["algorithm"].astype(str) == algorithm_name)
                            & (station_df["_period_key"] == period_key)
                        )
                        subset = station_df.loc[mask, metric_name].dropna()
                        if not subset.empty:
                            algo_vals[algorithm_name] = float(subset.iloc[0])
                    if not algo_vals:
                        continue
                    reverse = short_m in _higher_better
                    ranked = sorted(algo_vals.items(), key=lambda kv: kv[1], reverse=reverse)
                    best_name, best_val = ranked[0]
                    worst_name, worst_val = ranked[-1]
                    _period_wins.setdefault(best_name, {})[short_m] = (
                        _period_wins.get(best_name, {}).get(short_m, 0) + 1
                    )
                    txt_lines.append(
                        f"  {short_m}: best = {best_name} ({best_val:.3f})"
                        f",  worst = {worst_name} ({worst_val:.3f})"
                    )
                txt_lines.append("")

            txt_lines += ["Overall Summary (wins across all periods)", "=" * 70]
            total_periods = len(period_keys)
            for _, short_m in grouped_metrics:
                win_counts = {a: _period_wins.get(a, {}).get(short_m, 0) for a in pollutant_algorithms}
                best_a = max(win_counts, key=win_counts.get)
                txt_lines.append(
                    f"  {short_m}: {best_a} is best in {win_counts[best_a]}/{total_periods} periods"
                )
            txt_lines.append("")
            total_wins = {a: sum(_period_wins.get(a, {}).values()) for a in pollutant_algorithms}
            if total_wins:
                champion = max(total_wins, key=total_wins.get)
                max_possible = len(grouped_metrics) * total_periods
                txt_lines.append(
                    f"Strongest overall: {champion}"
                    f" ({total_wins[champion]} metric-period wins out of {max_possible} possible)"
                )
            txt_path = out_path.with_name(f"{out_path.stem}_conclusions.txt")
            txt_path.write_text("\n".join(txt_lines), encoding="utf-8")

    return written


# ---------------------------------------------------------------------------
# Run-directory helpers
# ---------------------------------------------------------------------------


def _build_eval_results_from_run_dirs(run_dirs: list[Path]) -> list[dict]:
    """Reconstruct eval-like result payloads from artifact directories for plot-time comparisons.

    This function avoids re-running evaluation; instead it reads the JSON and
    CSV artefacts written by a previous ``eval`` command and re-hydrates them
    into the same dict structure that :func:`.model_evaluation._build_eval_comparison`
    expects.

    Args:
        run_dirs: One or more completed run directories each containing at
            least ``run_metadata.json`` and ``validation/metrics.json``.

    Returns:
        A list of result dicts, one per run directory, suitable for passing
        to :func:`.model_evaluation._build_eval_comparison`.
    """

    def _infer_single_target_pollutant(metadata: dict) -> str | None:
        target_names = metadata.get("target_names")
        if isinstance(target_names, list) and len(target_names) == 1:
            target_name = str(target_names[0])
            if target_name.startswith("target_value__"):
                return target_name.split("__", 1)[1]

        algorithm_name = str(metadata.get("algorithm_name") or "").lower()
        config_path = str(metadata.get("config_path") or "").lower()
        run_dir_value = str(metadata.get("run_dir") or "").lower()
        haystack = " ".join([algorithm_name, config_path, run_dir_value])

        if "pm25" in haystack or "pm2.5" in haystack:
            return "PM2.5 (µg/m³)"
        if "no2" in haystack:
            return "NO2 (µg/m³)"
        return None

    def _coerce_single_target_metrics(metrics: dict[str, float], metadata: dict) -> dict[str, float]:
        # Multi-target payloads already have by_target.* keys and should pass through.
        if any(str(key).startswith("by_target.") for key in metrics.keys()):
            return metrics

        pollutant = _infer_single_target_pollutant(metadata)
        if not pollutant:
            return metrics

        requested_metrics = ["RMSE", "MAE", "MSE", "IA", "R2"]
        coerced = dict(metrics)
        for metric_name in requested_metrics:
            if metric_name in metrics:
                coerced[f"by_target.target_value__{pollutant}.{metric_name}"] = float(metrics[metric_name])
        return coerced

    def _coerce_baseline_payload(baseline_payload: dict, metadata: dict) -> dict:
        coerced = dict(baseline_payload)
        metrics = coerced.get("metrics")
        if isinstance(metrics, dict):
            numeric_metrics = {
                str(key): float(value)
                for key, value in metrics.items()
                if isinstance(value, (int, float)) and not isinstance(value, bool)
            }
            coerced["metrics"] = _coerce_single_target_metrics(numeric_metrics, metadata)
        return coerced

    results: list[dict] = []
    for run_dir in run_dirs:
        metadata_path = run_dir / "run_metadata.json"
        metadata: dict = {}
        if metadata_path.exists():
            with metadata_path.open("r", encoding="utf-8") as handle:
                metadata = json.load(handle)

        _meta_optim = metadata.get("model_optimization") or metadata.get("model_validation") or {}
        _skip_param_keys = {
            "verbose", "n_jobs", "random_state", "fallback_on_missing_backend",
            "validation_split", "patience",
        }
        _raw_best_params = _meta_optim.get("best_params") or metadata.get("model_params") or {}
        _best_params: dict | None = {k: v for k, v in _raw_best_params.items() if k not in _skip_param_keys} or None

        result: dict = {
            "config": metadata.get("config_path") or run_dir.name,
            "run_dir": str(run_dir),
            "target_station_id": str(metadata.get("target_station_id") or "unknown"),
            "algorithm": metadata.get("algorithm"),
            "algorithm_name": _resolve_algorithm_display_name_from_metadata(metadata),
            "best_params": _best_params,
        }

        period_summary_path = run_dir / "validation" / "by_period" / "metrics_summary.csv"
        if period_summary_path.exists():
            period_frame = pd.read_csv(period_summary_path)
            period_results: list[dict] = []

            reserved_columns = {
                "validation_period_name",
                "validation_period_start",
                "validation_period_end",
                "validation_period",
                "row_count",
                "target_station_id",
            }

            for _, row in period_frame.iterrows():
                metrics: dict[str, float] = {}
                for column in period_frame.columns:
                    if column in reserved_columns:
                        continue
                    value = row[column]
                    if pd.notna(value):
                        metrics[column] = float(value)

                metrics = _coerce_single_target_metrics(metrics, metadata)

                period_results.append(
                    {
                        "name": str(row.get("validation_period_name") or "all_periods"),
                        "period": {
                            "start": row.get("validation_period_start"),
                            "end": row.get("validation_period_end"),
                        },
                        "row_count": int(row.get("row_count") or 0),
                        "metrics": metrics,
                    }
                )

            if period_results:
                result["period_results"] = period_results

        baseline_period_results_path = run_dir / "validation" / "by_period" / "baseline_period_results.json"
        if baseline_period_results_path.exists():
            with baseline_period_results_path.open("r", encoding="utf-8") as handle:
                baseline_period_payload = json.load(handle)
            if isinstance(baseline_period_payload, list):
                normalized_baselines: list[dict] = []
                for baseline in baseline_period_payload:
                    if not isinstance(baseline, dict):
                        continue
                    normalized = {
                        key: value
                        for key, value in baseline.items()
                        if key != "period_results"
                    }
                    normalized_period_results: list[dict] = []
                    for period_item in baseline.get("period_results") or []:
                        if not isinstance(period_item, dict):
                            continue
                        metrics = period_item.get("metrics")
                        if isinstance(metrics, dict):
                            numeric_metrics = {
                                str(key): float(value)
                                for key, value in metrics.items()
                                if isinstance(value, (int, float)) and not isinstance(value, bool)
                            }
                            metrics = _coerce_single_target_metrics(numeric_metrics, metadata)
                        normalized_period_results.append(
                            {
                                **period_item,
                                "metrics": metrics,
                            }
                        )
                    normalized["period_results"] = normalized_period_results
                    normalized_baselines.append(normalized)

                if normalized_baselines:
                    result["baseline_period_results"] = normalized_baselines

        metrics_path = run_dir / "validation" / "metrics.json"
        if metrics_path.exists():
            with metrics_path.open("r", encoding="utf-8") as handle:
                metrics_payload = json.load(handle)
            if isinstance(metrics_payload, dict):
                typed_metrics = {
                    str(key): float(value)
                    for key, value in metrics_payload.items()
                    if isinstance(value, (int, float)) and not isinstance(value, bool)
                }
                result["metrics"] = _coerce_single_target_metrics(typed_metrics, metadata)

        baseline_results_path = run_dir / "validation" / "baseline_results.json"
        if baseline_results_path.exists():
            with baseline_results_path.open("r", encoding="utf-8") as handle:
                baseline_payload = json.load(handle)
            if isinstance(baseline_payload, list):
                normalized_baselines = [
                    _coerce_baseline_payload(item, metadata)
                    for item in baseline_payload
                    if isinstance(item, dict)
                ]
                if normalized_baselines:
                    result["baseline_results"] = normalized_baselines

        results.append(result)

    return results


def _validation_target_pairs(validation_df: pd.DataFrame) -> dict[str, tuple[str, str]]:
    """Return target labels mapped to (actual_col, prediction_col) pairs.

    Handles both single-target DataFrames (``actual`` / ``prediction`` columns)
    and multi-target DataFrames (``actual__<target>`` / ``prediction__<target>``
    column pairs).

    Args:
        validation_df: A DataFrame loaded from ``validation/validation.csv``.

    Returns:
        A dict mapping target label → ``(actual_col, prediction_col)``.
    """
    target_pairs: dict[str, tuple[str, str]] = {}
    if {"actual", "prediction"}.issubset(validation_df.columns):
        target_pairs["target"] = ("actual", "prediction")

    actual_prefix = "actual__"
    for column in validation_df.columns:
        if not column.startswith(actual_prefix):
            continue
        target_name = column[len(actual_prefix):]
        prediction_col = f"prediction__{target_name}"
        if prediction_col in validation_df.columns:
            target_pairs[target_name] = (column, prediction_col)

    return target_pairs


# ---------------------------------------------------------------------------
# Prediction plot series collection
# ---------------------------------------------------------------------------


def _collect_prediction_plot_series(run_dirs: list[Path]) -> list[dict]:
    """Load per-run validation outputs into a unified series payload for plotting.

    Reads ``validation/validation.csv`` and, when available,
    ``validation/baseline_predictions.csv`` from each run directory and
    assembles a list of dicts with keys ``run_dir``, ``algorithm``,
    ``station_id``, ``target``, and ``frame`` (a tidy time-indexed DataFrame).

    Args:
        run_dirs: Completed run directories to load series from.

    Returns:
        A list of series dicts ready for the interactive Plotly chart.

    Raises:
        FileNotFoundError: If ``validation.csv`` is absent in any run directory.
        ValueError: If ``validation.csv`` is missing required columns or if no
            series at all could be built from the provided directories.
    """

    def _infer_single_target_label(metadata: dict) -> str:
        target_names = metadata.get("target_names")
        if isinstance(target_names, list) and len(target_names) == 1:
            raw_name = str(target_names[0])
            if raw_name.startswith("target_value__"):
                return raw_name.split("__", 1)[1]

        config_path = str(metadata.get("config_path") or "").lower()
        run_dir_value = str(metadata.get("run_dir") or "").lower()
        haystack = " ".join([config_path, run_dir_value])
        if "pm25" in haystack or "pm2.5" in haystack:
            return "PM2.5"
        if "no2" in haystack:
            return "NO2"
        return "target"

    series: list[dict] = []
    for run_dir in run_dirs:
        metadata_path = run_dir / "run_metadata.json"
        validation_path = run_dir / "validation" / "validation.csv"
        if not validation_path.exists():
            raise FileNotFoundError(f"Validation file not found for run directory: {validation_path}")

        metadata = {}
        if metadata_path.exists():
            with metadata_path.open("r", encoding="utf-8") as handle:
                metadata = json.load(handle)

        algorithm = _resolve_algorithm_display_name_from_metadata(metadata)
        station_id = str(metadata.get("target_station_id") or "unknown")

        validation_df = pd.read_csv(validation_path)
        if "prediction_time" not in validation_df.columns:
            raise ValueError(f"validation.csv is missing prediction_time: {validation_path}")
        target_pairs = _validation_target_pairs(validation_df)
        if not target_pairs:
            raise ValueError(
                f"validation.csv does not contain prediction/actual columns: {validation_path}"
            )

        single_target_label = _infer_single_target_label(metadata)

        for target_name, (actual_col, prediction_col) in target_pairs.items():
            target_df = validation_df[["prediction_time", actual_col, prediction_col]].copy()
            target_df = target_df.rename(
                columns={
                    "prediction_time": "time",
                    actual_col: "actual",
                    prediction_col: "prediction",
                }
            )
            target_df = target_df.dropna(subset=["actual", "prediction"])
            if target_df.empty:
                continue

            target_df["time"] = pd.to_datetime(target_df["time"])
            target_df = target_df.sort_values("time").reset_index(drop=True)
            plot_target_name = single_target_label if str(target_name) == "target" else str(target_name)
            series.append(
                {
                    "run_dir": str(run_dir),
                    "algorithm": algorithm,
                    "station_id": station_id,
                    "target": plot_target_name,
                    "frame": target_df,
                }
            )

        baseline_predictions_path = run_dir / "validation" / "baseline_predictions.csv"
        baseline_results_path = run_dir / "validation" / "baseline_results.json"
        if baseline_predictions_path.exists() and baseline_results_path.exists():
            baseline_manifest: dict[str, str] = {}
            with baseline_results_path.open("r", encoding="utf-8") as handle:
                baseline_payload = json.load(handle)
            if isinstance(baseline_payload, list):
                for item in baseline_payload:
                    if not isinstance(item, dict):
                        continue
                    baseline_id = str(item.get("id") or "").strip()
                    if not baseline_id:
                        continue
                    baseline_manifest[baseline_id] = str(item.get("algorithm_name") or item.get("name") or baseline_id)

            if baseline_manifest:
                baseline_df = pd.read_csv(baseline_predictions_path)
                if "prediction_time" in baseline_df.columns:
                    baseline_df["prediction_time"] = pd.to_datetime(baseline_df["prediction_time"])

                    if "actual" in baseline_df.columns:
                        # Single-target baseline predictions.
                        for baseline_id, baseline_algorithm in baseline_manifest.items():
                            prediction_col = f"prediction__{baseline_id}"
                            if prediction_col not in baseline_df.columns:
                                continue
                            target_df = baseline_df[["prediction_time", "actual", prediction_col]].copy()
                            target_df = target_df.rename(
                                columns={
                                    "prediction_time": "time",
                                    "actual": "actual",
                                    prediction_col: "prediction",
                                }
                            )
                            target_df = target_df.dropna(subset=["actual", "prediction"])
                            if target_df.empty:
                                continue
                            target_df = target_df.sort_values("time").reset_index(drop=True)
                            series.append(
                                {
                                    "run_dir": str(run_dir),
                                    "algorithm": baseline_algorithm,
                                    "station_id": station_id,
                                    "target": single_target_label,
                                    "frame": target_df,
                                }
                            )
                    else:
                        # Multi-target baseline predictions.
                        actual_prefix = "actual__"
                        actual_targets = [
                            column[len(actual_prefix):]
                            for column in baseline_df.columns
                            if column.startswith(actual_prefix)
                        ]
                        for baseline_id, baseline_algorithm in baseline_manifest.items():
                            for target_name in actual_targets:
                                actual_col = f"actual__{target_name}"
                                prediction_col = f"prediction__{baseline_id}__{target_name}"
                                if actual_col not in baseline_df.columns or prediction_col not in baseline_df.columns:
                                    continue
                                target_df = baseline_df[["prediction_time", actual_col, prediction_col]].copy()
                                target_df = target_df.rename(
                                    columns={
                                        "prediction_time": "time",
                                        actual_col: "actual",
                                        prediction_col: "prediction",
                                    }
                                )
                                target_df = target_df.dropna(subset=["actual", "prediction"])
                                if target_df.empty:
                                    continue
                                target_df = target_df.sort_values("time").reset_index(drop=True)
                                series.append(
                                    {
                                        "run_dir": str(run_dir),
                                        "algorithm": baseline_algorithm,
                                        "station_id": station_id,
                                        "target": str(target_name),
                                        "frame": target_df,
                                    }
                                )

    if not series:
        raise ValueError("No prediction vs actual series were found in the provided run directories.")
    return series


# ---------------------------------------------------------------------------
# Output path helpers
# ---------------------------------------------------------------------------


def _default_plot_output_dir(
    config_paths: list[str] | None = None,
    config_dirs: list[str] | None = None,
) -> Path:
    """Return the default plot output directory inferred from config locations.

    Args:
        config_paths: Optional list of config file paths.
        config_dirs: Optional list of config directory paths.

    Returns:
        A :class:`~pathlib.Path` pointing to ``<base>/outputs/plots``.
    """
    if config_dirs:
        return Path(config_dirs[0]) / "outputs" / "plots"
    if config_paths:
        return Path(config_paths[0]).parent / "outputs" / "plots"
    return Path.cwd() / "outputs" / "plots"


def _default_interactive_plot_path(output_dir: Path) -> Path:
    """Return a timestamped default path for the interactive HTML chart.

    Args:
        output_dir: Directory in which the file will be placed.

    Returns:
        A :class:`~pathlib.Path` with a unique timestamped filename.
    """
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return output_dir / f"prediction_vs_actual_interactive_{stamp}.html"


def _timestamped_output_dir(
    base_output_dir: Path,
    *,
    prefix: str = "",
    label: str | None = None,
) -> Path:
    """Create a unique timestamped output folder under the provided base directory.

    Args:
        base_output_dir: Parent directory under which the new folder is created.
        prefix: Optional short string prepended to the folder name.
        label: Optional human-readable label appended after sanitisation.

    Returns:
        The :class:`~pathlib.Path` of the newly created directory.
    """
    effective_prefix = prefix
    if label and str(label).strip():
        sanitized_label = _sanitize_marker_suffix(str(label))
        effective_prefix = f"{prefix}_{sanitized_label}" if prefix else sanitized_label

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = 0
    while True:
        folder_name = f"{effective_prefix}_{stamp}_{suffix}" if effective_prefix else f"{stamp}_{suffix}"
        candidate = base_output_dir / folder_name
        if not candidate.exists():
            candidate.mkdir(parents=True, exist_ok=False)
            return candidate
        suffix += 1


def _comment_scoped_plot_base_dir(base_output_dir: Path, label: str | None) -> Path:
    """Scope plot outputs to a short top-level comment folder when a comment is provided.

    Args:
        base_output_dir: The base output directory.
        label: Optional comment string used to name the sub-directory.

    Returns:
        *base_output_dir* itself when *label* is blank, otherwise a sanitised
        sub-directory is created and returned.
    """
    if not label or not str(label).strip():
        return base_output_dir
    comment_dir = base_output_dir / _sanitize_marker_suffix(str(label))
    comment_dir.mkdir(parents=True, exist_ok=True)
    return comment_dir


def _resolve_plot_output_comment(run_dirs: list[Path], explicit_comment: str | None) -> str | None:
    """Resolve the comment string to use for scoping plot output directories.

    Uses the explicit CLI ``--comment`` value when provided; otherwise
    discovers the comment from run metadata files if all runs share exactly
    one common comment.

    Args:
        run_dirs: Run directories to inspect for run metadata.
        explicit_comment: The comment passed on the CLI, or ``None``.

    Returns:
        A comment string, or ``None`` if no unambiguous comment could be found.
    """
    if explicit_comment and str(explicit_comment).strip():
        return str(explicit_comment).strip()

    discovered_comments: set[str] = set()
    for run_dir in run_dirs:
        metadata_path = run_dir / "run_metadata.json"
        if not metadata_path.exists():
            continue
        with metadata_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        run_comment = str(metadata.get("run_comment") or "").strip()
        if run_comment:
            discovered_comments.add(run_comment)

    if len(discovered_comments) == 1:
        return next(iter(discovered_comments))
    return None


def _short_plot_file_stem(
    prefix: str,
    *,
    run_label: str,
    algorithm_label: str,
    station_label: str,
    target_label: str,
) -> str:
    """Build a deterministic, Windows-safe filename stem for static plot artefacts.

    Args:
        prefix: A short type token such as ``"prediction_plot"`` or
            ``"scatter_actual_vs_prediction"``.
        run_label: Sanitised run directory name token.
        algorithm_label: Algorithm display name token.
        station_label: Station identifier token.
        target_label: Target/pollutant token.

    Returns:
        A Windows-safe filename stem string.
    """
    prefix_part = _sanitize_marker_suffix(prefix)[:12].rstrip("_-") or "plot"
    algorithm_part = _sanitize_marker_suffix(algorithm_label)[:20].rstrip("_-") or "algo"
    station_part = _sanitize_marker_suffix(station_label)[:8].rstrip("_-") or "station"
    target_part = _sanitize_marker_suffix(target_label)[:10].rstrip("_-") or "target"
    return f"{prefix_part}_{station_part}_{target_part}_{algorithm_part}"


# ---------------------------------------------------------------------------
# Static prediction plots
# ---------------------------------------------------------------------------


def _write_static_prediction_plots(
    run_dirs: list[Path],
    output_dir: Path,
    *,
    title_templates: dict[str, str] | None = None,
) -> list[str]:
    """Export one static prediction-vs-actual PNG per target series into plot outputs.

    Reads ``validation/validation.csv`` from each run directory and calls
    :func:`.data.save_prediction_sample_plot` to produce a PNG.  Companion CSV
    files are also written.

    Args:
        run_dirs: Completed run directories to plot.
        output_dir: Root directory for output files.
        title_templates: Optional title format string overrides.

    Returns:
        Absolute paths of all PNG files written.
    """
    written: list[str] = []
    output_dir.mkdir(parents=True, exist_ok=True)
    title_templates = title_templates or {}

    for run_dir in run_dirs:
        validation_path = run_dir / "validation" / "validation.csv"
        if not validation_path.exists():
            continue

        metadata: dict = {}
        metadata_path = run_dir / "run_metadata.json"
        if metadata_path.exists():
            with metadata_path.open("r", encoding="utf-8") as handle:
                metadata = json.load(handle)

        validation_df = pd.read_csv(validation_path)
        if "prediction_time" not in validation_df.columns:
            continue

        pairs = _validation_target_pairs(validation_df)
        if not pairs:
            continue

        validation_df["prediction_time"] = pd.to_datetime(validation_df["prediction_time"])
        validation_df = validation_df.sort_values("prediction_time").reset_index(drop=True)

        algorithm_label = _resolve_algorithm_display_name_from_metadata(metadata)
        station_label = str(metadata.get("target_station_id") or "unknown")
        run_label = _sanitize_marker_suffix(run_dir.name)

        station_dir = output_dir / f"station_{_sanitize_marker_suffix(station_label)}" / "prediction"
        station_dir.mkdir(parents=True, exist_ok=True)

        for target_name, (actual_col, prediction_col) in pairs.items():
            target_label = str(target_name)
            file_stem = _short_plot_file_stem(
                "prediction_plot",
                run_label=run_label,
                algorithm_label=algorithm_label,
                station_label=station_label,
                target_label=target_label,
            )
            sample_df = validation_df[["prediction_time", actual_col, prediction_col]].copy()
            csv_path = station_dir / f"{file_stem}.csv"
            png_path = station_dir / f"{file_stem}.png"
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            sample_df.to_csv(csv_path, index=False)
            wrote_png = save_prediction_sample_plot(
                sample_df,
                datetime_col="prediction_time",
                series_by_target={target_label: (actual_col, prediction_col)},
                path=png_path,
                title=_render_plot_title(
                    title_templates.get(
                        "prediction_static",
                        "Prediction vs Actual ({target})\n{algorithm} | Station {station_id}",
                    ),
                    target=target_label,
                    algorithm=algorithm_label,
                    station_id=station_label,
                    run_dir=run_dir.name,
                    chart_type="prediction_static",
                ),
            )
            if wrote_png:
                written.append(str(png_path.resolve()))

    return written


# ---------------------------------------------------------------------------
# Scatter plots
# ---------------------------------------------------------------------------


def _write_static_actual_prediction_scatter_plots(
    run_dirs: list[Path],
    output_dir: Path,
    *,
    title_templates: dict[str, str] | None = None,
    group_output_by: str = "pollutant",
) -> list[str]:
    """Export actual-vs-prediction scatter plots per run directory and target.

    For each run directory a scatter chart is written showing both training
    (from ``validation/training_predictions.csv``) and evaluation (from
    ``validation/validation.csv``) splits as side-by-side subplots.  Baseline
    scatter charts are also written from ``validation/baseline_predictions.csv``
    when present.

    Args:
        run_dirs: Completed run directories to plot.
        output_dir: Root directory for output files.
        title_templates: Optional title format string overrides.
        group_output_by: Output organization: ``"pollutant"`` or ``"station"``.

    Returns:
        Absolute paths of all PNG files written.
    """
    written: list[str] = []
    output_dir.mkdir(parents=True, exist_ok=True)
    title_templates = title_templates or {}

    for run_dir in run_dirs:
        validation_path = run_dir / "validation" / "validation.csv"
        metadata_path = run_dir / "run_metadata.json"

        if not (validation_path.exists() and metadata_path.exists()):
            continue

        try:
            with metadata_path.open("r", encoding="utf-8") as handle:
                metadata = json.load(handle)

            validation_df = pd.read_csv(validation_path)

            pairs = _validation_target_pairs(validation_df)
            if not pairs:
                continue

            algorithm_label = _resolve_algorithm_display_name_from_metadata(metadata)
            station_label = str(metadata.get("target_station_id") or "unknown")
            run_label = _sanitize_marker_suffix(run_dir.name)

            def _scatter_station_dir(pollutant_token: str) -> Path:
                station_token = _sanitize_marker_suffix(station_label)
                if group_output_by == "pollutant":
                    return output_dir / f"pollutant_{pollutant_token}" / f"station_{station_token}"
                return output_dir / f"station_{station_token}"

            for target_name, (actual_col, prediction_col) in pairs.items():
                target_label = str(target_name)
                
                # Determine output directory based on grouping preference
                pollutant_token = _short_pollutant_name(target_name, metadata)
                station_dir = _scatter_station_dir(pollutant_token)
                station_dir.mkdir(parents=True, exist_ok=True)

                split_frames: list[pd.DataFrame] = []
                evaluation_scatter_df = pd.DataFrame(
                    {
                        "actual": pd.to_numeric(validation_df[actual_col], errors="coerce"),
                        "prediction": pd.to_numeric(validation_df[prediction_col], errors="coerce"),
                    }
                ).dropna()

                if not evaluation_scatter_df.empty:
                    split_frames.append(evaluation_scatter_df.assign(split="evaluation"))

                training_prediction_path = run_dir / "validation" / "training_predictions.csv"
                if training_prediction_path.exists():
                    training_df = pd.read_csv(training_prediction_path)
                    training_pairs = _validation_target_pairs(training_df)
                    if target_name in training_pairs:
                        train_actual_col, train_prediction_col = training_pairs[target_name]
                        training_scatter_df = pd.DataFrame(
                            {
                                "actual": pd.to_numeric(training_df[train_actual_col], errors="coerce"),
                                "prediction": pd.to_numeric(training_df[train_prediction_col], errors="coerce"),
                            }
                        ).dropna()
                        if not training_scatter_df.empty:
                            split_frames.append(training_scatter_df.assign(split="training"))

                if not split_frames:
                    continue

                scatter_df = pd.concat(split_frames, ignore_index=True)

                file_stem = _short_plot_file_stem(
                    "scatter_actual_vs_prediction",
                    run_label=run_label,
                    algorithm_label=algorithm_label,
                    station_label=station_label,
                    target_label=target_label,
                )

                scatter_csv = station_dir / f"{file_stem}.csv"
                scatter_png = station_dir / f"{file_stem}.png"
                scatter_csv.parent.mkdir(parents=True, exist_ok=True)

                scatter_df.to_csv(scatter_csv, index=False)

                splits_to_plot = [
                    s for s in ["training", "evaluation"]
                    if not scatter_df.loc[scatter_df["split"] == s].empty
                ]
                n_cols = len(splits_to_plot)
                split_colors = {"training": "tab:blue", "evaluation": "tab:orange"}

                fig, axes = plt.subplots(1, n_cols, figsize=(5.5 * n_cols, 5.5), squeeze=False)
                axes = axes[0]  # row 0, all columns

                min_value = float(min(scatter_df["actual"].min(), scatter_df["prediction"].min()))
                max_value = float(max(scatter_df["actual"].max(), scatter_df["prediction"].max()))

                for col_idx, split_name in enumerate(splits_to_plot):
                    axis = axes[col_idx]
                    subset = scatter_df.loc[scatter_df["split"] == split_name]
                    axis.scatter(
                        subset["actual"],
                        subset["prediction"],
                        alpha=0.55,
                        s=14,
                        color=split_colors[split_name],
                        label=f"n={len(subset)}",
                    )
                    axis.plot(
                        [min_value, max_value],
                        [min_value, max_value],
                        linestyle="--",
                        linewidth=1.2,
                        color="black",
                    )
                    axis.set_xlim(min_value, max_value)
                    axis.set_ylim(min_value, max_value)
                    axis.set_title(f"{split_name.title()} (n={len(subset)})", color=split_colors[split_name])
                    axis.set_xlabel("Actual")
                    axis.set_ylabel("Predicted")
                    axis.grid(alpha=0.25)

                fig.suptitle(
                    _render_plot_title(
                        title_templates.get(
                            "scatter_static",
                            "Actual vs Predicted - {target}\n{algorithm} | Station {station_id}\n"
                            "Evaluation rows from validation/validation.csv; training rows from validation/training_predictions.csv",
                        ),
                        target=target_label,
                        algorithm=algorithm_label,
                        station_id=station_label,
                        run_dir=run_dir.name,
                        chart_type="scatter_static",
                    )
                )
                fig.tight_layout(rect=[0, 0.02, 1, 0.92])
                fig.savefig(scatter_png, dpi=150)
                plt.close(fig)
                written.append(str(scatter_png.resolve()))

            # Baseline scatter plots from baseline_predictions.csv
            baseline_pred_path = run_dir / "validation" / "baseline_predictions.csv"
            if baseline_pred_path.exists():
                baseline_df = pd.read_csv(baseline_pred_path)
                if "actual" in baseline_df.columns:
                    actual_vals = pd.to_numeric(baseline_df["actual"], errors="coerce")
                    # Infer pollutant label from metadata for the title
                    _haystack = " ".join([
                        str(metadata.get("algorithm_name") or "").lower(),
                        str(metadata.get("config_path") or "").lower(),
                    ])
                    if "pm25" in _haystack or "pm2.5" in _haystack:
                        _pollutant_label = "PM2.5 (µg/m³)"
                    elif "no2" in _haystack:
                        _pollutant_label = "NO2 (µg/m³)"
                    else:
                        _pollutant_label = "target"
                    _baseline_pollutant_token = _short_pollutant_name(_pollutant_label, metadata)
                    baseline_station_dir = _scatter_station_dir(_baseline_pollutant_token)
                    baseline_station_dir.mkdir(parents=True, exist_ok=True)

                    for col in baseline_df.columns:
                        if not col.startswith("prediction__baseline_"):
                            continue
                        baseline_id = col[len("prediction__"):]
                        display_name = baseline_id.replace("_", " ").title()
                        pred_vals = pd.to_numeric(baseline_df[col], errors="coerce")
                        b_df = pd.DataFrame({"actual": actual_vals, "prediction": pred_vals}).dropna()
                        if b_df.empty:
                            continue

                        # Generate short baseline filename: scatter_b_rep_<station_id>_t_<target_station>_<pollutant>
                        # or: scatter_b_avg_t_<target_station>_<pollutant>
                        if "replace_with_station" in baseline_id:
                            # Extract station ID from baseline_replace_with_station_<id>
                            parts = baseline_id.split("_")
                            if len(parts) >= 4:  # baseline, replace, with, station, <id>
                                baseline_station_id = parts[-1]
                                b_prefix = f"scatter_b_rep_{baseline_station_id}"
                            else:
                                b_prefix = "scatter_b_replace"
                        elif "mean_of_other_stations" in baseline_id:
                            b_prefix = "scatter_b_avg"
                        else:
                            b_prefix = f"scatter_b_{_sanitize_marker_suffix(baseline_id)}"
                        
                        # Add target station and pollutant abbreviation to filename
                        pollutant_abbr = _baseline_pollutant_token
                        b_stem = f"{b_prefix}_t_{_sanitize_marker_suffix(station_label)}_{pollutant_abbr}"
                        b_png = baseline_station_dir / f"{b_stem}.png"
                        b_csv = baseline_station_dir / f"{b_stem}.csv"

                        # Skip if already written by an earlier run for the same station
                        if b_png.exists():
                            continue

                        b_df.to_csv(b_csv, index=False)

                        min_v = float(min(b_df["actual"].min(), b_df["prediction"].min()))
                        max_v = float(max(b_df["actual"].max(), b_df["prediction"].max()))

                        fig_b, ax_b = plt.subplots(1, 1, figsize=(5.5, 5.5))
                        ax_b.scatter(
                            b_df["actual"], b_df["prediction"],
                            alpha=0.55, s=14, color="tab:green", label=f"n={len(b_df)}",
                        )
                        ax_b.plot(
                            [min_v, max_v],
                            [min_v, max_v],
                            linestyle="--",
                            linewidth=1.2,
                            color="black",
                        )
                        ax_b.set_xlim(min_v, max_v)
                        ax_b.set_ylim(min_v, max_v)
                        ax_b.set_xlabel("Actual")
                        ax_b.set_ylabel("Predicted")
                        ax_b.grid(alpha=0.25)
                        fig_b.suptitle(
                            _render_plot_title(
                                title_templates.get(
                                    "baseline_scatter_static",
                                    "Actual vs Predicted - {target}\n{algorithm} | Station {station_id}",
                                ),
                                target=_pollutant_label,
                                algorithm=display_name,
                                station_id=station_label,
                                baseline_id=baseline_id,
                                run_dir=run_dir.name,
                                chart_type="baseline_scatter_static",
                            ),
                            y=0.98,
                        )
                        fig_b.tight_layout(rect=[0, 0.02, 1, 0.93])
                        fig_b.savefig(b_png, dpi=150)
                        plt.close(fig_b)
                        written.append(str(b_png.resolve()))

        except (Exception, KeyboardInterrupt) as exc:
            _report_note(
                f"Skipping scatter comparison for {run_dir.name} due to error: {type(exc).__name__}: {exc}"
            )
            continue

    return written


# ---------------------------------------------------------------------------
# Interactive Plotly chart
# ---------------------------------------------------------------------------


def _legend_layout_config(mode: str) -> tuple[bool, dict]:
    """Map a legend mode string to a Plotly showlegend flag and layout dict.

    Args:
        mode: One of ``"hidden"``, ``"none"``, ``"bottom"``, ``"top"``, or
            ``"right"`` (and any unrecognised string defaults to right).

    Returns:
        A ``(show_legend, legend_layout_dict)`` tuple.
    """
    normalized = str(mode or "hidden").strip().lower()
    if normalized in {"hidden", "none"}:
        return False, {}
    if normalized == "bottom":
        return True, {"orientation": "h", "x": 0.0, "xanchor": "left", "y": -0.25}
    if normalized == "top":
        return True, {"orientation": "h", "x": 0.0, "xanchor": "left", "y": 1.12}
    return True, {"orientation": "v", "x": 1.02, "xanchor": "left", "y": 1.0}


def _write_prediction_plot_data_csv(series: list[dict], csv_path: Path) -> None:
    """Export the prediction-plot series as a wide-format CSV.

    Each row corresponds to a unique ``prediction_time``.  Columns are
    ``actual_{station}_{target}`` (one per station/pollutant pair) followed by
    ``pred_{station}_{target}_{algorithm}`` for every model and baseline.

    Args:
        series: The series list returned by :func:`_collect_prediction_plot_series`.
        csv_path: Destination ``.csv`` file path.
    """

    def _sanitize(text: str) -> str:
        return re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_")

    frames: list[pd.DataFrame] = []
    actual_emitted: set[tuple[str, str]] = set()

    for item in series:
        station = _sanitize(str(item["station_id"]))
        target = _sanitize(str(item["target"]))
        algorithm = _sanitize(str(item["algorithm"]))
        frame = item["frame"][["time", "actual", "prediction"]].copy()

        actual_key = (station, target)
        if actual_key not in actual_emitted:
            actual_col = f"actual_{station}_{target}"
            sub = frame[["time", "actual"]].rename(columns={"actual": actual_col})
            frames.append(sub.set_index("time"))
            actual_emitted.add(actual_key)

        pred_col = f"pred_{station}_{target}_{algorithm}"
        sub = frame[["time", "prediction"]].rename(columns={"prediction": pred_col})
        frames.append(sub.set_index("time"))

    if not frames:
        return

    combined = pd.concat(frames, axis=1)
    combined = combined.loc[:, ~combined.columns.duplicated(keep="first")]
    combined = combined.sort_index()
    combined.index.name = "prediction_time"

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(csv_path)


def _write_interactive_prediction_plot(
    run_dirs: list[Path],
    output_path: Path,
    *,
    legend_mode: str = "hidden",
) -> dict:
    """Create one interactive Plotly chart for prediction-vs-actual comparisons.

    Reads all prediction series via :func:`_collect_prediction_plot_series`,
    builds a :class:`plotly.graph_objects.Figure` with per-algorithm dropdown
    filters and an optional legend position toggle, then writes a self-contained
    HTML file.

    Args:
        run_dirs: Completed run directories to include in the chart.
        output_path: Destination ``.html`` file path.
        legend_mode: One of ``"hidden"``, ``"none"``, ``"bottom"``, ``"top"``,
            or ``"right"``.

    Returns:
        A summary dict with keys ``path``, ``run_count``, ``series_count``,
        ``algorithms``, and ``legend_mode``.

    Raises:
        ModuleNotFoundError: If ``plotly`` is not installed.
        FileNotFoundError: If a required validation CSV is missing.
        ValueError: If no valid series are found.
    """
    try:
        import plotly.graph_objects as go
    except ModuleNotFoundError as exc:  # pragma: no cover - defensive import guard
        raise ModuleNotFoundError(
            "plotly is required for interactive charts. Install with: pip install plotly"
        ) from exc

    series = _collect_prediction_plot_series(run_dirs)
    series = sorted(
        series,
        key=lambda item: (
            str(item.get("station_id") or ""),
            str(item.get("target") or ""),
            str(item.get("algorithm") or ""),
        ),
    )

    # Keep actual/prediction for the same station-algorithm-target visually tied.
    palette = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]
    series_colors: dict[tuple[str, str, str], str] = {}
    actual_series_emitted: set[tuple[str, str]] = set()
    labels_series_emitted: set[tuple[str, str]] = set()

    figure = go.Figure()
    trace_algorithms: list[str] = []
    trace_stations: list[str] = []
    trace_targets: list[str] = []

    all_station_ids = sorted({str(item["station_id"]) for item in series})
    all_targets = sorted({str(item["target"]) for item in series})

    for item in series:
        station_label = str(item["station_id"])
        algorithm_label = str(item["algorithm"])
        target_label = str(item["target"])
        series_key = (station_label, algorithm_label, target_label)
        if series_key not in series_colors:
            series_colors[series_key] = palette[len(series_colors) % len(palette)]
        color = series_colors[series_key]

        label_suffix = f"station {station_label} | {target_label} | {algorithm_label}"
        actual_suffix = f"station {station_label} | {target_label}"
        frame = item["frame"]

        # Actual observations are identical across algorithms for a station/target,
        # so emit them once to avoid duplicated legend labels.
        actual_key = (station_label, target_label)
        if actual_key not in actual_series_emitted:
            figure.add_trace(
                go.Scattergl(
                    x=frame["time"],
                    y=frame["actual"],
                    mode="lines",
                    name=f"Actual | {actual_suffix}",
                    legendgroup=actual_suffix,
                    line={"width": 1.7, "color": "#111111", "dash": "solid"},
                    hovertemplate=(
                        "Time: %{x|%Y-%m-%d %H:%M:%S}<br>"
                        "Actual: %{y:.3f}<extra>Actual | " + actual_suffix + "</extra>"
                    ),
                )
            )
            trace_algorithms.append("__actual__")
            trace_stations.append(station_label)
            trace_targets.append(target_label)
            actual_series_emitted.add(actual_key)
        if label_suffix not in labels_series_emitted:
            figure.add_trace(
                go.Scattergl(
                    x=frame["time"],
                    y=frame["prediction"],
                    mode="lines",
                    name=f"Prediction | {label_suffix}",
                    legendgroup=label_suffix,
                    line={"width": 1.5, "color": color, "dash": "dash"},
                    hovertemplate=(
                        "Time: %{x|%Y-%m-%d %H:%M:%S}<br>"
                        "Predicted: %{y:.3f}<extra>Prediction | " + label_suffix + "</extra>"
                    ),
                )
            )
            labels_series_emitted.add(label_suffix)
            trace_algorithms.append(str(item["algorithm"]))
            trace_stations.append(station_label)
            trace_targets.append(target_label)

    unique_algorithms = sorted({value for value in trace_algorithms if value != "__actual__"})
    visibility_all = [True] * len(trace_algorithms)

    def _build_filter_visibility(station_filter: str | None = None, target_filter: str | None = None) -> list[bool]:
        visible: list[bool] = []
        for trace_station, trace_target in zip(trace_stations, trace_targets):
            station_ok = station_filter is None or trace_station == station_filter
            target_ok = target_filter is None or trace_target == target_filter
            visible.append(station_ok and target_ok)
        return visible

    def _filter_title(station_filter: str | None = None, target_filter: str | None = None) -> str:
        station_part = station_filter if station_filter is not None else "All Stations"
        target_part = target_filter if target_filter is not None else "All Pollutants"
        return f"Prediction vs Actual ({station_part} | {target_part})"

    filter_buttons = [
        {
            "label": "All Stations",
            "method": "update",
            "args": [
                {"visible": visibility_all},
                {"title": _filter_title()},
            ],
        }
    ]

    def _build_names(station_filter: str | None = None, target_filter: str | None = None) -> list[str]:
        """Compute per-trace legend labels for a given filter.

        For prediction traces the station and/or target context is omitted when
        those dimensions are already fixed by the active filter — keeping labels
        concise.  Actual traces always retain their full context.
        """
        names: list[str] = []
        for algo, s, t in zip(trace_algorithms, trace_stations, trace_targets):
            if algo == "__actual__":
                names.append(f"Actual | station {s} | {t}")
            else:
                parts: list[str] = []
                if station_filter is None:
                    parts.append(f"station {s}")
                if target_filter is None:
                    parts.append(t)
                parts.append(algo)
                names.append("Prediction | " + " | ".join(parts))
        return names

    # Patch the "All Stations" button to also include full names in args[0]
    filter_buttons[0]["args"][0]["name"] = _build_names()

    for station_id in all_station_ids:
        filter_buttons.append(
            {
                "label": f"Station {station_id}",
                "method": "update",
                "args": [
                    {
                        "visible": _build_filter_visibility(station_filter=station_id),
                        "name": _build_names(station_filter=station_id),
                    },
                    {"title": _filter_title(station_filter=station_id)},
                ],
            }
        )

    for target_label in all_targets:
        filter_buttons.append(
            {
                "label": f"Pollutant {target_label}",
                "method": "update",
                "args": [
                    {
                        "visible": _build_filter_visibility(target_filter=target_label),
                        "name": _build_names(target_filter=target_label),
                    },
                    {"title": _filter_title(target_filter=target_label)},
                ],
            }
        )

    if len(all_station_ids) > 1 or len(all_targets) > 1:
        for station_id in all_station_ids:
            for target_label in all_targets:
                filter_buttons.append(
                    {
                        "label": f"{station_id} | {target_label}",
                        "method": "update",
                        "args": [
                            {
                                "visible": _build_filter_visibility(
                                    station_filter=station_id,
                                    target_filter=target_label,
                                ),
                                "name": _build_names(
                                    station_filter=station_id,
                                    target_filter=target_label,
                                ),
                            },
                            {"title": _filter_title(station_filter=station_id, target_filter=target_label)},
                        ],
                    }
                )
    y_axis_title = "Concentration (µg/m³)"

    # y_axis_title = "Value"
    # for target_label in all_targets:
    #     lower_target = str(target_label).lower()
    #     if "µg/m³" in str(target_label) or "ug/m3" in lower_target or "ug/m^3" in lower_target:
    #         y_axis_title = "Concentration (µg/m³)"
    #         break

    legend_visible, legend_layout = _legend_layout_config(legend_mode)

    figure.update_layout(
        title=_filter_title(),
        template="plotly_white",
        margin={"b": 190},
        xaxis={
            "rangeslider": {"visible": True, "thickness": 0.01, "bgcolor": "#ffffff", "borderwidth": 0},
            "type": "date",
            "tickangle": 30,
            "tickformat": "%Y-%m-%d\n%H:%M",
            "tickformatstops": [
                {"dtickrange": [None, 60000], "value": "%H:%M:%S"},
                {"dtickrange": [60000, 86400000], "value": "%Y-%m-%d\n%H:%M"},
                {"dtickrange": [86400000, None], "value": "%Y-%m-%d"},
            ],
            "hoverformat": "%Y-%m-%d %H:%M:%S",
            "automargin": True,
            "showspikes": True,
            "spikemode": "across",
            "spikecolor": "#999999",
        },
        yaxis={"title": y_axis_title, "automargin": True, "showgrid": True},
        hovermode="x unified",
        updatemenus=[
            {
                "buttons": filter_buttons,
                "x": 0.0,
                "xanchor": "left",
                "y": 1.18,
                "yanchor": "top",
            },
            {
                "buttons": [
                    {"label": "Legend Hidden", "method": "relayout", "args": [{"showlegend": False}]},
                    {
                        "label": "Legend Right",
                        "method": "relayout",
                        "args": [
                            {"showlegend": True, "legend.orientation": "v", "legend.x": 1.02, "legend.y": 1.0}
                        ],
                    },
                    {
                        "label": "Legend Bottom",
                        "method": "relayout",
                        "args": [
                            {"showlegend": True, "legend.orientation": "h", "legend.x": 0.0, "legend.y": -0.25}
                        ],
                    },
                ],
                "x": 0.62,
                "xanchor": "left",
                "y": 1.18,
                "yanchor": "top",
            },
        ],
        showlegend=legend_visible,
        legend=legend_layout,
        annotations=[
            {
                "text": "Prediction Time",
                "xref": "paper",
                "yref": "paper",
                "x": 0.5,
                "y": -0.18,
                "xanchor": "center",
                "yanchor": "top",
                "showarrow": False,
                "font": {"size": 13},
            }
        ],
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(str(output_path), include_plotlyjs=True, full_html=True)

    csv_path = output_path.with_suffix(".csv")
    _write_prediction_plot_data_csv(series, csv_path)

    return {
        "path": str(output_path.resolve()),
        "plot_data_csv": str(csv_path.resolve()),
        "run_count": len(run_dirs),
        "series_count": len(series),
        "algorithms": unique_algorithms,
        "legend_mode": legend_mode,
    }


# ---------------------------------------------------------------------------
# CLI argument resolution
# ---------------------------------------------------------------------------


def _resolve_plot_run_dirs(
    config_paths: list[str] | None,
    run_dir_args: list[str] | None,
) -> list[Path]:
    """Resolve plot inputs into concrete artifact run directories.

    When *config_paths* are given the latest-marker mechanism is used to
    resolve the most recent run directory for each config.  When
    *run_dir_args* are given they are returned as-is.  At least one of the two
    arguments must be non-empty.

    Args:
        config_paths: Resolved config file paths from the CLI.
        run_dir_args: Explicit run directory paths from the CLI.

    Returns:
        A list of :class:`~pathlib.Path` objects pointing at run directories.

    Raises:
        ValueError: If neither *config_paths* nor *run_dir_args* are provided,
            or if the lengths of both lists are inconsistent.
    """
    run_dir_args = run_dir_args or []
    if config_paths:
        if run_dir_args and len(run_dir_args) != len(config_paths):
            raise ValueError(
                "When plotting multiple configs, provide the same number of --run-dir values as resolved configs."
            )

        resolved: list[Path] = []
        for index, config_path in enumerate(config_paths):
            cfgd, base = _load_cfg(config_path)
            artifacts_base = _artifact_base_dir(cfgd, base)
            marker_name = _config_latest_marker_name(config_path)
            resolved_run_arg = run_dir_args[index] if run_dir_args else None
            run_dir = _resolve_run_dir(
                artifacts_base,
                _resolve_eval_run_dir_arg(cfgd, resolved_run_arg),
                latest_markers=[marker_name],
                include_default_latest_marker=False,
            )
            resolved.append(run_dir)
        return resolved

    if run_dir_args:
        return [Path(path) for path in run_dir_args]

    raise ValueError("Provide at least one --config/--config-dir or --run-dir for plotting.")


def _resolve_plot_metric_graph_options(config_paths: list[str] | None) -> dict:
    """Resolve chart options from the first plotting config, if available.

    Reads ``plot.metrics_graph`` from the first config file and returns a dict
    with keys ``metrics``, ``comparison_mode``, and ``group_output_by``.
    
    Metrics resolution priority:
    1. ``plot.metrics_graph.metrics`` (if specified)
    2. Top-level ``metrics`` (if specified)
    3. None (uses all available metrics)

    Args:
        config_paths: Resolved config file paths; the first one is used.

    Returns:
        A dict of chart option key → resolved value.
    """
    defaults = {
        "metrics": None,
        "comparison_mode": "validation_periods",
        "group_output_by": "pollutant",
    }
    if not config_paths:
        return defaults

    cfgd, _ = _load_cfg(config_paths[0])
    plot_cfg = cfgd.get("plot") if isinstance(cfgd, dict) else None
    if not isinstance(plot_cfg, dict):
        plot_cfg = {}

    metrics_graph_cfg = plot_cfg.get("metrics_graph")
    if not isinstance(metrics_graph_cfg, dict):
        metrics_graph_cfg = {}

    resolved = dict(defaults)

    # Resolve metrics: priority to plot.metrics_graph.metrics, fallback to top-level metrics
    metrics_value = metrics_graph_cfg.get("metrics")
    if metrics_value is None and isinstance(cfgd, dict):
        # Fall back to top-level metrics config
        metrics_value = cfgd.get("metrics")
    
    if isinstance(metrics_value, str):
        resolved["metrics"] = [part.strip() for part in metrics_value.split(",") if part.strip()]
    elif isinstance(metrics_value, list):
        resolved["metrics"] = [str(item).strip() for item in metrics_value if str(item).strip()]

    comparison_mode = str(metrics_graph_cfg.get("comparison_mode") or "").strip().lower()
    if comparison_mode in {"validation_periods", "stations"}:
        resolved["comparison_mode"] = comparison_mode

    group_output_by = str(metrics_graph_cfg.get("group_output_by") or "").strip().lower()
    if group_output_by in {"station", "pollutant"}:
        resolved["group_output_by"] = group_output_by

    return resolved


def _resolve_plot_title_templates(config_paths: list[str] | None) -> dict[str, str]:
    """Resolve optional static image title templates from the first plotting config.

    Reads ``plot.title_templates`` from the first config file and merges any
    non-empty values over the hard-coded defaults.

    Args:
        config_paths: Resolved config file paths; the first one is used.

    Returns:
        A mapping of chart-type key → format string template.
    """
    defaults = {
        "prediction_static": "Prediction vs Actual ({target})\n{algorithm} | Station {station_id}",
        "scatter_static": (
            "Actual vs Predicted - {target}\n"
            "{algorithm} | Station {station_id}\n"
            "Evaluation rows from validation/validation.csv; training rows from validation/training_predictions.csv"
        ),
        "baseline_scatter_static": "Actual vs Predicted - {target}\n{algorithm} | Station {station_id}",
        "metrics_by_period": "Metrics by Period - Station {station_id} - {pollutant}",
        "metrics_by_station": "Metrics by Station - {pollutant} - {period_name}\n{period_start} to {period_end}",
    }
    if not config_paths:
        return defaults

    cfgd, _ = _load_cfg(config_paths[0])
    plot_cfg = cfgd.get("plot") if isinstance(cfgd, dict) else None
    if not isinstance(plot_cfg, dict):
        return defaults

    raw_templates = plot_cfg.get("title_templates")
    if not isinstance(raw_templates, dict):
        return defaults

    resolved = dict(defaults)
    for key in defaults:
        value = raw_templates.get(key)
        if isinstance(value, str) and value.strip():
            resolved[key] = value
    return resolved
