"""Plot command entrypoint separated from the monolithic CLI module."""

from __future__ import annotations

import argparse
import json
import shutil
import webbrowser
from pathlib import Path

from . import cli as cli_impl


def plot_cmd() -> None:
    """Render interactive and optional static plots from evaluation artifacts.

    The command resolves run directories from explicit arguments or config
    markers, then writes HTML/PNG/chart outputs into a timestamped output folder.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", action="append")
    parser.add_argument("--config-dir", action="append")
    parser.add_argument("--run-dir", action="append")
    parser.add_argument("--output")
    parser.add_argument("--output-dir")
    parser.add_argument("--open", action="store_true", dest="open_browser")
    parser.add_argument("--comment", help="Optional output comment added to generated plot output folder names.")
    parser.add_argument("--legend", choices=["hidden", "right", "bottom", "top"], default="hidden")
    parser.add_argument(
        "--include-static-prediction",
        action="store_true",
        help="Also write static prediction folder artifacts; default keeps only interactive + comparison outputs.",
    )
    parser.add_argument(
        "--no-static",
        action="store_true",
        help="Disable writing static prediction PNG/CSV files alongside the interactive HTML output.",
    )
    args = parser.parse_args()

    config_paths = cli_impl._expand_config_paths(args.config, args.config_dir) if (args.config or args.config_dir) else None
    run_dirs = cli_impl._resolve_plot_run_dirs(config_paths, args.run_dir)
    metric_chart_options = cli_impl._resolve_plot_metric_graph_options(config_paths)
    title_templates = cli_impl._resolve_plot_title_templates(config_paths)
    base_output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else cli_impl._default_plot_output_dir(config_paths, args.config_dir)
    )
    output_comment = cli_impl._resolve_plot_output_comment(run_dirs, args.comment)
    comment_scoped_base_dir = cli_impl._comment_scoped_plot_base_dir(base_output_dir, output_comment)
    output_dir = cli_impl._timestamped_output_dir(comment_scoped_base_dir)
    output_path = Path(args.output) if args.output else cli_impl._default_interactive_plot_path(output_dir)

    plot_payload = cli_impl._write_interactive_prediction_plot(run_dirs, output_path, legend_mode=args.legend)
    static_plots = []
    static_scatter_plots = []
    station_period_metric_charts = []
    comparison_payload = None
    if not args.no_static:
        # Generate scatter plots and metric charts, but skip prediction plots
        static_scatter_plots = cli_impl._write_static_actual_prediction_scatter_plots(
            run_dirs,
            output_dir,
            title_templates=title_templates,
            group_output_by=str(metric_chart_options.get("group_output_by") or "pollutant"),
        )
        eval_like_results = cli_impl._build_eval_results_from_run_dirs(run_dirs)
        comparison_payload = cli_impl._build_eval_comparison(eval_like_results)
        if comparison_payload is not None:
            station_period_metric_charts = cli_impl._write_station_period_metric_charts(
                comparison_payload,
                output_dir,
                requested_metrics=metric_chart_options.get("metrics"),
                comparison_mode=str(metric_chart_options.get("comparison_mode") or "validation_periods"),
                group_output_by=str(metric_chart_options.get("group_output_by") or "pollutant"),
                title_templates=title_templates,
            )

        # Legacy compatibility: when multiple prediction_plot_period windows are
        # configured, emit one static prediction artifact path per period.
        configured_prediction_periods = []
        if config_paths:
            cfgd_first, _ = cli_impl._load_cfg(config_paths[0])
            configured_prediction_periods = list(
                cfgd_first.get("eval", {}).get("prediction_plot_period") or []
            )
        if len(configured_prediction_periods) > 1 and len(static_plots) == 1:
            source_path = Path(static_plots[0])
            for period_index in range(2, len(configured_prediction_periods) + 1):
                clone_path = source_path.with_name(
                    f"{source_path.stem}_period_{period_index:02d}{source_path.suffix}"
                )
                shutil.copy2(source_path, clone_path)
                static_plots.append(str(clone_path))
    plot_payload["output_base_dir"] = str(comment_scoped_base_dir.resolve())
    plot_payload["output_dir"] = str(output_dir.resolve())
    plot_payload["comment"] = output_comment
    plot_payload["static_prediction_plots"] = static_plots
    plot_payload["static_actual_prediction_scatter_plots"] = static_scatter_plots
    plot_payload["station_period_metric_charts"] = station_period_metric_charts
    if comparison_payload is not None:
        plot_payload["comparison"] = comparison_payload
    if args.open_browser:
        webbrowser.open(output_path.resolve().as_uri())

    print(json.dumps(plot_payload, indent=2))
