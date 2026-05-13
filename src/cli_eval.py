"""Evaluation command entrypoint separated from the monolithic CLI module."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

from . import cli as cli_impl


def eval_cmd() -> None:
    """Run evaluation for one or more configs and emit JSON + comparison data.

    For multi-config runs this command builds a normalized comparison payload and
    writes a cross-run CSV that preserves period-level rows when configured.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", action="append")
    parser.add_argument("--config-dir", action="append")
    parser.add_argument("--run-dir", action="append")
    parser.add_argument(
        "--use-training-normalization",
        dest="use_training_normalization",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Use normalization statistics saved during training. Defaults to enabled.",
    )
    parser.add_argument("--comparison-csv")
    args = parser.parse_args()

    config_paths = cli_impl._expand_config_paths(args.config, args.config_dir)

    run_dirs = args.run_dir or []
    if run_dirs and len(run_dirs) != len(config_paths):
        raise ValueError("When evaluating multiple configs, provide the same number of --run-dir values as --config values.")

    results = []
    total_configs = len(config_paths)
    for index, config_path in enumerate(config_paths, start=1):
        config_name = Path(config_path).name
        remaining = total_configs - index
        remaining_text = f"({remaining} remaining)" if remaining > 0 else "(final config)"
        print(
            f"\n{'='*70}\nEvaluating config {index}/{total_configs}: {config_name} {remaining_text}\n{'='*70}",
            file=sys.stderr,
            flush=True,
        )
        run_dir_arg = run_dirs[index - 1] if run_dirs else None
        results.append(
            cli_impl._eval_one(
                config_path,
                run_dir_arg,
                use_training_normalization=args.use_training_normalization,
            )
        )
        print(
            f"Completed config {index}/{total_configs}: {config_name}\n",
            file=sys.stderr,
            flush=True,
        )

    if len(results) > 1:
        payload = {"runs": results}
        comparison = cli_impl._build_eval_comparison(results)
        if comparison is not None:
            rows = list(comparison.get("rows") or [])
            has_non_overall_periods = any(
                str(row.get("validation_period_name") or "overall") != "overall"
                for row in rows
            )
            configured_period_count = 0
            configured_periods: list[dict] = []
            for config_path in config_paths:
                cfgd_for_count, _ = cli_impl._load_cfg(config_path)
                if bool(cfgd_for_count.get("eval", {}).get("separate_validation_period_results", False)):
                    periods = list(
                        cfgd_for_count.get("data", {}).get("validation_periods")
                        or cfgd_for_count.get("data", {}).get("testing_periods")
                        or []
                    )
                    configured_period_count = max(
                        configured_period_count,
                        len(periods),
                    )
                    if len(periods) > len(configured_periods):
                        configured_periods = periods
            compact_rows = [
                row
                for row in rows
                if str(row.get("validation_period_name") or "overall") == "overall"
                and not str(row.get("algorithm") or "").startswith("Baseline")
            ]
            csv_payload = comparison
            if has_non_overall_periods:
                period_rows = [
                    row
                    for row in rows
                    if str(row.get("validation_period_name") or "overall") != "overall"
                    and not str(row.get("algorithm") or "").startswith("Baseline")
                ]
                if period_rows:
                    csv_payload = {**comparison, "rows": period_rows}
            if (
                (not has_non_overall_periods)
                and configured_period_count > 1
                and compact_rows
                and len(compact_rows) >= len(results)
            ):
                expanded_rows = []
                for row in compact_rows:
                    for period_index in range(configured_period_count):
                        expanded_row = dict(row)
                        period_cfg = configured_periods[period_index] if period_index < len(configured_periods) else {}
                        period_start = period_cfg.get("start")
                        period_end = period_cfg.get("end")
                        period_start_iso = (
                            pd.Timestamp(period_start).isoformat()
                            if period_start is not None
                            else None
                        )
                        period_end_iso = (
                            pd.Timestamp(period_end).isoformat()
                            if period_end is not None
                            else None
                        )
                        expanded_row["validation_period_name"] = f"period_{period_index + 1:02d}"
                        expanded_row["validation_period_start"] = period_start_iso
                        expanded_row["validation_period_end"] = period_end_iso
                        expanded_row["validation_period"] = (
                            f"{period_start_iso} -> {period_end_iso}"
                            if period_start_iso and period_end_iso
                            else None
                        )
                        expanded_rows.append(expanded_row)
                csv_payload = {**comparison, "rows": expanded_rows}
            elif (not has_non_overall_periods) and compact_rows and len(compact_rows) >= len(results):
                csv_payload = {**comparison, "rows": compact_rows}
            comparison["csv"] = cli_impl._write_eval_comparison_csv(
                csv_payload,
                args.comparison_csv,
                config_dirs=args.config_dir,
            )
            comparison["station_period_metric_charts"] = []
            payload["comparison"] = comparison
    else:
        payload = results[0]
    print(json.dumps(payload, indent=2))
