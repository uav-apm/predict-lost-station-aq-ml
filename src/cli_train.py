"""Train command entrypoint separated from the monolithic CLI module."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import cli as cli_impl


def train_cmd() -> None:
    """Run training for one or more configs and print a JSON summary payload.

    The command supports direct config paths and config-directory expansion.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", action="append")
    parser.add_argument("--config-dir", action="append")
    parser.add_argument("--comment", help="Optional run comment added to generated artifact folder names.")
    parser.add_argument(
        "--save-normalization",
        dest="save_normalization",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Persist training normalization statistics for later eval/predict reuse. Defaults to enabled.",
    )
    args = parser.parse_args()

    config_paths = cli_impl._expand_config_paths(args.config, args.config_dir)
    results = []
    total_configs = len(config_paths)

    for index, config_path in enumerate(config_paths, start=1):
        config_name = Path(config_path).name
        remaining = total_configs - index
        remaining_text = f"({remaining} remaining)" if remaining > 0 else "(final config)"
        print(
            f"\n{'='*70}\nTraining config {index}/{total_configs}: {config_name} {remaining_text}\n{'='*70}",
            file=sys.stderr,
            flush=True,
        )
        result = cli_impl._train_one(
            config_path,
            run_comment=args.comment,
            save_normalization=args.save_normalization,
        )
        results.append(result)
        print(
            f"Completed config {index}/{total_configs}: {config_name}\n",
            file=sys.stderr,
            flush=True,
        )

    if len(results) > 1:
        payload = {"runs": results}
    else:
        payload = results[0]
    print(json.dumps(payload, indent=2))
