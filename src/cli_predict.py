"""Predict command entrypoint separated from the monolithic CLI module."""

from __future__ import annotations

import argparse
import json

import pandas as pd

from . import cli as cli_impl


def predict_cmd() -> None:
    """Run scalar prediction for a trained run using CLI-provided feature values.

    The command validates feature counts against the active config, applies saved
    normalization statistics, and returns prediction + effective feature payload.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--input",
        required=True,
        help=(
            "Comma-separated neighbor-station feature values in this order: "
            "pollutant-major, station-major, lag-major (lag 0 first)."
        ),
    )
    parser.add_argument(
        "--target-history",
        help=(
            "Optional comma-separated target-history values. For single-target mode provide "
            "target_history_lags values. For multi-target mode you may provide either "
            "target_history_lags values (broadcast to all pollutants) or "
            "target_history_lags * num_pollutants values."
        ),
    )
    parser.add_argument("--run-dir")
    args = parser.parse_args()

    total_phases = 4
    cli_impl._report_phase("Loading configuration and resolving run directory", 1, total_phases)
    cfgd, base = cli_impl._load_cfg(args.config)
    artifacts_base = cli_impl._artifact_base_dir(cfgd, base)
    run_dir = cli_impl._resolve_run_dir(artifacts_base, args.run_dir)
    run_cfg = cli_impl._load_run_config(run_dir)
    cli_impl._ensure_config_alignment(cfgd, run_cfg)

    cli_impl._report_phase("Loading trained pipeline and normalization statistics", 2, total_phases)
    pipeline = cli_impl._load_chunked_joblib(run_dir / "pipeline.pkl")
    norm_stats = cli_impl._load_saved_normalization_stats(
        run_dir,
        normalization_method=cli_impl._normalization_type(cfgd["data"]),
    )

    cli_impl._report_phase("Preparing scalar input features and generating prediction", 3, total_phases)
    raw_values = cli_impl._parse_float_list(args.input, option_name="--input")
    target_pollutants = cli_impl.resolve_target_pollutants(cfgd["data"])
    station_history_lags = cli_impl._positive_lag_count(cfgd["data"], "station_history_lags", 1)
    target_history_lags = cli_impl._positive_lag_count(cfgd["data"], "target_history_lags", 1)
    expected_neighbors = len(cfgd["data"]["input_station_ids"]) * len(target_pollutants) * station_history_lags
    if len(raw_values) != expected_neighbors:
        if len(raw_values) > expected_neighbors and expected_neighbors > 0:
            # Backward compatibility: older workflows sometimes pass a wider
            # pollutant-major vector than the current model expects.
            raw_values = raw_values[:expected_neighbors]
            cli_impl._report_note(
                "Received extra --input values; using the first "
                f"{expected_neighbors} values required by the configured model."
            )
        else:
            raise ValueError(
                f"Expected {expected_neighbors} neighbor values in --input, received {len(raw_values)}. "
                f"Configured station_history_lags={station_history_lags}."
            )

    feature_values: dict[str, float] = {}
    cursor = 0
    is_multi_target = len(target_pollutants) > 1
    for pollutant in target_pollutants:
        for station_id in cfgd["data"]["input_station_ids"]:
            for lag in range(station_history_lags):
                if lag == 0:
                    feature_name = (
                        f"neighbor__{pollutant}__{station_id}"
                        if is_multi_target
                        else f"neighbor__{station_id}"
                    )
                else:
                    feature_name = (
                        f"neighbor__{pollutant}__{station_id}__lag_{lag}"
                        if is_multi_target
                        else f"neighbor__{station_id}__lag_{lag}"
                    )
                feature_values[feature_name] = raw_values[cursor]
                cursor += 1

    if bool(cfgd["data"].get("allow_target_history", False)):
        target_history_values = cli_impl._parse_float_list(args.target_history, option_name="--target-history")
        if not target_history_values:
            raise ValueError("--target-history is required when allow_target_history=true.")

        expected_per_target = target_history_lags
        expected_total = target_history_lags * len(target_pollutants)
        if is_multi_target and len(target_history_values) == expected_per_target:
            expanded_values = []
            for _ in target_pollutants:
                expanded_values.extend(target_history_values)
            target_history_values = expanded_values
        elif len(target_history_values) not in {expected_per_target, expected_total}:
            raise ValueError(
                "--target-history value count is invalid. "
                f"Expected {expected_per_target} (single-target or broadcast) or {expected_total} values."
            )

        if not is_multi_target and len(target_history_values) != expected_per_target:
            raise ValueError(
                f"Expected {expected_per_target} --target-history values for single-target mode, "
                f"received {len(target_history_values)}."
            )

        target_cursor = 0
        if is_multi_target:
            for pollutant in target_pollutants:
                for lag in range(1, target_history_lags + 1):
                    feature_values[f"target_history_lag_{lag}__{pollutant}"] = target_history_values[target_cursor]
                    target_cursor += 1
        else:
            for lag in range(1, target_history_lags + 1):
                feature_values[f"target_history_lag_{lag}"] = target_history_values[target_cursor]
                target_cursor += 1

    feature_frame = pd.DataFrame([feature_values])
    feature_frame = cli_impl.apply_norm_stats(feature_frame, norm_stats)
    prediction = pipeline.predict_scalar(feature_frame.iloc[0].to_dict())

    print(
        json.dumps(
            {
                "run_dir": str(run_dir),
                "prediction": prediction,
                "features": feature_values,
            },
            indent=2,
        )
    )
    cli_impl._report_phase("Prediction complete", 4, total_phases)
