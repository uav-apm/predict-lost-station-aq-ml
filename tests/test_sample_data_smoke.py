from pathlib import Path

import pandas as pd
import yaml

from src.cli import eval_cmd, plot_cmd, predict_cmd, train_cmd


def test_example_profile_runs_end_to_end(example_profile_copy, run_cli):
    train_output = run_cli(
        train_cmd,
        ["aq-spatial-reconstruction-train", "--config", str(example_profile_copy)],
    )
    run_dir = Path(train_output["run_dir"])

    eval_output = run_cli(
        eval_cmd,
        ["aq-spatial-reconstruction-eval", "--config", str(example_profile_copy)],
    )
    predict_output = run_cli(
        predict_cmd,
        [
            "aq-spatial-reconstruction-predict",
            "--config",
            str(example_profile_copy),
            "--input",
            "58.0,61.0,56.0,20.0,19.0,21.0",
        ],
    )

    assert Path(eval_output["validation"]).exists()
    assert Path(train_output["log_file"]).exists()
    assert Path(eval_output["log_file"]).exists()
    assert Path(eval_output["details"]).exists()
    assert "prediction" in predict_output
    assert predict_output["run_dir"] == str(run_dir)


def test_train_accepts_multiple_configs(example_profile_copy, run_cli):
    output = run_cli(
        train_cmd,
        [
            "aq-spatial-reconstruction-train",
            "--config",
            str(example_profile_copy),
            "--config",
            str(example_profile_copy),
        ],
    )

    assert len(output["runs"]) == 2
    assert Path(output["runs"][0]["run_dir"]).exists()
    assert Path(output["runs"][1]["run_dir"]).exists()


def test_eval_multiple_configs_writes_comparison_csv(example_profile_copy, run_cli):
    comparison_csv = Path(example_profile_copy).parent / "comparison.csv"

    run_cli(
        train_cmd,
        [
            "aq-spatial-reconstruction-train",
            "--config",
            str(example_profile_copy),
            "--config",
            str(example_profile_copy),
        ],
    )

    eval_output = run_cli(
        eval_cmd,
        [
            "aq-spatial-reconstruction-eval",
            "--config",
            str(example_profile_copy),
            "--config",
            str(example_profile_copy),
            "--comparison-csv",
            str(comparison_csv),
        ],
    )

    assert "comparison" in eval_output
    assert Path(eval_output["comparison"]["csv"]).exists()
    comparison_df = pd.read_csv(eval_output["comparison"]["csv"])
    assert len(comparison_df) == 2
    assert "RMSE" in comparison_df.columns


def test_eval_multiple_configs_keeps_validation_periods_separate_in_comparison(example_profile_copy, run_cli):
    cfg_path = Path(example_profile_copy)
    comparison_csv = cfg_path.parent / "comparison_by_period.csv"

    with cfg_path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)

    cfg["data"]["testing_periods"] = [
        {"start": "2025-01-15", "end": "2025-01-15"},
        {"start": "2025-01-15", "end": "2025-01-15"},
    ]
    cfg["eval"]["separate_validation_period_results"] = True
    with cfg_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg, handle, sort_keys=False)

    run_cli(
        train_cmd,
        [
            "aq-spatial-reconstruction-train",
            "--config",
            str(cfg_path),
            "--config",
            str(cfg_path),
        ],
    )

    eval_output = run_cli(
        eval_cmd,
        [
            "aq-spatial-reconstruction-eval",
            "--config",
            str(cfg_path),
            "--config",
            str(cfg_path),
            "--comparison-csv",
            str(comparison_csv),
        ],
    )

    comparison_df = pd.read_csv(eval_output["comparison"]["csv"])
    assert len(comparison_df) == 4
    assert "validation_period" in comparison_df.columns
    assert "validation_period_start" in comparison_df.columns
    assert "validation_period_end" in comparison_df.columns
    assert set(comparison_df["validation_period_name"]) == {"period_01", "period_02"}
    assert set(comparison_df["validation_period_start"]) == {"2025-01-15T00:00:00"}


def test_eval_can_take_run_dir_from_config(example_profile_copy, run_cli):
    train_output = run_cli(
        train_cmd,
        ["aq-spatial-reconstruction-train", "--config", str(example_profile_copy)],
    )

    cfg_path = Path(example_profile_copy)
    with cfg_path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    cfg["artifacts"]["eval_run_dir"] = train_output["run_dir"]
    with cfg_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg, handle, sort_keys=False)

    eval_output = run_cli(
        eval_cmd,
        ["aq-spatial-reconstruction-eval", "--config", str(example_profile_copy)],
    )

    assert eval_output["run_dir"] == train_output["run_dir"]


def test_example_profile_multi_output_targets(example_profile_copy, run_cli):
    cfg_path = Path(example_profile_copy)
    with cfg_path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)

    cfg["data"]["target_pollutant"] = ["PM2.5 (µg/m³)", "NO2 (µg/m³)"]
    cfg["model"]["target"] = ["PM2.5 (µg/m³)", "NO2 (µg/m³)"]
    cfg["data"]["impute_columns"] = ["PM2.5 (µg/m³)", "NO2 (µg/m³)"]
    with cfg_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg, handle, sort_keys=False)

    train_output = run_cli(
        train_cmd,
        ["aq-spatial-reconstruction-train", "--config", str(cfg_path)],
    )
    run_dir = Path(train_output["run_dir"])

    eval_output = run_cli(
        eval_cmd,
        ["aq-spatial-reconstruction-eval", "--config", str(cfg_path)],
    )
    validation_df = yaml.safe_load((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
    assert validation_df["target_names"] == ["target_value__PM2.5 (µg/m³)", "target_value__NO2 (µg/m³)"]

    predict_output = run_cli(
        predict_cmd,
        [
            "aq-spatial-reconstruction-predict",
            "--config",
            str(cfg_path),
            "--input",
            "58.0,61.0,56.0,20.0,19.0,21.0",
        ],
    )

    assert Path(eval_output["validation"]).exists()
    assert isinstance(predict_output["prediction"], dict)
    assert "target_value__PM2.5 (µg/m³)" in predict_output["prediction"]
    assert "target_value__NO2 (µg/m³)" in predict_output["prediction"]


def test_predict_supports_station_history_lags(example_profile_copy, run_cli):
    cfg_path = Path(example_profile_copy)
    with cfg_path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)

    cfg["data"]["station_history_lags"] = 2
    with cfg_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg, handle, sort_keys=False)

    run_cli(
        train_cmd,
        ["aq-spatial-reconstruction-train", "--config", str(cfg_path)],
    )

    predict_output = run_cli(
        predict_cmd,
        [
            "aq-spatial-reconstruction-predict",
            "--config",
            str(cfg_path),
            "--input",
            "58.0,57.0,61.0,60.0,56.0,55.0,20.0,19.0,19.0,18.0,21.0,20.0",
        ],
    )

    assert "prediction" in predict_output


def test_imputation_stats_csv_contains_per_station_rows(example_profile_copy, run_cli):
    train_output = run_cli(
        train_cmd,
        ["aq-spatial-reconstruction-train", "--config", str(example_profile_copy)],
    )

    run_dir = Path(train_output["run_dir"])
    stats_csv = run_dir / "imputation_stats.csv"
    stats_df = pd.read_csv(stats_csv)

    assert "station_id" in stats_df.columns
    assert "__all__" in set(stats_df["station_id"].astype(str))

    with Path(example_profile_copy).open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    expected_stations = set((cfg["data"].get("station_files") or {}).keys())
    observed_stations = set(stats_df["station_id"].astype(str)) - {"__all__"}
    assert expected_stations.issubset(observed_stations)


def test_eval_can_write_separate_results_per_validation_period(example_profile_copy, run_cli):
    cfg_path = Path(example_profile_copy)
    with cfg_path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)

    cfg["data"]["testing_periods"] = [
        {"start": "2025-01-15", "end": "2025-01-15"},
        {"start": "2025-01-15", "end": "2025-01-15"},
    ]
    cfg["eval"]["separate_validation_period_results"] = True
    with cfg_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg, handle, sort_keys=False)

    run_cli(
        train_cmd,
        ["aq-spatial-reconstruction-train", "--config", str(cfg_path)],
    )

    eval_output = run_cli(
        eval_cmd,
        ["aq-spatial-reconstruction-eval", "--config", str(cfg_path)],
    )

    assert Path(eval_output["validation"]).exists()
    assert "period_results" in eval_output
    assert len(eval_output["period_results"]) == 2
    assert Path(eval_output["period_results_summary"]).exists()
    for period_result in eval_output["period_results"]:
        assert Path(period_result["validation"]).exists()
        assert Path(period_result["metrics_file"]).exists()


def test_eval_can_disable_combined_output_when_writing_separate_period_results(example_profile_copy, run_cli):
    cfg_path = Path(example_profile_copy)
    with cfg_path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)

    cfg["data"]["testing_periods"] = [
        {"start": "2025-01-15", "end": "2025-01-15"},
        {"start": "2025-01-15", "end": "2025-01-15"},
    ]
    cfg["eval"]["separate_validation_period_results"] = True
    cfg["eval"]["only_separate_validation_period_results"] = True
    with cfg_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg, handle, sort_keys=False)

    train_output = run_cli(
        train_cmd,
        ["aq-spatial-reconstruction-train", "--config", str(cfg_path)],
    )

    eval_output = run_cli(
        eval_cmd,
        ["aq-spatial-reconstruction-eval", "--config", str(cfg_path)],
    )

    run_dir = Path(train_output["run_dir"])
    validation_dir = run_dir / "validation"
    assert eval_output["combined_result_disabled"] is True
    assert eval_output["validation"] is None
    assert eval_output["metrics"] is None
    assert not (validation_dir / "validation.csv").exists()
    assert not (validation_dir / "metrics.json").exists()
    assert len(eval_output["period_results"]) == 2
    for period_result in eval_output["period_results"]:
        assert Path(period_result["validation"]).exists()
        assert Path(period_result["metrics_file"]).exists()
