from pathlib import Path

import yaml

from src.cli import train_cmd


def test_training_creates_expected_artifacts(example_profile_copy, run_cli):
    output = run_cli(
        train_cmd,
        ["aq-spatial-reconstruction-train", "--config", str(example_profile_copy)],
    )

    run_dir = Path(output["run_dir"])
    pipeline_parts = sorted(run_dir.glob("pipeline.pkl.part*"))

    assert run_dir.exists()
    assert pipeline_parts
    assert (run_dir / "config_snapshot.yaml").exists()
    assert (run_dir / "run_metadata.json").exists()
    assert (run_dir / "norm_stats.json").exists()
    assert (run_dir / "column_statistics.json").exists()
    assert (run_dir / "validation").is_dir()
    assert (run_dir / "imputation_stats.json").exists()
    assert (run_dir / "imputation_details.json").exists()
    assert (run_dir / "imputed_dataset.csv").exists()
    assert (run_dir.parent / "latest_run.txt").read_text(encoding="utf-8").strip() == str(run_dir)


def test_training_can_dump_prefit_training_data(example_profile_copy, run_cli):
    cfg_path = Path(example_profile_copy)
    with cfg_path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    cfg["artifacts"]["dump_training_data"] = True
    cfg["model"]["algorithm"] = "sklearn.linear_model.LinearRegression"
    cfg["model"]["params"] = {"fit_intercept": True}
    with cfg_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg, handle, sort_keys=False)

    output = run_cli(
        train_cmd,
        ["aq-spatial-reconstruction-train", "--config", str(cfg_path)],
    )

    run_dir = Path(output["run_dir"])
    dump_dir = run_dir / "training_dump"

    assert (dump_dir / "training_supervised_rows.csv").exists()
    assert (dump_dir / "training_features_raw.csv").exists()
    assert (dump_dir / "training_features_model_input.csv").exists()
    assert (dump_dir / "training_target.csv").exists()
    assert (dump_dir / "training_station_matrix.csv").exists()
    assert (dump_dir / "training_summary.json").exists()
