from pathlib import Path
import json
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
import pytest

from src import cli
from src.forecasting import build_period_mask, normalize_periods, years_for_periods


def test_resolve_path_prefers_profile_directory(tmp_path):
    base = tmp_path / "profile"
    base.mkdir()
    csv_path = base / "sample.csv"
    csv_path.write_text("datetime,station_id,PM2.5\n2023-01-01,a,1\n", encoding="utf-8")

    assert cli._resolve_path(base, "sample.csv") == csv_path


def test_artifact_base_dir_uses_project_root_for_workspace_relative_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "PROJECT_ROOT", tmp_path)
    base = tmp_path / "configs" / "delhi" / "subconfigs" / "lstm" / "combined"
    base.mkdir(parents=True)
    cfgd = {"artifacts": {"dir": "configs/delhi/artifacts"}}

    resolved = cli._artifact_base_dir(cfgd, base)

    assert resolved == (tmp_path / "configs" / "delhi" / "artifacts")


def test_artifact_base_dir_keeps_simple_name_base_relative(tmp_path):
    base = tmp_path / "profile"
    base.mkdir(parents=True)
    cfgd = {"artifacts": {"dir": "artifacts"}}

    resolved = cli._artifact_base_dir(cfgd, base)

    assert resolved == (base / "artifacts")


def test_timestamped_run_dir_updates_latest_marker(tmp_path):
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()

    run_dir = cli._timestamped_run_dir(artifacts)

    assert run_dir.exists()
    assert (artifacts / "latest_run.txt").read_text(encoding="utf-8").strip() == str(run_dir)


def test_timestamped_run_dir_updates_config_specific_marker(tmp_path):
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()

    run_dir = cli._timestamped_run_dir(
        artifacts,
        latest_markers=["latest_run_example.txt"],
        run_label="example",
    )

    assert run_dir.exists()
    assert run_dir.name.startswith("example_")
    assert (artifacts / "latest_run.txt").read_text(encoding="utf-8").strip() == str(run_dir)
    assert (artifacts / "latest_run_example.txt").read_text(encoding="utf-8").strip() == str(run_dir)


def test_timestamped_run_dir_updates_all_marker_directories(tmp_path):
    artifacts = tmp_path / "artifacts"
    group_dir = artifacts / "campaign_1"
    artifacts.mkdir()

    run_dir = cli._timestamped_run_dir(
        group_dir,
        latest_markers=["latest_run_example.txt"],
        run_label="example",
        marker_dirs=[artifacts, group_dir],
    )

    assert run_dir.parent == group_dir
    assert (artifacts / "latest_run.txt").read_text(encoding="utf-8").strip() == str(run_dir)
    assert (artifacts / "latest_run_example.txt").read_text(encoding="utf-8").strip() == str(run_dir)
    assert (group_dir / "latest_run.txt").read_text(encoding="utf-8").strip() == str(run_dir)
    assert (group_dir / "latest_run_example.txt").read_text(encoding="utf-8").strip() == str(run_dir)


def test_compose_run_label_appends_comment():
    label = cli._compose_run_label("configs/example/example.yaml", run_comment="exp A")

    assert label == "example_exp A"


def test_comment_artifact_dir_creates_sanitized_group_folder(tmp_path):
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()

    comment_dir = cli._comment_artifact_dir(artifacts, "campaign 1 / April")

    assert comment_dir == artifacts / "campaign_1_April"
    assert comment_dir.exists()


def test_missing_value_strategy_always_returns_drop():
    assert cli._missing_value_strategy({}) == "drop"
    assert cli._missing_value_strategy({"missing_value_strategy": "drop"}) == "drop"


def test_normalization_type_supports_aliases_and_legacy_bool():
    assert cli._normalization_type({"normalization": "standard"}) == "zscore"
    assert cli._normalization_type({"normalization": "min-max"}) == "minmax"
    assert cli._normalization_type({"normalize": True}) == "zscore"
    assert cli._normalization_type({"normalize": False}) == "none"


def test_save_training_normalization_defaults_to_true_and_supports_override():
    assert cli._should_save_training_normalization({}) is True
    assert cli._should_save_training_normalization({"save_training_normalization": False}) is False
    assert cli._should_save_training_normalization({"save_training_normalization": False}, True) is True


def test_use_training_normalization_defaults_to_true_and_supports_override():
    assert cli._should_use_training_normalization({}) is True
    assert cli._should_use_training_normalization({"use_training_normalization": False}) is False
    assert cli._should_use_training_normalization({"use_training_normalization": False}, True) is True


def test_load_saved_normalization_stats_returns_identity_when_normalization_disabled(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    stats = cli._load_saved_normalization_stats(run_dir, normalization_method="none")

    assert stats == {"method": "none", "mu": {}, "sigma": {}}


def test_load_saved_normalization_stats_requires_file_for_active_normalization(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    with pytest.raises(FileNotFoundError, match="Saved normalization stats"):
        cli._load_saved_normalization_stats(run_dir, normalization_method="zscore")


def test_resolve_run_dir_prefers_config_specific_latest_marker(tmp_path):
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    generic_run = artifacts / "20260101_010101_0"
    specific_run = artifacts / "20260101_010102_0"
    generic_run.mkdir()
    specific_run.mkdir()
    (artifacts / "latest_run.txt").write_text(str(generic_run), encoding="utf-8")
    (artifacts / "latest_run_example.txt").write_text(str(specific_run), encoding="utf-8")

    resolved = cli._resolve_run_dir(artifacts, None, latest_markers=["latest_run_example.txt"])

    assert resolved == specific_run


def test_resolve_run_dir_falls_back_to_nested_snapshot_runs(tmp_path):
    artifacts = tmp_path / "artifacts"
    nested_run = artifacts / "campaign_1" / "example_20260101_010101_0"
    nested_run.mkdir(parents=True)
    (nested_run / "config_snapshot.yaml").write_text("model: {}\n", encoding="utf-8")

    resolved = cli._resolve_run_dir(artifacts, None)

    assert resolved == nested_run


def test_load_input_dataset_keeps_required_station_columns(tmp_path):
    base = tmp_path / "cfg"
    data_dir = base / "data"
    data_dir.mkdir(parents=True)
    (data_dir / "sample.csv").write_text(
        "datetime,PM2.5,NO2\n2023-01-01,1,2\n2023-01-01,2,3\n",
        encoding="utf-8",
    )
    cfg = {
        "data": {
            "folder": "data",
            "file_pattern": "*.csv",
            "datetime_col": "datetime",
            "sort_col": "datetime",
            "columns": ["NO2"],
            "station_id_col": None,
            "station_id_from": "filename",
            "target_station_id": "sample",
            "input_station_ids": ["sample"],
            "target_pollutant": "PM2.5",
            "impute_columns": ["PM2.5", "NO2"],
        },
        "model": {"target": "PM2.5"},
    }

    df, _ = cli._load_input_dataset(cfg, base)

    assert list(df.columns) == ["NO2", "datetime", "station_id", "PM2.5"]


def test_load_input_dataset_supports_station_file_mapping(tmp_path):
    base = tmp_path / "cfg"
    data_dir = base / "data"
    data_dir.mkdir(parents=True)
    (data_dir / "target_station_2024.csv").write_text(
        "Timestamp,PM2.5 (µg/m³),NO2 (µg/m³)\n2024-01-01 00:00:00,10,20\n",
        encoding="utf-8",
    )
    (data_dir / "station_a_2024.csv").write_text(
        "Timestamp,PM2.5 (µg/m³),NO2 (µg/m³)\n2024-01-01 00:00:00,11,21\n",
        encoding="utf-8",
    )
    cfg = {
        "data": {
            "folder": "data",
            "datetime_col": "Timestamp",
            "sort_col": "Timestamp",
            "columns": ["NO2 (µg/m³)"],
            "station_id_col": None,
            "target_station_id": "target_station",
            "input_station_ids": ["station_a"],
            "target_pollutant": "PM2.5 (µg/m³)",
            "station_files": {
                "target_station": ["target_station_2024.csv"],
                "station_a": ["station_a_2024.csv"],
            },
            "impute_columns": ["PM2.5 (µg/m³)", "NO2 (µg/m³)"],
        },
        "model": {"target": "PM2.5 (µg/m³)"},
    }

    df, source = cli._load_input_dataset(cfg, base)

    assert set(df["station_id"]) == {"target_station", "station_a"}
    assert source["mode"] == "station_files"


def test_drop_rows_with_missing_values_tracks_counts_by_station():
    frame = pd.DataFrame(
        {
            "station_id": ["a", "a", "b", "b"],
            "PM2.5": [1.0, None, 3.0, 4.0],
            "NO2": [10.0, 20.0, None, 40.0],
        }
    )

    dropped, details = cli._drop_rows_with_missing_values(
        frame,
        target_columns=["PM2.5", "NO2"],
        group_col="station_id",
    )

    assert len(dropped) == 2
    assert details["method"] == "drop_rows"
    assert details["dropped_rows"] == 2
    assert details["columns"]["PM2.5"]["missing_before"] == 1
    assert details["columns"]["PM2.5"]["missing_after"] == 0
    assert details["by_station"]["a"]["columns"]["PM2.5"]["missing_before"] == 1
    assert details["by_station"]["b"]["columns"]["NO2"]["missing_before"] == 1


def test_ensure_config_alignment_rejects_mismatch():
    cfgd = {"data": {"target_station_id": "target_station"}, "model": {"algorithm": "random_forest"}}
    run_cfg = {"data": {"target_station_id": "other_station"}, "model": {"algorithm": "random_forest"}}

    with pytest.raises(ValueError):
        cli._ensure_config_alignment(cfgd, run_cfg)


def test_save_chunked_joblib_splits_large_payload(tmp_path):
    path = tmp_path / "pipeline.pkl"
    payload = {"values": list(range(200_000))}

    parts = cli._save_chunked_joblib(payload, path, chunk_size_bytes=1024)

    assert not path.exists()
    assert len(parts) > 1
    assert all(part.exists() for part in parts)
    assert all(part.stat().st_size <= 1024 for part in parts[:-1])
    assert cli._load_chunked_joblib(path) == payload


def test_load_chunked_joblib_supports_legacy_single_file(tmp_path):
    path = tmp_path / "pipeline.pkl"
    payload = {"message": "legacy"}
    joblib.dump(payload, path)

    assert cli._load_chunked_joblib(path) == payload


def test_train_one_groups_commented_runs_and_updates_root_markers(tmp_path, monkeypatch):
    cfg_path = tmp_path / "example.yaml"
    cfg_path.write_text("data: {}\nmodel: {}\nartifacts: {}\n", encoding="utf-8")

    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    heatmap = cache_dir / "correlation_heatmap.png"
    heatmap.write_bytes(b"heatmap")
    dataset = cache_dir / "imputed_dataset.csv"
    dataset.write_text("value\n1\n", encoding="utf-8")
    details = cache_dir / "imputation_details.json"
    details.write_text("{}", encoding="utf-8")
    stats = cache_dir / "imputation_stats.json"
    stats.write_text("{}", encoding="utf-8")
    stats_csv = cache_dir / "imputation_stats.csv"
    stats_csv.write_text("column,missing_before,missing_after\n", encoding="utf-8")
    before_csv = cache_dir / "column_statistics_before.csv"
    before_csv.write_text("column\n", encoding="utf-8")
    after_csv = cache_dir / "column_statistics_after.csv"
    after_csv.write_text("column\n", encoding="utf-8")

    cfgd = {
        "artifacts": {"dir": "artifacts", "dump_training_data": False},
        "model": {"algorithm": "random_forest", "params": {}},
        "data": {
            "target_station_id": "station_a",
            "input_station_ids": ["station_b"],
            "allow_target_history": False,
            "normalization": "zscore",
        },
    }
    prepared = {
        "train_X": pd.DataFrame({"feature_a": [1.0, 2.0], "feature_b": [3.0, 4.0]}),
        "train_y": pd.Series([0.1, 0.2], name="target_value"),
        "target_names": ["target_value"],
        "train_rows": pd.DataFrame({"prediction_time": pd.to_datetime(["2024-01-01", "2024-01-02"])}),
        "data_source": {"mode": "folder", "path": "data", "files": []},
        "norm_stats": {"method": "zscore", "mu": {"feature_a": 1.5}, "sigma": {"feature_a": 0.5}},
        "column_stats_before": {},
        "column_stats_after": {},
        "correlation_analysis": {},
        "cache_paths": {
            "heatmap": heatmap,
            "dataset": dataset,
            "details": details,
            "stats": stats,
            "stats_csv": stats_csv,
            "column_stats_before_csv": before_csv,
            "column_stats_after_csv": after_csv,
        },
        "imputation_details": {"columns": {}},
        "used_cache": True,
        "cache_dir": cache_dir,
    }

    class DummyPipeline:
        def __init__(self, _config):
            self.feature_names_ = []
            self.model = SimpleNamespace(history_=None)

        def fit(self, X, y):
            self.feature_names_ = list(X.columns)
            return self

    def fake_save_chunked_joblib(_obj, path, *, chunk_size_bytes=cli.PICKLE_CHUNK_SIZE_BYTES):
        part_path = path.parent / f"{path.name}.part0001"
        part_path.write_bytes(b"pipeline")
        return [part_path]

    monkeypatch.setattr(cli, "_load_cfg", lambda raw_path: (cfgd, tmp_path))
    monkeypatch.setattr(cli, "_prepare_training_data", lambda loaded_cfg, base: prepared)
    monkeypatch.setattr(cli, "_build_model_config", lambda loaded_cfg: object())
    monkeypatch.setattr(cli, "StationForecastPipeline", DummyPipeline)
    monkeypatch.setattr(cli, "_save_chunked_joblib", fake_save_chunked_joblib)

    output = cli._train_one(str(cfg_path), run_comment="campaign 1")

    run_dir = Path(output["run_dir"])
    metadata = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
    artifacts_root = tmp_path / "artifacts"

    assert run_dir.parent == artifacts_root / "campaign_1"
    assert run_dir.name.startswith("example_")
    assert (artifacts_root / "latest_run.txt").read_text(encoding="utf-8").strip() == str(run_dir)
    assert metadata["run_group_dir"] == str(artifacts_root / "campaign_1")
    assert metadata["normalization_stats_file"] == str(run_dir / "norm_stats.json")
    assert (run_dir / "norm_stats.json").exists()


def test_train_one_can_skip_saving_normalization_stats(tmp_path, monkeypatch):
    cfg_path = tmp_path / "example.yaml"
    cfg_path.write_text("data: {}\nmodel: {}\nartifacts: {}\n", encoding="utf-8")

    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    heatmap = cache_dir / "correlation_heatmap.png"
    heatmap.write_bytes(b"heatmap")
    dataset = cache_dir / "imputed_dataset.csv"
    dataset.write_text("value\n1\n", encoding="utf-8")
    details = cache_dir / "imputation_details.json"
    details.write_text("{}", encoding="utf-8")
    stats = cache_dir / "imputation_stats.json"
    stats.write_text("{}", encoding="utf-8")
    stats_csv = cache_dir / "imputation_stats.csv"
    stats_csv.write_text("column,missing_before,missing_after\n", encoding="utf-8")
    before_csv = cache_dir / "column_statistics_before.csv"
    before_csv.write_text("column\n", encoding="utf-8")
    after_csv = cache_dir / "column_statistics_after.csv"
    after_csv.write_text("column\n", encoding="utf-8")

    cfgd = {
        "artifacts": {"dir": "artifacts", "dump_training_data": False},
        "model": {"algorithm": "random_forest", "params": {}},
        "data": {
            "target_station_id": "station_a",
            "input_station_ids": ["station_b"],
            "allow_target_history": False,
            "normalization": "zscore",
        },
    }
    prepared = {
        "train_X": pd.DataFrame({"feature_a": [1.0, 2.0], "feature_b": [3.0, 4.0]}),
        "train_y": pd.Series([0.1, 0.2], name="target_value"),
        "target_names": ["target_value"],
        "train_rows": pd.DataFrame({"prediction_time": pd.to_datetime(["2024-01-01", "2024-01-02"])}),
        "data_source": {"mode": "folder", "path": "data", "files": []},
        "norm_stats": {"method": "zscore", "mu": {"feature_a": 1.5}, "sigma": {"feature_a": 0.5}},
        "column_stats_before": {},
        "column_stats_after": {},
        "correlation_analysis": {},
        "cache_paths": {
            "heatmap": heatmap,
            "dataset": dataset,
            "details": details,
            "stats": stats,
            "stats_csv": stats_csv,
            "column_stats_before_csv": before_csv,
            "column_stats_after_csv": after_csv,
        },
        "imputation_details": {"columns": {}},
        "used_cache": True,
        "cache_dir": cache_dir,
    }

    class DummyPipeline:
        def __init__(self, _config):
            self.feature_names_ = []
            self.model = SimpleNamespace(history_=None)

        def fit(self, X, y):
            self.feature_names_ = list(X.columns)
            return self

    def fake_save_chunked_joblib(_obj, path, *, chunk_size_bytes=cli.PICKLE_CHUNK_SIZE_BYTES):
        part_path = path.parent / f"{path.name}.part0001"
        part_path.write_bytes(b"pipeline")
        return [part_path]

    monkeypatch.setattr(cli, "_load_cfg", lambda raw_path: (cfgd, tmp_path))
    monkeypatch.setattr(cli, "_prepare_training_data", lambda loaded_cfg, base: prepared)
    monkeypatch.setattr(cli, "_build_model_config", lambda loaded_cfg: object())
    monkeypatch.setattr(cli, "StationForecastPipeline", DummyPipeline)
    monkeypatch.setattr(cli, "_save_chunked_joblib", fake_save_chunked_joblib)

    output = cli._train_one(str(cfg_path), save_normalization=False)

    run_dir = Path(output["run_dir"])
    metadata = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))

    assert output["saved_training_normalization"] is False
    assert output["normalization_stats_file"] is None
    assert metadata["saved_training_normalization"] is False
    assert metadata["normalization_stats_file"] is None
    assert not (run_dir / "norm_stats.json").exists()


def test_evaluate_run_uses_saved_training_normalization_by_default(tmp_path, monkeypatch):
    run_dir = tmp_path / "run"
    validation_dir = run_dir / "validation"
    run_dir.mkdir()
    validation_dir.mkdir()
    (run_dir / "norm_stats.json").write_text(
        json.dumps({"method": "zscore", "mu": {"feature_a": 2.0}, "sigma": {"feature_a": 0.5}}),
        encoding="utf-8",
    )

    cfgd = {
        "data": {"normalization": "zscore"},
        "eval": {"store_details": False},
    }
    heatmap = tmp_path / "heatmap.png"
    heatmap.write_bytes(b"heatmap")
    captured: dict[str, object] = {}

    def fake_prepare(_cfgd, _base, norm_stats):
        captured["norm_stats"] = norm_stats
        return {
            "cache_paths": {"heatmap": heatmap},
            "imputation_details": {"columns": {}},
            "test_X": pd.DataFrame({"feature_a": [0.0, 1.0]}),
            "test_y": pd.Series([1.0, 2.0], name="target_value"),
            "test_rows": pd.DataFrame(
                {
                    "prediction_time": pd.to_datetime(["2025-01-01", "2025-01-02"]),
                    "feature_a": [10.0, 11.0],
                    "target_value": [1.0, 2.0],
                }
            ),
            "target_names": ["target_value"],
            "supervised_df": pd.DataFrame(
                {
                    "prediction_time": pd.to_datetime(["2025-01-01", "2025-01-02"]),
                    "feature_a": [10.0, 11.0],
                    "target_value": [1.0, 2.0],
                }
            ),
            "feature_names": ["feature_a"],
            "stats_before": {},
            "stats_after": {},
            "column_stats_before": {},
            "column_stats_after": {},
            "cache_dir": tmp_path,
            "normalization_source": "training_run",
        }

    class DummyPipeline:
        def evaluate(self, X, y):
            return {
                "prediction": np.array([1.1, 1.9]),
                "actual": np.array([1.0, 2.0]),
                "metrics": {"RMSE": 0.1},
            }

    monkeypatch.setattr(cli, "_prepare_evaluation_data", fake_prepare)
    monkeypatch.setattr(cli, "_load_chunked_joblib", lambda path: DummyPipeline())
    monkeypatch.setattr(cli.RUN_LOGGER, "_log_path", None)

    output = cli._evaluate_run(cfgd, tmp_path, run_dir)

    assert captured["norm_stats"] == {"method": "zscore", "mu": {"feature_a": 2.0}, "sigma": {"feature_a": 0.5}}
    assert output["normalization_source"] == "training_run"
    assert output["normalization_stats_file"] == str(run_dir / "norm_stats.json")


def test_evaluate_run_can_skip_saved_training_normalization(tmp_path, monkeypatch):
    run_dir = tmp_path / "run"
    validation_dir = run_dir / "validation"
    run_dir.mkdir()
    validation_dir.mkdir()

    cfgd = {
        "data": {"normalization": "zscore"},
        "eval": {"store_details": False},
    }
    heatmap = tmp_path / "heatmap.png"
    heatmap.write_bytes(b"heatmap")
    captured: dict[str, object] = {}

    def fake_prepare(_cfgd, _base, norm_stats):
        captured["norm_stats"] = norm_stats
        return {
            "cache_paths": {"heatmap": heatmap},
            "imputation_details": {"columns": {}},
            "test_X": pd.DataFrame({"feature_a": [0.0, 1.0]}),
            "test_y": pd.Series([1.0, 2.0], name="target_value"),
            "test_rows": pd.DataFrame(
                {
                    "prediction_time": pd.to_datetime(["2025-01-01", "2025-01-02"]),
                    "feature_a": [10.0, 11.0],
                    "target_value": [1.0, 2.0],
                }
            ),
            "target_names": ["target_value"],
            "supervised_df": pd.DataFrame(
                {
                    "prediction_time": pd.to_datetime(["2025-01-01", "2025-01-02"]),
                    "feature_a": [10.0, 11.0],
                    "target_value": [1.0, 2.0],
                }
            ),
            "feature_names": ["feature_a"],
            "stats_before": {},
            "stats_after": {},
            "column_stats_before": {},
            "column_stats_after": {},
            "cache_dir": tmp_path,
            "normalization_source": "evaluation_data",
        }

    class DummyPipeline:
        def evaluate(self, X, y):
            return {
                "prediction": np.array([1.1, 1.9]),
                "actual": np.array([1.0, 2.0]),
                "metrics": {"RMSE": 0.1},
            }

    monkeypatch.setattr(cli, "_prepare_evaluation_data", fake_prepare)
    monkeypatch.setattr(cli, "_load_chunked_joblib", lambda path: DummyPipeline())
    monkeypatch.setattr(cli.RUN_LOGGER, "_log_path", None)

    output = cli._evaluate_run(cfgd, tmp_path, run_dir, use_training_normalization=False)

    assert captured["norm_stats"] is None
    assert output["normalization_source"] == "evaluation_data"
    assert output["normalization_stats_file"] is None


def test_period_helpers_support_year_month_and_day_ranges():
    periods = normalize_periods(
        [
            "2024",
            {"start": "2025-01", "end": "2025-02"},
            {"start": "2025-03-03", "end": "2025-03-04"},
        ]
    )

    mask = build_period_mask(
        [
            "2024-06-01",
            "2025-01-15",
            "2025-02-28",
            "2025-03-04",
            "2025-04-01",
        ],
        periods,
    )

    assert years_for_periods(periods) == [2024, 2025]
    assert mask.tolist() == [True, True, True, True, False]


def test_load_cfg_supports_extends_with_nested_override(tmp_path):
    base_cfg = tmp_path / "base.yaml"
    derived_cfg = tmp_path / "derived.yaml"

    base_cfg.write_text(
        """
data:
  folder: data
  datetime_col: Timestamp
  target_station_id: "5024"
model:
  algorithm: random_forest
  params:
    n_estimators: 100
    random_state: 42
eval:
  horizon: 0
""".strip(),
        encoding="utf-8",
    )

    derived_cfg.write_text(
        """
extends: base.yaml
model:
  algorithm: extra_trees
  params:
    n_estimators: 200
""".strip(),
        encoding="utf-8",
    )

    cfg, base = cli._load_cfg(str(derived_cfg))

    assert base == derived_cfg.parent
    assert cfg["data"]["folder"] == "data"
    assert cfg["model"]["algorithm"] == "extra_trees"
    assert cfg["model"]["params"]["n_estimators"] == 200
    assert cfg["model"]["params"]["random_state"] == 42
    assert cfg["eval"]["horizon"] == 0


def test_load_cfg_rejects_extends_cycle(tmp_path):
    first = tmp_path / "first.yaml"
    second = tmp_path / "second.yaml"
    first.write_text("extends: second.yaml\n", encoding="utf-8")
    second.write_text("extends: first.yaml\n", encoding="utf-8")

    with pytest.raises(ValueError, match="cycle"):
        cli._load_cfg(str(first))


def test_build_eval_comparison_ranks_by_rmse():
    results = [
        {
            "config": "config_a.yaml",
            "run_dir": "artifacts/run_a",
            "metrics": {"MSE": 4.0, "RMSE": 2.0, "MAE": 1.5, "R2": 0.9, "IA": 0.8},
        },
        {
            "config": "config_b.yaml",
            "run_dir": "artifacts/run_b",
            "metrics": {"MSE": 1.0, "RMSE": 1.0, "MAE": 0.8, "R2": 0.95, "IA": 0.9},
        },
    ]

    comparison = cli._build_eval_comparison(results)

    assert comparison is not None
    assert comparison["ranked_by_rmse"] is True
    rows = comparison["rows"]
    by_config = {Path(row["config"]).name: row for row in rows}
    assert by_config["config_b.yaml"]["rank_by_rmse"] == 1
    assert by_config["config_a.yaml"]["rank_by_rmse"] == 2


def test_compute_station_replacement_baselines_single_target_emits_named_experiments():
    cfgd = {"data": {"input_station_ids": ["station_a", "station_b"]}}
    test_rows = pd.DataFrame(
        {
            "neighbor__station_a": [1.0, 3.0],
            "neighbor__station_b": [3.0, 1.0],
        }
    )
    actual = np.array([2.0, 2.0])

    baselines = cli._compute_station_replacement_baselines(
        cfgd,
        test_rows,
        actual,
        target_names=["target_value"],
    )

    by_name = {item["name"]: item for item in baselines}
    assert set(by_name.keys()) == {
        "Baseline: Average of Other Stations",
        "Baseline Replace with Station station_a",
        "Baseline Replace with Station station_b",
    }
    assert by_name["Baseline: Average of Other Stations"]["metrics"]["RMSE"] == pytest.approx(0.0)


def test_build_eval_comparison_includes_station_replacement_baselines():
    results = [
        {
            "config": "config_a.yaml",
            "run_dir": "artifacts/run_a",
            "target_station_id": "124",
            "algorithm_name": "extra_trees",
            "metrics": {"RMSE": 2.0, "MAE": 1.0},
            "baseline_results": [
                {
                    "id": "baseline_mean_of_other_stations",
                    "name": "Baseline: Average of Other Stations",
                    "algorithm_name": "Baseline: Average of Other Stations",
                    "metrics": {"RMSE": 2.5, "MAE": 1.3},
                },
                {
                    "id": "baseline_replace_with_station_1421",
                    "name": "Baseline Replace with Station 1421",
                    "algorithm_name": "Baseline Replace with Station 1421",
                    "metrics": {"RMSE": 3.0, "MAE": 1.6},
                },
            ],
        }
    ]

    comparison = cli._build_eval_comparison(results)

    assert comparison is not None
    algorithms = [row["algorithm"] for row in comparison["rows"]]
    assert "extra_trees" in algorithms
    assert "Baseline: Average of Other Stations" in algorithms
    assert "Baseline Replace with Station 1421" in algorithms


def test_build_eval_comparison_expands_separate_validation_period_results():
    results = [
        {
            "config": "config_a.yaml",
            "run_dir": "artifacts/run_a",
            "metrics": {"RMSE": 99.0},
            "period_results": [
                {
                    "name": "period_01",
                    "period": {"start": "2025-01-01T00:00:00", "end": "2025-01-31T23:59:59"},
                    "row_count": 12,
                    "metrics": {"RMSE": 2.0, "MAE": 1.0},
                },
                {
                    "name": "period_02",
                    "period": {"start": "2025-02-01T00:00:00", "end": "2025-02-28T23:59:59"},
                    "row_count": 8,
                    "metrics": {"RMSE": 1.0, "MAE": 0.5},
                },
            ],
        }
    ]

    comparison = cli._build_eval_comparison(results)

    assert comparison is not None
    assert comparison["ranked_by_rmse"] is True
    rows = comparison["rows"]
    assert len(rows) == 3
    period_rows = [row for row in rows if row["validation_period_name"] in {"period_01", "period_02"}]
    overall_rows = [row for row in rows if row["validation_period_name"] == "overall"]
    assert len(period_rows) == 2
    assert len(overall_rows) == 1
    assert period_rows[0]["validation_period"] == "2025-01-01T00:00:00 -> 2025-01-31T23:59:59"
    assert period_rows[1]["validation_period_start"] == "2025-02-01T00:00:00"
    assert period_rows[1]["validation_period_end"] == "2025-02-28T23:59:59"
    assert period_rows[0]["row_count"] == 12
    assert overall_rows[0]["RMSE"] == 99.0
    assert rows[0]["rank_by_rmse"] >= 1


def test_build_eval_comparison_includes_overall_baseline_rows_with_period_results():
    results = [
        {
            "config": "config_a.yaml",
            "run_dir": "artifacts/run_a",
            "algorithm_name": "extra_trees",
            "target_station_id": "124",
            "metrics": {"RMSE": 2.0, "MAE": 1.0},
            "period_results": [
                {
                    "name": "period_01",
                    "period": {"start": "2025-01-01T00:00:00", "end": "2025-01-31T23:59:59"},
                    "metrics": {"RMSE": 2.2, "MAE": 1.1},
                }
            ],
            "baseline_results": [
                {
                    "id": "baseline_mean_of_other_stations",
                    "algorithm_name": "Baseline: Average of Other Stations",
                    "metrics": {"RMSE": 2.8, "MAE": 1.5},
                }
            ],
        }
    ]

    comparison = cli._build_eval_comparison(results)

    assert comparison is not None
    rows = comparison["rows"]
    overall_rows = [row for row in rows if row.get("validation_period_name") == "overall"]
    assert len(overall_rows) == 2
    assert any(row["algorithm"] == "extra_trees" and row["RMSE"] == 2.0 for row in overall_rows)
    assert any(
        row["algorithm"] == "Baseline: Average of Other Stations" and row["RMSE"] == 2.8
        for row in overall_rows
    )


def test_build_eval_comparison_groups_rows_by_station_and_period_then_algorithm():
    period = {"start": "2025-01-01T00:00:00", "end": "2025-01-31T23:59:59"}
    results = [
        {
            "config": "delhi_lstm_124.yaml",
            "run_dir": "artifacts/run_lstm_124",
            "target_station_id": "124",
            "algorithm": "src.lstm_regressor.LocalLSTMRegressor",
            "period_results": [
                {
                    "name": "period_01",
                    "period": period,
                    "row_count": 12,
                    "metrics": {"RMSE": 2.0, "MAE": 1.0},
                }
            ],
        },
        {
            "config": "delhi_extra_trees_124.yaml",
            "run_dir": "artifacts/run_et_124",
            "target_station_id": "124",
            "algorithm": "extra_trees",
            "period_results": [
                {
                    "name": "period_01",
                    "period": period,
                    "row_count": 12,
                    "metrics": {"RMSE": 1.5, "MAE": 0.8},
                }
            ],
        },
        {
            "config": "delhi_extra_trees_1421.yaml",
            "run_dir": "artifacts/run_et_1421",
            "target_station_id": "1421",
            "algorithm": "extra_trees",
            "period_results": [
                {
                    "name": "period_01",
                    "period": period,
                    "row_count": 10,
                    "metrics": {"RMSE": 1.2, "MAE": 0.7},
                }
            ],
        },
    ]

    comparison = cli._build_eval_comparison(results)

    assert comparison is not None
    rows = comparison["rows"]
    assert [
        (row["target_station_id"], row["validation_period_name"], row["algorithm"])
        for row in rows
    ] == [
        ("124", "period_01", "extra_trees"),
        ("124", "period_01", "src.lstm_regressor.LocalLSTMRegressor"),
        ("1421", "period_01", "extra_trees"),
    ]


def test_expand_config_paths_supports_config_dirs_and_excludes_base(tmp_path):
    cfg_dir = tmp_path / "configs"
    cfg_dir.mkdir()
    (cfg_dir / "delhi_base.yaml").write_text("model: {}\n", encoding="utf-8")
    rf_cfg = cfg_dir / "delhi_random_forest.yaml"
    et_cfg = cfg_dir / "delhi_extra_trees.yaml"
    rf_cfg.write_text("model: {}\n", encoding="utf-8")
    et_cfg.write_text("model: {}\n", encoding="utf-8")

    expanded = cli._expand_config_paths(None, [str(cfg_dir)])

    assert expanded == [str(et_cfg), str(rf_cfg)]


def test_expand_config_paths_requires_non_base_yaml_in_directory(tmp_path):
    cfg_dir = tmp_path / "configs"
    cfg_dir.mkdir()
    (cfg_dir / "base.yaml").write_text("model: {}\n", encoding="utf-8")
    (cfg_dir / "city_base.yml").write_text("model: {}\n", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="No non-base YAML config files"):
        cli._expand_config_paths(None, [str(cfg_dir)])


def test_expand_config_paths_accepts_mixed_config_and_config_dir(tmp_path):
    cfg_dir = tmp_path / "configs"
    cfg_dir.mkdir()
    single_cfg = tmp_path / "single.yaml"
    dir_cfg = cfg_dir / "exp.yaml"
    single_cfg.write_text("model: {}\n", encoding="utf-8")
    dir_cfg.write_text("model: {}\n", encoding="utf-8")

    expanded = cli._expand_config_paths([str(single_cfg)], [str(cfg_dir)])

    assert expanded == [str(single_cfg), str(dir_cfg)]


def test_expand_config_paths_recurses_and_skips_excluded_dirs(tmp_path):
    cfg_dir = tmp_path / "configs"
    nested_dir = cfg_dir / "bilstm" / "combined"
    excluded_dir = cfg_dir / "unusedConfigs"
    nested_dir.mkdir(parents=True)
    excluded_dir.mkdir(parents=True)

    nested_cfg = nested_dir / "delhi_bilstm_124.yaml"
    ignored_cfg = excluded_dir / "old.yaml"
    nested_cfg.write_text("model: {}\n", encoding="utf-8")
    ignored_cfg.write_text("model: {}\n", encoding="utf-8")

    expanded = cli._expand_config_paths(None, [str(cfg_dir)])

    assert expanded == [str(nested_cfg)]


def test_write_eval_comparison_csv_persists_rows(tmp_path):
    comparison = {
        "rows": [
            {
                "config": "a.yaml",
                "run_dir": "artifacts/run_a",
                "RMSE": 1.0,
                "MAE": 0.5,
            },
            {
                "config": "b.yaml",
                "run_dir": "artifacts/run_b",
                "RMSE": 2.0,
                "MAE": 1.0,
            },
        ]
    }

    csv_path = tmp_path / "comparison.csv"
    written = cli._write_eval_comparison_csv(comparison, str(csv_path))

    assert Path(written).exists()
    frame = pd.read_csv(written)
    assert list(frame["config"]) == ["a.yaml", "b.yaml"]
    assert list(frame["RMSE"]) == [1.0, 2.0]


def test_write_eval_comparison_csv_defaults_to_config_dir_outputs(tmp_path):
    cfg_dir = tmp_path / "configs" / "delhi"
    cfg_dir.mkdir(parents=True)
    comparison = {
        "rows": [
            {
                "config": "a.yaml",
                "run_dir": "artifacts/run_a",
                "RMSE": 1.0,
            }
        ]
    }

    written = cli._write_eval_comparison_csv(comparison, None, config_dirs=[str(cfg_dir)])

    written_path = Path(written)
    assert written_path.exists()
    assert written_path.parent == (cfg_dir / "outputs")


def test_write_station_period_metric_charts_writes_one_image_per_station(tmp_path):
    comparison = {
        "rows": [
            {
                "config": "a.yaml",
                "run_dir": "artifacts/run_a",
                "target_station_id": "124",
                "algorithm": "extra_trees",
                "validation_period_name": "period_01",
                "validation_period_start": "2025-01-01T00:00:00",
                "validation_period_end": "2025-01-31T23:59:59",
                "RMSE": 1.2,
                "MAE": 0.9,
            },
            {
                "config": "b.yaml",
                "run_dir": "artifacts/run_b",
                "target_station_id": "124",
                "algorithm": "src.lstm_regressor.LocalLSTMRegressor",
                "validation_period_name": "period_01",
                "validation_period_start": "2025-01-01T00:00:00",
                "validation_period_end": "2025-01-31T23:59:59",
                "RMSE": 1.5,
                "MAE": 1.0,
            },
            {
                "config": "c.yaml",
                "run_dir": "artifacts/run_c",
                "target_station_id": "124",
                "algorithm": "extra_trees",
                "validation_period_name": "period_02",
                "validation_period_start": "2025-02-01T00:00:00",
                "validation_period_end": "2025-02-28T23:59:59",
                "RMSE": 1.1,
                "MAE": 0.8,
            },
            {
                "config": "d.yaml",
                "run_dir": "artifacts/run_d",
                "target_station_id": "1421",
                "algorithm": "extra_trees",
                "validation_period_name": "period_01",
                "validation_period_start": "2025-01-01T00:00:00",
                "validation_period_end": "2025-01-31T23:59:59",
                "RMSE": 0.9,
                "MAE": 0.7,
            },
        ]
    }

    written = cli._write_station_period_metric_charts(comparison, tmp_path)

    assert len(written) == 2
    assert all(Path(path).exists() for path in written)
    assert any(
        Path(path).name == "metrics_overall_124.png"
        and Path(path).parent.name == "pollutant_overall"
        for path in written
    )
    assert any(
        Path(path).name == "metrics_overall_1421.png"
        and Path(path).parent.name == "pollutant_overall"
        for path in written
    )


def test_write_station_period_metric_charts_writes_two_pollutant_files_per_station(tmp_path):
    comparison = {
        "rows": [
            {
                "config": "cfg_a.yaml",
                "run_dir": "run_a",
                "target_station_id": "124",
                "algorithm": "extra_trees",
                "validation_period_name": "period_01",
                "validation_period_start": "2025-01-01T00:00:00",
                "validation_period_end": "2025-01-31T23:59:59",
                "by_target.target_value__PM2.5 (µg/m³).RMSE": 10.0,
                "by_target.target_value__PM2.5 (µg/m³).MAE": 7.0,
                "by_target.target_value__PM2.5 (µg/m³).MSE": 100.0,
                "by_target.target_value__PM2.5 (µg/m³).IA": 0.90,
                "by_target.target_value__PM2.5 (µg/m³).R2": 0.81,
                "by_target.target_value__NO2 (µg/m³).RMSE": 9.0,
                "by_target.target_value__NO2 (µg/m³).MAE": 6.5,
                "by_target.target_value__NO2 (µg/m³).MSE": 81.0,
                "by_target.target_value__NO2 (µg/m³).IA": 0.91,
                "by_target.target_value__NO2 (µg/m³).R2": 0.82,
                "RMSE": 11.0,
                "MAE": 7.5,
            },
            {
                "config": "cfg_a.yaml",
                "run_dir": "run_a",
                "target_station_id": "124",
                "algorithm": "extra_trees",
                "validation_period_name": "period_02",
                "validation_period_start": "2025-04-01T00:00:00",
                "validation_period_end": "2025-04-30T23:59:59",
                "by_target.target_value__PM2.5 (µg/m³).RMSE": 12.0,
                "by_target.target_value__PM2.5 (µg/m³).MAE": 8.0,
                "by_target.target_value__PM2.5 (µg/m³).MSE": 144.0,
                "by_target.target_value__PM2.5 (µg/m³).IA": 0.88,
                "by_target.target_value__PM2.5 (µg/m³).R2": 0.79,
                "by_target.target_value__NO2 (µg/m³).RMSE": 10.0,
                "by_target.target_value__NO2 (µg/m³).MAE": 7.1,
                "by_target.target_value__NO2 (µg/m³).MSE": 100.0,
                "by_target.target_value__NO2 (µg/m³).IA": 0.89,
                "by_target.target_value__NO2 (µg/m³).R2": 0.80,
                "RMSE": 13.0,
                "MAE": 8.2,
            },
        ]
    }

    written = cli._write_station_period_metric_charts(comparison, tmp_path / "charts")

    assert len(written) == 2
    assert all(Path(path).exists() for path in written)
    assert any(Path(path).parent.name == "pollutant_PM2_5" for path in written)
    assert any(Path(path).parent.name == "pollutant_NO2" for path in written)
    assert all("pollutant_overall" not in Path(path).parent.name for path in written)


def test_collect_prediction_plot_series_reads_single_and_multi_target_validation(tmp_path):
    run_single = tmp_path / "run_single"
    run_single_validation = run_single / "validation"
    run_single_validation.mkdir(parents=True)
    (run_single / "run_metadata.json").write_text(
        json.dumps({"algorithm": "random_forest", "target_station_id": "124"}),
        encoding="utf-8",
    )
    pd.DataFrame(
        {
            "prediction_time": ["2025-01-01 00:00:00", "2025-01-01 01:00:00"],
            "actual": [10.0, 11.0],
            "prediction": [9.5, 10.8],
        }
    ).to_csv(run_single_validation / "validation.csv", index=False)

    run_multi = tmp_path / "run_multi"
    run_multi_validation = run_multi / "validation"
    run_multi_validation.mkdir(parents=True)
    (run_multi / "run_metadata.json").write_text(
        json.dumps(
            {
                "algorithm": "src.lstm_regressor.LocalLSTMRegressor",
                "target_station_id": "1421",
                "config_path": "configs/delhi/subconfigs/bilstm/pm25/delhi_bilstm_pm25_1421.yaml",
                "model_params": {"bidirectional": True},
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        {
            "prediction_time": ["2025-01-01 00:00:00", "2025-01-01 01:00:00"],
            "actual__PM2.5": [15.0, 16.0],
            "prediction__PM2.5": [14.8, 16.1],
            "actual__NO2": [7.0, 8.0],
            "prediction__NO2": [6.8, 8.1],
        }
    ).to_csv(run_multi_validation / "validation.csv", index=False)

    series = cli._collect_prediction_plot_series([run_single, run_multi])

    keys = sorted((item["algorithm"], item["station_id"], item["target"]) for item in series)
    assert keys == [
        ("BiLSTM (PM2.5)", "1421", "NO2"),
        ("BiLSTM (PM2.5)", "1421", "PM2.5"),
        ("random_forest", "124", "target"),
    ]


def test_collect_prediction_plot_series_includes_baseline_prediction_artifacts(tmp_path):
    run_dir = tmp_path / "run_a"
    validation_dir = run_dir / "validation"
    validation_dir.mkdir(parents=True)
    (run_dir / "run_metadata.json").write_text(
        json.dumps({"algorithm": "extra_trees", "target_station_id": "124"}),
        encoding="utf-8",
    )

    pd.DataFrame(
        {
            "prediction_time": ["2025-01-01 00:00:00", "2025-01-01 01:00:00"],
            "actual": [1.0, 2.0],
            "prediction": [1.1, 1.9],
        }
    ).to_csv(validation_dir / "validation.csv", index=False)

    (validation_dir / "baseline_results.json").write_text(
        json.dumps(
            [
                {
                    "id": "baseline_mean_of_other_stations",
                    "algorithm_name": "Baseline: Average of Other Stations",
                    "metrics": {"RMSE": 0.5},
                },
                {
                    "id": "baseline_replace_with_station_1421",
                    "algorithm_name": "Baseline Replace with Station 1421",
                    "metrics": {"RMSE": 0.7},
                },
            ]
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        {
            "prediction_time": ["2025-01-01 00:00:00", "2025-01-01 01:00:00"],
            "actual": [1.0, 2.0],
            "prediction__baseline_mean_of_other_stations": [1.05, 1.95],
            "prediction__baseline_replace_with_station_1421": [1.2, 1.8],
        }
    ).to_csv(validation_dir / "baseline_predictions.csv", index=False)

    series = cli._collect_prediction_plot_series([run_dir])

    keys = sorted((item["algorithm"], item["station_id"], item["target"]) for item in series)
    assert keys == [
        ("Baseline: Average of Other Stations", "124", "target"),
        ("Baseline Replace with Station 1421", "124", "target"),
        ("extra_trees", "124", "target"),
    ]


def test_collect_prediction_plot_series_infers_single_target_pollutant_label_from_config_path(tmp_path):
    run_dir = tmp_path / "run_pm25"
    validation_dir = run_dir / "validation"
    validation_dir.mkdir(parents=True)
    (run_dir / "run_metadata.json").write_text(
        json.dumps(
            {
                "algorithm": "extra_trees",
                "target_station_id": "124",
                "config_path": "configs/delhi/subconfigs/extra_trees/pm25/delhi_extra_trees_pm25_124.yaml",
            }
        ),
        encoding="utf-8",
    )

    pd.DataFrame(
        {
            "prediction_time": ["2025-01-01 00:00:00", "2025-01-01 01:00:00"],
            "actual": [1.0, 2.0],
            "prediction": [1.1, 1.9],
        }
    ).to_csv(validation_dir / "validation.csv", index=False)

    series = cli._collect_prediction_plot_series([run_dir])

    assert len(series) == 1
    assert series[0]["target"] == "PM2.5"


def test_write_interactive_prediction_plot_writes_html(tmp_path):
    pytest.importorskip("plotly")

    run_dir = tmp_path / "run_a"
    validation_dir = run_dir / "validation"
    validation_dir.mkdir(parents=True)
    (run_dir / "run_metadata.json").write_text(
        json.dumps({"algorithm": "extra_trees", "target_station_id": "124"}),
        encoding="utf-8",
    )
    pd.DataFrame(
        {
            "prediction_time": ["2025-01-01 00:00:00", "2025-01-01 01:00:00"],
            "actual": [1.0, 2.0],
            "prediction": [1.1, 1.9],
        }
    ).to_csv(validation_dir / "validation.csv", index=False)

    output_html = tmp_path / "plots" / "prediction_vs_actual.html"
    payload = cli._write_interactive_prediction_plot([run_dir], output_html)

    assert output_html.exists()
    assert payload["path"] == str(output_html.resolve())
    assert payload["run_count"] == 1
    assert payload["series_count"] == 1
    assert payload["algorithms"] == ["extra_trees"]

    content = output_html.read_text(encoding="utf-8")
    assert "All Algorithms" in content
    assert "Prediction vs Actual" in content


def test_default_plot_output_dir_uses_config_parent(tmp_path):
    config_path = tmp_path / "configs" / "example" / "example.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text("data: {}\n", encoding="utf-8")

    output_dir = cli._default_plot_output_dir([str(config_path)], None)

    assert output_dir == (config_path.parent / "outputs" / "plots")


def test_timestamped_output_dir_creates_unique_subfolders(tmp_path):
    base_output_dir = tmp_path / "outputs" / "plots"

    first = cli._timestamped_output_dir(base_output_dir)
    second = cli._timestamped_output_dir(base_output_dir)

    assert first.exists()
    assert second.exists()
    assert first != second
    assert first.parent == base_output_dir
    assert second.parent == base_output_dir


def test_timestamped_output_dir_includes_optional_comment_label(tmp_path):
    base_output_dir = tmp_path / "outputs" / "plots"

    output_dir = cli._timestamped_output_dir(base_output_dir, label="ablation 1")

    assert output_dir.exists()
    assert output_dir.name.startswith("ablation_1_")


def test_comment_scoped_plot_base_dir_uses_short_comment_folder(tmp_path):
    base_output_dir = tmp_path / "outputs" / "plots"

    scoped = cli._comment_scoped_plot_base_dir(base_output_dir, "campaign 1")

    assert scoped == (base_output_dir / "campaign_1")
    assert scoped.exists()


def test_comment_scoped_plot_base_dir_returns_base_without_comment(tmp_path):
    base_output_dir = tmp_path / "outputs" / "plots"

    scoped = cli._comment_scoped_plot_base_dir(base_output_dir, None)

    assert scoped == base_output_dir


def test_resolve_plot_output_comment_prefers_explicit_over_metadata(tmp_path):
    run_dir = tmp_path / "run_a"
    run_dir.mkdir(parents=True)
    (run_dir / "run_metadata.json").write_text(
        json.dumps({"run_comment": "from-metadata"}),
        encoding="utf-8",
    )

    resolved = cli._resolve_plot_output_comment([run_dir], "explicit")

    assert resolved == "explicit"


def test_resolve_plot_output_comment_uses_single_discovered_metadata_value(tmp_path):
    run_a = tmp_path / "run_a"
    run_b = tmp_path / "run_b"
    run_a.mkdir(parents=True)
    run_b.mkdir(parents=True)
    (run_a / "run_metadata.json").write_text(json.dumps({"run_comment": "campaign_1"}), encoding="utf-8")
    (run_b / "run_metadata.json").write_text(json.dumps({"run_comment": "campaign_1"}), encoding="utf-8")

    resolved = cli._resolve_plot_output_comment([run_a, run_b], None)

    assert resolved == "campaign_1"


def test_resolve_plot_output_comment_returns_none_when_multiple_values_found(tmp_path):
    run_a = tmp_path / "run_a"
    run_b = tmp_path / "run_b"
    run_a.mkdir(parents=True)
    run_b.mkdir(parents=True)
    (run_a / "run_metadata.json").write_text(json.dumps({"run_comment": "campaign_1"}), encoding="utf-8")
    (run_b / "run_metadata.json").write_text(json.dumps({"run_comment": "campaign_2"}), encoding="utf-8")

    resolved = cli._resolve_plot_output_comment([run_a, run_b], None)

    assert resolved is None


def test_write_static_prediction_plots_exports_pngs(tmp_path):
    run_dir = tmp_path / "run_a"
    validation_dir = run_dir / "validation"
    validation_dir.mkdir(parents=True)
    (run_dir / "run_metadata.json").write_text(
        json.dumps({"algorithm": "extra_trees", "target_station_id": "124"}),
        encoding="utf-8",
    )
    pd.DataFrame(
        {
            "prediction_time": ["2025-01-01 00:00:00", "2025-01-01 01:00:00"],
            "actual": [1.0, 2.0],
            "prediction": [1.1, 1.9],
        }
    ).to_csv(validation_dir / "validation.csv", index=False)

    output_dir = tmp_path / "outputs" / "plots"
    written = cli._write_static_prediction_plots([run_dir], output_dir)

    assert len(written) == 1
    assert Path(written[0]).exists()
    csv_files = sorted(output_dir.rglob("*.csv"))
    assert len(csv_files) == 1


def test_resolve_plot_title_templates_reads_config(tmp_path):
    cfg_path = tmp_path / "plot_cfg.yaml"
    cfg_path.write_text(
        """
plot:
  title_templates:
    prediction_static: "Pred {target} @ {station_id}"
    metrics_by_period: "Period chart {pollutant}"
""".strip(),
        encoding="utf-8",
    )

    templates = cli._resolve_plot_title_templates([str(cfg_path)])

    assert templates["prediction_static"] == "Pred {target} @ {station_id}"
    assert templates["metrics_by_period"] == "Period chart {pollutant}"
    assert "scatter_static" in templates


def test_write_static_prediction_plots_uses_title_template(tmp_path, monkeypatch):
    run_dir = tmp_path / "run_a"
    validation_dir = run_dir / "validation"
    validation_dir.mkdir(parents=True)
    (run_dir / "run_metadata.json").write_text(
        json.dumps({"algorithm": "extra_trees", "target_station_id": "124"}),
        encoding="utf-8",
    )
    pd.DataFrame(
        {
            "prediction_time": ["2025-01-01 00:00:00", "2025-01-01 01:00:00"],
            "actual": [1.0, 2.0],
            "prediction": [1.1, 1.9],
        }
    ).to_csv(validation_dir / "validation.csv", index=False)

    captured: dict[str, str] = {}

    def _fake_save_prediction_sample_plot(df, datetime_col, series_by_target, path, title):
        captured["title"] = title
        Path(path).write_text("png", encoding="utf-8")
        return True

    monkeypatch.setattr(cli, "save_prediction_sample_plot", _fake_save_prediction_sample_plot)

    cli._write_static_prediction_plots(
        [run_dir],
        tmp_path / "plots",
        title_templates={"prediction_static": "Pred {target} with {algorithm} at {station_id}"},
    )

    assert captured["title"] == "Pred target with extra_trees at 124"


def test_write_static_actual_prediction_scatter_plots_exports_pngs(tmp_path, monkeypatch):
    run_dir = tmp_path / "run_a"
    validation_dir = run_dir / "validation"
    train_dump_dir = run_dir / "training_dump"
    validation_dir.mkdir(parents=True)
    train_dump_dir.mkdir(parents=True)

    (run_dir / "run_metadata.json").write_text(
        json.dumps({"algorithm": "extra_trees", "target_station_id": "124"}),
        encoding="utf-8",
    )
    pd.DataFrame(
        {
            "prediction_time": ["2025-01-01 00:00:00", "2025-01-01 01:00:00"],
            "actual": [1.0, 2.0],
            "prediction": [1.1, 1.9],
        }
    ).to_csv(validation_dir / "validation.csv", index=False)

    pd.DataFrame(
        {
            "prediction_time": ["2024-01-01 00:00:00", "2024-01-01 01:00:00"],
            "actual": [0.9, 1.8],
            "prediction": [1.0, 1.7],
        }
    ).to_csv(validation_dir / "training_predictions.csv", index=False)

    pd.DataFrame({"f1": [1.0, 2.0], "f2": [2.0, 4.0]}).to_csv(
        train_dump_dir / "training_features_model_input.csv", index=False
    )
    pd.DataFrame({"target_value": [1.0, 2.0]}).to_csv(
        train_dump_dir / "training_target.csv", index=False
    )

    class DummyPipeline:
        feature_names_ = ["f1", "f2"]
        target_names_ = ["target"]

        def predict(self, X):
            return X["f1"].to_numpy()

    monkeypatch.setattr(cli, "_load_chunked_joblib", lambda path: DummyPipeline())

    output_dir = tmp_path / "outputs" / "plots"
    written = cli._write_static_actual_prediction_scatter_plots([run_dir], output_dir)

    assert len(written) == 1
    assert Path(written[0]).exists()
    csv_files = sorted(output_dir.rglob("*.csv"))
    assert len(csv_files) == 1
    frame = pd.read_csv(csv_files[0])
    assert set(frame["split"].tolist()) == {"evaluation", "training"}


def test_write_static_actual_prediction_scatter_plots_keeps_all_validation_rows(tmp_path):
    run_dir = tmp_path / "run_a"
    validation_dir = run_dir / "validation"
    validation_dir.mkdir(parents=True)

    (run_dir / "run_metadata.json").write_text(
        json.dumps({"algorithm": "extra_trees", "target_station_id": "124"}),
        encoding="utf-8",
    )
    (run_dir / "config_snapshot.yaml").write_text(
        """
data:
  validation_periods:
    - start: "2025"
      end: "2025"
    - start: "2024-01"
      end: "2024-01"
""".strip(),
        encoding="utf-8",
    )

    pd.DataFrame(
        {
            "prediction_time": [
                "2024-01-05 00:00:00",
                "2025-01-05 00:00:00",
                "2025-06-05 00:00:00",
            ],
            "actual": [9.0, 1.0, 2.0],
            "prediction": [8.8, 1.1, 1.9],
        }
    ).to_csv(validation_dir / "validation.csv", index=False)

    pd.DataFrame(
        {
            "prediction_time": ["2024-01-01 00:00:00", "2024-01-01 01:00:00"],
            "actual": [0.9, 1.8],
            "prediction": [1.0, 1.7],
        }
    ).to_csv(validation_dir / "training_predictions.csv", index=False)

    output_dir = tmp_path / "outputs" / "plots"
    written = cli._write_static_actual_prediction_scatter_plots([run_dir], output_dir)

    assert len(written) == 1
    csv_files = sorted(output_dir.rglob("*.csv"))
    assert len(csv_files) == 1
    frame = pd.read_csv(csv_files[0])
    evaluation_rows = frame.loc[frame["split"] == "evaluation"]
    assert len(evaluation_rows) == 3
    assert set(evaluation_rows["actual"].tolist()) == {9.0, 1.0, 2.0}


def test_write_static_actual_prediction_scatter_plots_groups_by_pollutant_then_station(tmp_path, monkeypatch):
    import matplotlib.figure as mpl_figure

    def _write_run(run_name: str, station_id: str, config_path: str) -> Path:
        run_dir = tmp_path / run_name
        validation_dir = run_dir / "validation"
        validation_dir.mkdir(parents=True)
        (run_dir / "run_metadata.json").write_text(
            json.dumps(
                {
                    "algorithm": "extra_trees",
                    "target_station_id": station_id,
                    "config_path": config_path,
                }
            ),
            encoding="utf-8",
        )
        pd.DataFrame(
            {
                "prediction_time": ["2025-01-01 00:00:00", "2025-01-01 01:00:00"],
                "actual": [1.0, 2.0],
                "prediction": [1.1, 1.9],
            }
        ).to_csv(validation_dir / "validation.csv", index=False)
        return run_dir

    no2_run = _write_run("run_no2", "124", "configs/delhi/subconfigs/extra_trees/no2/cfg.yaml")
    pm25_run = _write_run("run_pm25", "1421", "configs/delhi/subconfigs/extra_trees/pm25/cfg.yaml")
    pd.DataFrame(
        {
            "actual": [1.0, 2.0],
            "prediction__baseline_mean_of_other_stations": [1.2, 2.1],
        }
    ).to_csv(no2_run / "validation" / "baseline_predictions.csv", index=False)

    original_savefig = mpl_figure.Figure.savefig

    def _capture_savefig(self, fname, *args, **kwargs):
        Path(fname).parent.mkdir(parents=True, exist_ok=True)
        Path(fname).write_text("png", encoding="utf-8")

    monkeypatch.setattr(mpl_figure.Figure, "savefig", _capture_savefig)

    output_dir = tmp_path / "outputs" / "plots"
    written = cli._write_static_actual_prediction_scatter_plots([no2_run, pm25_run], output_dir)

    assert len(written) == 3
    assert (output_dir / "pollutant_NO2" / "station_124").exists()
    assert (output_dir / "pollutant_PM2_5" / "station_1421").exists()
    assert list((output_dir / "pollutant_NO2" / "station_124").glob("scatter*.png"))
    assert list((output_dir / "pollutant_PM2_5" / "station_1421").glob("scatter*.png"))
    assert not (output_dir / "station_124").exists()

    monkeypatch.setattr(mpl_figure.Figure, "savefig", original_savefig)


def test_write_static_scatter_plots_use_title_templates(tmp_path, monkeypatch):
    import matplotlib.figure as mpl_figure

    run_dir = tmp_path / "run_a"
    validation_dir = run_dir / "validation"
    validation_dir.mkdir(parents=True)

    (run_dir / "run_metadata.json").write_text(
        json.dumps({"algorithm": "extra_trees", "target_station_id": "124"}),
        encoding="utf-8",
    )
    pd.DataFrame(
        {
            "prediction_time": ["2025-01-01 00:00:00", "2025-01-01 01:00:00"],
            "actual": [1.0, 2.0],
            "prediction": [1.1, 1.9],
        }
    ).to_csv(validation_dir / "validation.csv", index=False)
    pd.DataFrame(
        {
            "prediction_time": ["2024-01-01 00:00:00", "2024-01-01 01:00:00"],
            "actual": [0.9, 1.8],
            "prediction": [1.0, 1.7],
        }
    ).to_csv(validation_dir / "training_predictions.csv", index=False)

    captured_titles: list[str] = []
    original_savefig = mpl_figure.Figure.savefig

    def _capture_savefig(self, fname, *args, **kwargs):
        captured_titles.append(self._suptitle.get_text() if self._suptitle is not None else "")
        Path(fname).parent.mkdir(parents=True, exist_ok=True)
        Path(fname).write_text("png", encoding="utf-8")

    monkeypatch.setattr(mpl_figure.Figure, "savefig", _capture_savefig)

    cli._write_static_actual_prediction_scatter_plots(
        [run_dir],
        tmp_path / "plots",
        title_templates={"scatter_static": "Scatter {target} :: {algorithm} :: {station_id}"},
    )

    assert any(title == "Scatter target :: extra_trees :: 124" for title in captured_titles)
    monkeypatch.setattr(mpl_figure.Figure, "savefig", original_savefig)


def test_build_eval_results_from_run_dirs_reads_period_metrics(tmp_path):
    run_dir = tmp_path / "run_a"
    period_dir = run_dir / "validation" / "by_period"
    period_dir.mkdir(parents=True)

    (run_dir / "run_metadata.json").write_text(
        json.dumps(
            {
                "config_path": "configs/delhi/subconfigs/extra_trees/combined/delhi_extra_trees_124.yaml",
                "target_station_id": "124",
                "algorithm": "extra_trees",
            }
        ),
        encoding="utf-8",
    )

    pd.DataFrame(
        {
            "validation_period_name": ["period_01", "period_02"],
            "validation_period_start": ["2025-01-01T00:00:00", "2025-04-01T00:00:00"],
            "validation_period_end": ["2025-01-31T23:59:59", "2025-04-30T23:59:59"],
            "row_count": [100, 120],
            "RMSE": [10.0, 12.0],
            "MAE": [7.0, 8.0],
            "MSE": [100.0, 144.0],
            "IA": [0.90, 0.88],
            "R2": [0.81, 0.79],
        }
    ).to_csv(period_dir / "metrics_summary.csv", index=False)

    results = cli._build_eval_results_from_run_dirs([run_dir])

    assert len(results) == 1
    result = results[0]
    assert result["target_station_id"] == "124"
    assert result["algorithm_name"] == "extra_trees"
    assert "period_results" in result
    assert len(result["period_results"]) == 2


def test_build_eval_results_from_run_dirs_maps_single_target_metrics_to_pollutant(tmp_path):
    run_dir = tmp_path / "run_bilstm_pm25"
    period_dir = run_dir / "validation" / "by_period"
    period_dir.mkdir(parents=True)

    (run_dir / "run_metadata.json").write_text(
        json.dumps(
            {
                "config_path": "configs/delhi/subconfigs/bilstm/pm25/delhi_bilstm_pm25_124.yaml",
                "algorithm": "src.lstm_regressor.LocalLSTMRegressor",
                "algorithm_name": "BiLSTM (PM2.5)",
                "target_station_id": "124",
            }
        ),
        encoding="utf-8",
    )

    pd.DataFrame(
        {
            "validation_period_name": ["period_01"],
            "validation_period_start": ["2025-01-01T00:00:00"],
            "validation_period_end": ["2025-01-31T23:59:59"],
            "row_count": [100],
            "RMSE": [10.0],
            "MAE": [7.0],
            "MSE": [100.0],
            "IA": [0.90],
            "R2": [0.81],
        }
    ).to_csv(period_dir / "metrics_summary.csv", index=False)

    results = cli._build_eval_results_from_run_dirs([run_dir])

    assert len(results) == 1
    period_result = results[0]["period_results"][0]
    metrics = period_result["metrics"]
    assert metrics["by_target.target_value__PM2.5 (µg/m³).RMSE"] == 10.0
    assert metrics["by_target.target_value__PM2.5 (µg/m³).MAE"] == 7.0


def test_build_eval_results_from_run_dirs_reads_persisted_baseline_results(tmp_path):
    run_dir = tmp_path / "run_a"
    validation_dir = run_dir / "validation"
    validation_dir.mkdir(parents=True)

    (run_dir / "run_metadata.json").write_text(
        json.dumps(
            {
                "config_path": "configs/delhi/subconfigs/bilstm/pm25/delhi_bilstm_pm25_124.yaml",
                "algorithm": "src.lstm_regressor.LocalLSTMRegressor",
                "algorithm_name": "BiLSTM (PM2.5)",
                "target_station_id": "124",
            }
        ),
        encoding="utf-8",
    )

    (validation_dir / "metrics.json").write_text(
        json.dumps({"RMSE": 10.0, "MAE": 7.0, "MSE": 100.0, "IA": 0.90, "R2": 0.81}),
        encoding="utf-8",
    )
    (validation_dir / "baseline_results.json").write_text(
        json.dumps(
            [
                {
                    "id": "baseline_mean_of_other_stations",
                    "name": "Baseline: Average of Other Stations",
                    "algorithm_name": "Baseline: Average of Other Stations",
                    "metrics": {"RMSE": 12.0, "MAE": 9.0, "MSE": 144.0, "IA": 0.85, "R2": 0.70},
                }
            ]
        ),
        encoding="utf-8",
    )

    results = cli._build_eval_results_from_run_dirs([run_dir])

    assert len(results) == 1
    baseline_results = results[0].get("baseline_results")
    assert isinstance(baseline_results, list)
    assert baseline_results[0]["algorithm_name"] == "Baseline: Average of Other Stations"
    baseline_metrics = baseline_results[0]["metrics"]
    assert baseline_metrics["by_target.target_value__PM2.5 (µg/m³).RMSE"] == 12.0


def test_build_eval_results_from_run_dirs_reads_persisted_baseline_period_results(tmp_path):
    run_dir = tmp_path / "run_a"
    period_dir = run_dir / "validation" / "by_period"
    period_dir.mkdir(parents=True)

    (run_dir / "run_metadata.json").write_text(
        json.dumps(
            {
                "config_path": "configs/delhi/subconfigs/bilstm/pm25/delhi_bilstm_pm25_124.yaml",
                "algorithm": "src.lstm_regressor.LocalLSTMRegressor",
                "algorithm_name": "BiLSTM (PM2.5)",
                "target_station_id": "124",
            }
        ),
        encoding="utf-8",
    )

    pd.DataFrame(
        {
            "validation_period_name": ["period_01"],
            "validation_period_start": ["2025-01-01T00:00:00"],
            "validation_period_end": ["2025-01-31T23:59:59"],
            "row_count": [100],
            "RMSE": [10.0],
            "MAE": [7.0],
            "MSE": [100.0],
            "IA": [0.90],
            "R2": [0.81],
        }
    ).to_csv(period_dir / "metrics_summary.csv", index=False)

    (period_dir / "baseline_period_results.json").write_text(
        json.dumps(
            [
                {
                    "id": "baseline_mean_of_other_stations",
                    "name": "Baseline: Average of Other Stations",
                    "algorithm_name": "Baseline: Average of Other Stations",
                    "period_results": [
                        {
                            "name": "period_01",
                            "period": {
                                "start": "2025-01-01T00:00:00",
                                "end": "2025-01-31T23:59:59",
                            },
                            "row_count": 100,
                            "metrics": {"RMSE": 12.0, "MAE": 9.0, "MSE": 144.0, "IA": 0.85, "R2": 0.70},
                        }
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )

    results = cli._build_eval_results_from_run_dirs([run_dir])

    assert len(results) == 1
    baseline_period_results = results[0].get("baseline_period_results")
    assert isinstance(baseline_period_results, list)
    assert baseline_period_results[0]["algorithm_name"] == "Baseline: Average of Other Stations"
    period_metrics = baseline_period_results[0]["period_results"][0]["metrics"]
    assert period_metrics["by_target.target_value__PM2.5 (µg/m³).RMSE"] == 12.0


def test_write_station_period_metric_charts_writes_requested_metrics_only(tmp_path):
    comparison = {
        "rows": [
            {
                "config": "cfg_a.yaml",
                "run_dir": "run_a",
                "target_station_id": "124",
                "algorithm": "extra_trees",
                "validation_period_name": "period_01",
                "validation_period_start": "2025-01-01T00:00:00",
                "validation_period_end": "2025-01-31T23:59:59",
                "by_target.target_value__PM2.5 (µg/m³).RMSE": 10.0,
                "by_target.target_value__PM2.5 (µg/m³).MAE": 7.0,
                "by_target.target_value__PM2.5 (µg/m³).MSE": 100.0,
                "by_target.target_value__PM2.5 (µg/m³).IA": 0.9,
                "by_target.target_value__PM2.5 (µg/m³).R2": 0.8,
                "by_target.target_value__PM2.5 (µg/m³).MAPE": 6.5,
            },
            {
                "config": "cfg_a.yaml",
                "run_dir": "run_a",
                "target_station_id": "124",
                "algorithm": "extra_trees",
                "validation_period_name": "period_02",
                "validation_period_start": "2025-04-01T00:00:00",
                "validation_period_end": "2025-04-30T23:59:59",
                "by_target.target_value__PM2.5 (µg/m³).RMSE": 12.0,
                "by_target.target_value__PM2.5 (µg/m³).MAE": 8.0,
                "by_target.target_value__PM2.5 (µg/m³).MSE": 144.0,
                "by_target.target_value__PM2.5 (µg/m³).IA": 0.88,
                "by_target.target_value__PM2.5 (µg/m³).R2": 0.79,
                "by_target.target_value__PM2.5 (µg/m³).MAPE": 7.2,
            },
        ]
    }

    written = cli._write_station_period_metric_charts(comparison, tmp_path / "charts")

    assert len(written) == 1
    assert Path(written[0]).exists()


def test_write_station_period_metric_charts_can_group_output_by_pollutant(tmp_path):
    comparison = {
        "rows": [
            {
                "config": "cfg_a.yaml",
                "run_dir": "run_a",
                "target_station_id": "124",
                "algorithm": "extra_trees",
                "validation_period_name": "period_01",
                "validation_period_start": "2025-01-01T00:00:00",
                "validation_period_end": "2025-01-31T23:59:59",
                "by_target.target_value__PM2.5 (µg/m³).RMSE": 10.0,
                "by_target.target_value__PM2.5 (µg/m³).MAE": 7.0,
            },
            {
                "config": "cfg_a.yaml",
                "run_dir": "run_a",
                "target_station_id": "124",
                "algorithm": "extra_trees",
                "validation_period_name": "period_02",
                "validation_period_start": "2025-04-01T00:00:00",
                "validation_period_end": "2025-04-30T23:59:59",
                "by_target.target_value__PM2.5 (µg/m³).RMSE": 12.0,
                "by_target.target_value__PM2.5 (µg/m³).MAE": 8.0,
            },
        ]
    }

    written = cli._write_station_period_metric_charts(
        comparison,
        tmp_path / "charts",
        group_output_by="pollutant",
    )

    assert len(written) == 1
    output_path = Path(written[0])
    assert output_path.exists()
    assert output_path.parent.name.startswith("pollutant_PM2_5")


def test_write_station_period_metric_charts_can_compare_stations(tmp_path):
    comparison = {
        "rows": [
            {
                "config": "cfg_a.yaml",
                "run_dir": "run_a",
                "target_station_id": "124",
                "algorithm": "extra_trees",
                "validation_period_name": "period_01",
                "validation_period_start": "2025-01-01T00:00:00",
                "validation_period_end": "2025-01-31T23:59:59",
                "by_target.target_value__PM2.5 (µg/m³).RMSE": 10.0,
                "by_target.target_value__PM2.5 (µg/m³).MAE": 7.0,
            },
            {
                "config": "cfg_b.yaml",
                "run_dir": "run_b",
                "target_station_id": "1421",
                "algorithm": "extra_trees",
                "validation_period_name": "period_01",
                "validation_period_start": "2025-01-01T00:00:00",
                "validation_period_end": "2025-01-31T23:59:59",
                "by_target.target_value__PM2.5 (µg/m³).RMSE": 11.0,
                "by_target.target_value__PM2.5 (µg/m³).MAE": 8.0,
            },
        ]
    }

    written = cli._write_station_period_metric_charts(
        comparison,
        tmp_path / "charts",
        comparison_mode="stations",
        group_output_by="pollutant",
    )

    assert len(written) == 1
    output_path = Path(written[0])
    assert output_path.exists()
    assert output_path.name.startswith("metrics_compare_stations_PM2_5")


def test_write_station_comparison_metric_title_omits_single_period_name(tmp_path, monkeypatch):
    import matplotlib.figure as mpl_figure

    comparison = {
        "rows": [
            {
                "config": "cfg_a.yaml",
                "run_dir": "run_a",
                "target_station_id": "124",
                "algorithm": "extra_trees",
                "validation_period_name": "period_01",
                "validation_period_start": "2025-01-01T00:00:00",
                "validation_period_end": "2025-01-31T23:59:59",
                "by_target.target_value__PM2.5 (µg/m³).RMSE": 10.0,
            },
            {
                "config": "cfg_b.yaml",
                "run_dir": "run_b",
                "target_station_id": "1421",
                "algorithm": "extra_trees",
                "validation_period_name": "period_01",
                "validation_period_start": "2025-01-01T00:00:00",
                "validation_period_end": "2025-01-31T23:59:59",
                "by_target.target_value__PM2.5 (µg/m³).RMSE": 11.0,
            },
        ]
    }

    captured_titles: list[str] = []
    original_savefig = mpl_figure.Figure.savefig

    def _capture_savefig(self, fname, *args, **kwargs):
        captured_titles.append(self._suptitle.get_text() if self._suptitle is not None else "")
        Path(fname).parent.mkdir(parents=True, exist_ok=True)
        Path(fname).write_text("png", encoding="utf-8")

    monkeypatch.setattr(mpl_figure.Figure, "savefig", _capture_savefig)

    cli._write_station_period_metric_charts(
        comparison,
        tmp_path / "charts",
        comparison_mode="stations",
        group_output_by="pollutant",
        requested_metrics=["RMSE"],
    )

    assert captured_titles
    assert "period_01" not in captured_titles[0]
    assert captured_titles[0] == "Metrics by Station - target_value__PM2.5 (µg/m³)\n2025-01-01 to 2025-01-31"
    monkeypatch.setattr(mpl_figure.Figure, "savefig", original_savefig)


def test_write_station_period_metric_charts_station_mode_has_legend_and_colors_best_worst(tmp_path, monkeypatch):
    import matplotlib.figure as mpl_figure

    comparison = {
        "rows": [
            {
                "config": "cfg_a.yaml",
                "run_dir": "run_a",
                "target_station_id": "124",
                "algorithm": "baseline_mean",
                "validation_period_name": "period_01",
                "validation_period_start": "2025-01-01T00:00:00",
                "validation_period_end": "2025-01-31T23:59:59",
                "by_target.target_value__PM2.5 (µg/m³).RMSE": 12.0,
                "by_target.target_value__PM2.5 (µg/m³).R2": 0.6,
            },
            {
                "config": "cfg_b.yaml",
                "run_dir": "run_b",
                "target_station_id": "124",
                "algorithm": "extra_trees",
                "validation_period_name": "period_01",
                "validation_period_start": "2025-01-01T00:00:00",
                "validation_period_end": "2025-01-31T23:59:59",
                "by_target.target_value__PM2.5 (µg/m³).RMSE": 10.0,
                "by_target.target_value__PM2.5 (µg/m³).R2": 0.75,
            },
            {
                "config": "cfg_a.yaml",
                "run_dir": "run_a",
                "target_station_id": "1421",
                "algorithm": "baseline_mean",
                "validation_period_name": "period_01",
                "validation_period_start": "2025-01-01T00:00:00",
                "validation_period_end": "2025-01-31T23:59:59",
                "by_target.target_value__PM2.5 (µg/m³).RMSE": 8.0,
                "by_target.target_value__PM2.5 (µg/m³).R2": 0.8,
            },
            {
                "config": "cfg_b.yaml",
                "run_dir": "run_b",
                "target_station_id": "1421",
                "algorithm": "extra_trees",
                "validation_period_name": "period_01",
                "validation_period_start": "2025-01-01T00:00:00",
                "validation_period_end": "2025-01-31T23:59:59",
                "by_target.target_value__PM2.5 (µg/m³).RMSE": 9.0,
                "by_target.target_value__PM2.5 (µg/m³).R2": 0.77,
            },
        ]
    }

    legend_counts: list[int] = []
    numeric_label_colors: list = []
    original_savefig = mpl_figure.Figure.savefig

    def _capture_savefig(self, fname, *args, **kwargs):
        legend_counts.append(len(self.legends))
        for ax in self.axes:
            for txt in ax.texts:
                label_text = txt.get_text().strip()
                if not label_text:
                    continue
                try:
                    float(label_text)
                except ValueError:
                    continue
                numeric_label_colors.append(txt.get_color())
        Path(fname).parent.mkdir(parents=True, exist_ok=True)
        Path(fname).write_text("png", encoding="utf-8")

    monkeypatch.setattr(mpl_figure.Figure, "savefig", _capture_savefig)

    cli._write_station_period_metric_charts(
        comparison,
        tmp_path / "charts",
        comparison_mode="stations",
        group_output_by="pollutant",
        requested_metrics=["RMSE", "R2"],
    )

    assert legend_counts and legend_counts[0] > 0
    assert "green" in numeric_label_colors
    assert "red" in numeric_label_colors
    monkeypatch.setattr(mpl_figure.Figure, "savefig", original_savefig)


def test_write_station_period_metric_charts_station_mode_clusters_baseline_before_models(tmp_path, monkeypatch):
    import matplotlib.axes as mpl_axes

    comparison = {
        "rows": [
            {
                "config": "cfg_a.yaml",
                "run_dir": "run_a",
                "target_station_id": "124",
                "algorithm": "baseline_mean",
                "validation_period_name": "period_01",
                "validation_period_start": "2025-01-01T00:00:00",
                "validation_period_end": "2025-01-31T23:59:59",
                "by_target.target_value__PM2.5 (µg/m³).RMSE": 12.0,
            },
            {
                "config": "cfg_b.yaml",
                "run_dir": "run_b",
                "target_station_id": "124",
                "algorithm": "extra_trees",
                "validation_period_name": "period_01",
                "validation_period_start": "2025-01-01T00:00:00",
                "validation_period_end": "2025-01-31T23:59:59",
                "by_target.target_value__PM2.5 (µg/m³).RMSE": 10.0,
            },
            {
                "config": "cfg_a.yaml",
                "run_dir": "run_a",
                "target_station_id": "1421",
                "algorithm": "baseline_mean",
                "validation_period_name": "period_01",
                "validation_period_start": "2025-01-01T00:00:00",
                "validation_period_end": "2025-01-31T23:59:59",
                "by_target.target_value__PM2.5 (µg/m³).RMSE": 8.0,
            },
            {
                "config": "cfg_b.yaml",
                "run_dir": "run_b",
                "target_station_id": "1421",
                "algorithm": "extra_trees",
                "validation_period_name": "period_01",
                "validation_period_start": "2025-01-01T00:00:00",
                "validation_period_end": "2025-01-31T23:59:59",
                "by_target.target_value__PM2.5 (µg/m³).RMSE": 9.0,
            },
        ]
    }

    call_order: list[str] = []
    captured_offsets: dict[str, tuple[list[float], float]] = {}
    original_bar = mpl_axes.Axes.bar

    def _capture_bar(self, x, height, *args, **kwargs):
        label = kwargs.get("label")
        if label in {"baseline_mean", "extra_trees"} and label not in captured_offsets:
            call_order.append(str(label))
            captured_offsets[str(label)] = ([float(v) for v in x], float(kwargs.get("width", args[0] if args else 0.0)))
        return original_bar(self, x, height, *args, **kwargs)

    monkeypatch.setattr(mpl_axes.Axes, "bar", _capture_bar)

    cli._write_station_period_metric_charts(
        comparison,
        tmp_path / "charts",
        comparison_mode="stations",
        group_output_by="pollutant",
        requested_metrics=["RMSE"],
    )

    assert "baseline_mean" in captured_offsets and "extra_trees" in captured_offsets
    assert call_order.index("baseline_mean") < call_order.index("extra_trees")
    baseline_x, baseline_width = captured_offsets["baseline_mean"]
    model_x, _ = captured_offsets["extra_trees"]
    assert abs(model_x[0] - baseline_x[0]) > baseline_width * 1.5


def test_write_station_period_metric_charts_falls_back_to_overall_metrics(tmp_path):
    comparison = {
        "rows": [
            {
                "config": "cfg_a.yaml",
                "run_dir": "run_a",
                "target_station_id": "124",
                "algorithm": "extra_trees",
                "validation_period_name": "overall",
                "validation_period_start": "",
                "validation_period_end": "",
                "by_target.target_value__PM2.5 (µg/m³).RMSE": 10.0,
                "by_target.target_value__PM2.5 (µg/m³).MAE": 7.0,
            },
        ]
    }

    written = cli._write_station_period_metric_charts(
        comparison,
        tmp_path / "charts",
        group_output_by="pollutant",
    )

    assert len(written) == 1
    output_path = Path(written[0])
    assert output_path.exists()
    assert output_path.parent.name.startswith("pollutant_PM2_5")


def test_write_station_period_metric_charts_falls_back_to_overall_for_station_comparison(tmp_path):
    comparison = {
        "rows": [
            {
                "config": "cfg_a.yaml",
                "run_dir": "run_a",
                "target_station_id": "124",
                "algorithm": "extra_trees",
                "validation_period_name": "overall",
                "validation_period_start": "",
                "validation_period_end": "",
                "by_target.target_value__PM2.5 (µg/m³).RMSE": 10.0,
                "by_target.target_value__PM2.5 (µg/m³).MAE": 7.0,
            },
            {
                "config": "cfg_b.yaml",
                "run_dir": "run_b",
                "target_station_id": "1421",
                "algorithm": "extra_trees",
                "validation_period_name": "overall",
                "validation_period_start": "",
                "validation_period_end": "",
                "by_target.target_value__PM2.5 (µg/m³).RMSE": 11.0,
                "by_target.target_value__PM2.5 (µg/m³).MAE": 8.0,
            },
        ]
    }

    written = cli._write_station_period_metric_charts(
        comparison,
        tmp_path / "charts",
        comparison_mode="stations",
        group_output_by="pollutant",
    )

    assert len(written) == 1
    output_path = Path(written[0])
    assert output_path.exists()
    assert output_path.parent.name.startswith("pollutant_PM2_5")


def test_write_station_period_metric_charts_uses_title_template(tmp_path, monkeypatch):
    import matplotlib.figure as mpl_figure

    comparison = {
        "rows": [
            {
                "config": "cfg_a.yaml",
                "run_dir": "run_a",
                "target_station_id": "124",
                "algorithm": "extra_trees",
                "validation_period_name": "period_01",
                "validation_period_start": "2025-01-01T00:00:00",
                "validation_period_end": "2025-01-31T23:59:59",
                "by_target.target_value__PM2.5 (µg/m³).RMSE": 10.0,
                "by_target.target_value__PM2.5 (µg/m³).MAE": 7.0,
            },
            {
                "config": "cfg_a.yaml",
                "run_dir": "run_a",
                "target_station_id": "124",
                "algorithm": "extra_trees",
                "validation_period_name": "period_02",
                "validation_period_start": "2025-04-01T00:00:00",
                "validation_period_end": "2025-04-30T23:59:59",
                "by_target.target_value__PM2.5 (µg/m³).RMSE": 12.0,
                "by_target.target_value__PM2.5 (µg/m³).MAE": 8.0,
            },
        ]
    }

    captured_titles: list[str] = []
    original_savefig = mpl_figure.Figure.savefig

    def _capture_savefig(self, fname, *args, **kwargs):
        captured_titles.append(self._suptitle.get_text() if self._suptitle is not None else "")
        Path(fname).parent.mkdir(parents=True, exist_ok=True)
        Path(fname).write_text("png", encoding="utf-8")

    monkeypatch.setattr(mpl_figure.Figure, "savefig", _capture_savefig)

    cli._write_station_period_metric_charts(
        comparison,
        tmp_path / "charts",
        title_templates={"metrics_by_period": "Metrics template {station_id} / {pollutant}"},
    )

    assert any(title == "Metrics template 124 / target_value__PM2.5 (µg/m³)" for title in captured_titles)
    monkeypatch.setattr(mpl_figure.Figure, "savefig", original_savefig)


def test_resolve_plot_metric_graph_options_defaults_without_config():
    options = cli._resolve_plot_metric_graph_options(None)

    assert options == {
        "metrics": None,
        "comparison_mode": "validation_periods",
        "group_output_by": "pollutant",
    }


def test_resolve_plot_metric_graph_options_reads_metrics_graph_config(tmp_path):
    cfg_path = tmp_path / "plot_cfg.yaml"
    cfg_path.write_text(
        """
plot:
  metrics_graph:
    metrics: [rmse, mae]
    comparison_mode: stations
    group_output_by: pollutant
""".strip(),
        encoding="utf-8",
    )

    options = cli._resolve_plot_metric_graph_options([str(cfg_path)])

    assert options["metrics"] == ["rmse", "mae"]
    assert options["comparison_mode"] == "stations"
    assert options["group_output_by"] == "pollutant"


def test_coerce_optimization_algorithm_aliases_and_none():
    assert cli._coerce_optimization_algorithm(None) == "none"
    assert cli._coerce_optimization_algorithm("none") == "none"
    assert cli._coerce_optimization_algorithm("off") == "none"
    assert cli._coerce_optimization_algorithm("random") == "random_search"
    assert cli._coerce_optimization_algorithm("grid") == "grid_search"


def test_optimize_model_params_returns_none_when_disabled():
    cfgd = {
        "model": {
            "algorithm": "extra_trees",
            "target": "target_value",
            "datetime_col": "Timestamp",
            "params": {"n_estimators": 10, "random_state": 42},
            "optimization": {"algorithm": "none"},
        }
    }
    X = pd.DataFrame({"a": [1, 2, 3, 4], "b": [4, 3, 2, 1]})
    y = pd.Series([1.0, 1.5, 2.0, 2.5], name="target_value")

    result = cli._optimize_model_params(cfgd, X, y)

    assert result is None


def test_optimize_model_params_selects_best_candidate(tmp_path):
    cfgd = {
        "model": {
            "algorithm": "extra_trees",
            "target": "target_value",
            "datetime_col": "Timestamp",
            "params": {"random_state": 42},
            "validation": {
                "algorithm": "grid_search",
                "metric": "RMSE",
                "param_grid": {
                    "n_estimators": [5, 20],
                    "max_depth": [3],
                },
            },
        }
    }
    X = pd.DataFrame({
        "f1": [1, 2, 3, 4, 5, 6, 7, 8],
        "f2": [8, 7, 6, 5, 4, 3, 2, 1],
    })
    y = pd.Series([1.1, 1.9, 3.2, 3.8, 5.1, 6.0, 7.0, 8.1], name="target_value")

    train_X = X.iloc[:6].reset_index(drop=True)
    train_y = y.iloc[:6].reset_index(drop=True)
    validation_X = X.iloc[6:].reset_index(drop=True)
    validation_y = y.iloc[6:].reset_index(drop=True)

    result = cli._optimize_model_params(
        cfgd,
        train_X,
        train_y,
        validation_X=validation_X,
        validation_y=validation_y,
    )

    assert result is not None
    assert result["algorithm"] == "grid_search"
    assert result["trial_count"] == 2
    assert result["validation_rows"] == 2
    assert "best_params" in result
    assert set(result["best_params"].keys()) >= {"random_state", "n_estimators", "max_depth"}


def test_optimize_model_params_uses_param_grid_random_state_values():
    cfgd = {
        "model": {
            "algorithm": "extra_trees",
            "target": "target_value",
            "datetime_col": "Timestamp",
            "params": {"n_estimators": 5, "random_state": 42},
            "validation": {
                "algorithm": "grid_search",
                "metric": "RMSE",
                "random_state": 42,
                "param_grid": {
                    "max_depth": [2],
                    "random_state": [42, 50, 100],
                },
            },
        }
    }
    X = pd.DataFrame({
        "f1": [1, 2, 3, 4, 5, 6],
        "f2": [6, 5, 4, 3, 2, 1],
    })
    y = pd.Series([1.0, 2.1, 2.9, 4.2, 5.0, 6.1], name="target_value")

    result = cli._optimize_model_params(
        cfgd,
        X.iloc[:4].reset_index(drop=True),
        y.iloc[:4].reset_index(drop=True),
        validation_X=X.iloc[4:].reset_index(drop=True),
        validation_y=y.iloc[4:].reset_index(drop=True),
    )

    assert result is not None
    assert result["trial_count"] == 3
    trial_states = {trial["params"]["random_state"] for trial in result["trials"]}
    assert trial_states == {42, 50, 100}
    assert result["best_params"]["random_state"] in {42, 50, 100}


def test_build_model_config_uses_first_random_state_when_list_without_tuning():
    cfgd = {
        "model": {
            "algorithm": "extra_trees",
            "target": "target_value",
            "datetime_col": "Timestamp",
            "params": {"n_estimators": 10, "random_state": [42, 50, 100]},
        }
    }

    model_config = cli._build_model_config(cfgd)

    assert model_config.params["random_state"] == 42
