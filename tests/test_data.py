import pandas as pd
import pytest

from src.data import (
    apply_norm_stats,
    column_statistics,
    correlation_analysis,
    identity_norm_stats,
    impute_targets_with_random_forest,
    infer_station_id_from_path,
    load_timeseries_csv,
    load_timeseries_folder,
    missing_value_stats,
    normalize_train_test,
)


def test_load_timeseries_csv_normalizes_datetime_name(tmp_path):
    path = tmp_path / "sample.csv"
    path.write_text(
        " DateTime , station_id , PM2.5 \n2023-01-02,a,2\n2023-01-01,a,1\n",
        encoding="utf-8",
    )

    df = load_timeseries_csv(path, datetime_col="datetime")

    assert list(df.columns) == ["datetime", "station_id", "PM2.5"]
    assert df.loc[0, "datetime"].isoformat() == "2023-01-01T00:00:00"


def test_load_timeseries_folder_combines_files(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    pd.DataFrame({"datetime": ["2023-01-02"], "PM2.5": [2]}).to_csv(
        data_dir / "b.csv",
        index=False,
    )
    pd.DataFrame({"datetime": ["2023-01-01"], "PM2.5": [1]}).to_csv(
        data_dir / "a.csv",
        index=False,
    )

    df, files = load_timeseries_folder(
        data_dir,
        datetime_col="datetime",
        sort_col="datetime",
        station_id_col="station_id",
        station_id_from="filename",
    )

    assert len(files) == 2
    assert df["PM2.5"].tolist() == [1, 2]
    assert df["station_id"].tolist() == ["a", "b"]


def test_load_timeseries_folder_ignores_cache_csvs(tmp_path):
    data_dir = tmp_path / "data"
    cache_dir = data_dir / ".aqsr_cache"
    data_dir.mkdir()
    cache_dir.mkdir()
    pd.DataFrame({"datetime": ["2023-01-01"], "PM2.5": [1]}).to_csv(
        data_dir / "station_a.csv",
        index=False,
    )
    pd.DataFrame({"stage": ["after"], "column": ["PM2.5"]}).to_csv(
        cache_dir / "column_statistics_after.csv",
        index=False,
    )

    df, files = load_timeseries_folder(
        data_dir,
        datetime_col="datetime",
        sort_col="datetime",
        pattern="**/*.csv",
        station_id_col="station_id",
        station_id_from="filename",
    )

    assert len(files) == 1
    assert files[0].endswith("station_a.csv")
    assert df["station_id"].tolist() == ["station_a"]


def test_normalize_and_apply_stats_round_trip():
    train = pd.DataFrame({"f1": [10.0, 12.0], "f2": [1.0, 3.0]})
    test = pd.DataFrame({"f1": [14.0], "f2": [5.0]})

    train_n, test_n, stats = normalize_train_test(train, test)
    restored = apply_norm_stats(test, stats)

    assert train_n["f1"].tolist() == pytest.approx([-1.0, 1.0])
    assert restored["f1"].tolist() == pytest.approx(test_n["f1"].tolist())


def test_missing_and_column_stats_capture_structure():
    df = pd.DataFrame({"station_id": ["a", "b"], "PM2.5": [1.0, None], "NO2": [2.0, 4.0]})

    missing = missing_value_stats(df, exclude=["station_id"])
    stats = column_statistics(df, exclude=["station_id"])

    assert missing["total_missing"] == 1
    assert stats["columns"]["PM2.5"]["present_count"] == 1
    assert stats["columns"]["PM2.5"]["present_pct_of_all_rows"] == pytest.approx(50.0)
    assert stats["columns"]["PM2.5"]["missing_pct_of_all_rows"] == pytest.approx(50.0)
    assert stats["columns"]["NO2"]["median"] == pytest.approx(3.0)


def test_infer_station_id_from_path_supports_filename_and_folder():
    assert infer_station_id_from_path("Raw_data_1Hr_2025_site_1426_Narela.csv", mode="filename") == "1426"
    assert infer_station_id_from_path("data\\1426\\sample.csv", mode="folder", folder_root="data") == "1426"


def test_impute_targets_with_random_forest_fills_missing_values():
    df = pd.DataFrame(
        {
            "PM2.5": [10.0, 12.0, None, 16.0],
            "NO2": [20.0, 21.0, 22.0, 23.0],
            "RH (%)": [50.0, 52.0, 54.0, 56.0],
        }
    )

    out, details, analysis = impute_targets_with_random_forest(
        df,
        target_columns=["PM2.5", "NO2"],
    )

    assert out["PM2.5"].isna().sum() == 0
    assert details["columns"]["PM2.5"]["method"] == "random_forest"
    assert "NO2" in analysis["columns"]


def test_correlation_analysis_returns_columns_and_matrix():
    df = pd.DataFrame({"PM2.5": [1.0, 2.0], "NO2": [3.0, 4.0]})

    analysis = correlation_analysis(df)

    assert analysis["columns"] == ["PM2.5", "NO2"]
    assert len(analysis["matrix"]) == 2


def test_correlation_analysis_drops_zero_signal_columns():
    df = pd.DataFrame(
        {
            "PM2.5": [1.0, 2.0, 3.0, 4.0],
            "NO2": [2.0, 4.0, 6.0, 8.0],
            "O Xylene": [5.0, 5.0, 5.0, 5.0],
        }
    )

    analysis = correlation_analysis(df)

    assert analysis["columns"] == ["PM2.5", "NO2"]
    assert len(analysis["matrix"]) == 2


def test_identity_norm_stats_builds_passthrough_maps():
    df = pd.DataFrame({"x": [5.0]})

    assert identity_norm_stats(df) == {"method": "none", "mu": {"x": 0.0}, "sigma": {"x": 1.0}}


def test_minmax_normalization_round_trip():
    train = pd.DataFrame({"f1": [10.0, 20.0], "f2": [2.0, 6.0]})
    test = pd.DataFrame({"f1": [15.0], "f2": [4.0]})

    train_n, test_n, stats = normalize_train_test(train, test, method="minmax")
    restored = apply_norm_stats(test, stats)

    assert stats["method"] == "minmax"
    assert train_n["f1"].tolist() == pytest.approx([0.0, 1.0])
    assert restored["f1"].tolist() == pytest.approx(test_n["f1"].tolist())


def test_robust_normalization_round_trip():
    train = pd.DataFrame({"f1": [0.0, 10.0, 20.0], "f2": [1.0, 1.0, 9.0]})
    test = pd.DataFrame({"f1": [15.0], "f2": [5.0]})

    _, test_n, stats = normalize_train_test(train, test, method="robust")
    restored = apply_norm_stats(test, stats)

    assert stats["method"] == "robust"
    assert restored["f1"].tolist() == pytest.approx(test_n["f1"].tolist())


def test_maxabs_normalization_round_trip():
    train = pd.DataFrame({"f1": [-2.0, 4.0], "f2": [-10.0, 5.0]})
    test = pd.DataFrame({"f1": [1.0], "f2": [-2.5]})

    _, test_n, stats = normalize_train_test(train, test, method="maxabs")
    restored = apply_norm_stats(test, stats)

    assert stats["method"] == "maxabs"
    assert restored["f2"].tolist() == pytest.approx(test_n["f2"].tolist())
