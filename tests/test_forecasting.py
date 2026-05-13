import pandas as pd

from src.forecasting import build_supervised_dataset, select_test_rows


def test_build_supervised_dataset_supports_neighbor_station_history_lags():
    timestamps = pd.to_datetime([
        "2025-01-01 00:00:00",
        "2025-01-01 01:00:00",
        "2025-01-01 02:00:00",
        "2025-01-01 03:00:00",
    ])
    station_matrix = pd.DataFrame(
        {
            "target_station": [10.0, 11.0, 12.0, 13.0],
            "station_a": [1.0, 2.0, 3.0, 4.0],
            "station_b": [5.0, 6.0, 7.0, 8.0],
        },
        index=timestamps,
    )

    cfg = {
        "target_station_id": "target_station",
        "input_station_ids": ["station_a", "station_b"],
        "target_pollutant": "PM2.5",
        "station_history_lags": 3,
    }

    supervised, feature_names, target_names = build_supervised_dataset(station_matrix, cfg, horizon=1)

    assert target_names == ["target_value"]
    assert "neighbor__station_a" in feature_names
    assert "neighbor__station_a__lag_1" in feature_names
    assert "neighbor__station_a__lag_2" in feature_names
    assert "neighbor__station_b" in feature_names
    assert "neighbor__station_b__lag_1" in feature_names
    assert "neighbor__station_b__lag_2" in feature_names

    first_row = supervised.iloc[0]
    assert first_row["neighbor__station_a"] == 3.0
    assert first_row["neighbor__station_a__lag_1"] == 2.0
    assert first_row["neighbor__station_a__lag_2"] == 1.0


def test_select_test_rows_uses_union_even_when_full_year_period_is_present():
    supervised_df = pd.DataFrame(
        {
            "prediction_time": pd.to_datetime(
                [
                    "2024-01-10 00:00:00",
                    "2025-01-10 00:00:00",
                    "2025-04-10 00:00:00",
                    "2025-10-10 00:00:00",
                ]
            )
        }
    )
    cfg = {
        "validation_periods": [
            {"start": "2025", "end": "2025"},
            {"start": "2024-01", "end": "2024-01"},
        ]
    }

    mask = select_test_rows(supervised_df, cfg)

    assert mask.tolist() == [True, True, True, True]


def test_select_test_rows_uses_union_when_no_full_year_period_exists():
    supervised_df = pd.DataFrame(
        {
            "prediction_time": pd.to_datetime(
                [
                    "2025-01-10 00:00:00",
                    "2025-02-10 00:00:00",
                    "2025-04-10 00:00:00",
                ]
            )
        }
    )
    cfg = {
        "validation_periods": [
            {"start": "2025-01", "end": "2025-01"},
            {"start": "2025-04", "end": "2025-04"},
        ]
    }

    mask = select_test_rows(supervised_df, cfg)

    assert mask.tolist() == [True, False, True]
