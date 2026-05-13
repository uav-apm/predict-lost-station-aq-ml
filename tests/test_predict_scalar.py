import pandas as pd

from src.pipeline import StationForecastConfig, StationForecastPipeline


def test_predict_scalar_returns_float():
    X = pd.DataFrame(
        {
            "neighbor__station_a": [1.0, 2.0, 3.0, 4.0],
            "neighbor__station_b": [2.0, 3.0, 4.0, 5.0],
            "neighbor__station_c": [3.0, 4.0, 5.0, 6.0],
        }
    )
    y = pd.Series([10.0, 12.0, 14.0, 16.0])

    pipe = StationForecastPipeline(
        StationForecastConfig(target="PM2.5", datetime_col="datetime", algorithm="random_forest")
    )
    pipe.fit(X, y)

    value = pipe.predict_scalar(
        {
            "neighbor__station_a": 4.0,
            "neighbor__station_b": 5.0,
            "neighbor__station_c": 6.0,
        }
    )

    assert isinstance(value, float)


def test_evaluate_multi_output_returns_only_by_target_metrics():
    X = pd.DataFrame(
        {
            "neighbor__station_a": [1.0, 2.0, 3.0, 4.0],
            "neighbor__station_b": [2.0, 3.0, 4.0, 5.0],
            "neighbor__station_c": [3.0, 4.0, 5.0, 6.0],
        }
    )
    y = pd.DataFrame(
        {
            "PM2.5 (ug/m3)": [10.0, 12.0, 14.0, 16.0],
            "NO2 (ug/m3)": [20.0, 22.0, 24.0, 26.0],
        }
    )

    pipe = StationForecastPipeline(
        StationForecastConfig(
            target="PM2.5",
            datetime_col="datetime",
            algorithm="random_forest",
        )
    )
    pipe.fit(X, y)
    result = pipe.evaluate(X, y)

    assert "by_target" in result["metrics"]
    assert "overall" not in result["metrics"]
    assert set(result["metrics"]["by_target"].keys()) == {"PM2.5 (ug/m3)", "NO2 (ug/m3)"}
