"""Utilities for building supervised forecasting windows from station matrices."""

from __future__ import annotations

from calendar import monthrange

import pandas as pd

from .reconstruction import resolve_target_pollutants


def _positive_int_config(data_cfg: dict, key: str, default: int) -> int:
    raw_value = data_cfg.get(key, default)
    try:
        value = int(raw_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"data.{key} must be a positive integer.") from exc
    if value < 1:
        raise ValueError(f"data.{key} must be >= 1.")
    return value


def build_supervised_dataset(station_matrix: pd.DataFrame, data_cfg: dict, horizon: int):
    """Build a horizon-based supervised dataset for the configured stations."""

    if horizon < 0:
        raise ValueError("horizon must be >= 0")

    target_station = data_cfg["target_station_id"]
    input_stations = list(data_cfg["input_station_ids"])
    station_history_lags = _positive_int_config(data_cfg, "station_history_lags", 1)
    target_pollutants = resolve_target_pollutants(data_cfg)
    is_multi_target = len(target_pollutants) > 1

    def _matrix_column(pollutant: str, station_id: str) -> str:
        return f"{pollutant}__{station_id}" if is_multi_target else station_id

    feature_frame = pd.DataFrame(index=station_matrix.index)
    feature_names: list[str] = []
    for pollutant in target_pollutants:
        for station_id in input_stations:
            source_col = _matrix_column(pollutant, station_id)
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
                feature_frame[feature_name] = station_matrix[source_col].shift(lag)
                feature_names.append(feature_name)

    # horizon=0 means reconstructing current target value (nowcast) at the same timestamp.
    target_columns: list[str] = []
    for pollutant in target_pollutants:
        source_col = _matrix_column(pollutant, target_station)
        target_name = f"target_value__{pollutant}" if is_multi_target else "target_value"
        feature_frame[target_name] = station_matrix[source_col].shift(-horizon)
        target_columns.append(target_name)
    feature_frame["prediction_time"] = station_matrix.index.to_series().shift(-horizon)
    feature_frame = feature_frame.dropna().reset_index().rename(columns={"index": "feature_time"})
    feature_frame = feature_frame.rename(columns={feature_frame.columns[0]: "feature_time"})
    return feature_frame, feature_names, target_columns


def _parse_period_boundary(value, *, is_end: bool) -> pd.Timestamp:
    text = str(value).strip()
    if not text:
        raise ValueError("Period boundaries cannot be empty.")

    if len(text) == 4 and text.isdigit():
        year = int(text)
        return pd.Timestamp(year=year, month=12 if is_end else 1, day=31 if is_end else 1, hour=23 if is_end else 0, minute=59 if is_end else 0, second=59 if is_end else 0)

    if len(text) == 7 and text[4] == "-":
        year = int(text[:4])
        month = int(text[5:7])
        day = monthrange(year, month)[1] if is_end else 1
        return pd.Timestamp(
            year=year,
            month=month,
            day=day,
            hour=23 if is_end else 0,
            minute=59 if is_end else 0,
            second=59 if is_end else 0,
        )

    stamp = pd.Timestamp(text)
    if len(text) == 10:
        return stamp.replace(hour=23 if is_end else 0, minute=59 if is_end else 0, second=59 if is_end else 0)
    return stamp


def normalize_periods(raw_periods) -> list[dict]:
    """Normalize configured period specs into inclusive start/end timestamps."""

    if raw_periods is None:
        return []
    if isinstance(raw_periods, (str, dict)):
        raw_periods = [raw_periods]

    periods: list[dict] = []
    for item in raw_periods:
        if isinstance(item, str):
            start = _parse_period_boundary(item, is_end=False)
            end = _parse_period_boundary(item, is_end=True)
        elif isinstance(item, dict):
            if "start" not in item and "end" not in item:
                raise ValueError("Period dictionaries must include at least one of 'start' or 'end'.")
            raw_start = item.get("start", item.get("end"))
            raw_end = item.get("end", item.get("start"))
            start = _parse_period_boundary(raw_start, is_end=False)
            end = _parse_period_boundary(raw_end, is_end=True)
        else:
            raise TypeError("Periods must be provided as strings or dictionaries.")

        if end < start:
            raise ValueError(f"Invalid period with end before start: {item}")
        periods.append({"start": start, "end": end})
    return periods


def resolve_validation_periods(data_cfg: dict) -> list[dict]:
    """Return validation periods from the modern range config or legacy test_days."""

    if data_cfg.get("validation_periods") is not None:
        return normalize_periods(data_cfg.get("validation_periods"))
    test_days = data_cfg.get("test_days") or []
    return normalize_periods(list(test_days))


def resolve_testing_periods(data_cfg: dict) -> list[dict]:
    """Return testing periods, falling back to validation periods or legacy test_days."""

    if data_cfg.get("testing_periods") is not None:
        return normalize_periods(data_cfg.get("testing_periods"))
    if data_cfg.get("validation_periods") is not None:
        return normalize_periods(data_cfg.get("validation_periods"))
    test_days = data_cfg.get("test_days") or []
    return normalize_periods(list(test_days))


def build_period_mask(values, periods) -> pd.Series:
    """Build an inclusive boolean mask for the provided timestamp-like values."""

    index = pd.to_datetime(values)
    normalized = normalize_periods(periods)
    series_index = values.index if hasattr(values, "index") and not callable(getattr(values, "index")) else None
    if not normalized:
        return pd.Series([False] * len(index), index=series_index)

    mask = pd.Series([False] * len(index), index=series_index)
    for period in normalized:
        mask |= (index >= period["start"]) & (index <= period["end"])
    return mask


def years_for_periods(periods) -> list[int]:
    """Return all calendar years touched by the configured periods."""

    years: set[int] = set()
    for period in normalize_periods(periods):
        for year in range(period["start"].year, period["end"].year + 1):
            years.add(year)
    return sorted(years)


def select_test_rows(supervised_df: pd.DataFrame, data_cfg: dict) -> pd.Series:
    """Select evaluation rows from configured testing periods."""

    prediction_time = pd.to_datetime(supervised_df["prediction_time"])
    testing_periods = resolve_testing_periods(data_cfg)

    if testing_periods:
        return build_period_mask(prediction_time, testing_periods)

    cutoff = max(1, int(len(supervised_df) * float(data_cfg.get("train_frac", 0.7))))
    mask = pd.Series([False] * len(supervised_df), index=supervised_df.index)
    mask.iloc[cutoff:] = True
    return mask


def split_supervised_rows(supervised_df: pd.DataFrame, data_cfg: dict):
    """Split the supervised dataset into train and configured test rows."""

    train_years = {int(year) for year in data_cfg.get("train_years") or []}
    if train_years:
        prediction_time = pd.to_datetime(supervised_df["prediction_time"])
        train_mask = prediction_time.dt.year.isin(train_years)
        test_mask = select_test_rows(supervised_df, data_cfg)
        train_df = supervised_df.loc[train_mask & ~test_mask].reset_index(drop=True)
        test_df = supervised_df.loc[test_mask].reset_index(drop=True)
        if len(train_df) == 0 or len(test_df) == 0:
            raise ValueError(
                "Configured train_years/test_days selection produced an empty train or test split."
            )
        return train_df, test_df

    test_mask = select_test_rows(supervised_df, data_cfg)
    if not bool(test_mask.any()):
        cutoff = max(1, min(len(supervised_df) - 1, int(len(supervised_df) * float(data_cfg.get("train_frac", 0.7)))))
        train_df = supervised_df.iloc[:cutoff].reset_index(drop=True)
        test_df = supervised_df.iloc[cutoff:].reset_index(drop=True)
        return train_df, test_df

    train_df = supervised_df.loc[~test_mask].reset_index(drop=True)
    test_df = supervised_df.loc[test_mask].reset_index(drop=True)
    return train_df, test_df
