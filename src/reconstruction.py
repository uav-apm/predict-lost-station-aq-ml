"""Station-selection and feature-building helpers for reconstruction tasks."""

from __future__ import annotations

import pandas as pd


def resolve_target_pollutants(data_cfg: dict) -> list[str]:
    """Normalize configured target pollutant(s) into a non-empty list."""

    raw = data_cfg.get("target_pollutant")
    if isinstance(raw, str):
        values = [raw]
    elif isinstance(raw, list):
        values = [str(item) for item in raw if str(item).strip()]
    else:
        raise TypeError("data.target_pollutant must be a string or list of strings.")

    if not values:
        raise ValueError("data.target_pollutant must include at least one pollutant.")
    return values


def required_raw_columns(data_cfg: dict, model_cfg: dict) -> list[str]:
    """Return raw columns that must be preserved when subsetting input CSVs."""

    columns: list[str] = []
    selected = data_cfg.get("columns") or []
    target_pollutants = resolve_target_pollutants(data_cfg)
    model_targets = model_cfg.get("target")
    if isinstance(model_targets, str):
        model_target_values = [model_targets]
    elif isinstance(model_targets, list):
        model_target_values = [str(item) for item in model_targets]
    else:
        model_target_values = []
    for value in [
        *selected,
        data_cfg["datetime_col"],
        data_cfg.get("sort_col", data_cfg["datetime_col"]),
        data_cfg.get("station_id_col") or "station_id",
        *target_pollutants,
        *model_target_values,
        *(data_cfg.get("impute_columns") or []),
    ]:
        if value and value not in columns:
            columns.append(value)
    return columns


def validate_station_roles(df: pd.DataFrame, data_cfg: dict):
    """Ensure configured station ids are present in the loaded dataset."""

    station_col = data_cfg.get("station_id_col") or "station_id"
    available = set(df[station_col].astype(str).unique())
    required = [data_cfg["target_station_id"], *data_cfg["input_station_ids"]]
    missing = [station for station in required if station not in available]
    if missing:
        raise KeyError("Configured station ids were not found in the dataset: " + ", ".join(missing))


def build_station_matrix(df: pd.DataFrame, data_cfg: dict) -> pd.DataFrame:
    """Pivot long-form pollutant readings into a wide time-indexed station matrix."""

    pollutants = resolve_target_pollutants(data_cfg)
    datetime_col = data_cfg["datetime_col"]
    station_col = data_cfg.get("station_id_col") or "station_id"
    if len(pollutants) == 1:
        pollutant = pollutants[0]
        subset = df[[datetime_col, station_col, pollutant]].copy()
        wide = subset.pivot_table(index=datetime_col, columns=station_col, values=pollutant, aggfunc="mean")
        wide = wide.sort_index()
        wide.columns = [str(col) for col in wide.columns]
        return wide

    subset = df[[datetime_col, station_col, *pollutants]].copy()
    melted = subset.melt(
        id_vars=[datetime_col, station_col],
        value_vars=pollutants,
        var_name="pollutant",
        value_name="value",
    )
    wide = melted.pivot_table(
        index=datetime_col,
        columns=["pollutant", station_col],
        values="value",
        aggfunc="mean",
    ).sort_index()
    wide.columns = [f"{str(pollutant)}__{str(station)}" for pollutant, station in wide.columns]
    return wide
