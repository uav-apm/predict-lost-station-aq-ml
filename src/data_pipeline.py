"""Data loading, preprocessing, imputation caching, and supervised-frame construction.

This module transforms raw CSV inputs into the normalised, supervised-learning
frames consumed by the model training and evaluation stages.  It sits above
``config_loader`` and ``artifact_io`` in the dependency hierarchy.

Responsibilities
----------------
- Config-driven data loading (``_load_input_dataset``).
- Missing-value strategy and normalization type resolution.
- Cache-aware imputation pipeline (``_load_or_build_imputed_dataset``).
- Supervised-frame construction via lag/horizon windowing (``_build_supervised_frame``).
- Training, validation, and cross-validation data preparation helpers.
- Period-slice utilities (``_filter_raw_df_by_periods``, ``_exclude_raw_df_by_periods``,
  ``_plot_period_df``, …).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from .artifact_io import (
    _filter_paths_by_years,
    _resolve_path,
    _save_json,
)
from .config_loader import _report_note
from .data import (
    apply_norm_stats,
    column_statistics,
    column_statistics_table,
    correlation_analysis,
    file_signature,
    imputation_stats_table,
    infer_station_id_from_path,
    load_timeseries_csv,
    load_timeseries_folder,
    missing_value_stats_by_group,
    missing_value_stats,
    normalize_train_test,
    save_correlation_heatmap,
)
from .forecasting import (
    build_period_mask,
    build_supervised_dataset,
    normalize_periods,
    resolve_testing_periods,
    resolve_validation_periods,
    select_test_rows,
    years_for_periods,
)
from .reconstruction import (
    build_station_matrix,
    required_raw_columns,
    validate_station_roles,
)


# ---------------------------------------------------------------------------
# Data-loading
# ---------------------------------------------------------------------------


def _load_input_dataset(
    cfgd: dict,
    base: Path,
    *,
    allowed_years: list[int] | None = None,
):
    """Load raw time-series data from disk according to the ``data`` config block.

    Supports three modes:
    * ``data.station_files`` — explicit per-station file lists.
    * ``data.folder`` — scan a directory for CSV files.
    * ``data.csv`` — single file.

    After loading, optional column selection and station-role validation are
    applied.

    Args:
        cfgd: Fully-resolved configuration dictionary.
        base: Directory containing the config file (used for relative path
            resolution).
        allowed_years: When non-empty, restrict loaded files to those whose
            inferred year appears in this list.

    Returns:
        A ``(DataFrame, source)`` tuple.  *source* is a plain dict with keys
        ``mode``, ``path``, and ``files`` describing where the data came from.

    Raises:
        KeyError: When the config specifies neither ``data.folder`` nor
            ``data.csv``.
        FileNotFoundError: When ``station_files`` is configured but produces
            no loadable files after year filtering.
        TypeError: When ``data.columns`` is not a list.
    """
    data_cfg = cfgd["data"]
    datetime_col = data_cfg["datetime_col"]
    sort_col = data_cfg.get("sort_col", datetime_col)
    station_col = data_cfg.get("station_id_col") or "station_id"
    station_id_from = data_cfg.get("station_id_from", "auto")
    station_id_regex = data_cfg.get("station_id_regex")

    if data_cfg.get("station_files"):
        folder = _resolve_path(base, data_cfg.get("folder", "."))
        frames: list[pd.DataFrame] = []
        files: list[str] = []
        for station_id, station_files in data_cfg["station_files"].items():
            resolved_files = _filter_paths_by_years(
                [_resolve_path(folder, file_name).as_posix() for file_name in station_files],
                allowed_years,
            )
            for raw_file in resolved_files:
                csv_path = Path(raw_file)
                df_piece = load_timeseries_csv(csv_path, datetime_col=datetime_col)
                df_piece[station_col] = station_id
                frames.append(df_piece)
                files.append(str(csv_path))
        if not frames:
            raise FileNotFoundError("data.station_files was provided but no files were loaded.")
        df = (
            pd.concat(frames, ignore_index=True)
            .sort_values(sort_col)
            .reset_index(drop=True)
        )
        source = {"mode": "station_files", "path": str(folder), "files": files}

    elif data_cfg.get("folder"):
        folder = _resolve_path(base, data_cfg["folder"])
        df, files = load_timeseries_folder(
            folder,
            datetime_col=datetime_col,
            sort_col=sort_col,
            pattern=data_cfg.get("file_pattern", "*.csv"),
            station_id_col=station_col,
            station_id_from=station_id_from,
            station_id_regex=station_id_regex,
        )
        files = _filter_paths_by_years(files, allowed_years)
        if allowed_years:
            filtered_frames = []
            for raw_file in files:
                frame = load_timeseries_csv(raw_file, datetime_col=datetime_col)
                frame[station_col] = infer_station_id_from_path(
                    raw_file,
                    mode=station_id_from,
                    regex=station_id_regex,
                    folder_root=folder,
                )
                filtered_frames.append(frame)
            df = (
                pd.concat(filtered_frames, ignore_index=True)
                .sort_values(sort_col)
                .reset_index(drop=True)
            )
        source = {"mode": "folder", "path": str(folder), "files": files}

    elif data_cfg.get("csv"):
        csv_path = _resolve_path(base, data_cfg["csv"])
        _filter_paths_by_years([str(csv_path)], allowed_years)
        df = load_timeseries_csv(csv_path, datetime_col=datetime_col)
        if station_col not in df.columns:
            df[station_col] = infer_station_id_from_path(
                csv_path,
                mode=station_id_from,
                regex=station_id_regex,
                folder_root=csv_path.parent,
            )
        df = df.sort_values(sort_col).reset_index(drop=True)
        source = {"mode": "csv", "path": str(csv_path), "files": [str(csv_path)]}

    else:
        raise KeyError("Config must provide either data.folder or data.csv.")

    selected = data_cfg.get("columns")
    if selected is not None:
        if not isinstance(selected, list):
            raise TypeError("data.columns must be a list when provided.")
        required = required_raw_columns(data_cfg, cfgd["model"])
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise KeyError(
                "Configured columns were not found in the dataset: " + ", ".join(missing)
            )
        df = df.loc[:, required].copy()

    if station_col not in df.columns:
        raise KeyError(
            "Station ids could not be resolved. Provide data.station_id_col when the column "
            "exists, or configure data.station_id_from / data.station_files for path-based "
            "inference."
        )

    if data_cfg.get("station_id_col") != station_col:
        data_cfg["station_id_col"] = station_col

    validate_station_roles(df, data_cfg)
    return df, source


# ---------------------------------------------------------------------------
# Config-value resolvers
# ---------------------------------------------------------------------------


def _missing_value_strategy(data_cfg: dict) -> str:
    """Return the missing-value strategy. Only 'drop' is supported."""
    return "drop"


def _normalization_type(data_cfg: dict) -> str:
    """Resolve the canonical normalization method from the data config.

    Accepts legacy string aliases (e.g. ``"standard"``, ``"min-max"``) and the
    deprecated boolean ``data.normalize``.

    Args:
        data_cfg: The ``data`` sub-dict of a resolved configuration.

    Returns:
        One of ``"none"``, ``"zscore"``, ``"minmax"``, ``"robust"``,
        ``"maxabs"``.

    Raises:
        ValueError: When the resolved value is not a recognised normalization
            type.
    """
    aliases = {
        "true": "zscore",
        "false": "none",
        "standard": "zscore",
        "standardize": "zscore",
        "standard_scaler": "zscore",
        "min-max": "minmax",
        "min_max": "minmax",
        "robust_scaler": "robust",
        "max-abs": "maxabs",
        "max_abs": "maxabs",
    }
    raw = data_cfg.get("normalization")
    if raw is None:
        raw = data_cfg.get("normalize", True)
    if isinstance(raw, bool):
        raw = "zscore" if raw else "none"
    normalized = str(raw).strip().lower()
    normalized = aliases.get(normalized, normalized)
    supported = {"none", "zscore", "minmax", "robust", "maxabs"}
    if normalized not in supported:
        raise ValueError(
            "data.normalization must be one of: none, zscore, minmax, robust, maxabs "
            "(or legacy boolean data.normalize)."
        )
    return normalized


def _positive_lag_count(data_cfg: dict, key: str, default: int = 1) -> int:
    """Read a strictly-positive integer lag count from the data config.

    Args:
        data_cfg: The ``data`` sub-dict of a resolved configuration.
        key: Config key to read (e.g. ``"input_lags"``).
        default: Value to use when the key is absent.

    Returns:
        Parsed integer value.

    Raises:
        ValueError: When the value is non-numeric or less than 1.
    """
    raw_value = data_cfg.get(key, default)
    try:
        value = int(raw_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"data.{key} must be a positive integer.") from exc
    if value < 1:
        raise ValueError(f"data.{key} must be >= 1.")
    return value


def _parse_float_list(raw_value: str | None, *, option_name: str) -> list[float]:
    """Parse a comma-separated string of floats from a CLI option.

    Args:
        raw_value: Raw string from a CLI argument, or ``None``.
        option_name: Human-readable option name used in error messages.

    Returns:
        List of parsed floats, or an empty list when *raw_value* is ``None`` or
        blank.

    Raises:
        ValueError: When any token in *raw_value* cannot be parsed as a float.
    """
    if raw_value is None:
        return []
    values = [item.strip() for item in str(raw_value).split(",") if item.strip()]
    if not values:
        return []
    try:
        return [float(value) for value in values]
    except ValueError as exc:
        raise ValueError(f"{option_name} must be a comma-separated list of numbers.") from exc


def _should_save_training_normalization(
    artifacts_cfg: dict | None,
    cli_override: bool | None = None,
) -> bool:
    """Return ``True`` when normalization statistics should be persisted after training.

    Args:
        artifacts_cfg: The ``artifacts`` sub-dict of a resolved configuration,
            or ``None``.
        cli_override: Explicit boolean from a ``--save-normalization`` /
            ``--no-save-normalization`` CLI flag.  Takes precedence when set.

    Returns:
        Whether to save normalization statistics.
    """
    if cli_override is not None:
        return bool(cli_override)
    if not isinstance(artifacts_cfg, dict):
        return True
    configured = artifacts_cfg.get("save_training_normalization")
    if configured is None:
        return True
    return bool(configured)


def _should_use_training_normalization(
    eval_cfg: dict | None,
    cli_override: bool | None = None,
) -> bool:
    """Return ``True`` when evaluation should reuse training normalization statistics.

    Args:
        eval_cfg: The ``eval`` sub-dict of a resolved configuration, or ``None``.
        cli_override: Explicit boolean from a ``--use-training-normalization`` /
            ``--no-use-training-normalization`` CLI flag.  Takes precedence when
            set.

    Returns:
        Whether to apply saved training normalization during evaluation.
    """
    if cli_override is not None:
        return bool(cli_override)
    if not isinstance(eval_cfg, dict):
        return True
    configured = eval_cfg.get("use_training_normalization")
    if configured is None:
        return True
    return bool(configured)


# ---------------------------------------------------------------------------
# Imputation cache helpers
# ---------------------------------------------------------------------------


def _cache_root_for_source(source: dict) -> Path:
    """Return the ``.aqsr_cache`` directory for the given data source.

    Args:
        source: Source descriptor dict with ``mode`` and ``path`` keys.

    Returns:
        Path to the cache root directory (not necessarily created yet).
    """
    source_mode = source["mode"]
    base = Path(source["path"])
    if source_mode in {"folder", "station_files"}:
        return base / ".aqsr_cache"
    return base.parent / ".aqsr_cache"


def _imputation_cache_signature(cfgd: dict, source: dict) -> dict:
    """Build a stable JSON-serialisable cache key for the imputation pipeline.

    The signature captures every input that would change the preprocessed output
    so that stale caches are automatically invalidated.

    Args:
        cfgd: Fully-resolved configuration dictionary.
        source: Source descriptor dict returned by :func:`_load_input_dataset`.

    Returns:
        A plain dict suitable for JSON serialisation and hashing.
    """
    data_cfg = cfgd["data"]
    return {
        "cache_format": 4,
        "files": file_signature(source["files"]),
        "datetime_col": data_cfg["datetime_col"],
        "sort_col": data_cfg.get("sort_col", data_cfg["datetime_col"]),
        "columns": data_cfg.get("columns"),
        "station_id_col": data_cfg.get("station_id_col") or "station_id",
        "station_id_from": data_cfg.get("station_id_from", "auto"),
        "station_id_regex": data_cfg.get("station_id_regex"),
        "missing_value_strategy": _missing_value_strategy(data_cfg),
        "drop_missing_values": bool(data_cfg.get("drop_missing_values", False)),
        "impute_columns": data_cfg.get("impute_columns") or [],
    }


def _cache_dir_for_signature(source: dict, signature: dict) -> Path:
    """Derive the deterministic cache directory path from *signature*.

    Args:
        source: Source descriptor dict with ``path`` and ``mode`` keys.
        signature: Dict produced by :func:`_imputation_cache_signature`.

    Returns:
        Path to the cache subdirectory for this exact signature.
    """
    cache_root = _cache_root_for_source(source)
    digest = hashlib.sha1(
        json.dumps(signature, sort_keys=True).encode("utf-8")
    ).hexdigest()[:12]
    return cache_root / digest


# ---------------------------------------------------------------------------
# Period slice utilities
# ---------------------------------------------------------------------------


def _periods_to_serializable(periods) -> list[dict]:
    """Convert a periods list to a JSON-safe list of ISO-8601 dicts.

    Args:
        periods: Value acceptable by :func:`~src.forecasting.normalize_periods`.

    Returns:
        List of ``{"start": …, "end": …}`` dicts with ISO-8601 strings.
    """
    return [
        {"start": period["start"].isoformat(), "end": period["end"].isoformat()}
        for period in normalize_periods(periods)
    ]


def _raw_datetime_mask(df: pd.DataFrame, cfgd: dict, periods) -> pd.Series:
    """Build a boolean mask selecting rows whose datetime falls in *periods*.

    Args:
        df: Raw dataset containing the datetime column.
        cfgd: Resolved configuration dictionary.
        periods: Normalized list of period dicts.

    Returns:
        Boolean :class:`pandas.Series` aligned with *df*.
    """
    return build_period_mask(df[cfgd["data"]["datetime_col"]], periods)


def _filter_raw_df_by_periods(df: pd.DataFrame, cfgd: dict, periods) -> pd.DataFrame:
    """Return a copy of *df* retaining only rows that fall within *periods*.

    Args:
        df: Raw dataset.
        cfgd: Resolved configuration dictionary.
        periods: Value acceptable by :func:`~src.forecasting.normalize_periods`.

    Returns:
        Filtered, sorted copy of *df*.
    """
    normalized = normalize_periods(periods)
    if not normalized:
        return df.copy()
    mask = _raw_datetime_mask(df, cfgd, normalized)
    return (
        df.loc[mask]
        .sort_values(cfgd["data"].get("sort_col", cfgd["data"]["datetime_col"]))
        .reset_index(drop=True)
    )


def _exclude_raw_df_by_periods(df: pd.DataFrame, cfgd: dict, periods) -> pd.DataFrame:
    """Return a copy of *df* excluding rows that fall within *periods*.

    Args:
        df: Raw dataset.
        cfgd: Resolved configuration dictionary.
        periods: Value acceptable by :func:`~src.forecasting.normalize_periods`.

    Returns:
        Filtered, sorted copy of *df*.
    """
    normalized = normalize_periods(periods)
    if not normalized:
        return df.copy()
    mask = _raw_datetime_mask(df, cfgd, normalized)
    return (
        df.loc[~mask]
        .sort_values(cfgd["data"].get("sort_col", cfgd["data"]["datetime_col"]))
        .reset_index(drop=True)
    )


def _extract_validation_years(data_cfg: dict) -> list[int]:
    """Extract the set of calendar years covered by the configured testing periods.

    Args:
        data_cfg: The ``data`` sub-dict of a resolved configuration.

    Returns:
        Sorted list of integer years.
    """
    periods = resolve_testing_periods(data_cfg)
    return years_for_periods(periods)


def _plot_period_df(df: pd.DataFrame, datetime_col: str, period_cfg) -> pd.DataFrame:
    """Filter *df* to rows within *period_cfg* for plot-slice generation.

    Args:
        df: Source dataframe.
        datetime_col: Name of the datetime column in *df*.
        period_cfg: Period configuration value.

    Returns:
        Filtered copy of *df*, or the full *df* if *period_cfg* is empty.
    """
    periods = normalize_periods(period_cfg)
    if not periods:
        return df.copy()
    mask = build_period_mask(df[datetime_col], periods)
    return df.loc[mask].reset_index(drop=True)


def _period_plot_suffix(period: dict, index: int) -> str:
    """Build a compact, filesystem-safe suffix string for period-scoped plots.

    Args:
        period: Period dict with ``"start"`` and ``"end"`` keys.
        index: One-based period index used as a sort prefix.

    Returns:
        A string like ``"01_20220101T000000_20221231T235959"``.
    """
    start = pd.Timestamp(period["start"]).strftime("%Y%m%dT%H%M%S")
    end = pd.Timestamp(period["end"]).strftime("%Y%m%dT%H%M%S")
    return f"{index:02d}_{start}_{end}"


def _period_display_label(period: dict, index: int) -> str:
    """Build a human-readable display label for a validation period.

    Args:
        period: Period dict with ``"start"`` and ``"end"`` keys.
        index: One-based period index.

    Returns:
        A string like ``"Period 1: 2022-01-01 to 2022-12-31"``.
    """
    start = pd.Timestamp(period["start"]).strftime("%Y-%m-%d")
    end = pd.Timestamp(period["end"]).strftime("%Y-%m-%d")
    return f"Period {index}: {start} to {end}"


def _validation_period_columns(
    period: dict | None,
    period_name: str | None = None,
) -> dict[str, str | None]:
    """Build a flat dict of period-metadata columns for comparison CSV rows.

    Args:
        period: Period dict with ``"start"`` / ``"end"`` keys, or ``None``.
        period_name: Human-readable name for the period.

    Returns:
        Dict with keys ``validation_period_name``, ``validation_period_start``,
        ``validation_period_end``, ``validation_period``.
    """
    if not isinstance(period, dict):
        return {
            "validation_period_name": period_name,
            "validation_period_start": None,
            "validation_period_end": None,
            "validation_period": None,
        }

    start = period.get("start")
    end = period.get("end")
    label = f"{start} -> {end}" if start and end else None
    return {
        "validation_period_name": period_name,
        "validation_period_start": start,
        "validation_period_end": end,
        "validation_period": label,
    }


# ---------------------------------------------------------------------------
# Imputation pipeline
# ---------------------------------------------------------------------------


def _export_imputed_view(cfgd: dict, df: pd.DataFrame) -> pd.DataFrame:
    """Limit persisted imputed datasets to configured columns plus required structural fields.

    Args:
        cfgd: Resolved configuration dictionary.
        df: Imputed dataframe.

    Returns:
        A copy of *df* restricted to the columns that are actually needed.
    """
    keep_columns = [
        col
        for col in required_raw_columns(cfgd["data"], cfgd["model"])
        if col in df.columns
    ]
    if not keep_columns:
        return df.copy()
    return df.loc[:, keep_columns].copy()


def _drop_rows_with_missing_values(
    df: pd.DataFrame,
    *,
    target_columns: list[str],
    group_col: str | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Remove rows where any of *target_columns* contains a missing value.

    Args:
        df: Input dataframe.
        target_columns: Columns to check for missing values.
        group_col: Optional station-ID column.  When provided, per-station
            statistics are included in the returned details dict.

    Returns:
        ``(cleaned_df, details)`` where *details* is a dict describing rows
        dropped per column (and, when *group_col* is given, per station).
    """
    available_columns = [column for column in target_columns if column in df.columns]
    if not available_columns:
        details: dict = {
            "method": "drop_rows",
            "dropped_rows": 0,
            "columns": {},
            "target_columns": [],
        }
        if group_col and group_col in df.columns:
            details["group_col"] = group_col
            details["by_station"] = {}
        return df.copy(), details

    missing_mask = df[available_columns].isna().any(axis=1)
    dropped_rows = int(missing_mask.sum())
    out = df.loc[~missing_mask].copy().reset_index(drop=True)

    details = {
        "method": "drop_rows",
        "dropped_rows": dropped_rows,
        "target_columns": available_columns,
        "columns": {},
    }
    for column in available_columns:
        details["columns"][column] = {
            "method": "drop_rows",
            "selected_features": [],
            "missing_before": int(df[column].isna().sum()),
            "missing_after": int(out[column].isna().sum()),
        }

    if group_col and group_col in df.columns:
        station_before = df[group_col].astype(str)
        details["group_col"] = group_col
        details["by_station"] = {}

        for station_id in sorted(station_before.unique()):
            station_before_mask = station_before == station_id
            station_before_df = df.loc[station_before_mask]
            station_after_df = out.loc[out[group_col].astype(str) == station_id]
            station_details: dict = {"columns": {}}
            for column in available_columns:
                station_details["columns"][column] = {
                    "method": "drop_rows",
                    "selected_features": [],
                    "missing_before": int(station_before_df[column].isna().sum()),
                    "missing_after": int(station_after_df[column].isna().sum()),
                }
            details["by_station"][station_id] = station_details

    return out, details


def _load_or_build_imputed_dataset(
    cfgd: dict,
    raw_df: pd.DataFrame,
    source: dict,
    *,
    cache_context: dict | None = None,
    include_visual_artifacts: bool = True,
) -> dict:
    """Return a preprocessed dataset, reading from cache when valid.

    Either loads an existing imputed dataset from the ```.aqsr_cache```
    directory (when the SHA-1 cache key matches and ``data.rebuild_cache``
    is not set) or runs the full imputation/drop pipeline and writes a fresh
    cache.

    Args:
        cfgd: Fully-resolved configuration dictionary.
        raw_df: Raw dataframe as returned by :func:`_load_input_dataset`.
        source: Source descriptor dict from :func:`_load_input_dataset`.
        cache_context: Optional extra dict merged into the cache signature to
            distinguish train vs eval slices.
        include_visual_artifacts: When ``True``, generate the correlation
            heatmap PNG alongside the cached CSV artefacts.

    Returns:
        A dict with keys: ``raw_df``, ``imputed_df``, ``imputation_details``,
        ``stats_before``, ``stats_after``, ``stats_before_by_station``,
        ``stats_after_by_station``, ``column_stats_before``,
        ``column_stats_after``, ``correlation_analysis``, ``cache_dir``,
        ``cache_paths``, ``used_cache``.
    """
    exclude = [
        cfgd["data"]["datetime_col"],
        cfgd["data"].get("station_id_col") or "station_id",
    ]
    station_col = cfgd["data"].get("station_id_col") or "station_id"
    raw_export_df = _export_imputed_view(cfgd, raw_df)
    stats_before = missing_value_stats(raw_export_df, exclude=exclude)
    stats_before_by_station = missing_value_stats_by_group(
        raw_export_df,
        group_col=station_col,
        exclude=exclude,
    )
    column_stats_before = column_statistics(raw_export_df, exclude=exclude)

    rebuild_cache = bool(cfgd["data"].get("rebuild_cache", False))
    cache_signature = _imputation_cache_signature(cfgd, source)
    if cache_context:
        cache_signature["slice"] = cache_context
    cache_dir = _cache_dir_for_signature(source, cache_signature)
    cache_dir.mkdir(parents=True, exist_ok=True)

    cache_meta_path = cache_dir / "cache_metadata.json"
    cache_data_path = cache_dir / "imputed_dataset.csv"
    cache_details_path = cache_dir / "imputation_details.json"
    cache_stats_path = cache_dir / "imputation_stats.json"
    cache_stats_csv_path = cache_dir / "imputation_stats.csv"
    cache_analysis_path = cache_dir / "correlation_analysis.json"
    cache_heatmap_path = cache_dir / "correlation_heatmap.png"
    cache_colstats_before_csv = cache_dir / "column_statistics_before.csv"
    cache_colstats_after_csv = cache_dir / "column_statistics_after.csv"

    all_cache_present = (
        cache_meta_path.exists()
        and cache_data_path.exists()
        and cache_details_path.exists()
        and cache_stats_path.exists()
        and cache_analysis_path.exists()
    )

    if not rebuild_cache and all_cache_present:
        with cache_meta_path.open("r", encoding="utf-8") as handle:
            cache_meta = json.load(handle)
        if cache_meta.get("signature") == cache_signature:
            _report_note(f"Reusing cached preprocessing from {cache_dir}")
            imputed_df = load_timeseries_csv(
                cache_data_path, datetime_col=cfgd["data"]["datetime_col"]
            )
            if (cfgd["data"].get("station_id_col") or "station_id") not in imputed_df.columns:
                imputed_df = pd.read_csv(
                    cache_data_path, na_values=["NA", ""], keep_default_na=True
                )
                imputed_df[cfgd["data"]["datetime_col"]] = pd.to_datetime(
                    imputed_df[cfgd["data"]["datetime_col"]]
                )
            with cache_details_path.open("r", encoding="utf-8") as handle:
                imputation_details = json.load(handle)
            with cache_stats_path.open("r", encoding="utf-8") as handle:
                stats_payload = json.load(handle)
            with cache_analysis_path.open("r", encoding="utf-8") as handle:
                analysis = json.load(handle)
            if include_visual_artifacts and not cache_heatmap_path.exists():
                _report_note(
                    f"Cached correlation heatmap missing. Rebuilding {cache_heatmap_path}"
                )
                wrote_heatmap = save_correlation_heatmap(analysis, cache_heatmap_path)
                if wrote_heatmap:
                    _report_note(f"Correlation heatmap written to {cache_heatmap_path}")
                else:
                    _report_note(
                        "Correlation heatmap was not written because the analysis had no "
                        "numeric matrix."
                    )
            column_stats_after = column_statistics(imputed_df, exclude=exclude)
            sort_key = cfgd["data"].get("sort_col", cfgd["data"]["datetime_col"])
            return {
                "raw_df": raw_export_df,
                "imputed_df": imputed_df.sort_values(sort_key).reset_index(drop=True),
                "imputation_details": imputation_details,
                "stats_before": stats_payload["before"],
                "stats_after": stats_payload["after"],
                "stats_before_by_station": stats_payload.get("before_by_station", {}),
                "stats_after_by_station": stats_payload.get("after_by_station", {}),
                "column_stats_before": column_stats_before,
                "column_stats_after": column_stats_after,
                "correlation_analysis": analysis,
                "cache_dir": cache_dir,
                "cache_paths": {
                    "dataset": cache_data_path,
                    "details": cache_details_path,
                    "stats": cache_stats_path,
                    "stats_csv": cache_stats_csv_path,
                    "analysis": cache_analysis_path,
                    "heatmap": cache_heatmap_path,
                    "column_stats_before_csv": cache_colstats_before_csv,
                    "column_stats_after_csv": cache_colstats_after_csv,
                },
                "used_cache": True,
            }

    elif rebuild_cache:
        _report_note(
            f"Rebuilding preprocessing cache because data.rebuild_cache=true: {cache_dir}"
        )

    impute_columns = list(cfgd["data"].get("impute_columns") or [])
    imputed_df, imputation_details = _drop_rows_with_missing_values(
        raw_df,
        target_columns=impute_columns,
        group_col=station_col,
    )
    analysis = correlation_analysis(imputed_df, exclude=[*exclude, station_col])
    _report_note(
        f"Built fresh preprocessing cache in {cache_dir} by dropping rows with missing "
        f"values in configured columns. "
        f"Dropped rows: {imputation_details.get('dropped_rows', 0)}"
    )

    export_df = _export_imputed_view(cfgd, imputed_df)
    stats_after = missing_value_stats(export_df, exclude=exclude)
    stats_after_by_station = missing_value_stats_by_group(
        export_df,
        group_col=station_col,
        exclude=exclude,
    )
    column_stats_after = column_statistics(export_df, exclude=exclude)

    export_df.to_csv(cache_data_path, index=False)
    _save_json(cache_details_path, imputation_details)
    _save_json(
        cache_stats_path,
        {
            "before": stats_before,
            "after": stats_after,
            "before_by_station": stats_before_by_station,
            "after_by_station": stats_after_by_station,
        },
    )
    imputation_stats_table(
        stats_before,
        stats_after,
        imputation_details,
        before_by_group=stats_before_by_station,
        after_by_group=stats_after_by_station,
    ).to_csv(cache_stats_csv_path, index=False)
    _save_json(cache_analysis_path, analysis)
    column_statistics_table(column_stats_before, "before").to_csv(
        cache_colstats_before_csv, index=False
    )
    column_statistics_table(column_stats_after, "after").to_csv(
        cache_colstats_after_csv, index=False
    )
    if include_visual_artifacts:
        _report_note(f"Writing correlation heatmap to {cache_heatmap_path}")
        wrote_heatmap = save_correlation_heatmap(analysis, cache_heatmap_path)
        if wrote_heatmap:
            _report_note(f"Correlation heatmap written to {cache_heatmap_path}")
        else:
            _report_note(
                "Correlation heatmap was not written because the analysis had no numeric matrix."
            )
    if not imputation_details.get("columns"):
        _report_note("No configured columns required imputation for this dataset.")
    _save_json(cache_meta_path, {"signature": cache_signature})

    return {
        "raw_df": raw_export_df,
        "imputed_df": export_df,
        "imputation_details": imputation_details,
        "stats_before": stats_before,
        "stats_after": stats_after,
        "stats_before_by_station": stats_before_by_station,
        "stats_after_by_station": stats_after_by_station,
        "column_stats_before": column_stats_before,
        "column_stats_after": column_stats_after,
        "correlation_analysis": analysis,
        "cache_dir": cache_dir,
        "cache_paths": {
            "dataset": cache_data_path,
            "details": cache_details_path,
            "stats": cache_stats_path,
            "stats_csv": cache_stats_csv_path,
            "analysis": cache_analysis_path,
            "heatmap": cache_heatmap_path,
            "column_stats_before_csv": cache_colstats_before_csv,
            "column_stats_after_csv": cache_colstats_after_csv,
        },
        "used_cache": False,
    }


# ---------------------------------------------------------------------------
# Supervised-frame construction
# ---------------------------------------------------------------------------


def _build_supervised_frame(
    cfgd: dict,
    raw_df: pd.DataFrame,
    source: dict,
    *,
    cache_context: dict | None = None,
    include_visual_artifacts: bool = True,
) -> dict:
    """Run imputation then build a lag/horizon supervised DataFrame.

    Combines :func:`_load_or_build_imputed_dataset` with
    :func:`~src.reconstruction.build_station_matrix` and
    :func:`~src.forecasting.build_supervised_dataset`.

    Args:
        cfgd: Resolved configuration dictionary.
        raw_df: Raw dataframe from :func:`_load_input_dataset`.
        source: Source descriptor dict.
        cache_context: Extra dict merged into the cache key.
        include_visual_artifacts: Passed through to
            :func:`_load_or_build_imputed_dataset`.

    Returns:
        All keys from :func:`_load_or_build_imputed_dataset` plus
        ``station_matrix``, ``supervised_df``, ``feature_names``,
        ``target_names``.
    """
    preprocessed = _load_or_build_imputed_dataset(
        cfgd,
        raw_df,
        source,
        cache_context=cache_context,
        include_visual_artifacts=include_visual_artifacts,
    )
    station_matrix = build_station_matrix(preprocessed["imputed_df"], cfgd["data"])
    supervised_df, feature_names, target_names = build_supervised_dataset(
        station_matrix,
        cfgd["data"],
        horizon=int(cfgd["eval"].get("horizon", 1)),
    )
    return {
        **preprocessed,
        "station_matrix": station_matrix,
        "supervised_df": supervised_df,
        "feature_names": feature_names,
        "target_names": target_names,
    }


# ---------------------------------------------------------------------------
# Training / evaluation data preparation
# ---------------------------------------------------------------------------


def _prepare_training_data(cfgd: dict, base: Path) -> dict:
    """Load, preprocess, and normalise supervised rows for model training.

    Handles training-period filtering, excluding validation and testing
    periods from the training set, and computing normalization statistics.

    Args:
        cfgd: Resolved configuration dictionary.
        base: Config file directory for relative path resolution.

    Returns:
        Dict extending :func:`_build_supervised_frame` output with keys
        ``data_source``, ``train_rows``, ``train_X_raw``, ``train_X``,
        ``train_y``, ``norm_stats``.
    """
    data_cfg = cfgd["data"]
    train_periods = normalize_periods(data_cfg.get("train_periods"))
    validation_periods = resolve_validation_periods(data_cfg)
    testing_periods = resolve_testing_periods(data_cfg)
    train_years = [int(year) for year in cfgd["data"].get("train_years") or []]
    allowed_years = years_for_periods(train_periods) or train_years or None
    raw_df, data_source = _load_input_dataset(cfgd, base, allowed_years=allowed_years)
    if train_periods:
        raw_df = _filter_raw_df_by_periods(raw_df, cfgd, train_periods)
    else:
        excluded_periods = [*validation_periods, *testing_periods]
        if excluded_periods:
            raw_df = _exclude_raw_df_by_periods(raw_df, cfgd, excluded_periods)

    prepared = _build_supervised_frame(
        cfgd,
        raw_df,
        data_source,
        cache_context={
            "mode": "train",
            "train_years": train_years,
            "train_periods": _periods_to_serializable(train_periods),
            "validation_periods_excluded": _periods_to_serializable(
                validation_periods if not train_periods else []
            ),
            "testing_periods_excluded": _periods_to_serializable(
                testing_periods if not train_periods else []
            ),
        },
    )
    train_rows = prepared["supervised_df"].reset_index(drop=True)
    train_X_raw = train_rows[prepared["feature_names"]].copy()
    normalization_method = _normalization_type(cfgd["data"])
    train_X, _, norm_stats = normalize_train_test(
        train_X_raw,
        train_X_raw,
        method=normalization_method,
    )
    target_names = prepared["target_names"]
    train_y = (
        train_rows[target_names[0]].copy()
        if len(target_names) == 1
        else train_rows[target_names].copy()
    )
    return {
        **prepared,
        "data_source": data_source,
        "train_rows": train_rows,
        "train_X_raw": train_X_raw,
        "train_X": train_X,
        "train_y": train_y,
        "norm_stats": norm_stats,
    }


def _prepare_validation_data_for_tuning(
    cfgd: dict,
    base: Path,
    norm_stats: dict,
) -> dict | None:
    """Load and normalise supervised rows for hyperparameter-tuning validation.

    Returns ``None`` when no ``data.validation_periods`` are configured.

    Args:
        cfgd: Resolved configuration dictionary.
        base: Config file directory.
        norm_stats: Normalization statistics from the training fold.

    Returns:
        Dict with keys ``validation_periods``, ``validation_rows``,
        ``validation_X_raw``, ``validation_X``, ``validation_y``, or
        ``None``.
    """
    validation_periods = resolve_validation_periods(cfgd["data"])
    if not validation_periods:
        return None

    allowed_years = years_for_periods(validation_periods) or None
    raw_df, data_source = _load_input_dataset(cfgd, base, allowed_years=allowed_years)
    prepared = _build_supervised_frame(
        cfgd,
        raw_df,
        data_source,
        cache_context={
            "mode": "train_validation_tuning",
            "validation_periods": _periods_to_serializable(validation_periods),
        },
        include_visual_artifacts=False,
    )

    prediction_time = pd.to_datetime(prepared["supervised_df"]["prediction_time"])
    tuning_mask = build_period_mask(prediction_time, validation_periods)
    validation_rows = prepared["supervised_df"].loc[tuning_mask].reset_index(drop=True)

    validation_X_raw = validation_rows[prepared["feature_names"]].copy()
    validation_X = apply_norm_stats(validation_X_raw, norm_stats)
    target_names = prepared["target_names"]
    validation_y = (
        validation_rows[target_names[0]].copy()
        if len(target_names) == 1
        else validation_rows[target_names].copy()
    )

    return {
        "validation_periods": validation_periods,
        "validation_rows": validation_rows,
        "validation_X_raw": validation_X_raw,
        "validation_X": validation_X,
        "validation_y": validation_y,
    }


def _prepare_evaluation_data(cfgd: dict, base: Path, norm_stats: dict | None) -> dict:
    """Load and normalise supervised rows for evaluation / inference.

    Args:
        cfgd: Resolved configuration dictionary.
        base: Config file directory.
        norm_stats: Training normalization statistics, or ``None`` to re-compute
            from the evaluation data itself.

    Returns:
        Dict extending :func:`_build_supervised_frame` output with keys
        ``data_source``, ``test_rows``, ``test_X_raw``, ``test_X``,
        ``test_y``, ``norm_stats``, ``normalization_source``.

    Raises:
        ValueError: When the period/step selection produces zero evaluation
            rows.
    """
    testing_periods = resolve_testing_periods(cfgd["data"])
    test_years = _extract_validation_years(cfgd["data"])
    raw_df, data_source = _load_input_dataset(cfgd, base, allowed_years=test_years or None)
    prepared = _build_supervised_frame(
        cfgd,
        raw_df,
        data_source,
        cache_context={
            "mode": "eval",
            "testing_periods": _periods_to_serializable(testing_periods),
        },
        include_visual_artifacts=False,
    )
    test_mask = select_test_rows(prepared["supervised_df"], cfgd["data"])
    test_rows = prepared["supervised_df"].loc[test_mask].reset_index(drop=True)

    if len(test_rows) == 0:
        prediction_times = pd.to_datetime(prepared["supervised_df"]["prediction_time"])
        available_days = sorted(
            prediction_times.dt.strftime("%Y-%m-%d").unique().tolist()
        )
        preview = available_days[:10]
        suffix = "" if len(available_days) <= 10 else f" ... ({len(available_days)} total)"
        _report_note(
            "Configured testing_periods/test_time_steps selection produced no evaluation rows; "
            "falling back to all available supervised rows for evaluation. "
            f"Requested testing_periods={cfgd['data'].get('testing_periods') or cfgd['data'].get('validation_periods') or cfgd['data'].get('test_days')}. "
            f"Available prediction days include: {preview}{suffix}"
        )
        test_rows = prepared["supervised_df"].reset_index(drop=True)

    test_X_raw = test_rows[prepared["feature_names"]].copy()
    if norm_stats is None:
        normalization_method = _normalization_type(cfgd["data"])
        _, test_X, effective_norm_stats = normalize_train_test(
            test_X_raw,
            test_X_raw,
            method=normalization_method,
        )
        normalization_source = "evaluation_data"
    else:
        test_X = apply_norm_stats(test_X_raw, norm_stats)
        effective_norm_stats = norm_stats
        normalization_source = "training_run"

    target_names = prepared["target_names"]
    test_y = (
        test_rows[target_names[0]].copy()
        if len(target_names) == 1
        else test_rows[target_names].copy()
    )
    return {
        **prepared,
        "data_source": data_source,
        "test_rows": test_rows,
        "test_X_raw": test_X_raw,
        "test_X": test_X,
        "test_y": test_y,
        "norm_stats": effective_norm_stats,
        "normalization_source": normalization_source,
    }


def _prepare_fold_training_inputs(
    cfgd: dict,
    base: Path,
    train_periods,
    validation_periods,
    fold_name: str,
) -> dict:
    """Prepare training inputs for one cross-validation fold.

    Args:
        cfgd: Resolved configuration dictionary.
        base: Config file directory.
        train_periods: Training periods for this fold.
        validation_periods: Validation periods for this fold (used only for
            cache-key tagging).
        fold_name: Human-readable fold identifier (e.g. ``"fold_1"``).

    Returns:
        Dict with ``data_source``, ``train_rows``, ``train_X``, ``train_y``,
        ``norm_stats`` and all keys from :func:`_build_supervised_frame`.
    """
    allowed_years = years_for_periods(train_periods) or None
    raw_df, data_source = _load_input_dataset(cfgd, base, allowed_years=allowed_years)
    train_raw_df = _filter_raw_df_by_periods(raw_df, cfgd, train_periods)
    prepared = _build_supervised_frame(
        cfgd,
        train_raw_df,
        data_source,
        cache_context={
            "mode": "cross_validation_train",
            "fold": fold_name,
            "train_periods": _periods_to_serializable(train_periods),
            "validation_periods": _periods_to_serializable(validation_periods),
        },
        include_visual_artifacts=False,
    )
    train_rows = prepared["supervised_df"].reset_index(drop=True)
    train_X_raw = train_rows[prepared["feature_names"]].copy()
    normalization_method = _normalization_type(cfgd["data"])
    train_X, _, norm_stats = normalize_train_test(
        train_X_raw,
        train_X_raw,
        method=normalization_method,
    )
    return {
        **prepared,
        "data_source": data_source,
        "train_rows": train_rows,
        "train_X": train_X,
        "train_y": (
            train_rows[prepared["target_names"][0]].copy()
            if len(prepared["target_names"]) == 1
            else train_rows[prepared["target_names"]].copy()
        ),
        "norm_stats": norm_stats,
    }


def _prepare_fold_validation_inputs(
    cfgd: dict,
    base: Path,
    norm_stats: dict,
    validation_periods,
    fold_name: str,
) -> dict:
    """Prepare validation inputs for one cross-validation fold.

    Args:
        cfgd: Resolved configuration dictionary.
        base: Config file directory.
        norm_stats: Normalization statistics from the training fold.
        validation_periods: Validation periods for this fold.
        fold_name: Human-readable fold identifier.

    Returns:
        Dict with ``data_source``, ``test_rows``, ``test_X``, ``test_y`` and
        all keys from :func:`_build_supervised_frame`.
    """
    allowed_years = years_for_periods(validation_periods) or None
    raw_df, data_source = _load_input_dataset(cfgd, base, allowed_years=allowed_years)
    prepared = _build_supervised_frame(
        cfgd,
        raw_df,
        data_source,
        cache_context={
            "mode": "cross_validation_eval",
            "fold": fold_name,
            "validation_periods": _periods_to_serializable(validation_periods),
        },
        include_visual_artifacts=False,
    )
    fold_data_cfg = dict(cfgd["data"])
    fold_data_cfg["validation_periods"] = _periods_to_serializable(validation_periods)
    test_mask = select_test_rows(prepared["supervised_df"], fold_data_cfg)
    test_rows = prepared["supervised_df"].loc[test_mask].reset_index(drop=True)
    test_X_raw = test_rows[prepared["feature_names"]].copy()
    test_X = apply_norm_stats(test_X_raw, norm_stats)
    return {
        **prepared,
        "data_source": data_source,
        "test_rows": test_rows,
        "test_X": test_X,
        "test_y": (
            test_rows[prepared["target_names"][0]].copy()
            if len(prepared["target_names"]) == 1
            else test_rows[prepared["target_names"]].copy()
        ),
    }
