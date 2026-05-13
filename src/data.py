"""Data loading and preprocessing helpers."""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor


def _clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(col).strip() for col in out.columns]
    return out


def _match_column(columns: list[str], name: str) -> str:
    lowered = {col.lower(): col for col in columns}
    key = name.strip().lower()
    if key not in lowered:
        raise KeyError(f"Expected column '{name}' in dataset columns: {columns}")
    return lowered[key]


def load_timeseries_csv(path, datetime_col: str) -> pd.DataFrame:
    """Load one CSV, normalize the datetime column name, and sort rows."""

    df = pd.read_csv(path, na_values=["NA", ""], keep_default_na=True)
    df = _clean_columns(df)
    matched = _match_column(list(df.columns), datetime_col)
    if matched != datetime_col:
        df = df.rename(columns={matched: datetime_col})
    df[datetime_col] = pd.to_datetime(df[datetime_col])
    return df.sort_values(datetime_col).reset_index(drop=True)


def infer_station_id_from_path(
    path,
    *,
    mode: str = "auto",
    regex: str | None = None,
    folder_root=None,
) -> str:
    """Infer a station id from a file path using filename or folder names."""

    file_path = Path(path)
    folder_root_path = Path(folder_root) if folder_root is not None else None

    if regex:
        match = re.search(regex, str(file_path))
        if match:
            return match.group(1) if match.groups() else match.group(0)

    def _from_filename() -> str:
        match = re.search(r"(?:^|[_-])site[_-]?([A-Za-z0-9]+)(?:[_-]|$)", file_path.stem, flags=re.IGNORECASE)
        if match:
            return match.group(1)
        numeric_match = re.search(r"(?<!\d)(\d{3,})(?!\d)", file_path.stem)
        if numeric_match:
            return numeric_match.group(1)
        return file_path.stem

    def _from_folder() -> str:
        if folder_root_path is not None:
            relative = file_path.parent.relative_to(folder_root_path)
            parts = [part for part in relative.parts if part not in {"", "."}]
            if parts:
                return parts[0]
        return file_path.parent.name or file_path.stem

    normalized = (mode or "auto").lower()
    if normalized == "filename":
        return _from_filename()
    if normalized == "folder":
        return _from_folder()
    if normalized == "auto":
        if folder_root_path is not None and file_path.parent != folder_root_path:
            return _from_folder()
        return _from_filename()
    raise ValueError("station_id_from must be one of: auto, filename, folder")


def infer_year_from_path(path) -> int | None:
    """Infer a 4-digit year from a file path when present."""

    file_path = Path(path)
    match = re.search(r"(?<!\d)((?:19|20)\d{2})(?!\d)", str(file_path))
    if not match:
        return None
    return int(match.group(1))


def load_timeseries_folder(
    folder,
    datetime_col: str,
    sort_col: str,
    pattern: str = "*.csv",
    *,
    station_id_col: str | None = None,
    station_id_from: str = "auto",
    station_id_regex: str | None = None,
):
    """Load and combine matching CSV files from a profile data folder."""

    folder_path = Path(folder)
    files = sorted(
        path
        for path in folder_path.glob(pattern)
        if path.is_file() and ".aqsr_cache" not in path.parts and "__pycache__" not in path.parts
    )
    if not files:
        raise FileNotFoundError(f"No CSV files matched '{pattern}' in {folder_path}")

    frames: list[pd.DataFrame] = []
    for path in files:
        frame = load_timeseries_csv(path, datetime_col=datetime_col)
        if station_id_col:
            if station_id_col in frame.columns:
                frame[station_id_col] = frame[station_id_col].astype(str)
            else:
                frame[station_id_col] = infer_station_id_from_path(
                    path,
                    mode=station_id_from,
                    regex=station_id_regex,
                    folder_root=folder_path,
                )
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    if sort_col not in combined.columns:
        raise KeyError(f"Sort column '{sort_col}' was not found in combined dataset.")
    combined = combined.sort_values(sort_col).reset_index(drop=True)
    return combined, [str(path) for path in files]


def identity_norm_stats(df: pd.DataFrame, exclude: list[str] | None = None) -> dict:
    """Return a pass-through normalization map for all numeric columns."""

    excluded = set(exclude or [])
    numeric_cols = [col for col in df.columns if col not in excluded and pd.api.types.is_numeric_dtype(df[col])]
    return {
        "method": "none",
        "mu": {col: 0.0 for col in numeric_cols},
        "sigma": {col: 1.0 for col in numeric_cols},
    }


def normalize_train_test(
    train: pd.DataFrame,
    test: pd.DataFrame,
    exclude: list[str] | None = None,
    *,
    method: str = "zscore",
):
    """Normalize numeric columns using train-set statistics for the selected method."""

    normalized_method = (method or "zscore").strip().lower()
    if normalized_method == "none":
        return train.copy(), test.copy(), identity_norm_stats(train, exclude=exclude)

    excluded = set(exclude or [])
    train_n = train.copy()
    test_n = test.copy()

    if normalized_method == "zscore":
        mu: dict[str, float] = {}
        sigma: dict[str, float] = {}
        for col in train.columns:
            if col in excluded or not pd.api.types.is_numeric_dtype(train[col]):
                continue
            col_mu = float(train[col].mean())
            col_sigma = float(train[col].std(ddof=0))
            if col_sigma == 0.0:
                col_sigma = 1.0
            mu[col] = col_mu
            sigma[col] = col_sigma
            train_n[col] = (train[col] - col_mu) / col_sigma
            test_n[col] = (test[col] - col_mu) / col_sigma
        return train_n, test_n, {"method": "zscore", "mu": mu, "sigma": sigma}

    if normalized_method == "minmax":
        mins: dict[str, float] = {}
        maxs: dict[str, float] = {}
        for col in train.columns:
            if col in excluded or not pd.api.types.is_numeric_dtype(train[col]):
                continue
            col_min = float(train[col].min())
            col_max = float(train[col].max())
            col_range = col_max - col_min
            if col_range == 0.0:
                col_range = 1.0
            mins[col] = col_min
            maxs[col] = col_max
            train_n[col] = (train[col] - col_min) / col_range
            test_n[col] = (test[col] - col_min) / col_range
        return train_n, test_n, {"method": "minmax", "min": mins, "max": maxs}

    if normalized_method == "robust":
        medians: dict[str, float] = {}
        iqrs: dict[str, float] = {}
        for col in train.columns:
            if col in excluded or not pd.api.types.is_numeric_dtype(train[col]):
                continue
            col_median = float(train[col].median())
            q1 = float(train[col].quantile(0.25))
            q3 = float(train[col].quantile(0.75))
            col_iqr = q3 - q1
            if col_iqr == 0.0:
                col_iqr = 1.0
            medians[col] = col_median
            iqrs[col] = col_iqr
            train_n[col] = (train[col] - col_median) / col_iqr
            test_n[col] = (test[col] - col_median) / col_iqr
        return train_n, test_n, {"method": "robust", "median": medians, "iqr": iqrs}

    if normalized_method == "maxabs":
        maxabs: dict[str, float] = {}
        for col in train.columns:
            if col in excluded or not pd.api.types.is_numeric_dtype(train[col]):
                continue
            col_maxabs = float(train[col].abs().max())
            if col_maxabs == 0.0:
                col_maxabs = 1.0
            maxabs[col] = col_maxabs
            train_n[col] = train[col] / col_maxabs
            test_n[col] = test[col] / col_maxabs
        return train_n, test_n, {"method": "maxabs", "maxabs": maxabs}

    raise ValueError("Unsupported normalization method. Expected one of: none, zscore, minmax, robust, maxabs")


def apply_norm_stats(df: pd.DataFrame, stats: dict, exclude: list[str] | None = None) -> pd.DataFrame:
    """Apply saved normalization statistics to a dataframe."""

    excluded = set(exclude or [])
    out = df.copy()
    method = str(stats.get("method", "zscore")).strip().lower()
    if method == "none":
        return out
    if method == "zscore":
        for col, col_mu in stats.get("mu", {}).items():
            if col in excluded or col not in out.columns:
                continue
            out[col] = (out[col] - float(col_mu)) / float(stats["sigma"][col])
        return out
    if method == "minmax":
        mins = stats.get("min", {})
        maxs = stats.get("max", {})
        for col, col_min in mins.items():
            if col in excluded or col not in out.columns:
                continue
            col_range = float(maxs[col]) - float(col_min)
            if col_range == 0.0:
                col_range = 1.0
            out[col] = (out[col] - float(col_min)) / col_range
        return out
    if method == "robust":
        medians = stats.get("median", {})
        iqrs = stats.get("iqr", {})
        for col, col_median in medians.items():
            if col in excluded or col not in out.columns:
                continue
            col_iqr = float(iqrs[col])
            if col_iqr == 0.0:
                col_iqr = 1.0
            out[col] = (out[col] - float(col_median)) / col_iqr
        return out
    if method == "maxabs":
        for col, col_maxabs in stats.get("maxabs", {}).items():
            if col in excluded or col not in out.columns:
                continue
            denom = float(col_maxabs)
            if denom == 0.0:
                denom = 1.0
            out[col] = out[col] / denom
        return out
    raise ValueError(f"Unsupported normalization method in stats: {method}")
    return out


def denormalize_array(values, stats: dict, column: str):
    """Map normalized predictions back to the original scale."""

    arr = np.asarray(values, dtype=float)
    method = str(stats.get("method", "zscore")).strip().lower()
    if method == "none":
        return arr
    if method == "zscore":
        return (arr * float(stats["sigma"][column])) + float(stats["mu"][column])
    if method == "minmax":
        col_min = float(stats["min"][column])
        col_max = float(stats["max"][column])
        return arr * (col_max - col_min) + col_min
    if method == "robust":
        return (arr * float(stats["iqr"][column])) + float(stats["median"][column])
    if method == "maxabs":
        return arr * float(stats["maxabs"][column])
    raise ValueError(f"Unsupported normalization method in stats: {method}")


def missing_value_stats(df: pd.DataFrame, exclude: list[str] | None = None) -> dict:
    """Summarize missing values by column."""

    excluded = set(exclude or [])
    missing = {col: int(df[col].isna().sum()) for col in df.columns if col not in excluded}
    return {"total_missing": int(sum(missing.values())), "missing_by_column": missing}


def missing_value_stats_by_group(
    df: pd.DataFrame,
    *,
    group_col: str,
    exclude: list[str] | None = None,
) -> dict[str, dict]:
    """Summarize missing values by column for each group value."""

    if group_col not in df.columns:
        return {}

    grouped_stats: dict[str, dict] = {}
    group_series = df[group_col].astype(str)
    for group_value in sorted(group_series.unique()):
        group_mask = group_series == group_value
        grouped_stats[group_value] = missing_value_stats(df.loc[group_mask], exclude=exclude)
    return grouped_stats


def column_statistics(df: pd.DataFrame, exclude: list[str] | None = None) -> dict:
    """Produce lightweight descriptive statistics for numeric and categorical columns."""

    excluded = set(exclude or [])
    columns: dict[str, dict] = {}
    for col in df.columns:
        if col in excluded:
            continue
        series = df[col]
        present_count = int(series.notna().sum())
        missing_count = int(series.isna().sum())
        record = {
            "dtype": str(series.dtype),
            "row_count": int(len(series)),
            "present_count": present_count,
            "missing_count": missing_count,
            "present_pct_of_all_rows": float((present_count / len(series)) * 100.0) if len(series) else 0.0,
            "missing_pct_of_all_rows": float((missing_count / len(series)) * 100.0) if len(series) else 0.0,
            "unique": int(series.nunique(dropna=True)),
        }
        if pd.api.types.is_numeric_dtype(series):
            non_null = series.dropna()
            record.update(
                {
                    "mean": float(non_null.mean()) if not non_null.empty else None,
                    "median": float(non_null.median()) if not non_null.empty else None,
                    "min": float(non_null.min()) if not non_null.empty else None,
                    "max": float(non_null.max()) if not non_null.empty else None,
                }
            )
        columns[col] = record
    return {"rows": int(len(df)), "columns": columns}


def column_statistics_table(stats: dict, stage: str) -> pd.DataFrame:
    """Convert JSON-style column statistics into a flat table."""

    rows = []
    for column, values in (stats.get("columns") or {}).items():
        row = {"stage": stage, "column": column}
        row.update(values)
        rows.append(row)
    return pd.DataFrame(rows)


def impute_numeric(df: pd.DataFrame, exclude: list[str] | None = None):
    """Median-impute numeric columns and capture details for artifacts."""

    excluded = set(exclude or [])
    out = df.copy()
    details: dict[str, dict] = {"columns": {}}
    for col in out.columns:
        if col in excluded or not pd.api.types.is_numeric_dtype(out[col]):
            continue
        missing_before = int(out[col].isna().sum())
        if missing_before == 0:
            continue
        fill_value = float(out[col].median())
        out[col] = out[col].fillna(fill_value)
        details["columns"][col] = {
            "method": "median",
            "fill_value": fill_value,
            "missing_before": missing_before,
            "missing_after": int(out[col].isna().sum()),
        }
    return out, details


def correlation_analysis(df: pd.DataFrame, exclude: list[str] | None = None) -> dict:
    """Compute a numeric correlation matrix suitable for JSON export."""

    excluded = set(exclude or [])
    numeric_cols = [
        col for col in df.columns if col not in excluded and pd.api.types.is_numeric_dtype(df[col])
    ]
    if not numeric_cols:
        return {"columns": [], "matrix": []}
    corr = df[numeric_cols].corr().fillna(0.0)
    if corr.empty:
        return {"columns": [], "matrix": []}

    keep_columns: list[str] = []
    for col in corr.columns:
        off_diag = corr.loc[col].drop(labels=[col], errors="ignore")
        if not off_diag.empty and bool((off_diag.abs() > 0.0).any()):
            keep_columns.append(col)

    if not keep_columns:
        return {"columns": [], "matrix": []}

    corr = corr.loc[keep_columns, keep_columns]
    return {
        "columns": list(corr.columns),
        "matrix": corr.round(6).values.tolist(),
    }


def save_correlation_heatmap(analysis: dict, path) -> bool:
    """Save a simple heatmap image for the correlation matrix.

    Returns True when an image was written and False when there was not enough
    numeric correlation data to render a heatmap.
    """

    columns = analysis.get("columns", [])
    matrix = analysis.get("matrix", [])
    if not columns or not matrix:
        return False

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(max(6, len(columns)), max(5, len(columns) * 0.6)))
    values = np.asarray(matrix, dtype=float)
    image = ax.imshow(values, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(columns)))
    ax.set_yticks(range(len(columns)))
    ax.set_xticklabels(columns, rotation=45, ha="right")
    ax.set_yticklabels(columns)
    ax.set_title("Correlation Heatmap")
    for row_idx in range(values.shape[0]):
        for col_idx in range(values.shape[1]):
            value = float(values[row_idx, col_idx])
            text_color = "white" if abs(value) >= 0.5 else "black"
            ax.text(
                col_idx,
                row_idx,
                f"{value:.2f}",
                ha="center",
                va="center",
                color=text_color,
                fontsize=8,
            )
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return True


def save_comparison_plot(
    df: pd.DataFrame,
    *,
    x_col: str,
    series_map: dict[str, str],
    path,
    title: str,
) -> bool:
    """Save a simple time-series comparison plot for selected columns."""

    if df.empty or not series_map:
        return False

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 4))
    for label, column in series_map.items():
        if column not in df.columns:
            continue
        ax.plot(df[x_col], df[column], label=label, linewidth=1.8)

    if not ax.lines:
        plt.close(fig)
        return False

    ax.set_title(title)
    ax.set_xlabel(x_col)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return True


def save_training_history_plot(history, *, path, title: str = "Training Loss vs Epoch") -> bool:
    """Save a training loss vs epoch plot from a Keras History-like object."""

    history_map = getattr(history, "history", history)
    if not isinstance(history_map, dict):
        return False

    loss_values = history_map.get("loss")
    if not loss_values:
        return False

    loss = np.asarray([np.nan if value is None else float(value) for value in loss_values], dtype=float)
    if loss.size == 0 or not np.isfinite(loss).any():
        return False

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    epochs = np.arange(1, loss.size + 1)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(epochs, loss, label="training loss", linewidth=2.0)

    val_loss_values = history_map.get("val_loss")
    if val_loss_values is not None:
        val_loss = np.asarray([np.nan if value is None else float(value) for value in val_loss_values], dtype=float)
        if val_loss.size == loss.size and np.isfinite(val_loss).any():
            ax.plot(epochs, val_loss, label="validation loss", linewidth=1.8)

    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_xticks(epochs)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return True


def save_imputation_sample_plot(
    before_df: pd.DataFrame,
    after_df: pd.DataFrame,
    *,
    datetime_col: str,
    target_columns: list[str],
    path,
    title: str,
) -> bool:
    """Save a plot showing original and imputed values for selected columns."""

    if before_df.empty or after_df.empty or not target_columns:
        return False

    available_columns = [col for col in target_columns if col in before_df.columns and col in after_df.columns]
    if not available_columns:
        return False

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(len(available_columns), 1, figsize=(10, max(4, 3 * len(available_columns))), sharex=True)
    if len(available_columns) == 1:
        axes = [axes]

    for axis, column in zip(axes, available_columns):
        axis.plot(after_df[datetime_col], after_df[column], label=f"{column} imputed", linewidth=1.8)
        axis.plot(before_df[datetime_col], before_df[column], label=f"{column} original", linewidth=1.2, alpha=0.8)
        missing_before = before_df[column].isna()
        if bool(missing_before.any()):
            axis.scatter(
                after_df.loc[missing_before, datetime_col],
                after_df.loc[missing_before, column],
                label=f"{column} filled points",
                s=18,
                zorder=3,
            )
        axis.set_title(column)
        axis.grid(alpha=0.3)
        axis.legend()

    axes[0].figure.suptitle(title)
    axes[-1].set_xlabel(datetime_col)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return True


def save_imputation_sample_plot_by_station(
    before_df: pd.DataFrame,
    after_df: pd.DataFrame,
    *,
    datetime_col: str,
    station_col: str,
    target_columns: list[str],
    path,
    title: str,
) -> bool:
    """Save one imputation plot image with station-specific panels."""

    if before_df.empty or after_df.empty or not target_columns:
        return False
    if station_col not in before_df.columns or station_col not in after_df.columns:
        return False

    available_columns = [col for col in target_columns if col in before_df.columns and col in after_df.columns]
    if not available_columns:
        return False

    before_station_ids = before_df[station_col].astype(str)
    after_station_ids = after_df[station_col].astype(str)
    station_ids = sorted(set(before_station_ids.unique()) & set(after_station_ids.unique()))
    if not station_ids:
        return False

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(
        len(available_columns),
        len(station_ids),
        figsize=(max(10, 4 * len(station_ids)), max(4, 3 * len(available_columns))),
        sharex=False,
    )
    axes = np.asarray(axes)
    if axes.ndim == 0:
        axes = axes.reshape(1, 1)
    elif axes.ndim == 1:
        if len(available_columns) == 1:
            axes = axes.reshape(1, -1)
        else:
            axes = axes.reshape(-1, 1)

    for row_idx, column in enumerate(available_columns):
        for col_idx, station_id in enumerate(station_ids):
            axis = axes[row_idx, col_idx]
            station_mask_before = before_station_ids == station_id
            station_mask_after = after_station_ids == station_id
            station_before = before_df.loc[station_mask_before].sort_values(datetime_col).reset_index(drop=True)
            station_after = after_df.loc[station_mask_after].sort_values(datetime_col).reset_index(drop=True)

            if station_before.empty or station_after.empty:
                axis.set_title(f"{station_id} | {column}")
                axis.grid(alpha=0.3)
                continue

            axis.plot(station_after[datetime_col], station_after[column], label="imputed", linewidth=1.8)
            axis.plot(station_before[datetime_col], station_before[column], label="original", linewidth=1.2, alpha=0.8)

            if len(station_before) == len(station_after):
                missing_before = station_before[column].isna().to_numpy()
                if bool(missing_before.any()):
                    axis.scatter(
                        station_after.loc[missing_before, datetime_col],
                        station_after.loc[missing_before, column],
                        label="filled points",
                        s=16,
                        zorder=3,
                    )

            axis.set_title(f"{station_id} | {column}")
            axis.grid(alpha=0.3)
            axis.legend(fontsize=8)

    axes[0, 0].figure.suptitle(title)
    for col_idx in range(len(station_ids)):
        axes[-1, col_idx].set_xlabel(datetime_col)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return True


def save_prediction_sample_plot(
    df: pd.DataFrame,
    *,
    datetime_col: str,
    series_by_target: dict[str, tuple[str, str]],
    path,
    title: str,
) -> bool:
    """Save a prediction vs actual plot with one panel per target."""

    if df.empty or not series_by_target:
        return False

    available_targets = [
        target
        for target, (actual_col, prediction_col) in series_by_target.items()
        if actual_col in df.columns and prediction_col in df.columns
    ]
    if not available_targets:
        return False

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(len(available_targets), 1, figsize=(10, max(4, 3 * len(available_targets))), sharex=True)
    if len(available_targets) == 1:
        axes = [axes]

    for axis, target in zip(axes, available_targets):
        actual_col, prediction_col = series_by_target[target]
        axis.plot(df[datetime_col], df[actual_col], label=f"{target} actual", linewidth=1.2, alpha=0.85)
        axis.plot(df[datetime_col], df[prediction_col], label=f"{target} prediction", linewidth=1.8)
        axis.set_title(target)
        axis.grid(alpha=0.3)
        axis.legend()

    axes[0].figure.suptitle(title)
    axes[-1].set_xlabel(datetime_col)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return True


def impute_targets_with_random_forest(
    df: pd.DataFrame,
    *,
    target_columns: list[str],
    exclude: list[str] | None = None,
    top_k: int = 3,
    group_col: str | None = None,
):
    """Impute configured target columns with Random Forest using top correlated features."""

    def _impute_single(
        frame: pd.DataFrame,
        *,
        target_columns: list[str],
        exclude: list[str] | None = None,
        top_k: int = 3,
    ):
        excluded = set(exclude or [])
        out = frame.copy()
        analysis = correlation_analysis(out, exclude=exclude)
        corr_df = pd.DataFrame(analysis["matrix"], index=analysis["columns"], columns=analysis["columns"])
        details: dict[str, dict] = {"columns": {}}

        for target in target_columns:
            if target not in out.columns:
                continue

            missing_mask = out[target].isna()
            missing_before = int(missing_mask.sum())
            if missing_before == 0:
                continue

            candidate_features = [col for col in corr_df.columns if col != target and col not in excluded]
            ranked = sorted(
                candidate_features,
                key=lambda col: abs(float(corr_df.loc[target, col])) if target in corr_df.index and col in corr_df.columns else 0.0,
                reverse=True,
            )
            selected_features = ranked[: max(1, min(top_k, len(ranked)))] if ranked else []
            if not selected_features:
                fill_value = float(out[target].median())
                out[target] = out[target].fillna(fill_value)
                details["columns"][target] = {
                    "method": "median_fallback",
                    "selected_features": [],
                    "missing_before": missing_before,
                    "missing_after": int(out[target].isna().sum()),
                }
                continue

            train_mask = out[target].notna()
            if int(train_mask.sum()) < 2:
                fill_value = float(out[target].dropna().median()) if not out[target].dropna().empty else 0.0
                out[target] = out[target].fillna(fill_value)
                details["columns"][target] = {
                    "method": "median_fallback",
                    "selected_features": selected_features,
                    "missing_before": missing_before,
                    "missing_after": int(out[target].isna().sum()),
                }
                continue

            train_X = out.loc[train_mask, selected_features].copy()
            pred_X = out.loc[missing_mask, selected_features].copy()

            train_fill = {col: float(train_X[col].median()) for col in selected_features}
            train_X = train_X.fillna(train_fill)
            pred_X = pred_X.fillna(train_fill)

            train_y = out.loc[train_mask, target].astype(float)
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(train_X, train_y)
            out.loc[missing_mask, target] = model.predict(pred_X)

            details["columns"][target] = {
                "method": "random_forest",
                "selected_features": selected_features,
                "feature_correlations": {
                    col: float(corr_df.loc[target, col]) if target in corr_df.index and col in corr_df.columns else 0.0
                    for col in selected_features
                },
                "missing_before": missing_before,
                "missing_after": int(out[target].isna().sum()),
            }

        return out, details, analysis

    if group_col and group_col in df.columns:
        excluded = list(dict.fromkeys([*(exclude or []), group_col]))
        out = df.copy()
        by_station: dict[str, dict] = {}
        merged_details: dict[str, dict] = {"columns": {}, "by_station": by_station, "group_col": group_col}

        station_series = out[group_col].astype(str)
        for station_id in sorted(station_series.unique()):
            station_mask = station_series == station_id
            station_frame = out.loc[station_mask].copy()
            station_imputed, station_details, _ = _impute_single(
                station_frame,
                target_columns=target_columns,
                exclude=excluded,
                top_k=top_k,
            )
            out.loc[station_mask, station_imputed.columns] = station_imputed.values
            by_station[station_id] = station_details

            for column, detail in station_details.get("columns", {}).items():
                aggregate = merged_details["columns"].setdefault(
                    column,
                    {
                        "method": "per_station",
                        "missing_before": 0,
                        "missing_after": 0,
                        "selected_features_by_station": {},
                    },
                )
                aggregate["missing_before"] += int(detail.get("missing_before", 0))
                aggregate["missing_after"] += int(detail.get("missing_after", 0))
                aggregate["selected_features_by_station"][station_id] = detail.get("selected_features", [])

        analysis = correlation_analysis(out, exclude=excluded)
        return out, merged_details, analysis

    return _impute_single(df, target_columns=target_columns, exclude=exclude, top_k=top_k)


def imputation_stats_table(
    before: dict,
    after: dict,
    details: dict,
    *,
    before_by_group: dict[str, dict] | None = None,
    after_by_group: dict[str, dict] | None = None,
) -> pd.DataFrame:
    """Convert imputation summaries into a flat CSV-friendly table."""

    before_map = before.get("missing_by_column", {}) if before else {}
    after_map = after.get("missing_by_column", {}) if after else {}
    detail_map = details.get("columns", {}) if details else {}
    all_columns = sorted(set(before_map) | set(after_map) | set(detail_map))

    rows = []
    for column in all_columns:
        detail = detail_map.get(column, {})
        rows.append(
            {
                "station_id": "__all__",
                "column": column,
                "missing_before": int(before_map.get(column, 0)),
                "missing_after": int(after_map.get(column, 0)),
                "method": detail.get("method"),
                "selected_features": ",".join(detail.get("selected_features", [])),
            }
        )

    by_station = details.get("by_station") if details else None
    station_ids = sorted(
        set(before_by_group or {})
        | set(after_by_group or {})
        | set(by_station or {})
    )
    for station_id in station_ids:
        station_before_map = (before_by_group or {}).get(station_id, {}).get("missing_by_column", {})
        station_after_map = (after_by_group or {}).get(station_id, {}).get("missing_by_column", {})
        station_detail_map = ((by_station or {}).get(station_id, {}) or {}).get("columns", {})
        station_columns = sorted(
            set(station_before_map)
            | set(station_after_map)
            | set(station_detail_map)
        )
        for column in station_columns:
            detail = station_detail_map.get(column, {})
            rows.append(
                {
                    "station_id": station_id,
                    "column": column,
                    "missing_before": int(station_before_map.get(column, 0)),
                    "missing_after": int(station_after_map.get(column, 0)),
                    "method": detail.get("method"),
                    "selected_features": ",".join(detail.get("selected_features", [])),
                }
            )
    return pd.DataFrame(rows)


def file_signature(paths: list[str]) -> list[dict]:
    """Return simple cache signatures for a list of source files."""

    signature: list[dict] = []
    for raw_path in paths:
        path = Path(raw_path)
        stat = path.stat()
        signature.append(
            {
                "path": str(path.resolve()),
                "size": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
            }
        )
    return signature


def save_json(path, payload: dict) -> None:
    """Write JSON with indentation."""

    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
