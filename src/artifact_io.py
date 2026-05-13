"""Artifact serialisation, run-directory management, and normalization persistence.

This module owns every operation that reads from or writes to the filesystem
for a training / evaluation run.  It is a dependency of ``data_pipeline``,
``model_training``, and ``model_evaluation`` and therefore must not import from
those modules.

Responsibilities
----------------
- Chunked joblib serialisation (``_save_chunked_joblib`` / ``_load_chunked_joblib``).
- Config-relative and project-relative path resolution (``_resolve_path``).
- Year-based file filtering (``_filter_paths_by_years``).
- Artifact directory helpers: ``_artifact_base_dir``, ``_comment_artifact_dir``,
  ``_config_latest_marker_name``, ``_compose_run_label``.
- Timestamped run directory creation and resolution (``_timestamped_run_dir``,
  ``_resolve_run_dir``).
- Config snapshot loading and alignment checking (``_load_run_config``,
  ``_ensure_config_alignment``).
- Normalization artifact helpers (``_normalization_artifact_path``,
  ``_identity_saved_norm_stats``, ``_load_saved_normalization_stats``).
- Thin JSON serialisation wrapper (``_save_json``).
"""

from __future__ import annotations

import json
import re
import shutil
import tempfile
from datetime import datetime
from pathlib import Path

import joblib
import yaml

from .config_loader import (
    PICKLE_CHUNK_SIZE_BYTES,
    PROJECT_ROOT,
    _sanitize_marker_suffix,
)
from .data import infer_year_from_path


# ---------------------------------------------------------------------------
# Chunked joblib helpers
# ---------------------------------------------------------------------------


def _pickle_part_paths(path: Path) -> list[Path]:
    """Return sorted list of ``.partNNNN`` files for a chunked joblib artifact.

    Args:
        path: Logical pickle path (without the ``.partNNNN`` suffix).

    Returns:
        Sorted list of existing part files.
    """
    return sorted(path.parent.glob(f"{path.name}.part*"))


def _save_chunked_joblib(
    obj,
    path: Path,
    *,
    chunk_size_bytes: int = PICKLE_CHUNK_SIZE_BYTES,
) -> list[Path]:
    """Serialise *obj* to disk as a sequence of fixed-size binary chunks.

    Splits the joblib-serialised byte stream into ``chunk_size_bytes``-sized
    files named ``<path>.part0001``, ``<path>.part0002``, …  This avoids
    Windows ``MAX_PATH`` and OS file-size constraints when persisting large
    sklearn / TensorFlow model artefacts.

    Args:
        obj: Any Python object accepted by :func:`joblib.dump`.
        path: Logical destination path.  The parent directory is created if it
            does not exist.
        chunk_size_bytes: Maximum byte size per chunk.  Must be positive.

    Returns:
        Ordered list of written part file paths.

    Raises:
        ValueError: When *chunk_size_bytes* is not positive, or when
            :func:`joblib.dump` produced no output.
    """
    if chunk_size_bytes <= 0:
        raise ValueError("chunk_size_bytes must be positive.")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.unlink(missing_ok=True)
    for part_path in _pickle_part_paths(path):
        part_path.unlink()

    written_parts: list[Path] = []
    with tempfile.TemporaryFile() as tmp_handle:
        joblib.dump(obj, tmp_handle)
        tmp_handle.seek(0)

        index = 1
        while True:
            chunk = tmp_handle.read(chunk_size_bytes)
            if not chunk:
                break
            part_path = path.parent / f"{path.name}.part{index:04d}"
            with part_path.open("wb") as part_handle:
                part_handle.write(chunk)
            written_parts.append(part_path)
            index += 1

    if not written_parts:
        raise ValueError(f"No pickle data was written for {path}.")

    return written_parts


def _load_chunked_joblib(path: Path):
    """Deserialise a joblib artifact written by :func:`_save_chunked_joblib`.

    Falls back to the original *path* first (single-file artefacts written by
    earlier versions of the code are still supported), then reassembles chunks
    in order.

    Args:
        path: Logical pickle path (without any ``.partNNNN`` suffix).

    Returns:
        The deserialised Python object.

    Raises:
        FileNotFoundError: When neither the monolithic file nor any part files
            exist.
    """
    if path.exists():
        return joblib.load(path)

    part_paths = _pickle_part_paths(path)
    if not part_paths:
        raise FileNotFoundError(f"Pickle artifact not found: {path}")

    with tempfile.TemporaryFile() as tmp_handle:
        for part_path in part_paths:
            with part_path.open("rb") as part_handle:
                shutil.copyfileobj(part_handle, tmp_handle)
        tmp_handle.seek(0)
        return joblib.load(tmp_handle)


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------


def _resolve_path(base: Path, raw_path: str) -> Path:
    """Resolve *raw_path* relative to *base*, falling back to cwd and project root.

    Absolute paths are returned as-is.  For relative paths the function tries
    ``base``, ``cwd``, and ``PROJECT_ROOT`` in that order and returns the first
    existing candidate.  When none exist it returns ``base / raw_path`` so the
    caller can create the file at the most natural location.

    Args:
        base: Directory to interpret relative paths against first.
        raw_path: Absolute or relative path string.

    Returns:
        A resolved :class:`~pathlib.Path` pointing to the most plausible
        location for *raw_path*.
    """
    path = Path(raw_path)
    if path.is_absolute():
        return path
    for prefix in (base, Path.cwd(), PROJECT_ROOT):
        candidate = prefix / path
        if candidate.exists():
            return candidate
    return base / path


def _filter_paths_by_years(
    paths: list[str],
    allowed_years: list[int] | None,
) -> list[str]:
    """Restrict file paths to those whose inferred year appears in *allowed_years*.

    Uses :func:`~src.data.infer_year_from_path` to extract a four-digit year
    from each path.  Paths whose year cannot be inferred are silently excluded
    when *allowed_years* is non-empty.

    Args:
        paths: List of file paths to filter.
        allowed_years: Whitelist of acceptable years.  When ``None`` or empty
            the original list is returned unchanged.

    Returns:
        Filtered list of file paths.

    Raises:
        FileNotFoundError: When *allowed_years* is non-empty but the filter
            produces an empty list.
    """
    if not allowed_years:
        return paths

    allowed = {int(year) for year in allowed_years}
    filtered = [path for path in paths if infer_year_from_path(path) in allowed]
    if not filtered:
        raise FileNotFoundError(
            f"No input files matched the configured year filter: {sorted(allowed)}"
        )
    return filtered


# ---------------------------------------------------------------------------
# Artifact directory helpers
# ---------------------------------------------------------------------------


def _artifact_base_dir(cfgd: dict, base: Path) -> Path:
    """Resolve and create the top-level artifact output directory.

    Supports absolute paths, explicit relative paths (treated as project-root
    relative), and short names (resolved via :func:`_resolve_path`).

    Args:
        cfgd: Fully-resolved configuration dictionary.
        base: Directory containing the active config file.

    Returns:
        Absolute path to the existing artifact output directory.
    """
    raw_dir = str(cfgd["artifacts"]["dir"])
    path = Path(raw_dir)
    if path.is_absolute():
        out_dir = path
    elif any(sep in raw_dir for sep in ("/", "\\")) and not raw_dir.startswith("."):
        out_dir = PROJECT_ROOT / path
    else:
        out_dir = _resolve_path(base, raw_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _config_latest_marker_name(config_path: str) -> str:
    """Return the ``latest_run`` marker filename for the given config path.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        A filename string of the form ``latest_run_<stem>.txt``.
    """
    stem = Path(config_path).stem
    return f"latest_run_{_sanitize_marker_suffix(stem)}.txt"


def _compose_run_label(config_path: str, run_comment: str | None = None) -> str:
    """Compose a human-readable label for a run directory.

    Args:
        config_path: Path to the YAML configuration file.
        run_comment: Optional free-text comment to append to the label.

    Returns:
        A label string combining the config stem and the comment (when given).
    """
    label_parts = [Path(config_path).stem]
    if run_comment and str(run_comment).strip():
        label_parts.append(str(run_comment).strip())
    return "_".join(label_parts)


def _comment_artifact_dir(base: Path, run_comment: str | None) -> Path:
    """Return (and create) a comment-scoped sub-directory under *base*.

    When *run_comment* is blank the function returns *base* unchanged.

    Args:
        base: Top-level artifact directory.
        run_comment: Optional comment string used as a sub-directory name.

    Returns:
        A directory path — either *base* itself or ``base / <sanitized_comment>``.
    """
    comment = str(run_comment or "").strip()
    if not comment:
        return base
    comment_dir = base / _sanitize_marker_suffix(comment)
    comment_dir.mkdir(parents=True, exist_ok=True)
    return comment_dir


# ---------------------------------------------------------------------------
# Run directory creation and resolution
# ---------------------------------------------------------------------------


def _timestamped_run_dir(
    base: Path,
    latest_markers: list[str] | None = None,
    *,
    run_label: str | None = None,
    marker_dirs: list[Path] | None = None,
) -> Path:
    """Create a unique timestamped directory under *base* for one training run.

    The directory name is ``[<label>_]<YYYYMMDD_HHMMSS>_<n>`` where ``<n>`` is
    a zero-based collision counter.  After the directory is created one or more
    ``latest_run*.txt`` marker files are written so that subsequent operations
    can auto-resolve the most recent run.

    Args:
        base: Parent directory that will contain the new run directory.
        latest_markers: Additional marker file names to write alongside the
            default ``latest_run.txt``.
        run_label: Optional label prefix embedded in the directory name.
        marker_dirs: Directories in which to write marker files.  Defaults to
            ``[base]``.

    Returns:
        Absolute path to the newly created run directory.
    """
    marker_names = ["latest_run.txt"]
    if latest_markers:
        for marker_name in latest_markers:
            if marker_name and marker_name not in marker_names:
                marker_names.append(marker_name)

    label_prefix = ""
    if run_label:
        label_prefix = f"{_sanitize_marker_suffix(run_label)}_"

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = 0
    while True:
        run_dir = base / f"{label_prefix}{stamp}_{suffix}"
        if not run_dir.exists():
            run_dir.mkdir(parents=True, exist_ok=False)
            target_marker_dirs = marker_dirs or [base]
            for marker_dir in target_marker_dirs:
                marker_dir.mkdir(parents=True, exist_ok=True)
                for marker_name in marker_names:
                    (marker_dir / marker_name).write_text(str(run_dir), encoding="utf-8")
            return run_dir
        suffix += 1


def _resolve_run_dir(
    base: Path,
    run_dir_arg: str | None,
    latest_markers: list[str] | None = None,
    *,
    include_default_latest_marker: bool = True,
) -> Path:
    """Resolve a run directory from an explicit argument or the most recent marker.

    Resolution order:
    1. Explicit *run_dir_arg* (absolute or relative to *base*).
    2. Most recently modified ``latest_run*.txt`` marker whose recorded path
       still exists on disk.
    3. Most recently modified directory containing a ``config_snapshot.yaml``.
    4. Most recently modified directory containing a ``run_metadata.json``.
    5. Lexicographically last immediate sub-directory of *base*.

    Args:
        base: Top-level artifact directory to search.
        run_dir_arg: Optional explicit path supplied via the CLI.
        latest_markers: Additional marker file names to check before the
            default ``latest_run.txt``.
        include_default_latest_marker: When ``True`` the default
            ``latest_run.txt`` marker is included in the search even when
            *latest_markers* is non-empty.

    Returns:
        Absolute path to the resolved run directory.

    Raises:
        FileNotFoundError: When no run directory can be found under *base*.
    """
    if run_dir_arg:
        run_dir = Path(run_dir_arg)
        return run_dir if run_dir.is_absolute() else base / run_dir

    marker_names = ["latest_run.txt"]
    if latest_markers:
        marker_names = [marker_name for marker_name in latest_markers if marker_name]
        if include_default_latest_marker and "latest_run.txt" not in marker_names:
            marker_names.append("latest_run.txt")

    marker_candidates: list[tuple[float, Path]] = []
    for marker_name in marker_names:
        direct_marker = base / marker_name
        if direct_marker.exists():
            marker_candidates.append((direct_marker.stat().st_mtime, direct_marker))

        for nested_marker in base.rglob(marker_name):
            marker_candidates.append((nested_marker.stat().st_mtime, nested_marker))

    for _, marker_path in sorted(marker_candidates, key=lambda item: item[0], reverse=True):
        raw = marker_path.read_text(encoding="utf-8").strip()
        if not raw:
            continue
        latest_path = Path(raw)
        if latest_path.exists():
            return latest_path

    snapshot_candidates = sorted({path.parent for path in base.rglob("config_snapshot.yaml")})
    if snapshot_candidates:
        return snapshot_candidates[-1]

    metadata_candidates = sorted({path.parent for path in base.rglob("run_metadata.json")})
    if metadata_candidates:
        return metadata_candidates[-1]

    candidates = sorted(path for path in base.iterdir() if path.is_dir())
    if not candidates:
        raise FileNotFoundError(f"No run directories found under {base}")
    return candidates[-1]


# ---------------------------------------------------------------------------
# Config snapshot helpers
# ---------------------------------------------------------------------------


def _load_run_config(run_dir: Path) -> dict | None:
    """Load the YAML config snapshot stored inside a run directory.

    Args:
        run_dir: Path to the training run directory.

    Returns:
        Parsed config dictionary, or ``None`` when no snapshot exists.
    """
    snapshot = run_dir / "config_snapshot.yaml"
    if not snapshot.exists():
        return None
    with snapshot.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _ensure_config_alignment(cfgd: dict, run_cfg: dict | None) -> None:
    """Raise when key fields in *cfgd* do not match the saved *run_cfg*.

    This guards against accidentally evaluating a model with a config that
    targets a different station, algorithm, or hyper-parameter set than the
    one used during training.

    Args:
        cfgd: Current (evaluation-time) configuration dictionary.
        run_cfg: Saved training config snapshot, or ``None`` (no check).

    Raises:
        ValueError: When *target_station_id*, *algorithm*, or *model.params*
            differ between the two configs.
    """
    if run_cfg is None:
        return
    if cfgd["data"]["target_station_id"] != run_cfg["data"]["target_station_id"]:
        raise ValueError("Configured target station id does not match the saved run.")
    if cfgd["model"]["algorithm"] != run_cfg["model"]["algorithm"]:
        raise ValueError("Configured algorithm does not match the saved run.")
    if cfgd["model"].get("params") != (run_cfg["model"].get("params") if run_cfg.get("model") else None):
        raise ValueError("Configured model params do not match the saved run.")


# ---------------------------------------------------------------------------
# Normalization artifact helpers
# ---------------------------------------------------------------------------


def _normalization_artifact_path(run_dir: Path) -> Path:
    """Return the canonical path for normalization statistics inside a run directory.

    Args:
        run_dir: Path to the training run directory.

    Returns:
        Path to ``norm_stats.json`` within *run_dir*.
    """
    return run_dir / "norm_stats.json"


def _identity_saved_norm_stats() -> dict:
    """Return a no-op normalization statistics dictionary.

    Used as a safe fallback when the run was trained without normalizing
    features (``normalization: none``) and no ``norm_stats.json`` was written.

    Returns:
        Dict with ``method="none"`` and empty ``mu`` / ``sigma`` mappings.
    """
    return {"method": "none", "mu": {}, "sigma": {}}


def _load_saved_normalization_stats(
    run_dir: Path,
    *,
    normalization_method: str,
) -> dict:
    """Load per-feature normalization statistics from a training run directory.

    Args:
        run_dir: Path to the training run directory.
        normalization_method: The normalization type declared in the eval config
            (e.g. ``"zscore"`` or ``"none"``).  When ``"none"`` and the stats
            file is absent, a no-op stats dict is returned instead of raising.

    Returns:
        Normalization statistics dictionary suitable for
        :func:`~src.data.apply_norm_stats`.

    Raises:
        FileNotFoundError: When the stats file is missing and normalization
            is not ``"none"``.
    """
    norm_stats_path = _normalization_artifact_path(run_dir)
    if norm_stats_path.exists():
        with norm_stats_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    if normalization_method == "none":
        return _identity_saved_norm_stats()
    raise FileNotFoundError(
        "Saved normalization stats were not found for this run. "
        "Re-train with --save-normalization (or artifacts.save_training_normalization: true), "
        "or run evaluation with --no-use-training-normalization."
    )


# ---------------------------------------------------------------------------
# JSON helper
# ---------------------------------------------------------------------------


def _save_json(path: Path, payload) -> None:
    """Serialise *payload* to *path* as pretty-printed JSON.

    Args:
        path: Destination file path.  Parent directories are created if they
            do not exist.
        payload: JSON-serialisable Python object.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)


# ---------------------------------------------------------------------------
# Algorithm display name helpers
# ---------------------------------------------------------------------------


def _algorithm_label(raw_algorithm: str | None) -> str:
    """Return a short, human-readable algorithm label suitable for charts.

    Strips any Python module prefix so ``src.lstm_regressor.LocalLSTMRegressor``
    becomes ``LocalLSTMRegressor``.

    Args:
        raw_algorithm: Raw algorithm identifier string, or ``None``.

    Returns:
        Cleaned algorithm label string.
    """
    if not raw_algorithm:
        return "unknown"
    return raw_algorithm.rsplit(".", 1)[-1]


def _resolve_algorithm_display_name(cfgd: dict) -> str:
    """Derive a display name for the algorithm from the live configuration.

    Prefers the explicit ``model.algorithm_name`` setting; falls back to the
    short form of ``model.algorithm``.

    Args:
        cfgd: Fully-resolved configuration dictionary.

    Returns:
        Human-readable algorithm display name.
    """
    model_cfg = cfgd.get("model", {})
    configured_name = str(model_cfg.get("algorithm_name") or "").strip()
    if configured_name:
        return configured_name
    return _algorithm_label(str(model_cfg.get("algorithm") or "unknown"))


def _resolve_algorithm_display_name_from_metadata(metadata: dict) -> str:
    """Derive a display name for the algorithm from a saved run-metadata dict.

    Applies heuristics for well-known algorithms (e.g. BiLSTM variants) when
    an explicit ``algorithm_name`` field is absent.

    Args:
        metadata: Run-metadata dict as loaded from ``run_metadata.json``.

    Returns:
        Human-readable algorithm display name.
    """
    configured_name = str(metadata.get("algorithm_name") or "").strip()
    if configured_name:
        return configured_name

    raw_algorithm = str(metadata.get("algorithm") or "unknown")
    short_algorithm = _algorithm_label(raw_algorithm)
    config_path = str(metadata.get("config_path") or "").lower()
    model_params = metadata.get("model_params") or {}
    is_bidirectional = bool(model_params.get("bidirectional", False))

    if raw_algorithm == "src.lstm_regressor.LocalLSTMRegressor":
        if is_bidirectional:
            if "bilstm_pm25" in config_path:
                return "BiLSTM (PM2.5)"
            if "bilstm_no2" in config_path:
                return "BiLSTM (NO2)"
            if "bilstm" in config_path:
                return "BiLSTM (PM2.5+NO2)"
            return "BiLSTM"
        if "lstm" in config_path:
            return "LSTM"

    return short_algorithm
