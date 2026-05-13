"""
Configuration loading, YAML inheritance resolution, and run-progress logging.

This module is the dependency root of the pipeline — it has no imports from other
``src`` submodules and only depends on the standard library plus PyYAML.

Responsibilities
----------------
- ``_RunLogger`` / ``RUN_LOGGER``: module-level singleton for timestamped progress
  messages written to stderr and optionally persisted to a log file.
- ``_deep_merge_dict`` / ``_load_cfg_with_inheritance`` / ``_load_cfg``: YAML loading
  with recursive ``extends`` inheritance and deep-merge semantics.
- ``_is_base_config_path`` / ``_expand_config_paths``: resolve ``--config`` and
  ``--config-dir`` CLI arguments into an ordered list of concrete YAML file paths.
- ``_sanitize_marker_suffix``: shared string-sanitisation helper used by artifact
  naming throughout the codebase.
- ``_report_phase`` / ``_report_note``: thin wrappers around ``RUN_LOGGER.emit`` used
  as structured progress checkpoints throughout training and evaluation.

Constants
---------
PROJECT_ROOT : Path
    Resolved absolute path of the repository root directory.
PICKLE_CHUNK_SIZE_BYTES : int
    Maximum byte size of a single joblib serialisation chunk (100 MB).
"""

from __future__ import annotations

import re
import sys
from datetime import datetime
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PICKLE_CHUNK_SIZE_BYTES = 100 * 1024 * 1024


class _RunLogger:
    """Emit timestamped run logs to stderr and optionally persist them to disk.

    A single module-level instance (``RUN_LOGGER``) is shared across the entire
    pipeline.  The ``start`` method resets the per-run buffer; ``set_log_file``
    flushes any buffered lines to disk and keeps the file open for subsequent
    ``emit`` calls.

    Attributes:
        _config_name: Stem of the active configuration file, used as a label in
            log lines.
        _buffer: Lines emitted before ``set_log_file`` was called.
        _log_path: Destination file path once the run directory is known.
    """

    def __init__(self) -> None:
        self._config_name: str | None = None
        self._buffer: list[str] = []
        self._log_path: Path | None = None

    def start(self, config_path: str) -> None:
        """Reset internal state for a new configuration run.

        Args:
            config_path: Path to the YAML config file being processed.  Only
                the filename stem is used as the per-line label.
        """
        self._config_name = Path(config_path).name
        self._buffer = []
        self._log_path = None

    def set_log_file(self, path: Path) -> None:
        """Direct future log lines to *path* and flush any buffered lines.

        Args:
            path: Destination log file.  Parent directories are created if they
                do not exist.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        self._log_path = path
        if self._buffer:
            with path.open("a", encoding="utf-8") as handle:
                for line in self._buffer:
                    handle.write(f"{line}\n")

    def emit(self, kind: str, message: str) -> None:
        """Write one log entry to stderr (always) and the log file (when set).

        Args:
            kind: Short category label, e.g. ``"PHASE"`` or ``"NOTE"``.
            message: Human-readable message body.
        """
        timestamp = datetime.now().isoformat(timespec="seconds")
        config_label = self._config_name or "unknown_config"
        line = f"{timestamp} [{config_label}] [{kind}] {message}"
        self._buffer.append(line)
        print(line, file=sys.stderr, flush=True)
        if self._log_path is not None:
            with self._log_path.open("a", encoding="utf-8") as handle:
                handle.write(f"{line}\n")


RUN_LOGGER = _RunLogger()


def _report_phase(phase: str, index: int, total: int) -> None:
    """Emit a ``phase`` log entry showing numerical progress through pipeline stages.

    Args:
        phase: Human-readable name of the current phase/step.
        index: One-based index of the current step.
        total: Total number of steps in the enclosing loop.
    """
    pct = int((index / total) * 100)
    RUN_LOGGER.emit("phase", f"{index}/{total} ({pct:>3}%) {phase}")


def _report_note(message: str) -> None:
    """Emit a ``note`` log entry for informational observations during a run.

    Args:
        message: Informational detail to include in the run log.
    """
    RUN_LOGGER.emit("note", message)

def _sanitize_marker_suffix(raw: str) -> str:
    """Normalise *raw* to a filesystem-safe identifier string.

    Replaces any run of characters outside ``[A-Za-z0-9_-]`` with a single
    underscore and strips leading/trailing underscores.

    Args:
        raw: Arbitrary string to sanitise (e.g. a config stem or comment).

    Returns:
        A filesystem-safe string, or ``"config"`` when *raw* is blank.
    """
    sanitized = re.sub(r"[^A-Za-z0-9_-]+", "_", raw.strip())
    return sanitized.strip("_") or "config"


def _deep_merge_dict(base: dict, override: dict) -> dict:
    """Recursively merge two dictionaries, giving *override* precedence.

    For keys present in both dicts whose values are both plain ``dict`` objects
    the merge recurses; for all other types the *override* value replaces the
    *base* value wholesale.

    Args:
        base: The lower-priority dictionary (e.g. values from a parent config).
        override: The higher-priority dictionary (e.g. values from a child
            config that ``extends`` the base).

    Returns:
        A new dict that is the deep merge of *base* and *override*.
    """
    merged: dict = dict(base)
    for key, value in override.items():
        base_value = merged.get(key)
        if isinstance(base_value, dict) and isinstance(value, dict):
            merged[key] = _deep_merge_dict(base_value, value)
        else:
            merged[key] = value
    return merged


def _load_cfg_with_inheritance(cfg_path: Path, seen: set[Path] | None = None) -> dict:
    """Load one YAML config file and recursively resolve optional ``extends`` inheritance.

    When a config file contains ``extends: path/to/base.yaml`` it is loaded first
    and the child values are deep-merged on top.  Cycle detection is performed
    using the ``seen`` set; a ``ValueError`` is raised if a cycle is found.

    Args:
        cfg_path: Path to the YAML configuration file to load.
        seen: Set of already-visited resolved paths used for cycle detection.
            Pass ``None`` (the default) at the call site; the function manages
            this set internally across recursive calls.

    Returns:
        A fully resolved configuration dictionary.

    Raises:
        ValueError: When a circular ``extends`` chain is detected.
        FileNotFoundError: When the file named in ``extends`` does not exist.
        TypeError: When the YAML file does not contain a top-level mapping.
    """
    resolved_path = cfg_path.resolve()
    visited = seen or set()
    if resolved_path in visited:
        chain = " -> ".join(str(path) for path in [*visited, resolved_path])
        raise ValueError(f"Config inheritance cycle detected: {chain}")

    visited.add(resolved_path)
    with cfg_path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle) or {}

    if not isinstance(cfg, dict):
        raise TypeError(f"Config file must contain a YAML mapping at top level: {cfg_path}")

    extends = cfg.get("extends")
    if not extends:
        visited.remove(resolved_path)
        return cfg

    parent_path = Path(extends)
    if not parent_path.is_absolute():
        parent_path = cfg_path.parent / parent_path
    parent_path = parent_path.resolve()
    if not parent_path.exists():
        raise FileNotFoundError(f"Base config referenced by extends was not found: {parent_path}")

    parent_cfg = _load_cfg_with_inheritance(parent_path, seen=visited)
    child_cfg = dict(cfg)
    child_cfg.pop("extends", None)
    merged = _deep_merge_dict(parent_cfg, child_cfg)
    visited.remove(resolved_path)
    return merged


def _load_cfg(path: str | Path) -> tuple[dict, Path]:
    """Load and fully resolve a YAML config file.

    Args:
        path: Path to the YAML configuration file.

    Returns:
        A ``(cfg_dict, config_dir)`` tuple where *cfg_dict* is the resolved
        configuration mapping and *config_dir* is the directory containing the
        config file (used for resolving relative paths within the config).

    Raises:
        FileNotFoundError: When *path* does not exist on disk.
    """
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    cfg = _load_cfg_with_inheritance(cfg_path)
    return cfg, cfg_path.parent


def _is_base_config_path(path: Path) -> bool:
    """Return ``True`` when a config filename follows the base-profile naming convention.

    Config files named ``base.yaml`` or ``*_base.yaml`` are treated as abstract
    parent configs that are not run directly.  They are skipped when
    ``--config-dir`` scans a directory for runnable configs.

    Args:
        path: Path to the candidate config file.

    Returns:
        ``True`` if the stem is ``"base"`` or ends with ``"_base"``.
    """
    stem = path.stem.lower()
    return stem == "base" or stem.endswith("_base")


def _expand_config_paths(
    config_paths: list[str] | None,
    config_dirs: list[str] | None,
) -> list[str]:
    """Resolve ``--config`` and ``--config-dir`` arguments into an ordered file list.

    Explicit ``--config`` paths are kept first in declaration order.  For each
    ``--config-dir`` entry the directory is scanned recursively for ``*.yaml``
    and ``*.yml`` files; files under ``artifacts/``, ``data/``, ``outputs/``,
    ``unusedconfigs/``, ``unused_configs/``, and ``__pycache__/`` sub-trees are
    skipped, as are base-profile files (see :func:`_is_base_config_path`).

    Args:
        config_paths: Explicit config file paths supplied via ``--config``.
            May be ``None``.
        config_dirs: Directory paths supplied via ``--config-dir`` to scan
            recursively.  May be ``None``.

    Returns:
        Deduplicated list of resolved config file paths in stable order.

    Raises:
        FileNotFoundError: When a directory from *config_dirs* does not exist
            or contains no runnable configs.
        ValueError: When neither *config_paths* nor *config_dirs* is non-empty.
    """
    excluded_dir_names = {
        "artifacts",
        "data",
        "outputs",
        "unusedconfigs",
        "unused_configs",
        "__pycache__",
    }

    def _is_excluded_candidate(path: Path, root: Path) -> bool:
        rel_parts = [part.lower() for part in path.relative_to(root).parts[:-1]]
        return any(part in excluded_dir_names for part in rel_parts)

    resolved: list[str] = []
    seen_from_dirs: set[Path] = set()

    for raw_path in config_paths or []:
        cfg_path = Path(raw_path)
        resolved.append(str(cfg_path))
        seen_from_dirs.add(cfg_path)

    for raw_dir in config_dirs or []:
        cfg_dir = Path(raw_dir)
        if not cfg_dir.exists() or not cfg_dir.is_dir():
            raise FileNotFoundError(f"Config directory not found: {cfg_dir}")

        candidates = sorted(cfg_dir.rglob("*.yaml"))
        candidates.extend(sorted(cfg_dir.rglob("*.yml")))
        added_from_dir = 0
        for candidate in candidates:
            if _is_excluded_candidate(candidate, cfg_dir):
                continue
            if _is_base_config_path(candidate):
                continue
            if candidate in seen_from_dirs:
                continue
            seen_from_dirs.add(candidate)
            resolved.append(str(candidate))
            added_from_dir += 1

        if added_from_dir == 0:
            raise FileNotFoundError(
                f"No non-base YAML config files found in directory: {cfg_dir}"
            )

    if not resolved:
        raise ValueError("Provide at least one --config or --config-dir.")

    return resolved
