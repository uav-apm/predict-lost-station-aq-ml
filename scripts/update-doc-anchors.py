#!/usr/bin/env python3
"""Refresh markdown source line anchors used in project docs.

This script updates selected docs links of the form:
- [symbol](../src/file.py#L123)
- click X href "../src/file.py#L123" "Open symbol"

It is intentionally label-aware to avoid accidental replacements on lines
containing multiple links.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class SymbolRef:
    source_rel_path: str
    pattern: str


SYMBOL_DEFS: dict[str, SymbolRef] = {
    "train_cmd()": SymbolRef("src/cli_train.py", r"^def\s+train_cmd\("),
    "_train_one()": SymbolRef("src/model_training.py", r"^def\s+_train_one\("),
    "_load_cfg()": SymbolRef("src/config_loader.py", r"^def\s+_load_cfg\("),
    "_prepare_training_data()": SymbolRef("src/data_pipeline.py", r"^def\s+_prepare_training_data\("),
    "_load_input_dataset()": SymbolRef("src/data_pipeline.py", r"^def\s+_load_input_dataset\("),
    "_filter_raw_df_by_periods()": SymbolRef("src/data_pipeline.py", r"^def\s+_filter_raw_df_by_periods\("),
    "_exclude_raw_df_by_periods()": SymbolRef("src/data_pipeline.py", r"^def\s+_exclude_raw_df_by_periods\("),
    "_build_supervised_frame()": SymbolRef("src/data_pipeline.py", r"^def\s+_build_supervised_frame\("),
    "_load_or_build_imputed_dataset()": SymbolRef("src/data_pipeline.py", r"^def\s+_load_or_build_imputed_dataset\("),
    "_dump_training_inputs()": SymbolRef("src/model_training.py", r"^def\s+_dump_training_inputs\("),
    "_write_imputation_plots()": SymbolRef("src/data_pipeline.py", r"^def\s+_write_imputation_plots\("),
    "_prepare_evaluation_data()": SymbolRef("src/data_pipeline.py", r"^def\s+_prepare_evaluation_data\("),
    "_prepare_fold_training_inputs()": SymbolRef("src/data_pipeline.py", r"^def\s+_prepare_fold_training_inputs\("),
    "_prepare_fold_validation_inputs()": SymbolRef("src/data_pipeline.py", r"^def\s+_prepare_fold_validation_inputs\("),
    "_resolve_eval_run_dir_arg()": SymbolRef("src/model_evaluation.py", r"^def\s+_resolve_eval_run_dir_arg\("),
    "_resolve_run_dir()": SymbolRef("src/artifact_io.py", r"^def\s+_resolve_run_dir\("),
    "_evaluate_run()": SymbolRef("src/model_evaluation.py", r"^def\s+_evaluate_run\("),
    "_write_prediction_plots()": SymbolRef("src/model_evaluation.py", r"^def\s+_write_prediction_plots\("),
    "_run_cross_validation()": SymbolRef("src/model_training.py", r"^def\s+_run_cross_validation\("),
    "_eval_one()": SymbolRef("src/model_evaluation.py", r"^def\s+_eval_one\("),
    "_baseline_metrics()": SymbolRef("src/model_evaluation.py", r"^def\s+_baseline_metrics\("),
    "_compute_station_replacement_baselines()": SymbolRef("src/model_evaluation.py", r"^def\s+_compute_station_replacement_baselines\("),
    "_load_chunked_joblib()": SymbolRef("src/artifact_io.py", r"^def\s+_load_chunked_joblib\("),
    "_load_saved_normalization_stats()": SymbolRef("src/artifact_io.py", r"^def\s+_load_saved_normalization_stats\("),
    "eval_cmd()": SymbolRef("src/cli_eval.py", r"^def\s+eval_cmd\("),
    "plot_cmd()": SymbolRef("src/cli_plot.py", r"^def\s+plot_cmd\("),
    "_resolve_plot_run_dirs()": SymbolRef("src/plot_writer.py", r"^def\s+_resolve_plot_run_dirs\("),
    "_collect_prediction_plot_series()": SymbolRef("src/plot_writer.py", r"^def\s+_collect_prediction_plot_series\("),
    "_write_interactive_prediction_plot()": SymbolRef("src/plot_writer.py", r"^def\s+_write_interactive_prediction_plot\("),
    "_resolve_algorithm_display_name_from_metadata()": SymbolRef("src/artifact_io.py", r"^def\s+_resolve_algorithm_display_name_from_metadata\("),
    "_write_static_prediction_plots()": SymbolRef("src/plot_writer.py", r"^def\s+_write_static_prediction_plots\("),
    "_write_static_actual_prediction_scatter_plots()": SymbolRef("src/plot_writer.py", r"^def\s+_write_static_actual_prediction_scatter_plots\("),
    "build_supervised_dataset()": SymbolRef("src/forecasting.py", r"^def\s+build_supervised_dataset\("),
    "normalize_periods()": SymbolRef("src/forecasting.py", r"^def\s+normalize_periods\("),
    "resolve_validation_periods()": SymbolRef("src/forecasting.py", r"^def\s+resolve_validation_periods\("),
    "years_for_periods()": SymbolRef("src/forecasting.py", r"^def\s+years_for_periods\("),
    "select_test_rows()": SymbolRef("src/forecasting.py", r"^def\s+select_test_rows\("),
    "identity_norm_stats()": SymbolRef("src/data.py", r"^def\s+identity_norm_stats\("),
    "normalize_train_test()": SymbolRef("src/data.py", r"^def\s+normalize_train_test\("),
    "impute_targets_with_random_forest()": SymbolRef("src/data.py", r"^def\s+impute_targets_with_random_forest\("),
    "apply_norm_stats()": SymbolRef("src/data.py", r"^def\s+apply_norm_stats\("),
    "validate_station_roles()": SymbolRef("src/reconstruction.py", r"^def\s+validate_station_roles\("),
    "build_station_matrix()": SymbolRef("src/reconstruction.py", r"^def\s+build_station_matrix\("),
    "make_regressor()": SymbolRef("src/pipeline.py", r"^def\s+make_regressor\("),
    "StationForecastPipeline": SymbolRef("src/pipeline.py", r"^class\s+StationForecastPipeline:"),
    "StationForecastPipeline.fit()": SymbolRef("src/pipeline.py", r"^\s+def\s+fit\("),
    "StationForecastPipeline.evaluate()": SymbolRef("src/pipeline.py", r"^\s+def\s+evaluate\("),
    "AttributionResult": SymbolRef("src/attribution.py", r"^class\s+AttributionResult:"),
    "compute_feature_attribution()": SymbolRef("src/attribution.py", r"^def\s+compute_feature_attribution\("),
}

TOOLTIP_ALIAS: dict[str, str] = {
    "artifact-writing section in _train_one()": "_train_one()",
    "norm_stats loading in _eval_one()": "_eval_one()",
}

DOCS = [
    ROOT / "docs" / "training-process.md",
    ROOT / "docs" / "evaluation.md",
    ROOT / "docs" / "plotting.md",
]


def find_line(source_rel_path: str, pattern: str) -> int:
    source_path = ROOT / source_rel_path
    regex = re.compile(pattern)
    with source_path.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle, start=1):
            if regex.search(line):
                return idx
    raise RuntimeError(f"Could not resolve line for pattern {pattern!r} in {source_rel_path}")


def build_anchor_map() -> dict[str, tuple[str, int]]:
    out: dict[str, tuple[str, int]] = {}
    required_symbols = collect_required_symbols()
    for label in sorted(required_symbols):
        ref = SYMBOL_DEFS.get(label)
        if ref is None:
            print(f"Warning: no symbol mapping defined for label '{label}'. Skipping.")
            continue
        try:
            out[label] = (ref.source_rel_path, find_line(ref.source_rel_path, ref.pattern))
        except RuntimeError as exc:
            print(f"Warning: {exc}. Skipping.")
    return out


def collect_required_symbols() -> set[str]:
    symbols: set[str] = set()
    link_pattern = re.compile(r"\[(`?[^`\]]+`?)\]\(\.\./src/[^)#]+#L\d+\)")
    click_pattern = re.compile(r'click\s+\w+\s+href\s+"\.\./src/[^"#]+#L\d+"\s+"Open\s+([^"]+)"')

    for doc in DOCS:
        if not doc.exists():
            continue
        text = doc.read_text(encoding="utf-8")
        for raw_label in link_pattern.findall(text):
            label = raw_label.strip("`").strip()
            if label:
                symbols.add(label)
        for tooltip in click_pattern.findall(text):
            symbol = TOOLTIP_ALIAS.get(tooltip, tooltip)
            if symbol:
                symbols.add(symbol)
    return symbols


def build_anchor_map_legacy() -> dict[str, tuple[str, int]]:
    out: dict[str, tuple[str, int]] = {}
    for label, ref in SYMBOL_DEFS.items():
        out[label] = (ref.source_rel_path, find_line(ref.source_rel_path, ref.pattern))
    return out


def rewrite_markdown_links(text: str, anchors: dict[str, tuple[str, int]]) -> str:
    for label, (source_rel_path, line_no) in anchors.items():
        target = f"../{source_rel_path}#L{line_no}"
        label_variants = [label, f"`{label}`"]
        for label_variant in label_variants:
            pattern = re.compile(rf"(\[{re.escape(label_variant)}\]\()\.\./src/[^)#]+#L\d+(\))")
            text = pattern.sub(rf"\1{target}\2", text)
    return text


def rewrite_mermaid_clicks(text: str, anchors: dict[str, tuple[str, int]]) -> str:
    pattern = re.compile(r'(click\s+\w+\s+href\s+")\.\./src/[^"#]+#L\d+("\s+"Open\s+([^"]+)".*)')

    def _replace_with_tooltip(match: re.Match[str]) -> str:
        prefix = match.group(1)
        suffix = match.group(2)
        tooltip = match.group(3)
        symbol = TOOLTIP_ALIAS.get(tooltip, tooltip)
        if symbol in anchors:
            source_rel_path, line_no = anchors[symbol]
            return f'{prefix}../{source_rel_path}#L{line_no}{suffix}'
        return match.group(0)

    return pattern.sub(_replace_with_tooltip, text)


def update_doc(path: Path, anchors: dict[str, tuple[str, int]]) -> bool:
    original = path.read_text(encoding="utf-8")
    updated = rewrite_markdown_links(original, anchors)
    updated = rewrite_mermaid_clicks(updated, anchors)
    if updated == original:
        return False
    path.write_text(updated, encoding="utf-8")
    return True


def main() -> int:
    anchors = build_anchor_map()
    changed: list[str] = []
    for doc in DOCS:
        if update_doc(doc, anchors):
            changed.append(str(doc.relative_to(ROOT)))

    if changed:
        print("Updated docs anchors:")
        for item in changed:
            print(f"- {item}")
    else:
        print("Docs anchors already up to date.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
