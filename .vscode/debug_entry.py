from __future__ import annotations

import sys

from src.cli import eval_cmd, plot_cmd, predict_cmd, train_cmd


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("Usage: debug_entry.py <train|eval|predict> [args...]")

    command = sys.argv[1].lower().strip()
    sys.argv = [f"aq-spatial-reconstruction-{command}", *sys.argv[2:]]

    if command == "train":
        train_cmd()
        return
    if command == "eval":
        eval_cmd()
        return
    if command == "predict":
        predict_cmd()
        return
    if command == "plot":    
        plot_cmd()
        return      
    raise SystemExit(f"Unsupported debug command: {command}")


if __name__ == "__main__":
    main()
