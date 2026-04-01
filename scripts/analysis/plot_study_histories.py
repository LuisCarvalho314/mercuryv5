from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mercury_runs.analysis.study_history_plots import generate_study_history_plots


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate study-level paper-precision history plots.")
    parser.add_argument("--study-root", type=str, required=True, help="Study directory under results/current/studies/<study_name>")
    parser.add_argument("--output-dir", type=str, default=None, help="Optional output directory for generated plots")
    args = parser.parse_args()

    outputs = generate_study_history_plots(
        study_root=Path(args.study_root),
        output_dir=(Path(args.output_dir) if args.output_dir else None),
    )
    for level, path in sorted(outputs.items()):
        print(f"Saved level {level}: {path}")


if __name__ == "__main__":
    main()
