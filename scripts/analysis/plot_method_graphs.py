from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mercury_runs.analysis.graph_plots import generate_method_graph_plots


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate unified internal graph plots for run methods.")
    parser.add_argument("--run_root", type=str, required=True, help="Run directory under results/current/studies/<study>/<run_id>")
    parser.add_argument("--datasets_root", type=str, default="datasets")
    args = parser.parse_args()

    outputs = generate_method_graph_plots(run_root=Path(args.run_root), datasets_root=Path(args.datasets_root))
    for name, path in outputs.items():
        print(f"Saved {name}: {path}")


if __name__ == "__main__":
    main()
