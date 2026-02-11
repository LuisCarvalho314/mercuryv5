# scripts/run_mercury_bundle.py
from __future__ import annotations

import argparse
from pathlib import Path

from src.mercury_runs.pipelines import run_all_bundled


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASETS_DIR = PROJECT_ROOT / "datasets"
RESULTS_DIR = PROJECT_ROOT / "results" / "mercury"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Mercury pipelines with bundled Parquet outputs.")
    parser.add_argument("--level", type=int, required=True)
    parser.add_argument("--sensor", type=str, default="cartesian")
    parser.add_argument("--sensor-range", type=int, default=None)
    parser.add_argument("--dataset-select", type=str, default="latest", choices=["latest", "run_id"])
    parser.add_argument("--dataset-run-id", type=str, default=None)
    parser.add_argument("--memory-length", type=int, default=10)
    parser.add_argument("--datasets-root", type=str, default=str(DATASETS_DIR))
    parser.add_argument("--results-root", type=str, default=str(RESULTS_DIR))
    parser.add_argument("--no-embed-metadata", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    run_all_bundled(
        datasets_root=Path(args.datasets_root),
        results_root=Path(args.results_root),
        level=args.level,
        sensor=args.sensor,
        sensor_range=args.sensor_range,
        dataset_select=args.dataset_select,
        dataset_run_id=args.dataset_run_id,
        memory_length=args.memory_length,
        embed_metadata_in_parquet=not args.no_embed_metadata,
    )


if __name__ == "__main__":
    main()
