from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

from mercury_runs.analysis.study_dataset import load_mercury_study_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Load a Mercury study into Polars dataframes.")
    parser.add_argument("--study-root", type=str, required=True, help="Study directory under results/current/studies/<study_name>")
    parser.add_argument(
        "--write-dir",
        type=str,
        default=None,
        help="Optional directory to cache the loaded dataframe bundle as parquet files",
    )
    args = parser.parse_args()

    dataset = load_mercury_study_dataset(Path(args.study_root))
    frames = {
        "runs": dataset.runs,
        "history": dataset.history,
        "states": dataset.states,
        "attribution": dataset.attribution,
        "artifacts": dataset.artifacts,
    }
    for name, frame in frames.items():
        print(f"{name}: rows={frame.height} cols={frame.width}")

    if args.write_dir:
        output_dir = Path(args.write_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        for name, frame in frames.items():
            path = output_dir / f"{name}.parquet"
            frame.write_parquet(path)
            print(f"wrote {path}")


if __name__ == "__main__":
    main()
