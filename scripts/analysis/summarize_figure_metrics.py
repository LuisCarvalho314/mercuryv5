from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

from mercury_runs.analysis.study_dataset import load_mercury_study_dataset


DEFAULT_REGIME_COLUMNS = (
    "level",
    "weight_memory",
    "rand_prob",
    "mercury_split_sensory_raw_latent_valid",
)

FIGURE_METRICS = (
    ("State precision", "precision"),
    ("State recall", "state_recall"),
    ("Mean total variation", "mean_total_variation"),
    ("Action-cond. mean TV", "action_conditioned_mean_total_variation"),
    ("Edge precision", "edge_precision"),
    ("Edge recall", "edge_recall"),
    ("Edge F1", "edge_f1"),
    ("Action-cond. edge precision", "action_conditioned_edge_precision"),
    ("Action-cond. edge recall", "action_conditioned_edge_recall"),
    ("Action-cond. edge F1", "action_conditioned_edge_f1"),
)


def state_recall_from_cooccurrence(matrix: Any) -> float | None:
    """Compute recall as best inferred-state coverage per ground-truth state."""
    if matrix is None:
        return None
    if hasattr(matrix, "to_list"):
        matrix = matrix.to_list()
    cooccurrence = np.asarray(matrix, dtype=np.float64)
    if cooccurrence.ndim != 2 or cooccurrence.size == 0:
        return None
    total = cooccurrence.sum()
    if total <= 0:
        return None
    return float(cooccurrence.max(axis=0).sum() / total)


def final_history_frame(history: pl.DataFrame) -> pl.DataFrame:
    if history.is_empty():
        return history
    final_steps = history.group_by("run_id").agg(pl.col("step").max().alias("step"))
    return history.join(final_steps, on=["run_id", "step"], how="inner")


def add_state_recall(history: pl.DataFrame) -> pl.DataFrame:
    return history.with_columns(
        pl.col("sensory_exact_state_cooccurrence_matrix")
        .map_elements(state_recall_from_cooccurrence, return_dtype=pl.Float64)
        .alias("sensory_state_recall"),
        pl.col("latent_exact_state_cooccurrence_matrix")
        .map_elements(state_recall_from_cooccurrence, return_dtype=pl.Float64)
        .alias("latent_state_recall"),
    )


def available_regime_columns(runs: pl.DataFrame, requested_columns: list[str]) -> list[str]:
    return [column for column in requested_columns if column in runs.columns]


def build_summary(study_root: Path, regime_columns: list[str], final_only: bool) -> pl.DataFrame:
    dataset = load_mercury_study_dataset(study_root)
    if dataset.history.is_empty():
        raise RuntimeError(f"No paper-precision history found under {study_root}")

    history = add_state_recall(dataset.history)
    if final_only:
        history = final_history_frame(history)

    group_columns = available_regime_columns(dataset.runs, regime_columns)
    if not group_columns:
        raise RuntimeError("None of the requested regime columns are available in the study summary.")
    if not final_only:
        group_columns = [*group_columns, "step"]

    metadata = dataset.runs.select(["run_id", *group_columns]).unique()
    joined = history.join(metadata, on="run_id", how="inner")

    rows: list[dict[str, Any]] = []
    for metric_index, (label, metric_name) in enumerate(FIGURE_METRICS):
        sensory_column = f"sensory_{metric_name}"
        latent_column = f"latent_{metric_name}"
        missing = [column for column in (sensory_column, latent_column) if column not in joined.columns]
        if missing:
            continue

        grouped = (
            joined.group_by(group_columns)
            .agg(
                pl.len().alias("n"),
                pl.col(sensory_column).mean().alias("sensory_mean"),
                pl.col(sensory_column).std().alias("sensory_sd"),
                pl.col(latent_column).mean().alias("latent_mean"),
                pl.col(latent_column).std().alias("latent_sd"),
            )
            .with_columns(
                pl.lit(metric_index).alias("metric_index"),
                pl.lit(label).alias("metric"),
            )
        )
        rows.extend(
            grouped.select(
                [*group_columns, "metric_index", "metric", "n", "sensory_mean", "sensory_sd", "latent_mean", "latent_sd"]
            ).iter_rows(named=True)
        )

    if not rows:
        raise RuntimeError("No figure metrics were found in the study history.")

    return pl.DataFrame(rows).sort([*group_columns, "metric_index"]).drop("metric_index")


def _format_stat(mean: float | None, sd: float | None) -> str:
    if mean is None:
        return "NA"
    if sd is None:
        return f"{mean:.3f} ± NA"
    return f"{mean:.3f} ± {sd:.3f}"


def to_markdown(summary: pl.DataFrame) -> str:
    regime_columns = [
        column
        for column in summary.columns
        if column not in {"metric_index", "metric", "n", "sensory_mean", "sensory_sd", "latent_mean", "latent_sd"}
    ]
    headers = [*regime_columns, "metric", "n", "sensory", "latent"]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in summary.iter_rows(named=True):
        values = [str(row[column]) for column in regime_columns]
        values.extend(
            [
                str(row["metric"]),
                str(row["n"]),
                _format_stat(row["sensory_mean"], row["sensory_sd"]),
                _format_stat(row["latent_mean"], row["latent_sd"]),
            ]
        )
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize all metrics shown in the notebook figure grid.")
    parser.add_argument("--study-root", required=True, type=Path, help="Study directory, e.g. results/current/studies/<study_name>")
    parser.add_argument(
        "--regime-columns",
        nargs="+",
        default=list(DEFAULT_REGIME_COLUMNS),
        help="Columns to preserve as regimes. Missing columns are ignored.",
    )
    parser.add_argument(
        "--all-steps",
        action="store_true",
        help="Summarize every checkpoint instead of only each run's final checkpoint.",
    )
    parser.add_argument("--output", type=Path, default=None, help="Optional output path.")
    parser.add_argument("--format", choices=("markdown", "csv"), default="markdown")
    args = parser.parse_args()

    summary = build_summary(
        study_root=args.study_root,
        regime_columns=args.regime_columns,
        final_only=not args.all_steps,
    )

    if args.format == "csv":
        if args.output is None:
            print(summary.write_csv())
        else:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            summary.write_csv(args.output)
            print(f"wrote {args.output}")
        return

    markdown = to_markdown(summary)
    if args.output is None:
        print(markdown)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(markdown + "\n", encoding="utf-8")
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
