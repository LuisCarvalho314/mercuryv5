"""Plot computational measurements from a study summary."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy.stats import t


SERIES_ORDER = (
    "mercury_cscg_hybrid",
    "tmgwr_cscg_hybrid",
    "vanilla_cscg",
    "oracle_capacity",
    "oracle_capacity_first_order",
    "oracle_capacity_structural",
)
SERIES_LABELS = {
    "mercury_cscg_hybrid": "MERCURY initialised",
    "tmgwr_cscg_hybrid": "TMGWR initialised",
    "vanilla_cscg": "Standard CSCG",
    "oracle_capacity": "Oracle capacity",
    "oracle_capacity_first_order": "Oracle capacity + first-order structure",
    "oracle_capacity_structural": "Oracle capacity + structure",
}
SERIES_COLORS = {
    "mercury_cscg_hybrid": "#CC79A7",
    "tmgwr_cscg_hybrid": "#7B61A8",
    "vanilla_cscg": "#264653",
    "oracle_capacity": "#0072B2",
    "oracle_capacity_first_order": "#56B4E9",
    "oracle_capacity_structural": "#009E73",
}
METRICS = (
    ("comp_total_wall_time_seconds", "Total wall time (s)"),
    ("comp_train_steps_per_second", "Training throughput (steps/s)"),
    ("comp_eval_steps_per_second", "Evaluation throughput (steps/s)"),
    ("comp_peak_rss_mb", "Peak RSS (MB)"),
    ("total_artifact_mb", "Persisted artifacts (MB)"),
    ("comp_precision_per_total_second", "Accuracy per total second"),
)


def _series_by_run(study_root: Path) -> dict[str, str]:
    series: dict[str, str] = {}
    for config_path in study_root.glob("*/run_config.json"):
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        cli = payload.get("cli_args") or {}
        method = str(cli.get("method") or payload.get("method") or "")
        value = str(cli.get("cscg_structural_preset") or "vanilla_cscg") if method == "cscg" else method
        series[str(payload.get("run_id") or config_path.parent.name)] = value
    return series


def _load_frame(study_root: Path) -> pl.DataFrame:
    frame = pl.read_parquet(study_root / "study_summary.parquet")
    series = _series_by_run(study_root)
    frame = frame.with_columns(
        pl.col("run_id").replace_strict(series, default=pl.col("method")).alias("series"),
        (pl.col("comp_total_artifact_bytes") / (1024.0 * 1024.0)).alias("total_artifact_mb"),
    )
    return frame.filter(pl.col("series").is_in(SERIES_ORDER))


def _summary(frame: pl.DataFrame) -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    for series in SERIES_ORDER:
        for level in sorted(frame.get_column("level").unique().to_list()):
            group = frame.filter((pl.col("series") == series) & (pl.col("level") == level))
            for metric, label in METRICS:
                values = np.asarray(group.get_column(metric).drop_nulls().to_list(), dtype=float)
                values = values[np.isfinite(values)]
                if not values.size:
                    continue
                ci95 = 0.0
                if values.size > 1:
                    ci95 = float(t.ppf(0.975, values.size - 1) * values.std(ddof=1) / np.sqrt(values.size))
                rows.append(
                    {
                        "series": series,
                        "series_label": SERIES_LABELS[series],
                        "level": int(level),
                        "metric": metric,
                        "metric_label": label,
                        "n": int(values.size),
                        "mean": float(values.mean()),
                        "sd": float(values.std(ddof=1)) if values.size > 1 else 0.0,
                        "ci95": ci95,
                    }
                )
    return pl.DataFrame(rows)


def _style() -> None:
    plt.style.use("default")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.edgecolor": "#374151",
            "axes.labelcolor": "#111827",
            "xtick.color": "#111827",
            "ytick.color": "#111827",
        }
    )


def _plot_profile(summary: pl.DataFrame, output_dir: Path) -> list[Path]:
    _style()
    fig, axes = plt.subplots(2, 3, figsize=(13.2, 7.4), dpi=220, constrained_layout=True)
    for axis, (metric, label) in zip(axes.ravel(), METRICS):
        metric_frame = summary.filter(pl.col("metric") == metric)
        for series in SERIES_ORDER:
            group = metric_frame.filter(pl.col("series") == series).sort("level")
            if group.is_empty():
                continue
            x = np.asarray(group["level"].to_list(), dtype=float)
            y = np.asarray(group["mean"].to_list(), dtype=float)
            err = np.asarray(group["ci95"].to_list(), dtype=float)
            axis.errorbar(
                x,
                y,
                yerr=err,
                label=SERIES_LABELS[series],
                color=SERIES_COLORS[series],
                marker="o",
                markersize=4.5,
                linewidth=1.7,
                capsize=2.5,
            )
        axis.set_title(label, fontsize=10.5, fontweight="bold")
        axis.set_xlabel("Maze level")
        axis.set_xticks(sorted(summary.get_column("level").unique().to_list()))
        axis.grid(axis="y", color="#D1D5DB", linewidth=0.7, alpha=0.75)
        axis.spines[["top", "right"]].set_visible(False)
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3, frameon=False, fontsize=9)
    fig.suptitle("Computational profile across maze complexity", fontsize=14, fontweight="bold")
    paths = [output_dir / "computational_profile.png", output_dir / "computational_profile.pdf"]
    for path in paths:
        fig.savefig(path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    return paths


def _plot_tradeoff(frame: pl.DataFrame, output_dir: Path) -> list[Path]:
    _style()
    levels = sorted(frame.get_column("level").unique().to_list())
    fig, axes = plt.subplots(1, len(levels), figsize=(13.2, 3.6), dpi=220, sharex=True, sharey=True, constrained_layout=True)
    for axis, level in zip(np.ravel(axes), levels):
        level_frame = frame.filter(pl.col("level") == level)
        for series in SERIES_ORDER:
            group = level_frame.filter(pl.col("series") == series).drop_nulls(
                ["comp_total_wall_time_seconds", "target_n1_accuracy"]
            )
            if group.is_empty():
                continue
            x = float(group.get_column("comp_total_wall_time_seconds").mean())
            y = float(group.get_column("target_n1_accuracy").mean())
            rss = float(group.get_column("comp_peak_rss_mb").mean())
            axis.scatter(x, y, s=max(30.0, rss * 0.22), color=SERIES_COLORS[series], alpha=0.9, edgecolor="white")
        axis.set_title(f"Level {level}", fontsize=10.5, fontweight="bold")
        axis.set_xlabel("Total wall time (s)")
        axis.grid(color="#D1D5DB", linewidth=0.7, alpha=0.75)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Next-state accuracy")
    handles = [
        plt.Line2D([], [], marker="o", linestyle="", color=SERIES_COLORS[s], label=SERIES_LABELS[s], markersize=6)
        for s in SERIES_ORDER
    ]
    fig.legend(handles=handles, loc="outside lower center", ncol=3, frameon=False, fontsize=9)
    fig.suptitle("Predictive accuracy versus computation", fontsize=14, fontweight="bold")
    paths = [output_dir / "accuracy_compute_tradeoff.png", output_dir / "accuracy_compute_tradeoff.pdf"]
    for path in paths:
        fig.savefig(path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", type=Path, default=Path("results/current/studies/main_study"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/current/notebook_plots/main_study/computational_metrics")
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame = _load_frame(args.study_root)
    summary = _summary(frame)
    csv_path = args.output_dir / "computational_metrics_summary.csv"
    summary.write_csv(csv_path)
    for path in [*_plot_profile(summary, args.output_dir), *_plot_tradeoff(frame, args.output_dir), csv_path]:
        print(path)


if __name__ == "__main__":
    main()
