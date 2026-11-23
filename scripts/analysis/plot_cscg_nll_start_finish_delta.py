from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t


LEVEL_LABELS = {13: "3x3", 20: "1x5", 21: "1x4", 22: "3x4"}
LEVEL_COLORS = {13: "#0072B2", 20: "#D55E00", 21: "#009E73", 22: "#CC79A7"}
LEVEL_MARKERS = {13: "o", 20: "s", 21: "^", 22: "D"}


def summarize_deltas(rows: pd.DataFrame, confidence: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {
        "run_id",
        "level",
        "seed",
        "batch_size",
        "em_nll_bps_initial",
        "em_nll_bps_final",
    }
    missing = sorted(required.difference(rows.columns))
    if missing:
        raise ValueError(f"Input is missing required columns: {', '.join(missing)}")

    by_seed = rows.copy()
    by_seed["level_label"] = by_seed["level"].map(LEVEL_LABELS).fillna(
        by_seed["level"].map(lambda value: f"Level {int(value)}")
    )
    by_seed["em_nll_bps_delta"] = (
        by_seed["em_nll_bps_initial"] - by_seed["em_nll_bps_final"]
    )

    summary = (
        by_seed.groupby(["level", "level_label", "batch_size"], sort=True)["em_nll_bps_delta"]
        .agg(["count", "mean", "std"])
        .reset_index()
        .rename(columns={"count": "n", "std": "sd"})
    )
    summary["sd"] = summary["sd"].fillna(0.0)
    summary["sem"] = summary["sd"] / np.sqrt(summary["n"].clip(lower=1))
    alpha = 1.0 - confidence
    summary["ci"] = [
        float(t.ppf(1.0 - alpha / 2.0, int(n) - 1) * sem) if int(n) > 1 else 0.0
        for n, sem in zip(summary["n"], summary["sem"])
    ]
    summary["ci_low"] = summary["mean"] - summary["ci"]
    summary["ci_high"] = summary["mean"] + summary["ci"]
    summary["confidence"] = confidence
    return by_seed, summary


def plot_deltas(summary: pd.DataFrame, output_dir: Path) -> list[Path]:
    batch_sizes = sorted(int(value) for value in summary["batch_size"].unique())
    levels = [level for level in LEVEL_LABELS if level in set(summary["level"])]
    levels.extend(
        int(level) for level in sorted(summary["level"].unique()) if level not in set(levels)
    )
    x = np.arange(len(batch_sizes), dtype=float)

    plt.style.use("default")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.edgecolor": "#374151",
            "axes.labelcolor": "#111827",
            "xtick.color": "#111827",
            "ytick.color": "#111827",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    figure, axis = plt.subplots(figsize=(7.2, 4.7), dpi=300)
    for level in levels:
        line = summary[summary["level"] == level].set_index("batch_size").reindex(batch_sizes)
        axis.errorbar(
            x,
            line["mean"].to_numpy(dtype=float),
            yerr=line["ci"].to_numpy(dtype=float),
            color=LEVEL_COLORS.get(level, "#4B5563"),
            marker=LEVEL_MARKERS.get(level, "o"),
            markersize=6.2,
            markeredgecolor="white",
            markeredgewidth=0.7,
            linewidth=2.0,
            elinewidth=1.1,
            capsize=3.0,
            label=LEVEL_LABELS.get(level, f"Level {level}"),
            zorder=3,
        )

    axis.axhline(0.0, color="#6B7280", linewidth=0.9, linestyle="--", zorder=1)
    axis.set_xticks(x, [str(value) for value in batch_sizes])
    axis.set_xlim(-0.16, len(batch_sizes) - 0.84)
    axis.set_xlabel("CSCG batch size, B")
    axis.set_ylabel("NLL reduction (initial − final, bits/symbol)")
    axis.grid(axis="y", color="#D1D5DB", linewidth=0.7, alpha=0.75)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.legend(
        title="Environment",
        loc="upper left",
        ncol=2,
        frameon=False,
        handlelength=2.2,
        columnspacing=1.5,
    )
    figure.suptitle("CSCG EM NLL reduction from start to finish", fontsize=13, fontweight="bold", y=0.98)
    figure.text(
        0.5,
        0.015,
        "Points show means across 30 seeds; error bars are 95% confidence intervals. Positive values indicate improvement.",
        ha="center",
        fontsize=8,
        color="#4B5563",
    )
    figure.subplots_adjust(left=0.13, right=0.98, bottom=0.18, top=0.88)

    output_dir.mkdir(parents=True, exist_ok=True)
    png = output_dir / "cscg_em_nll_start_finish_delta.png"
    pdf = output_dir / "cscg_em_nll_start_finish_delta.pdf"
    figure.savefig(png, bbox_inches="tight")
    figure.savefig(pdf, bbox_inches="tight")
    plt.close(figure)
    return [png, pdf]


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot the initial-to-final CSCG EM NLL reduction.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            "results/current/notebook_plots/cscg_batch_sweep/"
            "em_likelihood_trajectories/em_nll_trajectory_summary.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/current/notebook_plots/cscg_batch_sweep"),
    )
    parser.add_argument("--confidence", type=float, default=0.95)
    args = parser.parse_args()

    by_seed, summary = summarize_deltas(pd.read_csv(args.input), float(args.confidence))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    by_seed_path = args.output_dir / "cscg_em_nll_start_finish_delta_by_seed.csv"
    summary_path = args.output_dir / "cscg_em_nll_start_finish_delta_summary.csv"
    by_seed.to_csv(by_seed_path, index=False)
    summary.to_csv(summary_path, index=False)
    outputs = plot_deltas(summary, args.output_dir)
    for path in [by_seed_path, summary_path, *outputs]:
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
