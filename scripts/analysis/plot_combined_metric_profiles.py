#!/usr/bin/env python3
"""Render the three main-study compact metric profiles as one shared figure."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for import_root in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import polars as pl

from mercury_runs.analysis.plots import batch_summary_forest as forest
from scripts.analysis.study_selection import MAIN_STUDY_ROOT, PROTOCOL_ID


STUDY_ROOT = MAIN_STUDY_ROOT
OUTPUT_DIR = Path("results/current/notebook_plots/main_study/forest_matrix_profile")
OUTPUT_NAME = (
    "batch_40_checkpoints_2000_4000_final_selected_metric_profile_"
    "focused_transposed_compact_combined.png"
)

COMPARISONS = (
    ("checkpoint_2000", "(a) One pass", "mercury"),
    ("checkpoint_4000", "(b) Two passes", "hybrid"),
    ("final", "(c) Extensive", "hybrid"),
)
SERIES = (
    "target",
    "tmgwr_hybrid",
    "vanilla_cscg",
    "oracle_capacity",
    "oracle_capacity_first_order",
    "oracle_capacity_structural",
)
LABELS = {
    "target": "MERCURY",
    "tmgwr_hybrid": "TMGWR",
    "vanilla_cscg": "CSCG",
    "oracle_capacity": "Oracle-C",
    "oracle_capacity_first_order": "Oracle-C+1st",
    "oracle_capacity_structural": "Oracle-C+Struct",
}
# One deliberately high-separation palette shared by every checkpoint.  In
# particular, the four CSCG configurations no longer occupy adjacent blues.
COLORS = {
    "target": "#D55E00",
    "tmgwr_hybrid": "#E6A700",
    "vanilla_cscg": "#27313D",
    "oracle_capacity": "#0072B2",
    "oracle_capacity_first_order": "#CC79A7",
    "oracle_capacity_structural": "#009E73",
}


def _source_series(key: str, target_series: str) -> str:
    return target_series if key == "target" else key


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", type=Path, default=STUDY_ROOT)
    parser.add_argument("--protocol-id", default=PROTOCOL_ID)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    frame = forest._build_frame(
        args.study_root,
        args.protocol_id,
        protocol_ids_by_rand={0.1: args.protocol_id},
        batch_size=40,
    )
    if frame.is_empty():
        raise RuntimeError("No main-study evaluation records were found")

    plt.style.use("default")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.0,
            "axes.edgecolor": "#64748B",
            "axes.labelcolor": "#1F2933",
            "xtick.color": "#1F2933",
            "ytick.color": "#1F2933",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    row_specs = forest._transposed_row_specs(
        forest.PROFILE_FOCUSED_METRICS,
        include_nll_axis=False,
    )
    y_positions = np.arange(len(row_specs))[::-1].astype(float)
    row_y = {metric: y_positions[index] for index, (metric, _) in enumerate(row_specs)}
    method_offsets = forest._transposed_method_offsets(len(SERIES))

    figure, axes = plt.subplots(
        nrows=4,
        ncols=3,
        figsize=(7.25, 6.35),
        sharex=True,
        sharey=True,
        dpi=300,
        gridspec_kw={"hspace": 0.15, "wspace": 0.16},
    )

    for column, (comparison, caption, target_series) in enumerate(COMPARISONS):
        source = frame.filter(pl.col("comparison") == comparison)
        lookup = forest._summary_lookup(source)

        for row, level in enumerate(forest.LEVEL_ORDER):
            axis = axes[row, column]
            for metric_index, (metric, _) in enumerate(row_specs):
                base_y = row_y[metric]
                if metric_index > 0:
                    axis.axhline(
                        base_y + 0.5,
                        color="#D7DEE7",
                        linewidth=0.55,
                        zorder=0,
                    )

                for method_index, series_key in enumerate(SERIES):
                    series_name = _source_series(series_key, target_series)
                    result = lookup.get((level, series_name, metric))
                    if result is None:
                        continue
                    offset_y = base_y + float(method_offsets[method_index])
                    seed_values = (
                        source.filter(
                            (pl.col("level") == level)
                            & (pl.col("plot_series") == series_name)
                        )
                        .drop_nulls(["seed", metric])
                        .group_by("seed")
                        .agg(pl.col(metric).mean().alias("value"))
                        .sort("seed")
                    )
                    seed_offsets = np.linspace(-0.025, 0.025, len(seed_values))
                    color = COLORS[series_key]
                    axis.scatter(
                        seed_values["value"].to_numpy(),
                        offset_y + seed_offsets,
                        s=4.2,
                        color=color,
                        alpha=0.22,
                        linewidths=0,
                        zorder=1,
                    )
                    mean, half_width, _ = result
                    emphasized = method_index == 0
                    axis.errorbar(
                        mean,
                        offset_y,
                        xerr=half_width,
                        fmt="none",
                        color=color,
                        ecolor=color,
                        elinewidth=0.95 if emphasized else 0.65,
                        capsize=2.6,
                        capthick=0.95 if emphasized else 0.65,
                        zorder=3,
                    )
                    axis.scatter(
                        [mean],
                        [offset_y],
                        s=16.0 if emphasized else 10.6,
                        color=color,
                        edgecolors="none",
                        linewidths=0,
                        zorder=4,
                    )

            axis.set_xlim(0.0, 1.0)
            axis.set_xticks([0.0, 0.5, 1.0])
            axis.set_xticklabels(["0", "0.5", "1.0"])
            axis.set_ylim(y_positions[-1] - 0.48, y_positions[0] + 0.48)
            axis.set_axisbelow(True)
            axis.grid(
                axis="x",
                which="major",
                color="#D8DEE6",
                linewidth=0.55,
                alpha=0.8,
            )
            axis.tick_params(
                axis="x",
                labelsize=8.4,
                length=2.4,
                width=0.6,
                pad=1.8,
                labelbottom=row == len(forest.LEVEL_ORDER) - 1,
            )
            axis.tick_params(axis="y", length=0)
            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)
            axis.spines["left"].set_visible(False)
            axis.spines["bottom"].set_linewidth(0.6)
            if column == 0:
                axis.set_yticks(
                    y_positions,
                    [
                        forest.COMPACT_TRANSPOSED_METRIC_LABELS.get(metric, label)
                        for metric, label in row_specs
                    ],
                )
                axis.tick_params(axis="y", labelsize=8.7, pad=4.0)
            else:
                axis.tick_params(axis="y", labelleft=False, left=False)

    handles = [
        Line2D(
            [0],
            [0],
            color=COLORS[name],
            marker="o",
            linestyle="none",
            markersize=4.6,
            markeredgecolor="none",
            markeredgewidth=0.0,
            label=LABELS[name],
        )
        for name in SERIES
    ]
    figure.legend(
        handles,
        [handle.get_label() for handle in handles],
        loc="upper center",
        bbox_to_anchor=(0.52, 0.978),
        ncol=6,
        frameon=False,
        fontsize=8.4,
        handletextpad=0.30,
        columnspacing=0.95,
        borderaxespad=0.0,
    )
    figure.supxlabel("Metric value", fontsize=9.8, y=0.073)
    figure.subplots_adjust(left=0.125, right=0.992, top=0.925, bottom=0.145)

    # Calculate these only after subplot adjustment so every label is centered
    # against the final exported environment row rather than the default grid.
    figure.canvas.draw()
    environment_x = axes[0, 0].get_position().x0 - 0.088
    for row, level in enumerate(forest.LEVEL_ORDER):
        position = axes[row, 0].get_position()
        figure.text(
            environment_x,
            (position.y0 + position.y1) / 2,
            forest.LEVEL_LABELS[level].replace("x", "×"),
            rotation=90,
            rotation_mode="anchor",
            va="center",
            ha="center",
            fontsize=10.2,
            fontweight="bold",
            color="#27313D",
        )

    # Subfigure captions sit below the shared x-axis label.
    for column, (_, caption, _) in enumerate(COMPARISONS):
        position = axes[-1, column].get_position()
        figure.text(
            (position.x0 + position.x1) / 2,
            0.018,
            caption,
            ha="center",
            va="bottom",
            fontsize=12.6,
            fontweight="bold",
            color="#111827",
        )

    # Subtle separators clarify the three checkpoint figures without boxing them in.
    figure.canvas.draw()
    for left_column in (0, 1):
        left = axes[0, left_column].get_position()
        right = axes[0, left_column + 1].get_position()
        x = (left.x1 + right.x0) / 2
        figure.add_artist(
            Line2D(
                [x, x],
                [axes[-1, 0].get_position().y0, axes[0, 0].get_position().y1],
                transform=figure.transFigure,
                color="#CBD3DD",
                linewidth=0.65,
                alpha=0.8,
                zorder=0,
            )
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    png_path = args.output_dir / OUTPUT_NAME
    figure.savefig(png_path, bbox_inches="tight", pad_inches=0.035, dpi=600)
    figure.savefig(png_path.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.035)
    plt.close(figure)
    print(png_path)


if __name__ == "__main__":
    main()
