#!/usr/bin/env python3
"""Compact combined metric profile with environments encoded by marker shape."""
from __future__ import annotations

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
from mercury_runs.application.evaluation_protocol import EvaluationProtocol
from scripts.analysis.plot_combined_metric_profiles import (
    COLORS,
    COMPARISONS,
    LABELS,
    OUTPUT_DIR,
    SERIES,
    STUDY_ROOT,
    _source_series,
)


OUTPUT_STEM = (
    "batch_40_checkpoints_2000_4000_final_selected_metric_profile_"
    "focused_transposed_compact_combined_environment_shapes"
)
ENVIRONMENT_MARKERS = {13: "o", 21: "s", 20: "^", 22: "D"}
ENVIRONMENT_OFFSETS = {13: -0.057, 21: -0.019, 20: 0.019, 22: 0.057}


def _configure_style() -> None:
    plt.style.use("default")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.0,
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


def main() -> None:
    protocol = EvaluationProtocol(
        sensor="cardinal distance",
        sensor_range=1,
        rand_prob=0.1,
        evaluation_seed=0,
        number_of_walks=100,
        walk_length=10000,
        edge_thresholds=[1e-6],
    )
    frame = forest._build_frame(
        STUDY_ROOT,
        protocol.protocol_id(),
        protocol_ids_by_rand={0.1: protocol.protocol_id()},
        batch_size=40,
    )
    if frame.is_empty():
        raise RuntimeError("No main-study evaluation records were found")

    _configure_style()
    row_specs = forest._transposed_row_specs(
        forest.PROFILE_FOCUSED_METRICS,
        include_nll_axis=False,
    )
    _render(frame, row_specs, COMPARISONS)
    for comparison, suffix in zip(COMPARISONS, ("one_pass", "two_passes", "extensive")):
        _render(frame, row_specs, (comparison,), suffix=suffix)
        _render(frame, row_specs, (comparison,), suffix=f"{suffix}_no_oracles", no_oracles=True)


def _render(
    frame, row_specs, comparisons, *, suffix: str | None = None, no_oracles: bool = False
) -> None:
    standalone = suffix is not None
    series = SERIES[:3] if no_oracles else SERIES
    scale = 1.25 if no_oracles else 1.0
    font_scale = 0.5625 if no_oracles else 1.0
    point_scale = 1.5 if no_oracles else 1.0
    y_positions = np.arange(len(row_specs))[::-1].astype(float)
    row_y = {metric: y_positions[index] for index, (metric, _) in enumerate(row_specs)}
    # Reserve a distinct vertical lane for every method.  The four environment
    # offsets then remain inside that lane rather than colliding with the next
    # method's markers.
    method_limit = 0.36 if no_oracles else 0.38
    method_offsets = np.linspace(method_limit, -method_limit, len(series))
    environment_spread = 1.5 if no_oracles else 1.0

    figure, axes = plt.subplots(
        nrows=1,
        ncols=len(comparisons),
        figsize=(5.0, 3.8) if no_oracles else (
            (6.5, 5.5) if standalone else (8.15, 5.06)
        ),
        sharex=True,
        sharey=True,
        dpi=300,
        gridspec_kw={"wspace": 0.16},
    )
    axes = np.atleast_1d(axes)

    for column, (comparison, caption, target_series) in enumerate(comparisons):
        axis = axes[column]
        source = frame.filter(pl.col("comparison") == comparison)
        lookup = forest._summary_lookup(source)

        for metric_index, (metric, _) in enumerate(row_specs):
            base_y = row_y[metric]
            if metric_index > 0:
                axis.axhline(base_y + 0.5, color="#D7DEE7", linewidth=0.55 * scale, zorder=0)

            for method_index, series_key in enumerate(series):
                series_name = _source_series(series_key, target_series)
                color = COLORS[series_key]
                method_y = base_y + float(method_offsets[method_index])

                for level in forest.LEVEL_ORDER:
                    result = lookup.get((level, series_name, metric))
                    if result is None:
                        continue
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
                    environment_y = method_y + ENVIRONMENT_OFFSETS[level] * environment_spread
                    seed_offsets = np.linspace(-0.0035, 0.0035, len(seed_values))
                    marker = ENVIRONMENT_MARKERS[level]
                    axis.scatter(
                        seed_values["value"].to_numpy(),
                        environment_y + seed_offsets,
                        marker=marker,
                        s=5.0 * scale**2 * point_scale,
                        color=color,
                        alpha=0.16,
                        linewidths=0,
                        zorder=1,
                    )

                    mean, half_width, _ = result
                    emphasized = method_index == 0
                    axis.errorbar(
                        mean,
                        environment_y,
                        xerr=half_width,
                        fmt="none",
                        color=color,
                        ecolor=color,
                        elinewidth=(0.9 if emphasized else 0.65) * scale,
                        capsize=2.2 * scale,
                        capthick=(0.9 if emphasized else 0.65) * scale,
                        zorder=3,
                    )
                    axis.scatter(
                        [mean],
                        [environment_y],
                        marker=marker,
                        s=(19.0 if emphasized else 12.5) * scale**2 * point_scale,
                        color=color,
                        edgecolors="white",
                        linewidths=0.45 * scale,
                        zorder=4,
                    )

        if no_oracles:
            axis.set_xlim(-0.025, 1.04)
        else:
            axis.set_xlim(0.0, 1.0)
        axis.set_xticks([0.0, 0.5, 1.0], ["0", "0.5", "1.0"])
        y_padding = 0.54 if no_oracles else 0.48
        axis.set_ylim(y_positions[-1] - y_padding, y_positions[0] + y_padding)
        axis.set_axisbelow(True)
        axis.grid(axis="x", color="#D8DEE6", linewidth=0.55 * scale, alpha=0.8)
        axis.tick_params(
            axis="x", labelsize=11.0 * scale * font_scale if standalone else 8.2,
            length=2.4 * scale, width=0.6 * scale, pad=1.8 * scale,
        )
        axis.tick_params(axis="y", length=0)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.spines["left"].set_visible(False)
        axis.spines["bottom"].set_linewidth(0.6 * scale)
        if column == 0:
            axis.set_yticks(
                y_positions,
                [
                    forest.COMPACT_TRANSPOSED_METRIC_LABELS.get(metric, label)
                    for metric, label in row_specs
                ],
            )
            axis.tick_params(
                axis="y", labelsize=11.5 * scale * font_scale if standalone else 8.7,
                pad=4.0 * scale,
            )
        else:
            axis.tick_params(axis="y", labelleft=False, left=False)

        if not standalone:
            axis.set_title(
                caption,
                fontsize=10.2,
                fontweight="semibold",
                color="#111827",
                pad=5.0,
            )

    method_handles = [
        Line2D(
            [0], [0], marker="o", linestyle="none", markersize=4.8 * scale,
            markerfacecolor=COLORS[name], markeredgecolor="none", label=LABELS[name],
        )
        for name in series
    ]
    # Matplotlib fills multi-row legends down columns; reorder so the visible
    # rows follow the manuscript method order from left to right.
    method_order = (
        tuple(range(len(series))) if no_oracles else
        ((0, 2, 4, 1, 3, 5) if standalone else (0, 3, 1, 4, 2, 5))
    )
    method_handles = [method_handles[index] for index in method_order]
    environment_handles = [
        Line2D(
            [0], [0], marker=ENVIRONMENT_MARKERS[level], linestyle="none",
            markersize=4.8 * scale, markerfacecolor="#64748B", markeredgecolor="white",
            label=forest.LEVEL_LABELS[level].replace("x", "×"),
        )
        for level in forest.LEVEL_ORDER
    ]
    environment_handles = [environment_handles[index] for index in (0, 2, 1, 3)]
    if no_oracles:
        heading_size = 11.0 * scale * font_scale
        figure.text(
            0.11, 0.91, "Methods", ha="left", va="center",
            fontsize=heading_size, fontweight="semibold", color="#475569",
        )
        figure.legend(
            handles=method_handles, loc="center left", bbox_to_anchor=(0.21, 0.91),
            ncol=1, frameon=False,
            fontsize=10.5 * scale * font_scale,
            handletextpad=0.30,
            columnspacing=0.80, labelspacing=0.45, borderaxespad=0.0,
        )
        figure.text(
            0.52, 0.91, "Environments", ha="left", va="center",
            fontsize=heading_size, fontweight="semibold", color="#475569",
        )
        figure.legend(
            handles=environment_handles, loc="center left",
            bbox_to_anchor=(0.73, 0.91), ncol=2, frameon=False,
            fontsize=10.5 * scale * font_scale,
            handletextpad=0.30,
            columnspacing=0.85, labelspacing=0.45, borderaxespad=0.0,
        )
        figure.add_artist(
            Line2D(
                [0.50, 0.50], [0.83, 0.98], transform=figure.transFigure,
                color="#D7DEE7", linewidth=0.65 * scale,
            )
        )
    elif standalone:
        figure.legend(
            handles=method_handles, title="Methods", loc="upper center",
            bbox_to_anchor=(0.32, 0.87), ncol=2, frameon=False,
            fontsize=10.5 * scale * font_scale,
            title_fontsize=11.0 * scale * font_scale, handletextpad=0.30,
            columnspacing=0.80, labelspacing=0.45, borderaxespad=0.0,
        )
        figure.legend(
            handles=environment_handles, title="Environments", loc="upper center",
            bbox_to_anchor=(0.82, 0.87), ncol=2, frameon=False,
            fontsize=10.5 * scale * font_scale,
            title_fontsize=11.0 * scale * font_scale, handletextpad=0.30,
            columnspacing=0.85, labelspacing=0.45, borderaxespad=0.0,
        )
        figure.add_artist(
            Line2D(
                [0.65, 0.65], [0.745, 0.86], transform=figure.transFigure,
                color="#D7DEE7", linewidth=0.65 * scale,
            )
        )
    else:
        method_legend = figure.legend(
            handles=method_handles,
            loc="upper center",
            bbox_to_anchor=(0.37, 0.966),
            ncol=3,
            frameon=False,
            fontsize=8.8,
            handletextpad=0.30,
            columnspacing=0.80,
            labelspacing=0.45,
            borderaxespad=0.0,
        )
        figure.add_artist(method_legend)
        figure.legend(
            handles=environment_handles,
            loc="upper center",
            bbox_to_anchor=(0.88, 0.966),
            ncol=2,
            frameon=False,
            fontsize=8.8,
            handletextpad=0.30,
            columnspacing=0.85,
            labelspacing=0.45,
            borderaxespad=0.0,
        )
        figure.text(
            0.018, 0.925, "Methods", ha="left", va="center", fontsize=8.8,
            fontweight="semibold", color="#475569",
        )
        figure.text(
            0.655, 0.925, "Environments", ha="left", va="center", fontsize=8.8,
            fontweight="semibold", color="#475569",
        )
        figure.add_artist(
            Line2D(
                [0.64, 0.64], [0.885, 0.975], transform=figure.transFigure,
                color="#D7DEE7", linewidth=0.65,
            )
        )
    figure.supxlabel(
        "Metric value", fontsize=11.0 * scale * font_scale if standalone else 8.2,
        x=0.555 if no_oracles else 0.5, y=0.015 if no_oracles else 0.045,
    )
    # Shift the unchanged-width grid into the unused left margin.  The larger
    # canvas scales both dimensions together, preserving the figure aspect.
    if standalone:
        figure.subplots_adjust(
            left=0.16 if no_oracles else 0.15,
            right=0.95 if no_oracles else 0.97,
            top=0.82 if no_oracles else 0.72,
            bottom=0.10 if no_oracles else 0.14,
        )
    else:
        figure.subplots_adjust(left=0.080, right=0.967, top=0.80, bottom=0.14)

    figure.canvas.draw()
    for left_column in range(len(comparisons) - 1):
        left = axes[left_column].get_position()
        right = axes[left_column + 1].get_position()
        x = (left.x1 + right.x0) / 2
        figure.add_artist(
            Line2D(
                [x, x],
                [axes[0].get_position().y0, axes[0].get_position().y1],
                transform=figure.transFigure,
                color="#CBD3DD",
                linewidth=0.65,
                alpha=0.8,
                zorder=0,
            )
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = OUTPUT_STEM if suffix is None else f"{OUTPUT_STEM}_{suffix}"
    png_path = OUTPUT_DIR / f"{stem}.png"
    figure.savefig(png_path, bbox_inches="tight", pad_inches=0.035, dpi=600)
    figure.savefig(png_path.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.035)
    plt.close(figure)
    print(png_path)


if __name__ == "__main__":
    main()
