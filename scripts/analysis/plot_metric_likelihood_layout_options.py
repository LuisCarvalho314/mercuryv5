"""Render alternative layouts of the evaluation-v5 metric/likelihood figure.

The plots use the same condition means and checkpoint conventions as
plot_pass_2_to_1000_metric_delta.py, without regenerating its other outputs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd

from scripts.analysis.plot_config_metric_trajectories import (
    DEFAULT_LEVEL_ORDER,
    load_rows,
)
from scripts.analysis.plot_pass_2_to_1000_metric_delta import (
    COMPACT_SERIES_LABELS,
    PRIMARY_METRIC,
    SECONDARY_METRIC,
    SERIES_COLORS,
    SERIES_ORDER,
    three_checkpoint_seed_frame,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ANALYSIS_DIR = PROJECT_ROOT / (
    "results/current/action_alignment_rerun/notebook_plots/main_study_eval_v5/"
    "pass_2_to_1000_statistical_analysis"
)
DEFAULT_OUTPUT_DIR = ANALYSIS_DIR / "metric_likelihood_layout_options"
MANIFEST = PROJECT_ROOT / (
    "results/current/action_alignment_rerun/notebook_plots/main_study_eval_v5/"
    "evaluation_v5_manifest.json"
)
LEVEL_LABELS = {13: "3×3", 21: "1×4", 20: "1×5", 22: "3×4"}
METRICS = (
    ("latent_precision", r"Latent purity, $P$"),
    ("latent_recall", r"Latent concentration, $C$"),
    (PRIMARY_METRIC, r"Transition similarity, $S^A_{\mathrm{TV}}$"),
    (
        SECONDARY_METRIC,
        r"No-self transition similarity, $S^{A,\neg\mathrm{self}}_{\mathrm{TV}}$",
    ),
)
STAGES = ("initialisation", "pass_1", "pass_2", "pass_1000")
PASS_1_SERIES = {
    "vanilla_cscg",
    "oracle_capacity",
    "oracle_capacity_first_order",
    "oracle_capacity_structural",
}
SOURCE_PRETRAINED_SERIES = {"mercury_cscg_hybrid", "tmgwr_cscg_hybrid"}

# The fixed pairing 3×3/1×5 and 1×4/3×4 gave 40 close same-method,
# same-stage points across the four metrics (distance < 0.025 after scaling
# each metric's x and y ranges), versus 43 for scale pairing and 59 for
# dimensionality pairing. Its advantage over scale pairing is modest.
LAYOUTS = {
    "family_4x2": {
        "columns": ((21, 20), (13, 22)),
        "titles": ("1D corridors", "2D environments"),
        "metric_overlay": False,
        "width": 11.2,
    },
    "corridors_4x3": {
        "columns": ((13,), (21, 20), (22,)),
        "titles": ("3×3", "1D corridors: 1×4 / 1×5", "3×4"),
        "metric_overlay": False,
        "width": 14.7,
    },
    "metrics_2x4": {
        "columns": tuple((level,) for level in DEFAULT_LEVEL_ORDER),
        "titles": tuple(LEVEL_LABELS[level] for level in DEFAULT_LEVEL_ORDER),
        "metric_overlay": True,
        "width": 15.6,
    },
    "metrics_refined_2x4": {
        "columns": tuple((level,) for level in DEFAULT_LEVEL_ORDER),
        "titles": tuple(LEVEL_LABELS[level] for level in DEFAULT_LEVEL_ORDER),
        "metric_overlay": True,
        "refined": True,
        "width": 15.6,
    },
    "metrics_refined_half_page_2x4": {
        "columns": tuple((level,) for level in DEFAULT_LEVEL_ORDER),
        "titles": tuple(LEVEL_LABELS[level] for level in DEFAULT_LEVEL_ORDER),
        "metric_overlay": True,
        "refined": True,
        "half_page": True,
        "width": 7.16,
    },
    "metrics_structured_half_page_2x4": {
        "columns": tuple((level,) for level in DEFAULT_LEVEL_ORDER),
        "titles": tuple(LEVEL_LABELS[level] for level in DEFAULT_LEVEL_ORDER),
        "metric_overlay": True,
        "refined": True,
        "half_page": True,
        "structured_legend": True,
        "width": 7.16,
    },
    "metrics_compact_grouped_half_page_2x4": {
        "columns": tuple((level,) for level in DEFAULT_LEVEL_ORDER),
        "titles": tuple(LEVEL_LABELS[level] for level in DEFAULT_LEVEL_ORDER),
        "metric_overlay": True,
        "refined": True,
        "half_page": True,
        "compact_grouped": True,
        "width": 7.16,
    },
    "metrics_main_style_half_page_2x4": {
        "columns": tuple((level,) for level in DEFAULT_LEVEL_ORDER),
        "titles": tuple(LEVEL_LABELS[level] for level in DEFAULT_LEVEL_ORDER),
        "metric_overlay": True,
        "refined": True,
        "half_page": True,
        "compact_grouped": True,
        "main_style": True,
        "width": 7.16,
    },
    "metrics_main_style_final_half_page_2x4": {
        "columns": tuple((level,) for level in DEFAULT_LEVEL_ORDER),
        "titles": tuple(LEVEL_LABELS[level] for level in DEFAULT_LEVEL_ORDER),
        "metric_overlay": True,
        "refined": True,
        "half_page": True,
        "compact_grouped": True,
        "main_style": True,
        "final_tuning": True,
        "width": 7.16,
    },
    "scale_4x2": {
        "columns": ((13, 21), (20, 22)),
        "titles": ("Smaller environments", "Larger environments"),
        "metric_overlay": False,
        "width": 11.2,
    },
    "least_overlap_4x2": {
        "columns": ((13, 20), (21, 22)),
        "titles": ("3×3 and 1×5", "1×4 and 3×4"),
        "metric_overlay": False,
        "width": 11.2,
    },
}


def _condition_means(
    rows: pd.DataFrame, *, matched_budget: bool = False
) -> dict[str, pd.DataFrame]:
    return {
        metric: three_checkpoint_seed_frame(
            rows,
            metric=metric,
            pass_1_step=2_000,
            allow_missing_pass_1_series=(
                {"mercury_cscg_hybrid", "tmgwr_cscg_hybrid"} if matched_budget else None
            ),
        ).groupby(["series", "level"], sort=False).mean(numeric_only=True)
        for metric, _ in METRICS
    }


def _row_metrics(metric_overlay: bool) -> tuple[tuple[str, ...], ...]:
    if metric_overlay:
        return ((METRICS[0][0], METRICS[1][0]), (METRICS[2][0], METRICS[3][0]))
    return tuple((metric,) for metric, _ in METRICS)


def _axis_limits(
    means: dict[str, pd.DataFrame], columns: tuple[tuple[int, ...], ...],
    row_metrics: tuple[tuple[str, ...], ...],
) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    x_limits = []
    for levels in columns:
        values = [
            float(row[f"nll_{stage}"])
            for frame in means.values()
            for (series, level), row in frame.iterrows()
            if int(level) in levels
            for stage in STAGES
            if stage != "pass_1" or str(series) in PASS_1_SERIES
        ]
        upper = max(values)
        x_limits.append((-0.025 * upper, 1.045 * upper))
    y_limits = []
    for metrics in row_metrics:
        values = [
            float(row[f"quality_{stage}"])
            for metric in metrics
            for (series, _), row in means[metric].iterrows()
            for stage in STAGES
            if stage != "pass_1" or str(series) in PASS_1_SERIES
        ]
        padding = max(0.012, 0.035 * (max(values) - min(values)))
        lower = min(values) - padding
        upper = max(values) + padding
        y_limits.append((lower, upper))
    return x_limits, y_limits


def _draw_trajectory(
    axis: plt.Axes, row: pd.Series, colour: str, marker: str, show_pass_1: bool,
    *, source_pretrained: bool = False, marker_scale: float = 1.0, half_page: bool = False,
    compact_grouped: bool = False, main_style: bool = False,
) -> None:
    stages = ["initialisation"]
    if show_pass_1:
        stages.append("pass_1")
    stages.extend(("pass_2", "pass_1000"))
    points = [
        (float(row[f"nll_{stage}"]), float(row[f"quality_{stage}"]))
        for stage in stages
    ]
    for start, end in zip(points, points[1:]):
        axis.annotate(
            "", xy=end, xytext=start,
            arrowprops={
                "arrowstyle": "-|>", "color": colour,
                "linewidth": 0.55 if main_style else
                (0.58 if compact_grouped else (0.72 if half_page else 1.0)),
                "alpha": 0.79,
                "mutation_scale": 3.3 if main_style else (3.7 if compact_grouped else
                (5.5 if half_page else 7.0)),
                "shrinkA": 1.6 if compact_grouped else (2.0 if half_page else 2.5),
                "shrinkB": 1.6 if compact_grouped else (2.0 if half_page else 2.5),
            },
            zorder=3,
        )
    for stage, (x_value, y_value) in zip(stages, points):
        checkpoint = "pass_1" if source_pretrained and stage == "initialisation" else stage
        if checkpoint == "initialisation":
            facecolour, alpha, size, line_width = "white", 0.95, 28, 0.95
            edgecolour = colour
        else:
            facecolour = colour
            alpha = {"pass_1": 0.28, "pass_2": 0.48, "pass_1000": 0.95}[checkpoint]
            size = 29 if checkpoint == "pass_1000" else 27
            line_width = 0.5
            edgecolour = "white"
        axis.scatter(
            x_value, y_value, marker=marker,
            s=size * marker_scale * (
                0.8 if main_style and checkpoint == "initialisation" else
                (0.85 if compact_grouped and checkpoint == "initialisation" else 1.0)
            ),
            facecolors=facecolour, edgecolors=edgecolour,
            linewidths=line_width, alpha=alpha, zorder=4,
        )


def _legends(figure: plt.Figure, layout: dict, axes: np.ndarray) -> None:
    refined = layout.get("refined", False)
    half_page = layout.get("half_page", False)
    if layout.get("main_style", False):
        _main_style_legends(figure, final_tuning=layout.get("final_tuning", False))
        return
    if layout.get("compact_grouped", False):
        _compact_grouped_legends(figure)
        return
    if layout.get("structured_legend", False):
        _structured_legends(figure, axes)
        return
    method_handles = [
        Line2D(
            [0], [0], marker="o", linestyle="none",
            markersize=4.6 if half_page else 5.2,
            markerfacecolor=SERIES_COLORS[series], markeredgecolor="white",
            label=COMPACT_SERIES_LABELS.get(series, series),
        )
        for series in SERIES_ORDER
    ]
    figure.legend(
        handles=method_handles, loc="upper center",
        bbox_to_anchor=(0.5, 0.993 if half_page else (0.988 if refined else 0.995)),
        ncol=6, frameon=False, fontsize=7.0 if half_page else 8.0,
        handletextpad=0.25 if half_page else 0.35,
        columnspacing=0.6 if half_page else 0.9,
    )
    stage_handles = [
        Line2D(
            [0], [0], marker="o", linestyle="none",
            markersize=4.6 if half_page else 5.2,
            markerfacecolor=fill, markeredgecolor=edge, alpha=alpha, label=label,
        )
        for label, fill, edge, alpha in (
            ("Initialisation", "white", "#64748B", 1.0),
            ("Pass 1", "#94A3B8", "white", 0.38),
            ("Pass 2", "#94A3B8", "white", 0.65),
            ("Pass 1000", "#64748B", "white", 1.0),
        )
    ]
    figure.legend(
        handles=stage_handles, loc="upper center",
        bbox_to_anchor=(0.5, 0.953 if half_page else (0.955 if refined else 0.958)),
        ncol=4, frameon=False, fontsize=6.9 if half_page else 7.8,
        handletextpad=0.25 if half_page else 0.35,
        columnspacing=0.7 if half_page else 1.0,
    )
    if refined:
        row_keys = (
            ("Latent representation", (("o", r"$P$"), ("s", r"$C$"))),
            (
                "Transition structure",
                (("^", r"$S^A_{\mathrm{TV}}$"),
                 ("D", r"$S^{A,\neg\mathrm{self}}_{\mathrm{TV}}$")),
            ),
        )
        for row_index, (title, symbols) in enumerate(row_keys):
            key_y = axes[row_index, 0].get_position().y1 + (
                (0.068 if row_index == 0 else 0.036)
                if half_page else (0.042 if row_index == 0 else 0.033)
            )
            figure.text(
                0.07 if half_page else 0.075, key_y, title,
                ha="left", va="center",
                fontsize=7.3 if half_page else 9.2,
                fontweight="semibold",
            )
            handles = [
                Line2D([0], [0], marker=marker, linestyle="none",
                       markersize=5.0 if half_page else 6.1,
                       markerfacecolor="#64748B", markeredgecolor="white", label=label)
                for marker, label in symbols
            ]
            figure.legend(
                handles=handles, loc="center left",
                bbox_to_anchor=(0.25 if half_page else 0.215, key_y),
                ncol=2, frameon=False,
                fontsize=7.1 if half_page else 8.9, borderaxespad=0,
                handletextpad=0.25 if half_page else 0.35,
                columnspacing=0.8 if half_page else 1.15,
            )
        return
    if layout["metric_overlay"]:
        handles = [
            Line2D([0], [0], marker=marker, linestyle="none", markersize=5.2,
                   markerfacecolor="#64748B", markeredgecolor="white", label=label)
            for marker, label in (
                ("o", r"$P$ / $S^A_{\mathrm{TV}}$"),
                ("s", r"$C$ / $S^{A,\neg\mathrm{self}}_{\mathrm{TV}}$"),
            )
        ]
    else:
        handles = [
            Line2D([0], [0], marker=marker, linestyle="none", markersize=5.2,
                   markerfacecolor="#64748B", markeredgecolor="white",
                   label=LEVEL_LABELS[level])
            for levels in layout["columns"] if len(levels) > 1
            for marker, level in zip(("o", "s"), levels)
        ]
    if handles:
        figure.legend(
            handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.923),
            ncol=len(handles), frameon=False, fontsize=7.8,
            handletextpad=0.35, columnspacing=1.0,
        )


def _structured_legends(figure: plt.Figure, axes: np.ndarray) -> None:
    figure.text(0.11, 0.982, "Methods", ha="left", va="top",
                fontsize=7.1, fontweight="bold")
    figure.text(0.67, 0.982, "Optimisation stage", ha="left", va="top",
                fontsize=7.1, fontweight="bold")
    figure.add_artist(Line2D(
        [0.645, 0.645], [0.902, 0.985], transform=figure.transFigure,
        color="#D7DEE7", linewidth=0.5,
    ))
    method_handles = [
        Line2D([0], [0], marker="o", linestyle="none", markersize=4.5,
               markerfacecolor=SERIES_COLORS[series], markeredgecolor="white",
               label=COMPACT_SERIES_LABELS.get(series, series))
        for series in SERIES_ORDER
    ]
    # Matplotlib fills a multirow legend by column; reorder for reading order.
    method_handles = [method_handles[index] for index in (0, 3, 1, 4, 2, 5)]
    figure.legend(
        handles=method_handles, loc="upper left", bbox_to_anchor=(0.108, 0.958),
        ncol=3, frameon=False, fontsize=6.5, borderaxespad=0,
        handletextpad=0.25, columnspacing=0.7, labelspacing=0.28,
    )
    stage_handles = [
        Line2D([0], [0], marker="o", linestyle="none", markersize=4.6,
               markerfacecolor=fill, markeredgecolor=edge, alpha=alpha, label=label)
        for label, fill, edge, alpha in (
            ("Initialisation", "white", "#64748B", 1.0),
            ("Pass 1", "#94A3B8", "white", 0.38),
            ("Pass 2", "#94A3B8", "white", 0.65),
            ("Pass 1000", "#64748B", "white", 1.0),
        )
    ]
    stage_handles = [stage_handles[index] for index in (0, 2, 1, 3)]
    figure.legend(
        handles=stage_handles, loc="upper left", bbox_to_anchor=(0.668, 0.958),
        ncol=2, frameon=False, fontsize=6.5, borderaxespad=0,
        handletextpad=0.25, columnspacing=0.55, labelspacing=0.28,
    )
    row_symbols = (
        (("o", r"$P$"), ("s", r"$C$")),
        (("^", r"$S^A_{\mathrm{TV}}$"),
         ("D", r"$S^{A,\neg\mathrm{self}}_{\mathrm{TV}}$")),
    )
    for row_index, symbols in enumerate(row_symbols):
        key_y = axes[row_index, 0].get_position().y1 + (
            0.045 if row_index == 0 else 0.035
        )
        handles = [
            Line2D([0], [0], marker=marker, linestyle="none", markersize=4.8,
                   markerfacecolor="#64748B", markeredgecolor="white", label=label)
            for marker, label in symbols
        ]
        figure.legend(
            handles=handles, loc="center left", bbox_to_anchor=(0.11, key_y),
            ncol=2, frameon=False, fontsize=6.7, borderaxespad=0,
            handletextpad=0.25, columnspacing=0.7,
        )


def _compact_grouped_legends(figure: plt.Figure) -> None:
    groups = (("Methods", 0.11), ("Optimisation stage", 0.535), ("Metrics", 0.785))
    for heading, x_position in groups:
        figure.text(x_position, 0.985, heading, ha="left", va="top",
                    fontsize=6.7, fontweight="bold")
    for x_position in (0.515, 0.765):
        figure.add_artist(Line2D(
            [x_position, x_position], [0.908, 0.987],
            transform=figure.transFigure, color="#D7DEE7", linewidth=0.45,
        ))
    method_handles = [
        Line2D([0], [0], marker="o", linestyle="none", markersize=3.5,
               markerfacecolor=SERIES_COLORS[series], markeredgecolor="white",
               label=COMPACT_SERIES_LABELS.get(series, series))
        for series in SERIES_ORDER
    ]
    method_handles = [method_handles[index] for index in (0, 3, 1, 4, 2, 5)]
    figure.legend(
        handles=method_handles, loc="upper left", bbox_to_anchor=(0.108, 0.956),
        ncol=3, frameon=False, fontsize=5.9, borderaxespad=0,
        handletextpad=0.2, columnspacing=0.5, labelspacing=0.24,
    )
    stage_handles = [
        Line2D([0], [0], marker="o", linestyle="none", markersize=3.5,
               markerfacecolor=fill, markeredgecolor=edge, alpha=alpha, label=label)
        for label, fill, edge, alpha in (
            ("Initialisation", "white", "#64748B", 1.0),
            ("Pass 1", "#94A3B8", "white", 0.38),
            ("Pass 2", "#94A3B8", "white", 0.65),
            ("Pass 1000", "#64748B", "white", 1.0),
        )
    ]
    stage_handles = [stage_handles[index] for index in (0, 2, 1, 3)]
    figure.legend(
        handles=stage_handles, loc="upper left", bbox_to_anchor=(0.533, 0.956),
        ncol=2, frameon=False, fontsize=5.9, borderaxespad=0,
        handletextpad=0.2, columnspacing=0.4, labelspacing=0.24,
    )
    metric_handles = [
        Line2D([0], [0], marker=marker, linestyle="none", markersize=3.5,
               markerfacecolor="#64748B", markeredgecolor="white", label=label)
        for marker, label in (
            ("o", r"$P$"), ("s", r"$C$"),
            ("^", r"$S^A_{\mathrm{TV}}$"),
            ("D", r"$S^{A,\neg\mathrm{self}}_{\mathrm{TV}}$"),
        )
    ]
    metric_handles = [metric_handles[index] for index in (0, 2, 1, 3)]
    figure.legend(
        handles=metric_handles, loc="upper left", bbox_to_anchor=(0.783, 0.956),
        ncol=2, frameon=False, fontsize=5.9, borderaxespad=0,
        handletextpad=0.2, columnspacing=0.35, labelspacing=0.24,
    )


def _main_style_legends(figure: plt.Figure, *, final_tuning: bool = False) -> None:
    headings = (
        (("Conditions", 0.025), ("Metrics", 0.465), ("Checkpoints", 0.725))
        if final_tuning else
        (("Conditions", 0.29), ("Metrics", 0.64), ("Checkpoints", 0.88))
    )
    for heading, x_position in headings:
        figure.text(
            x_position, 0.947 if final_tuning else 0.989, heading,
            ha="left" if final_tuning else "center",
            va="center" if final_tuning else "top",
            fontsize=7.0 if final_tuning else 6.7,
            fontweight="semibold", color="#475569",
        )
    for x_position in ((0.45, 0.71) if final_tuning else (0.515, 0.765)):
        figure.add_artist(Line2D(
            [x_position, x_position], [0.900 if final_tuning else 0.910, 0.992],
            transform=figure.transFigure,
            color="#D7DEE7", linewidth=0.5,
        ))
    condition_handles = [
        Line2D([0], [0], marker="o", linestyle="none",
               markersize=4.2 if final_tuning else 3.3,
               markerfacecolor=SERIES_COLORS[series], markeredgecolor="white",
               label=COMPACT_SERIES_LABELS.get(series, series))
        for series in SERIES_ORDER
    ]
    condition_handles = [
        condition_handles[index]
        for index in ((0, 1, 2, 3, 4, 5) if final_tuning else (0, 3, 1, 4, 2, 5))
    ]
    figure.legend(
        handles=condition_handles, loc="upper left",
        bbox_to_anchor=(0.125, 0.988) if final_tuning else (0.075, 0.958),
        ncol=2 if final_tuning else 3, frameon=False,
        fontsize=7.0 if final_tuning else 6.0, borderaxespad=0,
        handletextpad=0.2, columnspacing=0.45 if final_tuning else 0.6,
        labelspacing=0.14 if final_tuning else 0.24,
    )
    metric_handles = [
        Line2D([0], [0], marker=marker, linestyle="none",
               markersize=5.6 if final_tuning else 3.3,
               markerfacecolor="#64748B", markeredgecolor="white", label=label)
        for marker, label in (
            ("o", r"$P$"), ("s", r"$C$"),
            ("^", r"$S^A_{\mathrm{TV}}$"),
            ("D", r"$S^A_{\mathrm{TV},\neg\mathrm{self}}$"),
        )
    ]
    metric_handles = [metric_handles[index] for index in (0, 2, 1, 3)]
    figure.legend(
        handles=metric_handles, loc="upper left",
        bbox_to_anchor=(0.54, 0.977) if final_tuning else (0.534, 0.958),
        ncol=2, frameon=False,
        fontsize=7.0 if final_tuning else 6.0, borderaxespad=0,
        handlelength=0.9 if final_tuning else 2.0,
        handletextpad=0.25 if final_tuning else 0.2,
        columnspacing=0.45 if final_tuning else 0.38,
        labelspacing=0.18 if final_tuning else 0.24,
    )
    checkpoint_handles = [
        Line2D([0], [0], marker="o", linestyle="none",
               markersize=4.2 if final_tuning else 3.3,
               markerfacecolor=fill, markeredgecolor=edge, alpha=alpha, label=label)
        for label, fill, edge, alpha in (
            ("Initialisation", "white", "#64748B", 0.95),
            ("Pass 1", "#94A3B8", "white", 0.38),
            ("Pass 2", "#94A3B8", "white", 0.65),
            ("Pass 1000", "#64748B", "white", 1.0),
        )
    ]
    if not final_tuning:
        checkpoint_handles = [checkpoint_handles[index] for index in (0, 2, 1, 3)]
    figure.legend(
        handles=checkpoint_handles, loc="upper left",
        bbox_to_anchor=(0.835, 0.991) if final_tuning else (0.783, 0.958),
        ncol=1 if final_tuning else 2, frameon=False,
        fontsize=7.0 if final_tuning else 6.0, borderaxespad=0,
        handletextpad=0.2, columnspacing=0.4,
        labelspacing=0.08 if final_tuning else 0.24,
    )


def render_layout(
    means: dict[str, pd.DataFrame], layout_name: str, output_dir: Path,
) -> tuple[Path, Path]:
    layout = LAYOUTS[layout_name]
    row_metrics = _row_metrics(layout["metric_overlay"])
    columns = layout["columns"]
    refined = layout.get("refined", False)
    half_page = layout.get("half_page", False)
    structured = layout.get("structured_legend", False)
    compact_grouped = layout.get("compact_grouped", False)
    main_style = layout.get("main_style", False)
    final_tuning = layout.get("final_tuning", False)
    height = (4.0 if final_tuning else 4.65) if half_page else (
        7.0 if layout["metric_overlay"] else 11.4
    )
    figure, axes = plt.subplots(
        len(row_metrics), len(columns), figsize=(layout["width"], height),
        sharex="col", sharey="row", squeeze=False,
        gridspec_kw={"height_ratios": (0.65, 1.0)} if final_tuning else {},
    )
    figure.subplots_adjust(
        left=0.075 if main_style else (0.11 if structured or compact_grouped else
        (0.095 if len(columns) <= 2 else (0.07 if half_page else 0.075))),
        right=0.99 if half_page else 0.985,
        bottom=0.125 if main_style else (0.14 if compact_grouped else
        (0.15 if half_page else (0.12 if layout["metric_overlay"] else 0.085))),
        top=(0.84 if final_tuning else 0.86) if main_style else (0.855 if compact_grouped else
        (0.83 if structured else (0.82 if half_page else (0.865 if refined else 0.84)))),
        wspace=0.17 if half_page else (0.14 if len(columns) <= 2 else 0.12),
        hspace=(0.12 if final_tuning else 0.18) if main_style else
        (0.22 if compact_grouped else (0.35 if half_page else
        (0.30 if refined else (0.16 if layout["metric_overlay"] else 0.18)))),
    )
    x_limits, y_limits = _axis_limits(means, columns, row_metrics)
    metric_labels = dict(METRICS)
    for row_index, metrics in enumerate(row_metrics):
        for column_index, levels in enumerate(columns):
            axis = axes[row_index, column_index]
            for metric_index, metric in enumerate(metrics):
                frame = means[metric]
                for (series, level), row in frame.iterrows():
                    if int(level) not in levels:
                        continue
                    marker = (
                        ("o", "s", "^", "D")[row_index * 2 + metric_index]
                        if refined else
                        ("o", "s")[metric_index]
                        if layout["metric_overlay"] else
                        ("o", "s")[levels.index(int(level))]
                    )
                    show_pass_1 = (
                        str(series) in PASS_1_SERIES
                        and np.isfinite(row["nll_pass_1"])
                        and np.isfinite(row["quality_pass_1"])
                    )
                    _draw_trajectory(
                        axis, row, SERIES_COLORS[str(series)], marker, show_pass_1,
                        source_pretrained=str(series) in SOURCE_PRETRAINED_SERIES,
                        marker_scale=(0.54 if main_style else (0.61 if compact_grouped else
                                      (1.08 if half_page else (1.25 if refined else 1.0)))),
                        half_page=half_page,
                        compact_grouped=compact_grouped,
                        main_style=main_style,
                    )
            axis.set_xlim(*x_limits[column_index])
            axis.set_ylim(*y_limits[row_index])
            axis.set_axisbelow(True)
            axis.grid(
                color="#CBD5E1",
                linewidth=0.32 if compact_grouped else
                (0.35 if half_page else (0.42 if refined else 0.45)),
                alpha=0.18 if compact_grouped else
                (0.22 if half_page else (0.25 if refined else 0.32)),
            )
            if main_style:
                for spine in axis.spines.values():
                    spine.set_edgecolor("#27313D")
                    spine.set_linewidth(0.65)
            axis.tick_params(
                labelsize=6.4 if main_style else
                (6.2 if compact_grouped else (6.6 if half_page else 8.0)),
                length=2.2 if half_page else 3,
                width=0.55 if half_page else 0.7,
                **({"colors": "#111827"} if main_style else {}),
            )
            axis.xaxis.set_major_locator(MaxNLocator(nbins=4 if half_page else 5))
            axis.yaxis.set_major_locator(MaxNLocator(nbins=3 if half_page else 4))
            if main_style:
                panel_label = f"({chr(ord('a') + row_index * 4 + column_index)})"
                if final_tuning:
                    axis.set_title(
                        panel_label, loc="left", fontsize=6.7,
                        fontweight="semibold", pad=4,
                    )
                    if row_index == 0:
                        axis.set_title(
                            layout["titles"][column_index],
                            loc="center", fontsize=7.0,
                            fontweight="semibold", pad=4,
                        )
                else:
                    axis.set_title(
                        f"{panel_label}  {layout['titles'][column_index]}",
                        loc="left", fontsize=7.2,
                        fontweight="semibold", pad=4,
                    )
            elif row_index == 0:
                axis.set_title(
                    layout["titles"][column_index],
                    fontsize=7.0 if compact_grouped else
                    (7.8 if half_page else 10.0),
                    fontweight="semibold", pad=5 if half_page else 9,
                )
            if refined and not main_style:
                # The structured variant uses compact in-panel labels. Earlier
                # variants place them above the axes because initial points
                # occupy the top-left corner.
                axis.text(
                    0.075 if structured or compact_grouped else 0.02,
                    0.975 if structured or compact_grouped else 1.01,
                    f"({chr(ord('a') + row_index * 4 + column_index)})",
                    transform=axis.transAxes, ha="left",
                    va="top" if structured or compact_grouped else "bottom",
                    fontsize=6.3 if compact_grouped else
                    (7.1 if half_page else 8.8),
                    fontweight="bold", clip_on=structured or compact_grouped,
                    bbox={"facecolor": "white", "edgecolor": "none",
                          "alpha": 0.60 if compact_grouped else 0.72,
                          "pad": 0.12 if compact_grouped else 0.15}
                    if structured or compact_grouped else None,
                )
            if column_index == 0:
                if main_style:
                    pass
                elif structured or compact_grouped:
                    axis.set_ylabel(
                        "Latent representation" if row_index == 0
                        else "Transition structure",
                        fontsize=6.6 if compact_grouped else 7.0,
                        labelpad=3.5 if compact_grouped else 5,
                    )
                elif not refined:
                    axis.set_ylabel(
                        ("Latent representation" if row_index == 0 else "Transition structure")
                        if layout["metric_overlay"] else metric_labels[metrics[0]],
                        fontsize=9.0, labelpad=9,
                    )
            else:
                axis.tick_params(labelleft=False)
            if row_index < len(row_metrics) - 1:
                axis.tick_params(labelbottom=False)
    figure.supxlabel(
        "Evaluation NLL ↓ (bits/symbol)" if final_tuning else
        "Evaluation NLL (bits/symbol; lower is better)",
        fontsize=7.1 if main_style else
        (7.0 if compact_grouped else (7.6 if half_page else 9.5)),
        y=0.035 if main_style else (0.05 if compact_grouped else
        (0.028 if half_page else (0.025 if layout["metric_overlay"] else 0.018))),
    )
    _legends(figure, layout, axes)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / f"metric_likelihood_{layout_name}"
    paths = (stem.with_suffix(".png"), stem.with_suffix(".pdf"))
    with plt.rc_context({"pdf.fonttype": 42, "ps.fonttype": 42}):
        for path in paths:
            figure.savefig(path, dpi=300)
    plt.close(figure)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--layout", choices=LAYOUTS)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    rows = load_rows(
        PROJECT_ROOT / manifest["source_shadow_study"], manifest["protocol_id"],
        40, set(DEFAULT_LEVEL_ORDER),
    )
    from scripts.analysis.plot_config_metric_trajectories import rows_at_comparison_steps

    comparison_rows = rows_at_comparison_steps(rows, manifest["protocol_id"])
    means = _condition_means(comparison_rows, matched_budget=comparison_rows is not rows)
    for layout_name in (args.layout,) if args.layout else LAYOUTS:
        for path in render_layout(means, layout_name, args.output_dir):
            print(path)


if __name__ == "__main__":
    main()
