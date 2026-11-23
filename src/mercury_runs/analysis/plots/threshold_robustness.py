"""Main threshold robustness plots.

Canonical: reads from threshold_edge_metrics_summary.csv files produced by
`mercury_runs.analysis.plots.threshold_edge_metrics`.
Accepts CLI arguments for all input/output paths.

Usage:
    python -m mercury_runs.analysis.plots.threshold_robustness \\
        --collapsed_summary results/..../threshold_edge_metrics_summary.csv \\
        --action_summary results/..../threshold_edge_metrics_summary.csv
"""
from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib
import numpy as np
import polars as pl

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from mercury_runs.analysis.plots import publication_style as pub


_DEFAULT_METRICS_ROOT = Path(
    "results/current/notebook_plots"
    "/mercury_cscg_hybrid_vs_cscg_batch_size_30_seed_threshold_edge_metrics_common_eval_walks2"
)
_DEFAULT_ACTION_ROOT = Path(
    "results/current/notebook_plots"
    "/mercury_cscg_hybrid_vs_cscg_batch_size_30_seed_action_conditioned_threshold_edge_metrics_common_eval_walks"
)
BASE_OUTPUT_DIR = _DEFAULT_METRICS_ROOT

LEVEL_ORDER = (13, 21, 20, 22)
LEVEL_LABELS = {13: "3x3", 21: "1x4", 20: "1x5", 22: "3x4"}
METHOD_LABELS = {
    "MERCURY": pub.method_label("mercury"),
    "POCML": "POCML",
    "TMGWR": pub.method_label("tmgwr"),
    "MERCURY-CSCG Hybrid": pub.method_label("hybrid"),
    "TMGWR-CSCG Hybrid": pub.method_label("tmgwr_hybrid"),
    "MERCURY initialised": pub.method_label("hybrid"),
    "TMGWR initialised": pub.method_label("tmgwr_hybrid"),
    "CSCG - vanilla": pub.method_label("vanilla_cscg"),
    "Standard CSCG": pub.method_label("vanilla_cscg"),
    "CSCG - oracle capacity": pub.method_label("oracle_capacity"),
    "Oracle capacity": pub.method_label("oracle_capacity"),
    "CSCG - oracle structural": pub.method_label("oracle_capacity_structural"),
    "Oracle capacity + structure": pub.method_label("oracle_capacity_structural"),
    "CSCG - oracle first-order": pub.method_label("oracle_capacity_first_order"),
    "Oracle capacity + first-order structure": pub.method_label("oracle_capacity_first_order"),
    "CSCG - oracle clones/state (x1)": "Oracle states x1",
    "CSCG - oracle clones/state (x2)": "Oracle states x2",
    "CSCG - observation clones (x4)": "Obs clones x4",
    "CSCG - observation clones (x8)": "Obs clones x8",
}
# Every method is drawn with the same weight and opacity: identity is carried by
# hue (Okabe-Ito, colour-blind safe) and, past the eight hues, by dash pattern.
LINEWIDTH = 1.9
LINE_ALPHA = 0.92
CI_ALPHA = 0.16
_DASHED = (0, (5.5, 2.2))
_FALLBACK_PALETTE = (
    "#111827",
    "#D55E00",
    "#0072B2",
    "#009E73",
    "#CC79A7",
    "#E69F00",
    "#56B4E9",
    "#7570B3",
    "#A6761D",
)
METHOD_STYLES = {
    "MERCURY": {"color": pub.method_color("mercury"), "linestyle": "-"},
    "POCML": {"color": "#CC79A7", "linestyle": "-"},
    "TMGWR": {"color": pub.method_color("tmgwr"), "linestyle": "-"},
    "MERCURY-CSCG Hybrid": {"color": pub.method_color("hybrid"), "linestyle": "-"},
    "TMGWR-CSCG Hybrid": {"color": pub.method_color("tmgwr_hybrid"), "linestyle": "-"},
    "MERCURY initialised": {"color": pub.method_color("hybrid"), "linestyle": "-"},
    "TMGWR initialised": {"color": pub.method_color("tmgwr_hybrid"), "linestyle": "-"},
    "CSCG - vanilla": {"color": pub.method_color("vanilla_cscg"), "linestyle": "-"},
    "Standard CSCG": {"color": pub.method_color("vanilla_cscg"), "linestyle": "-"},
    "CSCG - oracle capacity": {"color": pub.method_color("oracle_capacity"), "linestyle": "-"},
    "Oracle capacity": {"color": pub.method_color("oracle_capacity"), "linestyle": "-"},
    "CSCG - oracle structural": {"color": pub.method_color("oracle_capacity_structural"), "linestyle": "-"},
    "Oracle capacity + structure": {"color": pub.method_color("oracle_capacity_structural"), "linestyle": "-"},
    "CSCG - oracle first-order": {"color": pub.method_color("oracle_capacity_first_order"), "linestyle": "-"},
    "Oracle capacity + first-order structure": {"color": pub.method_color("oracle_capacity_first_order"), "linestyle": "-"},
    "CSCG - oracle clones/state (x1)": {"color": pub.method_color("ground_truth_state_1"), "linestyle": _DASHED},
    "CSCG - oracle clones/state (x2)": {"color": pub.method_color("ground_truth_state_2"), "linestyle": _DASHED},
    "CSCG - observation clones (x4)": {"color": pub.method_color("uniform_obs_4"), "linestyle": _DASHED},
    "CSCG - observation clones (x8)": {"color": pub.method_color("uniform_obs_8"), "linestyle": _DASHED},
}
METHOD_ORDER = {
    "MERCURY": 0,
    "POCML": 1,
    "TMGWR": 2,
    "MERCURY-CSCG Hybrid": 3,
    "TMGWR-CSCG Hybrid": 4,
    "MERCURY initialised": 3,
    "TMGWR initialised": 4,
    "CSCG - vanilla": 10,
    "Standard CSCG": 10,
    "CSCG - oracle capacity": 11,
    "Oracle capacity": 11,
    "CSCG - oracle first-order": 12,
    "Oracle capacity + first-order structure": 12,
    "CSCG - oracle structural": 13,
    "Oracle capacity + structure": 13,
    "CSCG - oracle clones/state (x1)": 14,
    "CSCG - oracle clones/state (x2)": 15,
    "CSCG - observation clones (x4)": 16,
    "CSCG - observation clones (x8)": 17,
}
METRIC_ORDER = ("edge_precision", "edge_recall", "edge_f1")
METRIC_LABELS = {
    "edge_precision": "Precision",
    "edge_recall": "Recall",
    "edge_f1": "F1",
}
COMPACT_CHECKPOINTS = (
    ("Single pass", "all_methods_checkpoint_2000"),
    ("Two passes", "all_methods_checkpoint_4000"),
    ("Extensive", "all_methods_final"),
)
COMPACT_OUTPUT = "threshold_sensitivity_action_edge_f1_compact_4x3.png"
COMPACT_LEGEND_LABELS = {
    "Standard CSCG": "CSCG",
    "Oracle capacity": "Oracle-C",
    "Oracle capacity + first-order structure": "Oracle-C+1st",
    "Oracle capacity + structure": "Oracle-C+Struct",
}
# Match the main-study combined metric profile exactly.  Keep this local to the
# compact six-method figure so broader diagnostic figures retain their existing
# extended palette.
COMPACT_METHOD_COLORS = {
    "MERCURY": "#D55E00",
    "MERCURY-CSCG Hybrid": "#D55E00",
    "MERCURY initialised": "#D55E00",
    "TMGWR": "#E6A700",
    "TMGWR-CSCG Hybrid": "#E6A700",
    "TMGWR initialised": "#E6A700",
    "CSCG - vanilla": "#27313D",
    "Standard CSCG": "#27313D",
    "CSCG - oracle capacity": "#0072B2",
    "Oracle capacity": "#0072B2",
    "CSCG - oracle first-order": "#CC79A7",
    "Oracle capacity + first-order structure": "#CC79A7",
    "CSCG - oracle structural": "#009E73",
    "Oracle capacity + structure": "#009E73",
}
SCENARIOS = {
    "all_methods_final": {
        "title": "Threshold robustness at final checkpoint",
        "subtitle": "All recorded methods",
        "paired_output": "threshold_robustness_all_methods_final_edge_f1_2x4.png",
    },
    "all_methods_checkpoint_4000": {
        "title": "Threshold robustness at checkpoint 4000",
        "subtitle": "All recorded methods",
        "paired_output": "threshold_robustness_all_methods_checkpoint_4000_edge_f1_2x4.png",
    },
    "all_methods_checkpoint_2000": {
        "title": "Threshold robustness at checkpoint 2000",
        "subtitle": "All recorded methods",
        "action_output": "threshold_robustness_all_methods_checkpoint_2000_action_f1_1x4.png",
        "paired_output": "threshold_robustness_all_methods_checkpoint_2000_edge_f1_2x4.png",
    },
    "final_hybrid_vs_cscg": {
        "title": "Threshold robustness of final edge recovery",
        "subtitle": "All recorded methods",
        "paired_output": "threshold_robustness_final_edge_f1_2x4.png",
    },
    "final_tmgwr_vs_cscg": {
        "title": "Threshold robustness of final TMGWR initialised edge recovery",
        "subtitle": "All recorded methods",
        "paired_output": "threshold_robustness_final_tmgwr_edge_f1_2x4.png",
    },
    "hybrid_vs_cscg_checkpoint_4000": {
        "title": "Threshold robustness at checkpoint 4000",
        "subtitle": "All recorded methods",
        "paired_output": "threshold_robustness_checkpoint_4000_edge_f1_2x4.png",
    },
    "tmgwr_hybrid_vs_cscg_checkpoint_4000": {
        "title": "Threshold robustness after two training passes for TMGWR initialised",
        "subtitle": "All recorded methods",
        "paired_output": "threshold_robustness_tmgwr_checkpoint_4000_edge_f1_2x4.png",
    },
    "mercury_vs_cscg_checkpoint_2000": {
        "title": "Threshold robustness at checkpoint 2000",
        "subtitle": "All recorded methods",
        "action_output": "threshold_robustness_mercury_action_f1_1x4.png",
        "paired_output": "threshold_robustness_checkpoint_2000_edge_f1_2x4.png",
    },
}


def _read_summary(path: Path, scope: str) -> pl.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pl.read_csv(path).with_columns(pl.lit(scope).alias("scope"))


def _method_rows(frame: pl.DataFrame, *, scenario: str, level: int, method: str) -> pl.DataFrame:
    return (
        frame.filter(
            (pl.col("scenario_key") == scenario)
            & (pl.col("level") == int(level))
            & (pl.col("method") == method)
        )
        .sort("threshold")
    )


def _line_arrays(rows: pl.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.asarray(rows["threshold"], dtype=float),
        np.asarray(rows["edge_f1_mean"], dtype=float),
        np.asarray(rows["edge_f1_ci95"], dtype=float),
    )


def _metric_line_arrays(rows: pl.DataFrame, metric: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.asarray(rows["threshold"], dtype=float),
        np.asarray(rows[f"{metric}_mean"], dtype=float),
        np.asarray(rows[f"{metric}_ci95"], dtype=float),
    )


def _method_style(method: str) -> dict[str, float | str]:
    style = METHOD_STYLES.get(method)
    if style is None:
        index = sum(ord(char) for char in method) % len(_FALLBACK_PALETTE)
        style = {"color": _FALLBACK_PALETTE[index], "linestyle": _DASHED}
    return {"linewidth": LINEWIDTH, "alpha": LINE_ALPHA, **style}


def _methods_for_panel(frame: pl.DataFrame, *, scenario: str, level: int) -> list[str]:
    rows = frame.filter(
        (pl.col("scenario_key") == scenario)
        & (pl.col("level") == int(level))
    )
    if rows.is_empty():
        return []
    methods = [str(method) for method in rows["method"].unique().to_list()]
    return sorted(methods, key=lambda method: (METHOD_ORDER.get(method, 99), method))


def _methods_for_scenario(*frames: pl.DataFrame, scenario: str) -> list[str]:
    methods: set[str] = set()
    for frame in frames:
        rows = frame.filter(pl.col("scenario_key") == scenario)
        if not rows.is_empty():
            methods.update(str(method) for method in rows["method"].unique().to_list())
    return sorted(methods, key=lambda method: (METHOD_ORDER.get(method, 99), method))


def _levels_for_scenario(*frames: pl.DataFrame, scenario: str) -> list[int]:
    levels: set[int] = set()
    for frame in frames:
        rows = frame.filter(pl.col("scenario_key") == scenario)
        if not rows.is_empty():
            levels.update(int(level) for level in rows["level"].unique().to_list())
    ordered_levels = [level for level in LEVEL_ORDER if level in levels]
    ordered_levels.extend(sorted(levels - set(ordered_levels)))
    return ordered_levels


def _plot_method(
    axis,
    rows: pl.DataFrame,
    method: str,
    *,
    show_ci: bool,
    label: str | None = None,
    metric: str = "edge_f1",
) -> None:
    if rows.is_empty():
        return
    x, y, ci = _metric_line_arrays(rows, metric)
    style = _method_style(method)
    axis.plot(
        x,
        y,
        color=style["color"],
        linestyle=style["linestyle"],
        linewidth=style["linewidth"],
        alpha=style["alpha"],
        label=label,
        solid_capstyle="round",
    )
    if show_ci:
        axis.fill_between(
            x,
            np.clip(y - ci, 0.0, 1.0),
            np.clip(y + ci, 0.0, 1.0),
            color=style["color"],
            alpha=CI_ALPHA,
            linewidth=0,
        )

def _style_axis(
    axis,
    *,
    level: int,
    row_label: str | None,
    show_xlabel: bool,
    show_ylabel: bool,
    show_title: bool,
    ylabel_mode: str = "f1_scope",
) -> None:
    if show_title:
        axis.set_title(LEVEL_LABELS.get(int(level), f"L{int(level)}"), fontsize=11, fontweight="bold")
    axis.set_xlim(0.0, 0.5)
    axis.set_ylim(-0.02, 1.02)
    axis.set_xticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
    axis.grid(True, which="major", color="#E5E7EB", linewidth=0.75)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.tick_params(labelsize=8.7)
    if show_xlabel:
        axis.set_xlabel(r"transition-mass cutoff ($\eta$)", fontsize=10)
    else:
        axis.tick_params(axis="x", labelbottom=False)
    if show_ylabel:
        if ylabel_mode == "metric":
            axis.set_ylabel("F1" if row_label is None else str(row_label), fontsize=10)
        else:
            axis.set_ylabel("F1" if row_label is None else f"{row_label}\nF1", fontsize=10)
    else:
        axis.tick_params(axis="y", labelleft=False)


def _legend_handles(methods: list[str]) -> list[Line2D]:
    handles = []
    for method in methods:
        style = _method_style(method)
        handles.append(
            Line2D(
                [0],
                [0],
                color=style["color"],
                linewidth=style["linewidth"],
                linestyle=style["linestyle"],
                alpha=style["alpha"],
                label=METHOD_LABELS.get(method, method),
            )
        )
    return handles


def _compact_method_style(method: str) -> dict[str, float | str]:
    style = _method_style(method)
    if method in COMPACT_METHOD_COLORS:
        style["color"] = COMPACT_METHOD_COLORS[method]
    if method in {"CSCG - oracle structural", "Oracle capacity + structure"}:
        style.update({"linewidth": 1.25, "alpha": 0.62, "linestyle": (0, (3.0, 1.8))})
    return style


def _compact_legend_handles(methods: list[str]) -> list[Line2D]:
    handles = []
    seen: set[str] = set()
    for method in methods:
        legend_method = {
            "MERCURY-CSCG Hybrid": "MERCURY",
            "MERCURY initialised": "MERCURY",
            "TMGWR-CSCG Hybrid": "TMGWR",
            "TMGWR initialised": "TMGWR",
        }.get(method, method)
        if legend_method in seen:
            continue
        seen.add(legend_method)
        style = _compact_method_style(legend_method)
        handles.append(
            Line2D(
                [0],
                [0],
                color=style["color"],
                linewidth=style["linewidth"],
                linestyle=style["linestyle"],
                alpha=style["alpha"],
                label=COMPACT_LEGEND_LABELS.get(
                    legend_method,
                    METHOD_LABELS.get(legend_method, legend_method),
                ),
            )
        )
    return handles


# Layout is laid out in inches and converted to figure fractions, so the header
# always reserves exactly the room the title, subtitle and legend need — the
# legend can never land on the panel titles, however many methods are plotted.
_PANEL_W_IN = 2.65
_WSPACE = 0.24
_HSPACE = 0.17
_RIGHT_PAD_IN = 0.18
_BOTTOM_IN = 0.70
_TOP_PAD_IN = 0.16
_TITLE_H_IN = 0.36
_SUBTITLE_H_IN = 0.30
_PANEL_TITLE_H_IN = 0.24
_LEGEND_ROW_IN = 0.25
_LEGEND_GAP_IN = 0.10
_LEGEND_NCOL_MAX = 4


def _new_grid(
    n_rows: int,
    n_cols: int,
    *,
    panel_h_in: float,
    n_legend_entries: int,
    ylabel_lines: int = 1,
):
    """Build a figure whose margins are sized for its own furniture."""
    n_cols = max(int(n_cols), 1)
    ncol = max(1, min(_LEGEND_NCOL_MAX, n_legend_entries))
    legend_rows = math.ceil(n_legend_entries / ncol) if n_legend_entries else 0
    legend_h = legend_rows * _LEGEND_ROW_IN + (_LEGEND_GAP_IN if legend_rows else 0.0)
    header_in = _TOP_PAD_IN + _TITLE_H_IN + _SUBTITLE_H_IN + legend_h + _PANEL_TITLE_H_IN
    left_in = 0.62 + 0.24 * ylabel_lines

    fig_w = left_in + _RIGHT_PAD_IN + n_cols * _PANEL_W_IN * (1.0 + _WSPACE) - _PANEL_W_IN * _WSPACE
    fig_h = header_in + n_rows * panel_h_in * (1.0 + _HSPACE) - panel_h_in * _HSPACE + _BOTTOM_IN

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(fig_w, fig_h),
        dpi=300,
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    fig.subplots_adjust(
        left=left_in / fig_w,
        right=1.0 - _RIGHT_PAD_IN / fig_w,
        top=1.0 - header_in / fig_h,
        bottom=_BOTTOM_IN / fig_h,
        wspace=_WSPACE,
        hspace=_HSPACE,
    )
    header = {
        "ncol": ncol,
        "title_y": 1.0 - _TOP_PAD_IN / fig_h,
        "subtitle_y": 1.0 - (_TOP_PAD_IN + _TITLE_H_IN) / fig_h,
        "legend_y": 1.0 - (_TOP_PAD_IN + _TITLE_H_IN + _SUBTITLE_H_IN) / fig_h,
    }
    return fig, axes, header


def _decorate(fig, header: dict, *, title: str, subtitle: str, methods: list[str]) -> None:
    fig.suptitle(title, fontsize=13, y=header["title_y"], va="top")
    fig.text(
        0.5,
        header["subtitle_y"],
        subtitle,
        ha="center",
        va="top",
        fontsize=9.5,
        color="#4B5563",
    )
    if methods:
        fig.legend(
            handles=_legend_handles(methods),
            loc="upper center",
            ncol=header["ncol"],
            frameon=False,
            fontsize=9,
            bbox_to_anchor=(0.5, header["legend_y"]),
            columnspacing=1.6,
            handlelength=2.2,
        )


def plot_final_paired(action: pl.DataFrame, collapsed: pl.DataFrame, *, scenario: str) -> Path:
    spec = SCENARIOS[scenario]
    legend_methods = _methods_for_scenario(action, collapsed, scenario=scenario)
    levels = _levels_for_scenario(action, collapsed, scenario=scenario)
    fig, axes, header = _new_grid(
        2,
        len(levels),
        panel_h_in=2.05,
        n_legend_entries=len(legend_methods),
        ylabel_lines=2,
    )
    for col_index, level in enumerate(levels):
        for row_index, (scope_frame, row_label) in enumerate(
            ((action, "Action-conditioned edge"), (collapsed, "Collapsed edge"))
        ):
            axis = axes[row_index, col_index]
            for method in _methods_for_panel(scope_frame, scenario=scenario, level=level):
                _plot_method(
                    axis,
                    _method_rows(scope_frame, scenario=scenario, level=level, method=method),
                    method,
                    show_ci=True,
                )
            _style_axis(
                axis,
                level=level,
                row_label=row_label,
                show_xlabel=row_index == 1,
                show_ylabel=col_index == 0,
                show_title=row_index == 0,
            )
    _decorate(
        fig,
        header,
        title=str(spec["title"]),
        subtitle=str(spec["subtitle"]),
        methods=legend_methods,
    )
    output_path = BASE_OUTPUT_DIR / str(spec["paired_output"])
    _save(fig, output_path)
    return output_path


def plot_full_metric_grid(frame: pl.DataFrame, *, scenario: str, scope_label: str, output_name: str) -> Path:
    spec = SCENARIOS[scenario]
    legend_methods = _methods_for_scenario(frame, scenario=scenario)
    levels = _levels_for_scenario(frame, scenario=scenario)
    fig, axes, header = _new_grid(
        3,
        len(levels),
        panel_h_in=2.05,
        n_legend_entries=len(legend_methods),
        ylabel_lines=1,
    )
    for col_index, level in enumerate(levels):
        for row_index, metric in enumerate(METRIC_ORDER):
            axis = axes[row_index, col_index]
            for method in _methods_for_panel(frame, scenario=scenario, level=level):
                _plot_method(
                    axis,
                    _method_rows(frame, scenario=scenario, level=level, method=method),
                    method,
                    show_ci=True,
                    metric=metric,
                )
            _style_axis(
                axis,
                level=level,
                row_label=METRIC_LABELS[metric],
                show_xlabel=row_index == len(METRIC_ORDER) - 1,
                show_ylabel=col_index == 0,
                show_title=row_index == 0,
                ylabel_mode="metric",
            )
    _decorate(
        fig,
        header,
        title=(
            f"{scope_label} thresholded edge metrics: "
            f"{str(spec['title']).replace('Threshold robustness ', '')}"
        ),
        subtitle=str(spec["subtitle"]),
        methods=legend_methods,
    )
    output_path = BASE_OUTPUT_DIR / output_name
    _save(fig, output_path)
    return output_path


def plot_mercury_action(action: pl.DataFrame, *, scenario: str) -> Path:
    spec = SCENARIOS[scenario]
    legend_methods = _methods_for_scenario(action, scenario=scenario)
    levels = _levels_for_scenario(action, scenario=scenario)
    fig, axes, header = _new_grid(
        1,
        len(levels),
        panel_h_in=2.25,
        n_legend_entries=len(legend_methods),
        ylabel_lines=1,
    )
    for col_index, (axis, level) in enumerate(zip(np.ravel(axes), levels)):
        for method in _methods_for_panel(action, scenario=scenario, level=level):
            _plot_method(
                axis,
                _method_rows(action, scenario=scenario, level=level, method=method),
                method,
                show_ci=True,
            )
        _style_axis(axis, level=level, row_label=None, show_xlabel=True, show_ylabel=col_index == 0, show_title=True)
    _decorate(
        fig,
        header,
        title=str(spec["title"]),
        subtitle=str(spec["subtitle"]),
        methods=legend_methods,
    )
    output_path = BASE_OUTPUT_DIR / str(spec["action_output"])
    _save(fig, output_path)
    return output_path


def plot_compact_action_f1(action: pl.DataFrame) -> Path:
    """Plot the three checkpoint sweeps without changing their aggregation."""
    missing_levels = [level for level in LEVEL_ORDER if level not in set(action["level"].to_list())]
    missing_scenarios = [
        scenario for _, scenario in COMPACT_CHECKPOINTS
        if action.filter(pl.col("scenario_key") == scenario).is_empty()
    ]
    if missing_levels or missing_scenarios:
        details = []
        if missing_levels:
            details.append(f"levels {missing_levels}")
        if missing_scenarios:
            details.append(f"scenarios {missing_scenarios}")
        raise ValueError("Compact figure requires " + " and ".join(details))

    methods: set[str] = set()
    for _, scenario in COMPACT_CHECKPOINTS:
        methods.update(_methods_for_scenario(action, scenario=scenario))
    legend_methods = sorted(methods, key=lambda method: (METHOD_ORDER.get(method, 99), method))

    fig, axes = plt.subplots(
        len(LEVEL_ORDER),
        len(COMPACT_CHECKPOINTS),
        figsize=(3.5, 5.3),
        dpi=300,
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    fig.subplots_adjust(left=0.15, right=0.99, top=0.875, bottom=0.095, wspace=0.14, hspace=0.10)

    for row_index, level in enumerate(LEVEL_ORDER):
        for col_index, (checkpoint_label, scenario) in enumerate(COMPACT_CHECKPOINTS):
            axis = axes[row_index, col_index]
            for method in _methods_for_panel(action, scenario=scenario, level=level):
                rows = _method_rows(action, scenario=scenario, level=level, method=method)
                x, y, ci = _line_arrays(rows)
                style = _compact_method_style(method)
                is_structural_oracle = method in {
                    "CSCG - oracle structural",
                    "Oracle capacity + structure",
                }
                axis.plot(
                    x,
                    y,
                    color=style["color"],
                    linestyle=style["linestyle"],
                    linewidth=style["linewidth"],
                    alpha=style["alpha"],
                    solid_capstyle="round",
                    zorder=1 if is_structural_oracle else 2,
                )
                axis.fill_between(
                    x,
                    np.clip(y - ci, 0.0, 1.0),
                    np.clip(y + ci, 0.0, 1.0),
                    color=style["color"],
                    alpha=0.05 if is_structural_oracle else 0.09,
                    linewidth=0,
                    zorder=0 if is_structural_oracle else 1,
                )

            axis.set_xlim(0.0, 0.5)
            axis.set_ylim(0.0, 1.0)
            axis.set_xticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
            axis.set_xticklabels(["0", ".1", ".2", ".3", ".4", ".5"])
            axis.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
            axis.set_yticklabels(["0", ".25", ".5", ".75", "1"])
            axis.grid(True, which="major", color="#ECEFF1", linewidth=0.45)
            axis.set_axisbelow(True)
            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)
            axis.tick_params(labelsize=6.0, length=2.0, width=0.55, pad=1.5)
            axis.tick_params(axis="x", labelbottom=row_index == len(LEVEL_ORDER) - 1)
            axis.tick_params(axis="y", labelleft=col_index == 0)
            if row_index == 0:
                axis.set_title(checkpoint_label, fontsize=7.0, fontweight="bold", pad=3)

    for row_index, level in enumerate(LEVEL_ORDER):
        position = axes[row_index, 0].get_position()
        fig.text(
            0.025,
            0.5 * (position.y0 + position.y1),
            LEVEL_LABELS[level].replace("x", "×"),
            ha="center",
            va="center",
            rotation=90,
            fontsize=6.2,
            fontweight="bold",
        )

    fig.legend(
        handles=_compact_legend_handles(legend_methods),
        loc="upper center",
        ncol=3,
        frameon=False,
        fontsize=8.0,
        bbox_to_anchor=(0.51, 0.995),
        columnspacing=0.55,
        labelspacing=0.20,
        handlelength=1.50,
        handletextpad=0.30,
    )
    fig.supxlabel(r"Transition-mass cutoff, $\tau$", fontsize=7.5, y=0.018)
    fig.supylabel("Action-conditioned edge F1", fontsize=6.8, x=0.068)

    output_path = BASE_OUTPUT_DIR / COMPACT_OUTPUT
    _save(fig, output_path)
    return output_path


def _save(fig, output_path: Path) -> None:
    # No bbox_inches="tight": the margins above are deliberate, and cropping them
    # away is what let the legend drift onto the panel titles.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, facecolor="white")
    fig.savefig(output_path.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--collapsed_summary",
        type=Path,
        default=_DEFAULT_METRICS_ROOT / "threshold_edge_metrics_summary.csv",
        help="Path to collapsed threshold_edge_metrics_summary.csv",
    )
    parser.add_argument(
        "--action_summary",
        type=Path,
        default=_DEFAULT_ACTION_ROOT / "threshold_edge_metrics_summary.csv",
        help="Path to action-conditioned threshold_edge_metrics_summary.csv",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=_DEFAULT_METRICS_ROOT,
        help="Directory where output figures are written",
    )
    parser.add_argument("--scenarios", type=str, default=",".join(SCENARIOS.keys()))
    parser.add_argument(
        "--compact",
        action="store_true",
        help="Also write the compact 4x3 action-conditioned edge-F1 checkpoint figure",
    )
    args = parser.parse_args()
    global BASE_OUTPUT_DIR
    BASE_OUTPUT_DIR = args.output_dir
    args.output_dir.mkdir(parents=True, exist_ok=True)

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
    action = _read_summary(args.action_summary, "action_conditioned")
    collapsed = _read_summary(args.collapsed_summary, "collapsed")
    scenario_keys = [item.strip() for item in str(args.scenarios).split(",") if item.strip()]
    paths = []
    for scenario in scenario_keys:
        if scenario not in SCENARIOS:
            raise ValueError(f"Unknown scenario: {scenario}")
        if "action_output" in SCENARIOS[scenario]:
            paths.append(plot_mercury_action(action, scenario=scenario))
        paths.append(plot_final_paired(action, collapsed, scenario=scenario))
        paths.append(
            plot_full_metric_grid(
                action,
                scenario=scenario,
                scope_label="Action-conditioned",
                output_name=f"threshold_metrics_{scenario}_action_conditioned_3x4.png",
            )
        )
        paths.append(
            plot_full_metric_grid(
                collapsed,
                scenario=scenario,
                scope_label="Collapsed",
                output_name=f"threshold_metrics_{scenario}_collapsed_3x4.png",
            )
        )
    if args.compact:
        paths.append(plot_compact_action_f1(action))
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
