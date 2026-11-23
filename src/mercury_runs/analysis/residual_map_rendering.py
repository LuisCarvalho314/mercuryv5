"""Rendering helpers for transition residual map figures.

This module owns figure assembly: layout, colorbars, TV-similarity bars, and
level labels. It does not perform any computation (no alignment, no projection,
no walk generation).

The public entry point is `save_selected_figure`, which takes a pre-computed
payload dict (as written by `selected_residual_maps.py`) and writes a PNG.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import numpy as np
import networkx as nx
from matplotlib import pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter
import re

from mercury_runs.analysis.transition_residual_maps import (
    TransitionResidualDiagnostics,
    _draw_residual_map,
    _linewidth_reference_values,
    _residual_colormap,
    _residual_width,
    padded_axis_limits,
)
from mercury_runs.infrastructure.paper_precision import exact_cartesian_reference_positions
from mercury_runs.analysis.plots import publication_style as pub


SUMMARY_NEGATIVE_COLOR = "#9ecae1"
SUMMARY_POSITIVE_COLOR = "#f4a582"
SUMMARY_SIMILARITY_COLOR = "#b8e3c2"
SUMMARY_TEXT_COLOR = "#1f2937"
LEVEL_ORDER = (13, 21, 20, 22)
LEVEL_LABELS = {13: "3x3", 21: "1x4", 20: "1x5", 22: "3x4"}
COMPACT_METHOD_ORDER = (
    "MERCURY",
    "TMGWR",
    "CSCG",
    "Oracle-C",
    "Oracle-C+1st",
    "Oracle-C+Struct",
)
TITLE_REPLACEMENTS = {
    "all methods final": "Matched-step transition residuals: final comparison",
    "all methods checkpoint 4000": "Matched-step transition residuals after two training passes",
    "all methods checkpoint 2000": "Matched-step transition residuals after one training pass",
    "final hybrid vs cscg": "Ground-truth-aligned transition residuals: MERCURY initialised versus CSCG",
    "final tmgwr vs cscg": "Ground-truth-aligned transition residuals: TMGWR initialised versus CSCG",
    "hybrid vs cscg checkpoint 4000": "Ground-truth-aligned transition residuals after two training passes",
    "tmgwr hybrid vs cscg checkpoint 4000": "Ground-truth-aligned transition residuals after two training passes: TMGWR initialised versus CSCG",
    "mercury vs cscg checkpoint 2000": "Ground-truth-aligned transition residuals after one training pass",
}
METHOD_REPLACEMENTS = {
    "Mercury": "MERCURY",
    "Hybrid CSCG": "MERCURY–CSCG Hybrid",
    "Vanilla CSCG": "Standard CSCG",
    "CSCG - vanilla": "Standard CSCG",
    "GT states x1": "CSCG — oracle clones/state (×1)",
    "GT states x2": "CSCG — oracle clones/state (×2)",
    "Obs clones x4": "CSCG — observation clones (×4)",
    "Obs clones x8": "CSCG — observation clones (×8)",
    "CSCG — oracle clones ×1": "CSCG — oracle clones/state (×1)",
    "CSCG — oracle clones ×2": "CSCG — oracle clones/state (×2)",
    "CSCG — observation clones ×4": "CSCG — observation clones (×4)",
    "CSCG — observation clones ×8": "CSCG — observation clones (×8)",
}
_LEVEL_RE = re.compile(r"^L(?P<level>\d+)/(?P<method>.+)$")


def diagnostic_from_json(panel: dict[str, Any], *, method: str) -> TransitionResidualDiagnostics:
    """Reconstruct TransitionResidualDiagnostics from a JSON panel dict."""
    transition_true = np.asarray(panel["P_gt"], dtype=np.float64)
    transition_predicted = np.asarray(panel["P_hat"], dtype=np.float64)
    residual = np.asarray(panel["R"], dtype=np.float64)
    return TransitionResidualDiagnostics(
        method=method,
        transition_true=transition_true,
        transition_predicted=transition_predicted,
        residual=residual,
        excess_mass=np.asarray(panel.get("E_plus", np.maximum(residual, 0.0)), dtype=np.float64),
        missing_mass=np.asarray(panel.get("E_minus", np.maximum(-residual, 0.0)), dtype=np.float64),
        matched_mass=np.minimum(transition_true, transition_predicted),
        tv_by_state=np.asarray(panel["tv_by_state"], dtype=np.float64),
        similarity_by_state=np.asarray(panel["similarity_by_state"], dtype=np.float64),
        uncovered_rows=np.asarray(panel["uncovered_rows"], dtype=bool),
        tv_sim=float(panel["tv_sim"]),
        residual_magnitude=(
            np.asarray(panel["mean_abs_R"], dtype=np.float64)
            if "mean_abs_R" in panel
            else np.abs(residual)
        ),
    )


def positions_for_level(level: int) -> dict[int, np.ndarray]:
    """Return normalised 2-D node positions for the given maze level."""
    reference_positions = exact_cartesian_reference_positions(int(level))
    positions = {
        int(idx): np.asarray([float(col), -float(row)], dtype=np.float64)
        for idx, (row, col) in enumerate(reference_positions.tolist())
    }
    if positions:
        coords = np.stack(list(positions.values()), axis=0)
        x_min, x_max = float(coords[:, 0].min()), float(coords[:, 0].max())
        y_min, y_max = float(coords[:, 1].min()), float(coords[:, 1].max())
        x_span = x_max - x_min
        if x_span > 0.0:
            scale = 4.0 / x_span
            center = np.asarray([0.5 * (x_min + x_max), 0.5 * (y_min + y_max)], dtype=np.float64)
            positions = {
                node: center + (np.asarray(pos, dtype=np.float64) - center) * scale
                for node, pos in positions.items()
            }
    return positions


def panel_aspect_for_level(level: int) -> float:
    """Return height/width of the padded data window drawn for the given level.

    Panels use `aspect="equal"`, so a row must be given this fraction of the
    column width in height for the maze to fill its cell instead of being
    squeezed vertically (which pulls nodes together).
    """
    positions, ((x_lo, x_hi), (y_lo, y_hi)) = _normal_panel_positions(int(level))
    if not positions:
        return 1.0
    width = max(float(x_hi - x_lo), 1e-9)
    return float(y_hi - y_lo) / width


def save_selected_figure(
    *,
    payload: dict[str, Any],
    output_path: Path,
    vis_threshold: float,
    max_residual_override: float | None = None,
    font_scale: float = 1.60,
    header_font_scale: float | None = None,
    omit_leading_zero: bool = False,
    show_summary_bar_labels: bool = True,
) -> None:
    """Assemble and save a multi-panel transition residual map figure.

    `payload` is the dict written by `selected_residual_maps.main()` alongside each PNG.
    """
    panel_payloads = dict(payload["panels"])
    levels: list[int] = []
    methods: list[str] = []
    parsed: dict[tuple[int, str], dict[str, Any]] = {}
    for panel_key, panel in panel_payloads.items():
        match = _LEVEL_RE.match(str(panel_key))
        if match is None:
            raise ValueError(f"Unexpected panel key: {panel_key!r}")
        level = int(match.group("level"))
        method = _display_method(str(match.group("method")))
        parsed[(level, method)] = dict(panel)
        if level not in levels:
            levels.append(level)
        if method not in methods:
            methods.append(method)
    order_index = {level: idx for idx, level in enumerate(LEVEL_ORDER)}
    levels.sort(key=lambda level: (order_index.get(level, len(order_index)), level))

    nrows = len(levels)
    ncols = len(methods)
    diagnostics_by_cell = {
        key: diagnostic_from_json(panel, method=f"L{key[0]} {key[1]}")
        for key, panel in parsed.items()
    }
    if max_residual_override is None:
        max_residual = max(
            max(
                float(np.max(np.abs(diagnostic.residual))),
                float(np.max(diagnostic.residual_magnitude)) if diagnostic.residual_magnitude is not None else 0.0,
            )
            for diagnostic in diagnostics_by_cell.values()
            if diagnostic.residual.size
        )
    else:
        max_residual = float(max_residual_override)
    max_residual = max(max_residual, float(vis_threshold), 1e-9)
    action_conditioned = bool((payload.get("visual_encoding") or {}).get("action_conditioned", False))
    resolved_header_font_scale = (
        float(font_scale) if header_font_scale is None else float(header_font_scale)
    )

    # Place axes in physical units.  This prevents Matplotlib's equal-aspect
    # adjustment from shrinking an axis inside an oversized GridSpec cell and
    # creating invisible horizontal gutters.
    fig_width = 19.50
    left_inches, right_inches, column_gap_inches = 0.25, 0.05, 0.06
    bottom_margin_inches, top_margin_inches = 2.20, 0.36
    row_footer_inches, row_gap_inches = 0.30, 0.08
    column_width = (
        fig_width - left_inches - right_inches - column_gap_inches * (ncols - 1)
    ) / ncols
    row_heights = [panel_aspect_for_level(level) * column_width for level in levels]
    fig_height = (
        bottom_margin_inches
        + top_margin_inches
        + sum(row_heights)
        + row_footer_inches * nrows
        + row_gap_inches * (nrows - 1)
    )
    fig = plt.figure(figsize=(fig_width, fig_height), dpi=300)
    axes_array = np.empty((nrows, ncols), dtype=object)
    row_tops_inches: list[float] = []
    row_top = fig_height - top_margin_inches
    for row_idx, row_height in enumerate(row_heights):
        row_tops_inches.append(row_top)
        row_top -= row_height + row_footer_inches
        if row_idx < nrows - 1:
            row_top -= row_gap_inches

    node_collection = None
    bar_specs: list[tuple[int, plt.Axes, TransitionResidualDiagnostics]] = []
    row_axes: dict[int, list[plt.Axes]] = {}
    for row_idx, level in enumerate(levels):
        pos, panel_limits = _normal_panel_positions(level)
        row_axes[level] = []
        for col_idx, method in enumerate(methods):
            axis_left = left_inches + col_idx * (column_width + column_gap_inches)
            axis_bottom = row_tops_inches[row_idx] - row_heights[row_idx]
            ax = fig.add_axes(
                [
                    axis_left / fig_width,
                    axis_bottom / fig_height,
                    column_width / fig_width,
                    row_heights[row_idx] / fig_height,
                ]
            )
            axes_array[row_idx, col_idx] = ax
            row_axes[level].append(ax)
            diagnostic = diagnostics_by_cell[(level, method)]
            node_collection = _draw_residual_map(
                ax=ax,
                diagnostic=diagnostic,
                positions=pos,
                vis_threshold=float(vis_threshold),
                max_residual=max_residual,
                show_edge_labels=False,
                font_scale=float(font_scale),
            )
            ax.set_xlim(*panel_limits[0])
            ax.set_ylim(*panel_limits[1])
            if row_idx == 0:
                method_title = _compact_method_label(method)
                ax.set_title(
                    method_title,
                    fontsize=9.8 * resolved_header_font_scale,
                    fontweight="bold",
                    pad=3,
                )
            else:
                ax.set_title("")
            bar_specs.append((row_idx, ax, diagnostic))
    fig.canvas.draw()
    for level, axes_for_row in row_axes.items():
        _draw_level_label(
            fig,
            axes_for_row,
            LEVEL_LABELS.get(int(level), f"L{int(level)}"),
            font_scale=resolved_header_font_scale,
        )
    row_bar_y = {
        row_idx: (row_tops_inches[row_idx] - row_heights[row_idx] - 0.27) / fig_height
        for row_idx in range(nrows)
    }
    for row_idx, ax, diagnostic in bar_specs:
        _draw_tv_similarity_bar(
            fig,
            ax,
            diagnostic,
            font_scale=1.20 * float(font_scale),
            omit_leading_zero=bool(omit_leading_zero),
            show_labels=bool(show_summary_bar_labels),
            y0_override=row_bar_y[row_idx],
            height_inches=0.28,
        )

    residual_colorbar_bottom = 1.30 / fig_height
    node_colorbar_bottom = 0.55 / fig_height
    colorbar_height = 0.34 / fig_height
    legend_left = 0.06
    legend_right = 0.97
    legend_column_gap = 0.015
    legend_content_width = legend_right - legend_left - legend_column_gap
    right_legend_fraction = min(0.50, float(font_scale) / 4.80)
    right_legend_width = legend_content_width * right_legend_fraction
    colorbar_width = legend_content_width - right_legend_width
    colorbar_left = legend_left
    right_legend_left = colorbar_left + colorbar_width + legend_column_gap
    residual_cax = fig.add_axes([colorbar_left, residual_colorbar_bottom, colorbar_width, colorbar_height])
    node_cax = fig.add_axes([colorbar_left, node_colorbar_bottom, colorbar_width, colorbar_height])
    residual_norm = TwoSlopeNorm(vmin=-max_residual, vcenter=0.0, vmax=max_residual)
    residual_mappable = ScalarMappable(norm=residual_norm, cmap=_residual_colormap())
    residual_mappable.set_array([])
    residual_colorbar = fig.colorbar(
        residual_mappable,
        cax=residual_cax,
        orientation="horizontal",
    )
    residual_colorbar.ax.tick_params(labelsize=8 * float(font_scale), pad=2)
    if omit_leading_zero:
        residual_colorbar.ax.xaxis.set_major_formatter(
            FuncFormatter(lambda value, _position: _format_fraction(value, ".1f"))
        )
    fig.text(
        colorbar_left - 0.015,
        residual_colorbar_bottom + 0.5 * colorbar_height,
        r"$\bar{R}^{A}_{ij}$" if action_conditioned else r"$R_{ij}$",
        ha="right",
        va="center",
        fontsize=9 * float(font_scale),
        fontweight="semibold",
    )
    if node_collection is not None:
        node_colorbar = fig.colorbar(
            node_collection,
            cax=node_cax,
            orientation="horizontal",
        )
        node_colorbar.ax.tick_params(labelsize=8 * float(font_scale), pad=2)
        if omit_leading_zero:
            node_colorbar.ax.xaxis.set_major_formatter(
                FuncFormatter(lambda value, _position: _format_fraction(value, ".1f"))
            )
        fig.text(
            colorbar_left - 0.015,
            node_colorbar_bottom + 0.5 * colorbar_height,
            r"$\bar{S}^{A}_{i}$" if action_conditioned else r"$S_i$",
            ha="right",
            va="center",
            fontsize=9 * float(font_scale),
            fontweight="semibold",
        )

    width_values = _linewidth_reference_values(max_residual)
    edge_handles = []
    for value in width_values:
        formatted_value = (
            _format_fraction(value, ".2g") if omit_leading_zero else f"{value:.2g}"
        )
        edge_handles.append(
            Line2D(
                [0],
                [0],
                color="#4b5563",
                lw=_residual_width(value, max_residual),
                label=fr"$|R_{{ij}}|={formatted_value}$",
            )
        )
    edge_legend = fig.legend(
        handles=(
            edge_handles
            + [Line2D([0], [0], color="#d1d5db", lw=1.05, label="ground-truth edge")]
        ),
        loc="lower left",
        bbox_to_anchor=(right_legend_left, 1.31 / fig_height, right_legend_width, 0.0),
        mode="expand",
        borderaxespad=0.0,
        ncol=min(4, len(width_values) + 1),
        fontsize=8 * float(font_scale), frameon=True, borderpad=0.25, labelspacing=0.25, handlelength=1.4, handletextpad=0.3, columnspacing=0.4,
    )
    fig.add_artist(edge_legend)
    _draw_summary_bar_key(
        fig,
        x0=right_legend_left,
        y0=0.55 / fig_height,
        width=right_legend_width,
        height=0.34 / fig_height,
        font_scale=1.15 * float(font_scale),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    fig.savefig(output_path.with_suffix(".pdf"))
    plt.close(fig)


def _normal_panel_positions(
    level: int,
) -> tuple[dict[int, np.ndarray], tuple[tuple[float, float], tuple[float, float]]]:
    """Return standard node geometry and the safe padded drawing window."""
    positions = positions_for_level(int(level))
    return positions, padded_axis_limits(positions)


def save_selected_figure_transposed(
    *,
    payload: dict[str, Any],
    output_path: Path,
    vis_threshold: float,
    max_residual_override: float | None = None,
    font_scale: float = 1.45,
) -> None:
    """Save the normal residual grammar with methods as rows."""
    parsed: dict[tuple[int, str], dict[str, Any]] = {}
    levels: list[int] = []
    methods: list[str] = []
    for panel_key, panel in dict(payload["panels"]).items():
        match = _LEVEL_RE.match(str(panel_key))
        if match is None:
            raise ValueError(f"Unexpected panel key: {panel_key!r}")
        level = int(match.group("level"))
        method = _display_method(str(match.group("method")))
        parsed[(level, method)] = dict(panel)
        if level not in levels:
            levels.append(level)
        if method not in methods:
            methods.append(method)
    order_index = {level: index for index, level in enumerate(LEVEL_ORDER)}
    levels.sort(key=lambda level: (order_index.get(level, len(order_index)), level))
    methods.sort(key=lambda method: COMPACT_METHOD_ORDER.index(_compact_method_label(method)))
    diagnostics = {
        key: diagnostic_from_json(panel, method=f"L{key[0]} {key[1]}")
        for key, panel in parsed.items()
    }
    if max_residual_override is None:
        max_residual = max(
            max(
                float(np.max(np.abs(diagnostic.residual))),
                float(np.max(diagnostic.residual_magnitude))
                if diagnostic.residual_magnitude is not None
                else 0.0,
            )
            for diagnostic in diagnostics.values()
        )
    else:
        max_residual = float(max_residual_override)
    max_residual = max(max_residual, float(vis_threshold), 1e-9)
    action_conditioned = bool((payload.get("visual_encoding") or {}).get("action_conditioned", False))

    figure_width = 9.40
    left_inches, right_inches, column_gap_inches = 0.58, 0.06, 0.06
    bottom_margin_inches, top_margin_inches = 1.55, 0.36
    row_footer_inches, row_gap_inches = 0.30, 0.08
    environment_spans = np.asarray([2.0, 3.0, 4.0, 3.0], dtype=np.float64)
    available_width = (
        figure_width
        - left_inches
        - right_inches
        - column_gap_inches * (len(levels) - 1)
    )
    unit_width = available_width / float(environment_spans.sum())
    column_widths = (environment_spans * unit_width).tolist()
    column_lefts: list[float] = []
    column_left = left_inches
    for column_width in column_widths:
        column_lefts.append(column_left)
        column_left += column_width + column_gap_inches
    axis_heights = [
        panel_aspect_for_level(level) * column_widths[level_index]
        for level_index, level in enumerate(levels)
    ]
    row_height = max(axis_heights)
    figure_height = (
        bottom_margin_inches
        + top_margin_inches
        + row_height * len(methods)
        + row_footer_inches * len(methods)
        + row_gap_inches * (len(methods) - 1)
    )
    figure = plt.figure(figsize=(figure_width, figure_height), dpi=300)
    axes_array = np.empty((len(methods), len(levels)), dtype=object)
    row_tops_inches: list[float] = []
    row_top = figure_height - top_margin_inches
    for method_index in range(len(methods)):
        row_tops_inches.append(row_top)
        row_top -= row_height + row_footer_inches
        if method_index < len(methods) - 1:
            row_top -= row_gap_inches
    node_collection = None
    bar_specs: list[tuple[int, plt.Axes, TransitionResidualDiagnostics]] = []
    for method_index, method in enumerate(methods):
        for level_index, level in enumerate(levels):
            axis_height = axis_heights[level_index]
            axis_bottom = row_tops_inches[method_index] - 0.5 * (row_height + axis_height)
            axis = figure.add_axes(
                [
                    column_lefts[level_index] / figure_width,
                    axis_bottom / figure_height,
                    column_widths[level_index] / figure_width,
                    axis_height / figure_height,
                ]
            )
            axes_array[method_index, level_index] = axis
            positions, panel_limits = _normal_panel_positions(level)
            diagnostic = diagnostics[(level, method)]
            node_collection = _draw_residual_map(
                ax=axis,
                diagnostic=diagnostic,
                positions=positions,
                vis_threshold=float(vis_threshold),
                max_residual=max_residual,
                show_edge_labels=False,
                font_scale=float(font_scale),
            )
            axis.set_xlim(*panel_limits[0])
            axis.set_ylim(*panel_limits[1])
            axis.set_title("")
            bar_specs.append((method_index, axis, diagnostic))

    figure.canvas.draw()
    for level_index, level in enumerate(levels):
        figure.text(
            (column_lefts[level_index] + 0.5 * column_widths[level_index]) / figure_width,
            1.0 - 0.10 / figure_height,
            LEVEL_LABELS.get(level, f"L{level}"),
            ha="center",
            va="top",
            fontsize=10.5 * float(font_scale),
            fontweight="bold",
        )
    for method_index, method in enumerate(methods):
        row_center = (row_tops_inches[method_index] - 0.5 * row_height) / figure_height
        figure.text(
            0.018,
            row_center,
            _compact_method_label(method),
            rotation=90,
            ha="center",
            va="center",
            fontsize=9.5 * float(font_scale),
            fontweight="bold",
        )
    row_bar_y = {
        method_index: (row_tops_inches[method_index] - row_height - 0.17) / figure_height
        for method_index in range(len(methods))
    }
    for method_index, axis, diagnostic in bar_specs:
        _draw_tv_similarity_bar(
            figure,
            axis,
            diagnostic,
            font_scale=float(font_scale),
            y0_override=row_bar_y[method_index],
            width_override=0.12,
        )

    residual_cax = figure.add_axes([0.18, 1.00 / figure_height, 0.64, 0.10 / figure_height])
    node_cax = figure.add_axes([0.18, 0.55 / figure_height, 0.64, 0.10 / figure_height])
    residual_norm = TwoSlopeNorm(vmin=-max_residual, vcenter=0.0, vmax=max_residual)
    residual_mappable = ScalarMappable(norm=residual_norm, cmap=_residual_colormap())
    residual_mappable.set_array([])
    residual_colorbar = figure.colorbar(residual_mappable, cax=residual_cax, orientation="horizontal")
    residual_colorbar.set_label(
        r"$\bar{R}^{A}_{ij}=\langle\hat{P}(j|i,a)-P^{GT}(j|i,a)\rangle_a$"
        if action_conditioned
        else r"$R_{ij}=\hat{P}(j|i)-P^{GT}(j|i)$",
        fontsize=9 * float(font_scale),
    )
    residual_colorbar.ax.xaxis.set_label_position("top")
    residual_colorbar.ax.tick_params(labelsize=7.5 * float(font_scale), pad=2)
    if node_collection is not None:
        node_colorbar = figure.colorbar(node_collection, cax=node_cax, orientation="horizontal")
        node_colorbar.set_label(
            r"$\bar{S}^{A}_i=\langle1-TV(i,a)\rangle_a$"
            if action_conditioned
            else r"$S_i=1-TV(i)$",
            fontsize=9 * float(font_scale),
        )
        node_colorbar.ax.xaxis.set_label_position("top")
        node_colorbar.ax.tick_params(labelsize=7.5 * float(font_scale), pad=2)

    width_values = _linewidth_reference_values(max_residual)
    edge_legend = figure.legend(
        handles=(
            [
                Line2D(
                    [0],
                    [0],
                    color="#4b5563",
                    lw=_residual_width(value, max_residual),
                    label=fr"$|R_{{ij}}|={value:.2g}$",
                )
                for value in width_values
            ]
            + [Line2D([0], [0], color="#d1d5db", lw=1.05, label="ground-truth edge")]
        ),
        loc="lower center",
        bbox_to_anchor=(0.34, 0.06 / figure_height),
        ncol=min(4, len(width_values) + 1),
        fontsize=7.5 * float(font_scale),
        frameon=True,
    )
    figure.add_artist(edge_legend)
    _draw_summary_bar_key(
        figure,
        x0=0.72,
        y0=0.07 / figure_height,
        font_scale=float(font_scale),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.04)
    figure.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.04)
    plt.close(figure)


def save_selected_figure_compact(
    *,
    payload: dict[str, Any],
    output_path: Path,
    vis_threshold: float,
    max_residual_override: float | None = None,
    supplementary: bool = False,
    top_k: int | None = None,
) -> None:
    """Save a condensed residual matrix without altering the original renderer.

    All four environments and all six methods are retained. Within each
    environment, learned/baseline methods occupy the first 1x3 row and oracle
    controls occupy the second, giving each network useful horizontal space.
    """
    parsed: dict[tuple[int, str], dict[str, Any]] = {}
    available_levels: set[int] = set()
    available_methods: set[str] = set()
    for panel_key, panel in dict(payload["panels"]).items():
        match = _LEVEL_RE.match(str(panel_key))
        if match is None:
            raise ValueError(f"Unexpected panel key: {panel_key!r}")
        level = int(match.group("level"))
        method = _compact_method_label(_display_method(str(match.group("method"))))
        parsed[(level, method)] = dict(panel)
        available_levels.add(level)
        available_methods.add(method)

    levels = [level for level in LEVEL_ORDER if level in available_levels]
    methods = [method for method in COMPACT_METHOD_ORDER if method in available_methods]
    if not levels or not methods:
        raise ValueError("Compact residual figure has no matching panels.")

    diagnostics = {
        key: diagnostic_from_json(panel, method=f"L{key[0]} {key[1]}")
        for key, panel in parsed.items()
        if key[0] in levels and key[1] in methods
    }
    if max_residual_override is None:
        max_residual = max(
            float(np.max(diagnostic.residual_magnitude))
            if diagnostic.residual_magnitude is not None
            else float(np.max(np.abs(diagnostic.residual)))
            for diagnostic in diagnostics.values()
        )
    else:
        max_residual = float(max_residual_override)
    max_residual = max(max_residual, float(vis_threshold), 1e-9)

    fig_width = 7.15 if supplementary else 3.50
    fig_height = 7.6 if supplementary else 6.15
    fig = plt.figure(figsize=(fig_width, fig_height), dpi=300)
    outer_grid = fig.add_gridspec(
        len(levels),
        1,
        left=0.09 if supplementary else 0.13,
        right=0.992,
        bottom=0.115,
        top=0.955,
        hspace=0.18,
        height_ratios=[2.0 * panel_aspect_for_level(level) + 0.20 for level in levels],
    )
    axes_by_level: dict[int, list[plt.Axes]] = {}
    for level_index, level in enumerate(levels):
        method_grid = outer_grid[level_index].subgridspec(2, 3, hspace=0.24, wspace=0.06)
        axes_by_level[level] = []
        positions = positions_for_level(level)
        for method_index, method in enumerate(methods):
            axis = fig.add_subplot(method_grid[method_index // 3, method_index % 3])
            axes_by_level[level].append(axis)
            diagnostic = diagnostics[(level, method)]
            _draw_compact_residual_panel(
                axis,
                diagnostic=diagnostic,
                positions=positions,
                vis_threshold=float(vis_threshold),
                max_residual=max_residual,
                top_k=top_k,
                supplementary=supplementary,
            )
            axis.text(
                0.02,
                0.02,
                f"S={diagnostic.tv_sim:.2f}",
                transform=axis.transAxes,
                ha="left",
                va="bottom",
                fontsize=4.9 if supplementary else 4.0,
                color=SUMMARY_TEXT_COLOR,
                zorder=30,
            )
            axis.set_title(
                method,
                fontsize=6.8 if supplementary else 5.5,
                fontweight="bold",
                pad=0.7,
            )

    fig.canvas.draw()
    for level in levels:
        boxes = [axis.get_position() for axis in axes_by_level[level]]
        y_center = 0.5 * (min(box.y0 for box in boxes) + max(box.y1 for box in boxes))
        fig.text(
            0.022 if supplementary else 0.028,
            y_center,
            LEVEL_LABELS.get(level, f"L{level}"),
            rotation=90,
            ha="center",
            va="center",
            fontsize=7.2 if supplementary else 6.2,
            fontweight="bold",
        )

    residual_handles = [
        Line2D([0], [0], color=_residual_colormap()(0.12), lw=1.2, label="missing"),
        Line2D([0], [0], color=_residual_colormap()(0.88), lw=1.2, label="excess"),
        Line2D([0], [0], color=_residual_colormap()(0.50), lw=2.2, label="action cancellation"),
        Line2D([0], [0], color="#cbd5e1", lw=0.55, label="GT topology"),
    ]
    fig.legend(
        handles=residual_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.055),
        ncol=4,
        frameon=False,
        title="edge width = |residual|",
        title_fontsize=4.8 if supplementary else 4.2,
        fontsize=5.4 if supplementary else 4.7,
        handlelength=1.5,
        handletextpad=0.35,
        columnspacing=0.65,
    )
    node_cax = fig.add_axes([0.33, 0.022, 0.34, 0.008 if supplementary else 0.006])
    node_mappable = ScalarMappable(norm=Normalize(vmin=0.0, vmax=1.0), cmap=plt.get_cmap("viridis"))
    node_mappable.set_array([])
    node_colorbar = fig.colorbar(node_mappable, cax=node_cax, orientation="horizontal")
    node_colorbar.set_ticks([0.0, 0.5, 1.0])
    node_colorbar.ax.tick_params(labelsize=4.8 if supplementary else 4.2, length=1.5, pad=1)
    node_colorbar.set_label("node similarity", fontsize=5.4 if supplementary else 4.7, labelpad=1)
    node_colorbar.outline.set_linewidth(0.35)

    fig.suptitle(
        _compact_title(str(payload.get("scenario", "transition residuals"))),
        fontsize=8.2 if supplementary else 7.2,
        fontweight="semibold",
        y=0.985,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=600, bbox_inches="tight", pad_inches=0.025)
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.025)
    plt.close(fig)


def _draw_compact_residual_panel(
    ax: plt.Axes,
    *,
    diagnostic: TransitionResidualDiagnostics,
    positions: dict[int, np.ndarray],
    vis_threshold: float,
    max_residual: float,
    top_k: int | None,
    supplementary: bool,
) -> None:
    n_nodes = int(diagnostic.transition_true.shape[0])
    graph = nx.DiGraph()
    graph.add_nodes_from(range(n_nodes))
    pos = {
        node: np.asarray(positions.get(node, [float(node), 0.0]), dtype=np.float64)
        for node in range(n_nodes)
    }

    # Collapse reciprocal ground-truth support into a faint topology skeleton.
    skeleton = {
        (min(int(source), int(target)), max(int(source), int(target)))
        for source, target in np.argwhere(diagnostic.transition_true > 0.0).tolist()
        if int(source) != int(target)
    }
    if skeleton:
        nx.draw_networkx_edges(
            graph,
            pos=pos,
            edgelist=sorted(skeleton),
            edge_color="#cbd5e1",
            width=0.45 if supplementary else 0.32,
            alpha=0.72,
            arrows=False,
            ax=ax,
        )

    node_size = 34 if supplementary else 22
    linewidths = [0.75 if bool(diagnostic.uncovered_rows[node]) else 0.25 for node in range(n_nodes)]
    nx.draw_networkx_nodes(
        graph,
        pos=pos,
        node_color=diagnostic.similarity_by_state.tolist(),
        cmap=plt.get_cmap("viridis"),
        vmin=0.0,
        vmax=1.0,
        node_size=node_size,
        linewidths=linewidths,
        edgecolors="#334155",
        ax=ax,
    )
    nx.draw_networkx_labels(
        graph,
        pos=pos,
        labels={node: str(node) for node in range(n_nodes)},
        font_size=3.1 if supplementary else 2.55,
        font_color="#111827",
        ax=ax,
    )

    residual = np.asarray(diagnostic.residual, dtype=np.float64)
    magnitude = (
        np.asarray(diagnostic.residual_magnitude, dtype=np.float64)
        if diagnostic.residual_magnitude is not None
        else np.abs(residual)
    )
    candidates = [
        (float(magnitude[source, target]), int(source), int(target))
        for source, target in np.argwhere(magnitude > float(vis_threshold)).tolist()
    ]
    candidates.sort(reverse=True)
    selected = candidates if top_k is None else candidates[: max(int(top_k), 0)]
    residual_norm = TwoSlopeNorm(vmin=-max_residual, vcenter=0.0, vmax=max_residual)
    residual_cmap = _residual_colormap()
    edge_groups: dict[bool, dict[str, list[Any]]] = {}
    for edge_magnitude, source, target in selected:
        signed_value = float(residual[source, target])
        relative = min(edge_magnitude / max_residual, 1.0)
        key = source < target
        group = edge_groups.setdefault(key, {"edges": [], "colors": [], "widths": []})
        group["edges"].append((source, target))
        base_color = residual_cmap(residual_norm(signed_value))
        group["colors"].append((*base_color[:3], 0.55 + 0.4 * relative))
        group["widths"].append(0.35 + (1.25 if supplementary else 0.95) * relative)
    for forward, group in edge_groups.items():
        nx.draw_networkx_edges(
            graph,
            pos=pos,
            edgelist=group["edges"],
            edge_color=group["colors"],
            width=group["widths"],
            style="-",
            arrows=True,
            arrowstyle="-|>",
            arrowsize=5.0 if supplementary else 3.8,
            connectionstyle=("arc3,rad=0.10" if forward else "arc3,rad=-0.10"),
            ax=ax,
        )

    ax.set_axis_off()
    ax.set_aspect("equal", adjustable="box")
    if pos:
        x_limits, y_limits = padded_axis_limits(pos)
        ax.set_xlim(*x_limits)
        ax.set_ylim(*y_limits)


def _compact_method_label(method: str) -> str:
    aliases = {
        "MERCURY initialised": "MERCURY",
        "MERCURY–CSCG Hybrid": "MERCURY",
        "MERCURY-CSCG Hybrid": "MERCURY",
        "TMGWR initialised": "TMGWR",
        "TMGWR-CSCG Hybrid": "TMGWR",
        "Standard CSCG": "CSCG",
        "Oracle capacity": "Oracle-C",
        "Oracle capacity + first-order structure": "Oracle-C+1st",
        "Oracle capacity + structure": "Oracle-C+Struct",
    }
    return aliases.get(str(method), str(method))


def _format_fraction(value: float, format_spec: str) -> str:
    text = format(float(value), format_spec)
    if float(value) == 0.0:
        return "0"
    if text.startswith("-0."):
        return f"-.{text[3:]}"
    if text.startswith("0."):
        return f".{text[2:]}"
    return text


def _compact_title(title: str) -> str:
    normalized = str(title).lower()
    if "one training pass" in normalized or "checkpoint 2000" in normalized:
        compact = "Action-conditioned residuals — one pass"
    elif "two training passes" in normalized or "checkpoint 4000" in normalized:
        compact = "Action-conditioned residuals — two passes"
    elif "final" in normalized:
        compact = "Action-conditioned residuals — final"
    else:
        compact = "Action-conditioned transition residuals"
    if "self-loops included" in normalized:
        compact += " (with self-loops)"
    return compact


def _draw_tv_similarity_bar(
    fig: plt.Figure,
    ax: plt.Axes,
    diagnostic: TransitionResidualDiagnostics,
    *,
    font_scale: float = 1.0,
    omit_leading_zero: bool = False,
    show_labels: bool = True,
    y0_override: float | None = None,
    width_override: float | None = None,
    height_inches: float = 0.21,
) -> None:
    tv_sim = float(np.clip(diagnostic.tv_sim, 0.0, 1.0))
    tv_error = float(np.clip(1.0 - tv_sim, 0.0, 1.0))
    n_states = max(int(diagnostic.residual.shape[0]), 1)
    positive_mass = float(np.asarray(diagnostic.excess_mass, dtype=np.float64).sum() / n_states)
    negative_mass = float(np.asarray(diagnostic.missing_mass, dtype=np.float64).sum() / n_states)
    total_signed_error = positive_mass + negative_mass
    if total_signed_error > 0.0:
        negative_share = tv_error * negative_mass / total_signed_error
        positive_share = tv_error * positive_mass / total_signed_error
    else:
        negative_share = 0.5 * tv_error
        positive_share = 0.5 * tv_error

    bbox = ax.get_position()
    fig_height = float(fig.get_figheight())
    width = 0.92 * float(bbox.width) if width_override is None else float(width_override)
    height = float(height_inches) / fig_height
    x0 = float(bbox.x0 + 0.5 * bbox.width - 0.5 * width)
    y0 = (
        float(bbox.y0 - 0.17 / fig_height)
        if y0_override is None
        else float(y0_override)
    )

    ax.add_patch(Rectangle((x0, y0), width, height, transform=fig.transFigure, facecolor=SUMMARY_SIMILARITY_COLOR, edgecolor="#9ca3af", linewidth=0.45, clip_on=False, zorder=20))
    if negative_share > 0.0:
        ax.add_patch(Rectangle((x0, y0), width * negative_share, height, transform=fig.transFigure, facecolor=SUMMARY_NEGATIVE_COLOR, edgecolor="none", clip_on=False, zorder=21))
    if positive_share > 0.0:
        ax.add_patch(Rectangle((x0 + width * negative_share, y0), width * positive_share, height, transform=fig.transFigure, facecolor=SUMMARY_POSITIVE_COLOR, edgecolor="none", clip_on=False, zorder=21))

    if not show_labels:
        return

    similarity_share = max(1.0 - negative_share - positive_share, 0.0)
    format_share = (
        (lambda value: _format_fraction(value, ".2f"))
        if omit_leading_zero
        else (lambda value: f"{value:.2f}")
    )
    label_specs = (
        (0.0, negative_share, format_share(negative_share), SUMMARY_NEGATIVE_COLOR, 0.14),
        (negative_share, positive_share, format_share(positive_share), SUMMARY_POSITIVE_COLOR, 0.36),
        (negative_share + positive_share, similarity_share, format_share(similarity_share), SUMMARY_TEXT_COLOR, 0.78),
    )
    inside_threshold = 0.13 * max(1.0, float(font_scale) / 1.92)
    for start, component_width, label, color, fallback_relative_x in label_specs:
        inside = component_width >= inside_threshold
        relative_x = start + 0.5 * component_width if inside else fallback_relative_x
        ax.text(x0 + width * relative_x, y0 + (0.5 * height if inside else -0.004), label, transform=fig.transFigure, ha="center", va=("center" if inside else "top"), fontsize=5.8 * float(font_scale), fontweight="bold", color=SUMMARY_TEXT_COLOR, clip_on=False, zorder=22)


def _draw_summary_bar_key(
    fig: plt.Figure,
    *,
    x0: float,
    y0: float,
    width: float = 0.25,
    height: float | None = None,
    font_scale: float = 1.0,
) -> None:
    resolved_height = 0.26 / float(fig.get_figheight()) if height is None else float(height)
    for segment_width, label, color in zip(
        (0.29, 0.29, 0.42),
        ("missing", "excess", "similarity"),
        (SUMMARY_NEGATIVE_COLOR, SUMMARY_POSITIVE_COLOR, SUMMARY_SIMILARITY_COLOR),
    ):
        rect_width = width * segment_width
        fig.patches.append(Rectangle((x0, y0), rect_width, resolved_height, transform=fig.transFigure, facecolor=color, edgecolor="#9ca3af", linewidth=0.45, clip_on=False, zorder=30))
        fig.text(x0 + 0.5 * rect_width, y0 + 0.5 * resolved_height, label, ha="center", va="center", fontsize=7.2 * float(font_scale), fontweight="bold", color=SUMMARY_TEXT_COLOR, zorder=31)
        x0 += rect_width


def _draw_level_label(
    fig: plt.Figure,
    axes_for_row: list[plt.Axes],
    label: str,
    *,
    font_scale: float = 1.0,
) -> None:
    if not axes_for_row:
        return
    boxes = [ax.get_position() for ax in axes_for_row]
    y0 = min(float(box.y0) for box in boxes)
    y1 = max(float(box.y1) for box in boxes)
    fig.text(0.011, 0.5 * (y0 + y1), label, rotation=90, va="center", ha="center", fontsize=12 * float(font_scale), fontweight="bold")


def _display_method(method: str) -> str:
    replaced = METHOD_REPLACEMENTS.get(str(method), str(method))
    return pub.method_label(replaced)


def _display_title(title: str) -> str:
    normalized = str(title).replace("_", " ")
    return TITLE_REPLACEMENTS.get(normalized, normalized)
