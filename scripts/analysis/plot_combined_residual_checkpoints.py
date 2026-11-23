#!/usr/bin/env python3
"""Render the one-/two-pass residual comparison as one native Matplotlib figure.

Unlike the original composite, this module never reads a PNG.  It reads the two
JSON sidecars produced by ``selected_residual_maps`` and draws every node, edge,
summary bar, label, colourbar, and legend directly into one Matplotlib figure.

The layout constants immediately below are intentionally collected in one place
so the publication figure can be adjusted without touching the data pipeline.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for import_root in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-residual-combined")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp")

import matplotlib

matplotlib.use("Agg", force=True)
from matplotlib import pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
import numpy as np

from mercury_runs.analysis.residual_map_rendering import (
    LEVEL_LABELS,
    LEVEL_ORDER,
    _compact_method_label,
    _draw_level_label,
    _draw_summary_bar_key,
    _draw_tv_similarity_bar,
    _format_fraction,
    _normal_panel_positions,
    diagnostic_from_json,
    panel_aspect_for_level,
)
from mercury_runs.analysis.transition_residual_maps import (
    _draw_residual_map,
    _linewidth_reference_values,
    _residual_colormap,
    _residual_width,
)


# Publication layout controls.  These are the main values to tweak.
FIGURE_SIZE = (19.20, 10.80)
PNG_DPI = 440
LEFT_IN = 0.38
RIGHT_IN = 0.16
TOP_IN = 0.92
BOTTOM_LEGEND_IN = 1.42
GROUP_GAP_IN = 0.25
COLUMN_GAP_IN = 0.08
ROW_FOOTER_IN = 0.30
ROW_GAP_IN = 0.06
PANEL_FONT_SCALE = 2.30
HEADER_FONT_SCALE = 1.92
VIS_THRESHOLD = 0.02

# Shared colourbar and legend layout.
LEGEND_LEFT = 0.095
LEGEND_RIGHT = 0.975
LEGEND_GAP = 0.030
LEGEND_RIGHT_WIDTH = 0.210
RESIDUAL_LEGEND_BOTTOM_IN = 0.94
NODE_LEGEND_BOTTOM_IN = 0.36
COLOURBAR_HEIGHT_IN = 0.24

# Spacing inside the edge-width legend.
EDGE_LEGEND_BORDER_PAD = 0.20
EDGE_LEGEND_HANDLE_LENGTH = 1.20
EDGE_LEGEND_HANDLE_TEXT_PAD = 0.25
EDGE_LEGEND_COLUMN_SPACING = 0.55

ONE_PASS_METHODS = ("MERCURY", "TMGWR", "Standard CSCG")
TWO_PASS_METHODS = (
    "MERCURY initialised",
    "Standard CSCG",
    "Oracle capacity",
    "Oracle capacity + first-order structure",
)


def _panel_lookup(payload: dict[str, Any]) -> dict[tuple[int, str], dict[str, Any]]:
    lookup: dict[tuple[int, str], dict[str, Any]] = {}
    for key, panel in dict(payload["panels"]).items():
        level_text, method = str(key).split("/", 1)
        lookup[(int(level_text.removeprefix("L")), method)] = dict(panel)
    return lookup


def _global_residual_scale(*payloads: dict[str, Any]) -> float:
    declared = [
        float(encoding["global_residual_scale"])
        for payload in payloads
        if (encoding := dict(payload.get("visual_encoding") or {})).get(
            "global_residual_scale"
        )
        is not None
    ]
    if declared:
        # Both checkpoint payloads are produced together with the same scale.
        return max(declared)
    magnitudes = []
    for payload in payloads:
        for panel in dict(payload["panels"]).values():
            magnitudes.append(np.abs(np.asarray(panel["R"], dtype=np.float64)))
            if "mean_abs_R" in panel:
                magnitudes.append(np.asarray(panel["mean_abs_R"], dtype=np.float64))
    return max(VIS_THRESHOLD, max(float(np.max(value)) for value in magnitudes))


def render_combined_residual_checkpoints(
    *,
    one_pass_payload: dict[str, Any],
    two_pass_payload: dict[str, Any],
    output_stem: Path,
) -> list[Path]:
    """Draw both checkpoints into one figure and save native PNG/PDF outputs."""
    groups = (
        ("(a) One pass", ONE_PASS_METHODS, _panel_lookup(one_pass_payload)),
        ("(b) Two passes", TWO_PASS_METHODS, _panel_lookup(two_pass_payload)),
    )
    levels = [
        level
        for level in LEVEL_ORDER
        if all(
            all((level, method) in lookup for method in methods)
            for _title, methods, lookup in groups
        )
    ]
    if not levels:
        raise ValueError("The checkpoint payloads have no complete common levels.")

    figure_width, figure_height = FIGURE_SIZE
    method_count = sum(len(methods) for _title, methods, _lookup in groups)
    column_width = (
        figure_width
        - LEFT_IN
        - RIGHT_IN
        - GROUP_GAP_IN
        - COLUMN_GAP_IN * (method_count - len(groups))
    ) / method_count
    row_heights = [panel_aspect_for_level(level) * column_width for level in levels]
    required_height = (
        TOP_IN
        + BOTTOM_LEGEND_IN
        + sum(row_heights)
        + ROW_FOOTER_IN * len(levels)
        + ROW_GAP_IN * (len(levels) - 1)
    )
    if required_height > figure_height:
        raise ValueError(
            f"FIGURE_SIZE height must be at least {required_height:.2f} inches for this layout."
        )

    figure = plt.figure(figsize=FIGURE_SIZE)
    max_residual = _global_residual_scale(one_pass_payload, two_pass_payload)
    row_top = figure_height - TOP_IN
    row_tops: list[float] = []
    for row_height in row_heights:
        row_tops.append(row_top)
        row_top -= row_height + ROW_FOOTER_IN + ROW_GAP_IN

    x_positions: list[float] = []
    group_bounds: list[tuple[float, float]] = []
    x = LEFT_IN
    for group_index, (_title, methods, _lookup) in enumerate(groups):
        group_left = x
        for method_index, _method in enumerate(methods):
            x_positions.append(x)
            x += column_width
            if method_index < len(methods) - 1:
                x += COLUMN_GAP_IN
        group_bounds.append((group_left, x))
        if group_index < len(groups) - 1:
            x += GROUP_GAP_IN

    node_collection = None
    row_axes: dict[int, list[plt.Axes]] = {level: [] for level in levels}
    column_index = 0
    for group_title, methods, lookup in groups:
        for method in methods:
            for row_index, level in enumerate(levels):
                axis = figure.add_axes(
                    [
                        x_positions[column_index] / figure_width,
                        (row_tops[row_index] - row_heights[row_index]) / figure_height,
                        column_width / figure_width,
                        row_heights[row_index] / figure_height,
                    ]
                )
                diagnostic = diagnostic_from_json(
                    lookup[(level, method)], method=f"L{level} {method}"
                )
                positions, limits = _normal_panel_positions(level)
                node_collection = _draw_residual_map(
                    ax=axis,
                    diagnostic=diagnostic,
                    positions=positions,
                    vis_threshold=VIS_THRESHOLD,
                    max_residual=max_residual,
                    show_edge_labels=False,
                    font_scale=PANEL_FONT_SCALE,
                )
                axis.set_xlim(*limits[0])
                axis.set_ylim(*limits[1])
                axis.set_title(
                    _compact_method_label(method) if row_index == 0 else "",
                    fontsize=9.8 * HEADER_FONT_SCALE,
                    fontweight="bold",
                    pad=3,
                )
                row_axes[level].append(axis)
                _draw_tv_similarity_bar(
                    figure,
                    axis,
                    diagnostic,
                    font_scale=1.20 * PANEL_FONT_SCALE,
                    omit_leading_zero=True,
                    show_labels=False,
                    y0_override=(
                        row_tops[row_index] - row_heights[row_index] - 0.27
                    )
                    / figure_height,
                    height_inches=0.22,
                )
            column_index += 1

    figure.canvas.draw()
    for level in levels:
        _draw_level_label(
            figure,
            row_axes[level],
            LEVEL_LABELS.get(level, f"L{level}"),
            font_scale=HEADER_FONT_SCALE,
        )

    for (title, _methods, _lookup), (group_left, group_right) in zip(
        groups, group_bounds, strict=True
    ):
        figure.text(
            0.5 * (group_left + group_right) / figure_width,
            1.0 - 0.18 / figure_height,
            title,
            ha="center",
            va="top",
            fontsize=18,
            fontweight="bold",
        )

    separator_x = (group_bounds[0][1] + 0.5 * GROUP_GAP_IN) / figure_width
    figure.add_artist(
        Line2D(
            [separator_x, separator_x],
            [BOTTOM_LEGEND_IN / figure_height, 1.0],
            transform=figure.transFigure,
            color="#d1d5db",
            linewidth=0.7,
        )
    )

    # Shared native legends.
    colourbar_width = (
        LEGEND_RIGHT - LEGEND_LEFT - LEGEND_GAP - LEGEND_RIGHT_WIDTH
    )
    right_left = LEGEND_LEFT + colourbar_width + LEGEND_GAP
    residual_bottom = RESIDUAL_LEGEND_BOTTOM_IN / figure_height
    node_bottom = NODE_LEGEND_BOTTOM_IN / figure_height
    colourbar_height = COLOURBAR_HEIGHT_IN / figure_height

    residual_axis = figure.add_axes(
        [LEGEND_LEFT, residual_bottom, colourbar_width, colourbar_height]
    )
    residual_norm = TwoSlopeNorm(vmin=-max_residual, vcenter=0.0, vmax=max_residual)
    residual_mappable = ScalarMappable(norm=residual_norm, cmap=_residual_colormap())
    residual_mappable.set_array([])
    residual_bar = figure.colorbar(
        residual_mappable, cax=residual_axis, orientation="horizontal"
    )
    residual_bar.ax.tick_params(labelsize=15, pad=2)
    residual_bar.ax.xaxis.set_major_formatter(
        FuncFormatter(lambda value, _position: _format_fraction(value, ".1f"))
    )
    figure.text(
        LEGEND_LEFT - 0.012,
        residual_bottom + 0.5 * colourbar_height,
        r"$\bar{R}^{A, \neg\mathrm{self}}_{ij}$",
        ha="right",
        va="center",
        fontsize=21,
        fontweight="semibold",
    )

    node_axis = figure.add_axes(
        [LEGEND_LEFT, node_bottom, colourbar_width, colourbar_height]
    )
    if node_collection is not None:
        node_bar = figure.colorbar(node_collection, cax=node_axis, orientation="horizontal")
        node_bar.ax.tick_params(labelsize=15, pad=2)
        node_bar.ax.xaxis.set_major_formatter(
            FuncFormatter(lambda value, _position: _format_fraction(value, ".1f"))
        )
    figure.text(
        LEGEND_LEFT - 0.012,
        node_bottom + 0.5 * colourbar_height,
        r"$S^{A, \neg\mathrm{self}}_{i}$",
        ha="right",
        va="center",
        fontsize=21,
        fontweight="semibold",
    )

    width_values = _linewidth_reference_values(max_residual)
    handles = [
        Line2D(
            [0],
            [0],
            color="#4b5563",
            lw=_residual_width(value, max_residual),
            label=fr"$E|R|={_format_fraction(value, '.2g')}$",
        )
        for value in width_values
    ]
    handles.append(Line2D([0], [0], color="#d1d5db", lw=1.05, label="GT"))
    figure.legend(
        handles=handles,
        loc="lower left",
        bbox_to_anchor=(right_left, residual_bottom, LEGEND_RIGHT_WIDTH, 0.0),
        mode="expand",
        borderaxespad=0.0,
        ncol=len(handles),
        fontsize=11.5,
        frameon=True,
        borderpad=EDGE_LEGEND_BORDER_PAD,
        handlelength=EDGE_LEGEND_HANDLE_LENGTH,
        handletextpad=EDGE_LEGEND_HANDLE_TEXT_PAD,
        columnspacing=EDGE_LEGEND_COLUMN_SPACING,
    )
    _draw_summary_bar_key(
        figure,
        x0=right_left,
        y0=node_bottom,
        width=LEGEND_RIGHT_WIDTH,
        height=colourbar_height,
        font_scale=2.0,
    )

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    outputs = [output_stem.with_suffix(".png"), output_stem.with_suffix(".pdf")]
    figure.savefig(outputs[0], dpi=PNG_DPI, facecolor="white")
    figure.savefig(outputs[1], facecolor="white")
    plt.close(figure)
    return outputs


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--one-pass-json", required=True, type=Path)
    parser.add_argument("--two-pass-json", required=True, type=Path)
    parser.add_argument("--output-stem", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    outputs = render_combined_residual_checkpoints(
        one_pass_payload=json.loads(args.one_pass_json.read_text(encoding="utf-8")),
        two_pass_payload=json.loads(args.two_pass_json.read_text(encoding="utf-8")),
        output_stem=args.output_stem,
    )
    for output in outputs:
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
