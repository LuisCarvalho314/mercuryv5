#!/usr/bin/env python3
"""Combine the one-pass metric profile and metric/likelihood trajectories."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
from typing import Any

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
from matplotlib.ticker import MaxNLocator
import numpy as np
import polars as pl

from mercury_runs.analysis.plots import batch_summary_forest as forest
from scripts.analysis.plot_combined_metric_profiles import SERIES, _source_series
from scripts.analysis.plot_config_metric_trajectories import rows_at_comparison_steps
from scripts.analysis.plot_pass_2_to_1000_metric_delta import (
    COMPACT_SERIES_LABELS,
    DEFAULT_LEVEL_ORDER,
    LEVEL_LABELS,
    PRIMARY_METRIC,
    SECONDARY_METRIC,
    SERIES_COLORS,
    SERIES_ORDER,
    load_rows,
    three_checkpoint_seed_frame,
)


ENVIRONMENT_MARKERS = {13: "o", 21: "s", 20: "^", 22: "D"}
ENVIRONMENT_OFFSETS = {13: -0.048, 21: -0.016, 20: 0.016, 22: 0.048}
PROFILE_METRICS = (
    ("latent_precision", "Purity", r"Latent purity, $P$", r"$P_Z$"),
    ("latent_recall", "Concentration", r"Latent concentration, $C$", r"$C_Z$"),
    (
        "latent_action_conditioned_mean_total_variation_similarity",
        "Transition similarity",
        r"Transition similarity, $S^A_{\mathrm{TV}}$",
        r"$S^A_{\mathrm{TV}}$",
    ),
    (
        "latent_action_conditioned_non_self_loop_total_variation_similarity",
        "No-self similarity",
        r"No-self similarity, $S^{A,\neg\mathrm{self}}_{\mathrm{TV}}$",
        r"$S^{A,\neg\mathrm{self}}_{\mathrm{TV}}$",
    ),
)
LIKELIHOOD_METRICS = (
    "latent_precision",
    "latent_recall",
    PRIMARY_METRIC,
    SECONDARY_METRIC,
)
PASS_1_LIKELIHOOD_SERIES = {
    "vanilla_cscg",
    "oracle_capacity",
    "oracle_capacity_first_order",
    "oracle_capacity_structural",
}
PROFILE_TO_LIKELIHOOD_SERIES = {
    "target": "mercury_cscg_hybrid",
    "tmgwr_hybrid": "tmgwr_cscg_hybrid",
    "vanilla_cscg": "vanilla_cscg",
    "oracle_capacity": "oracle_capacity",
    "oracle_capacity_first_order": "oracle_capacity_first_order",
    "oracle_capacity_structural": "oracle_capacity_structural",
}
OUTPUT_STEM = "pass_2_to_1000_one_pass_metric_profile_with_metric_likelihood_spaces"


def _configure_style() -> None:
    plt.style.use("default")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.0,
            "axes.edgecolor": "#27313D",
            "axes.labelcolor": "#111827",
            "xtick.color": "#111827",
            "ytick.color": "#111827",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _draw_one_pass_profile(
    axis: plt.Axes,
    source: pl.DataFrame,
    lookup: dict[tuple[int, str, str], tuple[float, float, float]],
    *,
    metric: str,
    show_x_labels: bool,
) -> None:
    method_offsets = np.linspace(0.34, -0.34, len(SERIES))
    for method_index, series_key in enumerate(SERIES):
        series_name = _source_series(series_key, "mercury")
        colour = SERIES_COLORS[PROFILE_TO_LIKELIHOOD_SERIES[series_key]]
        method_y = float(method_offsets[method_index])
        for level in forest.LEVEL_ORDER:
            result = lookup.get((level, series_name, metric))
            if result is None:
                continue
            seed_values = (
                source.filter(
                    (pl.col("level") == level) & (pl.col("plot_series") == series_name)
                )
                .drop_nulls(["seed", metric])
                .group_by("seed")
                .agg(pl.col(metric).mean().alias("value"))
                .sort("seed")
            )
            y = method_y + ENVIRONMENT_OFFSETS[level]
            seed_offsets = np.linspace(-0.003, 0.003, len(seed_values))
            marker = ENVIRONMENT_MARKERS[level]
            axis.scatter(
                seed_values["value"].to_numpy(),
                y + seed_offsets,
                marker=marker,
                s=5.0,
                color=colour,
                alpha=0.16,
                linewidths=0,
                zorder=1,
            )
            mean, half_width, _ = result
            emphasized = method_index == 0
            axis.errorbar(
                mean,
                y,
                xerr=half_width,
                fmt="none",
                color=colour,
                elinewidth=0.9 if emphasized else 0.65,
                capsize=2.2,
                capthick=0.9 if emphasized else 0.65,
                zorder=3,
            )
            axis.scatter(
                mean,
                y,
                marker=marker,
                s=20.0 if emphasized else 14.0,
                color=colour,
                edgecolors="white",
                linewidths=0.45,
                zorder=4,
            )

    axis.set_xlim(0.0, 1.0)
    axis.set_xticks([0.0, 0.5, 1.0], ["0", "0.5", "1.0"])
    axis.set_ylim(-0.46, 0.46)
    axis.set_yticks([])
    axis.grid(axis="x", color="#D8DEE6", linewidth=0.55, alpha=0.8)
    axis.set_axisbelow(True)
    axis.tick_params(axis="x", labelsize=9.0, labelbottom=show_x_labels)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_visible(False)
    axis.spines["bottom"].set_linewidth(0.6)


def _draw_likelihood_space(
    axis: plt.Axes,
    spec: dict[str, Any],
    row_index: int,
    *,
    show_ci: bool = False,
) -> None:
    for (series, level), seed_values in spec["seed_values"].groupby(
        ["series", "level"], sort=False
    ):
        trajectory = seed_values.mean(numeric_only=True)
        colour = SERIES_COLORS[str(series)]
        marker = ENVIRONMENT_MARKERS[int(level)]
        checkpoints = [
            (
                float(trajectory["nll_initialisation"]),
                float(trajectory["quality_initialisation"]),
                "initialisation",
            )
        ]
        if str(series) in PASS_1_LIKELIHOOD_SERIES:
            checkpoints.append(
                (float(trajectory["nll_pass_1"]), float(trajectory["quality_pass_1"]), "pass_1")
            )
        checkpoints.extend(
            [
                (float(trajectory["nll_pass_2"]), float(trajectory["quality_pass_2"]), "pass_2"),
                (
                    float(trajectory["nll_pass_1000"]),
                    float(trajectory["quality_pass_1000"]),
                    "pass_1000",
                ),
            ]
        )
        for start, end in zip(checkpoints, checkpoints[1:]):
            axis.annotate(
                "",
                xy=end[:2],
                xytext=start[:2],
                arrowprops={
                    "arrowstyle": "-|>",
                    "color": colour,
                    "linewidth": 1.0,
                    "alpha": 0.82,
                    "mutation_scale": 7.0,
                    "shrinkA": 2.5,
                    "shrinkB": 2.5,
                },
                zorder=3,
            )
        checkpoint_style = {
            "initialisation": ("white", 0.95, 0.9),
            "pass_1": (colour, 0.28, 0.55),
            "pass_2": (colour, 0.48, 0.55),
            "pass_1000": (colour, 0.95, 0.45),
        }
        for x, y, checkpoint in checkpoints:
            facecolour, alpha, linewidth = checkpoint_style[checkpoint]
            if show_ci:
                nll_values = seed_values[f"nll_{checkpoint}"].to_numpy(float)
                quality_values = seed_values[f"quality_{checkpoint}"].to_numpy(float)
                axis.errorbar(
                    x,
                    y,
                    xerr=forest._ci95_half_width(len(nll_values), float(np.std(nll_values, ddof=1))),
                    yerr=forest._ci95_half_width(
                        len(quality_values), float(np.std(quality_values, ddof=1))
                    ),
                    fmt="none",
                    ecolor=colour,
                    elinewidth=0.7,
                    capsize=1.8,
                    capthick=0.7,
                    alpha=0.62,
                    zorder=3.5,
                )
            axis.scatter(
                x,
                y,
                marker=marker,
                s=25 if checkpoint != "pass_1000" else 27,
                facecolors=facecolour,
                edgecolors=colour if checkpoint == "initialisation" else "white",
                linewidths=linewidth,
                alpha=alpha,
                zorder=4,
            )

    axis.grid(True, color="#D1D5DB", linewidth=0.42, alpha=0.52, zorder=0)
    axis.tick_params(labelsize=9.0)
    axis.yaxis.set_major_locator(MaxNLocator(nbins=4))
    axis.xaxis.set_major_locator(MaxNLocator(nbins=5))
    axis.set_ylabel(spec["absolute_axis_label"], fontsize=10.0)
    axis.set_title(
        f"({chr(ord('b') + row_index)})  {spec['panel_label']}/likelihood space",
        loc="left",
        fontsize=11.0,
        fontweight="semibold",
    )


def plot_combined(
    profile_frame: pl.DataFrame,
    likelihood_specs: list[dict[str, Any]],
    output_dir: Path,
    *,
    show_likelihood_ci: bool = False,
) -> list[Path]:
    _configure_style()
    source = profile_frame.filter(pl.col("comparison") == "checkpoint_2000")
    lookup = forest._summary_lookup(source)
    figure_height = 8.85
    figure, axes = plt.subplots(
        4,
        2,
        figsize=(9.3, figure_height),
        dpi=300,
        gridspec_kw={"width_ratios": [0.25, 0.75], "hspace": 0.30, "wspace": 0.18},
    )
    figure.subplots_adjust(
        left=0.065,
        right=0.99,
        bottom=0.53 / figure_height,
        top=1.0 - 1.06 / figure_height,
    )

    for row_index, ((metric, _, _, _), spec) in enumerate(
        zip(PROFILE_METRICS, likelihood_specs)
    ):
        _draw_one_pass_profile(
            axes[row_index, 0],
            source,
            lookup,
            metric=metric,
            show_x_labels=row_index == len(PROFILE_METRICS) - 1,
        )
        _draw_likelihood_space(
            axes[row_index, 1], spec, row_index, show_ci=show_likelihood_ci
        )

    axes[0, 0].set_title(
        "(a)  One-pass acquisition", loc="left", fontsize=11.0, fontweight="semibold"
    )
    axes[-1, 0].set_xlabel("Metric value", fontsize=10.0)
    axes[-1, 1].set_xlabel(
        "Evaluation NLL (bits/symbol; lower is better)", fontsize=10.0
    )

    method_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markersize=6.2,
            markerfacecolor=SERIES_COLORS[series],
            markeredgecolor="white",
            label=COMPACT_SERIES_LABELS.get(series, series),
        )
        for series in SERIES_ORDER
    ]
    method_handles = [method_handles[index] for index in (0, 3, 1, 4, 2, 5)]
    method_legend = figure.legend(
        handles=method_handles,
        loc="center",
        bbox_to_anchor=(0.25, 0.947),
        ncol=3,
        frameon=False,
        fontsize=9.8,
        handletextpad=0.30,
        columnspacing=0.80,
        labelspacing=0.45,
    )
    figure.add_artist(method_legend)
    environment_handles = [
        Line2D(
            [0],
            [0],
            marker=ENVIRONMENT_MARKERS[level],
            linestyle="none",
            markersize=5.8,
            markerfacecolor="#6B7280",
            markeredgecolor="white",
            label=LEVEL_LABELS.get(level, f"L{level}").replace("x", r"$\times$"),
        )
        for level in DEFAULT_LEVEL_ORDER
    ]
    environment_handles = [environment_handles[index] for index in (0, 2, 1, 3)]
    figure.legend(
        handles=environment_handles,
        loc="center",
        bbox_to_anchor=(0.62, 0.947),
        ncol=2,
        frameon=False,
        fontsize=9.8,
        handletextpad=0.30,
        columnspacing=0.85,
        labelspacing=0.45,
    )
    figure.text(
        0.25, 0.992, "Conditions", ha="center", va="top",
        fontsize=9.8, fontweight="semibold", color="#475569",
    )
    figure.text(
        0.62, 0.992, "Environments", ha="center", va="top",
        fontsize=9.8, fontweight="semibold", color="#475569",
    )
    figure.add_artist(
        Line2D(
            [0.49, 0.49], [0.912, 0.995], transform=figure.transFigure,
            color="#D7DEE7", linewidth=0.65,
        )
    )
    checkpoint_handles = [
        Line2D(
            [0], [0], marker="o", linestyle="none", markersize=5.5,
            markerfacecolor=facecolour, markeredgecolor=edgecolour, alpha=alpha, label=label,
        )
        for label, facecolour, edgecolour, alpha in (
            ("Initialisation", "white", "#64748B", 0.95),
            ("Pass 1", "#94A3B8", "white", 0.38),
            ("Pass 2", "#94A3B8", "white", 0.65),
            ("Pass 1000", "#64748B", "white", 1.0),
        )
    ]
    figure.text(
        0.86, 0.992, "Checkpoints", ha="center", va="top",
        fontsize=9.8, fontweight="semibold", color="#475569",
    )
    figure.legend(
        handles=checkpoint_handles,
        loc="center",
        bbox_to_anchor=(0.86, 0.947),
        ncol=2,
        frameon=False,
        fontsize=8.8,
        handletextpad=0.35,
        columnspacing=0.75,
    )
    figure.add_artist(
        Line2D(
            [0.735, 0.735], [0.912, 0.995], transform=figure.transFigure,
            color="#D7DEE7", linewidth=0.65,
        )
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    output_stem = OUTPUT_STEM + ("_95ci_trial" if show_likelihood_ci else "")
    paths = [output_dir / f"{output_stem}.png", output_dir / f"{output_stem}.pdf"]
    for path in paths:
        figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    caption_path = output_dir / f"{output_stem}_caption.tex"
    likelihood_ci_description = (
        " Horizontal and vertical error bars show two-sided 95\\% Student-$t$ confidence "
        "intervals for NLL and the structural metric, respectively."
        if show_likelihood_ci
        else ""
    )
    caption_path.write_text(
        "\\caption{One-pass acquisition profiles and metric/likelihood trajectories. The left "
        "column shows seed values and condition means with 95\\% confidence intervals after one "
        "pass (step 2000). The right column shows condition-mean trajectories from initialisation "
        "through pass 1000. Pass 1 is omitted from the MERCURY and TMGWR likelihood trajectories "
        "because their source one-pass representations have no comparable NLL. Colour identifies "
        "condition and marker shape identifies environment in both columns."
        f"{likelihood_ci_description}}}\\n",
        encoding="utf-8",
    )
    return [*paths, caption_path]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", type=Path, required=True)
    parser.add_argument("--protocol-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=40)
    parser.add_argument(
        "--likelihood-ci-trial",
        action="store_true",
        help="Add 95% confidence intervals to likelihood checkpoints under a trial stem.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    profile_frame = forest._build_frame(
        args.study_root,
        args.protocol_id,
        protocol_ids_by_rand={0.1: args.protocol_id},
        batch_size=args.batch_size,
    )
    if profile_frame.is_empty():
        raise RuntimeError("No evaluation records were found")
    rows = load_rows(
        args.study_root,
        args.protocol_id,
        args.batch_size,
        set(DEFAULT_LEVEL_ORDER),
    )
    comparison_rows = rows_at_comparison_steps(rows, args.protocol_id)
    likelihood_specs = [
        {
            "panel_label": panel_label,
            "absolute_axis_label": axis_label,
            "seed_values": three_checkpoint_seed_frame(
                comparison_rows,
                metric=metric,
                pass_1_step=2_000,
                allow_missing_pass_1_series=(
                    {"mercury_cscg_hybrid", "tmgwr_cscg_hybrid"}
                    if comparison_rows is not rows else None
                ),
            ),
        }
        for metric, (_, panel_label, axis_label, _) in zip(
            LIKELIHOOD_METRICS, PROFILE_METRICS
        )
    ]
    for path in plot_combined(
        profile_frame,
        likelihood_specs,
        args.output_dir,
        show_likelihood_ci=args.likelihood_ci_trial,
    ):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
