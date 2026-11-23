"""Canonical plots for the ambiguity-threshold/decay ablation study.

All parameter combinations are rendered with equal visual weight. The module
consumes canonical evaluation records and recomputes thresholded edge metrics
from their saved transition matrices.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable

import matplotlib
import numpy as np
import polars as pl

matplotlib.use("Agg", force=True)
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

from mercury_runs.analysis.evaluation_records import (
    diagnostics_from_record,
    edge_metrics_at_threshold,
    load_evaluation_step,
)
from mercury_runs.analysis.plots import batch_summary_forest as forest
from mercury_runs.analysis.plots import batch_tv_similarity as tv
from mercury_runs.analysis.plots import threshold_edge_metrics as threshold_data
from mercury_runs.analysis.plots import threshold_robustness as robustness
from mercury_runs.analysis.plots.selected_residual_maps import _aggregate_diagnostics
from mercury_runs.analysis.residual_map_rendering import save_selected_figure


PARAMETER_ORDER: tuple[tuple[int, float], ...] = tuple(
    (threshold, decay)
    for threshold in (10, 20, 30, 40)
    for decay in (0.99, 1.0)
)
SCENARIO_KEY = "age_ambiguity_ablation"
DEFAULT_THRESHOLDS = threshold_data.DEFAULT_THRESHOLDS


def _series_key(threshold: int, decay: float) -> str:
    decay_slug = f"{float(decay):g}".replace(".", "p")
    return f"threshold_{int(threshold)}_decay_{decay_slug}"


SERIES_ORDER = tuple(_series_key(threshold, decay) for threshold, decay in PARAMETER_ORDER)
SERIES_LABELS = {
    _series_key(threshold, decay): f"threshold={threshold}, decay={decay:g}"
    for threshold, decay in PARAMETER_ORDER
}
# Each pair shares a hue family while remaining distinguishable in point plots.
SERIES_COLORS = {
    _series_key(10, 0.99): "#0072B2",
    _series_key(10, 1.0): "#56B4E9",
    _series_key(20, 0.99): "#009E73",
    _series_key(20, 1.0): "#66C2A5",
    _series_key(30, 0.99): "#D55E00",
    _series_key(30, 1.0): "#E69F00",
    _series_key(40, 0.99): "#7B4DCC",
    _series_key(40, 1.0): "#CC79A7",
}

ENVIRONMENT_MARKERS = {13: "o", 21: "s", 20: "^", 22: "D"}
ENVIRONMENT_OFFSETS = {13: -0.045, 21: -0.015, 20: 0.015, 22: 0.045}
THRESHOLD_COLORS = {
    10: "#0072B2",
    20: "#009E73",
    30: "#D55E00",
    40: "#7B4DCC",
}

PARAMETER_RESPONSE_METRICS = tuple(forest.PROFILE_FOCUSED_METRICS)
PARAMETER_RESPONSE_COLORS = {
    "latent_precision": "#0072B2",
    "latent_recall": "#009E73",
    "latent_action_conditioned_mean_total_variation_similarity": "#D55E00",
    "latent_action_conditioned_non_self_loop_total_variation_similarity": "#7B4DCC",
}
PARAMETER_RESPONSE_LABELS = {
    metric: forest.COMPACT_TRANSPOSED_METRIC_LABELS[metric]
    for metric in PARAMETER_RESPONSE_METRICS
}


def _parameter_response_style() -> None:
    plt.style.use("default")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.0,
            "axes.edgecolor": "#64748B",
            "axes.labelcolor": "#1F2933",
            "axes.titlecolor": "#1F2933",
            "xtick.color": "#1F2933",
            "ytick.color": "#1F2933",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _save_parameter_response_figure(figure: plt.Figure, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=600)
    figure.savefig(output_path.with_suffix(".pdf"))
    plt.close(figure)
    return output_path


def _metric_legend_handles(
    *,
    show_line: bool = True,
    marker_size: float = 4.8,
) -> list[Line2D]:
    return [
        Line2D(
            [0], [0], color=PARAMETER_RESPONSE_COLORS[metric], marker="o",
            linestyle="-" if show_line else "none",
            linewidth=1.15, markersize=marker_size, markeredgewidth=0,
            label=PARAMETER_RESPONSE_LABELS[metric],
        )
        for metric in PARAMETER_RESPONSE_METRICS
    ]


def _decay_legend_handles(*, marker_size: float = 5.5) -> list[Line2D]:
    return [
        Line2D(
            [0], [0], marker="o", linestyle="none", markersize=marker_size,
            markerfacecolor="#64748B" if filled else "white",
            markeredgecolor="#64748B", markeredgewidth=1.0, label=label,
        )
        for filled, label in ((True, "0.99"), (False, "1"))
    ]


def _parameter_response_ylim(
    lookup: dict[tuple[int, str, str], tuple[float, float, float]],
) -> tuple[float, float]:
    bounds = [
        value
        for (level, _, metric), (mean, half_width, _) in lookup.items()
        if level in forest.LEVEL_ORDER and metric in PARAMETER_RESPONSE_METRICS
        for value in (mean - half_width, mean + half_width)
    ]
    lower, upper = min(bounds), max(bounds)
    padding = max(0.02, 0.06 * (upper - lower))
    return lower - padding, upper + padding


def _format_parameter_response_axis(
    axis: plt.Axes,
    *,
    metric_limits: tuple[float, float],
) -> None:
    axis.set_xlim(6.0, 44.0)
    axis.set_xticks([10, 20, 30, 40])
    axis.set_ylim(*metric_limits)
    axis.set_axisbelow(True)
    axis.grid(axis="x", color="#D8DEE6", linewidth=0.55, alpha=0.75)
    axis.grid(axis="y", color="#E6EAF0", linewidth=0.55, alpha=0.8)
    axis.tick_params(length=2.8, width=0.65)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_linewidth(0.65)
    axis.spines["bottom"].set_linewidth(0.65)


def _plot_parameter_response_faceted(frame: pl.DataFrame, *, output_path: Path) -> Path:
    """Facet environments and make ambiguity threshold the primary axis."""
    _parameter_response_style()
    figure, axes = plt.subplots(
        2, 2, figsize=(8.8, 7.1), dpi=300, sharex=True, sharey=True,
        gridspec_kw={"hspace": 0.22, "wspace": 0.13},
    )
    metric_offsets = dict(zip(PARAMETER_RESPONSE_METRICS, (1.65, 0.55, -0.55, -1.65)))
    decay_offsets = {0.99: 0.12, 1.0: -0.12}
    lookup = forest._summary_lookup(frame)
    metric_limits = _parameter_response_ylim(lookup)

    for axis, level in zip(axes.flat, forest.LEVEL_ORDER):
        for metric in PARAMETER_RESPONSE_METRICS:
            color = PARAMETER_RESPONSE_COLORS[metric]
            for decay in (0.99, 1.0):
                filled = decay == 0.99
                trajectory_theta: list[float] = []
                trajectory_value: list[float] = []
                for threshold in (10, 20, 30, 40):
                    series_key = _series_key(threshold, decay)
                    result = lookup.get((level, series_key, metric))
                    if result is None:
                        continue
                    x = threshold + metric_offsets[metric] + decay_offsets[decay]
                    mean, half_width, _ = result
                    trajectory_theta.append(x)
                    trajectory_value.append(mean)
                    axis.errorbar(
                        x, mean, yerr=half_width, fmt="none", color=color,
                        elinewidth=1.0, capsize=2.5, capthick=0.9, zorder=3,
                    )
                    axis.scatter(
                        [x], [mean], marker="o", s=31,
                        facecolors=color if filled else "white", edgecolors=color,
                        linewidths=1.05, zorder=4,
                    )
                axis.plot(
                    trajectory_theta, trajectory_value, color=color, linewidth=0.75,
                    alpha=0.40, linestyle="-" if filled else (0, (3.0, 2.0)),
                    zorder=2,
                )
        _format_parameter_response_axis(axis, metric_limits=metric_limits)
        axis.set_title(forest.LEVEL_LABELS[level].replace("x", r"$\times$"), fontsize=10.5, fontweight="semibold", pad=5)

    figure.supxlabel(r"Ambiguity threshold, $\theta$", fontsize=10.0, y=0.055)
    figure.supylabel("Metric value", fontsize=10.0, x=0.04)
    metric_legend = figure.legend(
        handles=_metric_legend_handles(show_line=False, marker_size=7.0),
        loc="upper center", bbox_to_anchor=(0.39, 0.975),
        ncol=4, frameon=False, fontsize=12.5,
        columnspacing=0.70, handletextpad=0.30, borderaxespad=0.0,
    )
    figure.add_artist(metric_legend)
    figure.legend(
        handles=_decay_legend_handles(marker_size=7.0),
        loc="upper center", bbox_to_anchor=(0.89, 0.975),
        ncol=2, frameon=False, fontsize=12.5, handletextpad=0.30,
        columnspacing=0.85, borderaxespad=0.0,
    )
    figure.text(
        0.018, 0.953, "Metrics", ha="left", va="center", fontsize=12.5,
        fontweight="semibold", color="#475569",
    )
    figure.text(
        0.775, 0.953, r"$\gamma$", ha="left", va="center", fontsize=12.5,
        fontweight="semibold", color="#475569",
    )
    figure.add_artist(
        Line2D(
            [0.755, 0.755], [0.925, 0.982], transform=figure.transFigure,
            color="#D7DEE7", linewidth=0.65,
        )
    )
    figure.subplots_adjust(left=0.105, right=0.985, top=0.84, bottom=0.12)
    return _save_parameter_response_figure(figure, output_path)


def _plot_parameter_response_combined(frame: pl.DataFrame, *, output_path: Path) -> Path:
    """Combine environments by marker shape while retaining threshold rows."""
    _parameter_response_style()
    figure, axis = plt.subplots(figsize=(7.6, 6.5), dpi=300)
    metric_offsets = dict(zip(PARAMETER_RESPONSE_METRICS, (2.25, 0.75, -0.75, -2.25)))
    environment_offsets = dict(zip(forest.LEVEL_ORDER, (-0.27, -0.09, 0.09, 0.27)))
    decay_offsets = {0.99: 0.045, 1.0: -0.045}
    lookup = forest._summary_lookup(frame)
    metric_limits = _parameter_response_ylim(lookup)

    for metric in PARAMETER_RESPONSE_METRICS:
        color = PARAMETER_RESPONSE_COLORS[metric]
        for level in forest.LEVEL_ORDER:
            marker = ENVIRONMENT_MARKERS.get(level, "o")
            for decay in (0.99, 1.0):
                filled = decay == 0.99
                trajectory_theta: list[float] = []
                trajectory_value: list[float] = []
                for threshold in (10, 20, 30, 40):
                    series_key = _series_key(threshold, decay)
                    result = lookup.get((level, series_key, metric))
                    if result is None:
                        continue
                    x = (
                        threshold + metric_offsets[metric]
                        + environment_offsets[level] + decay_offsets[decay]
                    )
                    mean, half_width, _ = result
                    trajectory_theta.append(x)
                    trajectory_value.append(mean)
                    axis.errorbar(
                        x, mean, yerr=half_width, fmt="none", color=color,
                        elinewidth=0.8, capsize=2.0, capthick=0.75, alpha=0.9, zorder=3,
                    )
                    axis.scatter(
                        [x], [mean], marker=marker, s=24,
                        facecolors=color if filled else "white", edgecolors=color,
                        linewidths=0.9, zorder=4,
                    )
                axis.plot(
                    trajectory_theta, trajectory_value, color=color, linewidth=0.55,
                    alpha=0.25, linestyle="-" if filled else (0, (3.0, 2.0)),
                    zorder=2,
                )

    _format_parameter_response_axis(axis, metric_limits=metric_limits)
    axis.set_xlabel(r"Ambiguity threshold, $\theta$", fontsize=10.0, labelpad=7)
    axis.set_ylabel("Metric value", fontsize=10.0, labelpad=7)
    metric_legend = figure.legend(
        handles=_metric_legend_handles(), loc="upper left", bbox_to_anchor=(0.08, 0.995),
        ncol=4, frameon=False, fontsize=8.5, handlelength=1.5,
        columnspacing=1.0, handletextpad=0.4, title="Metric", title_fontsize=8.5,
    )
    figure.add_artist(metric_legend)
    environment_handles = [
        Line2D(
            [0], [0], marker=ENVIRONMENT_MARKERS.get(level, "o"), linestyle="none",
            markersize=5.4, markerfacecolor="white", markeredgecolor="#64748B",
            markeredgewidth=0.9, label=forest.LEVEL_LABELS[level].replace("x", "×"),
        )
        for level in forest.LEVEL_ORDER
    ]
    environment_legend = figure.legend(
        handles=environment_handles, loc="upper left", bbox_to_anchor=(0.08, 0.936),
        ncol=4, frameon=False, fontsize=8.4, handletextpad=0.35,
        columnspacing=1.05, title="Environment", title_fontsize=8.5,
    )
    figure.add_artist(environment_legend)
    figure.legend(
        handles=_decay_legend_handles(), loc="upper right", bbox_to_anchor=(0.96, 0.936),
        ncol=2, frameon=False, fontsize=8.4, handletextpad=0.35,
        columnspacing=0.8, title=r"$\gamma$", title_fontsize=8.5,
    )
    figure.subplots_adjust(left=0.12, right=0.98, top=0.80, bottom=0.11)
    return _save_parameter_response_figure(figure, output_path)


def _plot_focused_profile_environment_shapes(
    frame: pl.DataFrame,
    *,
    output_path: Path,
) -> Path:
    """Render seed observations, means, and 95% CIs in the compact style."""
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
    row_specs = forest._transposed_row_specs(
        forest.PROFILE_FOCUSED_METRICS,
        include_nll_axis=False,
    )
    row_gap = 1.18
    y_positions = np.arange(len(row_specs))[::-1].astype(float) * row_gap
    row_y = {metric: y_positions[index] for index, (metric, _) in enumerate(row_specs)}
    threshold_offsets = dict(zip((10, 20, 30, 40), np.linspace(0.36, -0.36, 4)))
    decay_offsets = {0.99: 0.055, 1.0: -0.055}
    lookup = forest._summary_lookup(frame)

    # Match the width:height ratio of one panel in the three-column combined
    # profile (approximately 0.65), rather than that of the full figure.
    figure, axis = plt.subplots(figsize=(3.0, 4.6), dpi=300)
    for metric_index, (metric, _) in enumerate(row_specs):
        base_y = row_y[metric]
        if metric_index > 0:
            axis.axhline(base_y + row_gap / 2, color="#D7DEE7", linewidth=0.55, zorder=0)

        for series_key, (threshold, decay) in zip(SERIES_ORDER, PARAMETER_ORDER):
            color = THRESHOLD_COLORS[threshold]
            filled = decay == 0.99
            setting_y = base_y + threshold_offsets[threshold] + decay_offsets[decay]
            for level in forest.LEVEL_ORDER:
                result = lookup.get((level, series_key, metric))
                if result is None:
                    continue
                environment_y = setting_y + ENVIRONMENT_OFFSETS.get(level, 0.0)
                marker = ENVIRONMENT_MARKERS.get(level, "o")
                seed_values = (
                    frame.filter(
                        (pl.col("level") == level)
                        & (pl.col("plot_series") == series_key)
                    )
                    .drop_nulls(["seed", metric])
                    .group_by("seed")
                    .agg(pl.col(metric).mean().alias("value"))
                    .sort("seed")
                )
                seed_offsets = np.linspace(-0.003, 0.003, len(seed_values))
                axis.scatter(
                    seed_values["value"].to_numpy(),
                    environment_y + seed_offsets,
                    marker=marker,
                    s=3.0,
                    facecolors=color if filled else "none",
                    edgecolors="none" if filled else color,
                    alpha=0.12,
                    linewidths=0 if filled else 0.30,
                    zorder=1,
                )
                mean, half_width, _ = result
                axis.errorbar(
                    mean,
                    environment_y,
                    xerr=half_width,
                    fmt="none",
                    color=color,
                    ecolor=color,
                    elinewidth=0.72,
                    capsize=2.2,
                    capthick=0.72,
                    zorder=3,
                )
                axis.scatter(
                    [mean],
                    [environment_y],
                    marker=marker,
                    s=12.0,
                    facecolors=color if filled else "none",
                    edgecolors="white" if filled else color,
                    linewidths=0.45 if filled else 0.85,
                    zorder=4,
                )

    axis.set_xlim(0.0, 1.0)
    axis.set_xticks([0.0, 0.5, 1.0], ["0", "0.5", "1.0"])
    axis.set_ylim(y_positions[-1] - 0.54, y_positions[0] + 0.54)
    axis.set_yticks(
        y_positions,
        [
            forest.COMPACT_TRANSPOSED_METRIC_LABELS.get(metric, label)
            for metric, label in row_specs
        ],
    )
    axis.set_axisbelow(True)
    axis.grid(axis="x", color="#D8DEE6", linewidth=0.55, alpha=0.8)
    axis.tick_params(axis="x", labelsize=8.2, length=2.4, width=0.6, pad=1.8)
    axis.tick_params(axis="y", labelsize=8.7, length=0, pad=4.0)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_visible(False)
    axis.spines["bottom"].set_linewidth(0.6)
    threshold_handles = [
        Line2D(
            [0], [0], marker="o", linestyle="none", markersize=4.8,
            markerfacecolor=THRESHOLD_COLORS[threshold], markeredgecolor="none",
            label=str(threshold),
        )
        for threshold in (10, 20, 30, 40)
    ]
    threshold_legend = figure.legend(
        handles=threshold_handles,
        loc="center left",
        bbox_to_anchor=(0.12, 0.958),
        ncol=4,
        frameon=False,
        fontsize=6.0,
        handletextpad=0.12,
        columnspacing=0.30,
        borderaxespad=0.0,
    )
    figure.add_artist(threshold_legend)
    decay_handles = [
        Line2D(
            [0], [0], marker="o", linestyle="none", markersize=4.8,
            markerfacecolor="#64748B" if filled else "none",
            markeredgecolor="#64748B", markeredgewidth=0.8,
            label=label,
        )
        for filled, label in ((True, "0.99"), (False, "1"))
    ]
    decay_legend = figure.legend(
        handles=decay_handles,
        loc="center left",
        bbox_to_anchor=(0.69, 0.958),
        ncol=2,
        frameon=False,
        fontsize=6.0,
        handletextpad=0.12,
        columnspacing=0.28,
        borderaxespad=0.0,
    )
    figure.add_artist(decay_legend)
    environment_handles = [
        Line2D(
            [0], [0], marker=ENVIRONMENT_MARKERS.get(level, "o"), linestyle="none",
            markersize=4.8, markerfacecolor="none", markeredgecolor="#64748B",
            markeredgewidth=0.8,
            label=forest.LEVEL_LABELS[level].replace("x", "×"),
        )
        for level in forest.LEVEL_ORDER
    ]
    figure.legend(
        handles=environment_handles,
        loc="center left",
        bbox_to_anchor=(0.19, 0.905),
        ncol=4,
        frameon=False,
        fontsize=5.8,
        handletextpad=0.12,
        columnspacing=0.26,
        borderaxespad=0.0,
    )
    figure.text(
        0.04, 0.958, r"$\theta$:",
        ha="left", va="center", fontsize=6.6, fontweight="semibold", color="#475569",
    )
    figure.text(
        0.61, 0.958, r"$\gamma$:",
        ha="left", va="center", fontsize=6.6, fontweight="semibold", color="#475569",
    )
    figure.text(
        0.04, 0.905, "Env.:",
        ha="left", va="center", fontsize=6.4, fontweight="semibold", color="#475569",
    )
    figure.supxlabel("Metric value", fontsize=8.2, y=0.035)
    figure.subplots_adjust(left=0.20, right=0.97, top=0.865, bottom=0.11)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=600)
    figure.savefig(output_path.with_suffix(".pdf"))
    plt.close(figure)
    return output_path


def _load_rows(study_root: Path, protocol_id: str) -> list[dict[str, Any]]:
    manifest_path = study_root / "study_manifest.jsonl"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Study manifest not found: {manifest_path}")

    allowed = set(PARAMETER_ORDER)
    rows: list[dict[str, Any]] = []
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        config = dict(json.loads(line)["config"])
        if str(config.get("method")) != "mercury":
            continue
        parameters = (int(config["ambiguity_threshold"]), float(config["ambiguity_decay"]))
        if parameters not in allowed:
            continue
        run_id = str(config.get("requested_run_id") or config.get("run_id"))
        run_root = study_root / run_id
        record = load_evaluation_step(run_root, protocol_id, method="mercury")
        if record is None:
            continue
        threshold, decay = parameters
        rows.append(
            {
                "run_id": run_id,
                "run_root": run_root,
                "level": int(config["level"]),
                "seed": int(config["seed"]),
                "rand_prob": float(config["rand_prob"]),
                "ambiguity_threshold": threshold,
                "ambiguity_decay": decay,
                "series": _series_key(threshold, decay),
                "record": record,
            }
        )
    if not rows:
        raise RuntimeError(f"No MERCURY records for protocol {protocol_id} under {study_root}")
    return rows


def _configure_styles(levels: list[int]) -> None:
    forest.LEVEL_ORDER[:] = levels
    forest.SERIES_LABELS.update(SERIES_LABELS)
    forest.SERIES_COLORS.update(SERIES_COLORS)
    threshold_data.METHOD_COLORS.update(
        {SERIES_LABELS[key]: SERIES_COLORS[key] for key in SERIES_ORDER}
    )
    robustness.LEVEL_ORDER = tuple(levels)
    for index, (threshold, decay) in enumerate(PARAMETER_ORDER):
        key = _series_key(threshold, decay)
        label = SERIES_LABELS[key]
        robustness.METHOD_LABELS[label] = label
        robustness.METHOD_STYLES[label] = {
            "color": SERIES_COLORS[key],
            "linestyle": "-" if decay == 0.99 else (0, (5.5, 2.2)),
        }
        robustness.METHOD_ORDER[label] = index
    robustness.SCENARIOS[SCENARIO_KEY] = {
        "title": "Threshold robustness of ambiguity threshold and decay",
        "subtitle": "Final checkpoint; ribbons show 95% confidence intervals",
        "paired_output": "threshold_robustness_age_ambiguity_edge_f1_2x4.png",
    }


def _frame(rows: list[dict[str, Any]]) -> pl.DataFrame:
    return pl.DataFrame(
        [
            {
                "comparison": "final",
                "level": row["level"],
                "seed": row["seed"],
                "rand_prob": row["rand_prob"],
                "ambiguity_threshold": row["ambiguity_threshold"],
                "ambiguity_decay": row["ambiguity_decay"],
                "plot_series": row["series"],
                **forest._metrics_from_record(row["record"]),
            }
            for row in rows
        ],
        infer_schema_length=None,
    )


def _summary_plots(frame: pl.DataFrame, output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    title = "Final ambiguity threshold and decay ablation"
    common = {
        "frame": frame,
        "series_order": list(SERIES_ORDER),
        "title": title,
        "plot_dir": output_dir,
        "series_labels": SERIES_LABELS,
    }
    return [
        forest.plot_forest(
            output_name="age_ambiguity_forest_6x4.png",
            emphasize_first=False,
            **common,
        ),
        forest.plot_metric_matrix(
            output_name="age_ambiguity_metric_matrix.png",
            gain_label=None,
            emphasize_first=False,
            show_gain=False,
            **common,
        ),
        forest.plot_metric_profile(
            output_name="age_ambiguity_metric_profile.png",
            emphasize_first=False,
            **common,
        ),
        _plot_focused_profile_environment_shapes(
            frame,
            output_path=output_dir / "age_ambiguity_metric_profile_focused.png",
        ),
        _plot_parameter_response_faceted(
            frame,
            output_path=output_dir / "age_ambiguity_parameter_response_faceted.png",
        ),
        _plot_parameter_response_combined(
            frame,
            output_path=output_dir / "age_ambiguity_parameter_response_combined.png",
        ),
        forest.plot_metric_profile_transposed(
            output_name="age_ambiguity_metric_profile_focused_transposed.png",
            metric_keys=forest.PROFILE_FOCUSED_METRICS,
            emphasize_first=False,
            **common,
        ),
    ]


def _tv_plots(frame: pl.DataFrame, output_dir: Path) -> tuple[list[Path], list[dict[str, Any]]]:
    rand_prob = float(frame["rand_prob"][0])
    outputs: list[Path] = []
    csv_rows: list[dict[str, Any]] = []
    for spec in tv.METRIC_SPECS:
        path = (
            output_dir
            / "final"
            / spec["family"]
            / spec["self_loop_policy"]
            / f"age_ambiguity_{spec['family']}_{spec['self_loop_policy']}.png"
        )
        outputs.append(
            tv._plot_one(
                frame,
                output_path=path,
                rand_prob=rand_prob,
                comparison_label="Final ambiguity threshold and decay ablation",
                metric=str(spec["column"]),
                metric_label=str(spec["label"]),
                self_loop_policy=str(spec["self_loop_policy"]),
                series_order=list(SERIES_ORDER),
                primary_series=None,
                series_labels=SERIES_LABELS,
                series_colors=SERIES_COLORS,
            )
        )
        csv_rows.extend(
            tv._summaries_for_csv(
                frame,
                rand_prob=rand_prob,
                comparison="final",
                metric=str(spec["column"]),
                metric_label=str(spec["label"]),
                self_loop_policy=str(spec["self_loop_policy"]),
                series_order=list(SERIES_ORDER),
                series_labels=SERIES_LABELS,
            )
        )
    return outputs, csv_rows


def _residual_plots(rows: list[dict[str, Any]], output_dir: Path, levels: list[int]) -> list[Path]:
    outputs: list[Path] = []
    for action_conditioned in (False, True):
        for include_self_loops in (False, True):
            panels: dict[str, Any] = {}
            for level in levels:
                for key in SERIES_ORDER:
                    diagnostics = [
                        diagnostics_from_record(
                            row["record"],
                            method=SERIES_LABELS[key],
                            action_conditioned=action_conditioned,
                            ignore_self_loops=not include_self_loops,
                        )
                        for row in rows
                        if row["level"] == level and row["series"] == key
                    ]
                    if diagnostics:
                        panels[f"L{level}/{SERIES_LABELS[key]}"] = _aggregate_diagnostics(diagnostics).to_json_dict()
            scope = "action_conditioned" if action_conditioned else "collapsed"
            loops = "with_self_loops" if include_self_loops else "without_self_loops"
            path = output_dir / scope / loops / f"age_ambiguity_transition_residual_{scope}_{loops}.png"
            payload = {
                "scenario": "Ground-truth-aligned transition residuals: ambiguity threshold and decay",
                "panels": panels,
                "visual_encoding": {"action_conditioned": action_conditioned},
            }
            save_selected_figure(payload=payload, output_path=path, vis_threshold=0.02)
            path.with_suffix(".json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
            outputs.append(path)
    return outputs


def _threshold_rows(
    rows: list[dict[str, Any]],
    *,
    thresholds: Iterable[float],
    action_conditioned: bool,
    include_self_loops: bool,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        label = SERIES_LABELS[row["series"]]
        for threshold in thresholds:
            metrics = edge_metrics_at_threshold(
                row["record"],
                float(threshold),
                action_conditioned=action_conditioned,
                include_self_loops=include_self_loops,
            )
            output.append(
                {
                    "scenario_key": SCENARIO_KEY,
                    "scenario_title": "Ambiguity threshold and decay ablation",
                    "level": row["level"],
                    "seed": row["seed"],
                    "method": label,
                    "run_kind": row["series"],
                    "run_root": str(row["run_root"]),
                    "metric_scope": "action_conditioned" if action_conditioned else "collapsed",
                    "threshold": float(threshold),
                    **metrics,
                }
            )
    return output


def _threshold_plots(
    rows: list[dict[str, Any]],
    output_dir: Path,
    *,
    thresholds: tuple[float, ...],
) -> list[Path]:
    outputs: list[Path] = []
    for include_self_loops in (False, True):
        loops = "with_self_loops" if include_self_loops else "without_self_loops"
        summary_paths: dict[str, Path] = {}
        for action_conditioned in (False, True):
            scope = "action_conditioned" if action_conditioned else "collapsed"
            scope_dir = output_dir / scope / loops
            scope_dir.mkdir(parents=True, exist_ok=True)
            raw = _threshold_rows(
                rows,
                thresholds=thresholds,
                action_conditioned=action_conditioned,
                include_self_loops=include_self_loops,
            )
            aggregate = threshold_data._aggregate_rows(raw)
            raw_path = scope_dir / "threshold_edge_metrics_by_seed.csv"
            summary_path = scope_dir / "threshold_edge_metrics_summary.csv"
            threshold_data._write_csv(raw_path, raw)
            threshold_data._write_csv(summary_path, aggregate)
            summary_paths[scope] = summary_path
            outputs.extend((raw_path, summary_path))

        plot_dir = output_dir / "robustness" / loops
        robustness.BASE_OUTPUT_DIR = plot_dir
        action = robustness._read_summary(summary_paths["action_conditioned"], "action_conditioned")
        collapsed = robustness._read_summary(summary_paths["collapsed"], "collapsed")
        outputs.append(robustness.plot_final_paired(action, collapsed, scenario=SCENARIO_KEY))
        outputs.append(
            robustness.plot_full_metric_grid(
                action,
                scenario=SCENARIO_KEY,
                scope_label="Action-conditioned",
                output_name="threshold_metrics_age_ambiguity_action_conditioned_3x4.png",
            )
        )
        outputs.append(
            robustness.plot_full_metric_grid(
                collapsed,
                scenario=SCENARIO_KEY,
                scope_label="Policy-collapsed",
                output_name="threshold_metrics_age_ambiguity_collapsed_3x4.png",
            )
        )
    return outputs


def generate_plots(
    *,
    study_root: Path,
    output_dir: Path,
    protocol_id: str,
    thresholds: tuple[float, ...] = DEFAULT_THRESHOLDS,
) -> list[Path]:
    rows = _load_rows(study_root, protocol_id)
    canonical_order = [13, 21, 20, 22]
    available = {int(row["level"]) for row in rows}
    levels = [level for level in canonical_order if level in available]
    levels.extend(sorted(available - set(levels)))
    _configure_styles(levels)
    frame = _frame(rows)
    rand_slug = "rand_" + f"{float(frame['rand_prob'][0]):g}".replace("-", "m").replace(".", "p")

    outputs = _summary_plots(frame, output_dir / "forest_matrix_profile")
    tv_outputs, tv_rows = _tv_plots(frame, output_dir / "tv_similarity" / rand_slug)
    outputs.extend(tv_outputs)
    tv_summary = output_dir / "tv_similarity" / rand_slug / "tv_similarity_summary.csv"
    if tv_rows:
        tv_summary.parent.mkdir(parents=True, exist_ok=True)
        with tv_summary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(tv_rows[0]))
            writer.writeheader()
            writer.writerows(tv_rows)
        outputs.append(tv_summary)
    outputs.extend(_residual_plots(rows, output_dir / "residual_maps" / rand_slug, levels))
    outputs.extend(
        _threshold_plots(
            rows,
            output_dir / "threshold_edge_robustness" / rand_slug,
            thresholds=thresholds,
        )
    )
    # Every raster figure produced by this suite has a vector companion.  Include
    # both in the manifest so a stale PDF cannot silently survive a PNG-only rerun.
    manifested: list[Path] = []
    for path in outputs:
        manifested.append(path)
        if path.suffix.lower() == ".png":
            pdf_path = path.with_suffix(".pdf")
            if not pdf_path.is_file():
                raise FileNotFoundError(f"missing PDF companion for {path}")
            manifested.append(pdf_path)
    return manifested


def _parse_thresholds(raw: str) -> tuple[float, ...]:
    values = tuple(float(part.strip()) for part in raw.replace(" ", ",").split(",") if part.strip())
    if not values:
        raise argparse.ArgumentTypeError("at least one threshold is required")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study_root", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--protocol_id", required=True)
    parser.add_argument(
        "--thresholds",
        type=_parse_thresholds,
        default=DEFAULT_THRESHOLDS,
        help="Comma-separated edge thresholds",
    )
    args = parser.parse_args()
    for path in generate_plots(
        study_root=args.study_root,
        output_dir=args.output_dir,
        protocol_id=args.protocol_id,
        thresholds=tuple(args.thresholds),
    ):
        print(path)


if __name__ == "__main__":
    main()
