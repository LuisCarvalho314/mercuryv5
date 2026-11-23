"""Mercury-only structural trajectories and sensory-template-length distribution.

Companion to ``plot_config_metric_trajectories.py``: same study, same visual
language, but restricted to the Mercury phase of every ``mercury_cscg_hybrid``
run (plus standalone ``mercury`` runs when present), because the quantities
plotted here only exist while Mercury is growing its graph.

Two figures are produced:

1. ``<stem>.png`` -- per-step trajectories, one column per level:
   * learned/true node ratio (``latent_node_count / n_true_states``),
   * cumulative cloning events (``clone_created_cumulative``),
   * cumulative node-deletion events (``node_prune_cumulative``).
2. ``<template-stem>.png`` -- distribution of sensory template lengths
   from the final latent graph: a jittered scatter of every node's template
   length with the mean and a 95% CI over seed-level means.

A node's sensory template is the ``mem_adj`` row reshaped to
(sensory_channels, memory_length); its length is the number of memory lags that
carry any non-zero weight. Note this is one more than ``bmu_memory_depth`` in
the per-step metrics, which reports the deepest active lag *index*.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
from scipy import stats
from mercury_runs.analysis.plots import publication_style as pub


DEFAULT_STUDY_ROOT = Path("results/current/studies/default_matrix_smoke_30_seed_onehot_clean_more_mazes")
DEFAULT_PLOT_DIR = Path("results/current/notebook_plots/default_matrix_smoke_30_seed_onehot_clean_more_mazes/metric_trajectories")
DEFAULT_PROTOCOL_ID = "canonical-v4-8922ca1c45c19cc7"

LEVEL_LABELS = {13: "3x3", 20: "1x5", 21: "1x4", 22: "3x4"}
DEFAULT_LEVEL_ORDER = (13, 21, 20, 22)

# The graph-learning phase uses the shared warm MERCURY identity colour.
MERCURY_COLOR = pub.method_color("mercury")
CLONE_COLOR = "#009E73"
DELETION_COLOR = "#D55E00"
REFERENCE_COLOR = "#94A3B8"

# Methods whose bundles contain a Mercury learning phase, mapped to the bundle
# sub-directory that holds it (`None` when the bundle root is already Mercury).
MERCURY_PHASE_METHODS = {
    "mercury_cscg_hybrid": "mercury",
    "mercury": None,
}

METRICS = (
    ("node_ratio", "Learned / true\nnodes", MERCURY_COLOR),
    ("clone_created_cumulative", "Cloning events\n(cumulative)", CLONE_COLOR),
    ("node_prune_cumulative", "Deletion events\n(cumulative)", DELETION_COLOR),
)

PARQUET_COLUMNS = (
    "latent_node_count",
    "clone_created_cumulative",
    "node_prune_cumulative",
)


@dataclass(frozen=True)
class RunEntry:
    run_id: str
    method: str
    level: int
    seed: int
    rand_prob: float
    batch_size: int
    memory_length: int
    run_root: Path


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_manifest(study_root: Path, batch_size: int, levels: set[int] | None) -> list[RunEntry]:
    manifest_path = study_root / "study_manifest.jsonl"
    entries: list[RunEntry] = []
    with manifest_path.open(encoding="utf-8") as handle:
        for raw in handle:
            if not raw.strip():
                continue
            config = dict(json.loads(raw)["config"])
            method = str(config.get("method", ""))
            if method not in MERCURY_PHASE_METHODS:
                continue
            if int(config.get("cscg_batch_size", batch_size)) != int(batch_size):
                continue
            level = int(config["level"])
            if levels is not None and level not in levels:
                continue
            run_id = str(config["requested_run_id"])
            entries.append(
                RunEntry(
                    run_id=run_id,
                    method=method,
                    level=level,
                    seed=int(config["seed"]),
                    rand_prob=float(config.get("rand_prob", 0.0)),
                    batch_size=int(config.get("cscg_batch_size", batch_size)),
                    memory_length=int(config.get("memory_length", 0)),
                    run_root=study_root / run_id,
                )
            )
    return entries


def _mercury_bundle_file(entry: RunEntry, suffix: str) -> Path | None:
    phase_dir = MERCURY_PHASE_METHODS[entry.method]
    pattern = (
        f"bundles/{entry.method}/{phase_dir}/**/*{suffix}"
        if phase_dir is not None
        else f"bundles/{entry.method}/**/*{suffix}"
    )
    candidates = sorted(entry.run_root.glob(pattern))
    return candidates[0] if candidates else None


def _true_state_count(entry: RunEntry, protocol_id: str) -> int | None:
    """Ground-truth state count, preferring the canonical evaluation record."""
    snapshot_root = entry.run_root / "evaluations" / protocol_id / "snapshots"
    namespaces = [f"{entry.method}_mercury", entry.method, "mercury"]
    for namespace in namespaces:
        directory = snapshot_root / namespace
        if not directory.exists():
            continue
        for path in sorted(directory.glob("step=*.json")):
            record = _read_json(path)
            n_true = record.get("state_representation", {}).get("n_true_states")
            if n_true:
                return int(n_true)
    ground_truth = entry.run_root / "ground_truth" / f"{entry.run_id}_ground_truth_states.parquet"
    if ground_truth.exists():
        frame = pd.read_parquet(ground_truth, columns=["ground_truth_state_id"])
        return int(frame["ground_truth_state_id"].nunique())
    return None


def _template_lengths(entry: RunEntry) -> np.ndarray | None:
    """Number of memory lags carrying weight, per node of the final graph."""
    graph_path = _mercury_bundle_file(entry, "_latent_graph.npz")
    if graph_path is None or entry.memory_length <= 0:
        return None
    with np.load(graph_path, allow_pickle=True) as archive:
        if "node__mem_adj" not in archive:
            return None
        node_count = int(archive["n"])
        memory = np.asarray(archive["node__mem_adj"], dtype=np.float32)[:node_count]
    if node_count == 0 or memory.size == 0:
        return None
    if memory.shape[1] % entry.memory_length != 0:
        raise ValueError(
            f"{entry.run_id}: mem_adj width {memory.shape[1]} is not divisible by "
            f"memory_length={entry.memory_length}"
        )
    lags = memory.reshape(node_count, -1, entry.memory_length) > 0.0
    return lags.any(axis=1).sum(axis=1).astype(np.int32)


def load_trajectory_rows(study_root: Path, protocol_id: str, batch_size: int, levels: set[int] | None) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for entry in _load_manifest(study_root, batch_size=batch_size, levels=levels):
        states_path = _mercury_bundle_file(entry, "_states.parquet")
        if states_path is None:
            continue
        n_true = _true_state_count(entry, protocol_id)
        if not n_true:
            continue
        states = pd.read_parquet(states_path, columns=list(PARQUET_COLUMNS))
        frame = pd.DataFrame(
            {
                "step": np.arange(1, len(states) + 1, dtype=np.int64),
                "node_ratio": states["latent_node_count"].to_numpy(dtype=float) / float(n_true),
                "clone_created_cumulative": states["clone_created_cumulative"].to_numpy(dtype=float),
                "node_prune_cumulative": states["node_prune_cumulative"].to_numpy(dtype=float),
            }
        )
        frame = frame.melt(id_vars="step", var_name="metric", value_name="value")
        frame["run_id"] = entry.run_id
        frame["method"] = entry.method
        frame["level"] = entry.level
        frame["level_label"] = LEVEL_LABELS.get(entry.level, f"L{entry.level}")
        frame["seed"] = entry.seed
        frame["rand_prob"] = entry.rand_prob
        frame["batch_size"] = entry.batch_size
        frame["n_true_states"] = n_true
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_template_rows(study_root: Path, batch_size: int, levels: set[int] | None) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for entry in _load_manifest(study_root, batch_size=batch_size, levels=levels):
        lengths = _template_lengths(entry)
        if lengths is None:
            continue
        for node_index, length in enumerate(lengths.tolist()):
            rows.append(
                {
                    "run_id": entry.run_id,
                    "method": entry.method,
                    "level": entry.level,
                    "level_label": LEVEL_LABELS.get(entry.level, f"L{entry.level}"),
                    "seed": entry.seed,
                    "rand_prob": entry.rand_prob,
                    "batch_size": entry.batch_size,
                    "memory_length": entry.memory_length,
                    "node_index": node_index,
                    "template_length": int(length),
                }
            )
    return pd.DataFrame(rows)


def _confidence_interval_half_width(sd: np.ndarray, n: np.ndarray, confidence: float = 0.95) -> np.ndarray:
    """Half-width of a t-based CI; zero wherever fewer than two samples exist."""
    sd = np.asarray(sd, dtype=float)
    n = np.asarray(n, dtype=float)
    half_width = np.zeros_like(sd, dtype=float)
    usable = n >= 2
    if np.any(usable):
        critical = stats.t.ppf(0.5 + confidence / 2.0, n[usable] - 1.0)
        half_width[usable] = critical * sd[usable] / np.sqrt(n[usable])
    return half_width


def summarise_trajectories(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return rows
    grouped = (
        rows.groupby(["level", "level_label", "rand_prob", "metric", "step"], dropna=False)["value"]
        .agg(["count", "mean", "std"])
        .reset_index()
        .rename(columns={"count": "n", "std": "sd"})
    )
    grouped["sd"] = grouped["sd"].fillna(0.0)
    grouped["sem"] = grouped["sd"] / np.sqrt(grouped["n"].clip(lower=1))
    grouped["ci95"] = _confidence_interval_half_width(grouped["sd"].to_numpy(), grouped["n"].to_numpy())
    return grouped


def summarise_templates(rows: pd.DataFrame) -> pd.DataFrame:
    """Pool nodes for the cloud, but base the CI on seed-level means.

    Nodes within a run are not independent samples, so a CI over all nodes would
    be far too tight; the seed means are the independent replicates.
    """
    if rows.empty:
        return rows
    seed_means = (
        rows.groupby(["level", "level_label", "run_id", "seed"], dropna=False)["template_length"]
        .mean()
        .reset_index()
    )
    summary = (
        seed_means.groupby(["level", "level_label"], dropna=False)["template_length"]
        .agg(["count", "mean", "std"])
        .reset_index()
        .rename(columns={"count": "n_runs", "mean": "mean", "std": "sd"})
    )
    summary["sd"] = summary["sd"].fillna(0.0)
    summary["ci95"] = _confidence_interval_half_width(summary["sd"].to_numpy(), summary["n_runs"].to_numpy())
    node_stats = (
        rows.groupby(["level", "level_label"], dropna=False)["template_length"]
        .agg(["count", "median", "min", "max"])
        .reset_index()
        .rename(columns={"count": "n_nodes", "median": "node_median", "min": "node_min", "max": "node_max"})
    )
    return summary.merge(node_stats, on=["level", "level_label"], how="left")


def _apply_style() -> None:
    plt.style.use("default")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.edgecolor": "#2F3740",
            "axes.labelcolor": "#1F2933",
            "xtick.color": "#1F2933",
            "ytick.color": "#1F2933",
        }
    )


def _ordered_levels(values: set[int]) -> list[int]:
    levels = [level for level in DEFAULT_LEVEL_ORDER if level in values]
    levels.extend(sorted(values.difference(levels)))
    return levels


def _style_panel(axis: plt.Axes) -> None:
    axis.grid(axis="y", color="#CBD5E1", linewidth=0.55, alpha=0.55)
    axis.grid(axis="x", color="#E5E7EB", linewidth=0.45, alpha=0.42)
    axis.tick_params(axis="both", labelsize=6.9, length=2.5, pad=1.5)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def plot_structural_trajectories(
    summary: pd.DataFrame,
    plot_dir: Path,
    *,
    output_stem: str,
    title: str,
    x_scale: str = "linear",
) -> list[Path]:
    if summary.empty:
        raise ValueError("No Mercury trajectory rows available to plot.")

    _apply_style()
    levels = _ordered_levels(set(summary["level"]))
    rand_values = sorted(summary["rand_prob"].unique())
    rand_label = f"rand={rand_values[0]:g}" if len(rand_values) == 1 else "all rand"

    n_rows = len(METRICS)
    n_cols = len(levels)
    figure, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(2.75 * n_cols + 1.9, 1.55 * n_rows + 1.55),
        sharex=True,
        dpi=300,
        gridspec_kw={"hspace": 0.2, "wspace": 0.12},
    )
    axes = np.asarray(axes).reshape(n_rows, n_cols)

    for row_index, (metric, metric_label, color) in enumerate(METRICS):
        metric_frame = summary[summary["metric"] == metric]
        upper = float((metric_frame["mean"] + metric_frame["ci95"]).max())
        lower = float((metric_frame["mean"] - metric_frame["ci95"]).min())
        if metric == "node_ratio":
            upper = max(upper, 1.0)
            lower = min(lower, 1.0)
        padding = 0.06 * max(upper - lower, 1e-6)
        y_limits = (max(0.0, lower - padding), upper + padding)
        for col_index, level in enumerate(levels):
            axis = axes[row_index, col_index]
            panel = metric_frame[metric_frame["level"] == level].sort_values("step")
            if not panel.empty:
                x = panel["step"].to_numpy(dtype=float)
                y = panel["mean"].to_numpy(dtype=float)
                ci = panel["ci95"].to_numpy(dtype=float)
                axis.fill_between(x, y - ci, y + ci, color=color, alpha=0.16, linewidth=0)
                axis.plot(x, y, color=color, linewidth=1.35, alpha=0.96)
            if metric == "node_ratio":
                axis.axhline(1.0, color=REFERENCE_COLOR, linewidth=0.7, linestyle=(0, (2.0, 2.2)), alpha=0.8, zorder=0)
            axis.set_xscale(x_scale)
            axis.set_ylim(*y_limits)
            _style_panel(axis)
            if col_index == 0:
                axis.set_ylabel(metric_label, fontsize=7.7, fontweight="bold", ha="center", va="center")
                axis.yaxis.set_label_coords(-0.235, 0.5)
            else:
                axis.tick_params(axis="y", labelleft=False)
                axis.spines["left"].set_visible(False)
            if row_index == 0:
                axis.set_title(LEVEL_LABELS.get(int(level), f"L{level}"), fontsize=9.2, fontweight="bold", pad=4)
            if row_index < n_rows - 1:
                axis.tick_params(axis="x", labelbottom=False)

    seed_counts = sorted(set(summary["n"]))
    seed_label = f"n={seed_counts[0]} seeds" if len(seed_counts) == 1 else f"n={seed_counts[0]}-{seed_counts[-1]} seeds"
    figure.suptitle(f"{title} ({rand_label}, {seed_label}; mean ± 95% CI)", fontsize=11.5, y=0.992)
    scale_note = " (log scale)" if x_scale == "log" else ""
    figure.supxlabel(f"Mercury learning step{scale_note}", fontsize=8.8, y=0.025)
    figure.subplots_adjust(left=0.092, right=0.995, top=0.885, bottom=0.095)

    output = plot_dir / f"{output_stem}.png"
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)
    return [output]


def plot_template_lengths(
    rows: pd.DataFrame,
    summary: pd.DataFrame,
    plot_dir: Path,
    *,
    output_stem: str,
    title: str,
    seed: int = 0,
) -> list[Path]:
    if rows.empty:
        raise ValueError("No Mercury sensory templates available to plot.")

    _apply_style()
    levels = _ordered_levels(set(rows["level"]))
    rand_values = sorted(rows["rand_prob"].unique())
    rand_label = f"rand={rand_values[0]:g}" if len(rand_values) == 1 else "all rand"

    figure, axis = plt.subplots(figsize=(1.15 * len(levels) + 2.2, 3.6), dpi=300)
    generator = np.random.default_rng(seed)

    for position, level in enumerate(levels):
        values = rows.loc[rows["level"] == level, "template_length"].to_numpy(dtype=float)
        if values.size == 0:
            continue
        # Jitter both axes: template lengths are small integers, so identical
        # points would otherwise stack invisibly on one pixel.
        x_jitter = position - 0.1 + generator.uniform(-0.16, 0.16, size=values.size)
        y_jitter = values + generator.uniform(-0.09, 0.09, size=values.size)
        axis.scatter(
            x_jitter,
            y_jitter,
            s=6.0,
            color=MERCURY_COLOR,
            alpha=0.26,
            linewidths=0,
            zorder=2,
        )

    for position, level in enumerate(levels):
        stats_row = summary[summary["level"] == level]
        if stats_row.empty:
            continue
        mean = float(stats_row["mean"].iloc[0])
        ci95 = float(stats_row["ci95"].iloc[0])
        axis.errorbar(
            position + 0.26,
            mean,
            yerr=ci95,
            fmt="o",
            markersize=4.6,
            color="#1F2933",
            ecolor="#1F2933",
            elinewidth=1.2,
            capsize=3.2,
            capthick=1.2,
            zorder=3,
        )
        axis.annotate(
            f"{mean:.2f}",
            xy=(position + 0.26, mean + ci95),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            fontsize=7.0,
            color="#1F2933",
        )

    axis.set_xticks(range(len(levels)))
    axis.set_xticklabels([LEVEL_LABELS.get(int(level), f"L{level}") for level in levels], fontsize=8.4, fontweight="bold")
    axis.set_xlim(-0.5, len(levels) - 0.4)
    max_length = float(rows["template_length"].max())
    axis.set_ylim(0.4, max_length + 0.6)
    axis.yaxis.set_major_locator(MaxNLocator(integer=True))
    axis.set_ylabel("Sensory template length (memory lags)", fontsize=8.4)
    axis.set_xlabel("Maze", fontsize=8.4)
    _style_panel(axis)
    axis.grid(axis="x", visible=False)

    node_total = int(summary["n_nodes"].sum())
    run_total = int(summary["n_runs"].sum())
    figure.suptitle(
        f"{title} ({rand_label}; {node_total} nodes over {run_total} runs)",
        fontsize=10.4,
        y=0.99,
    )
    figure.text(
        0.5,
        0.943,
        "Each dot is one latent node (jittered); black marker is the mean over seed means with 95% CI",
        ha="center",
        fontsize=7.0,
        color="#4B5563",
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.925))

    output = plot_dir / f"{output_stem}.png"
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)
    return [output]


def _parse_levels(raw: str | None) -> set[int] | None:
    if raw is None or not raw.strip():
        return None
    return {int(item.strip()) for item in raw.split(",") if item.strip()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Mercury-only structural trajectories and sensory template lengths.")
    parser.add_argument("--study-root", type=Path, default=DEFAULT_STUDY_ROOT)
    parser.add_argument("--plot-dir", type=Path, default=DEFAULT_PLOT_DIR)
    parser.add_argument("--protocol-id", type=str, default=DEFAULT_PROTOCOL_ID)
    parser.add_argument("--batch-size", type=int, default=40)
    parser.add_argument("--levels", type=str, default=",".join(str(level) for level in DEFAULT_LEVEL_ORDER))
    parser.add_argument("--output-stem", type=str, default="mercury_structural_trajectories")
    parser.add_argument("--template-output-stem", type=str, default="mercury_sensory_template_lengths")
    parser.add_argument("--title", type=str, default="Mercury graph growth")
    parser.add_argument("--template-title", type=str, default="Mercury sensory template lengths")
    parser.add_argument("--x-scale", type=str, choices=("linear", "log"), default="linear")
    args = parser.parse_args()

    args.plot_dir.mkdir(parents=True, exist_ok=True)
    levels = _parse_levels(args.levels)

    rows = load_trajectory_rows(args.study_root, args.protocol_id, args.batch_size, levels)
    if rows.empty:
        raise SystemExit(f"No Mercury-phase state bundles found under {args.study_root}.")
    summary = summarise_trajectories(rows)

    template_rows = load_template_rows(args.study_root, args.batch_size, levels)
    if template_rows.empty:
        raise SystemExit(f"No Mercury latent graphs found under {args.study_root}.")
    template_summary = summarise_templates(template_rows)

    saved: list[Path] = []
    for frame, name in (
        (summary, f"{args.output_stem}_summary.csv"),
        (template_rows, f"{args.template_output_stem}_rows.csv"),
        (template_summary, f"{args.template_output_stem}_summary.csv"),
    ):
        path = args.plot_dir / name
        frame.to_csv(path, index=False)
        saved.append(path)

    saved.extend(
        plot_structural_trajectories(
            summary,
            args.plot_dir,
            output_stem=args.output_stem,
            title=args.title,
            x_scale=args.x_scale,
        )
    )
    saved.extend(
        plot_template_lengths(
            template_rows,
            template_summary,
            args.plot_dir,
            output_stem=args.template_output_stem,
            title=args.template_title,
        )
    )

    for path in saved:
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
