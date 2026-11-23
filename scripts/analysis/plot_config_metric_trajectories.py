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
import numpy as np
import pandas as pd
from mercury_runs.analysis.plots import publication_style as pub


DEFAULT_STUDY_ROOT = Path("results/current/studies/default_matrix_smoke_30_seed_onehot_clean_more_mazes")
DEFAULT_PLOT_DIR = Path("results/current/notebook_plots/default_matrix_smoke_30_seed_onehot_clean_more_mazes/metric_trajectories")
DEFAULT_PROTOCOL_ID = "canonical-v4-8922ca1c45c19cc7"
DEFAULT_EDGE_THRESHOLDS = (
    0.0,
    1e-6,
    1e-5,
    1e-4,
    1e-3,
    2e-3,
    5e-3,
    1e-2,
    2e-2,
    5e-2,
    0.1,
    0.15,
    0.2,
    0.25,
    0.3,
    0.4,
    0.5,
)

LEVEL_LABELS = {13: "3x3", 20: "1x5", 21: "1x4", 22: "3x4"}
DEFAULT_LEVEL_ORDER = (13, 21, 20, 22)

SERIES_ORDER = (
    "mercury_cscg_hybrid",
    "tmgwr_cscg_hybrid",
    *pub.CSCG_VARIANT_ORDER,
)
SERIES_LABELS = {name: pub.method_label(name) for name in SERIES_ORDER}
SERIES_COLORS = {name: pub.method_color(name) for name in SERIES_ORDER}
SERIES_MARKERS = {
    "mercury_cscg_hybrid": "o",
    "tmgwr_cscg_hybrid": "s",
    "vanilla_cscg": "o",
    "oracle_capacity": "^",
    "oracle_capacity_structural": "D",
    "oracle_capacity_first_order": "v",
}
BATCH_COLORS = {
    10: "#2563EB",
    40: "#EA580C",
    100: "#16A34A",
}
BATCH_LINESTYLES = ("-", "--", "-.", ":")

# Row labels are wrapped so they stay shorter than a panel's height; an
# unwrapped label runs past its own row and collides with its neighbours.
METRICS = (
    ("latent_precision", pub.metric_label("latent_precision").replace(" ", "\n"), "score"),
    ("latent_recall", pub.metric_label("latent_recall").replace(" ", "\n"), "score"),
    ("action_edge_f1_auc", "Edge F1\nAUC", "score"),
    ("action_no_self_loop_edge_f1_auc", "No-loop\nedge F1 AUC", "score"),
    ("action_tv_similarity", "Action\nTV sim.", "score"),
    ("action_no_self_loop_tv_similarity", "Action no-loop\nTV sim.", "score"),
    ("eval_nll_bps", "Eval NLL", "nll"),
)


@dataclass(frozen=True)
class RunEntry:
    run_id: str
    method: str
    series: str
    level: int
    seed: int
    rand_prob: float
    batch_size: int
    handoff_step: int
    run_root: Path


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _series_from_config(config: dict[str, Any]) -> str | None:
    method = str(config.get("method", ""))
    if method == "cscg":
        preset = config.get("cscg_structural_preset") or config.get("cscg_clone_preset")
        return str(preset) if preset in SERIES_LABELS else None
    if method in {"mercury_cscg_hybrid", "tmgwr_cscg_hybrid"}:
        return method
    return None


def _load_manifest(study_root: Path, batch_size: int, levels: set[int] | None) -> list[RunEntry]:
    manifest_path = study_root / "study_manifest.jsonl"
    entries: list[RunEntry] = []
    actual_candidates: dict[tuple[Any, ...], list[Path]] = {}
    for config_path in study_root.glob("*/run_config.json"):
        try:
            persisted = _read_json(config_path)
            config = dict(persisted.get("cli_args") or persisted.get("run_payload") or {})
            series = _series_from_config(config)
            if series is None:
                continue
            key = (
                str(config.get("method")),
                int(config["level"]),
                int(config["seed"]),
                float(config.get("rand_prob", 0.0)),
                int(config.get("cscg_batch_size", batch_size)),
                series,
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue
        actual_candidates.setdefault(key, []).append(config_path.parent)

    with manifest_path.open(encoding="utf-8") as handle:
        for raw in handle:
            if not raw.strip():
                continue
            config = dict(json.loads(raw)["config"])
            if int(config.get("cscg_batch_size", batch_size)) != int(batch_size):
                continue
            level = int(config["level"])
            if levels is not None and level not in levels:
                continue
            series = _series_from_config(config)
            if series is None:
                continue
            run_id = str(config["requested_run_id"])
            run_root = study_root / run_id
            if not (run_root / "run_status.json").exists():
                key = (
                    str(config["method"]),
                    level,
                    int(config["seed"]),
                    float(config.get("rand_prob", 0.0)),
                    int(config.get("cscg_batch_size", batch_size)),
                    series,
                )
                candidates = actual_candidates.get(key, [])
                if len(candidates) != 1:
                    continue
                run_root = candidates[0]
                run_id = run_root.name
            entries.append(
                RunEntry(
                    run_id=run_id,
                    method=str(config["method"]),
                    series=series,
                    level=level,
                    seed=int(config["seed"]),
                    rand_prob=float(config.get("rand_prob", 0.0)),
                    batch_size=int(config.get("cscg_batch_size", batch_size)),
                    handoff_step=(int(config.get("num_steps", 0)) if str(config.get("method", "")) in {"mercury_cscg_hybrid", "tmgwr_cscg_hybrid"} else 0),
                    run_root=run_root,
                )
            )
    return entries


def _snapshot_dir(entry: RunEntry, protocol_id: str) -> Path:
    return entry.run_root / "evaluations" / protocol_id / "snapshots" / entry.method


def _snapshot_paths(entry: RunEntry, protocol_id: str) -> list[Path]:
    directory = _snapshot_dir(entry, protocol_id)
    if not directory.exists():
        return []
    paths = []
    for path in directory.glob("step=*.json"):
        try:
            int(path.stem.split("=", 1)[1])
        except ValueError:
            continue
        paths.append(path)
    return sorted(paths, key=lambda path: int(path.stem.split("=", 1)[1]))


def _action_edge_f1(
    *,
    p_true: np.ndarray,
    p_hat: np.ndarray,
    valid_rows: np.ndarray,
    threshold: float,
    include_self_loops: bool,
) -> float:
    n_true = int(valid_rows.shape[1])
    gt_edges: set[tuple[int, int, int]] = set()
    hat_edges: set[tuple[int, int, int]] = set()
    for action_id, source_state in np.argwhere(valid_rows).tolist():
        for target_state in range(p_true.shape[2]):
            if not include_self_loops and int(target_state) == int(source_state):
                continue
            if target_state < n_true and float(p_true[action_id, source_state, target_state]) > float(threshold):
                gt_edges.add((int(action_id), int(source_state), int(target_state)))
            if float(p_hat[action_id, source_state, target_state]) > float(threshold):
                hat_edges.add((int(action_id), int(source_state), int(target_state)))
    true_positive = len(gt_edges & hat_edges)
    precision = true_positive / len(hat_edges) if hat_edges else 0.0
    recall = true_positive / len(gt_edges) if gt_edges else 0.0
    return (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0.0 else 0.0


def _action_edge_f1_auc(record: dict[str, Any], *, include_self_loops: bool) -> float | None:
    action = record.get("action_conditioned", {})
    if not {"P_true_full", "P_hat_full", "valid_rows"}.issubset(action):
        return None
    p_true = np.asarray(action["P_true_full"], dtype=np.float64)
    p_hat = np.asarray(action["P_hat_full"], dtype=np.float64)
    valid_rows = np.asarray(action["valid_rows"], dtype=bool)
    if p_true.ndim != 3 or p_hat.ndim != 3 or valid_rows.ndim != 2:
        return None
    thresholds = np.asarray(DEFAULT_EDGE_THRESHOLDS, dtype=np.float64)
    f1_values = np.asarray(
        [
            _action_edge_f1(
                p_true=p_true,
                p_hat=p_hat,
                valid_rows=valid_rows,
                threshold=float(threshold),
                include_self_loops=include_self_loops,
            )
            for threshold in thresholds
        ],
        dtype=np.float64,
    )
    span = float(thresholds[-1] - thresholds[0])
    if span <= 0.0:
        return float(np.mean(f1_values))
    return float(np.trapezoid(f1_values, thresholds) / span)


def _metrics_from_record(record: dict[str, Any]) -> dict[str, float]:
    state = record.get("state_representation", {})
    action = record.get("action_conditioned", {})
    model_fit = record.get("model_fit") or {}

    metrics = {
        "latent_precision": state.get("latent_precision"),
        "latent_recall": state.get("latent_recall", state.get("node_recall")),
        "action_edge_f1_auc": _action_edge_f1_auc(record, include_self_loops=True),
        "action_no_self_loop_edge_f1_auc": _action_edge_f1_auc(record, include_self_loops=False),
        "action_tv_similarity": action.get("tv_similarity_occupancy"),
        "action_no_self_loop_tv_similarity": action.get("tv_non_self_loop_similarity_occupancy"),
        "eval_nll_bps": model_fit.get("negative_log2_likelihood_mean", model_fit.get("bits_per_symbol")),
    }
    return {key: float(value) for key, value in metrics.items() if value is not None}


def load_rows(study_root: Path, protocol_id: str, batch_size: int, levels: set[int] | None) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for entry in _load_manifest(study_root, batch_size=batch_size, levels=levels):
        for path in _snapshot_paths(entry, protocol_id):
            record = _read_json(path)
            step = int(record.get("snapshot_step", path.stem.split("=", 1)[1]))
            observed_samples = int(record.get("observed_samples", step))
            phase = "cscg" if entry.method in {"mercury_cscg_hybrid", "tmgwr_cscg_hybrid"} else "native"
            plot_step = observed_samples + int(entry.handoff_step)
            # Hybrid step zero is the CSCG structure immediately after transfer
            # from a source trained for handoff_step samples. Keep zero as its
            # analysis-stage key; plot_step records its real cumulative exposure.
            comparison_step = 0 if step == 0 else plot_step
            metric_values = _metrics_from_record(record)
            if not metric_values:
                continue
            for metric, value in metric_values.items():
                rows.append(
                    {
                        "run_id": entry.run_id,
                        "method": entry.method,
                        "series": entry.series,
                        "series_label": SERIES_LABELS[entry.series],
                        "level": entry.level,
                        "level_label": LEVEL_LABELS.get(entry.level, f"L{entry.level}"),
                        "seed": entry.seed,
                        "rand_prob": entry.rand_prob,
                        "batch_size": entry.batch_size,
                        "phase": phase,
                        "handoff_step": entry.handoff_step,
                        "step": step,
                        "observed_samples": observed_samples,
                        "plot_step": plot_step,
                        "comparison_step": comparison_step,
                        "metric": metric,
                        "value": value,
                    }
                )
    return pd.DataFrame(rows)


def rows_at_comparison_steps(rows: pd.DataFrame, protocol_id: str) -> pd.DataFrame:
    """Select total-sample checkpoints for matched-budget v5 analyses."""
    if not protocol_id.startswith("evaluation-v5-nonself-matched-budget-v1-"):
        return rows
    aligned = rows.copy()
    aligned["step"] = aligned["comparison_step"]
    return aligned


def summarise_rows(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return rows
    grouped = (
        rows.groupby(
            [
                "series",
                "series_label",
                "level",
                "level_label",
                "rand_prob",
                "batch_size",
                "phase",
                "handoff_step",
                "step",
                "observed_samples",
                "plot_step",
                "metric",
            ],
            dropna=False,
        )["value"]
        .agg(["count", "mean", "std"])
        .reset_index()
        .rename(columns={"count": "n", "mean": "mean", "std": "sd"})
    )
    grouped["sd"] = grouped["sd"].fillna(0.0)
    grouped["sem"] = grouped["sd"] / np.sqrt(grouped["n"].clip(lower=1))
    return grouped


def _ordered_present(values: set[str], preferred: tuple[str, ...]) -> list[str]:
    return [value for value in preferred if value in values] + sorted(values.difference(preferred))


def _axis_limits(metric_frame: pd.DataFrame, metric_kind: str) -> tuple[float, float]:
    values = metric_frame["mean"].to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return (0.0, 1.0) if metric_kind == "score" else (0.0, 2.0)
    if metric_kind == "score":
        low = max(0.0, float(values.min()) - 0.035)
        high = min(1.0, float(values.max()) + 0.035)
        if high - low < 0.12:
            midpoint = (high + low) / 2.0
            low = max(0.0, midpoint - 0.06)
            high = min(1.0, midpoint + 0.06)
        return low, high
    value_range = float(np.ptp(values))
    low = max(0.0, float(values.min()) - 0.08 * float(value_range if value_range > 0 else values.max()))
    high = float(values.max()) + 0.10 * float(value_range if value_range > 0 else max(values.max(), 1.0))
    return low, high


def plot_compact_trajectories(summary: pd.DataFrame, plot_dir: Path, *, output_stem: str, title: str) -> list[Path]:
    if summary.empty:
        raise ValueError("No trajectory rows available to plot.")

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

    levels = [level for level in DEFAULT_LEVEL_ORDER if level in set(summary["level"])]
    levels.extend(sorted(set(summary["level"]).difference(levels)))
    series_order = _ordered_present(set(summary["series"]), SERIES_ORDER)
    batch_sizes = sorted(int(value) for value in summary["batch_size"].unique())
    batch_linestyles = {
        batch_size: BATCH_LINESTYLES[index % len(BATCH_LINESTYLES)]
        for index, batch_size in enumerate(batch_sizes)
    }
    batch_colors = {
        batch_size: BATCH_COLORS.get(batch_size, plt.get_cmap("tab10")(index % 10))
        for index, batch_size in enumerate(batch_sizes)
    }
    batch_comparison = len(batch_sizes) > 1
    rand_values = sorted(summary["rand_prob"].unique())
    rand_label = f"rand={rand_values[0]:g}" if len(rand_values) == 1 else "all rand"
    handoff_steps = sorted(
        int(value)
        for value in summary["handoff_step"].dropna().unique()
        if int(value) > 0
    )

    n_rows = len(METRICS)
    n_cols = len(levels)
    figure, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(2.75 * n_cols + 1.9, 1.22 * n_rows + 1.55),
        sharex=True,
        dpi=300,
        gridspec_kw={"hspace": 0.18, "wspace": 0.12},
    )
    axes = np.asarray(axes).reshape(n_rows, n_cols)

    for row_index, (metric, metric_label, metric_kind) in enumerate(METRICS):
        metric_frame = summary[summary["metric"] == metric]
        y_limits = _axis_limits(metric_frame, metric_kind)
        for col_index, level in enumerate(levels):
            axis = axes[row_index, col_index]
            panel = metric_frame[metric_frame["level"] == level]
            for series in series_order:
                for batch_size in batch_sizes:
                    line = panel[
                        (panel["series"] == series) & (panel["batch_size"] == batch_size)
                    ].sort_values("plot_step")
                    if line.empty:
                        continue
                    x = line["plot_step"].to_numpy(dtype=float)
                    y = line["mean"].to_numpy(dtype=float)
                    sem = line["sem"].to_numpy(dtype=float)
                    color = (
                        batch_colors[batch_size]
                        if batch_comparison and len(series_order) == 1
                        else SERIES_COLORS.get(series, "#555555")
                    )
                    marker = SERIES_MARKERS.get(series, "o")
                    if batch_comparison:
                        series_label = SERIES_LABELS.get(series, series)
                        label = (
                            f"Batch {batch_size}"
                            if len(series_order) == 1
                            else f"{series_label}, batch {batch_size}"
                        )
                    else:
                        label = SERIES_LABELS.get(series, series)
                    axis.plot(
                        x,
                        y,
                        color=color,
                        linestyle=batch_linestyles[batch_size],
                        marker=marker,
                        markersize=2.8,
                        linewidth=1.25,
                        alpha=0.96,
                        label=label if row_index == 0 and col_index == 0 else None,
                    )
                    if len(x) <= 8:
                        axis.fill_between(x, y - sem, y + sem, color=color, alpha=0.075, linewidth=0)
            for handoff_step in handoff_steps:
                axis.axvline(float(handoff_step), color="#94A3B8", linewidth=0.65, linestyle=(0, (2.0, 2.2)), alpha=0.55, zorder=0)
            axis.set_xscale("log")
            axis.set_ylim(*y_limits)
            axis.grid(axis="y", color="#CBD5E1", linewidth=0.55, alpha=0.55)
            axis.grid(axis="x", color="#E5E7EB", linewidth=0.45, alpha=0.42)
            axis.tick_params(axis="both", labelsize=6.9, length=2.5, pad=1.5)
            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)
            if col_index == 0:
                axis.set_ylabel(metric_label, fontsize=7.7, fontweight="bold", ha="center", va="center")
                # Pin every row label to the same x so they form a clean column
                # instead of drifting with each row's tick-label width.
                axis.yaxis.set_label_coords(-0.235, 0.5)
            else:
                axis.tick_params(axis="y", labelleft=False)
                axis.spines["left"].set_visible(False)
            if row_index == 0:
                axis.set_title(LEVEL_LABELS.get(int(level), f"L{level}"), fontsize=9.2, fontweight="bold", pad=4)
            if row_index < n_rows - 1:
                axis.tick_params(axis="x", labelbottom=False)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.945),
        ncol=3,
        frameon=False,
        fontsize=7.6,
        handlelength=1.6,
        columnspacing=1.0,
        handletextpad=0.35,
    )
    figure.suptitle(f"{title} ({rand_label})", fontsize=11.5, y=0.992)
    figure.supxlabel("Cumulative exposure / snapshot step (hybrids offset by source pre-pass; log scale)", fontsize=8.8, y=0.025)
    figure.subplots_adjust(left=0.092, right=0.995, top=0.885, bottom=0.085)

    output = plot_dir / f"{output_stem}.png"
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)
    return [output]


def _parse_levels(raw: str | None) -> set[int] | None:
    if raw is None or not raw.strip():
        return None
    return {int(item.strip()) for item in raw.split(",") if item.strip()}


def _parse_batch_sizes(raw: str | None, fallback: int) -> list[int]:
    if raw is None or not raw.strip():
        return [int(fallback)]
    values = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("--batch-sizes must contain at least one integer")
    return list(dict.fromkeys(values))


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot compact metric trajectories for every config in a canonical study.")
    parser.add_argument("--study-root", type=Path, default=DEFAULT_STUDY_ROOT)
    parser.add_argument("--plot-dir", type=Path, default=DEFAULT_PLOT_DIR)
    parser.add_argument("--protocol-id", type=str, default=DEFAULT_PROTOCOL_ID)
    parser.add_argument("--batch-size", type=int, default=40)
    parser.add_argument(
        "--batch-sizes",
        type=str,
        default=None,
        help="Optional comma-separated batch sizes to overlay on the same figure.",
    )
    parser.add_argument("--levels", type=str, default=",".join(str(level) for level in DEFAULT_LEVEL_ORDER))
    parser.add_argument("--output-stem", type=str, default="all_config_metric_trajectories")
    parser.add_argument("--title", type=str, default="All-config metric trajectories")
    args = parser.parse_args()

    args.plot_dir.mkdir(parents=True, exist_ok=True)
    batch_sizes = _parse_batch_sizes(args.batch_sizes, args.batch_size)
    frames = [
        load_rows(args.study_root, args.protocol_id, batch_size, _parse_levels(args.levels))
        for batch_size in batch_sizes
    ]
    rows = pd.concat([frame for frame in frames if not frame.empty], ignore_index=True) if any(
        not frame.empty for frame in frames
    ) else pd.DataFrame()
    if rows.empty:
        raise SystemExit(
            f"No evaluation records found under {args.study_root} for protocol {args.protocol_id} "
            f"and batch sizes {batch_sizes}."
        )
    summary = summarise_rows(rows)

    rows_path = args.plot_dir / f"{args.output_stem}_rows.csv"
    summary_path = args.plot_dir / f"{args.output_stem}_summary.csv"
    rows.to_csv(rows_path, index=False)
    summary.to_csv(summary_path, index=False)
    outputs = plot_compact_trajectories(summary, args.plot_dir, output_stem=args.output_stem, title=args.title)

    print(f"Saved {rows_path}")
    print(f"Saved {summary_path}")
    for output in outputs:
        print(f"Saved {output}")


if __name__ == "__main__":
    main()
