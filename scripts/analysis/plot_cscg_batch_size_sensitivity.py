from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.stats import t

from scripts.analysis.plot_config_metric_trajectories import (
    DEFAULT_LEVEL_ORDER,
    LEVEL_LABELS,
    load_rows,
)
from scripts.analysis.study_selection import BATCH_SWEEP_ROOT, PROTOCOL_ID


METRICS = (
    ("latent_precision", "Latent purity", True),
    ("action_tv_similarity", "Action-conditioned TV similarity", True),
    ("action_no_self_loop_tv_similarity", "Action-conditioned no-loop TV similarity", True),
    ("eval_nll_bps", "Evaluation NLL (bits/symbol)", False),
)
METRIC_COLORS = {
    "latent_precision": "#0072B2",
    "action_tv_similarity": "#009E73",
    "action_no_self_loop_tv_similarity": "#E69F00",
    "eval_nll_bps": "#CC79A7",
}
LEVEL_MARKERS = {13: "o", 21: "s", 20: "^", 22: "D"}


def parse_int_list(raw: str) -> list[int]:
    return list(dict.fromkeys(int(item.strip()) for item in raw.split(",") if item.strip()))


def final_checkpoint_rows(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return rows
    required = {metric for metric, _, _ in METRICS}
    selected = rows[rows["metric"].isin(required)].copy()
    selected = selected.sort_values(["run_id", "metric", "step"])
    return selected.drop_duplicates(["run_id", "metric"], keep="last").reset_index(drop=True)


def aggregate_final_rows(rows: pd.DataFrame, confidence: float = 0.95) -> pd.DataFrame:
    grouped = (
        rows.groupby(["metric", "level", "level_label", "batch_size"], dropna=False)["value"]
        .agg(["count", "mean", "std"])
        .reset_index()
        .rename(columns={"count": "n", "std": "sd"})
    )
    grouped["sd"] = grouped["sd"].fillna(0.0)
    grouped["sem"] = grouped["sd"] / np.sqrt(grouped["n"].clip(lower=1))
    alpha = 1.0 - float(confidence)
    critical = grouped["n"].map(
        lambda n: float(t.ppf(1.0 - alpha / 2.0, int(n) - 1)) if int(n) > 1 else 0.0
    )
    grouped["ci"] = critical * grouped["sem"]
    grouped["ci_low"] = grouped["mean"] - grouped["ci"]
    grouped["ci_high"] = grouped["mean"] + grouped["ci"]
    grouped["confidence"] = float(confidence)
    return grouped


def adjacent_gain_rows(rows: pd.DataFrame, batch_sizes: list[int], confidence: float = 0.95) -> pd.DataFrame:
    if len(batch_sizes) < 2:
        return pd.DataFrame()
    wide = rows.pivot_table(
        index=["metric", "level", "level_label", "seed"],
        columns="batch_size",
        values="value",
        aggfunc="first",
    )
    records: list[dict[str, float | int | str]] = []
    higher_is_better = {metric: direction for metric, _, direction in METRICS}
    alpha = 1.0 - float(confidence)
    for left, right in zip(batch_sizes[:-1], batch_sizes[1:]):
        if left not in wide.columns or right not in wide.columns:
            continue
        paired = wide[[left, right]].dropna()
        for (metric, level, level_label), group in paired.groupby(level=["metric", "level", "level_label"]):
            delta = group[right] - group[left]
            improvement = delta if higher_is_better[str(metric)] else -delta
            n = int(improvement.count())
            mean = float(improvement.mean())
            sd = float(improvement.std(ddof=1)) if n > 1 else 0.0
            sem = sd / np.sqrt(max(n, 1))
            ci = float(t.ppf(1.0 - alpha / 2.0, n - 1) * sem) if n > 1 else 0.0
            records.append(
                {
                    "metric": str(metric),
                    "level": int(level),
                    "level_label": str(level_label),
                    "from_batch": int(left),
                    "to_batch": int(right),
                    "n": n,
                    "mean_improvement": mean,
                    "ci": ci,
                    "ci_low": mean - ci,
                    "ci_high": mean + ci,
                    "confidence": float(confidence),
                }
            )
    return pd.DataFrame(records)


def _mean_environment_gain(summary: pd.DataFrame, metric: str, left: int, right: int, higher_is_better: bool) -> float:
    panel = summary[summary["metric"] == metric].pivot(index="level", columns="batch_size", values="mean")
    if left not in panel.columns or right not in panel.columns:
        return float("nan")
    delta = panel[right] - panel[left]
    return float(delta.mean() if higher_is_better else -delta.mean())


def plot_sensitivity(
    summary: pd.DataFrame,
    *,
    batch_sizes: list[int],
    levels: list[int],
    output_dir: Path,
) -> list[Path]:
    plt.style.use("default")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.0,
            "axes.edgecolor": "#374151",
            "axes.labelcolor": "#111827",
            "xtick.color": "#111827",
            "ytick.color": "#111827",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    x = np.arange(len(batch_sizes), dtype=float)
    figure, axis = plt.subplots(figsize=(6.8, 4.5), dpi=300)
    nll_axis = axis.twinx()

    for metric, _, _ in METRICS:
        target_axis = nll_axis if metric == "eval_nll_bps" else axis
        panel = summary[summary["metric"] == metric]
        for level in levels:
            line = panel[panel["level"] == level].set_index("batch_size").reindex(batch_sizes)
            if line["mean"].isna().all():
                continue
            target_axis.plot(
                x,
                line["mean"].to_numpy(dtype=float),
                color=METRIC_COLORS[metric],
                linestyle="-",
                marker=LEVEL_MARKERS.get(level, "o"),
                markersize=5.0,
                markeredgecolor="white",
                markeredgewidth=0.65,
                linewidth=1.65,
                dash_capstyle="round",
                zorder=3,
            )

    axis.set_xticks(x, [str(batch_size) for batch_size in batch_sizes])
    axis.set_xlim(-0.14, len(batch_sizes) - 0.86)
    axis.set_ylim(0.0, 1.02)
    axis.set_yticks(np.linspace(0.0, 1.0, 6))
    # NLL is non-negative.  Use a zero baseline so low-NLL environments are
    # not clipped as batch size increases.
    nll_axis.set_ylim(0.0, 2.5)
    nll_axis.set_yticks(np.arange(0.0, 2.51, 0.5))
    axis.set_xlabel("CSCG batch size, B", fontsize=10.5, labelpad=5)
    axis.set_ylabel("Structural quality", fontsize=10.5, labelpad=6)
    nll_axis.set_ylabel("Evaluation NLL (bits/symbol)", fontsize=10.5, labelpad=7)
    axis.grid(axis="y", color="#D1D5DB", linewidth=0.55, alpha=0.65)
    axis.axvline(x[1], color="#6B7280", linewidth=0.8, alpha=0.38, zorder=1)
    axis.text(
        x[1] + 0.035,
        0.72,
        "selected: B=40",
        ha="left",
        va="center",
        fontsize=7.5,
        color="#4B5563",
        zorder=5,
    )
    for current_axis in (axis, nll_axis):
        current_axis.spines["top"].set_visible(False)
        current_axis.tick_params(labelsize=9.5)
    axis.spines["right"].set_visible(False)
    nll_axis.spines["left"].set_visible(False)
    nll_axis.spines["right"].set_color("#374151")

    metric_handles = [
        Line2D(
            [0],
            [0],
            color=METRIC_COLORS[metric],
            linestyle="-",
            linewidth=2.0,
            label=label,
        )
        for metric, label in (
            ("latent_precision", r"Latent purity, $P_Z$"),
            ("action_tv_similarity", r"TV similarity, $S_{\mathrm{TV}}^{A}$"),
            (
                "action_no_self_loop_tv_similarity",
                r"No-loop TV, $S_{\mathrm{TV}}^{A,\neg\mathrm{self}}$",
            ),
            ("eval_nll_bps", "Evaluation NLL ↓"),
        )
    ]
    environment_handles = [
        Line2D(
            [0],
            [0],
            color="#64748B",
            linestyle="none",
            marker=LEVEL_MARKERS.get(level, "o"),
            markersize=5.5,
            markerfacecolor="white",
            markeredgewidth=1.15,
            label=LEVEL_LABELS.get(level, f"Level {level}"),
        )
        for level in levels
    ]
    metric_legend = figure.legend(
        handles=metric_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=4,
        frameon=False,
        fontsize=8.2,
        handlelength=1.8,
        handletextpad=0.45,
        columnspacing=0.9,
        borderaxespad=0.0,
    )
    figure.add_artist(metric_legend)
    figure.legend(
        handles=environment_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.91),
        ncol=4,
        frameon=False,
        fontsize=8.2,
        handlelength=1.0,
        handletextpad=0.3,
        columnspacing=0.9,
        borderaxespad=0.0,
    )
    figure.subplots_adjust(left=0.105, right=0.895, bottom=0.13, top=0.84)

    output_dir.mkdir(parents=True, exist_ok=True)
    png = output_dir / "cscg_batch_size_sensitivity_dual_axis.png"
    pdf = output_dir / "cscg_batch_size_sensitivity_dual_axis.pdf"
    figure.savefig(png, bbox_inches="tight")
    figure.savefig(pdf, bbox_inches="tight")
    plt.close(figure)
    return [png, pdf]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot final-checkpoint CSCG batch-size sensitivity.")
    parser.add_argument("--study-root", type=Path, default=BATCH_SWEEP_ROOT)
    parser.add_argument("--protocol-id", type=str, default=PROTOCOL_ID)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-sizes", type=str, default="10,40,100")
    parser.add_argument("--levels", type=str, default="13,20,21,22")
    parser.add_argument("--confidence", type=float, default=0.95)
    return parser


def main() -> None:
    args = build_parser().parse_args()

    batch_sizes = parse_int_list(args.batch_sizes)
    requested_levels = parse_int_list(args.levels)
    level_set = set(requested_levels)
    frames = [load_rows(args.study_root, args.protocol_id, batch_size, level_set) for batch_size in batch_sizes]
    rows = pd.concat([frame for frame in frames if not frame.empty], ignore_index=True) if any(
        not frame.empty for frame in frames
    ) else pd.DataFrame()
    if rows.empty:
        raise SystemExit(f"No evaluation records found under {args.study_root} for protocol {args.protocol_id}.")

    final = final_checkpoint_rows(rows)
    summary = aggregate_final_rows(final, confidence=float(args.confidence))
    gains = adjacent_gain_rows(final, batch_sizes, confidence=float(args.confidence))
    levels = [level for level in DEFAULT_LEVEL_ORDER if level in set(requested_levels)]
    levels.extend(level for level in requested_levels if level not in levels)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    final_path = args.output_dir / "cscg_batch_size_final_by_seed.csv"
    summary_path = args.output_dir / "cscg_batch_size_final_summary.csv"
    gains_path = args.output_dir / "cscg_batch_size_adjacent_gains.csv"
    final.to_csv(final_path, index=False)
    summary.to_csv(summary_path, index=False)
    gains.to_csv(gains_path, index=False)
    outputs = plot_sensitivity(summary, batch_sizes=batch_sizes, levels=levels, output_dir=args.output_dir)

    for path in [final_path, summary_path, gains_path, *outputs]:
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
