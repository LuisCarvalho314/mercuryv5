"""Learning curves for the colour-tile CSCG sweep (scripts/run_cscg_colour_tile_seeds.sh).

PYTHONPATH=.:src .venv/bin/python scripts/analysis/plot_cscg_colour_tile_learning_curves.py \
    --sweep results/diagnostics/cscg_vs_jepa_colour_tiles_3seed_20k

Training-matched decoder, student model (the EMA teacher is left out). Writes
learning_curves.csv, learning_curves_mean_sd.png and learning_curves_per_seed.png.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# (run folder, method, label, colour, line style); colours: validated categorical slots 1-5.
ARMS = [
    ("em_baselines", "short", "Short EM", "#2a78d6", (0, (1, 1.5))),
    ("em_baselines", "long", "Long EM", "#eb6834", (0, (4, 2))),
    ("jepa_mix1", "predictive", "EM-JEPA", "#1baf7a", "-"),
    ("gradcscg_lr0.01_adam10", "long", "gradCSCG (long EM)", "#eda100", (0, (6, 2, 1, 2))),
    ("gradcscg_jepa_mix1_lr0.01_adam10", "predictive", "gradCSCG-JEPA", "#e87ba4", (0, (2, 1))),
]
METRICS = [
    ("action_tv_similarity", "Action TV similarity ↑"),
    ("action_no_self_loop_tv_similarity", "Non-self-loop TV similarity ↑"),
    ("latent_precision", "Latent precision ↑"),
    ("eval_nll_bps", "Held-out NLL (bits/step) ↓"),
    ("state_coverage", "State coverage ↑"),
]
LEVELS = {24: "Level 24 — colour level0 (4 colours, Latin square)",
          25: "Level 25 — colour level1 (10 colours)"}
INK, MUTED, GRID = "#2b2b29", "#6b6a64", "#e4e3de"


def load(sweep: Path) -> pd.DataFrame:
    frames = []
    for folder, method, label, *_ in ARMS:
        path = sweep / folder / "evaluation_v5_training_matched_walks10x500" / "canonical_metrics_all.csv"
        frame = pd.read_csv(path)
        frame = frame[(frame.method == method) & (frame.decoder == "training_matched")].copy()
        frame["arm"] = label
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def style(ax, title, xlabel):
    ax.set_title(title, fontsize=9.5, color=INK, loc="left")
    ax.set_xlabel(xlabel, fontsize=8.5, color=MUTED)
    ax.grid(True, color=GRID, lw=0.6)
    ax.tick_params(colors=MUTED, labelsize=8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v / 1000:g}k"))


def figure(data: pd.DataFrame, per_seed: bool, path: Path) -> None:
    levels = sorted(data.level.unique())
    fig, axes = plt.subplots(len(levels), len(METRICS), figsize=(4.1 * len(METRICS), 3.6 * len(levels)),
                             squeeze=False, constrained_layout=True)
    for row, level in enumerate(levels):
        for col, (metric, title) in enumerate(METRICS):
            ax = axes[row, col]
            ends = []
            for _, _, label, colour, dash in ARMS:
                part = data[(data.level == level) & (data.arm == label)]
                if per_seed:
                    for _, seed in part.groupby("seed"):
                        seed = seed.sort_values("step")
                        ax.plot(seed.step, seed[metric], color=colour, ls=dash, lw=1.3, alpha=0.85)
                    final = part[part.step == part.step.max()][metric].mean()
                else:
                    stats = part.groupby("step")[metric].agg(["mean", "std"]).reset_index()
                    ax.fill_between(stats.step, stats["mean"] - stats["std"], stats["mean"] + stats["std"],
                                    color=colour, alpha=0.14, lw=0)
                    ax.plot(stats.step, stats["mean"], color=colour, ls=dash, lw=2)
                    final = stats["mean"].iloc[-1]
                ends.append((final, label, colour))
            # Direct labels at the line ends, nudged apart so they never collide.
            low, high = ax.get_ylim()
            gap = 0.075 * (high - low)
            placed = []
            for value, label, colour in sorted(ends):
                y = max(value, placed[-1] + gap) if placed else value
                placed.append(y)
                ax.annotate(label, xy=(data.step.max(), value), xytext=(data.step.max() * 1.03, y),
                            fontsize=7, color=INK, va="center", annotation_clip=False,
                            arrowprops=dict(arrowstyle="-", color=colour, lw=0.8))
            ax.set_xlim(0, data.step.max() * 1.0)
            style(ax, title if row == 0 else title, "Training observations")
            if col == 0:
                ax.set_ylabel(LEVELS.get(level, f"Level {level}"), fontsize=9, color=INK)
    handles = [matplotlib.lines.Line2D([], [], color=c, ls=d, lw=2, label=l) for _, _, l, c, d in ARMS]
    fig.legend(handles=handles, loc="outside lower center", ncol=len(ARMS), frameon=False, fontsize=9)
    kind = "each seed (3 seeds)" if per_seed else "mean ± sd over 3 seeds"
    fig.suptitle(f"Colour-tile CSCG learning curves — {kind}; training-matched decoder, 10 × 500 held-out walks",
                 fontsize=11, color=INK)
    fig.savefig(path, dpi=150, facecolor="#fcfcfb")
    plt.close(fig)
    print(f"Wrote {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep", type=Path, default=Path("results/diagnostics/cscg_vs_jepa_colour_tiles_3seed_20k"))
    args = parser.parse_args()
    data = load(args.sweep)
    data.to_csv(args.sweep / "learning_curves.csv", index=False)
    figure(data, per_seed=False, path=args.sweep / "learning_curves_mean_sd.png")
    figure(data, per_seed=True, path=args.sweep / "learning_curves_per_seed.png")


if __name__ == "__main__":
    main()
