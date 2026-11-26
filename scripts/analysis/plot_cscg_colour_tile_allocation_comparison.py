"""Ground-truth vs uniform clone allocation on the colour-tile mazes, every metric (EM-CSCG arms).

PYTHONPATH=.:src .venv/bin/python scripts/analysis/plot_cscg_colour_tile_allocation_comparison.py

Needs learning_curves.csv in both sweeps (plot_cscg_colour_tile_learning_curves.py) and
concentration_gini.csv (plot_cscg_colour_tile_concentration_gini.py). Writes, in the uniform
sweep folder:
  allocation_comparison_level{L}.png      learning curves, mean ± sd over seeds
  allocation_comparison_final.png         final-step GT vs uniform per arm (seeds as dots)
  allocation_comparison_final.csv         the final-step table
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SWEEPS = {"Ground-truth clones": ROOT / "results/diagnostics/cscg_vs_jepa_colour_tiles_3seed_20k",
          "Uniform 10 clones": ROOT / "results/diagnostics/cscg_vs_jepa_colour_tiles_uniform10_3seed_20k"}
OUT = SWEEPS["Uniform 10 clones"]
ALLOC_STYLE = {"Ground-truth clones": (0, (4, 2)), "Uniform 10 clones": "-"}
ARMS = {"Short EM": "#2a78d6", "Long EM": "#eb6834", "EM-JEPA": "#1baf7a"}   # validated slots 1-3
METRICS = [
    ("action_tv_similarity", "Action TV similarity", "↑"),
    ("action_no_self_loop_tv_similarity", "Non-self-loop TV similarity", "↑"),
    ("action_edge_f1_auc", "Action edge F1 AUC", "↑"),
    ("latent_precision", "Latent precision", "↑"),
    ("concentration", "Latent concentration", "↑"),
    ("eval_nll_bps", "Held-out NLL (bits/step)", "↓"),
    ("state_coverage", "State coverage", "↑"),
    ("clone_utilisation", "Clone utilisation", "↑"),
    ("unmatched_mass", "Unmatched mass", "↓"),
    ("gini_weighted", "Transition Gini (occupancy-weighted)", "↓"),
]
LEVELS = {24: "Level 24 — colour level0 (4 colours, Latin square)", 25: "Level 25 — colour level1 (10 colours)"}
INK, MUTED, GRID = "#2b2b29", "#6b6a64", "#e4e3de"


def load() -> pd.DataFrame:
    frames = []
    for alloc, sweep in SWEEPS.items():
        frame = pd.read_csv(sweep / "learning_curves.csv")
        frame = frame[frame.arm.isin(ARMS)].copy()
        frame["allocation"] = alloc
        frames.append(frame)
    data = pd.concat(frames, ignore_index=True)
    extra = pd.read_csv(OUT / "concentration_gini.csv")[
        ["allocation", "arm", "level", "seed", "step", "concentration", "gini_weighted"]]
    merged = data.merge(extra, on=["allocation", "arm", "level", "seed", "step"], how="left", validate="one_to_one")
    if merged[["concentration", "gini_weighted"]].isna().any().any():
        raise ValueError("concentration_gini.csv does not cover every curve point")
    return merged


def style(ax, title, xlabel=None):
    ax.set_title(title, fontsize=9.5, color=INK, loc="left")
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=8.5, color=MUTED)
    ax.grid(True, color=GRID, lw=0.6)
    ax.tick_params(colors=MUTED, labelsize=8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)


def legend_handles():
    Line = matplotlib.lines.Line2D
    return ([Line([], [], color=c, lw=2, label=a) for a, c in ARMS.items()] +
            [Line([], [], color=INK, ls=d, lw=2, label=al) for al, d in ALLOC_STYLE.items()])


def curves(data: pd.DataFrame, level: int) -> None:
    ncol = 5
    nrow = int(np.ceil(len(METRICS) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.3 * ncol, 3.5 * nrow), squeeze=False, constrained_layout=True)
    part = data[data.level == level]
    for ax, (metric, title, sense) in zip(axes.flat, METRICS):
        for alloc, dash in ALLOC_STYLE.items():
            for arm, colour in ARMS.items():
                sub = part[(part.allocation == alloc) & (part.arm == arm)]
                stats = sub.groupby("step")[metric].agg(["mean", "std"]).reset_index()
                ax.fill_between(stats.step, stats["mean"] - stats["std"], stats["mean"] + stats["std"],
                                color=colour, alpha=0.10, lw=0)
                ax.plot(stats.step, stats["mean"], color=colour, ls=dash, lw=1.9)
        style(ax, f"{title} {sense}", "Training observations")
        ax.set_xlim(0, part.step.max())
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v / 1000:g}k"))
    for ax in list(axes.flat)[len(METRICS):]:
        ax.set_visible(False)
    fig.legend(handles=legend_handles(), loc="outside lower center", ncol=5, frameon=False, fontsize=9)
    fig.suptitle(f"{LEVELS[level]}: ground-truth vs uniform (10 per colour) clone allocation — EM-CSCG, "
                 "mean ± sd over 3 seeds, training-matched decoder, 10 × 500 held-out walks",
                 fontsize=11, color=INK)
    path = OUT / f"allocation_comparison_level{level}.png"
    fig.savefig(path, dpi=150, facecolor="#fcfcfb")
    plt.close(fig)
    print(f"Wrote {path}")


def final(data: pd.DataFrame) -> None:
    last = data[data.step == data.step.max()]
    last[["level", "arm", "allocation", "seed"] + [m for m, *_ in METRICS]].to_csv(
        OUT / "allocation_comparison_final.csv", index=False)
    levels = sorted(last.level.unique())
    fig, axes = plt.subplots(len(levels), len(METRICS), figsize=(2.3 * len(METRICS), 3.3 * len(levels)),
                             squeeze=False, constrained_layout=True)
    offsets = {"Ground-truth clones": -0.17, "Uniform 10 clones": 0.17}
    markers = {"Ground-truth clones": "o", "Uniform 10 clones": "s"}
    for r, level in enumerate(levels):
        for c, (metric, title, sense) in enumerate(METRICS):
            ax = axes[r, c]
            for x, (arm, colour) in enumerate(ARMS.items()):
                means = {}
                for alloc in ALLOC_STYLE:
                    values = last[(last.level == level) & (last.arm == arm) & (last.allocation == alloc)][metric]
                    xs = np.full(len(values), x + offsets[alloc])
                    filled = alloc == "Uniform 10 clones"
                    ax.scatter(xs, values, s=14, marker=markers[alloc], zorder=3, lw=1,
                               facecolor=colour if filled else "white", edgecolor=colour, alpha=0.55)
                    means[alloc] = values.mean()
                    ax.scatter([x + offsets[alloc]], [means[alloc]], s=60, marker=markers[alloc], zorder=4, lw=1.6,
                               facecolor=colour if filled else "white", edgecolor=colour)
                ax.plot([x + offsets[a] for a in ALLOC_STYLE], [means[a] for a in ALLOC_STYLE],
                        color=colour, lw=1.2, zorder=2)
            ax.set_xticks(range(len(ARMS)), list(ARMS), rotation=35, ha="right", fontsize=7.5)
            ax.set_xlim(-0.6, len(ARMS) - 0.4)
            short = {"Transition Gini (occupancy-weighted)": "Transition Gini*",
                     "Non-self-loop TV similarity": "Non-self-loop TV sim.",
                     "Held-out NLL (bits/step)": "Held-out NLL (bits)"}.get(title, title)
            style(ax, f"{short} {sense}" if r == 0 else "")
            ax.title.set_fontsize(8)
            ax.title.set_wrap(True)
            if c == 0:
                ax.set_ylabel(LEVELS[level].split(" — ")[0], fontsize=9.5, color=INK)
    Line = matplotlib.lines.Line2D
    handles = [Line([], [], ls="", marker="o", markersize=7, markerfacecolor="white", markeredgecolor=INK,
                    label="Ground-truth clones (open circle)"),
               Line([], [], ls="", marker="s", markersize=7, markerfacecolor=INK, markeredgecolor=INK,
                    label="Uniform 10 clones (filled square)"),
               Line([], [], ls="", marker="o", markersize=4, markerfacecolor=MUTED, markeredgecolor=MUTED,
                    alpha=0.6, label="Small marks: individual seeds")]
    fig.legend(handles=handles, loc="outside lower center", ncol=3, frameon=False, fontsize=9)
    fig.suptitle("Final step (20k): ground-truth vs uniform clone allocation for every metric — EM-CSCG, "
                 "large marks = mean over 3 seeds; *Gini weighted by decoder occupancy (true transitions: 0)", fontsize=11, color=INK)
    path = OUT / "allocation_comparison_final.png"
    fig.savefig(path, dpi=150, facecolor="#fcfcfb")
    plt.close(fig)
    print(f"Wrote {path}")


def main():
    data = load()
    for level in sorted(data.level.unique()):
        curves(data, level)
    final(data)


if __name__ == "__main__":
    main()
