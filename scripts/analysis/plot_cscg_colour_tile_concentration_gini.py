"""Latent concentration and transition Gini impurity for the colour-tile EM-CSCG runs.

PYTHONPATH=.:src .venv/bin/python scripts/analysis/plot_cscg_colour_tile_concentration_gini.py

Concentration = canonical latent recall (``latent_recall``) of the training-matched decoder.
Gini impurity of a transition row = 1 - sum_j T[a, i, j]^2, the quantity the gradCSCG
regulariser penalises (torch_gradient.py). Reported two ways:
  occupancy-weighted  rows weighted by how often the decoder visits clone i on the held-out
                      walks (unused clones carry no weight) -- the headline number;
  all rows            plain mean over every (action, clone) row, as in the regulariser.
Reference: the same occupancy-weighted Gini of the empirical ground-truth transitions.
Student models only (no EMA teacher); EM-optimised arms only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SWEEPS = {"Ground-truth clones": "results/diagnostics/cscg_vs_jepa_colour_tiles_3seed_20k",
          "Uniform 10 clones": "results/diagnostics/cscg_vs_jepa_colour_tiles_uniform10_3seed_20k"}
ALLOC_STYLE = {"Ground-truth clones": (0, (4, 2)), "Uniform 10 clones": "-"}
# (folder, method, label, colour): validated categorical slots 1-3.
ARMS = [("em_baselines", "short", "Short EM", "#2a78d6"),
        ("em_baselines", "long", "Long EM", "#eb6834"),
        ("jepa_mix1", "predictive", "EM-JEPA", "#1baf7a")]
EVAL = "evaluation_v5_training_matched_walks10x500"
LEVELS = {24: "Level 24 — 4 colours", 25: "Level 25 — 10 colours"}
INK, MUTED, GRID = "#2b2b29", "#6b6a64", "#e4e3de"


def row_gini(T):
    return 1.0 - (np.asarray(T) ** 2).sum(axis=-1)


def collect() -> pd.DataFrame:
    rows = []
    for alloc, sweep in SWEEPS.items():
        for folder, method, label, _ in ARMS:
            for case in sorted((ROOT / sweep / folder).glob(f"level=*/seed=*/{method}")):
                level = int(case.parent.parent.name.split("=")[1])
                seed = int(case.parent.name.split("=")[1])
                with np.load(case / "snapshots.npz") as saved:
                    steps, tensors = saved["steps"], saved["T"]
                for step, T in zip(steps, tensors):
                    result = json.loads((case / EVAL / "training_matched" / f"step={int(step):05d}.json").read_text())
                    occupancy = np.asarray(result["alignment"]["cooccurrence"], dtype=float).sum(axis=1)
                    gini = row_gini(T)                                    # (actions, clones)
                    weights = np.broadcast_to(occupancy, gini.shape)
                    truth = result["action_conditioned"]
                    true_gini = row_gini(truth["P_true_full"])
                    true_weights = np.asarray(truth["evaluation_occupancy_weights"], dtype=float)
                    rows.append(dict(
                        allocation=alloc, arm=label, level=level, seed=seed, step=int(step),
                        n_states=int(T.shape[1]),
                        concentration=result["state_representation"]["node_recall"],
                        gini_weighted=float((gini * weights).sum() / weights.sum()),
                        gini_all_rows=float(gini.mean()),
                        gini_true=float((true_gini * true_weights).sum() / true_weights.sum())))
    return pd.DataFrame(rows)


def plot(data: pd.DataFrame, path: Path) -> None:
    panels = [("concentration", "Latent concentration ↑"),
              ("gini_weighted", "Transition Gini, occupancy-weighted"),
              ("gini_all_rows", "Transition Gini, all rows")]
    levels = sorted(data.level.unique())
    fig, axes = plt.subplots(len(levels), len(panels), figsize=(5.2 * len(panels), 3.8 * len(levels)),
                             squeeze=False, constrained_layout=True)
    for r, level in enumerate(levels):
        for c, (metric, title) in enumerate(panels):
            ax = axes[r, c]
            for alloc, dash in ALLOC_STYLE.items():
                for _, _, label, colour in ARMS:
                    part = data[(data.level == level) & (data.arm == label) & (data.allocation == alloc)]
                    stats = part.groupby("step")[metric].agg(["mean", "std"]).reset_index()
                    ax.fill_between(stats.step, stats["mean"] - stats["std"], stats["mean"] + stats["std"],
                                    color=colour, alpha=0.12, lw=0)
                    ax.plot(stats.step, stats["mean"], color=colour, ls=dash, lw=2)
            if metric.startswith("gini"):
                ref = data[data.level == level].groupby("step").gini_true.mean()
                ax.axhline(ref.mean(), color=MUTED, lw=1.2, ls=(0, (1, 1.5)))
                ax.annotate(f"true transitions {ref.mean():.2f}", xy=(data.step.max(), ref.mean()),
                            xytext=(-4, 4), textcoords="offset points", ha="right", fontsize=7.5, color=MUTED)
            ax.set_title(title, fontsize=9.5, color=INK, loc="left")
            ax.set_xlabel("Training observations", fontsize=8.5, color=MUTED)
            if c == 0:
                ax.set_ylabel(LEVELS.get(level, f"Level {level}"), fontsize=9, color=INK)
            ax.grid(True, color=GRID, lw=0.6)
            ax.tick_params(colors=MUTED, labelsize=8)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            for side in ("left", "bottom"):
                ax.spines[side].set_color(GRID)
            ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v / 1000:g}k"))
            ax.set_xlim(0, data.step.max())
    Line = matplotlib.lines.Line2D
    handles = [Line([], [], color=colour, lw=2, label=label) for _, _, label, colour in ARMS]
    handles += [Line([], [], color=INK, ls=dash, lw=2, label=alloc) for alloc, dash in ALLOC_STYLE.items()]
    handles += [Line([], [], color=MUTED, ls=(0, (1, 1.5)), lw=1.2, label="True-transition Gini")]
    fig.legend(handles=handles, loc="outside lower center", ncol=len(handles), frameon=False, fontsize=9)
    fig.suptitle("Colour-tile EM-CSCG: latent concentration and transition Gini impurity "
                 "(mean ± sd over 3 seeds; training-matched decoder, 10 × 500 held-out walks)",
                 fontsize=11, color=INK)
    fig.savefig(path, dpi=150, facecolor="#fcfcfb")
    plt.close(fig)
    print(f"Wrote {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results/diagnostics/cscg_vs_jepa_colour_tiles_uniform10_3seed_20k")
    args = parser.parse_args()
    data = collect()
    data.to_csv(args.output / "concentration_gini.csv", index=False)
    plot(data, args.output / "concentration_gini.png")
    final = data[data.step == data.step.max()]
    summary = final.groupby(["level", "allocation", "arm"])[
        ["concentration", "gini_weighted", "gini_all_rows", "gini_true"]].agg(["mean", "std"]).round(3)
    pd.set_option("display.width", 200)
    print(summary)


if __name__ == "__main__":
    main()
