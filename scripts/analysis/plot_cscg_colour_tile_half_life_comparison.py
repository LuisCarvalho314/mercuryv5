"""EM-JEPA teacher half-life comparison on the colour-tile mazes, every metric, both clone allocations.

PYTHONPATH=.:src .venv/bin/python scripts/analysis/plot_cscg_colour_tile_half_life_comparison.py --half-lives 40 80
    or, for any EM-JEPA run folders (e.g. teacher contexts):
    ... --runs "Teacher context 40=jepa_mix1" "Teacher context 80=jepa_mix1_tw80" --name teacher_context_comparison
    and for any method, LABEL=FOLDER:METHOD (default method predictive; the teacher is read for predictive only):
    ... --runs "5 updates=em_baselines:long" "20 updates=em_baselines_lu20:long" --name local_updates_long

Half-life H lives in <sweep>/jepa_mix1 (H = 40) or <sweep>/jepa_mix1_hl<H>. Metrics come from
the training-matched evaluation (canonical_metrics_all.csv); concentration = latent recall and
the occupancy-weighted transition Gini 1 - sum_j T[a,i,j]^2 come from the per-step evaluation
records and snapshots.npz (as in plot_cscg_colour_tile_concentration_gini.py). Curves show the
student; the final-step CSV and table also carry the EMA teacher. Writes into the uniform sweep:
  half_life_comparison_level{L}.png, half_life_comparison_final.csv
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
SWEEPS = {"Ground-truth clones": ROOT / "results/diagnostics/cscg_vs_jepa_colour_tiles_3seed_20k",
          "Uniform 10 clones": ROOT / "results/diagnostics/cscg_vs_jepa_colour_tiles_uniform10_3seed_20k"}
OUT = SWEEPS["Uniform 10 clones"]
ALLOC_STYLE = {"Ground-truth clones": (0, (4, 2)), "Uniform 10 clones": "-"}
COLOURS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]   # validated categorical slots, fixed order
EVAL = "evaluation_v5_training_matched_walks10x500"
MODELS = {"student": ("predictive", "training_matched", "T"),
          "teacher": ("predictive", "teacher_training_matched", "teacher_T")}
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


def folder(half_life: int) -> str:
    return "jepa_mix1" if half_life == 40 else f"jepa_mix1_hl{half_life}"


def load(variants, expected_half_life=None) -> pd.DataFrame:
    """variants: [(label, run folder name[:method])]; expected_half_life: {label: half-life} to verify, if given."""
    frames = []
    for alloc, sweep in SWEEPS.items():
        for hl, spec in variants:
            name, _, method = spec.partition(":")
            method = method or "predictive"
            models = MODELS if method == "predictive" else {"student": (method, "training_matched", "T")}
            run = sweep / name
            manifest = json.loads((run / "experiment.json").read_text())
            if expected_half_life and float(manifest["experiment"]["teacher_half_life"]) != float(expected_half_life[hl]):
                raise ValueError(f"{run} has half-life {manifest['experiment']['teacher_half_life']}, expected {hl}")
            metrics = pd.read_csv(run / EVAL / "canonical_metrics_all.csv")
            for model, (method_folder, decoder, key) in models.items():
                part = metrics[(metrics.method == ("teacher" if model == "teacher" else method_folder)) &
                               (metrics.decoder == "training_matched")].copy()
                extra = []
                for case in sorted(run.glob(f"level=*/seed=*/{method_folder}")):
                    level = int(case.parent.parent.name.split("=")[1])
                    seed = int(case.parent.name.split("=")[1])
                    with np.load(case / "snapshots.npz") as saved:
                        steps, tensors = saved["steps"], saved[key]
                    for step, T in zip(steps, tensors):
                        record = json.loads((case / EVAL / decoder / f"step={int(step):05d}.json").read_text())
                        occupancy = np.asarray(record["alignment"]["cooccurrence"], dtype=float).sum(axis=1)
                        gini = 1.0 - (T ** 2).sum(axis=-1)
                        weights = np.broadcast_to(occupancy, gini.shape)
                        extra.append(dict(level=level, seed=seed, step=int(step),
                                          concentration=record["state_representation"]["node_recall"],
                                          gini_weighted=float((gini * weights).sum() / weights.sum())))
                part = part.merge(pd.DataFrame(extra), on=["level", "seed", "step"], how="left", validate="one_to_one")
                if part[["concentration", "gini_weighted"]].isna().any().any() or part.empty:
                    raise ValueError(f"Incomplete {model} records in {run}")
                part["allocation"], part["half_life"], part["model"] = alloc, hl, model
                frames.append(part)
    return pd.concat(frames, ignore_index=True)


def curves(data: pd.DataFrame, level: int, half_lives, name: str) -> None:
    colours = dict(zip(half_lives, COLOURS))
    ncol = 5
    nrow = int(np.ceil(len(METRICS) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.3 * ncol, 3.5 * nrow), squeeze=False, constrained_layout=True)
    part = data[(data.level == level) & (data.model == "student")]
    for ax, (metric, title, sense) in zip(axes.flat, METRICS):
        for alloc, dash in ALLOC_STYLE.items():
            for hl in half_lives:
                stats = part[(part.allocation == alloc) & (part.half_life == hl)].groupby("step")[metric] \
                    .agg(["mean", "std"]).reset_index()
                ax.fill_between(stats.step, stats["mean"] - stats["std"], stats["mean"] + stats["std"],
                                color=colours[hl], alpha=0.10, lw=0)
                ax.plot(stats.step, stats["mean"], color=colours[hl], ls=dash, lw=1.9)
        ax.set_title(f"{title} {sense}", fontsize=9.5, color=INK, loc="left")
        ax.set_xlabel("Training observations", fontsize=8.5, color=MUTED)
        ax.grid(True, color=GRID, lw=0.6)
        ax.tick_params(colors=MUTED, labelsize=8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.set_xlim(0, part.step.max())
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v / 1000:g}k"))
    Line = matplotlib.lines.Line2D
    handles = ([Line([], [], color=colours[hl], lw=2, label=hl) for hl in half_lives] +
               [Line([], [], color=INK, ls=d, lw=2, label=a) for a, d in ALLOC_STYLE.items()])
    fig.legend(handles=handles, loc="outside lower center", ncol=len(handles), frameon=False, fontsize=9)
    fig.suptitle(f"{LEVELS[level]}: EM-JEPA student, "
                 f"{' vs '.join(map(str, half_lives))} — mean ± sd over 3 seeds, training-matched decoder, "
                 "10 × 500 held-out walks", fontsize=11, color=INK)
    path = OUT / f"{name}_level{level}.png"
    fig.savefig(path, dpi=150, facecolor="#fcfcfb")
    plt.close(fig)
    print(f"Wrote {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--half-lives", type=int, nargs="+", default=[40, 80])
    parser.add_argument("--runs", nargs="+", metavar="LABEL=FOLDER",
                        help="compare these EM-JEPA run folders instead of --half-lives")
    parser.add_argument("--name", default="half_life_comparison", help="output file prefix")
    args = parser.parse_args()
    if args.runs:
        variants = [tuple(item.split("=", 1)) for item in args.runs]
        expected = None
    else:
        variants = [(f"Teacher half-life {hl}", folder(hl)) for hl in args.half_lives]
        expected = {f"Teacher half-life {hl}": hl for hl in args.half_lives}
    labels = [label for label, _ in variants]
    data = load(variants, expected)
    for level in sorted(data.level.unique()):
        curves(data, level, labels, args.name)
    last = data[data.step == data.step.max()]
    columns = [m for m, *_ in METRICS]
    last[["level", "allocation", "half_life", "model", "seed"] + columns].to_csv(
        OUT / f"{args.name}_final.csv", index=False)
    summary = last.groupby(["level", "allocation", "model", "half_life"], sort=False)[columns].mean().round(3)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 20)
    print(summary.to_string())


if __name__ == "__main__":
    main()
