"""Aggregate per-seed offline JEPA-CSCG runs (one <root>/seed=*/ directory per seed).

PYTHONPATH=.:src .venv/bin/python scripts/analysis/cscg_offline_jepa_seed_summary.py <root>
Reads each seed's learning_curves.csv; writes <root>/aggregate.md, paired_differences.csv
and mean_learning_curves.png. Seed is the paired unit; checkpoints are not replicates.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

METRICS = dict(similarity="Similarity", nonself_similarity="No-self sim.", edge_f1_auc="Edge F1 AUC",
               purity="Purity", concentration="Concentration", nll_bits="NLL bits")
LABELS = dict(offline="Offline CSCG", patch="Patch CSCG", ema_control="EMA control", jepa="JEPA-CSCG")
PAIRS = [("jepa", "patch"), ("jepa", "offline"), ("patch", "offline")]


def load(root):
    frames = []
    for folder in sorted(root.glob("seed=*")):
        df = pd.read_csv(folder / "learning_curves.csv")
        manifest = json.loads((folder / "experiment.json").read_text())
        frames.append(df.assign(seed=manifest["seed"]))
    df = pd.concat(frames, ignore_index=True)
    return df[df.role == "student"], manifest


def main():
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    root = p.parse_args().root.resolve()
    curves, manifest = load(root)
    final_pass = int(curves.pass_index.max())
    lengths = sorted(curves.patch_length.dropna().unique())
    # Offline does not depend on L: replicate it under each patch length for pairing.
    expanded = pd.concat([pd.concat([curves[curves.patch_length == L],
                                     curves[curves.method == "offline"].assign(patch_length=L)]) for L in lengths])
    final = expanded[expanded.pass_index == final_pass]
    seeds = sorted(int(s) for s in final.seed.unique())
    if any(len(part) != len(seeds) for _, part in final.groupby(["patch_length", "method"])):
        raise ValueError("Every seed needs every arm at the final pass")
    area = expanded.sort_values("pass_index").groupby(["patch_length", "method", "seed"]).apply(
        lambda part: np.trapezoid(part.similarity, part.pass_index) / final_pass, include_groups=False)
    final = final.merge(area.rename("time_mean_similarity").reset_index(), on=["patch_length", "method", "seed"])
    metrics = [*METRICS, "time_mean_similarity"]
    rows = []
    for L in lengths:
        by = final[final.patch_length == L].set_index(["method", "seed"])
        for left, right in PAIRS:
            for metric in metrics:
                delta = by.loc[left, metric] - by.loc[right, metric]
                better = -delta if metric == "nll_bits" else delta
                rows.append(dict(patch_length=int(L), comparison=f"{left}-{right}", metric=metric,
                                 mean=delta.mean(), sd=delta.std(ddof=1), wins=int((better > 0).sum()),
                                 losses=int((better < 0).sum()), **{f"seed{s}": delta[s] for s in seeds}))
    paired = pd.DataFrame(rows)
    paired.to_csv(root / "paired_differences.csv", index=False)

    config = manifest["config"]
    lines = ["# Offline JEPA-CSCG: multi-seed summary", "",
             f"Level {manifest['level']}, seeds {seeds}, E = {config['passes']} passes x J = "
             f"{config['coupled_iterations_per_pass']}, lambda = {config['jepa_weight']}, beta = {config['ema']}. "
             f"Student endpoints after pass {final_pass}; mean (seed SD).", "",
             "| L | Model | " + " | ".join(METRICS.values()) + " | Mean similarity over passes |",
             "| ---: | --- |" + " ---: |" * (len(METRICS) + 1)]
    for L in lengths:
        for method in ["offline", "patch", "jepa"]:
            part = final[(final.patch_length == L) & (final.method == method)]
            cells = [f"{part[m].mean():.4f} ({part[m].std(ddof=1):.4f})" for m in metrics]
            lines.append(f"| {int(L)} | {LABELS[method]} | " + " | ".join(cells) + " |")
    lines += ["", "EMA control students equal patch CSCG exactly (asserted per run) and are omitted.", "",
              "## Paired differences (mean, seed SD, wins/losses; a win is an improvement, lower NLL counts as better)", "",
              "| L | Comparison | Similarity | Edge F1 AUC | Concentration | NLL bits | Mean similarity over passes |",
              "| ---: | --- | ---: | ---: | ---: | ---: | ---: |"]
    for (L, comparison), part in paired.groupby(["patch_length", "comparison"], sort=False):
        part = part.set_index("metric")
        cells = [f"{part.loc[m, 'mean']:+.4f} ({part.loc[m, 'sd']:.4f}) {part.loc[m, 'wins']}/{part.loc[m, 'losses']}"
                 for m in ["similarity", "edge_f1_auc", "concentration", "nll_bits", "time_mean_similarity"]]
        lines.append(f"| {L} | {comparison} | " + " | ".join(cells) + " |")
    lines += ["", "Per-seed values for every metric are in paired_differences.csv. Five seeds: descriptive, no significance test."]
    (root / "aggregate.md").write_text("\n".join(lines) + "\n")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    shown = ["similarity", "edge_f1_auc", "concentration", "nll_bits"]
    fig, axes = plt.subplots(len(lengths), len(shown), figsize=(4.2 * len(shown), 3.4 * len(lengths)), squeeze=False)
    for i, L in enumerate(lengths):
        for j, metric in enumerate(shown):
            ax = axes[i, j]
            for method, color in [("offline", "black"), ("patch", "tab:blue"), ("jepa", "tab:red")]:
                agg = expanded[(expanded.patch_length == L) & (expanded.method == method)].groupby("pass_index")[metric].agg(["mean", "std"])
                ax.plot(agg.index, agg["mean"], color=color, label=LABELS[method])
                ax.fill_between(agg.index, agg["mean"] - agg["std"], agg["mean"] + agg["std"], color=color, alpha=.12)
            ax.set(title=f"L={int(L)}: {METRICS[metric]}", xlabel="Data pass (student EM updates)")
            ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(root / "mean_learning_curves.png", dpi=150)
    plt.close(fig)
    print(f"Summary: {root / 'aggregate.md'}")


if __name__ == "__main__":
    main()
