"""PCA trajectories of projected transition tensors: CSCG (long EM) vs EM-JEPA (student and EMA teacher).

Reads the training-matched evaluation records written by cscg_distillation_matched_eval.py. Each
checkpoint's ``P_hat_full`` (learned transitions projected into ground-truth state space with the
record's canonical alignment, plus the unmatched sink) is one point; one PCA is fitted per level to
all arms, seeds and checkpoints, and the record's ``P_true_full`` is projected as the GT reference.
A second figure repeats this for the no-self-loop movement distributions, using the same
conditioning as cscg_transition_phase_space.py.

  PYTHONPATH=.:src .venv/bin/python scripts/analysis/cscg_jepa_pca_trajectories.py \
      --case 23:<em_baselines dir>:<jepa_mix1 dir> [--case ...] --output DIR [--walks 10 --walk-length 500]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from mercury_runs.infrastructure.paper_precision import condition_non_selfloop_action_distributions
from scripts.analysis.cscg_distillation_matched_eval import output_name

# (label, run key, method folder, decoder folder, colour, line style)
ARMS = [("CSCG (long EM)", "em", "long", "training_matched", "#2a78d6", "-"),
        ("EM-JEPA student", "jepa", "predictive", "training_matched", "#eb6834", "-"),
        ("EM-JEPA teacher", "jepa", "predictive", "teacher_training_matched", "#eb6834", "--")]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def load_case(level, runs, evaluation):
    rows, full, nonself, gt_full, gt_nonself = [], [], [], None, None
    for label, key, method, decoder, _, _ in ARMS:
        for seed_dir in sorted(runs[key].glob(f"level={level}/seed=*")):
            seed = int(seed_dir.name.split("=")[1])
            records = sorted((seed_dir / method / evaluation / decoder).glob("step=*.json"))
            if not records:
                raise FileNotFoundError(f"No {decoder} records for {label}, seed {seed}: {seed_dir / method}")
            for path in records:
                record = json.loads(path.read_text())
                action = record["action_conditioned"]
                q = np.asarray(action["P_hat_full"], dtype=np.float64)
                gt = np.asarray(action["P_true_full"], dtype=np.float64)
                valid = np.asarray(action["valid_rows"], dtype=bool)
                q_ns, gt_ns, move = condition_non_selfloop_action_distributions(q, gt, valid)
                if gt_full is None:
                    gt_full, gt_nonself, move_rows = gt, gt_ns[move], move
                elif not (np.array_equal(gt, gt_full) and np.array_equal(move, move_rows)):
                    raise ValueError(f"Ground truth differs across records: {path}")
                full.append(q.ravel())
                nonself.append(q_ns[move].ravel())
                rows.append(dict(level=level, arm=label, seed=seed, step=int(record["snapshot_step"]),
                                 tv_similarity=float(action["tv_similarity_occupancy"]),
                                 tv_similarity_nonself=float(action["tv_non_self_loop_similarity_occupancy"])))
    frame = pd.DataFrame(rows)
    if frame.duplicated(["arm", "seed", "step"]).any():
        raise ValueError(f"Duplicate checkpoint records at level {level}")
    gts = {"full": gt_full.ravel(), "nonself": gt_nonself.ravel()}
    variance = {}
    for name, matrix in (("full", np.stack(full)), ("nonself", np.stack(nonself))):
        if not np.isfinite(matrix).all():
            raise ValueError(f"Non-finite {name} features at level {level}")
        pca = PCA(n_components=2, svd_solver="full")
        frame[[f"pc1_{name}", f"pc2_{name}"]] = pca.fit_transform(matrix)
        frame.attrs[f"gt_{name}"] = pca.transform(gts[name].reshape(1, -1))[0]
        frame[f"dist_gt_{name}"] = np.linalg.norm(matrix - gts[name], axis=1)   # in the full feature space
        variance[name] = pca.explained_variance_ratio_
    frame.attrs["variance"] = variance
    return frame


def panel(ax, frame, name):
    x, y = f"pc1_{name}", f"pc2_{name}"
    lines = []
    for label, _, _, _, colour, ls in ARMS:
        part = frame[frame.arm == label]
        for _, seed in part.groupby("seed"):
            seed = seed.sort_values("step")
            ax.plot(seed[x], seed[y], color=colour, ls=ls, lw=.8, alpha=.28, zorder=1)
        mean = part.groupby("step")[[x, y, f"dist_gt_{name}"]].mean().sort_index()
        ax.plot(mean[x], mean[y], color=colour, ls=ls, lw=2.4, zorder=3)
        for i in range(0, len(mean) - 1, max(1, len(mean) // 6)):   # direction of travel
            ax.annotate("", xy=mean.iloc[i + 1][[x, y]], xytext=mean.iloc[i][[x, y]], zorder=3,
                        arrowprops=dict(arrowstyle="-|>", color=colour, lw=1.4, mutation_scale=11))
        ax.scatter(mean[x].iloc[0], mean[y].iloc[0], s=55, facecolor="white", edgecolor=colour, lw=1.8, zorder=4)
        ax.scatter(mean[x].iloc[-1], mean[y].iloc[-1], s=60, marker="s", color=colour, edgecolor="white", lw=1, zorder=5)
        lines.append(f"{label}: {mean[f'dist_gt_{name}'].iloc[-1]:.2f}")
    gt = frame.attrs[f"gt_{name}"]
    ax.scatter(*gt, marker="*", s=260, color=INK, edgecolor="white", lw=1.2, zorder=6)
    ax.annotate("GT", gt, xytext=(6, 6), textcoords="offset points", fontsize=9, weight="bold", color=INK)
    ax.text(.02, .98, "Final ‖P̂ − P_GT‖:\n" + "\n".join(lines), transform=ax.transAxes, fontsize=7, color=MUTED,
            va="top", zorder=7, bbox=dict(facecolor="white", edgecolor=GRID, alpha=.88, boxstyle="round,pad=.3"))
    variance = frame.attrs["variance"][name]
    ax.set_xlabel(f"PC1 ({variance[0]:.0%})", fontsize=8, color=MUTED)
    ax.set_ylabel(f"PC2 ({variance[1]:.0%})", fontsize=8, color=MUTED)
    ax.set_title(f"Level {int(frame.level.iloc[0])}", fontsize=10, color=INK, loc="left")
    ax.tick_params(labelsize=7, colors=MUTED, length=0)
    ax.grid(color=GRID, lw=.8); ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)


def figure(frames, name, title, path):
    columns = 3
    rows = (len(frames) + 1 + columns - 1) // columns   # one spare cell holds the legend
    fig, axes = plt.subplots(rows, columns, figsize=(5.6 * columns, 4.9 * rows), squeeze=False)
    axes = axes.ravel()
    for ax, frame in zip(axes, frames):
        panel(ax, frame, name)
    for ax in axes[len(frames):]:
        ax.axis("off")
    handles = [Line2D([0], [0], color=c, ls=ls, lw=2.4, label=label) for label, _, _, _, c, ls in ARMS]
    handles += [Line2D([0], [0], marker="o", color="none", markerfacecolor="white", markeredgecolor=MUTED, label="start (step 0; projection depends on each arm's decoder)"),
                Line2D([0], [0], marker="s", color="none", markerfacecolor=MUTED, label="end (step 20,000)"),
                Line2D([0], [0], marker="*", color="none", markerfacecolor=INK, markersize=12, label="ground truth")]
    axes[len(frames)].legend(handles=handles, loc="center left", frameon=False, fontsize=10)
    fig.suptitle(title, fontsize=11, color=INK, x=.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, .96))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case", action="append", required=True, help="LEVEL:EM_BASELINES_DIR:JEPA_DIR")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--walks", type=int, default=10)
    p.add_argument("--walk-length", type=int, default=500)
    p.add_argument("--rand-prob", type=float, default=.1)
    args = p.parse_args()
    evaluation = output_name(args.walks, args.walk_length, args.rand_prob)
    frames = []
    for case in args.case:
        level, em, jepa = case.split(":")
        frames.append(load_case(int(level), {"em": Path(em), "jepa": Path(jepa)}, evaluation))
    args.output.mkdir(parents=True, exist_ok=True)
    coordinates = []
    for frame in frames:   # attrs hold per-level arrays (GT coordinates); pandas cannot concat them
        plain = frame.copy()
        plain.attrs = {}
        coordinates.append(plain)
    pd.concat(coordinates, ignore_index=True).to_csv(args.output / "pca_trajectory_coordinates.csv", index=False)
    common = "3 seeds, 20,000 steps; thin = seeds, thick = mean over seeds; one PCA per level"
    figure(frames, "full", f"Projected transition tensors, PCA trajectories: CSCG vs EM-JEPA ({common})",
           args.output / "pca_trajectories_full.png")
    figure(frames, "nonself", f"No-self-loop movement distributions, PCA trajectories: CSCG vs EM-JEPA ({common})",
           args.output / "pca_trajectories_nonself.png")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
