#!/usr/bin/env python3
"""Spatial spectral development from frozen GT-projected evaluation records."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd

from mercury_runs.analysis.residual_map_rendering import positions_for_level
from mercury_runs.analysis.spectral_transitions import read_method_records
from mercury_runs.analysis.spatial_spectra import spatial_representation, align_spatial_modes, compare_spatial
from scripts.analysis.cscg_spectral_development import DEFAULT_DATASET, COLORS, LABELS, _save

ENVIRONMENTS = {13: "3x3", 21: "1x4", 20: "1x5", 22: "3x4"}
MODELS = ("mercury", "vanilla_cscg", "oracle_capacity")
COLUMNS = ((0, "initial"), (2000, "source"), (2000, "handoff"), (4000, "trained"), (2000000, "trained"))
COLUMN_LABELS = ("0 samples\nCSCG initialisation", "2,000 samples\nsource / pure CSCG", "2,000 samples\nCSCG handoff", "4,000 samples", "2,000,000 samples")


def _column(step, phase):
    if step == 0:
        return 0
    if phase == "handoff":
        return 2
    return {2000: 1, 4000: 3, 2000000: 4}[step]


def _layout(values, level, state_order):
    positions = positions_for_level(level)
    xs = sorted({float(p[0]) for p in positions.values()})
    ys = sorted({float(p[1]) for p in positions.values()}, reverse=True)
    grid = np.full((len(ys), len(xs)), np.nan)
    for index, state in enumerate(state_order):
        x, y = positions[int(state)]
        grid[ys.index(float(y)), xs.index(float(x))] = values[index]
    return grid


def _blank(ax, label="Unavailable"):
    ax.set_facecolor("#f1f1f1")
    ax.text(.5, .5, label, ha="center", va="center", transform=ax.transAxes, fontsize=7, color="0.4")
    ax.set(xticks=[], yticks=[])


def render_heatmaps(summaries, reference, level, state_order, output, modes):
    shape = _layout(np.zeros(len(state_order)), level, state_order).shape
    aspect = shape[0] / shape[1]
    fig, axes = plt.subplots(4, 5 * modes, figsize=(3.4 * 5, max(5.3, 4 * 1.7 * aspect + 2.5)), squeeze=False, layout="constrained")
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("#444444")
    valid = [s["modes"] for s in summaries.values() if s["modes"] is not None]
    limit = max(np.abs(reference["basis"]).max(), max((np.abs(v).max() for v in valid), default=0))
    for row, model in enumerate(("gt", *MODELS)):
        for column in range(5):
            item = {"modes": reference["basis"], "n_modes": 1, "n": 1} if model == "gt" else summaries.get((model, column))
            for mode in range(modes):
                ax = axes[row, column * modes + mode]
                ax.set_box_aspect(aspect)
                if item is None:
                    _blank(ax)
                elif item["modes"] is None:
                    _blank(ax, "Unresolved\neigenspace")
                else:
                    ax.imshow(_layout(item["modes"][:, mode], level, state_order), cmap=cmap, vmin=-limit, vmax=limit)
                    ax.set(xticks=[], yticks=[])
                    if model != "gt" and mode == 0:
                        ax.set_xlabel(f"{item['n_modes']}/{item['n']} seeds resolved", fontsize=7)
                if row == 0:
                    ax.set_title(f"{COLUMN_LABELS[column]}\ncoordinate {mode + 1}", fontsize=8)
                if column == 0 and mode == 0:
                    ax.set_ylabel("Ground truth" if model == "gt" else LABELS[model], fontsize=10)
    fig.colorbar(ScalarMappable(norm=Normalize(-limit, limit), cmap="RdBu_r"), ax=axes, shrink=.65,
                 label="GT-aligned spectral coordinate (median across resolved seeds)")
    fig.suptitle(f"{ENVIRONMENTS[level]} · emergence of global spatial modes\nSigns and rotations aligned to GT; columns span the leading nonprincipal eigenspace", fontsize=13)
    _save(fig, output, "eigenvector_heatmaps")


def render_convergence(frame, output, level):
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), layout="constrained")
    panels = (("symmetric_eigenvalue_rmse", "Eigenvalue RMSE · symmetric operator"),
              ("symmetric_subspace_error", "Spectral-subspace error (max sin θ)"),
              ("action_tv_similarity", "Action-conditioned TV similarity"))
    positions = [0, 1, 1, 2, 3]
    for ax, (metric, label) in zip(axes, panels):
        for model in MODELS:
            group = frame[frame.model == model].groupby("column")[metric]
            median = group.median()
            x = [positions[c] for c in median.index]
            ax.plot(x, median, "o-", color=COLORS[model], lw=2, label=LABELS[model])
            ax.fill_between(x, group.quantile(.25), group.quantile(.75), color=COLORS[model], alpha=.18)
            for c, marker in ((1, "s"), (2, "D")):
                if model == "mercury" and c in median.index:
                    ax.plot(positions[c], median.loc[c], marker=marker, markerfacecolor="white", color=COLORS[model], ms=7)
        ax.set(xlabel="Total training samples (categorical spacing)", ylabel=label,
               xticks=range(4), xticklabels=["0", "2,000", "4,000", "2,000,000"])
        ax.grid(alpha=.15)
    axes[0].legend(fontsize=8)
    fig.suptitle(f"{ENVIRONMENTS[level]} · medians and interquartile ranges\nHollow square: MERCURY source; hollow diamond: CSCG handoff at the same sample budget", fontsize=12)
    _save(fig, output, "spectral_convergence")


def render_distances(summaries, reference, level, output, residual=False):
    rows = MODELS if residual else ("gt", *MODELS)
    fig, axes = plt.subplots(len(rows), 5, figsize=(15, 3 * len(rows)), squeeze=False, layout="constrained")
    gt = reference["distances"]
    matrices = [s["distances"] for s in summaries.values() if s["distances"] is not None]
    limit = max((np.abs(m - gt).max() for m in matrices), default=1) if residual else max([gt.max(), *[m.max() for m in matrices]])
    limit = max(limit, 1e-12)
    for row, model in enumerate(rows):
        for column in range(5):
            ax = axes[row, column]
            item = {"distances": gt, "n_distances": 1, "n": 1} if model == "gt" else summaries.get((model, column))
            if item is None:
                _blank(ax)
            elif item["distances"] is None:
                _blank(ax, "Unresolved\nprincipal eigenspace")
            else:
                matrix = item["distances"] - gt if residual else item["distances"]
                ax.imshow(matrix, cmap="RdBu_r" if residual else "viridis", vmin=-limit if residual else 0, vmax=limit)
                ticks = range(len(gt))
                ax.set(xticks=ticks, yticks=ticks)
                ax.tick_params(labelsize=5)
                if model != "gt":
                    ax.set_xlabel(f"{item['n_distances']}/{item['n']} seeds resolved", fontsize=7)
            if row == 0:
                ax.set_title(COLUMN_LABELS[column], fontsize=9)
            if column == 0:
                ax.set_ylabel("Ground truth" if model == "gt" else LABELS[model], fontsize=10)
    fig.colorbar(ScalarMappable(norm=Normalize(-limit if residual else 0, limit), cmap="RdBu_r" if residual else "viridis"),
                 ax=axes, shrink=.65, label="Learned − GT distance" if residual else "Spectral diffusion distance")
    fig.suptitle(f"{ENVIRONMENTS[level]} · {'structural distance discrepancies' if residual else 'pairwise state distances in spectral space'}\nFixed GT state order; entrywise medians across resolved seeds", fontsize=13)
    _save(fig, output, "spectral_distance_residuals" if residual else "spectral_distance_matrices")


def run(study, protocol, level, output, modes=2, time=3.):
    samples = [s for s in read_method_records(study, protocol, level) if s["condition"] in MODELS]
    reference = spatial_representation(samples[0]["truth"], modes=modes, time=time)
    if reference["basis"] is None or reference["distances"] is None:
        raise ValueError("GT spatial eigenspace is not separated at the requested mode count")
    # Fix the arbitrary GT signs once, before any cross-run alignment.
    basis = reference["basis"]
    basis *= np.where(basis[np.argmax(abs(basis), axis=0), np.arange(modes)] < 0, -1, 1)
    state_order = samples[0]["record"]["ground_truth"]["state_order"]
    rows, representations, groups = [], [], {}
    for index, sample in enumerate(samples):
        r = sample["record"]
        representation = spatial_representation(sample["operator"], modes=modes, time=time)
        aligned = align_spatial_modes(representation["basis"], basis)
        column = _column(sample["checkpoint_step"], sample["phase"])
        row = {"model": sample["condition"], "phase": sample["phase"], "column": column,
               "checkpoint_step": sample["checkpoint_step"], "snapshot_step": r["snapshot_step"],
               "run_id": sample["run_id"], "seed": sample["config"]["seed"], "record_path": sample["record_path"],
               "modes_defined": representation["modes_defined"], "distances_defined": representation["principal_defined"],
               "action_tv_similarity": r["action_conditioned"]["tv_similarity_occupancy"],
               "unmatched_mass": float(np.mean(sample["leakage"])), **compare_spatial(reference, representation)}
        rows.append(row)
        representations.append({**representation, "aligned": aligned})
        groups.setdefault((sample["condition"], column), []).append(index)
    frame = pd.DataFrame(rows)
    summaries = {}
    for key, indices in groups.items():
        aligned = [representations[i]["aligned"] for i in indices if representations[i]["aligned"] is not None]
        distances = [representations[i]["distances"] for i in indices if representations[i]["distances"] is not None]
        summaries[key] = {"n": len(indices), "n_modes": len(aligned), "n_distances": len(distances),
                          "modes": np.median(aligned, axis=0) if aligned else None,
                          "distances": np.median(distances, axis=0) if distances else None}
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "spatial_metrics.csv", index=False)
    n = len(basis)
    np.savez_compressed(output / "spatial_representations.npz", gt_operator=reference["operator"],
                        gt_eigenvalues=reference["eigenvalues"], gt_modes=basis, gt_distances=reference["distances"],
                        state_order=state_order, operators=np.stack([r["operator"] for r in representations]),
                        aligned_modes=np.stack([r["aligned"] if r["aligned"] is not None else np.full((n, modes), np.nan) for r in representations]),
                        distances=np.stack([r["distances"] if r["distances"] is not None else np.full((n, n), np.nan) for r in representations]))
    render_heatmaps(summaries, reference, level, state_order, output, modes)
    render_convergence(frame, output, level)
    render_distances(summaries, reference, level, output)
    render_distances(summaries, reference, level, output, residual=True)
    fig, ax = plt.subplots(figsize=(4, 3), layout="constrained")
    grid = _layout(np.asarray(state_order), level, state_order)
    cmap = plt.get_cmap("Greys").copy()
    cmap.set_bad("#444444")
    ax.imshow(np.where(np.isfinite(grid), 0, np.nan), cmap=cmap, vmin=0, vmax=1)
    for (y, x), value in np.ndenumerate(grid):
        if np.isfinite(value):
            ax.text(x, y, str(int(value)), ha="center", va="center")
    ax.set(xticks=[], yticks=[], title="GT state IDs used by distance matrices")
    _save(fig, output, "state_layout")
    _write_report(frame, output, modes, time, protocol)
    return frame


def _write_report(frame, output, modes, time, protocol):
    metadata = {"protocol_id": protocol, "operator": "W=(P+P.T)/2; leakage=1-P@1; D=diag(W@1+leakage); L=I-D^-1/2 W D^-1/2",
                "displayed_modes": modes, "diffusion_time": time, "degeneracy_tolerance": 1e-7,
                "distance_embedding": "All nonprincipal eigenvectors weighted by exp(-time*eigenvalue)",
                "alignment": "Orthogonal Procrustes against fixed GT basis; columns are aligned eigenspace coordinates",
                "aggregation": "Entrywise medians across resolved seeds; counts shown. Median maps are not eigenvectors of a median operator; median dissimilarities need not form a Euclidean distance matrix.",
                "records": len(frame), "source_and_handoff": "Same staged model and sample budget; representation transfer, not additional training"}
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    lines = ["# Spatial emergence of global relational structure", "",
             "Primary figures: [eigenvector heatmaps](eigenvector_heatmaps.png), [spectral convergence](spectral_convergence.png), [state-distance matrices](spectral_distance_matrices.png), [distance discrepancies](spectral_distance_residuals.png). [State-ID layout](state_layout.png). All figures also have PDF versions.", "",
             "Spatial diagnostics use a symmetric normalised Laplacian of the canonical GT-projected transitions. Unmatched probability enters the degree as leakage and is never redistributed among matched targets. This is a separate diagnostic operator: existing directed spectral results and canonical action-conditioned TV are unchanged.", "",
             f"The lowest eigenmode is excluded. In closed GT dynamics it is stationary; with unmatched mass it is a survival/coverage mode. The next {modes} modes define the displayed subspace. Signs and rotations are aligned to a fixed GT basis by Procrustes. Display columns are aligned eigenspace coordinates and may mix individual eigenvectors; they are not persistent identities for distinct eigenvalues. Ground-truth alignment supplies a common frame, not extra training evidence.", "",
             "No eigenspace is split at a repeated boundary (tolerance 1e-7). An unresolved lowest boundary prevents both maps and distances; an unresolved upper display boundary prevents maps only. Resolved-seed counts appear on the plots. Conditional medians can omit poorly resolved runs, so use the counts alongside the summaries.", "",
             f"Distance matrices use all nonprincipal modes weighted by exp(−{time:g}λ). They do not depend on eigenvector signs, rotations within repeated eigenspaces, or the displayed mode count. A common colour scale is used across every model and checkpoint; residuals show learned minus GT distances. Positive residuals separate states too much, negative residuals bring them too close. Entrywise medians summarise runs rather than a single learned embedding.", "",
             "MERCURY and its CSCG continuation are one staged trajectory. Source and handoff have the same 2,000-sample budget and distinct columns; the transfer can change structural diagnostics without new observations. Later points use cumulative source-plus-CSCG samples. Source training checkpoints before 2,000 are unavailable. Changing GT alignments can contribute to motion and are not causally separated from learning.", "",
             "Convergence uses per-run symmetric eigenvalue RMSE, displayed-subspace principal-angle error, and the stored action-conditioned TV similarity. Lines show medians; bands show interquartile ranges. The x-axis uses categorical spacing, explicitly labelled with total sample counts. Directed eigenvalue errors must not be compared numerically with these Laplacian eigenvalue errors.", "",
             "## Median diagnostics", "", "| Model | Samples | Stage | Eigenvalue RMSE | Subspace error | Distance RMSE | TV similarity | Resolved maps |", "|---|---:|---|---:|---:|---:|---:|---:|"]
    for (model, column), group in frame.groupby(["model", "column"]):
        median = group.median(numeric_only=True)
        fmt = lambda x: f"{x:.4g}" if np.isfinite(x) else "—"
        values = " | ".join(fmt(median[k]) for k in ("symmetric_eigenvalue_rmse", "symmetric_subspace_error", "spectral_distance_rmse", "action_tv_similarity"))
        lines.append(f"| {LABELS[model]} | {int(group.checkpoint_step.iloc[0]):,} | {group.phase.iloc[0]} | {values} | {int(group.modes_defined.sum())}/{len(group)} |")
    directed = f"../../model_spectral_development/{output.name}"
    lines += ["", f"Supplementary directed diagnostics: [complex-eigenvalue scatter]({directed}/eigenvalue_spectra.png), [directed report and PCA]({directed}/findings.md). These directed results remain separate from the symmetric spatial view.", ""]
    (output / "README.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--levels", nargs="+", type=int, choices=list(ENVIRONMENTS), default=[13, 21, 20, 22])
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    manifest = json.loads((args.dataset_root / "notebook_plots/main_study_eval_v5/evaluation_v5_manifest.json").read_text())
    study = Path(manifest["source_shadow_study"])
    study = study if study.is_absolute() else ROOT / study
    if not study.resolve().is_relative_to(args.dataset_root.resolve()):
        parser.error("Manifest redirects outside selected dataset")
    output = args.output_dir or args.dataset_root / "notebook_plots/main_study_eval_v5/spatial_spectral_development"
    for level in args.levels:
        frame = run(study, manifest["protocol_id"], level, output / ENVIRONMENTS[level])
        print(f"{ENVIRONMENTS[level]}: {len(frame)} records; {int(frame.modes_defined.sum())} resolved spatial maps; {output / ENVIRONMENTS[level]}")
    lines = ["# Spatial spectral analysis", "", "Primary visualisations of global relational structure in MERCURY→CSCG, standard CSCG and oracle-capacity CSCG. Each environment uses the same symmetric-operator construction, fixed GT state order and colour scales shared across its checkpoints. See the per-environment report for operator definitions, degeneracy handling and resolved-seed counts.", "",
             "| Environment | Eigenvector heatmaps | Convergence | State distances | Discrepancies | Report |", "|---|---|---|---|---|---|"]
    for environment in ENVIRONMENTS.values():
        if (output / environment / "README.md").exists():
            lines.append(f"| {environment} | [Maps]({environment}/eigenvector_heatmaps.png) | [Median + IQR]({environment}/spectral_convergence.png) | [Distances]({environment}/spectral_distance_matrices.png) | [Residuals]({environment}/spectral_distance_residuals.png) | [Methods and results]({environment}/README.md) |")
    lines += ["", "Each figure is also saved as PDF. Per-run metrics and NumPy arrays accompany the figures. Complex-eigenvalue scatters remain supplementary directed diagnostics; the existing directed analysis and established TV metric are unchanged.", "",
              "Reproduce: `.venv/bin/python scripts/analysis/plot_spatial_spectral_development.py`.", ""]
    (output / "README.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
