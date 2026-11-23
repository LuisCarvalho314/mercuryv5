#!/usr/bin/env python3
"""Compare projected transition spectra across capacities and staged model trajectories."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from mercury_runs.analysis.plots.publication_style import METHOD_COLORS
from mercury_runs.analysis.spectral_transitions import (
    dominant_subspace, read_capacity_records, read_method_records, spectral_comparison, subspace_error,
)

DEFAULT_DATASET = ROOT / "results/current/action_alignment_rerun_matched_budget_lambda099_20261004"
ENVIRONMENTS = {13: "3x3", 21: "1x4", 20: "1x5", 22: "3x4"}
COLORS = {"oracle_capacity": METHOD_COLORS["oracle_capacity"], "vanilla_cscg": METHOD_COLORS["standard_cscg"]}
COLORS.update({"mercury": METHOD_COLORS["mercury"], "tmgwr": METHOD_COLORS["tmgwr"],
               "mercury_cscg_hybrid": METHOD_COLORS["mercury_initialised"], "tmgwr_cscg_hybrid": METHOD_COLORS["tmgwr_initialised"]})
LABELS = {"oracle_capacity": "Oracle capacity", "vanilla_cscg": "Standard CSCG", "mercury": "MERCURY",
          "tmgwr": "TMGWR → CSCG", "mercury_cscg_hybrid": "MERCURY → CSCG", "tmgwr_cscg_hybrid": "TMGWR → CSCG"}
LABELS["mercury"] = "MERCURY → CSCG"
MARKERS = dict(zip(LABELS, ["o", "^", "s", "D", "P", "X"]))
METRICS = ["eigenvalue_rmse", "gap_error", "subspace_sin_max"]


def _label(frame: pd.DataFrame, preset: str) -> str:
    n = frame.loc[frame.preset == preset, "n_learned_states"].unique()
    capacity = str(n[0]) if len(n) == 1 else f"{min(n)}–{max(n)}"
    return f"{LABELS[preset]} ({capacity} states)"


def _save(fig, output: Path, name: str) -> None:
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"{name}.{suffix}", dpi=180, bbox_inches="tight")
    plt.close(fig)


def _correlations(frame: pd.DataFrame, groups: list[str], x_columns: list[str], y_columns: list[str]) -> pd.DataFrame:
    rows = []
    for key, group in frame.groupby(groups, sort=True):
        identity = dict(zip(groups, key if isinstance(key, tuple) else (key,)))
        for x in x_columns:
            for y in y_columns:
                pair = group[[x, y]].replace([np.inf, -np.inf], np.nan).dropna()
                defined = len(pair) >= 3 and pair[x].nunique() > 1 and pair[y].nunique() > 1
                rows.append({**identity, "x": x, "y": y, "n_pairs": len(pair),
                             "spearman_rho": pair[x].corr(pair[y], method="spearman") if defined else np.nan})
    return pd.DataFrame(rows)


def analyse_environment(study: Path, protocol: str, level: int, output: Path, rank: int = 2, *, all_methods: bool = False) -> pd.DataFrame:
    samples = read_method_records(study, protocol, level) if all_methods else read_capacity_records(study, protocol, level)
    rows, eigenvalues = [], []
    previous = {}
    for sample in samples:
        r, c = sample["record"], sample["config"]
        stats, gt_values, values = spectral_comparison(sample["truth"], sample["operator"], rank=rank)
        counts = np.asarray(r["action_conditioned"]["source_action_counts"], dtype=float)
        state_weights = counts.sum(axis=0) / counts.sum()
        alignment = np.asarray(r["alignment"]["learned_to_true"])
        step = r["snapshot_step"]
        condition = sample.get("condition", c["cscg_structural_preset"])
        checkpoint_step = sample.get("checkpoint_step", step)
        row = {"level": level, "environment": ENVIRONMENTS[level], "run_id": sample["run_id"],
               "preset": condition, "seed": c["seed"], "snapshot_step": step, "checkpoint_step": checkpoint_step,
               "namespace": r["method"], "phase": sample.get("phase", "cscg"),
               "training_pass": checkpoint_step / c["num_steps"],
               "phase_observed_samples": r.get("observed_samples", step),
               "n_learned_states": r["state_representation"]["n_learned_states"],
               "n_true_states": len(sample["truth"]), "record_path": sample["record_path"],
               "action_tv_similarity": r["action_conditioned"]["tv_similarity_occupancy"],
               "nll_bits_per_symbol": (r.get("model_fit") or {}).get("bits_per_symbol", np.nan),
               "unmatched_mass": float(state_weights @ sample["leakage"]),
               "unmatched_learned_states": int((alignment < 0).sum()),
               "alignment_change_fraction": np.nan, "subspace_step_sin_max": np.nan, **stats}
        trajectory = (sample["run_id"], condition)
        prior = previous.get(trajectory)
        if prior is not None:
            if prior["namespace"] == r["method"]:
                if prior["alignment"].shape != alignment.shape:
                    raise ValueError("Learned state IDs changed size within a CSCG run")
                row["alignment_change_fraction"] = float(np.mean(prior["alignment"] != alignment))
            row["subspace_step_sin_max"] = subspace_error(prior["basis"], dominant_subspace(sample["operator"], rank))
        previous[trajectory] = {"alignment": alignment, "basis": dominant_subspace(sample["operator"], rank), "namespace": r["method"]}
        rows.append(row)
        for mode, (target, value) in enumerate(zip(gt_values, values)):
            eigenvalues.append({"run_id": sample["run_id"], "preset": row["preset"], "seed": row["seed"],
                                "snapshot_step": step, "checkpoint_step": checkpoint_step, "phase": row["phase"], "mode": mode, "gt_real": target.real, "gt_imag": target.imag,
                                "real": value.real, "imag": value.imag, "absolute_error": abs(target - value)})
    frame = pd.DataFrame(rows)
    # One shared environment-wise basis, following the existing transition PCA.
    vectors = np.stack([np.asarray(s["record"]["action_conditioned"]["P_hat_full"]).ravel() for s in samples])
    gt_vector = np.asarray(samples[0]["record"]["action_conditioned"]["P_true_full"]).ravel()
    pca = PCA(n_components=2, svd_solver="full")
    frame[["pc1", "pc2"]] = pca.fit_transform(vectors)
    gt_coordinate = pca.transform(gt_vector[None])[0]
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "operators_and_pca.npz", operators=np.stack([s["operator"] for s in samples]),
                        ground_truth=samples[0]["truth"], learned_vectors=vectors, ground_truth_vector=gt_vector,
                        components=pca.components_, feature_mean=pca.mean_, gt_coordinate=gt_coordinate,
                        explained_variance_ratio=pca.explained_variance_ratio_)
    frame.to_csv(output / "checkpoint_metrics.csv", index=False)
    pd.DataFrame(eigenvalues).to_csv(output / "eigenvalues.csv", index=False)
    interval_rows = []
    change_columns = METRICS + ["action_tv_similarity", "nll_bits_per_symbol", "unmatched_mass"]
    for _, group in frame.groupby(["run_id", "preset"]):
        ordered = group.sort_values("checkpoint_step", kind="stable").to_dict("records")
        for old, new in zip(ordered[:-1], ordered[1:]):
            interval_rows.append({**{k: new[k] for k in ("run_id", "preset", "seed", "alignment_change_fraction", "subspace_step_sin_max")},
                                  "from_step": old["checkpoint_step"], "to_step": new["checkpoint_step"],
                                  "from_phase": old["phase"], "to_phase": new["phase"],
                                  **{f"delta_{k}": new[k] - old[k] for k in change_columns}})
    intervals = pd.DataFrame(interval_rows)
    intervals.to_csv(output / "interval_changes.csv", index=False)
    _correlations(frame, ["preset", "checkpoint_step", "phase"], METRICS, ["action_tv_similarity", "nll_bits_per_symbol", "unmatched_mass"]).to_csv(output / "checkpoint_associations.csv", index=False)
    _correlations(intervals, ["preset", "from_step", "to_step", "from_phase", "to_phase"], [f"delta_{k}" for k in METRICS],
                  ["delta_action_tv_similarity", "delta_nll_bits_per_symbol", "alignment_change_fraction"]).to_csv(output / "interval_associations.csv", index=False)
    metadata = {"dataset_study": str(study), "protocol_id": protocol, "level": level,
                "n_records": len(frame), "n_seeds": int(frame.seed.nunique()),
                "steps": sorted(frame.checkpoint_step.unique().tolist()), "subspace_rank": rank,
                "comparison": "all_methods" if all_methods else "capacity",
                "checkpoint_clock": "Cumulative training samples, including hybrid source stage",
                "operator": "Directed evaluation-policy collapse, matched GT targets only; no row renormalisation; self-loops retained",
                "gaps": {"unit_second_gap": "1 - |lambda_2|", "modulus_gap": "|lambda_1| - |lambda_2|"},
                "subspace": "Dominant right complex-Schur invariant subspace, including Perron mode; largest sine of principal angles; undefined at modulus ties within 1e-7",
                "eigenvalues": "Minimum total complex-plane distance assignment against GT; mode IDs are GT anchors, not tracked learned modes",
                "pca": "One centred fit to learned action tensors including sink; GT excluded; no feature scaling",
                "pca_explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
                "training_settings": {key: samples[0]["config"].get(key) for key in ("num_steps", "cscg_n_iter", "cscg_batch_size", "cscg_online_lambda", "cscg_pseudocount", "cscg_training_mode")},
                "undefined_subspaces": int((~frame.subspace_defined).sum()),
                "limits": ["Substochastic gaps are not Markov mixing rates; leakage retained as row deficit.",
                           "Alignment changes are diagnostics, not a causal decomposition of spectral change.",
                           "Two allocations differ in total capacity and clone distribution across observations.",
                           "Associations are descriptive, within condition and checkpoint/interval; no pooled independent-sample inference.",
                           "Policy collapse can hide action-specific errors; action TV and full-tensor PCA remain complementary."]}
    if all_methods:
        metadata["training_settings"] = "Method-specific; see original run configs linked by record paths"
        metadata["limits"][2] = "MERCURY/TMGWR are staged trajectories, not capacity sweeps. Only their source endpoint at 2000 samples is saved. Source NLL is unavailable."
        metadata["inference_by_namespace"] = {s["record"]["method"]: s["record"].get("inference", {}) for s in samples}
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    _plot_trajectories(frame, output)
    _plot_eigenvalues(pd.DataFrame(eigenvalues), frame, output)
    _plot_relationships(frame, intervals, output)
    _plot_pca(frame, gt_coordinate, pca.explained_variance_ratio_, output)
    if all_methods:
        _write_method_findings(frame, metadata, output)
    else:
        _write_findings(frame, metadata, output)
    return frame


def _plot_trajectories(frame, output):
    panels = [("eigenvalue_rmse", "Eigenvalue RMSE"), ("modulus_gap", "Modulus gap"),
              ("subspace_sin_max", "Subspace error (max sin θ)"), ("action_tv_similarity", "Action-conditioned TV similarity"),
              ("nll_bits_per_symbol", "NLL (bits/symbol)"), ("unmatched_mass", "Unmatched-target probability")]
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.5), layout="constrained")
    steps = sorted(frame.checkpoint_step.unique())
    for ax, (metric, label) in zip(axes.flat, panels):
        for preset, group in frame.groupby("preset"):
            for _, seed in group.groupby("seed"):
                seed = seed.sort_values("checkpoint_step", kind="stable")
                ax.plot([steps.index(s) for s in seed.checkpoint_step], seed[metric], color=COLORS[preset], alpha=.1, lw=.7)
            grouped = group.groupby(["checkpoint_step", "phase"])[metric]
            median = grouped.median()
            positions = [steps.index(step) for step, _ in median.index]
            # Source precedes handoff at the same sample count, including in medians.
            order = sorted(range(len(median)), key=lambda i: (positions[i], {"source": 0, "handoff": 1}.get(median.index[i][1], 2)))
            positions = np.asarray(positions)[order]
            ax.plot(positions, median.iloc[order], "o-", color=COLORS[preset], label=_label(frame, preset))
            ax.fill_between(positions, grouped.quantile(.25).iloc[order], grouped.quantile(.75).iloc[order], color=COLORS[preset], alpha=.15)
            for i, (step, phase) in enumerate(median.index):
                if phase in ("source", "handoff"):
                    ax.plot(steps.index(step), median.iloc[i], marker="s" if phase == "source" else "D",
                            markerfacecolor="white", markeredgecolor=COLORS[preset], ls="", markersize=5)
        if metric == "modulus_gap":
            ax.axhline(frame.gt_modulus_gap.iloc[0], color="black", ls="--", lw=1, label="Ground truth")
        ax.set(xlabel="Total training samples (categorical spacing)", ylabel=label, xticks=range(len(steps)),
               xticklabels=[f"{v:,}" for v in steps])
        ax.grid(alpha=.15)
    axes.flat[0].legend(fontsize=8)
    if frame.phase.isin(["source", "handoff"]).any():
        axes.flat[1].legend(handles=[Line2D([], [], marker="s", markerfacecolor="white", color="0.4", ls="", label="Source"),
                                     Line2D([], [], marker="D", markerfacecolor="white", color="0.4", ls="", label="CSCG handoff")], fontsize=8)
    fig.suptitle(f"{frame.environment.iloc[0]} · median, interquartile range and individual seeds")
    _save(fig, output, "spectral_trajectories")


def _plot_eigenvalues(values, frame, output):
    steps = sorted(values.checkpoint_step.unique())
    fig, axes = plt.subplots(1, len(steps), figsize=(3.2 * len(steps), 3.3), layout="constrained", squeeze=False)
    angle = np.linspace(0, 2 * np.pi, 200)
    for ax, step in zip(axes.flat, steps):
        ax.plot(np.cos(angle), np.sin(angle), color="0.8", lw=.6)
        for (preset, phase), group in values[values.checkpoint_step == step].groupby(["preset", "phase"]):
            ax.scatter(group.real, group.imag, s=10, color=COLORS[preset], alpha=.25,
                       marker="x" if phase == "source" else "o", label=f"{LABELS[preset]} ({phase})")
        gt = values[values.checkpoint_step == step].drop_duplicates("mode")
        ax.scatter(gt.gt_real, gt.gt_imag, marker="x", color="black", s=45, label="GT")
        ax.set(aspect="equal", xlim=(-1.08, 1.08), ylim=(-1.08, 1.08), xlabel="Real eigenvalue", ylabel="Imaginary eigenvalue",
               title=f"Samples {step:,}")
        ax.axhline(0, color="0.8", lw=.5)
    handles = [Line2D([], [], color=COLORS[p], marker="o", ls="", label=LABELS[p]) for p in frame.preset.unique()]
    handles.append(Line2D([], [], color="0.4", marker="x", ls="", label="Source stage (where available)"))
    fig.legend(handles=handles, loc="outside lower center", ncol=3, fontsize=7)
    fig.suptitle(f"Supplementary · {frame.environment.iloc[0]} directed complex spectra (all seeds)")
    _save(fig, output, "eigenvalue_spectra")


def _plot_relationships(frame, intervals, output):
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.5), layout="constrained")
    markers = ["o", "s", "^", "D", "v", "P"]
    steps = sorted(frame.checkpoint_step.unique())
    for col, metric in enumerate(METRICS):
        for preset, group in frame.groupby("preset"):
            for step, checkpoint in group.groupby("checkpoint_step"):
                i = steps.index(step)
                axes[0, col].scatter(checkpoint.action_tv_similarity, checkpoint[metric], color=COLORS[preset],
                                     marker=markers[i % len(markers)], s=15, alpha=.5)
            changes = intervals[intervals.preset == preset]
            axes[1, col].scatter(changes.delta_nll_bits_per_symbol, changes[f"delta_{metric}"], color=COLORS[preset], s=15, alpha=.4)
        axes[0, col].set(xlabel="Action-conditioned TV similarity", ylabel=metric.replace("_", " "))
        axes[1, col].set(xlabel="Δ NLL (bits/symbol)", ylabel="Δ " + metric.replace("_", " "))
        axes[1, col].axhline(0, color="0.7", lw=.6)
        axes[1, col].axvline(0, color="0.7", lw=.6)
    fig.suptitle(f"{frame.environment.iloc[0]} · descriptive associations; CSVs stratify checkpoint/interval")
    handles = [Line2D([], [], color=COLORS[p], marker="o", ls="", label=_label(frame, p)) for p in frame.preset.unique()]
    handles += [Line2D([], [], color="0.4", marker=markers[i % len(markers)], ls="", label=f"Samples {step:,}")
                for i, (step, _) in enumerate(frame.groupby("checkpoint_step"))]
    fig.legend(handles=handles, loc="outside lower center", ncol=3, fontsize=7)
    _save(fig, output, "spectral_relationships")


def _plot_pca(frame, gt, variance, output):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), layout="constrained")
    for ax, metric in zip(axes, ["action_tv_similarity", "nll_bits_per_symbol", "unmatched_mass"]):
        norm = Normalize(frame[metric].min(), frame[metric].max())
        for preset, group in frame.groupby("preset"):
            for _, seed in group.groupby("seed"):
                ax.plot(seed.pc1, seed.pc2, color=COLORS[preset], lw=.6, alpha=.15)
            points = ax.scatter(group.pc1, group.pc2, c=group[metric], norm=norm, cmap="viridis", s=17,
                                marker=MARKERS[preset], alpha=.7, label=_label(frame, preset))
            missing = group[group[metric].isna()]
            if len(missing):
                ax.scatter(missing.pc1, missing.pc2, color="0.5", marker=MARKERS[preset], s=20)
        ax.scatter(*gt, color="red", marker="*", s=110, label="GT")
        fig.colorbar(points, ax=ax, label=metric.replace("_", " "), shrink=.75)
        ax.set(xlabel=f"PC1 ({variance[0]:.1%})", ylabel=f"PC2 ({variance[1]:.1%})")
    axes[0].legend(fontsize=6)
    fig.suptitle(f"{frame.environment.iloc[0]} · shared PCA of action-conditioned projected tensors (including sink)")
    _save(fig, output, "transition_pca")


def _write_findings(frame, metadata, output):
    lines = [f"# {frame.environment.iloc[0]} CSCG spectral development", "",
             f"{len(frame)} records; {metadata['n_seeds']} paired seeds per capacity. Protocol: `{metadata['protocol_id']}`.", "",
             "Directed GT evaluation-policy operator, self-loops retained, unmatched mass retained as row deficit. Gaps describe this substochastic operator; they are not estimates of mixing time.", "",
             "| Allocation | Pass | Eigenvalue RMSE (median) | Modulus gap (median) | Subspace error (median) | Defined subspaces | TV similarity (median) | NLL (median) | Unmatched mass (median) |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for (preset, training_pass), group in frame.groupby(["preset", "training_pass"]):
        med = group.median(numeric_only=True)
        lines.append(f"| {_label(frame, preset)} | {training_pass:g} | {med.eigenvalue_rmse:.4g} | {med.modulus_gap:.4g} | {med.subspace_sin_max:.4g} | {int(group.subspace_defined.sum())}/{len(group)} | {med.action_tv_similarity:.4g} | {med.nll_bits_per_symbol:.4g} | {med.unmatched_mass:.4g} |")
    lines += ["", f"Dominant subspace rank: {metadata['subspace_rank']}, including the Perron mode. A missing value marks a tied modulus boundary in either operator. Subspace errors ignore signs and rotations. GT-anchored eigenvalue matching does not establish persistent mode identities across training.", "",
              "`checkpoint_associations.csv` reports Spearman correlations within capacity and checkpoint. `interval_associations.csv` relates paired changes within capacity and interval, including alignment-change fractions. These are descriptive repeated-measurement summaries without pooled significance tests.", "",
              "`interval_changes.csv` records changing alignments. Alignment changes can contribute to apparent spectral motion; these diagnostics do not separate them from transition learning. Unmatched targets are never redistributed among matched states.", "",
              "PCA is fitted once per environment to both learned capacity conditions, with GT excluded. Its basis is saved and its variance fractions are in metadata; it is not the basis from the earlier six-condition PCA.", "",
              "The comparison concerns two clone allocations, including their different distributions across observations. It does not establish a continuous capacity-response curve. Action collapse can conceal action-specific errors even when spectra agree.", "",
              "Figures: [trajectories](spectral_trajectories.png), [complex spectra](eigenvalue_spectra.png), [relationships](spectral_relationships.png), [tensor PCA](transition_pca.png).", ""]
    associations = pd.read_csv(output / "checkpoint_associations.csv")
    changes = pd.read_csv(output / "interval_changes.csv")
    lines += ["## Initial descriptive findings", ""]
    for preset, group in frame.groupby("preset"):
        final = group[group.snapshot_step == group.snapshot_step.max()]
        final_corr = associations[(associations.preset == preset) & (associations.checkpoint_step == final.snapshot_step.iloc[0])
                                  & (associations.x == "eigenvalue_rmse")].set_index("y")
        rho_tv = final_corr.loc["action_tv_similarity", "spearman_rho"]
        rho_nll = final_corr.loc["nll_bits_per_symbol", "spearman_rho"]
        lines.append(f"- {_label(frame, preset)}: final eigenvalue error versus TV similarity has Spearman ρ={rho_tv:.3f}; versus NLL, ρ={rho_nll:.3f} ({len(final)} seeds).")
        for (start, end), interval in changes[changes.preset == preset].groupby(["from_step", "to_step"]):
            spectral = interval.delta_eigenvalue_rmse < 0
            nll = interval.delta_nll_bits_per_symbol < 0
            tv = interval.delta_action_tv_similarity > 0
            changing = int((interval.alignment_change_fraction > 0).sum())
            lines.append(f"  Samples {start}→{end}: eigenvalue error decreases in {int(spectral.sum())}/{len(interval)} seeds; NLL decreases in {int(nll.sum())}/{len(interval)}; TV similarity increases in {int(tv.sum())}/{len(interval)}. Both eigenvalue error and NLL decrease in {int((spectral & nll).sum())}/{len(interval)}. Alignments change in {changing}/{len(interval)}.")
    final = frame[frame.snapshot_step == frame.snapshot_step.max()].groupby("preset").median(numeric_only=True)
    if len(final) == 2:
        lines += ["", f"Final median eigenvalue error is {final.loc['oracle_capacity', 'eigenvalue_rmse']:.4g} at oracle capacity and {final.loc['vanilla_cscg', 'eigenvalue_rmse']:.4g} for standard CSCG. Median subspace error is {final.loc['oracle_capacity', 'subspace_sin_max']:.4g} and {final.loc['vanilla_cscg', 'subspace_sin_max']:.4g}, respectively. Compare these separately with TV and NLL above: closeness of eigenvalues alone does not establish structural agreement.", ""]
    (output / "findings.md").write_text("\n".join(lines))


def _write_method_findings(frame, metadata, output):
    lines = [f"# {frame.environment.iloc[0]} spectral development across model stages", "",
             f"Primary spatial figures: [eigenvector maps, convergence and state distances](../../spatial_spectral_development/{output.name}/README.md). The complex-eigenvalue scatters below are supplementary directed diagnostics.", "",
             f"{len(frame)} records; {metadata['n_seeds']} paired seeds. Protocol `{metadata['protocol_id']}`.", "",
             "MERCURY and its CSCG continuation form one model trajectory; TMGWR and its continuation form another. Each trajectory has a source endpoint and a CSCG handoff at 2,000 total samples, followed by 4,000 and 2,000,000 total samples. The handoff consumes no new training samples. Pure CSCG capacities provide reference trajectories.", "",
             "The source models have only their 2,000-sample checkpoint saved. Their NLL is absent in the evaluation records and remains missing here. The source-stage decoder is frozen WTA, while the CSCG stages use posterior marginal decoding; transfer changes both representation and inference. No source training curve or source NLL is inferred from the hybrid records.", "",
             "| Model | Stage | Total samples | Eigenvalue RMSE (median) | Modulus gap (median) | Subspace error (median) | Defined subspaces | TV similarity (median) | NLL (median) | Unmatched mass (median) |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    phase_order = {"source": 0, "handoff": 1, "hybrid": 2, "cscg": 2}
    groups = list(frame.groupby(["preset", "checkpoint_step", "phase"]))
    groups.sort(key=lambda item: (item[0][0], item[0][1], phase_order[item[0][2]]))
    number = lambda value: f"{value:.4g}" if np.isfinite(value) else "—"
    for (preset, step, phase), group in groups:
        med = group.median(numeric_only=True)
        stats = " | ".join(number(med[k]) for k in ["eigenvalue_rmse", "modulus_gap", "subspace_sin_max"])
        rest = " | ".join(number(med[k]) for k in ["action_tv_similarity", "nll_bits_per_symbol", "unmatched_mass"])
        lines.append(f"| {_label(frame, preset)} | {phase} | {step:,} | {stats} | {int(group.subspace_defined.sum())}/{len(group)} | {rest} |")
    lines += ["", "The directed evaluation-policy operator and GT state order are shared across all methods. Self-loops are retained; unmatched targets remain a probability deficit in the square matched-state block. Gaps of this substochastic operator are not mixing-time estimates. Rank-two right invariant subspaces include the Perron mode. Undefined boundary ties are omitted, and principal angles remove arbitrary basis rotations.", "",
              "`interval_changes.csv` follows each staged model in order, including the source→handoff transition at the same sample count. Alignment IDs are not assumed to survive the representation transfer: alignment-change fractions at that boundary are missing. Within the CSCG continuation they are computed from stable clone IDs. GT-space subspace changes can still be measured across the transfer. Source→handoff spectral changes must not be attributed to additional training.", "",
              "PCA is fitted once per environment across all four model trajectories and all saved stages, including unmatched-target features; GT is excluded from fitting. The source and its continuation share a colour and marker and are connected in sequence. Grey PCA points in the NLL panel have missing source NLL. This basis differs from the earlier two-capacity PCA. In trajectory plots, hollow squares mark source endpoints and hollow diamonds mark the CSCG handoff at the same sample count.", "",
              "## Relationships with TV and NLL", "",
              "Spearman correlations are descriptive and stratified by model, sample count and stage. Paired-change correlations additionally distinguish source→handoff from subsequent learning. No independent-sample significance tests pool repeated checkpoints.", ""]
    associations = pd.read_csv(output / "checkpoint_associations.csv")
    for preset, group in frame.groupby("preset"):
        final_step = group.checkpoint_step.max()
        phase = "hybrid" if preset in ("mercury", "tmgwr") else "cscg"
        selected = associations[(associations.preset == preset) & (associations.checkpoint_step == final_step)
                                & (associations.phase == phase) & (associations.x == "eigenvalue_rmse")].set_index("y")
        n = int(selected.loc['action_tv_similarity', 'n_pairs'])
        lines.append(f"- {LABELS[preset]}, final stage: eigenvalue error versus TV similarity ρ={selected.loc['action_tv_similarity', 'spearman_rho']:.3f}; versus NLL ρ={selected.loc['nll_bits_per_symbol', 'spearman_rho']:.3f} ({n} seeds).")
    lines += ["", "Figures: [trajectories](spectral_trajectories.png), [complex spectra](eigenvalue_spectra.png), [relationships](spectral_relationships.png), [shared tensor PCA](transition_pca.png).", ""]
    (output / "findings.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--levels", type=int, nargs="+", choices=list(ENVIRONMENTS), default=[21, 22])
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--subspace-rank", type=int, default=2)
    parser.add_argument("--all-methods", action="store_true", help="Include MERCURY/TMGWR source stages and their CSCG continuations as staged model trajectories")
    args = parser.parse_args()
    manifest = json.loads((args.dataset_root / "notebook_plots/main_study_eval_v5/evaluation_v5_manifest.json").read_text())
    study = Path(manifest["source_shadow_study"])
    if not study.is_absolute():
        study = ROOT / study
    # A manifest must not silently redirect an explicitly selected dataset.
    if not study.resolve().is_relative_to(args.dataset_root.resolve()):
        parser.error("Manifest study lies outside the selected dataset")
    name = "model_spectral_development" if args.all_methods else "cscg_spectral_development"
    output = args.output_dir or args.dataset_root / "notebook_plots/main_study_eval_v5" / name
    for level in args.levels:
        frame = analyse_environment(study, manifest["protocol_id"], level, output / ENVIRONMENTS[level], args.subspace_rank, all_methods=args.all_methods)
        print(f"{ENVIRONMENTS[level]}: {len(frame)} records; {int(frame.subspace_defined.sum())} defined subspaces; {output / ENVIRONMENTS[level]}")


if __name__ == "__main__":
    main()
