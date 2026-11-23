"""Compare fixed versus oracle clone capacity using frozen canonical records."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.analysis.cscg_distillation_canonical_report import (
    LABELS, METHODS, METRICS, METRIC_LABELS, PRIMARY, canonical_metrics,
    final_f1_curve, residual_plots, summarise,
)

ROOT = Path(__file__).resolve().parents[2]
ALLOCATION_LABELS = {"fixed_10": "10 clones/observation", "ground_truth": "Ground-truth allocation"}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fixed", type=Path, default=ROOT / "results/diagnostics/cscg_distillation_coupled5_v1")
    p.add_argument("--ground-truth", type=Path, default=ROOT / "results/diagnostics/cscg_distillation_coupled5_ground_truth_v1")
    p.add_argument("--ground-truth-only", action="store_true", help="Report a ground-truth-only seed cohort")
    args = p.parse_args()
    roots = ({"ground_truth": args.ground_truth} if args.ground_truth_only else
             {"fixed_10": args.fixed, "ground_truth": args.ground_truth})
    configs = {name: json.loads((root / "experiment.json").read_text()) for name, root in roots.items()}
    implementations = {name: json.loads((root / "implementation.json").read_text()) for name, root in roots.items()}
    if not args.ground_truth_only and implementations["fixed_10"]["algorithm_sha256"] != implementations["ground_truth"]["algorithm_sha256"]:
        raise ValueError("Allocation experiments use different learning implementations")
    for key in ["experiment", "levels", "seeds", "observations", "checkpoints", "rand_prob", "evaluation_walks",
                "walk_length", "evaluation_seed", "methods", "local_parameter_updates"]:
        if not args.ground_truth_only and configs["fixed_10"][key] != configs["ground_truth"][key]:
            raise ValueError(f"Allocation comparison has a mismatched setting: {key}")
    if configs["ground_truth"].get("clone_allocation") != "ground_truth":
        raise ValueError("Expected ground-truth clone allocation")
    output = args.ground_truth / "canonical_metrics"
    output.mkdir(parents=True, exist_ok=True)
    curves, summaries, thresholds, final_records, old_provenance, hashes = [], [], [], {}, {}, {}
    clone_allocations = {}
    keys_by_allocation = {}
    for condition, (allocation, root) in enumerate(roots.items()):
        rows, keys = [], set()
        for path in sorted(root.glob("level=*/seed=*/*/rows.json")):
            saved_rows = json.loads(path.read_text())
            identity = tuple(saved_rows[0][key] for key in ["level", "seed", "method"])
            provenance = json.loads((path.parent / "training.json").read_text())["provenance"]
            capacity_key = (allocation, identity[0])
            capacity = provenance["clone_allocation"]
            if capacity_key in clone_allocations and clone_allocations[capacity_key] != capacity:
                raise ValueError("Clone allocation varies within an environment/condition")
            clone_allocations[capacity_key] = capacity
            comparable = {k: v for k, v in provenance.items() if k not in {"clone_allocation", "initial_T_sha256"}}
            if allocation == "fixed_10":
                old_provenance[identity] = comparable
            elif not args.ground_truth_only and comparable != old_provenance[identity]:
                raise ValueError(f"Training input or canonical evaluation differs: {identity}")
            for original in saved_rows:
                record_path = path.parent / "evaluations" / f"step={original['step']:04d}.json"
                record = json.loads(record_path.read_text())
                keys.add((*identity, original["step"]))
                scores = canonical_metrics(record)
                rows.append(dict(iterations=5, allocation=allocation, level=record["level"], seed=original["seed"],
                                 method=record["method"], step=original["step"], train_seconds=original["train_seconds"],
                                 clone_count=sum(provenance["clone_allocation"]), **scores))
                hashes[str(record_path)] = hashlib.sha256(record_path.read_bytes()).hexdigest()
                if original["step"] == configs[allocation]["observations"]:
                    final_records[(condition, *identity)] = record
                    for score in final_f1_curve(record):
                        thresholds.append(dict(allocation=allocation, level=record["level"], seed=original["seed"],
                                               method=record["method"], step=original["step"], **score))
        df = pd.DataFrame(rows)
        summaries.append(summarise(df).assign(allocation=allocation))
        curves.append(df); keys_by_allocation[allocation] = keys
    expected = len(configs["ground_truth"]["levels"]) * len(configs["ground_truth"]["seeds"]) * len(METHODS) * len(configs["ground_truth"]["checkpoints"])
    if (not args.ground_truth_only and keys_by_allocation["fixed_10"] != keys_by_allocation["ground_truth"]) or len(keys_by_allocation["ground_truth"]) != expected:
        raise ValueError("Incomplete paired allocation comparison")
    for path in ([] if args.ground_truth_only else (args.fixed / "evaluation_caches").glob("*/walks/*.npz")):
        with np.load(path) as old, np.load(args.ground_truth / path.relative_to(args.fixed)) as new:
            if set(old.files) != set(new.files) or any(not np.array_equal(old[key], new[key]) for key in old.files):
                raise ValueError("Held-out walks differ between allocations")
    curves = pd.concat(curves, ignore_index=True)
    summaries = pd.concat(summaries, ignore_index=True)
    curves.to_csv(output / "learning_curves.csv", index=False)
    summaries.to_csv(output / "per_seed_summary.csv", index=False)
    means = summaries.groupby(["allocation", "level", "method"]).mean(numeric_only=True)
    means.to_csv(output / "summary.csv")
    differences = []
    numeric = [f"{prefix}_{metric}" for prefix in ["final", "time_mean"] for metric in METRICS] + ["train_seconds"]
    indexed = summaries.set_index(["allocation", "level", "seed", "method"], drop=False)
    def pair(label, left, right):
        for metric in numeric:
            differences.append(dict(comparison=label, allocation=left.allocation, reference_allocation=right.allocation,
                                    level=int(left.level), seed=int(left.seed), method=left.method, reference=right.method,
                                    metric=metric, value=float(left[metric]), reference_value=float(right[metric]),
                                    delta=float(left[metric] - right[metric])))
    if not args.ground_truth_only:
        for _, row in summaries[summaries.allocation == "ground_truth"].iterrows():
            pair("ground_truth_minus_fixed", row, indexed.loc[("fixed_10", row.level, row.seed, row.method)])
    for (allocation, level, seed), part in summaries.groupby(["allocation", "level", "seed"]):
        part = part.set_index("method", drop=False)
        for method in ["predictive", "posterior", "nll_control"]:
            pair("method_minus_long", part.loc[method], part.loc["long"])
        pair("predictive_minus_nll", part.loc["predictive"], part.loc["nll_control"])
    paired = pd.DataFrame(differences)
    paired.to_csv(output / "per_seed_paired_differences.csv", index=False)
    lower_is_better = paired.metric.str.endswith(("eval_nll_bps", "unmatched_mass")) | (paired.metric == "train_seconds")
    paired["benefit_delta"] = np.where(lower_is_better, -paired.delta, paired.delta)
    aggregate = paired.groupby(["comparison", "allocation", "level", "method", "reference", "metric"]).agg(
        mean=("delta", "mean"), std=("delta", "std"), count=("delta", "count"),
        wins=("benefit_delta", lambda x: int((x > 1e-12).sum())),
        losses=("benefit_delta", lambda x: int((x < -1e-12).sum())),
        ties=("benefit_delta", lambda x: int((x.abs() <= 1e-12).sum())))
    aggregate.to_csv(output / "paired_summary.csv")

    import matplotlib.pyplot as plt
    levels = sorted(curves.level.unique())
    diagnostic_labels = dict(state_coverage="Physical-state coverage", unmatched_mass="Unmatched mass",
                             eval_nll_bps="Held-out NLL (bits/observation)")
    for filename, panels in [("learning_curves", PRIMARY), ("purity_and_recall", ("latent_precision", "latent_recall")),
                             ("coverage_unmatched_nll", tuple(diagnostic_labels))]:
        fig, axes = plt.subplots(len(levels), len(panels), figsize=(4.5*len(panels), 4*len(levels)), squeeze=False)
        for i, level in enumerate(levels):
            for j, metric in enumerate(panels):
                ax = axes[i, j]
                for method in METHODS:
                    for allocation, style in [(key, "-" if key == "ground_truth" else "--") for key in roots]:
                        part = curves[(curves.level == level) & (curves.method == method) & (curves.allocation == allocation)]
                        agg = part.groupby("step")[metric].agg(["mean", "std"])
                        color = f"C{METHODS.index(method)}"
                        ax.plot(agg.index, agg["mean"], color=color, linestyle=style,
                                label=(LABELS[method] if args.ground_truth_only else f"{LABELS[method]}: {ALLOCATION_LABELS[allocation]}"))
                        sd = agg["std"].fillna(0)
                        ax.fill_between(agg.index, agg["mean"]-sd, agg["mean"]+sd, color=color, alpha=.08)
                ax.set(title=f"Level {level}: {METRIC_LABELS.get(metric, diagnostic_labels.get(metric))}", xlabel="Observations")
                if metric != "eval_nll_bps":
                    ax.set_ylim(0, 1)
                ax.grid(alpha=.2)
        axes[0, 0].legend(fontsize=6)
        fig.tight_layout()
        fig.savefig(output / f"{filename}.png", dpi=160); fig.savefig(output / f"{filename}.pdf")
        plt.close(fig)
    fig, axes = plt.subplots(len(levels), len(PRIMARY), figsize=(18, 4*len(levels)), squeeze=False)
    for i, level in enumerate(levels):
        for j, metric in enumerate(PRIMARY):
            ax = axes[i, j]
            for method, offset in [("predictive", -.1), ("posterior", .1)]:
                part = paired[(paired.comparison == "method_minus_long") & (paired.allocation == "ground_truth")
                              & (paired.level == level) & (paired.method == method) & (paired.metric == "final_"+metric)]
                ax.scatter(part.seed + offset, part.delta, label=LABELS[method], color=f"C{METHODS.index(method)}")
            ax.axhline(0, color="grey", linewidth=.8)
            ax.set(title=f"Level {level}: {METRIC_LABELS[metric]}", xlabel="Paired seed", ylabel="KL minus long EM")
            ax.set_xticks(configs["ground_truth"]["seeds"])
            ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "paired_seed_differences.png", dpi=160)
    fig.savefig(output / "paired_seed_differences.pdf")
    plt.close(fig)
    threshold_df = pd.DataFrame(thresholds)
    threshold_df.to_csv(output / "final_step_f1_by_seed.csv", index=False)
    fig, axes = plt.subplots(len(levels), 2, figsize=(12, 4*len(levels)), squeeze=False)
    for i, level in enumerate(levels):
        for j, include_self in enumerate([True, False]):
            ax = axes[i, j]
            for method in METHODS:
                for allocation, style in [(key, "-" if key == "ground_truth" else "--") for key in roots]:
                    part = threshold_df[(threshold_df.level == level) & (threshold_df.method == method)
                                        & (threshold_df.allocation == allocation) & (threshold_df.include_self_loops == include_self)]
                    agg = part.groupby("threshold").edge_f1.agg(["mean", "std"])
                    color = f"C{METHODS.index(method)}"
                    ax.plot(agg.index, agg["mean"], color=color, linestyle=style,
                            label=(LABELS[method] if args.ground_truth_only else f"{LABELS[method]}: {ALLOCATION_LABELS[allocation]}"))
                    sd = agg["std"].fillna(0)
                    ax.fill_between(agg.index, agg["mean"]-sd, agg["mean"]+sd, color=color, alpha=.08)
            ax.set(title=f"Level {level}: action-conditioned F1, {'with' if include_self else 'without'} self-loops",
                   xlabel="Probability threshold", ylabel=f"Edge F1 at observation {configs['ground_truth']['observations']:,}", ylim=(0, 1))
            ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(output / "final_step_action_conditioned_edge_f1.png", dpi=160)
    fig.savefig(output / "final_step_action_conditioned_edge_f1.pdf")
    plt.close(fig)
    residual_plots(final_records, output, conditions={i: key for i, key in enumerate(roots)})

    lines = ["# Ground-truth clone allocation: five coupled E/M iterations", "",
             f"Ground-truth allocation uses {len(configs['ground_truth']['seeds'])} seeds per environment and {expected // len(configs['ground_truth']['checkpoints'])} completed runs.",
             f"Horizon: {configs['ground_truth']['observations']:,} observations; {len(configs['ground_truth']['checkpoints'])} checkpoints including the initial tensor.",
             "One clone is allocated per physical state, grouped by its observation symbol.",
             "No oracle transitions, latent identities or trajectory labels are supplied to learning. Initial tensors are random with paired seeds; dimensions necessarily differ across allocations.",
             "All arms use the same trajectories, five coupled iterations, cumulative counts, teacher half-life, windows, auxiliary weights, per-edge pseudocount and held-out canonical evaluation.",
             "Transition similarity, action-conditioned edge F1, purity and recall use the exact reference extractors. Higher structural scores are better.", "",
             "## Clone allocations", "",
             "| Allocation | Level | Total clones | Clones per observation symbol |",
             "| --- | --- | ---: | --- |"]
    for (allocation, level), capacity in sorted(clone_allocations.items()):
        lines.append(f"| {allocation} | {level} | {sum(capacity)} | {capacity} |")
    lines += ["", "## Canonical final metrics", "",
             "| Allocation | Level | Model | Similarity | No-self similarity | Edge F1 threshold AUC | No-self edge F1 threshold AUC | Purity | Recall | NLL bits | Train seconds |",
             "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for (allocation, level, method), row in means.iterrows():
        metrics = (*PRIMARY, "latent_precision", "latent_recall", "eval_nll_bps")
        values = " | ".join(f"{row['final_'+metric]:.4f}" for metric in metrics)
        lines.append(f"| {allocation} | {level} | {LABELS[method]} | {values} | {row.train_seconds:.2f} |")
    lines += ["", "## Coverage and unmatched-mass diagnostics", "",
              "Physical-state coverage is distinct from canonical clone recall above; unmatched mass is the canonical macro average.", "",
              "| Allocation | Level | Model | Physical-state coverage | Unmatched mass | Clone utilisation |",
              "| --- | --- | --- | ---: | ---: | ---: |"]
    for (allocation, level, method), row in means.iterrows():
        lines.append(f"| {allocation} | {level} | {LABELS[method]} | {row.final_state_coverage:.4f} | {row.final_unmatched_mass:.4f} | {row.final_clone_utilisation:.4f} |")
    lines += ["", "## Integrated metrics (temporal means)", "",
              "Trapezoidal integration uses the actual checkpoint observation positions and divides by the run horizon. F1 threshold AUC and integration over training observations are separate operations.", "",
              "| Allocation | Level | Model | Similarity | No-self similarity | Edge F1 threshold AUC | No-self edge F1 threshold AUC | Purity | Recall | Coverage | Unmatched mass | NLL bits |",
              "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for (allocation, level, method), row in means.iterrows():
        metrics = (*PRIMARY, "latent_precision", "latent_recall", "state_coverage", "unmatched_mass", "eval_nll_bps")
        values = " | ".join(f"{row['time_mean_'+metric]:.4f}" for metric in metrics)
        lines.append(f"| {allocation} | {level} | {LABELS[method]} | {values} |")
    lines += ["", "## Paired final differences against long EM", "",
              "Positive structural differences favour the model; wins/losses count paired seeds, not checkpoints.", "",
              "| Level | Model | Metric | Mean difference | Seed SD | Wins | Losses | Ties |",
              "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for key, row in aggregate.iterrows():
        comparison, allocation, level, method, reference, metric = key
        if comparison == "method_minus_long" and allocation == "ground_truth" and metric in ["final_"+m for m in PRIMARY]:
            lines.append(f"| {level} | {LABELS[method]} | {metric} | {row['mean']:+.4f} | {row['std']:.4f} | {int(row.wins)} | {int(row.losses)} | {int(row.ties)} |")
    lines += ["", "Per-seed final and temporal-mean metrics and paired differences are in the CSVs.",
              "final_step_action_conditioned_edge_f1.png shows actual final-checkpoint F1 versus threshold; purity_and_recall.png shows canonical representation metrics.",
              "coverage_unmatched_nll.png shows physical-state coverage, unmatched mass and held-out NLL throughout training.",
              "paired_seed_differences.png shows every KL-minus-long-EM seed difference on the four primary canonical scores; points above zero favour the KL model.",
              "Residual plots use the reference renderer and include separate per-action matrices with unmatched mass.",
              "The per-entry pseudocount is held fixed, so oracle capacity also reduces total pseudocount mass; this is part of the capacity intervention, not an isolated change in state count at fixed total regularisation.",
              "The decoder remains carried causal filtering. Metric definitions match the reference study; the walk/decoder protocols are not pooled across studies.",
              "Auxiliary M-steps are validated generalized-EM descent steps, not exact marginal argmin solutions. Seed-level differences and variability accompany all means.",
              "Curves show the seed mean and one seed standard deviation. The historical three-seed fixed-allocation cohort is not pooled into an unmatched larger cohort."]
    (output / "results.md").write_text("\n".join(lines)+"\n")
    provenance = dict(fixed=(None if args.ground_truth_only else str(args.fixed)), ground_truth=str(args.ground_truth),
                      ground_truth_only=args.ground_truth_only, seeds=configs["ground_truth"]["seeds"],
                      training_and_evaluation_matched=True, held_out_walk_arrays_equal=(None if args.ground_truth_only else True), records=len(curves),
                      training_implementations=implementations,
                      metric_implementation_sha256=hashlib.sha256((ROOT / "scripts/analysis/plot_config_metric_trajectories.py").read_bytes()).hexdigest(),
                      canonical_report_helpers_sha256=hashlib.sha256((ROOT / "scripts/analysis/cscg_distillation_canonical_report.py").read_bytes()).hexdigest(),
                      implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      evaluation_record_sha256=hashes)
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2)+"\n")
    print(f"Allocation comparison: {output / 'results.md'}")


if __name__ == "__main__":
    main()
