"""Run the EGM research pilot without changing production runs or studies.

PYTHONPATH=.:src .venv/bin/python scripts/run_cscg_distillation_experiment.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from mercury_runs.algorithms.cscg.distillation import (
    METHODS, STUDENT_INFERENCE, CarriedFilter, ExperimentConfig, row_normalize, train_experiment,
)
from mercury_runs.algorithms.cscg.prepare import action_to_indices, map_to_indices, to_indices
from mercury_runs.application.evaluation_protocol import default_canonical_v4_protocol
from mercury_runs.application.evaluation_service import (
    build_empirical_ground_truth, generate_and_cache_walks, serialise_evaluation_result,
)
from mercury_runs.infrastructure.paper_precision import (
    compute_fixed_map_precision_details, compute_purity_from_cooccurrence, compute_transition_statistics,
    evaluate_transition_structure_canonical, exact_cartesian_reference_positions,
    exact_valid_sensorimotor_transitions, infer_num_actions_from_valid_transitions,
)

ROOT = Path(__file__).resolve().parents[1]
LABELS = {"short": "Short EM", "long": "Long EM", "predictive": "EMA predictive KL",
          "posterior": "EMA posterior KL", "nll_control": "Replay NLL control"}
SUMMARY_METRICS = ("action_tv", "nonself_tv", "nll_bits", "purity", "edge_f1",
                   "state_coverage", "unmatched_mass", "clone_utilisation", "node_recall")


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def load_training(level, seed, rand_prob, steps):
    import polars as pl
    directory = ROOT / f"datasets/level={level}/sensor=cardinal distance/range=1"
    candidates = sorted(directory.glob(f"*_seed{seed}_p{rand_prob}_steps{max(2000, steps)}_allsteps_randomstart_*.parquet"))
    if len(candidates) != 1:
        raise ValueError(f"Expected one existing paired trajectory: {candidates}")
    path = candidates[0]
    frame = pl.read_parquet(path).head(steps)
    if len(frame) != steps:
        raise ValueError(f"Insufficient training observations: {path}")
    obs = frame.select([c for c in frame.columns if c.startswith("observation_")]).to_numpy()
    act = frame.select([c for c in frame.columns if c.startswith("action_")]).to_numpy()
    x, vocabulary = to_indices(obs)
    incoming, action_vocabulary = action_to_indices(act)
    return dict(x=x, actions=incoming[1:], vocabulary=vocabulary, action_vocabulary=action_vocabulary,
                path=str(path.relative_to(ROOT)), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                collisions=int(frame["collision"].sum()))


def ground_truth_clone_allocation(level, vocabulary):
    """Oracle capacity only: one clone per physical state sharing a symbol.

    Enumerate the full environment so capacity is independent of which states a
    particular training walk visited. No transition/clone identity is transferred.
    """
    from mercury_runs.algorithms.cscg.structural_init import ground_truth_capacity
    from mercury_runs.infrastructure.paper_precision import _init_env, _set_start_position
    positions = exact_cartesian_reference_positions(level)
    environment = _init_env(level_index=level, sensor="cardinal distance", sensor_range=1, seed=0)
    observations = []
    for position in positions:
        _set_start_position(environment, tuple(map(int, position)))
        observations.append(np.asarray(environment.agent.observation).copy())
    obs_idx = map_to_indices(vocabulary, np.asarray(observations))
    allocation = ground_truth_capacity(n_obs=len(vocabulary), obs_idx=obs_idx,
                                       ground_truth_states=np.arange(len(positions)), clones_per_state=1)
    if int(allocation.sum()) != len(positions):
        raise ValueError("Oracle allocation does not cover the full environment")
    return allocation


def indexed_evaluation(walks, vocabulary, action_vocabulary):
    indexed = []
    for x, actions, truth, gt_actions in zip(walks["filtered_obs_per_walk"], walks["filtered_actions_per_walk"],
                                            walks["gt_exact_per_walk"], walks["gt_action_per_walk"]):
        if x is None:
            raise ValueError("Empty held-out evaluation walk")
        # Unknown vocabulary is a failed comparison, never silently dropped.
        indexed.append((map_to_indices(vocabulary, x), map_to_indices(action_vocabulary, actions), truth, gt_actions))
    return indexed


def evaluate_snapshot(T, n_clones, indexed, gt, protocol, level, method, step, destination):
    decoded, truths, action_walks, losses = [], [], [], []
    for x, incoming, truth, gt_actions in indexed:
        filt = CarriedFilter(n_clones)
        # Exercise the same carried implementation across arbitrary I/O chunks.
        parts = [filt.batch(T, x[s:s + 40], incoming[s:s + 40]) for s in range(0, len(x), 40)]
        beliefs = np.concatenate([part[1] for part in parts])
        decoded.append(beliefs.argmax(axis=-1)); truths.append(truth); action_walks.append(gt_actions)
        losses.append(np.concatenate([part[2] for part in parts]))
    C, H, N = compute_transition_statistics(
        decoded_walks=decoded, true_walks=truths, action_walks=action_walks,
        n_learned=T.shape[1], n_true=gt["n_true"], n_actions=T.shape[0])
    result = evaluate_transition_structure_canonical(
        T_hat=T, T_true=gt["T_GT"], cooccurrence=C, action_source_cooccurrence=H,
        source_action_counts=N, edge_thresholds=protocol.edge_thresholds)
    purity = compute_purity_from_cooccurrence(C) / 100
    used = C.sum(axis=1) > 0
    coverage = len(np.unique(result.alignment[used])) / gt["n_true"]
    details = compute_fixed_map_precision_details(
        inferred_states=np.concatenate(decoded), ground_truth_states=np.concatenate(truths),
        metric_name="latent_precision", inferred_state_count=T.shape[1], ground_truth_state_count=gt["n_true"])
    recall = details["latent_recall"]
    nll = float(np.concatenate(losses).mean() / np.log(2))
    record = serialise_evaluation_result(
        protocol=protocol, level=level, method=method, step=step, observed_samples=step,
        walk_cache_id=protocol.walk_cache_key(), gt=gt, result=result,
        inference_meta=dict(decoder="causal_filter_argmax", batch_size=40, carried_belief=True),
        latent_precision=purity, node_recall=recall, n_learned=T.shape[1], n_true=gt["n_true"],
        n_actions=T.shape[0], model_fit=dict(bits_per_symbol=nll, initial_observation_included=True))
    write_json(destination, record)
    return dict(action_tv=result.tv_occupancy, nonself_tv=result.tv_non_self_loop_occupancy,
                action_tv_macro=result.tv_macro, nonself_tv_macro=result.tv_non_self_loop_macro,
                purity=purity, clone_utilisation=float(used.mean()), node_recall=recall, state_coverage=coverage,
                edge_f1=result.primary_metrics()["action_probability_edge_f1"], nll_bits=nll,
                unmatched_mass=result.total_unmatched_target_mass)


def summarise_rows(df):
    per_seed = []
    for (level, seed, method), part in df.groupby(["level", "seed", "method"]):
        part = part.sort_values("step")
        if part.step.duplicated().any() or part.step.iloc[0] != 0 or part.step.iloc[-1] <= 0:
            raise ValueError("Each learning curve requires unique checkpoints from zero to a positive endpoint")
        final = part.iloc[-1]
        value = dict(level=int(level), seed=int(seed), method=method,
                     train_seconds=float(final.train_seconds), final_step=int(final.step),
                     work_entries=int(final.work_entries))
        for metric in SUMMARY_METRICS:
            value["final_" + metric] = float(final[metric])
            value["auc_" + metric] = float(np.trapezoid(part[metric], part.step) / final.step)
        per_seed.append(value)
    return pd.DataFrame(per_seed)


def report(output, rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = pd.DataFrame(rows)
    df.to_csv(output / "learning_curves.csv", index=False)
    metrics = ["action_tv", "nonself_tv", "nll_bits", "purity"]
    levels = sorted(df.level.unique())
    fig, axes = plt.subplots(len(levels), len(metrics), figsize=(17, 4 * len(levels)), squeeze=False)
    for row, level in enumerate(levels):
        for col, metric in enumerate(metrics):
            ax = axes[row, col]
            for method in METHODS:
                subset = df[(df.level == level) & (df.method == method)]
                agg = subset.groupby("step")[metric].agg(["mean", "std"])
                if agg.empty:
                    continue
                line, = ax.plot(agg.index, agg["mean"], label=LABELS[method])
                sd = agg["std"].fillna(0)
                ax.fill_between(agg.index, agg["mean"] - sd, agg["mean"] + sd, color=line.get_color(), alpha=.13)
            ax.set(title=f"Level {level}: {metric}", xlabel="Observations")
            if metric != "nll_bits":
                ax.set_ylim(0, 1)
            ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "learning_curves.png", dpi=160)
    fig.savefig(output / "learning_curves.pdf")
    plt.close(fig)
    fig, axes = plt.subplots(len(levels), 2, figsize=(11, 4 * len(levels)), squeeze=False)
    for i, level in enumerate(levels):
        for j, metric in enumerate(["action_tv", "nonself_tv"]):
            for method in METHODS:
                part = df[(df.level == level) & (df.method == method)]
                aggregate = part.groupby("step")[[metric, "work_entries"]].mean()
                axes[i, j].plot(aggregate.work_entries, aggregate[metric], label=LABELS[method])
            axes[i, j].set(title=f"Level {level}: {metric}", xlabel="Counted tensor/block entries", ylim=(0, 1))
            axes[i, j].grid(alpha=.2)
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(output / "work_curves.png", dpi=160); plt.close(fig)

    seeds = summarise_rows(df)
    seeds.to_csv(output / "per_seed_summary.csv", index=False)
    summary = seeds.groupby(["level", "method"]).mean(numeric_only=True)
    summary.to_csv(output / "summary.csv")
    paired = []
    for level in levels:
        subset = seeds[seeds.level == level].set_index(["seed", "method"])
        present = set(df[df.level == level].method)
        for baseline in [b for b in ("short", "long", "nll_control") if b in present]:
            for method in [m for m in ("predictive", "posterior") if m in present]:
                for metric in ["final_action_tv", "final_nonself_tv", "auc_action_tv", "auc_nonself_tv"]:
                    delta = subset.xs(method, level="method")[metric] - subset.xs(baseline, level="method")[metric]
                    paired.append(dict(level=int(level), method=method, baseline=baseline, metric=metric,
                                       mean_delta=float(delta.mean()), sd_delta=float(delta.std(ddof=0)),
                                       paired_seeds=len(delta), seed_deltas={str(k): float(v) for k, v in delta.items()}))
    write_json(output / "paired_differences.json", paired)
    # Equal consumed-work comparison: interpolate each method's saved curve at
    # the smallest final work among methods in a seed. Exposure then differs.
    work_rows = []
    for (level, seed), subset in df.groupby(["level", "seed"]):
        budget = subset.groupby("method").work_entries.max().min()
        for method, part in subset.groupby("method"):
            part = part.sort_values("work_entries")
            work_rows.append(dict(level=int(level), seed=int(seed), method=method, budget=int(budget),
                                  observations=float(np.interp(budget, part.work_entries, part.step)),
                                  action_tv=float(np.interp(budget, part.work_entries, part.action_tv)),
                                  nonself_tv=float(np.interp(budget, part.work_entries, part.nonself_tv))))
    pd.DataFrame(work_rows).to_csv(output / "equal_consumed_work.csv", index=False)
    manifest = json.loads((output / "experiment.json").read_text())
    updates = manifest["local_parameter_updates"]
    coupled = manifest["experiment"].get("coupled_auxiliary", False)
    steps = int(df.step.max())
    lines = ["# CSCG self-distillation pilot results", "",
             "Pilot evidence for EGM brainstorming; no confirmatory statistical claim.", "",
             f"All five arms receive identical {steps:,}-observation streams, clone allocations, initial tensors,",
             f"forgetting factors and {updates} parameter-update opportunities per transition. No teacher is fitted by EM.",
             (f"All arms take {updates} coupled E/M iterations; every KL/NLL M-step includes its auxiliary objective."
              if coupled else f"Baselines take {updates} EM passes; KL/NLL arms take {updates - 1} EM passes and one penalty update."),
             "Auxiliary M-steps use EM-like majorization or an analytical-gradient generalized M-step, not an exact marginal argmin.",
             "Running statistics are committed once per arrival; teacher targets stay fixed and EMA runs once afterwards.",
             "Actual compute differs; entry counts and training time are recorded. Shading is one seed standard deviation.", "",
             "Predictive versus replay-NLL control isolates the added posterior-KL term at equal NLL coefficient.",
             "The posterior arm uses one preconditioned analytic-gradient step with at most eight loss-only backtracks.",
             "Consequently differences between the two KL arms also include optimiser differences.", "",
             "| Level | Method | Final action TV | Final nonself TV | Mean action TV over observations | Training seconds |",
             "| --- | --- | ---: | ---: | ---: | ---: |"]
    for (level, method), r in summary.iterrows():
        lines.append(f"| {level} | {LABELS[method]} | {r.final_action_tv:.4f} | {r.final_nonself_tv:.4f} | {r.auc_action_tv:.4f} | {r.train_seconds:.2f} |")
    lines += ["", "| Level | Method | Held-out NLL (bits) | Purity | Edge F1 |",
              "| --- | --- | ---: | ---: | ---: |"]
    for (level, method), r in summary.iterrows():
        lines.append(f"| {level} | {LABELS[method]} | {r.final_nll_bits:.4f} | {r.final_purity:.4f} | {r.final_edge_f1:.4f} |")
    lines += ["", "## Alignment and coverage diagnostics", "",
              "Canonical projection sends mass targeting unassigned clones to the unmatched sink.",
              "A decrease in TV can therefore reflect improved clone assignment and coverage as well as changed transitions.",
              "The following unmatched-mass statistic is macro averaged over valid action/source rows; primary TV is occupancy weighted.", "",
              "| Level | Method | Used clone fraction | Macro unmatched mass | Ground-truth state coverage |",
              "| --- | --- | ---: | ---: | ---: |"]
    for (level, method), part in df[df.step == df.step.max()].groupby(["level", "method"]):
        r = part.mean(numeric_only=True)
        lines.append(f"| {level} | {LABELS[method]} | {r.clone_utilisation:.4f} | {r.unmatched_mass:.4f} | {r.state_coverage:.4f} |")
    lines += ["", "Lower TV and integrated mean TV are better. Held-out NLL includes the first observation of each walk.",
              "See paired_differences.json for seed-wise distilled-minus-control differences; negative favours distillation.",
              "The work-axis comparison uses counted entries as a proxy, not FLOPs. Equal-consumed-work interpolation",
              "uses different observation counts and is not a simultaneously exposure-matched comparison.", "",
              "All primary results use causal carried-filter decoding and canonical v5 structural projection,",
              "including unmatched mass and conditional nonself movement support. Protocol identity records this decoder.",
              "The available trajectory randomness changes action persistence; no sensor-corruption experiment was run.",
              "The brief describes the objective, count recursion, pilot scope, and remaining attribution limitations."]
    (output / "results.md").write_text("\n".join(lines) + "\n")


def compare_local_updates(output, baseline):
    """Read-only comparison with the completed two-update artifacts."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    roots = {2: baseline, 5: output}
    manifests = {k: json.loads((root / "experiment.json").read_text()) for k, root in roots.items()}
    if [manifests[k]["local_parameter_updates"] for k in [2, 5]] != [2, 5]:
        raise ValueError("Comparison requires a two-update reference and a five-update experiment")
    for key in ["levels", "seeds", "observations", "clones_per_observation", "checkpoints", "rand_prob",
                "evaluation_walks", "walk_length", "evaluation_seed", "methods", "penalty", "inference", "vocabulary"]:
        if manifests[2][key] != manifests[5][key]:
            raise ValueError(f"Incompatible comparison setting: {key}")
    for key, value in manifests[2]["experiment"].items():
        if key not in {"local_updates", "coupled_auxiliary"} and value != manifests[5]["experiment"][key]:
            raise ValueError(f"Incompatible hyperparameter: {key}")
    frames, summary_frames, provenance = [], [], {}
    for updates, root in roots.items():
        rows = []
        for path in sorted(root.glob("level=*/seed=*/*/rows.json")):
            part = json.loads(path.read_text())
            key = (part[0]["level"], part[0]["seed"], part[0]["method"])
            run_provenance = json.loads((path.parent / "training.json").read_text())["provenance"]
            with np.load(path.parent / "snapshots.npz") as saved:
                initial_digest = hashlib.sha256(saved["T"][0].tobytes()).hexdigest()
            if initial_digest != run_provenance["initial_T_sha256"]:
                raise ValueError(f"Initial snapshot provenance mismatch: {key}")
            if updates == 2:
                provenance[key] = run_provenance
            elif key not in provenance or provenance[key] != run_provenance:
                raise ValueError(f"Unmatched trajectory, initial tensor or evaluation protocol: {key}")
            rows.extend(part)
        df = pd.DataFrame(rows)
        expected_runs = len(manifests[updates]["levels"]) * len(manifests[updates]["seeds"]) * len(METHODS)
        summary = summarise_rows(df)
        if len(summary) != expected_runs:
            raise ValueError("Comparison requires every declared paired run")
        df["local_updates"] = updates
        summary["local_updates"] = updates
        frames.append(df); summary_frames.append(summary)
    # Verify the actual held-out walk arrays, not only the protocol identifier.
    for old_cache in (baseline / "evaluation_caches").glob("*/walks/*.npz"):
        new_cache = output / old_cache.relative_to(baseline)
        with np.load(old_cache) as old, np.load(new_cache) as new:
            if set(old.files) != set(new.files) or any(not np.array_equal(old[k], new[k]) for k in old.files):
                raise ValueError("Held-out walk arrays differ between update budgets")
    curves = pd.concat(frames, ignore_index=True)
    summaries = pd.concat(summary_frames, ignore_index=True)
    destination = output / "comparison_2_vs_5"
    destination.mkdir(exist_ok=True)
    curves.to_csv(destination / "learning_curves.csv", index=False)
    summaries.to_csv(destination / "per_seed_summary.csv", index=False)
    paired = []
    metrics = [f"{prefix}_{metric}" for prefix in ["final", "auc"] for metric in SUMMARY_METRICS]
    metrics += ["train_seconds", "work_entries"]
    def add_pair(comparison, updates, left, right, method, reference):
        for metric in metrics:
            paired.append(dict(comparison=comparison, local_updates=updates, level=int(left.level), seed=int(left.seed),
                               reference_updates=int(right.local_updates),
                               method=method, reference=reference, metric=metric, value=float(left[metric]),
                               reference_value=float(right[metric]), delta=float(left[metric] - right[metric])))
    old = summaries[summaries.local_updates == 2].set_index(["level", "seed", "method"], drop=False)
    for _, row in summaries[summaries.local_updates == 5].iterrows():
        reference = old.loc[(row.level, row.seed, row.method)]
        add_pair("five_minus_two", 5, row, reference, row.method, row.method)
    for (updates, level, seed), part in summaries.groupby(["local_updates", "level", "seed"]):
        by_method = part.set_index("method", drop=False)
        for method in ["predictive", "posterior", "nll_control"]:
            add_pair("method_minus_long", int(updates), by_method.loc[method], by_method.loc["long"], method, "long")
        add_pair("predictive_minus_nll", int(updates), by_method.loc["predictive"], by_method.loc["nll_control"],
                 "predictive", "nll_control")
    long_five = summaries[(summaries.local_updates == 5) & (summaries.method == "long")].set_index(["level", "seed"])
    for _, row in summaries[(summaries.local_updates == 2) & (summaries.method == "predictive")].iterrows():
        add_pair("predictive_two_minus_long_five", 2, row, long_five.loc[(row.level, row.seed)], "predictive", "long")
    differences = pd.DataFrame(paired)
    differences.to_csv(destination / "per_seed_paired_differences.csv", index=False)
    grouped = differences.groupby(["comparison", "local_updates", "level", "method", "reference", "metric"])
    aggregates = grouped.delta.agg(["mean", "std", "count"])
    aggregates.to_csv(destination / "paired_summary.csv")
    mean_summary = summaries.groupby(["local_updates", "level", "method"]).mean(numeric_only=True)
    mean_summary.to_csv(destination / "summary.csv")

    coupled = manifests[5]["experiment"].get("coupled_auxiliary", False)
    def plot_metrics(filename, metrics_to_plot, methods):
        levels = manifests[5]["levels"]
        fig, axes = plt.subplots(len(levels), len(metrics_to_plot), figsize=(5 * len(metrics_to_plot), 4 * len(levels)), squeeze=False)
        for i, level in enumerate(levels):
            for j, metric in enumerate(metrics_to_plot):
                ax = axes[i, j]
                for method in methods:
                    color = f"C{METHODS.index(method)}"
                    for updates, linestyle in [(2, "--"), (5, "-")]:
                        part = curves[(curves.level == level) & (curves.method == method) & (curves.local_updates == updates)]
                        agg = part.groupby("step")[metric].agg(["mean", "std"])
                        schedule = "5 coupled iterations" if coupled and updates == 5 else f"{updates} updates"
                        ax.plot(agg.index, agg["mean"], color=color, linestyle=linestyle,
                                label=f"{LABELS[method]}, {schedule}")
                        sd = agg["std"].fillna(0)
                        ax.fill_between(agg.index, agg["mean"] - sd, agg["mean"] + sd, color=color, alpha=.07)
                ax.set(title=f"Level {level}: {metric}", xlabel="Observations")
                if metric in {"action_tv", "nonself_tv", "purity", "state_coverage", "unmatched_mass"}:
                    ax.set_ylim(0, 1)
                ax.grid(alpha=.2)
        axes[0, 0].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(destination / f"{filename}.png", dpi=160)
        fig.savefig(destination / f"{filename}.pdf")
        plt.close(fig)
    plot_metrics("predictive_vs_long", ["action_tv", "nonself_tv"], ["long", "predictive"])
    plot_metrics("learning_curves", ["action_tv", "nonself_tv", "purity", "nll_bits"], METHODS)
    plot_metrics("coverage_and_time", ["state_coverage", "unmatched_mass", "train_seconds"], METHODS)
    lines = ["# Two local updates versus five coupled CSCG E/M iterations" if coupled else "# Two versus five local CSCG updates", "",
             "Research extension of the EGM brainstorm; means over three paired seeds per environment.",
             ("Each run sees 2,000 observations. Every five-iteration E-step uses the latest student parameters; every auxiliary M-step incorporates its penalty."
              if coupled else "Each run sees 2,000 observations. Five-update baselines take five EM passes; the other arms take four EM passes plus one penalty step."),
             "The predictive/NLL auxiliary-count step and posterior analytical-gradient step are generalized M-steps with descent; they do not claim an exact marginal argmin.",
             "Teacher targets are frozen before the local updates, sufficient statistics commit once, and EMA runs once afterwards.",
             "Initial tensors, trajectories, allocations, hyperparameters, actual held-out walks and evaluation protocols match the completed two-update experiment.", "",
             "TV areas are divided by 2,000 (integrated mean TV). Lower TV, NLL and unmatched mass are better; higher purity and coverage are better.", "",
             "| Level | Updates | Method | Final action TV | Integrated action TV | Final nonself TV | Integrated nonself TV |",
             "| --- | ---: | --- | ---: | ---: | ---: | ---: |"]
    for (updates, level, method), row in mean_summary.iterrows():
        lines.append(f"| {level} | {updates} | {LABELS[method]} | {row.final_action_tv:.4f} | {row.auc_action_tv:.4f} | {row.final_nonself_tv:.4f} | {row.auc_nonself_tv:.4f} |")
    lines += ["", "## Predictive KL minus long EM: every paired seed", "",
              "Negative values favour predictive KL. These are descriptive paired differences, not independent checkpoint replicates.", "",
              "| Level | Updates | Seed | Final action TV difference | Integrated action TV difference | Final nonself TV difference | Integrated nonself TV difference |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    core = differences[(differences.comparison == "method_minus_long") & (differences.method == "predictive")]
    for (level, updates, seed), part in core.groupby(["level", "local_updates", "seed"]):
        values = part.set_index("metric").delta
        lines.append(f"| {level} | {updates} | {seed} | {values.final_action_tv:+.4f} | {values.auc_action_tv:+.4f} | {values.final_nonself_tv:+.4f} | {values.auc_nonself_tv:+.4f} |")
    lines += ["", "## Supplementary endpoints", "",
              "| Level | Updates | Method | Purity | State coverage | Macro unmatched mass | NLL (bits) | Train seconds |",
              "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |"]
    for (updates, level, method), row in mean_summary.iterrows():
        lines.append(f"| {level} | {updates} | {LABELS[method]} | {row.final_purity:.4f} | {row.final_state_coverage:.4f} | {row.final_unmatched_mass:.4f} | {row.final_nll_bits:.4f} | {row.train_seconds:.2f} |")
    lines += ["", "Per-seed final and integrated differences for every requested metric, within-model five-minus-two differences,",
              "predictive-minus-long differences, and predictive-minus-NLL differences are in per_seed_paired_differences.csv.",
              "Integrated purity, state coverage, clone utilisation, canonical node recall, NLL and unmatched mass are also in per_seed_summary.csv.",
              "Coverage is distinct from canonical node recall. Unmatched mass uses the existing macro mean over valid source/action rows.", "",
              "The two-update training times come from the completed experiment, while five-update times were measured in this extension;",
              "ratios are descriptive and not a controlled hardware benchmark. Objective-only line searches are not additional parameter updates.",
              "A lower canonical TV can reflect changed clone alignment/coverage as well as transition learning. No full map-recovery claim is implied.",
              "Three paired seeds do not support a confirmatory claim; raw seed differences accompany every aggregate."]
    (destination / "results.md").write_text("\n".join(lines) + "\n")
    write_json(destination / "provenance.json", dict(
        reference=str(baseline), five_update_experiment=str(output),
        reference_manifest_sha256=hashlib.sha256((baseline / "experiment.json").read_bytes()).hexdigest(),
        reference_implementation=json.loads((baseline / "implementation.json").read_text()),
        new_implementation=json.loads((output / "implementation.json").read_text()),
        paired_runs_per_budget=len(summary_frames[0]), held_out_walk_arrays_equal=True))


def run_case(output, level, seed, method, config, checkpoints, data, n_clones, initial, indexed, gt, protocol, provenance):
    folder = output / f"level={level}" / f"seed={seed}" / method
    folder.mkdir(parents=True, exist_ok=True)
    complete = folder / "rows.json"
    if complete.exists():
        print(f"skip level={level} seed={seed} {method}", flush=True)
        return json.loads(complete.read_text())
    print(f"train level={level} seed={seed} {method}", flush=True)
    metadata = {}
    result = train_experiment(
        initial, n_clones, data["x"], data["actions"], method, config, checkpoints,
        callback=lambda step, T, meta: metadata.update({step: meta}))
    np.savez_compressed(folder / "snapshots.npz", steps=np.array(checkpoints),
                        T=np.stack([result["snapshots"][s] for s in checkpoints]), n_clones=n_clones,
                        obs_uniques=data["vocabulary"], action_uniques=data["action_vocabulary"])
    write_json(folder / "training.json", dict(provenance=provenance, diagnostics=result["diagnostics"],
                                              checkpoint_metadata=metadata))
    method_rows = []
    for step in checkpoints:
        metrics = evaluate_snapshot(result["snapshots"][step], n_clones, indexed, gt, protocol,
                                    level, method, step, folder / "evaluations" / f"step={step:04d}.json")
        meta = metadata[step]
        work = sum(meta[k] for k in ["filter_block_entries", "smoothing_block_entries", "replay_block_entries", "tensor_entries"])
        method_rows.append(dict(level=level, seed=seed, method=method, step=step,
                                work_entries=work, **meta, **metrics))
    write_json(complete, method_rows)
    print(f"done {method}: TV={method_rows[-1]['action_tv']:.4f} nonself={method_rows[-1]['nonself_tv']:.4f}; train={result['train_seconds']:.2f}s", flush=True)
    return method_rows


def checkpoint_schedule(steps):
    """Scale the original 22-checkpoint schedule with the requested horizon."""
    original = [0, 40, *range(100, 2001, 100)]
    return sorted({round(step * steps / 2000) for step in original})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--levels", nargs="+", type=int, default=[21, 22])
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--workers", type=int, default=1, help="Independent training/evaluation processes")
    p.add_argument("--clones", type=int, default=10)
    p.add_argument("--clone-allocation", choices=["uniform", "ground_truth"], default="uniform")
    p.add_argument("--forgetting", type=float, default=1.0)
    p.add_argument("--pseudocount", type=float, default=.002)
    p.add_argument("--weight", type=float, default=1.0)
    p.add_argument("--window", type=int, default=40)
    p.add_argument("--teacher-window", type=int, help="Teacher smoothing context (default: --window)")
    p.add_argument("--half-life", type=float, default=40.)
    p.add_argument("--local-updates", type=int, choices=[2, 5], default=2)
    p.add_argument("--coupled-auxiliary", action="store_true", help="Include the auxiliary objective in every E/M iteration")
    p.add_argument("--student-inference", choices=STUDENT_INFERENCE, default="filter",
                   help="Replay source belief: carried filter, or smoothed over the student's x_{u:u+1}")
    p.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS),
                   help="Subset of arms to train (default: all five)")
    p.add_argument("--rand-prob", type=float, default=.3)
    p.add_argument("--walks", type=int, default=10)
    p.add_argument("--walk-length", type=int, default=500)
    p.add_argument("--output", type=Path)
    p.add_argument("--compare-to", type=Path, help="Completed two-update experiment to compare without modifying it")
    p.add_argument("--report-only", action="store_true", help="Regenerate plots and tables from saved evaluation rows")
    args = p.parse_args()
    if args.steps < 2:
        p.error("steps must be at least 2")
    if args.workers < 1:
        p.error("workers must be positive")
    if args.coupled_auxiliary and args.local_updates != 5:
        p.error("the coupled experiment requires --local-updates 5")
    config = ExperimentConfig(window=args.window, forgetting=args.forgetting,
                              pseudocount=args.pseudocount, penalty_weight=args.weight,
                              teacher_half_life=args.half_life, local_updates=args.local_updates,
                              coupled_auxiliary=args.coupled_auxiliary,
                              student_inference=args.student_inference, teacher_window=args.teacher_window)
    default_output = ("cscg_distillation_coupled5_v1" if args.coupled_auxiliary else
                      "cscg_distillation_v1" if args.local_updates == 2 else "cscg_distillation_updates5_v1")
    if args.clone_allocation == "ground_truth":
        default_output = default_output.replace("_v1", "_ground_truth_v1")
    if args.student_inference == "smooth":
        default_output = default_output.replace("_v1", "_smooth_student_v1")
    output = (args.output or ROOT / "results/diagnostics" / default_output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.report_only:
        saved = [json.loads(path.read_text()) for path in sorted(output.glob("level=*/seed=*/*/rows.json"))]
        if not saved:
            p.error("No saved evaluation rows in the output directory")
        report(output, [row for part in saved for row in part])
        if args.compare_to:
            compare_local_updates(output, args.compare_to.resolve())
        return
    algorithm_path = ROOT / "src/mercury_runs/algorithms/cscg/distillation.py"
    implementation = dict(algorithm_sha256=hashlib.sha256(algorithm_path.read_bytes()).hexdigest(),
                          runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), numpy_version=np.__version__)
    implementation_path = output / "implementation.json"
    if implementation_path.exists() and json.loads(implementation_path.read_text()) != implementation:
        raise ValueError("Implementation differs from the saved experiment; use a fresh output directory")
    write_json(implementation_path, implementation)
    checkpoints = checkpoint_schedule(args.steps)
    manifest = dict(experiment=asdict(config), levels=args.levels, seeds=args.seeds,
                    observations=args.steps, clones_per_observation=(args.clones if args.clone_allocation == "uniform" else None),
                    clone_allocation=args.clone_allocation, checkpoints=checkpoints,
                    rand_prob=args.rand_prob, evaluation_walks=args.walks, walk_length=args.walk_length,
                    evaluation_seed=10000, methods=[m for m in METHODS if m in args.methods], local_parameter_updates=args.local_updates,
                    penalty="current frozen window, coefficient eta * original running count mass",
                    inference="causal carried filter; historical boundary beliefs; destination-normalized T",
                    vocabulary="observation/action vocabulary scanned from the shared training stream",
                    schema_version=1)
    manifest_path = output / "experiment.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError("Output directory contains a different experiment")
    write_json(manifest_path, manifest)
    rows, tasks = [], []
    for level in args.levels:
        protocol = default_canonical_v4_protocol(
            level=level, sensor="cardinal distance", sensor_range=1, rand_prob=args.rand_prob,
            evaluation_seed=10000, number_of_walks=args.walks, walk_length=args.walk_length,
            valid_trajectories_only=False, edge_thresholds=[1e-6])
        protocol.decoder = "causal_clone_filter_argmax"
        protocol.decoder_configuration_source = "shared_experiment_filter"
        protocol.inference_state_initialisation = "uniform_clone_prior"
        walks = generate_and_cache_walks(study_root=output, protocol=protocol, level=level)
        gt = build_empirical_ground_truth(
            walk_data=walks, n_true=len(exact_cartesian_reference_positions(level)),
            n_actions=infer_num_actions_from_valid_transitions(exact_valid_sensorimotor_transitions(level_index=level)),
            level=level)
        for seed in args.seeds:
            data = load_training(level, seed, args.rand_prob, args.steps)
            n_clones = (ground_truth_clone_allocation(level, data["vocabulary"]) if args.clone_allocation == "ground_truth"
                        else np.full(len(data["vocabulary"]), args.clones, dtype=int))
            n_states = int(n_clones.sum())
            initial = row_normalize(np.random.RandomState(seed).rand(len(data["action_vocabulary"]), n_states, n_states) + args.pseudocount)
            indexed = indexed_evaluation(walks, data["vocabulary"], data["action_vocabulary"])
            provenance = {k: data[k] for k in ["path", "sha256", "collisions"]}
            provenance.update(clone_allocation=n_clones.tolist(), initial_T_sha256=hashlib.sha256(initial.tobytes()).hexdigest(),
                              evaluation_protocol=protocol.protocol_id(), ground_truth_checksum=gt["checksum"])
            for method in [m for m in METHODS if m in args.methods]:
                tasks.append((output, level, seed, method, config, checkpoints, data, n_clones,
                              initial, indexed, gt, protocol, provenance))
    if args.workers == 1:
        for task in tasks:
            rows.extend(run_case(*task))
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(run_case, *task) for task in tasks]
            for future in as_completed(futures):
                rows.extend(future.result())
    report(output, rows)
    if args.compare_to:
        compare_local_updates(output, args.compare_to.resolve())
    print(f"Results: {output / 'results.md'}", flush=True)


if __name__ == "__main__":
    main()
