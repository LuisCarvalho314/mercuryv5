"""Canonical metric reporting for saved CSCG distillation evaluation records.

Uses the exact metric extractor used by postprocess_evaluation_v5.py in the
reference study. No training, decoding, model loading, or alignment is performed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.analysis.plot_config_metric_trajectories import (
    DEFAULT_EDGE_THRESHOLDS, _action_edge_f1, _metrics_from_record,
)

ROOT = Path(__file__).resolve().parents[2]
METHODS = ("short", "long", "predictive", "posterior", "nll_control")
LABELS = dict(short="Short EM", long="Long EM", predictive="Predictive KL",
              posterior="Posterior KL", nll_control="NLL control")
PRIMARY = ("action_tv_similarity", "action_no_self_loop_tv_similarity",
           "action_edge_f1_auc", "action_no_self_loop_edge_f1_auc")
METRIC_LABELS = dict(zip(PRIMARY, ("Action transition similarity", "No-self-loop transition similarity",
                                  "Action edge F1 (threshold AUC)", "No-self-loop edge F1 (threshold AUC)")))
METRIC_LABELS.update(latent_precision="Clone purity", latent_recall="Canonical clone recall")
METRICS = (*PRIMARY, "latent_precision", "latent_recall", "eval_nll_bps",
           "state_coverage", "unmatched_mass", "clone_utilisation")


def canonical_metrics(record):
    scores = _metrics_from_record(record)
    for metric in PRIMARY:
        if metric not in scores or not np.isfinite(scores[metric]):
            raise ValueError(f"Missing or invalid canonical metric: {metric}")
    action = record["action_conditioned"]
    for similarity, tv in [("action_tv_similarity", "tv_occupancy"),
                           ("action_no_self_loop_tv_similarity", "tv_non_self_loop_occupancy")]:
        if not np.isclose(scores[similarity], 1 - action[tv], atol=1e-12, rtol=0):
            raise ValueError("Stored canonical TV and similarity disagree")
    C = np.asarray(record["alignment"]["cooccurrence"])
    mapping = np.asarray(record["alignment"]["learned_to_true"])
    used = C.sum(axis=1) > 0
    represented = mapping[used & (mapping >= 0)]
    scores.update(state_coverage=len(np.unique(represented)) / C.shape[1],
                  unmatched_mass=float(action["total_unmatched_target_mass"]),
                  clone_utilisation=float(used.mean()))
    return scores


def summarise(curves):
    rows = []
    for key, part in curves.groupby(["iterations", "level", "seed", "method"]):
        part = part.sort_values("step")
        if part.step.duplicated().any() or part.step.iloc[0] != 0:
            raise ValueError("Canonical learning curves require unique checkpoints starting at zero")
        final = part.iloc[-1]
        row = dict(zip(["iterations", "level", "seed", "method"], key))
        row.update(train_seconds=float(final.train_seconds), final_step=int(final.step))
        for metric in METRICS:
            row["final_" + metric] = float(final[metric])
            row["time_mean_" + metric] = float(np.trapezoid(part[metric], part.step) / final.step)
        rows.append(row)
    return pd.DataFrame(rows)


def final_f1_curve(record):
    """Actual F1 at each threshold at one checkpoint, using the reference rule."""
    action = record["action_conditioned"]
    truth = np.asarray(action["P_true_full"], dtype=float)
    prediction = np.asarray(action["P_hat_full"], dtype=float)
    valid = np.asarray(action["valid_rows"], dtype=bool)
    return [dict(threshold=float(threshold), include_self_loops=include_self,
                 edge_f1=_action_edge_f1(p_true=truth, p_hat=prediction, valid_rows=valid,
                                        threshold=threshold, include_self_loops=include_self))
            for include_self in [True, False] for threshold in DEFAULT_EDGE_THRESHOLDS]


def residual_plots(records, output, conditions=None):
    """Reference-rendered maps plus uncollapsed per-action probability residuals."""
    import matplotlib.pyplot as plt
    from mercury_runs.analysis.evaluation_records import action_matrices, diagnostics_from_record
    from mercury_runs.analysis.residual_map_rendering import save_selected_figure
    from mercury_runs.infrastructure.paper_precision import condition_non_selfloop_action_distributions

    checkpoints = {record["snapshot_step"] for record in records.values()}
    if len(checkpoints) != 1:
        raise ValueError("Residual plots require a common checkpoint")
    checkpoint = checkpoints.pop()
    directory = output / "residuals"
    directory.mkdir(exist_ok=True)
    levels = sorted({key[1] for key in records})
    seeds = sorted({key[2] for key in records})
    paths = []
    conditions = conditions or {2: "budget2", 5: "budget5"}
    for iterations, condition in conditions.items():
        for seed in seeds:
            for ignore_self in [False, True]:
                policy = "without_self_loops" if ignore_self else "with_self_loops"
                panels = {}
                for level in levels:
                    for method in METHODS:
                        record = records[(iterations, level, seed, method)]
                        diagnostic = diagnostics_from_record(record, method=method, action_conditioned=True,
                                                             ignore_self_loops=ignore_self)
                        panels[f"L{level}/{LABELS[method]}"] = dict(
                            P_gt=diagnostic.transition_true.tolist(), P_hat=diagnostic.transition_predicted.tolist(),
                            R=diagnostic.residual.tolist(), mean_abs_R=diagnostic.residual_magnitude.tolist(),
                            E_plus=diagnostic.excess_mass.tolist(), E_minus=diagnostic.missing_mass.tolist(),
                            tv_by_state=diagnostic.tv_by_state.tolist(), similarity_by_state=diagnostic.similarity_by_state.tolist(),
                            uncovered_rows=diagnostic.uncovered_rows.tolist(), tv_sim=float(diagnostic.tv_sim))
                payload = dict(scenario=f"Final action-conditioned residuals; {condition}, seed {seed}, {policy}",
                               visual_encoding=dict(action_conditioned=True), panels=panels)
                stem = f"canonical_residuals_{condition}_seed{seed}_{policy}"
                (directory / f"{stem}.json").write_text(json.dumps(payload, indent=2) + "\n")
                for suffix in ["png", "pdf"]:
                    target = directory / f"{stem}.{suffix}"
                    save_selected_figure(payload=payload, output_path=target, vis_threshold=.02,
                                         max_residual_override=1., font_scale=1., header_font_scale=1.)
                    paths.append(str(target))
                # Seed zero also gets every action separately, retaining the sink.
                if seed != seeds[0]:
                    continue
                for level in levels:
                    fig, axes = plt.subplots(4, len(METHODS), figsize=(16, 11), squeeze=False, layout="constrained")
                    for col, method in enumerate(METHODS):
                        record = records[(iterations, level, seed, method)]
                        truth, prediction, valid = action_matrices(record, include_sink=True)
                        if ignore_self:
                            prediction, truth, valid = condition_non_selfloop_action_distributions(prediction, truth, valid)
                        residual = prediction - truth
                        n = truth.shape[1]
                        shown_states = list(range(0, n, max(1, int(np.ceil(n / 12)))))
                        for action in range(4):
                            values = np.where(valid[action, :, None], residual[action], np.nan)
                            artist = axes[action, col].imshow(values, vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")
                            axes[action, col].set_xticks([*shown_states, n], [*map(str, shown_states), "⊥"], fontsize=6)
                            axes[action, col].set_yticks(shown_states, list(map(str, shown_states)), fontsize=6)
                            if action == 0:
                                axes[action, col].set_title(LABELS[method])
                            if col == 0:
                                axes[action, col].set_ylabel(f"{['North', 'East', 'South', 'West'][action]}\nSource state")
                            if action == 3:
                                axes[action, col].set_xlabel("Target state (⊥ = unmatched)")
                    fig.colorbar(artist, ax=axes.ravel().tolist(), shrink=.7, label="Canonical P(model) − P(GT)")
                    fig.suptitle(f"Level {level}: final action-conditioned residuals, {condition}, seed {seed}, {policy}")
                    stem = f"action_residuals_level{level}_{condition}_seed{seed}_{policy}"
                    for suffix in ["png", "pdf"]:
                        target = directory / f"{stem}.{suffix}"
                        fig.savefig(target, dpi=150)
                        paths.append(str(target))
                    plt.close(fig)
    (directory / "manifest.json").write_text(json.dumps(dict(
        checkpoint=checkpoint, reference_renderer="mercury_runs.analysis.residual_map_rendering.save_selected_figure",
        diagnostics="mercury_runs.analysis.evaluation_records.diagnostics_from_record",
        map_policy="canonical evaluation-occupancy weighted action display; includes aggregate absolute residual magnitude",
        matrix_policy="separate N/E/S/W actions; explicit sink retained; invalid rows masked",
        visual_threshold=.02, metrics_threshold_unchanged=True, conditions=conditions, files=paths), indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--two", type=Path, default=ROOT / "results/diagnostics/cscg_distillation_v1")
    parser.add_argument("--five", type=Path, default=ROOT / "results/diagnostics/cscg_distillation_coupled5_v1")
    parser.add_argument("--reference", type=Path, default=ROOT / "results/current/action_alignment_rerun_matched_budget_lambda099_20261004")
    args = parser.parse_args()
    output = args.five / "canonical_metrics"
    output.mkdir(parents=True, exist_ok=True)
    rows, threshold_rows, protocols, source_hashes, final_records = [], [], {}, {}, {}
    expected_keys = None
    for iterations, root in [(2, args.two), (5, args.five)]:
        keys = set()
        config = json.loads((root / "experiment.json").read_text())
        for row_path in sorted(root.glob("level=*/seed=*/*/rows.json")):
            for original in json.loads(row_path.read_text()):
                path = row_path.parent / "evaluations" / f"step={original['step']:04d}.json"
                record = json.loads(path.read_text())
                keys.add((record["level"], original["seed"], record["method"], record["snapshot_step"]))
                protocols[record["protocol_id"]] = record["protocol"]
                scores = canonical_metrics(record)
                # Primary similarity is only a change of reporting direction.
                assert np.isclose(scores["action_tv_similarity"], 1 - original["action_tv"])
                assert np.isclose(scores["action_no_self_loop_tv_similarity"], 1 - original["nonself_tv"])
                rows.append(dict(iterations=iterations, level=record["level"], seed=original["seed"],
                                 method=record["method"], step=record["snapshot_step"],
                                 train_seconds=original["train_seconds"], **scores))
                if record["snapshot_step"] == config["observations"]:
                    final_records[(iterations, record["level"], original["seed"], record["method"])] = record
                    for threshold_score in final_f1_curve(record):
                        threshold_rows.append(dict(iterations=iterations, level=record["level"], seed=original["seed"],
                                                   method=record["method"], step=record["snapshot_step"], **threshold_score))
                source_hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        expected_count = len(config["levels"]) * len(config["seeds"]) * len(config["methods"]) * len(config["checkpoints"])
        if len(keys) != expected_count or (expected_keys is not None and keys != expected_keys):
            raise ValueError("Incomplete or unmatched canonical comparison records")
        expected_keys = keys
    curves = pd.DataFrame(rows)
    seeds = summarise(curves)
    curves.to_csv(output / "learning_curves.csv", index=False)
    seeds.to_csv(output / "per_seed_summary.csv", index=False)
    means = seeds.groupby(["iterations", "level", "method"]).mean(numeric_only=True)
    means.to_csv(output / "summary.csv")
    threshold_scores = pd.DataFrame(threshold_rows)
    threshold_scores.to_csv(output / "final_step_f1_by_seed.csv", index=False)
    threshold_scores.groupby(["iterations", "level", "method", "include_self_loops", "threshold"]).edge_f1.agg(
        ["mean", "std", "count"]).to_csv(output / "final_step_f1_summary.csv")
    differences = []
    numeric = [f"{prefix}_{metric}" for prefix in ["final", "time_mean"] for metric in METRICS] + ["train_seconds"]
    def pair(label, left, right):
        for metric in numeric:
            differences.append(dict(comparison=label, iterations=int(left.iterations), reference_iterations=int(right.iterations),
                                    level=int(left.level), seed=int(left.seed), method=left.method, reference=right.method,
                                    metric=metric, value=float(left[metric]), reference_value=float(right[metric]),
                                    delta=float(left[metric] - right[metric])))
    by_key = seeds.set_index(["iterations", "level", "seed", "method"], drop=False)
    for _, row in seeds[seeds.iterations == 5].iterrows():
        pair("five_minus_two", row, by_key.loc[(2, row.level, row.seed, row.method)])
    for (iterations, level, seed), part in seeds.groupby(["iterations", "level", "seed"]):
        indexed = part.set_index("method", drop=False)
        for method in ["predictive", "posterior", "nll_control"]:
            pair("method_minus_long", indexed.loc[method], indexed.loc["long"])
        pair("predictive_minus_nll", indexed.loc["predictive"], indexed.loc["nll_control"])
        if iterations == 2:
            pair("predictive_two_minus_long_five", indexed.loc["predictive"], by_key.loc[(5, level, seed, "long")])
    paired = pd.DataFrame(differences)
    paired.to_csv(output / "per_seed_paired_differences.csv", index=False)
    paired.groupby(["comparison", "iterations", "reference_iterations", "level", "method", "reference", "metric"]).delta.agg(
        ["mean", "std", "count"]).to_csv(output / "paired_summary.csv")

    import matplotlib.pyplot as plt
    for filename, methods, panels in [("learning_curves", METHODS, PRIMARY),
                                      ("predictive_vs_long", ("long", "predictive"), PRIMARY),
                                      ("purity_and_recall", METHODS, ("latent_precision", "latent_recall"))]:
        levels = sorted(curves.level.unique())
        fig, axes = plt.subplots(len(levels), len(panels), figsize=(4.5 * len(panels), 4 * len(levels)), squeeze=False)
        for i, level in enumerate(levels):
            for j, metric in enumerate(panels):
                ax = axes[i, j]
                for method in methods:
                    for iterations, linestyle in [(2, "--"), (5, "-")]:
                        values = curves[(curves.level == level) & (curves.method == method) & (curves.iterations == iterations)]
                        agg = values.groupby("step")[metric].agg(["mean", "std"])
                        color = f"C{METHODS.index(method)}"
                        schedule = "2 local updates" if iterations == 2 else "5 coupled iterations"
                        ax.plot(agg.index, agg["mean"], color=color, linestyle=linestyle, label=f"{LABELS[method]}: {schedule}")
                        sd = agg["std"].fillna(0)
                        ax.fill_between(agg.index, agg["mean"]-sd, agg["mean"]+sd, color=color, alpha=.08)
                ax.set(title=f"Level {level}: {METRIC_LABELS[metric]}", xlabel="Observations", ylim=(0, 1))
                ax.grid(alpha=.2)
        axes[0, 0].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(output / f"{filename}.png", dpi=160)
        fig.savefig(output / f"{filename}.pdf")
        plt.close(fig)

    fig, axes = plt.subplots(len(levels), 2, figsize=(12, 4 * len(levels)), squeeze=False)
    for i, level in enumerate(levels):
        for j, include_self in enumerate([True, False]):
            ax = axes[i, j]
            for method in METHODS:
                for iterations, linestyle in [(2, "--"), (5, "-")]:
                    part = threshold_scores[(threshold_scores.level == level) & (threshold_scores.method == method)
                                            & (threshold_scores.iterations == iterations)
                                            & (threshold_scores.include_self_loops == include_self)]
                    agg = part.groupby("threshold").edge_f1.agg(["mean", "std"])
                    color = f"C{METHODS.index(method)}"
                    ax.plot(agg.index, agg["mean"], color=color, linestyle=linestyle,
                            label=f"{LABELS[method]}: {iterations}", marker=".", markersize=3)
                    sd = agg["std"].fillna(0)
                    ax.fill_between(agg.index, agg["mean"]-sd, agg["mean"]+sd, color=color, alpha=.08)
            policy = "with self-loops" if include_self else "without self-loops"
            ax.set(title=f"Level {level}: action-conditioned edge F1, {policy}",
                   xlabel="Action-conditioned transition probability threshold",
                   ylabel="Action-conditioned edge F1 at observation 2,000", ylim=(0, 1))
            ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output / "final_step_f1.png", dpi=160)
    fig.savefig(output / "final_step_f1.pdf")
    fig.savefig(output / "final_step_action_conditioned_edge_f1.png", dpi=160)
    fig.savefig(output / "final_step_action_conditioned_edge_f1.pdf")
    plt.close(fig)
    residual_plots(final_records, output)

    reference_manifest = json.loads((args.reference / "notebook_plots/main_study_eval_v5/evaluation_v5_manifest.json").read_text())
    shadow = args.reference / "evaluation_v5/studies/main_study"
    reference_path = next(shadow.glob(f"*/evaluations/{reference_manifest['protocol_id']}/snapshots/cscg/step=00002000.json"))
    reference_record = json.loads(reference_path.read_text())
    provenance = dict(reference_dataset=str(args.reference), reference_record=str(reference_path),
                      reference_protocol=reference_record["protocol"], experiment_protocols=protocols,
                      metric_implementation="scripts.analysis.plot_config_metric_trajectories._metrics_from_record",
                      metric_implementation_sha256=hashlib.sha256((ROOT / "scripts/analysis/plot_config_metric_trajectories.py").read_bytes()).hexdigest(),
                      report_implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      edge_thresholds=list(DEFAULT_EDGE_THRESHOLDS), edge_threshold_auc_normalized_by_span=True,
                      edge_sink_policy="explicit sink counts as false positive",
                      no_self_edge_policy="exclude physical diagonal without reconditioning probabilities",
                      no_self_similarity_policy="canonical conditional nonself distributions and GT movement support",
                      evaluation_records=len(rows), evaluation_record_sha256=source_hashes,
                      training_or_inference_changed=False)
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    lines = ["# CSCG distillation: canonical transition similarity and edge F1", "",
             "Metrics reuse the exact extractor used by the supplied reference dataset's evaluation-v5 postprocessing.",
             "Transition similarity is 1 minus occupancy-weighted canonical TV. Edge F1 is normalized area over the reference probability-threshold grid [0, 0.5].",
             "Both metrics include unmatched-target penalties. All four primary scores are higher-is-better.",
             "No-self transition similarity conditions on movement; no-self edge F1 removes diagonal edges without renormalizing probabilities.", "",
             "The experiment's carried causal decoder, ten 500-step walks, evaluation seed 10000 and policy randomness 0.3 are unchanged.",
             "The reference dataset uses batch forward-backward decoding and a different walk protocol. Metric definitions match; evaluation protocols do not.",
             "These results are an internal two-versus-five comparison, not a pooled comparison against the reference dataset's numerical scores.", "",
             "F1 threshold AUC and temporal mean are different integrations: the former summarizes thresholds at one checkpoint;",
             "time_mean_* integrates a learning curve over observations and divides by 2,000. Shading shows one seed standard deviation.", "",
             "| Level | Budget | Model | Transition similarity | No-self similarity | Edge F1 threshold AUC | No-self edge F1 threshold AUC |",
             "| --- | --- | --- | ---: | ---: | ---: | ---: |"]
    for (iterations, level, method), row in means.iterrows():
        budget = "2 local" if iterations == 2 else "5 coupled"
        values = " | ".join(f"{row['final_' + m]:.4f}" for m in PRIMARY)
        lines.append(f"| {level} | {budget} | {LABELS[method]} | {values} |")
    lines += ["", "## Mean scores over the first 2,000 observations", "",
              "| Level | Budget | Model | Transition similarity | No-self similarity | Edge F1 threshold AUC | No-self edge F1 threshold AUC |",
              "| --- | --- | --- | ---: | ---: | ---: | ---: |"]
    for (iterations, level, method), row in means.iterrows():
        values = " | ".join(f"{row['time_mean_' + m]:.4f}" for m in PRIMARY)
        lines.append(f"| {level} | {iterations} | {LABELS[method]} | {values} |")
    lines += ["", "## Supplementary endpoints", "",
              "| Level | Budget | Model | Purity | Canonical recall | GT state coverage | Unmatched mass | NLL bits | Train seconds |",
              "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for (iterations, level, method), row in means.iterrows():
        values = " | ".join(f"{row['final_' + m]:.4f}" for m in ("latent_precision", "latent_recall", "state_coverage", "unmatched_mass", "eval_nll_bps"))
        lines.append(f"| {level} | {iterations} | {LABELS[method]} | {values} | {row.train_seconds:.2f} |")
    lines += ["", "Per-seed final and temporal-mean paired differences are saved in per_seed_paired_differences.csv.",
              "Dedicated purity_and_recall.png/pdf plots show canonical clone purity and recall over observations.",
              "final_step_action_conditioned_edge_f1.png/pdf plots actual action-conditioned edge F1 versus threshold at observation 2,000, with and without self-loops, rather than an AUC.",
              "Edges retain their action label: true positives, false positives and false negatives are counted over (action, source, destination) triples.",
              "final_step_f1_by_seed.csv and final_step_f1_summary.csv retain every seed, threshold and update budget.",
              "residuals/ contains the canonical reference-style residual maps for every seed/budget, with and without self-loops,",
              "and separate per-action residual matrices with the unmatched sink for seed zero. Residuals are model minus GT probabilities.",
              "Positive paired differences favour a method for similarity, F1, purity and recall; lower NLL, unmatched mass and time are better.",
              "Canonical recall is column-max cooccurrence recall, not the fraction of represented physical states; fragmentation can lower recall despite complete state coverage.",
              "The old F1 at threshold 1e-6 remains in the historical reports but is not substituted for this reference F1 threshold AUC.",
              "The five coupled auxiliary M-steps provide generalized-EM descent rather than exact marginal argmin solutions.",
              "Three paired seeds per environment are pilot evidence; substantial unmatched mass and protocol differences limit broader claims."]
    (output / "results.md").write_text("\n".join(lines) + "\n")
    print(f"Canonical report: {output / 'results.md'}")


if __name__ == "__main__":
    main()
