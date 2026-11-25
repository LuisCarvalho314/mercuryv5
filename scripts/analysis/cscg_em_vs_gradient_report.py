"""Report for scripts/run_cscg_em_vs_gradient.sh: EM-CSCG vs projected-gradient CSCG.

  PYTHONPATH=.:src .venv/bin/python scripts/analysis/cscg_em_vs_gradient_report.py --sweep DIR \
      [--walks 10 --walk-length 500 --rand-prob 0.1]

Primary metrics: training-matched decoder (carried filter + lag W - 1, the long arm's training
inference). The pure filter (lag 0) is reported as a decoder-sensitivity check only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.analysis.cscg_distillation_matched_eval import output_name

COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300")
INK, MUTED = "#0b0b0b", "#52514e"
CURVES = (("action_tv_similarity", "Action transition similarity"),
          ("action_no_self_loop_tv_similarity", "No-self-loop transition similarity"),
          ("eval_nll_bps", "Held-out NLL (bits / symbol)"),
          ("latent_precision", "Clone purity"),
          ("active_clones", "Active clones"),
          ("transition_entropy_bits", "Mean row entropy of T (bits)"))


FAMILIES = {"gradcscg": "gradCSCG (logits + Adam)", "projected_adam": "Projected Adam",
            "pgd": "PGD", "pgd_plain": "PGD fixed step"}
ORDER = ("em", "gradcscg", "projected_adam", "pgd", "pgd_plain")
PRIMARY_ARMS = ("em", "gradcscg")


def family(name):
    return "em" if name == "em" else name.rsplit("_lr", 1)[0]


def arm_key(name):
    """EM, then the primary gradCSCG arms, then the secondary variants; learning rates ascending."""
    return (ORDER.index(family(name)), 0. if name == "em" else float(name.rsplit("_lr", 1)[1]))


def label(name):
    if name == "em":
        return "EM"
    return f"{FAMILIES[family(name)]} η={name.rsplit('_lr', 1)[1]}"


def snapshot_statistics(path):
    with np.load(path) as saved:
        steps, Ts = saved["steps"], saved["T"].astype(np.float64)
    out = {}
    for step, T in zip(steps, Ts):
        with np.errstate(divide="ignore", invalid="ignore"):
            entropy = -np.where(T > 0, T * np.log2(T), 0.).sum(-1)
        out[int(step)] = dict(transition_entropy_bits=float(entropy.mean()), exact_zeros=int(np.sum(T == 0)),
                              min_T=float(T.min()), row_sum_error=float(np.abs(T.sum(-1) - 1).max()),
                              finite=bool(np.all(np.isfinite(T))))
    return out


def load(sweep, budget):
    curves, failures, objectives = [], [], []
    for context_dir in sorted(sweep.glob("context*")):
        if not context_dir.is_dir():
            continue
        context = int(context_dir.name.removeprefix("context"))
        for arm_dir in sorted(p for p in context_dir.iterdir() if p.is_dir()):
            manifest = json.loads((arm_dir / "experiment.json").read_text())["experiment"]
            for failure in arm_dir.glob("level=*/seed=*/long/failure.json"):
                f = json.loads(failure.read_text())
                failures.append(dict(context=context, arm=arm_dir.name, error=f["error"],
                                     last_checkpoint=f["last_checkpoint"]))
            case = next(iter(arm_dir.glob("level=*/seed=*/long")), None)
            if case is None or not (case / "rows.json").exists():
                continue
            rows = pd.DataFrame(json.loads((case / "rows.json").read_text()))
            metrics = pd.read_csv(arm_dir / output_name(*budget) / "canonical_metrics_all.csv")
            stats = snapshot_statistics(case / "snapshots.npz")
            training = json.loads((case / "training.json").read_text())
            for record in training.get("objective_diagnostics", []):
                its = record["local_iterations"]
                objectives.append(dict(context=context, arm=arm_dir.name, step=record["step"],
                                       objective_start=its[0]["objective_before"],
                                       objective_end=its[-1]["objective_after"],
                                       backtracks=sum(i.get("backtracks", 0) for i in its),
                                       rejected=sum(not i.get("accepted", True) for i in its)))
            cost = rows[["step", "train_seconds", "transitions", "em_passes", "gradient_steps",
                         "line_search_evaluations", "backtracks", "rejected_steps", "smoothing_block_entries"]]
            for decoder, part in metrics.groupby("decoder"):
                part = part.merge(cost, on="step", how="left")
                for column in ("transition_entropy_bits", "exact_zeros", "min_T", "row_sum_error", "finite"):
                    part[column] = [stats[int(s)][column] for s in part.step]
                # Clones visited by the decoder on the held-out walks.
                part["active_clones"] = part.clone_utilisation * sum(training["provenance"]["clone_allocation"])
                part = part.assign(context=context, arm=arm_dir.name, optimizer=manifest["optimizer"],
                                   learning_rate=manifest["gradient_learning_rate"],
                                   max_backtracks=manifest["gradient_backtracks"])
                # Optimisation iterations: local iterations; evaluations add Armijo trial forward passes.
                part["iterations"] = part.transitions * manifest["local_updates"]
                part["objective_evaluations"] = part.em_passes + part.line_search_evaluations
                curves.append(part)
    return pd.concat(curves, ignore_index=True), pd.DataFrame(failures), pd.DataFrame(objectives)


def plot(curves, x, xlabel, metrics, filename, output, log_x=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    contexts = sorted(curves.context.unique())
    arms = sorted(curves.arm.unique(), key=arm_key)
    fig, axes = plt.subplots(len(contexts), len(metrics), figsize=(4.2 * len(metrics), 3.4 * len(contexts)),
                             squeeze=False)
    for i, context in enumerate(contexts):
        for j, (metric, name) in enumerate(metrics):
            ax = axes[i, j]
            for k, arm in enumerate(arms):
                part = curves[(curves.context == context) & (curves.arm == arm)].sort_values("step")
                if part.empty:
                    continue
                values = part[x].clip(lower=1e-3) if log_x else part[x]
                ax.plot(values, part[metric], color=COLORS[k % len(COLORS)], lw=2,
                        ls="--" if arm.startswith("pgd_plain") else "-", label=label(arm))
            ax.set_title(f"Context {context}: {name}", fontsize=9, color=INK)
            ax.set_xlabel(xlabel, fontsize=8, color=MUTED)
            if log_x:
                ax.set_xscale("log")
            ax.tick_params(labelsize=7, colors=MUTED)
            ax.grid(alpha=.2)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
    axes[0, 0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(output / f"{filename}.png", dpi=150)
    plt.close(fig)


def table(frame, columns, formats):
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" if f is None else "---:" for f in formats) + " |"]
    for _, row in frame.iterrows():
        lines.append("| " + " | ".join(str(row[c]) if f is None else format(row[c], f) for c, f in zip(columns, formats)) + " |")
    return lines


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sweep", type=Path, required=True)
    p.add_argument("--walks", type=int, default=10)
    p.add_argument("--walk-length", type=int, default=500)
    p.add_argument("--rand-prob", type=float, default=.1)
    args = p.parse_args()
    sweep = args.sweep.resolve()
    output = sweep / "report"; output.mkdir(exist_ok=True)
    curves, failures, objectives = load(sweep, (args.walks, args.walk_length, args.rand_prob))
    curves.to_csv(output / "learning_curves.csv", index=False)
    objectives.to_csv(output / "arrival_objectives.csv", index=False)
    failures.to_csv(output / "failures.csv", index=False)
    matched = curves[curves.decoder == "training_matched"]
    primary = matched[matched.arm.map(family).isin(PRIMARY_ARMS)]
    secondary = matched[(matched.arm == "em") | ~matched.arm.map(family).isin(PRIMARY_ARMS)]
    short = (CURVES[0], CURVES[2])
    for frame, suffix in ((primary, ""), (secondary, "_secondary")):
        if frame.arm.nunique() < 2:
            continue
        plot(frame, "step", "Environment steps", CURVES, f"metrics_vs_steps{suffix}", output)
        plot(frame, "iterations", "Optimisation iterations (local)", short, f"metrics_vs_iterations{suffix}", output)
        plot(frame, "objective_evaluations", "Window passes (E-steps + line-search evaluations)", short,
             f"metrics_vs_evaluations{suffix}", output, log_x=True)
        plot(frame, "train_seconds", "Training wall-clock (s)", short, f"metrics_vs_wallclock{suffix}", output,
             log_x=True)
    primary = matched

    final = primary[primary.step == primary.step.max()].copy()
    final["arm_order"] = final.arm.map(arm_key)
    final = final.sort_values(["context", "arm_order"])
    final["label"] = final.arm.map(label)
    final["seconds_per_arrival_ms"] = 1e3 * final.train_seconds / final.transitions
    final["evaluations_per_iteration"] = final.objective_evaluations / final.iterations
    final["rejected_fraction"] = final.rejected_steps / final.gradient_steps.replace(0, np.nan)
    filt = curves[(curves.decoder == "filter") & (curves.step == curves.step.max())].set_index(["context", "arm"])
    final["filter_action_tv_similarity"] = [filt.loc[(c, a), "action_tv_similarity"] for c, a in zip(final.context, final.arm)]
    final.to_csv(output / "final.csv", index=False)
    last = objectives[objectives.step == objectives.step.max()] if not objectives.empty else objectives
    lines = ["# EM-CSCG vs projected-gradient CSCG", "",
             "Level 23, seed 0, 20,000 training steps (rand_prob 0.3), ground-truth clone allocation, RandomState(0)",
             "initial tensor, forgetting 1.0, pseudocount 0.002, five local iterations per arrival. Context = transitions",
             "in the long-EM window (W = context + 1 observations) smoothed from the carried historical filter belief.",
             "Both arms minimise the same per-arrival objective from the same running statistics recursion",
             "(docs/gradient_cscg_formulation.md); only the optimiser differs.", "",
             f"Primary evaluation: {args.walks} walks × {args.walk_length} steps (rand_prob {args.rand_prob:g}), "
             "training-matched decoder (carried filter + lag W − 1). NLL is the carried filter's causal loss.", "",
             "## Final structural metrics (training-matched decoder)", ""]
    lines += table(final, ["context", "label", "action_tv_similarity", "action_no_self_loop_tv_similarity",
                           "action_edge_f1_auc", "latent_precision", "latent_recall", "eval_nll_bps",
                           "active_clones", "transition_entropy_bits"],
                   [None, None, ".4f", ".4f", ".4f", ".4f", ".4f", ".4f", ".0f", ".3f"])
    lines += ["", "## Computational cost and numerical health", "",
              "Wall-clock includes contention: all arms trained concurrently on one 10-core machine. Window passes",
              "count E-step forward-backward passes plus forward-only Armijo trial evaluations.", ""]
    lines += table(final, ["context", "label", "train_seconds", "seconds_per_arrival_ms", "evaluations_per_iteration",
                           "backtracks", "rejected_fraction", "exact_zeros", "min_T", "row_sum_error"],
                   [None, None, ".1f", ".2f", ".2f", ".0f", ".3f", ".0f", ".2e", ".1e"])
    if not last.empty:
        last = last.assign(label=last.arm.map(label), order=last.arm.map(arm_key)).sort_values(["context", "order"])
        lines += ["", f"## Per-arrival objective at arrival {int(last.step.max())}", "",
                  "Mass-normalised objective G before the first and after the fifth local iteration. Each arm evaluates",
                  "G on its own statistics and boundary belief (trajectories differ), so compare the reduction, not levels.", ""]
        lines += table(last.assign(reduction=last.objective_start - last.objective_end),
                       ["context", "label", "objective_start", "objective_end", "reduction", "backtracks", "rejected"],
                       [None, None, ".6f", ".6f", ".2e", ".0f", ".0f"])
    lines += ["", "## Failures", ""]
    lines += (["None."] if failures.empty else
              table(failures.assign(label=failures.arm.map(label)), ["context", "label", "last_checkpoint", "error"],
                    [None, None, None, None]))
    lines += ["", "Decoder sensitivity (pure filter, lag 0) final action similarity is in final.csv",
              "(`filter_action_tv_similarity`); it is not the primary evaluation.", "",
              "Figures: metrics_vs_{steps,iterations,evaluations,wallclock}.png compare EM with the primary",
              "reference-style gradCSCG arms; the *_secondary.png figures compare EM with projected Adam and PGD."]
    (output / "results.md").write_text("\n".join(lines) + "\n")
    print(f"Wrote {output / 'results.md'}")


if __name__ == "__main__":
    main()
