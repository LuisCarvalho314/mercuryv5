"""Report a jepa_mix sweep (scripts/run_cscg_jepa_mix_sweep.sh): read-only over saved runs.

  PYTHONPATH=.:src .venv/bin/python scripts/analysis/cscg_jepa_mix_sweep_report.py --sweep DIR \
      [--walks 100 --walk-length 10000]

Inputs per arm directory: rows.json (runner evaluation + recorded support margins),
training.json (per-checkpoint arrival-objective terms and per-row count contributions),
snapshots.npz (checkpoint tensors), failure.json (recorded support failures), and the
training-matched canonical metrics for the requested evaluation budget
(<evaluation dir>/canonical_metrics_all.csv). Writes DIR/report/ (DIR/report_walks<W>x<L>/
for a non-canonical budget).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.analysis.cscg_distillation_matched_eval import CANONICAL_BUDGET, OUTPUT, output_name

METRICS = ("action_tv_similarity", "action_no_self_loop_tv_similarity", "action_edge_f1_auc",
           "action_no_self_loop_edge_f1_auc", "latent_precision", "latent_recall", "eval_nll_bps",
           "state_coverage", "unmatched_mass", "clone_utilisation")
LOWER_IS_BETTER = ("eval_nll_bps", "unmatched_mass")
BASELINES = {"short": ("Short EM", "#eb6834"), "long": ("Long EM", "#1baf7a"),
             "legacy": ("Legacy coupled, eta=1", "#6250d6")}
RHO_RAMP = ["#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"]   # one hue, light -> dark


def arm_of(directory, method):
    name = directory.name
    if name == "em_baselines":
        return method, None
    if name == "legacy_eta1":
        return "legacy", None
    if name.startswith("jepa_mix"):
        return name, float(name[len("jepa_mix"):])
    return name, None


def structural_change(path):
    """Mean row TV between consecutive checkpoint tensors, keyed by the later step."""
    with np.load(path) as saved:
        steps, Ts = saved["steps"], saved["T"]
    tv = .5 * np.abs(np.diff(Ts, axis=0)).sum(-1).mean(axis=(1, 2))
    return dict(zip(steps[1:].tolist(), tv.tolist()))


def load(sweep, evaluation):
    curves, rows_out, failures = [], [], []
    for directory in sorted(p for p in sweep.iterdir() if p.is_dir() and not p.name.startswith("report")):
        matched_path = directory / evaluation / "canonical_metrics_all.csv"
        matched = pd.read_csv(matched_path) if matched_path.exists() else pd.DataFrame()
        eval_seconds = pd.Series(dtype=float)
        if not matched.empty:
            if "evaluation_seconds" in matched:
                eval_seconds = matched.groupby(["level", "seed", "method", "decoder"]).evaluation_seconds.sum()
            common = matched[matched.decoder == "common_lag1"]
            matched = matched[matched.decoder == "training_matched"]
        else:
            common = matched
        for failure in directory.glob("level=*/seed=*/*/failure.json"):
            f = json.loads(failure.read_text())
            arm, rho = arm_of(directory, f["method"])
            failures.append(dict(arm=arm, rho=rho, level=f["level"], seed=f["seed"], stage=f["stage"],
                                 error=f["error"], last_checkpoint=f["last_checkpoint"]))
        for path in sorted(directory.glob("level=*/seed=*/*/rows.json")):
            rows = json.loads(path.read_text())
            training = json.loads((path.parent / "training.json").read_text())
            diag = {d["step"]: d for d in training["diagnostics"]}
            level, seed, method = rows[0]["level"], rows[0]["seed"], rows[0]["method"]
            arm, rho = arm_of(directory, method)
            change = structural_change(path.parent / "snapshots.npz")
            seconds = {decoder: float(eval_seconds.get((level, seed, method, decoder), np.nan))
                       for decoder in ("training_matched", "filter")}
            first = True
            for row in rows:
                step = row["step"]
                record = dict(arm=arm, rho=rho, level=level, seed=seed, step=step,
                              train_seconds=row["train_seconds"], min_T=row.get("min_T"),
                              min_teacher_T=row.get("min_teacher_T"),
                              min_student_evidence=row.get("min_student_evidence"),
                              min_teacher_evidence=row.get("min_teacher_evidence"),
                              structural_change=change.get(step),
                              # Whole-run evaluation time, carried once per case (on its first row).
                              eval_seconds_matched=seconds["training_matched"] if first else np.nan,
                              eval_seconds_filter=seconds["filter"] if first else np.nan)
                first = False
                d = diag.get(step)
                if d is not None:
                    final = d["local_iterations"][-1]
                    mix = final["jepa_mix"]
                    nll = final["data_nll_after"]
                    record.update(effective_mix=mix, objective=final["objective_after"],
                                  prior_term=final["prior_term_after"], data_nll=nll,
                                  jepa_term=final["jepa_term_after"],
                                  weighted_data_nll=(1 - mix) * nll if nll is not None else 0.,
                                  weighted_jepa=mix * final["jepa_term_after"], predictive_kl=d["predictive_kl"],
                                  posterior_kl=d["posterior_kl"], replay_nll=d["replay_nll"],
                                  min_window_student_prediction=d["min_window_student_prediction"],
                                  min_window_teacher_target=d["min_window_teacher_target"],
                                  iteration_max_increase=max(i["objective_after"] - i["objective_before"]
                                                             for i in d["local_iterations"]))
                    em_rows, jepa_rows = np.asarray(d["em_row_mass"]), np.asarray(d["jepa_row_mass"])
                    history = np.asarray(d["history_row_mass"])
                    touched = jepa_rows > 0
                    new = em_rows + jepa_rows
                    share = np.divide(jepa_rows, new, out=np.zeros_like(new), where=new > 0)
                    record.update(rows_total=int(share.size), rows_jepa_touched=int(touched.sum()),
                                  jepa_share_median_touched=float(np.median(share[touched])) if touched.any() else None,
                                  rows_jepa_dominant=int((share > .5).sum()),
                                  em_mass_on_untouched_rows=float(em_rows[~touched].sum()),
                                  new_to_history_mass=float(new.sum() / history.sum()))
                    for (a, i), e in np.ndenumerate(em_rows):
                        rows_out.append(dict(arm=arm, rho=rho, level=level, seed=seed, step=step, action=a, source=i,
                                             em_mass=float(e), jepa_mass=float(jepa_rows[a, i])))
                hit = matched[(matched.level == level) & (matched.seed == seed) & (matched.method == method)
                              & (matched.step == step)] if not matched.empty else matched
                if len(hit) == 1:
                    record.update({m: float(hit.iloc[0][m]) for m in METRICS})
                same = common[(common.level == level) & (common.seed == seed) & (common.method == method)
                              & (common.step == step)] if not common.empty else common
                if len(same) == 1:
                    record.update({"common_" + m: float(same.iloc[0][m]) for m in METRICS})
                curves.append(record)
    return pd.DataFrame(curves), pd.DataFrame(rows_out), pd.DataFrame(failures)


def order(arms):
    fixed = [a for a in ("short", "long", "legacy") if a in arms]
    return fixed + sorted((a for a in arms if a.startswith("jepa_mix")), key=lambda a: float(a[8:]))


def label(arm):
    return BASELINES[arm][0] if arm in BASELINES else f"jepa_mix={arm[8:]}"


def style(arm, arms):
    if arm in BASELINES:
        return dict(color=BASELINES[arm][1], linestyle="--")
    mixes = [a for a in order(arms) if a.startswith("jepa_mix")]
    ramp = RHO_RAMP[-len(mixes):] if len(mixes) <= len(RHO_RAMP) else RHO_RAMP
    return dict(color=ramp[mixes.index(arm) % len(ramp)], linestyle="-")


def convergence(part, metric):
    """Integrated mean over observations and t90: first checkpoint within 10% of final change."""
    part = part.sort_values("step")
    steps, values = part.step.to_numpy(float), part[metric].to_numpy(float)
    auc = float(np.trapezoid(values, steps) / steps[-1])
    target = values[0] + .9 * (values[-1] - values[0])
    reached = values <= target if values[-1] < values[0] else values >= target
    return auc, float(steps[np.argmax(reached)]) if reached.any() else float("nan")


def summarise(curves):
    rows = []
    for (arm, level, seed), part in curves.groupby(["arm", "level", "seed"]):
        part = part.sort_values("step"); final = part.iloc[-1]
        value = dict(arm=arm, level=level, seed=seed, final_step=int(final.step), train_seconds=final.train_seconds,
                     eval_seconds_matched=part.eval_seconds_matched.sum(min_count=1),
                     eval_seconds_filter=part.eval_seconds_filter.sum(min_count=1))
        late = part[part.step >= final.step / 2]
        value["late_mean_structural_change"] = float(late.structural_change.mean())
        value["late_max_structural_change"] = float(late.structural_change.max())
        for metric in METRICS:
            if metric in part and part[metric].notna().all():
                value["final_" + metric] = final[metric]
                value["auc_" + metric], value["t90_" + metric] = convergence(part, metric)
                # Late-phase variability: SD over the second half of the checkpoints (0 = settled).
                late = part[part.step >= final.step / 2][metric]
                value["late_sd_" + metric] = float(late.std(ddof=0))
            if "common_" + metric in part and pd.notna(final.get("common_" + metric)):
                value["final_common_" + metric] = final["common_" + metric]
        for col in ("min_T", "min_teacher_T", "min_student_evidence", "min_teacher_evidence",
                    "min_window_student_prediction", "min_window_teacher_target"):
            if col in part and part[col].notna().any():
                value["overall_" + col] = float(part[col].min())
        for col in ("effective_mix", "objective", "prior_term", "data_nll", "jepa_term", "weighted_data_nll",
                    "weighted_jepa", "predictive_kl", "posterior_kl", "replay_nll", "rows_jepa_touched", "rows_total",
                    "rows_jepa_dominant", "jepa_share_median_touched", "em_mass_on_untouched_rows",
                    "new_to_history_mass", "iteration_max_increase"):
            if col in part and pd.notna(final.get(col)):
                value["final_" + col] = final[col]
        if "iteration_max_increase" in part and part.iteration_max_increase.notna().any():
            value["worst_iteration_increase"] = float(part.iteration_max_increase.max())
        rows.append(value)
    return pd.DataFrame(rows)


def fmt(mean, sd, digits=4, sci=False):
    if pd.isna(mean):
        return "—"
    body = f"{mean:.2e}" if sci else f"{mean:.{digits}f}"
    return body if pd.isna(sd) or sd == 0 else body + (f" ± {sd:.1e}" if sci else f" ± {sd:.{digits}f}")


def table(per_seed, columns, arms, sci=()):
    lines = ["| Level | Arm | " + " | ".join(h for h, _ in columns) + " |",
             "| --- | --- | " + " | ".join("---:" for _ in columns) + " |"]
    for level in sorted(per_seed.level.unique()):
        for arm in arms:
            part = per_seed[(per_seed.level == level) & (per_seed.arm == arm)]
            if part.empty:
                continue
            cells = [fmt(part[c].mean(), part[c].std(ddof=0) if len(part) > 1 else np.nan, sci=c in sci)
                     if c in part else "—" for _, c in columns]
            lines.append(f"| {level} | {label(arm)} | " + " | ".join(cells) + " |")
    return lines


def figures(curves, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    arms = order(set(curves.arm)); levels = sorted(curves.level.unique())
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#8a8a84",
                         "axes.labelcolor": "#3d3d3a", "xtick.color": "#5f5e5a", "ytick.color": "#5f5e5a",
                         "font.size": 9, "lines.linewidth": 2})

    def panel(filename, metrics, log=()):
        present = [m for m in metrics if m in curves and curves[m].notna().any()]
        if not present:
            return
        fig, axes = plt.subplots(len(levels), len(present), figsize=(4.2 * len(present), 3.4 * len(levels)),
                                 squeeze=False)
        for i, level in enumerate(levels):
            for j, (metric, title) in enumerate(present_titles(present, metrics)):
                ax = axes[i, j]
                for arm in arms:
                    part = curves[(curves.level == level) & (curves.arm == arm)].dropna(subset=[metric])
                    if part.empty:
                        continue
                    agg = part.groupby("step")[metric].agg(["mean", "min", "max"])
                    st = style(arm, arms)
                    ax.plot(agg.index, agg["mean"], label=label(arm), **st)
                    if part.seed.nunique() > 1:
                        ax.fill_between(agg.index, agg["min"], agg["max"], color=st["color"], alpha=.12, linewidth=0)
                if metric in log:
                    ax.set_yscale("log")
                ax.set(title=f"Level {level}: {title}", xlabel="Training observations")
                ax.grid(alpha=.25, linewidth=.6)
        axes[0, 0].legend(fontsize=7, frameon=False)
        fig.tight_layout(); fig.savefig(output / f"{filename}.png", dpi=160); plt.close(fig)

    def present_titles(present, metrics):
        return [(m, metrics[m]) for m in metrics if m in present]

    panel("learning_curves", {"action_tv_similarity": "action TV similarity (higher better)",
                              "action_no_self_loop_tv_similarity": "non-self-loop TV similarity",
                              "eval_nll_bps": "held-out NLL, bits/step (lower better)",
                              "latent_precision": "latent precision"})
    panel("support_margins", {"min_T": "min student T entry", "min_teacher_T": "min teacher T entry",
                              "min_student_evidence": "min student p(x_t | x_<t) since last checkpoint",
                              "min_window_student_prediction": "min window prediction p_u(j; T)"},
          log=("min_T", "min_teacher_T", "min_student_evidence", "min_window_student_prediction"))
    panel("teacher_matching", {"predictive_kl": "teacher-matching loss L (predictive KL)",
                               "replay_nll": "observation-likelihood component -log Z",
                               "posterior_kl": "latent component KL(q || p/Z)",
                               "data_nll": "short arrival NLL -log p(x_t | x_<t)"})
    panel("structure_and_metrics", {"structural_change": "mean row TV between checkpoints",
                                    "action_edge_f1_auc": "action-conditioned edge F1 AUC",
                                    "latent_precision": "latent purity (precision)",
                                    "latent_recall": "latent concentration (recall)"},
          log=("structural_change",))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep", type=Path, required=True)
    parser.add_argument("--walks", type=int, default=CANONICAL_BUDGET[0])
    parser.add_argument("--walk-length", type=int, default=CANONICAL_BUDGET[1])
    parser.add_argument("--rand-prob", type=float, default=CANONICAL_BUDGET[2],
                        help="Random-action probability of the evaluation walks")
    args = parser.parse_args()
    budget = (args.walks, args.walk_length, args.rand_prob)
    sweep = args.sweep.resolve()
    output = sweep / output_name(*budget).replace(OUTPUT, "report")
    output.mkdir(exist_ok=True)
    curves, row_masses, failures = load(sweep, output_name(*budget))
    curves.to_csv(output / "curves.csv", index=False)
    row_masses.to_csv(output / "row_contributions.csv", index=False)
    failures.to_csv(output / "failures.csv", index=False)
    per_seed = summarise(curves)
    per_seed.to_csv(output / "per_seed_summary.csv", index=False)
    arms = order(set(curves.arm))
    figures(curves, output)
    scope = ", ".join(f"level {l}: seeds {sorted(curves[curves.level == l].seed.unique().tolist())}"
                      for l in sorted(curves.level.unique()))
    reduced = budget[:2] != CANONICAL_BUDGET[:2]
    lines = [f"# Accumulated predictive JEPA-CSCG vs EM: {sweep.name}", "",
             f"Scope: {scope}; final step {int(curves.step.max()):,}. Means ± population SD over seeds.",
             f"Canonical metrics use the training-matched decoder; evaluation budget {budget[0]} walks × "
             f"{budget[1]:,} steps = {budget[0] * budget[1]:,} steps per checkpoint (evaluation-v5 seed 0,",
             f"rand_prob {budget[2]:g}), identical walks for every arm and checkpoint.",
             *(["", f"**Non-canonical evaluation rand_prob {budget[2]:g}** (canonical 0.1)."]
               if budget[2] != CANONICAL_BUDGET[2] else []),
             *(["", "**Reduced evaluation budget** (canonical: 100 × 10,000). For rapid methodological comparison",
                "between these arms only; not comparable with full-budget results."] if reduced else []),
             "", "All margins and contributions are recorded, never fed back into learning.", "",
             "## Support failures", ""]
    if failures.empty:
        lines.append("None: no student or teacher filter, and no replay target, met zero predictive support.")
    else:
        lines += ["| Arm | Level | Seed | Stage | Last checkpoint | Error |", "| --- | ---: | ---: | --- | ---: | --- |"]
        lines += [f"| {label(r.arm)} | {r.level} | {r.seed} | {r.stage} | {r.last_checkpoint} | {r.error} |"
                  for r in failures.itertuples()]
    lines += ["", "## Final canonical metrics (training-matched decoder)", ""]
    lines += table(per_seed, [("Action TV sim.", "final_action_tv_similarity"),
                              ("Non-self TV sim.", "final_action_no_self_loop_tv_similarity"),
                              ("Edge F1 AUC", "final_action_edge_f1_auc"),
                              ("Non-self edge F1", "final_action_no_self_loop_edge_f1_auc"),
                              ("Purity (precision)", "final_latent_precision"),
                              ("Concentration (recall)", "final_latent_recall"),
                              ("NLL (bits)", "final_eval_nll_bps"), ("Coverage", "final_state_coverage"),
                              ("Unmatched mass", "final_unmatched_mass")], arms)
    if "final_common_action_tv_similarity" in per_seed:
        lines += ["", "## Final canonical metrics (common decoder: carried filter + lag-one smoothing, every arm)", "",
                  "One decoder configuration for all arms, so differences reflect the learned transition tensor rather",
                  "than each arm's inference context. Identical to the training-matched decoder except for long EM",
                  "(training-matched lag W - 1).", ""]
        lines += table(per_seed, [("Action TV sim.", "final_common_action_tv_similarity"),
                                  ("Non-self TV sim.", "final_common_action_no_self_loop_tv_similarity"),
                                  ("Edge F1 AUC", "final_common_action_edge_f1_auc"),
                                  ("Non-self edge F1", "final_common_action_no_self_loop_edge_f1_auc"),
                                  ("Purity (precision)", "final_common_latent_precision"),
                                  ("Concentration (recall)", "final_common_latent_recall"),
                                  ("NLL (bits)", "final_common_eval_nll_bps"), ("Coverage", "final_common_state_coverage"),
                                  ("Unmatched mass", "final_common_unmatched_mass")], arms)
    lines += ["", "## Learning trajectory and convergence", "",
              "Integrated mean = area under the checkpoint curve / final step. t90 = first checkpoint that",
              "has covered 90% of the change from step 0 to the final value (lower = faster); it is only",
              "meaningful when the late-phase SD (SD over checkpoints in the second half of training) is small.", ""]
    lines += table(per_seed, [("Integrated TV sim.", "auc_action_tv_similarity"),
                              ("t90 TV sim.", "t90_action_tv_similarity"),
                              ("Integrated non-self TV", "auc_action_no_self_loop_tv_similarity"),
                              ("t90 non-self TV", "t90_action_no_self_loop_tv_similarity"),
                              ("Late SD TV sim.", "late_sd_action_tv_similarity"),
                              ("Late SD non-self TV", "late_sd_action_no_self_loop_tv_similarity"),
                              ("Integrated NLL", "auc_eval_nll_bps"), ("t90 NLL", "t90_eval_nll_bps")], arms)
    lines += ["", "## Late-stage variability and structural oscillation (second half of training)", "",
              "Late SD = population SD over checkpoints with step >= final/2. Structural change = mean over rows of",
              "TV(T_k[a,i,:], T_{k-1}[a,i,:]) between consecutive checkpoint tensors.", ""]
    lines += table(per_seed, [("Late SD TV sim.", "late_sd_action_tv_similarity"),
                              ("Late SD non-self TV", "late_sd_action_no_self_loop_tv_similarity"),
                              ("Late SD edge F1", "late_sd_action_edge_f1_auc"),
                              ("Late SD purity", "late_sd_latent_precision"),
                              ("Late SD NLL", "late_sd_eval_nll_bps"),
                              ("Late mean struct. change", "late_mean_structural_change"),
                              ("Late max struct. change", "late_max_structural_change")], arms,
                   sci={"late_mean_structural_change", "late_max_structural_change"})
    lines += ["", "## Runtime (seconds, measured separately)", "",
              "Train = learner wall time over the full stream, excluding evaluation. Evaluation = sum over all",
              "checkpoints of the training-matched decoder pass, and of the lag-0 filter check.", ""]
    lines += table(per_seed, [("Train", "train_seconds"), ("Eval, matched decoder", "eval_seconds_matched"),
                              ("Eval, filter check", "eval_seconds_filter")], arms)
    lines += ["", "## Minimum probabilities and support margins (minimum over all checkpoints)", "",
              "Student/teacher evidence: smallest p(x_t | x_<t) of the carried filter at any training arrival.",
              "Window prediction: smallest p_u(j; T) over every window transition and destination clone at",
              "checkpoint arrivals; teacher target: smallest q_{u+1}(j) there.", ""]
    lines += table(per_seed, [("min T", "overall_min_T"), ("min teacher T", "overall_min_teacher_T"),
                              ("min student evidence", "overall_min_student_evidence"),
                              ("min teacher evidence", "overall_min_teacher_evidence"),
                              ("min window prediction", "overall_min_window_student_prediction"),
                              ("min teacher target", "overall_min_window_teacher_target")], arms,
                   sci={"overall_min_T", "overall_min_teacher_T", "overall_min_student_evidence",
                        "overall_min_teacher_evidence", "overall_min_window_student_prediction",
                        "overall_min_window_teacher_target"})
    lines += ["", "## Teacher matching and the per-arrival objective (final checkpoint, last local iteration)", "",
              "Teacher-matching loss L = mean_u KL(q_{u+1} || p_u(T)) over the current window = latent KL(q || p/Z)",
              "+ observation-likelihood component -log Z (replay NLL). Per-arrival objective F = -<prior + kappa,",
              "log T> + (1 - rho) NLL_t + rho L (lambda = 1). Worst increase = largest F_after - F_before over every",
              "recorded local iteration (MM descent check; <= 0 expected).", ""]
    lines += table(per_seed, [("L (pred. KL)", "final_predictive_kl"), ("obs. component", "final_replay_nll"),
                              ("latent KL", "final_posterior_kl"), ("NLL_t", "final_data_nll"),
                              ("(1-rho) NLL_t", "final_weighted_data_nll"), ("rho L", "final_weighted_jepa"),
                              ("F", "final_objective"), ("worst increase", "worst_iteration_increase")],
                   arms, sci={"worst_iteration_increase"})
    lines += ["", "## New evidence per transition row (final checkpoint)", "",
              "Per row (a, i) this arrival adds data mass (1 - rho) sum_j xi_aij and teacher mass rho sum_j D_aij",
              "to the running statistics; teacher share = teacher / (data + teacher). New-to-history = mass added",
              "by the arrival / mass of the pre-arrival statistics plus pseudocounts. Full per-row values at",
              "every checkpoint: row_contributions.csv.", ""]
    lines += table(per_seed, [("rows", "final_rows_total"), ("teacher-touched rows", "final_rows_jepa_touched"),
                              ("teacher-dominant rows", "final_rows_jepa_dominant"),
                              ("median share (touched)", "final_jepa_share_median_touched"),
                              ("data mass, untouched rows", "final_em_mass_on_untouched_rows"),
                              ("new / history mass", "final_new_to_history_mass")], arms,
                   sci={"final_em_mass_on_untouched_rows", "final_new_to_history_mass"})
    lines += ["", "Figures: learning_curves.png, support_margins.png, teacher_matching.png, structure_and_metrics.png",
              "(seed range shaded). Tables: curves.csv, per_seed_summary.csv, row_contributions.csv, failures.csv."]
    (output / "results.md").write_text("\n".join(lines) + "\n")
    print(f"Wrote {output / 'results.md'}")


if __name__ == "__main__":
    main()
