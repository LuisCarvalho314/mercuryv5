"""Report a jepa_mix sweep (scripts/run_cscg_jepa_mix_sweep.sh): read-only over saved runs.

  PYTHONPATH=.:src .venv/bin/python scripts/analysis/cscg_jepa_mix_sweep_report.py --sweep DIR

Inputs per arm directory: rows.json (runner evaluation + recorded support margins),
training.json (per-checkpoint EM/JEPA terms and per-row count contributions),
failure.json (recorded support failures), and the training-matched canonical metrics
evaluation_v5_training_matched/canonical_metrics_all.csv. Writes DIR/report/.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

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


def load(sweep):
    curves, rows_out, failures = [], [], []
    for directory in sorted(p for p in sweep.iterdir() if p.is_dir() and p.name != "report"):
        matched_path = directory / "evaluation_v5_training_matched" / "canonical_metrics_all.csv"
        matched = pd.read_csv(matched_path) if matched_path.exists() else pd.DataFrame()
        if not matched.empty:
            matched = matched[matched.decoder == "training_matched"]
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
            for row in rows:
                step = row["step"]
                record = dict(arm=arm, rho=rho, level=level, seed=seed, step=step,
                              train_seconds=row["train_seconds"], min_T=row.get("min_T"),
                              min_teacher_T=row.get("min_teacher_T"),
                              min_student_evidence=row.get("min_student_evidence"),
                              min_teacher_evidence=row.get("min_teacher_evidence"))
                d = diag.get(step)
                if d is not None:
                    final = d["local_iterations"][-1]
                    mix = final["jepa_mix"]
                    em, jepa = final["em_term_after"], final["jepa_term_after"]
                    record.update(effective_mix=mix, em_term=em, jepa_term=jepa,
                                  weighted_em=(1 - mix) * em if em is not None else 0.,
                                  weighted_jepa=mix * jepa, predictive_kl=d["predictive_kl"],
                                  posterior_kl=d["posterior_kl"], replay_nll=d["replay_nll"],
                                  min_window_student_prediction=d["min_window_student_prediction"],
                                  min_window_teacher_target=d["min_window_teacher_target"],
                                  iteration_max_increase=max(i["objective_after"] - i["objective_before"]
                                                             for i in d["local_iterations"]))
                    em_rows, jepa_rows = np.asarray(d["em_row_mass"]), np.asarray(d["jepa_row_mass"])
                    touched = jepa_rows > 0
                    share = jepa_rows / (em_rows + jepa_rows)
                    record.update(rows_total=int(share.size), rows_jepa_touched=int(touched.sum()),
                                  jepa_share_median_touched=float(np.median(share[touched])) if touched.any() else None,
                                  jepa_share_min_touched=float(share[touched].min()) if touched.any() else None,
                                  rows_jepa_dominant=int((share > .5).sum()),
                                  em_mass_on_untouched_rows=float(em_rows[~touched].sum()))
                    for (a, i), e in np.ndenumerate(em_rows):
                        rows_out.append(dict(arm=arm, rho=rho, level=level, seed=seed, step=step, action=a, source=i,
                                             em_mass=float(e), jepa_mass=float(jepa_rows[a, i])))
                hit = matched[(matched.level == level) & (matched.seed == seed) & (matched.method == method)
                              & (matched.step == step)] if not matched.empty else matched
                if len(hit) == 1:
                    record.update({m: float(hit.iloc[0][m]) for m in METRICS})
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
        value = dict(arm=arm, level=level, seed=seed, final_step=int(final.step), train_seconds=final.train_seconds)
        for metric in METRICS:
            if metric in part and part[metric].notna().all():
                value["final_" + metric] = final[metric]
                value["auc_" + metric], value["t90_" + metric] = convergence(part, metric)
                # Late-phase variability: SD over the second half of the checkpoints (0 = settled).
                late = part[part.step >= final.step / 2][metric]
                value["late_sd_" + metric] = float(late.std(ddof=0))
        for col in ("min_T", "min_teacher_T", "min_student_evidence", "min_teacher_evidence",
                    "min_window_student_prediction", "min_window_teacher_target"):
            if col in part and part[col].notna().any():
                value["overall_" + col] = float(part[col].min())
        for col in ("effective_mix", "em_term", "jepa_term", "weighted_em", "weighted_jepa", "predictive_kl",
                    "posterior_kl", "replay_nll", "rows_jepa_touched", "rows_total", "rows_jepa_dominant",
                    "jepa_share_median_touched", "jepa_share_min_touched", "em_mass_on_untouched_rows",
                    "iteration_max_increase"):
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
    panel("objective_terms", {"em_term": "normalised EM term -<B,log T>/<B,1>",
                              "jepa_term": "JEPA predictive loss L",
                              "weighted_em": "(1 - rho) x EM term", "weighted_jepa": "rho x JEPA term"},
          log=("weighted_em",))
    panel("count_contributions", {"jepa_share_median_touched": "median JEPA share, JEPA-touched rows",
                                  "rows_jepa_dominant": "rows with JEPA share > 0.5",
                                  "em_mass_on_untouched_rows": "EM mass on rows without JEPA counts"},
          log=("em_mass_on_untouched_rows",))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep", type=Path, required=True)
    args = parser.parse_args()
    sweep = args.sweep.resolve(); output = sweep / "report"; output.mkdir(exist_ok=True)
    curves, row_masses, failures = load(sweep)
    curves.to_csv(output / "curves.csv", index=False)
    row_masses.to_csv(output / "row_contributions.csv", index=False)
    failures.to_csv(output / "failures.csv", index=False)
    per_seed = summarise(curves)
    per_seed.to_csv(output / "per_seed_summary.csv", index=False)
    arms = order(set(curves.arm))
    figures(curves, output)
    scope = ", ".join(f"level {l}: seeds {sorted(curves[curves.level == l].seed.unique().tolist())}"
                      for l in sorted(curves.level.unique()))
    lines = [f"# JEPA-dominated optimisation sweep: {sweep.name}", "",
             f"Scope: {scope}; final step {int(curves.step.max()):,}. Means ± population SD over seeds.",
             "Canonical metrics use the training-matched decoder (evaluation-v5 walks, 100 × 10,000 steps).",
             "The learning algorithm is unchanged; all margins and contributions are recorded, never fed back.", "",
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
                              ("Latent precision", "final_latent_precision"), ("Latent recall", "final_latent_recall"),
                              ("NLL (bits)", "final_eval_nll_bps"), ("Coverage", "final_state_coverage"),
                              ("Unmatched mass", "final_unmatched_mass")], arms)
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
                              ("Integrated NLL", "auc_eval_nll_bps"), ("t90 NLL", "t90_eval_nll_bps"),
                              ("Train seconds", "train_seconds")], arms)
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
    lines += ["", "## EM and JEPA objective contributions (final checkpoint, last local iteration)", "",
              "EM term = -<B,log T>/<B,1> (normalised); JEPA term = predictive loss L. The objective is",
              "(1 - rho) EM + rho L; legacy uses its effective rho_t = w/(<B,1> + w). Worst increase = largest",
              "objective_after - objective_before over every recorded local iteration (MM descent check).", ""]
    lines += table(per_seed, [("rho (effective)", "final_effective_mix"), ("EM term", "final_em_term"),
                              ("JEPA term", "final_jepa_term"), ("(1-rho) EM", "final_weighted_em"),
                              ("rho L", "final_weighted_jepa"), ("latent KL", "final_posterior_kl"),
                              ("replay NLL", "final_replay_nll"), ("worst increase", "worst_iteration_increase")],
                   arms, sci={"final_weighted_em", "worst_iteration_increase"})
    lines += ["", "## Effective count contributions per transition row (final checkpoint)", "",
              "Per row (a, i): EM mass (1 - rho) sum_j B_aij / <B,1> and JEPA mass rho sum_j D_aij; JEPA share =",
              "JEPA / (EM + JEPA). Rows outside the window's source support receive EM counts only. Full per-row",
              "values at every checkpoint: row_contributions.csv.", ""]
    lines += table(per_seed, [("rows", "final_rows_total"), ("JEPA-touched rows", "final_rows_jepa_touched"),
                              ("JEPA-dominant rows", "final_rows_jepa_dominant"),
                              ("median share (touched)", "final_jepa_share_median_touched"),
                              ("min share (touched)", "final_jepa_share_min_touched"),
                              ("EM mass, untouched rows", "final_em_mass_on_untouched_rows")], arms,
                   sci={"final_em_mass_on_untouched_rows"})
    lines += ["", "Figures: learning_curves.png, support_margins.png, objective_terms.png, count_contributions.png",
              "(seed range shaded). Tables: curves.csv, per_seed_summary.csv, row_contributions.csv, failures.csv."]
    (output / "results.md").write_text("\n".join(lines) + "\n")
    print(f"Wrote {output / 'results.md'}")


if __name__ == "__main__":
    main()
