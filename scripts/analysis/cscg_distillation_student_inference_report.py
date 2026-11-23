"""Canonical-metric report: filtered versus smoothed student in online JEPA-CSCG.

Reads two completed run_cscg_distillation_experiment.py outputs that differ only
in --student-inference, verifies they are paired, and writes canonical metrics,
paired differences and plots to <setting>/canonical_metrics. Metrics come from
the same extractor as cscg_distillation_canonical_report.py (evaluation-v5
postprocessing). No training, decoding or alignment is run.

PYTHONPATH=.:src .venv/bin/python scripts/analysis/cscg_distillation_student_inference_report.py \
    --filter <setting>/filter --smooth <setting>/smooth [--reference <historical filtered dir>]

Cross-setting overview of completed reports:

PYTHONPATH=.:src .venv/bin/python scripts/analysis/cscg_distillation_student_inference_report.py \
    --overview <output dir> --setting "Label=<setting>/canonical_metrics" ...
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.analysis.cscg_distillation_canonical_report import (
    LABELS, METHODS, METRICS, METRIC_LABELS, PRIMARY, canonical_metrics, final_f1_curve, residual_plots, summarise,
)

ROOT = Path(__file__).resolve().parents[2]
INFERENCE = ("filter", "smooth")
INFERENCE_LABELS = dict(filter="filtered student", smooth="smoothed student")
LINESTYLE = dict(filter="--", smooth="-")
# Validated categorical slots (dataviz reference palette, light surface), fixed per method.
COLORS = dict(zip(METHODS, ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4")))
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e6e5e0"
METRIC_LABELS = {**METRIC_LABELS, "latent_recall": "Latent concentration (canonical recall)",
                 "latent_precision": "Latent purity", "eval_nll_bps": "Held-out NLL (bits/symbol)",
                 "state_coverage": "Physical-state coverage", "unmatched_mass": "Unmatched target mass",
                 "clone_utilisation": "Clone utilisation"}
LOWER_IS_BETTER = ("eval_nll_bps", "unmatched_mass")
HEADLINE = (*PRIMARY, "latent_precision", "latent_recall", "eval_nll_bps")
UPDATE_KEYS = ("transitions", "base_msteps", "refinements", "em_passes", "penalty_updates",
               "statistics_commits", "teacher_updates")
NO_REPLAY = ("short", "long")   # student smoothing provably cannot change these arms


def style(ax):
    ax.grid(color=GRID, linewidth=.8); ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.title.set_color(INK)


def load(root):
    """Saved rows, provenance, snapshots and canonical records for every run."""
    runs = {}
    for rows_path in sorted(root.glob("level=*/seed=*/*/rows.json")):
        folder, rows = rows_path.parent, json.loads(rows_path.read_text())
        key = (rows[0]["level"], rows[0]["seed"], rows[0]["method"])
        with np.load(folder / "snapshots.npz") as saved:
            T = saved["T"].copy()
        records = {}
        for row in rows:
            path = folder / "evaluations" / f"step={row['step']:04d}.json"
            records[row["step"]] = (path, json.loads(path.read_text()))
        runs[key] = dict(rows=rows, training=json.loads((folder / "training.json").read_text()), T=T, records=records)
    if not runs:
        raise ValueError(f"No completed runs in {root}")
    return runs


def verify(roots, runs, manifests, reference_root):
    if [manifests[i]["experiment"].get("student_inference", "filter") for i in INFERENCE] != list(INFERENCE):
        raise ValueError("Expected one filtered and one smoothed experiment")
    strip = lambda m: {**m, "experiment": {k: v for k, v in m["experiment"].items() if k != "student_inference"}}
    if strip(manifests["filter"]) != strip(manifests["smooth"]):
        raise ValueError("Experiments differ in more than student_inference")
    expected = len(manifests["filter"]["levels"]) * len(manifests["filter"]["seeds"]) * len(METHODS)
    if runs["filter"].keys() != runs["smooth"].keys() or len(runs["filter"]) != expected:
        raise ValueError("Incomplete or unpaired filtered/smoothed runs")
    final = lambda run: run["training"]["checkpoint_metadata"][str(max(map(int, run["training"]["checkpoint_metadata"])))]
    for key, f in runs["filter"].items():
        s = runs["smooth"][key]
        if f["training"]["provenance"] != s["training"]["provenance"]:
            raise ValueError(f"Unmatched trajectory, initial tensor, allocation or protocol: {key}")
        if any(final(f)[k] != final(s)[k] for k in UPDATE_KEYS):
            raise ValueError(f"Learning-update counts differ: {key}")
        if key[2] in NO_REPLAY and not np.array_equal(f["T"], s["T"]):
            raise ValueError(f"Smoothing changed an arm without a replay penalty: {key}")
    for path in roots["filter"].glob("evaluation_caches/*/walks/*.npz"):
        with np.load(path) as a, np.load(roots["smooth"] / path.relative_to(roots["filter"])) as b:
            if set(a.files) != set(b.files) or any(not np.array_equal(a[k], b[k]) for k in a.files):
                raise ValueError("Held-out walk arrays differ")
    checks = dict(paired_runs=len(runs["filter"]), update_counts_equal=True,
                  no_replay_arms_bitwise_equal=True, held_out_walks_equal=True)
    if reference_root:
        reference = {}
        for rows_path in sorted(reference_root.glob("level=*/seed=*/*/rows.json")):
            rows = json.loads(rows_path.read_text())
            key = (rows[0]["level"], rows[0]["seed"], rows[0]["method"])
            if key in runs["filter"]:
                with np.load(rows_path.parent / "snapshots.npz") as saved:
                    reference[key] = (json.loads((rows_path.parent / "training.json").read_text())["provenance"], saved["T"].copy())
        if not reference:
            raise ValueError("Reference shares no runs with the filtered experiment")
        for key, (provenance, T) in reference.items():
            if provenance != runs["filter"][key]["training"]["provenance"] or not np.array_equal(T, runs["filter"][key]["T"]):
                raise ValueError(f"Filtered run does not reproduce the reference bit for bit: {key}")
        checks.update(reference=str(reference_root.relative_to(ROOT) if reference_root.is_relative_to(ROOT) else reference_root),
                      reference_runs_bitwise_reproduced=len(reference),
                      reference_snapshots_compared=int(sum(len(T) for _, T in reference.values())))
    return checks


def collect(runs, manifests):
    curves, thresholds, finals, hashes = [], [], {}, {}
    for inference in INFERENCE:
        final_step = manifests[inference]["observations"]
        for (level, seed, method), run in runs[inference].items():
            smoothing = {d["step"]: d["source_smoothing_tv"] for d in run["training"]["diagnostics"] if "source_smoothing_tv" in d}
            for row in run["rows"]:
                path, record = run["records"][row["step"]]
                scores = canonical_metrics(record)
                assert np.isclose(scores["action_tv_similarity"], 1 - row["action_tv"])
                curves.append(dict(iterations=manifests[inference]["local_parameter_updates"], inference=inference,
                                   level=level, seed=seed, method=method, step=row["step"],
                                   train_seconds=row["train_seconds"], source_smoothing_tv=smoothing.get(row["step"], np.nan),
                                   **{m: scores[m] for m in METRICS}))
                hashes[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
                if row["step"] == final_step:
                    finals[(inference, level, seed, method)] = record
                    thresholds += [dict(inference=inference, level=level, seed=seed, method=method, **score)
                                   for score in final_f1_curve(record)]
    curves = pd.DataFrame(curves)
    seeds = pd.concat([summarise(curves[curves.inference == i].drop(columns="inference")).assign(inference=i)
                       for i in INFERENCE], ignore_index=True)
    smoothing = curves.groupby(["inference", "level", "seed", "method"]).source_smoothing_tv.mean().rename("mean_source_smoothing_tv")
    seeds = seeds.join(smoothing, on=["inference", "level", "seed", "method"])
    return curves, seeds, pd.DataFrame(thresholds), finals, hashes


def paired_differences(seeds):
    numeric = [f"{p}_{m}" for p in ("final", "time_mean") for m in METRICS] + ["train_seconds"]
    table = seeds.set_index(["inference", "level", "seed", "method"])
    rows = []
    def add(contrast, family, left, right, level, seed, method):
        for metric in numeric:
            rows.append(dict(contrast=contrast, family=family, level=level, seed=seed, method=method, metric=metric,
                             value=float(left[metric]), reference_value=float(right[metric]),
                             delta=float(left[metric] - right[metric])))
    for (level, seed), _ in seeds.groupby(["level", "seed"]):
        get = lambda i, m: table.loc[(i, level, seed, m)]
        for method in METHODS:
            add(f"smooth_minus_filter:{method}", "smoothing_effect", get("smooth", method), get("filter", method), level, seed, method)
        for inference in INFERENCE:
            for baseline in ("short", "nll_control", "long"):
                add(f"predictive_minus_{baseline}:{inference}", "jepa_effect", get(inference, "predictive"),
                    get(inference, baseline), level, seed, "predictive")
        for baseline in ("short", "nll_control"):   # difference in differences
            add(f"interaction_vs_{baseline}", "interaction",
                get("smooth", "predictive") - get("smooth", baseline),
                get("filter", "predictive") - get("filter", baseline), level, seed, "predictive")
    frame = pd.DataFrame(rows)
    lower = frame.metric.str.endswith(LOWER_IS_BETTER) | (frame.metric == "train_seconds")
    frame["benefit_delta"] = np.where(lower, -frame.delta, frame.delta)
    summary = frame.groupby(["family", "contrast", "level", "metric"]).agg(
        mean=("delta", "mean"), std=("delta", "std"), count=("delta", "count"),
        wins=("benefit_delta", lambda x: int((x > 1e-12).sum())),
        losses=("benefit_delta", lambda x: int((x < -1e-12).sum())),
        ties=("benefit_delta", lambda x: int((x.abs() <= 1e-12).sum()))).reset_index()
    return frame, summary


def plot_curves(curves, output, filename, panels, methods, title):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    levels = sorted(curves.level.unique())
    fig, axes = plt.subplots(len(levels), len(panels), figsize=(4.3 * len(panels), 3.5 * len(levels)), squeeze=False)
    for i, level in enumerate(levels):
        for j, metric in enumerate(panels):
            ax = axes[i, j]; style(ax)
            for method in methods:
                for inference in INFERENCE:
                    part = curves[(curves.level == level) & (curves.method == method) & (curves.inference == inference)]
                    agg = part.groupby("step")[metric].agg(["mean", "std"])
                    ax.plot(agg.index, agg["mean"], color=COLORS[method], linestyle=LINESTYLE[inference], linewidth=2)
                    if inference == "smooth":
                        sd = agg["std"].fillna(0)
                        ax.fill_between(agg.index, agg["mean"] - sd, agg["mean"] + sd, color=COLORS[method], alpha=.10, linewidth=0)
            ax.set_title(f"Level {level}: {METRIC_LABELS[metric]}", fontsize=9)
            ax.set_xlabel("Observations", fontsize=8, color=MUTED)
            if metric not in ("eval_nll_bps",):
                ax.set_ylim(0, 1)
    handles = [Line2D([], [], color=COLORS[m], linewidth=2, label=LABELS[m]) for m in methods]
    handles += [Line2D([], [], color=MUTED, linewidth=2, linestyle=LINESTYLE[i], label=INFERENCE_LABELS[i].capitalize())
                for i in INFERENCE]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False, fontsize=8)
    fig.suptitle(title, color=INK, fontsize=11)
    fig.tight_layout(rect=(0, .05, 1, .97))
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"{filename}.{suffix}", dpi=160)
    plt.close(fig)


def plot_final_f1(thresholds, output, final_step):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    levels = sorted(thresholds.level.unique())
    fig, axes = plt.subplots(len(levels), 2, figsize=(11, 3.6 * len(levels)), squeeze=False)
    for i, level in enumerate(levels):
        for j, include_self in enumerate([True, False]):
            ax = axes[i, j]; style(ax)
            for method in METHODS:
                for inference in INFERENCE:
                    part = thresholds[(thresholds.level == level) & (thresholds.method == method) &
                                      (thresholds.inference == inference) & (thresholds.include_self_loops == include_self)]
                    agg = part.groupby("threshold").edge_f1.mean()
                    ax.plot(agg.index, agg.values, color=COLORS[method], linestyle=LINESTYLE[inference], linewidth=2)
            policy = "with self-loops" if include_self else "without self-loops"
            ax.set_title(f"Level {level}: action-conditioned edge F1, {policy}", fontsize=9)
            ax.set_xlabel("Transition probability threshold", fontsize=8, color=MUTED)
            ax.set_ylabel(f"Edge F1 at observation {final_step:,}", fontsize=8, color=MUTED)
            ax.set_ylim(0, 1)
    handles = [Line2D([], [], color=COLORS[m], linewidth=2, label=LABELS[m]) for m in METHODS]
    handles += [Line2D([], [], color=MUTED, linewidth=2, linestyle=LINESTYLE[i], label=INFERENCE_LABELS[i].capitalize()) for i in INFERENCE]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, .05, 1, 1))
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"final_step_action_conditioned_edge_f1.{suffix}", dpi=160)
    plt.close(fig)


SMOOTHING_ROWS = [(f"smooth_minus_filter:{m}", f"{LABELS[m]}: smooth − filter", COLORS[m]) for m in METHODS]
BASELINE_LABELS = dict(short="Short EM", long="Long EM", nll_control="replay-NLL control")
JEPA_ROWS = [(f"predictive_minus_{b}:{i}", f"JEPA − {BASELINE_LABELS[b]}, {INFERENCE_LABELS[i]}", COLORS[b])
             for b in ("short", "long", "nll_control") for i in INFERENCE]


def plot_paired(differences, output, rows, filename, title):
    """Per-seed paired differences (dots) and their mean (black tick), final checkpoint.

    Hollow markers: filtered-student contrasts; filled: smoothed-student or smoothing contrasts.
    """
    import matplotlib.pyplot as plt
    levels = sorted(differences.level.unique())
    fig, axes = plt.subplots(len(levels), len(HEADLINE), figsize=(2.6 * len(HEADLINE) + 2.6, .5 * len(rows) * len(levels) + 1.6),
                             squeeze=False, sharey=True)
    for i, level in enumerate(levels):
        for j, metric in enumerate(HEADLINE):
            ax = axes[i, j]; style(ax); ax.grid(axis="y", visible=False)
            ax.axvline(0, color=MUTED, linewidth=1)
            for y, (contrast, _, color) in enumerate(rows):
                part = differences[(differences.level == level) & (differences.contrast == contrast) &
                                   (differences.metric == f"final_{metric}")]
                filled = "smooth" in contrast or contrast.startswith("smooth_minus")
                ax.scatter(part.delta, np.full(len(part), y), s=36, color=color if filled else "white",
                           edgecolor=color, linewidth=1.5, zorder=3)
                ax.plot([part.delta.mean()] * 2, [y - .3, y + .3], color=INK, linewidth=2, zorder=4)
            if i == 0:
                arrow = " (← better)" if metric in LOWER_IS_BETTER else " (better →)"
                ax.set_title(METRIC_LABELS[metric] + arrow, fontsize=8)
            if i == len(levels) - 1:
                ax.set_xlabel("Paired difference", fontsize=8, color=MUTED)
        axes[i, 0].set_yticks(range(len(rows)), [label for _, label, _ in rows], fontsize=8)
        axes[i, 0].set_ylabel(f"Level {level}", color=INK)
        axes[i, 0].invert_yaxis()
    fig.suptitle(title + " — final checkpoint, per seed (dots) and mean (black tick)", color=INK, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, .96))
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"{filename}.{suffix}", dpi=160)
    plt.close(fig)


def contrast_table(summary, family, order=None):
    part = summary[summary.family == family].set_index(["contrast", "level", "metric"])
    keys = sorted({k[:2] for k in part.index}, key=lambda k: (order.index(k[0]) if order else 0, k[1], k[0]))
    lines = ["Final checkpoint: mean ± SD, wins/n (wins = paired seeds where the left model is better).", "",
             "| Contrast | Level | " + " | ".join(METRIC_LABELS[m] for m in HEADLINE) + " |",
             "| --- | ---: |" + " ---: |" * len(HEADLINE)]
    for contrast, level in keys:
        cells = [part.loc[(contrast, level, f"final_{m}")] for m in HEADLINE]
        lines.append(f"| {contrast} | {level} | " + " | ".join(
            f"{c['mean']:+.4f} ± {c['std']:.4f} ({int(c.wins)}/{int(c['count'])})" for c in cells) + " |")
    return lines


def write_markdown(output, seeds, summary, checks, title, manifest):
    jepa_lines = contrast_table(summary, "jepa_effect", [c for c, _, _ in JEPA_ROWS])
    means = seeds.groupby(["level", "method", "inference"]).mean(numeric_only=True)
    fmt = lambda r, p: " | ".join(f"{r[p + m]:.4f}" for m in HEADLINE)
    head = ("| Level | Model | Student | " + " | ".join(METRIC_LABELS[m] for m in HEADLINE) + " |",
            "| ---: | --- | --- |" + " ---: |" * len(HEADLINE))
    lines = [f"# {title}", "",
             f"Setting: {manifest['local_parameter_updates']} local updates"
             f"{' (coupled auxiliary)' if manifest['experiment'].get('coupled_auxiliary') else ''}, "
             f"{manifest['clone_allocation']} clone allocation, {manifest['observations']:,} observations, "
             f"seeds {manifest['seeds']}.",
             "Canonical metrics reuse the evaluation-v5 extractor (`plot_config_metric_trajectories._metrics_from_record`).",
             "Transition similarity = 1 − occupancy-weighted canonical TV; edge F1 = normalised threshold AUC; both include",
             "unmatched-target penalties. Concentration is canonical latent recall. Lower is better only for NLL and unmatched mass.", "",
             "## Primary result: JEPA effect (predictive − baseline)", "",
             "The teacher contribution is JEPA − replay-NLL control (identical except for the teacher target);",
             "JEPA − Short EM is the total effect of the replay penalty; JEPA − Long EM compares against full-history EM.",
             "The filtered student is the original configuration; the smoothed student is a robustness check.", "",
             *jepa_lines, "",
             "## Verification", "", *[f"- {k}: {v}" for k, v in checks.items()], "",
             "## Final checkpoint (mean over seeds)", "", *head]
    lines += [f"| {l} | {LABELS[m]} | {i} | {fmt(r, 'final_')} |" for (l, m, i), r in means.iterrows()]
    lines += ["", "## Mean over the learning curve (time_mean)", "", *head]
    lines += [f"| {l} | {LABELS[m]} | {i} | {fmt(r, 'time_mean_')} |" for (l, m, i), r in means.iterrows()]
    lines += ["", "## Supplementary canonical endpoints (final)", "",
              "| Level | Model | Student | Physical-state coverage | Unmatched mass | Clone utilisation | Source smoothing TV |",
              "| ---: | --- | --- | ---: | ---: | ---: | ---: |"]
    lines += [f"| {l} | {LABELS[m]} | {i} | {r.final_state_coverage:.4f} | {r.final_unmatched_mass:.4f} | "
              f"{r.final_clone_utilisation:.4f} | " + ("–" if np.isnan(r.mean_source_smoothing_tv) else f"{r.mean_source_smoothing_tv:.4f}") + " |"
              for (l, m, i), r in means.iterrows()]
    for heading, family in (("Secondary: smoothing effect within each model (smooth − filter)", "smoothing_effect"),
                            ("Secondary: interaction, smoothed JEPA effect − filtered JEPA effect", "interaction")):
        lines += ["", f"## {heading}", "", *contrast_table(summary, family)]
    lines += ["", "## Files", "",
              "- `learning_curves.png/pdf`: four primary canonical metrics, every model; dashed = filtered, solid = smoothed (±1 SD band).",
              "- `jepa_filter_vs_smooth.png/pdf`: Short EM, replay-NLL control and JEPA only.",
              "- `purity_and_recall.png/pdf`, `nll_and_coverage.png/pdf`: representation metrics and held-out NLL.",
              "- `final_step_action_conditioned_edge_f1.png/pdf`: actual edge F1 against threshold at the final checkpoint.",
              "- `paired_jepa_effect.png/pdf` (primary): JEPA − Short EM / Long EM / replay-NLL control per seed, both students.",
              "- `paired_smoothing_effect.png/pdf` (secondary): smooth − filter within each model.",
              "- `residuals/`: canonical residual maps for every seed and student, with and without self-loops.",
              "- CSV: `learning_curves`, `per_seed_summary`, `summary`, `per_seed_paired_differences`, `paired_summary`, `final_step_f1_by_seed`."]
    (output / "results.md").write_text("\n".join(lines) + "\n")


def report(args):
    import matplotlib
    matplotlib.use("Agg")
    roots = dict(filter=args.filter.resolve(), smooth=args.smooth.resolve())
    output = (args.output or roots["smooth"].parent / "canonical_metrics").resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifests = {i: json.loads((roots[i] / "experiment.json").read_text()) for i in INFERENCE}
    runs = {i: load(roots[i]) for i in INFERENCE}
    checks = verify(roots, runs, manifests, args.reference.resolve() if args.reference else None)
    curves, seeds, thresholds, finals, hashes = collect(runs, manifests)
    differences, summary = paired_differences(seeds)
    curves.to_csv(output / "learning_curves.csv", index=False)
    seeds.to_csv(output / "per_seed_summary.csv", index=False)
    seeds.groupby(["level", "method", "inference"]).mean(numeric_only=True).to_csv(output / "summary.csv")
    differences.to_csv(output / "per_seed_paired_differences.csv", index=False)
    summary.to_csv(output / "paired_summary.csv", index=False)
    thresholds.to_csv(output / "final_step_f1_by_seed.csv", index=False)
    title = args.title
    plot_curves(curves, output, "learning_curves", PRIMARY, METHODS, title)
    plot_curves(curves, output, "jepa_filter_vs_smooth", (*PRIMARY, "eval_nll_bps"), ("short", "nll_control", "predictive"), title)
    plot_curves(curves, output, "purity_and_recall", ("latent_precision", "latent_recall"), METHODS, title)
    plot_curves(curves, output, "nll_and_coverage", ("eval_nll_bps", "state_coverage", "unmatched_mass"), METHODS, title)
    plot_final_f1(thresholds, output, manifests["filter"]["observations"])
    plot_paired(differences, output, SMOOTHING_ROWS, "paired_smoothing_effect", "Smoothing effect within each model")
    plot_paired(differences, output, JEPA_ROWS, "paired_jepa_effect", "JEPA effect under each student")
    residual_plots(finals, output, conditions=dict(filter="filtered_student", smooth="smoothed_student"))
    (output / "verification.json").write_text(json.dumps(checks, indent=2) + "\n")
    (output / "provenance.json").write_text(json.dumps(dict(
        filter=str(roots["filter"].relative_to(ROOT)), smooth=str(roots["smooth"].relative_to(ROOT)),
        metric_implementation="scripts.analysis.plot_config_metric_trajectories._metrics_from_record",
        metric_implementation_sha256=hashlib.sha256((ROOT / "scripts/analysis/plot_config_metric_trajectories.py").read_bytes()).hexdigest(),
        report_implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        implementations={i: json.loads((roots[i] / "implementation.json").read_text()) for i in INFERENCE},
        evaluation_records=len(hashes), evaluation_record_sha256=hashes, training_or_inference_changed=False), indent=2) + "\n")
    write_markdown(output, seeds, summary, checks, title, manifests["filter"])
    print(f"Canonical report: {output / 'results.md'}")


def overview(args):
    """One figure answering the primary question across settings and levels."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    output = args.overview.resolve(); output.mkdir(parents=True, exist_ok=True)
    frames = []
    for item in args.setting:
        label, path = item.split("=", 1)
        frames.append(pd.read_csv(Path(path) / "per_seed_paired_differences.csv").assign(setting=label))
    data = pd.concat(frames, ignore_index=True)
    data.to_csv(output / "overview_paired_differences.csv", index=False)
    groups = [(s, l) for s in dict.fromkeys(data.setting) for l in sorted(data[data.setting == s].level.unique())]
    metrics = ("action_tv_similarity", "action_edge_f1_auc", "eval_nll_bps")
    series = [("predictive_minus_short:filter", "JEPA − Short EM, filtered student", COLORS["predictive"], False),
              ("predictive_minus_short:smooth", "JEPA − Short EM, smoothed student", COLORS["predictive"], True),
              ("smooth_minus_filter:nll_control", "Smoothing effect without teacher (replay-NLL control)", COLORS["nll_control"], True)]
    fig, axes = plt.subplots(1, len(metrics), figsize=(5.2 * len(metrics), .55 * len(groups) * len(series) + 1.8), sharey=True)
    for ax, metric in zip(axes, metrics):
        style(ax); ax.grid(axis="y", visible=False); ax.axvline(0, color=MUTED, linewidth=1)
        ticks = []
        for g, (setting, level) in enumerate(groups):
            for k, (contrast, _, color, filled) in enumerate(series):
                y = g * (len(series) + 1) + k
                part = data[(data.setting == setting) & (data.level == level) & (data.contrast == contrast) &
                            (data.metric == f"final_{metric}")]
                ax.scatter(part.delta, np.full(len(part), y), s=36, color=color if filled else "white",
                           edgecolor=color, linewidth=1.5, zorder=3)
                ax.plot([part.delta.mean()] * 2, [y - .35, y + .35], color=INK, linewidth=2, zorder=4)
            ticks.append((g * (len(series) + 1) + 1, f"{setting}\nlevel {level}"))
        arrow = " (← better)" if metric in LOWER_IS_BETTER else " (better →)"
        ax.set_title(METRIC_LABELS[metric] + arrow, fontsize=9)
        ax.set_xlabel("Final-checkpoint paired difference", fontsize=8, color=MUTED)
    axes[0].set_yticks([t for t, _ in ticks], [l for _, l in ticks], fontsize=8)
    axes[0].invert_yaxis()
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], marker="o", linestyle="", markersize=7, markeredgewidth=1.5, markeredgecolor=c,
                      markerfacecolor=c if f else "white", label=l) for _, l, c, f in series]
    handles.append(Line2D([], [], color=INK, linewidth=2, label="Mean over seeds"))
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False, fontsize=8)
    fig.suptitle("Online JEPA–CSCG: JEPA gain with a filtered vs smoothed student, and smoothing without a teacher",
                 color=INK, fontsize=11)
    fig.tight_layout(rect=(0, .06, 1, .96))
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"overview_filter_vs_smooth.{suffix}", dpi=160)
    plt.close(fig)
    print(f"Overview: {output / 'overview_filter_vs_smooth.png'}")
    jepa_overview(data, groups, output)


def jepa_overview(data, groups, output):
    """Primary figure: JEPA − each baseline (rows) for each canonical metric (columns), every setting and level."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    metrics = ("action_tv_similarity", "action_edge_f1_auc", "latent_precision", "latent_recall", "eval_nll_bps")
    baselines = ("short", "long", "nll_control")
    step = len(INFERENCE) + 1
    fig, axes = plt.subplots(len(baselines), len(metrics), figsize=(3.4 * len(metrics) + 1.2, .2 * step * len(groups) * len(baselines) + 2),
                             sharey=True, squeeze=False)
    for r, baseline in enumerate(baselines):
        for c, metric in enumerate(metrics):
            ax = axes[r, c]; style(ax); ax.grid(axis="y", visible=False); ax.axvline(0, color=MUTED, linewidth=1)
            for g, (setting, level) in enumerate(groups):
                for k, inference in enumerate(INFERENCE):
                    y = g * step + k
                    part = data[(data.setting == setting) & (data.level == level) & (data.metric == f"final_{metric}") &
                                (data.contrast == f"predictive_minus_{baseline}:{inference}")]
                    color = COLORS[baseline]
                    ax.scatter(part.delta, np.full(len(part), y), s=30, color=color if inference == "smooth" else "white",
                               edgecolor=color, linewidth=1.5, zorder=3)
                    ax.plot([part.delta.mean()] * 2, [y - .4, y + .4], color=INK, linewidth=2, zorder=4)
            if r == 0:
                arrow = " (← better)" if metric in LOWER_IS_BETTER else " (better →)"
                ax.set_title(METRIC_LABELS[metric] + arrow, fontsize=8)
            if c == 0:
                ax.set_ylabel(f"JEPA − {BASELINE_LABELS[baseline]}", color=INK, fontsize=9)
        axes[r, 0].set_yticks([g * step + .5 for g in range(len(groups))], [f"{s} L{l}" for s, l in groups], fontsize=7)
    axes[0, 0].invert_yaxis()
    handles = [Line2D([], [], marker="o", linestyle="", markersize=7, markeredgewidth=1.5, markeredgecolor=MUTED,
                      markerfacecolor=MUTED if i == "smooth" else "white", label=INFERENCE_LABELS[i].capitalize())
               for i in INFERENCE] + [Line2D([], [], color=INK, linewidth=2, label="Mean over seeds")]
    handles += [Line2D([], [], marker="o", linestyle="", markersize=7, color=COLORS[b], label=f"vs {BASELINE_LABELS[b]}")
                for b in baselines]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False, fontsize=8)
    fig.suptitle("Online JEPA–CSCG: effect of the JEPA teacher, final checkpoint, per seed (dots) and mean (tick)",
                 color=INK, fontsize=11)
    fig.tight_layout(rect=(0, .04, 1, .97))
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"overview_jepa_effect.{suffix}", dpi=160)
    plt.close(fig)
    print(f"Overview: {output / 'overview_jepa_effect.png'}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--filter", type=Path)
    p.add_argument("--smooth", type=Path)
    p.add_argument("--reference", type=Path, help="Historical filtered output to reproduce bit for bit")
    p.add_argument("--output", type=Path, help="Default: <setting>/canonical_metrics")
    p.add_argument("--title", default="Online JEPA-CSCG: filtered versus smoothed student")
    p.add_argument("--overview", type=Path, help="Write a cross-setting overview figure to this directory")
    p.add_argument("--setting", action="append", default=[], help="Label=<canonical_metrics dir>, for --overview")
    args = p.parse_args()
    if args.overview:
        if not args.setting:
            p.error("--overview requires at least one --setting")
        overview(args)
    elif args.filter and args.smooth:
        report(args)
    else:
        p.error("give --filter and --smooth, or --overview with --setting")


if __name__ == "__main__":
    main()
