"""Canonical-metric report: JEPA teacher smoothing window 40 versus a longer window.

Reads two completed run_cscg_distillation_experiment.py outputs that differ only
in the teacher context (--teacher-window, or --window, which also sets the replay
and long-EM windows) and possibly --methods, verifies they are paired, and writes
canonical metrics, paired differences and plots. Short EM must be bit-for-bit
identical; long EM and the replay-NLL control too when --window is unchanged.

PYTHONPATH=.:src .venv/bin/python scripts/analysis/cscg_distillation_teacher_window_report.py \
    --base <teacher-window-40 dir> --long <longer teacher-window dir> [--output DIR] [--title T]

Cross-setting overview of completed reports:

PYTHONPATH=.:src .venv/bin/python scripts/analysis/cscg_distillation_teacher_window_report.py \
    --overview <output dir> --setting "Label=<report dir>" ...
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.analysis.cscg_distillation_canonical_report import LABELS, METRICS, canonical_metrics, summarise
from scripts.analysis.cscg_distillation_student_inference_report import (
    COLORS, HEADLINE, INK, LOWER_IS_BETTER, METRIC_LABELS, MUTED, UPDATE_KEYS, load, style,
)

ROOT = Path(__file__).resolve().parents[2]
NO_TEACHER = ("short", "long", "nll_control")
BASELINE_LABELS = dict(short="Short EM", long="Long EM", nll_control="replay-NLL control")
OVERVIEW_METRICS = ("action_tv_similarity", "action_edge_f1_auc", "latent_precision", "latent_recall", "eval_nll_bps")


def teacher_window(manifest):
    experiment = manifest["experiment"]
    return experiment.get("teacher_window") or experiment["window"]


def verify(roots, runs, manifests):
    windows = {c: teacher_window(manifests[c]) for c in roots}
    if not windows["base"] < windows["long"]:
        raise ValueError(f"Expected a longer teacher window in --long: {windows}")
    strip = lambda m: {**{k: v for k, v in m.items() if k != "methods"},
                       "experiment": {k: v for k, v in m["experiment"].items() if k not in ("teacher_window", "window")}}
    if strip(manifests["base"]) != strip(manifests["long"]):
        raise ValueError("Experiments differ in more than the context windows and the method subset")
    replay = {c: manifests[c]["experiment"]["window"] for c in roots}
    unchanged = ("short",) if replay["base"] != replay["long"] else NO_TEACHER
    methods = manifests["long"]["methods"]
    if "predictive" not in methods or not set(methods) <= set(manifests["base"]["methods"]):
        raise ValueError("The longer-window run must include predictive and a subset of the base methods")
    expected = len(manifests["long"]["levels"]) * len(manifests["long"]["seeds"]) * len(methods)
    if len(runs["long"]) != expected or not set(runs["long"]) <= set(runs["base"]):
        raise ValueError("Incomplete or unpaired runs")
    final = lambda run: run["training"]["checkpoint_metadata"][str(max(map(int, run["training"]["checkpoint_metadata"])))]
    for key, run in runs["long"].items():
        base = runs["base"][key]
        if run["training"]["provenance"] != base["training"]["provenance"]:
            raise ValueError(f"Unmatched trajectory, initial tensor, allocation or protocol: {key}")
        if any(final(run)[k] != final(base)[k] for k in UPDATE_KEYS):
            raise ValueError(f"Learning-update counts differ: {key}")
        if key[2] in unchanged and not np.array_equal(run["T"], base["T"]):
            raise ValueError(f"Window change altered an arm it cannot affect: {key}")
    walks = 0
    for path in roots["long"].glob("evaluation_caches/*/walks/*.npz"):
        with np.load(path) as a, np.load(roots["base"] / path.relative_to(roots["long"])) as b:
            if set(a.files) != set(b.files) or any(not np.array_equal(a[k], b[k]) for k in a.files):
                raise ValueError("Held-out walk arrays differ")
        walks += 1
    return dict(teacher_windows=windows, replay_and_long_em_windows=replay, student_inference=manifests["long"]["experiment"]["student_inference"],
                methods=methods, paired_runs=len(runs["long"]), update_counts_equal=True,
                bitwise_equal_arms=list(unchanged), bitwise_equal_runs=sum(k[2] in unchanged for k in runs["long"]),
                held_out_walk_files_equal=walks)


def collect(runs, manifests):
    rows = []
    methods = manifests["long"]["methods"]
    for condition in ("base", "long"):
        for (level, seed, method), run in runs[condition].items():
            if method not in methods:
                continue
            for row in run["rows"]:
                scores = canonical_metrics(run["records"][row["step"]][1])
                assert np.isclose(scores["action_tv_similarity"], 1 - row["action_tv"])
                rows.append(dict(iterations=manifests[condition]["local_parameter_updates"], condition=condition,
                                 teacher_window=teacher_window(manifests[condition]), level=level, seed=seed,
                                 method=method, step=row["step"], train_seconds=row["train_seconds"],
                                 **{m: scores[m] for m in METRICS}))
    curves = pd.DataFrame(rows)
    seeds = pd.concat([summarise(curves[curves.condition == c].drop(columns=["condition", "teacher_window"])).assign(condition=c)
                       for c in ("base", "long")], ignore_index=True)
    return curves, seeds


def paired_differences(seeds, windows):
    numeric = [f"{p}_{m}" for p in ("final", "time_mean") for m in METRICS] + ["train_seconds"]
    table = seeds.set_index(["condition", "level", "seed", "method"])
    rows = []
    def add(contrast, family, left, right, level, seed):
        for metric in numeric:
            rows.append(dict(contrast=contrast, family=family, level=level, seed=seed, metric=metric,
                             value=float(left[metric]), reference_value=float(right[metric]),
                             delta=float(left[metric] - right[metric])))
    methods = set(seeds.method)
    for (level, seed), _ in seeds.groupby(["level", "seed"]):
        get = lambda c, m: table.loc[(c, level, seed, m)]
        for condition in ("base", "long"):
            for baseline in (b for b in NO_TEACHER if b in methods):
                add(f"predictive_minus_{baseline}:W{windows[condition]}", "jepa_effect",
                    get(condition, "predictive"), get(condition, baseline), level, seed)
        add(f"predictive:W{windows['long']}_minus_W{windows['base']}", "teacher_window_effect",
            get("long", "predictive"), get("base", "predictive"), level, seed)
    frame = pd.DataFrame(rows)
    lower = frame.metric.str.endswith(LOWER_IS_BETTER) | (frame.metric == "train_seconds")
    frame["benefit_delta"] = np.where(lower, -frame.delta, frame.delta)
    summary = frame.groupby(["family", "contrast", "level", "metric"]).agg(
        mean=("delta", "mean"), std=("delta", "std"), count=("delta", "count"),
        wins=("benefit_delta", lambda x: int((x > 1e-12).sum())),
        losses=("benefit_delta", lambda x: int((x < -1e-12).sum())),
        ties=("benefit_delta", lambda x: int((x.abs() <= 1e-12).sum()))).reset_index()
    return frame, summary


def contrast_table(summary, family, order):
    part = summary[summary.family == family].set_index(["contrast", "level", "metric"])
    keys = sorted({k[:2] for k in part.index}, key=lambda k: (order.index(k[0]), k[1]))
    lines = ["Final checkpoint: mean ± SD, wins/n (wins = paired seeds where the left side is better).", "",
             "| Contrast | Level | " + " | ".join(METRIC_LABELS[m] for m in HEADLINE) + " |",
             "| --- | ---: |" + " ---: |" * len(HEADLINE)]
    for contrast, level in keys:
        cells = [part.loc[(contrast, level, f"final_{m}")] for m in HEADLINE]
        lines.append(f"| {contrast} | {level} | " + " | ".join(
            f"{c['mean']:+.4f} ± {c['std']:.4f} ({int(c.wins)}/{int(c['count'])})" for c in cells) + " |")
    return lines


def plot_curves(curves, output, windows, title, unchanged):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    levels = sorted(curves.level.unique())
    panels = OVERVIEW_METRICS
    methods = [m for m in ("short", "long", "nll_control", "predictive") if m in set(curves.method)]
    fig, axes = plt.subplots(len(levels), len(panels), figsize=(4 * len(panels), 3.3 * len(levels)), squeeze=False)
    for i, level in enumerate(levels):
        for j, metric in enumerate(panels):
            ax = axes[i, j]; style(ax)
            for method in methods:
                for condition, dash in (("base", "--"), ("long", "-")):
                    if method in unchanged and condition == "base":
                        continue   # bitwise identical to the longer-window run
                    part = curves[(curves.level == level) & (curves.method == method) & (curves.condition == condition)]
                    agg = part.groupby("step")[metric].agg(["mean", "std"])
                    ax.plot(agg.index, agg["mean"], color=COLORS[method], linestyle=dash, linewidth=2)
                    if condition == "long":
                        sd = agg["std"].fillna(0)
                        ax.fill_between(agg.index, agg["mean"] - sd, agg["mean"] + sd, color=COLORS[method], alpha=.10, linewidth=0)
            ax.set_title(f"Level {level}: {METRIC_LABELS[metric]}", fontsize=9)
            ax.set_xlabel("Observations", fontsize=8, color=MUTED)
            if metric != "eval_nll_bps":
                ax.set_ylim(0, 1)
    handles = [Line2D([], [], color=COLORS[m], linewidth=2, label=LABELS[m]) for m in methods]
    handles += [Line2D([], [], color=MUTED, linewidth=2, linestyle="--", label=f"Context {windows['base']}"),
                Line2D([], [], color=MUTED, linewidth=2, label=f"Context {windows['long']}")]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False, fontsize=8)
    fig.suptitle(title + f" — mean ± 1 SD over seeds ({', '.join(LABELS[m] for m in unchanged)} identical under both)", color=INK, fontsize=11)
    fig.tight_layout(rect=(0, .04, 1, .97))
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"learning_curves.{suffix}", dpi=160)
    plt.close(fig)


def dot_figure(data, groups, rows, output, filename, title):
    """rows: (label, [(contrast, filled)], color). Groups: (setting, level). Dots = seeds, tick = mean."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    step = max(len(series) for _, series, _ in rows) + 1
    fig, axes = plt.subplots(len(rows), len(OVERVIEW_METRICS),
                             figsize=(3.4 * len(OVERVIEW_METRICS) + 1.2, .2 * step * len(groups) * len(rows) + 2.2),
                             sharey=True, squeeze=False)
    for r, (label, series, color) in enumerate(rows):
        for c, metric in enumerate(OVERVIEW_METRICS):
            ax = axes[r, c]; style(ax); ax.grid(axis="y", visible=False); ax.axvline(0, color=MUTED, linewidth=1)
            for g, (setting, level) in enumerate(groups):
                for k, (contrast, filled) in enumerate(series):
                    y = g * step + k
                    part = data[(data.setting == setting) & (data.level == level) & (data.contrast == contrast) &
                                (data.metric == f"final_{metric}")]
                    ax.scatter(part.delta, np.full(len(part), y), s=30, color=color if filled else "white",
                               edgecolor=color, linewidth=1.5, zorder=3)
                    if len(part):
                        ax.plot([part.delta.mean()] * 2, [y - .4, y + .4], color=INK, linewidth=2, zorder=4)
            if r == 0:
                ax.set_title(METRIC_LABELS[metric] + (" (← better)" if metric in LOWER_IS_BETTER else " (better →)"), fontsize=8)
            if c == 0:
                ax.set_ylabel(label, color=INK, fontsize=9)
        axes[r, 0].set_yticks([g * step + (step - 2) / 2 for g in range(len(groups))],
                              [f"{s} L{l}" if s else f"Level {l}" for s, l in groups], fontsize=7)
    axes[0, 0].invert_yaxis()
    return fig, axes


def finish(fig, handles, output, filename, title):
    import matplotlib.pyplot as plt
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False, fontsize=8)
    fig.suptitle(title, color=INK, fontsize=11)
    fig.tight_layout(rect=(0, .04, 1, .97))
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"{filename}.{suffix}", dpi=160)
    plt.close(fig)


def jepa_rows(windows, baselines):
    rows = [(f"JEPA − {BASELINE_LABELS[b]}", [(f"predictive_minus_{b}:W{windows['base']}", False),
                                               (f"predictive_minus_{b}:W{windows['long']}", True)], COLORS[b])
            for b in baselines]
    rows.append((f"JEPA: W{windows['long']} − W{windows['base']}",
                 [(f"predictive:W{windows['long']}_minus_W{windows['base']}", True)], COLORS["predictive"]))
    return rows


def legend_handles(windows, baselines):
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], marker="o", linestyle="", markersize=7, markeredgewidth=1.5, markeredgecolor=MUTED,
                      markerfacecolor=MUTED if filled else "white", label=f"Context {windows[c]}")
               for c, filled in (("base", False), ("long", True))]
    handles.append(Line2D([], [], color=INK, linewidth=2, label="Mean over seeds"))
    handles += [Line2D([], [], marker="o", linestyle="", markersize=7, color=COLORS[b], label=f"vs {BASELINE_LABELS[b]}")
                for b in baselines]
    handles.append(Line2D([], [], marker="o", linestyle="", markersize=7, color=COLORS["predictive"], label="Window effect on JEPA"))
    return handles


def context_lines(checks, manifest):
    windows, replay = checks["teacher_windows"], checks["replay_and_long_em_windows"]
    if replay["base"] == replay["long"]:
        return [f"The teacher smooths over the last {windows['long']} (vs {windows['base']}) observations from its cached boundary belief;",
                f"only its targets for the student's {replay['long']}-observation replay window are used. Long EM, the replay window,",
                "EMA half-life and every other setting are unchanged; arms without a teacher are bit-for-bit identical."]
    return [f"Context {windows['long']} vs {windows['base']} observations for the teacher's smoothing window, the JEPA/replay-NLL penalty",
            f"(teacher targets for all {replay['long'] - 1} vs {replay['base'] - 1} window transitions) and long EM's E-step window.",
            "The student is unchanged: its own E-step uses only the newest transition, and its replay source beliefs are its cached",
            f"{manifest['experiment']['student_inference']}ed one-transition beliefs. EMA half-life and all other settings are unchanged;",
            "short EM is bit-for-bit identical."]


def write_markdown(output, seeds, summary, checks, title, manifest, windows):
    order = sorted(set(summary.contrast), key=lambda c: (c.startswith("predictive:"), c))
    means = seeds.groupby(["level", "method", "condition"]).mean(numeric_only=True)
    head = ("| Level | Model | Teacher window | " + " | ".join(METRIC_LABELS[m] for m in HEADLINE) + " |",
            "| ---: | --- | ---: |" + " ---: |" * len(HEADLINE))
    lines = [f"# {title}", "",
             f"Setting: {manifest['local_parameter_updates']} local updates"
             f"{' (coupled auxiliary)' if manifest['experiment'].get('coupled_auxiliary') else ''}, "
             f"{manifest['clone_allocation']} clone allocation, {manifest['observations']:,} observations, seeds {manifest['seeds']}, "
             f"{manifest['experiment']['student_inference']} student.",
             *context_lines(checks, manifest),
             "Canonical metrics reuse the evaluation-v5 extractor. Lower is better only for NLL.", "",
             f"## Primary result: JEPA effect with teacher window {windows['base']} and {windows['long']}", "",
             *contrast_table(summary, "jepa_effect", order), "",
             f"## Teacher-window effect on JEPA (W{windows['long']} − W{windows['base']})", "",
             *contrast_table(summary, "teacher_window_effect", order), "",
             "## Verification", "", *[f"- {k}: {v}" for k, v in checks.items()], "",
             "## Final checkpoint (mean over seeds)", "", *head]
    fmt = lambda r: " | ".join(f"{r['final_' + m]:.4f}" for m in HEADLINE)
    lines += [f"| {l} | {LABELS[m]} | {windows[c]} | {fmt(r)} |" for (l, m, c), r in means.iterrows()]
    lines += ["", "## Files", "",
              "- `learning_curves.png/pdf`: canonical metrics over training; dashed = shorter context, solid = longer.",
              "- `paired_jepa_effect.png/pdf`: per-seed paired differences (JEPA − each baseline under both windows; window effect).",
              "- CSV: `learning_curves`, `per_seed_summary`, `summary`, `per_seed_paired_differences`, `paired_summary`."]
    (output / "results.md").write_text("\n".join(lines) + "\n")


def report(args):
    import matplotlib
    matplotlib.use("Agg")
    roots = dict(base=args.base.resolve(), long=args.long.resolve())
    output = (args.output or roots["long"] / "canonical_metrics").resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifests = {c: json.loads((roots[c] / "experiment.json").read_text()) for c in roots}
    runs = {c: load(roots[c]) for c in roots}
    checks = verify(roots, runs, manifests)
    windows = checks["teacher_windows"]
    curves, seeds = collect(runs, manifests)
    differences, summary = paired_differences(seeds, windows)
    curves.to_csv(output / "learning_curves.csv", index=False)
    seeds.to_csv(output / "per_seed_summary.csv", index=False)
    seeds.groupby(["level", "method", "condition"]).mean(numeric_only=True).to_csv(output / "summary.csv")
    differences.to_csv(output / "per_seed_paired_differences.csv", index=False)
    summary.to_csv(output / "paired_summary.csv", index=False)
    (output / "verification.json").write_text(json.dumps(checks, indent=2) + "\n")
    (output / "provenance.json").write_text(json.dumps(dict(
        **{c: str(r.relative_to(ROOT) if r.is_relative_to(ROOT) else r) for c, r in roots.items()},
        metric_implementation="scripts.analysis.plot_config_metric_trajectories._metrics_from_record",
        implementations={c: json.loads((roots[c] / "implementation.json").read_text()) for c in roots}), indent=2) + "\n")
    plot_curves(curves, output, windows, args.title, checks["bitwise_equal_arms"])
    baselines = [b for b in NO_TEACHER if b in set(seeds.method)]
    groups = [("", l) for l in sorted(differences.level.unique())]
    fig, _ = dot_figure(differences.assign(setting=""), groups, jepa_rows(windows, baselines), output, "", "")
    finish(fig, legend_handles(windows, baselines), output, "paired_jepa_effect",
           args.title + " — final checkpoint, per seed (dots) and mean (tick)")
    write_markdown(output, seeds, summary, checks, args.title, manifests["long"], windows)
    print(f"Teacher-window report: {output / 'results.md'}")


def overview(args):
    import matplotlib
    matplotlib.use("Agg")
    output = args.overview.resolve(); output.mkdir(parents=True, exist_ok=True)
    frames, windows = [], None
    for item in args.setting:
        label, path = item.split("=", 1)
        frames.append(pd.read_csv(Path(path) / "per_seed_paired_differences.csv").assign(setting=label))
        found = json.loads((Path(path) / "verification.json").read_text())["teacher_windows"]
        if windows not in (None, found):
            raise ValueError("Settings use different teacher windows")
        windows = found
    data = pd.concat(frames, ignore_index=True)
    data.to_csv(output / "overview_paired_differences.csv", index=False)
    groups = [(s, l) for s in dict.fromkeys(data.setting) for l in sorted(data[data.setting == s].level.unique())]
    baselines = [b for b in NO_TEACHER if data.contrast.str.startswith(f"predictive_minus_{b}:").any()]
    fig, _ = dot_figure(data, groups, jepa_rows(windows, baselines), output, "", "")
    finish(fig, legend_handles(windows, baselines), output, "overview_teacher_window",
           f"Online JEPA–CSCG: JEPA effect with context {windows['base']} vs {windows['long']}, final checkpoint")
    print(f"Overview: {output / 'overview_teacher_window.png'}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base", type=Path, help="Run with the shorter (historical) teacher window")
    p.add_argument("--long", type=Path, help="Run with the longer teacher window")
    p.add_argument("--output", type=Path, help="Default: <long>/canonical_metrics")
    p.add_argument("--title", default="Online JEPA-CSCG: teacher smoothing window")
    p.add_argument("--overview", type=Path)
    p.add_argument("--setting", action="append", default=[], help="Label=<report dir>, for --overview")
    args = p.parse_args()
    if args.overview:
        if not args.setting:
            p.error("--overview requires at least one --setting")
        overview(args)
    elif args.base and args.long:
        report(args)
    else:
        p.error("give --base and --long, or --overview with --setting")


if __name__ == "__main__":
    main()
