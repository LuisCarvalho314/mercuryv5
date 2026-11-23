"""Compute thresholded edge precision/recall/F1 from canonical evaluation records.

Canonical: consumes evaluation records produced by `evaluate_study`.
Must not load model snapshots, generate walks, compute alignments, or project transitions.

Usage:
    python -m mercury_runs.analysis.plots.threshold_edge_metrics \\
        --study_root results/current/studies/<study> \\
        --protocol_id canonical-v4-<hash>
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib
import numpy as np
from matplotlib import pyplot as plt

from mercury_runs.analysis.evaluation_records import (
    edge_metrics_at_threshold,
    load_evaluation_step,
    load_study_manifest,
)
from mercury_runs.analysis.plots.selected_residual_maps import (
    BATCH_SIZE,
    CSCG_PRESETS,
    LEVELS,
    SCENARIOS,
    SEEDS,
    _KIND_TO_RUN,
    _cscg_run,
    _hybrid_run,
    _rand_prob_suffix,
)
from mercury_runs.analysis.plots import publication_style as pub


DEFAULT_OUTPUT_DIR = Path(
    "results/current/notebook_plots/"
    "mercury_cscg_hybrid_vs_cscg_batch_size_30_seed_threshold_edge_metrics2"
)
DEFAULT_ACTION_OUTPUT_DIR = Path(
    "results/current/notebook_plots/"
    "mercury_cscg_hybrid_vs_cscg_batch_size_30_seed_action_conditioned_threshold_edge_metrics2"
)
DEFAULT_THRESHOLDS = (
    0.0, 1e-6, 1e-5, 1e-4, 1e-3, 2e-3, 5e-3,
    1e-2, 2e-2, 5e-2, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5,
)
METRICS = ("edge_precision", "edge_recall", "edge_f1")
ENVIRONMENT_ORDER = (13, 21, 20, 22)
ENVIRONMENT_LABELS = {13: "3x3", 21: "1x4", 20: "1x5", 22: "3x4"}
METHOD_COLORS = {
    pub.method_label("mercury"): pub.method_color("mercury"),
    pub.method_label("tmgwr"): pub.method_color("tmgwr"),
    pub.method_label("hybrid"): pub.method_color("hybrid"),
    pub.method_label("tmgwr_hybrid"): pub.method_color("tmgwr_hybrid"),
    pub.method_label("vanilla_cscg"): pub.method_color("vanilla_cscg"),
    pub.method_label("oracle_capacity"): pub.method_color("oracle_capacity"),
    pub.method_label("oracle_capacity_structural"): pub.method_color("oracle_capacity_structural"),
    pub.method_label("oracle_capacity_first_order"): pub.method_color("oracle_capacity_first_order"),
    "CSCG - oracle clones/state (x1)": pub.method_color("ground_truth_state_1"),
    "CSCG - oracle clones/state (x2)": pub.method_color("ground_truth_state_2"),
    "CSCG - observation clones (x4)": pub.method_color("uniform_obs_4"),
    "CSCG - observation clones (x8)": pub.method_color("uniform_obs_8"),
}


def _load_record_for_kind(run_root: Path, kind: str, protocol_id: str) -> dict[str, Any] | None:
    spec = _KIND_TO_RUN[kind]
    return load_evaluation_step(run_root, protocol_id, spec.phase_step, method=spec.evaluation_method)


def _rows_for_method(
    manifest: dict[tuple[Any, ...], Path],
    scenario: dict[str, Any],
    *,
    level: int,
    seeds: Iterable[int],
    thresholds: Iterable[float],
    method_label: str,
    run_kind: str,
    run_resolver: Any,
    diagnostic_kind: str,
    protocol_id: str,
    include_self_loops: bool,
    action_conditioned: bool,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        try:
            run_root = Path(run_resolver(int(seed)))
        except KeyError:
            continue
        record = _load_record_for_kind(run_root, diagnostic_kind, protocol_id)
        if record is None:
            print(
                f"  Warning: no eval record for {diagnostic_kind!r} seed={seed} level={level} — skipping.",
                file=sys.stderr,
            )
            continue
        for threshold in thresholds:
            metrics = edge_metrics_at_threshold(
                record,
                float(threshold),
                action_conditioned=bool(action_conditioned),
                include_self_loops=bool(include_self_loops),
            )
            rows.append(
                {
                    "scenario_key": str(scenario["key"]),
                    "scenario_title": str(scenario["title"]),
                    "level": int(level),
                    "seed": int(seed),
                    "method": _ascii_label(method_label),
                    "run_kind": str(run_kind),
                    "run_root": str(run_root),
                    "metric_scope": "action_conditioned" if action_conditioned else "collapsed",
                    "threshold": float(threshold),
                    **metrics,
                }
            )
    return rows


def _aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            row["scenario_key"],
            row["scenario_title"],
            row["metric_scope"],
            row["level"],
            row["method"],
            row["run_kind"],
            row["threshold"],
        )
        grouped[key].append(row)

    out: list[dict[str, Any]] = []
    for key, items in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][3], item[0][5], item[0][6])):
        scenario_key, scenario_title, metric_scope, level, method, run_kind, threshold = key
        summary: dict[str, Any] = {
            "scenario_key": scenario_key,
            "scenario_title": scenario_title,
            "metric_scope": metric_scope,
            "level": int(level),
            "method": method,
            "run_kind": run_kind,
            "threshold": float(threshold),
            "seed_count": len(items),
        }
        for metric in METRICS:
            values = np.asarray([float(item[metric]) for item in items], dtype=np.float64)
            std = float(values.std(ddof=1)) if values.size > 1 else 0.0
            sem = float(std / np.sqrt(values.size)) if values.size else 0.0
            ci95 = float(1.96 * sem) if values.size > 1 else 0.0
            summary[f"{metric}_mean"] = float(values.mean()) if values.size else 0.0
            summary[f"{metric}_std"] = std
            summary[f"{metric}_sem"] = sem
            summary[f"{metric}_ci95"] = ci95
        for count_key in ("true_positive_count", "predicted_edge_count", "true_edge_count"):
            values = np.asarray([float(item[count_key]) for item in items], dtype=np.float64)
            summary[f"{count_key}_mean"] = float(values.mean()) if values.size else 0.0
        out.append(summary)
    return out


def _plot_summary(rows: list[dict[str, Any]], *, output_dir: Path) -> list[Path]:
    by_scenario: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_scenario[str(row["scenario_key"])].append(row)

    paths: list[Path] = []
    for scenario_key, scenario_rows in by_scenario.items():
        available_levels = {int(row["level"]) for row in scenario_rows}
        levels = [level for level in ENVIRONMENT_ORDER if level in available_levels]
        levels.extend(sorted(available_levels - set(levels)))
        fig, axes = plt.subplots(
            len(METRICS), len(levels), figsize=(13.8, 8.0), dpi=180, squeeze=False, sharex=True, sharey="row",
        )
        metric_scope = str(scenario_rows[0].get("metric_scope", "collapsed"))
        for col_index, level in enumerate(levels):
            level_rows = [row for row in scenario_rows if int(row["level"]) == int(level)]
            methods = _ordered_methods({str(row["method"]) for row in level_rows})
            for row_index, metric in enumerate(METRICS):
                ax = axes[row_index, col_index]
                for method in methods:
                    method_rows = sorted(
                        [row for row in level_rows if str(row["method"]) == method],
                        key=lambda row: float(row["threshold"]),
                    )
                    x = np.asarray([float(row["threshold"]) for row in method_rows], dtype=np.float64)
                    y = np.asarray([float(row[f"{metric}_mean"]) for row in method_rows], dtype=np.float64)
                    yerr = np.asarray([float(row[f"{metric}_ci95"]) for row in method_rows], dtype=np.float64)
                    color = METHOD_COLORS.get(method)
                    ax.plot(x, y, marker="o", markersize=3.0, linewidth=1.8, label=method, color=color)
                    ax.fill_between(x, np.clip(y - yerr, 0.0, 1.0), np.clip(y + yerr, 0.0, 1.0), color=color, alpha=0.13, linewidth=0)
                ax.set_ylim(-0.02, 1.02)
                t_min = min(float(row["threshold"]) for row in scenario_rows)
                t_max = max(float(row["threshold"]) for row in scenario_rows)
                if t_min == t_max:
                    t_min -= 0.01
                    t_max += 0.01
                ax.set_xlim(t_min, t_max)
                ax.set_xscale("linear")
                ax.grid(True, which="major", color="#e5e7eb", linewidth=0.75)
                ax.set_axisbelow(True)
                if row_index == 0:
                    ax.set_title(_environment_label(level), fontsize=11, fontweight="bold")
                if col_index == 0:
                    ax.set_ylabel(_metric_title(metric), fontsize=10)
                if row_index == len(METRICS) - 1:
                    ax.set_xlabel("threshold")
                ax.tick_params(labelsize=8.5)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=min(3, len(labels)), fontsize=8.5, frameon=True, edgecolor="#d1d5db")
        scope_title = "Action-conditioned" if metric_scope == "action_conditioned" else "Collapsed"
        title = str(scenario_rows[0]["scenario_title"])
        fig.suptitle(f"{scope_title} thresholded edge metrics: {title} (ribbons: 95% CI)", fontsize=12, y=0.985)
        fig.subplots_adjust(left=0.075, right=0.99, top=0.91, bottom=0.17, wspace=0.12, hspace=0.18)
        output_path = output_dir / f"threshold_edge_metrics_{scenario_key}.png"
        fig.savefig(output_path)
        plt.close(fig)
        paths.append(output_path)
    return paths


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _parse_float_list(raw: str) -> list[float]:
    return [float(v.strip()) for v in raw.split(",") if v.strip()]


def _parse_int_list(raw: str) -> list[int]:
    values: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-", 1)
            values.extend(range(int(a), int(b) + 1))
        elif part:
            values.append(int(part))
    return values


def _ascii_label(label: str) -> str:
    return str(label).replace("–", "-").replace("—", "-").replace("×", "x")


def _ordered_methods(methods: set[str]) -> list[str]:
    order = {
        pub.method_label("mercury"): 0, "POCML": 1, pub.method_label("tmgwr"): 2,
        pub.method_label("hybrid"): 3, pub.method_label("tmgwr_hybrid"): 4,
        pub.method_label("vanilla_cscg"): 10, pub.method_label("oracle_capacity"): 11,
        pub.method_label("oracle_capacity_first_order"): 12, pub.method_label("oracle_capacity_structural"): 13,
        "CSCG - oracle clones/state (x1)": 14, "CSCG - oracle clones/state (x2)": 15,
        "CSCG - observation clones (x4)": 16, "CSCG - observation clones (x8)": 17,
    }
    return sorted(methods, key=lambda m: (order.get(m, 99), m))


def _metric_title(metric: str) -> str:
    return {"edge_precision": "Edge Precision", "edge_recall": "Edge Recall", "edge_f1": "Edge F1"}[metric]


def _environment_label(level: int) -> str:
    return ENVIRONMENT_LABELS.get(int(level), f"Level {int(level)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study_root", type=Path,
                        default=Path("results/current/studies/mercury_cscg_hybrid_vs_cscg_batch_size_30_seed_common_eval_walks2"))
    parser.add_argument("--protocol_id", type=str, required=True,
                        help="Canonical-v4 protocol ID (e.g. canonical-v4-<hash>)")
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--rand_prob", type=float, default=0.3)
    parser.add_argument("--levels", type=str, default=",".join(str(l) for l in LEVELS))
    parser.add_argument("--seeds", type=str, default=f"0-{max(SEEDS)}")
    parser.add_argument("--thresholds", type=str, default=",".join(f"{v:g}" for v in DEFAULT_THRESHOLDS))
    parser.add_argument("--scenarios", type=str, default=",".join(str(s["key"]) for s in SCENARIOS))
    parser.add_argument("--include_self_loops", action="store_true")
    parser.add_argument("--action_conditioned", action="store_true")
    parser.add_argument("--no_plots", action="store_true", help="Write CSV summaries only; skip threshold-edge figures.")
    args = parser.parse_args()

    if bool(args.action_conditioned) and args.output_dir == DEFAULT_OUTPUT_DIR:
        args.output_dir = DEFAULT_ACTION_OUTPUT_DIR

    matplotlib.use("Agg", force=True)
    levels = tuple(_parse_int_list(args.levels))
    seeds = tuple(_parse_int_list(args.seeds))
    thresholds = tuple(_parse_float_list(args.thresholds))
    scenario_keys = set(s.strip() for s in args.scenarios.split(","))
    scenarios = [s for s in SCENARIOS if str(s["key"]) in scenario_keys]

    manifest = load_study_manifest(args.study_root)
    rand_prob = float(args.rand_prob)

    rows: list[dict[str, Any]] = []
    for scenario in scenarios:
        print(f"Evaluating {scenario['key']}")
        for level in levels:
            targets = tuple(scenario.get("targets") or ((scenario["target_label"], scenario["target_kind"]),))
            for target_label, target_kind in targets:
                target_kind = str(target_kind)
                target_run_key = _KIND_TO_RUN[target_kind].manifest_run_key
                if target_run_key == "tmgwr_hybrid":
                    target_method = "tmgwr_cscg_hybrid"
                elif target_run_key == "hybrid":
                    target_method = "mercury_cscg_hybrid"
                else:
                    raise ValueError(f"Unsupported target run key for threshold metrics: {target_run_key}")
                rows.extend(
                    _rows_for_method(
                        manifest, scenario, level=int(level), seeds=seeds, thresholds=thresholds,
                        method_label=str(target_label), run_kind=target_kind,
                        run_resolver=lambda seed, method=target_method, level=int(level): _hybrid_run(
                            manifest,
                            level,
                            int(seed),
                            rand_prob=rand_prob,
                            method=method,
                        ),
                        diagnostic_kind=target_kind,
                        protocol_id=args.protocol_id,
                        include_self_loops=bool(args.include_self_loops),
                        action_conditioned=bool(args.action_conditioned),
                    )
                )
            for label, preset in CSCG_PRESETS:
                rows.extend(
                    _rows_for_method(
                        manifest, scenario, level=int(level), seeds=seeds, thresholds=thresholds,
                        method_label=str(label), run_kind=str(preset),
                        run_resolver=lambda seed, preset=str(preset), level=int(level): _cscg_run(
                            manifest, level, int(seed), preset, rand_prob=rand_prob
                        ),
                        diagnostic_kind=str(scenario["cscg_kind"]),
                        protocol_id=args.protocol_id,
                        include_self_loops=bool(args.include_self_loops),
                        action_conditioned=bool(args.action_conditioned),
                    )
                )

    if not rows:
        print("No rows produced. Check that evaluate_study has been run.", file=sys.stderr)
        sys.exit(1)

    aggregate_rows = _aggregate_rows(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_csv = args.output_dir / "threshold_edge_metrics_by_seed.csv"
    aggregate_csv = args.output_dir / "threshold_edge_metrics_summary.csv"
    _write_csv(raw_csv, rows)
    _write_csv(aggregate_csv, aggregate_rows)

    manifest_info = {
        "study_root": str(args.study_root),
        "protocol_id": args.protocol_id,
        "rand_prob": rand_prob,
        "batch_size": BATCH_SIZE,
        "levels": list(levels),
        "seeds": list(seeds),
        "thresholds": list(thresholds),
        "scenario_keys": [s["key"] for s in scenarios],
        "include_self_loops": bool(args.include_self_loops),
        "action_conditioned": bool(args.action_conditioned),
        "raw_csv": str(raw_csv),
        "summary_csv": str(aggregate_csv),
        "plots_enabled": not bool(args.no_plots),
    }
    (args.output_dir / "threshold_edge_metrics_manifest.json").write_text(
        json.dumps(manifest_info, indent=2), encoding="utf-8"
    )

    plot_paths = [] if bool(args.no_plots) else _plot_summary(aggregate_rows, output_dir=args.output_dir)
    print(raw_csv)
    print(aggregate_csv)
    for path in plot_paths:
        print(path)


if __name__ == "__main__":
    main()
