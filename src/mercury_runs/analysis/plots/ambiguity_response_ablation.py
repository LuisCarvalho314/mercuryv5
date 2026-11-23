"""Canonical plot suite adapted to the four ambiguity-response modes.

This module only adapts study-record selection and labels. Figure rendering is
delegated to the same canonical plot functions used by evaluate_and_plot.sh.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib
import polars as pl

matplotlib.use("Agg", force=True)

from mercury_runs.analysis.evaluation_records import (
    diagnostics_from_record,
    edge_metrics_at_threshold,
    load_evaluation_step,
)
from mercury_runs.analysis.plots import batch_summary_forest as forest
from mercury_runs.analysis.plots import batch_tv_similarity as tv
from mercury_runs.analysis.plots import threshold_edge_metrics as threshold_data
from mercury_runs.analysis.plots import threshold_robustness as robustness
from mercury_runs.analysis.plots.selected_residual_maps import _aggregate_diagnostics
from mercury_runs.analysis.residual_map_rendering import save_selected_figure


MODE_ORDER = (
    "full_mercury",
    "refinement_without_reconstruction",
    "local_repair_only",
    "no_ambiguity_response",
)
MODE_LABELS = {
    "full_mercury": "Full MERCURY",
    "refinement_without_reconstruction": "Refinement without reconstruction",
    "local_repair_only": "Local repair only",
    "no_ambiguity_response": "No ambiguity response",
}
MODE_COLORS = {
    "full_mercury": "#009E73",
    "refinement_without_reconstruction": "#D55E00",
    "local_repair_only": "#0072B2",
    "no_ambiguity_response": "#5F6368",
}
COMPACT_MODE_LABELS = {
    "full_mercury": "Full MERCURY",
    "refinement_without_reconstruction": "No reconstruction",
    "local_repair_only": "Local repair",
    "no_ambiguity_response": "No response",
}
SCENARIO_KEY = "ambiguity_response_ablation"
DEFAULT_THRESHOLDS = threshold_data.DEFAULT_THRESHOLDS


def _load_rows(study_root: Path, protocol_id: str) -> list[dict[str, Any]]:
    manifest_path = study_root / "study_manifest.jsonl"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Study manifest not found: {manifest_path}")

    rows: list[dict[str, Any]] = []
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        config = dict(json.loads(line)["config"])
        mode = str(config.get("ambiguity_response_mode", ""))
        if str(config.get("method")) != "mercury" or mode not in MODE_ORDER:
            continue
        run_id = str(config.get("requested_run_id") or config.get("run_id"))
        run_root = study_root / run_id
        record = load_evaluation_step(run_root, protocol_id, method="mercury")
        if record is None:
            continue
        rows.append(
            {
                "run_id": run_id,
                "run_root": run_root,
                "level": int(config["level"]),
                "seed": int(config["seed"]),
                "rand_prob": float(config["rand_prob"]),
                "mode": mode,
                "record": record,
            }
        )
    if not rows:
        raise RuntimeError(f"No MERCURY records for protocol {protocol_id} under {study_root}")
    return rows


def _configure_canonical_styles(levels: list[int]) -> None:
    forest.LEVEL_ORDER[:] = levels
    forest.SERIES_LABELS.update(MODE_LABELS)
    forest.SERIES_COLORS.update(MODE_COLORS)
    threshold_data.METHOD_COLORS.update({MODE_LABELS[key]: value for key, value in MODE_COLORS.items()})
    robustness.LEVEL_ORDER = tuple(levels)
    robustness.METHOD_LABELS.update({MODE_LABELS[key]: MODE_LABELS[key] for key in MODE_ORDER})
    robustness.METHOD_STYLES.update(
        {
            MODE_LABELS[key]: {"color": MODE_COLORS[key], "linestyle": "-"}
            for key in MODE_ORDER
        }
    )
    robustness.METHOD_ORDER.update({MODE_LABELS[key]: index for index, key in enumerate(MODE_ORDER)})
    robustness.SCENARIOS[SCENARIO_KEY] = {
        "title": "Threshold robustness of ambiguity-response ablations",
        "subtitle": "Final checkpoint; ribbons show 95% confidence intervals",
        "paired_output": "threshold_robustness_edge_f1_2x4.png",
    }


def _frame(rows: list[dict[str, Any]]) -> pl.DataFrame:
    return pl.DataFrame(
        [
            {
                "comparison": "final",
                "level": row["level"],
                "seed": row["seed"],
                "rand_prob": row["rand_prob"],
                "plot_series": row["mode"],
                **forest._metrics_from_record(row["record"]),
            }
            for row in rows
        ],
        infer_schema_length=None,
    )


def _summary_plots(frame: pl.DataFrame, output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    title = "Final ambiguity-response ablation"
    kwargs = {
        "frame": frame,
        "series_order": list(MODE_ORDER),
        "title": title,
        "plot_dir": output_dir,
        "series_labels": MODE_LABELS,
        "series_colors": MODE_COLORS,
    }
    outputs = [
        forest.plot_forest(output_name="ambiguity_response_forest_6x4.png", **kwargs),
        forest.plot_metric_matrix(
            output_name="ambiguity_response_metric_matrix.png",
            gain_label="Full MERCURY gain",
            **kwargs,
        ),
        forest.plot_metric_profile(output_name="ambiguity_response_metric_profile.png", **kwargs),
        forest.plot_metric_profile(
            output_name="ambiguity_response_metric_profile_focused.png",
            metric_keys=forest.PROFILE_FOCUSED_METRICS,
            **kwargs,
        ),
        forest.plot_metric_profile_transposed(
            output_name="ambiguity_response_metric_profile_focused_transposed.png",
            metric_keys=forest.PROFILE_FOCUSED_METRICS,
            **kwargs,
        ),
        forest.plot_metric_profile_transposed_compact(
            output_name="ambiguity_response_metric_profile_focused_transposed_compact.png",
            metric_keys=forest.PROFILE_FOCUSED_METRICS,
            **{**kwargs, "series_labels": COMPACT_MODE_LABELS},
        ),
    ]
    return outputs


def _tv_plots(frame: pl.DataFrame, output_dir: Path) -> tuple[list[Path], list[dict[str, Any]]]:
    rand_prob = float(frame["rand_prob"][0])
    outputs: list[Path] = []
    csv_rows: list[dict[str, Any]] = []
    for spec in tv.METRIC_SPECS:
        path = output_dir / spec["family"] / spec["self_loop_policy"] / f"ambiguity_response_{spec['family']}_{spec['self_loop_policy']}.png"
        outputs.append(
            tv._plot_one(
                frame,
                output_path=path,
                rand_prob=rand_prob,
                comparison_label="Final ambiguity-response ablation",
                metric=str(spec["column"]),
                metric_label=str(spec["label"]),
                self_loop_policy=str(spec["self_loop_policy"]),
                series_order=list(MODE_ORDER),
                primary_series="full_mercury",
                series_labels=MODE_LABELS,
                series_colors=MODE_COLORS,
            )
        )
        csv_rows.extend(
            tv._summaries_for_csv(
                frame,
                rand_prob=rand_prob,
                comparison="final",
                metric=str(spec["column"]),
                metric_label=str(spec["label"]),
                self_loop_policy=str(spec["self_loop_policy"]),
                series_order=list(MODE_ORDER),
                series_labels=MODE_LABELS,
            )
        )
    return outputs, csv_rows


def _residual_plots(rows: list[dict[str, Any]], output_dir: Path, levels: list[int]) -> list[Path]:
    outputs: list[Path] = []
    for action_conditioned in (False, True):
        for include_self_loops in (False, True):
            panels: dict[str, Any] = {}
            for level in levels:
                for mode in MODE_ORDER:
                    diagnostics = [
                        diagnostics_from_record(
                            row["record"],
                            method=MODE_LABELS[mode],
                            action_conditioned=action_conditioned,
                            ignore_self_loops=not include_self_loops,
                        )
                        for row in rows
                        if row["level"] == level and row["mode"] == mode
                    ]
                    if diagnostics:
                        panels[f"L{level}/{MODE_LABELS[mode]}"] = _aggregate_diagnostics(diagnostics).to_json_dict()
            scope = "action_conditioned" if action_conditioned else "collapsed"
            loops = "with_self_loops" if include_self_loops else "without_self_loops"
            path = output_dir / scope / loops / f"ambiguity_response_transition_residual_{scope}_{loops}.png"
            payload = {
                "scenario": "Ground-truth-aligned transition residuals: ambiguity-response ablation",
                "panels": panels,
                "visual_encoding": {"action_conditioned": action_conditioned},
            }
            save_selected_figure(payload=payload, output_path=path, vis_threshold=0.02)
            path.with_suffix(".json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
            outputs.append(path)
    return outputs


def _threshold_rows(
    rows: list[dict[str, Any]],
    *,
    thresholds: tuple[float, ...],
    action_conditioned: bool,
    include_self_loops: bool,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        for threshold in thresholds:
            metrics = edge_metrics_at_threshold(
                row["record"],
                threshold,
                action_conditioned=action_conditioned,
                include_self_loops=include_self_loops,
            )
            output.append(
                {
                    "scenario_key": SCENARIO_KEY,
                    "scenario_title": "Ambiguity-response ablation",
                    "level": row["level"],
                    "seed": row["seed"],
                    "method": MODE_LABELS[row["mode"]],
                    "run_kind": row["mode"],
                    "run_root": str(row["run_root"]),
                    "metric_scope": "action_conditioned" if action_conditioned else "collapsed",
                    "threshold": threshold,
                    **metrics,
                }
            )
    return output


def _threshold_plots(rows: list[dict[str, Any]], output_dir: Path) -> list[Path]:
    outputs: list[Path] = []
    for include_self_loops in (False, True):
        loop_dir = output_dir / ("with_self_loops" if include_self_loops else "without_self_loops")
        summary_paths: dict[str, Path] = {}
        for action_conditioned in (False, True):
            scope = "action_conditioned" if action_conditioned else "collapsed"
            scope_dir = loop_dir / scope
            scope_dir.mkdir(parents=True, exist_ok=True)
            raw = _threshold_rows(
                rows,
                thresholds=DEFAULT_THRESHOLDS,
                action_conditioned=action_conditioned,
                include_self_loops=include_self_loops,
            )
            aggregate = threshold_data._aggregate_rows(raw)
            raw_path = scope_dir / "threshold_edge_metrics_by_seed.csv"
            summary_path = scope_dir / "threshold_edge_metrics_summary.csv"
            threshold_data._write_csv(raw_path, raw)
            threshold_data._write_csv(summary_path, aggregate)
            summary_paths[scope] = summary_path
            outputs.extend((raw_path, summary_path))

        plot_dir = loop_dir / "robustness"
        robustness.BASE_OUTPUT_DIR = plot_dir
        action = robustness._read_summary(summary_paths["action_conditioned"], "action_conditioned")
        collapsed = robustness._read_summary(summary_paths["collapsed"], "collapsed")
        outputs.append(robustness.plot_final_paired(action, collapsed, scenario=SCENARIO_KEY))
        outputs.append(
            robustness.plot_full_metric_grid(
                action,
                scenario=SCENARIO_KEY,
                scope_label="Action-conditioned",
                output_name="threshold_metrics_action_conditioned_3x4.png",
            )
        )
        outputs.append(
            robustness.plot_full_metric_grid(
                collapsed,
                scenario=SCENARIO_KEY,
                scope_label="Policy-collapsed",
                output_name="threshold_metrics_collapsed_3x4.png",
            )
        )
    return outputs


def generate_plots(*, study_root: Path, output_dir: Path, protocol_id: str) -> list[Path]:
    rows = _load_rows(study_root, protocol_id)
    canonical_order = [13, 21, 20, 22]
    available = {int(row["level"]) for row in rows}
    levels = [level for level in canonical_order if level in available]
    levels.extend(sorted(available - set(levels)))
    _configure_canonical_styles(levels)
    frame = _frame(rows)

    outputs = _summary_plots(frame, output_dir / "forest_matrix_profile")
    tv_outputs, tv_rows = _tv_plots(frame, output_dir / "tv_similarity")
    outputs.extend(tv_outputs)
    tv_summary = output_dir / "tv_similarity" / "tv_similarity_summary.csv"
    if tv_rows:
        import csv

        with tv_summary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(tv_rows[0]))
            writer.writeheader()
            writer.writerows(tv_rows)
        outputs.append(tv_summary)
    outputs.extend(_residual_plots(rows, output_dir / "residual_maps", levels))
    outputs.extend(_threshold_plots(rows, output_dir / "threshold_edge_robustness"))
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study_root", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--protocol_id", required=True)
    args = parser.parse_args()
    for path in generate_plots(
        study_root=args.study_root,
        output_dir=args.output_dir,
        protocol_id=args.protocol_id,
    ):
        print(path)


if __name__ == "__main__":
    main()
