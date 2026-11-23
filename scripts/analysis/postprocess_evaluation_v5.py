#!/usr/bin/env python3
"""Render evaluation-v5 figures and decoder-dependent statistical analyses."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for import_root in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-evaluation-v5")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp")

import matplotlib

matplotlib.use("Agg", force=True)
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

from scripts.analysis.study_selection import PROTOCOL_ID, STUDIES_ROOT, load_study_protocol
from scripts.analysis.evaluate_v5 import (
    EvaluationV5MatchedBudgetProtocol,
    EvaluationV5Protocol,
)
from scripts.analysis.plot_config_metric_trajectories import (
    DEFAULT_LEVEL_ORDER,
    LEVEL_LABELS,
    SERIES_COLORS,
    SERIES_LABELS,
    SERIES_ORDER,
    load_rows,
    rows_at_comparison_steps,
)
from scripts.analysis import planned_paired_contrasts
from scripts.analysis import mercury_source_initialisation_paired_contrasts
from scripts.analysis import plot_pass_2_to_1000_metric_delta
from mercury_runs.analysis.residual_map_rendering import save_selected_figure
from mercury_runs.analysis.plots.selected_residual_maps import SCENARIOS


FOCUSED_METRICS = (
    ("latent_precision", "Latent purity", True),
    ("action_edge_f1_auc", "Action edge F1 AUC", True),
    ("action_tv_similarity", "Action TV similarity", True),
    ("eval_nll_bps", "Evaluation NLL", False),
)
CHECKPOINTS = (0, 2_000, 4_000, 2_000_000)
CHECKPOINT_LABELS = {0: "Initial", 2_000: "2k", 4_000: "4k", 2_000_000: "2m"}


def _run(command: list[str]) -> None:
    print("+ " + " ".join(command), flush=True)
    environment = os.environ.copy()
    python_path = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = f"{PROJECT_ROOT}:{PROJECT_ROOT / 'src'}" + (
        f":{python_path}" if python_path else ""
    )
    subprocess.run(command, cwd=PROJECT_ROOT, env=environment, check=True)


def _series_order(rows: pd.DataFrame) -> list[str]:
    present = set(str(value) for value in rows["series"].unique())
    return [value for value in SERIES_ORDER if value in present]


def plot_initial_profile(rows: pd.DataFrame, output_dir: Path) -> list[Path]:
    initial = rows[rows["step"] == 0].copy()
    if initial.empty:
        raise RuntimeError("No evaluation-v5 step-0 rows were found.")
    series_order = _series_order(initial)
    figure, axes = plt.subplots(1, len(FOCUSED_METRICS), figsize=(10.2, 3.35), dpi=300)
    for axis, (metric, label, higher_is_better) in zip(np.ravel(axes), FOCUSED_METRICS):
        panel = initial[initial["metric"] == metric]
        values = [panel.loc[panel["series"] == series, "value"].to_numpy(float) for series in series_order]
        means = [float(np.mean(value)) if value.size else np.nan for value in values]
        cis = [
            float(1.96 * np.std(value, ddof=1) / np.sqrt(value.size)) if value.size > 1 else 0.0
            for value in values
        ]
        x = np.arange(len(series_order))
        axis.bar(
            x,
            means,
            yerr=cis,
            color=[SERIES_COLORS[series] for series in series_order],
            capsize=2.2,
            linewidth=0,
        )
        axis.set_xticks(x, [SERIES_LABELS[series] for series in series_order], rotation=42, ha="right")
        axis.set_title(label + (" ↑" if higher_is_better else " ↓"), fontsize=9)
        axis.grid(axis="y", color="#D7DEE7", linewidth=0.6)
        axis.set_axisbelow(True)
        if higher_is_better:
            axis.set_ylim(0.0, 1.0)
    figure.suptitle("Evaluation-v5: frozen CSCG structures before training", fontsize=11, fontweight="bold")
    figure.subplots_adjust(left=0.055, right=0.99, bottom=0.31, top=0.80, wspace=0.30)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = [
        output_dir / "evaluation_v5_initial_structure_profile.png",
        output_dir / "evaluation_v5_initial_structure_profile.pdf",
    ]
    for path in paths:
        figure.savefig(path)
    plt.close(figure)
    return paths


def plot_checkpoint_trajectories(
    rows: pd.DataFrame, output_dir: Path, *, matched_budget: bool = False
) -> list[Path]:
    checkpoints = CHECKPOINTS
    checkpoint_labels = CHECKPOINT_LABELS
    selected = rows[rows["step"].isin(checkpoints)].copy()
    series_order = _series_order(selected)
    levels = [value for value in DEFAULT_LEVEL_ORDER if value in set(selected["level"])]
    figure, axes = plt.subplots(
        len(FOCUSED_METRICS),
        len(levels),
        figsize=(2.55 * len(levels) + 1.4, 1.65 * len(FOCUSED_METRICS) + 1.1),
        sharex=True,
        dpi=300,
    )
    axes = np.asarray(axes).reshape(len(FOCUSED_METRICS), len(levels))
    for row_index, (metric, label, higher_is_better) in enumerate(FOCUSED_METRICS):
        for column, level in enumerate(levels):
            axis = axes[row_index, column]
            panel = selected[(selected["metric"] == metric) & (selected["level"] == level)]
            for series in series_order:
                line = panel[panel["series"] == series]
                summary = line.groupby("step")["value"].agg(["mean", "std", "count"]).reindex(checkpoints)
                if summary["mean"].isna().all():
                    continue
                ci = 1.96 * summary["std"].fillna(0.0) / np.sqrt(summary["count"].clip(lower=1))
                x = np.arange(len(checkpoints))
                axis.errorbar(
                    x,
                    summary["mean"].to_numpy(float),
                    yerr=ci.to_numpy(float),
                    color=SERIES_COLORS[series],
                    marker="o",
                    markersize=2.7,
                    linewidth=1.15,
                    capsize=1.8,
                    label=SERIES_LABELS[series],
                )
            axis.grid(axis="y", color="#D7DEE7", linewidth=0.55)
            axis.set_axisbelow(True)
            if row_index == 0:
                axis.set_title(LEVEL_LABELS.get(level, f"L{level}"), fontsize=9, fontweight="bold")
            if column == 0:
                axis.set_ylabel(label + (" ↑" if higher_is_better else " ↓"), fontsize=7.8)
            if higher_is_better:
                axis.set_ylim(0.0, 1.0)
            if row_index == len(FOCUSED_METRICS) - 1:
                axis.set_xticks(np.arange(len(checkpoints)), [checkpoint_labels[value] for value in checkpoints])
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, legend_labels, loc="upper center", ncol=max(1, len(legend_labels)), frameon=False, fontsize=7.4)
    figure.suptitle(
        "Evaluation-v5: total-sample checkpoint trajectories"
        if matched_budget else "Evaluation-v5: CSCG training-checkpoint trajectories",
        y=0.995, fontsize=10.5, fontweight="bold",
    )
    figure.subplots_adjust(left=0.10, right=0.995, bottom=0.075, top=0.91, hspace=0.18, wspace=0.14)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = "evaluation_v5_checkpoints_0_2000_4000_2000000_trajectories"
    paths = [output_dir / f"{stem}.png", output_dir / f"{stem}.pdf"]
    for path in paths:
        figure.savefig(path)
    plt.close(figure)
    return paths


def render_combined_metric_profile(*, study_root: Path, output_dir: Path, protocol: Any) -> list[Path]:
    from scripts.analysis import plot_combined_metric_profiles_environment_shapes as profile

    profile.STUDY_ROOT = study_root
    profile.OUTPUT_DIR = output_dir
    profile.EvaluationProtocol = lambda **_kwargs: protocol
    profile.main()
    stem = profile.OUTPUT_STEM
    return [
        output_dir / f"{stem}{suffix}.{extension}"
        for suffix in (
            "", "_one_pass", "_two_passes", "_extensive",
            "_one_pass_no_oracles", "_two_passes_no_oracles", "_extensive_no_oracles",
        )
        for extension in ("png", "pdf")
    ]


def _select_residual_panels(
    payload: dict[str, Any], *, methods: tuple[str, ...], scenario: str, output_name: str
) -> dict[str, Any]:
    panels = dict(payload.get("panels") or {})
    levels = sorted(
        {int(key.split("/", 1)[0][1:]) for key in panels},
        key=lambda level: DEFAULT_LEVEL_ORDER.index(level),
    )
    selected: dict[str, Any] = {}
    for level in levels:
        for method in methods:
            key = f"L{level}/{method}"
            if key not in panels:
                raise KeyError(f"Required residual panel is missing: {key}")
            selected[key] = panels[key]
    return {
        **payload,
        "scenario": scenario,
        "output_name": output_name,
        "panels": selected,
    }


def _write_main_study_residual_panel(
    *, source_json: Path, output_dir: Path, methods: tuple[str, ...], checkpoint: int
) -> Path:
    payload = _select_residual_panels(
        json.loads(source_json.read_text(encoding="utf-8")),
        methods=methods,
        scenario=(
            "Matched-step action-conditioned transition residuals after one training pass"
            if int(checkpoint) == 2_000
            else "Ground-truth-aligned action-conditioned transition residuals after two training passes"
        ),
        output_name=(
            "transition_residual_action_conditioned_rand_0p1_batch_40_selected_"
            f"main_study_checkpoint_{int(checkpoint)}.png"
        ),
    )
    output_path = output_dir / str(payload["output_name"])
    payload["figure"] = str(output_path)
    payload["json"] = str(output_path.with_suffix(".json"))
    output_path.with_suffix(".json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    visual_encoding = dict(payload.get("visual_encoding") or {})
    save_selected_figure(
        payload=payload,
        output_path=output_path,
        vis_threshold=float(visual_encoding.get("vis_threshold", 0.02)),
        max_residual_override=float(visual_encoding["global_residual_scale"]),
        font_scale=2.30,
        header_font_scale=1.92,
        omit_leading_zero=True,
        show_summary_bar_labels=False,
    )
    return output_path


def combine_residual_checkpoints(input_dir: Path, output_dir: Path) -> list[Path]:
    from scripts.analysis.plot_combined_residual_checkpoints import (
        render_combined_residual_checkpoints,
    )

    all_methods_stem = (
        "transition_residual_action_conditioned_rand_0p1_batch_40_selected_"
        "all_methods_checkpoint_"
    )
    one_pass = _write_main_study_residual_panel(
        source_json=input_dir / f"{all_methods_stem}2000.json",
        output_dir=input_dir,
        methods=("MERCURY", "TMGWR", "Standard CSCG"),
        checkpoint=2_000,
    )
    two_passes = _write_main_study_residual_panel(
        source_json=input_dir / f"{all_methods_stem}4000.json",
        output_dir=input_dir,
        methods=(
            "MERCURY initialised",
            "Standard CSCG",
            "Oracle capacity",
            "Oracle capacity + first-order structure",
        ),
        checkpoint=4_000,
    )
    stem = (
        "transition_residual_action_conditioned_rand_0p1_batch_40_selected_"
        "main_study_checkpoints_2000_4000_combined"
    )
    return render_combined_residual_checkpoints(
        one_pass_payload=json.loads(one_pass.with_suffix(".json").read_text(encoding="utf-8")),
        two_pass_payload=json.loads(two_passes.with_suffix(".json").read_text(encoding="utf-8")),
        output_stem=output_dir / stem,
    )


def run_main_study_products(
    *, study_root: Path, output_root: Path, protocol: Any, python: str
) -> list[Path]:
    protocol_id = protocol.protocol_id()
    outputs: list[Path] = []
    rows = load_rows(study_root, protocol_id, 40, set(DEFAULT_LEVEL_ORDER))
    outputs.extend(plot_initial_profile(rows, output_root / "evaluation_v5"))
    comparison_rows = rows_at_comparison_steps(rows, protocol_id)
    outputs.extend(
        plot_checkpoint_trajectories(
            comparison_rows,
            output_root / "evaluation_v5",
            matched_budget=comparison_rows is not rows,
        )
    )
    outputs.extend(
        render_combined_metric_profile(
            study_root=study_root,
            output_dir=output_root / "forest_matrix_profile",
            protocol=protocol,
        )
    )

    residual_dir = output_root / "residual_maps" / "rand_0p1" / "action_conditioned" / "without_self_loops"
    residual_run_started_ns = time.time_ns()
    _run(
        [
            python,
            "-m",
            "mercury_runs.analysis.plots.selected_residual_maps",
            "--study_root",
            str(study_root),
            "--protocol_id",
            protocol_id,
            "--output_dir",
            str(residual_dir),
            "--rand_prob",
            "0.1",
            "--action_conditioned",
            "--compact_variants",
        ]
    )
    residual_summary_path = residual_dir / "summary_action_conditioned_rand_0p1.json"
    residual_summary = json.loads(residual_summary_path.read_text(encoding="utf-8"))
    figures = dict(residual_summary.get("figures") or {})
    if len(figures) != len(SCENARIOS):
        raise RuntimeError(
            f"Residual suite produced {len(figures)} scenarios; expected {len(SCENARIOS)}."
        )
    residual_outputs = [residual_summary_path]
    for figure_outputs in figures.values():
        for path_text in figure_outputs.values():
            path = Path(path_text)
            residual_outputs.append(path)
            if path.suffix.lower() == ".png":
                residual_outputs.append(path.with_suffix(".pdf"))
    missing_or_stale = [
        path
        for path in residual_outputs
        if not path.is_file() or path.stat().st_mtime_ns < residual_run_started_ns
    ]
    if missing_or_stale:
        raise RuntimeError(f"Residual outputs missing or stale after regeneration: {missing_or_stale}")
    outputs.extend(dict.fromkeys(residual_outputs))
    outputs.extend(combine_residual_checkpoints(residual_dir, residual_dir))

    stats_dir = output_root / "statistical_analysis"
    if planned_paired_contrasts.main(
        [
            "--study-root",
            str(study_root),
            "--output-dir",
            str(stats_dir),
            "--protocol-id",
            protocol_id,
        ]
    ) != 0:
        raise RuntimeError("Planned paired contrasts failed.")

    pass_dir = output_root / "pass_2_to_1000_statistical_analysis"
    if plot_pass_2_to_1000_metric_delta.main(
        [
            "--study-root",
            str(study_root),
            "--output-dir",
            str(pass_dir),
            "--protocol-id",
            protocol_id,
            "--batch-size",
            "40",
        ]
    ) != 0:
        raise RuntimeError("Pass-2-to-1000 analysis failed.")

    likelihood_profile_command = [
        python,
        str(PROJECT_ROOT / "scripts" / "analysis" / "plot_one_pass_profile_with_metric_likelihood_spaces.py"),
        "--study-root",
        str(study_root),
        "--protocol-id",
        protocol_id,
        "--output-dir",
        str(pass_dir),
        "--batch-size",
        "40",
    ]
    _run(likelihood_profile_command)
    _run([*likelihood_profile_command, "--likelihood-ci-trial"])
    for suffix in ("", "_95ci_trial"):
        stem = pass_dir / f"pass_2_to_1000_one_pass_metric_profile_with_metric_likelihood_spaces{suffix}"
        outputs.extend((stem.with_suffix(".png"), stem.with_suffix(".pdf")))
        outputs.append(pass_dir / f"{stem.name}_caption.tex")

    source_initialisation_stats_dir = (
        pass_dir / "mercury_source_vs_initialisation_statistical_analysis"
    )
    if mercury_source_initialisation_paired_contrasts.main(
        [
            "--study-root",
            str(study_root),
            "--output-dir",
            str(source_initialisation_stats_dir),
            "--protocol-id",
            protocol_id,
            "--batch-size",
            "40",
        ]
    ) != 0:
        raise RuntimeError("MERCURY source/initialisation paired contrasts failed.")

    threshold_root = output_root / "threshold_edge_robustness" / "rand_0p1"
    scenarios = "all_methods_checkpoint_2000,all_methods_checkpoint_4000,all_methods_final"
    for include_self_loops in (False, True):
        loop_slug = "with_self_loops" if include_self_loops else "without_self_loops"
        summaries: dict[str, Path] = {}
        for action_conditioned in (False, True):
            scope = "action_conditioned" if action_conditioned else "collapsed"
            metrics_dir = threshold_root / "metrics" / loop_slug / scope
            command = [
                python,
                "-m",
                "mercury_runs.analysis.plots.threshold_edge_metrics",
                "--study_root",
                str(study_root),
                "--protocol_id",
                protocol_id,
                "--output_dir",
                str(metrics_dir),
                "--rand_prob",
                "0.1",
                "--scenarios",
                scenarios,
                "--no_plots",
            ]
            if include_self_loops:
                command.append("--include_self_loops")
            if action_conditioned:
                command.append("--action_conditioned")
            _run(command)
            summaries[scope] = metrics_dir / "threshold_edge_metrics_summary.csv"
        robustness_dir = threshold_root / "robustness" / loop_slug
        _run(
            [
                python,
                "-m",
                "mercury_runs.analysis.plots.threshold_robustness",
                "--collapsed_summary",
                str(summaries["collapsed"]),
                "--action_summary",
                str(summaries["action_conditioned"]),
                "--output_dir",
                str(robustness_dir),
                "--scenarios",
                scenarios,
                "--compact",
            ]
        )
        outputs.append(robustness_dir / "threshold_sensitivity_action_edge_f1_compact_4x3.png")
    return outputs


def run_batch_sweep_products(
    *, study_root: Path, output_root: Path, protocol: Any, python: str
) -> list[Path]:
    output_dir = output_root / "batch_size_sensitivity"
    _run(
        [
            python,
            "scripts/analysis/plot_cscg_batch_size_sensitivity.py",
            "--study-root",
            str(study_root),
            "--protocol-id",
            protocol.protocol_id(),
            "--output-dir",
            str(output_dir),
            "--batch-sizes",
            "10,40,100",
            "--levels",
            "13,20,21,22",
        ]
    )
    return [
        output_dir / "cscg_batch_size_final_by_seed.csv",
        output_dir / "cscg_batch_size_final_summary.csv",
        output_dir / "cscg_batch_size_adjacent_gains.csv",
        output_dir / "cscg_batch_size_sensitivity_dual_axis.png",
        output_dir / "cscg_batch_size_sensitivity_dual_axis.pdf",
    ]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--evaluation-studies-root",
        type=Path,
        default=STUDIES_ROOT,
    )
    parser.add_argument(
        "--notebook-plots-root", type=Path, default=Path("results/current/notebook_plots")
    )
    parser.add_argument("--studies", nargs="+", default=["main_study", "cscg_batch_sweep"])
    parser.add_argument("--protocol-id", default=None, help="Expected protocol ID (defaults to the selected v5 ID).")
    return parser


def _expected_protocol_id(
    study_root: Path,
    evaluation_studies_root: Path,
    study_name: str,
    explicit_protocol_id: str | None,
) -> str | None:
    if explicit_protocol_id is not None:
        return explicit_protocol_id
    if evaluation_studies_root.resolve() != STUDIES_ROOT.resolve():
        return None
    schema_version = json.loads((study_root / "protocol.json").read_text(encoding="utf-8")).get("schema_version")
    if schema_version == EvaluationV5Protocol.schema_version:
        return PROTOCOL_ID
    if study_name == "main_study" and schema_version == EvaluationV5MatchedBudgetProtocol.schema_version:
        return None
    raise ValueError(f"Unsupported evaluation-v5 protocol schema for {study_name}: {schema_version}")


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    python = sys.executable
    all_outputs: dict[str, list[str]] = {}
    for study_name in args.studies:
        study_root = Path(args.evaluation_studies_root) / study_name
        expected_protocol_id = _expected_protocol_id(
            study_root,
            Path(args.evaluation_studies_root),
            study_name,
            args.protocol_id,
        )
        protocol = load_study_protocol(study_root, expected_protocol_id)
        output_root = Path(args.notebook_plots_root) / f"{study_name}_eval_v5"
        if study_name == "main_study":
            outputs = run_main_study_products(
                study_root=study_root,
                output_root=output_root,
                protocol=protocol,
                python=python,
            )
        elif study_name == "cscg_batch_sweep":
            outputs = run_batch_sweep_products(
                study_root=study_root,
                output_root=output_root,
                protocol=protocol,
                python=python,
            )
        else:
            raise ValueError(f"No evaluation-v5 post-processing definition for {study_name!r}.")
        missing = [path for path in outputs if not path.exists()]
        if missing:
            raise RuntimeError(f"Missing expected {study_name} outputs: {missing}")
        all_outputs[study_name] = [str(path) for path in outputs]
        output_root.mkdir(parents=True, exist_ok=True)
        (output_root / "evaluation_v5_manifest.json").write_text(
            json.dumps(
                {
                    "source_shadow_study": str(study_root),
                    "protocol_id": protocol.protocol_id(),
                    "outputs": all_outputs[study_name],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    print(json.dumps(all_outputs, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
