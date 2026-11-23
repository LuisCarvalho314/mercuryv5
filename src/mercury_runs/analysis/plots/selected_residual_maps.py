"""Generate ground-truth-aligned transition residual map figures.

Canonical: consumes evaluation records produced by `evaluate_study`.
Must not load model snapshots, generate walks, or decode sequences.

Usage:
    python -m mercury_runs.analysis.plots.selected_residual_maps \\
        --study_root results/current/studies/<study> \\
        --protocol_id canonical-v4-<hash>
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import replace, dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib
import numpy as np

from mercury_runs.analysis.evaluation_records import (
    diagnostics_from_record,
    load_evaluation_step,
    load_study_manifest,
)
from mercury_runs.analysis.residual_map_rendering import (
    save_selected_figure,
    save_selected_figure_compact,
)
from mercury_runs.analysis.transition_residual_maps import TransitionResidualDiagnostics
from mercury_runs.analysis.plots import publication_style as pub


DEFAULT_STUDY_ROOT = Path(
    "results/current/studies"
    "/mercury_cscg_hybrid_vs_cscg_batch_size_30_seed_common_eval_walks2"
)
DEFAULT_OUTPUT_DIR = Path(
    "results/current/notebook_plots/"
    "mercury_cscg_hybrid_vs_cscg_batch_size_30_seed_common_eval_walk_transition_residual_maps"
)
LEVELS = (13, 20, 21, 22)
SEEDS = tuple(range(30))
BATCH_SIZE = int(os.environ.get("PLOT_CSCG_BATCH_SIZE", "40"))
VIS_THRESHOLD = 0.02
ALL_CSCG_PRESETS = (
    *((pub.method_label(preset), preset) for preset in pub.CSCG_VARIANT_ORDER),
    ("CSCG — oracle clones/state (×1)", "ground_truth_state_1"),
    ("CSCG — oracle clones/state (×2)", "ground_truth_state_2"),
    ("CSCG — observation clones (×4)", "uniform_obs_4"),
    ("CSCG — observation clones (×8)", "uniform_obs_8"),
)
_CSCG_PRESET_LOOKUP = {preset: (label, preset) for label, preset in ALL_CSCG_PRESETS}


def _selected_cscg_presets() -> tuple[tuple[str, str], ...]:
    raw = os.environ.get("PLOT_CSCG_PRESETS")
    if not raw:
        return ALL_CSCG_PRESETS
    selected = []
    for preset in (item.strip() for item in raw.split(",") if item.strip()):
        if preset not in _CSCG_PRESET_LOOKUP:
            raise ValueError(f"Unknown PLOT_CSCG_PRESETS item: {preset}")
        selected.append(_CSCG_PRESET_LOOKUP[preset])
    return tuple(selected)


CSCG_PRESETS = _selected_cscg_presets()


@dataclass(frozen=True)
class ArtifactSpec:
    manifest_run_key: str
    phase_step: int | None
    evaluation_method: str
    cumulative_step: int | None
    phase: str

# Maps scenario kind name to (manifest_run_key, step_or_None, evaluation_method_namespace)
# manifest_run_key selects the run from manifest: "hybrid" = mercury_cscg_hybrid runs, "cscg" = cscg runs.
_KIND_TO_RUN: dict[str, ArtifactSpec] = {
    "mercury_checkpoint_2000": ArtifactSpec(
        manifest_run_key="hybrid",
        phase_step=2000,
        evaluation_method="mercury_cscg_hybrid_mercury",
        cumulative_step=2000,
        phase="mercury",
    ),

    "hybrid_cscg_checkpoint_4000": ArtifactSpec(
        manifest_run_key="hybrid",
        phase_step=2000,
        evaluation_method="mercury_cscg_hybrid",
        cumulative_step=4000,
        phase="cscg",
    ),

    "hybrid_final": ArtifactSpec(
        manifest_run_key="hybrid",
        phase_step=None,
        evaluation_method="mercury_cscg_hybrid",
        cumulative_step=None,
        phase="cscg",
    ),

    # TMGWR-phase records are written under the "tmgwr_cscg_hybrid_tmgwr"
    # namespace, mirroring "mercury_cscg_hybrid_mercury" for the MERCURY phase.
    "tmgwr_hybrid_cscg_checkpoint_2000": ArtifactSpec(
        manifest_run_key="tmgwr_hybrid",
        phase_step=2000,
        evaluation_method="tmgwr_cscg_hybrid_tmgwr",
        cumulative_step=2000,
        phase="tmgwr",
    ),

    "tmgwr_hybrid_cscg_checkpoint_4000": ArtifactSpec(
        manifest_run_key="tmgwr_hybrid",
        phase_step=2000,
        evaluation_method="tmgwr_cscg_hybrid",
        cumulative_step=4000,
        phase="cscg",
    ),

    "tmgwr_hybrid_final": ArtifactSpec(
        manifest_run_key="tmgwr_hybrid",
        phase_step=None,
        evaluation_method="tmgwr_cscg_hybrid",
        cumulative_step=None,
        phase="cscg",
    ),

    "cscg_checkpoint_2000": ArtifactSpec(
        manifest_run_key="cscg",
        phase_step=2000,
        evaluation_method="cscg",
        cumulative_step=2000,
        phase="cscg",
    ),

    "cscg_checkpoint_4000": ArtifactSpec(
        manifest_run_key="cscg",
        phase_step=4000,
        evaluation_method="cscg",
        cumulative_step=4000,
        phase="cscg",
    ),

    "cscg_final": ArtifactSpec(
        manifest_run_key="cscg",
        phase_step=None,
        evaluation_method="cscg",
        cumulative_step=None,
        phase="cscg",
    ),
}

SCENARIOS = (
    {
        "key": "all_methods_final",
        "title": "Matched-step transition residuals: final comparison",
        "targets": (
            (pub.method_label("hybrid"), "hybrid_final"),
            (pub.method_label("tmgwr_hybrid"), "tmgwr_hybrid_final"),
        ),
        "cscg_kind": "cscg_final",
        "output_name": "transition_residual_batch_40_selected_all_methods_final.png",
    },
    {
        "key": "all_methods_checkpoint_4000",
        "title": "Matched-step transition residuals after two training passes",
        "targets": (
            (pub.method_label("hybrid"), "hybrid_cscg_checkpoint_4000"),
            (pub.method_label("tmgwr_hybrid"), "tmgwr_hybrid_cscg_checkpoint_4000"),
        ),
        "cscg_kind": "cscg_checkpoint_4000",
        "output_name": "transition_residual_batch_40_selected_all_methods_checkpoint_4000.png",
    },
    {
        "key": "all_methods_checkpoint_2000",
        "title": "Matched-step transition residuals after one training pass",
        "targets": (
            ("MERCURY", "mercury_checkpoint_2000"),
            ("TMGWR", "tmgwr_hybrid_cscg_checkpoint_2000"),
        ),
        "cscg_kind": "cscg_checkpoint_2000",
        "font_scale": 2.30,
        "header_font_scale": 1.60,
        "omit_leading_zero": True,
        "output_name": "transition_residual_batch_40_selected_all_methods_checkpoint_2000.png",
    },
    {
        "key": "final_hybrid_vs_cscg",
        "title": "Ground-truth-aligned transition residuals: MERCURY initialised versus CSCG",
        "target_label": pub.method_label("hybrid"),
        "target_kind": "hybrid_final",
        "cscg_kind": "cscg_final",
        "font_scale": 2.30,
        "header_font_scale": 1.92,
        "omit_leading_zero": True,
        "output_name": "transition_residual_batch_40_selected_final_hybrid_vs_cscg.png",
    },
    {
        "key": "final_tmgwr_vs_cscg",
        "title": "Ground-truth-aligned transition residuals: TMGWR initialised versus CSCG",
        "target_label": pub.method_label("tmgwr_hybrid"),
        "target_kind": "tmgwr_hybrid_final",
        "cscg_kind": "cscg_final",
        "output_name": "transition_residual_batch_40_selected_final_tmgwr_vs_cscg.png",
    },
    {
        "key": "tmgwr_hybrid_vs_cscg_checkpoint_4000",
        "title": "Ground-truth-aligned transition residuals after two training passes: TMGWR initialised versus CSCG",
        "target_label": pub.method_label("tmgwr_hybrid"),
        "target_kind": "tmgwr_hybrid_cscg_checkpoint_4000",
        "cscg_kind": "cscg_checkpoint_4000",
        "output_name": "transition_residual_batch_40_selected_tmgwr_hybrid_vs_cscg_checkpoint_4000.png",
    },
    {
        "key": "hybrid_vs_cscg_checkpoint_4000",
        "title": "Ground-truth-aligned transition residuals after two training passes",
        "target_label": pub.method_label("hybrid"),
        "target_kind": "hybrid_cscg_checkpoint_4000",
        "cscg_kind": "cscg_checkpoint_4000",
        "font_scale": 2.30,
        "header_font_scale": 1.92,
        "omit_leading_zero": True,
        "output_name": "transition_residual_batch_40_selected_hybrid_vs_cscg_checkpoint_4000.png",
    },
    {
        "key": "mercury_vs_cscg_checkpoint_2000",
        "title": "Ground-truth-aligned transition residuals after one training pass",
        "target_label": "MERCURY",
        "target_kind": "mercury_checkpoint_2000",
        "cscg_kind": "cscg_checkpoint_2000",
        "font_scale": 2.30,
        "header_font_scale": 1.92,
        "omit_leading_zero": True,
        "output_name": "transition_residual_batch_40_selected_mercury_vs_cscg_checkpoint_2000.png",
    },
)




def _hybrid_run(
    manifest: dict[tuple[Any, ...], Path],
    level: int,
    seed: int,
    *,
    rand_prob: float,
    method: str = "mercury_cscg_hybrid",
) -> Path:
    return manifest[(str(method), int(level), int(seed), float(rand_prob), BATCH_SIZE, None)]


def _cscg_run(
    manifest: dict[tuple[Any, ...], Path],
    level: int,
    seed: int,
    preset: str,
    *,
    rand_prob: float,
) -> Path:
    return manifest[("cscg", int(level), int(seed), float(rand_prob), BATCH_SIZE, str(preset))]


def _diagnostic_for_run(
    run_root: Path,
    kind: str,
    label: str,
    protocol_id: str,
    *,
    include_self_loops: bool,
    action_conditioned: bool,
) -> TransitionResidualDiagnostics:
    spec = _KIND_TO_RUN[kind]

    record = load_evaluation_step(
        run_root,
        protocol_id,
        spec.phase_step,
        method=spec.evaluation_method,
    )

    if record is None:
        raise FileNotFoundError(
            f"No evaluation record for kind={kind!r}\n"
            f"  run_root={run_root}\n"
            f"  phase={spec.phase!r}\n"
            f"  phase_step={spec.phase_step!r}\n"
            f"  cumulative_step={spec.cumulative_step!r}\n"
            f"  evaluation_method={spec.evaluation_method!r}\n"
            f"Run evaluate_study first with protocol_id={protocol_id!r}."
        )

    return diagnostics_from_record(
        record,
        method=label,
        action_conditioned=bool(action_conditioned),
        ignore_self_loops=not bool(include_self_loops),
    )


def _aggregate_diagnostics(diagnostics: list[TransitionResidualDiagnostics]) -> TransitionResidualDiagnostics:
    if not diagnostics:
        raise ValueError("No diagnostics to aggregate")
    count = float(len(diagnostics))
    transition_true = sum((item.transition_true for item in diagnostics), np.zeros_like(diagnostics[0].transition_true)) / count
    transition_predicted = (
        sum((item.transition_predicted for item in diagnostics), np.zeros_like(diagnostics[0].transition_predicted)) / count
    )
    residual = sum((item.residual for item in diagnostics), np.zeros_like(diagnostics[0].residual)) / count
    residual_magnitude = (
        sum(
            (
                np.asarray(item.residual_magnitude, dtype=np.float64)
                if item.residual_magnitude is not None
                else np.abs(item.residual)
                for item in diagnostics
            ),
            np.zeros_like(diagnostics[0].residual),
        )
        / count
    )
    tv_by_state = sum((item.tv_by_state for item in diagnostics), np.zeros_like(diagnostics[0].tv_by_state)) / count
    similarity_by_state = (
        sum((item.similarity_by_state for item in diagnostics), np.zeros_like(diagnostics[0].similarity_by_state)) / count
    )
    uncovered_fraction = (
        sum((item.uncovered_rows.astype(np.float64) for item in diagnostics), np.zeros_like(diagnostics[0].uncovered_rows, dtype=np.float64))
        / count
    )
    return replace(
        diagnostics[0],
        transition_true=transition_true,
        transition_predicted=transition_predicted,
        residual=residual,
        residual_magnitude=residual_magnitude,
        excess_mass=(
            sum((item.excess_mass for item in diagnostics), np.zeros_like(diagnostics[0].excess_mass))
            / count
        ),
        missing_mass=(
            sum((item.missing_mass for item in diagnostics), np.zeros_like(diagnostics[0].missing_mass))
            / count
        ),
        matched_mass=np.minimum(transition_true, transition_predicted),
        tv_by_state=tv_by_state,
        similarity_by_state=similarity_by_state,
        uncovered_rows=uncovered_fraction > 0.5,
        tv_sim=float(np.mean([item.tv_sim for item in diagnostics])),
    )


def _rand_prob_suffix(rand_prob: float) -> str:
    if abs(float(rand_prob)) <= 1e-12:
        return ""
    text = f"{float(rand_prob):g}".replace("-", "m").replace(".", "p")
    return f"_rand_{text}"


def _build_payload(
    manifest: dict[tuple[Any, ...], Path],
    scenario: dict[str, Any],
    protocol_id: str,
    *,
    rand_prob: float,
    include_self_loops: bool,
    action_conditioned: bool,
    levels: tuple[int, ...],
    seeds: tuple[int, ...],
) -> dict[str, Any]:
    panels: dict[str, Any] = {}
    raw_targets = scenario.get("targets")
    if raw_targets is None:
        raw_targets = ((str(scenario["target_label"]), str(scenario["target_kind"])),)
    targets = tuple((str(label), str(kind)) for label, kind in raw_targets)
    available_cscg_presets = {
        str(preset)
        for method, level, _seed, row_rand_prob, bs, preset in manifest
        if method == "cscg"
        and int(level) in set(levels)
        and abs(float(row_rand_prob) - float(rand_prob)) <= 1e-12
        and int(bs) == BATCH_SIZE
        and preset is not None
    }
    cscg_presets = tuple(
        (label, preset)
        for label, preset in CSCG_PRESETS
        if str(preset) in available_cscg_presets
    )
    for level in levels:
        for target_label, target_kind in targets:
            target_diags = []

            for seed in seeds:
                try:
                    spec = _KIND_TO_RUN[target_kind]

                    target_method = (
                        "tmgwr_cscg_hybrid"
                        if spec.manifest_run_key == "tmgwr_hybrid"
                        else "mercury_cscg_hybrid"
                    )

                    run_root = _hybrid_run(
                        manifest,
                        level,
                        seed,
                        rand_prob=rand_prob,
                        method=target_method,
                    )

                    diag = _diagnostic_for_run(
                        run_root,
                        target_kind,
                        target_label,
                        protocol_id,
                        include_self_loops=include_self_loops,
                        action_conditioned=action_conditioned,
                    )
                    target_diags.append(diag)

                except (KeyError, FileNotFoundError) as exc:
                    print(
                        f"  Warning: skipping {target_label} "
                        f"level={level} seed={seed}: {exc}",
                        file=sys.stderr,
                    )

            if target_diags:
                panels[f"L{level}/{target_label}"] = (
                    _aggregate_diagnostics(target_diags).to_json_dict()
                )

        for label, preset in cscg_presets:
            cscg_diags = []

            for seed in seeds:
                try:
                    run_root = _cscg_run(
                        manifest,
                        level,
                        seed,
                        preset,
                        rand_prob=rand_prob,
                    )

                    diag = _diagnostic_for_run(
                        run_root,
                        str(scenario["cscg_kind"]),
                        label,
                        protocol_id,
                        include_self_loops=include_self_loops,
                        action_conditioned=action_conditioned,
                    )
                    cscg_diags.append(diag)

                except (KeyError, FileNotFoundError) as exc:
                    print(
                        f"  Warning: skipping cscg preset={preset} "
                        f"level={level} seed={seed}: {exc}",
                        file=sys.stderr,
                    )

            if cscg_diags:
                panels[f"L{level}/{label}"] = (
                    _aggregate_diagnostics(cscg_diags).to_json_dict()
                )

        for label, preset in cscg_presets:
            cscg_diags = []
            for seed in seeds:
                try:
                    run_root = _cscg_run(manifest, level, seed, preset, rand_prob=rand_prob)
                    diag = _diagnostic_for_run(
                        run_root,
                        str(scenario["cscg_kind"]),
                        label,
                        protocol_id,
                        include_self_loops=include_self_loops,
                        action_conditioned=action_conditioned,
                    )
                    cscg_diags.append(diag)
                except (KeyError, FileNotFoundError) as exc:
                    print(f"  Warning: skipping cscg preset={preset} level={level} seed={seed}: {exc}", file=sys.stderr)
            if cscg_diags:
                panels[f"L{level}/{label}"] = _aggregate_diagnostics(cscg_diags).to_json_dict()

    output_name = str(scenario["output_name"])
    title = str(scenario["title"])
    output_name = output_name.replace("_batch_40_", f"{_rand_prob_suffix(rand_prob)}_batch_{BATCH_SIZE}_")
    if include_self_loops:
        output_name = output_name.replace("transition_residual_", "transition_residual_self_loops_")
        title = f"{title} (self-loops included)"
    if action_conditioned:
        output_name = output_name.replace("transition_residual_", "transition_residual_action_conditioned_")
        title = title.replace("transition residuals", "action-conditioned transition residuals")
    return {
        "scenario": title,
        "output_name": output_name,
        "font_scale": float(scenario.get("font_scale", 1.60)),
        "header_font_scale": float(
            scenario.get("header_font_scale", scenario.get("font_scale", 1.60))
        ),
        "omit_leading_zero": bool(scenario.get("omit_leading_zero", False)),
        "panels": panels,
        "visual_encoding": {
            "node_colour": "mean per-seed state-wise transition similarity",
            "edge_colour": "mean per-seed signed residual R_ij",
            "edge_width": "mean per-seed absolute residual |R_ij|",
            "residual_colormap": "RdBu_r",
            "vis_threshold": VIS_THRESHOLD,
            "ground_truth_source": f"evaluation records for protocol {protocol_id}",
            "rand_prob": float(rand_prob),
            "self_loops_included": bool(include_self_loops),
            "action_conditioned": bool(action_conditioned),
            "protocol_id": protocol_id,
        },
    }

def _record_method(record: dict[str, Any]) -> str | None:
    for key in (
        "method",
        "evaluation_method",
        "method_namespace",
        "evaluation_method_namespace",
    ):
        value = record.get(key)
        if value is not None:
            return str(value)
    return None

def _summary_path(output_dir: Path, *, include_self_loops: bool, action_conditioned: bool, rand_prob: float) -> Path:
    metric_prefix = "summary_action_conditioned" if action_conditioned else "summary"
    rand_suffix = _rand_prob_suffix(float(rand_prob))
    suffix = "_self_loops" if include_self_loops else ""
    return output_dir / f"{metric_prefix}{rand_suffix}{suffix}.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study_root", type=Path, default=DEFAULT_STUDY_ROOT)
    parser.add_argument("--protocol_id", type=str, required=True,
                        help="Canonical-v4 protocol ID (e.g. canonical-v4-<hash>)")
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--rand_prob", type=float, default=0.0)
    parser.add_argument("--include_self_loops", action="store_true")
    parser.add_argument("--action_conditioned", action="store_true")
    parser.add_argument(
        "--compact_variants",
        action="store_true",
        help="Also write separate single-column and supplementary condensed residual figures.",
    )
    parser.add_argument("--levels", type=str, default=",".join(str(l) for l in LEVELS))
    parser.add_argument("--seeds", type=str, default=f"0-{max(SEEDS)}")
    parser.add_argument("--scenarios", type=str, default=",".join(str(s["key"]) for s in SCENARIOS))
    args = parser.parse_args()

    matplotlib.use("Agg", force=True)

    levels = tuple(int(x) for x in args.levels.split(","))
    raw_seeds = str(args.seeds)
    seeds: list[int] = []
    for part in raw_seeds.split(","):
        if "-" in part:
            a, b = part.split("-", 1)
            seeds.extend(range(int(a), int(b) + 1))
        else:
            seeds.append(int(part))
    seeds_tuple = tuple(seeds)

    manifest = load_study_manifest(args.study_root)
    scenario_keys = {item.strip() for item in str(args.scenarios).split(",") if item.strip()}
    payloads: list[dict[str, Any]] = []
    for scenario in SCENARIOS:
        if str(scenario["key"]) not in scenario_keys:
            continue
        payloads.append(
            _build_payload(
                manifest,
                scenario,
                args.protocol_id,
                rand_prob=float(args.rand_prob),
                include_self_loops=bool(args.include_self_loops),
                action_conditioned=bool(args.action_conditioned),
                levels=levels,
                seeds=seeds_tuple,
            )
        )

    if not any(p["panels"] for p in payloads):
        print("No panels produced. Check that evaluate_study has been run.", file=sys.stderr)
        sys.exit(1)

    panels_with_data = [
        panel
        for payload in payloads
        for panel in payload["panels"].values()
        if panel.get("R")
    ]
    global_max = max(
        max(
            float(np.max(np.abs(np.asarray(panel["R"], dtype=np.float64)))),
            float(np.max(np.asarray(panel.get("mean_abs_R", panel["R"]), dtype=np.float64))),
        )
        for panel in panels_with_data
    ) if panels_with_data else VIS_THRESHOLD
    global_max = max(global_max, float(VIS_THRESHOLD), 1e-9)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "output_dir": str(args.output_dir),
        "vis_threshold": VIS_THRESHOLD,
        "global_residual_scale": global_max,
        "protocol_id": args.protocol_id,
        "rand_prob": float(args.rand_prob),
        "self_loops_included": bool(args.include_self_loops),
        "action_conditioned": bool(args.action_conditioned),
        "figures": {},
    }
    for payload in payloads:
        if not payload["panels"]:
            continue
        payload["visual_encoding"]["global_residual_scale"] = global_max
        output_path = args.output_dir / payload["output_name"]
        json_path = output_path.with_suffix(".json")
        payload["figure"] = str(output_path)
        payload["json"] = str(json_path)
        json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        save_selected_figure(
            payload=payload,
            output_path=output_path,
            vis_threshold=VIS_THRESHOLD,
            max_residual_override=global_max,
            font_scale=float(payload.get("font_scale", 1.60)),
            header_font_scale=float(
                payload.get("header_font_scale", payload.get("font_scale", 1.60))
            ),
            omit_leading_zero=bool(payload.get("omit_leading_zero", False)),
        )
        figure_outputs = {"png": str(output_path), "json": str(json_path)}
        if bool(args.compact_variants) and "all_methods" in output_path.stem:
            compact_path = output_path.with_name(f"{output_path.stem}_compact{output_path.suffix}")
            supplementary_path = output_path.with_name(
                f"{output_path.stem}_compact_supplementary{output_path.suffix}"
            )
            save_selected_figure_compact(
                payload=payload,
                output_path=compact_path,
                vis_threshold=VIS_THRESHOLD,
                max_residual_override=global_max,
                supplementary=False,
                top_k=None,
            )
            save_selected_figure_compact(
                payload=payload,
                output_path=supplementary_path,
                vis_threshold=VIS_THRESHOLD,
                max_residual_override=global_max,
                supplementary=True,
                top_k=None,
            )
            figure_outputs.update(
                {
                    "compact_png": str(compact_path),
                    "compact_pdf": str(compact_path.with_suffix(".pdf")),
                    "compact_supplementary_png": str(supplementary_path),
                    "compact_supplementary_pdf": str(supplementary_path.with_suffix(".pdf")),
                }
            )
            print(compact_path)
            print(supplementary_path)
        summary["figures"][payload["scenario"]] = figure_outputs
        print(output_path)

    summary_path = _summary_path(
        args.output_dir,
        include_self_loops=bool(args.include_self_loops),
        action_conditioned=bool(args.action_conditioned),
        rand_prob=float(args.rand_prob),
    )
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
