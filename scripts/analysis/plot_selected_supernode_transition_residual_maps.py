from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import ListedColormap, Normalize, TwoSlopeNorm
from matplotlib.patches import Circle, FancyArrowPatch
from matplotlib.lines import Line2D

from mercury.graph.core import Graph
from maze_environment.LEVEL import levels as MAZE_LEVELS
from maze_environment.agent import Agent
from mercury_runs.analysis.transition_residual_maps import (
    TransitionResidualDiagnostics,
    _draw_residual_map,
    _ground_truth_action_structure,
    _ground_truth_structure,
    _paper_precision_payload,
    _residual_colormap,
    _residual_width,
    _linewidth_reference_values,
    compute_action_conditioned_transition_residual_diagnostics,
    compute_transition_residual_diagnostics,
)
from mercury_runs.infrastructure.paper_precision import (
    collapse_action_transition_tensor_to_adjacency,
    compute_alignment,
    exact_cartesian_reference_positions,
    row_normalize,
    zero_diagonal,
)
from mercury_runs.algorithms.cscg.prepare import action_to_indices, filter_valid_cscg_rows, load_dataset_arrays, to_indices
from scripts.analysis.plot_selected_transition_residual_maps import (
    BATCH_SIZE,
    CSCG_PRESETS,
    LEVELS,
    OUTPUT_DIR,
    RAND_PROB,
    SCENARIOS,
    SEEDS,
    VIS_THRESHOLD,
    _artifact,
    _cached_ground_truth_action_structure,
    _cached_ground_truth_structure,
    _cscg_run,
    _cscg_config,
    _diagnostic_for_run,
    _ground_truth_cache_key,
    _history_row,
    _hybrid_run,
    _learned_for_graph,
    _learned_for_model,
    _load_manifest,
    _rand_prob_suffix,
    _recomputed_eval_cooccurrence,
    _viterbi_processed_learned_and_cooccurrence,
)
from scripts.analysis.rerender_selected_transition_residual_maps import (
    FINAL_TITLE,
    LEVEL_LABELS,
    LEVEL_ORDER,
    METHOD_REPLACEMENTS,
    _draw_summary_bar_key,
    SUMMARY_NEGATIVE_COLOR,
    SUMMARY_POSITIVE_COLOR,
    SUMMARY_SIMILARITY_COLOR,
    SUMMARY_TEXT_COLOR,
    VITERBI_SUFFIX,
)


SUPER_OUTPUT_DIR = OUTPUT_DIR
NODE_CMAP = plt.get_cmap("viridis")
OBS_CMAP = ListedColormap(
    [
        "#b2182b",
        "#ef8a62",
        "#f4a582",
        "#ca0020",
        "#fdae61",
        "#c51b7d",
        "#de77ae",
        "#8c510a",
        "#bf812d",
        "#a6611a",
        "#d6604d",
        "#f1b6da",
    ],
    name="warm_observation_categories",
)
LEARNED_EDGE_THRESHOLD = 0.01
SUPER_EDGE_THRESHOLD = 0.02
MAX_LEARNED_EDGES_PER_PAIR = 2
MAX_LEARNED_EDGES_PER_PANEL = 80


@dataclass(frozen=True)
class NativePanel:
    level: int
    method: str
    seed: int
    run_root: Path
    diagnostic: TransitionResidualDiagnostics
    learned_transition: np.ndarray
    alignment: np.ndarray
    occupancy: np.ndarray
    observation_ids: np.ndarray
    observation_values: list[Any]
    ground_truth_observation_ids: np.ndarray
    source_denominators: np.ndarray
    learned_edges: list[dict[str, Any]]
    missing_super_edges: list[dict[str, Any]]

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "level": int(self.level),
            "seed": int(self.seed),
            "run_root": str(self.run_root),
            "tv_sim": float(self.diagnostic.tv_sim),
            "tv_by_state": self.diagnostic.tv_by_state.tolist(),
            "similarity_by_state": self.diagnostic.similarity_by_state.tolist(),
            "uncovered_rows": self.diagnostic.uncovered_rows.astype(bool).tolist(),
            "alignment": self.alignment.astype(int).tolist(),
            "occupancy": self.occupancy.astype(float).tolist(),
            "observation_ids": self.observation_ids.astype(int).tolist(),
            "observation_values": self.observation_values,
            "ground_truth_observation_ids": self.ground_truth_observation_ids.astype(int).tolist(),
            "observation_alignment_mismatches": _observation_alignment_mismatches(
                alignment=self.alignment,
                occupancy=self.occupancy,
                learned_observation_ids=self.observation_ids,
                gt_observation_ids=self.ground_truth_observation_ids,
            ),
            "source_denominators": self.source_denominators.astype(float).tolist(),
            "P_gt": self.diagnostic.transition_true.tolist(),
            "P_hat": self.diagnostic.transition_predicted.tolist(),
            "R": self.diagnostic.residual.tolist(),
            "learned_edges": self.learned_edges,
            "missing_super_edges": self.missing_super_edges,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate super-node learned-edge transition residual maps.")
    parser.add_argument("--viterbi-processed", action="store_true", help="Use Viterbi-processed CSCG snapshots.")
    parser.add_argument("--include-self-loops", action="store_true", help="Retain self-loop transitions in diagnostics.")
    parser.add_argument("--action-conditioned", action="store_true", help="Use action-conditioned residuals for edge colour.")
    parser.add_argument("--rand-prob", type=float, default=RAND_PROB, help="Study rand_prob value to select.")
    parser.add_argument("--vis-threshold", type=float, default=VIS_THRESHOLD, help="Residual threshold for display.")
    parser.add_argument(
        "--learned-edge-threshold",
        type=float,
        default=LEARNED_EDGE_THRESHOLD,
        help="Minimum native learned-edge contribution to draw.",
    )
    args = parser.parse_args()

    matplotlib.use("Agg", force=True)
    manifest = _load_manifest()
    payloads = [
        _build_supernode_payload(
            manifest,
            scenario,
            rand_prob=float(args.rand_prob),
            viterbi_processed=bool(args.viterbi_processed),
            include_self_loops=bool(args.include_self_loops),
            action_conditioned=bool(args.action_conditioned),
            learned_edge_threshold=float(args.learned_edge_threshold),
            super_edge_threshold=max(float(args.vis_threshold), 0.0),
        )
        for scenario in SCENARIOS
    ]
    global_residual = max(
        [
            float(np.max(np.abs(np.asarray(panel["R"], dtype=np.float64))))
            for payload in payloads
            for panel in payload["panels"].values()
        ]
        or [1e-9]
    )
    global_contribution = max(
        [
            float(edge["contribution"])
            for payload in payloads
            for panel in payload["panels"].values()
            for edge in panel["learned_edges"]
        ]
        or [1e-9]
    )
    global_residual = max(global_residual, float(args.vis_threshold), 1e-9)
    global_contribution = max(global_contribution, float(args.learned_edge_threshold), 1e-9)

    SUPER_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = {
        "output_dir": str(SUPER_OUTPUT_DIR),
        "rand_prob": float(args.rand_prob),
        "viterbi_processed": bool(args.viterbi_processed),
        "self_loops_included": bool(args.include_self_loops),
        "action_conditioned": bool(args.action_conditioned),
        "representative_seed_rule": "seed with per-seed TV_sim closest to the 30-seed mean for each level/method/scenario panel",
        "global_residual_scale": float(global_residual),
        "global_contribution_scale": float(global_contribution),
        "figures": {},
    }
    for payload in payloads:
        payload["visual_encoding"]["global_residual_scale"] = float(global_residual)
        payload["visual_encoding"]["global_contribution_scale"] = float(global_contribution)
        output_path = SUPER_OUTPUT_DIR / payload["output_name"]
        json_path = output_path.with_suffix(".json")
        payload["figure"] = str(output_path)
        payload["json"] = str(json_path)
        json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        _save_supernode_figure(
            payload=payload,
            output_path=output_path,
            max_residual=global_residual,
            max_contribution=global_contribution,
        )
        level_outputs = {}
        for level in LEVELS:
            level_payload = _payload_for_level(payload, int(level))
            level_output_path = output_path.with_name(output_path.stem + f"_level_{int(level)}" + output_path.suffix)
            level_json_path = level_output_path.with_suffix(".json")
            level_payload["figure"] = str(level_output_path)
            level_payload["json"] = str(level_json_path)
            level_json_path.write_text(json.dumps(level_payload, indent=2), encoding="utf-8")
            _save_supernode_figure(
                payload=level_payload,
                output_path=level_output_path,
                max_residual=global_residual,
                max_contribution=global_contribution,
            )
            level_outputs[f"L{int(level)}"] = {"png": str(level_output_path), "json": str(level_json_path)}
            print(level_output_path)
        summary["figures"][payload["scenario"]] = {
            "png": str(output_path),
            "json": str(json_path),
            "level_figures": level_outputs,
        }
        print(output_path)
    summary_path = SUPER_OUTPUT_DIR / _summary_name(
        rand_prob=float(args.rand_prob),
        viterbi_processed=bool(args.viterbi_processed),
        include_self_loops=bool(args.include_self_loops),
        action_conditioned=bool(args.action_conditioned),
    )
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def _build_supernode_payload(
    manifest: dict[tuple[Any, ...], Path],
    scenario: dict[str, Any],
    *,
    rand_prob: float,
    viterbi_processed: bool,
    include_self_loops: bool,
    action_conditioned: bool,
    learned_edge_threshold: float,
    super_edge_threshold: float,
) -> dict[str, Any]:
    panels: dict[str, Any] = {}
    for level in LEVELS:
        target = _representative_panel(
            manifest,
            scenario=scenario,
            level=int(level),
            method_label=str(scenario["target_label"]),
            kind=str(scenario["target_kind"]),
            preset=None,
            rand_prob=rand_prob,
            viterbi_processed=viterbi_processed,
            include_self_loops=include_self_loops,
            action_conditioned=action_conditioned,
            learned_edge_threshold=learned_edge_threshold,
            super_edge_threshold=super_edge_threshold,
        )
        panels[f"L{level}/{scenario['target_label']}"] = target.to_json_dict()
        for label, preset in CSCG_PRESETS:
            panel = _representative_panel(
                manifest,
                scenario=scenario,
                level=int(level),
                method_label=label,
                kind=str(scenario["cscg_kind"]),
                preset=preset,
                rand_prob=rand_prob,
                viterbi_processed=viterbi_processed,
                include_self_loops=include_self_loops,
                action_conditioned=action_conditioned,
                learned_edge_threshold=learned_edge_threshold,
                super_edge_threshold=super_edge_threshold,
            )
            panels[f"L{level}/{label}"] = panel.to_json_dict()

    output_name = str(scenario["output_name"]).replace("transition_residual_", "supernode_transition_residual_")
    output_name = output_name.replace("_batch_40_", f"{_rand_prob_suffix(rand_prob)}_batch_40_")
    title = str(scenario["title"])
    if viterbi_processed:
        output_name = output_name.replace("supernode_transition_residual_", "supernode_transition_residual_viterbi100_smooth1em09_")
        title = f"{title} {VITERBI_SUFFIX}"
    if include_self_loops:
        output_name = output_name.replace("supernode_transition_residual_", "supernode_transition_residual_self_loops_")
        title = f"{title} (self-loops included)"
    if action_conditioned:
        output_name = output_name.replace("supernode_transition_residual_", "supernode_transition_residual_action_conditioned_")
        title = title.replace("transition residuals", "action-conditioned transition residuals")
    return {
        "scenario": _display_title(title),
        "output_name": output_name,
        "panels": panels,
        "visual_encoding": {
            "super_nodes": "ground-truth states at the fixed ground-truth layout",
            "super_node_colour": "seed-averaged state-wise transition similarity S_i from the matching transition residual map",
            "learned_nodes": "representative-seed native learned states arranged radially inside the aligned ground-truth super-node",
            "learned_edges": "not drawn; residual edges are aggregated ground-truth-coordinate residuals from the matching transition residual map",
            "learned_node_colour": "native learned-node fill colour indicates source observation id",
            "missing_super_edges": "not drawn separately; missing mass is encoded by negative aggregated residual edges",
            "residual_colormap": "RdBu_r",
            "learned_edge_threshold": float(learned_edge_threshold),
            "super_edge_threshold": float(super_edge_threshold),
            "rand_prob": float(rand_prob),
            "metric_layer": "seed-averaged transition residual JSON",
            "subnode_layer": "representative seed with per-seed TV_sim closest to the 30-seed mean",
        },
    }


def _payload_for_level(payload: dict[str, Any], level: int) -> dict[str, Any]:
    prefix = f"L{int(level)}/"
    filtered = {
        key: value
        for key, value in dict(payload["panels"]).items()
        if str(key).startswith(prefix)
    }
    level_label = LEVEL_LABELS.get(int(level), f"L{int(level)}")
    return {
        **payload,
        "scenario": f"{payload['scenario']} — {level_label}",
        "panels": filtered,
        "ground_truth_observation_ids": {
            f"L{int(level)}": dict(payload.get("ground_truth_observation_ids", {})).get(f"L{int(level)}", [])
        },
    }


def _representative_panel(
    manifest: dict[tuple[Any, ...], Path],
    *,
    scenario: dict[str, Any],
    level: int,
    method_label: str,
    kind: str,
    preset: str | None,
    rand_prob: float,
    viterbi_processed: bool,
    include_self_loops: bool,
    action_conditioned: bool,
    learned_edge_threshold: float,
    super_edge_threshold: float,
) -> NativePanel:
    candidates: list[tuple[int, Path, TransitionResidualDiagnostics]] = []
    for seed in SEEDS:
        run_root = (
            _hybrid_run(manifest, level, int(seed), rand_prob=rand_prob)
            if preset is None
            else _cscg_run(manifest, level, int(seed), str(preset), rand_prob=rand_prob)
        )
        diagnostic = _diagnostic_for_run(
            run_root,
            kind,
            method_label,
            viterbi_processed=viterbi_processed,
            include_self_loops=include_self_loops,
            action_conditioned=action_conditioned,
        )
        candidates.append((int(seed), run_root, diagnostic))
    mean_tv_sim = float(np.mean([diagnostic.tv_sim for _, _, diagnostic in candidates]))
    seed, run_root, diagnostic = min(candidates, key=lambda item: (abs(float(item[2].tv_sim) - mean_tv_sim), item[0]))
    learned, cooccurrence, run_config, observation_ids, observation_values = _native_learned_for_run(
        run_root=run_root,
        kind=kind,
        viterbi_processed=viterbi_processed,
        action_conditioned=action_conditioned,
    )
    gt_observation_ids = _ground_truth_observation_ids_for_values(
        level,
        observation_values=observation_values,
    )
    learned_matrix = _learned_matrix_for_edges(learned, action_conditioned=action_conditioned)
    settings = _settings_for_run(run_config, learned, action_conditioned=action_conditioned)
    ignore_self_loops = bool(settings["ignore_self_loops"]) and not bool(include_self_loops)
    if ignore_self_loops:
        learned_matrix = zero_diagonal(learned_matrix)
    learned_transition = row_normalize(learned_matrix)
    cooccurrence_matrix = _pad_cooccurrence(
        np.asarray(cooccurrence, dtype=np.int64),
        n_learned=int(learned_transition.shape[0]),
        n_true=int(diagnostic.residual.shape[0]),
    )
    alignment = compute_alignment(cooccurrence_matrix)
    occupancy = cooccurrence_matrix.sum(axis=1).astype(np.float64)
    alignment = np.asarray(alignment, dtype=np.int64)
    alignment[occupancy <= 0.0] = -1
    denominators = _source_denominators(alignment, occupancy, n_true=diagnostic.residual.shape[0])
    learned_edges = _learned_edge_payloads(
        learned_transition=learned_transition,
        diagnostic=diagnostic,
        alignment=alignment,
        occupancy=occupancy,
        denominators=denominators,
        threshold=learned_edge_threshold,
        max_edges_per_pair=MAX_LEARNED_EDGES_PER_PAIR,
        max_edges_per_panel=MAX_LEARNED_EDGES_PER_PANEL,
    )
    missing_super_edges = _missing_super_edge_payloads(
        learned_edges=learned_edges,
        diagnostic=diagnostic,
        threshold=super_edge_threshold,
    )
    return NativePanel(
        level=level,
        method=method_label,
        seed=seed,
        run_root=run_root,
        diagnostic=diagnostic,
        learned_transition=learned_transition,
        alignment=alignment,
        occupancy=occupancy,
        observation_ids=_pad_observation_ids(observation_ids, n_learned=int(learned_transition.shape[0])),
        observation_values=observation_values,
        ground_truth_observation_ids=gt_observation_ids,
        source_denominators=denominators,
        learned_edges=learned_edges,
        missing_super_edges=missing_super_edges,
    )


def _native_learned_for_run(
    *,
    run_root: Path,
    kind: str,
    viterbi_processed: bool,
    action_conditioned: bool,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any], np.ndarray, list[Any]]:
    run_status = json.loads((run_root / "run_status.json").read_text(encoding="utf-8"))
    run_config = dict(run_status.get("config") or {})
    if kind == "mercury_checkpoint_2000":
        paper_path = _artifact(run_status, "mercury_cscg_hybrid_paper_precision_json")
        row = _history_row(_paper_precision_payload(paper_path), phase="mercury", step=2000)
        graph_path = run_root / "bundles" / "snapshots" / "mercury" / "paper_precision" / "step=2000" / "latent_graph.npz"
        graph = Graph.from_npz(str(graph_path))
        return (
            _learned_for_graph(run_config, graph_path, action_conditioned=action_conditioned),
            np.asarray(row["latent_exact_state_cooccurrence_matrix"], dtype=np.int64),
            run_config,
            _mercury_observation_ids(graph, memory_length=int(run_config.get("memory_length", 10))),
            _observation_values_from_snapshot_dir(graph_path.parent)
            or _mercury_observation_values(graph, memory_length=int(run_config.get("memory_length", 10))),
        )
    if viterbi_processed and kind in {"hybrid_cscg_checkpoint_4000", "hybrid_final", "cscg_checkpoint_2000", "cscg_checkpoint_4000", "cscg_final"}:
        method = "mercury_cscg_hybrid" if kind.startswith("hybrid") else "cscg"
        selected_step = 2000 if kind in {"hybrid_cscg_checkpoint_4000", "cscg_checkpoint_2000"} else 4000 if kind == "cscg_checkpoint_4000" else None
        learned, cooccurrence = _viterbi_processed_learned_and_cooccurrence(
            run_root=run_root,
            run_status=run_status,
            method=method,
            selected_step=selected_step,
            action_conditioned=action_conditioned,
        )
        observation_ids, observation_values = _observation_ids_for_viterbi_snapshot(run_root, method, selected_step)
        return learned, cooccurrence, run_config, observation_ids, observation_values
    if kind == "hybrid_cscg_checkpoint_4000":
        paper_path = _artifact(run_status, "mercury_cscg_hybrid_paper_precision_json")
        row = _history_row(_paper_precision_payload(paper_path), phase="cscg", cumulative_step=4000)
        model_path = run_root / "snapshots" / "mercury_cscg_hybrid" / "paper_precision" / "step=2000" / "model.npz"
        obs_values = _observation_values_from_model(model_path, run_config=run_config)
        return (
            _learned_for_model(model_path, action_conditioned=action_conditioned),
            _recomputed_eval_cooccurrence(
                run_root=run_root,
                run_config=run_config,
                method="mercury_cscg_hybrid",
                model_path=model_path,
            ),
            run_config,
            _observation_ids_from_model(model_path),
            obs_values,
        )
    if kind == "hybrid_final":
        paper_path = _artifact(run_status, "mercury_cscg_hybrid_paper_precision_json")
        row = _history_row(_paper_precision_payload(paper_path), phase="cscg")
        model_path = _artifact(run_status, "mercury_cscg_hybrid_model_npz")
        obs_values = _observation_values_from_model(model_path, run_config=run_config)
        return (
            _learned_for_model(model_path, action_conditioned=action_conditioned),
            _recomputed_eval_cooccurrence(
                run_root=run_root,
                run_config=run_config,
                method="mercury_cscg_hybrid",
                model_path=model_path,
            ),
            run_config,
            _observation_ids_from_model(model_path),
            obs_values,
        )
    if kind == "cscg_checkpoint_2000":
        paper_path = _artifact(run_status, "cscg_paper_precision_json")
        row = _history_row(_paper_precision_payload(paper_path), step=2000)
        model_path = run_root / "snapshots" / "cscg" / "paper_precision" / "step=2000" / "model.npz"
        obs_values = _observation_values_from_model(model_path, run_config=run_config)
        return (
            _learned_for_model(model_path, action_conditioned=action_conditioned),
            _recomputed_eval_cooccurrence(
                run_root=run_root,
                run_config=run_config,
                method="cscg",
                model_path=model_path,
            ),
            run_config,
            _observation_ids_from_model(model_path),
            obs_values,
        )
    if kind == "cscg_checkpoint_4000":
        paper_path = _artifact(run_status, "cscg_paper_precision_json")
        row = _history_row(_paper_precision_payload(paper_path), step=4000)
        model_path = run_root / "snapshots" / "cscg" / "paper_precision" / "step=4000" / "model.npz"
        obs_values = _observation_values_from_model(model_path, run_config=run_config)
        return (
            _learned_for_model(model_path, action_conditioned=action_conditioned),
            _recomputed_eval_cooccurrence(
                run_root=run_root,
                run_config=run_config,
                method="cscg",
                model_path=model_path,
            ),
            run_config,
            _observation_ids_from_model(model_path),
            obs_values,
        )
    if kind == "cscg_final":
        paper_path = _artifact(run_status, "cscg_paper_precision_json")
        row = _history_row(_paper_precision_payload(paper_path))
        model_path = _artifact(run_status, "cscg_model_npz")
        obs_values = _observation_values_from_model(model_path, run_config=run_config)
        return (
            _learned_for_model(model_path, action_conditioned=action_conditioned),
            _recomputed_eval_cooccurrence(
                run_root=run_root,
                run_config=run_config,
                method="cscg",
                model_path=model_path,
            ),
            run_config,
            _observation_ids_from_model(model_path),
            obs_values,
        )
    raise ValueError(f"Unknown kind: {kind}")


def _observation_ids_from_model(path: Path) -> np.ndarray:
    model = np.load(path, allow_pickle=False)
    return _observation_ids_from_n_clones(np.asarray(model["n_clones"], dtype=np.int64))


def _observation_values_from_model(path: Path, *, run_config: dict[str, Any] | None = None) -> list[Any]:
    model = np.load(path, allow_pickle=False)
    if "obs_uniques" in model.files:
        return _jsonable_observation_values(np.asarray(model["obs_uniques"]))
    vocab_path = path.parent / "observation_vocabulary.npz"
    if vocab_path.exists():
        vocab = np.load(vocab_path, allow_pickle=False)
        if "obs_uniques" in vocab.files:
            return _jsonable_observation_values(np.asarray(vocab["obs_uniques"]))
    if run_config is not None:
        return _training_observation_values(run_config)
    n_obs = int(np.asarray(model["n_clones"], dtype=np.int64).shape[0])
    return [int(idx) for idx in range(n_obs)]


def _observation_ids_for_viterbi_snapshot(run_root: Path, method: str, selected_step: int | None) -> tuple[np.ndarray, list[Any]]:
    snapshot_method = "mercury_cscg_hybrid" if method == "mercury_cscg_hybrid" else "cscg"
    snapshot_root = run_root / "snapshots" / snapshot_method / "paper_precision"
    if selected_step is None:
        snapshots = sorted(
            [path for path in snapshot_root.iterdir() if path.is_dir() and path.name.startswith("step=")],
            key=lambda path: int(path.name.split("=", 1)[1]),
        )
        snapshot_dir = snapshots[-1]
    else:
        snapshot_dir = snapshot_root / f"step={int(selected_step)}"
    model = np.load(snapshot_dir / "model.npz", allow_pickle=False)
    return _observation_ids_from_n_clones(np.asarray(model["n_clones"], dtype=np.int64)), _observation_values_from_model(snapshot_dir / "model.npz")


def _observation_values_from_snapshot_dir(snapshot_dir: Path) -> list[Any] | None:
    vocab_path = Path(snapshot_dir) / "observation_vocabulary.npz"
    if not vocab_path.exists():
        return None
    vocab = np.load(vocab_path, allow_pickle=False)
    if "obs_uniques" not in vocab.files:
        return None
    return _jsonable_observation_values(np.asarray(vocab["obs_uniques"]))


def _training_observation_values(run_config: dict[str, Any]) -> list[Any]:
    observations, actions, collisions, _, _ = load_dataset_arrays(
        datasets_root=PROJECT_ROOT / str(run_config.get("datasets_root", "datasets")),
        level=int(run_config["level"]),
        sensor=str(run_config.get("sensor", "cartesian")),
        sensor_range=run_config.get("sensor_range"),
        dataset_select=str(run_config.get("dataset_select", "latest")),
        dataset_run_id=run_config.get("dataset_run_id"),
    )
    observations, _, _, _ = filter_valid_cscg_rows(
        observations=observations,
        actions=actions,
        collisions=collisions,
        valid_trajectories_only=bool(run_config.get("cscg_valid_trajectories_only", run_config.get("valid_trajectories_only", False))),
    )
    _, obs_values = to_indices(np.asarray(observations))
    return _jsonable_observation_values(np.asarray(obs_values))


def _action_values_from_model(path: Path, *, run_config: dict[str, Any]) -> np.ndarray:
    model = np.load(path, allow_pickle=False)
    if "action_uniques" in model.files:
        return np.asarray(model["action_uniques"])
    vocab_path = path.parent / "observation_vocabulary.npz"
    if vocab_path.exists():
        vocab = np.load(vocab_path, allow_pickle=False)
        if "action_uniques" in vocab.files:
            return np.asarray(vocab["action_uniques"])
    observations, actions, collisions, _, _ = load_dataset_arrays(
        datasets_root=PROJECT_ROOT / str(run_config.get("datasets_root", "datasets")),
        level=int(run_config["level"]),
        sensor=str(run_config.get("sensor", "cartesian")),
        sensor_range=run_config.get("sensor_range"),
        dataset_select=str(run_config.get("dataset_select", "latest")),
        dataset_run_id=run_config.get("dataset_run_id"),
    )
    _, actions, _, _ = filter_valid_cscg_rows(
        observations=observations,
        actions=actions,
        collisions=collisions,
        valid_trajectories_only=bool(run_config.get("cscg_valid_trajectories_only", run_config.get("valid_trajectories_only", False))),
    )
    _, action_values = action_to_indices(np.asarray(actions))
    return np.asarray(action_values)


def _observation_ids_from_n_clones(n_clones: np.ndarray) -> np.ndarray:
    values: list[int] = []
    for obs_id, count in enumerate(np.asarray(n_clones, dtype=np.int64).tolist()):
        values.extend([int(obs_id)] * int(count))
    return np.asarray(values, dtype=np.int64)


def _mercury_observation_ids(graph: Graph, *, memory_length: int) -> np.ndarray:
    node_features = getattr(graph, "node_features", {})
    if "mem_adj" not in node_features:
        return np.zeros(int(graph.n), dtype=np.int64)
    mem_adj = np.asarray(node_features["mem_adj"], dtype=np.float64)[: int(graph.n)]
    memory_length = max(int(memory_length), 1)
    if mem_adj.ndim != 2 or mem_adj.shape[1] % memory_length != 0:
        return np.zeros(int(graph.n), dtype=np.int64)
    n_sensory = int(mem_adj.shape[1] // memory_length)
    current = mem_adj.reshape(int(graph.n), n_sensory, memory_length)[:, :, 0]
    observation_ids = np.argmax(current, axis=1).astype(np.int64, copy=False)
    observation_ids[current.sum(axis=1) <= 0.0] = 0
    return observation_ids


def _mercury_observation_values(graph: Graph, *, memory_length: int) -> list[Any]:
    mem_adj = np.asarray(getattr(graph, "node_features", {}).get("mem_adj", np.zeros((int(graph.n), 0))), dtype=np.float64)
    memory_length = max(int(memory_length), 1)
    if mem_adj.ndim != 2 or mem_adj.shape[1] == 0 or mem_adj.shape[1] % memory_length != 0:
        return [int(idx) for idx in range(max(int(graph.n), 1))]
    n_sensory = int(mem_adj.shape[1] // memory_length)
    return [int(idx) for idx in range(n_sensory)]


def _pad_observation_ids(observation_ids: np.ndarray, *, n_learned: int) -> np.ndarray:
    values = np.asarray(observation_ids, dtype=np.int64).reshape(-1)
    padded = np.zeros(int(n_learned), dtype=np.int64)
    count = min(int(n_learned), int(values.shape[0]))
    padded[:count] = values[:count]
    return padded


def _jsonable_observation_values(values: np.ndarray) -> list[Any]:
    arr = np.asarray(values)
    if arr.ndim == 1:
        return [int(value) if np.issubdtype(arr.dtype, np.integer) else float(value) for value in arr.tolist()]
    return [
        [int(item) if isinstance(item, (int, np.integer)) else float(item) for item in row]
        for row in arr.tolist()
    ]


def _observation_key(value: Any) -> tuple[Any, ...]:
    if isinstance(value, list):
        return tuple(_observation_key(item)[0] if len(_observation_key(item)) == 1 else _observation_key(item) for item in value)
    if isinstance(value, tuple):
        return tuple(value)
    if isinstance(value, np.ndarray):
        return tuple(np.asarray(value).reshape(-1).tolist())
    return (value,)


def _observation_alignment_mismatches(
    *,
    alignment: np.ndarray,
    occupancy: np.ndarray,
    learned_observation_ids: np.ndarray,
    gt_observation_ids: np.ndarray,
) -> dict[str, Any]:
    mismatches = []
    for learned_idx, true_state in enumerate(np.asarray(alignment, dtype=np.int64).tolist()):
        if learned_idx >= int(occupancy.shape[0]) or float(occupancy[learned_idx]) <= 0.0:
            continue
        if true_state < 0:
            continue
        if learned_idx >= int(learned_observation_ids.shape[0]) or true_state >= int(gt_observation_ids.shape[0]):
            continue
        learned_obs = int(learned_observation_ids[learned_idx])
        gt_obs = int(gt_observation_ids[true_state])
        if learned_obs != gt_obs:
            mismatches.append(
                {
                    "learned_state": int(learned_idx),
                    "aligned_gt_state": int(true_state),
                    "learned_observation_id": learned_obs,
                    "gt_observation_id": gt_obs,
                }
            )
    return {"count": len(mismatches), "examples": mismatches[:20]}


def _settings_for_run(run_config: dict[str, Any], learned: np.ndarray, *, action_conditioned: bool) -> dict[str, Any]:
    if action_conditioned:
        _, settings = _cached_ground_truth_action_structure(
            _ground_truth_cache_key(run_config) + ("action", int(np.asarray(learned).shape[0])),
            json.dumps(run_config, sort_keys=True),
            int(np.asarray(learned).shape[0]),
        )
        return settings
    _, settings = _cached_ground_truth_structure(_ground_truth_cache_key(run_config), json.dumps(run_config, sort_keys=True))
    return settings


def _learned_matrix_for_edges(learned: np.ndarray, *, action_conditioned: bool) -> np.ndarray:
    weights = np.asarray(learned, dtype=np.float64)
    if action_conditioned:
        return collapse_action_transition_tensor_to_adjacency(weights)
    return weights


def _pad_cooccurrence(cooccurrence: np.ndarray, *, n_learned: int, n_true: int) -> np.ndarray:
    matrix = np.asarray(cooccurrence, dtype=np.int64)
    if matrix.ndim != 2:
        raise ValueError(f"Expected cooccurrence matrix, got {matrix.shape!r}")
    padded = np.zeros((int(n_learned), int(n_true)), dtype=np.int64)
    rows = min(int(n_learned), int(matrix.shape[0]))
    cols = min(int(n_true), int(matrix.shape[1]))
    padded[:rows, :cols] = matrix[:rows, :cols]
    return padded


def _source_denominators(alignment: np.ndarray, occupancy: np.ndarray, *, n_true: int) -> np.ndarray:
    denominators = np.zeros(int(n_true), dtype=np.float64)
    for true_state in range(int(n_true)):
        denominators[true_state] = float(occupancy[np.asarray(alignment) == true_state].sum())
    return denominators


def _learned_edge_payloads(
    *,
    learned_transition: np.ndarray,
    diagnostic: TransitionResidualDiagnostics,
    alignment: np.ndarray,
    occupancy: np.ndarray,
    denominators: np.ndarray,
    threshold: float,
    max_edges_per_pair: int,
    max_edges_per_panel: int,
) -> list[dict[str, Any]]:
    edges: list[dict[str, Any]] = []
    transition = np.asarray(learned_transition, dtype=np.float64)
    residual = np.asarray(diagnostic.residual, dtype=np.float64)
    n_true = int(residual.shape[0])
    candidates: list[dict[str, Any]] = []
    for source, target in np.argwhere(transition > 0.0).tolist():
        true_source = int(alignment[int(source)])
        true_target = int(alignment[int(target)])
        if true_source < 0 or true_source >= n_true or true_target < 0 or true_target >= n_true:
            continue
        denom = float(denominators[true_source])
        if denom <= 0.0:
            continue
        contribution = float(occupancy[int(source)] / denom * transition[int(source), int(target)])
        signed_residual = float(residual[true_source, true_target])
        if contribution < float(threshold) and abs(signed_residual) < float(threshold):
            continue
        if abs(signed_residual) <= 1e-12:
            continue
        candidates.append(
            {
                "source": int(source),
                "target": int(target),
                "aligned_source": true_source,
                "aligned_target": true_target,
                "learned_probability": float(transition[int(source), int(target)]),
                "source_occupancy": float(occupancy[int(source)]),
                "contribution": contribution,
                "residual": signed_residual,
            }
        )
    by_pair: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for edge in candidates:
        key = (int(edge["aligned_source"]), int(edge["aligned_target"]))
        by_pair.setdefault(key, []).append(edge)
    for pair_edges in by_pair.values():
        pair_edges.sort(
            key=lambda edge: (float(edge["contribution"]) * abs(float(edge["residual"])), float(edge["contribution"])),
            reverse=True,
        )
        edges.extend(pair_edges[: max(int(max_edges_per_pair), 1)])
    edges.sort(
        key=lambda edge: (float(edge["contribution"]) * abs(float(edge["residual"])), float(edge["contribution"])),
        reverse=True,
    )
    return edges[: max(int(max_edges_per_panel), 1)]


def _missing_super_edge_payloads(
    *,
    learned_edges: list[dict[str, Any]],
    diagnostic: TransitionResidualDiagnostics,
    threshold: float,
) -> list[dict[str, Any]]:
    supported_pairs = {(int(edge["aligned_source"]), int(edge["aligned_target"])) for edge in learned_edges}
    missing: list[dict[str, Any]] = []
    residual = np.asarray(diagnostic.residual, dtype=np.float64)
    magnitude = (
        np.asarray(diagnostic.residual_magnitude, dtype=np.float64)
        if diagnostic.residual_magnitude is not None
        else np.abs(residual)
    )
    for source, target in np.argwhere(residual < -float(threshold)).tolist():
        pair = (int(source), int(target))
        if pair in supported_pairs:
            continue
        missing.append(
            {
                "source": int(source),
                "target": int(target),
                "residual": float(residual[int(source), int(target)]),
                "magnitude": float(magnitude[int(source), int(target)]),
            }
        )
    return missing


def _save_supernode_figure(
    *,
    payload: dict[str, Any],
    output_path: Path,
    max_residual: float,
    max_contribution: float,
) -> None:
    metric_payload = _matching_metric_payload(payload, output_path=output_path)
    panel_payloads = _metric_overlay_panels(payload, metric_payload)
    if metric_payload is not None:
        metric_scale = ((metric_payload.get("visual_encoding") or {}).get("global_residual_scale"))
        if metric_scale is not None:
            max_residual = float(metric_scale)
    parsed: dict[tuple[int, str], dict[str, Any]] = {}
    levels: list[int] = []
    methods: list[str] = []
    for key, panel in panel_payloads.items():
        level_text, method = str(key).split("/", 1)
        level = int(level_text.removeprefix("L"))
        method = _display_method(method)
        parsed[(level, method)] = dict(panel)
        if level not in levels:
            levels.append(level)
        if method not in methods:
            methods.append(method)
    order_index = {level: idx for idx, level in enumerate(LEVEL_ORDER)}
    levels.sort(key=lambda item: (order_index.get(item, len(order_index)), item))

    nrows = len(levels)
    ncols = len(methods)
    fig_width = max(11.4, 3.05 * ncols + 1.2)
    fig_height = max(6.05, 1.74 * nrows + 0.82)
    height_ratios = [2.05 if int(level) == 13 else 1.0 for level in levels]
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(fig_width, fig_height),
        dpi=220,
        gridspec_kw={"height_ratios": height_ratios},
    )
    axes_array = np.asarray(axes, dtype=object).reshape(nrows, ncols)
    fig.subplots_adjust(left=0.02, right=0.835, bottom=0.105, top=0.835, wspace=0.012, hspace=0.015)

    row_axes: dict[int, list[plt.Axes]] = {}
    bar_specs: list[tuple[plt.Axes, dict[str, Any]]] = []
    for row_idx, level in enumerate(levels):
        positions = _positions_for_level(level)
        row_axes[level] = []
        for col_idx, method in enumerate(methods):
            ax = axes_array[row_idx, col_idx]
            row_axes[level].append(ax)
            panel = parsed[(level, method)]
            _draw_supernode_panel(
                ax=ax,
                panel=panel,
                positions=positions,
                max_residual=max_residual,
                vis_threshold=float((payload.get("visual_encoding") or {}).get("super_edge_threshold", VIS_THRESHOLD)),
            )
            if row_idx == 0:
                ax.set_title(method, fontsize=9.8, fontweight="bold", pad=3)
            else:
                ax.set_title("")
            bar_specs.append((ax, panel))

    fig.canvas.draw()
    for level, axes_for_row in row_axes.items():
        _draw_level_label(fig, axes_for_row, LEVEL_LABELS.get(int(level), f"L{int(level)}"))
    for ax, panel in bar_specs:
        _draw_panel_summary_bar(fig, ax, panel)

    residual_cax = fig.add_axes([0.89, 0.105, 0.018, 0.73])
    node_cax = fig.add_axes([0.94, 0.105, 0.018, 0.73])
    residual_norm = TwoSlopeNorm(vmin=-max_residual, vcenter=0.0, vmax=max_residual)
    residual_mappable = ScalarMappable(norm=residual_norm, cmap=_residual_colormap())
    residual_mappable.set_array([])
    residual_cb = fig.colorbar(residual_mappable, cax=residual_cax, label=r"$R_{ij}=\hat{P}(j|i)-P^{GT}(j|i)$")
    residual_cb.ax.yaxis.set_ticks_position("left")
    residual_cb.ax.yaxis.set_label_position("left")
    residual_cb.ax.tick_params(labelsize=8, pad=2)
    node_mappable = ScalarMappable(norm=Normalize(vmin=0.0, vmax=1.0), cmap=NODE_CMAP)
    node_mappable.set_array([])
    node_cb = fig.colorbar(node_mappable, cax=node_cax, label=r"$S_i = 1 - TV(i)$")
    node_cb.ax.tick_params(labelsize=8, pad=2)
    _draw_supernode_legend(fig, max_residual=max_residual)
    _draw_summary_bar_key(fig, x0=0.56, y0=0.012)
    fig.suptitle(str(payload["scenario"]), fontsize=12, y=0.972)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def _matching_metric_payload(payload: dict[str, Any], *, output_path: Path) -> dict[str, Any] | None:
    candidate_names = _metric_sidecar_names(payload, output_path=output_path)
    for candidate in candidate_names:
        if candidate.exists():
            return json.loads(candidate.read_text(encoding="utf-8"))
    return None


def _metric_sidecar_names(payload: dict[str, Any], *, output_path: Path) -> list[Path]:
    output = Path(output_path)
    stem = output.stem
    suffix = output.suffix or ".png"
    candidates: list[Path] = []
    if stem.startswith("supernode_transition_residual_"):
        candidates.append(output.with_name(stem.replace("supernode_transition_residual_", "transition_residual_", 1) + ".json"))
    if "_level_" in stem:
        base_stem, level_suffix = stem.rsplit("_level_", 1)
        if base_stem.startswith("supernode_transition_residual_"):
            metric_stem = base_stem.replace("supernode_transition_residual_", "transition_residual_", 1)
            candidates.append(output.with_name(f"{metric_stem}.json"))
    payload_output = payload.get("output_name")
    if payload_output:
        payload_stem = Path(str(payload_output)).stem
        if payload_stem.startswith("supernode_transition_residual_"):
            candidates.append(output.with_name(payload_stem.replace("supernode_transition_residual_", "transition_residual_", 1) + ".json"))
    return list(dict.fromkeys(candidates))


def _metric_overlay_panels(payload: dict[str, Any], metric_payload: dict[str, Any] | None) -> dict[str, Any]:
    panels = {str(key): dict(value) for key, value in dict(payload["panels"]).items()}
    if metric_payload is None:
        return panels
    metric_panels = {str(key): dict(value) for key, value in dict(metric_payload.get("panels", {})).items()}
    metric_fields = (
        "tv_sim",
        "tv_by_state",
        "similarity_by_state",
        "uncovered_rows",
        "P_gt",
        "P_hat",
        "R",
        "E_plus",
        "E_minus",
        "M",
        "mean_abs_R",
    )
    for panel_key, panel in panels.items():
        metric_panel = metric_panels.get(panel_key)
        if metric_panel is None:
            continue
        for field in metric_fields:
            if field in metric_panel:
                panel[field] = metric_panel[field]
        panel["metric_layer_source"] = "seed_averaged_transition_residual_json"
        panel["subnode_layer_source"] = "representative_seed_supernode_json"
    return panels


def _draw_supernode_panel(
    *,
    ax: plt.Axes,
    panel: dict[str, Any],
    positions: dict[int, np.ndarray],
    max_residual: float,
    vis_threshold: float,
) -> None:
    n_true = len(panel["P_gt"])
    pos_true = {node: np.asarray(positions[node], dtype=np.float64) for node in range(n_true)}
    diagnostic = _diagnostic_from_panel(panel)
    _draw_residual_map(
        ax=ax,
        diagnostic=diagnostic,
        positions=pos_true,
        vis_threshold=float(vis_threshold),
        max_residual=float(max_residual),
        show_edge_labels=False,
    )
    learned_pos = _learned_positions(panel, pos_true)
    observation_ids = np.asarray(panel.get("observation_ids", []), dtype=np.int64)
    for learned_idx, point in learned_pos.items():
        obs_id = int(observation_ids[int(learned_idx)]) if int(learned_idx) < observation_ids.shape[0] else 0
        ax.add_patch(
            Circle(
                point,
                radius=_subnode_radius(n_true),
                facecolor=OBS_CMAP(int(obs_id) % OBS_CMAP.N),
                edgecolor="#374151",
                linewidth=0.22,
                alpha=0.92,
                zorder=12,
            )
        )
    _redraw_state_labels(ax=ax, diagnostic=diagnostic, positions=pos_true)


def _diagnostic_from_panel(panel: dict[str, Any]) -> TransitionResidualDiagnostics:
    transition_true = np.asarray(panel["P_gt"], dtype=np.float64)
    transition_predicted = np.asarray(panel["P_hat"], dtype=np.float64)
    residual = np.asarray(panel["R"], dtype=np.float64)
    return TransitionResidualDiagnostics(
        method=str(panel.get("method", "supernode residual map")),
        transition_true=transition_true,
        transition_predicted=transition_predicted,
        residual=residual,
        excess_mass=np.maximum(residual, 0.0),
        missing_mass=np.maximum(-residual, 0.0),
        matched_mass=np.minimum(transition_true, transition_predicted),
        tv_by_state=np.asarray(panel["tv_by_state"], dtype=np.float64),
        similarity_by_state=np.asarray(panel["similarity_by_state"], dtype=np.float64),
        uncovered_rows=np.asarray(panel.get("uncovered_rows", np.zeros(transition_true.shape[0], dtype=bool)), dtype=bool),
        tv_sim=float(panel["tv_sim"]),
        residual_magnitude=(
            np.asarray(panel["mean_abs_R"], dtype=np.float64)
            if "mean_abs_R" in panel
            else np.abs(residual)
        ),
    )


def _redraw_state_labels(
    *,
    ax: plt.Axes,
    diagnostic: TransitionResidualDiagnostics,
    positions: dict[int, np.ndarray],
) -> None:
    for node, center in positions.items():
        similarity = float(diagnostic.similarity_by_state[int(node)])
        ax.text(
            center[0],
            center[1],
            str(node),
            ha="center",
            va="center",
            fontsize=7.0,
            fontweight="bold",
            color="white" if similarity < 0.76 else "#111827",
            zorder=20,
        )


def _draw_panel_summary_bar(fig: plt.Figure, ax: plt.Axes, panel: dict[str, Any]) -> None:
    tv_sim = float(np.clip(panel["tv_sim"], 0.0, 1.0))
    tv_error = float(np.clip(1.0 - tv_sim, 0.0, 1.0))
    residual = np.asarray(panel["R"], dtype=np.float64)
    n_states = max(int(residual.shape[0]), 1)
    positive_mass = float(np.maximum(residual, 0.0).sum() / n_states)
    negative_mass = float(np.maximum(-residual, 0.0).sum() / n_states)
    total_error = positive_mass + negative_mass
    if total_error > 0.0:
        missing_share = tv_error * negative_mass / total_error
        excess_share = tv_error * positive_mass / total_error
    else:
        missing_share = 0.5 * tv_error
        excess_share = 0.5 * tv_error
    similarity_share = max(1.0 - missing_share - excess_share, 0.0)

    bbox = ax.get_position()
    width = 0.088
    height = 0.021
    x0 = float(bbox.x0 + 0.5 * bbox.width - 0.5 * width)
    y0 = float(bbox.y0 - 0.022)
    ax.add_patch(
        plt.Rectangle(
            (x0, y0),
            width,
            height,
            transform=fig.transFigure,
            facecolor=SUMMARY_SIMILARITY_COLOR,
            edgecolor="#9ca3af",
            linewidth=0.45,
            clip_on=False,
            zorder=20,
        )
    )
    segments = (
        (0.0, missing_share, SUMMARY_NEGATIVE_COLOR),
        (missing_share, excess_share, SUMMARY_POSITIVE_COLOR),
        (missing_share + excess_share, similarity_share, SUMMARY_SIMILARITY_COLOR),
    )
    for start, value, color in segments:
        if value <= 0.0:
            continue
        ax.add_patch(
            plt.Rectangle(
                (x0 + width * start, y0),
                width * value,
                height,
                transform=fig.transFigure,
                facecolor=color,
                edgecolor="none",
                clip_on=False,
                zorder=21,
            )
        )
    label_specs = (
        (0.0, missing_share, f"{missing_share:.2f}", 0.14),
        (missing_share, excess_share, f"{excess_share:.2f}", 0.36),
        (missing_share + excess_share, similarity_share, f"{similarity_share:.2f}", 0.78),
    )
    for start, component_width, label, fallback_relative_x in label_specs:
        inside = component_width >= 0.13
        relative_x = start + 0.5 * component_width if inside else fallback_relative_x
        ax.text(
            x0 + width * relative_x,
            y0 + (0.5 * height if inside else -0.004),
            label,
            transform=fig.transFigure,
            ha="center",
            va=("center" if inside else "top"),
            fontsize=5.8,
            fontweight="bold",
            color=SUMMARY_TEXT_COLOR,
            clip_on=False,
            zorder=22,
        )


def _learned_positions(panel: dict[str, Any], pos_true: dict[int, np.ndarray]) -> dict[int, np.ndarray]:
    alignment = np.asarray(panel["alignment"], dtype=np.int64)
    occupancy = np.asarray(panel["occupancy"], dtype=np.float64)
    learned_pos: dict[int, np.ndarray] = {}
    n_true = len(pos_true)
    radius = 0.72 * _super_radius(n_true)
    for true_state in range(n_true):
        members = np.where(alignment == true_state)[0].tolist()
        if not members:
            continue
        members.sort(key=lambda idx: (-float(occupancy[int(idx)]) if int(idx) < occupancy.size else 0.0, int(idx)))
        center = pos_true[true_state]
        offsets = _radial_offsets(len(members), radius)
        for learned_idx, offset in zip(members, offsets, strict=True):
            learned_pos[int(learned_idx)] = np.asarray(center, dtype=np.float64) + offset
    return learned_pos


def _radial_offsets(count: int, radius: float) -> list[np.ndarray]:
    if int(count) <= 1:
        return [np.asarray([0.0, radius], dtype=np.float64)]
    offsets: list[np.ndarray] = []
    remaining = int(count)
    ring_idx = 0
    start_angle = -0.5 * math.pi
    while remaining > 0:
        capacity = 8 if ring_idx == 0 else 14 + 6 * (ring_idx - 1)
        n_ring = min(remaining, capacity)
        ring_radius = radius * (1.0 + 0.48 * ring_idx)
        angle_offset = start_angle + (0.5 * math.pi / max(n_ring, 1) if ring_idx % 2 else 0.0)
        for idx in range(n_ring):
            angle = angle_offset + 2.0 * math.pi * idx / float(n_ring)
            offsets.append(
                np.asarray(
                    [ring_radius * math.cos(angle), ring_radius * math.sin(angle)],
                    dtype=np.float64,
                )
            )
        remaining -= n_ring
        ring_idx += 1
    return offsets


def _draw_arrow(
    ax: plt.Axes,
    start: np.ndarray,
    end: np.ndarray,
    *,
    color: Any,
    width: float,
    alpha: float,
    rad: float,
    zorder: int,
    shrink: float,
    linestyle: str = "-",
) -> None:
    start_arr = np.asarray(start, dtype=np.float64)
    end_arr = np.asarray(end, dtype=np.float64)
    if float(np.linalg.norm(end_arr - start_arr)) <= 1e-9:
        end_arr = start_arr + np.asarray([0.001, 0.001], dtype=np.float64)
    patch = FancyArrowPatch(
        start_arr,
        end_arr,
        arrowstyle="-|>",
        mutation_scale=5.5 + 1.0 * float(width),
        linewidth=float(width),
        color=color,
        alpha=float(alpha),
        linestyle=linestyle,
        connectionstyle=f"arc3,rad={float(rad)}",
        shrinkA=float(shrink) * 72.0,
        shrinkB=float(shrink) * 72.0,
        zorder=int(zorder),
    )
    ax.add_patch(patch)


def _draw_supernode_legend(fig: plt.Figure, *, max_residual: float) -> None:
    width_values = _linewidth_reference_values(max_residual)
    edge_legend = fig.legend(
        handles=(
            [
                Line2D(
                    [0],
                    [0],
                    color="#4b5563",
                    lw=_residual_width(value, max_residual),
                    label=fr"$|R_{{ij}}|={value:.2g}$",
                )
                for value in width_values
            ]
            + [Line2D([0], [0], color="#d1d5db", lw=1.05, label="ground-truth edge")]
        ),
        loc="lower center",
        bbox_to_anchor=(0.33, 0.011),
        ncol=min(4, len(width_values) + 1),
        fontsize=8,
        frameon=True,
        borderpad=0.25,
        labelspacing=0.25,
        handlelength=2.1,
        handletextpad=0.5,
        columnspacing=0.9,
    )
    fig.add_artist(edge_legend)
    fig.text(
        0.56,
        0.050,
        "sub-node colour = observation id",
        ha="left",
        va="center",
        fontsize=7.2,
        fontweight="bold",
        color=SUMMARY_TEXT_COLOR,
    )


def _contribution_width(value: float, max_value: float) -> float:
    return 0.35 + 3.65 * float(value) / max(float(max_value), 1e-9)


def _contribution_reference_values(max_value: float) -> list[float]:
    candidates = [0.05, 0.10, 0.25, 0.50]
    values = [value for value in candidates if value <= float(max_value) * 1.05]
    if values:
        return values[:3]
    return [float(max_value)]


def _super_radius(n_true: int) -> float:
    return 0.39 if int(n_true) <= 8 else 0.30


def _subnode_radius(n_true: int) -> float:
    return 0.058 if int(n_true) <= 8 else 0.048


def _pad_axis(ax: plt.Axes, points: list[np.ndarray]) -> None:
    coords = np.stack([np.asarray(point, dtype=np.float64) for point in points], axis=0)
    x_min, y_min = coords.min(axis=0)
    x_max, y_max = coords.max(axis=0)
    x_span = max(float(x_max - x_min), 1.0)
    y_span = max(float(y_max - y_min), 1.0)
    ax.set_xlim(float(x_min - 0.25 * x_span), float(x_max + 0.25 * x_span))
    ax.set_ylim(float(y_min - 0.30 * y_span), float(y_max + 0.34 * y_span))


def _positions_for_level(level: int) -> dict[int, np.ndarray]:
    reference_positions = exact_cartesian_reference_positions(int(level))
    positions = {
        int(idx): np.asarray([float(col), -float(row)], dtype=np.float64)
        for idx, (row, col) in enumerate(reference_positions.tolist())
    }
    if positions:
        coords = np.stack(list(positions.values()), axis=0)
        x_min = float(coords[:, 0].min())
        x_max = float(coords[:, 0].max())
        y_min = float(coords[:, 1].min())
        y_max = float(coords[:, 1].max())
        x_span = x_max - x_min
        if x_span > 0.0:
            scale = 5.25 / x_span
            center = np.asarray([0.5 * (x_min + x_max), 0.5 * (y_min + y_max)], dtype=np.float64)
            positions = {node: center + (np.asarray(pos) - center) * scale for node, pos in positions.items()}
    return positions


def _ground_truth_observation_ids_for_values(level: int, *, observation_values: list[Any]) -> np.ndarray:
    positions = exact_cartesian_reference_positions(int(level))
    maze = _maze_array(int(level))
    observation_to_id = {
        _observation_key(value): int(obs_id)
        for obs_id, value in enumerate(observation_values)
    }
    ids = []
    for row, col in positions.tolist():
        agent = Agent(initial_position=(int(row), int(col)), maze=maze, sensors={"sensor": "cardinal distance", "range": 1})
        observation = _observation_key(agent.make_observation(maze))
        ids.append(int(observation_to_id.get(observation, -1)))
    return np.asarray(ids, dtype=np.int64)


def _maze_array(level: int) -> np.ndarray:
    rows = MAZE_LEVELS[int(level)]
    return np.asarray([[1 if str(value) == "X" else 0 for value in row] for row in rows], dtype=np.int64)


def _display_method(method: str) -> str:
    return METHOD_REPLACEMENTS.get(str(method), str(method))


def _display_title(title: str) -> str:
    normalized = str(title)
    normalized = normalized.replace(
        "Ground-truth-aligned transition residuals: MERCURY initialised versus CSCG",
        FINAL_TITLE,
    )
    return normalized


def _draw_level_label(fig: plt.Figure, axes_for_row: list[plt.Axes], label: str) -> None:
    if not axes_for_row:
        return
    boxes = [ax.get_position() for ax in axes_for_row]
    y0 = min(float(box.y0) for box in boxes)
    y1 = max(float(box.y1) for box in boxes)
    fig.text(
        0.011,
        0.5 * (y0 + y1),
        label,
        rotation=90,
        va="center",
        ha="center",
        fontsize=12,
        fontweight="bold",
    )


def _summary_name(
    *,
    rand_prob: float,
    viterbi_processed: bool,
    include_self_loops: bool,
    action_conditioned: bool,
) -> str:
    prefix = "summary_supernode_action_conditioned" if action_conditioned else "summary_supernode"
    suffix = _rand_prob_suffix(rand_prob)
    if viterbi_processed:
        suffix += "_viterbi100_smooth1em09"
    if include_self_loops:
        suffix += "_self_loops"
    return f"{prefix}{suffix}.json"


if __name__ == "__main__":
    main()
