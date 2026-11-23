"""Study-level snapshot evaluation service for the canonical protocol.

Implements the First-Class Snapshot Evaluation Architecture:

1. Training produces immutable model snapshots.
2. This service consumes those snapshots and produces versioned evaluation artefacts.
3. Plotting scripts consume evaluation artefacts only.

Usage:
    python -m mercury_runs evaluate-study \\
        --study-root results/current/studies/<study> \\
        --level 13 --sensor cartesian --rand-prob 0.3
"""
from __future__ import annotations

import hashlib
import json
import logging
import multiprocessing
import os
import queue
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .evaluation_protocol import EvaluationProtocol, default_canonical_v4_protocol
from ..infrastructure.paper_precision import (
    action_ids_from_action_vectors,
    compute_transition_statistics,
    evaluate_transition_structure_canonical,
    exact_cartesian_reference_positions,
    exact_cartesian_state_ids_for_level,
    generate_random_start_walks,
    normalize_action_rows,
    validate_evaluation_walk_action_alignment,
    cartesian_state_ids,
)
from ..infrastructure.runtime import atomic_write_json

logger = logging.getLogger(__name__)

# Bootstrap-phase evaluation namespaces. A hybrid run trains a bootstrap model
# first, then CSCG on top of it; the bootstrap model is evaluated under its own
# method name so its records never collide with the CSCG-phase ones, which
# restart their step counter at 0.
_HYBRID_PHASE_METHODS = {
    "mercury_cscg_hybrid_mercury": "mercury_cscg_hybrid",
    "tmgwr_cscg_hybrid_tmgwr": "tmgwr_cscg_hybrid",
}
# Bootstrap snapshots live under bundles/, keyed by the stage that wrote them.
_HYBRID_BOOTSTRAP_SNAPSHOT_DIR = {
    "mercury_cscg_hybrid_mercury": "mercury",
    "tmgwr_cscg_hybrid_tmgwr": "sensory",
}

ProgressCallback = Callable[[dict[str, Any]], None]


# ---------------------------------------------------------------------------
# Walk-cache utilities
# ---------------------------------------------------------------------------


def _walk_cache_dir(study_root: Path, protocol: EvaluationProtocol) -> Path:
    return study_root / "evaluation_caches" / protocol.protocol_id() / "walks"


def _walk_cache_filename(protocol: EvaluationProtocol, level: int) -> str:
    sr = f"_range={protocol.sensor_range}" if protocol.sensor_range is not None else ""
    return f"level={level}_sensor={protocol.sensor}{sr}_rand={protocol.rand_prob}_eval-seed={protocol.evaluation_seed}.npz"


def _save_walk_cache(cache_dir: Path, filename: str, walk_data: dict[str, Any]) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    save_dict: dict[str, Any] = {
        "n_walks": np.array(walk_data["n_walks"], dtype=np.int64),
    }
    for i, (obs, act, gt_exact, gt_cart, gt_act, coll) in enumerate(zip(
        walk_data["filtered_obs_per_walk"],
        walk_data["filtered_actions_per_walk"],
        walk_data["gt_exact_per_walk"],
        walk_data.get("gt_cartesian_per_walk", [None] * walk_data["n_walks"]),
        walk_data["gt_action_per_walk"],
        walk_data.get("collisions_per_walk", [None] * walk_data["n_walks"]),
    )):
        prefix = f"walk_{i:05d}"
        if obs is None:
            save_dict[f"{prefix}_valid"] = np.array(False)
        else:
            save_dict[f"{prefix}_valid"] = np.array(True)
            save_dict[f"{prefix}_obs"] = np.asarray(obs)
            save_dict[f"{prefix}_act"] = np.asarray(act)
            save_dict[f"{prefix}_gt_exact"] = np.asarray(gt_exact)
            if gt_cart is not None:
                save_dict[f"{prefix}_gt_cart"] = np.asarray(gt_cart)
            save_dict[f"{prefix}_gt_action"] = np.asarray(gt_act)
            if coll is not None:
                save_dict[f"{prefix}_collisions"] = np.asarray(coll)
    np.savez_compressed(cache_dir / filename, **save_dict)


def _load_walk_cache(cache_dir: Path, filename: str) -> dict[str, Any]:
    data = np.load(cache_dir / filename, allow_pickle=False)
    n_walks = int(data["n_walks"])
    result: dict[str, Any] = {"n_walks": n_walks}
    keys = ["filtered_obs_per_walk", "filtered_actions_per_walk", "gt_exact_per_walk",
            "gt_cartesian_per_walk", "gt_action_per_walk", "collisions_per_walk"]
    for key in keys:
        result[key] = []

    for i in range(n_walks):
        prefix = f"walk_{i:05d}"
        if not bool(data.get(f"{prefix}_valid", np.array(False))):
            for key in keys:
                result[key].append(None)
        else:
            result["filtered_obs_per_walk"].append(data[f"{prefix}_obs"])
            result["filtered_actions_per_walk"].append(data[f"{prefix}_act"])
            result["gt_exact_per_walk"].append(data[f"{prefix}_gt_exact"])
            result["gt_cartesian_per_walk"].append(data.get(f"{prefix}_gt_cart"))
            result["gt_action_per_walk"].append(data[f"{prefix}_gt_action"])
            result["collisions_per_walk"].append(data.get(f"{prefix}_collisions"))
    return result


def generate_and_cache_walks(
    *,
    study_root: Path,
    protocol: EvaluationProtocol,
    level: int,
    force_regenerate: bool = False,
) -> dict[str, Any]:
    """Return cached walk data, generating and persisting it if not present."""
    from ..algorithms.cscg.prepare import filter_valid_cscg_rows

    cache_dir = _walk_cache_dir(study_root, protocol)
    filename = _walk_cache_filename(protocol, level)
    cache_path = cache_dir / filename

    if cache_path.exists() and not force_regenerate:
        logger.info("Loading walk cache: %s", cache_path)
        return _load_walk_cache(cache_dir, filename)

    logger.info("Generating %d evaluation walks (level=%d)…", protocol.number_of_walks, level)
    walks = generate_random_start_walks(
        level=level,
        sensor=protocol.sensor,
        sensor_range=protocol.sensor_range,
        rand_prob=protocol.rand_prob,
        num_walks=protocol.number_of_walks,
        walk_length=protocol.walk_length,
        base_seed=protocol.evaluation_seed,
        valid_trajectories_only=protocol.valid_trajectories_only,
    )

    valid_traj = protocol.valid_trajectories_only
    filtered_obs_list = []
    filtered_actions_list = []
    gt_exact_list = []
    gt_cartesian_list = []
    gt_action_list = []
    collisions_list = []

    for walk in walks:
        validate_evaluation_walk_action_alignment(
            positions=np.asarray(walk.cartesian_observations),
            actions=np.asarray(walk.actions),
        )
        filtered_obs, filtered_actions, _, _ = filter_valid_cscg_rows(
            observations=np.asarray(walk.observations),
            actions=np.asarray(walk.actions),
            collisions=np.asarray(walk.collisions, dtype=bool),
            valid_trajectories_only=valid_traj,
        )
        if filtered_obs.shape[0] == 0 or filtered_actions.shape[0] == 0:
            filtered_obs_list.append(None)
            filtered_actions_list.append(None)
            gt_exact_list.append(None)
            gt_cartesian_list.append(None)
            gt_action_list.append(None)
            collisions_list.append(None)
            continue

        gt_collisions = np.asarray(walk.collisions, dtype=bool)
        mask = (~gt_collisions) if valid_traj else slice(None)
        cartesian_positions = np.asarray(walk.cartesian_observations)[mask]
        gt_exact = exact_cartesian_state_ids_for_level(level_index=level, positions=cartesian_positions)
        gt_cart = _cartesian_state_ids_local(cartesian_positions)
        gt_act = action_ids_from_action_vectors(filtered_actions)
        coll = np.asarray(walk.collisions, dtype=bool)[mask] if valid_traj else np.asarray(walk.collisions, dtype=bool)

        min_len = min(filtered_obs.shape[0], int(gt_exact.shape[0]))
        filtered_obs_list.append(filtered_obs[:min_len])
        filtered_actions_list.append(filtered_actions[:min_len])
        gt_exact_list.append(gt_exact[:min_len].astype(np.int64, copy=False))
        gt_cartesian_list.append(gt_cart[:min_len].astype(np.int64, copy=False))
        gt_action_list.append(gt_act[:min_len])
        collisions_list.append(coll[:min_len])

    walk_data = {
        "filtered_obs_per_walk": filtered_obs_list,
        "filtered_actions_per_walk": filtered_actions_list,
        "gt_exact_per_walk": gt_exact_list,
        "gt_cartesian_per_walk": gt_cartesian_list,
        "gt_action_per_walk": gt_action_list,
        "collisions_per_walk": collisions_list,
        "n_walks": len(walks),
    }

    _save_walk_cache(cache_dir, filename, walk_data)

    # Save protocol alongside the cache.
    proto_path = cache_dir.parent / "protocol.json"
    if not proto_path.exists():
        atomic_write_json(proto_path, protocol.to_dict())

    return walk_data


def _cartesian_state_ids_local(positions: np.ndarray) -> np.ndarray:
    return cartesian_state_ids(positions)


# ---------------------------------------------------------------------------
# Empirical ground-truth construction
# ---------------------------------------------------------------------------


def build_empirical_ground_truth(
    *,
    walk_data: dict[str, Any],
    n_true: int,
    n_actions: int,
    level: int,
) -> dict[str, Any]:
    """Construct T_GT, P_GT_action, valid_action_rows from the shared held-out walks.

    The same empirical counts are reused across all methods and checkpoints.
    """
    T_GT = np.zeros((n_actions, n_true, n_true), dtype=np.int64)

    for gt_exact, gt_action in zip(
        walk_data["gt_exact_per_walk"],
        walk_data["gt_action_per_walk"],
    ):
        if gt_exact is None or gt_action is None:
            continue
        states = np.asarray(gt_exact, dtype=np.int64)
        actions = np.asarray(gt_action, dtype=np.int64)
        length = min(states.shape[0], actions.shape[0])
        if length < 2:
            continue
        # Transition: states[t] -> states[t+1] under action actions[t+1]
        # (post-step action convention).
        action_idx = actions[1:length]
        source_idx = states[:length - 1]
        target_idx = states[1:length]
        valid = (
            (0 <= action_idx) & (action_idx < n_actions)
            & (0 <= source_idx) & (source_idx < n_true)
            & (0 <= target_idx) & (target_idx < n_true)
        )
        np.add.at(T_GT, (action_idx[valid], source_idx[valid], target_idx[valid]), 1)

    row_sums = T_GT.sum(axis=2)  # (n_actions, n_true)
    valid_rows = row_sums > 0  # (n_actions, n_true) bool
    P_GT = normalize_action_rows(T_GT.astype(np.float64))

    # Checksum for verifying shared GT across evaluations.
    checksum = hashlib.sha256(T_GT.tobytes()).hexdigest()[:16]

    return {
        "T_GT": T_GT,
        "P_GT_action": P_GT,
        "valid_action_rows": valid_rows,
        "N_ai": row_sums,
        "n_true": n_true,
        "n_actions": n_actions,
        "checksum": checksum,
        "level": level,
    }


# ---------------------------------------------------------------------------
# Evaluation result serialisation
# ---------------------------------------------------------------------------


def _arrays_to_lists(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _arrays_to_lists(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_arrays_to_lists(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def serialise_evaluation_result(
    *,
    protocol: EvaluationProtocol,
    level: int,
    method: str,
    step: int,
    observed_samples: int,
    walk_cache_id: str,
    gt: dict[str, Any],
    result: Any,
    inference_meta: dict[str, Any],
    latent_precision: float,
    node_recall: float,
    n_learned: int,
    n_true: int,
    n_actions: int,
    model_fit: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the full v4 evaluation record."""
    r = result
    edge_by_thresh: dict[str, Any] = {}
    for thresh, metrics in r.action_edge_metrics_by_threshold.items():
        edge_by_thresh[str(thresh)] = metrics
    coll_edge: dict[str, Any] = {}
    for thresh, metrics in r.collapsed_eval_edge_by_threshold.items():
        coll_edge[str(thresh)] = metrics

    record = {
        "schema": "transition_structure_evaluation_v4",
        "protocol_id": protocol.protocol_id(),
        "protocol": protocol.to_dict(),
        "method": method,
        "level": level,
        "snapshot_step": step,
        "observed_samples": observed_samples,
        "walk_cache_id": walk_cache_id,
        "ground_truth": {
            "source": protocol.ground_truth_source,
            "transition_count_checksum": gt["checksum"],
            "state_order": list(range(gt["n_true"])),
            "action_order": list(range(gt["n_actions"])),
            "T_true_action": _arrays_to_lists(gt["T_GT"]),
            "P_true_action": _arrays_to_lists(gt["P_GT_action"]),
            "valid_action_rows": _arrays_to_lists(gt["valid_action_rows"]),
            "N_ai": _arrays_to_lists(gt["N_ai"]),
        },
        "inference": inference_meta,
        "alignment": {
            "cooccurrence": _arrays_to_lists(r.cooccurrence),
            "action_source_cooccurrence": _arrays_to_lists(r.action_source_cooccurrence),
            "learned_to_true": _arrays_to_lists(r.alignment),
            "true_to_learned": _arrays_to_lists(
                np.argmax(r.cooccurrence, axis=0) if r.cooccurrence.size > 0 else np.array([])
            ),
            "aligned_learned_states": _arrays_to_lists(r.aligned_learned_mask),
            "unmatched_learned_states": r.unaligned_learned_states,
            "tie_breaking": protocol.alignment_tie_breaking,
        },
        "state_representation": {
            "latent_precision": float(latent_precision),
            "node_recall": float(node_recall),
            "n_learned_states": n_learned,
            "n_true_states": n_true,
            "n_actions": n_actions,
        },
        "action_conditioned": {
            "P_true_full": _arrays_to_lists(r.P_true_full),
            "P_hat_full": _arrays_to_lists(r.P_hat_full),
            "valid_rows": _arrays_to_lists(r.valid_action_rows),
            "valid_non_self_loop_rows": _arrays_to_lists(r.valid_non_self_loop_rows),
            "source_action_counts": _arrays_to_lists(r.source_action_counts),
            "evaluation_occupancy_weights": _arrays_to_lists(r.evaluation_occupancy_weights),
            "tv_by_pair_full": _arrays_to_lists(r.tv_by_pair_full),
            "tv_by_pair_non_self_loop": _arrays_to_lists(r.tv_by_pair_non_self_loop),
            "tv_by_state_occupancy": _arrays_to_lists(r.tv_by_state_occupancy),
            "tv_by_state_macro": _arrays_to_lists(r.tv_by_state_macro),
            "tv_occupancy": float(r.tv_occupancy),
            "tv_similarity_occupancy": float(1.0 - r.tv_occupancy),
            "tv_macro": float(r.tv_macro),
            "tv_non_self_loop_occupancy": float(r.tv_non_self_loop_occupancy),
            "tv_non_self_loop_similarity_occupancy": float(1.0 - r.tv_non_self_loop_occupancy),
            "tv_non_self_loop_macro": float(r.tv_non_self_loop_macro),
            "tv_non_self_loop_by_state_occupancy": _arrays_to_lists(r.tv_non_self_loop_by_state_occupancy),
            "tv_non_self_loop_by_state_macro": _arrays_to_lists(r.tv_non_self_loop_by_state_macro),
            "unmatched_target_mass": _arrays_to_lists(r.unmatched_target_mass),
            "total_unmatched_target_mass": float(r.total_unmatched_target_mass),
            "edge_metrics_by_threshold": edge_by_thresh,
        },
        "policy_collapsed": {
            "action_weights_evaluation_policy": _arrays_to_lists(r.action_weights_evaluation_policy),
            "state_occupancy_weights": _arrays_to_lists(r.state_occupancy_weights),
            "P_true": _arrays_to_lists(r.collapsed_eval_P_true),
            "P_hat": _arrays_to_lists(r.collapsed_eval_P_hat),
            "tv_by_state": _arrays_to_lists(r.collapsed_eval_tv_by_state),
            "tv_occupancy": float(r.collapsed_eval_tv_occupancy),
            "tv_similarity_occupancy": float(1.0 - r.collapsed_eval_tv_occupancy),
            "tv_macro": float(r.collapsed_eval_tv_macro),
            "tv_non_self_loop_by_state": _arrays_to_lists(r.collapsed_eval_tv_non_self_loop_by_state),
            "tv_non_self_loop_occupancy": float(r.collapsed_eval_tv_non_self_loop_occupancy),
            "tv_non_self_loop_similarity_occupancy": float(1.0 - r.collapsed_eval_tv_non_self_loop_occupancy),
            "tv_non_self_loop_macro": float(r.collapsed_eval_tv_non_self_loop_macro),
            "edge_metrics_by_threshold": coll_edge,
        },
    }
    if model_fit is not None:
        record["model_fit"] = _arrays_to_lists(model_fit)
    return record


def _evaluation_snapshot_dir(run_root: Path, protocol: EvaluationProtocol, method: str | None = None) -> Path:
    eval_dir = run_root / "evaluations" / protocol.protocol_id() / "snapshots"
    return eval_dir / str(method) if method else eval_dir


def save_evaluation_result(
    *,
    run_root: Path,
    protocol: EvaluationProtocol,
    step: int,
    record: dict[str, Any],
    method: str | None = None,
) -> Path:
    eval_dir = _evaluation_snapshot_dir(run_root, protocol, method)
    eval_dir.mkdir(parents=True, exist_ok=True)
    path = eval_dir / f"step={step:08d}.json"
    atomic_write_json(path, record)
    return path


def is_evaluated(*, run_root: Path, protocol: EvaluationProtocol, step: int, method: str | None = None) -> bool:
    path = _evaluation_snapshot_dir(run_root, protocol, method) / f"step={step:08d}.json"
    if path.exists():
        return True
    if method in {"cscg", "mercury", "mercury_cscg_hybrid", "tmgwr_cscg_hybrid"}:
        legacy_path = _evaluation_snapshot_dir(run_root, protocol, None) / f"step={step:08d}.json"
        return legacy_path.exists()
    return False


def save_evaluation_status(
    *,
    run_root: Path,
    protocol: EvaluationProtocol,
    status: str,
    evaluated_steps: list[int],
    method: str | None = None,
) -> None:
    status_dir = run_root / "evaluations" / protocol.protocol_id()
    if method:
        status_dir = status_dir / str(method)
    status_path = status_dir / "status.json"
    status_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(status_path, {
        "status": status,
        "evaluated_steps": sorted(evaluated_steps),
        "protocol_id": protocol.protocol_id(),
    })


# ---------------------------------------------------------------------------
# Study-level snapshot discovery
# ---------------------------------------------------------------------------


def discover_runs(study_root: Path) -> list[Path]:
    """Find all run directories under study_root.

    Checks study_root/runs/ first (canonical layout); falls back to scanning
    study_root directly for directories that contain run_status.json (flat layout).
    """
    runs_dir = study_root / "runs"
    if runs_dir.exists():
        return sorted(p for p in runs_dir.iterdir() if p.is_dir())
    # Flat layout: runs are direct subdirectories of study_root.
    return sorted(p for p in study_root.iterdir() if p.is_dir() and (p / "run_status.json").exists())


def manifest_run_ids_for_filters(
    *,
    study_root: Path,
    method: str,
    level: int,
    rand_prob: float | None,
    cscg_batch_size: int | None,
    cscg_clone_preset: str | None,
) -> set[str] | None:
    if cscg_batch_size is None and cscg_clone_preset is None:
        return None
    manifest_path = Path(study_root) / "study_manifest.jsonl"
    if not manifest_path.exists():
        return None
    manifest_method = _HYBRID_PHASE_METHODS.get(str(method), str(method))
    selected: set[str] = set()
    with manifest_path.open(encoding="utf-8") as handle:
        for raw in handle:
            raw = raw.strip()
            if not raw:
                continue
            try:
                config = json.loads(raw)["config"]
            except Exception:
                continue
            config_method = str(config.get("method"))
            embedded_baseline_cscg = (
                method == "cscg"
                and config_method == "mercury_cscg_hybrid"
                and bool(config.get("baseline_cscg", False))
            )
            if config_method != manifest_method and not embedded_baseline_cscg:
                continue
            if int(config.get("level", -1)) != int(level):
                continue
            if rand_prob is not None and float(config.get("rand_prob", "nan")) != float(rand_prob):
                continue
            if cscg_batch_size is not None and int(config.get("cscg_batch_size", -1)) != int(cscg_batch_size):
                continue
            if cscg_clone_preset is not None and config.get("cscg_clone_preset") != str(cscg_clone_preset):
                continue
            run_id = config.get("requested_run_id")
            if run_id is not None:
                selected.add(str(run_id))

    # A study launched with reuse_existing_run may record the newly requested
    # identity in the manifest while the worker reuses a semantically matching
    # run under an older identity. Resolve those effective run directories from
    # their persisted configuration as well; otherwise any manifest-backed
    # filter (notably cscg_batch_size) rejects every reused run.
    for run_root in discover_runs(Path(study_root)):
        try:
            persisted = json.loads((run_root / "run_config.json").read_text(encoding="utf-8"))
            config = dict(persisted.get("cli_args") or persisted.get("run_payload") or {})
        except Exception:
            continue
        config_method = str(config.get("method"))
        embedded_baseline_cscg = (
            method == "cscg"
            and config_method == "mercury_cscg_hybrid"
            and bool(config.get("baseline_cscg", False))
        )
        if config_method != manifest_method and not embedded_baseline_cscg:
            continue
        if int(config.get("level", -1)) != int(level):
            continue
        if rand_prob is not None and float(config.get("rand_prob", "nan")) != float(rand_prob):
            continue
        if cscg_batch_size is not None and int(config.get("cscg_batch_size", -1)) != int(cscg_batch_size):
            continue
        if cscg_clone_preset is not None and config.get("cscg_clone_preset") != str(cscg_clone_preset):
            continue
        selected.add(run_root.name)
    return selected


def discover_snapshots(run_root: Path, method: str) -> list[Path]:
    """Find sorted snapshot directories for a method in a run."""
    from ..infrastructure.paper_precision_snapshots import list_paper_precision_snapshots
    bootstrap_dir = _HYBRID_BOOTSTRAP_SNAPSHOT_DIR.get(str(method))
    if bootstrap_dir is not None:
        root = Path(run_root) / "bundles" / "snapshots" / bootstrap_dir / "paper_precision"
        if not root.exists():
            return []
        return sorted(
            [path for path in root.iterdir() if path.is_dir() and path.name.startswith("step=")],
            key=lambda path: int(path.name.split("=", 1)[1]),
        )
    return list_paper_precision_snapshots(run_root, method)


def run_level(run_root: Path) -> int | None:
    """Return the configured maze level for a run directory."""
    try:
        cfg = json.loads((run_root / "run_config.json").read_text(encoding="utf-8"))
        return int(cfg.get("run_payload", {}).get("level"))
    except Exception:
        return None


def run_rand_prob(run_root: Path) -> float | None:
    """Return the configured training rand_prob for a run directory."""
    try:
        cfg = json.loads((run_root / "run_config.json").read_text(encoding="utf-8"))
        return float(cfg.get("run_payload", {}).get("rand_prob"))
    except Exception:
        return None


def run_cscg_batch_size(run_root: Path) -> int | None:
    """Return the configured CSCG batch size for a run directory."""
    try:
        cfg = json.loads((run_root / "run_config.json").read_text(encoding="utf-8"))
        value = cfg.get("run_payload", {}).get("cscg_batch_size")
        return None if value is None else int(value)
    except Exception:
        return None


def run_cscg_clone_preset(run_root: Path) -> str | None:
    """Return the configured CSCG clone preset for a run directory."""
    try:
        cfg = json.loads((run_root / "run_config.json").read_text(encoding="utf-8"))
        value = cfg.get("run_payload", {}).get("cscg_clone_preset")
        return None if value is None else str(value)
    except Exception:
        return None


def snapshot_step_hint(snapshot_path: Path) -> int | None:
    """Parse step=N snapshot directory names without loading the snapshot."""
    try:
        return int(snapshot_path.name.split("=", 1)[1])
    except Exception:
        return None


def run_evaluation_log_path(*, log_dir: Path, run_root: Path) -> Path:
    return log_dir / f"run-{run_root.name}.log"


def filter_evaluation_runs(
    *,
    runs: list[Path],
    method: str,
    level: int,
    protocol: EvaluationProtocol,
    rand_prob: float | None = None,
    snapshot_steps: set[int] | None = None,
    cscg_batch_size: int | None = None,
    cscg_clone_preset: str | None = None,
    run_ids: set[str] | None = None,
    resume: bool = True,
    force: bool = False,
) -> list[Path]:
    """Return runs matching level/method that still need evaluation."""
    selected: list[Path] = []
    for run_root in runs:
        if run_ids is not None and run_root.name not in run_ids:
            continue
        if run_level(run_root) != level:
            continue
        if rand_prob is not None and run_rand_prob(run_root) != float(rand_prob):
            continue
        if run_ids is None and cscg_batch_size is not None and run_cscg_batch_size(run_root) != int(cscg_batch_size):
            continue
        if run_ids is None and cscg_clone_preset is not None and run_cscg_clone_preset(run_root) != str(cscg_clone_preset):
            continue
        snapshots = discover_snapshots(run_root, method)
        if snapshot_steps is not None:
            snapshots = [path for path in snapshots if snapshot_step_hint(path) in snapshot_steps]
        if not snapshots:
            continue
        if resume and not force:
            step_hints = [snapshot_step_hint(path) for path in snapshots]
            known_steps = [step for step in step_hints if step is not None]
            if known_steps and len(known_steps) == len(snapshots) and all(
                is_evaluated(run_root=run_root, protocol=protocol, step=step, method=method) for step in known_steps
            ):
                continue
        selected.append(run_root)
    return selected


# ---------------------------------------------------------------------------
# Main evaluation service
# ---------------------------------------------------------------------------


def evaluate_run_snapshots(
    *,
    run_root: Path,
    method: str,
    protocol: EvaluationProtocol,
    level: int,
    walk_data: dict[str, Any],
    gt: dict[str, Any],
    walk_cache_id: str,
    resume: bool = True,
    force: bool = False,
    snapshot_steps: set[int] | None = None,
    progress_callback: ProgressCallback | None = None,
) -> list[dict[str, Any]]:
    """Evaluate all snapshots for one run under the given protocol."""
    def emit(event: str, **payload: Any) -> None:
        if progress_callback is not None:
            progress_callback({"event": event, "run": run_root.name, **payload})

    snapshots = discover_snapshots(run_root, method)
    if snapshot_steps is not None:
        snapshots = [path for path in snapshots if snapshot_step_hint(path) in snapshot_steps]
    if not snapshots:
        logger.info("No snapshots found for method=%s in %s", method, run_root)
        return []

    if method in {"cscg", "mercury_cscg_hybrid", "tmgwr_cscg_hybrid"}:
        from ..algorithms.cscg.snapshot_adapter import CSCGSnapshotAdapter
        adapter = CSCGSnapshotAdapter()
    elif method in {"mercury", "mercury_cscg_hybrid_mercury"}:
        from ..algorithms.mercury.snapshot_adapter import MercurySnapshotAdapter
        adapter = MercurySnapshotAdapter()
    elif method in {"sensory", "tmgwr_cscg_hybrid_tmgwr"}:
        from ..algorithms.sensory.snapshot_adapter import SensorySnapshotAdapter
        adapter = SensorySnapshotAdapter()
    else:
        logger.warning("No adapter for method=%s, skipping.", method)
        return []

    n_true = gt["n_true"]
    n_actions = gt["n_actions"]
    evaluated_steps: list[int] = []
    results: list[dict[str, Any]] = []
    emit("run_start", snapshots=len(snapshots))

    for snapshot_path in snapshots:
        try:
            step_hint = int(snapshot_path.name.split("=", 1)[1])
        except Exception:
            step_hint = None

        if step_hint is not None and resume and not force and is_evaluated(
            run_root=run_root, protocol=protocol, step=step_hint, method=method
        ):
            logger.info("  step=%d already evaluated, skipping.", step_hint)
            evaluated_steps.append(step_hint)
            emit("step_skipped", step=step_hint)
            continue

        snapshot = adapter.load_snapshot(snapshot_path)
        step = adapter.step(snapshot)
        observed_samples = adapter.observed_samples(snapshot)

        if resume and not force and is_evaluated(run_root=run_root, protocol=protocol, step=step, method=method):
            logger.info("  step=%d already evaluated, skipping.", step)
            evaluated_steps.append(step)
            emit("step_skipped", step=step)
            continue

        logger.info("  Evaluating step=%d …", step)
        emit("step_start", step=step)
        if progress_callback is None:
            print(f"  evaluating step={step}", flush=True)

        # Determine batch size for CSCG.
        meta = snapshot.get("meta", {})
        batch_size = int(meta.get("batch_size", meta.get("paper_precision", {}).get("walk_length", 1024)))

        # Decode walks.
        emit("decode_start", step=step)
        if progress_callback is None:
            print(f"    decoding evaluation walks for step={step}", flush=True)
        if method in {"cscg", "mercury_cscg_hybrid", "tmgwr_cscg_hybrid"}:
            decoded = adapter.decode_walks(snapshot, walk_data, batch_size=batch_size)
            inference_meta = adapter.inference_metadata(snapshot, batch_size=batch_size)
            T_hat = adapter.transition_tensor(snapshot)
            model_fit = adapter.negative_log2_likelihood(snapshot, walk_data)
        else:
            decoded = adapter.decode_walks(snapshot, walk_data, level=level)
            inference_meta = adapter.inference_metadata(snapshot)
            T_hat = adapter.transition_tensor(snapshot, n_actions=n_actions)
            model_fit = None

        decoded_walks = decoded["decoded_walks"]
        gt_exact_walks = decoded["gt_exact_walks"]
        gt_action_walks = decoded["gt_action_walks"]
        emit("decode_done", step=step, walks=len(decoded_walks))
        if progress_callback is None:
            print(f"    decoded {len(decoded_walks)} walk(s) for step={step}", flush=True)

        if not decoded_walks:
            logger.warning("  No decoded walks for step=%d, skipping.", step)
            continue

        n_learned = adapter.n_learned_states(snapshot)

        # Compute C, H, N_ai from decoded walks.
        C, H, N_ai = compute_transition_statistics(
            decoded_walks=decoded_walks,
            true_walks=gt_exact_walks,
            action_walks=gt_action_walks,
            n_learned=n_learned,
            n_true=n_true,
            n_actions=n_actions,
        )

        # Latent precision and node recall from same cooccurrence.
        from ..infrastructure.paper_precision import (
            concatenate_walk_series,
            compute_fixed_map_precision_details,
            compute_hard_alignment,
        )
        inferred_series = concatenate_walk_series(decoded_walks)
        gt_series = concatenate_walk_series(gt_exact_walks)
        prec_details = compute_fixed_map_precision_details(
            inferred_states=inferred_series,
            ground_truth_states=gt_series,
            metric_name="latent_precision",
            inferred_state_count=n_learned,
            ground_truth_state_count=n_true,
        )
        latent_precision = float(prec_details["latent_precision"])
        node_recall = float(prec_details["latent_recall"])

        # Canonical evaluation.
        result = evaluate_transition_structure_canonical(
            T_hat=T_hat,
            T_true=gt["T_GT"].astype(np.float64),
            cooccurrence=C,
            action_source_cooccurrence=H,
            source_action_counts=N_ai,
            edge_thresholds=tuple(float(e) for e in protocol.edge_thresholds),
        )

        record = serialise_evaluation_result(
            protocol=protocol,
            level=level,
            method=method,
            step=step,
            observed_samples=observed_samples,
            walk_cache_id=walk_cache_id,
            gt=gt,
            result=result,
            inference_meta=inference_meta,
            latent_precision=latent_precision,
            node_recall=node_recall,
            n_learned=n_learned,
            n_true=n_true,
            n_actions=n_actions,
            model_fit=model_fit,
        )

        save_evaluation_result(run_root=run_root, protocol=protocol, step=step, record=record, method=method)
        evaluated_steps.append(step)
        results.append(record)
        emit("step_done", step=step)

        m = result.primary_metrics(threshold=protocol.edge_thresholds[0])
        logger.info(
            "  step=%d | tv_occ=%.4f | precision=%.3f | recall=%.3f",
            step,
            m["action_tv_occupancy_weighted"],
            latent_precision,
            node_recall,
        )

    save_evaluation_status(
        run_root=run_root,
        protocol=protocol,
        status="complete",
        evaluated_steps=evaluated_steps,
        method=method,
    )
    emit("run_done", evaluated_steps=len(evaluated_steps), new_records=len(results))
    return results


def _evaluate_run_worker(payload: dict[str, Any]) -> dict[str, Any]:
    """ProcessPool entry point for one run evaluation."""
    run_root = Path(payload["run_root"])
    log_path = run_evaluation_log_path(log_dir=Path(payload["log_dir"]), run_root=run_root)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    progress_queue = payload.get("progress_queue")

    def progress_callback(event: dict[str, Any]) -> None:
        if progress_queue is not None:
            progress_queue.put(event)

    with log_path.open("w", encoding="utf-8") as log_handle:
        with redirect_stdout(log_handle), redirect_stderr(log_handle):
            logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
            records = evaluate_run_snapshots(
                run_root=run_root,
                method=str(payload["method"]),
                protocol=payload["protocol"],
                level=int(payload["level"]),
                walk_data=payload["walk_data"],
                gt=payload["gt"],
                walk_cache_id=str(payload["walk_cache_id"]),
                resume=bool(payload["resume"]),
                force=bool(payload["force"]),
                snapshot_steps=payload.get("snapshot_steps"),
                progress_callback=progress_callback,
            )
    return {"run": run_root.name, "records": len(records), "log_path": str(log_path)}


def _clear_status_line() -> None:
    if sys.stdout.isatty() or os.environ.get("MERCURY_FORCE_TTY_PROGRESS") == "1":
        print("\r\033[K", end="", flush=True)


def _render_parallel_progress(
    *,
    active: dict[str, str],
    completed: int,
    total: int,
    last_done: str | None,
) -> None:
    if not (sys.stdout.isatty() or os.environ.get("MERCURY_FORCE_TTY_PROGRESS") == "1"):
        return
    fragments = [f"runs {completed}/{total}"]
    if last_done:
        fragments.append(f"last={last_done}")
    for index, (run_name, state) in enumerate(sorted(active.items())[:4], start=1):
        fragments.append(f"w{index}:{run_name} {state}")
    _clear_status_line()
    print("  " + " | ".join(fragments)[:180], end="", flush=True)


def _event_state(event: dict[str, Any]) -> str:
    name = str(event.get("event", ""))
    step = event.get("step")
    if name == "run_start":
        return f"start {event.get('snapshots', '?')} snapshots"
    if name == "step_skipped":
        return f"skip step={step}"
    if name == "step_start":
        return f"step={step}"
    if name == "decode_start":
        return f"decode step={step}"
    if name == "decode_done":
        return f"decoded step={step} walks={event.get('walks', '?')}"
    if name == "step_done":
        return f"done step={step}"
    if name == "run_done":
        return f"complete {event.get('evaluated_steps', '?')} steps"
    return name


def _drain_progress_events(
    progress_queue: Any,
    *,
    active: dict[str, str],
    completed: int,
    total: int,
    last_done: str | None,
) -> None:
    while True:
        try:
            event = progress_queue.get_nowait()
        except queue.Empty:
            break
        run_name = str(event.get("run", "?"))
        if event.get("event") == "run_done":
            active.pop(run_name, None)
        else:
            active[run_name] = _event_state(event)
        _render_parallel_progress(active=active, completed=completed, total=total, last_done=last_done)


def _evaluate_runs_parallel(
    *,
    runs: list[Path],
    method: str,
    protocol: EvaluationProtocol,
    level: int,
    walk_data: dict[str, Any],
    gt: dict[str, Any],
    walk_cache_id: str,
    resume: bool,
    force: bool,
    snapshot_steps: set[int] | None,
    max_workers: int,
    log_dir: Path,
) -> None:
    effective_workers = max(1, min(int(max_workers), len(runs)))
    print(f"Evaluating {len(runs)} run(s) with {effective_workers} worker(s)")
    print(f"Worker logs: {log_dir}")

    manager = multiprocessing.Manager()
    progress_queue = manager.Queue()
    payloads = [
        {
            "run_root": str(run_root),
            "method": method,
            "protocol": protocol,
            "level": level,
            "walk_data": walk_data,
            "gt": gt,
            "walk_cache_id": walk_cache_id,
            "resume": resume,
            "force": force,
            "snapshot_steps": snapshot_steps,
            "log_dir": str(log_dir),
            "progress_queue": progress_queue,
        }
        for run_root in runs
    ]
    active: dict[str, str] = {}
    completed = 0
    last_done: str | None = None
    last_render = 0.0
    with ProcessPoolExecutor(max_workers=effective_workers) as pool:
        future_map = {pool.submit(_evaluate_run_worker, payload): payload for payload in payloads}
        pending = set(future_map)
        while pending:
            done, pending = wait(pending, timeout=0.5, return_when=FIRST_COMPLETED)
            _drain_progress_events(
                progress_queue,
                active=active,
                completed=completed,
                total=len(runs),
                last_done=last_done,
            )
            now = time.monotonic()
            if now - last_render >= 0.5:
                _render_parallel_progress(active=active, completed=completed, total=len(runs), last_done=last_done)
                last_render = now
            for future in done:
                result = future.result()
                completed += 1
                last_done = str(result["run"])
                active.pop(last_done, None)
                if not (sys.stdout.isatty() or os.environ.get("MERCURY_FORCE_TTY_PROGRESS") == "1"):
                    print(f"  done run={last_done} records={result['records']} log={result['log_path']}")
                _render_parallel_progress(active=active, completed=completed, total=len(runs), last_done=last_done)
        _drain_progress_events(
            progress_queue,
            active=active,
            completed=completed,
            total=len(runs),
            last_done=last_done,
        )
    if sys.stdout.isatty() or os.environ.get("MERCURY_FORCE_TTY_PROGRESS") == "1":
        _clear_status_line()
        print(f"  completed {completed}/{len(runs)} run(s)")


def evaluate_study(
    *,
    study_root: Path,
    level: int,
    method: str,
    protocol: EvaluationProtocol | None = None,
    resume: bool = True,
    force: bool = False,
    max_workers: int = 1,
    no_parallel: bool = False,
    log_dir: Path | None = None,
    run_rand_prob: float | None = None,
    snapshot_steps: set[int] | None = None,
    cscg_batch_size: int | None = None,
    cscg_clone_preset: str | None = None,
) -> None:
    """Top-level entry point: evaluate all runs in a study under the canonical protocol.

    Walk cache and empirical GT are shared across all runs.
    """
    study_root = Path(study_root)

    if protocol is None:
        protocol = EvaluationProtocol()

    n_true = int(exact_cartesian_reference_positions(level).shape[0])

    # Shared walk cache.
    walk_data = generate_and_cache_walks(
        study_root=study_root,
        protocol=protocol,
        level=level,
        force_regenerate=force,
    )

    # Determine n_actions from valid transitions.
    from ..infrastructure.paper_precision import (
        exact_valid_sensorimotor_transitions,
        infer_num_actions_from_valid_transitions,
    )
    valid_transitions = exact_valid_sensorimotor_transitions(level_index=level)
    n_actions = infer_num_actions_from_valid_transitions(valid_transitions)

    # Shared empirical ground truth.
    logger.info("Building empirical ground truth from cached walks...")
    gt = build_empirical_ground_truth(
        walk_data=walk_data,
        n_true=n_true,
        n_actions=n_actions,
        level=level,
    )
    walk_cache_id = protocol.walk_cache_key()

    logger.info(
        "Built empirical ground truth: %d true states, %d actions, %d valid (a,i) rows",
        n_true, n_actions, int(gt["valid_action_rows"].sum()),
    )
    print(
        f"Built empirical ground truth: {n_true} true states, {n_actions} actions, "
        f"{int(gt['valid_action_rows'].sum())} valid (a,i) rows",
        flush=True,
    )

    runs = discover_runs(study_root)
    if not runs:
        logger.warning("No runs found under %s/runs/", study_root)
        return

    selected_run_ids = manifest_run_ids_for_filters(
        study_root=study_root,
        method=method,
        level=level,
        rand_prob=run_rand_prob,
        cscg_batch_size=cscg_batch_size,
        cscg_clone_preset=cscg_clone_preset,
    )
    runs = filter_evaluation_runs(
        runs=runs,
        method=method,
        level=level,
        protocol=protocol,
        rand_prob=run_rand_prob,
        snapshot_steps=snapshot_steps,
        cscg_batch_size=cscg_batch_size,
        cscg_clone_preset=cscg_clone_preset,
        run_ids=selected_run_ids,
        resume=resume,
        force=force,
    )
    if not runs:
        logger.warning("No unevaluated runs matching level=%d method=%s found under %s", level, method, study_root)
        return

    logger.info("Found %d run(s) for level=%d method=%s", len(runs), level, method)
    print(f"Found {len(runs)} run(s) for level={level} method={method}", flush=True)

    if log_dir is None:
        log_dir = study_root / "evaluation_logs" / protocol.protocol_id() / f"level={level}_method={method}"

    if int(max_workers) > 1 and not bool(no_parallel):
        _evaluate_runs_parallel(
            runs=runs,
            method=method,
            protocol=protocol,
            level=level,
            walk_data=walk_data,
            gt=gt,
            walk_cache_id=walk_cache_id,
            resume=resume,
            force=force,
            snapshot_steps=snapshot_steps,
            max_workers=int(max_workers),
            log_dir=log_dir,
        )
        logger.info("Study evaluation complete: %s", study_root)
        return

    for run_root in runs:
        logger.info("Evaluating run: %s", run_root.name)
        print(f"Evaluating run: {run_root.name}", flush=True)
        evaluate_run_snapshots(
            run_root=run_root,
            method=method,
            protocol=protocol,
            level=level,
            walk_data=walk_data,
            gt=gt,
            walk_cache_id=walk_cache_id,
            resume=resume,
            force=force,
            snapshot_steps=snapshot_steps,
        )

    logger.info("Study evaluation complete: %s", study_root)
