from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from mercury_runs.algorithms.cscg.prepare import filter_valid_cscg_rows
from mercury_runs.infrastructure.paper_precision_snapshots import (
    load_mercury_paper_precision_snapshot,
)
from mercury_runs.infrastructure.paper_precision import (
    aggregate_aligned_weighted_adjacency,
    build_weighted_adjacency_from_valid_transitions,
    build_weighted_adjacency_from_walks,
    cartesian_state_ids,
    collapse_action_transition_tensor_to_adjacency,
    compute_alignment,
    compute_mean_tv,
    exact_cartesian_reference_positions,
    exact_cartesian_state_ids_for_level,
    exact_valid_sensorimotor_transitions,
    generate_random_start_walks,
    remap_position_walks_to_global_ids,
    row_normalize,
    zero_diagonal,
)
from mercury_runs.infrastructure.reporting import write_study_summary
from mercury_runs.infrastructure.runtime import atomic_write_json


METRIC_CONTEXT_KEYS = {
    "method",
    "series_label",
    "phase",
    "stage",
    "iteration",
    "step",
    "observed_samples",
    "phase_step",
    "cumulative_step",
    "handoff_step",
}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _paper_precision_settings(run_payload: dict[str, Any], protocol: dict[str, Any]) -> dict[str, Any]:
    paper = ((run_payload.get("evaluation") or {}).get("paper_precision") or {})
    structure = paper.get("structure_metrics") or {}
    return {
        "paper_precision_enabled": bool(paper.get("enabled", True)),
        "paper_precision_mode": str(protocol.get("mode", paper.get("mode", "per_iteration"))),
        "paper_precision_eval_interval": int(protocol.get("eval_interval", paper.get("eval_interval", 1))),
        "paper_precision_eval_checkpoints": protocol.get("eval_checkpoints", paper.get("eval_checkpoints")),
        "paper_precision_num_points": protocol.get("num_points", paper.get("num_points")),
        "paper_precision_schedule": str(paper.get("schedule", "linear")),
        "paper_precision_num_walks": int(protocol.get("num_walks", paper.get("num_walks", 100))),
        "paper_precision_walk_length": int(protocol.get("walk_length", paper.get("walk_length", 10_000))),
        "paper_precision_seed": int(protocol.get("base_seed", run_payload.get("seed", 0))),
        "paper_precision_rand_prob": float(protocol.get("rand_prob", run_payload.get("rand_prob", 0.3))),
        "structure_metrics_enabled": bool(structure.get("enabled", True)),
        "structure_metrics_epsilon": float(structure.get("epsilon", 1e-6)),
        "structure_metrics_epsilons": structure.get("thresholds"),
        "structure_metrics_ignore_self_loops": bool(structure.get("ignore_self_loops", True)),
        "structure_metrics_scope": str(structure.get("scope", "exact_level")),
        "structure_metrics_ground_truth_source": str(structure.get("ground_truth_source", "empirical_walks")),
    }


def _cscg_config(run_root: Path, run_payload: dict[str, Any], protocol: dict[str, Any]) -> SimpleNamespace:
    cscg = ((run_payload.get("baselines") or {}).get("cscg") or {})
    seed = cscg.get("seed")
    return SimpleNamespace(
        datasets_root=Path("datasets"),
        output_root=run_root / "bundles" / "cscg",
        run_id=run_root.name,
        level=int(run_payload["level"]),
        sensor=str(run_payload["sensor"]),
        sensor_range=run_payload.get("sensor_range"),
        ground_truth_states_parquet=run_root / "ground_truth" / f"{run_root.name}_ground_truth_states.parquet",
        window_length=int(run_payload.get("window_length", 100)),
        valid_trajectories_only=bool(cscg.get("valid_trajectories_only", run_payload.get("valid_trajectories_only", False))),
        clones_per_obs=int(cscg.get("clones_per_obs", 10)),
        clone_allocation=str(cscg.get("clone_allocation", "uniform")),
        clones_per_state=int(cscg.get("clones_per_state", 1)),
        n_iter=int(cscg.get("n_iter", 1000)),
        pseudocount=float(cscg.get("pseudocount", 0.002)),
        term_early=bool(cscg.get("term_early", True)),
        train_algorithm=str(cscg.get("train_algorithm", "em")),
        training_mode=str(cscg.get("training_mode", "full")),
        batch_size=int(cscg.get("batch_size", 1024)),
        online_lambda=float(cscg.get("online_lambda", 1.0)),
        eval_viterbi=bool(cscg.get("eval_viterbi", False)),
        seed=int(seed if seed is not None else run_payload.get("seed", 0)),
        **_paper_precision_settings(run_payload, protocol),
    )


def _hybrid_cscg_config(run_root: Path, run_payload: dict[str, Any], protocol: dict[str, Any]) -> SimpleNamespace:
    cscg = ((run_payload.get("baselines") or {}).get("cscg") or {})
    return SimpleNamespace(
        datasets_root=Path("datasets"),
        output_root=run_root / "bundles" / "mercury_cscg_hybrid",
        run_id=run_root.name,
        level=int(run_payload["level"]),
        sensor=str(run_payload["sensor"]),
        sensor_range=run_payload.get("sensor_range"),
        ground_truth_states_parquet=run_root / "ground_truth" / f"{run_root.name}_ground_truth_states.parquet",
        memory_length=int(run_payload.get("memory_length", 1)),
        window_length=int(run_payload.get("window_length", 100)),
        seed=int(run_payload.get("seed", 0)),
        rand_prob=float(run_payload.get("rand_prob", 0.3)),
        num_steps=int(run_payload.get("num_steps", 1)),
        valid_trajectories_only=bool(run_payload.get("valid_trajectories_only", False)),
        mercury_valid_trajectories_only=bool(run_payload.get("mercury_valid_trajectories_only", False)),
        mercury_split_sensory_raw_latent_valid=bool(run_payload.get("mercury_split_sensory_raw_latent_valid", False)),
        mercury_input_mode=str(run_payload.get("mercury_input_mode", "pretrained_raw")),
        sensory=dict(run_payload.get("sensory") or {}),
        latent=dict(run_payload.get("latent") or {}),
        action_map=dict(run_payload.get("action_map") or {}),
        n_iter=int(cscg.get("n_iter", 1000)),
        pseudocount=float(cscg.get("pseudocount", 0.002)),
        term_early=bool(cscg.get("term_early", True)),
        train_algorithm=str(cscg.get("train_algorithm", "em")),
        training_mode=str(cscg.get("training_mode", "full")),
        batch_size=int(cscg.get("batch_size", 1024)),
        online_lambda=float(cscg.get("online_lambda", 1.0)),
        mercury_prior_weight=float(cscg.get("mercury_cscg_prior_weight", 1.0)),
        eval_viterbi=bool(cscg.get("eval_viterbi", False)),
        **_paper_precision_settings(run_payload, protocol),
    )


def _mercury_config(run_root: Path, run_payload: dict[str, Any], protocol: dict[str, Any]) -> SimpleNamespace:
    settings = _paper_precision_settings(run_payload, protocol)
    return SimpleNamespace(
        datasets_root=Path("datasets"),
        results_root=run_root / "bundles" / "mercury",
        run_id=run_root.name,
        level=int(run_payload["level"]),
        sensor=str(run_payload["sensor"]),
        sensor_range=run_payload.get("sensor_range"),
        memory_length=int(run_payload.get("memory_length", 1)),
        window_length=int(run_payload.get("window_length", 100)),
        seed=int(protocol.get("base_seed", run_payload.get("seed", 0))),
        rand_prob=float(protocol.get("rand_prob", run_payload.get("rand_prob", 0.3))),
        num_steps=int(run_payload.get("num_steps", 1)),
        valid_trajectories_only=bool(run_payload.get("valid_trajectories_only", False)),
        mercury_valid_trajectories_only=bool(run_payload.get("mercury_valid_trajectories_only", False)),
        mercury_split_sensory_raw_latent_valid=bool(run_payload.get("mercury_split_sensory_raw_latent_valid", False)),
        mercury_input_mode=str(run_payload.get("mercury_input_mode", "pretrained_raw")),
        sensory=dict(run_payload.get("sensory") or {}),
        latent=dict(run_payload.get("latent") or {}),
        action_map=dict(run_payload.get("action_map") or {}),
        paper_precision_enabled=settings["paper_precision_enabled"],
        paper_precision_mode=settings["paper_precision_mode"],
        paper_precision_eval_interval=settings["paper_precision_eval_interval"],
        paper_precision_eval_checkpoints=settings["paper_precision_eval_checkpoints"],
        paper_precision_num_points=settings["paper_precision_num_points"],
        paper_precision_num_walks=settings["paper_precision_num_walks"],
        paper_precision_walk_length=settings["paper_precision_walk_length"],
        structure_metrics_enabled=settings["structure_metrics_enabled"],
        structure_metrics_epsilon=settings["structure_metrics_epsilon"],
        structure_metrics_epsilons=settings["structure_metrics_epsilons"],
        structure_metrics_ignore_self_loops=settings["structure_metrics_ignore_self_loops"],
        structure_metrics_scope=settings["structure_metrics_scope"],
        structure_metrics_ground_truth_source=settings["structure_metrics_ground_truth_source"],
    )


def _load_cscg_snapshot(snapshot_dir: Path) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    model = np.load(snapshot_dir / "model.npz", allow_pickle=False)
    if "obs_uniques" in model and "action_uniques" in model:
        obs_uniques = np.asarray(model["obs_uniques"])
        action_uniques = np.asarray(model["action_uniques"])
    else:
        vocab = np.load(snapshot_dir / "observation_vocabulary.npz", allow_pickle=False)
        obs_uniques = np.asarray(vocab["obs_uniques"])
        action_uniques = np.asarray(vocab["action_uniques"])
    n_clones = np.asarray(model["n_clones"], dtype=np.int64)
    model_npz = {"T": np.asarray(model["T"]), "n_clones": n_clones}
    return model_npz, n_clones, obs_uniques, action_uniques


def _structure_epsilons(config: SimpleNamespace) -> list[float]:
    epsilons = getattr(config, "structure_metrics_epsilons", None)
    if epsilons:
        return [float(value) for value in epsilons]
    return [float(getattr(config, "structure_metrics_epsilon", 1e-6))]


def _epsilon_key(epsilon: float) -> str:
    return f"{float(epsilon):.12g}"


def _true_structure(config: SimpleNamespace, *, cscg_filter: bool) -> tuple[list[np.ndarray], np.ndarray, int]:
    walks = generate_random_start_walks(
        level=config.level,
        sensor=config.sensor,
        sensor_range=config.sensor_range,
        rand_prob=float(getattr(config, "paper_precision_rand_prob", getattr(config, "rand_prob", 0.3))),
        num_walks=int(config.paper_precision_num_walks),
        walk_length=int(config.paper_precision_walk_length),
        base_seed=int(getattr(config, "paper_precision_seed", getattr(config, "seed", 0))),
        valid_trajectories_only=bool(getattr(config, "valid_trajectories_only", False)),
    )
    exact_walks: list[np.ndarray] = []
    position_walks: list[np.ndarray] = []
    for walk in walks:
        positions = np.asarray(walk.cartesian_observations)
        if cscg_filter:
            _, _, filtered_gt, _ = filter_valid_cscg_rows(
                observations=walk.observations,
                actions=walk.actions,
                collisions=np.asarray(walk.collisions, dtype=bool),
                ground_truth_bmu=positions,
                valid_trajectories_only=bool(getattr(config, "valid_trajectories_only", False)),
            )
            positions = np.asarray(filtered_gt)
        exact_walks.append(
            exact_cartesian_state_ids_for_level(
                level_index=int(config.level),
                positions=positions,
            ).astype(np.int64, copy=False)
        )
        position_walks.append(positions)

    scope = str(getattr(config, "structure_metrics_scope", "exact_level")).strip().lower()
    source = str(getattr(config, "structure_metrics_ground_truth_source", "empirical_walks")).strip().lower()
    if scope == "walk_local":
        true_walks = remap_position_walks_to_global_ids(position_walks)
        n_true = int(len(np.unique(np.concatenate(true_walks)))) if true_walks else 0
        if source == "maze_topology":
            exact_series = np.concatenate(exact_walks) if exact_walks else np.asarray([], dtype=np.int64)
            local_series = np.concatenate(true_walks) if true_walks else np.asarray([], dtype=np.int64)
            exact_to_local = {
                int(exact_state): int(local_state)
                for exact_state, local_state in zip(exact_series.tolist(), local_series.tolist(), strict=False)
            }
            W_true = build_weighted_adjacency_from_valid_transitions(
                exact_valid_sensorimotor_transitions(level_index=int(config.level)),
                n_states=n_true,
                state_mapping=exact_to_local,
            )
        else:
            W_true = build_weighted_adjacency_from_walks(true_walks, n_states=n_true)
        return true_walks, W_true, n_true

    n_true = int(exact_cartesian_reference_positions(int(config.level)).shape[0])
    if source == "maze_topology":
        W_true = build_weighted_adjacency_from_valid_transitions(
            exact_valid_sensorimotor_transitions(level_index=int(config.level)),
            n_states=n_true,
        )
    else:
        W_true = build_weighted_adjacency_from_walks(exact_walks, n_states=n_true)
    return exact_walks, W_true, n_true


def _tv_from_cooccurrence(
    *,
    cooccurrence: list[list[int]] | np.ndarray,
    W_hat: np.ndarray,
    W_true: np.ndarray,
    n_true: int,
    ignore_self_loops: bool,
) -> float:
    contingency = np.asarray(cooccurrence, dtype=np.int64)
    learned_size = int(np.asarray(W_hat).shape[0])
    if contingency.shape[0] < learned_size:
        padded = np.zeros((learned_size, contingency.shape[1]), dtype=contingency.dtype)
        padded[: contingency.shape[0], : contingency.shape[1]] = contingency
        contingency = padded
    alignment = compute_alignment(contingency)
    occupancy = contingency.sum(axis=1).astype(np.float64, copy=False)
    aligned = aggregate_aligned_weighted_adjacency(
        W_hat=np.asarray(W_hat, dtype=np.float64),
        A=alignment,
        mu=occupancy,
        n_true=int(n_true),
    )
    truth = np.asarray(W_true, dtype=np.float64)
    if ignore_self_loops:
        truth = zero_diagonal(truth)
        aligned = zero_diagonal(aligned)
    return compute_mean_tv(row_normalize(truth), row_normalize(aligned))


def _patch_tv_fields(row: dict[str, Any], prefix: str, tv: float, config: SimpleNamespace) -> None:
    key = f"{prefix}_mean_total_variation"
    row[key] = float(tv)
    by_epsilon = row.get("metrics_by_structure_epsilon")
    if isinstance(by_epsilon, dict):
        for epsilon in _structure_epsilons(config):
            bucket = by_epsilon.get(_epsilon_key(epsilon))
            if isinstance(bucket, dict):
                bucket[key] = float(tv)


def _recompute_cscg_payload(
    *,
    run_root: Path,
    payload_path: Path,
    snapshot_method: str,
    config: SimpleNamespace,
    dry_run: bool,
) -> int:
    payload = _load_json(payload_path)
    history = payload.get("history") if isinstance(payload.get("history"), list) else []
    changed_rows = 0
    _, W_true, n_true = _true_structure(config, cscg_filter=True)
    ignore_self_loops = bool(getattr(config, "structure_metrics_ignore_self_loops", True))
    snapshot_root = run_root / "snapshots" / snapshot_method / "paper_precision"
    new_history: list[dict[str, Any]] = []
    for row in history:
        if not isinstance(row, dict):
            new_history.append(row)
            continue
        if snapshot_method == "mercury_cscg_hybrid" and str(row.get("phase", "cscg")) != "cscg":
            new_history.append(row)
            continue
        step = int(row.get("step", row.get("observed_samples", 0)))
        snapshot_dir = snapshot_root / f"step={step}"
        if not snapshot_dir.exists():
            new_history.append(row)
            continue
        model_npz, n_clones, obs_uniques, action_uniques = _load_cscg_snapshot(snapshot_dir)
        del n_clones, obs_uniques, action_uniques
        tv = _tv_from_cooccurrence(
            cooccurrence=row["latent_exact_state_cooccurrence_matrix"],
            W_hat=collapse_action_transition_tensor_to_adjacency(np.asarray(model_npz["T"], dtype=np.float64)),
            W_true=W_true,
            n_true=n_true,
            ignore_self_loops=ignore_self_loops,
        )
        patched_row = dict(row)
        _patch_tv_fields(patched_row, "latent", tv, config)
        new_history.append(patched_row)
        changed_rows += 1
    if changed_rows:
        payload["history"] = new_history
        latest = next((row for row in reversed(new_history) if isinstance(row, dict) and "latent_mean_total_variation" in row), None)
        if latest is not None:
            _patch_tv_fields(payload["metrics"], "latent", float(latest["latent_mean_total_variation"]), config)
        if not dry_run:
            atomic_write_json(payload_path, payload)
    return changed_rows


def _recompute_mercury_payload(
    *,
    run_root: Path,
    run_payload: dict[str, Any],
    payload_path: Path,
    dry_run: bool,
) -> tuple[int, list[dict[str, Any]]]:
    if not payload_path.exists():
        return 0, []
    payload = _load_json(payload_path)
    protocol = payload.get("protocol") or {}
    config = _mercury_config(run_root, run_payload, protocol)
    snapshot_root = run_root / "bundles" / "snapshots" / "mercury" / "paper_precision"
    history = payload.get("history") if isinstance(payload.get("history"), list) else []
    changed_rows = 0
    _, W_true, n_true = _true_structure(config, cscg_filter=False)
    ignore_self_loops = bool(getattr(config, "structure_metrics_ignore_self_loops", True))
    new_history: list[dict[str, Any]] = []
    for row in history:
        if not isinstance(row, dict):
            new_history.append(row)
            continue
        step = int(row.get("step", row.get("observed_samples", 0)))
        snapshot_dir = snapshot_root / f"step={step}"
        if not snapshot_dir.exists():
            new_history.append(row)
            continue
        snapshot = load_mercury_paper_precision_snapshot(snapshot_dir)
        patched_row = dict(row)
        sensory_tv = _tv_from_cooccurrence(
            cooccurrence=row["sensory_exact_state_cooccurrence_matrix"],
            W_hat=np.asarray(snapshot["sensory_graph"].adj, dtype=np.float64),
            W_true=W_true,
            n_true=n_true,
            ignore_self_loops=ignore_self_loops,
        )
        latent_tv = _tv_from_cooccurrence(
            cooccurrence=row["latent_exact_state_cooccurrence_matrix"],
            W_hat=np.asarray(snapshot["latent_graph"].adj, dtype=np.float64),
            W_true=W_true,
            n_true=n_true,
            ignore_self_loops=ignore_self_loops,
        )
        _patch_tv_fields(patched_row, "sensory", sensory_tv, config)
        _patch_tv_fields(patched_row, "latent", latent_tv, config)
        new_history.append(patched_row)
        changed_rows += 1
    if changed_rows:
        payload["history"] = new_history
        latest = next((row for row in reversed(new_history) if isinstance(row, dict) and "latent_mean_total_variation" in row), None)
        if latest is not None:
            _patch_tv_fields(payload["metrics"], "sensory", float(latest["sensory_mean_total_variation"]), config)
            _patch_tv_fields(payload["metrics"], "latent", float(latest["latent_mean_total_variation"]), config)
        if not dry_run:
            atomic_write_json(payload_path, payload)
    return changed_rows, new_history


def _patch_hybrid_mercury_rows(run_root: Path, mercury_history: list[dict[str, Any]], dry_run: bool) -> int:
    hybrid_path = run_root / "metrics" / "mercury_cscg_hybrid" / f"{run_root.name}_paper_precision.json"
    if not hybrid_path.exists() or not mercury_history:
        return 0
    payload = _load_json(hybrid_path)
    history = payload.get("history") if isinstance(payload.get("history"), list) else []
    by_step = {int(row.get("step", row.get("observed_samples", -1))): row for row in mercury_history if isinstance(row, dict)}
    changed = 0
    patched: list[dict[str, Any]] = []
    for row in history:
        if not isinstance(row, dict) or str(row.get("phase")) != "mercury":
            patched.append(row)
            continue
        step = int(row.get("step", row.get("observed_samples", -1)))
        source = by_step.get(step)
        if source is None:
            patched.append(row)
            continue
        merged = dict(source)
        merged.update({key: row[key] for key in METRIC_CONTEXT_KEYS if key in row})
        patched.append(merged)
        changed += 1
    if changed and not dry_run:
        payload["history"] = patched
        atomic_write_json(hybrid_path, payload)
    return changed


def _run_dirs(study_dir: Path) -> list[Path]:
    return sorted(path for path in study_dir.iterdir() if path.is_dir() and (path / "run_config.json").exists())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--study-root", type=Path, default=Path("results/current/studies"))
    parser.add_argument("--study-name", default="mercury_cscg_hybrid_vs_cscg_batch_size_30_seed")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-summary", action="store_true")
    args = parser.parse_args()

    study_dir = args.study_root / args.study_name
    runs = _run_dirs(study_dir)
    if args.limit is not None:
        runs = runs[: int(args.limit)]

    stats = {"runs": 0, "cscg_rows": 0, "hybrid_cscg_rows": 0, "mercury_rows": 0, "hybrid_mercury_rows": 0}
    for index, run_root in enumerate(runs, start=1):
        run_config = _load_json(run_root / "run_config.json")
        run_payload = run_config.get("run_payload") or {}
        method = str(run_payload.get("method", ""))
        stats["runs"] += 1
        if method == "cscg":
            payload_path = run_root / "metrics" / "cscg" / f"{run_root.name}_paper_precision.json"
            if payload_path.exists():
                protocol = (_load_json(payload_path).get("protocol") or {})
                config = _cscg_config(run_root, run_payload, protocol)
                stats["cscg_rows"] += _recompute_cscg_payload(
                    run_root=run_root,
                    payload_path=payload_path,
                    snapshot_method="cscg",
                    config=config,
                    dry_run=bool(args.dry_run),
                )
        elif method == "mercury_cscg_hybrid":
            mercury_path = run_root / "bundles" / "metrics" / "mercury" / f"{run_root.name}_paper_precision.json"
            mercury_rows, mercury_history = _recompute_mercury_payload(
                run_root=run_root,
                run_payload=run_payload,
                payload_path=mercury_path,
                dry_run=bool(args.dry_run),
            )
            stats["mercury_rows"] += mercury_rows
            stats["hybrid_mercury_rows"] += _patch_hybrid_mercury_rows(run_root, mercury_history, bool(args.dry_run))
            payload_path = run_root / "metrics" / "mercury_cscg_hybrid" / f"{run_root.name}_paper_precision.json"
            if payload_path.exists():
                protocol = (_load_json(payload_path).get("protocol") or {})
                config = _hybrid_cscg_config(run_root, run_payload, protocol)
                stats["hybrid_cscg_rows"] += _recompute_cscg_payload(
                    run_root=run_root,
                    payload_path=payload_path,
                    snapshot_method="mercury_cscg_hybrid",
                    config=config,
                    dry_run=bool(args.dry_run),
                )
        if index % 25 == 0:
            print(f"processed {index}/{len(runs)} runs: {stats}", flush=True)

    if not args.dry_run and not args.skip_summary:
        summary_path = write_study_summary(study_root=args.study_root, study_name=args.study_name)
        print(f"wrote summary: {summary_path}")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
