from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from ...infrastructure.paper_precision import (
    action_ids_from_action_vectors,
    build_action_conditioned_tensor_from_graph,
    build_action_conditioned_tensor_from_valid_transitions,
    build_action_conditioned_tensor_from_walks,
    build_weighted_adjacency_from_valid_transitions,
    cartesian_state_ids,
    compute_action_conditioned_structure_metrics,
    compute_fixed_map_precision_details,
    compute_purity_from_cooccurrence,
    compute_sensorimotor_link_error,
    compute_weighted_structure_metrics,
    concatenate_walk_series,
    dominant_ground_truth_mapping_from_cooccurrence,
    exact_cartesian_state_ids_for_level,
    exact_cartesian_reference_positions,
    exact_valid_sensorimotor_transitions,
    generate_random_start_walks,
    infer_num_actions_from_valid_transitions,
    project_graph_edges_to_ground_truth,
    remap_position_walks_to_global_ids,
    write_paper_precision_artifact,
)
from mercury.action_map.adapter import ActionMap
from mercury.latent.state import LatentState, init_latent_state, latent_step_predict_only
from mercury.latent.params import LatentParams
from mercury.memory.state import add_memory, init_mem, update_memory
from mercury.sensory.state import SensoryState, init_global_context, sensory_step_predict_only
from mercury.sensory.params import SensoryParams
from ...infrastructure.reporting import summarize_mercury_native_metrics


def _action_codebook_size(action_map: ActionMap, *, fallback: int) -> int:
    codebook = getattr(getattr(action_map, "state", None), "codebook", None)
    if codebook is None:
        return int(fallback)
    return int(np.asarray(codebook).shape[0])


def _init_latent_state_compat(mem, *, n_actions: int):
    try:
        return init_latent_state(mem, n_actions=n_actions)
    except TypeError:
        return init_latent_state(mem)


def _supports_exact_action_semantics(*, config, action_map: ActionMap | None) -> bool:
    if bool((config.action_map or {}).get("identity_for_one_hot", False)):
        return True
    return bool(getattr(action_map, "identity_mapping", False))


def compute_mercury_precision(*, config) -> None:
    from scripts.analysis.compute_precision import compute_metrics

    compute_metrics(
        level=config.level,
        window_length=config.window_length,
        sensor=config.sensor,
        sensor_range=config.sensor_range,
        results_dir=Path(config.results_root),
        out_dir=Path(config.results_root).parents[1] / "metrics" / "mercury",
        use_run_id=config.run_id,
    )


def compute_mercury_paper_precision(
    *,
    config,
    sensory_graph: Any,
    latent_graph: Any,
    action_map: ActionMap,
    show_progress: bool = True,
) -> Path:
    final_metrics = compute_mercury_paper_precision_metrics(
        config=config,
        sensory_graph=sensory_graph,
        latent_graph=latent_graph,
        action_map=action_map,
        show_progress=show_progress,
        progress_desc="Mercury Paper Precision",
    )
    return write_mercury_paper_precision_payload(
        config=config,
        metrics=final_metrics,
        history=[],
        schedule_unit="final_only",
    )


def _run_mercury_snapshot_on_walk(
    *,
    walk,
    sensory_graph: Any,
    latent_graph: Any,
    action_map: ActionMap,
    sensory_params: SensoryParams,
    latent_params: LatentParams,
    memory_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    observation_dim = int(np.asarray(walk.observations).shape[1])
    sensory_state = SensoryState(
        gs=sensory_graph,
        global_context=init_global_context(observation_dim),
        mapping=np.arange(sensory_graph.n, dtype=np.int32),
    )
    mem = init_mem(sensory_graph.n, int(memory_length))
    latent_state = _init_latent_state_compat(
        mem,
        n_actions=_action_codebook_size(action_map, fallback=int(np.asarray(walk.actions).shape[1])),
    )
    latent_state.g = latent_graph
    latent_state.mapping = np.arange(latent_graph.n, dtype=np.int32)
    previous_latent_bmu = None
    state_memory: list[int] = []
    sensory_bmus: list[int] = []
    latent_bmus: list[int] = []
    for observation, action, collision in zip(walk.observations, walk.actions, walk.collisions, strict=False):
        action_vector = np.atleast_1d(action).astype(np.float32)
        action_bmu = int(action_map.predict(action=action_vector))
        sensory_state, activation_vector = sensory_step_predict_only(
            np.asarray(observation, dtype=np.float32),
            sensory_state,
            sensory_params,
        )
        sensory_bmus.append(int(sensory_state.prev_bmu))
        sensory_graph_size = int(sensory_state.gs.n)
        if mem.gs.n != sensory_graph_size * int(memory_length):
            mem = init_mem(sensory_graph_size, length=int(memory_length))
        # if previous_latent_bmu is None or not bool(collision):
        if True:
            mem = update_memory(mem)
            mem = add_memory(mem, activation_vector)
            latent_state, _, state_memory = latent_step_predict_only(
                mem,
                latent_state,
                int(action_bmu),
                latent_params,
                state_memory,
            )
        latent_bmus.append(int(latent_state.prev_bmu))
        previous_latent_bmu = latent_state.prev_bmu
    return (
        np.asarray(sensory_bmus, dtype=np.int64),
        np.asarray(latent_bmus, dtype=np.int64),
    )


def compute_mercury_paper_precision_metrics(
    *,
    config,
    sensory_graph: Any | None = None,
    latent_graph: Any | None = None,
    action_map: ActionMap | None = None,
    show_progress: bool = False,
    progress_desc: str | None = None,
    progress_callback: Any = None,
) -> dict[str, Any]:
    walks = generate_random_start_walks(
        level=config.level,
        sensor=config.sensor,
        sensor_range=config.sensor_range,
        # rand_prob=float(config.rand_prob),
        rand_prob=float(1.0),
        num_walks=int(config.paper_precision_num_walks),
        walk_length=int(config.paper_precision_walk_length),
        base_seed=int(config.seed),
        valid_trajectories_only=bool(getattr(config, "mercury_valid_trajectories_only", False)),
    )
    gt_walks: list[np.ndarray] = []
    gt_exact_walks: list[np.ndarray] = []
    gt_position_walks: list[np.ndarray] = []
    gt_action_walks: list[np.ndarray] = []
    sensory_walks: list[np.ndarray] = []
    latent_walks: list[np.ndarray] = []
    sensory_params = SensoryParams(**config.sensory)
    latent_params = LatentParams(**config.latent)
    walk_iterator = walks
    if show_progress:
        walk_iterator = tqdm(walks, total=len(walks), desc=(progress_desc or "Mercury Paper Precision"), position=0)
    total_walks = len(walks)
    for walk_index, walk in enumerate(walk_iterator, start=1):
        ground_truth_state_ids = cartesian_state_ids(walk.cartesian_observations)
        exact_ground_truth_state_ids = exact_cartesian_state_ids_for_level(
            level_index=int(config.level),
            positions=walk.cartesian_observations,
        )
        if sensory_graph is None or latent_graph is None or action_map is None:
            raise RuntimeError("Mercury paper precision requires a frozen sensory graph, latent graph, and action map.")
        sensory_bmu, latent_bmu = _run_mercury_snapshot_on_walk(
            walk=walk,
            sensory_graph=deepcopy(sensory_graph),
            latent_graph=deepcopy(latent_graph),
            action_map=deepcopy(action_map),
            sensory_params=sensory_params,
            latent_params=latent_params,
            memory_length=int(config.memory_length),
        )
        length = min(len(ground_truth_state_ids), len(sensory_bmu), len(latent_bmu))
        gt_walks.append(ground_truth_state_ids[:length].astype(np.int64, copy=False))
        gt_exact_walks.append(exact_ground_truth_state_ids[:length].astype(np.int64, copy=False))
        gt_position_walks.append(np.asarray(walk.cartesian_observations)[:length])
        gt_action_walks.append(action_ids_from_action_vectors(np.asarray(walk.actions))[:length])
        sensory_walks.append(sensory_bmu[:length].astype(np.int64, copy=False))
        latent_walks.append(latent_bmu[:length].astype(np.int64, copy=False))
        if progress_callback is not None and (walk_index == 1 or walk_index == total_walks or walk_index % max(1, total_walks // 20) == 0):
            progress_callback(current=int(walk_index), total=int(total_walks), message="Paper precision walks")
    gt_series = concatenate_walk_series(gt_walks)
    gt_exact_series = concatenate_walk_series(gt_exact_walks)
    sensory_series = concatenate_walk_series(sensory_walks)
    latent_series = concatenate_walk_series(latent_walks)
    sensory_details = compute_fixed_map_precision_details(
        inferred_states=sensory_series,
        ground_truth_states=gt_series,
        metric_name="sensory_precision",
    )
    latent_details = compute_fixed_map_precision_details(
        inferred_states=latent_series,
        ground_truth_states=gt_series,
        metric_name="latent_precision",
    )
    sensory_exact_details = compute_fixed_map_precision_details(
        inferred_states=sensory_series,
        ground_truth_states=gt_exact_series,
        metric_name="sensory_exact_precision",
    )
    latent_exact_details = compute_fixed_map_precision_details(
        inferred_states=latent_series,
        ground_truth_states=gt_exact_series,
        metric_name="latent_exact_precision",
    )
    metrics: dict[str, Any] = {
        "sensory_precision": sensory_details["sensory_precision"],
        "sensory_evaluation_steps": sensory_details["evaluation_steps"],
        "sensory_inferred_state_count": sensory_details["inferred_state_count"],
        "sensory_ground_truth_state_count": sensory_details["ground_truth_state_count"],
        "sensory_cooccurrence_matrix": sensory_details["cooccurrence_matrix"],
        "latent_precision": latent_details["latent_precision"],
        "latent_evaluation_steps": latent_details["evaluation_steps"],
        "latent_inferred_state_count": latent_details["inferred_state_count"],
        "latent_ground_truth_state_count": latent_details["ground_truth_state_count"],
        "latent_cooccurrence_matrix": latent_details["cooccurrence_matrix"],
    }
    metrics["sensory_purity"] = compute_purity_from_cooccurrence(sensory_exact_details["cooccurrence_matrix"])
    metrics["latent_purity"] = compute_purity_from_cooccurrence(latent_exact_details["cooccurrence_matrix"])
    metrics["sensory_exact_state_cooccurrence_matrix"] = sensory_exact_details["cooccurrence_matrix"]
    metrics["latent_exact_state_cooccurrence_matrix"] = latent_exact_details["cooccurrence_matrix"]
    if bool(getattr(config, "structure_metrics_enabled", True)):
        structure_scope = str(getattr(config, "structure_metrics_scope", "exact_level")).strip().lower()
        structure_source = str(getattr(config, "structure_metrics_ground_truth_source", "empirical_walks")).strip().lower()
        structure_epsilon = float(getattr(config, "structure_metrics_epsilon", 1e-6))
        if structure_scope == "walk_local":
            true_structure_walks = remap_position_walks_to_global_ids(gt_position_walks)
            n_true_states = int(len(np.unique(concatenate_walk_series(true_structure_walks)))) if true_structure_walks else 0
            if structure_source == "maze_topology":
                exact_series = concatenate_walk_series(gt_exact_walks)
                walk_local_series = concatenate_walk_series(true_structure_walks)
                exact_to_local = {
                    int(exact_state): int(local_state)
                    for exact_state, local_state in zip(exact_series.tolist(), walk_local_series.tolist(), strict=False)
                }
                W_true = build_weighted_adjacency_from_valid_transitions(
                    exact_valid_sensorimotor_transitions(level_index=int(config.level)),
                    n_states=n_true_states,
                    state_mapping=exact_to_local,
                )
            else:
                W_true = None
        else:
            true_structure_walks = gt_exact_walks
            n_true_states = int(exact_cartesian_reference_positions(int(config.level)).shape[0])
            W_true = (
                build_weighted_adjacency_from_valid_transitions(
                    exact_valid_sensorimotor_transitions(level_index=int(config.level)),
                    n_states=n_true_states,
                )
                if structure_source == "maze_topology"
                else None
            )
        sensory_structure_metrics = compute_weighted_structure_metrics(
            decoded_walks=sensory_walks,
            true_walks=true_structure_walks,
            W_hat=np.asarray(sensory_graph.adj, dtype=np.float64),
            n_true=n_true_states,
            W_true=W_true,
            epsilon=structure_epsilon,
        )
        latent_structure_metrics = compute_weighted_structure_metrics(
            decoded_walks=latent_walks,
            true_walks=true_structure_walks,
            W_hat=np.asarray(latent_graph.adj, dtype=np.float64),
            n_true=n_true_states,
            W_true=W_true,
            epsilon=structure_epsilon,
        )
        metrics["sensory_mean_total_variation"] = sensory_structure_metrics["mean_total_variation"]
        metrics["sensory_edge_precision"] = sensory_structure_metrics["edge_precision"]
        metrics["sensory_edge_recall"] = sensory_structure_metrics["edge_recall"]
        metrics["sensory_edge_f1"] = sensory_structure_metrics["edge_f1"]
        metrics["latent_mean_total_variation"] = latent_structure_metrics["mean_total_variation"]
        metrics["latent_edge_precision"] = latent_structure_metrics["edge_precision"]
        metrics["latent_edge_recall"] = latent_structure_metrics["edge_recall"]
        metrics["latent_edge_f1"] = latent_structure_metrics["edge_f1"]

    if _supports_exact_action_semantics(config=config, action_map=action_map):
        valid_transitions = exact_valid_sensorimotor_transitions(level_index=int(config.level))
        n_actions = max(infer_num_actions_from_valid_transitions(valid_transitions), _action_codebook_size(action_map, fallback=4))
        if bool(getattr(config, "structure_metrics_enabled", True)):
            if structure_scope == "walk_local":
                if structure_source == "maze_topology":
                    T_true = build_action_conditioned_tensor_from_valid_transitions(
                        valid_transitions,
                        n_states=n_true_states,
                        n_actions=n_actions,
                        state_mapping=exact_to_local,
                    )
                else:
                    T_true = build_action_conditioned_tensor_from_walks(
                        state_walks=true_structure_walks,
                        action_walks=gt_action_walks,
                        n_states=n_true_states,
                        n_actions=n_actions,
                    )
            else:
                T_true = (
                    build_action_conditioned_tensor_from_valid_transitions(
                        valid_transitions,
                        n_states=n_true_states,
                        n_actions=n_actions,
                    )
                    if structure_source == "maze_topology"
                    else build_action_conditioned_tensor_from_walks(
                        state_walks=true_structure_walks,
                        action_walks=gt_action_walks,
                        n_states=n_true_states,
                        n_actions=n_actions,
                    )
                )
            sensory_action_metrics = compute_action_conditioned_structure_metrics(
                decoded_walks=sensory_walks,
                true_walks=true_structure_walks,
                action_walks=gt_action_walks,
                T_hat=build_action_conditioned_tensor_from_graph(
                    sensory_graph,
                    n_actions=n_actions,
                    epsilon=structure_epsilon,
                ),
                n_true=n_true_states,
                T_true=T_true,
                epsilon=structure_epsilon,
            )
            latent_action_metrics = compute_action_conditioned_structure_metrics(
                decoded_walks=latent_walks,
                true_walks=true_structure_walks,
                action_walks=gt_action_walks,
                T_hat=build_action_conditioned_tensor_from_graph(
                    latent_graph,
                    n_actions=n_actions,
                    epsilon=structure_epsilon,
                ),
                n_true=n_true_states,
                T_true=T_true,
                epsilon=structure_epsilon,
            )
            metrics["sensory_action_conditioned_mean_total_variation"] = sensory_action_metrics["mean_total_variation"]
            metrics["sensory_action_conditioned_edge_precision"] = sensory_action_metrics["edge_precision"]
            metrics["sensory_action_conditioned_edge_recall"] = sensory_action_metrics["edge_recall"]
            metrics["sensory_action_conditioned_edge_f1"] = sensory_action_metrics["edge_f1"]
            metrics["latent_action_conditioned_mean_total_variation"] = latent_action_metrics["mean_total_variation"]
            metrics["latent_action_conditioned_edge_precision"] = latent_action_metrics["edge_precision"]
            metrics["latent_action_conditioned_edge_recall"] = latent_action_metrics["edge_recall"]
            metrics["latent_action_conditioned_edge_f1"] = latent_action_metrics["edge_f1"]
        sensory_mapping = dominant_ground_truth_mapping_from_cooccurrence(sensory_exact_details["cooccurrence_matrix"])
        latent_mapping = dominant_ground_truth_mapping_from_cooccurrence(latent_exact_details["cooccurrence_matrix"])
        sensory_edges = project_graph_edges_to_ground_truth(
            graph=sensory_graph,
            node_to_ground_truth=sensory_mapping,
        )
        latent_edges = project_graph_edges_to_ground_truth(
            graph=latent_graph,
            node_to_ground_truth=latent_mapping,
        )
        metrics["sensory_sensorimotor_link_error"] = compute_sensorimotor_link_error(
            edges=sensory_edges,
            valid_transitions=valid_transitions,
        )
        metrics["latent_sensorimotor_link_error"] = compute_sensorimotor_link_error(
            edges=latent_edges,
            valid_transitions=valid_transitions,
        )
    else:
        reason = "action_map_does_not_preserve_true_action_ids"
        metrics["sensory_action_conditioned_mean_total_variation"] = None
        metrics["sensory_action_conditioned_edge_precision"] = None
        metrics["sensory_action_conditioned_edge_recall"] = None
        metrics["sensory_action_conditioned_edge_f1"] = None
        metrics["latent_action_conditioned_mean_total_variation"] = None
        metrics["latent_action_conditioned_edge_precision"] = None
        metrics["latent_action_conditioned_edge_recall"] = None
        metrics["latent_action_conditioned_edge_f1"] = None
        metrics["sensory_action_conditioned_mean_total_variation_reason"] = reason
        metrics["sensory_action_conditioned_edge_precision_reason"] = reason
        metrics["sensory_action_conditioned_edge_recall_reason"] = reason
        metrics["sensory_action_conditioned_edge_f1_reason"] = reason
        metrics["latent_action_conditioned_mean_total_variation_reason"] = reason
        metrics["latent_action_conditioned_edge_precision_reason"] = reason
        metrics["latent_action_conditioned_edge_recall_reason"] = reason
        metrics["latent_action_conditioned_edge_f1_reason"] = reason
        metrics["sensory_sensorimotor_link_error"] = None
        metrics["latent_sensorimotor_link_error"] = None
        metrics["sensory_sensorimotor_link_error_reason"] = reason
        metrics["latent_sensorimotor_link_error_reason"] = reason
    return metrics


def write_mercury_paper_precision_payload(
    *,
    config,
    metrics: dict[str, Any],
    history: list[dict[str, Any]],
    schedule_unit: str,
) -> Path:
    payload = {
        "method": "mercury",
        "schema": "paper_precision_v1",
        "protocol": {
            "num_walks": int(config.paper_precision_num_walks),
            "walk_length": int(config.paper_precision_walk_length),
            "rand_prob": float(config.rand_prob),
            "base_seed": int(config.seed),
            "valid_trajectories_only": bool(getattr(config, "mercury_valid_trajectories_only", False)),
            "ground_truth_state_series": "cartesian_state_id",
            "mode": str(config.paper_precision_mode),
            "eval_interval": int(config.paper_precision_eval_interval),
            "num_points": config.paper_precision_num_points,
            "schedule_unit": schedule_unit,
            "x_axis_unit": "observed_samples",
        },
        "metrics": metrics,
        "history": history,
    }
    out_path = Path(config.results_root).parents[1] / "metrics" / "mercury" / f"{config.run_id}_paper_precision.json"
    return write_paper_precision_artifact(path=out_path, payload=payload)


def summarize_mercury_run(states_parquet: Path) -> dict:
    return summarize_mercury_native_metrics(states_parquet)
