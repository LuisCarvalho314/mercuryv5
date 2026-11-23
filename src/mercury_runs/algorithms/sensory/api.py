from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

from mercury.action_map.adapter import ActionMap
from mercury.sensory.params import SensoryParams
from mercury.sensory.state import SensoryState, init_global_context, init_state, sensory_step, sensory_step_predict_only

from ...infrastructure.observation_encoding import (
    encode_observations_with_vocabulary,
    fit_cscg_one_hot_observations,
)
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
    resolve_structure_metric_epsilons,
    structure_epsilon_key,
    write_paper_precision_artifact,
)
from ...infrastructure.paper_precision import resolve_eval_checkpoints
from ...infrastructure.paper_precision_snapshots import write_sensory_paper_precision_snapshot
from ...io_parquet import LoadedDataset, ParquetConfig, load_level_parquet
from ...pipelines import _action_dim, _build_action_map, filter_mercury_dataset, params_to_dict, utc_timestamp
from ...save_results import write_bundle_parquet
from ...schemas_results import ResultBundleMeta, SourceDatasetRef
from ..mercury.evaluate import _action_codebook_size, _supports_exact_action_semantics
from .config import SensoryConfig


def _results_output_dir(root: Path, level: int, sensor: str, sensor_range: int | None) -> Path:
    base = root / f"level={level}" / f"sensor={sensor}"
    if sensor == "cardinal distance":
        return base / f"range={sensor_range}"
    return base


def _one_hot_dataset(dataset: LoadedDataset, obs_uniques: np.ndarray) -> LoadedDataset:
    encoded = encode_observations_with_vocabulary(
        observations=np.asarray(dataset.observations),
        obs_uniques=obs_uniques,
    )
    return LoadedDataset(
        observations=encoded.observations,
        actions=np.asarray(dataset.actions),
        collisions=np.asarray(dataset.collisions),
        source_metadata={**dict(dataset.source_metadata), "observation_mode": "cscg_one_hot"},
        parquet_path=dataset.parquet_path,
    )


def _load_training_dataset(config: SensoryConfig) -> tuple[LoadedDataset, np.ndarray | None]:
    dataset = load_level_parquet(
        ParquetConfig(
            root=Path(config.datasets_root),
            level=config.level,
            sensor=config.sensor,
            sensor_range=config.sensor_range,
            select=config.dataset_select,
            run_id=config.dataset_run_id,
        )
    )
    if bool(config.valid_trajectories_only):
        dataset = filter_mercury_dataset(dataset=dataset, valid_trajectories_only=True)
    mode = str(config.observation_mode).strip().lower()
    if mode == "raw":
        return dataset, None
    if mode == "cscg_one_hot":
        fitted = fit_cscg_one_hot_observations(np.asarray(dataset.observations))
        return _one_hot_dataset(dataset, fitted.obs_uniques), fitted.obs_uniques
    raise ValueError(f"Unsupported sensory observation_mode: {config.observation_mode!r}")


def _run_sensory_on_walk(
    *,
    observations: np.ndarray,
    actions: np.ndarray,
    sensory_graph: Any,
    action_map: ActionMap,
    sensory_params: SensoryParams,
    obs_uniques: np.ndarray | None,
) -> np.ndarray:
    eval_observations = np.asarray(observations)
    if obs_uniques is not None:
        eval_observations = encode_observations_with_vocabulary(
            observations=eval_observations,
            obs_uniques=obs_uniques,
        ).observations
    state = SensoryState(
        gs=sensory_graph,
        global_context=init_global_context(int(eval_observations.shape[1])),
        mapping=np.arange(sensory_graph.n, dtype=np.int32),
    )
    bmus: list[int] = []
    for observation, action in zip(eval_observations, actions, strict=False):
        if hasattr(action_map, "predict"):
            action_map.predict(action=np.atleast_1d(action).astype(np.float32))
        state, _ = sensory_step_predict_only(np.asarray(observation, dtype=np.float32), state, sensory_params)
        bmus.append(int(state.prev_bmu))
    return np.asarray(bmus, dtype=np.int64)


def compute_sensory_paper_precision_metrics(
    *,
    config: SensoryConfig,
    sensory_graph: Any,
    action_map: ActionMap,
    obs_uniques: np.ndarray | None = None,
) -> dict[str, Any]:
    walks = generate_random_start_walks(
        level=config.level,
        sensor=config.sensor,
        sensor_range=config.sensor_range,
        rand_prob=float(config.rand_prob),
        num_walks=int(config.paper_precision_num_walks),
        walk_length=int(config.paper_precision_walk_length),
        base_seed=int(config.seed),
        valid_trajectories_only=bool(config.valid_trajectories_only),
    )
    latent_walks: list[np.ndarray] = []
    gt_walks: list[np.ndarray] = []
    gt_exact_walks: list[np.ndarray] = []
    gt_position_walks: list[np.ndarray] = []
    gt_action_walks: list[np.ndarray] = []
    unsupported_walks = 0
    params = SensoryParams(**config.sensory)
    for walk in walks:
        try:
            sensory_bmu = _run_sensory_on_walk(
                observations=np.asarray(walk.observations),
                actions=np.asarray(walk.actions),
                sensory_graph=deepcopy(sensory_graph),
                action_map=deepcopy(action_map),
                sensory_params=params,
                obs_uniques=obs_uniques,
            )
        except ValueError:
            unsupported_walks += 1
            continue
        gt = cartesian_state_ids(walk.cartesian_observations)
        gt_exact = exact_cartesian_state_ids_for_level(level_index=int(config.level), positions=walk.cartesian_observations)
        length = min(len(gt), len(sensory_bmu))
        latent_walks.append(sensory_bmu[:length].astype(np.int64, copy=False))
        gt_walks.append(gt[:length].astype(np.int64, copy=False))
        gt_exact_walks.append(gt_exact[:length].astype(np.int64, copy=False))
        gt_position_walks.append(np.asarray(walk.cartesian_observations)[:length])
        gt_action_walks.append(action_ids_from_action_vectors(np.asarray(walk.actions))[:length])
    if not latent_walks:
        return {
            "latent_precision": 0.0,
            "latent_recall": 0.0,
            "latent_evaluation_steps": 0,
            "unsupported_walks": int(unsupported_walks),
            "decoded_walks": 0,
        }
    latent_series = concatenate_walk_series(latent_walks)
    gt_series = concatenate_walk_series(gt_walks)
    gt_exact_series = concatenate_walk_series(gt_exact_walks)
    details = compute_fixed_map_precision_details(
        inferred_states=latent_series,
        ground_truth_states=gt_series,
        metric_name="latent_precision",
    )
    exact_details = compute_fixed_map_precision_details(
        inferred_states=latent_series,
        ground_truth_states=gt_exact_series,
        metric_name="latent_exact_precision",
    )
    metrics: dict[str, Any] = {
        "latent_precision": details["latent_precision"],
        "latent_recall": details["latent_recall"],
        "latent_evaluation_steps": details["evaluation_steps"],
        "latent_inferred_state_count": details["inferred_state_count"],
        "latent_ground_truth_state_count": details["ground_truth_state_count"],
        "latent_cooccurrence_matrix": details["cooccurrence_matrix"],
        "latent_purity": compute_purity_from_cooccurrence(exact_details["cooccurrence_matrix"]),
        "latent_exact_state_cooccurrence_matrix": exact_details["cooccurrence_matrix"],
        "unsupported_walks": int(unsupported_walks),
        "decoded_walks": int(len(latent_walks)),
    }
    structure_scope = str(getattr(config, "structure_metrics_scope", "exact_level")).strip().lower()
    structure_source = str(getattr(config, "structure_metrics_ground_truth_source", "empirical_walks")).strip().lower()
    structure_epsilons = resolve_structure_metric_epsilons(config)
    structure_epsilon = float(structure_epsilons[0])
    ignore_self_loops = bool(getattr(config, "structure_metrics_ignore_self_loops", True))
    true_structure_walks: list[np.ndarray] = []
    n_true_states = 0
    W_true = None
    exact_to_local: dict[int, int] | None = None
    if bool(getattr(config, "structure_metrics_enabled", True)):
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
        metrics_by_epsilon: dict[str, Any] = {}
        graph_adjacency = np.asarray(sensory_graph.adj, dtype=np.float64)
        for epsilon in structure_epsilons:
            structure_metrics = compute_weighted_structure_metrics(
                decoded_walks=latent_walks,
                true_walks=true_structure_walks,
                W_hat=graph_adjacency,
                n_true=n_true_states,
                W_true=W_true,
                epsilon=float(epsilon),
                ignore_self_loops=ignore_self_loops,
            )
            metrics_by_epsilon[structure_epsilon_key(float(epsilon))] = {
                "latent_mean_total_variation": structure_metrics["mean_total_variation"],
                "latent_edge_precision": structure_metrics["edge_precision"],
                "latent_edge_recall": structure_metrics["edge_recall"],
                "latent_edge_f1": structure_metrics["edge_f1"],
            }
        primary_structure_metrics = metrics_by_epsilon[structure_epsilon_key(structure_epsilon)]
        metrics["latent_mean_total_variation"] = primary_structure_metrics["latent_mean_total_variation"]
        metrics["latent_edge_precision"] = primary_structure_metrics["latent_edge_precision"]
        metrics["latent_edge_recall"] = primary_structure_metrics["latent_edge_recall"]
        metrics["latent_edge_f1"] = primary_structure_metrics["latent_edge_f1"]
        metrics["metrics_by_structure_epsilon"] = metrics_by_epsilon

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
            for epsilon in structure_epsilons:
                action_metrics = compute_action_conditioned_structure_metrics(
                    decoded_walks=latent_walks,
                    true_walks=true_structure_walks,
                    action_walks=gt_action_walks,
                    T_hat=build_action_conditioned_tensor_from_graph(
                        sensory_graph,
                        n_actions=n_actions,
                        epsilon=float(epsilon),
                    ),
                    n_true=n_true_states,
                    T_true=T_true,
                    epsilon=float(epsilon),
                    ignore_self_loops=ignore_self_loops,
                )
                threshold_metrics = metrics["metrics_by_structure_epsilon"][structure_epsilon_key(float(epsilon))]
                threshold_metrics["latent_action_conditioned_mean_total_variation"] = action_metrics["mean_total_variation"]
                threshold_metrics["latent_action_conditioned_edge_precision"] = action_metrics["edge_precision"]
                threshold_metrics["latent_action_conditioned_edge_recall"] = action_metrics["edge_recall"]
                threshold_metrics["latent_action_conditioned_edge_f1"] = action_metrics["edge_f1"]
            primary_action_metrics = metrics["metrics_by_structure_epsilon"][structure_epsilon_key(structure_epsilon)]
            metrics["latent_action_conditioned_mean_total_variation"] = primary_action_metrics["latent_action_conditioned_mean_total_variation"]
            metrics["latent_action_conditioned_edge_precision"] = primary_action_metrics["latent_action_conditioned_edge_precision"]
            metrics["latent_action_conditioned_edge_recall"] = primary_action_metrics["latent_action_conditioned_edge_recall"]
            metrics["latent_action_conditioned_edge_f1"] = primary_action_metrics["latent_action_conditioned_edge_f1"]
        latent_mapping = dominant_ground_truth_mapping_from_cooccurrence(exact_details["cooccurrence_matrix"])
        latent_edges = project_graph_edges_to_ground_truth(
            graph=sensory_graph,
            node_to_ground_truth=latent_mapping,
        )
        metrics["latent_sensorimotor_link_error"] = compute_sensorimotor_link_error(
            edges=latent_edges,
            valid_transitions=valid_transitions,
        )
    else:
        reason = "action_map_does_not_preserve_true_action_ids"
        metrics["latent_action_conditioned_mean_total_variation"] = None
        metrics["latent_action_conditioned_edge_precision"] = None
        metrics["latent_action_conditioned_edge_recall"] = None
        metrics["latent_action_conditioned_edge_f1"] = None
        metrics["latent_action_conditioned_mean_total_variation_reason"] = reason
        metrics["latent_action_conditioned_edge_precision_reason"] = reason
        metrics["latent_action_conditioned_edge_recall_reason"] = reason
        metrics["latent_action_conditioned_edge_f1_reason"] = reason
        metrics["latent_sensorimotor_link_error"] = None
        metrics["latent_sensorimotor_link_error_reason"] = reason
    return metrics


def write_sensory_paper_precision_payload(
    *,
    config: SensoryConfig,
    metrics: dict[str, Any],
    history: list[dict[str, Any]],
    schedule_unit: str,
) -> Path:
    payload = {
        "method": "sensory",
        "schema": "paper_precision_v1",
        "protocol": {
            "num_walks": int(config.paper_precision_num_walks),
            "walk_length": int(config.paper_precision_walk_length),
            "rand_prob": float(config.rand_prob),
            "base_seed": int(config.seed),
            "valid_trajectories_only": bool(config.valid_trajectories_only),
            "mode": str(config.paper_precision_mode),
            "eval_interval": int(config.paper_precision_eval_interval),
            "eval_checkpoints": config.paper_precision_eval_checkpoints,
            "num_points": config.paper_precision_num_points,
            "schedule_unit": schedule_unit,
            "observation_mode": str(config.observation_mode),
        },
        "metrics": metrics,
        "history": history,
    }
    out_path = Path(config.output_root).parents[1] / "metrics" / "sensory" / f"{config.run_id}_paper_precision.json"
    return write_paper_precision_artifact(path=out_path, payload=payload)


def run_sensory(config: SensoryConfig, *, progress_callback=None) -> Path:
    dataset, obs_uniques = _load_training_dataset(config)
    observations = np.asarray(dataset.observations)
    actions = np.asarray(dataset.actions)
    collisions = np.asarray(dataset.collisions)
    sensory_params = SensoryParams(**config.sensory)
    sensory_state = init_state(int(observations.shape[1]))
    action_map = _build_action_map(_action_dim(actions), dict(config.action_map))
    checkpoints = set(
        resolve_eval_checkpoints(
            total_units=int(observations.shape[0]),
            schedule=config.paper_precision_schedule,
            num_points=config.paper_precision_num_points,
            eval_interval=int(config.paper_precision_eval_interval),
            explicit_checkpoints=config.paper_precision_eval_checkpoints,
        )
    )
    bmus: list[int] = []
    for index, (observation, action, collision) in enumerate(zip(observations, actions, collisions, strict=False), start=1):
        action_bmu, _ = action_map.step(np.atleast_1d(action).astype(np.float32))
        sensory_state = sensory_step(np.asarray(observation, dtype=np.float32), int(action_bmu), sensory_state, sensory_params, action_map)
        bmus.append(int(sensory_state.prev_bmu))
        if progress_callback is not None and (index == 1 or index == int(observations.shape[0]) or index % max(1, int(observations.shape[0]) // 100) == 0):
            progress_callback(stage="train", current=index, total=int(observations.shape[0]), message="Sensory pass", extra={"sensory_nodes": int(sensory_state.gs.n)})
        if bool(config.paper_precision_enabled) and str(config.paper_precision_mode).strip().lower() == "per_iteration" and index in checkpoints:
            write_sensory_paper_precision_snapshot(
                config=config,
                step=index,
                observed_samples=index,
                sensory_graph=sensory_state.gs,
                action_map=action_map,
                obs_uniques=obs_uniques,
            )
    output_dir = _results_output_dir(Path(config.output_root), config.level, config.sensor, config.sensor_range)
    output_dir.mkdir(parents=True, exist_ok=True)
    sensory_state.gs.to_npz(str(output_dir / f"{config.run_id}_sensory_graph.npz"))
    action_map.to_npz(str(output_dir / f"{config.run_id}_action_map.npz"))
    if obs_uniques is not None:
        np.savez_compressed(output_dir / f"{config.run_id}_observation_vocabulary.npz", obs_uniques=np.asarray(obs_uniques))
    if bool(config.paper_precision_enabled):
        if str(config.paper_precision_mode).strip().lower() == "final":
            write_sensory_paper_precision_snapshot(
                config=config,
                step=int(observations.shape[0]),
                observed_samples=int(observations.shape[0]),
                sensory_graph=sensory_state.gs,
                action_map=action_map,
                obs_uniques=obs_uniques,
            )
    meta = ResultBundleMeta(
        run_id=config.run_id,
        timestamp_utc=utc_timestamp(),
        source=SourceDatasetRef(
            level=config.level,
            sensor=config.sensor,
            sensor_range=config.sensor_range,
            select=config.dataset_select if config.dataset_select in {"latest", "run_id"} else "latest",
            dataset_run_id=config.dataset_run_id,
            dataset_parquet_name=dataset.parquet_path.name,
        ),
        sensory_params=params_to_dict(sensory_params),
        latent_params=None,
        action_map_params={"dim": _action_dim(actions), **dict(config.action_map)},
        memory_length=None,
        run_parameters={
            **dict(config.run_parameters),
            "sensory_eval": {
                "observation_mode": str(config.observation_mode),
                "steps": int(len(bmus)),
                "num_sensory_nodes": int(sensory_state.gs.n),
            },
            "observation_vocabulary": (obs_uniques.tolist() if obs_uniques is not None else None),
        },
        source_dataset_metadata=dict(dataset.source_metadata),
        notes=config.notes,
    )
    return write_bundle_parquet(
        output_dir=output_dir,
        run_id=config.run_id,
        bundle_name="states",
        columns={
            "sensory_bmu": np.asarray(bmus, dtype=np.int32),
            "sensory_node_count": np.full((len(bmus),), int(sensory_state.gs.n), dtype=np.int32),
        },
        meta=meta,
        embed_metadata_in_parquet=bool(config.embed_metadata_in_parquet),
    )
