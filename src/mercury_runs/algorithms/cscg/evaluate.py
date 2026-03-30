from __future__ import annotations

from typing import Any
from pathlib import Path

import numpy as np

from ...infrastructure.paper_precision import (
    action_ids_from_action_vectors,
    build_action_conditioned_tensor_from_valid_transitions,
    build_action_conditioned_tensor_from_walks,
    build_weighted_adjacency_from_valid_transitions,
    cartesian_state_ids,
    collapse_action_transition_tensor_to_adjacency,
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
    map_values_with_reference,
    remap_position_walks_to_global_ids,
    resolve_dataset_parquet_from_bundle,
    support_action_transition_tensor,
    write_paper_precision_artifact,
)
from ...io_parquet import ParquetConfig, load_level_parquet
from ...infrastructure.reporting import rollout_n_step_observation_prediction_metrics
from .prepare import filter_valid_cscg_rows, import_cscg_module


def decode_cscg_states(model: Any, obs_idx: np.ndarray, act_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    decode_log2_lik, clone_states = model.decode(obs_idx, act_idx)
    return np.asarray(decode_log2_lik, dtype=np.float64), np.asarray(clone_states, dtype=np.int32)


def summarize_cscg_observation_prediction_accuracy(
    *,
    clone_states: np.ndarray,
    ground_truth_bmu: np.ndarray,
    act_idx: np.ndarray,
    horizons: list[int],
    show_progress: bool = True,
) -> dict[str, Any]:
    metrics = rollout_n_step_observation_prediction_metrics(
        state_series=clone_states,
        gt_series=ground_truth_bmu,
        action_series=act_idx,
        horizons=horizons,
        show_progress=show_progress,
        progress_desc="CSCG Native Eval",
    )
    return {
        "n_step_observation_prediction_accuracy": metrics["n_step_observation_prediction_accuracy"],
        "trajectory_log_likelihood_sum": metrics["trajectory_log_likelihood_sum"],
        "trajectory_log_likelihood_mean": metrics["trajectory_log_likelihood_mean"],
        "steps": metrics["steps"],
    }


def compute_cscg_precision(*, config) -> None:
    from scripts.analysis.compute_precision import compute_metrics_for_states

    states_path = Path(config.output_root) / f"level={config.level}" / f"sensor={config.sensor}"
    if config.sensor == "cardinal distance":
        states_path = states_path / f"range={config.sensor_range}"
    compute_metrics_for_states(
        states_path=states_path / f"{config.run_id}_states.parquet",
        out_dir=Path(config.output_root).parents[1] / "metrics" / "cscg",
        run_id=config.run_id,
        window_length=int(config.window_length),
        precision_columns={"latent_precision": "cscg_state_id"},
        method_metadata={
            "method": "cscg",
            "level": config.level,
            "latent_sensor": config.sensor,
            "latent_sensor_range": config.sensor_range,
        },
    )


def compute_cscg_paper_precision(*, config) -> Path:
    states_path = Path(config.output_root) / f"level={config.level}" / f"sensor={config.sensor}"
    if config.sensor == "cardinal distance":
        states_path = states_path / f"range={config.sensor_range}"
    states_path = states_path / f"{config.run_id}_states.parquet"
    dataset_path = resolve_dataset_parquet_from_bundle(states_path, datasets_root=Path(config.datasets_root))
    loaded = load_level_parquet(
        ParquetConfig(
            root=Path(config.datasets_root),
            level=config.level,
            sensor=config.sensor,
            sensor_range=config.sensor_range,
            select="run_id",
            run_id=dataset_path.stem,
        )
    )
    train_obs, train_actions, _, _ = filter_valid_cscg_rows(
        observations=np.asarray(loaded.observations),
        actions=np.asarray(loaded.actions),
        collisions=np.asarray(loaded.collisions, dtype=bool),
        valid_trajectories_only=bool(config.valid_trajectories_only),
    )
    obs_uniques = np.unique(train_obs, axis=0)
    action_uniques = np.unique(train_actions, axis=0)
    model_npz = np.load(states_path.with_name(f"{config.run_id}_cscg_model.npz"))
    n_clones = np.asarray(model_npz["n_clones"], dtype=np.int64)
    chmm_mod = import_cscg_module(Path.cwd())
    metrics = compute_cscg_paper_precision_metrics_from_model(
        model_npz=model_npz,
        n_clones=n_clones,
        config=config,
        obs_uniques=obs_uniques,
        action_uniques=action_uniques,
        chmm_mod=chmm_mod,
    )
    return write_cscg_paper_precision_payload(config=config, metrics=metrics, history=[], mode=str(config.paper_precision_mode))


def compute_cscg_paper_precision_metrics_from_model(
    *,
    model_npz: Any,
    n_clones: np.ndarray,
    config,
    obs_uniques: np.ndarray,
    action_uniques: np.ndarray,
    chmm_mod: Any,
    progress_callback=None,
) -> dict[str, Any]:
    walks = generate_random_start_walks(
        level=config.level,
        sensor=config.sensor,
        sensor_range=config.sensor_range,
        rand_prob=float(config.paper_precision_rand_prob),
        num_walks=int(config.paper_precision_num_walks),
        walk_length=int(config.paper_precision_walk_length),
        base_seed=int(config.paper_precision_seed),
    )
    inferred_walks: list[np.ndarray] = []
    gt_walks: list[np.ndarray] = []
    gt_exact_walks: list[np.ndarray] = []
    gt_position_walks: list[np.ndarray] = []
    gt_action_walks: list[np.ndarray] = []
    requested_walks = int(len(walks))
    skipped_short_walks = 0
    unsupported_walks = 0
    total_walks = int(len(walks))
    for index, walk in enumerate(walks, start=1):
        filtered_obs, filtered_actions, _, _ = filter_valid_cscg_rows(
            observations=walk.observations,
            actions=walk.actions,
            collisions=walk.collisions,
            valid_trajectories_only=bool(config.valid_trajectories_only),
        )
        obs_idx = map_values_with_reference(reference_uniques=obs_uniques, values=filtered_obs).astype(np.int64, copy=False)
        act_idx = map_values_with_reference(reference_uniques=action_uniques, values=filtered_actions).astype(np.int64, copy=False)
        if obs_idx.shape[0] == 0 or act_idx.shape[0] == 0:
            skipped_short_walks += 1
            continue
        model = chmm_mod.CHMM(
            n_clones=n_clones,
            x=obs_idx,
            a=act_idx,
            pseudocount=float(config.pseudocount),
            dtype=np.float32,
            seed=int(config.seed),
        )
        model.T = np.asarray(model_npz["T"], dtype=np.float32)
        try:
            _, clone_states = model.decode(obs_idx, act_idx)
        except AssertionError:
            unsupported_walks += 1
            continue
        gt_collisions = np.asarray(walk.collisions, dtype=bool)
        gt = cartesian_state_ids(
            np.asarray(walk.cartesian_observations)[(~gt_collisions) if bool(config.valid_trajectories_only) else slice(None)]
        )
        gt_exact = exact_cartesian_state_ids_for_level(
            level_index=int(config.level),
            positions=np.asarray(walk.cartesian_observations)[(~gt_collisions) if bool(config.valid_trajectories_only) else slice(None)],
        )
        length = min(len(clone_states), len(gt))
        inferred_walks.append(np.asarray(clone_states[:length], dtype=np.int64))
        gt_walks.append(gt[:length].astype(np.int64, copy=False))
        gt_exact_walks.append(gt_exact[:length].astype(np.int64, copy=False))
        gt_position_walks.append(np.asarray(walk.cartesian_observations)[(~gt_collisions) if bool(config.valid_trajectories_only) else slice(None)][:length])
        gt_action_walks.append(action_ids_from_action_vectors(filtered_actions)[:length])
        if progress_callback is not None and ((index % 5) == 0 or index == total_walks):
            progress_callback(current=index, total=total_walks, message="paper precision")
    metrics = compute_fixed_map_precision_details(
        inferred_states=concatenate_walk_series(inferred_walks),
        ground_truth_states=concatenate_walk_series(gt_walks),
        metric_name="latent_precision",
    )
    exact_metrics = compute_fixed_map_precision_details(
        inferred_states=concatenate_walk_series(inferred_walks),
        ground_truth_states=concatenate_walk_series(gt_exact_walks),
        metric_name="latent_exact_precision",
    )
    attempted_walks = requested_walks - skipped_short_walks
    decoded_walks = attempted_walks - unsupported_walks
    metrics.update(
        {
            "requested_walks": requested_walks,
            "attempted_walks": attempted_walks,
            "decoded_walks": decoded_walks,
            "unsupported_walks": unsupported_walks,
            "skipped_short_walks": skipped_short_walks,
            "decode_coverage": (float(decoded_walks) / float(attempted_walks) if attempted_walks > 0 else 0.0),
        }
    )
    metrics["latent_purity"] = compute_purity_from_cooccurrence(exact_metrics["cooccurrence_matrix"])
    metrics["latent_exact_state_cooccurrence_matrix"] = exact_metrics["cooccurrence_matrix"]
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
        structure_metrics = compute_weighted_structure_metrics(
            decoded_walks=inferred_walks,
            true_walks=true_structure_walks,
            W_hat=collapse_action_transition_tensor_to_adjacency(np.asarray(model_npz["T"], dtype=np.float64)),
            n_true=n_true_states,
            W_true=W_true,
            epsilon=structure_epsilon,
        )
        metrics["latent_mean_total_variation"] = structure_metrics["mean_total_variation"]
        metrics["latent_edge_precision"] = structure_metrics["edge_precision"]
        metrics["latent_edge_recall"] = structure_metrics["edge_recall"]
        metrics["latent_edge_f1"] = structure_metrics["edge_f1"]
        valid_transitions = exact_valid_sensorimotor_transitions(level_index=int(config.level))
        n_actions = max(infer_num_actions_from_valid_transitions(valid_transitions), int(np.asarray(model_npz["T"]).shape[0]))
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
        action_structure_metrics = compute_action_conditioned_structure_metrics(
            decoded_walks=inferred_walks,
            true_walks=true_structure_walks,
            action_walks=gt_action_walks,
            T_hat=support_action_transition_tensor(np.asarray(model_npz["T"], dtype=np.float64), epsilon=structure_epsilon),
            n_true=n_true_states,
            T_true=T_true,
            epsilon=structure_epsilon,
        )
        metrics["latent_action_conditioned_mean_total_variation"] = action_structure_metrics["mean_total_variation"]
        metrics["latent_action_conditioned_edge_precision"] = action_structure_metrics["edge_precision"]
        metrics["latent_action_conditioned_edge_recall"] = action_structure_metrics["edge_recall"]
        metrics["latent_action_conditioned_edge_f1"] = action_structure_metrics["edge_f1"]
    state_to_gt = dominant_ground_truth_mapping_from_cooccurrence(exact_metrics["cooccurrence_matrix"])
    valid_transitions = exact_valid_sensorimotor_transitions(level_index=int(config.level))
    transitions = set()
    transition_tensor = np.asarray(model_npz["T"], dtype=np.float32)
    for action_idx in range(int(transition_tensor.shape[0])):
        for source_idx, target_idx in np.argwhere(transition_tensor[action_idx] > 0.0).tolist():
            if int(source_idx) not in state_to_gt or int(target_idx) not in state_to_gt:
                continue
            transitions.add(
                (
                    int(state_to_gt[int(source_idx)]),
                    int(action_idx),
                    int(state_to_gt[int(target_idx)]),
                )
            )
    metrics["latent_sensorimotor_link_error"] = compute_sensorimotor_link_error(
        edges=transitions,
        valid_transitions=valid_transitions,
    )
    return metrics


def write_cscg_paper_precision_payload(*, config, metrics: dict[str, Any], history: list[dict[str, Any]], mode: str) -> Path:
    schedule_unit = "training_step" if history else "final_only"
    payload = {
        "method": "cscg",
        "schema": "paper_precision_v1",
        "protocol": {
            "num_walks": int(config.paper_precision_num_walks),
            "walk_length": int(config.paper_precision_walk_length),
            "rand_prob": float(config.paper_precision_rand_prob),
            "base_seed": int(config.paper_precision_seed),
            "inferred_state_series": "cscg_state_id",
            "ground_truth_state_series": "cartesian_state_id",
            "state_decode_algorithm": "viterbi",
            "valid_trajectories_only": bool(config.valid_trajectories_only),
            "mode": mode,
            "eval_interval": int(config.paper_precision_eval_interval),
            "num_points": config.paper_precision_num_points,
            "schedule_unit": schedule_unit,
            "x_axis_unit": "observed_samples",
        },
        "metrics": metrics,
        "history": history,
    }
    out_dir = Path(config.output_root).parents[1] / "metrics" / "cscg"
    return write_paper_precision_artifact(path=out_dir / f"{config.run_id}_paper_precision.json", payload=payload)
