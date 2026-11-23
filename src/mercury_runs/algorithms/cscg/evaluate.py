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
    project_and_evaluate_transition_structure,
    remap_position_walks_to_global_ids,
    resolve_structure_metric_epsilons,
    resolve_dataset_parquet_from_bundle,
    structure_epsilon_key,
    write_paper_precision_artifact,
)
from ...infrastructure.paper_precision_snapshots import load_cscg_paper_precision_snapshot
from ...io_parquet import ParquetConfig, load_level_parquet
from ...infrastructure.reporting import rollout_n_step_observation_prediction_metrics
from .prepare import (
    action_to_indices,
    align_post_step_actions_to_cscg,
    filter_valid_cscg_rows,
    import_cscg_module,
    map_to_indices,
)


def decode_cscg_states(model: Any, obs_idx: np.ndarray, act_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    decode_log2_lik, clone_states = model.decode(obs_idx, align_post_step_actions_to_cscg(act_idx))
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
    _, action_uniques = action_to_indices(train_actions)
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
        valid_trajectories_only=bool(config.valid_trajectories_only),
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
        act_idx = map_to_indices(action_uniques, filtered_actions).astype(np.int64, copy=False)
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
            _, clone_states = model.decode(obs_idx, align_post_step_actions_to_cscg(act_idx))
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
        inferred_state_count=int(np.asarray(n_clones, dtype=np.int64).sum()),
        ground_truth_state_count=int(exact_cartesian_reference_positions(int(config.level)).shape[0]),
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
        structure_epsilons = resolve_structure_metric_epsilons(config)
        structure_epsilon = float(structure_epsilons[0])
        ignore_self_loops = bool(getattr(config, "structure_metrics_ignore_self_loops", True))
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
        metrics_by_epsilon: dict[str, Any] = {}
        transition_tensor = np.asarray(model_npz["T"], dtype=np.float64)
        collapsed_adjacency = collapse_action_transition_tensor_to_adjacency(transition_tensor)
        for epsilon in structure_epsilons:
            structure_metrics = compute_weighted_structure_metrics(
                decoded_walks=inferred_walks,
                true_walks=true_structure_walks,
                W_hat=collapsed_adjacency,
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
        for epsilon in structure_epsilons:
            action_structure_metrics = compute_action_conditioned_structure_metrics(
                decoded_walks=inferred_walks,
                true_walks=true_structure_walks,
                action_walks=gt_action_walks,
                T_hat=transition_tensor,
                n_true=n_true_states,
                T_true=T_true,
                epsilon=float(epsilon),
                ignore_self_loops=ignore_self_loops,
            )
            threshold_metrics = metrics["metrics_by_structure_epsilon"][structure_epsilon_key(float(epsilon))]
            threshold_metrics["latent_action_conditioned_mean_total_variation"] = action_structure_metrics["mean_total_variation"]
            threshold_metrics["latent_action_conditioned_edge_precision"] = action_structure_metrics["edge_precision"]
            threshold_metrics["latent_action_conditioned_edge_recall"] = action_structure_metrics["edge_recall"]
            threshold_metrics["latent_action_conditioned_edge_f1"] = action_structure_metrics["edge_f1"]
        primary_action_metrics = metrics["metrics_by_structure_epsilon"][structure_epsilon_key(structure_epsilon)]
        metrics["latent_action_conditioned_mean_total_variation"] = primary_action_metrics["latent_action_conditioned_mean_total_variation"]
        metrics["latent_action_conditioned_edge_precision"] = primary_action_metrics["latent_action_conditioned_edge_precision"]
        metrics["latent_action_conditioned_edge_recall"] = primary_action_metrics["latent_action_conditioned_edge_recall"]
        metrics["latent_action_conditioned_edge_f1"] = primary_action_metrics["latent_action_conditioned_edge_f1"]
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
    training_mode = str(getattr(config, "training_mode", "full")).strip().lower()
    schedule_unit = "final_only" if not history else ("training_step" if training_mode == "online_em" else "em_iteration")
    x_axis_unit = "observed_samples" if training_mode == "online_em" else schedule_unit
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
            "eval_checkpoints": config.paper_precision_eval_checkpoints,
            "num_points": config.paper_precision_num_points,
            "schedule_unit": schedule_unit,
            "x_axis_unit": x_axis_unit,
            "structure_metrics_epsilons": resolve_structure_metric_epsilons(config),
            "eval_viterbi_enabled": bool(getattr(config, "eval_viterbi", False)),
            "eval_viterbi_n_iter": (int(getattr(config, "n_iter", 0)) if bool(getattr(config, "eval_viterbi", False)) else None),
            "eval_viterbi_pseudocount": (0.0 if bool(getattr(config, "eval_viterbi", False)) else None),
        },
        "metrics": metrics,
        "history": history,
    }
    out_dir = Path(config.output_root).parents[1] / "metrics" / "cscg"
    return write_paper_precision_artifact(path=out_dir / f"{config.run_id}_paper_precision.json", payload=payload)


class WalkCache:
    """Walk cache keyed by eval protocol parameters. Shared across snapshots and runs."""

    def __init__(self) -> None:
        self._cache: dict[tuple, dict] = {}

    def get_or_generate(self, *, config) -> dict:
        key = self._make_key(config)
        if key not in self._cache:
            self._cache[key] = _generate_cscg_eval_walk_data(config=config)
        return self._cache[key]

    @staticmethod
    def _make_key(config) -> tuple:
        return (
            int(config.level),
            str(getattr(config, "sensor", "cartesian")),
            getattr(config, "sensor_range", None),
            float(config.paper_precision_rand_prob),
            int(config.paper_precision_seed),
            int(config.paper_precision_num_walks),
            int(config.paper_precision_walk_length),
            bool(getattr(config, "valid_trajectories_only", False)),
        )


def _generate_cscg_eval_walk_data(*, config) -> dict:
    """Generate environment walks and extract model-independent ground truth once."""
    walks = generate_random_start_walks(
        level=int(config.level),
        sensor=str(getattr(config, "sensor", "cartesian")),
        sensor_range=getattr(config, "sensor_range", None),
        rand_prob=float(config.paper_precision_rand_prob),
        num_walks=int(config.paper_precision_num_walks),
        walk_length=int(config.paper_precision_walk_length),
        base_seed=int(config.paper_precision_seed),
        valid_trajectories_only=bool(getattr(config, "valid_trajectories_only", False)),
    )
    valid_traj_only = bool(getattr(config, "valid_trajectories_only", False))
    level = int(config.level)

    filtered_obs_list: list[np.ndarray | None] = []
    filtered_actions_list: list[np.ndarray | None] = []
    gt_exact_list: list[np.ndarray | None] = []
    gt_cartesian_list: list[np.ndarray | None] = []
    gt_action_list: list[np.ndarray | None] = []

    for walk in walks:
        filtered_obs, filtered_actions, _, _ = filter_valid_cscg_rows(
            observations=np.asarray(walk.observations),
            actions=np.asarray(walk.actions),
            collisions=np.asarray(walk.collisions, dtype=bool),
            valid_trajectories_only=valid_traj_only,
        )
        if filtered_obs.shape[0] == 0 or filtered_actions.shape[0] == 0:
            filtered_obs_list.append(None)
            filtered_actions_list.append(None)
            gt_exact_list.append(None)
            gt_cartesian_list.append(None)
            gt_action_list.append(None)
            continue
        gt_collisions = np.asarray(walk.collisions, dtype=bool)
        mask = (~gt_collisions) if valid_traj_only else slice(None)
        cartesian_positions = np.asarray(walk.cartesian_observations)[mask]
        gt_exact = exact_cartesian_state_ids_for_level(level_index=level, positions=cartesian_positions)
        gt_cart = cartesian_state_ids(cartesian_positions)
        gt_act = action_ids_from_action_vectors(filtered_actions)
        min_len = min(filtered_obs.shape[0], int(gt_exact.shape[0]))
        filtered_obs_list.append(filtered_obs[:min_len])
        filtered_actions_list.append(filtered_actions[:min_len])
        gt_exact_list.append(gt_exact[:min_len].astype(np.int64, copy=False))
        gt_cartesian_list.append(gt_cart[:min_len].astype(np.int64, copy=False))
        gt_action_list.append(gt_act[:min_len])

    return {
        "filtered_obs_per_walk": filtered_obs_list,
        "filtered_actions_per_walk": filtered_actions_list,
        "gt_exact_per_walk": gt_exact_list,
        "gt_cartesian_per_walk": gt_cartesian_list,
        "gt_action_per_walk": gt_action_list,
        "n_walks": len(walks),
    }


def _decode_model_on_walk_cache(
    *,
    walk_data: dict,
    model_npz: Any,
    n_clones: np.ndarray,
    obs_uniques: np.ndarray,
    action_uniques: np.ndarray,
    config,
    chmm_mod: Any,
) -> dict:
    """Decode all cached walks with a model snapshot. Model-specific step."""
    inferred_walks: list[np.ndarray] = []
    gt_exact_aligned: list[np.ndarray] = []
    gt_cartesian_aligned: list[np.ndarray] = []
    gt_action_aligned: list[np.ndarray] = []
    skipped = 0
    unsupported = 0

    for filtered_obs, filtered_actions, gt_exact, gt_cart, gt_action in zip(
        walk_data["filtered_obs_per_walk"],
        walk_data["filtered_actions_per_walk"],
        walk_data["gt_exact_per_walk"],
        walk_data["gt_cartesian_per_walk"],
        walk_data["gt_action_per_walk"],
    ):
        if filtered_obs is None:
            skipped += 1
            continue
        obs_idx = map_values_with_reference(reference_uniques=obs_uniques, values=filtered_obs).astype(np.int64, copy=False)
        act_idx = map_to_indices(action_uniques, filtered_actions).astype(np.int64, copy=False)
        if obs_idx.shape[0] == 0 or act_idx.shape[0] == 0:
            skipped += 1
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
            _, clone_states = model.decode(obs_idx, align_post_step_actions_to_cscg(act_idx))
        except AssertionError:
            unsupported += 1
            continue
        length = min(len(clone_states), int(gt_exact.shape[0]))
        inferred_walks.append(np.asarray(clone_states[:length], dtype=np.int64))
        gt_exact_aligned.append(gt_exact[:length])
        gt_cartesian_aligned.append(gt_cart[:length])
        gt_action_aligned.append(gt_action[:length])

    return {
        "inferred_walks": inferred_walks,
        "gt_exact_walks": gt_exact_aligned,
        "gt_cartesian_walks": gt_cartesian_aligned,
        "gt_action_walks": gt_action_aligned,
        "skipped_short_walks": skipped,
        "unsupported_walks": unsupported,
        "requested_walks": int(walk_data["n_walks"]),
    }


def _compute_cscg_canonical_metrics(
    *,
    decoded: dict,
    model_npz: Any,
    config,
    n_clones: np.ndarray,
) -> dict[str, Any]:
    """Compute canonical metrics using project_and_evaluate_transition_structure."""
    inferred_walks = decoded["inferred_walks"]
    gt_exact_walks = decoded["gt_exact_walks"]
    gt_cartesian_walks = decoded["gt_cartesian_walks"]
    gt_action_walks = decoded["gt_action_walks"]

    inferred_series = concatenate_walk_series(inferred_walks)
    gt_exact_series = concatenate_walk_series(gt_exact_walks)
    gt_cartesian_series = concatenate_walk_series(gt_cartesian_walks)

    requested = decoded["requested_walks"]
    attempted = requested - decoded["skipped_short_walks"]
    decoded_count = attempted - decoded["unsupported_walks"]

    n_learned = int(np.asarray(n_clones, dtype=np.int64).sum())
    n_true = int(exact_cartesian_reference_positions(int(config.level)).shape[0])

    precision_details = compute_fixed_map_precision_details(
        inferred_states=inferred_series,
        ground_truth_states=gt_cartesian_series,
        metric_name="latent_precision",
    )
    exact_details = compute_fixed_map_precision_details(
        inferred_states=inferred_series,
        ground_truth_states=gt_exact_series,
        metric_name="latent_exact_precision",
        inferred_state_count=n_learned,
        ground_truth_state_count=n_true,
    )
    cooccurrence = np.asarray(exact_details["cooccurrence_matrix"], dtype=np.int64)

    metrics: dict[str, Any] = {
        "latent_precision": precision_details["latent_precision"],
        "latent_recall": precision_details["latent_recall"],
        "latent_evaluation_steps": precision_details["evaluation_steps"],
        "latent_inferred_state_count": precision_details["inferred_state_count"],
        "latent_ground_truth_state_count": precision_details["ground_truth_state_count"],
        "latent_cooccurrence_matrix": precision_details["cooccurrence_matrix"],
        "latent_exact_precision": exact_details["latent_exact_precision"],
        "latent_exact_recall": exact_details["latent_exact_recall"],
        "latent_purity": compute_purity_from_cooccurrence(cooccurrence),
        "latent_exact_state_cooccurrence_matrix": cooccurrence.astype(int).tolist(),
        "requested_walks": requested,
        "attempted_walks": attempted,
        "decoded_walks": decoded_count,
        "unsupported_walks": decoded["unsupported_walks"],
        "skipped_short_walks": decoded["skipped_short_walks"],
        "decode_coverage": (float(decoded_count) / float(attempted) if attempted > 0 else 0.0),
    }

    if bool(getattr(config, "structure_metrics_enabled", True)):
        structure_source = str(getattr(config, "structure_metrics_ground_truth_source", "empirical_walks")).strip().lower()
        ignore_self_loops = bool(getattr(config, "structure_metrics_ignore_self_loops", True))
        structure_epsilons = resolve_structure_metric_epsilons(config)
        structure_epsilon = float(structure_epsilons[0])

        valid_transitions = exact_valid_sensorimotor_transitions(level_index=int(config.level))
        n_actions = max(
            infer_num_actions_from_valid_transitions(valid_transitions),
            int(np.asarray(model_npz["T"]).shape[0]),
        )

        if structure_source == "maze_topology":
            T_true = build_action_conditioned_tensor_from_valid_transitions(
                valid_transitions, n_states=n_true, n_actions=n_actions
            )
        else:
            T_true = build_action_conditioned_tensor_from_walks(
                state_walks=gt_exact_walks,
                action_walks=gt_action_walks,
                n_states=n_true,
                n_actions=n_actions,
            )

        T_hat = np.asarray(model_npz["T"], dtype=np.float64)
        eval_result = project_and_evaluate_transition_structure(
            T_hat=T_hat,
            T_true=T_true,
            cooccurrence=cooccurrence,
            ignore_self_loops=ignore_self_loops,
            edge_thresholds=tuple(float(e) for e in structure_epsilons),
        )

        metrics_by_epsilon: dict[str, Any] = {}
        for epsilon in structure_epsilons:
            t = float(epsilon)
            a = eval_result.action_edge_metrics_by_threshold.get(t, {})
            c = eval_result.collapsed_edge_metrics_by_threshold.get(t, {})
            metrics_by_epsilon[structure_epsilon_key(t)] = {
                "latent_mean_total_variation": eval_result.collapsed_tv,
                "latent_edge_precision": c.get("edge_precision", float("nan")),
                "latent_edge_recall": c.get("edge_recall", float("nan")),
                "latent_edge_f1": c.get("edge_f1", float("nan")),
                "latent_action_conditioned_mean_total_variation": eval_result.action_tv,
                "latent_action_conditioned_edge_precision": a.get("edge_precision", float("nan")),
                "latent_action_conditioned_edge_recall": a.get("edge_recall", float("nan")),
                "latent_action_conditioned_edge_f1": a.get("edge_f1", float("nan")),
            }

        primary = metrics_by_epsilon[structure_epsilon_key(structure_epsilon)]
        metrics["latent_mean_total_variation"] = primary["latent_mean_total_variation"]
        metrics["latent_edge_precision"] = primary["latent_edge_precision"]
        metrics["latent_edge_recall"] = primary["latent_edge_recall"]
        metrics["latent_edge_f1"] = primary["latent_edge_f1"]
        metrics["latent_action_conditioned_mean_total_variation"] = primary["latent_action_conditioned_mean_total_variation"]
        metrics["latent_action_conditioned_edge_precision"] = primary["latent_action_conditioned_edge_precision"]
        metrics["latent_action_conditioned_edge_recall"] = primary["latent_action_conditioned_edge_recall"]
        metrics["latent_action_conditioned_edge_f1"] = primary["latent_action_conditioned_edge_f1"]
        metrics["metrics_by_structure_epsilon"] = metrics_by_epsilon

        state_to_gt = dominant_ground_truth_mapping_from_cooccurrence(cooccurrence)
        transitions: set[tuple[int, int, int]] = set()
        for action_idx in range(int(T_hat.shape[0])):
            for source_idx, target_idx in np.argwhere(T_hat[action_idx] > 0.0).tolist():
                s, t_state = int(source_idx), int(target_idx)
                if s in state_to_gt and t_state in state_to_gt:
                    transitions.add((int(state_to_gt[s]), action_idx, int(state_to_gt[t_state])))
        metrics["latent_sensorimotor_link_error"] = compute_sensorimotor_link_error(
            edges=transitions, valid_transitions=valid_transitions
        )

    return metrics


def evaluate_cscg_snapshots_canonical(
    *,
    snapshots: list[Path],
    config,
    chmm_mod: Any,
    walk_cache: "WalkCache | None" = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Canonical evaluation: shared walks, single alignment, project_and_evaluate_transition_structure.

    Generates walks once (or retrieves from cache), then for each snapshot decodes walks,
    builds cooccurrence, and calls the canonical evaluation pipeline.  Returns
    (history, final_metrics) ready for write_cscg_paper_precision_payload.
    """
    if not snapshots:
        return [], {}

    if walk_cache is not None:
        walk_data = walk_cache.get_or_generate(config=config)
    else:
        walk_data = _generate_cscg_eval_walk_data(config=config)

    history: list[dict[str, Any]] = []
    for snapshot_path in snapshots:
        snapshot = load_cscg_paper_precision_snapshot(snapshot_path)
        meta = snapshot["meta"]
        model_npz = snapshot["model_npz"]
        n_clones = np.asarray(snapshot["n_clones"], dtype=np.int64)
        obs_uniques = np.asarray(snapshot["obs_uniques"])
        action_uniques = np.asarray(snapshot["action_uniques"])

        decoded = _decode_model_on_walk_cache(
            walk_data=walk_data,
            model_npz=model_npz,
            n_clones=n_clones,
            obs_uniques=obs_uniques,
            action_uniques=action_uniques,
            config=config,
            chmm_mod=chmm_mod,
        )
        metrics = _compute_cscg_canonical_metrics(
            decoded=decoded,
            model_npz=model_npz,
            config=config,
            n_clones=n_clones,
        )
        history.append({
            "stage": str(meta.get("stage", "em")),
            "iteration": int(meta.get("iteration", meta.get("step", 0))),
            "step": int(meta.get("step", 0)),
            "observed_samples": int(meta.get("observed_samples", meta.get("step", 0))),
            **metrics,
        })

    final_metrics = {
        k: v for k, v in history[-1].items()
        if k not in {"stage", "iteration", "step", "observed_samples"}
    }
    return history, final_metrics
