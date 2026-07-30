from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

from .artifacts import write_cscg_artifacts
from .config import CSCGConfig
from .evaluate import (
    compute_cscg_precision,
    decode_cscg_states,
    summarize_cscg_observation_prediction_accuracy,
)
from .prepare import (
    action_to_indices,
    collect_streaming_cscg_metadata,
    filter_valid_cscg_rows,
    fit_state_to_gt_map,
    import_cscg_module,
    iter_cscg_training_batches,
    load_dataset_arrays,
    load_ground_truth_states,
    map_to_indices,
    resolve_dataset_path_and_metadata,
    to_indices,
)
from .train import objective_scores, run_cscg_online_em, train_cscg_model, viterbi_refined_eval_model_npz
from .structural_init import (
    apply_observation_tensor_to_clone_counts,
    clone_edges_from_oracle_topology,
    fixed_or_ground_truth_capacity,
    normalize_structural_options,
    observation_edges_from_empirical_ground_truth,
    observation_edges_from_oracle_topology,
)
from ...infrastructure.paper_precision import resolve_eval_checkpoints
from ...infrastructure.paper_precision_snapshots import write_cscg_paper_precision_snapshot
from ...infrastructure.runtime import progress_enabled


def _online_sample_schedule_total(*, explicit_checkpoints: list[int] | None, default_total: int) -> int:
    if explicit_checkpoints is None:
        return int(default_total)
    valid = [int(value) for value in explicit_checkpoints if 0 < int(value) <= int(default_total)]
    if not valid:
        return int(default_total)
    return max(valid)


def _consume_crossed_checkpoints(checkpoints: list[int], next_index: int, current_step: int) -> tuple[int, bool]:
    should_record = False
    while next_index < len(checkpoints) and int(current_step) >= int(checkpoints[next_index]):
        should_record = True
        next_index += 1
    return next_index, should_record


def _resolve_cscg_clone_allocation(
    *,
    config: CSCGConfig,
    obs_idx: np.ndarray,
    ground_truth_states: np.ndarray | None,
    n_obs: int,
) -> np.ndarray:
    allocation = str(config.clone_allocation).strip().lower()
    if allocation == "uniform":
        return np.full(int(n_obs), int(config.clones_per_obs), dtype=np.int64)
    if allocation not in {"ground_truth", "oracle_ground_truth"}:
        raise ValueError(f"Unsupported CSCG clone_allocation: {config.clone_allocation}")
    if ground_truth_states is None:
        raise ValueError("CSCG ground-truth clone allocation requires aligned ground-truth states.")
    obs_idx = np.asarray(obs_idx, dtype=np.int64)
    ground_truth_states = np.asarray(ground_truth_states)
    if obs_idx.shape[0] != ground_truth_states.shape[0]:
        raise ValueError(
            "CSCG ground-truth clone allocation requires observation and ground-truth arrays with matching lengths."
        )
    n_clones = np.zeros(int(n_obs), dtype=np.int64)
    for obs in range(int(n_obs)):
        states = np.unique(ground_truth_states[obs_idx == obs])
        n_clones[obs] = max(1, int(states.shape[0])) * int(config.clones_per_state)
    return n_clones


def _resolve_cscg_capacity(
    *,
    config: CSCGConfig,
    obs_idx: np.ndarray,
    ground_truth_states: np.ndarray | None,
    n_obs: int,
) -> np.ndarray:
    capacity_mode = str(config.clone_capacity_mode).strip().lower()
    if capacity_mode in {"fixed", "ground_truth"}:
        return fixed_or_ground_truth_capacity(
            mode=capacity_mode,
            clone_allocation=str(config.clone_allocation),
            clones_per_obs=int(config.clones_per_obs),
            clones_per_state=int(config.clones_per_state),
            n_obs=int(n_obs),
            obs_idx=obs_idx,
            ground_truth_states=ground_truth_states,
        )
    return _resolve_cscg_clone_allocation(
        config=config,
        obs_idx=obs_idx,
        ground_truth_states=ground_truth_states,
        n_obs=n_obs,
    )


def _initialize_cscg_structure(
    *,
    config: CSCGConfig,
    model: Any,
    n_clones: np.ndarray,
    obs_idx: np.ndarray,
    act_idx: np.ndarray,
    ground_truth_states: np.ndarray | None,
    n_obs: int,
    n_actions: int,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    transfer_level, init_mode, structural_source, capacity_mode = normalize_structural_options(
        transfer_level=config.structural_transfer_level,
        init_mode=config.structural_init_mode,
        structural_source=config.structural_source,
        capacity_mode=config.clone_capacity_mode,
    )
    summary = {
        "source_method": None,
        "source_graph_path": None,
        "transfer_level": transfer_level,
        "init_mode": init_mode,
        "structural_source": structural_source,
        "capacity_mode": capacity_mode,
        "clone_counts": [int(value) for value in np.asarray(n_clones, dtype=np.int64).tolist()],
        "oracle_source": structural_source if structural_source.startswith("oracle_") else None,
    }
    if init_mode == "none" or structural_source == "none":
        return None, summary
    if structural_source.startswith("oracle_") and ground_truth_states is None:
        raise ValueError("Oracle CSCG structural initialization requires aligned ground-truth states.")
    if transfer_level == "clone_state":
        if structural_source != "oracle_maze_topology":
            raise ValueError("Pure CSCG clone/state transfer currently supports oracle_maze_topology.")
        counts, source_n_clones = clone_edges_from_oracle_topology(
            level=int(config.level),
            obs_idx=obs_idx,
            action_idx=act_idx,
            ground_truth_states=np.asarray(ground_truth_states),
            n_obs=int(n_obs),
            n_actions=int(n_actions),
            clones_per_state=int(config.clones_per_state),
            mode=init_mode,
        )
        if counts.shape != model.C.shape:
            raise ValueError(f"Structural counts shape {counts.shape} does not match model.C shape {model.C.shape}.")
        summary["clone_counts"] = [int(value) for value in source_n_clones.tolist()]
    else:
        if structural_source == "oracle_maze_topology":
            obs_tensor = observation_edges_from_oracle_topology(
                level=int(config.level),
                obs_idx=obs_idx,
                action_idx=act_idx,
                ground_truth_states=np.asarray(ground_truth_states),
                n_obs=int(n_obs),
                n_actions=int(n_actions),
                mode=init_mode,
            )
        elif structural_source == "oracle_empirical_ground_truth":
            obs_tensor = observation_edges_from_empirical_ground_truth(
                obs_idx=obs_idx,
                action_idx=act_idx,
                ground_truth_states=np.asarray(ground_truth_states),
                n_obs=int(n_obs),
                n_actions=int(n_actions),
                mode=init_mode,
            )
        else:
            obs_tensor = np.zeros((int(n_actions), int(n_obs), int(n_obs)), dtype=np.float32)
        counts = apply_observation_tensor_to_clone_counts(observation_tensor=obs_tensor, n_clones=n_clones)
        if counts.shape != model.C.shape:
            raise ValueError(f"Structural counts shape {counts.shape} does not match model.C shape {model.C.shape}.")
    if not np.any(counts):
        summary["empty"] = True
        return None, summary
    model.C = counts.astype(model.dtype, copy=False)
    model.update_T()
    return counts.astype(np.float32, copy=True), summary


def run_cscg(config: CSCGConfig, *, progress_callback: Optional[Callable[..., None]] = None):
    show_progress = progress_enabled(default=True)
    if progress_callback is not None:
        progress_callback(stage="jit_warmup_start", current=0, total=1, message="Starting CSCG warmup")
    chmm_mod = import_cscg_module(Path.cwd())
    if progress_callback is not None:
        progress_callback(stage="jit_warmup_end", current=1, total=1, message="CSCG warmup ready")
    algo = str(config.train_algorithm).strip().lower()
    training_mode = str(config.training_mode).strip().lower()
    if training_mode not in {"full", "online_em"}:
        raise ValueError(f"Unsupported CSCG training_mode: {config.training_mode}")

    dataset_metadata: dict[str, Any]
    dataset_parquet_name: str
    obs_values: np.ndarray
    action_values: np.ndarray
    online_training_summary = {
        "training_mode": training_mode,
        "batch_size": None if training_mode == "full" else int(config.batch_size),
        "online_lambda": None if training_mode == "full" else float(config.online_lambda),
        "num_batches": None,
        "final_viterbi_refinement": (False if training_mode == "online_em" else None),
    }
    initial_counts: np.ndarray | None = None
    structural_summary: dict[str, Any] = {}

    if training_mode == "online_em":
        dataset_path, dataset_metadata = resolve_dataset_path_and_metadata(
            datasets_root=config.datasets_root,
            level=config.level,
            sensor=config.sensor,
            sensor_range=config.sensor_range,
            dataset_select=config.dataset_select,
            dataset_run_id=config.dataset_run_id,
        )
        dataset_parquet_name = dataset_path.name
        obs_values, action_values, training_stats, available_batch_count = collect_streaming_cscg_metadata(
            parquet_path=dataset_path,
            batch_size=int(config.batch_size),
            valid_trajectories_only=bool(config.valid_trajectories_only),
        )
        n_obs = int(obs_values.shape[0])
        n_actions = int(action_values.shape[0])
        if n_obs < 1 or n_actions < 1:
            raise ValueError("CSCG online EM requires at least one observation and one action.")
        allocation_ground_truth = None
        allocation_obs_idx = None
        allocation_act_idx = None
        needs_ground_truth = (
            str(config.clone_allocation).strip().lower() in {"ground_truth", "oracle_ground_truth"}
            or str(config.clone_capacity_mode).strip().lower() == "ground_truth"
            or str(config.structural_source).strip().lower().startswith("oracle_")
        )
        if needs_ground_truth:
            allocation_observations, allocation_actions, allocation_collisions, _, _ = load_dataset_arrays(
                datasets_root=config.datasets_root,
                level=config.level,
                sensor=config.sensor,
                sensor_range=config.sensor_range,
                dataset_select=config.dataset_select,
                dataset_run_id=config.dataset_run_id,
            )
            allocation_ground_truth_all = load_ground_truth_states(config.ground_truth_states_parquet)
            allocation_observations, allocation_actions, allocation_ground_truth, _ = filter_valid_cscg_rows(
                observations=allocation_observations,
                actions=allocation_actions,
                collisions=allocation_collisions,
                ground_truth_bmu=allocation_ground_truth_all,
                valid_trajectories_only=bool(config.valid_trajectories_only),
            )
            allocation_obs_idx = map_to_indices(obs_values, allocation_observations)
            allocation_act_idx = map_to_indices(action_values, allocation_actions)
        allocation_obs = allocation_obs_idx if allocation_obs_idx is not None else np.arange(n_obs, dtype=np.int64)
        n_clones = _resolve_cscg_capacity(
            config=config,
            obs_idx=allocation_obs,
            ground_truth_states=allocation_ground_truth,
            n_obs=n_obs,
        )
        seed_obs = np.asarray([0, min(1, n_obs - 1), 0], dtype=np.int64)
        seed_actions = np.asarray([0, max(0, n_actions - 1), 0], dtype=np.int64)
        model = chmm_mod.CHMM(
            n_clones=n_clones,
            x=seed_obs,
            a=seed_actions,
            pseudocount=float(config.pseudocount),
            dtype=np.float32,
            seed=int(config.seed),
        )
        initial_counts, structural_summary = _initialize_cscg_structure(
            config=config,
            model=model,
            n_clones=n_clones,
            obs_idx=allocation_obs,
            act_idx=(allocation_act_idx if allocation_act_idx is not None else np.zeros_like(allocation_obs, dtype=np.int64)),
            ground_truth_states=allocation_ground_truth,
            n_obs=n_obs,
            n_actions=n_actions,
        )
    else:
        observations, actions, collisions, dataset_metadata, dataset_parquet_name = load_dataset_arrays(
            datasets_root=config.datasets_root,
            level=config.level,
            sensor=config.sensor,
            sensor_range=config.sensor_range,
            dataset_select=config.dataset_select,
            dataset_run_id=config.dataset_run_id,
        )
        ground_truth_bmu = load_ground_truth_states(config.ground_truth_states_parquet)
        observations, actions, training_ground_truth_bmu, training_stats = filter_valid_cscg_rows(
            observations=observations,
            actions=actions,
            collisions=collisions,
            ground_truth_bmu=ground_truth_bmu,
            valid_trajectories_only=bool(config.valid_trajectories_only),
        )
        obs_idx_seed, obs_values = to_indices(observations)
        act_idx_seed, action_values = action_to_indices(actions)
        if obs_idx_seed.shape[0] < 3 or act_idx_seed.shape[0] < 3:
            raise ValueError("CSCG baseline could not align enough samples between dataset and ground truth.")
        n_obs = int(obs_idx_seed.max()) + 1
        n_clones = _resolve_cscg_capacity(
            config=config,
            obs_idx=obs_idx_seed,
            ground_truth_states=training_ground_truth_bmu,
            n_obs=n_obs,
        )
        model = chmm_mod.CHMM(
            n_clones=n_clones,
            x=obs_idx_seed.astype(np.int64, copy=False),
            a=act_idx_seed.astype(np.int64, copy=False),
            pseudocount=float(config.pseudocount),
            dtype=np.float32,
            seed=int(config.seed),
        )
        initial_counts, structural_summary = _initialize_cscg_structure(
            config=config,
            model=model,
            n_clones=n_clones,
            obs_idx=obs_idx_seed,
            act_idx=act_idx_seed,
            ground_truth_states=training_ground_truth_bmu,
            n_obs=n_obs,
            n_actions=int(action_values.shape[0]),
        )

    paper_precision_history: list[dict[str, Any]] = []
    em_schedule_total_units = int(config.n_iter)
    if training_mode == "online_em":
        em_schedule_total_units = _online_sample_schedule_total(
            explicit_checkpoints=config.paper_precision_eval_checkpoints,
            default_total=max(1, int(config.n_iter)),
        )
    em_eval_checkpoints = resolve_eval_checkpoints(
        total_units=em_schedule_total_units,
        schedule=config.paper_precision_schedule,
        num_points=config.paper_precision_num_points,
        eval_interval=int(config.paper_precision_eval_interval),
        explicit_checkpoints=config.paper_precision_eval_checkpoints,
    )
    em_eval_checkpoint_set = set(em_eval_checkpoints)
    next_online_eval_index = 0
    last_em_iteration = 0
    last_em_step = 0

    def _record_paper_precision(*, stage: str, iteration: int, current_step: int, observed_samples: int, model) -> None:
        model_npz = {"T": np.asarray(model.T), "n_clones": n_clones}
        write_cscg_paper_precision_snapshot(
            config=config,
            stage=stage,
            iteration=int(iteration),
            step=int(current_step),
            observed_samples=int(observed_samples),
            model_npz=model_npz,
            obs_uniques=np.asarray(obs_values),
            action_uniques=np.asarray(action_values),
        )
        paper_precision_history.append(
            {
                "stage": stage,
                "iteration": int(iteration),
                "step": int(current_step),
                "observed_samples": int(observed_samples),
            }
        )

    def _paper_precision_callback(*, stage: str, iteration: int, model, step: int | None = None) -> None:
        nonlocal last_em_iteration, last_em_step, next_online_eval_index
        if str(stage).strip().lower() != "em":
            return
        current_step = int(step if training_mode == "online_em" and step is not None else iteration)
        last_em_iteration = int(iteration)
        last_em_step = current_step
        if progress_callback is not None:
            if training_mode == "online_em":
                progress_callback(
                    stage=f"train_{stage}",
                    current=min(current_step, int(config.n_iter)),
                    total=int(config.n_iter),
                    message=f"observed samples {current_step} (batch {iteration})",
                )
            else:
                progress_callback(stage=f"train_{stage}", current=int(iteration), total=int(config.n_iter), message=f"{stage} iter {iteration}")
        if training_mode == "online_em":
            next_online_eval_index, should_record = _consume_crossed_checkpoints(
                em_eval_checkpoints,
                next_online_eval_index,
                current_step,
            )
            if not should_record:
                return
        elif int(iteration) not in em_eval_checkpoint_set:
            return
        _record_paper_precision(
            stage=stage,
            iteration=int(iteration),
            current_step=current_step,
            observed_samples=current_step,
            model=model,
        )

    training_callback = (
        _paper_precision_callback if bool(config.paper_precision_enabled) and str(config.paper_precision_mode).strip().lower() == "per_iteration" else None
    )
    if training_mode == "online_em":
        convergence_by_stage = {
            "em": run_cscg_online_em(
                model,
                chmm_mod=chmm_mod,
                batch_iterator_factory=lambda: iter_cscg_training_batches(
                    parquet_path=dataset_path,
                    batch_size=int(config.batch_size),
                    include_prev=True,
                    valid_trajectories_only=bool(config.valid_trajectories_only),
                ),
                map_observations=lambda values: map_to_indices(obs_values, values),
                map_actions=lambda values: map_to_indices(action_values, values),
                online_lambda=float(config.online_lambda),
                max_batches=int(config.n_iter),
                term_early=bool(config.term_early),
                max_steps=int(config.n_iter),
                initial_counts=initial_counts,
                repeat_batches=True,
                iteration_callback=training_callback,
            )
        }
        training_stages = ["em"]
        online_training_summary["num_batches"] = int(len(convergence_by_stage["em"]))
    else:
        if progress_callback is not None and training_callback is None:
            def _training_progress_callback(*, stage: str, iteration: int, model, step: int | None = None) -> None:
                progress_callback(stage=f"train_{stage}", current=int(iteration), total=int(config.n_iter), message=f"{stage} iter {iteration}")
            training_callback = _training_progress_callback
        observations, actions, collisions, _, _ = load_dataset_arrays(
            datasets_root=config.datasets_root,
            level=config.level,
            sensor=config.sensor,
            sensor_range=config.sensor_range,
            dataset_select=config.dataset_select,
            dataset_run_id=config.dataset_run_id,
        )
        observations, actions, _, _ = filter_valid_cscg_rows(
            observations=observations,
            actions=actions,
            collisions=collisions,
            valid_trajectories_only=bool(config.valid_trajectories_only),
        )
        obs_idx_train = map_to_indices(obs_values, observations).astype(np.int64, copy=False)
        act_idx_train = map_to_indices(action_values, actions).astype(np.int64, copy=False)
        convergence_by_stage, training_stages = train_cscg_model(
            model,
            algo=algo,
            obs_idx=obs_idx_train,
            act_idx=act_idx_train,
            n_iter=int(config.n_iter),
            term_early=bool(config.term_early),
            iteration_callback=training_callback,
        )

    paper_precision_mode = str(config.paper_precision_mode).strip().lower()
    if bool(config.paper_precision_enabled) and paper_precision_mode == "per_iteration" and last_em_iteration > 0:
        last_recorded_step = int(paper_precision_history[-1]["step"]) if paper_precision_history else -1
        explicit_online_schedule_complete = (
            training_mode == "online_em"
            and config.paper_precision_eval_checkpoints is not None
            and next_online_eval_index >= len(em_eval_checkpoints)
        )
        if last_recorded_step < last_em_step and not explicit_online_schedule_complete:
            _record_paper_precision(
                stage="em",
                iteration=last_em_iteration,
                current_step=last_em_step,
                observed_samples=last_em_step,
                model=model,
            )

    observations, actions, collisions, _, _ = load_dataset_arrays(
        datasets_root=config.datasets_root,
        level=config.level,
        sensor=config.sensor,
        sensor_range=config.sensor_range,
        dataset_select=config.dataset_select,
        dataset_run_id=config.dataset_run_id,
    )
    ground_truth_bmu = load_ground_truth_states(config.ground_truth_states_parquet)
    observations, actions, filtered_ground_truth_bmu, valid_transition_stats = filter_valid_cscg_rows(
        observations=observations,
        actions=actions,
        collisions=collisions,
        ground_truth_bmu=ground_truth_bmu,
        valid_trajectories_only=bool(config.valid_trajectories_only),
    )
    ground_truth_bmu = np.asarray(filtered_ground_truth_bmu, dtype=np.int32)
    obs_idx = map_to_indices(obs_values, observations).astype(np.int64, copy=False)
    act_idx = map_to_indices(action_values, actions).astype(np.int64, copy=False)
    length = min(int(ground_truth_bmu.shape[0]), int(obs_idx.shape[0]), int(act_idx.shape[0]))
    if length < 3:
        raise ValueError("CSCG baseline could not align enough samples between dataset and ground truth.")
    obs_idx = obs_idx[:length]
    act_idx = act_idx[:length]
    ground_truth_bmu = ground_truth_bmu[:length]
    valid_transition_stats = dict(training_stats if training_mode == "online_em" else valid_transition_stats)
    online_training_summary["structural_initialization"] = structural_summary

    if training_mode == "online_em" and algo == "viterbi":
        model.pseudocount = 0.0
        convergence_by_stage["viterbi"] = np.asarray(
            model.learn_viterbi_T(obs_idx, act_idx, n_iter=int(config.n_iter)),
            dtype=np.float32,
        )
        training_stages = ["em", "viterbi"]
        online_training_summary["final_viterbi_refinement"] = True

    objective_name, objective_scores_arr = objective_scores(model, algo, obs_idx, act_idx)
    if progress_callback is not None:
        progress_callback(stage="native_eval", current=0, total=1, message="Starting native eval")
    decode_log2_lik_arr, clone_states = decode_cscg_states(model, obs_idx, act_idx)
    state_to_gt = fit_state_to_gt_map(clone_states, ground_truth_bmu)
    mapped_bmu = state_to_gt[clone_states]
    observation_prediction_eval = summarize_cscg_observation_prediction_accuracy(
        clone_states=clone_states,
        ground_truth_bmu=ground_truth_bmu,
        act_idx=act_idx,
        horizons=[1, 3, 5],
        show_progress=show_progress,
    )
    latent_node_count = np.full(length, fill_value=int(n_clones.sum()), dtype=np.int32)
    states_path = write_cscg_artifacts(
        config=config,
        model=model,
        dataset_parquet_name=dataset_parquet_name,
        dataset_metadata=dataset_metadata,
        act_idx=act_idx,
        n_obs=n_obs,
        n_clones=n_clones,
        convergence_by_stage=convergence_by_stage,
        training_stages=training_stages,
        objective_name=objective_name,
        objective_scores=objective_scores_arr,
        decode_log2_lik=decode_log2_lik_arr,
        ground_truth_bmu=ground_truth_bmu,
        valid_transition_stats=valid_transition_stats,
        training_summary=online_training_summary,
        mapped_bmu=mapped_bmu,
        latent_node_count=latent_node_count,
        clone_states=clone_states,
        observation_prediction_eval=observation_prediction_eval,
    )
    if bool(config.paper_precision_enabled) and paper_precision_mode == "final":
        final_step = int(last_em_step) if int(last_em_step) > 0 else int(config.n_iter)
        final_iteration = int(last_em_iteration) if int(last_em_iteration) > 0 else int(config.n_iter)
        if bool(getattr(config, "eval_viterbi", False)):
            final_stage = "viterbi_eval"
            final_model_npz = viterbi_refined_eval_model_npz(
                model,
                obs_idx=obs_idx,
                act_idx=act_idx,
                n_clones=n_clones,
                n_iter=int(config.n_iter),
            )
        else:
            final_stage = "final"
            final_model_npz = {"T": np.asarray(model.T), "n_clones": n_clones}
        write_cscg_paper_precision_snapshot(
            config=config,
            stage=final_stage,
            iteration=final_iteration,
            step=final_step,
            observed_samples=final_step,
            model_npz=final_model_npz,
            obs_uniques=np.asarray(obs_values),
            action_uniques=np.asarray(action_values),
        )
    compute_cscg_precision(config=config)
    if progress_callback is not None:
        progress_callback(stage="native_eval", current=1, total=1, message="Native eval complete")
    return states_path


__all__ = ["CSCGConfig", "compute_cscg_precision", "run_cscg"]
