from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

from .artifacts import write_cscg_artifacts
from .config import CSCGConfig
from .evaluate import (
    compute_cscg_paper_precision,
    compute_cscg_paper_precision_metrics_from_model,
    compute_cscg_precision,
    decode_cscg_states,
    summarize_cscg_observation_prediction_accuracy,
    write_cscg_paper_precision_payload,
)
from .prepare import (
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
from .train import objective_scores, run_cscg_online_em, train_cscg_model
from ...infrastructure.paper_precision import resolve_eval_checkpoints
from ...infrastructure.runtime import progress_enabled


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
        n_clones = np.full(n_obs, int(config.clones_per_obs), dtype=np.int64)
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
    else:
        observations, actions, collisions, dataset_metadata, dataset_parquet_name = load_dataset_arrays(
            datasets_root=config.datasets_root,
            level=config.level,
            sensor=config.sensor,
            sensor_range=config.sensor_range,
            dataset_select=config.dataset_select,
            dataset_run_id=config.dataset_run_id,
        )
        observations, actions, _, training_stats = filter_valid_cscg_rows(
            observations=observations,
            actions=actions,
            collisions=collisions,
            valid_trajectories_only=bool(config.valid_trajectories_only),
        )
        obs_idx_seed, obs_values = to_indices(observations)
        act_idx_seed, action_values = to_indices(actions)
        if obs_idx_seed.shape[0] < 3 or act_idx_seed.shape[0] < 3:
            raise ValueError("CSCG baseline could not align enough samples between dataset and ground truth.")
        n_obs = int(obs_idx_seed.max()) + 1
        n_clones = np.full(n_obs, int(config.clones_per_obs), dtype=np.int64)
        model = chmm_mod.CHMM(
            n_clones=n_clones,
            x=obs_idx_seed.astype(np.int64, copy=False),
            a=act_idx_seed.astype(np.int64, copy=False),
            pseudocount=float(config.pseudocount),
            dtype=np.float32,
            seed=int(config.seed),
        )

    paper_precision_history: list[dict[str, Any]] = []
    em_schedule_total_units = int(config.n_iter)
    if training_mode == "online_em":
        em_schedule_total_units = max(1, min(int(config.n_iter), int(available_batch_count)))
    em_eval_checkpoints = set(
        resolve_eval_checkpoints(
            total_units=em_schedule_total_units,
            num_points=config.paper_precision_num_points,
            eval_interval=int(config.paper_precision_eval_interval),
        )
    )
    last_em_iteration = 0
    last_em_step = 0

    def _record_paper_precision(*, stage: str, iteration: int, current_step: int, observed_samples: int, model) -> None:
        paper_precision_history.append(
            {
                "stage": stage,
                "iteration": int(iteration),
                "step": int(current_step),
                "observed_samples": int(observed_samples),
                **compute_cscg_paper_precision_metrics_from_model(
                    model_npz={"T": np.asarray(model.T), "n_clones": n_clones},
                    n_clones=n_clones,
                    config=config,
                    obs_uniques=np.asarray(obs_values),
                    action_uniques=np.asarray(action_values),
                    chmm_mod=chmm_mod,
                ),
            }
        )

    def _paper_precision_callback(*, stage: str, iteration: int, model, step: int | None = None) -> None:
        nonlocal last_em_iteration, last_em_step
        if str(stage).strip().lower() != "em":
            return
        current_step = int(step if step is not None else iteration)
        last_em_iteration = int(iteration)
        last_em_step = current_step
        if progress_callback is not None:
            progress_callback(stage=f"train_{stage}", current=int(iteration), total=int(config.n_iter), message=f"{stage} iter {iteration}")
        if int(iteration) not in em_eval_checkpoints:
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
        if last_recorded_step < last_em_step:
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
    if bool(config.paper_precision_enabled):
        if paper_precision_mode == "per_iteration":
            write_cscg_paper_precision_payload(
                config=config,
                metrics=(
                    {
                        key: value
                        for key, value in paper_precision_history[-1].items()
                        if key not in {"stage", "iteration", "step", "observed_samples"}
                    }
                    if paper_precision_history
                    else {}
                ),
                history=paper_precision_history,
                mode=str(config.paper_precision_mode),
            )
        else:
            if progress_callback is not None:
                progress_callback(stage="paper_precision", current=0, total=1, message="Starting paper precision")
            compute_cscg_paper_precision(config=config)
    compute_cscg_precision(config=config)
    if progress_callback is not None:
        progress_callback(stage="native_eval", current=1, total=1, message="Native eval complete")
    return states_path


__all__ = ["CSCGConfig", "compute_cscg_precision", "compute_cscg_paper_precision", "run_cscg"]
