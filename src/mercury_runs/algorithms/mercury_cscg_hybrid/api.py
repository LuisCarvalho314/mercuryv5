from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

from ..cscg.evaluate import (
    decode_cscg_states,
    summarize_cscg_observation_prediction_accuracy,
)
from ..cscg.prepare import (
    align_post_step_actions_to_cscg,
    fit_state_to_gt_map,
    import_cscg_module,
    iter_cscg_training_batches,
    map_to_indices,
    resolve_dataset_path_and_metadata,
)
from ..cscg.train import objective_scores, run_cscg_online_em, train_cscg_model, viterbi_refined_eval_model_npz
from ..mercury import MercuryConfig, run_mercury
from ..sensory import SensoryConfig, run_sensory
from mercury.graph.core import Graph
from .artifacts import write_hybrid_artifacts
from .config import MercuryCSCGHybridConfig
from .prepare import (
    build_hybrid_assignment,
    build_source_graph_observation_assignment,
    load_hybrid_training_arrays,
    load_mercury_latent_graph,
    map_eval_arrays,
    mercury_output_dir,
)
from .train import initialize_cscg_from_mercury_graph
from ...infrastructure.paper_precision import resolve_eval_checkpoints, write_paper_precision_artifact
from ...infrastructure.runtime import progress_enabled


def _mercury_config(config: MercuryCSCGHybridConfig) -> MercuryConfig:
    return MercuryConfig(
        datasets_root=config.datasets_root,
        results_root=config.output_root / "mercury",
        run_id=config.run_id,
        level=config.level,
        sensor=config.sensor,
        sensor_range=config.sensor_range,
        memory_length=config.memory_length,
        window_length=config.window_length,
        seed=config.seed,
        rand_prob=config.rand_prob,
        num_steps=config.num_steps,
        valid_trajectories_only=config.valid_trajectories_only,
        mercury_valid_trajectories_only=config.mercury_valid_trajectories_only,
        mercury_split_sensory_raw_latent_valid=config.mercury_split_sensory_raw_latent_valid,
        reuse_existing_dataset=config.reuse_existing_dataset,
        reuse_existing_run=config.reuse_existing_run,
        mercury_input_mode=config.mercury_input_mode,
        sensory=config.sensory,
        latent=config.latent,
        action_map=config.action_map,
        paper_precision_enabled=bool(config.paper_precision_enabled),
        paper_precision_mode=str(config.paper_precision_mode),
        paper_precision_eval_interval=int(config.paper_precision_eval_interval),
        paper_precision_eval_checkpoints=config.paper_precision_eval_checkpoints,
        paper_precision_num_points=config.paper_precision_num_points,
        paper_precision_schedule=str(config.paper_precision_schedule),
        paper_precision_num_walks=int(config.paper_precision_num_walks),
        paper_precision_walk_length=int(config.paper_precision_walk_length),
        structure_metrics_enabled=config.structure_metrics_enabled,
        structure_metrics_epsilon=config.structure_metrics_epsilon,
        structure_metrics_epsilons=config.structure_metrics_epsilons,
        structure_metrics_ignore_self_loops=config.structure_metrics_ignore_self_loops,
        structure_metrics_scope=config.structure_metrics_scope,
        structure_metrics_ground_truth_source=config.structure_metrics_ground_truth_source,
        run_parameters={"hybrid_parent_run_id": config.run_id},
        random_start=config.random_start,
        embed_metadata_in_parquet=config.embed_metadata_in_parquet,
        notes="mercury stage for mercury_cscg_hybrid",
    )


def _sensory_config(config: MercuryCSCGHybridConfig) -> SensoryConfig:
    return SensoryConfig(
        datasets_root=config.datasets_root,
        output_root=config.output_root / "sensory",
        run_id=config.run_id,
        level=config.level,
        sensor=config.sensor,
        sensor_range=config.sensor_range,
        seed=config.seed,
        rand_prob=config.rand_prob,
        num_steps=config.num_steps,
        window_length=config.window_length,
        valid_trajectories_only=config.valid_trajectories_only,
        observation_mode="raw",
        dataset_select=config.dataset_select,
        dataset_run_id=config.dataset_run_id,
        random_start=config.random_start,
        reuse_existing_dataset=True,
        sensory=config.sensory,
        action_map=config.action_map,
        paper_precision_enabled=bool(config.paper_precision_enabled),
        paper_precision_mode=str(config.paper_precision_mode),
        paper_precision_eval_interval=int(config.paper_precision_eval_interval),
        paper_precision_eval_checkpoints=config.paper_precision_eval_checkpoints,
        paper_precision_num_points=config.paper_precision_num_points,
        paper_precision_schedule=str(config.paper_precision_schedule),
        paper_precision_num_walks=int(config.paper_precision_num_walks),
        paper_precision_walk_length=int(config.paper_precision_walk_length),
        structure_metrics_enabled=config.structure_metrics_enabled,
        structure_metrics_epsilon=config.structure_metrics_epsilon,
        structure_metrics_epsilons=config.structure_metrics_epsilons,
        structure_metrics_ignore_self_loops=config.structure_metrics_ignore_self_loops,
        structure_metrics_scope=config.structure_metrics_scope,
        structure_metrics_ground_truth_source=config.structure_metrics_ground_truth_source,
        run_parameters={"hybrid_parent_run_id": config.run_id},
        embed_metadata_in_parquet=config.embed_metadata_in_parquet,
        notes=f"sensory stage for {config.method_name}",
    )


def write_hybrid_paper_precision_payload(*, config, metrics: dict[str, Any], history: list[dict[str, Any]], mode: str) -> Path:
    training_mode = str(getattr(config, "training_mode", "full")).strip().lower()
    schedule_unit = "final_only" if not history else ("training_step" if training_mode == "online_em" else "em_iteration")
    x_axis_unit = "observed_samples" if training_mode == "online_em" else schedule_unit
    method_name = str(getattr(config, "method_name", "mercury_cscg_hybrid"))
    payload = {
        "method": method_name,
        "schema": "paper_precision_v1",
        "protocol": {
            "num_walks": int(config.paper_precision_num_walks),
            "walk_length": int(config.paper_precision_walk_length),
            "rand_prob": float(config.paper_precision_rand_prob),
            "base_seed": int(config.paper_precision_seed),
            "inferred_state_series": f"{method_name}_state_id",
            "ground_truth_state_series": "cartesian_state_id",
            "state_decode_algorithm": "viterbi",
            "valid_trajectories_only": bool(config.valid_trajectories_only),
            "mode": mode,
            "eval_interval": int(config.paper_precision_eval_interval),
            "eval_checkpoints": config.paper_precision_eval_checkpoints,
            "num_points": config.paper_precision_num_points,
            "schedule_unit": schedule_unit,
            "x_axis_unit": x_axis_unit,
            "eval_viterbi_enabled": bool(getattr(config, "eval_viterbi", False)),
            "eval_viterbi_n_iter": (int(getattr(config, "n_iter", 0)) if bool(getattr(config, "eval_viterbi", False)) else None),
            "eval_viterbi_pseudocount": (0.0 if bool(getattr(config, "eval_viterbi", False)) else None),
        },
        "metrics": metrics,
        "history": history,
    }
    out_dir = Path(config.output_root).parents[1] / "metrics" / method_name
    return write_paper_precision_artifact(path=out_dir / f"{config.run_id}_paper_precision.json", payload=payload)


def _consume_crossed_checkpoints(checkpoints: list[int], next_index: int, current_step: int) -> tuple[int, bool]:
    should_record = False
    while next_index < len(checkpoints) and int(current_step) >= int(checkpoints[next_index]):
        should_record = True
        next_index += 1
    return next_index, should_record


def _online_sample_schedule_total(*, explicit_checkpoints: list[int] | None, default_total: int) -> int:
    if explicit_checkpoints is None:
        return int(default_total)
    valid = [int(value) for value in explicit_checkpoints if 0 < int(value) <= int(default_total)]
    if not valid:
        return int(default_total)
    return max(valid)


def _write_hybrid_paper_precision_snapshot(
    *,
    config,
    stage: str,
    iteration: int,
    step: int,
    observed_samples: int,
    model_npz: dict[str, np.ndarray],
    obs_uniques: np.ndarray,
    action_uniques: np.ndarray,
) -> None:
    method_name = str(getattr(config, "method_name", "mercury_cscg_hybrid"))
    out_dir = Path(config.output_root).parents[1] / "snapshots" / method_name / "paper_precision" / f"step={int(step)}"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / "model.npz", **model_npz)
    np.savez_compressed(out_dir / "observation_vocabulary.npz", obs_uniques=np.asarray(obs_uniques), action_uniques=np.asarray(action_uniques))
    (out_dir / "snapshot.json").write_text(
        json.dumps(
            {
                "method": method_name,
                "stage": str(stage),
                "iteration": int(iteration),
                "step": int(step),
                "observed_samples": int(observed_samples),
                "model_npz": "model.npz",
                "observation_vocabulary_npz": "observation_vocabulary.npz",
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _internal_mercury_paper_precision_path(mercury_config: MercuryConfig) -> Path:
    return Path(mercury_config.results_root).parents[1] / "metrics" / "mercury" / f"{mercury_config.run_id}_paper_precision.json"


def _hybrid_mercury_history(*, mercury_config: MercuryConfig, hybrid_config: MercuryCSCGHybridConfig) -> list[dict[str, Any]]:
    paper_precision_path = _internal_mercury_paper_precision_path(mercury_config)
    if not paper_precision_path.exists():
        return []
    payload = json.loads(paper_precision_path.read_text(encoding="utf-8"))
    history = payload.get("history")
    if not isinstance(history, list):
        return []
    rows: list[dict[str, Any]] = []
    for item in history:
        if not isinstance(item, dict):
            continue
        step = int(item.get("step", item.get("observed_samples", 0)))
        row = dict(item)
        row.update(
            {
                "method": "mercury_cscg_hybrid_mercury",
                "series_label": "Hybrid Mercury",
                "phase": "mercury",
                "stage": "mercury",
                "iteration": step,
                "step": step,
                "observed_samples": int(item.get("observed_samples", step)),
                "phase_step": step,
                "cumulative_step": step,
                "handoff_step": int(hybrid_config.num_steps),
            }
        )
        rows.append(row)
    return rows


def run_mercury_cscg_hybrid(
    config: MercuryCSCGHybridConfig,
    *,
    progress_callback: Optional[Callable[..., None]] = None,
) -> Path:
    training_mode = str(config.training_mode).strip().lower()
    if training_mode not in {"full", "online_em"}:
        raise ValueError(f"Unsupported Mercury-CSCG hybrid training_mode: {config.training_mode}")
    show_progress = progress_enabled(default=True)
    source_method = str(getattr(config, "source_method", "mercury")).strip().lower()
    mercury_states_path: Path | None = None
    mercury_latent_graph_path: Path | None = None
    source_states_path: Path | None = None
    source_graph_path: Path
    if source_method == "mercury":
        if progress_callback is not None:
            progress_callback(stage="mercury_start", current=0, total=1, message="Starting Mercury bootstrap")
        mercury_config = _mercury_config(config)
        mercury_states_path = run_mercury(
            mercury_config,
            progress_callback=(
                (lambda **kwargs: progress_callback(stage=f"mercury_{kwargs.pop('stage', 'progress')}", **kwargs))
                if progress_callback is not None
                else None
            ),
        )
        mercury_dir = mercury_output_dir(config.output_root / "mercury", config.level, config.sensor, config.sensor_range)
        mercury_latent_graph_path = mercury_dir / f"{config.run_id}_latent_graph.npz"
        source_states_path = mercury_states_path
        source_graph_path = mercury_latent_graph_path
    elif source_method in {"tmgwr", "sensory"}:
        if progress_callback is not None:
            progress_callback(stage="sensory_start", current=0, total=1, message="Starting TMGWR bootstrap")
        sensory_config = _sensory_config(config)
        source_states_path = run_sensory(
            sensory_config,
            progress_callback=(
                (lambda **kwargs: progress_callback(stage=f"sensory_{kwargs.pop('stage', 'progress')}", **kwargs))
                if progress_callback is not None
                else None
            ),
        )
        sensory_dir = mercury_output_dir(config.output_root / "sensory", config.level, config.sensor, config.sensor_range)
        source_graph_path = sensory_dir / f"{config.run_id}_sensory_graph.npz"
    else:
        raise ValueError(f"Unsupported CSCG hybrid source_method: {config.source_method}")
    if progress_callback is not None:
        progress_callback(stage="jit_warmup_start", current=0, total=1, message="Starting CSCG warmup")
    chmm_mod = import_cscg_module(Path.cwd())
    if progress_callback is not None:
        progress_callback(stage="jit_warmup_end", current=1, total=1, message="CSCG warmup ready")

    training = load_hybrid_training_arrays(config)
    latent_graph = load_mercury_latent_graph(source_graph_path) if source_method == "mercury" else Graph.from_npz(str(source_graph_path))
    if source_method == "mercury" and str(config.clone_capacity_mode).strip().lower() == "source":
        assignment = build_hybrid_assignment(
            latent_graph=latent_graph,
            obs_idx=training["obs_idx"],
            n_obs=int(training["obs_values"].shape[0]),
            memory_length=int(config.memory_length),
        )
    else:
        assignment = build_source_graph_observation_assignment(
            graph=latent_graph,
            obs_values=np.asarray(training["obs_values"]),
            n_obs=int(training["obs_values"].shape[0]),
            clone_capacity_mode=str(config.clone_capacity_mode),
            clones_per_obs=1,
            memory_length=(int(config.memory_length) if source_method == "mercury" else None),
        )
    init_ground_truth: np.ndarray | None = None
    if str(config.structural_source).strip().lower().startswith("oracle_"):
        init_arrays = map_eval_arrays(config, training["obs_values"], training["action_values"])
        init_ground_truth = np.asarray(init_arrays["ground_truth_bmu"], dtype=np.int32)
    model = initialize_cscg_from_mercury_graph(
        chmm_mod=chmm_mod,
        latent_graph=latent_graph,
        assignment=assignment,
        obs_idx=training["obs_idx"],
        act_idx=training["act_idx"],
        pseudocount=float(config.pseudocount),
        seed=int(config.seed),
        transfer_level=str(config.structural_transfer_level),
        init_mode=str(config.structural_init_mode),
        structural_source=str(config.structural_source),
        level=int(config.level),
        ground_truth_states=init_ground_truth,
    )
    mercury_initial_counts = np.asarray(model.C, dtype=np.float32).copy()

    paper_precision_history: list[dict[str, Any]] = []
    online_batches_per_epoch = max(1, int(np.ceil(float(config.num_steps) / float(config.batch_size))))
    online_sample_budget = max(1, int(config.n_iter))
    online_total_batches = max(1, int(config.n_iter))
    online_total_observed_samples = online_sample_budget
    online_schedule_total_units = _online_sample_schedule_total(
        explicit_checkpoints=config.paper_precision_eval_checkpoints,
        default_total=online_total_observed_samples,
    )
    em_eval_checkpoints = resolve_eval_checkpoints(
        total_units=(online_schedule_total_units if training_mode == "online_em" else int(config.n_iter)),
        schedule=config.paper_precision_schedule,
        num_points=config.paper_precision_num_points,
        eval_interval=int(config.paper_precision_eval_interval),
        explicit_checkpoints=config.paper_precision_eval_checkpoints,
    )
    em_eval_checkpoint_set = set(em_eval_checkpoints)
    next_online_eval_index = 0
    last_em_iteration = 0
    last_em_step = 0

    def _record_paper_precision(*, stage: str, iteration: int, model, current_step: int | None = None) -> None:
        nonlocal last_em_step
        model_npz = {"T": np.asarray(model.T), "n_clones": assignment.n_clones}
        step = int(current_step if current_step is not None else iteration)
        last_em_step = max(last_em_step, step)
        cumulative_step = int(config.num_steps) + step
        _write_hybrid_paper_precision_snapshot(
            config=config,
            stage=stage,
            iteration=int(iteration),
            step=step,
            observed_samples=step,
            model_npz=model_npz,
            obs_uniques=np.asarray(training["obs_values"]),
            action_uniques=np.asarray(training["action_values"]),
        )
        paper_precision_history.append(
            {
                "method": f"{config.method_name}_cscg",
                "series_label": "Hybrid CSCG",
                "phase": "cscg",
                "stage": stage,
                "iteration": int(iteration),
                "step": step,
                "observed_samples": step,
                "phase_step": step,
                "cumulative_step": cumulative_step,
                "handoff_step": int(config.num_steps),
            }
        )

    def _training_callback(*, stage: str, iteration: int, model, step: int | None = None) -> None:
        nonlocal last_em_iteration, last_em_step, next_online_eval_index
        if str(stage).strip().lower() != "em":
            return
        last_em_iteration = int(iteration)
        last_em_step = int(step if training_mode == "online_em" and step is not None else iteration)
        if progress_callback is not None:
            if training_mode == "online_em":
                progress_callback(
                    stage=f"train_{stage}",
                    current=min(last_em_step, online_sample_budget),
                    total=online_sample_budget,
                    message=f"observed samples {last_em_step} (batch {iteration})",
                )
            else:
                progress_callback(stage=f"train_{stage}", current=int(iteration), total=int(config.n_iter), message=f"{stage} iter {iteration}")
        if not bool(config.paper_precision_enabled) or str(config.paper_precision_mode).strip().lower() != "per_iteration":
            return
        if training_mode == "online_em":
            next_online_eval_index, should_record = _consume_crossed_checkpoints(
                em_eval_checkpoints,
                next_online_eval_index,
                last_em_step,
            )
            if should_record:
                _record_paper_precision(stage=stage, iteration=int(iteration), model=model, current_step=last_em_step)
            return
        if int(iteration) in em_eval_checkpoint_set:
            _record_paper_precision(stage=stage, iteration=int(iteration), model=model, current_step=last_em_step)

    training_callback = _training_callback if (progress_callback is not None or bool(config.paper_precision_enabled)) else None
    if training_mode == "online_em":
        if str(config.train_algorithm).strip().lower() not in {"em", "viterbi"}:
            raise ValueError(f"Unsupported CSCG train_algorithm: {config.train_algorithm}")
        dataset_path, _ = resolve_dataset_path_and_metadata(
            datasets_root=config.datasets_root,
            level=config.level,
            sensor=config.sensor,
            sensor_range=config.sensor_range,
            dataset_select=config.dataset_select,
            dataset_run_id=config.dataset_run_id,
        )
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
                map_observations=lambda values: map_to_indices(training["obs_values"], values),
                map_actions=lambda values: map_to_indices(training["action_values"], values),
                online_lambda=float(config.online_lambda),
                max_batches=online_total_batches,
                term_early=bool(config.term_early),
                max_steps=online_sample_budget,
                initial_counts=mercury_initial_counts * float(config.mercury_prior_weight),
                repeat_batches=True,
                iteration_callback=training_callback,
            )
        }
        training_stages = ["em"]
    else:
        convergence_by_stage, training_stages = train_cscg_model(
            model,
            algo=str(config.train_algorithm).strip().lower(),
            obs_idx=training["obs_idx"],
            act_idx=training["act_idx"],
            n_iter=int(config.n_iter),
            term_early=bool(config.term_early),
            iteration_callback=training_callback,
        )
    if bool(config.paper_precision_enabled) and str(config.paper_precision_mode).strip().lower() == "per_iteration" and last_em_iteration > 0:
        last_cscg_iteration = max(
            (int(row.get("iteration", 0)) for row in paper_precision_history if row.get("phase") == "cscg"),
            default=0,
        )
        explicit_online_schedule_complete = (
            training_mode == "online_em"
            and config.paper_precision_eval_checkpoints is not None
            and next_online_eval_index >= len(em_eval_checkpoints)
        )
        if last_cscg_iteration < last_em_iteration and not explicit_online_schedule_complete:
            _record_paper_precision(stage="em", iteration=last_em_iteration, model=model, current_step=last_em_step)

    eval_arrays = map_eval_arrays(config, training["obs_values"], training["action_values"])
    if training_mode == "online_em" and str(config.train_algorithm).strip().lower() == "viterbi":
        model.pseudocount = 0.0
        convergence_by_stage["viterbi"] = np.asarray(
            model.learn_viterbi_T(
                eval_arrays["obs_idx"],
                align_post_step_actions_to_cscg(eval_arrays["act_idx"]),
                n_iter=int(config.n_iter),
            ),
            dtype=np.float32,
        )
        training_stages = ["em", "viterbi"]
    objective_name, objective_scores_arr = objective_scores(
        model,
        str(config.train_algorithm).strip().lower(),
        eval_arrays["obs_idx"],
        eval_arrays["act_idx"],
    )
    if progress_callback is not None:
        progress_callback(stage="native_eval", current=0, total=1, message="Starting native eval")
    decode_log2_lik_arr, clone_states = decode_cscg_states(model, eval_arrays["obs_idx"], eval_arrays["act_idx"])
    state_to_gt = fit_state_to_gt_map(clone_states, eval_arrays["ground_truth_bmu"])
    mapped_bmu = state_to_gt[clone_states]
    observation_prediction_eval = summarize_cscg_observation_prediction_accuracy(
        clone_states=clone_states,
        ground_truth_bmu=eval_arrays["ground_truth_bmu"],
        act_idx=eval_arrays["act_idx"],
        horizons=[1, 3, 5],
        show_progress=show_progress,
    )
    latent_node_count = np.full(int(clone_states.shape[0]), fill_value=int(assignment.n_clones.sum()), dtype=np.int32)
    states_path = write_hybrid_artifacts(
        config=config,
        model=model,
        dataset_parquet_name=training["dataset_parquet_name"],
        dataset_metadata=training["dataset_metadata"],
        act_idx=eval_arrays["act_idx"],
        n_obs=int(assignment.n_clones.shape[0]),
        n_clones=assignment.n_clones,
        convergence_by_stage=convergence_by_stage,
        training_stages=training_stages,
        objective_name=objective_name,
        objective_scores=objective_scores_arr,
        decode_log2_lik=decode_log2_lik_arr,
        ground_truth_bmu=eval_arrays["ground_truth_bmu"],
        valid_transition_stats=eval_arrays["valid_transition_stats"],
        training_summary={
            "training_mode": training_mode,
            "batch_size": None if training_mode == "full" else int(config.batch_size),
            "online_lambda": None if training_mode == "full" else float(config.online_lambda),
            "mercury_prior_weight": None if training_mode == "full" else float(config.mercury_prior_weight),
            "online_batches": None if training_mode == "full" else int(len(convergence_by_stage.get("em", []))),
            "online_batches_per_epoch": None if training_mode == "full" else int(online_batches_per_epoch),
            "online_sample_budget": None if training_mode == "full" else int(online_sample_budget),
            "bootstrap_source": "mercury_latent_graph",
            "num_mercury_graph_nodes": int(latent_graph.n),
            "num_fallback_clone_states": int(np.count_nonzero(assignment.fallback_state_mask)),
        },
        mapped_bmu=mapped_bmu,
        latent_node_count=latent_node_count,
        clone_states=clone_states,
        observation_prediction_eval=observation_prediction_eval,
        state_to_node=assignment.state_to_node,
        state_to_observation=assignment.state_to_observation,
        fallback_state_mask=assignment.fallback_state_mask,
        mercury_states_path=mercury_states_path,
        mercury_latent_graph_path=mercury_latent_graph_path,
        source_states_path=source_states_path,
        source_graph_path=source_graph_path,
    )
    if bool(config.paper_precision_enabled) and str(config.paper_precision_mode).strip().lower() == "final":
        _write_hybrid_paper_precision_snapshot(
            config=config,
            stage="final",
            iteration=int(last_em_iteration) if int(last_em_iteration) > 0 else int(config.n_iter),
            step=int(last_em_step) if int(last_em_step) > 0 else int(config.n_iter),
            observed_samples=int(last_em_step) if int(last_em_step) > 0 else int(config.n_iter),
            model_npz={"T": np.asarray(model.T), "n_clones": assignment.n_clones},
            obs_uniques=np.asarray(training["obs_values"]),
            action_uniques=np.asarray(training["action_values"]),
        )
    if progress_callback is not None:
        progress_callback(stage="native_eval", current=1, total=1, message="Native eval complete")
    return states_path


__all__ = ["MercuryCSCGHybridConfig", "run_mercury_cscg_hybrid", "write_hybrid_paper_precision_payload"]
