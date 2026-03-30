from __future__ import annotations

import inspect
from typing import Any, Callable, Optional

import numpy as np

from .artifacts import capacity_run_id, output_dir, save_pocml_checkpoint, save_pocml_embeddings, write_pocml_artifacts
from .config import POCMLConfig
from .evaluate import (
    compute_pocml_paper_precision_metrics_from_model,
    compute_pocml_paper_precision,
    compute_pocml_precision,
    compute_direct_observation_prediction_accuracy,
    compute_ground_truth_observation_prediction_accuracy,
    evaluate_pocml_sequence,
    resolve_pocml_precision_capacities,
    write_pocml_paper_precision_payload,
)
from .prepare import fit_obs_to_gt_map, fit_state_to_gt_map
from .train import train_pocml_model
from ...infrastructure.runtime import progress_enabled


def _supports_keyword_argument(func: Callable[..., Any], argument_name: str) -> bool:
    parameters = inspect.signature(func).parameters.values()
    return any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD or parameter.name == argument_name
        for parameter in parameters
    )


def _normalize_explicit_capacities(config: POCMLConfig) -> list[int]:
    capacities = config.precision_capacities or []
    normalized: list[int] = []
    for value in capacities:
        capacity = int(value)
        if capacity not in normalized:
            normalized.append(capacity)
    return normalized


def _run_single_pocml_capacity(
    *,
    config: POCMLConfig,
    precision_capacities: list[int],
    progress_callback: Optional[Callable[..., None]] = None,
) -> dict[str, Any]:
    show_progress = progress_enabled(default=True)
    paper_precision_history: list[dict[str, Any]] = []

    def _paper_precision_callback(*, model, step: int, observed_samples: int, obs_values, action_values, torch_module) -> None:
        paper_precision_history.append(
            {
                "step": int(step),
                "observed_samples": int(observed_samples),
                **compute_pocml_paper_precision_metrics_from_model(
                    model=model,
                    config=config,
                    obs_uniques=np.asarray(obs_values),
                    action_uniques=np.asarray(action_values),
                    torch_module=torch_module,
                ),
            }
        )

    paper_precision_mode = str(config.paper_precision_mode).strip().lower()
    trained = train_pocml_model(
        config=config,
        paper_precision_callback=(
            _paper_precision_callback if paper_precision_mode == "per_iteration" else None
        ),
        progress_callback=progress_callback,
    )
    if progress_callback is not None:
        progress_callback(stage="native_eval", current=0, total=1, message="Starting native eval")
    model = trained["model"]
    torch_module = trained["torch"]
    native_eval = evaluate_pocml_sequence(
        model=model,
        obs_idx=trained["obs_idx"],
        action_idx=trained["actions_for_eval"],
        n_obs=trained["n_obs"],
        n_actions=trained["n_actions"],
        torch_module=torch_module,
        functional_module=trained["functional"],
        show_progress=show_progress,
        progress_desc="POCML Native Eval",
        progress_callback=(
            (lambda **kwargs: progress_callback(stage="native_eval", **kwargs))
            if progress_callback is not None
            else None
        ),
    )
    obs_to_gt = fit_obs_to_gt_map(trained["obs_idx"], trained["ground_truth_bmu"])
    state_sequence_np = native_eval["sensory_proxy_state_ids"].astype(np.int32, copy=False)
    sensory_proxy_state_ids_np = native_eval["sensory_proxy_state_ids"][: len(trained["ground_truth_bmu"])].astype(
        np.int32, copy=False
    )
    latent_proxy_state_ids_np = native_eval["latent_proxy_state_ids"][: len(trained["ground_truth_bmu"])].astype(
        np.int32, copy=False
    )
    sensory_state_to_gt = fit_state_to_gt_map(
        sensory_proxy_state_ids_np, trained["ground_truth_bmu"][: len(sensory_proxy_state_ids_np)]
    )
    latent_state_to_gt = fit_state_to_gt_map(
        latent_proxy_state_ids_np, trained["ground_truth_bmu"][: len(latent_proxy_state_ids_np)]
    )
    sensory_bmu = sensory_state_to_gt[sensory_proxy_state_ids_np]
    latent_bmu = latent_state_to_gt[latent_proxy_state_ids_np]
    latent_node_count = np.full(len(trained["ground_truth_bmu"]), fill_value=trained["n_states"], dtype=np.int32)
    if progress_callback is not None:
        progress_callback(stage="direct_eval", current=0, total=1, message="Starting direct eval")
    direct_observation_prediction_accuracy = {
        **native_eval["direct_observation_prediction_accuracy"],
        **compute_direct_observation_prediction_accuracy(
            model=model,
            obs_idx=trained["obs_idx"],
            action_idx=trained["actions_for_eval"],
            horizons=[3, 5],
            n_obs=trained["n_obs"],
            n_actions=trained["n_actions"],
            torch_module=torch_module,
            show_progress=show_progress,
        ),
    }
    if progress_callback is not None:
        progress_callback(stage="gt_eval", current=0, total=1, message="Starting GT eval")
    n_step_observation_prediction_accuracy = compute_ground_truth_observation_prediction_accuracy(
        model=model,
        obs_idx=trained["obs_idx"],
        action_idx=trained["actions_for_eval"],
        ground_truth_bmu=trained["ground_truth_bmu"],
        obs_to_gt=obs_to_gt,
        horizons=[1, 3, 5],
        n_obs=trained["n_obs"],
        n_actions=trained["n_actions"],
        torch_module=torch_module,
        show_progress=show_progress,
    )
    obs_given_state = torch_module.softmax(10 * model.M.detach().cpu(), dim=1).numpy()[0]
    state_obs_from_memory = obs_given_state.argmax(axis=0).astype(np.int32, copy=False)
    output_dir_path = output_dir(config.output_root, config.level, config.sensor, config.sensor_range)
    embeddings_path = save_pocml_embeddings(
        output_dir_path=output_dir_path,
        run_id=config.run_id,
        q_matrix=model.Q.detach().cpu().numpy(),
        v_matrix=model.V.detach().cpu().numpy(),
        obs_to_gt=obs_to_gt,
        state_obs_from_memory=state_obs_from_memory,
        state_sequence=state_sequence_np,
        action_sequence=trained["actions_for_eval"][: max(0, len(state_sequence_np) - 1)].astype(np.int32, copy=False),
        observation_sequence=trained["obs_idx"][: len(state_sequence_np)].astype(np.int32, copy=False),
    )
    checkpoint_path = save_pocml_checkpoint(
        output_dir_path=output_dir_path,
        run_id=config.run_id,
        torch_module=torch_module,
        model=model,
    )
    resolved_precision_capacities = [int(value) for value in (precision_capacities or [int(trained["n_states"])])]
    pocml_eval: dict[str, Any] = {
        "n_step_observation_prediction_accuracy": n_step_observation_prediction_accuracy,
        "direct_observation_prediction_accuracy": direct_observation_prediction_accuracy,
        "valid_transition_filter": bool(config.valid_trajectories_only),
        **trained["valid_transition_stats"],
        "paper_precision_metric_name": "paper_precision_by_capacity",
        "precision_state_series": "latent_proxy_state_id",
        "primary_precision_capacity": int(trained["n_states"]),
        "precision_capacities": resolved_precision_capacities,
        "next_obs_confidence_mean": native_eval["next_obs_confidence_mean"],
        "next_obs_confidence_std": native_eval["next_obs_confidence_std"],
        "trajectory_log_likelihood_sum": native_eval["trajectory_log_likelihood_sum"],
        "trajectory_log_likelihood_mean": native_eval["trajectory_log_likelihood_mean"],
        "next_obs_steps": native_eval["next_obs_steps"],
    }
    states_path = write_pocml_artifacts(
        config=config,
        dataset_parquet_name=trained["dataset_parquet_name"],
        dataset_metadata=trained["dataset_metadata"],
        n_actions=trained["n_actions"],
        valid_transition_stats=trained["valid_transition_stats"],
        effective_batch_size=trained["effective_batch_size"],
        train_trajectories_count=len(trained["train_trajectories"]),
        training_drop_last=trained["training_drop_last"],
        embeddings_path=embeddings_path,
        checkpoint_path=checkpoint_path,
        ground_truth_bmu=trained["ground_truth_bmu"],
        sensory_bmu=sensory_bmu,
        latent_bmu=latent_bmu,
        latent_node_count=latent_node_count,
        sensory_proxy_state_ids=sensory_proxy_state_ids_np,
        latent_proxy_state_ids=latent_proxy_state_ids_np,
        pocml_eval=pocml_eval,
        precision_capacities=resolved_precision_capacities,
    )
    final_metrics = compute_pocml_paper_precision_metrics_from_model(
        model=model,
        config=config,
        obs_uniques=trained["obs_values"],
        action_uniques=trained["action_values"],
        torch_module=torch_module,
        progress_callback=(
            (lambda **kwargs: progress_callback(stage="paper_precision", **kwargs))
            if progress_callback is not None
            else None
        ),
    )
    if paper_precision_mode == "per_iteration" and (
        not paper_precision_history or int(paper_precision_history[-1]["step"]) != int(config.epochs)
    ):
        paper_precision_history.append(
            {
                "step": int(config.epochs),
                "observed_samples": int(config.epochs) * int(trained["observed_samples_per_epoch"]),
                **final_metrics,
            }
        )
    return {
        "states_path": states_path,
        "resolved_capacity": int(trained["n_states"]),
        "paper_precision_metrics": final_metrics,
        "paper_precision_history": paper_precision_history,
        "checkpoint_path": checkpoint_path,
        "embeddings_path": embeddings_path,
    }


def run_pocml(config: POCMLConfig, *, progress_callback: Optional[Callable[..., None]] = None):
    explicit_capacities = _normalize_explicit_capacities(config)
    requested_capacities = explicit_capacities if explicit_capacities else [config.n_states]
    primary_requested_capacity = requested_capacities[0]
    primary_config = config.model_copy(update={"n_states": primary_requested_capacity})
    primary_kwargs = {
        "config": primary_config,
        "precision_capacities": (
            [int(value) for value in explicit_capacities]
            if explicit_capacities
            else ([int(primary_requested_capacity)] if primary_requested_capacity is not None else [])
        ),
    }
    if progress_callback is not None and _supports_keyword_argument(_run_single_pocml_capacity, "progress_callback"):
        primary_kwargs["progress_callback"] = progress_callback
    primary_result = _run_single_pocml_capacity(**primary_kwargs)
    resolved_capacities = [int(primary_result["resolved_capacity"])] if not explicit_capacities else [int(value) for value in explicit_capacities]
    primary_capacity = int(primary_result["resolved_capacity"])
    if explicit_capacities and primary_capacity != int(explicit_capacities[0]):
        raise RuntimeError("Primary POCML precision capacity did not match the requested explicit capacity.")

    results_by_capacity: dict[int, dict[str, Any]] = {primary_capacity: primary_result}
    for capacity in resolved_capacities[1:]:
        capacity_id = capacity_run_id(base_run_id=config.run_id, capacity=int(capacity), primary_capacity=primary_capacity)
        capacity_kwargs = {
            "config": config.model_copy(update={"run_id": capacity_id, "n_states": int(capacity)}),
            "precision_capacities": resolved_capacities,
        }
        if progress_callback is not None and _supports_keyword_argument(_run_single_pocml_capacity, "progress_callback"):
            capacity_kwargs["progress_callback"] = progress_callback
        capacity_result = _run_single_pocml_capacity(**capacity_kwargs)
        results_by_capacity[int(capacity)] = capacity_result

    if not explicit_capacities:
        resolved_capacities = resolve_pocml_precision_capacities(config=primary_config.model_copy(update={"n_states": primary_capacity}))
    resolved_capacities = [int(value) for value in resolved_capacities]

    primary_state_path = primary_result["states_path"]
    if bool(config.paper_precision_enabled):
        write_pocml_paper_precision_payload(
            config=config,
            metrics_by_capacity={
                str(capacity): dict(results_by_capacity[int(capacity)]["paper_precision_metrics"])
                for capacity in resolved_capacities
            },
            history_by_capacity={
                str(capacity): list(results_by_capacity[int(capacity)]["paper_precision_history"])
                for capacity in resolved_capacities
            },
            mode=str(config.paper_precision_mode),
            resolved_capacities=resolved_capacities,
            primary_capacity=primary_capacity,
        )
    compute_pocml_precision(config=config)
    return primary_state_path


__all__ = ["POCMLConfig", "compute_pocml_paper_precision", "compute_pocml_precision", "evaluate_pocml_sequence", "run_pocml"]
