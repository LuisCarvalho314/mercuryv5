from __future__ import annotations

from typing import Any
from pathlib import Path

import numpy as np
from tqdm import tqdm

from ...infrastructure.paper_precision import (
    DEFAULT_PAPER_PRECISION_NUM_WALKS,
    DEFAULT_PAPER_PRECISION_WALK_LENGTH,
    cartesian_state_ids,
    compute_fixed_map_precision_details,
    concatenate_walk_series,
    generate_random_start_walks,
    map_values_with_reference,
    resolve_dataset_parquet_from_bundle,
    write_paper_precision_artifact,
)
from ...io_parquet import load_level_parquet, ParquetConfig
from ...save_results import read_bundle_meta
from .artifacts import capacity_run_id
from .prepare import filter_valid_pocml_rows, import_pocml_modules
from .train import prepare_model_for_single_sequence_eval


def _rollout_predicted_observation(
    *,
    model: Any,
    obs_idx: np.ndarray,
    action_idx: np.ndarray,
    t: int,
    h: int,
    n_obs: int,
    n_actions: int,
    torch_module: Any,
) -> int:
    model.init_time()
    oh_start = torch_module.nn.functional.one_hot(torch_module.tensor([int(obs_idx[t])]), num_classes=n_obs).to(
        torch_module.float32
    )
    model.init_state(obs=oh_start)
    for k in range(h):
        a_k = int(action_idx[t + k])
        oh_a = torch_module.nn.functional.one_hot(torch_module.tensor([a_k]), num_classes=n_actions).to(torch_module.float32)
        hd_state = model.update_state(oh_a)
        oh_u_next = model.get_expected_state(hd_state)
        if k < h - 1:
            model.u = oh_u_next
            model.clean_up()
            model.inc_time()
            continue
        oh_o_pred = model.get_obs_from_memory(oh_u_next)
        return int(torch_module.argmax(oh_o_pred, dim=1).item())
    raise RuntimeError("POCML rollout did not produce a predicted observation.")


def compute_direct_observation_prediction_accuracy(
    *,
    model: Any,
    obs_idx: np.ndarray,
    action_idx: np.ndarray,
    horizons: list[int],
    n_obs: int,
    n_actions: int,
    torch_module: Any,
    show_progress: bool = False,
    progress_desc: str | None = None,
) -> dict[str, float]:
    accuracies: dict[str, float] = {}
    for horizon in horizons:
        h = int(horizon)
        if h <= 0:
            continue
        total = 0
        correct = 0
        iterator = range(0, len(obs_idx) - h)
        if show_progress:
            iterator = tqdm(iterator, total=max(0, len(obs_idx) - h), desc=(progress_desc or f"POCML Direct Eval n{h}"), position=0)
        for t in iterator:
            pred_obs = _rollout_predicted_observation(
                model=model,
                obs_idx=obs_idx,
                action_idx=action_idx,
                t=t,
                h=h,
                n_obs=n_obs,
                n_actions=n_actions,
                torch_module=torch_module,
            )
            if pred_obs == int(obs_idx[t + h]):
                correct += 1
            total += 1
        accuracies[f"n{h}"] = (float(correct) / float(total)) if total > 0 else 0.0
    return accuracies


def compute_ground_truth_observation_prediction_accuracy(
    *,
    model: Any,
    obs_idx: np.ndarray,
    action_idx: np.ndarray,
    ground_truth_bmu: np.ndarray,
    obs_to_gt: np.ndarray,
    horizons: list[int],
    n_obs: int,
    n_actions: int,
    torch_module: Any,
    show_progress: bool = False,
    progress_desc: str | None = None,
) -> dict[str, float]:
    accuracies: dict[str, float] = {}
    for horizon in horizons:
        h = int(horizon)
        if h <= 0:
            continue
        total = 0
        correct = 0
        iterator = range(0, len(obs_idx) - h)
        if show_progress:
            iterator = tqdm(iterator, total=max(0, len(obs_idx) - h), desc=(progress_desc or f"POCML GT Eval n{h}"), position=0)
        for t in iterator:
            pred_obs = _rollout_predicted_observation(
                model=model,
                obs_idx=obs_idx,
                action_idx=action_idx,
                t=t,
                h=h,
                n_obs=n_obs,
                n_actions=n_actions,
                torch_module=torch_module,
            )
            pred_gt = int(obs_to_gt[pred_obs])
            if pred_gt == int(ground_truth_bmu[t + h]):
                correct += 1
            total += 1
        accuracies[f"n{h}"] = (float(correct) / float(total)) if total > 0 else 0.0
    return accuracies


def evaluate_pocml_sequence(
    *,
    model: Any,
    obs_idx: np.ndarray,
    action_idx: np.ndarray,
    n_obs: int,
    n_actions: int,
    torch_module: Any,
    functional_module: Any,
    show_progress: bool = False,
    progress_desc: str | None = None,
    progress_callback=None,
) -> dict[str, Any]:
    predicted_obs: list[int] = [int(obs_idx[0])]
    latent_proxy_state_ids: list[int] = []
    sensory_proxy_state_ids: list[int] = []
    next_obs_correct = 0
    next_obs_total = 0
    next_obs_confidences: list[float] = []
    trajectory_log_likelihood = 0.0

    model.init_time()
    oh_o_first = functional_module.one_hot(torch_module.tensor([int(obs_idx[0])]), num_classes=n_obs).to(torch_module.float32)
    model.init_state(obs=oh_o_first)
    initial_state_id = int(torch_module.argmax(model.u, dim=1).item())
    latent_proxy_state_ids.append(initial_state_id)
    sensory_proxy_state_ids.append(initial_state_id)

    iterator = range(action_idx.shape[0])
    if show_progress:
        iterator = tqdm(iterator, total=int(action_idx.shape[0]), desc=(progress_desc or "POCML Native Eval"), position=0)
    for step in iterator:
        action_i = int(action_idx[step])
        next_obs_i = int(obs_idx[step + 1])
        oh_a = functional_module.one_hot(torch_module.tensor([action_i]), num_classes=n_actions).to(torch_module.float32)
        oh_o_next = functional_module.one_hot(torch_module.tensor([next_obs_i]), num_classes=n_obs).to(torch_module.float32)
        hd_state = model.update_state(oh_a)
        oh_u_next = model.get_expected_state(hd_state)
        latent_proxy_state_ids.append(int(torch_module.argmax(oh_u_next, dim=1).item()))
        oh_o_next_pred = model.get_obs_from_memory(oh_u_next)
        pred_next_obs = int(torch_module.argmax(oh_o_next_pred, dim=1).item())
        predicted_obs.append(pred_next_obs)
        next_obs_total += 1
        if pred_next_obs == next_obs_i:
            next_obs_correct += 1
        prob_true = float(oh_o_next_pred[0, next_obs_i].item())
        next_obs_confidences.append(prob_true)
        trajectory_log_likelihood += float(np.log(max(prob_true, 1e-12)))
        model.update_state_given_obs(oh_o_next)
        sensory_proxy_state_ids.append(int(torch_module.argmax(model.u, dim=1).item()))
        model.clean_up()
        model.inc_time()
        if progress_callback is not None and (((step + 1) % 250) == 0 or (step + 1) == int(action_idx.shape[0])):
            progress_callback(current=step + 1, total=int(action_idx.shape[0]), message="native eval")

    next_obs_accuracy = (float(next_obs_correct) / float(next_obs_total)) if next_obs_total > 0 else 0.0
    next_obs_confidence_mean = float(np.mean(next_obs_confidences)) if next_obs_confidences else 0.0
    next_obs_confidence_std = float(np.std(next_obs_confidences)) if next_obs_confidences else 0.0
    trajectory_log_likelihood_mean = float(trajectory_log_likelihood) / float(next_obs_total) if next_obs_total > 0 else 0.0
    return {
        "predicted_obs": np.asarray(predicted_obs, dtype=np.int64),
        "latent_proxy_state_ids": np.asarray(latent_proxy_state_ids, dtype=np.int32),
        "sensory_proxy_state_ids": np.asarray(sensory_proxy_state_ids, dtype=np.int32),
        "direct_observation_prediction_accuracy": {"n1": next_obs_accuracy},
        "next_obs_confidence_mean": next_obs_confidence_mean,
        "next_obs_confidence_std": next_obs_confidence_std,
        "trajectory_log_likelihood_sum": float(trajectory_log_likelihood),
        "trajectory_log_likelihood_mean": trajectory_log_likelihood_mean,
        "next_obs_steps": int(next_obs_total),
    }


def compute_pocml_precision(*, config) -> None:
    from scripts.analysis.compute_precision import compute_metrics_for_states

    states_path = Path(config.output_root) / f"level={config.level}" / f"sensor={config.sensor}"
    if config.sensor == "cardinal distance":
        states_path = states_path / f"range={config.sensor_range}"
    compute_metrics_for_states(
        states_path=states_path / f"{config.run_id}_states.parquet",
        out_dir=Path(config.output_root).parents[1] / "metrics" / "pocml",
        run_id=config.run_id,
        window_length=int(config.window_length),
        precision_columns={"capacity_precision": "latent_proxy_state_id"},
        method_metadata={
            "method": "pocml",
            "level": config.level,
            "latent_sensor": config.sensor,
            "latent_sensor_range": config.sensor_range,
        },
    )


def load_pocml_checkpoint(*, config, n_obs: int, n_actions: int, run_id: str | None = None, n_states: int | None = None) -> tuple[Any, Any, Any]:
    model_mod, _ = import_pocml_modules(Path.cwd())
    import torch

    checkpoint_path = Path(config.output_root) / f"level={config.level}" / f"sensor={config.sensor}"
    if config.sensor == "cardinal distance":
        checkpoint_path = checkpoint_path / f"range={config.sensor_range}"
    checkpoint_path = checkpoint_path / f"{(run_id or config.run_id)}_pocml_model.pt"
    model = model_mod.POCML(
        n_obs=int(n_obs),
        n_states=int(n_states if n_states is not None else (config.n_states or n_obs)),
        n_actions=int(n_actions),
        state_dim=int(config.state_dim),
        random_feature_dim=int(config.random_feature_dim),
        alpha=float(config.alpha),
        batch_size=1,
        memory_bias=bool(config.memory_bias),
    )
    try:
        state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except TypeError:  # pragma: no cover
        state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict)
    return prepare_model_for_single_sequence_eval(model), model_mod, torch


def compute_pocml_paper_precision(*, config) -> Path:
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
    train_obs, train_actions, _, _ = filter_valid_pocml_rows(
        observations=np.asarray(loaded.observations),
        actions=np.asarray(loaded.actions),
        collisions=np.asarray(loaded.collisions, dtype=bool),
        valid_trajectories_only=bool(config.valid_trajectories_only),
    )
    obs_uniques = np.unique(train_obs, axis=0)
    action_uniques = np.unique(train_actions, axis=0)
    resolved_capacities = resolve_pocml_precision_capacities(config=config, states_path=states_path)
    primary_capacity = int(resolved_capacities[0])
    metrics_by_capacity: dict[str, dict[str, Any]] = {}
    for capacity in resolved_capacities:
        model, _, torch_module = load_pocml_checkpoint(
            config=config,
            n_obs=int(obs_uniques.shape[0]),
            n_actions=int(action_uniques.shape[0]),
            run_id=capacity_run_id(base_run_id=config.run_id, capacity=int(capacity), primary_capacity=primary_capacity),
            n_states=int(capacity),
        )
        metrics_by_capacity[str(int(capacity))] = compute_pocml_paper_precision_metrics_from_model(
            model=model,
            config=config,
            obs_uniques=obs_uniques,
            action_uniques=action_uniques,
            torch_module=torch_module,
        )
    return write_pocml_paper_precision_payload(
        config=config,
        metrics_by_capacity=metrics_by_capacity,
        history_by_capacity={str(int(capacity)): [] for capacity in resolved_capacities},
        mode=str(config.paper_precision_mode),
        resolved_capacities=resolved_capacities,
        primary_capacity=primary_capacity,
    )


def compute_pocml_paper_precision_metrics_from_model(
    *,
    model: Any,
    config,
    obs_uniques: np.ndarray,
    action_uniques: np.ndarray,
    torch_module: Any,
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
    total_walks = int(len(walks))
    for index, walk in enumerate(walks, start=1):
        filtered_obs, filtered_actions, _, _ = filter_valid_pocml_rows(
            observations=walk.observations,
            actions=walk.actions,
            collisions=walk.collisions,
            valid_trajectories_only=bool(config.valid_trajectories_only),
        )
        obs_idx = map_values_with_reference(reference_uniques=obs_uniques, values=filtered_obs)
        act_idx_full = map_values_with_reference(reference_uniques=action_uniques, values=filtered_actions)
        if obs_idx.shape[0] < 2 or act_idx_full.shape[0] < 2:
            continue
        walk_eval = evaluate_pocml_sequence(
            model=model,
            obs_idx=obs_idx.astype(np.int64, copy=False),
            action_idx=act_idx_full[:-1].astype(np.int64, copy=False),
            n_obs=int(obs_uniques.shape[0]),
            n_actions=int(action_uniques.shape[0]),
            torch_module=torch_module,
            functional_module=torch_module.nn.functional,
        )
        gt_collisions = np.asarray(walk.collisions, dtype=bool)
        gt = cartesian_state_ids(
            np.asarray(walk.cartesian_observations)[(~gt_collisions) if bool(config.valid_trajectories_only) else slice(None)]
        )
        inferred_walks.append(walk_eval["latent_proxy_state_ids"][: len(gt)].astype(np.int64, copy=False))
        gt_walks.append(gt.astype(np.int64, copy=False))
        if progress_callback is not None and ((index % 5) == 0 or index == total_walks):
            progress_callback(current=index, total=total_walks, message="paper precision")
    return compute_fixed_map_precision_details(
        inferred_states=concatenate_walk_series(inferred_walks),
        ground_truth_states=concatenate_walk_series(gt_walks),
        metric_name="capacity_precision",
    )


def resolve_pocml_precision_capacities(*, config, states_path: Path | None = None) -> list[int]:
    if config.precision_capacities:
        ordered: list[int] = []
        for value in config.precision_capacities:
            normalized = int(value)
            if normalized not in ordered:
                ordered.append(normalized)
        return ordered
    if config.n_states is not None:
        return [int(config.n_states)]
    candidate_states_path = states_path
    if candidate_states_path is None:
        candidate_states_path = Path(config.output_root) / f"level={config.level}" / f"sensor={config.sensor}"
        if config.sensor == "cardinal distance":
            candidate_states_path = candidate_states_path / f"range={config.sensor_range}"
        candidate_states_path = candidate_states_path / f"{config.run_id}_states.parquet"
    meta = read_bundle_meta(candidate_states_path)
    pocml_eval = ((meta.run_parameters or {}).get("pocml_eval") or {}) if meta is not None else {}
    primary_capacity = pocml_eval.get("primary_precision_capacity")
    if primary_capacity is not None:
        return [int(primary_capacity)]
    raise RuntimeError("Could not resolve POCML precision capacities from config or bundle metadata.")


def write_pocml_paper_precision_payload(
    *,
    config,
    metrics_by_capacity: dict[str, dict[str, Any]],
    history_by_capacity: dict[str, list[dict[str, Any]]],
    mode: str,
    resolved_capacities: list[int],
    primary_capacity: int,
) -> Path:
    primary_history = list(history_by_capacity.get(str(int(primary_capacity))) or [])
    observed_samples_per_epoch = None
    if primary_history:
        latest = primary_history[-1]
        if int(latest.get("step", 0)) > 0 and "observed_samples" in latest:
            observed_samples_per_epoch = int(latest["observed_samples"]) // int(latest["step"])
    payload = {
        "method": "pocml",
        "schema": "paper_precision_v2",
        "protocol": {
            "num_walks": int(config.paper_precision_num_walks),
            "walk_length": int(config.paper_precision_walk_length),
            "rand_prob": float(config.paper_precision_rand_prob),
            "base_seed": int(config.paper_precision_seed),
            "valid_trajectories_only": bool(config.valid_trajectories_only),
            "inferred_state_series": "latent_proxy_state_id",
            "ground_truth_state_series": "cartesian_state_id",
            "capacity_dependent": True,
            "mode": mode,
            "eval_interval": int(config.paper_precision_eval_interval),
            "num_points": config.paper_precision_num_points,
            "resolved_capacities": [int(value) for value in resolved_capacities],
            "primary_capacity": int(primary_capacity),
            "x_axis_unit": "observed_samples",
            "observed_samples_per_epoch": observed_samples_per_epoch,
        },
        "metrics_by_capacity": metrics_by_capacity,
        "history_by_capacity": history_by_capacity,
    }
    out_dir = Path(config.output_root).parents[1] / "metrics" / "pocml"
    return write_paper_precision_artifact(path=out_dir / f"{config.run_id}_paper_precision.json", payload=payload)
