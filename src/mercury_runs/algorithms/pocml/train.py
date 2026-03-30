from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .prepare import (
    build_train_trajectories,
    filter_valid_pocml_rows,
    import_pocml_modules,
    load_dataset_arrays,
    load_ground_truth_states,
    to_indices,
)
from ...infrastructure.paper_precision import resolve_eval_checkpoints


def prepare_model_for_single_sequence_eval(model: Any) -> Any:
    if getattr(model, "batch_size", 1) == 1:
        return model
    model.batch_size = 1
    memory = getattr(model, "M", None)
    if memory is not None:
        model.init_memory(memory=memory[:1].detach().clone())
    return model


def _trajectory_length(trajectory: Any) -> int:
    shape = getattr(trajectory, "shape", None)
    if shape is not None and len(shape) > 0:
        return int(shape[0])
    array = getattr(trajectory, "arr", None)
    if array is not None:
        return int(np.asarray(array).shape[0])
    return int(len(trajectory))


def train_pocml_model(*, config, paper_precision_callback=None, progress_callback=None):
    model_mod, trainer_mod = import_pocml_modules(Path.cwd())
    try:
        import torch
        from torch.utils.data import DataLoader
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("POCML baseline requires torch. Install with `pip install torch`.") from exc

    observations, actions, collisions, dataset_metadata, dataset_parquet_name = load_dataset_arrays(
        datasets_root=config.datasets_root,
        level=config.level,
        sensor=config.sensor,
        sensor_range=config.sensor_range,
        dataset_select=config.dataset_select,
        dataset_run_id=config.dataset_run_id,
    )
    if observations.shape[0] < 3:
        raise ValueError("POCML baseline requires at least 3 observation rows.")

    ground_truth_bmu = load_ground_truth_states(config.ground_truth_states_parquet)
    observations, actions, filtered_ground_truth_bmu, valid_transition_stats = filter_valid_pocml_rows(
        observations=observations,
        actions=actions,
        collisions=collisions,
        ground_truth_bmu=ground_truth_bmu,
        valid_trajectories_only=bool(config.valid_trajectories_only),
    )
    ground_truth_bmu = np.asarray(filtered_ground_truth_bmu, dtype=np.int32)
    obs_idx, obs_values = to_indices(observations)
    act_idx, action_values = to_indices(actions)
    length = min(int(ground_truth_bmu.shape[0]), int(obs_idx.shape[0]), int(act_idx.shape[0]))
    ground_truth_bmu = ground_truth_bmu[:length]
    obs_idx = obs_idx[:length]
    act_idx = act_idx[:length]
    if length < 3:
        raise ValueError("POCML baseline could not align enough samples between dataset and ground truth.")

    o_pre = obs_idx[:-1]
    o_next = obs_idx[1:]
    a = act_idx[:-1]
    gt_pre = ground_truth_bmu[:-1].astype(np.int64, copy=False)
    gt_next = ground_truth_bmu[1:].astype(np.int64, copy=False)
    node_pre = gt_pre if bool(config.use_ground_truth_node_ids) else o_pre
    node_next = gt_next if bool(config.use_ground_truth_node_ids) else o_next
    n_obs = int(obs_idx.max()) + 1
    n_actions = int(act_idx.max()) + 1
    n_states = int(config.n_states or n_obs)
    if bool(config.use_ground_truth_node_ids):
        n_states = max(n_states, int(ground_truth_bmu.max()) + 1)
    if bool(config.memory_bias) and n_states > n_obs:
        raise ValueError(
            "POCML memory_bias=True is incompatible with n_states > n_obs in the vendored upstream implementation. "
            f"Resolved n_states={n_states}, n_obs={n_obs}. Use --pocml_memory_bias False or choose capacities <= {n_obs}."
        )

    train_trajectories = build_train_trajectories(
        o_pre=o_pre,
        a=a,
        o_next=o_next,
        node_pre=node_pre,
        node_next=node_next,
        trajectory_length=int(config.trajectory_length),
        max_trajectories=config.max_trajectories,
        torch_module=torch,
    )
    effective_batch_size = min(int(config.batch_size), max(1, len(train_trajectories)))
    training_drop_last = len(train_trajectories) > effective_batch_size
    train_loader_kwargs = {
        "batch_size": effective_batch_size,
        "shuffle": True,
        "drop_last": training_drop_last,
    }
    try:
        train_loader = DataLoader(train_trajectories, **train_loader_kwargs)
    except TypeError:
        train_loader_kwargs.pop("drop_last", None)
        train_loader = DataLoader(train_trajectories, **train_loader_kwargs)

    model = model_mod.POCML(
        n_obs=n_obs,
        n_states=n_states,
        n_actions=n_actions,
        state_dim=int(config.state_dim),
        batch_size=effective_batch_size,
        random_feature_dim=int(config.random_feature_dim),
        alpha=float(config.alpha),
        memory_bias=bool(config.memory_bias),
    )
    trainer = trainer_mod.POCMLTrainer(
        model=model,
        train_loader=train_loader,
        device=torch.device("cpu"),
        lr_Q=float(config.lr_q),
        lr_V=float(config.lr_v),
        lr_all=float(config.lr_all),
        lr_M=float(config.lr_m),
        reg_M=float(config.reg_m),
        max_iter_M=int(config.max_iter_m),
        eps_M=float(config.eps_m),
        log=False,
    )
    trajectory_lengths = [_trajectory_length(trajectory) for trajectory in train_trajectories]
    batches_per_epoch = (
        (len(train_trajectories) // effective_batch_size)
        if training_drop_last
        else int(np.ceil(len(train_trajectories) / effective_batch_size))
    )
    if training_drop_last:
        trajectories_seen_per_epoch = batches_per_epoch * effective_batch_size
        observed_samples_per_epoch = int(sum(trajectory_lengths[:trajectories_seen_per_epoch]))
    else:
        observed_samples_per_epoch = int(sum(trajectory_lengths))
    if str(config.paper_precision_mode).strip().lower() == "per_iteration" and paper_precision_callback is not None:
        paper_precision_checkpoints = set(
            resolve_eval_checkpoints(
                total_units=int(config.epochs),
                num_points=config.paper_precision_num_points,
                eval_interval=int(config.paper_precision_eval_interval),
            )
        )
        loss_epochs: list[np.ndarray] = []
        best_model = None
        best_loss = float("inf")
        for epoch in range(int(config.epochs)):
            epoch_loss_record, epoch_model = trainer.train(epochs=1)
            epoch_loss_record = np.asarray(epoch_loss_record, dtype=np.float64)
            if epoch_loss_record.size == 0 or not np.isfinite(epoch_loss_record).all() or epoch_model is None:
                raise RuntimeError("POCML training failed to converge to a finite model. No baseline artifacts were written.")
            loss_epochs.append(epoch_loss_record.reshape(1, -1))
            epoch_loss = float(epoch_loss_record.mean())
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model = prepare_model_for_single_sequence_eval(epoch_model)
            if progress_callback is not None:
                progress_callback(stage="train", current=epoch + 1, total=int(config.epochs), message=f"epoch {epoch + 1}/{int(config.epochs)}")
            if (epoch + 1) in paper_precision_checkpoints:
                paper_precision_callback(
                    model=prepare_model_for_single_sequence_eval(epoch_model),
                    step=epoch + 1,
                    observed_samples=(epoch + 1) * observed_samples_per_epoch,
                    obs_values=np.asarray(obs_values),
                    action_values=np.asarray(action_values),
                    torch_module=torch,
                )
        loss_record = np.concatenate(loss_epochs, axis=0) if loss_epochs else np.zeros((0, 0), dtype=np.float64)
    elif progress_callback is not None:
        loss_epochs: list[np.ndarray] = []
        best_model = None
        best_loss = float("inf")
        for epoch in range(int(config.epochs)):
            epoch_loss_record, epoch_model = trainer.train(epochs=1)
            epoch_loss_record = np.asarray(epoch_loss_record, dtype=np.float64)
            if epoch_loss_record.size == 0 or not np.isfinite(epoch_loss_record).all() or epoch_model is None:
                raise RuntimeError("POCML training failed to converge to a finite model. No baseline artifacts were written.")
            loss_epochs.append(epoch_loss_record.reshape(1, -1))
            epoch_loss = float(epoch_loss_record.mean())
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model = prepare_model_for_single_sequence_eval(epoch_model)
            progress_callback(stage="train", current=epoch + 1, total=int(config.epochs), message=f"epoch {epoch + 1}/{int(config.epochs)}")
        loss_record = np.concatenate(loss_epochs, axis=0) if loss_epochs else np.zeros((0, 0), dtype=np.float64)
    else:
        loss_record, best_model = trainer.train(epochs=int(config.epochs))
    loss_record = np.asarray(loss_record, dtype=np.float64)
    if loss_record.size == 0 or not np.isfinite(loss_record).all() or best_model is None:
        raise RuntimeError("POCML training failed to converge to a finite model. No baseline artifacts were written.")
    return {
        "torch": torch,
        "functional": torch.nn.functional,
        "model": prepare_model_for_single_sequence_eval(best_model),
        "obs_idx": obs_idx,
        "act_idx": act_idx,
        "obs_values": np.asarray(obs_values),
        "action_values": np.asarray(action_values),
        "ground_truth_bmu": ground_truth_bmu,
        "n_obs": n_obs,
        "n_actions": n_actions,
        "n_states": n_states,
        "dataset_metadata": dataset_metadata,
        "dataset_parquet_name": dataset_parquet_name,
        "valid_transition_stats": valid_transition_stats,
        "train_trajectories": train_trajectories,
        "effective_batch_size": effective_batch_size,
        "training_drop_last": training_drop_last,
        "observed_samples_per_epoch": observed_samples_per_epoch,
        "actions_for_eval": a,
        "loss_record": loss_record,
    }
