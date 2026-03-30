from __future__ import annotations

import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np

from ...save_results import write_bundle_parquet
from ...schemas_results import ResultBundleMeta, SourceDatasetRef


def utc_timestamp() -> str:
    return datetime.datetime.now(datetime.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def output_dir(root: Path, level: int, sensor: str, sensor_range: Optional[int]) -> Path:
    base = root / f"level={level}" / f"sensor={sensor}"
    if sensor == "cardinal distance":
        return base / f"range={sensor_range}"
    return base


def capacity_run_id(*, base_run_id: str, capacity: int, primary_capacity: int) -> str:
    if int(capacity) == int(primary_capacity):
        return base_run_id
    return f"{base_run_id}_capacity={int(capacity)}"


def save_pocml_embeddings(
    *,
    output_dir_path: Path,
    run_id: str,
    q_matrix: np.ndarray,
    v_matrix: np.ndarray,
    obs_to_gt: np.ndarray,
    state_obs_from_memory: np.ndarray,
    state_sequence: np.ndarray,
    action_sequence: np.ndarray,
    observation_sequence: np.ndarray,
) -> Path:
    output_dir_path.mkdir(parents=True, exist_ok=True)
    out_path = output_dir_path / f"{run_id}_embeddings.npz"
    np.savez_compressed(
        out_path,
        Q=q_matrix,
        V=v_matrix,
        obs_to_gt=obs_to_gt,
        state_obs_from_memory=state_obs_from_memory,
        state_sequence=state_sequence,
        action_sequence=action_sequence,
        observation_sequence=observation_sequence,
    )
    return out_path


def save_pocml_checkpoint(*, output_dir_path: Path, run_id: str, torch_module: Any, model: Any) -> Path:
    output_dir_path.mkdir(parents=True, exist_ok=True)
    out_path = output_dir_path / f"{run_id}_pocml_model.pt"
    torch_module.save(model.state_dict(), out_path)
    return out_path


def write_pocml_artifacts(
    *,
    config,
    dataset_parquet_name: str,
    dataset_metadata: dict[str, Any],
    n_actions: int,
    valid_transition_stats: dict[str, int],
    effective_batch_size: int,
    train_trajectories_count: int,
    training_drop_last: bool,
    embeddings_path: Path,
    checkpoint_path: Path,
    ground_truth_bmu: np.ndarray,
    sensory_bmu: np.ndarray,
    latent_bmu: np.ndarray,
    latent_node_count: np.ndarray,
    sensory_proxy_state_ids: np.ndarray,
    latent_proxy_state_ids: np.ndarray,
    pocml_eval: dict[str, Any],
    precision_capacities: list[int],
) -> Path:
    output_dir_path = output_dir(config.output_root, config.level, config.sensor, config.sensor_range)
    meta = ResultBundleMeta(
        run_id=config.run_id,
        timestamp_utc=utc_timestamp(),
        source=SourceDatasetRef(
            level=config.level,
            sensor=config.sensor,
            sensor_range=config.sensor_range,
            select=config.dataset_select if config.dataset_select in {"latest", "run_id"} else "latest",
            dataset_run_id=config.dataset_run_id,
            dataset_parquet_name=dataset_parquet_name,
        ),
        sensory_params={},
        latent_params={},
        action_map_params={"method": "pocml", "n_actions": n_actions},
        memory_length=None,
        run_parameters={
            "baseline_method": "pocml",
            "pocml": config.model_dump(mode="json"),
            "training_data": {
                "mode": "finite_trajectory_chunks",
                "valid_transition_filter": bool(config.valid_trajectories_only),
                **valid_transition_stats,
                "trajectory_length": int(config.trajectory_length),
                "num_trajectories": int(train_trajectories_count),
                "max_trajectories": config.max_trajectories,
                "batch_size": effective_batch_size,
                "drop_last": training_drop_last,
                "observation_mode": "one_hot_observation_ids",
                "node_id_source": ("cartesian_proxy_bmu" if bool(config.use_ground_truth_node_ids) else "observation_ids"),
                "model_selection": "best_model_by_mean_epoch_loss",
                "memory_bias": bool(config.memory_bias),
            },
            "artifacts": {"embeddings_npz": str(embeddings_path), "pocml_model_checkpoint_pt": str(checkpoint_path)},
            "pocml_eval": pocml_eval,
            "bmu_proxy": {
                "method": "posterior_argmax_state",
                "description": (
                    "Per-step proxy BMU uses argmax over the model's state posterior; latent from predictive "
                    "state and sensory from observation-corrected state."
                ),
            },
            "precision_capacities": [int(value) for value in precision_capacities],
        },
        source_dataset_metadata=dataset_metadata,
        notes="baseline=pocml",
    )
    return write_bundle_parquet(
        output_dir=output_dir_path,
        run_id=config.run_id,
        bundle_name="states",
        columns={
            "cartesian_proxy_bmu": ground_truth_bmu,
            "sensory_bmu": sensory_bmu.astype(np.int32, copy=False),
            "latent_bmu": latent_bmu.astype(np.int32, copy=False),
            "latent_node_count": latent_node_count,
            "sensory_proxy_state_id": sensory_proxy_state_ids,
            "latent_proxy_state_id": latent_proxy_state_ids,
        },
        meta=meta,
        embed_metadata_in_parquet=True,
    )
