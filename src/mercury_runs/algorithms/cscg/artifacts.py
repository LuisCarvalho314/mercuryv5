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


def write_cscg_artifacts(
    *,
    config,
    model: Any,
    dataset_parquet_name: str,
    dataset_metadata: dict[str, Any],
    act_idx: np.ndarray,
    n_obs: int,
    n_clones: np.ndarray,
    convergence_by_stage: dict[str, np.ndarray],
    training_stages: list[str],
    objective_name: str,
    objective_scores: np.ndarray,
    decode_log2_lik: np.ndarray,
    ground_truth_bmu: np.ndarray,
    valid_transition_stats: dict[str, int],
    training_summary: dict[str, Any],
    mapped_bmu: np.ndarray,
    latent_node_count: np.ndarray,
    clone_states: np.ndarray,
    observation_prediction_eval: dict[str, Any],
) -> Path:
    output_dir_path = output_dir(config.output_root, config.level, config.sensor, config.sensor_range)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    model_npz = output_dir_path / f"{config.run_id}_cscg_model.npz"
    np.savez_compressed(
        model_npz,
        T=model.T.astype(np.float32, copy=False),
        n_clones=n_clones,
        convergence=np.concatenate([stage for stage in convergence_by_stage.values()]),
        **{f"convergence_{stage}": values for stage, values in convergence_by_stage.items()},
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
            dataset_parquet_name=dataset_parquet_name,
        ),
        sensory_params={},
        latent_params={},
        action_map_params={"method": "cscg", "n_actions": int(act_idx.max()) + 1},
        memory_length=None,
        run_parameters={
            "baseline_method": "cscg",
            "cscg": config.model_dump(mode="json"),
            "training_stages": training_stages,
            "training_data": {
                "observation_mode": "discrete_observation_ids",
                "action_mode": "discrete_action_ids",
                "valid_transition_filter": bool(config.valid_trajectories_only),
                **valid_transition_stats,
                "sequence_length": int(len(ground_truth_bmu)),
                **{key: value for key, value in training_summary.items() if value is not None},
            },
            "artifacts": {"cscg_model_npz": str(model_npz)},
            "cscg_eval": {
                "train_algorithm": str(config.train_algorithm).strip().lower(),
                **{key: value for key, value in training_summary.items() if value is not None},
                "objective_metric": objective_name,
                "objective_bits_per_symbol_mean": float(objective_scores.mean()),
                "objective_negative_log2_likelihood_sum": float(objective_scores.sum()),
                "state_decode_algorithm": "viterbi",
                "state_decode_negative_log2_likelihood_sum": float(decode_log2_lik.sum()),
                "state_decode_negative_log2_likelihood_mean": float(decode_log2_lik.mean()),
                "num_clone_states": int(n_clones.sum()),
                "num_observations": int(n_obs),
                "num_actions": int(act_idx.max()) + 1,
                "convergence_steps": int(sum(len(stage) for stage in convergence_by_stage.values())),
                "convergence_steps_by_stage": {stage: int(len(values)) for stage, values in convergence_by_stage.items()},
                "precision_metric_name": "paper_precision",
                "precision_state_series": "cscg_state_id",
                "valid_transition_filter": bool(config.valid_trajectories_only),
                **valid_transition_stats,
                **observation_prediction_eval,
            },
        },
        source_dataset_metadata=dataset_metadata,
        notes="baseline=cscg",
    )
    return write_bundle_parquet(
        output_dir=output_dir_path,
        run_id=config.run_id,
        bundle_name="states",
        columns={
            "cartesian_proxy_bmu": ground_truth_bmu.astype(np.int32, copy=False),
            "sensory_bmu": mapped_bmu.astype(np.int32, copy=False),
            "latent_bmu": mapped_bmu.astype(np.int32, copy=False),
            "latent_node_count": latent_node_count,
            "cscg_state_id": clone_states,
        },
        meta=meta,
        embed_metadata_in_parquet=True,
    )
