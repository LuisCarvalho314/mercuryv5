from __future__ import annotations

from pathlib import Path

from mercury.latent.params import LatentParams
from mercury.sensory.params import SensoryParams

from ...pipelines import run_all_bundled


def _sensor_dataset_key(sensor: str, sensor_range) -> str:
    if sensor == "cardinal distance":
        return f"{sensor}:range={sensor_range}"
    return sensor


def run_mercury_bundle(*, config, dataset_run_ids, latent_step_callback=None, show_progress: bool = True, progress_callback=None) -> Path:
    dataset_run_id = None
    ground_truth_dataset_run_id = None
    if isinstance(dataset_run_ids, dict):
        dataset_run_id = dataset_run_ids.get(_sensor_dataset_key(config.sensor, config.sensor_range))
        ground_truth_dataset_run_id = dataset_run_ids.get("cartesian")
    return run_all_bundled(
        datasets_root=Path(config.datasets_root),
        results_root=Path(config.results_root),
        level=config.level,
        sensor=config.sensor,
        sensor_range=config.sensor_range,
        dataset_select=("run_id" if dataset_run_id else "latest"),
        dataset_run_id=dataset_run_id,
        ground_truth_dataset_run_id=ground_truth_dataset_run_id,
        mercury_valid_trajectories_only=bool(config.mercury_valid_trajectories_only),
        mercury_split_sensory_raw_latent_valid=bool(config.mercury_split_sensory_raw_latent_valid),
        memory_length=config.memory_length,
        embed_metadata_in_parquet=bool(config.embed_metadata_in_parquet),
        sensory_params=SensoryParams(**config.sensory),
        latent_params=LatentParams(**config.latent),
        action_map_params=dict(config.action_map),
        run_parameters={
            **dict(config.run_parameters),
            "dataset_runs": dataset_run_ids,
            "execution": {
                "reuse_existing_run": bool(config.reuse_existing_run),
                "reuse_existing_dataset": bool(config.reuse_existing_dataset),
                "mercury_valid_trajectories_only": bool(config.mercury_valid_trajectories_only),
                "mercury_split_sensory_raw_latent_valid": bool(config.mercury_split_sensory_raw_latent_valid),
                "dataset_select": ("run_id" if dataset_run_id else "latest"),
                "dataset_run_id": dataset_run_id,
                "ground_truth_dataset_run_id": ground_truth_dataset_run_id,
            },
        },
        latent_step_callback=latent_step_callback,
        show_progress=show_progress,
        progress_callback=progress_callback,
        run_id=config.run_id,
        notes=config.notes,
    )
