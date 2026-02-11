# src/mercury_runs/pipelines.py
from __future__ import annotations

import dataclasses
import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from data_helper.csv_loader import iter_sequence
from mercury.action_map.adapter import ActionMap
from mercury.latent.params import LatentParams
from mercury.latent.state import init_latent_state, latent_step
from mercury.memory.state import add_memory, init_mem, update_memory
from mercury.sensory.params import SensoryParams
from mercury.sensory.state import init_state, sensory_step, sensory_step_frozen

from .io_parquet import LoadedDataset, ParquetConfig, load_level_parquet
from .save_results import write_bundle_parquet
from .schemas_results import ResultBundleMeta, SourceDatasetRef


def utc_timestamp() -> str:
    return datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def params_to_dict(params_object: Any) -> Dict[str, Any]:
    if params_object is None:
        return {}
    if hasattr(params_object, "model_dump"):  # pydantic v2
        return params_object.model_dump()
    if hasattr(params_object, "dict"):  # pydantic v1
        return params_object.dict()
    if dataclasses.is_dataclass(params_object):
        return dataclasses.asdict(params_object)
    return dict(vars(params_object))


def _action_dim(actions: np.ndarray) -> int:
    return int(actions.shape[1]) if actions.ndim == 2 else 1


def _results_output_dir(results_root: Path, level: int, sensor: str, sensor_range: Optional[int]) -> Path:
    base = results_root / f"level={level}" / f"sensor={sensor}"
    if sensor == "cardinal distance":
        return base / f"range={sensor_range}"
    return base


def _build_action_map(action_dim: int, action_map_params: Dict[str, Any]) -> ActionMap:
    return ActionMap.random(
        n_codebook=int(action_map_params["n_codebook"]),
        dim=int(action_dim),
        lr=float(action_map_params["lr"]),
        sigma=float(action_map_params["sigma"]),
        key=int(action_map_params["key"]),
    )


def run_ground_truth_cartesian(
    dataset_cartesian: LoadedDataset,
    sensory_params: SensoryParams,
    action_map_params: Dict[str, Any],
) -> np.ndarray:
    observations, actions, collisions = dataset_cartesian.observations, dataset_cartesian.actions, dataset_cartesian.collisions
    observation_dim = int(observations.shape[1])
    action_dim = _action_dim(actions)

    sensory_state = init_state(observation_dim)
    action_map = _build_action_map(action_dim, action_map_params)

    bmus: list[int] = []
    for observation, action, collision in iter_sequence(observations, actions, collisions):
        action_vector = np.atleast_1d(action).astype(np.float32)
        action_bmu, _ = action_map.step(action_vector)

        sensory_state = sensory_step(
            observation.astype(np.float32),
            int(action_bmu),
            sensory_state,
            sensory_params,
            action_map,
        )
        bmus.append(int(sensory_state.prev_bmu))

    return np.asarray(bmus, dtype=np.int32)


def run_latent(
    dataset_latent: LoadedDataset,
    sensory_params: SensoryParams,
    latent_params: LatentParams,
    action_map_params: Dict[str, Any],
    memory_length: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    observations, actions, collisions = dataset_latent.observations, dataset_latent.actions, dataset_latent.collisions
    observation_dim = int(observations.shape[1])
    action_dim = _action_dim(actions)

    sensory_state = init_state(observation_dim)
    action_map = _build_action_map(action_dim, action_map_params)

    sensory_bmus: list[int] = []
    for observation, action, collision in iter_sequence(observations, actions, collisions):
        action_vector = np.atleast_1d(action).astype(np.float32)
        action_bmu, _ = action_map.step(action_vector)

        sensory_state = sensory_step(
            observation.astype(np.float32),
            int(action_bmu),
            sensory_state,
            sensory_params,
            action_map,
        )
        sensory_bmus.append(int(sensory_state.prev_bmu))

    mem = init_mem(sensory_state.gs.n, memory_length)
    latent_state = init_latent_state(mem)

    previous_latent_bmu = None
    memory_vectors: list[np.ndarray] = []
    action_memory: list[int] = []
    state_memory: list[int] = []

    latent_bmus: list[int] = []
    latent_node_counts: list[int] = []

    for observation, action, collision in iter_sequence(observations, actions, collisions):
        action_vector = np.atleast_1d(action).astype(np.float32)
        action_bmu, _ = action_map.step(action_vector)

        sensory_state = sensory_step_frozen(
            observation.astype(np.float32),
            int(action_bmu),
            sensory_state,
            sensory_params,
            action_map,
        )

        sensory_graph_size = int(sensory_state.gs.n)
        if mem.gs.n != sensory_graph_size * memory_length:
            mem = init_mem(sensory_graph_size, length=memory_length)

        if previous_latent_bmu is None or not bool(collision):
            activation_vector = np.asarray(sensory_state.gs.node_features["activation"], dtype=np.float32)
            mem = update_memory(mem)
            mem = add_memory(mem, activation_vector)

            memory_vectors.append(activation_vector)
            action_memory.append(int(action_bmu))

            latent_state, _, state_memory = latent_step(
                mem,
                memory_vectors,
                latent_state,
                int(action_bmu),
                latent_params,
                action_map,
                action_memory,
                state_memory,
            )

        latent_bmus.append(int(latent_state.prev_bmu))
        latent_node_counts.append(int(latent_state.g.n))
        previous_latent_bmu = latent_state.prev_bmu

    return (
        np.asarray(sensory_bmus, dtype=np.int32),
        np.asarray(latent_bmus, dtype=np.int32),
        np.asarray(latent_node_counts, dtype=np.int32),
    )


def run_all_bundled(
    *,
    datasets_root: Path,
    results_root: Path,
    level: int,
    sensor: str,
    sensor_range: Optional[int] = None,
    dataset_select: str = "latest",
    dataset_run_id: Optional[str] = None,
    memory_length: int = 10,
    embed_metadata_in_parquet: bool = True,
    sensory_params: SensoryParams,
    latent_params: LatentParams,
    action_map_params: Dict[str, Any],
    run_id: str,
    notes: Optional[str] = None,
) -> Path:
    dataset_latent = load_level_parquet(
        ParquetConfig(
            root=datasets_root,
            level=level,
            sensor=sensor,
            sensor_range=sensor_range,
            select=dataset_select,
            run_id=dataset_run_id,
        )
    )
    dataset_cartesian = load_level_parquet(
        ParquetConfig(
            root=datasets_root,
            level=level,
            sensor="cartesian",
            sensor_range=None,
            select="latest",
            run_id=None,
        )
    )

    output_dir = _results_output_dir(results_root, level, sensor, sensor_range)

    ground_truth_bmu = run_ground_truth_cartesian(dataset_cartesian, sensory_params, action_map_params)
    sensory_bmu, latent_bmu, latent_node_count = run_latent(dataset_latent, sensory_params, latent_params, action_map_params, memory_length)

    length = min(
        int(ground_truth_bmu.shape[0]),
        int(sensory_bmu.shape[0]),
        int(latent_bmu.shape[0]),
        int(latent_node_count.shape[0]),
    )
    ground_truth_bmu = ground_truth_bmu[:length]
    sensory_bmu = sensory_bmu[:length]
    latent_bmu = latent_bmu[:length]
    latent_node_count = latent_node_count[:length]

    source_ref = SourceDatasetRef(
        level=level,
        sensor=sensor,
        sensor_range=sensor_range,
        select=dataset_select,
        dataset_run_id=dataset_run_id,
        dataset_parquet_name=dataset_latent.parquet_path.name,
    )

    meta = ResultBundleMeta(
        run_id=run_id,
        timestamp_utc=utc_timestamp(),
        source=source_ref,
        sensory_params=params_to_dict(sensory_params),
        latent_params=params_to_dict(latent_params),
        action_map_params={"dim": _action_dim(dataset_latent.actions), **action_map_params},
        memory_length=memory_length,
        notes=notes,
    )

    return write_bundle_parquet(
        output_dir=output_dir,
        run_id=run_id,
        bundle_name="states",
        columns={
            "ground_truth_bmu": ground_truth_bmu,
            "sensory_bmu": sensory_bmu,
            "latent_bmu": latent_bmu,
            "latent_node_count": latent_node_count,
        },
        meta=meta,
        embed_metadata_in_parquet=embed_metadata_in_parquet,
    )
