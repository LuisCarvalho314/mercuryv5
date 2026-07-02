# src/mercury_runs/pipelines.py
from __future__ import annotations

import dataclasses
import datetime
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Tuple

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator
from tqdm import tqdm

from data_helper.csv_loader import iter_sequence
from mercury.action_map.adapter import ActionMap
from mercury.graph.core import Graph
from mercury.latent.params import LatentParams
from mercury.latent.state import (
    active_edge_ages,
    active_node_ages,
    bmu_memory_depth,
    distribution_summary,
    init_latent_state,
    latent_step,
)
from mercury.memory.state import add_memory, init_mem, update_memory
from mercury.sensory.params import SensoryParams
from mercury.sensory.state import SensoryState, init_state, sensory_step, sensory_step_frozen

from .io_parquet import LoadedDataset, ParquetConfig, load_level_parquet
from .infrastructure.observation_encoding import (
    encode_observations_with_vocabulary,
    fit_cscg_one_hot_observations,
)
from .save_results import write_bundle_parquet
from .schemas_results import ResultBundleMeta, SourceDatasetRef


def filter_mercury_dataset(*, dataset: LoadedDataset, valid_trajectories_only: bool) -> LoadedDataset:
    from .algorithms.mercury.prepare import filter_mercury_dataset as _filter_mercury_dataset

    return _filter_mercury_dataset(
        dataset=dataset,
        valid_trajectories_only=valid_trajectories_only,
    )


class ActionMapParamsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    n_codebook: int = Field(gt=0)
    lr: float = Field(ge=0.0)
    sigma: float = Field(ge=0.0)
    key: int = Field(ge=0)
    identity_for_one_hot: bool = False


class RunAllBundledConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    datasets_root: Path
    results_root: Path
    level: int = Field(ge=0)
    sensor: str = Field(min_length=1)
    sensor_range: Optional[int] = Field(default=None, ge=0)
    dataset_select: Literal["latest", "run_id"] = "latest"
    dataset_run_id: Optional[str] = None
    ground_truth_dataset_run_id: Optional[str] = None
    mercury_valid_trajectories_only: bool = False
    mercury_split_sensory_raw_latent_valid: bool = False
    mercury_input_mode: Literal["pretrained_raw", "pretrained_cscg_one_hot", "direct_cscg_one_hot"] = "pretrained_raw"
    memory_length: int = Field(gt=0)
    embed_metadata_in_parquet: bool = True
    sensory_params: SensoryParams
    latent_params: LatentParams
    action_map_params: ActionMapParamsConfig
    run_parameters: Dict[str, Any] = Field(default_factory=dict)
    run_id: str = Field(min_length=1)
    notes: Optional[str] = None

    @model_validator(mode="after")
    def _validate_dataset_selection(self) -> "RunAllBundledConfig":
        if self.dataset_select == "run_id" and not self.dataset_run_id:
            raise ValueError("dataset_select='run_id' requires dataset_run_id.")
        if self.sensor == "cardinal distance" and self.sensor_range is None:
            raise ValueError("sensor='cardinal distance' requires sensor_range.")
        if self.mercury_input_mode in {"pretrained_cscg_one_hot", "direct_cscg_one_hot"}:
            if self.sensor != "cardinal distance":
                raise ValueError(f"mercury_input_mode={self.mercury_input_mode!r} requires sensor='cardinal distance'.")
            if self.sensor_range is None:
                raise ValueError(f"mercury_input_mode={self.mercury_input_mode!r} requires sensor_range.")
        return self


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
    cfg = ActionMapParamsConfig.model_validate(action_map_params)
    if bool(cfg.identity_for_one_hot):
        if int(action_dim) != int(cfg.n_codebook):
            raise ValueError(
                f"identity_for_one_hot requires action_dim == n_codebook; got {action_dim} and {cfg.n_codebook}"
            )
        return ActionMap.identity(dim=int(action_dim))
    return ActionMap.random(
        n_codebook=int(cfg.n_codebook),
        dim=int(action_dim),
        lr=float(cfg.lr),
        sigma=float(cfg.sigma),
        key=int(cfg.key),
    )


def _action_codebook_size(action_map: Any, *, fallback: int) -> int:
    codebook = getattr(getattr(action_map, "state", None), "codebook", None)
    if codebook is None:
        return int(fallback)
    return int(np.asarray(codebook).shape[0])


def _init_latent_state_compat(mem: Any, *, n_actions: int):
    try:
        return init_latent_state(mem, n_actions=n_actions)
    except TypeError:
        return init_latent_state(mem)


def _one_hot_dataset(dataset: LoadedDataset, obs_uniques: np.ndarray) -> LoadedDataset:
    encoded = encode_observations_with_vocabulary(
        observations=np.asarray(dataset.observations),
        obs_uniques=obs_uniques,
    )
    return LoadedDataset(
        observations=encoded.observations,
        actions=np.asarray(dataset.actions),
        collisions=np.asarray(dataset.collisions),
        source_metadata={
            **dict(dataset.source_metadata),
            "observation_mode": "cscg_one_hot",
            "observation_vocabulary_size": int(np.asarray(obs_uniques).shape[0]),
        },
        parquet_path=dataset.parquet_path,
    )


def _build_identity_sensory_graph(n_symbols: int) -> Graph:
    graph = Graph(directed=True)
    graph.register_node_feature("weight", int(n_symbols))
    graph.register_node_feature("context", int(n_symbols))
    graph.register_node_feature("activation", 1)
    graph.register_edge_feature("age", 1, dtype=np.int32, init_value=0)
    graph.register_edge_feature("action", 1, dtype=np.int32, init_value=0)
    basis = np.eye(int(n_symbols), dtype=np.float32)
    for index in range(int(n_symbols)):
        graph.add_node(
            node_feat={
                "weight": basis[index],
                "context": np.zeros((int(n_symbols),), dtype=np.float32),
                "activation": np.array(0.0, dtype=np.float32),
            },
            edge_defaults_for_new_node={
                "age": np.array([0], dtype=np.int32),
                "action": np.array([0], dtype=np.int32),
            },
        )
    return graph


def _set_identity_sensory_activation(graph: Graph, bmu: int) -> Graph:
    activation = np.zeros((graph.n,), dtype=np.float32)
    activation[int(bmu)] = 1.0
    graph.set_node_feat("activation", activation)
    return graph


def run_ground_truth_cartesian(
    dataset_cartesian: LoadedDataset,
    sensory_params: SensoryParams,
    action_map_params: Dict[str, Any],
    *,
    show_progress: bool = False,
    progress_desc: Optional[str] = None,
    progress_callback: Any = None,
) -> np.ndarray:
    observations, actions, collisions = dataset_cartesian.observations, dataset_cartesian.actions, dataset_cartesian.collisions
    observation_dim = int(observations.shape[1])
    action_dim = _action_dim(actions)

    sensory_state = init_state(observation_dim)
    action_map = _build_action_map(action_dim, action_map_params)

    iterator = iter_sequence(observations, actions, collisions)
    if show_progress:
        iterator = tqdm(iterator, total=int(observations.shape[0]), desc=(progress_desc or "Mercury GT"), position=0)

    bmus: list[int] = []
    total = int(observations.shape[0])
    report_every = max(1, total // 100)
    for index, (observation, action, collision) in enumerate(iterator, start=1):
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
        if progress_callback is not None and (index == 1 or index == total or index % report_every == 0):
            progress_callback(current=int(index), total=total, message="Ground truth pass")

    return np.asarray(bmus, dtype=np.int32)


def _attribution_row_for_step(
    *,
    step: int,
    latent_bmu: int,
    latent_node_count: int,
    attribution: Any | None,
) -> dict[str, float | int]:
    row: dict[str, float | int] = {
        "step": int(step),
        "latent_bmu": int(latent_bmu),
        "latent_node_count": int(latent_node_count),
    }
    default_fields = {
        "admissible_node_count": 0,
        "selected_memory_drive_raw": 0.0,
        "selected_undirected_drive_raw": 0.0,
        "selected_baseline_drive_raw": 0.0,
        "selected_action_drive_raw": 0.0,
        "selected_memory_drive_weighted": 0.0,
        "selected_undirected_drive_weighted": 0.0,
        "selected_baseline_drive_weighted": 0.0,
        "selected_action_drive_weighted": 0.0,
        "selected_memory_drive_share": 0.0,
        "selected_undirected_drive_share": 0.0,
        "selected_baseline_drive_share": 0.0,
        "selected_action_drive_share": 0.0,
        "selected_trace_penalty": 0.0,
        "selected_total_support_pre_trace": 0.0,
        "selected_total_support_post_trace": 0.0,
    }
    if attribution is None:
        row.update(default_fields)
        return row

    row["admissible_node_count"] = int(getattr(attribution, "admissible_node_count", 0))
    for field_name, default_value in default_fields.items():
        if field_name == "admissible_node_count":
            continue
        row[field_name] = float(getattr(attribution, field_name, default_value))
    return row


def _structural_metric_row_for_step(
    *,
    step: int,
    latent_state: Any,
    latent_bmu: int,
    memory_length: int,
) -> dict[str, Any]:
    graph = latent_state.g
    node_ages = active_node_ages(graph, step)
    edge_ages = active_edge_ages(graph, step)
    row: dict[str, Any] = {
        "bmu_memory_depth": int(bmu_memory_depth(graph, latent_bmu, memory_length)),
        "clone_created_count": int(getattr(latent_state, "last_clone_created_count", 0)),
        "clone_created_cumulative": int(getattr(latent_state, "clone_created_count", 0)),
        "edge_prune_count": int(getattr(latent_state, "last_edge_prune_count", 0)),
        "edge_prune_cumulative": int(getattr(latent_state, "edge_prune_count", 0)),
        "node_prune_count": int(getattr(latent_state, "last_node_prune_count", 0)),
        "node_prune_cumulative": int(getattr(latent_state, "node_prune_count", 0)),
        "node_age_distribution": node_ages.astype(np.int32, copy=False).tolist(),
        "edge_age_distribution": edge_ages.astype(np.int32, copy=False).tolist(),
    }
    row.update(distribution_summary(node_ages, "node_age"))
    row.update(distribution_summary(edge_ages, "edge_age"))
    return row


def _empty_structural_metric_arrays(length: int) -> dict[str, np.ndarray | list[list[int]]]:
    return {
        "bmu_memory_depth": np.zeros((length,), dtype=np.int32),
        "clone_created_count": np.zeros((length,), dtype=np.int32),
        "clone_created_cumulative": np.zeros((length,), dtype=np.int32),
        "edge_prune_count": np.zeros((length,), dtype=np.int32),
        "edge_prune_cumulative": np.zeros((length,), dtype=np.int32),
        "node_prune_count": np.zeros((length,), dtype=np.int32),
        "node_prune_cumulative": np.zeros((length,), dtype=np.int32),
        "node_age_mean": np.zeros((length,), dtype=np.float32),
        "node_age_median": np.zeros((length,), dtype=np.float32),
        "node_age_std": np.zeros((length,), dtype=np.float32),
        "node_age_q25": np.zeros((length,), dtype=np.float32),
        "node_age_q75": np.zeros((length,), dtype=np.float32),
        "node_age_q90": np.zeros((length,), dtype=np.float32),
        "edge_age_mean": np.zeros((length,), dtype=np.float32),
        "edge_age_median": np.zeros((length,), dtype=np.float32),
        "edge_age_std": np.zeros((length,), dtype=np.float32),
        "edge_age_q25": np.zeros((length,), dtype=np.float32),
        "edge_age_q75": np.zeros((length,), dtype=np.float32),
        "edge_age_q90": np.zeros((length,), dtype=np.float32),
        "node_age_distribution": [[] for _ in range(length)],
        "edge_age_distribution": [[] for _ in range(length)],
    }


def _structural_metric_columns(rows: list[dict[str, Any]], length: int) -> dict[str, Any]:
    if not rows:
        return _empty_structural_metric_arrays(length)
    columns: dict[str, Any] = {}
    keys = list(rows[0].keys())
    int_keys = {
        "bmu_memory_depth",
        "clone_created_count",
        "clone_created_cumulative",
        "edge_prune_count",
        "edge_prune_cumulative",
        "node_prune_count",
        "node_prune_cumulative",
    }
    list_keys = {"node_age_distribution", "edge_age_distribution"}
    for key in keys:
        values = [row[key] for row in rows[:length]]
        if key in list_keys:
            columns[key] = values
        elif key in int_keys:
            columns[key] = np.asarray(values, dtype=np.int32)
        else:
            columns[key] = np.asarray(values, dtype=np.float32)
    return columns


def run_latent(
    dataset_latent: LoadedDataset,
    sensory_params: SensoryParams,
    latent_params: LatentParams,
    action_map_params: Dict[str, Any],
    memory_length: int,
    step_callback: Any = None,
    dataset_sensory: LoadedDataset | None = None,
    latent_valid_trajectories_only: bool = False,
    direct_one_hot_observations: bool = False,
    obs_uniques: np.ndarray | None = None,
    *,
    show_progress: bool = False,
    progress_desc: Optional[str] = None,
    progress_callback: Any = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Graph, list[dict[str, float | int]], list[dict[str, Any]]]:
    sensory_dataset = dataset_sensory or dataset_latent
    sensory_observations = sensory_dataset.observations
    sensory_actions = sensory_dataset.actions
    sensory_collisions = sensory_dataset.collisions
    latent_observations = dataset_latent.observations
    latent_actions = dataset_latent.actions
    latent_collisions = dataset_latent.collisions

    observation_dim = int(sensory_observations.shape[1])
    action_dim = _action_dim(sensory_actions)

    action_map = _build_action_map(action_dim, action_map_params)
    if bool(direct_one_hot_observations):
        if obs_uniques is None:
            raise ValueError("direct_one_hot_observations requires obs_uniques.")
        sensory_state = SensoryState(
            gs=_build_identity_sensory_graph(observation_dim),
            global_context=np.zeros((observation_dim,), dtype=np.float32),
            mapping=np.arange(observation_dim, dtype=np.int32),
        )
    else:
        sensory_state = init_state(observation_dim)

    training_sensory_bmus: list[int] = []
    if not bool(direct_one_hot_observations):
        sensory_iterator = iter_sequence(sensory_observations, sensory_actions, sensory_collisions)
        if show_progress:
            sensory_iterator = tqdm(
                sensory_iterator,
                total=int(sensory_observations.shape[0]),
                desc=(f"{progress_desc or 'Mercury Latent'} Sensory"),
                position=0,
            )
        sensory_total = int(sensory_observations.shape[0])
        sensory_report_every = max(1, sensory_total // 100)
        for sensory_index, (observation, action, collision) in enumerate(sensory_iterator, start=1):
            action_vector = np.atleast_1d(action).astype(np.float32)
            action_bmu, _ = action_map.step(action_vector)

            sensory_state = sensory_step(
                observation.astype(np.float32),
                int(action_bmu),
                sensory_state,
                sensory_params,
                action_map,
            )
            training_sensory_bmus.append(int(sensory_state.prev_bmu))
            if progress_callback is not None and (
                sensory_index == 1 or sensory_index == sensory_total or sensory_index % sensory_report_every == 0
            ):
                progress_callback(
                    stage="latent_sensory",
                    current=int(sensory_index),
                    total=sensory_total,
                    message="Latent sensory pass",
                    extra={"sensory_nodes": int(sensory_state.gs.n)},
                )

    mem = init_mem(sensory_state.gs.n, memory_length)
    latent_state = _init_latent_state_compat(mem, n_actions=_action_codebook_size(action_map, fallback=action_dim))

    previous_latent_bmu = None
    memory_vectors: list[np.ndarray] = []
    action_memory: list[int] = []
    state_memory: list[int] = []

    frozen_sensory_bmus: list[int] = []
    latent_bmus: list[int] = []
    latent_node_counts: list[int] = []
    attribution_rows: list[dict[str, float | int]] = []
    structural_metric_rows: list[dict[str, Any]] = []

    latent_iterator = iter_sequence(latent_observations, latent_actions, latent_collisions)
    if show_progress:
        latent_iterator = tqdm(
            latent_iterator,
            total=int(latent_observations.shape[0]),
            desc=(progress_desc or "Mercury Latent"),
            position=0,
        )
    latent_total = int(latent_observations.shape[0])
    latent_report_every = max(1, latent_total // 100)
    for latent_index, (observation, action, collision) in enumerate(latent_iterator, start=1):
        action_vector = np.atleast_1d(action).astype(np.float32)
        if hasattr(action_map, "predict"):
            action_bmu = int(action_map.predict(action=action_vector))
        else:
            action_bmu, _ = action_map.step(action_vector)
            action_bmu = int(action_bmu)

        if bool(direct_one_hot_observations):
            activation_vector = np.asarray(observation, dtype=np.float32)
            if activation_vector.ndim != 1 or activation_vector.shape[0] != sensory_state.gs.n:
                raise ValueError("direct one-hot Mercury requires one-hot observations with width equal to sensory graph size.")
            if not np.isclose(float(activation_vector.sum()), 1.0) or np.count_nonzero(activation_vector > 0.5) != 1:
                raise ValueError("direct one-hot Mercury requires one-hot observations.")
            sensory_bmu = int(np.argmax(activation_vector))
            if sensory_state.prev_bmu is not None:
                sensory_state.gs.add_edge(
                    int(sensory_state.prev_bmu),
                    sensory_bmu,
                    weight=float(sensory_state.gs.adj[int(sensory_state.prev_bmu), sensory_bmu]) + 1.0,
                    edge_feat={"action": np.array([int(action_bmu)], dtype=np.int32)},
                )
            sensory_state.gs = _set_identity_sensory_activation(sensory_state.gs, sensory_bmu)
            sensory_state.prev_bmu = sensory_bmu
            sensory_state.step_idx += 1
            frozen_sensory_bmus.append(sensory_bmu)
        else:
            sensory_state = sensory_step_frozen(
                observation.astype(np.float32),
                int(action_bmu),
                sensory_state,
                sensory_params,
                action_map,
            )
            activation_vector = np.asarray(sensory_state.gs.node_features["activation"], dtype=np.float32)
            frozen_sensory_bmus.append(int(sensory_state.prev_bmu))

        sensory_graph_size = int(sensory_state.gs.n)
        if mem.gs.n != sensory_graph_size * memory_length:
            mem = init_mem(sensory_graph_size, length=memory_length)

        should_update_latent = not (bool(latent_valid_trajectories_only) and bool(collision))
        if should_update_latent:
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

        current_latent_bmu = latent_state.prev_bmu
        if current_latent_bmu is None:
            current_latent_bmu = 0
        latent_bmus.append(int(current_latent_bmu))
        latent_node_counts.append(int(latent_state.g.n))
        attribution_rows.append(
            _attribution_row_for_step(
                step=len(latent_bmus),
                latent_bmu=int(current_latent_bmu),
                latent_node_count=int(latent_state.g.n),
                attribution=getattr(latent_state, "last_bmu_attribution", None),
            )
        )
        structural_metric_rows.append(
            _structural_metric_row_for_step(
                step=len(latent_bmus),
                latent_state=latent_state,
                latent_bmu=int(current_latent_bmu),
                memory_length=memory_length,
            )
        )
        previous_latent_bmu = latent_state.prev_bmu
        if show_progress and hasattr(latent_iterator, "set_postfix"):
            latent_iterator.set_postfix(nodes=int(latent_state.g.n))
        if progress_callback is not None and (
            latent_index == 1 or latent_index == latent_total or latent_index % latent_report_every == 0
        ):
            progress_callback(
                stage="latent",
                current=int(latent_index),
                total=latent_total,
                message="Latent pass",
                extra={
                    "latent_nodes": int(latent_state.g.n),
                    "sensory_nodes": int(sensory_state.gs.n),
                },
            )
        if step_callback is not None:
            callback_kwargs = {
                "step": int(len(latent_bmus)),
                "sensory_graph": sensory_state.gs,
                "latent_graph": latent_state.g,
                "action_map": action_map,
                "obs_uniques": obs_uniques,
            }
            try:
                step_callback(**callback_kwargs)
            except TypeError:
                callback_kwargs.pop("obs_uniques", None)
                step_callback(**callback_kwargs)

    return (
        np.asarray(frozen_sensory_bmus, dtype=np.int32),
        np.asarray(latent_bmus, dtype=np.int32),
        np.asarray(latent_node_counts, dtype=np.int32),
        latent_state.g,
        attribution_rows,
        structural_metric_rows,
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
    ground_truth_dataset_run_id: Optional[str] = None,
    mercury_valid_trajectories_only: bool = False,
    mercury_split_sensory_raw_latent_valid: bool = False,
    mercury_input_mode: str = "pretrained_raw",
    memory_length: int = 10,
    embed_metadata_in_parquet: bool = True,
    sensory_params: SensoryParams,
    latent_params: LatentParams,
    action_map_params: Dict[str, Any],
    run_parameters: Optional[Dict[str, Any]] = None,
    run_id: str,
    notes: Optional[str] = None,
    latent_step_callback: Any = None,
    show_progress: bool = False,
    progress_callback: Any = None,
) -> Path:
    try:
        config = RunAllBundledConfig.model_validate(
            {
                "datasets_root": datasets_root,
                "results_root": results_root,
                "level": level,
                "sensor": sensor,
                "sensor_range": sensor_range,
                "dataset_select": dataset_select,
                "dataset_run_id": dataset_run_id,
                "ground_truth_dataset_run_id": ground_truth_dataset_run_id,
                "mercury_valid_trajectories_only": mercury_valid_trajectories_only,
                "mercury_split_sensory_raw_latent_valid": mercury_split_sensory_raw_latent_valid,
                "mercury_input_mode": mercury_input_mode,
                "memory_length": memory_length,
                "embed_metadata_in_parquet": embed_metadata_in_parquet,
                "sensory_params": sensory_params,
                "latent_params": latent_params,
                "action_map_params": action_map_params,
                "run_parameters": run_parameters or {},
                "run_id": run_id,
                "notes": notes,
            }
        )
    except ValidationError as exc:
        raise ValueError(f"Invalid pipeline configuration: {exc}") from exc

    dataset_latent_raw = load_level_parquet(
        ParquetConfig(
            root=config.datasets_root,
            level=config.level,
            sensor=config.sensor,
            sensor_range=config.sensor_range,
            select=config.dataset_select,
            run_id=config.dataset_run_id,
        )
    )
    dataset_cartesian_raw = load_level_parquet(
        ParquetConfig(
            root=config.datasets_root,
            level=config.level,
            sensor="cartesian",
            sensor_range=None,
            select=("run_id" if config.ground_truth_dataset_run_id else "latest"),
            run_id=config.ground_truth_dataset_run_id,
        )
    )
    dataset_sensory = dataset_latent_raw
    dataset_latent = dataset_latent_raw
    dataset_cartesian = dataset_cartesian_raw
    split_sensory_latent = bool(config.mercury_split_sensory_raw_latent_valid)
    latent_valid_trajectories_only = False
    obs_uniques: np.ndarray | None = None

    if bool(config.mercury_valid_trajectories_only):
        dataset_sensory = filter_mercury_dataset(dataset=dataset_latent_raw, valid_trajectories_only=True)
        dataset_latent = dataset_sensory
        dataset_cartesian = filter_mercury_dataset(dataset=dataset_cartesian_raw, valid_trajectories_only=True)
        split_sensory_latent = False
    elif split_sensory_latent:
        latent_valid_trajectories_only = True

    mercury_input_mode = str(config.mercury_input_mode).strip().lower()
    if mercury_input_mode in {"pretrained_cscg_one_hot", "direct_cscg_one_hot"}:
        fitted = fit_cscg_one_hot_observations(np.asarray(dataset_sensory.observations))
        obs_uniques = fitted.obs_uniques
        dataset_sensory = _one_hot_dataset(dataset_sensory, obs_uniques)
        dataset_latent = _one_hot_dataset(dataset_latent, obs_uniques)
    elif mercury_input_mode != "pretrained_raw":
        raise ValueError(f"Unsupported mercury_input_mode: {config.mercury_input_mode!r}")

    action_map_params_dict = config.action_map_params.model_dump()
    output_dir = _results_output_dir(config.results_root, config.level, config.sensor, config.sensor_range)

    ground_truth_bmu = run_ground_truth_cartesian(
        dataset_cartesian,
        config.sensory_params,
        action_map_params_dict,
        show_progress=show_progress,
        progress_desc="Mercury GT",
        progress_callback=(
            (lambda **kwargs: progress_callback(stage="ground_truth", **kwargs))
            if progress_callback is not None
            else None
        ),
    )
    latent_result = run_latent(
        dataset_latent,
        config.sensory_params,
        config.latent_params,
        action_map_params_dict,
        config.memory_length,
        step_callback=latent_step_callback,
        dataset_sensory=dataset_sensory,
        latent_valid_trajectories_only=latent_valid_trajectories_only,
        direct_one_hot_observations=(mercury_input_mode == "direct_cscg_one_hot"),
        obs_uniques=obs_uniques,
        show_progress=show_progress,
        progress_desc="Mercury Latent",
        progress_callback=progress_callback,
    )
    if len(latent_result) == 5:
        sensory_bmu, latent_bmu, latent_node_count, latent_graph, attribution_rows = latent_result
        structural_metric_rows = []
    else:
        (
            sensory_bmu,
            latent_bmu,
            latent_node_count,
            latent_graph,
            attribution_rows,
            structural_metric_rows,
        ) = latent_result

    length = min(
        int(ground_truth_bmu.shape[0]),
        int(sensory_bmu.shape[0]),
        int(latent_bmu.shape[0]),
        int(latent_node_count.shape[0]),
    )
    cartesian_proxy_bmu = ground_truth_bmu[:length]
    sensory_bmu = sensory_bmu[:length]
    latent_bmu = latent_bmu[:length]
    latent_node_count = latent_node_count[:length]
    structural_metric_columns = _structural_metric_columns(structural_metric_rows, length)

    source_ref = SourceDatasetRef(
        level=config.level,
        sensor=config.sensor,
        sensor_range=config.sensor_range,
        select=config.dataset_select,
        dataset_run_id=config.dataset_run_id,
        dataset_parquet_name=dataset_latent.parquet_path.name,
    )

    meta = ResultBundleMeta(
        run_id=config.run_id,
        timestamp_utc=utc_timestamp(),
        source=source_ref,
        sensory_params=params_to_dict(config.sensory_params),
        latent_params=params_to_dict(config.latent_params),
        action_map_params={"dim": _action_dim(dataset_latent.actions), **action_map_params_dict},
        memory_length=config.memory_length,
        run_parameters={
            **dict(config.run_parameters),
            "observation_vocabulary": (obs_uniques.tolist() if obs_uniques is not None else None),
            "ground_truth_proxy": {
                "series_name": "cartesian_proxy_bmu",
                "legacy_series_name": "ground_truth_bmu",
                "description": "Sensory BMU sequence induced by cartesian observations; not raw cartesian coordinates.",
            },
            "execution": {
                **dict((config.run_parameters or {}).get("execution") or {}),
                "mercury_input_mode": mercury_input_mode,
                "observation_vocabulary_size": (int(obs_uniques.shape[0]) if obs_uniques is not None else None),
            },
        },
        source_dataset_metadata=dataset_latent.source_metadata,
        ground_truth_dataset_metadata=dataset_cartesian.source_metadata,
        notes=config.notes,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    latent_graph.to_npz(str(output_dir / f"{config.run_id}_latent_graph.npz"))
    write_bundle_parquet(
        output_dir=output_dir,
        run_id=config.run_id,
        bundle_name="attribution",
        columns={
            key: np.asarray([row[key] for row in attribution_rows])
            for key in attribution_rows[0].keys()
        } if attribution_rows else {"step": np.asarray([], dtype=np.int32)},
        meta=meta,
        embed_metadata_in_parquet=config.embed_metadata_in_parquet,
    )

    return write_bundle_parquet(
        output_dir=output_dir,
        run_id=config.run_id,
        bundle_name="states",
        columns={
            "cartesian_proxy_bmu": cartesian_proxy_bmu,
            "sensory_bmu": sensory_bmu,
            "latent_bmu": latent_bmu,
            "latent_node_count": latent_node_count,
            **structural_metric_columns,
        },
        meta=meta,
        embed_metadata_in_parquet=config.embed_metadata_in_parquet,
    )
