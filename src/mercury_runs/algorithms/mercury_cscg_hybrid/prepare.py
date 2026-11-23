from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from mercury.graph.core import Graph

from ..cscg.prepare import action_to_indices, filter_valid_cscg_rows, load_dataset_arrays, map_to_indices, to_indices
from ..cscg.structural_init import infer_graph_node_observations, source_graph_capacity, state_offsets


@dataclass(frozen=True)
class HybridAssignment:
    n_clones: np.ndarray
    node_to_state: np.ndarray
    node_to_observation: np.ndarray
    state_to_node: np.ndarray
    state_to_observation: np.ndarray
    fallback_state_mask: np.ndarray


def mercury_output_dir(root: Path, level: int, sensor: str, sensor_range: int | None) -> Path:
    base = root / f"level={level}" / f"sensor={sensor}"
    if sensor == "cardinal distance":
        return base / f"range={sensor_range}"
    return base


def load_mercury_latent_graph(path: Path) -> Graph:
    if not path.exists():
        raise FileNotFoundError(f"Missing Mercury latent graph: {path}")
    return Graph.from_npz(str(path))


def load_mercury_latent_bmus(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Missing Mercury states parquet: {path}")
    frame = pl.read_parquet(path)
    if "latent_bmu" not in frame.columns:
        raise ValueError(f"Mercury states parquet has no 'latent_bmu' column: {path}")
    return frame.get_column("latent_bmu").to_numpy().astype(np.int64, copy=False)


def load_hybrid_training_arrays(config: Any) -> dict[str, Any]:
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
    obs_idx, obs_values = to_indices(observations)
    act_idx, action_values = action_to_indices(actions)
    if obs_idx.shape[0] < 3 or act_idx.shape[0] < 3:
        raise ValueError("Mercury-CSCG hybrid could not align enough samples for CSCG training.")
    return {
        "observations": observations,
        "actions": actions,
        "obs_idx": obs_idx.astype(np.int64, copy=False),
        "act_idx": act_idx.astype(np.int64, copy=False),
        "obs_values": obs_values,
        "action_values": action_values,
        "dataset_metadata": dataset_metadata,
        "dataset_parquet_name": dataset_parquet_name,
        "training_stats": training_stats,
    }


def _handoff_node_observations(
    *,
    latent_graph: Graph,
    n_nodes: int,
    memory_length: int,
) -> tuple[np.ndarray, int]:
    mem_adj = _latent_node_sensory_templates(latent_graph)
    memory_length = int(memory_length)
    if memory_length <= 0:
        raise ValueError("memory_length must be positive for Mercury mem_adj handoff.")
    if mem_adj.shape[1] % memory_length != 0:
        raise ValueError(
            f"Mercury latent mem_adj width {mem_adj.shape[1]} is not divisible by memory_length {memory_length}."
        )
    n_sensory = int(mem_adj.shape[1] // memory_length)
    current_timestep_templates = mem_adj.reshape(n_nodes, n_sensory, memory_length)[:, :, 0]
    if current_timestep_templates.shape != (n_nodes, n_sensory):
        raise ValueError("Mercury latent graph current-timestep mem_adj has an invalid shape.")
    node_obs = np.argmax(current_timestep_templates, axis=1).astype(np.int64, copy=False)
    has_current_evidence = current_timestep_templates.sum(axis=1) > 0.0
    node_obs[~has_current_evidence] = 0
    return node_obs, n_sensory


def _latent_node_sensory_templates(latent_graph: Graph) -> np.ndarray:
    node_features = getattr(latent_graph, "node_features", {})
    if "mem_adj" not in node_features:
        raise ValueError("Mercury latent graph has no 'mem_adj' node feature for CSCG emission handoff.")
    mem_adj = np.asarray(node_features["mem_adj"], dtype=np.float32)
    if mem_adj.ndim != 2 or mem_adj.shape[0] < int(latent_graph.n):
        raise ValueError("Mercury latent graph 'mem_adj' has an invalid shape.")
    return mem_adj[: int(latent_graph.n)]


def build_hybrid_assignment(
    *,
    latent_graph: Graph,
    obs_idx: np.ndarray,
    n_obs: int,
    memory_length: int,
) -> HybridAssignment:
    n_nodes = int(latent_graph.n)
    if n_nodes <= 0:
        raise ValueError("Mercury latent graph has no nodes.")
    node_observations, handoff_n_obs = _handoff_node_observations(
        latent_graph=latent_graph,
        n_nodes=n_nodes,
        memory_length=int(memory_length),
    )
    n_obs = max(int(n_obs), int(handoff_n_obs))

    nodes_by_obs: list[list[int]] = [[] for _ in range(int(n_obs))]
    for node_id, obs_id in enumerate(node_observations.tolist()):
        nodes_by_obs[int(obs_id)].append(int(node_id))

    n_clones = np.asarray([max(1, len(nodes)) for nodes in nodes_by_obs], dtype=np.int64)
    total_states = int(n_clones.sum())
    node_to_state = np.full(n_nodes, -1, dtype=np.int64)
    state_to_node = np.full(total_states, -1, dtype=np.int64)
    state_to_observation = np.empty(total_states, dtype=np.int64)
    fallback_state_mask = np.zeros(total_states, dtype=bool)

    state_offset = 0
    for obs_id, nodes in enumerate(nodes_by_obs):
        if nodes:
            for clone_offset, node_id in enumerate(nodes):
                state_id = state_offset + clone_offset
                node_to_state[int(node_id)] = state_id
                state_to_node[state_id] = int(node_id)
                state_to_observation[state_id] = int(obs_id)
        else:
            fallback_state_mask[state_offset] = True
            state_to_observation[state_offset] = int(obs_id)
        state_offset += int(n_clones[obs_id])

    if np.any(node_to_state < 0):
        raise ValueError("Failed to assign every Mercury latent graph node to a CSCG clone state.")
    return HybridAssignment(
        n_clones=n_clones,
        node_to_state=node_to_state,
        node_to_observation=node_observations.astype(np.int64, copy=False),
        state_to_node=state_to_node,
        state_to_observation=state_to_observation,
        fallback_state_mask=fallback_state_mask,
    )


def build_source_graph_observation_assignment(
    *,
    graph: Graph,
    obs_values: np.ndarray,
    n_obs: int,
    clone_capacity_mode: str,
    clones_per_obs: int,
    memory_length: int | None = None,
) -> HybridAssignment:
    node_observations = infer_graph_node_observations(
        graph=graph,
        obs_values=obs_values,
        memory_length=memory_length,
    )
    capacity_mode = str(clone_capacity_mode).strip().lower()
    if capacity_mode == "source":
        n_clones = source_graph_capacity(node_to_observation=node_observations, n_obs=int(n_obs))
    else:
        n_clones = np.full(int(n_obs), int(clones_per_obs), dtype=np.int64)
    offsets = state_offsets(n_clones)
    nodes_seen_by_obs = np.zeros(int(n_obs), dtype=np.int64)
    total_states = int(n_clones.sum())
    node_to_state = np.zeros(int(graph.n), dtype=np.int64)
    state_to_node = np.full(total_states, -1, dtype=np.int64)
    state_to_observation = np.repeat(np.arange(int(n_obs), dtype=np.int64), n_clones)
    fallback_state_mask = np.ones(total_states, dtype=bool)
    for node_id, obs_id_raw in enumerate(node_observations.tolist()):
        obs_id = int(obs_id_raw)
        if obs_id < 0 or obs_id >= int(n_obs):
            obs_id = 0
        slot = min(int(nodes_seen_by_obs[obs_id]), int(n_clones[obs_id]) - 1)
        state_id = int(offsets[obs_id]) + slot
        nodes_seen_by_obs[obs_id] += 1
        node_to_state[int(node_id)] = state_id
        state_to_node[state_id] = int(node_id)
        fallback_state_mask[state_id] = False
    return HybridAssignment(
        n_clones=n_clones,
        node_to_state=node_to_state,
        node_to_observation=node_observations.astype(np.int64, copy=False),
        state_to_node=state_to_node,
        state_to_observation=state_to_observation,
        fallback_state_mask=fallback_state_mask,
    )


def map_eval_arrays(config: Any, obs_values: np.ndarray, action_values: np.ndarray) -> dict[str, Any]:
    observations, actions, collisions, _, _ = load_dataset_arrays(
        datasets_root=config.datasets_root,
        level=config.level,
        sensor=config.sensor,
        sensor_range=config.sensor_range,
        dataset_select=config.dataset_select,
        dataset_run_id=config.dataset_run_id,
    )
    from ..cscg.prepare import load_ground_truth_states

    ground_truth_bmu = load_ground_truth_states(config.ground_truth_states_parquet)
    observations, actions, filtered_ground_truth_bmu, valid_transition_stats = filter_valid_cscg_rows(
        observations=observations,
        actions=actions,
        collisions=collisions,
        ground_truth_bmu=ground_truth_bmu,
        valid_trajectories_only=bool(config.valid_trajectories_only),
    )
    obs_idx = map_to_indices(obs_values, observations).astype(np.int64, copy=False)
    act_idx = map_to_indices(action_values, actions).astype(np.int64, copy=False)
    ground_truth_bmu = np.asarray(filtered_ground_truth_bmu, dtype=np.int32)
    length = min(int(ground_truth_bmu.shape[0]), int(obs_idx.shape[0]), int(act_idx.shape[0]))
    if length < 3:
        raise ValueError("Mercury-CSCG hybrid could not align enough samples for evaluation.")
    return {
        "obs_idx": obs_idx[:length],
        "act_idx": act_idx[:length],
        "ground_truth_bmu": ground_truth_bmu[:length],
        "valid_transition_stats": valid_transition_stats,
    }
