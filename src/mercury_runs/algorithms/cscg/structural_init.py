from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ...infrastructure.paper_precision import (
    action_ids_from_action_vectors,
    exact_valid_sensorimotor_transitions,
)


@dataclass(frozen=True)
class StructuralInitResult:
    counts: np.ndarray
    n_clones: np.ndarray
    state_to_observation: np.ndarray
    source: str
    transfer_level: str
    init_mode: str
    capacity_mode: str
    source_method: str | None = None
    source_graph_path: str | None = None
    oracle_source: str | None = None


def normalize_structural_options(
    *,
    transfer_level: str,
    init_mode: str,
    structural_source: str,
    capacity_mode: str,
) -> tuple[str, str, str, str]:
    level = str(transfer_level).strip().lower()
    mode = str(init_mode).strip().lower()
    source = str(structural_source).strip().lower()
    capacity = str(capacity_mode).strip().lower()
    if level not in {"clone_state", "first_order_observation"}:
        raise ValueError(f"Unsupported CSCG structural_transfer_level: {transfer_level}")
    if mode not in {"none", "binary", "count"}:
        raise ValueError(f"Unsupported CSCG structural_init_mode: {init_mode}")
    if source not in {"none", "oracle_maze_topology", "oracle_empirical_ground_truth", "source_graph"}:
        raise ValueError(f"Unsupported CSCG structural_source: {structural_source}")
    if capacity not in {"source", "ground_truth", "fixed"}:
        raise ValueError(f"Unsupported CSCG clone_capacity_mode: {capacity}")
    return level, mode, source, capacity


def state_offsets(n_clones: np.ndarray) -> np.ndarray:
    counts = np.asarray(n_clones, dtype=np.int64)
    return np.concatenate([[0], np.cumsum(counts[:-1])]).astype(np.int64, copy=False)


def block_state_to_observation(n_clones: np.ndarray) -> np.ndarray:
    return np.repeat(np.arange(int(len(n_clones)), dtype=np.int64), np.asarray(n_clones, dtype=np.int64))


def fixed_or_ground_truth_capacity(
    *,
    mode: str,
    clone_allocation: str,
    clones_per_obs: int,
    clones_per_state: int,
    n_obs: int,
    obs_idx: np.ndarray,
    ground_truth_states: np.ndarray | None,
) -> np.ndarray:
    capacity = str(mode).strip().lower()
    allocation = str(clone_allocation).strip().lower()
    if capacity == "fixed":
        if allocation in {"ground_truth", "oracle_ground_truth"} and ground_truth_states is not None:
            return ground_truth_capacity(
                n_obs=n_obs,
                obs_idx=obs_idx,
                ground_truth_states=ground_truth_states,
                clones_per_state=clones_per_state,
            )
        return np.full(int(n_obs), int(clones_per_obs), dtype=np.int64)
    if capacity == "ground_truth":
        if ground_truth_states is None:
            raise ValueError("CSCG clone_capacity_mode='ground_truth' requires aligned ground-truth states.")
        return ground_truth_capacity(
            n_obs=n_obs,
            obs_idx=obs_idx,
            ground_truth_states=ground_truth_states,
            clones_per_state=clones_per_state,
        )
    raise ValueError("source capacity is only available from a source graph assignment.")


def ground_truth_capacity(
    *,
    n_obs: int,
    obs_idx: np.ndarray,
    ground_truth_states: np.ndarray,
    clones_per_state: int,
) -> np.ndarray:
    obs = np.asarray(obs_idx, dtype=np.int64)
    gt = np.asarray(ground_truth_states)
    if obs.shape[0] != gt.shape[0]:
        raise ValueError("Ground-truth clone capacity requires observation and ground-truth arrays with matching lengths.")
    n_clones = np.zeros(int(n_obs), dtype=np.int64)
    for obs_id in range(int(n_obs)):
        n_clones[obs_id] = max(1, int(np.unique(gt[obs == obs_id]).shape[0])) * int(clones_per_state)
    return n_clones


def observation_edges_from_empirical_ground_truth(
    *,
    obs_idx: np.ndarray,
    action_idx: np.ndarray,
    ground_truth_states: np.ndarray,
    n_obs: int,
    n_actions: int,
    mode: str,
) -> np.ndarray:
    obs = np.asarray(obs_idx, dtype=np.int64)
    actions = np.asarray(action_idx, dtype=np.int64)
    gt = np.asarray(ground_truth_states)
    tensor = np.zeros((int(n_actions), int(n_obs), int(n_obs)), dtype=np.float32)
    for i in range(max(0, obs.shape[0] - 1)):
        if gt[i] == gt[i + 1] and obs[i] == obs[i + 1]:
            continue
        action_id = int(actions[i + 1] if actions.shape[0] == obs.shape[0] else actions[i])
        if 0 <= action_id < int(n_actions):
            tensor[action_id, int(obs[i]), int(obs[i + 1])] += 1.0
    return _apply_init_mode(tensor, mode)


def observation_edges_from_oracle_topology(
    *,
    level: int,
    obs_idx: np.ndarray,
    action_idx: np.ndarray,
    ground_truth_states: np.ndarray,
    n_obs: int,
    n_actions: int,
    mode: str,
) -> np.ndarray:
    gt_to_obs: dict[int, int] = {}
    for obs_id, gt_state in zip(np.asarray(obs_idx, dtype=np.int64).tolist(), np.asarray(ground_truth_states).tolist(), strict=False):
        gt_to_obs.setdefault(int(gt_state), int(obs_id))
    tensor = np.zeros((int(n_actions), int(n_obs), int(n_obs)), dtype=np.float32)
    try:
        valid_transitions = exact_valid_sensorimotor_transitions(level_index=int(level))
    except Exception:
        valid_transitions = set()
    for source_state, action_id, target_state in valid_transitions:
        if int(source_state) not in gt_to_obs or int(target_state) not in gt_to_obs:
            continue
        if 0 <= int(action_id) < int(n_actions):
            tensor[int(action_id), gt_to_obs[int(source_state)], gt_to_obs[int(target_state)]] += 1.0
    return _apply_init_mode(tensor, mode)


def clone_edges_from_oracle_topology(
    *,
    level: int,
    obs_idx: np.ndarray,
    action_idx: np.ndarray,
    ground_truth_states: np.ndarray,
    n_obs: int,
    n_actions: int,
    clones_per_state: int,
    mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    gt_by_obs: list[list[int]] = []
    obs = np.asarray(obs_idx, dtype=np.int64)
    gt = np.asarray(ground_truth_states)
    for obs_id in range(int(n_obs)):
        values = sorted(int(value) for value in np.unique(gt[obs == obs_id]).tolist())
        gt_by_obs.append(values or [-1])
    n_clones = np.asarray([max(1, len(values)) * int(clones_per_state) for values in gt_by_obs], dtype=np.int64)
    offsets = state_offsets(n_clones)
    gt_to_states: dict[int, list[int]] = {}
    for obs_id, gt_values in enumerate(gt_by_obs):
        base = int(offsets[obs_id])
        cursor = 0
        for gt_state in gt_values:
            states = [base + cursor + clone for clone in range(int(clones_per_state))]
            cursor += int(clones_per_state)
            if int(gt_state) >= 0:
                gt_to_states[int(gt_state)] = states
    counts = np.zeros((int(n_actions), int(n_clones.sum()), int(n_clones.sum())), dtype=np.float32)
    try:
        valid_transitions = exact_valid_sensorimotor_transitions(level_index=int(level))
    except Exception:
        valid_transitions = set()
    for source_state, action_id, target_state in valid_transitions:
        if int(source_state) not in gt_to_states or int(target_state) not in gt_to_states:
            continue
        if not (0 <= int(action_id) < int(n_actions)):
            continue
        for src in gt_to_states[int(source_state)]:
            for dst in gt_to_states[int(target_state)]:
                counts[int(action_id), int(src), int(dst)] += 1.0 / float(len(gt_to_states[int(source_state)]) * len(gt_to_states[int(target_state)]))
    return _apply_init_mode(counts, mode), n_clones


def observation_edges_from_graph(
    *,
    graph: Any,
    node_to_observation: np.ndarray,
    n_obs: int,
    n_actions: int,
    mode: str,
) -> np.ndarray:
    adj = np.asarray(graph.adj, dtype=np.float32)
    action_edges = _graph_action_edges(graph)
    tensor = np.zeros((int(n_actions), int(n_obs), int(n_obs)), dtype=np.float32)
    for source_node, target_node in np.argwhere(adj[: int(graph.n), : int(graph.n)] != 0.0).tolist():
        source_obs = int(node_to_observation[int(source_node)])
        target_obs = int(node_to_observation[int(target_node)])
        action_id = int(action_edges[int(source_node), int(target_node)]) if action_edges is not None else 0
        if 0 <= source_obs < int(n_obs) and 0 <= target_obs < int(n_obs) and 0 <= action_id < int(n_actions):
            tensor[action_id, source_obs, target_obs] += float(adj[int(source_node), int(target_node)])
    return _apply_init_mode(tensor, mode)


def clone_edges_from_graph(
    *,
    graph: Any,
    node_to_state: np.ndarray,
    n_states: int,
    n_actions: int,
    mode: str,
) -> np.ndarray:
    adj = np.asarray(graph.adj, dtype=np.float32)
    action_edges = _graph_action_edges(graph)
    counts = np.zeros((int(n_actions), int(n_states), int(n_states)), dtype=np.float32)
    for source_node, target_node in np.argwhere(adj[: int(graph.n), : int(graph.n)] != 0.0).tolist():
        source_state = int(node_to_state[int(source_node)])
        target_state = int(node_to_state[int(target_node)])
        action_id = int(action_edges[int(source_node), int(target_node)]) if action_edges is not None else 0
        if 0 <= source_state < int(n_states) and 0 <= target_state < int(n_states) and 0 <= action_id < int(n_actions):
            counts[action_id, source_state, target_state] += float(adj[int(source_node), int(target_node)])
    return _apply_init_mode(counts, mode)


def apply_observation_tensor_to_clone_counts(
    *,
    observation_tensor: np.ndarray,
    n_clones: np.ndarray,
) -> np.ndarray:
    obs_tensor = np.asarray(observation_tensor, dtype=np.float32)
    offsets = state_offsets(n_clones)
    counts = np.zeros((obs_tensor.shape[0], int(np.asarray(n_clones).sum()), int(np.asarray(n_clones).sum())), dtype=np.float32)
    for action_id, source_obs, target_obs in np.argwhere(obs_tensor != 0.0).tolist():
        src_start = int(offsets[int(source_obs)])
        dst_start = int(offsets[int(target_obs)])
        src_n = int(n_clones[int(source_obs)])
        dst_n = int(n_clones[int(target_obs)])
        counts[int(action_id), src_start : src_start + src_n, dst_start : dst_start + dst_n] += (
            float(obs_tensor[int(action_id), int(source_obs), int(target_obs)]) / float(src_n * dst_n)
        )
    return counts


def infer_graph_node_observations(*, graph: Any, obs_values: np.ndarray, memory_length: int | None = None) -> np.ndarray:
    if "mem_adj" in getattr(graph, "node_features", {}):
        mem_adj = np.asarray(graph.node_features["mem_adj"], dtype=np.float32)[: int(graph.n)]
        if memory_length is None or int(memory_length) <= 0 or mem_adj.shape[1] % int(memory_length) != 0:
            return np.argmax(mem_adj, axis=1).astype(np.int64, copy=False)
        n_sensory = int(mem_adj.shape[1] // int(memory_length))
        templates = mem_adj.reshape(int(graph.n), n_sensory, int(memory_length))[:, :, 0]
        return np.argmax(templates, axis=1).astype(np.int64, copy=False)
    if "weight" not in getattr(graph, "node_features", {}):
        raise ValueError("Source graph needs either 'mem_adj' or 'weight' node features for observation transfer.")
    weights = np.asarray(graph.node_features["weight"], dtype=np.float32)[: int(graph.n)]
    obs = np.asarray(obs_values, dtype=np.float32)
    if obs.ndim == 1:
        obs = obs.reshape(-1, 1)
    if weights.ndim == 1:
        weights = weights.reshape(-1, 1)
    distances = ((weights[:, None, :] - obs[None, :, :]) ** 2).sum(axis=2)
    return np.argmin(distances, axis=1).astype(np.int64, copy=False)


def source_graph_capacity(*, node_to_observation: np.ndarray, n_obs: int) -> np.ndarray:
    counts = np.zeros(int(n_obs), dtype=np.int64)
    for obs_id in np.asarray(node_to_observation, dtype=np.int64).tolist():
        if 0 <= int(obs_id) < int(n_obs):
            counts[int(obs_id)] += 1
    counts[counts == 0] = 1
    return counts


def empirical_observation_tensor(*, obs_idx: np.ndarray, action_idx: np.ndarray, n_obs: int, n_actions: int, mode: str) -> np.ndarray:
    obs = np.asarray(obs_idx, dtype=np.int64)
    actions = np.asarray(action_idx, dtype=np.int64)
    tensor = np.zeros((int(n_actions), int(n_obs), int(n_obs)), dtype=np.float32)
    for i in range(max(0, obs.shape[0] - 1)):
        action_id = int(actions[i + 1] if actions.shape[0] == obs.shape[0] else actions[i])
        if 0 <= action_id < int(n_actions):
            tensor[action_id, int(obs[i]), int(obs[i + 1])] += 1.0
    return _apply_init_mode(tensor, mode)


def _graph_action_edges(graph: Any) -> np.ndarray | None:
    if "action" not in getattr(graph, "edge_features", {}):
        return None
    action_feature = np.asarray(graph.edge_features["action"])
    return action_feature[:, :, 0] if action_feature.ndim == 3 else action_feature


def _apply_init_mode(values: np.ndarray, mode: str) -> np.ndarray:
    if str(mode).strip().lower() == "none":
        return np.zeros_like(values, dtype=np.float32)
    if str(mode).strip().lower() == "binary":
        return (np.asarray(values) != 0.0).astype(np.float32)
    if str(mode).strip().lower() == "count":
        return np.asarray(values, dtype=np.float32)
    raise ValueError(f"Unsupported CSCG structural_init_mode: {mode}")
