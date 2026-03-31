from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence

import numpy as np

from src.maze_environment.LEVEL import levels
from src.maze_environment.agent import Agent
from src.maze_environment.maze_environment_v3 import MazeEnvironment
from src.utils.metrics import compute_cooccurrence_matrix, compute_precision

from ..io_parquet import LoadedDataset
from ..save_results import read_bundle_meta
from .runtime import atomic_write_json

DEFAULT_PAPER_PRECISION_NUM_WALKS = 100
DEFAULT_PAPER_PRECISION_WALK_LENGTH = 10_000


def resolve_eval_checkpoints(*, total_units: int, num_points: int | None, eval_interval: int) -> list[int]:
    total = int(total_units)
    if total <= 0:
        return []
    if num_points is not None:
        target_points = max(1, min(int(num_points), total))
        checkpoints = np.linspace(1, total, num=target_points, dtype=np.int64).tolist()
        if checkpoints[-1] != total:
            checkpoints.append(total)
        return sorted({int(value) for value in checkpoints if int(value) > 0})
    interval = max(1, int(eval_interval))
    checkpoints = list(range(interval, total + 1, interval))
    if not checkpoints or checkpoints[-1] != total:
        checkpoints.append(total)
    return sorted({int(value) for value in checkpoints if int(value) > 0})


@dataclass(frozen=True)
class EvaluationWalk:
    observations: np.ndarray
    cartesian_observations: np.ndarray
    actions: np.ndarray
    collisions: np.ndarray
    start_position: tuple[int, int]
    seed: int


def _open_positions(level_index: int) -> list[tuple[int, int]]:
    level = levels[int(level_index)]
    positions: list[tuple[int, int]] = []
    for row_index, row in enumerate(level):
        for col_index, value in enumerate(row):
            if value != "X":
                positions.append((row_index, col_index))
    if not positions:
        raise ValueError(f"Level {level_index} does not contain any accessible positions.")
    return positions


def _init_env(*, level_index: int, sensor: str, sensor_range: int | None, seed: int) -> MazeEnvironment:
    sensors: dict[str, Any] = {"sensor": sensor}
    if sensor == "cardinal distance":
        sensors["range"] = sensor_range
    return MazeEnvironment(level=levels[int(level_index)], plotting=False, agent_sensors=sensors, seed=int(seed))


def _set_start_position(env: MazeEnvironment, position: tuple[int, int]) -> None:
    env.agent_position = position
    env.agent.position = position
    env.agent.make_observation(env.maze)


def generate_random_start_walks(
    *,
    level: int,
    sensor: str,
    sensor_range: int | None,
    rand_prob: float,
    num_walks: int = DEFAULT_PAPER_PRECISION_NUM_WALKS,
    walk_length: int = DEFAULT_PAPER_PRECISION_WALK_LENGTH,
    base_seed: int = 0,
    valid_trajectories_only: bool = False,
) -> list[EvaluationWalk]:
    walks: list[EvaluationWalk] = []
    candidate_starts = _open_positions(level)
    for walk_index in range(int(num_walks)):
        walk_seed = int(base_seed) + walk_index
        chooser = random.Random(walk_seed)
        start_position = candidate_starts[chooser.randrange(len(candidate_starts))]
        env = _init_env(level_index=level, sensor=sensor, sensor_range=sensor_range, seed=walk_seed)
        _set_start_position(env, start_position)
        action = env.random_action()
        collision = True
        observations: list[Any] = []
        cartesian_observations: list[tuple[int, int]] = []
        actions: list[np.ndarray] = []
        collisions: list[bool] = []
        while len(observations) < int(walk_length):
            action = env.random_policy(action, collision, rand_prob)
            observation, action_vec, collision = env.step(action)
            if bool(valid_trajectories_only) and bool(collision):
                continue
            observations.append(observation)
            cartesian_observations.append(tuple(int(value) for value in env.agent.position))
            actions.append(np.asarray(action_vec))
            collisions.append(bool(collision))
        walks.append(
            EvaluationWalk(
                observations=np.asarray(observations),
                cartesian_observations=np.asarray(cartesian_observations),
                actions=np.asarray(actions),
                collisions=np.asarray(collisions, dtype=bool),
                start_position=start_position,
                seed=walk_seed,
            )
        )
    return walks


def loaded_dataset_from_walk(
    *,
    observations: np.ndarray,
    actions: np.ndarray,
    collisions: np.ndarray,
) -> LoadedDataset:
    return LoadedDataset(
        observations=np.asarray(observations),
        actions=np.asarray(actions),
        collisions=np.asarray(collisions, dtype=bool),
        source_metadata={},
        parquet_path=Path("<evaluation-walk>"),
    )


def cartesian_state_ids(positions: np.ndarray) -> np.ndarray:
    positions_arr = np.asarray(positions)
    if positions_arr.ndim != 2 or positions_arr.shape[1] != 2:
        raise ValueError(f"Expected cartesian positions with shape (N, 2), got {positions_arr.shape!r}")
    unique_positions = sorted({(int(row), int(col)) for row, col in positions_arr.tolist()})
    mapping = {position: index for index, position in enumerate(unique_positions)}
    return np.asarray([mapping[(int(row), int(col))] for row, col in positions_arr.tolist()], dtype=np.int64)


def exact_cartesian_reference_positions(level_index: int) -> np.ndarray:
    return np.asarray(_open_positions(level_index), dtype=np.int64)


def exact_cartesian_state_ids_for_level(*, level_index: int, positions: np.ndarray) -> np.ndarray:
    positions_arr = np.asarray(positions)
    if positions_arr.ndim != 2 or positions_arr.shape[1] != 2:
        raise ValueError(f"Expected cartesian positions with shape (N, 2), got {positions_arr.shape!r}")
    mapping = build_exact_indexer(exact_cartesian_reference_positions(level_index))
    try:
        return np.asarray(
            [mapping[(int(row), int(col))] for row, col in positions_arr.tolist()],
            dtype=np.int64,
        )
    except KeyError as exc:  # pragma: no cover
        raise ValueError(f"Encountered position not present in level {level_index}: {exc.args[0]!r}") from exc


def compute_fixed_map_precision_details(
    *,
    inferred_states: np.ndarray,
    ground_truth_states: np.ndarray,
    metric_name: str,
) -> dict[str, Any]:
    z_hat = np.asarray(inferred_states, dtype=np.int64)
    z_true = np.asarray(ground_truth_states, dtype=np.int64)
    if z_hat.shape[0] != z_true.shape[0]:
        raise ValueError("Inferred and ground-truth state sequences must have matching lengths.")
    if z_hat.size == 0:
        return {
            metric_name: 0.0,
            "evaluation_steps": 0,
            "inferred_state_count": 0,
            "ground_truth_state_count": 0,
            "cooccurrence_matrix": [],
        }
    cooccurrence = compute_cooccurrence_matrix(z_hat.tolist(), z_true.tolist())
    return {
        metric_name: float(compute_precision(z_hat.tolist(), z_true.tolist())),
        "evaluation_steps": int(z_hat.shape[0]),
        "inferred_state_count": int(len(np.unique(z_hat))),
        "ground_truth_state_count": int(len(np.unique(z_true))),
        "cooccurrence_matrix": cooccurrence.astype(int).tolist(),
    }


def compute_purity_from_cooccurrence(cooccurrence_matrix: Sequence[Sequence[int]]) -> float:
    matrix = np.asarray(cooccurrence_matrix, dtype=np.float64)
    if matrix.size == 0:
        return 0.0
    total = float(matrix.sum())
    if total <= 0.0:
        return 0.0
    return float(100.0 * np.max(matrix, axis=1).sum() / total)


def dominant_ground_truth_mapping_from_cooccurrence(cooccurrence_matrix: Sequence[Sequence[int]]) -> dict[int, int]:
    matrix = np.asarray(cooccurrence_matrix, dtype=np.int64)
    if matrix.size == 0:
        return {}
    return {
        int(node_index): int(np.argmax(matrix[node_index]))
        for node_index in range(int(matrix.shape[0]))
        if matrix.shape[1] > 0
    }


def exact_valid_sensorimotor_transitions(*, level_index: int) -> set[tuple[int, int, int]]:
    env = _init_env(level_index=level_index, sensor="cartesian", sensor_range=None, seed=0)
    state_map = build_exact_indexer(exact_cartesian_reference_positions(level_index))
    transitions: set[tuple[int, int, int]] = set()
    for position in _open_positions(level_index):
        env.agent.position = position
        env.agent_position = position
        env.agent.make_observation(env.maze)
        source_state = int(state_map[(int(position[0]), int(position[1]))])
        for action_name in env.action_keys:
            env.agent.position = position
            env.agent_position = position
            _observation, action_vec, _collision = env.step(action_name)
            next_position = tuple(int(value) for value in env.agent.position)
            action_id = int(np.argmax(np.asarray(action_vec, dtype=np.int32)))
            transitions.add((source_state, action_id, int(state_map[next_position])))
    return transitions


def project_graph_edges_to_ground_truth(
    *,
    graph: Any,
    node_to_ground_truth: dict[int, int],
) -> set[tuple[int, int, int]]:
    adj = np.asarray(graph.adj)
    if adj.ndim != 2:
        raise ValueError("Expected graph adjacency with shape (n, n).")
    action_feature = np.asarray(graph.edge_features["action"])
    actions = action_feature[:, :, 0] if action_feature.ndim == 3 else action_feature
    transitions: set[tuple[int, int, int]] = set()
    for source, target in np.argwhere(adj != 0).tolist():
        source_idx = int(source)
        target_idx = int(target)
        if source_idx not in node_to_ground_truth or target_idx not in node_to_ground_truth:
            continue
        transitions.add(
            (
                int(node_to_ground_truth[source_idx]),
                int(actions[source_idx, target_idx]),
                int(node_to_ground_truth[target_idx]),
            )
        )
    return transitions


def compute_sensorimotor_link_error(
    edges: Iterable[tuple[int, int, int]],
    valid_transitions: set[tuple[int, int, int]],
    tau: float = 1e-4,
) -> float:
    edge_set = {tuple(int(value) for value in edge) for edge in edges}
    invalid = sum(1 for edge in edge_set if edge not in valid_transitions)
    return float((invalid + float(tau)) / (len(edge_set) + float(tau)))


def compute_contingency_matrix(
    decoded_states: np.ndarray,
    true_states: np.ndarray,
    n_learned: int,
    n_true: int,
) -> np.ndarray:
    decoded = np.asarray(decoded_states, dtype=np.int64)
    truth = np.asarray(true_states, dtype=np.int64)
    if decoded.shape[0] != truth.shape[0]:
        raise ValueError("Decoded and true state sequences must have matching lengths.")
    matrix = np.zeros((int(n_learned), int(n_true)), dtype=np.int64)
    for decoded_state, true_state in zip(decoded.tolist(), truth.tolist(), strict=False):
        matrix[int(decoded_state), int(true_state)] += 1
    return matrix


def compute_alignment(C: np.ndarray) -> np.ndarray:
    contingency = np.asarray(C, dtype=np.int64)
    if contingency.ndim != 2:
        raise ValueError(f"Expected contingency matrix with shape (n_learned, n_true), got {contingency.shape!r}")
    if contingency.shape[1] == 0:
        return np.zeros(contingency.shape[0], dtype=np.int64)
    return np.argmax(contingency, axis=1).astype(np.int64, copy=False)


def compute_occupancy(decoded_states: np.ndarray, n_learned: int) -> np.ndarray:
    decoded = np.asarray(decoded_states, dtype=np.int64)
    occupancy = np.zeros(int(n_learned), dtype=np.float64)
    for decoded_state in decoded.tolist():
        occupancy[int(decoded_state)] += 1.0
    return occupancy


def aggregate_aligned_weighted_adjacency(
    W_hat: np.ndarray,
    A: np.ndarray,
    mu: np.ndarray,
    n_true: int,
) -> np.ndarray:
    learned = np.asarray(W_hat, dtype=np.float64)
    alignment = np.asarray(A, dtype=np.int64)
    occupancy = np.asarray(mu, dtype=np.float64)
    if learned.ndim != 2 or learned.shape[0] != learned.shape[1]:
        raise ValueError(f"Expected square learned adjacency, got {learned.shape!r}")
    if alignment.shape[0] != learned.shape[0] or occupancy.shape[0] != learned.shape[0]:
        raise ValueError("Alignment and occupancy must match the learned adjacency size.")
    n_true_states = int(n_true)
    aligned = np.zeros((n_true_states, n_true_states), dtype=np.float64)
    if n_true_states == 0:
        return aligned
    for true_state in range(n_true_states):
        source_indices = np.where(alignment == true_state)[0]
        if source_indices.size == 0:
            continue
        denom = float(occupancy[source_indices].sum())
        if denom <= 0.0:
            continue
        for source_index in source_indices.tolist():
            source_weight = float(occupancy[int(source_index)])
            if source_weight <= 0.0:
                continue
            for learned_target in range(learned.shape[1]):
                aligned[true_state, int(alignment[learned_target])] += source_weight * float(learned[source_index, learned_target])
        aligned[true_state] /= denom
    return aligned


def row_normalize(W: np.ndarray) -> np.ndarray:
    weights = np.asarray(W, dtype=np.float64)
    if weights.ndim != 2:
        raise ValueError(f"Expected matrix with shape (n, n), got {weights.shape!r}")
    row_sums = weights.sum(axis=1, keepdims=True)
    normalized = np.zeros_like(weights, dtype=np.float64)
    nonzero_rows = row_sums[:, 0] > 0.0
    if np.any(nonzero_rows):
        normalized[nonzero_rows] = weights[nonzero_rows] / row_sums[nonzero_rows]
    return normalized


def zero_diagonal(W: np.ndarray) -> np.ndarray:
    weights = np.asarray(W, dtype=np.float64).copy()
    if weights.ndim != 2 or weights.shape[0] != weights.shape[1]:
        raise ValueError(f"Expected square matrix, got {weights.shape!r}")
    np.fill_diagonal(weights, 0.0)
    return weights


def zero_action_tensor_diagonal(T: np.ndarray) -> np.ndarray:
    transitions = np.asarray(T, dtype=np.float64).copy()
    if transitions.ndim != 3 or transitions.shape[1] != transitions.shape[2]:
        raise ValueError(f"Expected action-conditioned tensor with shape (a, n, n), got {transitions.shape!r}")
    diag_indices = np.arange(transitions.shape[1])
    transitions[:, diag_indices, diag_indices] = 0.0
    return transitions


def compute_mean_tv(T_true: np.ndarray, T_aligned: np.ndarray) -> float:
    transition_true = np.asarray(T_true, dtype=np.float64)
    transition_aligned = np.asarray(T_aligned, dtype=np.float64)
    if transition_true.shape != transition_aligned.shape:
        raise ValueError("Transition matrices must have matching shapes.")
    if transition_true.size == 0:
        return 0.0
    return float(np.mean(0.5 * np.abs(transition_true - transition_aligned).sum(axis=1)))


def compute_edge_metrics(W_true: np.ndarray, W_aligned: np.ndarray, epsilon: float = 1e-6) -> dict[str, float]:
    weights_true = np.asarray(W_true, dtype=np.float64)
    weights_aligned = np.asarray(W_aligned, dtype=np.float64)
    if weights_true.shape != weights_aligned.shape:
        raise ValueError("Weighted adjacency matrices must have matching shapes.")
    threshold = float(epsilon)
    true_edges = set(map(tuple, np.argwhere(weights_true > threshold).tolist()))
    predicted_edges = set(map(tuple, np.argwhere(weights_aligned > threshold).tolist()))
    true_positives = len(true_edges & predicted_edges)
    precision = float(true_positives / len(predicted_edges)) if predicted_edges else 0.0
    recall = float(true_positives / len(true_edges)) if true_edges else 0.0
    f1 = float((2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0.0 else 0.0)
    return {
        "edge_precision": precision,
        "edge_recall": recall,
        "edge_f1": f1,
    }


def infer_num_actions_from_valid_transitions(valid_transitions: Iterable[tuple[int, int, int]]) -> int:
    action_ids = [int(action_id) for _source_state, action_id, _target_state in valid_transitions]
    return (max(action_ids) + 1) if action_ids else 0


def build_action_conditioned_tensor_from_valid_transitions(
    valid_transitions: Iterable[tuple[int, int, int]],
    n_states: int,
    n_actions: int | None = None,
    state_mapping: dict[int, int] | None = None,
) -> np.ndarray:
    transitions = [tuple(int(value) for value in transition) for transition in valid_transitions]
    resolved_n_actions = int(n_actions) if n_actions is not None else infer_num_actions_from_valid_transitions(transitions)
    tensor = np.zeros((resolved_n_actions, int(n_states), int(n_states)), dtype=np.float64)
    mapping = state_mapping or {}
    for source_state, action_id, target_state in transitions:
        source_idx = int(mapping.get(source_state, source_state))
        target_idx = int(mapping.get(target_state, target_state))
        if source_idx < 0 or target_idx < 0 or source_idx >= int(n_states) or target_idx >= int(n_states):
            continue
        if action_id < 0 or action_id >= resolved_n_actions:
            continue
        tensor[int(action_id), source_idx, target_idx] = 1.0
    return tensor


def action_ids_from_action_vectors(actions: np.ndarray) -> np.ndarray:
    action_array = np.asarray(actions)
    if action_array.ndim == 1:
        return action_array.astype(np.int64, copy=False)
    return np.argmax(action_array, axis=1).astype(np.int64, copy=False)


def build_action_conditioned_tensor_from_walks(
    *,
    state_walks: Iterable[np.ndarray],
    action_walks: Iterable[np.ndarray],
    n_states: int,
    n_actions: int,
) -> np.ndarray:
    tensor = np.zeros((int(n_actions), int(n_states), int(n_states)), dtype=np.float64)
    for state_walk, action_walk in zip(state_walks, action_walks, strict=False):
        states = np.asarray(state_walk, dtype=np.int64)
        actions = np.asarray(action_walk, dtype=np.int64)
        if states.shape[0] < 2:
            continue
        if actions.shape[0] == states.shape[0]:
            # Evaluation walks store action[t] alongside the post-step observation/state[t].
            # The transition states[t] -> states[t+1] is therefore conditioned by action[t+1].
            transition_actions = actions[1:]
        elif actions.shape[0] == states.shape[0] - 1:
            # Standard convention: one action per state transition.
            transition_actions = actions
        else:
            raise ValueError(
                "Action walks must either match state walk length (post-step action convention) "
                "or be exactly one shorter (transition action convention)."
            )
        for source_state, action_id, target_state in zip(
            states[:-1].tolist(),
            transition_actions.tolist(),
            states[1:].tolist(),
            strict=False,
        ):
            if action_id < 0 or action_id >= int(n_actions):
                continue
            tensor[int(action_id), int(source_state), int(target_state)] += 1.0
    return tensor


def build_action_conditioned_tensor_from_graph(
    graph: Any,
    *,
    n_actions: int,
    epsilon: float = 1e-6,
) -> np.ndarray:
    adj = np.asarray(graph.adj, dtype=np.float64)
    if adj.ndim != 2 or adj.shape[0] != adj.shape[1]:
        raise ValueError(f"Expected graph adjacency with shape (n, n), got {adj.shape!r}")
    action_feature = np.asarray(graph.edge_features["action"])
    actions = action_feature[:, :, 0] if action_feature.ndim == 3 else action_feature
    tensor = np.zeros((int(n_actions), int(adj.shape[0]), int(adj.shape[1])), dtype=np.float64)
    for source_idx, target_idx in np.argwhere(adj > float(epsilon)).tolist():
        action_id = int(actions[int(source_idx), int(target_idx)])
        if 0 <= action_id < int(n_actions):
            tensor[action_id, int(source_idx), int(target_idx)] = 1.0
    return tensor


def support_action_transition_tensor(
    transition_tensor: np.ndarray,
    *,
    epsilon: float = 1e-6,
) -> np.ndarray:
    transitions = np.asarray(transition_tensor, dtype=np.float64)
    if transitions.ndim != 3:
        raise ValueError(f"Expected transition tensor with shape (a, n, n), got {transitions.shape!r}")
    return (transitions > float(epsilon)).astype(np.float64, copy=False)


def _pad_action_tensor_to_n_actions(tensor: np.ndarray, n_actions: int) -> np.ndarray:
    padded = np.asarray(tensor, dtype=np.float64)
    if padded.ndim != 3:
        raise ValueError(f"Expected action-conditioned tensor with shape (a, n, n), got {padded.shape!r}")
    if padded.shape[0] == int(n_actions):
        return padded
    if padded.shape[0] > int(n_actions):
        return padded[: int(n_actions)]
    out = np.zeros((int(n_actions), padded.shape[1], padded.shape[2]), dtype=np.float64)
    out[: padded.shape[0]] = padded
    return out


def normalize_action_rows(tensor: np.ndarray) -> np.ndarray:
    weights = np.asarray(tensor, dtype=np.float64)
    if weights.ndim != 3:
        raise ValueError(f"Expected action-conditioned tensor with shape (a, n, n), got {weights.shape!r}")
    row_sums = weights.sum(axis=2, keepdims=True)
    normalized = np.zeros_like(weights, dtype=np.float64)
    nonzero_rows = row_sums > 0.0
    np.divide(weights, row_sums, out=normalized, where=nonzero_rows)
    return normalized


def aggregate_aligned_action_transition_tensor(
    T_hat: np.ndarray,
    A: np.ndarray,
    mu: np.ndarray,
    n_true: int,
) -> np.ndarray:
    learned = np.asarray(T_hat, dtype=np.float64)
    alignment = np.asarray(A, dtype=np.int64)
    occupancy = np.asarray(mu, dtype=np.float64)
    if learned.ndim != 3 or learned.shape[1] != learned.shape[2]:
        raise ValueError(f"Expected action-conditioned tensor with shape (a, n, n), got {learned.shape!r}")
    if alignment.shape[0] != learned.shape[1] or occupancy.shape[0] != learned.shape[1]:
        raise ValueError("Alignment and occupancy must match the learned state count.")
    n_actions = int(learned.shape[0])
    n_true_states = int(n_true)
    aligned = np.zeros((n_actions, n_true_states, n_true_states), dtype=np.float64)
    if n_true_states == 0:
        return aligned
    for true_state in range(n_true_states):
        source_indices = np.where(alignment == true_state)[0]
        if source_indices.size == 0:
            continue
        denom = float(occupancy[source_indices].sum())
        if denom <= 0.0:
            continue
        for source_index in source_indices.tolist():
            source_weight = float(occupancy[int(source_index)])
            if source_weight <= 0.0:
                continue
            for action_idx in range(n_actions):
                row = learned[action_idx, int(source_index)]
                if not np.any(row > 0.0):
                    continue
                for learned_target, value in enumerate(row.tolist()):
                    if value <= 0.0:
                        continue
                    aligned[action_idx, true_state, int(alignment[int(learned_target)])] += source_weight * float(value)
        aligned[:, true_state, :] /= denom
    return aligned


def compute_action_conditioned_edge_metrics(
    T_true: np.ndarray,
    T_aligned: np.ndarray,
    epsilon: float = 1e-6,
) -> dict[str, float]:
    truth = np.asarray(T_true, dtype=np.float64)
    predicted = np.asarray(T_aligned, dtype=np.float64)
    if truth.shape != predicted.shape:
        raise ValueError("Action-conditioned tensors must have matching shapes.")
    threshold = float(epsilon)
    true_edges = truth > threshold
    predicted_edges = predicted > threshold
    true_positives = int(np.count_nonzero(true_edges & predicted_edges))
    predicted_count = int(np.count_nonzero(predicted_edges))
    true_count = int(np.count_nonzero(true_edges))
    precision = float(true_positives / predicted_count) if predicted_count > 0 else 0.0
    recall = float(true_positives / true_count) if true_count > 0 else 0.0
    f1 = float((2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0.0 else 0.0)
    return {
        "edge_precision": precision,
        "edge_recall": recall,
        "edge_f1": f1,
    }


def compute_action_conditioned_mean_tv(
    T_true: np.ndarray,
    T_aligned: np.ndarray,
    epsilon: float = 1e-6,
) -> float:
    truth = np.asarray(T_true, dtype=np.float64)
    predicted = np.asarray(T_aligned, dtype=np.float64)
    if truth.shape != predicted.shape:
        raise ValueError("Action-conditioned tensors must have matching shapes.")
    normalized_truth = normalize_action_rows(truth)
    normalized_predicted = normalize_action_rows(predicted)
    valid_pairs = truth.sum(axis=2) > float(epsilon)
    if not np.any(valid_pairs):
        return 0.0
    tv_values: list[float] = []
    for action_idx, source_idx in np.argwhere(valid_pairs).tolist():
        predicted_row = predicted[int(action_idx), int(source_idx)]
        if float(predicted_row.sum()) <= float(epsilon):
            tv_values.append(1.0)
            continue
        delta = np.abs(
            normalized_predicted[int(action_idx), int(source_idx)]
            - normalized_truth[int(action_idx), int(source_idx)]
        )
        tv_values.append(float(0.5 * delta.sum()))
    return float(np.mean(tv_values)) if tv_values else 0.0


def build_weighted_adjacency_from_walks(state_walks: Iterable[np.ndarray], n_states: int) -> np.ndarray:
    adjacency = np.zeros((int(n_states), int(n_states)), dtype=np.float64)
    for walk in state_walks:
        states = np.asarray(walk, dtype=np.int64)
        if states.shape[0] < 2:
            continue
        for source_state, target_state in zip(states[:-1].tolist(), states[1:].tolist(), strict=False):
            adjacency[int(source_state), int(target_state)] += 1.0
    return adjacency


def collapse_action_transition_tensor_to_adjacency(transition_tensor: np.ndarray) -> np.ndarray:
    transitions = np.asarray(transition_tensor, dtype=np.float64)
    if transitions.ndim == 2:
        if transitions.shape[0] != transitions.shape[1]:
            raise ValueError(f"Expected square transition matrix, got {transitions.shape!r}")
        return transitions
    if transitions.ndim != 3:
        raise ValueError(f"Expected transition tensor with shape (a, n, n), got {transitions.shape!r}")
    return transitions.sum(axis=0)


def build_weighted_adjacency_from_valid_transitions(
    valid_transitions: Iterable[tuple[int, int, int]],
    n_states: int,
    state_mapping: dict[int, int] | None = None,
) -> np.ndarray:
    adjacency = np.zeros((int(n_states), int(n_states)), dtype=np.float64)
    mapping = state_mapping or {}
    for source_state, _action_id, target_state in valid_transitions:
        source_idx = int(mapping.get(int(source_state), int(source_state)))
        target_idx = int(mapping.get(int(target_state), int(target_state)))
        if source_idx < 0 or target_idx < 0 or source_idx >= int(n_states) or target_idx >= int(n_states):
            continue
        adjacency[source_idx, target_idx] = 1.0
    return adjacency


def remap_position_walks_to_global_ids(position_walks: Iterable[np.ndarray]) -> list[np.ndarray]:
    walks = [np.asarray(walk) for walk in position_walks]
    if not walks:
        return []
    lengths = [int(walk.shape[0]) for walk in walks]
    concatenated = np.concatenate(walks, axis=0)
    global_ids = cartesian_state_ids(concatenated)
    remapped: list[np.ndarray] = []
    offset = 0
    for length in lengths:
        remapped.append(np.asarray(global_ids[offset:offset + length], dtype=np.int64))
        offset += length
    return remapped


def compute_weighted_structure_metrics(
    *,
    decoded_walks: Iterable[np.ndarray],
    true_walks: Iterable[np.ndarray],
    W_hat: np.ndarray,
    n_true: int,
    W_true: np.ndarray | None = None,
    epsilon: float = 1e-6,
    ignore_self_loops: bool = False,
) -> dict[str, float]:
    decoded_series = concatenate_walk_series(decoded_walks)
    true_series = concatenate_walk_series(true_walks)
    learned = np.asarray(W_hat, dtype=np.float64)
    if learned.ndim != 2 or learned.shape[0] != learned.shape[1]:
        raise ValueError(f"Expected square learned adjacency, got {learned.shape!r}")
    n_learned = int(learned.shape[0])
    n_true_states = int(n_true)
    contingency = compute_contingency_matrix(
        decoded_states=decoded_series,
        true_states=true_series,
        n_learned=n_learned,
        n_true=n_true_states,
    )
    alignment = compute_alignment(contingency)
    occupancy = compute_occupancy(decoded_series, n_learned=n_learned)
    aligned_adjacency = aggregate_aligned_weighted_adjacency(
        W_hat=learned,
        A=alignment,
        mu=occupancy,
        n_true=n_true_states,
    )
    true_adjacency = (
        np.asarray(W_true, dtype=np.float64)
        if W_true is not None
        else build_weighted_adjacency_from_walks(true_walks, n_states=n_true_states)
    )
    if ignore_self_loops:
        true_adjacency = zero_diagonal(true_adjacency)
        aligned_adjacency = zero_diagonal(aligned_adjacency)
    return {
        "mean_total_variation": compute_mean_tv(
            row_normalize(true_adjacency),
            row_normalize(aligned_adjacency),
        ),
        **compute_edge_metrics(true_adjacency, aligned_adjacency, epsilon=float(epsilon)),
    }


def compute_action_conditioned_structure_metrics(
    *,
    decoded_walks: Iterable[np.ndarray],
    true_walks: Iterable[np.ndarray],
    action_walks: Iterable[np.ndarray],
    T_hat: np.ndarray,
    n_true: int,
    T_true: np.ndarray | None = None,
    epsilon: float = 1e-6,
    ignore_self_loops: bool = False,
) -> dict[str, float]:
    decoded_series = concatenate_walk_series(decoded_walks)
    true_series = concatenate_walk_series(true_walks)
    learned_support = support_action_transition_tensor(T_hat, epsilon=float(epsilon))
    n_learned = int(learned_support.shape[1])
    n_true_states = int(n_true)
    contingency = compute_contingency_matrix(
        decoded_states=decoded_series,
        true_states=true_series,
        n_learned=n_learned,
        n_true=n_true_states,
    )
    alignment = compute_alignment(contingency)
    occupancy = compute_occupancy(decoded_series, n_learned=n_learned)
    learned_uniform = normalize_action_rows(learned_support)
    aligned_tensor = aggregate_aligned_action_transition_tensor(
        learned_uniform,
        alignment,
        occupancy,
        n_true=n_true_states,
    )
    true_tensor = (
        np.asarray(T_true, dtype=np.float64)
        if T_true is not None
        else build_action_conditioned_tensor_from_walks(
            state_walks=true_walks,
            action_walks=action_walks,
            n_states=n_true_states,
            n_actions=int(learned_support.shape[0]),
        )
    )
    resolved_n_actions = max(int(true_tensor.shape[0]), int(aligned_tensor.shape[0]))
    true_tensor = _pad_action_tensor_to_n_actions(true_tensor, resolved_n_actions)
    aligned_tensor = _pad_action_tensor_to_n_actions(aligned_tensor, resolved_n_actions)
    if ignore_self_loops:
        true_tensor = zero_action_tensor_diagonal(true_tensor)
        aligned_tensor = zero_action_tensor_diagonal(aligned_tensor)
    return {
        "mean_total_variation": compute_action_conditioned_mean_tv(
            true_tensor,
            aligned_tensor,
            epsilon=float(epsilon),
        ),
        **compute_action_conditioned_edge_metrics(
            true_tensor,
            aligned_tensor,
            epsilon=float(epsilon),
        ),
    }


def write_paper_precision_artifact(*, path: Path, payload: Dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(path, payload)
    return path


def summarize_paper_precision_artifact(path: Path) -> Dict[str, Any]:
    import json

    payload = json.loads(path.read_text(encoding="utf-8"))
    metrics_by_capacity = dict(payload.get("metrics_by_capacity") or {})
    if metrics_by_capacity:
        protocol = dict(payload.get("protocol") or {})
        summarized_by_capacity: dict[str, Dict[str, Any]] = {}
        for capacity, metrics in metrics_by_capacity.items():
            metric_payload = dict(metrics or {})
            summary = {key: value for key, value in metric_payload.items() if isinstance(value, (int, float))}
            for key in ("evaluation_steps", "inferred_state_count", "ground_truth_state_count"):
                if key in metric_payload:
                    summary[key] = metric_payload[key]
            summarized_by_capacity[str(capacity)] = summary
        return {
            "resolved_capacities": [int(value) for value in (protocol.get("resolved_capacities") or [])],
            "primary_capacity": protocol.get("primary_capacity"),
            "metrics_by_capacity": summarized_by_capacity,
        }
    metrics = dict(payload.get("metrics") or {})
    protocol = dict(payload.get("protocol") or {})
    summary = {key: value for key, value in metrics.items() if isinstance(value, (int, float))}
    for key in ("evaluation_steps", "inferred_state_count", "ground_truth_state_count"):
        if key in metrics:
            summary[key] = metrics[key]
    for key in ("num_walks", "walk_length"):
        if key in protocol:
            summary[key] = protocol[key]
    return summary


def resolve_dataset_parquet_from_bundle(states_path: Path, datasets_root: Path = Path("datasets")) -> Path:
    meta = read_bundle_meta(states_path)
    if meta is None or not meta.source.dataset_parquet_name:
        raise RuntimeError(f"Missing source dataset metadata in {states_path}")
    candidates = sorted(datasets_root.rglob(meta.source.dataset_parquet_name))
    if not candidates:
        raise FileNotFoundError(f"Dataset parquet {meta.source.dataset_parquet_name} not found under {datasets_root}")
    return candidates[-1]


def build_exact_indexer(reference_values: np.ndarray) -> dict[Any, int]:
    if reference_values.ndim == 1:
        return {int(value): index for index, value in enumerate(np.asarray(reference_values).tolist())}
    return {tuple(np.asarray(row).tolist()): index for index, row in enumerate(np.asarray(reference_values).tolist())}


def map_values_with_reference(*, reference_uniques: np.ndarray, values: np.ndarray) -> np.ndarray:
    mapping = build_exact_indexer(np.asarray(reference_uniques))
    if values.ndim == 1:
        try:
            return np.asarray([mapping[int(value)] for value in np.asarray(values).tolist()], dtype=np.int64)
        except KeyError as exc:  # pragma: no cover
            raise ValueError(f"Encountered evaluation value not present in the training vocabulary: {exc.args[0]!r}") from exc
    try:
        return np.asarray([mapping[tuple(np.asarray(row).tolist())] for row in np.asarray(values).tolist()], dtype=np.int64)
    except KeyError as exc:  # pragma: no cover
        raise ValueError(f"Encountered evaluation row not present in the training vocabulary: {exc.args[0]!r}") from exc


def concatenate_walk_series(walk_values: Iterable[np.ndarray]) -> np.ndarray:
    series = [np.asarray(values) for values in walk_values]
    if not series:
        return np.asarray([], dtype=np.int64)
    return np.concatenate(series, axis=0)
