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


def resolve_eval_checkpoints(
    *,
    total_units: int,
    num_points: int | None,
    eval_interval: int,
    schedule: str = "linear",
    min_checkpoint: int = 10,
    explicit_checkpoints: Sequence[int] | None = None,
) -> list[int]:
    total = int(total_units)
    if total <= 0:
        return []

    if explicit_checkpoints is not None:
        checkpoints = sorted({int(value) for value in explicit_checkpoints if 0 < int(value) <= total})
        if not checkpoints:
            raise ValueError("paper_precision_eval_checkpoints must contain at least one checkpoint within the run length.")
        if checkpoints[-1] != total:
            checkpoints.append(total)
        return checkpoints

    if num_points is not None:
        target_points = max(1, min(int(num_points), total))

        if schedule == "log":
            start = max(1, min(int(min_checkpoint), total))
            raw = np.geomspace(start, total, num=target_points)
            checkpoints = np.rint(raw).astype(np.int64).tolist()
        else:
            checkpoints = np.linspace(1, total, num=target_points, dtype=np.int64).tolist()

        return sorted({int(value) for value in checkpoints if int(value) > 0})

    interval = max(1, int(eval_interval))
    checkpoints = list(range(interval, total + 1, interval))
    checkpoints.append(total)
    return sorted({int(value) for value in checkpoints if int(value) > 0})

def resolve_structure_metric_epsilons(config: Any) -> list[float]:
    raw_values = getattr(config, "structure_metrics_epsilons", None)
    if raw_values is None:
        raw_values = [getattr(config, "structure_metrics_epsilon", 1e-6)]
    elif isinstance(raw_values, (int, float)):
        raw_values = [raw_values]

    resolved: list[float] = []
    for raw_value in raw_values:
        value = float(raw_value)
        if value < 0.0:
            raise ValueError("Structure metric epsilons must be non-negative.")
        if value not in resolved:
            resolved.append(value)
    if not resolved:
        resolved.append(float(getattr(config, "structure_metrics_epsilon", 1e-6)))
    return resolved


def structure_epsilon_key(epsilon: float) -> str:
    return f"{float(epsilon):.12g}"


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
    inferred_state_count: int | None = None,
    ground_truth_state_count: int | None = None,
) -> dict[str, Any]:
    z_hat = np.asarray(inferred_states, dtype=np.int64)
    z_true = np.asarray(ground_truth_states, dtype=np.int64)
    if z_hat.shape[0] != z_true.shape[0]:
        raise ValueError("Inferred and ground-truth state sequences must have matching lengths.")
    if z_hat.size == 0:
        recall_name = metric_name.replace("precision", "recall")
        return {
            metric_name: 0.0,
            recall_name: 0.0,
            "evaluation_steps": 0,
            "inferred_state_count": 0,
            "ground_truth_state_count": 0,
            "cooccurrence_matrix": [],
        }
    if inferred_state_count is not None or ground_truth_state_count is not None:
        cooccurrence = dense_integer_cooccurrence_matrix(
            z_hat,
            z_true,
            n_inferred=inferred_state_count,
            n_true=ground_truth_state_count,
        )
    else:
        cooccurrence = compute_cooccurrence_matrix(z_hat.tolist(), z_true.tolist())
    total = float(z_hat.shape[0])
    recall_name = metric_name.replace("precision", "recall")
    return {
        metric_name: float(np.max(cooccurrence, axis=1).sum() / total) if total > 0.0 else 0.0,
        recall_name: float(np.max(cooccurrence, axis=0).sum() / total) if total > 0.0 else 0.0,
        "evaluation_steps": int(z_hat.shape[0]),
        "inferred_state_count": int(len(np.unique(z_hat))),
        "ground_truth_state_count": int(len(np.unique(z_true))),
        "cooccurrence_matrix": cooccurrence.astype(int).tolist(),
    }


def dense_integer_cooccurrence_matrix(
    inferred_states: np.ndarray,
    ground_truth_states: np.ndarray,
    *,
    n_inferred: int | None = None,
    n_true: int | None = None,
) -> np.ndarray:
    inferred = np.asarray(inferred_states, dtype=np.int64)
    truth = np.asarray(ground_truth_states, dtype=np.int64)
    if inferred.shape[0] != truth.shape[0]:
        raise ValueError("Inferred and ground-truth state sequences must have matching lengths.")
    inferred_size = int(n_inferred) if n_inferred is not None else (int(inferred.max()) + 1 if inferred.size else 0)
    truth_size = int(n_true) if n_true is not None else (int(truth.max()) + 1 if truth.size else 0)
    matrix = np.zeros((max(inferred_size, 0), max(truth_size, 0)), dtype=np.int64)
    valid = (
        (0 <= inferred) & (inferred < matrix.shape[0])
        & (0 <= truth) & (truth < matrix.shape[1])
    )
    np.add.at(matrix, (inferred[valid], truth[valid]), 1)
    return matrix


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


def compute_mean_tv(T_true: np.ndarray, T_aligned: np.ndarray, epsilon: float = 1e-6) -> float:
    transition_true = np.asarray(T_true, dtype=np.float64)
    transition_aligned = np.asarray(T_aligned, dtype=np.float64)
    if transition_true.shape != transition_aligned.shape:
        raise ValueError("Transition matrices must have matching shapes.")
    if transition_true.size == 0:
        return 0.0
    valid_rows = transition_true.sum(axis=1) > float(epsilon)
    if not np.any(valid_rows):
        return 0.0
    tv_values: list[float] = []
    for source_idx in np.flatnonzero(valid_rows).tolist():
        predicted_row = transition_aligned[int(source_idx)]
        if float(predicted_row.sum()) <= float(epsilon):
            tv_values.append(1.0)
            continue
        delta = np.abs(predicted_row - transition_true[int(source_idx)])
        tv_values.append(float(0.5 * delta.sum()))
    return float(np.mean(tv_values)) if tv_values else 0.0


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


_ACTION_ID_BY_CARTESIAN_DISPLACEMENT = {
    (-1, 0): 0,  # north
    (0, 1): 1,   # east
    (1, 0): 2,   # south
    (0, -1): 3,  # west
}


def action_id_from_cartesian_displacement(source: Sequence[int], target: Sequence[int]) -> int | None:
    """Return the environment action ID for a realised unit displacement.

    A self transition returns ``None`` because position alone cannot identify
    which action collided with a wall.
    """
    source_position = np.asarray(source, dtype=np.int64).reshape(-1)
    target_position = np.asarray(target, dtype=np.int64).reshape(-1)
    if source_position.shape != (2,) or target_position.shape != (2,):
        raise ValueError("Cartesian source and target positions must each contain two coordinates.")
    displacement = tuple((target_position - source_position).tolist())
    if displacement == (0, 0):
        return None
    try:
        return int(_ACTION_ID_BY_CARTESIAN_DISPLACEMENT[displacement])
    except KeyError as exc:
        raise ValueError(f"Expected a unit cardinal displacement, got {displacement!r}.") from exc


def validate_evaluation_walk_action_alignment(
    *,
    positions: np.ndarray,
    actions: np.ndarray,
) -> int:
    """Require exact post-step action agreement with every realised move."""
    position_array = np.asarray(positions)
    action_ids = action_ids_from_action_vectors(np.asarray(actions))
    if position_array.ndim != 2 or position_array.shape[1] != 2:
        raise ValueError(f"Expected positions with shape (N, 2), got {position_array.shape!r}.")
    if position_array.shape[0] != action_ids.shape[0]:
        raise ValueError("Evaluation positions and actions must have matching lengths.")

    checked = 0
    for timestep in range(max(0, position_array.shape[0] - 1)):
        expected = action_id_from_cartesian_displacement(
            position_array[timestep],
            position_array[timestep + 1],
        )
        if expected is None:
            continue
        actual = int(action_ids[timestep + 1])
        if actual != expected:
            raise ValueError(
                "Evaluation action alignment invariant failed at transition "
                f"{timestep}->{timestep + 1}: expected action {expected} from realised "
                f"displacement, found post-step action {actual}."
            )
        checked += 1
    if checked == 0:
        raise ValueError("Evaluation action alignment invariant checked no non-self transitions.")
    return checked


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
    learned = np.asarray(T_hat, dtype=np.float64)
    learned_support = support_action_transition_tensor(learned, epsilon=float(epsilon))
    n_learned = int(learned.shape[1])
    n_true_states = int(n_true)
    contingency = compute_contingency_matrix(
        decoded_states=decoded_series,
        true_states=true_series,
        n_learned=n_learned,
        n_true=n_true_states,
    )
    alignment = compute_alignment(contingency)
    occupancy = compute_occupancy(decoded_series, n_learned=n_learned)
    learned_distribution = normalize_action_rows(learned)
    aligned_distribution_tensor = aggregate_aligned_action_transition_tensor(
        learned_distribution,
        alignment,
        occupancy,
        n_true=n_true_states,
    )
    learned_support_distribution = normalize_action_rows(learned_support)
    aligned_support_tensor = aggregate_aligned_action_transition_tensor(
        learned_support_distribution,
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
            n_actions=int(learned.shape[0]),
        )
    )
    resolved_n_actions = max(
        int(true_tensor.shape[0]),
        int(aligned_distribution_tensor.shape[0]),
        int(aligned_support_tensor.shape[0]),
    )
    true_tensor = _pad_action_tensor_to_n_actions(true_tensor, resolved_n_actions)
    aligned_distribution_tensor = _pad_action_tensor_to_n_actions(aligned_distribution_tensor, resolved_n_actions)
    aligned_support_tensor = _pad_action_tensor_to_n_actions(aligned_support_tensor, resolved_n_actions)
    if ignore_self_loops:
        true_tensor = zero_action_tensor_diagonal(true_tensor)
        aligned_distribution_tensor = zero_action_tensor_diagonal(aligned_distribution_tensor)
        aligned_support_tensor = zero_action_tensor_diagonal(aligned_support_tensor)
    return {
        "mean_total_variation": compute_action_conditioned_mean_tv(
            true_tensor,
            aligned_distribution_tensor,
            epsilon=float(epsilon),
        ),
        **compute_action_conditioned_edge_metrics(
            true_tensor,
            aligned_support_tensor,
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
    reference = np.asarray(reference_uniques)
    values_arr = np.asarray(values)

    if reference.dtype != object and values_arr.dtype != object:
        try:
            if values_arr.ndim == 1:
                reference_1d = np.asarray(reference).reshape(-1)
                if reference_1d.size == 0 and values_arr.size > 0:
                    raise KeyError(None)
                if np.unique(reference_1d).shape[0] != reference_1d.shape[0]:
                    raise TypeError("duplicate reference values require dictionary mapping")
                values_1d = values_arr.astype(reference_1d.dtype, copy=False)
                sorter = np.argsort(reference_1d, kind="stable")
                sorted_reference = reference_1d[sorter]
                positions = np.searchsorted(sorted_reference, values_1d)
                valid = (
                    (positions < sorted_reference.shape[0])
                    & (sorted_reference[np.minimum(positions, sorted_reference.shape[0] - 1)] == values_1d)
                )
                if not np.all(valid):
                    missing = values_1d[~valid][0]
                    raise KeyError(int(missing) if np.issubdtype(values_1d.dtype, np.integer) else missing.item())
                return sorter[positions].astype(np.int64, copy=False)

            reference_rows = np.ascontiguousarray(reference.reshape(reference.shape[0], -1))
            value_rows = np.ascontiguousarray(values_arr.astype(reference_rows.dtype, copy=False).reshape(values_arr.shape[0], -1))
            row_dtype = np.dtype((np.void, reference_rows.dtype.itemsize * reference_rows.shape[1]))
            reference_keys = reference_rows.view(row_dtype).reshape(-1)
            value_keys = value_rows.view(row_dtype).reshape(-1)
            if reference_keys.size == 0 and value_keys.size > 0:
                raise KeyError(None)
            if np.unique(reference_keys).shape[0] != reference_keys.shape[0]:
                raise TypeError("duplicate reference rows require dictionary mapping")
            sorter = np.argsort(reference_keys, kind="stable")
            sorted_reference = reference_keys[sorter]
            positions = np.searchsorted(sorted_reference, value_keys)
            valid = (
                (positions < sorted_reference.shape[0])
                & (sorted_reference[np.minimum(positions, sorted_reference.shape[0] - 1)] == value_keys)
            )
            if not np.all(valid):
                missing = values_arr.reshape(values_arr.shape[0], -1)[~valid][0]
                raise KeyError(tuple(np.asarray(missing).tolist()))
            return sorter[positions].astype(np.int64, copy=False)
        except KeyError as exc:  # pragma: no cover
            raise ValueError(f"Encountered evaluation value not present in the training vocabulary: {exc.args[0]!r}") from exc
        except (TypeError, ValueError):
            pass

    mapping = build_exact_indexer(reference)
    if values_arr.ndim == 1:
        try:
            return np.asarray([mapping[int(value)] for value in values_arr.tolist()], dtype=np.int64)
        except KeyError as exc:  # pragma: no cover
            raise ValueError(f"Encountered evaluation value not present in the training vocabulary: {exc.args[0]!r}") from exc
    try:
        return np.asarray([mapping[tuple(np.asarray(row).tolist())] for row in values_arr.tolist()], dtype=np.int64)
    except KeyError as exc:  # pragma: no cover
        raise ValueError(f"Encountered evaluation row not present in the training vocabulary: {exc.args[0]!r}") from exc


def concatenate_walk_series(walk_values: Iterable[np.ndarray]) -> np.ndarray:
    series = [np.asarray(values) for values in walk_values]
    if not series:
        return np.asarray([], dtype=np.int64)
    return np.concatenate(series, axis=0)


@dataclass(frozen=True)
class TransitionStructureEvaluation:
    """All transition structure metrics derived from one consistent alignment.

    Collapsed and action-conditioned metrics share the same cooccurrence matrix,
    alignment, occupancy, and projected tensors.  The collapsed adjacency is
    derived by summing the projected action tensor, not by projecting an
    independently collapsed adjacency.
    """

    n_learned: int
    n_true: int
    n_actions: int
    cooccurrence: np.ndarray            # (n_learned, n_true)  int64
    alignment: np.ndarray               # (n_learned,)  int64
    occupancy: np.ndarray               # (n_learned,)  float64
    # Row-normalised distributions in ground-truth space, post-diagonal-mask.
    projected_action_transitions: np.ndarray     # (n_actions, n_true, n_true)
    projected_collapsed_adjacency: np.ndarray    # (n_true, n_true)
    ground_truth_action_transitions: np.ndarray  # (n_actions, n_true, n_true)
    ground_truth_collapsed_adjacency: np.ndarray # (n_true, n_true)
    valid_action_rows: np.ndarray       # (n_actions, n_true)  bool
    valid_collapsed_rows: np.ndarray    # (n_true,)  bool
    action_tv: float
    collapsed_tv: float
    action_edge_metrics_by_threshold: dict
    collapsed_edge_metrics_by_threshold: dict

    def primary_metrics(self, threshold: float = 1e-6) -> dict[str, float]:
        a = self.action_edge_metrics_by_threshold.get(float(threshold), {})
        c = self.collapsed_edge_metrics_by_threshold.get(float(threshold), {})
        return {
            "action_tv": self.action_tv,
            "collapsed_tv": self.collapsed_tv,
            "action_edge_precision": a.get("edge_precision", float("nan")),
            "action_edge_recall": a.get("edge_recall", float("nan")),
            "action_edge_f1": a.get("edge_f1", float("nan")),
            "collapsed_edge_precision": c.get("edge_precision", float("nan")),
            "collapsed_edge_recall": c.get("edge_recall", float("nan")),
            "collapsed_edge_f1": c.get("edge_f1", float("nan")),
        }


def project_and_evaluate_transition_structure(
    *,
    T_hat: np.ndarray,
    T_true: np.ndarray,
    cooccurrence: np.ndarray,
    ignore_self_loops: bool = True,
    edge_thresholds: Sequence[float] = (1e-6,),
    epsilon: float = 1e-6,
) -> TransitionStructureEvaluation:
    """Canonical single-alignment evaluation of learned transition structure.

    Pipeline:
    1. Derive alignment and occupancy from ``cooccurrence`` once.
    2. Project the full learned action tensor into GT space.
    3. Derive collapsed adjacency from the projected tensor (not independently).
    4. Remove self-loops after projection, never before.
    5. Normalise rows to probability distributions.
    6. Compute action-conditioned TV, collapsed TV, and edge metrics at every
       requested threshold from the same projected tensors.
    """
    T_hat_arr = np.asarray(T_hat, dtype=np.float64)
    T_true_arr = np.asarray(T_true, dtype=np.float64)
    contingency = np.asarray(cooccurrence, dtype=np.int64)

    if T_hat_arr.ndim != 3 or T_hat_arr.shape[1] != T_hat_arr.shape[2]:
        raise ValueError(f"Expected learned tensor with shape (a, n, n), got {T_hat_arr.shape!r}")
    if T_true_arr.ndim != 3 or T_true_arr.shape[1] != T_true_arr.shape[2]:
        raise ValueError(f"Expected ground-truth tensor with shape (a, n, n), got {T_true_arr.shape!r}")
    if contingency.ndim != 2:
        raise ValueError(f"Expected cooccurrence matrix with shape (n_learned, n_true), got {contingency.shape!r}")

    n_actions = max(int(T_hat_arr.shape[0]), int(T_true_arr.shape[0]))
    n_learned = int(T_hat_arr.shape[1])
    n_true = int(T_true_arr.shape[1])

    T_hat_arr = _pad_action_tensor_to_n_actions(T_hat_arr, n_actions)
    T_true_arr = _pad_action_tensor_to_n_actions(T_true_arr, n_actions)

    if contingency.shape[0] < n_learned or contingency.shape[1] < n_true:
        padded = np.zeros(
            (max(n_learned, contingency.shape[0]), max(n_true, contingency.shape[1])),
            dtype=np.int64,
        )
        padded[: contingency.shape[0], : contingency.shape[1]] = contingency
        contingency = padded
    contingency = contingency[:n_learned, :n_true]

    # Step 1: alignment and occupancy derived once, shared by all metrics.
    alignment = compute_alignment(contingency)
    occupancy = contingency.sum(axis=1).astype(np.float64, copy=False)

    # Step 2: project learned tensor into GT space (normalise rows, then aggregate).
    T_hat_GT = aggregate_aligned_action_transition_tensor(
        normalize_action_rows(T_hat_arr), alignment, occupancy, n_true=n_true
    )

    # Step 3: collapsed adjacency derived from projected action tensor.
    W_hat_GT = collapse_action_transition_tensor_to_adjacency(T_hat_GT)
    W_true = collapse_action_transition_tensor_to_adjacency(T_true_arr)

    # Step 4: self-loops removed after projection.
    if ignore_self_loops:
        T_hat_GT = zero_action_tensor_diagonal(T_hat_GT)
        T_true_arr = zero_action_tensor_diagonal(T_true_arr)
        W_hat_GT = zero_diagonal(W_hat_GT)
        W_true = zero_diagonal(W_true)

    # Step 5: normalise to probability distributions.
    P_hat_action = normalize_action_rows(T_hat_GT)
    P_true_action = normalize_action_rows(T_true_arr)
    P_hat_collapsed = row_normalize(W_hat_GT)
    P_true_collapsed = row_normalize(W_true)

    valid_action_rows = T_true_arr.sum(axis=2) > float(epsilon)
    valid_collapsed_rows = W_true.sum(axis=1) > float(epsilon)

    # Step 6: compute metrics from shared normalised projections.
    action_tv = _tv_over_valid_pairs(P_true_action, P_hat_action, valid_action_rows, epsilon=float(epsilon))
    collapsed_tv = _tv_over_valid_rows(P_true_collapsed, P_hat_collapsed, valid_collapsed_rows, epsilon=float(epsilon))

    action_edge_by_threshold: dict[float, dict[str, float]] = {}
    collapsed_edge_by_threshold: dict[float, dict[str, float]] = {}
    for threshold in edge_thresholds:
        t = float(threshold)
        action_edge_by_threshold[t] = compute_action_conditioned_edge_metrics(P_true_action, P_hat_action, epsilon=t)
        collapsed_edge_by_threshold[t] = compute_edge_metrics(P_true_collapsed, P_hat_collapsed, epsilon=t)

    return TransitionStructureEvaluation(
        n_learned=n_learned,
        n_true=n_true,
        n_actions=n_actions,
        cooccurrence=contingency,
        alignment=alignment,
        occupancy=occupancy,
        projected_action_transitions=P_hat_action,
        projected_collapsed_adjacency=P_hat_collapsed,
        ground_truth_action_transitions=P_true_action,
        ground_truth_collapsed_adjacency=P_true_collapsed,
        valid_action_rows=valid_action_rows,
        valid_collapsed_rows=valid_collapsed_rows,
        action_tv=action_tv,
        collapsed_tv=collapsed_tv,
        action_edge_metrics_by_threshold=action_edge_by_threshold,
        collapsed_edge_metrics_by_threshold=collapsed_edge_by_threshold,
    )


def _tv_over_valid_pairs(
    P_true: np.ndarray,
    P_hat: np.ndarray,
    valid_pairs: np.ndarray,
    *,
    epsilon: float,
) -> float:
    tv_values: list[float] = []
    for action_idx, state_idx in np.argwhere(valid_pairs).tolist():
        row_hat = P_hat[int(action_idx), int(state_idx)]
        if float(row_hat.sum()) <= float(epsilon):
            tv_values.append(1.0)
        else:
            tv_values.append(float(0.5 * np.abs(row_hat - P_true[int(action_idx), int(state_idx)]).sum()))
    return float(np.mean(tv_values)) if tv_values else 0.0


def _tv_over_valid_rows(
    P_true: np.ndarray,
    P_hat: np.ndarray,
    valid_rows: np.ndarray,
    *,
    epsilon: float,
) -> float:
    tv_values: list[float] = []
    for state_idx in np.flatnonzero(valid_rows).tolist():
        row_hat = P_hat[int(state_idx)]
        if float(row_hat.sum()) <= float(epsilon):
            tv_values.append(1.0)
        else:
            tv_values.append(float(0.5 * np.abs(row_hat - P_true[int(state_idx)]).sum()))
    return float(np.mean(tv_values)) if tv_values else 0.0


# ---------------------------------------------------------------------------
# Canonical transition structure evaluation (First-Class Snapshot Architecture)
# ---------------------------------------------------------------------------


def compute_hard_alignment(C: np.ndarray) -> np.ndarray:
    """Hard alignment returning -1 for unobserved learned states.

    A(k) = argmax_i C[k, i]  when sum_i C[k, i] > 0
    A(k) = -1                 when sum_i C[k, i] = 0

    Prevents zero-sum rows from being mapped to true state 0 via argmax.
    """
    contingency = np.asarray(C, dtype=np.int64)
    alignment = np.full(contingency.shape[0], -1, dtype=np.int64)
    if contingency.shape[1] == 0:
        return alignment
    row_sums = contingency.sum(axis=1)
    observed = row_sums > 0
    if np.any(observed):
        alignment[observed] = np.argmax(contingency[observed], axis=1)
    return alignment


def compute_transition_statistics(
    *,
    decoded_walks: Iterable[np.ndarray],
    true_walks: Iterable[np.ndarray],
    action_walks: Iterable[np.ndarray],
    n_learned: int,
    n_true: int,
    n_actions: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute C, H, N_ai from evaluation walks.

    Evaluation walks store action[t] as the post-step action (the action that
    led to state[t]).  The source-state transition (state[t] -> state[t+1]) is
    therefore conditioned on action[t+1].

    Returns:
        C:    (n_learned, n_true) int64  -- total state cooccurrence C[k, i]
        H:    (n_actions, n_learned, n_true) int64  -- action-source cooccurrence H[a, k, i]
        N_ai: (n_actions, n_true) int64  -- true state-action counts
    """
    C = np.zeros((n_learned, n_true), dtype=np.int64)
    H = np.zeros((n_actions, n_learned, n_true), dtype=np.int64)
    N_ai = np.zeros((n_actions, n_true), dtype=np.int64)

    for decoded_walk, true_walk, action_walk in zip(decoded_walks, true_walks, action_walks, strict=False):
        decoded = np.asarray(decoded_walk, dtype=np.int64)
        truth = np.asarray(true_walk, dtype=np.int64)
        actions = np.asarray(action_walk, dtype=np.int64)
        length = min(decoded.shape[0], truth.shape[0])

        # Total cooccurrence over all steps.
        decoded_steps = decoded[:length]
        truth_steps = truth[:length]
        valid_steps = (
            (0 <= decoded_steps) & (decoded_steps < n_learned)
            & (0 <= truth_steps) & (truth_steps < n_true)
        )
        np.add.at(C, (decoded_steps[valid_steps], truth_steps[valid_steps]), 1)

        # Transitions: source step t, target step t+1, action actions[t+1].
        transition_length = min(length - 1, actions.shape[0] - 1)
        if transition_length < 1:
            continue
        # action that caused the move from state[t] to state[t+1]
        action_idx = actions[1:transition_length + 1]
        decoded_src = decoded[:transition_length]
        truth_src = truth[:transition_length]
        valid_transitions = (
            (0 <= decoded_src) & (decoded_src < n_learned)
            & (0 <= truth_src) & (truth_src < n_true)
            & (0 <= action_idx) & (action_idx < n_actions)
        )
        np.add.at(
            H,
            (action_idx[valid_transitions], decoded_src[valid_transitions], truth_src[valid_transitions]),
            1,
        )
        np.add.at(N_ai, (action_idx[valid_transitions], truth_src[valid_transitions]), 1)

    return C, H, N_ai


def compute_action_projection_weights(
    H: np.ndarray,
    alignment: np.ndarray,
    n_true: int,
) -> np.ndarray:
    """Action-specific clone weights ω[a, k, i] = H[a,k,i] / sum_{k' in K_i} H[a,k',i].

    Returns (n_actions, n_learned, n_true).  Zero when denominator is zero.
    """
    H_f = np.asarray(H, dtype=np.float64)
    n_actions, n_learned, _ = H_f.shape
    omega = np.zeros((n_actions, n_learned, n_true), dtype=np.float64)
    for i in range(n_true):
        K_i = np.where(alignment == i)[0]
        if K_i.size == 0:
            continue
        denom = H_f[:, K_i, i].sum(axis=1)  # (n_actions,)
        pos = denom > 0
        if not np.any(pos):
            continue
        for a in np.where(pos)[0].tolist():
            omega[a, K_i, i] = H_f[a, K_i, i] / denom[a]
    return omega


def _build_target_mapping_matrix(alignment: np.ndarray, n_true: int) -> np.ndarray:
    """target_mat[l, j] = 1 if alignment[l] == j; target_mat[l, n_true] = 1 if unaligned."""
    n_learned = alignment.shape[0]
    target_mat = np.zeros((n_learned, n_true + 1), dtype=np.float64)
    for l in range(n_learned):
        j = int(alignment[l])
        if j < 0:
            target_mat[l, n_true] = 1.0
        elif j < n_true:
            target_mat[l, j] = 1.0
    return target_mat


def project_transition_tensor_canonical(
    *,
    T_hat_norm: np.ndarray,
    alignment: np.ndarray,
    omega: np.ndarray,
    n_true: int,
) -> np.ndarray:
    """Canonical projection into GT space with unmatched (⊥) target column.

    P_proj[a, i, j] for j < n_true: matched target mass.
    P_proj[a, i, n_true]: unmatched (⊥) target mass.

    Returns shape (n_actions, n_true, n_true+1).
    """
    target_mat = _build_target_mapping_matrix(alignment, n_true)
    # P_by_learned[a, k, j] = sum_{l: A(l)=j} T_hat_norm[a, k, l]
    P_by_learned = np.einsum("akl,lj->akj", T_hat_norm, target_mat)
    # P_proj[a, i, j] = sum_k omega[a, k, i] * P_by_learned[a, k, j]
    return np.einsum("aki,akj->aij", omega, P_by_learned)


def complete_and_normalize_projected_rows(
    P_proj: np.ndarray,
    valid_rows: np.ndarray,
) -> np.ndarray:
    """Row-complete and normalise the projected distribution.

    For each valid (a, i):
      - If the row has positive mass: normalise over S^GT ∪ {⊥}.
      - If the row has no mass: assign all to ⊥ (TV = 1 vs any GT row).
    Invalid rows are zeroed.

    Returns shape identical to P_proj.
    """
    P_hat = np.zeros_like(P_proj)
    row_sums = P_proj.sum(axis=2)  # (n_actions, n_true)
    safe_sums = np.where(row_sums > 0, row_sums, 1.0)
    normalized = P_proj / safe_sums[:, :, np.newaxis]

    has_mass = valid_rows & (row_sums > 0)
    no_mass = valid_rows & ~has_mass

    P_hat[has_mass] = normalized[has_mass]
    for a, i in np.argwhere(no_mass).tolist():
        P_hat[a, i, -1] = 1.0  # all mass to ⊥

    return P_hat


def _action_tv_by_pair(
    P_hat: np.ndarray,
    P_true: np.ndarray,
    valid_rows: np.ndarray,
) -> np.ndarray:
    """0.5 * |P_hat[a,i,:] - P_true[a,i,:]|_1 for valid (a, i) pairs."""
    tv = np.zeros(valid_rows.shape, dtype=np.float64)
    delta = np.abs(P_hat - P_true)  # (n_actions, n_true, n_true+1)
    tv_raw = 0.5 * delta.sum(axis=2)  # (n_actions, n_true)
    tv[valid_rows] = tv_raw[valid_rows]
    return tv


def compute_occupancy_weighted_action_tv(
    tv_by_pair: np.ndarray,
    valid_rows: np.ndarray,
    N_ai: np.ndarray,
) -> tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
    """Compute occupancy-weighted and macro action-conditioned TV.

    Returns:
        tv_occ: global occupancy-weighted TV (primary metric)
        tv_macro: global macro-average TV
        occ_weights: (n_actions, n_true) evaluation occupancy weights
        tv_by_state_occ: (n_true,) per-state occupancy-weighted TV
        tv_by_state_macro: (n_true,) per-state macro TV
    """
    n_actions, n_true = valid_rows.shape
    N_valid = np.where(valid_rows, np.asarray(N_ai, dtype=np.float64), 0.0)
    total = float(N_valid.sum())
    if total > 0:
        occ_weights = N_valid / total
    else:
        n_valid = int(np.count_nonzero(valid_rows))
        occ_weights = np.where(valid_rows, 1.0 / n_valid if n_valid > 0 else 0.0, 0.0)

    tv_occ = float((tv_by_pair * occ_weights).sum())
    n_valid_pairs = int(np.count_nonzero(valid_rows))
    tv_macro = float(tv_by_pair[valid_rows].mean()) if n_valid_pairs > 0 else 0.0

    tv_by_state_occ = np.zeros(n_true, dtype=np.float64)
    tv_by_state_macro = np.zeros(n_true, dtype=np.float64)
    for i in range(n_true):
        valid_a = valid_rows[:, i]
        if not np.any(valid_a):
            continue
        N_i = float(N_ai[valid_a, i].sum())
        if N_i > 0:
            w_i = N_ai[valid_a, i].astype(np.float64) / N_i
            tv_by_state_occ[i] = float((tv_by_pair[valid_a, i] * w_i).sum())
        tv_by_state_macro[i] = float(tv_by_pair[valid_a, i].mean())

    return tv_occ, tv_macro, occ_weights, tv_by_state_occ, tv_by_state_macro


def condition_non_selfloop_action_distributions(
    P_hat: np.ndarray,
    P_true: np.ndarray,
    valid_rows: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Condition action rows on making a non-self transition.

    V_move = {(a, i): sum_{j != i} P_GT(j | i, a) > 0}.
    Rows outside V_move are zero.  Predicted rows with no remaining mass are
    assigned to the explicit sink so that a missing learned row has TV = 1.

    Returns (P_hat_nonself, P_true_nonself, valid_move_rows).
    """
    P_hat_arr = np.asarray(P_hat, dtype=np.float64)
    P_true_arr = np.asarray(P_true, dtype=np.float64)
    valid_arr = np.asarray(valid_rows, dtype=bool)
    if P_hat_arr.shape != P_true_arr.shape or P_hat_arr.ndim != 3:
        raise ValueError("P_hat and P_true must be matching (action, source, target) tensors.")
    n_actions, n_true, n_ext = P_hat_arr.shape
    if valid_arr.shape != (n_actions, n_true):
        raise ValueError("valid_rows must match the action/source dimensions.")
    if n_ext != n_true + 1:
        raise ValueError("Conditional no-self rows require an explicit sink target column.")

    P_hat_nonself = np.zeros_like(P_hat_arr)
    P_true_nonself = np.zeros_like(P_true_arr)
    valid_move_rows = np.zeros_like(valid_arr)

    for a, i in np.argwhere(valid_arr).tolist():
        gt_row = P_true_arr[a, i].copy()
        gt_row[i] = 0.0
        gt_total = float(gt_row[:n_true].sum())
        if gt_total <= 0.0:
            continue
        valid_move_rows[a, i] = True
        P_true_nonself[a, i] = gt_row / gt_total

        hat_row = P_hat_arr[a, i].copy()
        hat_row[i] = 0.0
        hat_total = float(hat_row.sum())
        if hat_total <= 0.0:
            P_hat_nonself[a, i, -1] = 1.0
        else:
            P_hat_nonself[a, i] = hat_row / hat_total

    return P_hat_nonself, P_true_nonself, valid_move_rows


def compute_non_selfloop_action_tv(
    P_hat: np.ndarray,
    P_true: np.ndarray,
    valid_rows: np.ndarray,
) -> np.ndarray:
    """Conditional non-self-loop TV diagnostic.

    Rows outside V_move are returned as zero and must be excluded from metric
    aggregation using the mask from
    :func:`condition_non_selfloop_action_distributions`.
    """
    P_hat_nonself, P_true_nonself, valid_move_rows = condition_non_selfloop_action_distributions(
        P_hat, P_true, valid_rows
    )
    return _action_tv_by_pair(P_hat_nonself, P_true_nonself, valid_move_rows)


def compute_action_edge_metrics_canonical(
    P_hat: np.ndarray,
    P_true: np.ndarray,
    valid_rows: np.ndarray,
    thresholds: Sequence[float],
) -> dict:
    """Action-probability thresholded edge metrics.

    GT edge set: (a, i, j) with j < n_true and P_GT(j|i,a) > threshold.
    Predicted edge set: (a, i, j) with P_hat(j|i,a) > threshold (may include ⊥).
    ⊥ edges count as false positives only.
    """
    n_true = P_true.shape[1]
    results: dict[float, dict] = {}
    for eps in thresholds:
        t = float(eps)
        gt_edges: set[tuple] = set()
        hat_edges: set[tuple] = set()
        for a, i in np.argwhere(valid_rows).tolist():
            for j in range(P_true.shape[2]):
                if j < n_true and float(P_true[a, i, j]) > t:
                    gt_edges.add((a, i, j))
                if float(P_hat[a, i, j]) > t:
                    hat_edges.add((a, i, j))
        tp = len(gt_edges & hat_edges)
        prec = tp / len(hat_edges) if hat_edges else 0.0
        rec = tp / len(gt_edges) if gt_edges else 0.0
        f1 = (2.0 * prec * rec / (prec + rec)) if (prec + rec) > 0.0 else 0.0
        results[t] = {
            "action_probability_edge_precision": prec,
            "action_probability_edge_recall": rec,
            "action_probability_edge_f1": f1,
        }
    return results


def compute_collapsed_edge_metrics_canonical(
    P_hat_coll: np.ndarray,
    P_true_coll: np.ndarray,
    valid_rows: np.ndarray,
    thresholds: Sequence[float],
) -> dict:
    """Collapsed probability-thresholded edge metrics."""
    n_true = P_true_coll.shape[0]
    results: dict[float, dict] = {}
    for eps in thresholds:
        t = float(eps)
        gt_edges: set[tuple] = set()
        hat_edges: set[tuple] = set()
        for i in np.flatnonzero(valid_rows).tolist():
            for j in range(P_true_coll.shape[1]):
                if j < n_true and float(P_true_coll[i, j]) > t:
                    gt_edges.add((i, j))
                if float(P_hat_coll[i, j]) > t:
                    hat_edges.add((i, j))
        tp = len(gt_edges & hat_edges)
        prec = tp / len(hat_edges) if hat_edges else 0.0
        rec = tp / len(gt_edges) if gt_edges else 0.0
        f1 = (2.0 * prec * rec / (prec + rec)) if (prec + rec) > 0.0 else 0.0
        results[t] = {
            "action_probability_edge_precision": prec,
            "action_probability_edge_recall": rec,
            "action_probability_edge_f1": f1,
        }
    return results


def compute_collapsed_distributions_canonical(
    P_hat: np.ndarray,
    P_true: np.ndarray,
    valid_rows: np.ndarray,
    N_ai: np.ndarray,
    thresholds: Sequence[float] = (1e-6,),
) -> dict:
    """Compute evaluation-policy and uniform-valid-action collapsed TV and edge metrics.

    Both GT and predicted are marginalised from the same action-conditioned tensors
    using the same explicit action prior q(a | i).
    """
    n_actions, n_true, n_ext = P_hat.shape

    # q_eval(a|i) = N_ai[a,i] / N_i  for (a,i) in V
    # q_uniform(a|i) = 1 / |A_i|     for a in A_i
    q_eval = np.zeros((n_actions, n_true), dtype=np.float64)
    q_unif = np.zeros((n_actions, n_true), dtype=np.float64)
    for i in range(n_true):
        valid_a = valid_rows[:, i]
        if not np.any(valid_a):
            continue
        N_i = float(N_ai[valid_a, i].sum())
        n_va = int(valid_a.sum())
        if N_i > 0:
            q_eval[valid_a, i] = N_ai[valid_a, i].astype(np.float64) / N_i
        q_unif[valid_a, i] = 1.0 / n_va

    # Collapsed: einsum('ai,aij->ij', q, P)
    P_hat_eval = np.einsum("ai,aij->ij", q_eval, P_hat)
    P_true_eval = np.einsum("ai,aij->ij", q_eval, P_true)
    P_hat_unif = np.einsum("ai,aij->ij", q_unif, P_hat)
    P_true_unif = np.einsum("ai,aij->ij", q_unif, P_true)

    # Valid collapsed rows: any valid action for that state
    valid_collapsed = valid_rows.any(axis=0)  # (n_true,)
    N_i_arr = N_ai.sum(axis=0).astype(np.float64)  # (n_true,)
    total_N = float(N_i_arr[valid_collapsed].sum())
    if total_N > 0:
        d_state = np.where(valid_collapsed, N_i_arr / total_N, 0.0)
    else:
        n_vc = int(valid_collapsed.sum())
        d_state = np.where(valid_collapsed, 1.0 / n_vc if n_vc > 0 else 0.0, 0.0)

    def _tv_stats(P_hat_c: np.ndarray, P_true_c: np.ndarray) -> dict:
        tv_by_state = np.zeros(n_true, dtype=np.float64)
        for i in np.flatnonzero(valid_collapsed).tolist():
            tv_by_state[i] = 0.5 * float(np.abs(P_hat_c[i] - P_true_c[i]).sum())
        tv_occ = float((tv_by_state * d_state).sum())
        n_vc = int(valid_collapsed.sum())
        tv_macro = float(tv_by_state[valid_collapsed].mean()) if n_vc > 0 else 0.0
        return {"tv_by_state": tv_by_state, "tv_occ": tv_occ, "tv_macro": tv_macro}

    def _non_self_loop_tv_stats(P_hat_c: np.ndarray, P_true_c: np.ndarray) -> dict:
        hat_ns, true_ns, valid_move_2d = condition_non_selfloop_action_distributions(
            P_hat_c[np.newaxis, :, :],
            P_true_c[np.newaxis, :, :],
            valid_collapsed[np.newaxis, :],
        )
        valid_move = valid_move_2d[0]
        tv_by_state = _action_tv_by_pair(hat_ns, true_ns, valid_move_2d)[0]
        move_count = int(valid_move.sum())
        move_total = float(N_i_arr[valid_move].sum())
        if move_total > 0.0:
            move_weights = np.where(valid_move, N_i_arr / move_total, 0.0)
        else:
            move_weights = np.where(valid_move, 1.0 / move_count if move_count else 0.0, 0.0)
        tv_occ = float((tv_by_state * move_weights).sum())
        tv_macro = float(tv_by_state[valid_move].mean()) if move_count else 0.0
        return {"tv_non_self_loop_by_state": tv_by_state, "tv_non_self_loop_occ": tv_occ, "tv_non_self_loop_macro": tv_macro}

    eval_tv = _tv_stats(P_hat_eval, P_true_eval)
    unif_tv = _tv_stats(P_hat_unif, P_true_unif)
    eval_nonsl_tv = _non_self_loop_tv_stats(P_hat_eval, P_true_eval)
    unif_nonsl_tv = _non_self_loop_tv_stats(P_hat_unif, P_true_unif)

    eval_edge = compute_collapsed_edge_metrics_canonical(P_hat_eval, P_true_eval, valid_collapsed, thresholds)
    unif_edge = compute_collapsed_edge_metrics_canonical(P_hat_unif, P_true_unif, valid_collapsed, thresholds)

    return {
        "q_eval": q_eval,
        "q_uniform": q_unif,
        "valid_collapsed_rows": valid_collapsed,
        "state_occupancy_weights": d_state,
        "evaluation_policy": {
            "P_hat": P_hat_eval,
            "P_true": P_true_eval,
            **eval_tv,
            **eval_nonsl_tv,
            "edge_metrics_by_threshold": eval_edge,
        },
        "uniform_valid_actions": {
            "P_hat": P_hat_unif,
            "P_true": P_true_unif,
            **unif_tv,
            **unif_nonsl_tv,
            "edge_metrics_by_threshold": unif_edge,
        },
    }


@dataclass(frozen=True)
class CanonicalTransitionEvaluationResult:
    """Full canonical transition structure evaluation (First-Class Snapshot Architecture).

    All action-conditioned and collapsed metrics share one cooccurrence matrix,
    one alignment, and one set of projected distributions.

    The primary metric is TV_action_occ: expected local transition error under the
    held-out evaluation process, with evaluation-occupancy weighting and self-loops
    included in the comparison target space (S^GT ∪ {⊥}).
    """

    n_learned: int
    n_true: int
    n_actions: int

    # --- Alignment ---
    cooccurrence: np.ndarray                   # (n_learned, n_true) int64
    action_source_cooccurrence: np.ndarray     # (n_actions, n_learned, n_true) int64
    alignment: np.ndarray                      # (n_learned,) int64, -1 = unaligned
    aligned_learned_mask: np.ndarray           # (n_learned,) bool
    unaligned_learned_states: list             # list[int]
    source_action_counts: np.ndarray           # (n_actions, n_true) int64 — N_ai

    # --- Projected distributions (extended space: S^GT ∪ {⊥}) ---
    P_true_full: np.ndarray                    # (n_actions, n_true, n_true+1)
    P_hat_full: np.ndarray                     # (n_actions, n_true, n_true+1)
    valid_action_rows: np.ndarray              # (n_actions, n_true) bool
    valid_non_self_loop_rows: np.ndarray       # (n_actions, n_true) bool -- V_move

    # --- Unmatched target mass ---
    unmatched_target_mass: np.ndarray          # (n_actions, n_true) float64
    total_unmatched_target_mass: float

    # --- Action-conditioned TV variants ---
    tv_by_pair_full: np.ndarray                # (n_actions, n_true)
    tv_by_pair_non_self_loop: np.ndarray       # (n_actions, n_true)
    evaluation_occupancy_weights: np.ndarray   # (n_actions, n_true)
    tv_by_state_occupancy: np.ndarray          # (n_true,)
    tv_by_state_macro: np.ndarray              # (n_true,)
    tv_occupancy: float                        # includes self-loops
    tv_macro: float
    tv_non_self_loop_occupancy: float
    tv_non_self_loop_macro: float
    tv_non_self_loop_by_state_occupancy: np.ndarray
    tv_non_self_loop_by_state_macro: np.ndarray
    action_edge_metrics_by_threshold: dict

    # --- Collapsed metrics ---
    action_weights_evaluation_policy: np.ndarray   # (n_actions, n_true) q_eval
    action_weights_uniform_valid: np.ndarray       # (n_actions, n_true) q_uniform
    valid_collapsed_rows: np.ndarray               # (n_true,) bool
    state_occupancy_weights: np.ndarray            # (n_true,)

    collapsed_eval_P_true: np.ndarray              # (n_true, n_true+1)
    collapsed_eval_P_hat: np.ndarray               # (n_true, n_true+1)
    collapsed_eval_tv_by_state: np.ndarray         # (n_true,)
    collapsed_eval_tv_occupancy: float
    collapsed_eval_tv_macro: float
    collapsed_eval_tv_non_self_loop_by_state: np.ndarray
    collapsed_eval_tv_non_self_loop_occupancy: float
    collapsed_eval_tv_non_self_loop_macro: float
    collapsed_eval_edge_by_threshold: dict

    collapsed_unif_P_true: np.ndarray              # (n_true, n_true+1)
    collapsed_unif_P_hat: np.ndarray               # (n_true, n_true+1)
    collapsed_unif_tv_by_state: np.ndarray         # (n_true,)
    collapsed_unif_tv_occupancy: float
    collapsed_unif_tv_macro: float
    collapsed_unif_tv_non_self_loop_by_state: np.ndarray
    collapsed_unif_tv_non_self_loop_occupancy: float
    collapsed_unif_tv_non_self_loop_macro: float
    collapsed_unif_edge_by_threshold: dict

    def primary_metrics(self, threshold: float = 1e-6) -> dict[str, float]:
        a_edge = self.action_edge_metrics_by_threshold.get(float(threshold), {})
        c_edge = self.collapsed_eval_edge_by_threshold.get(float(threshold), {})
        return {
            "action_tv_occupancy_weighted": self.tv_occupancy,
            "action_tv_macro": self.tv_macro,
            "action_non_self_loop_tv_occupancy_weighted": self.tv_non_self_loop_occupancy,
            "action_non_self_loop_tv_macro": self.tv_non_self_loop_macro,
            "collapsed_evaluation_policy_tv_occupancy": self.collapsed_eval_tv_occupancy,
            "collapsed_evaluation_policy_tv_macro": self.collapsed_eval_tv_macro,
            "collapsed_evaluation_policy_non_self_loop_tv_occupancy": self.collapsed_eval_tv_non_self_loop_occupancy,
            "collapsed_evaluation_policy_non_self_loop_tv_macro": self.collapsed_eval_tv_non_self_loop_macro,
            "collapsed_uniform_action_tv_occupancy": self.collapsed_unif_tv_occupancy,
            "action_probability_edge_precision": a_edge.get("action_probability_edge_precision", float("nan")),
            "action_probability_edge_recall": a_edge.get("action_probability_edge_recall", float("nan")),
            "action_probability_edge_f1": a_edge.get("action_probability_edge_f1", float("nan")),
            "collapsed_eval_edge_precision": c_edge.get("action_probability_edge_precision", float("nan")),
            "collapsed_eval_edge_recall": c_edge.get("action_probability_edge_recall", float("nan")),
            "collapsed_eval_edge_f1": c_edge.get("action_probability_edge_f1", float("nan")),
        }


def evaluate_transition_structure_canonical(
    *,
    T_hat: np.ndarray,
    T_true: np.ndarray,
    cooccurrence: np.ndarray,
    action_source_cooccurrence: np.ndarray,
    source_action_counts: np.ndarray,
    edge_thresholds: Sequence[float] = (1e-6,),
) -> CanonicalTransitionEvaluationResult:
    """Canonical single-alignment evaluation of learned transition structure.

    Implements the First-Class Snapshot Evaluation Architecture:

    1.  Hard alignment: A(k) = argmax_i C[k,i]; -1 if no evaluation evidence.
    2.  Action-specific projection weights ω[a,k,i] from H[a,k,i].
    3.  Canonical projection into S^GT ∪ {⊥}: targets with A(l)=-1 become ⊥.
    4.  Row completion: empty projected rows are assigned entirely to ⊥ (TV=1).
    5.  Primary TV includes self-loops; non-self-loop TV is a separate diagnostic.
    6.  Occupancy weights derived from N_ai (true state-action counts), never from
        decoded model output.
    7.  Collapsed metrics use evaluation-policy or uniform action prior explicitly;
        truth and prediction marginalise from the same action-conditioned tensors.

    Args:
        T_hat: (n_actions, n_learned, n_learned) — raw learned transition counts/probs.
        T_true: (n_actions, n_true, n_true) — ground-truth transition counts/probs.
        cooccurrence: (n_learned, n_true) int64 — C[k,i] from evaluation walks.
        action_source_cooccurrence: (n_actions, n_learned, n_true) int64 — H[a,k,i].
        source_action_counts: (n_actions, n_true) int64 — N_ai from evaluation walks.
        edge_thresholds: probability thresholds for edge metrics.

    Returns:
        CanonicalTransitionEvaluationResult with all metrics and distributions.
    """
    T_hat_arr = np.asarray(T_hat, dtype=np.float64)
    T_true_arr = np.asarray(T_true, dtype=np.float64)
    C = np.asarray(cooccurrence, dtype=np.int64)
    H = np.asarray(action_source_cooccurrence, dtype=np.int64)
    N_ai = np.asarray(source_action_counts, dtype=np.int64)

    if T_hat_arr.ndim != 3 or T_hat_arr.shape[1] != T_hat_arr.shape[2]:
        raise ValueError(f"Expected T_hat with shape (a, n, n), got {T_hat_arr.shape!r}")
    if T_true_arr.ndim != 3 or T_true_arr.shape[1] != T_true_arr.shape[2]:
        raise ValueError(f"Expected T_true with shape (a, n, n), got {T_true_arr.shape!r}")
    if C.ndim != 2:
        raise ValueError(f"Expected cooccurrence with shape (n_learned, n_true), got {C.shape!r}")
    if H.ndim != 3:
        raise ValueError(f"Expected action_source_cooccurrence with shape (a, n_learned, n_true), got {H.shape!r}")

    n_actions = max(int(T_hat_arr.shape[0]), int(T_true_arr.shape[0]))
    n_learned = int(T_hat_arr.shape[1])
    n_true = int(T_true_arr.shape[1])

    T_hat_arr = _pad_action_tensor_to_n_actions(T_hat_arr, n_actions)
    T_true_arr = _pad_action_tensor_to_n_actions(T_true_arr, n_actions)
    H = _pad_action_tensor_to_n_actions(H.astype(np.float64), n_actions).astype(np.int64)
    N_ai_padded = np.zeros((n_actions, n_true), dtype=np.int64)
    N_ai_padded[:N_ai.shape[0], :N_ai.shape[1]] = N_ai[:n_actions, :n_true]
    N_ai = N_ai_padded

    # Pad cooccurrence to (n_learned, n_true).
    if C.shape[0] < n_learned or C.shape[1] < n_true:
        padded = np.zeros((max(n_learned, C.shape[0]), max(n_true, C.shape[1])), dtype=np.int64)
        padded[:C.shape[0], :C.shape[1]] = C
        C = padded
    C = C[:n_learned, :n_true]

    # --- Step 1: Hard alignment ---
    alignment = compute_hard_alignment(C)
    aligned_mask = alignment >= 0
    unaligned = [int(k) for k in np.where(~aligned_mask)[0]]

    # --- Step 2: Action-specific projection weights ---
    omega = compute_action_projection_weights(H, alignment, n_true)

    # --- Step 3: Normalise learned transitions ---
    T_hat_norm = normalize_action_rows(T_hat_arr)

    # --- Step 4: Canonical projection ---
    P_proj = project_transition_tensor_canonical(
        T_hat_norm=T_hat_norm,
        alignment=alignment,
        omega=omega,
        n_true=n_true,
    )

    # --- Step 5: GT extended distribution (⊥ col = 0) ---
    T_true_norm = normalize_action_rows(T_true_arr)
    P_true_ext = np.zeros((n_actions, n_true, n_true + 1), dtype=np.float64)
    P_true_ext[:, :, :n_true] = T_true_norm

    # --- Step 6: Valid rows ---
    valid_rows = T_true_arr.sum(axis=2) > 0  # (n_actions, n_true)

    # --- Step 7: Complete and normalise projected rows ---
    P_hat_full = complete_and_normalize_projected_rows(P_proj, valid_rows)

    # --- Step 8: Unmatched target mass ---
    unmatched_mass = np.where(valid_rows, P_hat_full[:, :, n_true], 0.0)
    n_valid = int(np.count_nonzero(valid_rows))
    total_unmatched = float(unmatched_mass[valid_rows].sum()) / max(1.0, float(n_valid))

    # --- Step 9: Action-conditioned TV (primary: full distribution, with self-loops) ---
    tv_full = _action_tv_by_pair(P_hat_full, P_true_ext, valid_rows)
    tv_occ, tv_macro, occ_weights, tv_state_occ, tv_state_macro = compute_occupancy_weighted_action_tv(
        tv_full, valid_rows, N_ai
    )

    # --- Step 10: Non-self-loop diagnostic ---
    P_hat_nonsl, P_true_nonsl, valid_move_rows = condition_non_selfloop_action_distributions(
        P_hat_full, P_true_ext, valid_rows
    )
    tv_nonsl = _action_tv_by_pair(P_hat_nonsl, P_true_nonsl, valid_move_rows)
    tv_nonsl_occ, tv_nonsl_macro, _, tv_nonsl_state_occ, tv_nonsl_state_macro = compute_occupancy_weighted_action_tv(
        tv_nonsl, valid_move_rows, N_ai
    )

    # --- Step 11: Action edge metrics ---
    action_edge = compute_action_edge_metrics_canonical(P_hat_full, P_true_ext, valid_rows, edge_thresholds)

    # --- Step 12: Collapsed metrics ---
    coll = compute_collapsed_distributions_canonical(
        P_hat_full, P_true_ext, valid_rows, N_ai, thresholds=edge_thresholds
    )

    return CanonicalTransitionEvaluationResult(
        n_learned=n_learned,
        n_true=n_true,
        n_actions=n_actions,
        cooccurrence=C,
        action_source_cooccurrence=H,
        alignment=alignment,
        aligned_learned_mask=aligned_mask,
        unaligned_learned_states=unaligned,
        source_action_counts=N_ai,
        P_true_full=P_true_ext,
        P_hat_full=P_hat_full,
        valid_action_rows=valid_rows,
        valid_non_self_loop_rows=valid_move_rows,
        unmatched_target_mass=unmatched_mass,
        total_unmatched_target_mass=total_unmatched,
        tv_by_pair_full=tv_full,
        tv_by_pair_non_self_loop=tv_nonsl,
        evaluation_occupancy_weights=occ_weights,
        tv_by_state_occupancy=tv_state_occ,
        tv_by_state_macro=tv_state_macro,
        tv_occupancy=tv_occ,
        tv_macro=tv_macro,
        tv_non_self_loop_occupancy=tv_nonsl_occ,
        tv_non_self_loop_macro=tv_nonsl_macro,
        tv_non_self_loop_by_state_occupancy=tv_nonsl_state_occ,
        tv_non_self_loop_by_state_macro=tv_nonsl_state_macro,
        action_edge_metrics_by_threshold=action_edge,
        action_weights_evaluation_policy=coll["q_eval"],
        action_weights_uniform_valid=coll["q_uniform"],
        valid_collapsed_rows=coll["valid_collapsed_rows"],
        state_occupancy_weights=coll["state_occupancy_weights"],
        collapsed_eval_P_true=coll["evaluation_policy"]["P_true"],
        collapsed_eval_P_hat=coll["evaluation_policy"]["P_hat"],
        collapsed_eval_tv_by_state=coll["evaluation_policy"]["tv_by_state"],
        collapsed_eval_tv_occupancy=coll["evaluation_policy"]["tv_occ"],
        collapsed_eval_tv_macro=coll["evaluation_policy"]["tv_macro"],
        collapsed_eval_tv_non_self_loop_by_state=coll["evaluation_policy"]["tv_non_self_loop_by_state"],
        collapsed_eval_tv_non_self_loop_occupancy=coll["evaluation_policy"]["tv_non_self_loop_occ"],
        collapsed_eval_tv_non_self_loop_macro=coll["evaluation_policy"]["tv_non_self_loop_macro"],
        collapsed_eval_edge_by_threshold=coll["evaluation_policy"]["edge_metrics_by_threshold"],
        collapsed_unif_P_true=coll["uniform_valid_actions"]["P_true"],
        collapsed_unif_P_hat=coll["uniform_valid_actions"]["P_hat"],
        collapsed_unif_tv_by_state=coll["uniform_valid_actions"]["tv_by_state"],
        collapsed_unif_tv_occupancy=coll["uniform_valid_actions"]["tv_occ"],
        collapsed_unif_tv_macro=coll["uniform_valid_actions"]["tv_macro"],
        collapsed_unif_tv_non_self_loop_by_state=coll["uniform_valid_actions"]["tv_non_self_loop_by_state"],
        collapsed_unif_tv_non_self_loop_occupancy=coll["uniform_valid_actions"]["tv_non_self_loop_occ"],
        collapsed_unif_tv_non_self_loop_macro=coll["uniform_valid_actions"]["tv_non_self_loop_macro"],
        collapsed_unif_edge_by_threshold=coll["uniform_valid_actions"]["edge_metrics_by_threshold"],
    )
