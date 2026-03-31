from __future__ import annotations

import numpy as np
import pytest

from mercury.graph.core import Graph
from mercury_runs.infrastructure.paper_precision import (
    build_action_conditioned_tensor_from_valid_transitions,
    build_action_conditioned_tensor_from_walks,
    exact_cartesian_reference_positions,
    exact_valid_sensorimotor_transitions,
)
from mercury_runs.infrastructure.paper_precision import compute_action_conditioned_structure_metrics


def test_action_conditioned_structure_metrics_perfect_graph() -> None:
    metrics = compute_action_conditioned_structure_metrics(
        decoded_walks=[np.array([0, 1], dtype=np.int64)],
        true_walks=[np.array([0, 1], dtype=np.int64)],
        action_walks=[np.array([0, 0], dtype=np.int64)],
        T_hat=np.array([[[0.0, 1.0], [0.0, 0.0]]], dtype=np.float64),
        n_true=2,
        T_true=np.array([[[0.0, 1.0], [0.0, 0.0]]], dtype=np.float64),
    )

    assert metrics["edge_precision"] == pytest.approx(1.0)
    assert metrics["edge_recall"] == pytest.approx(1.0)
    assert metrics["edge_f1"] == pytest.approx(1.0)
    assert metrics["mean_total_variation"] == pytest.approx(0.0)


def test_action_conditioned_structure_metrics_missing_correct_edge_gives_max_tv() -> None:
    metrics = compute_action_conditioned_structure_metrics(
        decoded_walks=[np.array([0, 1], dtype=np.int64)],
        true_walks=[np.array([0, 1], dtype=np.int64)],
        action_walks=[np.array([0, 0], dtype=np.int64)],
        T_hat=np.zeros((1, 2, 2), dtype=np.float64),
        n_true=2,
        T_true=np.array([[[0.0, 1.0], [0.0, 0.0]]], dtype=np.float64),
    )

    assert metrics["mean_total_variation"] == pytest.approx(1.0)
    assert metrics["edge_f1"] == pytest.approx(0.0)


def test_action_conditioned_structure_metrics_multiple_successors_increase_tv() -> None:
    metrics = compute_action_conditioned_structure_metrics(
        decoded_walks=[np.array([0, 1], dtype=np.int64)],
        true_walks=[np.array([0, 1], dtype=np.int64)],
        action_walks=[np.array([0, 0], dtype=np.int64)],
        T_hat=np.array([[[1.0, 1.0], [0.0, 0.0]]], dtype=np.float64),
        n_true=2,
        T_true=np.array([[[0.0, 1.0], [0.0, 0.0]]], dtype=np.float64),
    )

    assert metrics["mean_total_variation"] == pytest.approx(0.5)
    assert metrics["edge_precision"] == pytest.approx(0.5)
    assert metrics["edge_recall"] == pytest.approx(1.0)
    assert metrics["edge_f1"] == pytest.approx(2.0 / 3.0)


def test_action_conditioned_structure_metrics_can_ignore_self_loops() -> None:
    decoded_walks = [np.array([0, 1], dtype=np.int64)]
    true_walks = [np.array([0, 1], dtype=np.int64)]
    action_walks = [np.array([0, 0], dtype=np.int64)]
    true_tensor = np.array([[[1.0, 1.0], [0.0, 0.0]]], dtype=np.float64)
    learned_tensor = np.array([[[0.0, 1.0], [0.0, 0.0]]], dtype=np.float64)

    metrics = compute_action_conditioned_structure_metrics(
        decoded_walks=decoded_walks,
        true_walks=true_walks,
        action_walks=action_walks,
        T_hat=learned_tensor,
        n_true=2,
        T_true=true_tensor,
        ignore_self_loops=False,
    )
    ignored_metrics = compute_action_conditioned_structure_metrics(
        decoded_walks=decoded_walks,
        true_walks=true_walks,
        action_walks=action_walks,
        T_hat=learned_tensor,
        n_true=2,
        T_true=true_tensor,
        ignore_self_loops=True,
    )

    assert metrics["edge_recall"] == pytest.approx(0.5)
    assert metrics["edge_f1"] == pytest.approx(2.0 / 3.0)
    assert metrics["mean_total_variation"] == pytest.approx(0.5)
    assert ignored_metrics == {
        "mean_total_variation": pytest.approx(0.0),
        "edge_precision": pytest.approx(1.0),
        "edge_recall": pytest.approx(1.0),
        "edge_f1": pytest.approx(1.0),
    }


def test_level_13_true_action_conditioned_transitions_include_boundary_self_loops() -> None:
    positions = exact_cartesian_reference_positions(13)
    transitions = exact_valid_sensorimotor_transitions(level_index=13)
    tensor = build_action_conditioned_tensor_from_valid_transitions(
        transitions,
        n_states=int(positions.shape[0]),
        n_actions=4,
    )

    # Level 13 uses environment action IDs: north=0, east=1, south=2, west=3.
    # State 0 corresponds to position (1, 1), a top-left corner in the open area.
    assert tuple(positions[0].tolist()) == (1, 1)
    assert np.flatnonzero(tensor[0, 0]).tolist() == [0]
    assert np.flatnonzero(tensor[1, 0]).tolist() == [0]
    assert np.flatnonzero(tensor[2, 0]).tolist() == [2]
    assert np.flatnonzero(tensor[3, 0]).tolist() == [0]

    # State 2 corresponds to position (2, 1), where south and west are blocked.
    assert tuple(positions[2].tolist()) == (2, 1)
    assert np.flatnonzero(tensor[0, 2]).tolist() == [0]
    assert np.flatnonzero(tensor[1, 2]).tolist() == [3]
    assert np.flatnonzero(tensor[2, 2]).tolist() == [2]
    assert np.flatnonzero(tensor[3, 2]).tolist() == [2]


def test_graph_edge_action_label_is_overwritten_for_same_source_target_pair() -> None:
    graph = Graph(directed=True)
    graph.register_edge_feature("action", dim=1, dtype=np.int32, init_value=0)
    graph.add_node()
    graph.add_node()

    graph.add_edge(0, 0, weight=1.0, edge_feat={"action": np.array([0], dtype=np.int32)})
    graph.add_edge(0, 0, weight=1.0, edge_feat={"action": np.array([3], dtype=np.int32)})

    assert float(graph.adj[0, 0]) == pytest.approx(1.0)
    assert int(graph.edge_features["action"][0, 0, 0]) == 3


def test_build_action_conditioned_tensor_from_walks_uses_post_step_action_alignment() -> None:
    tensor = build_action_conditioned_tensor_from_walks(
        state_walks=[np.array([0, 2, 3], dtype=np.int64)],
        action_walks=[np.array([3, 2, 1], dtype=np.int64)],
        n_states=4,
        n_actions=4,
    )

    assert np.flatnonzero(tensor[2, 0]).tolist() == [2]
    assert np.flatnonzero(tensor[1, 2]).tolist() == [3]
    assert not np.any(tensor[3, 0])
    assert not np.any(tensor[2, 2])


def test_build_action_conditioned_tensor_from_walks_keeps_level_13_rows_deterministic() -> None:
    # Stored evaluation walks pair each action with the post-step state.
    # This path visits state 2 and then state 3 in level 13, with east as the
    # action leaving state 2. The empirical action-conditioned row for
    # (state=2, action=east) should therefore stay deterministic.
    tensor = build_action_conditioned_tensor_from_walks(
        state_walks=[np.array([0, 2, 3], dtype=np.int64)],
        action_walks=[np.array([0, 0, 1], dtype=np.int64)],
        n_states=7,
        n_actions=4,
    )

    assert np.flatnonzero(tensor[1, 2]).tolist() == [3]
    assert not np.any(tensor[0, 2])
    assert not np.any(tensor[2, 2])
