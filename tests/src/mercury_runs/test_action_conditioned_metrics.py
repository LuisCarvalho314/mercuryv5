from __future__ import annotations

import numpy as np
import pytest

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
