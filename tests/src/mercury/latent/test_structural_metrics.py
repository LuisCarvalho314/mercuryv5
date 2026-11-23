import numpy as np
import pytest

import mercury.latent.state as latent_state_mod
from mercury.graph.core import Graph
from mercury.latent.state import (
    _add_node,
    _register_features,
    active_edge_ages,
    active_node_ages,
    add_temporal_edge_at_first_gap_clone,
    bmu_memory_depth,
    distribution_summary,
)


class _StubMemory:
    def __init__(self, sensory_n_nodes: int, length: int) -> None:
        self.sensory_n_nodes = sensory_n_nodes
        self.length = length
        self.gs = type("GS", (), {"n": sensory_n_nodes * length})()


def test_bmu_memory_depth_uses_deepest_positive_lag() -> None:
    memory = _StubMemory(sensory_n_nodes=2, length=4)
    graph = Graph(directed=True)
    _register_features(graph, memory.gs.n, n_actions=1)
    graph = _add_node(graph, memory, u=0, t=0, created_step=0)
    graph.node_features["mem_adj"][0, 1] = np.float32(1.0)
    graph.node_features["mem_adj"][0, 7] = np.float32(1.0)

    assert bmu_memory_depth(graph, 0, memory.length) == 3


def test_active_age_distributions_use_creation_timestamps() -> None:
    memory = _StubMemory(sensory_n_nodes=2, length=2)
    graph = Graph(directed=True)
    _register_features(graph, memory.gs.n, n_actions=1)
    graph = _add_node(graph, memory, u=0, t=0, created_step=2)
    graph = _add_node(graph, memory, u=1, t=0, created_step=5)
    graph.add_edge(
        0,
        1,
        weight=1.0,
        edge_feat={"edge_created_at": np.array([4], dtype=np.int32)},
    )

    assert active_node_ages(graph, timestep=7).tolist() == [5, 2]
    assert active_edge_ages(graph, timestep=7).tolist() == [3]


def test_distribution_summary_returns_expected_quantiles() -> None:
    summary = distribution_summary(np.asarray([0, 2, 4], dtype=np.int32), "node_age")

    assert summary["node_age_mean"] == 2.0
    assert summary["node_age_median"] == 2.0
    assert summary["node_age_q25"] == 1.0
    assert summary["node_age_q75"] == 3.0


def test_clone_creation_reports_reuse_without_new_node(monkeypatch: pytest.MonkeyPatch) -> None:
    memory = _StubMemory(sensory_n_nodes=2, length=3)
    graph = Graph(directed=True)
    _register_features(graph, memory.gs.n, n_actions=1)
    graph = _add_node(graph, memory, u=0, t=0, created_step=0)

    monkeypatch.setattr(
        latent_state_mod,
        "activations_at_t",
        lambda ms, t: np.asarray([0, 1], dtype=np.int32) if t == 1 else np.asarray([0, 0], dtype=np.int32),
    )

    graph, first_clone, created = add_temporal_edge_at_first_gap_clone(
        graph,
        memory,
        v=0,
        preds=np.asarray([], dtype=np.int32),
        created_step=4,
    )
    graph, reused_clone, reused_created = add_temporal_edge_at_first_gap_clone(
        graph,
        memory,
        v=0,
        preds=np.asarray([], dtype=np.int32),
        created_step=5,
    )

    assert created is True
    assert reused_created is False
    assert reused_clone == first_clone
    assert graph.n == 2
    assert int(graph.node_features["node_created_at"][first_clone]) == 4
