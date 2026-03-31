import numpy as np
import pytest

from mercury.graph.core import Graph
from mercury.latent.params import LatentParams
from mercury.latent.state import LatentState, _compute_bmu_energy_attribution, _register_features, compute_bmu
from mercury.memory.state import add_memory, init_mem


def test_compute_bmu_uses_configured_lambda_trace() -> None:
    memory_state = add_memory(init_mem(1, length=2), np.array([1.0], dtype=np.float32))

    graph = Graph(directed=True)
    _register_features(graph, mem_len=memory_state.gs.n, n_actions=1)
    for _ in range(2):
        graph.add_node()
    graph.node_features["mem_adj"] = np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32)

    no_penalty = compute_bmu(
        state=LatentState(
            g=graph,
            mapping=np.array([0, 1]),
            prev_activations=np.array([100.0, 0.0], dtype=np.float32),
        ),
        memory_state=memory_state,
        action_bmu=0,
        cfg=LatentParams(
            allow_self_loops=False,
            trace_decay=0.99,
            lambda_trace=0.0,
            weight_memory=1.0,
            weight_undirected=0.0,
            weight_base=0.0,
            weight_action=0.0,
        ),
    )
    with_penalty = compute_bmu(
        state=LatentState(
            g=graph,
            mapping=np.array([0, 1]),
            prev_activations=np.array([100.0, 0.0], dtype=np.float32),
        ),
        memory_state=memory_state,
        action_bmu=0,
        cfg=LatentParams(
            allow_self_loops=False,
            trace_decay=0.99,
            lambda_trace=0.05,
            weight_memory=1.0,
            weight_undirected=0.0,
            weight_base=0.0,
            weight_action=0.0,
        ),
    )

    assert no_penalty == 0
    assert with_penalty == 1


def test_compute_bmu_attribution_exposes_weighted_components_and_shares() -> None:
    memory_state = add_memory(init_mem(1, length=2), np.array([1.0], dtype=np.float32))

    graph = Graph(directed=True)
    _register_features(graph, mem_len=memory_state.gs.n, n_actions=1)
    for _ in range(2):
        graph.add_node()
    graph.node_features["mem_adj"] = np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    graph.adj = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=np.float32)

    computation = _compute_bmu_energy_attribution(
        state=LatentState(
            g=graph,
            mapping=np.array([0, 1]),
            prev_activations=np.array([1.0, 0.0], dtype=np.float32),
        ),
        memory_state=memory_state,
        action_bmu=0,
        cfg=LatentParams(
            allow_self_loops=False,
            trace_decay=0.99,
            lambda_trace=0.05,
            weight_memory=0.5,
            weight_undirected=0.25,
            weight_base=0.25,
            weight_action=0.0,
        ),
    )

    attribution = computation.attribution
    assert attribution.winning_latent_unit == 1
    assert attribution.selected_memory_drive_raw == 1.0
    assert attribution.selected_undirected_drive_raw == 1.0
    assert attribution.selected_baseline_drive_raw == 0.0
    assert attribution.selected_action_drive_raw == 0.0
    assert attribution.selected_total_support_pre_trace == pytest.approx(0.75)
    assert attribution.selected_trace_penalty == 0.0
    assert attribution.selected_total_support_post_trace == pytest.approx(0.75)
    assert attribution.selected_memory_drive_share == pytest.approx(2.0 / 3.0)
    assert attribution.selected_undirected_drive_share == pytest.approx(1.0 / 3.0)
    assert attribution.selected_baseline_drive_share == 0.0
    assert attribution.selected_action_drive_share == 0.0
