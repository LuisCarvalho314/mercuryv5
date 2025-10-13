# tests/test_latent_temporal_edge.py
import numpy as np
import pytest

from mercury.graph.core import Graph
from mercury.latent.state import (
    _register_features,
    _add_node,
    _add_temporal_edge,
    build_initial_graph,
    LatentState,
    compute_bmu,
)
from mercury.memory.state import mem_id


# --------- helpers ---------

class StubMS:
    def __init__(self, sensory_n_nodes: int, L: int):
        self.length = L
        self.sensory_n_nodes = sensory_n_nodes

        class Gs:
            n = sensory_n_nodes * L  # mem_len = S*L
        self.gs = Gs()


def make_graph(mem_len: int) -> Graph:
    g = Graph(directed=True)
    _register_features(g, mem_len)
    return g


def make_ls_from_rows(mem_len: int, rows: list[np.ndarray]) -> LatentState:
    n_lat = len(rows)
    g = Graph(directed=True)
    _register_features(g, mem_len)
    for _ in range(n_lat):
        g.add_node(
            node_feat={
                "activation": np.array(0.0, np.float32),
                "mem_adj": np.zeros(mem_len, np.float32),
            },
            edge_defaults_for_new_node={
                "age": np.array([0], np.int32),
                "action": np.array([0], np.int32),
            },
        )
    g.set_node_feat("mem_adj", np.stack(rows).astype(np.float32))
    return LatentState(g=g, mapping=np.arange(n_lat))


# --------- mem_adj encoding tests ---------

def test_add_temporal_edge_sets_one_hot_no_accumulate():
    S, L = 3, 4
    g = make_graph(mem_len=S * L)

    ms = StubMS(S, L)
    g = _add_node(g, ms, u=0)
    v = 0

    idx0 = mem_id(0, 0, L)
    assert g.node_features["mem_adj"][v, idx0] == np.float32(1.0)

    idx = mem_id(2, 3, L)
    g = _add_temporal_edge(g, u=2, v=v, t=3, L=L)
    assert g.node_features["mem_adj"][v, idx] == np.float32(1.0)

    g = _add_temporal_edge(g, u=2, v=v, t=3, L=L)
    assert g.node_features["mem_adj"][v, idx] == np.float32(1.0)

    row = g.node_features["mem_adj"][v]
    assert np.count_nonzero(row) == 2
    assert row[idx0] == np.float32(1.0)
    assert row[idx] == np.float32(1.0)


def test_add_temporal_edge_bounds_checks():
    S, L = 2, 3
    g = make_graph(mem_len=S * L)
    g = _add_node(g, StubMS(S, L), u=0)

    with pytest.raises(ValueError):
        _add_temporal_edge(g, u=0, v=1, t=L, L=L)
    with pytest.raises(ValueError):
        _add_temporal_edge(g, u=0, v=1, t=-1, L=L)
    with pytest.raises(ValueError):
        _add_temporal_edge(g, u=0, v=g.n, t=0, L=L)


def test_add_temporal_edge_index_overflow_raises_when_mem_adj_too_short():
    L = 3
    g = make_graph(mem_len=2)  # too short on purpose
    ms = StubMS(sensory_n_nodes=2, L=L)

    g = _add_node(g, ms, u=0)  # safe: mem_id(0,0,3)=0 fits
    with pytest.raises(ValueError):
        _add_temporal_edge(g, u=1, v=0, t=0, L=L)  # mem_id(1,0,3)=3 >= 2


def test_add_node_initializes_mem_adj_full_length_and_links_u_to_new_v_at_t0():
    S, L = 4, 2
    ms = StubMS(S, L)
    g = make_graph(mem_len=ms.gs.n)

    g = _add_node(g, ms, u=3)
    v = 0
    idx = mem_id(3, 0, L)

    assert g.node_features["mem_adj"].shape == (1, ms.gs.n)
    assert g.node_features["mem_adj"][v, idx] == np.float32(1.0)
    assert np.count_nonzero(g.node_features["mem_adj"][v]) == 1


def test_build_initial_graph_creates_S_nodes_and_one_hot_links_from_u_at_t0():
    S, L = 5, 3
    ms = StubMS(S, L)
    g = build_initial_graph(ms)

    assert g.n == S
    assert g.node_features["mem_adj"].shape == (S, ms.gs.n)
    for v in range(S):
        idx = mem_id(v, 0, L)
        row = g.node_features["mem_adj"][v]
        assert row[idx] == np.float32(1.0)
        assert np.count_nonzero(row) == 1


# --------- BMU computation tests (compute_bmu returns int) ---------

class StubMS_Act:
    # default length=1 so contender mask is neutral for argmax sanity tests
    def __init__(self, activations: np.ndarray, length: int = 1):
        self.activations = activations
        self.length = length


def test_step_state_returns_bmu_and_sets_activation():
    mem_len = 6
    row0 = np.array([1, 0, 0, 0, 0, 0], np.float32)
    row1 = np.zeros(mem_len, np.float32)
    row2 = np.array([0, 0, 0, 1, 1, 1], np.float32)
    ls = make_ls_from_rows(mem_len, [row0, row1, row2])

    ms = StubMS_Act(activations=np.array([1.0, 0, 0, 0.5, 0.5, 0.5], np.float32), length=1)

    expected = ms.activations @ ls.g.node_features["mem_adj"].T
    bmu = compute_bmu(ls, ms)

    act = ls.g.node_features["activation"].astype(np.float32)
    np.testing.assert_allclose(act, expected, rtol=0, atol=0)
    assert bmu == int(np.argmax(expected))  # 2


def test_step_state_dtype_and_shape_stability():
    mem_len = 4
    ls = make_ls_from_rows(mem_len, [
        np.array([1, 0, 1, 0], np.float32),
        np.array([0, 1, 0, 1], np.float32),
    ])
    ms = StubMS_Act(activations=np.array([0.2, 0.4, 0.6, 0.8], np.float32), length=1)

    bmu = compute_bmu(ls, ms)
    act = ls.g.node_features["activation"]
    assert act.shape == (2,)
    assert act.dtype == np.float32
    assert bmu in (0, 1)


def test_step_state_raises_on_bad_shapes():
    mem_len = 5
    ls = make_ls_from_rows(mem_len, [np.ones(mem_len, np.float32)])
    ms = StubMS_Act(activations=np.ones(4, np.float32), length=1)
    with pytest.raises(ValueError):
        _ = compute_bmu(ls, ms)


# --------- masking rules: t0 required + consecutive prefix 0..k ---------

class StubMS_ActL:
    def __init__(self, activations: np.ndarray, L: int):
        self.activations = activations
        self.length = L


def test_step_state_requires_t0_and_consecutive_timesteps():
    # S=2, L=5 -> mem_len=10; per-sensor columns: s0=[0..4], s1=[5..9]
    L = 5
    mem_len = 10
    row0 = np.array([1,1,1,1,0,  0,0,0,0,0], np.float32)           # valid 0..3
    row1 = np.array([1,1,1,0,1,  1,0,0,0,0], np.float32)           # gap at t=3 -> invalid
    row2 = np.array([0,1,1,1,0,  0,0,0,0,0], np.float32)           # no t0 -> invalid

    ls = make_ls_from_rows(mem_len, [row0, row1, row2])
    ms = StubMS_ActL(activations=np.ones(mem_len, np.float32), L=L)

    bmu = compute_bmu(ls, ms)
    assert bmu == 0


def test_step_state_prefers_higher_activation_among_valid_consecutive():
    L = 4
    mem_len = 8  # S=2
    row0 = np.array([1,1,1,0,  0,0,0,0], np.float32) * 1.0   # valid 0..2
    row1 = np.array([1,1,1,0,  0,0,0,0], np.float32) * 2.0   # valid 0..2, larger
    ls = make_ls_from_rows(mem_len, [row0, row1])
    ms = StubMS_ActL(activations=np.ones(mem_len, np.float32), L=L)

    bmu = compute_bmu(ls, ms)
    assert bmu == 1


def test_step_state_all_invalid_does_not_crash():
    L = 3
    mem_len = 6  # S=2
    rows = [
        np.array([1,0,1,  0,0,0], np.float32),  # gap
        np.array([0,1,1,  0,0,0], np.float32),  # no t0
    ]
    ls = make_ls_from_rows(mem_len, rows)
    ms = StubMS_ActL(activations=np.ones(mem_len, np.float32), L=L)

    bmu = compute_bmu(ls, ms)
    assert bmu in (0, 1)  # fallback used
