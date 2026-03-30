# tests/test_latent_temporal_edge.py
import numpy as np
import pytest

from mercury.graph.core import Graph
from mercury.latent.state import (
    _register_features,
    _add_node,
    _add_temporal_edge,
    _update_edges_with_actions,
    build_initial_graph,
    LatentState,
    compute_bmu,
    _predict
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


# replace the old test with this:
def test_add_temporal_edge_rejects_mem_adj_width_not_multiple_of_L():
    L = 3
    g = make_graph(mem_len=2)  # 2 % 3 != 0
    ms = StubMS(sensory_n_nodes=2, L=L)
    with pytest.raises(ValueError, match="not divisible"):
        _add_node(g, ms, u=0)

# add a true overflow test where mem_len is a multiple of L but idx >= mem_len:
def test_add_temporal_edge_index_overflow_raises_when_idx_oob_but_width_valid():
    L = 3
    g = make_graph(mem_len=3)       # multiple of L
    ms = StubMS(sensory_n_nodes=1, L=L)

    g = _add_node(g, ms, u=0)       # create v=0 safely (idx=0)
    # now force overflow: mem_id(1,0,3)=3 which is == mem_len -> oob
    with pytest.raises(ValueError):
        _add_temporal_edge(g, u=1, v=0, t=0, L=L)



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


# --- additional tests to append ---

from mercury.latent.state import _set_activation, latent_step
from mercury.latent.params import LatentParams

def test_compute_bmu_validates_memlen_multiple_of_L():
    # mem_len=6 but L=4 -> not multiple => error
    ls = make_ls_from_rows(6, [np.ones(6, np.float32)])
    class MSBad:
        def __init__(self):
            self.activations = np.ones(6, np.float32)
            self.length = 4
    with pytest.raises(ValueError):
        _ = compute_bmu(ls, MSBad())

def test_set_activation_writes_one_hot_row():
    ls = make_ls_from_rows(3, [np.zeros(3, np.float32)])
    g = _set_activation(ls.g, bmu=0)
    row = g.node_features["activation"]
    assert row.shape == (1,)
    assert row.dtype == np.float32
    assert float(row[0]) == 1.0

def test_compute_bmu_consecutive_check_aggregates_across_sensors():
    # S=2, L=4 -> mem_len=8; columns: s0=[0..3], s1=[4..7]
    L = 4
    mem_len = 8
    # Row A: t=0 from s0(0) and s1(4), t=1 from s1(5), t=2 from s0(2) => active {0,1,2} -> valid
    row_valid = np.array([1,0,1,0, 1,1,0,0], np.float32)
    # Row B: t=0 from s0(0), t=2 from s0(2) only => gap at t=1 overall -> invalid
    row_gap   = np.array([1,0,1,0, 0,0,0,0], np.float32)
    ls = make_ls_from_rows(mem_len, [row_valid, row_gap])
    ms = StubMS_ActL(activations=np.ones(mem_len, np.float32), L=L)

    bmu = compute_bmu(ls, ms)
    assert bmu == 0

def test_latent_step_smoke_updates_prev_bmu_step_idx_and_activation(monkeypatch):
    rows = [
        np.array([1,0,0,0], np.float32),
        np.array([0,1,0,0], np.float32),
    ]
    ls = make_ls_from_rows(4, rows)
    state = LatentState(g=ls.g, mapping=ls.mapping.copy(), prev_bmu=0, step_idx=0)

    ms = StubMS_ActL(activations=np.ones(4, np.float32), L=1)

    class AMStub:
        def __init__(self):
            self.state = type("S", (), {})()
            self.state.codebook = np.eye(3, dtype=np.float32)
        @property
        def params(self):
            return type("P", (), {"dim": 3})()
    am = AMStub()

    from mercury.latent.params import LatentParams
    cfg = LatentParams(gaussian_shape=1, max_age=2)

    import mercury.latent.state as latent_state_mod

    # force current BMU = 1
    monkeypatch.setattr(latent_state_mod, "compute_bmu", lambda *_: 1)

    # stub replay resolver to return (graph, resolved_prev=0)
    def fake_resolve_prev(**kwargs):
        return kwargs["g"], 0
    monkeypatch.setattr(latent_state_mod, "_resolve_prev_with_memory_replay", fake_resolve_prev)

    # stub edge updater no-op
    def fake_update_edges(*, g_in, prev_bmu, bmu, action_bmu, action_map, gaussian_shape):
        return g_in
    monkeypatch.setattr(latent_state_mod, "_update_edges_with_actions", fake_update_edges)

    out, bmu_now = latent_step(
        ms,
        state,
        action_bmu=0,
        cfg=cfg,
        action_map=am,
        action_mem=[0],
    )

    # BMU now is 1
    assert bmu_now == 1

    # prev_bmu updated to resolved_prev (0)
    assert out.prev_bmu == 0

    # activation one-hot at node 1
    act = out.g.node_features["activation"]
    assert int(np.argmax(act)) == 1

    # step advanced
    assert out.step_idx == 1

    # mapping valid
    assert np.array_equal(out.mapping, np.arange(out.g.n))


def test_update_edges_with_actions_uses_passed_beta(monkeypatch):
    g = make_graph(mem_len=4)
    for _ in range(2):
        g.add_node(
            node_feat={
                "activation": np.array(0.0, np.float32),
                "mem_adj": np.zeros(4, np.float32),
                "ambiguity_score": np.zeros(3, np.float32),
            },
            edge_defaults_for_new_node={
                "age": np.array([0], np.int32),
                "action": np.array([0], np.int32),
            },
        )

    class AMStub:
        def __init__(self):
            self.state = type("S", (), {})()
            self.state.codebook = np.eye(3, dtype=np.float32)

        @property
        def params(self):
            return type("P", (), {"dim": 3})()

    recorded: dict[str, float] = {}

    def fake_update_edge(g_in, u, v, prior_row, observed_row, action_bmu, beta, gaussian_shape):
        recorded["beta"] = float(beta)
        return g_in

    monkeypatch.setattr(latent_state_mod, "_update_edge", fake_update_edge)

    out = _update_edges_with_actions(
        g=g,
        prev_bmu=0,
        bmu=1,
        action_bmu=2,
        action_map=AMStub(),
        gaussian_shape=1.0,
        beta=0.37,
    )

    assert out is g
    assert recorded["beta"] == pytest.approx(0.37)



# append at end of file

# --------- _predict tests ---------

def _make_graph_with_actions(mem_len: int, n_lat: int, action_matrix: np.ndarray) -> Graph:
    """
    Build a graph with:
    - n_lat nodes
    - fully connected edges i->j with weight 1.0
    - edge_features['action'][i,j,0] = action_matrix[i,j]

    This matches _predict() which also checks g.adj[i,j] != 0.
    """
    g = Graph(directed=True)
    _register_features(g, mem_len)

    # create nodes
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

    assert action_matrix.shape == (n_lat, n_lat)

    # add edges and set action labels
    for i in range(n_lat):
        for j in range(n_lat):
            g.add_edge(
                i,
                j,
                weight=1.0,
                edge_feat={"action": np.array([action_matrix[i, j]], dtype=np.int32)},
            )

    return g



def test_predict_single_match_row_slice():
    action = np.array([
        [1, 2, 3],
        [3, 3, 0],
        [4, 5, 6],
    ], dtype=np.int32)
    g = _make_graph_with_actions(mem_len=4, n_lat=3, action_matrix=action)
    out = _predict(g, bmu=0, action_bmu=3)
    # row0 == [1,2,3] -> matches index 2
    np.testing.assert_array_equal(out, np.array([2], np.int32))

def test_predict_multiple_matches_row_slice():
    action = np.array([
        [9, 9, 1],
        [3, 3, 3],
        [0, 1, 2],
    ], dtype=np.int32)
    g = _make_graph_with_actions(mem_len=3, n_lat=3, action_matrix=action)
    out = _predict(g, bmu=1, action_bmu=3)
    # row1 == [3,3,3] -> matches indices [0,1,2]
    np.testing.assert_array_equal(out, np.array([0, 1, 2], np.int32))

def test_predict_no_match_returns_empty():
    action = np.array([
        [1, 2],
        [3, 4],
    ], dtype=np.int32)
    g = _make_graph_with_actions(mem_len=2, n_lat=2, action_matrix=action)
    out = _predict(g, bmu=0, action_bmu=7)
    assert out.size == 0
    assert out.dtype == np.int32

def test_predict_all_match_entire_row():
    action = np.array([
        [5, 5],
        [0, 1],
    ], dtype=np.int32)
    g = _make_graph_with_actions(mem_len=4, n_lat=2, action_matrix=action)
    out = _predict(g, bmu=0, action_bmu=5)
    # row0 == [5,5] -> matches [0,1]
    assert out.shape == (2,)
    np.testing.assert_array_equal(out, np.array([0, 1], np.int32))


from mercury.latent.state import _remove_aliased_connections

def test_remove_aliased_connections_removes_expected_edges():
    g = _make_graph_with_actions(mem_len=1, n_lat=4, action_matrix=np.zeros((4,4), np.int32))
    # seed edges
    g.add_edge(1, 2)
    g.add_edge(1, 3)
    g.add_edge(0, 2)

    g = _remove_aliased_connections(g, bmu_prev=1, predicted_nodes=np.array([2, 3], np.int32))

    assert g.adj[1, 2] == 0.0
    assert g.adj[1, 3] == 0.0
    # unrelated edge untouched
    assert g.adj[0, 2] == 1.0

def test_remove_aliased_connections_noop_on_empty():
    g = _make_graph_with_actions(mem_len=1, n_lat=3, action_matrix=np.zeros((3,3), np.int32))
    g.add_edge(1, 2)
    before = g.adj.copy()

    g = _remove_aliased_connections(g, bmu_prev=1, predicted_nodes=np.array([], np.int32))

    np.testing.assert_array_equal(g.adj, before)

def test_remove_aliased_connections_handles_duplicates():
    g = _make_graph_with_actions(mem_len=1, n_lat=4, action_matrix=np.zeros((4,4), np.int32))
    g.add_edge(2, 0)
    g.add_edge(2, 1)
    g.add_edge(2, 3)

    g = _remove_aliased_connections(g, bmu_prev=2, predicted_nodes=np.array([1, 1, 3], np.int32))

    assert g.adj[2, 1] == 0.0
    assert g.adj[2, 3] == 0.0
    # non-aliased edge remains
    assert g.adj[2, 0] == 1.0

def test_remove_aliased_connections_accepts_lists():
    g = _make_graph_with_actions(mem_len=1, n_lat=3, action_matrix=np.zeros((3,3), np.int32))
    g.add_edge(0, 1)
    g.add_edge(0, 2)

    g = _remove_aliased_connections(g, bmu_prev=0, predicted_nodes=[1])

    assert g.adj[0, 1] == 0.0
    assert g.adj[0, 2] == 1.0

# --------- _resolve_ambiguity_add_action_edges tests ---------

import numpy as np
import pytest

import mercury.latent.state as latent_state_mod
from mercury.latent.state import (
    _register_features,
    _resolve_ambiguity_add_action_edges,
)
from mercury.graph.core import Graph


class StubMS_LenOnly:
    """Minimal ms stub for _resolve_ambiguity_add_action_edges."""
    def __init__(self, length: int):
        self.length = length
        # fields used by real activations_at_t are monkeypatched in tests
        # so we do not need ms.gs etc here


class DummyActionMap:
    def __init__(self):
        # matches shape expectations of downstream codepaths if accessed
        self.state = type("S", (), {})()
        self.state.codebook = np.eye(3, dtype=np.float32)


def _make_basic_graph(mem_len: int, n_latent: int) -> Graph:
    """Graph with registered features and n_latent blank nodes."""
    g = Graph(directed=True)
    _register_features(g, mem_len)
    for _ in range(n_latent):
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
    return g
