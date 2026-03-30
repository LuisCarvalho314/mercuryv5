import numpy as np
import pytest

import mercury.latent.state as latent_state_mod
from mercury.latent.state import (
    _make_past_memory_view,
    _extend_prev_from_memory_once,
    _resolve_prev_with_memory_replay,
    _update_edges_with_actions,
    latent_step,
    LatentState,
    _register_features,
    _set_activation,
)
from mercury.graph.core import Graph


# ---------- shared helpers ----------

class _StubGS:
    def __init__(self, n: int):
        self.n = n


class StubMSFull:
    """
    Minimal MemoryState-like stub for calls that expect:
    - length
    - activations (flattened S*L)
    - gs.n
    - sensory_n_nodes
    """
    def __init__(self, activ_mat: np.ndarray):
        # activ_mat shape (S, L)
        assert activ_mat.ndim == 2
        S, L = activ_mat.shape
        self.length = L
        self.sensory_n_nodes = S
        self.gs = _StubGS(n=S * L)
        self.activations = activ_mat.reshape(S * L).astype(np.float32)


def _make_graph_with_nodes(mem_len: int, n_lat: int) -> Graph:
    """
    Build latent graph with registered node/edge features.
    Add n_lat blank nodes with zero mem_adj.
    """
    g = Graph(directed=True)
    _register_features(g, mem_len)
    for _ in range(n_lat):
        g.add_node(
            node_feat={
                "activation": np.array(0.0, dtype=np.float32),
                "mem_adj": np.zeros(mem_len, dtype=np.float32),
            },
            edge_defaults_for_new_node={
                "age": np.array([0], dtype=np.int32),
                "action": np.array([0], dtype=np.int32),
            },
        )
    return g


def _make_state_with_graph(g: Graph) -> LatentState:
    return LatentState(g=g, mapping=np.arange(g.n, dtype=np.int32), prev_bmu=None, step_idx=0)


class StubActionMap:
    """
    Matches ActionMap interface used by _update_edges_with_actions.
    """
    def __init__(self, dim: int, n_codes: int):
        self.state = type("S", (), {})()
        # simple codebook = identity
        self.state.codebook = np.eye(n_codes, dtype=np.float32)
        self._params = type("P", (), {"dim": dim})()

    @property
    def params(self):
        return self._params


class StubCfg:
    def __init__(self, gaussian_shape: float):
        self.gaussian_shape = gaussian_shape
        self.max_age = 9999


# ---------- _make_past_memory_view tests ----------

def test_make_past_memory_view_shifts_time_axis_and_zeros_future():
    # activ_mat[S,L]:
    # sensor 0: [10,11,12,13]
    # sensor 1: [20,21,22,23]
    activ_mat = np.array(
        [
            [10, 11, 12, 13],
            [20, 21, 22, 23],
        ],
        dtype=np.float32,
    )
    ms = StubMSFull(activ_mat)

    # k = 0  => no shift
    view0 = _make_past_memory_view(ms, 0)
    assert view0.length == ms.length
    np.testing.assert_allclose(
        view0.activations.reshape(2, 4),
        activ_mat,
        rtol=0,
        atol=0,
    )

    # k = 1  => new_t0 = old_t1, new_t1 = old_t2, new_t2 = old_t3, new_t3 = 0
    view1 = _make_past_memory_view(ms, 1)
    shifted_expected = np.array(
        [
            [11, 12, 13, 0],
            [21, 22, 23, 0],
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(
        view1.activations.reshape(2, 4),
        shifted_expected,
        rtol=0,
        atol=0,
    )

    # k >= L  => all zeros
    view_big = _make_past_memory_view(ms, 99)
    assert np.count_nonzero(view_big.activations) == 0
    assert view_big.length == ms.length
    assert view_big.gs.n == ms.gs.n
    assert view_big.sensory_n_nodes == ms.sensory_n_nodes


# ---------- _extend_prev_from_memory_once tests ----------

def test_extend_prev_from_memory_once_progresses_and_spawns(monkeypatch):
    """
    We start with a node that encodes only local t0 for sensor0.
    We fake activations_at_t so that an older global timestep activates sensor1.
    We expect:
    - progressed == True
    - new node spawned
    - new node encodes both local t0(sensor0) and new local t1(sensor1)
    """
    # S=2, L=3. We'll pretend deeper past at global_t=2.
    activ_mat = np.array(
        [
            [1, 0, 0],  # sensor0 timeline
            [0, 1, 1],  # sensor1 timeline (note activity at global_t=2)
        ],
        dtype=np.float32,
    )
    ms = StubMSFull(activ_mat)  # length=3, gs.n=6

    # Build graph with one node whose mem_adj marks only local t0 for sensor0
    g = _make_graph_with_nodes(mem_len=ms.gs.n, n_lat=1)
    # mark sensor0,t0 bit manually
    g.node_features["mem_adj"][0, 0] = np.float32(1.0)  # sensor0,t0 index

    # Fake activations_at_t so that global_t==2 -> sensor1 active
    def fake_activations_at_t(ms_in, t_in):
        if t_in == 2:
            return np.array([0, 1], dtype=np.int32)  # sensor1 active
        return np.array([0, 0], dtype=np.int32)

    monkeypatch.setattr(latent_state_mod, "activations_at_t", fake_activations_at_t)

    # Call
    g_out, new_prev_idx, progressed = _extend_prev_from_memory_once(g, ms, prev_idx=0)

    # We should have progressed and spawned
    assert progressed is True
    assert g_out is g
    assert g.n == 2
    assert new_prev_idx == 1  # new node index

    # New node encodes original local t0(sensor0) and new local t1(sensor1)
    row_new = g.node_features["mem_adj"][new_prev_idx].reshape(ms.sensory_n_nodes, ms.length)
    assert row_new[0, 0] == 1.0  # sensor0 at local t0
    assert row_new[1, 1] == 1.0  # sensor1 at new local t1


def test_extend_prev_from_memory_once_no_progress_when_horizon_hit(monkeypatch):
    """
    If local timeline already hits the horizon or no deeper past sensors fire,
    progressed == False and no node is spawned.
    """
    # S=1, L=2. Node already encodes local t0 and t1.
    activ_mat = np.array([[1, 1]], dtype=np.float32)  # shape (1,2)
    ms = StubMSFull(activ_mat)  # length=2, gs.n=2

    g = _make_graph_with_nodes(mem_len=ms.gs.n, n_lat=1)
    g.node_features["mem_adj"][0, 0] = np.float32(1.0)  # sensor0,t0
    g.node_features["mem_adj"][0, 1] = np.float32(1.0)  # sensor0,t1

    def fake_activations_at_t(ms_in, t_in):
        return np.array([1], dtype=np.int32)

    monkeypatch.setattr(latent_state_mod, "activations_at_t", fake_activations_at_t)

    g_out, new_prev_idx, progressed = _extend_prev_from_memory_once(g, ms, prev_idx=0)

    assert progressed is False
    assert g_out is g
    assert new_prev_idx == 0
    assert g.n == 1  # no spawn


# ---------- _resolve_prev_with_memory_replay tests ----------

def test_resolve_prev_with_memory_replay_unaliased_path_no_change(monkeypatch):
    """
    If _predict returns <=1 dest for the newest action, replay stops immediately.
    We expect:
    - same prev_bmu returned
    - no pruning
    - no extend
    - no edge backfill
    """
    activ_mat = np.array([[1, 0, 0]], dtype=np.float32)  # S=1,L=3 gs.n=3
    ms = StubMSFull(activ_mat)
    g = _make_graph_with_nodes(mem_len=ms.gs.n, n_lat=2)

    state = LatentState(g=g, mapping=np.arange(g.n), prev_bmu=0, step_idx=0)
    prev_bmu = 0
    action_mem = [5, 6, 7]  # arbitrary

    def fake_predict(g_in, node_idx, action_label):
        return np.array([1], dtype=np.int32)  # unaliased (size 1)

    def fake_remove_alias(*args, **kwargs):
        raise AssertionError("should not prune in unaliased branch")

    def fake_extend(*args, **kwargs):
        raise AssertionError("should not extend in unaliased branch")

    def fake_update_edges(*args, **kwargs):
        raise AssertionError("should not wire in unaliased branch")

    monkeypatch.setattr(latent_state_mod, "_predict", fake_predict)
    monkeypatch.setattr(latent_state_mod, "_remove_aliased_connections", fake_remove_alias)
    monkeypatch.setattr(latent_state_mod, "_extend_prev_from_memory_once", fake_extend)
    monkeypatch.setattr(latent_state_mod, "_update_edges_with_actions", fake_update_edges)

    g_out, resolved_prev = _resolve_prev_with_memory_replay(
        g=g,
        state=state,
        ms=ms,
        prev_bmu=prev_bmu,
        action_mem=action_mem,
        action_map=StubActionMap(dim=3, n_codes=4),
        gaussian_shape=1.0,
        max_replay=ms.length,
    )

    assert g_out is g
    assert resolved_prev == prev_bmu
    assert g.n == 2  # unchanged


def test_resolve_prev_with_memory_replay_alias_spawns_and_wires(monkeypatch):
    """
    Aliased case:
    - First iteration: _predict says aliased (>1). We prune and extend.
      _extend_prev_from_memory_once returns (g, new_idx, True) and grows g.
      We then call _update_edges_with_actions(new_prev -> old_prev).
      cur_prev := new_prev.
    - Second iteration: _predict now returns <=1. We stop.
    We expect:
    - node count up by 1
    - returned prev == spawned node
    - _remove_aliased_connections and _update_edges_with_actions both called
    """
    activ_mat = np.array([[1, 0, 0]], dtype=np.float32)  # S=1,L=3
    ms = StubMSFull(activ_mat)
    g = _make_graph_with_nodes(mem_len=ms.gs.n, n_lat=2)  # nodes 0,1 exist
    state = LatentState(g=g, mapping=np.arange(g.n), prev_bmu=0, step_idx=0)

    call_log = {
        "remove_aliased": [],
        "update_edges": [],
    }

    predict_calls = {"count": 0}

    def fake_predict(g_in, node_idx, action_label):
        # first call aliased, second call unaliased
        if predict_calls["count"] == 0:
            predict_calls["count"] += 1
            return np.array([1, 2], dtype=np.int32)  # aliased (>1)
        return np.array([1], dtype=np.int32)         # unaliased

    def fake_remove_alias(g_in, bmu_prev, predicted_nodes):
        call_log["remove_aliased"].append((bmu_prev, predicted_nodes.copy()))
        return g_in

    def fake_extend(g_in, ms_in, prev_idx_in):
        # simulate spawn
        new_idx = g_in.n
        g_in.add_node(
            node_feat={
                "activation": np.array(0.0, np.float32),
                "mem_adj": np.zeros(ms_in.gs.n, np.float32),
            },
            edge_defaults_for_new_node={
                "age": np.array([0], np.int32),
                "action": np.array([0], np.int32),
            },
        )
        return g_in, new_idx, True  # progressed=True

    def fake_update_edges(*,  # accept kwargs from real call
                          g_in,
                          prev_bmu,
                          bmu,
                          action_bmu,
                          action_map,
                          gaussian_shape):
        call_log["update_edges"].append(
            (prev_bmu, bmu, action_bmu, gaussian_shape)
        )
        return g_in

    monkeypatch.setattr(latent_state_mod, "_predict", fake_predict)
    monkeypatch.setattr(latent_state_mod, "_remove_aliased_connections", fake_remove_alias)
    monkeypatch.setattr(latent_state_mod, "_extend_prev_from_memory_once", fake_extend)
    monkeypatch.setattr(latent_state_mod, "_update_edges_with_actions", fake_update_edges)

    prev_bmu = 0
    action_mem = [42, 99]  # most recent action = 99
    gaussian_shape = 3.14

    g_out, resolved_prev = _resolve_prev_with_memory_replay(
        g=g,
        state=state,
        ms=ms,
        prev_bmu=prev_bmu,
        action_mem=action_mem,
        action_map=StubActionMap(dim=3, n_codes=4),
        gaussian_shape=gaussian_shape,
        max_replay=ms.length,
    )

    # graph is same object but gained one node
    assert g_out is g
    assert g.n == 3
    # resolved_prev should be new node index 2
    assert resolved_prev == 2

    # we pruned once
    assert len(call_log["remove_aliased"]) == 1
    assert call_log["remove_aliased"][0][0] == prev_bmu

    # we wired historical edge once with latest action (99)
    assert len(call_log["update_edges"]) == 1
    wired_prev, wired_cur, wired_label, wired_gauss = call_log["update_edges"][0]
    assert wired_prev == 2        # new node
    assert wired_cur == prev_bmu  # old prev
    assert wired_label == 99
    assert wired_gauss == pytest.approx(gaussian_shape)


def test_update_edges_with_actions_uses_passed_beta(monkeypatch):
    g = _make_graph_with_nodes(mem_len=3, n_lat=2)
    action_map = StubActionMap(dim=3, n_codes=3)
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
        action_map=action_map,
        gaussian_shape=1.5,
        beta=0.37,
    )

    assert out is g
    assert recorded["beta"] == pytest.approx(0.37)


# ---------- latent_step integration tests ----------

def test_latent_step_updates_prev_bmu_and_sets_activation(monkeypatch):
    """
    latent_step:
    - compute_bmu -> bmu_now
    - resolve_prev_with_memory_replay -> resolved_prev
    - writes edge resolved_prev -> bmu_now via _update_edges_with_actions
    - sets activation one-hot at bmu_now
    - updates prev_bmu, step_idx, mapping
    """
    # 3 nodes
    mem_len = 4
    g = _make_graph_with_nodes(mem_len=mem_len, n_lat=3)
    state = LatentState(
        g=g,
        mapping=np.arange(g.n),
        prev_bmu=1,
        step_idx=7,
    )

    # Memory stub with length=2
    activ_mat = np.array(
        [
            [1, 0],
            [0, 1],
        ],
        dtype=np.float32,
    )
    ms = StubMSFull(activ_mat)

    am = StubActionMap(dim=2, n_codes=5)
    cfg = StubCfg(gaussian_shape=2.5)

    log = {
        "resolve_prev": [],
        "update_edges": [],
    }

    # force BMU for "now"
    monkeypatch.setattr(latent_state_mod, "compute_bmu", lambda *_args, **_kw: 2)

    # stub replay resolver
    def fake_resolve_prev(**kwargs):
        # kwargs: g, state, ms, prev_bmu, action_mem, action_map, gaussian_shape, max_replay
        log["resolve_prev"].append(kwargs["prev_bmu"])
        return kwargs["g"], 0  # resolved_prev = 0

    monkeypatch.setattr(latent_state_mod, "_resolve_prev_with_memory_replay", fake_resolve_prev)

    # stub edge updater
    def fake_update_edges(*, g_in, prev_bmu, bmu, action_bmu, action_map, gaussian_shape):
        log["update_edges"].append(
            (prev_bmu, bmu, action_bmu, gaussian_shape)
        )
        return g_in

    monkeypatch.setattr(latent_state_mod, "_update_edges_with_actions", fake_update_edges)

    action_bmu_now = 4
    action_mem = [3, 4, 5]

    state_out, bmu_now = latent_step(
        ms=ms,
        state=state,
        action_bmu=action_bmu_now,
        cfg=cfg,
        action_map=am,
        action_mem=action_mem,
    )

    # returned BMU is forced 2
    assert bmu_now == 2

    # step_idx increments
    assert state_out.step_idx == 8

    # prev_bmu updated to resolved_prev (0)
    assert state_out.prev_bmu == 0

    # mapping updated to reflect graph size
    assert np.array_equal(state_out.mapping, np.arange(state_out.g.n))

    # activation one-hot at bmu_now=2
    act_vec = state_out.g.node_features["activation"]
    assert act_vec.shape == (state_out.g.n,)
    assert int(np.argmax(act_vec)) == 2

    # resolver saw original prev_bmu (1)
    assert log["resolve_prev"] == [1]

    # update_edges_with_actions called once with resolved_prev -> bmu_now
    assert len(log["update_edges"]) == 1
    prev_arg, bmu_arg, action_arg, gauss_arg = log["update_edges"][0]
    assert prev_arg == 0
    assert bmu_arg == 2
    assert action_arg == action_bmu_now
    assert gauss_arg == cfg.gaussian_shape


def test_set_activation_sets_single_hot_in_place():
    g = _make_graph_with_nodes(mem_len=6, n_lat=3)
    g2 = _set_activation(g, bmu=1)
    assert g2 is g
    vec = g.node_features["activation"]
    assert vec.dtype == np.float32
    assert vec.shape == (3,)
    assert np.count_nonzero(vec) == 1
    assert vec[1] == 1.0
    assert vec[0] == 0.0 and vec[2] == 0.0
