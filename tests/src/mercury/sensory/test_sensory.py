# tests/test_sensory_numpy.py
from __future__ import annotations
import numpy as np
import pytest

from mercury.sensory.state import (
    build_initial_graph, init_state, SensoryState,
    _calc_activation, _should_create_node, _pick_bmu,
    _compute_dn, _topological_similarity, _meets_threshold,
    _update_winning_node, _update_neighbours, _update_edge, _edge_action_row,
    sensory_step,
    sensory_step_frozen,
)
from mercury.sensory.params import SensoryParams
from mercury.action_map.adapter import ActionMap
from mercury.graph.core import Graph

Array = np.ndarray


# ---------- build_initial_graph ----------
def test_build_initial_graph():
    data_dim = 4
    g = build_initial_graph(data_dim)
    assert isinstance(g, Graph)
    assert g.n == 2
    assert g.adj.shape == (2, 2)
    np.testing.assert_array_equal(g.node_features["weight"], np.zeros((2, data_dim), np.float32))
    np.testing.assert_array_equal(g.node_features["context"], np.zeros((2, data_dim), np.float32))
    np.testing.assert_array_equal(g.node_features["activation"], np.zeros((2,), np.float32))
    np.testing.assert_array_equal(g.edge_features["age"], np.zeros((2, 2, 1), np.int32))
    np.testing.assert_array_equal(g.edge_features["action"], np.zeros((2, 2, 1), np.int32))


# ---------- _calc_activation ----------
def test_calc_activation():
    d_bmu = 0.0
    gaussian_shape = 1.0
    activation = _calc_activation(d_bmu, gaussian_shape)
    assert activation == 1.0
    assert isinstance(activation, float)


# ---------- _should_create_node ----------
@pytest.mark.parametrize(
    "activation,thr,n,cap,expected",
    [
        (0.9, 0.8, 9, 10, False),
        (0.9, 0.8, 11, 10, False),
        (0.7, 0.8, 9, 10, True),
        (0.7, 0.8, 11, 10, False),
        (0.8, 0.8, 9, 10, True),
        (0.9, 0.8, 10, 10, False),
        (0.8, 0.8, 10, 10, False),
    ],
)
def test_should_create_node(activation, thr, n, cap, expected):
    out = _should_create_node(activation, thr, n, cap)
    assert isinstance(out, bool)
    assert out is expected


# ---------- _pick_bmu ----------
def test_pick_bmu_weight_only():
    weights = np.array([[0.0, 0.0],
                        [1.0, 0.0]], dtype=np.float32)
    contexts = np.array([[0.0, 0.0],
                         [0.0, 0.0]], dtype=np.float32)
    observation = np.array([0.9, 0.0], dtype=np.float32)
    global_context = np.array([0.0, 0.0], dtype=np.float32)

    bmu, resid, _ = _pick_bmu(weights, contexts, observation, global_context, sensory_weighting=1.0)

    assert bmu == 1
    expected_resid = np.sum((weights - observation) ** 2, axis=1)
    np.testing.assert_array_equal(resid, expected_resid)


def test_pick_bmu_context_only():
    weights = np.array([[0.0, 0.0],
                        [0.0, 0.0]], dtype=np.float32)
    contexts = np.array([[0.0, 1.0],
                         [1.0, 0.0]], dtype=np.float32)
    observation = np.array([0.0, 0.0], dtype=np.float32)
    global_context = np.array([1.0, 0.0], dtype=np.float32)

    bmu, resid, _ = _pick_bmu(weights, contexts, observation, global_context, sensory_weighting=0.0)

    assert bmu == 1
    np.testing.assert_array_equal(resid, np.sum((weights - observation) ** 2, axis=1))


def test_pick_bmu_tie_breaks_to_lower_index():
    weights = np.array([[1.0, 0.0],
                        [0.0, 1.0]], dtype=np.float32)
    contexts = np.array([[0.0, 1.0],
                         [1.0, 0.0]], dtype=np.float32)
    observation = np.array([0.5, 0.5], dtype=np.float32)
    global_context = np.array([0.5, 0.5], dtype=np.float32)

    bmu, _, _ = _pick_bmu(weights, contexts, observation, global_context, sensory_weighting=0.5)
    assert bmu == 0  # np.argmin picks first on ties


def test_pick_bmu_output_types_and_shapes():
    weights = np.zeros((3, 4), np.float32)
    contexts = np.zeros((3, 4), np.float32)
    observation = np.zeros((4,), np.float32)
    global_context = np.zeros((4,), np.float32)

    bmu, resid, _ = _pick_bmu(weights, contexts, observation, global_context, sensory_weighting=0.3)

    assert isinstance(bmu, int)
    assert resid.shape == (3,)
    assert resid.dtype == np.float32


# ---------- graph helper ----------
def _make_graph(n: int = 3, dim: int = 2) -> Graph:
    g = build_initial_graph(data_dim=dim, n=n)
    return g


# ---------- _update_winning_node ----------
def test_update_node_updates_only_selected_row():
    dim, n = 3, 4
    g = _make_graph(n=n, dim=dim)

    init_w = (np.arange(n * dim, dtype=np.float32).reshape(n, dim) * 0.1).astype(np.float32)
    init_c = np.flip(init_w, axis=0).astype(np.float32)
    g.set_node_feat("weight", init_w)
    g.set_node_feat("context", init_c)

    bmu = 2
    x = np.array([1.0, -1.0, 0.5], np.float32)
    gc = np.array([-0.5, 0.25, 2.0], np.float32)
    lr = 0.3

    g2 = _update_winning_node(g, bmu=bmu, observation=x, global_context=gc, lr=lr)

    expected_w_bmu = init_w[bmu] + lr * (x - init_w[bmu])
    expected_c_bmu = init_c[bmu] + lr * (gc - init_c[bmu])

    for i in range(n):
        if i == bmu:
            np.testing.assert_allclose(g2.node_features["weight"][i], expected_w_bmu)
            np.testing.assert_allclose(g2.node_features["context"][i], expected_c_bmu)
        else:
            np.testing.assert_array_equal(g2.node_features["weight"][i], init_w[i])
            np.testing.assert_array_equal(g2.node_features["context"][i], init_c[i])


def test_update_node_lr_zero_is_no_op():
    dim, n = 2, 3
    g = _make_graph(n=n, dim=dim)

    init_w = np.array([[0.1, 0.2],
                       [0.3, 0.4],
                       [0.5, 0.6]], dtype=np.float32)
    init_c = np.array([[0.6, 0.5],
                       [0.4, 0.3],
                       [0.2, 0.1]], dtype=np.float32)
    g.set_node_feat("weight", init_w)
    g.set_node_feat("context", init_c)

    g2 = _update_winning_node(
        g, bmu=1,
        observation=np.array([9.9, 9.9], np.float32),
        global_context=np.array([-9.9, -9.9], np.float32),
        lr=0.0,
    )

    np.testing.assert_array_equal(g2.node_features["weight"], init_w)
    np.testing.assert_array_equal(g2.node_features["context"], init_c)


def test_update_node_preserves_shapes_and_dtypes():
    dim, n = 5, 2
    g = _make_graph(n=n, dim=dim)

    x = np.linspace(0.0, 1.0, dim, dtype=np.float32)
    gc = np.linspace(1.0, 2.0, dim, dtype=np.float32)

    g2 = _update_winning_node(g, bmu=0, observation=x, global_context=gc, lr=0.5)

    w2 = g2.node_features["weight"]
    c2 = g2.node_features["context"]

    assert w2.shape == (n, dim)
    assert c2.shape == (n, dim)
    assert w2.dtype == np.float32
    assert c2.dtype == np.float32


# ---------- _compute_dn ----------
def test_compute_dn_weight_vs_context_limits():
    n, d = 3, 2
    w = np.array([[0., 0.],
                  [1., 0.],
                  [0., 1.]], np.float32)
    c = np.array([[0., 0.],
                  [0., 1.],
                  [1., 0.]], np.float32)
    x = np.array([1., 0.], np.float32)
    gc = np.array([0., 1.], np.float32)

    dn_w1, wt1, _ = _compute_dn(1.0, x, gc, w, c)
    dn_w0, wt0, _ = _compute_dn(0.0, x, gc, w, c)

    np.testing.assert_allclose(dn_w1, wt1)
    ctx_term = np.sum((c - gc) ** 2, axis=1)
    np.testing.assert_allclose(dn_w0, ctx_term)


# ---------- _topological_similarity / _meets_threshold ----------
def test_topological_similarity_threshold_and_sigma_guard():
    dist = np.array([0.2, 0.5, 1.0], np.float32)
    thr = 0.5
    sigma = 0.3
    sim = _topological_similarity(dist, thr, sigma)
    assert sim.shape == (3,)
    assert sim[0] > 0.0
    # threshold boundary behavior: _meets_threshold returns x <= thr
    # so 0.5 is kept; with small sigma similarity is nonzero.
    assert float(sim[1]) != 0.0
    assert float(sim[2]) == 0.0


def test_meets_threshold_vectorized():
    x = np.array([0.2, 0.5, 0.7], np.float32)
    mask = _meets_threshold(x, 0.5)
    assert mask.dtype == np.bool_
    assert mask.tolist() == [True, True, False]


# ---------- _update_neighbours ----------
def test_update_neighbours_no_neighbors_noop():
    n, d = 3, 2
    g = _make_graph(n, d)
    # no edges, so BMU has no neighbors
    w0 = np.ones((n, d), np.float32)
    c0 = np.zeros((n, d), np.float32)
    g.set_node_feat("weight", w0)
    g.set_node_feat("context", c0)

    g2 = _update_neighbours(
        g, bmu=1,
        observation=np.array([2., 2.], np.float32),
        diff_weights=np.zeros_like(w0),
        gaussian_shape=0.5,
        distance_threshold=0.2,
        lr=0.3,
    )
    np.testing.assert_array_equal(g2.node_features["weight"], w0)
    np.testing.assert_array_equal(g2.node_features["context"], c0)


def test_update_neighbours_updates_only_neighbors_and_excludes_bmu():
    n, d = 4, 2
    g = _make_graph(n, d)
    # neighbors for node 1: connect 1->0 and 1->2
    g.add_edge(1, 0, weight=1.0)
    g.add_edge(1, 2, weight=1.0)

    w = np.array([[0., 0.],
                  [1., 1.],
                  [2., 2.],
                  [9., 9.]], np.float32)
    c = np.zeros_like(w)
    g.set_node_feat("weight", w)
    g.set_node_feat("context", c)

    x = np.array([1., 1.], np.float32)
    d_weight = w - x
    g2 = _update_neighbours(
        g, bmu=1, observation=x, diff_weights=d_weight,
        gaussian_shape=1.0, distance_threshold=2.0, lr=0.5
    )

    for i in [0, 2]:
        assert not np.array_equal(g2.node_features["weight"][i], w[i])
        assert not np.array_equal(g2.node_features["context"][i], c[i])
    # BMU and non-neighbor unchanged
    np.testing.assert_array_equal(g2.node_features["weight"][1], w[1])
    np.testing.assert_array_equal(g2.node_features["context"][1], c[1])
    np.testing.assert_array_equal(g2.node_features["weight"][3], w[3])
    np.testing.assert_array_equal(g2.node_features["context"][3], c[3])


# ---------- _update_winning_node sanity ----------
def test_update_winning_node_moves_toward_inputs():
    g = _make_graph(2, 3)
    w = np.array([[0., 0., 0.],
                  [2., 2., 2.]], np.float32)
    c = np.array([[0., 0., 0.],
                  [2., 2., 2.]], np.float32)
    g.set_node_feat("weight", w)
    g.set_node_feat("context", c)

    x = np.array([1., 1., 1.], np.float32)
    gc = np.array([3., 3., 3.], np.float32)
    lr = 0.25
    g2 = _update_winning_node(g, 1, x, gc, lr)
    np.testing.assert_allclose(g2.node_features["weight"][1], w[1] + lr * (x - w[1]))
    np.testing.assert_allclose(g2.node_features["context"][1], c[1] + lr * (gc - c[1]))


# ---------- _update_edge / _edge_action_row ----------
def test_update_edge_sets_weight_and_action_feature():
    g = _make_graph(2, 2)
    u, v = 0, 1
    prior = np.zeros((2,), np.float32)
    observed = np.zeros((2,), np.float32)
    g2 = _update_edge(g, u, v, prior, observed, action_bmu=7, beta=0.5, gaussian_shape=1.0)
    assert float(g2.adj[u, v]) == pytest.approx(0.5, rel=0, abs=1e-6)
    assert int(g2.edge_features["action"][u, v, 0]) == 7


# ---------- pick + activation sanity ----------
def test_pick_bmu_and_activation_sanity():
    w = np.array([[0., 0.], [2., 0.]], np.float32)
    c = np.zeros_like(w)
    x = np.array([1.9, 0.0], np.float32)
    gc = np.array([0., 0.], np.float32)

    bmu, d_weight, _ = _pick_bmu(w, c, x, gc, sensory_weighting=1.0)
    assert bmu == 1
    act = _calc_activation(np.sqrt(d_weight[bmu]), 1.0)
    assert isinstance(act, float)
    assert 0.0 < act <= 1.0


# ---------- init_state ----------
def test_init_state_sets_global_context_and_shapes():
    s = init_state(data_dim=3, n=2)
    assert isinstance(s, SensoryState)
    assert s.gs.n == 2
    assert s.global_context.shape == (3,)
    assert np.all(s.global_context == 0.0)


# ---------- edge case: single neighbor shape safety ----------
def test_update_neighbours_single_neighbor_shape_safe():
    g = Graph(directed=True)
    g.register_node_feature("weight", dim=3, dtype=np.float32, init_value=0.0)
    g.register_node_feature("context", dim=3, dtype=np.float32, init_value=0.0)
    g.register_edge_feature("age", dim=1, dtype=np.int32, init_value=0)
    g.register_edge_feature("action", dim=1, dtype=np.int32, init_value=0)
    g.add_node(); g.add_node()         # nodes 0,1
    g.add_edge(0, 1)                   # 0 -> 1 neighbor

    weights = np.array([[0., 0., 0.],
                        [1., 1., 1.]], dtype=np.float32)
    contexts = np.array([[0., 0., 0.],
                         [0., 0., 0.]], dtype=np.float32)
    g.set_node_feat("weight", weights)
    g.set_node_feat("context", contexts)

    bmu = 0
    observation = np.array([3., 0., 0.], np.float32)
    diff_weights = np.array([[0., 0., 0.],
                             [2., 0., 0.]], np.float32)
    gaussian_shape = 1.0
    distance_threshold = 10.0
    lr = 0.5

    g2 = _update_neighbours(
        g, bmu, observation, diff_weights,
        gaussian_shape=gaussian_shape,
        distance_threshold=distance_threshold,
        lr=lr
    )

    dist = np.linalg.norm(np.array([2., 0., 0.], np.float32))
    topo = _topological_similarity(np.array([dist], np.float32), distance_threshold, gaussian_shape)[0]
    expected_w1 = np.array([1., 1., 1.], np.float32) + lr * np.array([2., 0., 0.], np.float32) * topo
    expected_c1 = np.array([0., 0., 0.], np.float32) + lr * (observation - np.array([0., 0., 0.], np.float32)) * topo

    np.testing.assert_array_equal(g2.node_features["weight"][0], weights[0])
    np.testing.assert_array_equal(g2.node_features["context"][0], contexts[0])
    assert g2.node_features["weight"].shape == (2, 3)
    assert g2.node_features["context"].shape == (2, 3)
    np.testing.assert_allclose(g2.node_features["weight"][1], expected_w1, atol=1e-6)
    np.testing.assert_allclose(g2.node_features["context"][1], expected_c1, atol=1e-6)


# ---------- _topological_similarity extra checks ----------
def test_topological_similarity_vectorizes_and_masks():
    dist = np.array([0.0, 1.0, 3.0], np.float32)
    sim = _topological_similarity(dist, distance_threshold=2.0, gaussian_shape=1.0)
    expected = np.array([1.0, np.exp(-0.5 * 1.0 ** 2), 0.0], np.float32)
    np.testing.assert_allclose(sim, expected, atol=1e-6)


def test_topological_similarity_sigma_le_zero_returns_zeros():
    dist = np.array([0.0, 1.0], np.float32)
    sim0 = _topological_similarity(dist, distance_threshold=5.0, gaussian_shape=0.0)
    sim_neg = _topological_similarity(dist, distance_threshold=5.0, gaussian_shape=-1.0)
    np.testing.assert_array_equal(sim0, np.zeros_like(dist))
    np.testing.assert_array_equal(sim_neg, np.zeros_like(dist))


def test_topological_similarity_scalar_input_returns_length1():
    dist = np.array(1.0, np.float32)
    sim = _topological_similarity(dist, distance_threshold=2.0, gaussian_shape=1.0)
    assert sim.shape == (1,)
    expected = np.array([np.exp(-0.5 * 1.0 ** 2)], np.float32)
    np.testing.assert_allclose(sim, expected, atol=1e-6)


def test_sensory_step_frozen_updates_context_without_learning_edges():
    state = init_state(data_dim=2, n=2)
    weights = np.array([[0.0, 0.0], [2.0, 2.0]], dtype=np.float32)
    contexts = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)
    state.gs.set_node_feat("weight", weights)
    state.gs.set_node_feat("context", contexts)
    state.global_context = np.array([1.0, 1.0], dtype=np.float32)
    state.prev_bmu = 0
    state.gs.add_edge(0, 1, weight=0.25, edge_feat={"action": np.array([3], dtype=np.int32)})

    adj_before = state.gs.adj.copy()
    edge_action_before = state.gs.edge_features["action"].copy()
    cfg = SensoryParams(sensory_weighting=0.0, global_context_lr=0.25)
    action_map = ActionMap.identity(dim=2)

    next_state = sensory_step_frozen(
        observation=np.array([9.0, 9.0], dtype=np.float32),
        action_bmu=1,
        state=state,
        cfg=cfg,
        action_map=action_map,
    )

    assert next_state.prev_bmu == 1
    np.testing.assert_array_equal(next_state.gs.adj, adj_before)
    np.testing.assert_array_equal(next_state.gs.edge_features["action"], edge_action_before)
    np.testing.assert_array_equal(next_state.gs.node_features["activation"], np.array([0.0, 1.0], dtype=np.float32))
    np.testing.assert_allclose(next_state.global_context, np.array([1.25, 1.25], dtype=np.float32))


def test_sensory_step_uses_configured_action_lr(monkeypatch):
    state = init_state(data_dim=2, n=2)
    state.gs.set_node_feat("weight", np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32))
    state.gs.set_node_feat("context", np.zeros((2, 2), dtype=np.float32))
    state.prev_bmu = 0

    recorded: dict[str, float] = {}

    def fake_update_edge(g, u, v, prior_row, observed_row, action_bmu, beta, gaussian_shape):
        recorded["beta"] = float(beta)
        return g

    monkeypatch.setattr("mercury.sensory.state._update_edge", fake_update_edge)
    monkeypatch.setattr("mercury.sensory.state.age_maintenance", lambda u, v, g, p: (g, np.arange(g.n, dtype=np.int32)))

    action_map = ActionMap.identity(dim=2)
    cfg = SensoryParams(action_lr=0.23, gaussian_shape=2)

    next_state = sensory_step(
        observation=np.array([0.0, 0.0], dtype=np.float32),
        action_bmu=1,
        state=state,
        cfg=cfg,
        action_map=action_map,
    )

    assert next_state.prev_bmu == 0
    assert recorded["beta"] == pytest.approx(0.23)
