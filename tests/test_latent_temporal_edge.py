# tests/test_latent_temporal_edge.py
import numpy as np
import pytest

from mercury.graph.core import Graph
from mercury.latent.state import (
    _register_features,
    _add_node,
    _add_temporal_edge,
    build_initial_graph,
)
from mercury.memory.state import mem_id


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

def test_add_temporal_edge_sets_one_hot_no_accumulate():
    S, L = 3, 4
    g = make_graph(mem_len=S * L)

    # create one latent node; this writes mem_id(u=0, t=0)
    ms = StubMS(S, L)
    g = _add_node(g, ms, u=0)
    v = 0

    idx0 = mem_id(0, 0, L)
    assert g.node_features["mem_adj"][v, idx0] == np.float32(1.0)

    # first write at (u=2,t=3)
    idx = mem_id(2, 3, L)
    g = _add_temporal_edge(g, u=2, v=v, t=3, L=L)
    assert g.node_features["mem_adj"][v, idx] == np.float32(1.0)

    # second write to same slot should not accumulate
    g = _add_temporal_edge(g, u=2, v=v, t=3, L=L)
    assert g.node_features["mem_adj"][v, idx] == np.float32(1.0)

    # exactly two ones now: (0,0) from _add_node and (2,3) from above
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
    g = make_graph(mem_len=2)                     # too short on purpose
    ms = StubMS(sensory_n_nodes=2, L=L)

    # Safe init: mem_id(0,0,3)=0 fits in mem_len=2
    g = _add_node(g, ms, u=0)

    # Now trigger overflow: mem_id(1,0,3)=3 >= 2
    with pytest.raises(ValueError):
        _add_temporal_edge(g, u=1, v=0, t=0, L=L)



def test_add_node_initializes_mem_adj_full_length_and_links_u_to_new_v_at_t0():
    S, L = 4, 2
    ms = StubMS(S, L)
    g = make_graph(mem_len=ms.gs.n)

    # add node with u=3, link should set mem_adj[v, mem_id(3,0,L)] = 1
    g = _add_node(g, ms, u=3)
    v = 0
    idx = mem_id(3, 0, L)

    assert g.node_features["mem_adj"].shape == (1, ms.gs.n)
    assert g.node_features["mem_adj"][v, idx] == np.float32(1.0)
    # one-hot
    assert np.count_nonzero(g.node_features["mem_adj"][v]) == 1


def test_build_initial_graph_creates_S_nodes_and_one_hot_links_from_u_at_t0():
    S, L = 5, 3
    ms = StubMS(S, L)
    g = build_initial_graph(ms)

    assert g.n == S
    assert g.node_features["mem_adj"].shape == (S, ms.gs.n)

    # for each latent node v added in order, link comes from u=v at t=0
    for v in range(S):
        idx = mem_id(v, 0, L)
        row = g.node_features["mem_adj"][v]
        assert row[idx] == np.float32(1.0)
        assert np.count_nonzero(row) == 1
