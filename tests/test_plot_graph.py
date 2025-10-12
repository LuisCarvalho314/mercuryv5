# tests/test_plot_graph.py
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless backend before importing pyplot
import pytest
import networkx as nx

from mercury.graph.plot_graph import to_networkx, plot_graph

# Import your Graph class
from mercury.graph.core import Graph  # adjust if your module path differs


def _make_graph():
    g = Graph(directed=True)
    # register features
    g.register_node_feature("bias", dim=1)
    g.register_node_feature("state", dim=2)
    g.register_edge_feature("age", dim=1)
    g.register_edge_feature("attr", dim=2)
    # nodes
    i0 = g.add_node(node_feat={"bias": 0.5, "state": np.array([1.0, 2.0], np.float32)})
    i1 = g.add_node(node_feat={"bias": 1.5, "state": np.array([3.0, 4.0], np.float32)})
    assert (i0, i1) == (0, 1)
    # edge with features
    g.add_edge(0, 1, weight=2.0,
               edge_feat={
                   "age":  np.array([7.0], np.float32),
                   "attr": np.array([9.0, 10.0], np.float32),
               })
    return g


def test_to_networkx_nodes_and_edges_and_attrs():
    g = _make_graph()
    G = to_networkx(g)

    # structure
    assert isinstance(G, nx.DiGraph)
    assert G.number_of_nodes() == 2
    assert G.number_of_edges() == 1
    assert (0, 1) in G.edges

    # node attributes
    n0 = G.nodes[0]
    n1 = G.nodes[1]
    # scalar kept under key "bias"
    assert pytest.approx(n0.get("bias", 0.0)) == 0.5
    assert pytest.approx(n1.get("bias", 0.0)) == 1.5
    # vector splits into [i] keys
    assert pytest.approx(n0.get("state[0]", 0.0)) == 1.0
    assert pytest.approx(n0.get("state[1]", 0.0)) == 2.0
    assert pytest.approx(n1.get("state[0]", 0.0)) == 3.0
    assert pytest.approx(n1.get("state[1]", 0.0)) == 4.0

    # edge attributes
    e01 = G[0][1]
    assert pytest.approx(e01.get("weight", 0.0)) == 2.0
    # age dim=1 stored under "age"
    assert pytest.approx(e01.get("age", 0.0)) == 7.0
    # attr dim=2 split
    assert pytest.approx(e01.get("attr[0]", 0.0)) == 9.0
    assert pytest.approx(e01.get("attr[1]", 0.0)) == 10.0


def test_plot_graph_runs_headless_with_color_keys():
    g = _make_graph()
    # Should not raise
    plot_graph(
        g,
        layout="spring",
        node_color_key="bias",
        node_label_key="bias",
        edge_color_key="age",
        edge_width_key="weight",
        with_labels=True,
        arrows=True,
        figsize=(4, 3),
    )


def test_plot_graph_accepts_other_layouts():
    g = _make_graph()
    for lay in ["kamada_kawai", "circular", "spectral", "spring", "unknown"]:
        # unknown falls back to spring
        plot_graph(g, layout=lay, with_labels=False)
