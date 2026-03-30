# tests/test_maintenance.py
from __future__ import annotations

import copy
import numpy as np

from mercury.graph.core import Graph
from mercury.graph.maintenance import (
    MaintenanceParams,
    age_maintenance,
    update_ages,
    prune_old_edges,
    remove_lonely_nodes,
)

Array = np.ndarray


def make_graph(n: int = 3) -> Graph:
    """Make graph with n nodes and edge feature 'age'."""
    g = Graph(directed=True)
    g.register_edge_feature("age", dim=1, dtype=np.int32, init_value=0)
    for _ in range(n):
        g.add_node()
    g.assert_invariants()
    return g


def assert_metadata_equal(a: Graph, b: Graph):
    assert a.n == b.n, "n mismatch"
    assert a.self_loop_allowed == b.self_loop_allowed, "self_loop_allowed mismatch"
    assert a.directed == b.directed, "directed flag mismatch"


def assert_adjacency_matrices_equal(a: Graph, b: Graph):
    assert np.array_equal(a.adj, b.adj), "adjacency differs"


def _assert_feature_maps_equal(
    map_a: dict[str, np.ndarray],
    map_b: dict[str, np.ndarray],
    map_name: str,
):
    keys_a = set(map_a.keys())
    keys_b = set(map_b.keys())
    only_in_a = keys_a - keys_b
    only_in_b = keys_b - keys_a
    assert keys_a == keys_b, f"{map_name} keys differ: only_in_a={only_in_a}, only_in_b={only_in_b}"
    for k in sorted(keys_a):
        a = map_a[k]; b = map_b[k]
        assert a.shape == b.shape, f"{map_name}['{k}'] shape {a.shape}!={b.shape}"
        assert a.dtype == b.dtype, f"{map_name}['{k}'] dtype {a.dtype}!={b.dtype}"
        assert np.array_equal(a, b), f"{map_name}['{k}'] values differ"


def assert_node_feature_maps_equal(a: Graph, b: Graph):
    _assert_feature_maps_equal(a.node_features, b.node_features, "node_features")


def assert_edge_feature_maps_equal(a: Graph, b: Graph):
    _assert_feature_maps_equal(a.edge_features, b.edge_features, "edge_features")


def assert_graphs_equal(a: Graph, b: Graph):
    assert_metadata_equal(a, b)
    assert_adjacency_matrices_equal(a, b)
    assert_node_feature_maps_equal(a, b)
    assert_edge_feature_maps_equal(a, b)


def test_mercury_age_maintenance_prunes_and_compacts():
    g = make_graph(n=3)
    p = MaintenanceParams(max_age=10)

    g.add_edge(0, 1, edge_feat={"age": np.array([0], np.int32)})
    g.add_edge(0, 2, edge_feat={"age": np.array([p.max_age - 1], np.int32)})
    g.add_edge(1, 0, edge_feat={"age": np.array([0], np.int32)})

    g_after, old_to_new = age_maintenance(u=0, v=1, g=g, p=p)

    assert np.array_equal(old_to_new, np.array([0, 1, -1], np.int32))
    assert g_after.n == 2
    assert float(g_after.adj[0, 1]) == 1.0
    assert int(np.count_nonzero(g_after.adj)) == 2
    assert int(g_after.edge_features["age"][0, 1, 0]) == 0


def test_mercury_age_maintenance_no_prune_when_under_threshold():
    g = make_graph(n=3)
    p = MaintenanceParams(max_age=1000)

    g.add_edge(0, 1, edge_feat={"age": np.array([0], np.int32)})
    g.add_edge(0, 2, edge_feat={"age": np.array([0], np.int32)})
    g.add_edge(1, 0, edge_feat={"age": np.array([0], np.int32)})

    g_before = copy.deepcopy(g)

    g_after, old_to_new = age_maintenance(u=0, v=1, g=g, p=p)

    assert np.array_equal(old_to_new, np.array([0, 1, 2], np.int32))
    assert g_after.n == 3
    assert int(g_after.edge_features["age"][0, 2, 0]) == 1
    assert int(g_after.edge_features["age"][0, 1, 0]) == 0
    assert np.array_equal(g_after.adj, g_before.adj)


def test_age_updating():
    g = make_graph()
    g.add_edge(0, 1)
    g.add_edge(0, 2)

    update_ages(0, 1, g)

    assert int(g.edge_features["age"][0, 1, 0]) == 0
    assert int(g.edge_features["age"][0, 2, 0]) == 1


def test_age_pruning():
    g = make_graph(n=4)
    p = MaintenanceParams(max_age=10)
    g.add_edge(0, 1)
    g.add_edge(0, 3)

    old_g = copy.deepcopy(g)

    g.add_edge(0, 2, edge_feat={"age": np.array([10], np.int32)})
    prune_old_edges(g, p)

    assert_graphs_equal(old_g, g)


def test_remove_lonely_nodes_drops_only_isolated_tail():
    g = make_graph(n=3)
    g.add_edge(0, 1)
    g.add_edge(0, 2)
    g.add_edge(1, 0)

    old_g = copy.deepcopy(g)

    g.add_node()  # lonely node index 3
    g_after, mapping = remove_lonely_nodes(g)

    assert_graphs_equal(old_g, g_after)
    assert g_after.n == 3
    assert np.array_equal(mapping, np.array([0, 1, 2, -1], np.int32))


def test_remove_lonely_nodes_middle_index_remap():
    g = make_graph(n=4)
    g.add_edge(0, 3)        # lonely: 1,2
    g.add_edge(0, 2)
    g.add_edge(2, 0)        # now 2 is not lonely

    g_after, mapping = remove_lonely_nodes(g)

    assert g_after.n == 3
    assert np.array_equal(mapping, np.array([0, -1, 1, 2], np.int32)), "Mapping does not match"
    assert float(g_after.adj[0, 2]) == 1.0
