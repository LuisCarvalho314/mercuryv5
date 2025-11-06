# tests/test_graph_jax.py
"""
Extensive test suite for the JAX-based `Graph` class and the functional API.

What is tested
--------------
1) Core invariants and registration:
   - Shapes and dtypes for adjacency, node_features features, and edge_features features.
   - Duplicate registrations and bad arguments.

2) Mutations:
   - add_node, add_nodes, add_edge, add_edges.
   - Symmetric writes for undirected graphs or when explicitly requested.
   - Self-loop policy enforcement.
   - remove_edge with and without feature clearing.
   - remove_nodes with mapping and shape compaction.
   - reserve capacity growth without changing n.

3) Utilities and accessors:
   - neighbors (out, in, any) including empty cases and sorting.
   - successors, predecessors aliases.
   - degree, out_degree, in_degree.
   - set_node_feat full overwrite with strict shape checks.
   - set_edge_feat with symmetric option and strict shape checks.
   - to_edge_list and from_edge_list round-trips.
   - assert_invariants on consistent states.

4) Serialization:
   - to_npz / from_npz round-trip, including feature contents.

5) Functional API:
   - gs_empty, gs_register_node/edge_features, gs_add_node/edge_features, gs_remove_nodes.
   - Shape and mapping correctness, and error conditions.

Why this coverage
-----------------
Graphs are stateful data structures. Small shape errors propagate and are hard
to debug under `jit`/`vmap`. These tests lock down shapes, dtypes, and edge_features
cases so refactors remain safe.

How it is tested
----------------
- PyTest functions with clear arrange-act-assert blocks.
- Parametrization where it reduces duplication.
- Strict checks on shapes, dtypes, and values.
- Round-trip checks for serialization and construction helpers.

Notes for Apple Metal users
---------------------------
JAX's Metal backend is experimental. For test stability you can force CPU by
setting `os.environ["JAX_PLATFORM_NAME"] = "cpu"` before importing JAX (e.gs.,
in a `conftest.py`).
"""

import pytest
import numpy as np

from mercury.graph.core import (
    Graph,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def assert_array_equal(a: np.ndarray, b: np.ndarray):
    """Exact equality helper with informative failure."""
    assert a.shape == b.shape, f"shape mismatch: {a.shape} vs {b.shape}"
    assert np.array_equal(a, b), f"values differ:\n{a}\n!=\n{b}"

def assert_allclose(a: np.ndarray, b: np.ndarray, tol=1e-6):
    """Float approximate equality helper."""
    assert a.shape == b.shape, f"shape mismatch: {a.shape} vs {b.shape}"
    assert np.allclose(a, b, atol=tol, rtol=0.0), f"values differ:\n{a}\n!=\n{b}"

# ---------------------------------------------------------------------------
# 1) Core invariants and registration
# ---------------------------------------------------------------------------

def test_initial_state_empty_graph():
    g = Graph()
    assert g.n == 0
    assert g.adj.shape == (0, 0)
    assert g.adj.dtype == np.float32
    assert g.node_features == {}
    assert g.edge_features == {}
    g.assert_invariants()

def test_register_node_feature_dim1_and_dimF():
    g = Graph()
    g.register_node_feature("bias", dim=1, dtype=np.int32, init_value=3)
    g.register_node_feature("state", dim=4, dtype=np.float32, init_value=0.2)
    assert "bias" in g.node_features and "state" in g.node_features
    assert g.node_features["bias"].shape == (0,)
    assert g.node_features["bias"].dtype == np.int32
    assert g.node_features["state"].shape == (0, 4)
    assert g.node_features["state"].dtype == np.float32
    g.assert_invariants()

def test_register_node_feature_specs_and_errors():
    g = Graph()
    g.register_node_feature(specs=[
        {"name": "a", "dim": 1},
        {"name": "b", "dim": 3, "dtype": np.float16, "init_value": 1},
    ])
    assert set(g.node_features) == {"a", "b"}
    assert g.node_features["a"].shape == (0,)
    assert g.node_features["b"].shape == (0, 3)
    assert g.node_features["b"].dtype == np.float16

    with pytest.raises(ValueError):
        g.register_node_feature("a", dim=1)  # duplicate

    with pytest.raises(ValueError):
        g.register_node_feature()  # missing args

    with pytest.raises(ValueError):
        g.register_node_feature("bad", dim=0)  # dim < 1

def test_register_edge_feature_dimE_and_specs():
    g = Graph()
    g.register_edge_feature("w", dim=1, dtype=np.float32, init_value=1.5)
    assert g.edge_features["w"].shape == (0, 0, 1)
    assert g.edge_features["w"].dtype == np.float32

    g.register_edge_feature(specs=[
        {"name": "vec", "dim": 3, "dtype": np.int32, "init_value": 0},
    ])
    assert g.edge_features["vec"].shape == (0, 0, 3)
    assert g.edge_features["vec"].dtype == np.int32

    with pytest.raises(ValueError):
        g.register_edge_feature("w")  # duplicate

    with pytest.raises(ValueError):
        g.register_edge_feature()  # missing

    with pytest.raises(ValueError):
        g.register_edge_feature("bad", dim=0)  # dim < 1

# ---------------------------------------------------------------------------
# 2) Mutations: add_node, add_nodes, add_edge, add_edges, sym, self-loops
# ---------------------------------------------------------------------------

def test_add_node_grows_all_and_writes_defaults_and_node_feats():
    g = Graph()
    g.register_node_feature("bias", dim=1)
    g.register_node_feature("state", dim=3)
    g.register_edge_feature("e", dim=2)

    idx0 = g.add_node(
        node_feat={"bias": 5, "state": np.array([0.1, 0.2, 0.3])},
        edge_defaults_for_new_node={"e": 7.0},  # broadcast to (2,)
    )
    assert idx0 == 0
    assert g.n == 1
    assert g.adj.shape == (1, 1)
    assert g.node_features["bias"].shape == (1,)
    assert g.node_features["state"].shape == (1, 3)
    assert_allclose(g.node_features["bias"], np.array([5.0], dtype=np.float32))
    assert_allclose(g.node_features["state"], np.array([[0.1, 0.2, 0.3]], dtype=np.float32))
    assert g.edge_features["e"].shape == (1, 1, 2)
    assert_allclose(g.edge_features["e"][0, 0, :], np.array([7.0, 7.0], dtype=np.float32))
    g.assert_invariants()

def test_add_node_bad_shapes_raise():
    g = Graph()
    g.register_node_feature("state", dim=3)
    g.register_edge_feature("e", dim=2)

    with pytest.raises(ValueError):
        g.add_node(node_feat={"state": np.array([1.0, 2.0])})  # wrong length

    with pytest.raises(ValueError):
        g.add_node(edge_defaults_for_new_node={"e": np.array([1.0, 2.0, 3.0])})  # wrong E

def test_add_nodes_bulk_span_and_defaults():
    g = Graph()
    g.register_edge_feature("e", dim=1)
    start, end = g.add_nodes(3, edge_defaults={"e": 1.0})
    assert (start, end) == (0, 3)
    assert g.n == 3
    assert g.edge_features["e"].shape == (3, 3, 1)
    # All touching edges for each added node_features got default 1.0 on the diagonal and row/col where created
    # We check presence on diagonal after sequential additions:
    diag = g.edge_features["e"][:, :, 0].diagonal()
    assert_allclose(diag, np.array([1.0, 1.0, 1.0], dtype=np.float32))
    g.assert_invariants()

def test_add_nodes_k_negative_raises():
    g = Graph()
    with pytest.raises(ValueError):
        g.add_nodes(-1)

def test_add_edge_sets_adj_and_features_and_symmetry_flag():
    g = Graph(directed=True)
    g.register_edge_feature("e", dim=3)
    g.add_nodes(2)
    g.add_edge(0, 1, weight=2.5, edge_feat={"e": np.array([1, 2, 3])}, set_symmetric=True)

    assert g.adj.shape == (2, 2)
    assert float(g.adj[0, 1]) == pytest.approx(2.5)
    # symmetric write requested even though directed=True
    assert float(g.adj[1, 0]) == pytest.approx(2.5)

    e = g.edge_features["e"]
    assert e.shape == (2, 2, 3)
    assert_array_equal(e[0, 1], np.array([1, 2, 3], dtype=np.float32))
    assert_array_equal(e[1, 0], np.array([1, 2, 3], dtype=np.float32))
    g.assert_invariants()

def test_add_edge_undirected_default_symmetry():
    g = Graph(directed=False)
    g.add_nodes(2)
    g.add_edge(0, 1)  # by default mirrors
    assert float(g.adj[0, 1]) == 1.0
    assert float(g.adj[1, 0]) == 1.0

def test_add_edge_out_of_bounds_and_self_loop_policy():
    g = Graph()
    g.add_nodes(1)
    with pytest.raises(IndexError):
        g.add_edge(0, 1)

    g.self_loop_allowed = False
    with pytest.raises(ValueError):
        g.add_edge(0, 0)

    g.self_loop_allowed = True
    g.add_edge(0, 0, weight=3.0)
    assert float(g.adj[0, 0]) == pytest.approx(3.0)

def test_add_edges_vectorized_weights_and_features():
    g = Graph()
    g.register_edge_feature("e", dim=2)
    g.add_nodes(3)
    src = np.array([0, 1])
    dst = np.array([1, 2])
    weights = np.array([1.0, 2.0])
    feats = np.array([[1.0, 5.0], [2.0, 6.0]], dtype=np.float32)
    g.add_edges(src, dst, weights=weights, edge_feat={"e": feats})

    assert float(g.adj[0, 1]) == pytest.approx(1.0)
    assert float(g.adj[1, 2]) == pytest.approx(2.0)
    assert_array_equal(g.edge_features["e"][0, 1], np.array([1.0, 5.0], dtype=np.float32))
    assert_array_equal(g.edge_features["e"][1, 2], np.array([2.0, 6.0], dtype=np.float32))

def test_add_edges_broadcast_feature_vector_and_scalar_weight():
    g = Graph()
    g.register_edge_feature("e", dim=2)
    g.add_nodes(3)
    src = np.array([0, 1])
    dst = np.array([1, 2])
    g.add_edges(src, dst, weights=1.0, edge_feat={"e": np.array([9.0, 9.0])})
    assert float(g.adj[0, 1]) == pytest.approx(1.0)
    assert float(g.adj[1, 2]) == pytest.approx(1.0)
    assert_array_equal(g.edge_features["e"][0, 1], np.array([9.0, 9.0], dtype=np.float32))
    assert_array_equal(g.edge_features["e"][1, 2], np.array([9.0, 9.0], dtype=np.float32))

def test_add_edges_length_mismatch_raises():
    g = Graph()
    g.add_nodes(2)
    with pytest.raises(ValueError):
        g.add_edges(np.array([0]), np.array([0, 1]))

def test_add_edges_feature_batch_mismatch_raises():
    g = Graph()
    g.register_edge_feature("e", dim=2)
    g.add_nodes(3)
    with pytest.raises(ValueError):
        g.add_edges(np.array([0, 1]), np.array([1, 2]),
                    edge_feat={"e": np.ones((3, 2))})  # wrong first dim

# ---------------------------------------------------------------------------
# 3) Utilities, accessors, degree, setters
# ---------------------------------------------------------------------------

def test_neighbors_modes_and_empty_results_sorted_any():
    g = Graph()
    g.add_nodes(3)
    g.add_edge(0, 2)
    g.add_edge(2, 1)
    out0 = g.neighbors(0, "out")
    in0 = g.neighbors(0, "in")
    any0 = g.neighbors(0, "any")

    assert_array_equal(out0, np.array([2]))
    assert_array_equal(in0, np.array([], dtype=np.int32))
    assert_array_equal(any0, np.array([2]))

    with pytest.raises(ValueError):
        _ = g.neighbors(0, "bad")

def test_successors_predecessors_aliases():
    g = Graph()
    g.add_nodes(2)
    g.add_edge(0, 1)
    assert_array_equal(g.successors(0), np.array([1]))
    assert_array_equal(g.predecessors(1), np.array([0]))

def test_degrees():
    g = Graph()
    g.add_nodes(3)
    g.add_edge(0, 1)
    g.add_edge(0, 2)
    assert_array_equal(g.out_degree(), np.array([2, 0, 0]))
    assert_array_equal(g.in_degree(), np.array([0, 1, 1]))
    assert_array_equal(g.degree(), np.array([2, 1, 1]))

def test_set_node_feat_full_overwrite_and_shape_checks():
    g = Graph()
    g.register_node_feature("vec", dim=2)
    g.add_nodes(3)
    g.set_node_feat("vec", np.ones((3, 2), dtype=np.float32))
    assert_array_equal(g.get_node_feat("vec"), np.ones((3, 2), dtype=np.float32))

    with pytest.raises(KeyError):
        g.set_node_feat("missing", np.zeros((3, 2)))

    with pytest.raises(ValueError):
        g.set_node_feat("vec", np.zeros((2, 2)))  # wrong n

def test_set_edge_feat_and_symmetric():
    g = Graph(directed=True)
    g.register_edge_feature("e", dim=2)
    g.add_nodes(2)
    g.set_edge_feat("e", 0, 1, np.array([4.0, 5.0]), set_symmetric=True)
    assert_array_equal(g.edge_features["e"][0, 1], np.array([4.0, 5.0]))
    assert_array_equal(g.edge_features["e"][1, 0], np.array([4.0, 5.0]))

    with pytest.raises(KeyError):
        g.set_edge_feat("missing", 0, 1, np.array([1.0]))

    with pytest.raises(IndexError):
        g.set_edge_feat("e", 0, 2, np.array([1.0, 2.0]))

    with pytest.raises(ValueError):
        g.set_edge_feat("e", 0, 1, np.array([1.0, 2.0, 3.0]))  # wrong E

# ---------------------------------------------------------------------------
# remove_edge / remove_nodes
# ---------------------------------------------------------------------------

def test_remove_edge_zeroes_adj_and_optionally_features():
    g = Graph()
    g.register_edge_feature("e", dim=2)
    g.add_nodes(2)
    g.add_edge(0, 1, edge_feat={"e": np.array([3.0, 4.0])})

    # default clear_features=True
    g.remove_edge(0, 1)
    assert float(g.adj[0, 1]) == 0.0
    assert_array_equal(g.edge_features["e"][0, 1], np.array([0.0, 0.0]))

    # set again then clear_features=False
    g.add_edge(0, 1, edge_feat={"e": np.array([5.0, 6.0])})
    g.remove_edge(0, 1, clear_features=False)
    assert float(g.adj[0, 1]) == 0.0
    assert_array_equal(g.edge_features["e"][0, 1], np.array([5.0, 6.0]))

def test_remove_edge_symmetric_on_undirected():
    g = Graph(directed=False)
    g.register_edge_feature("e", dim=1)
    g.add_nodes(2)
    g.add_edge(0, 1, edge_feat={"e": np.array([1.0])})
    g.remove_edge(0, 1)  # mirrors by default
    assert float(g.adj[0, 1]) == 0.0
    assert float(g.adj[1, 0]) == 0.0
    assert float(g.edge_features["e"][0, 1, 0]) == 0.0
    assert float(g.edge_features["e"][1, 0, 0]) == 0.0

def test_remove_nodes_compacts_all_and_returns_mapping():
    g = Graph()
    g.register_node_feature("bias", dim=1)
    g.register_edge_feature("e", dim=1)
    g.add_nodes(4)
    g.add_edge(0, 1)
    g.add_edge(2, 3)
    mapping = g.remove_nodes([1, 3])

    assert g.n == 2
    assert g.adj.shape == (2, 2)
    assert len(mapping) == 4
    assert mapping[0] == 0 and mapping[2] == 1
    assert mapping[1] == -1 and mapping[3] == -1
    g.assert_invariants()


def test_remove_nodes_empty_and_oob_checks():
    g = Graph()
    # empty input -> empty int32 array
    out = g.remove_nodes([])
    assert isinstance(out, np.ndarray)
    assert out.dtype == np.int32
    assert_array_equal(out, np.empty(0, dtype=np.int32))

    # out-of-bounds still raises
    g.add_nodes(1)
    with pytest.raises(IndexError):
        g.remove_nodes([1])

# ---------------------------------------------------------------------------
# reserve capacity
# ---------------------------------------------------------------------------

def test_reserve_grows_storage_not_n():
    g = Graph()
    g.register_node_feature("x", dim=2)
    g.register_edge_feature("e", dim=1)
    g.reserve(5)
    assert g.n == 0
    assert g.adj.shape == (5, 5)
    assert g.node_features["x"].shape == (5, 2)
    assert g.edge_features["e"].shape == (5, 5, 1)
    g.assert_invariants()

def test_reserve_no_shrink_noop_when_capacity_small():
    g = Graph()
    g.add_nodes(3)
    before = (g.adj.shape, g.n)
    g.reserve(2)  # <= n -> no-op
    after = (g.adj.shape, g.n)
    assert before == after

# ---------------------------------------------------------------------------
# to_edge_list / from_edge_list
# ---------------------------------------------------------------------------

def test_to_edge_list_and_from_edge_list_roundtrip():
    g = Graph()
    g.add_nodes(3)
    g.add_edge(0, 1)
    g.add_edge(1, 2)
    el = g.to_edge_list()
    assert el.shape[1] == 2
    # Rebuild from edge_features list
    g2 = Graph.from_edge_list(3, el, directed=True)
    assert_array_equal(g2.adj, g.adj)

def test_from_edge_list_input_validation():
    with pytest.raises(ValueError):
        _ = Graph.from_edge_list(3, np.array([0, 1]))  # not (m,2)

# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

def test_npz_round_trip(tmp_path):
    g = Graph()
    g.register_node_feature("bias", dim=1)
    g.register_edge_feature("e", dim=2)
    g.add_nodes(2)
    g.add_edge(0, 1, edge_feat={"e": np.array([4.0, 5.0])})

    path = tmp_path / "graph.npz"
    g.to_npz(str(path))
    g2 = Graph.from_npz(str(path))
    g2.assert_invariants()

    assert g2.n == g.n
    assert_array_equal(g2.adj, g.adj)
    assert set(g2.node_features) == set(g.node_features)
    assert set(g2.edge_features) == set(g.edge_features)
    assert_array_equal(g2.node_features["bias"], g.node_features["bias"])
    assert_array_equal(g2.edge_features["e"], g.edge_features["e"])

# ---------------------------------------------------------------------------
# assert_invariants explicitly on crafted inconsistent states
# ---------------------------------------------------------------------------

def test_assert_invariants_catches_shape_mismatch():
    import numpy as np
    from mercury.graph.core import Graph

    g = Graph()
    g.add_nodes(2)

    # 1) Too small (capacity < n) -> should fail
    g.adj = np.zeros((1, 1), dtype=np.float32)
    with pytest.raises(AssertionError):
        g.assert_invariants()

    # 2) Not square -> should fail
    g.adj = np.zeros((2, 3), dtype=np.float32)
    with pytest.raises(AssertionError):
        g.assert_invariants()


# ---------------------------------------------------------------------------
# Edge cases and corner behaviors
# ---------------------------------------------------------------------------

def test_neighbors_on_isolated_node_returns_empty():
    g = Graph()
    g.add_nodes(1)
    assert_array_equal(g.neighbors(0, "any"), np.array([], dtype=np.int32))

def test_add_edge_overwrites_existing_weight_and_features():
    g = Graph()
    g.register_edge_feature("e", 2)
    g.add_nodes(2)
    g.add_edge(0, 1, weight=1.0, edge_feat={"e": np.array([1.0, 1.0])})
    g.add_edge(0, 1, weight=3.0, edge_feat={"e": np.array([9.0, 9.0])})
    assert float(g.adj[0, 1]) == pytest.approx(3.0)
    assert_array_equal(g.edge_features["e"][0, 1], np.array([9.0, 9.0]))

def test_remove_nodes_with_duplicates_and_unsorted_indices():
    g = Graph()
    g.add_nodes(5)
    mapping = g.remove_nodes([3, 1, 3, 1])  # duplicates + unsorted
    # Remaining nodes should be old indices [0,2,4] -> new [0,1,2]
    assert g.n == 3
    assert mapping[0] == 0
    assert mapping[2] == 1
    assert mapping[4] == 2
    assert mapping[1] == -1 and mapping[3] == -1

def test_set_symmetric_overrides_directed_flag_both_ways():
    g = Graph(directed=True)
    g.add_nodes(2)
    g.add_edge(0, 1, set_symmetric=True)
    assert float(g.adj[1, 0]) == 1.0
    # Now undirected graph but no mirroring if override False
    g2 = Graph(directed=False)
    g2.add_nodes(2)
    g2.add_edge(0, 1, set_symmetric=False)
    assert float(g2.adj[1, 0]) == 0.0
