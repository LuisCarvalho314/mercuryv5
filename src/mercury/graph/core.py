"""
Graph (NumPy version)
=====================

Dense, directed/undirected graph with per-node and per-edge feature storage.

Key invariants
--------------
- `adj`: shape `(n, n)`, dtype float32. Nonzero => edge present.
- `node_features[name]`:
    * `(n,)` if registered with `dim==1`
    * `(n, F)` if registered with `dim==F>1`
- `edge_features[name]`:
    * `(n, n, E)` with `E>=1` channels per edge

Quickstart
----------
>>> import numpy as np
>>> g = Graph(directed=True)
>>> g.register_node_feature("bias", dim=1)
>>> g.register_node_feature("state", dim=3)
>>> g.register_edge_feature("attr", dim=2)
>>> g.add_node(node_feat={"bias": 1.0, "state": np.array([0.1, 0.2, 0.3])},
...            edge_defaults_for_new_node={"attr": 5.0})
0
>>> g.add_node()
1
>>> g.add_edge(0, 1, weight=2.0, edge_feat={"attr": np.array([7.0, 8.0])})
>>> g.neighbors(0, "out").tolist()
[1]
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np

Array = np.ndarray


@dataclass(frozen=True)
class _Dims:
    """Internal helper for feature dimensions."""
    n: int
    F: Optional[int] = None
    E: Optional[int] = None


class Graph:
    """Dense graph with per-node and per-edge feature storage."""

    def __init__(self, directed: bool = True):
        """
        Initialize an empty dense graph.

        Parameters
        ----------
        directed : bool, default True
            If True, edges are directed. If False, you may mirror operations
            by passing `set_symmetric=True` to edge-mutating methods.

        Notes
        -----
        Creates empty storages:
        `adj` with shape (0, 0),
        `node_features` and `edge_features` as empty dicts.
        """
        self.directed: bool = bool(directed)
        self.n: int = 0
        self.adj: Array = np.zeros((0, 0), dtype=np.float32)
        self.node_features: Dict[str, Array] = {}
        self.edge_features: Dict[str, Array] = {}
        self.node_schema: Dict[str, Dict[str, Union[int, str]]] = {}
        self.edge_schema: Dict[str, Dict[str, Union[int, str]]] = {}
        self.self_loop_allowed: bool = True

    # ---------- registration ----------

    def register_node_feature(
        self,
        name: Optional[str] = None,
        dim: Optional[int] = None,
        dtype: np.dtype = np.float32,
        init_value: Union[float, int] = 0.0,
        specs: Optional[List[dict]] = None,
    ) -> None:
        """
        Register a node feature and allocate storage.

        Parameters
        ----------
        name : str, optional
            Feature name. Required if `specs` is None.
        dim : int, optional
            Feature dimension. 1 for scalar, >1 for vector. Required if `specs` is None.
        dtype : numpy.dtype, default numpy.float32
            Storage dtype.
        init_value : float or int, default 0.0
            Initial fill value for existing nodes.
        specs : list of dict, optional
            Bulk registration. Each dict requires `name` and `dim` and may
            include `dtype` and `init_value`.

        Raises
        ------
        ValueError
            If arguments are missing, duplicated, or invalid.

        Notes
        -----
        When `dim==1` the array has shape `(n,)`. Otherwise `(n, dim)`.
        Existing nodes are filled with `init_value`.
        """
        if specs is not None:
            for spec in specs:
                self.register_node_feature(
                    name=spec["name"],
                    dim=int(spec["dim"]),
                    dtype=spec.get("dtype", np.float32),
                    init_value=spec.get("init_value", 0.0),
                )
            return

        if name is None or dim is None:
            raise ValueError("Provide (name, dim) or specs=[...].")
        if name in self.node_features:
            raise ValueError(f"Node feature '{name}' already registered.")
        if dim < 1:
            raise ValueError("dim must be >= 1")

        if dim == 1:
            arr = np.full((self.n,), init_value, dtype=dtype)
        else:
            arr = np.full((self.n, dim), init_value, dtype=dtype)
        self.node_features[name] = arr
        self.node_schema[name] = {"dim": 1 if dim == 1 else int(dim), "dtype": str(dtype)}

    def register_edge_feature(
        self,
        name: Optional[str] = None,
        dim: int = 1,
        dtype: np.dtype = np.float32,
        init_value: Union[float, int] = 0.0,
        specs: Optional[List[dict]] = None,
    ) -> None:
        """
        Register an edge feature and allocate storage.

        Parameters
        ----------
        name : str, optional
            Feature name. Required if `specs` is None.
        dim : int, default 1
            Channel count per edge. Must be >= 1.
        dtype : numpy.dtype, default numpy.float32
            Storage dtype.
        init_value : float or int, default 0.0
            Initial fill value for existing edges.
        specs : list of dict, optional
            Bulk registration. Each dict may include `name`, `dim`, `dtype`, `init_value`.

        Raises
        ------
        ValueError
            If arguments are missing, duplicated, or invalid.

        Notes
        -----
        Storage shape is `(n, n, dim)`.
        """
        if specs is not None:
            for spec in specs:
                self.register_edge_feature(
                    name=spec["name"],
                    dim=int(spec.get("dim", 1)),
                    dtype=spec.get("dtype", np.float32),
                    init_value=spec.get("init_value", 0.0),
                )
            return

        if name is None:
            raise ValueError("Provide (name) or specs=[...].")
        if name in self.edge_features:
            raise ValueError(f"Edge feature '{name}' already registered.")
        if dim < 1:
            raise ValueError("dim must be >= 1")

        arr = np.full((self.n, self.n, dim), init_value, dtype=dtype)
        self.edge_features[name] = arr
        self.edge_schema[name] = {"dim": int(dim), "dtype": str(dtype)}

    # ---------- public mutators ----------

    def add_node(
        self,
        node_feat: Optional[Dict[str, Array]] = None,
        edge_defaults_for_new_node: Optional[Dict[str, Union[float, Array]]] = None,
    ) -> int:
        """
        Add a single node and grow all storages.

        Parameters
        ----------
        node_feat : dict[str, ndarray], optional
            Values for the new node. Scalars for dim==1 or 1D arrays of shape `(F,)`.
        edge_defaults_for_new_node : dict[str, float or ndarray], optional
            Per-edge-feature defaults for edges touching the new node. Scalars
            broadcast to all channels; otherwise `(E,)`.

        Returns
        -------
        int
            Index of the new node.

        Raises
        ------
        ValueError
            If provided feature shapes are incompatible.

        Notes
        -----
        Updates both the new row and column in registered edge features.
        """
        node_feat = node_feat or {}
        edge_defaults_for_new_node = edge_defaults_for_new_node or {}

        old_n = self.n
        new_n = old_n + 1

        # Grow adjacency
        self.adj = self._pad_adj(self.adj, new_n)

        # Grow node features and set new row
        for name, mat in list(self.node_features.items()):
            dims = self._node_dims(mat)
            mat = self._pad_node(mat, new_n)

            if name in node_feat:
                val = np.asarray(node_feat[name], dtype=mat.dtype)
                if dims.F == 1 or mat.ndim == 1:
                    if val.shape == (1,):
                        val = val.reshape(())
                    if val.shape not in [(), (0,)] and val.shape != ():
                        self._ensure_shape(val, (), f"node_feat['{name}']")
                    mat[old_n] = val if val.shape == () else val.item()
                else:
                    self._ensure_shape(val, (dims.F,), f"node_feat['{name}']")
                    mat[old_n, :] = val

            self.node_features[name] = mat

        # Grow edge features and initialize touching edges
        for name, mat in list(self.edge_features.items()):
            dims = self._edge_dims(mat)
            mat = self._pad_edge(mat, new_n)

            default = edge_defaults_for_new_node.get(name, 0.0)
            default = np.asarray(default, dtype=mat.dtype)
            if default.shape == ():
                default = np.full((dims.E,), default, dtype=mat.dtype)
            self._ensure_shape(default, (dims.E,), f"edge_defaults_for_new_node['{name}']")

            mat[old_n, :new_n, :] = default
            mat[:new_n, old_n, :] = default
            self.edge_features[name] = mat

        self.n = new_n
        return old_n

    def _assert_self_loop_policy(self, u: int, v: int):
        """Raise if self-loops are disabled and `u == v`."""
        if not self.self_loop_allowed and u == v:
            raise ValueError("Self-loops are disabled. Set `self_loop_allowed=True`.")

    def add_edge(
        self,
        u: int,
        v: int,
        weight: Union[float, int] = 1.0,
        edge_feat: Optional[Dict[str, Array]] = None,
        set_symmetric: Optional[bool] = None,
    ) -> None:
        """
        Add or overwrite a single edge.

        Parameters
        ----------
        u, v : int
            Source and destination indices.
        weight : float, default 1.0
            Edge weight written to `adj[u, v]`.
        edge_feat : dict[str, ndarray], optional
            Per-edge-feature values of shape `(E,)`.
        set_symmetric : bool, optional
            If True, mirror to `(v, u)`. If None, uses `not directed`.

        Raises
        ------
        IndexError
            If `u` or `v` is out of bounds.
        ValueError
            If self-loops are disallowed or feature shapes mismatch.
        """
        self._ensure_in_bounds(u)
        self._ensure_in_bounds(v)
        self._assert_self_loop_policy(u, v)
        sym = (not self.directed) if set_symmetric is None else bool(set_symmetric)

        self.adj[u, v] = weight
        if sym:
            self.adj[v, u] = weight

        edge_feat = edge_feat or {}
        for name, mat in list(self.edge_features.items()):
            dims = self._edge_dims(mat)
            if name in edge_feat and edge_feat[name] is not None:
                val = np.asarray(edge_feat[name], dtype=mat.dtype)
                self._ensure_shape(val, (dims.E,), f"edge_feat['{name}']")
                mat[u, v, :] = val
                if sym:
                    mat[v, u, :] = val
                self.edge_features[name] = mat

    def remove_edge(
        self,
        u: int,
        v: int,
        *,
        clear_features: bool = True,
        set_symmetric: Optional[bool] = None,
    ) -> None:
        """
        Remove an edge and optionally clear its features.

        Parameters
        ----------
        u, v : int
            Source and destination indices.
        clear_features : bool, default True
            If True, zero all registered edge-feature channels at `(u, v)`.
        set_symmetric : bool, optional
            If True, also remove `(v, u)`. If None, uses `not directed`.

        Raises
        ------
        IndexError
            If `u` or `v` is out of bounds.
        """
        self._ensure_in_bounds(u)
        self._ensure_in_bounds(v)
        sym = (not self.directed) if set_symmetric is None else bool(set_symmetric)

        self.adj[u, v] = 0.0
        if sym:
            self.adj[v, u] = 0.0

        if not clear_features:
            return

        for name, mat in list(self.edge_features.items()):
            E = mat.shape[2]
            zeros = np.zeros((E,), dtype=mat.dtype)
            mat[u, v, :] = zeros
            if sym:
                mat[v, u, :] = zeros
            self.edge_features[name] = mat

    def remove_nodes(self, indices: Union[List[int], Tuple[int, ...], Array]) -> Array:
        """
        Remove multiple nodes and compact all storages.

        Parameters
        ----------
        indices : array-like of int
            Node indices to remove. Duplicates and order are ignored.

        Returns
        -------
        numpy.ndarray
            Mapping array of shape `(old_n,)`, dtype int32. Kept nodes map to
            new indices, removed nodes map to `-1`. Returns `array([], int32)`
            if `indices` is empty.

        Raises
        ------
        IndexError
            If any index is out of bounds.

        Notes
        -----
        This operation is O(n^2) due to dense compaction.
        """
        idx = np.asarray(indices, dtype=np.int32).ravel()
        if idx.size == 0:
            return np.empty(0, dtype=np.int32)

        idx = np.unique(idx)
        if int(idx[0]) < 0 or int(idx[-1]) >= self.n:
            raise IndexError("remove_nodes: index out of bounds.")

        keep = np.ones((self.n,), dtype=bool)
        keep[idx] = False

        seq = np.cumsum(keep.astype(np.int32)) - 1
        old_to_new_arr = np.where(keep, seq, -np.ones_like(seq, dtype=np.int32)).astype(np.int32)

        self.adj = self.adj[keep][:, keep]

        for name, mat in list(self.node_features.items()):
            self.node_features[name] = mat[keep] if mat.ndim == 1 else mat[keep, :]

        for name, mat in list(self.edge_features.items()):
            self.edge_features[name] = mat[keep][:, keep, :]

        self.n = int(keep.sum())
        return old_to_new_arr

    # ---------- accessors / updates ----------

    def get_node_feat(self, name: str) -> Array:
        """
        Return the full storage for a node feature.

        Parameters
        ----------
        name : str
            Feature name.

        Returns
        -------
        ndarray
            Shape `(n,)` or `(n, F)`.

        Raises
        ------
        KeyError
            If the feature is unknown.
        """
        if name not in self.node_features:
            raise KeyError(f"Unknown node feature '{name}'.")
        return self.node_features[name]

    def set_node_feat(self, name: str, values: Array) -> None:
        """
        Overwrite a node feature storage.

        Parameters
        ----------
        name : str
            Feature name.
        values : ndarray
            Shape `(n,)` for dim==1 or `(n, F)` for dim>1. Dtype is cast to storage dtype.

        Raises
        ------
        KeyError
            If the feature is unknown.
        ValueError
            If `values` has incompatible shape.
        """
        if name not in self.node_features:
            raise KeyError(f"Unknown node feature '{name}'.")
        mat = self.node_features[name]
        values = np.asarray(values, dtype=mat.dtype)
        if mat.ndim == 1:
            self._ensure_shape(values, (self.n,), f"values for '{name}'")
        else:
            self._ensure_shape(values, (self.n, mat.shape[1]), f"values for '{name}'")
        self.node_features[name] = values

    def set_edge_feat(
        self,
        name: str,
        u: int,
        v: int,
        value: Array,
        *,
        set_symmetric: Optional[bool] = None,
    ) -> None:
        """
        Set edge-feature channels for a single edge.

        Parameters
        ----------
        name : str
            Edge feature name.
        u, v : int
            Edge endpoints.
        value : ndarray
            Shape `(E,)` with `E` equal to the registered channel count.
        set_symmetric : bool, optional
            If True, mirror to `(v, u)`. If None, uses `not directed`.

        Raises
        ------
        KeyError
            If the feature is unknown.
        IndexError
            If indices are out of bounds.
        ValueError
            If `value` has wrong shape.
        """
        self._ensure_in_bounds(u)
        self._ensure_in_bounds(v)
        if name not in self.edge_features:
            raise KeyError(f"Unknown edge feature '{name}'.")
        mat = self.edge_features[name]
        dims = self._edge_dims(mat)
        val = np.asarray(value, dtype=mat.dtype)
        self._ensure_shape(val, (dims.E,), f"value for edge feature '{name}'")
        sym = (not self.directed) if set_symmetric is None else bool(set_symmetric)
        mat[u, v, :] = val
        if sym:
            mat[v, u, :] = val
        self.edge_features[name] = mat

    # ---------- bulk ops ----------

    def add_nodes(
        self,
        k: int,
        node_feat: Optional[Dict[str, Array]] = None,
        edge_defaults: Optional[Dict[str, Union[float, Array]]] = None,
    ) -> Tuple[int, int]:
        """
        Add `k` nodes.

        Parameters
        ----------
        k : int
            Number of nodes to add. Must be >= 0.
        node_feat : dict[str, ndarray], optional
            Values for each new node, applied identically to all newly added nodes.
        edge_defaults : dict[str, float or ndarray], optional
            Defaults for edges touching each new node.

        Returns
        -------
        (int, int)
            `(start, end)` index range of the added nodes.

        Raises
        ------
        ValueError
            If `k < 0`.
        """
        if k < 0:
            raise ValueError("k must be >= 0")
        start = self.n
        for _ in range(k):
            self.add_node(node_feat=node_feat, edge_defaults_for_new_node=edge_defaults)
        return start, self.n

    def add_edges(
        self,
        src: Array,
        dst: Array,
        *,
        weights: Union[float, Array] = 1.0,
        edge_feat: Optional[Dict[str, Array]] = None,
        set_symmetric: Optional[bool] = None,
    ) -> None:
        """
        Vectorized edge insertion.

        Parameters
        ----------
        src, dst : array-like
            Source and destination indices, same length.
        weights : float or array-like, default 1.0
            Scalar or per-edge weights.
        edge_feat : dict[str, ndarray], optional
            For each feature name, either a 1D array `(E,)` broadcast to all edges
            or a 2D array `(m, E)` with per-edge rows.
        set_symmetric : bool, optional
            If True, mirror each edge.

        Raises
        ------
        ValueError
            If inputs have mismatched lengths or feature batch shapes.
        IndexError
            If any index is out of bounds.
        """
        src = np.asarray(src).ravel()
        dst = np.asarray(dst).ravel()
        if src.size != dst.size:
            raise ValueError("src and dst must have same size.")
        if isinstance(weights, (int, float)):
            W = None
            w_scalar = float(weights)
        else:
            W = np.asarray(weights).ravel()
            if W.size != src.size:
                raise ValueError("weights must be scalar or match number of edges.")
            w_scalar = None

        ef_norm: Dict[str, Tuple[Array, bool]] = {}
        if edge_feat:
            for k, v in edge_feat.items():
                arr = np.asarray(v)
                if arr.ndim == 1:
                    ef_norm[k] = (arr, True)
                else:
                    if arr.shape[0] != src.size:
                        raise ValueError(f"edge_feat['{k}'] first dim must match num edges.")
                    ef_norm[k] = (arr, False)

        for i in range(int(src.size)):
            wi = w_scalar if W is None else float(W[i])
            per_edge = None
            if ef_norm:
                per_edge = {}
                for k, (arr, is_bcast) in ef_norm.items():
                    per_edge[k] = arr if is_bcast else arr[i]
            self.add_edge(int(src[i]), int(dst[i]), weight=wi,
                          edge_feat=per_edge, set_symmetric=set_symmetric)

    def remove_edges(
        self,
        src: Array,
        dst: Array,
        *,
        set_symmetric: Optional[bool] = None,
        clear_features: bool = True,
    ) -> None:
        """
        Vectorized edge removal.

        Parameters
        ----------
        src, dst : array-like
            Source and destination indices, same length.
        set_symmetric : bool, optional
            If True, also remove mirrored edges.
        clear_features : bool, default True
            If True, zero edge features on removed edges.

        Raises
        ------
        ValueError
            If inputs have mismatched lengths.
        IndexError
            If any index is out of bounds.
        """
        src = np.asarray(src).ravel()
        dst = np.asarray(dst).ravel()
        if src.size != dst.size:
            raise ValueError("src and dst must have same size.")
        for i in range(int(src.size)):
            self.remove_edge(int(src[i]), int(dst[i]),
                             set_symmetric=set_symmetric,
                             clear_features=clear_features)

    # ---------- utilities ----------

    def neighbors(self, i: int, mode: str = "any") -> Array:
        """
        Return neighbor indices of a node.

        Parameters
        ----------
        i : int
            Node index.
        mode : {'out', 'in', 'any'}, default 'any'
            Neighbor type. 'any' returns the union of in- and out-neighbors.

        Returns
        -------
        ndarray
            Sorted unique neighbor indices, dtype int32.

        Raises
        ------
        IndexError
            If `i` is out of bounds.
        ValueError
            If `mode` is invalid.
        """
        self._ensure_in_bounds(i)
        if mode not in {"out", "in", "any"}:
            raise ValueError("mode must be 'out', 'in', or 'any'")

        nz = self.adj != 0
        if mode == "out":
            return np.nonzero(nz[i, :])[0].astype(np.int32)
        if mode == "in":
            return np.nonzero(nz[:, i])[0].astype(np.int32)

        out_idx = np.nonzero(nz[i, :])[0]
        in_idx = np.nonzero(nz[:, i])[0]
        if out_idx.size + in_idx.size == 0:
            return np.array([], dtype=np.int32)
        return np.unique(np.concatenate([out_idx, in_idx])).astype(np.int32)

    def successors(self, i: int) -> Array:
        """
        Return out-neighbors of a node.

        Parameters
        ----------
        i : int
            Node index.

        Returns
        -------
        ndarray
            Indices of nodes `j` with `adj[i, j] != 0`, dtype int32.
        """
        return self.neighbors(i, "out")

    def predecessors(self, i: int) -> Array:
        """
        Return in-neighbors of a node.

        Parameters
        ----------
        i : int
            Node index.

        Returns
        -------
        ndarray
            Indices of nodes `j` with `adj[j, i] != 0`, dtype int32.
        """
        return self.neighbors(i, "in")

    # ---------- degree helpers ----------

    def out_degree(self) -> Array:
        """
        Compute out-degree per node.

        Returns
        -------
        ndarray
            Shape `(n,)`, number of nonzero entries per row of `adj`.
        """
        return np.sum(self.adj != 0, axis=1)

    def in_degree(self) -> Array:
        """
        Compute in-degree per node.

        Returns
        -------
        ndarray
            Shape `(n,)`, number of nonzero entries per column of `adj`.
        """
        return np.sum(self.adj != 0, axis=0)

    def degree(self) -> Array:
        """
        Compute total degree per node.

        Returns
        -------
        ndarray
            Shape `(n,)`, `out_degree + in_degree`.
        """
        return self.out_degree() + self.in_degree()

    # ---------- validation ----------

    def assert_invariants(self) -> None:
        """
        Validate internal storage shapes.

        Checks
        ------
        - `adj` is square and at least `(n, n)`.
        - Node features are 1D or 2D and have first dim >= `n`.
        - Edge features are 3D `(n, n, E)` with square leading dims and first dim >= `n`.

        Raises
        ------
        AssertionError
            If any invariant is violated.
        """
        if self.adj.shape[0] != self.adj.shape[1]:
            raise AssertionError(f"adj not square: {self.adj.shape}")
        if self.adj.shape[0] < self.n:
            raise AssertionError(f"adj too small {self.adj.shape} < ({self.n},{self.n})")

        for name, mat in self.node_features.items():
            if mat.ndim not in (1, 2):
                raise AssertionError(f"node feature '{name}' ndim {mat.ndim} not in {{1,2}}")
            if mat.shape[0] < self.n:
                raise AssertionError(f"node feature '{name}' dim0 {mat.shape[0]} < {self.n}")

        for name, mat in self.edge_features.items():
            if mat.ndim != 3:
                raise AssertionError(f"edge feature '{name}' ndim {mat.ndim} != 3")
            if mat.shape[0] != mat.shape[1]:
                raise AssertionError(f"edge feature '{name}' not square in leading dims: {mat.shape[:2]}")
            if mat.shape[0] < self.n:
                raise AssertionError(f"edge feature '{name}' too small {mat.shape[:2]} < ({self.n},{self.n})")

    # ---------- serialization ----------

    def to_npz(self, path: str) -> None:
        """
        Save graph to a compressed NPZ file.

        Parameters
        ----------
        path : str
            Output path.

        Notes
        -----
        Stores:
        - 'n', 'adj'
        - 'node__{name}' for each node feature
        - 'edge__{name}' for each edge feature
        """
        data = {"n": np.array(self.n), "adj": np.asarray(self.adj)}
        for k, v in self.node_features.items():
            data[f"node__{k}"] = np.asarray(v)
        for k, v in self.edge_features.items():
            data[f"edge__{k}"] = np.asarray(v)
        np.savez_compressed(path, **data)

    @staticmethod
    def from_npz(path: str) -> "Graph":
        """
        Load graph from a compressed NPZ file.

        Parameters
        ----------
        path : str
            Input path produced by `to_npz`.

        Returns
        -------
        Graph
            Reconstructed graph.

        Raises
        ------
        AssertionError
            If loaded arrays violate invariants.
        """
        z = np.load(path, allow_pickle=False)
        g = Graph()
        g.n = int(z["n"])
        g.adj = np.asarray(z["adj"])
        for key in z.files:
            if key.startswith("node__"):
                name = key.split("__", 1)[1]
                arr = np.asarray(z[key])
                g.node_features[name] = arr
                g.node_schema[name] = {"dim": 1 if arr.ndim == 1 else int(arr.shape[1]),
                                       "dtype": str(arr.dtype)}
            elif key.startswith("edge__"):
                name = key.split("__", 1)[1]
                arr = np.asarray(z[key])
                g.edge_features[name] = arr
                g.edge_schema[name] = {"dim": int(arr.shape[2]), "dtype": str(arr.dtype)}
        g.assert_invariants()
        return g

    # ---------- interoperability & capacity ----------

    def to_edge_list(self) -> Array:
        """
        Return the list of present edges.

        Returns
        -------
        ndarray
            Shape `(m, 2)` int32 array of `(u, v)` pairs. Empty array with shape
            `(0, 2)` if no edges exist.

        Notes
        -----
        Scans the dense adjacency, O(n^2).
        """
        nz0, nz1 = np.nonzero(self.adj != 0)
        if nz0.size == 0:
            return np.zeros((0, 2), dtype=np.int32)
        return np.stack([nz0, nz1], axis=1).astype(np.int32)

    @staticmethod
    def from_edge_list(n: int, edges: Array, *, directed: bool = True) -> "Graph":
        """
        Build a graph from an edge list.

        Parameters
        ----------
        n : int
            Number of nodes.
        edges : ndarray
            Shape `(m, 2)` int-like array of directed pairs.
        directed : bool, default True
            If False, edges are mirrored.

        Returns
        -------
        Graph

        Raises
        ------
        ValueError
            If `edges` does not have shape `(m, 2)`.
        """
        g = Graph(directed=directed)
        g.add_nodes(n)
        if edges.size:
            edges = np.asarray(edges)
            if edges.ndim != 2 or edges.shape[1] != 2:
                raise ValueError("edges must have shape (m, 2).")
            g.add_edges(edges[:, 0], edges[:, 1], set_symmetric=not directed)
        return g

    def reserve(self, capacity_n: int) -> None:
        """
        Grow internal storages to at least `capacity_n`.

        Parameters
        ----------
        capacity_n : int
            Target capacity in nodes. Must be >= 0.

        Notes
        -----
        No shrink occurs. Does not change `n`.
        """
        if capacity_n < 0:
            raise ValueError("capacity_n must be >= 0")
        if capacity_n <= self.n:
            return
        self.adj = self._pad_adj(self.adj, capacity_n)
        for name, mat in list(self.node_features.items()):
            self.node_features[name] = self._pad_node(mat, capacity_n)
        for name, mat in list(self.edge_features.items()):
            self.edge_features[name] = self._pad_edge(mat, capacity_n)

    # ---------- helpers ----------

    @staticmethod
    def _pad_adj(adj: Array, new_n: int) -> Array:
        """Pad adjacency to `(new_n, new_n)` with zeros."""
        if adj.shape == (new_n, new_n):
            return adj
        dn0 = new_n - adj.shape[0]
        dn1 = new_n - adj.shape[1]
        return np.pad(adj, ((0, dn0), (0, dn1)), mode="constant", constant_values=0)

    @staticmethod
    def _pad_node(mat: Array, new_n: int) -> Array:
        """Pad node-feature storage to length `new_n` along axis 0."""
        if mat.ndim == 1:
            n = mat.shape[0]
            return mat if n == new_n else np.pad(mat, (0, new_n - n), mode="constant", constant_values=0)
        if mat.ndim == 2:
            n, _ = mat.shape
            return mat if n == new_n else np.pad(mat, ((0, new_n - n), (0, 0)), mode="constant", constant_values=0)
        raise ValueError(f"Node feature must be 1-D or 2-D; got {mat.shape}")

    @staticmethod
    def _pad_edge(mat: Array, new_n: int) -> Array:
        """Pad edge-feature storage to `(new_n, new_n, E)`."""
        if mat.ndim != 3:
            raise ValueError(f"Edge feature must be 3-D (n, n, E); got {mat.shape}")
        n, _, _ = mat.shape
        if n == new_n:
            return mat
        dn = new_n - n
        return np.pad(mat, ((0, dn), (0, dn), (0, 0)), mode="constant", constant_values=0)

    @staticmethod
    def _ensure_shape(arr: Array, expected: Tuple[int, ...], label: str):
        """Raise if `arr.shape != expected`."""
        if arr.shape != expected:
            raise ValueError(f"{label} must have shape {expected}, got {arr.shape}")

    def _ensure_in_bounds(self, i: int):
        """Raise if node index `i` is out of bounds."""
        if not (0 <= int(i) < self.n):
            raise IndexError(f"Node index {i} out of bounds (n={self.n}).")

    def _node_dims(self, mat: Array) -> _Dims:
        """Return `_Dims(n, F)` for a node-feature storage."""
        if mat.ndim == 1:
            return _Dims(n=mat.shape[0], F=1)
        if mat.ndim == 2:
            n, F = mat.shape
            return _Dims(n=n, F=F)
        raise ValueError(f"Node feature storage must be 1-D or 2-D; got {mat.shape}")

    @staticmethod
    def _edge_dims(mat: Array) -> _Dims:
        """Return `_Dims(n, E)` for an edge-feature storage."""
        if mat.ndim != 3:
            raise ValueError(f"Edge feature storage must be 3-D (n, n, E); got {mat.shape}")
        n0, n1, E = mat.shape
        if n0 != n1:
            raise ValueError(f"Edge feature first two dims must match (n,n,E); got {(n0, n1, E)}")
        return _Dims(n=n0, E=E)
