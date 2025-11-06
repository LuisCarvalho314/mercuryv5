from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Optional, Union, Protocol, runtime_checkable

import numpy as np

from mercury.graph.core import Graph  # replace with actual import

Array = np.ndarray


@runtime_checkable
class HasMaxAge(Protocol):
    """
    Protocol for maintenance config.

    Attributes
    ----------
    max_age : int
        Threshold at or above which an edge is pruned.
    """
    max_age: int


@dataclass(frozen=True)
class MaintenanceParams:
    """
    Hyperparameters for edge-age maintenance.

    Parameters
    ----------
    max_age : int, default 18
        Edges with age >= `max_age` are removed by `prune_old_edges`.
    """
    max_age: int = 18


from typing import Iterable

def age_maintenance(
    u: int,
    v: int,
    g: Graph,
    p: HasMaxAge,
    exclude: Optional[Iterable[int]] = None,
) -> Tuple[Graph, Array]:
    """
    Age edges from node `u`, prune old edges, drop isolated nodes.

    Pipeline:
    1) update_ages(u, v, g)
    2) prune_old_edges(g, p)
    3) remove_lonely_nodes(g, exclude=keep_nodes)

    keep_nodes must be an ndarray[int32] of node indices that are immune
    to orphan-pruning.

    Returns:
        (g_after, old_to_new_mapping)
    """
    assert "age" in g.edge_features, "Graph has no 'age' edge feature"

    update_ages(u, v, g)
    prune_old_edges(g, p)

    # normalize exclude
    if exclude is None:
        base_keep = set()
    elif isinstance(exclude, (int, np.integer)):
        base_keep = {int(exclude)}
    else:
        base_keep = {int(x) for x in exclude}

    # also force-keep v (target of the current transition)
    base_keep.add(int(v))

    keep_nodes = np.asarray(sorted(base_keep), dtype=np.int32)

    return remove_lonely_nodes(g, exclude=keep_nodes)



def update_ages(u: int, v: int, g: Graph) -> None:
    """
    Increment ages on row `u` except column `v`.

    Operation
    ---------
    For each `j` with `g.adj[u, j] != 0` and `j != v`, perform
    `g.edge_features['age'][u, j, :] += 1`.

    Parameters
    ----------
    u : int
        Row index. Must satisfy `0 <= u < g.n`.
    v : int
        Column index to exclude. Must satisfy `0 <= v < g.n`.
    g : Graph
        Graph holding an `"age"` edge feature `(n, n, E)`.

    Returns
    -------
    None
        Modifies `g` in place.

    Raises
    ------
    IndexError
        If `u` or `v` is out of bounds.
    KeyError
        If `"age"` is not registered.
    """
    g._ensure_in_bounds(u); g._ensure_in_bounds(v)
    age = g.edge_features["age"]        # (n, n, E)
    n = g.n
    mask_row = (g.adj[u] != 0)          # (n,)
    if 0 <= v < n:
        mask_row[v] = False
    if not np.any(mask_row):
        return
    age[u, mask_row, :] = age[u, mask_row, :] + 1  # broadcast over E


def prune_old_edges(g: Graph, p: HasMaxAge) -> None:
    """
    Remove edges whose age is at or above the threshold.

    Policy
    ------
    If `age[i, j, 0] >= p.max_age`:
      - set `g.adj[i, j] = 0.0`
      - set `g.edge_features['age'][i, j, :] = 0`

    Parameters
    ----------
    g : Graph
        Graph with `"age"` edge feature `(n, n, E)`, `E >= 1`.
    p : HasMaxAge
        Configuration with `max_age` threshold.

    Returns
    -------
    None
        Modifies `g` in place.

    Raises
    ------
    KeyError
        If `"age"` is not registered.
    """
    age = g.edge_features["age"]               # (n, n, E)
    prune_mask = age[..., 0] >= int(p.max_age) # (n, n)
    if not np.any(prune_mask):
        return
    g.adj[prune_mask] = 0.0
    age[prune_mask, :] = 0
    g.edge_features["age"] = age


def remove_lonely_nodes(
    g: Graph,
    *,
    exclude: Optional[Union[int, Array]] = None,
    min_nodes: int = 1,
) -> Tuple[Graph, Array]:
    """
    Remove nodes with zero in-degree, optionally keeping some indices.

    Parameters
    ----------
    g : Graph
        Input graph. Modified in place.
    exclude : int or ndarray, optional
        Node index or indices to force-keep even if in-degree is zero.
    min_nodes : int, default 1
        Minimum number of nodes to keep in the graph. If removing all lonely
        nodes would drop below this, only a prefix of lonely nodes is removed.

    Returns
    -------
    (Graph, ndarray)
        The same graph instance and an `old_to_new` mapping `(old_n,)` int32.
        Kept nodes map to new indices; removed nodes map to `-1`.
        If nothing is removed, returns the identity mapping `arange(n)`.

    Notes
    -----
    Uses `g.in_degree()` for loneliness. Out-degree is ignored here.
    """
    indeg = g.in_degree().astype(np.int32)
    lonely = indeg == 0

    if exclude is not None and g.n:
        ex = np.asarray(exclude, dtype=np.int32).ravel()
        ex = ex[(ex >= 0) & (ex < g.n)]
        if ex.size:
            lonely[ex] = False

    idx = np.nonzero(lonely)[0]

    if min_nodes is not None:
        max_drop = max(0, g.n - int(min_nodes))
        if idx.size > max_drop:
            idx = idx[:max_drop]

    if idx.size == 0:
        # identity mapping
        return g, np.arange(g.n, dtype=np.int32)

    mapping = g.remove_nodes(idx)
    return g, mapping
