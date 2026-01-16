# mercury/memory/state.py
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from typing import List
from mercury.graph.core import Graph
from mercury.sensory.state import SensoryState

Array = np.ndarray


@dataclass(frozen=True)
class MemoryState:
    gs: Graph
    length: int                 # strip length L
    sensory_n_nodes: int        # number of sensory nodes at init time (S)

    @property
    def activations(self) -> Array:
        return self.gs.node_features["activation"]

def mem_id(s: int, t: int, length: int) -> int:
    return s * length + t


def activations_at_t(ms: MemoryState, t: int) -> np.ndarray:
    """Return activations at strip time t, shape (S,)."""
    L = int(ms.length)
    if not (0 <= t < L):
        raise ValueError(f"t={t} out of range [0,{L-1}]")
    n = ms.gs.n
    S = n // L
    idx = np.arange(S, dtype=np.int32) * L + t
    act = ms.gs.node_features["activation"]  # (S*L,)
    return act[idx].copy()


# def init_mem(ss: SensoryState, length: int = 5) -> MemoryState:
#     """Create S strips of length L with edges t->t-1. S = ss.gs.n."""
#     L = int(length)
#     if L <= 0:
#         raise ValueError("length must be >= 1")
#
#     g = Graph(directed=True)
#     g.register_node_feature("activation", dim=1)
#
#     S = int(ss.gs.n)  # sensory node count
#     for _ in range(S):
#         base = g.add_node()
#         for _ in range(1, L):
#             g.add_node()
#         for t in range(1, L):
#             g.add_edge(base + t, base + t - 1, weight=1.0)
#
#     return MemoryState(gs=g, length=L, sensory_n_nodes=S)

def init_mem(n: int, length: int = 5) -> MemoryState:
    """Create S strips of length L with edges t->t+1. S = ss.gs.n."""
    strip_length = int(length)
    if strip_length <= 0:
        raise ValueError("length must be >= 1")

    g = Graph(directed=True)
    g.register_node_feature("activation", dim=1)

    sensory_node_count = n  # number of sensory nodes
    for _ in range(sensory_node_count):
        base = g.add_node()
        for _ in range(1, strip_length):
            g.add_node()
        for t in range(0, strip_length - 1):
            # g.add_edge(base + t, base + t + 1, weight=1.0)
            g.add_edge(base + t, base + t + 1, weight=1)


    return MemoryState(gs=g, length=strip_length, sensory_n_nodes=sensory_node_count)


# def add_memory(ms: MemoryState, memory: Array) -> MemoryState:
#     """Write activations into the last node of each strip. memory.shape == (S,)."""
#     g, L = ms.gs, int(ms.length)
#     S = g.n // L
#     mem = np.ravel(np.asarray(memory, dtype=np.float32))
#     if mem.shape[0] != S:
#         raise ValueError(f"len(memory)={mem.shape[0]} must equal S={S} (gs.n//length)")
#
#     idx = np.arange(S, dtype=np.int32) * L + (L - 1)
#     act = g.node_features.get("activation")
#     if act is None or act.shape != (g.n,):
#         raise KeyError("Memory graph missing node feature 'activation' with shape (n,)")
#
#     new_act = act.copy()
#     new_act[idx] = mem
#     g.set_node_feat("activation", new_act)
#     return MemoryState(gs=g, length=L, sensory_n_nodes=ms.sensory_n_nodes)

def add_memory(ms: MemoryState, memory: Array) -> MemoryState:
    """Write activations into the first node (t=0) of each strip. memory.shape == (S,)."""
    g, strip_length = ms.gs, int(ms.length)
    sensory_count = g.n // strip_length

    mem = np.ravel(np.asarray(memory, dtype=np.float32))
    if mem.shape[0] != sensory_count:
        raise ValueError(f"len(memory)={mem.shape[0]} must equal S={sensory_count} (gs.n//length)")

    # Indices of t=0 nodes for each strip
    idx = np.arange(sensory_count, dtype=np.int32) * strip_length

    act = g.node_features.get("activation")
    if act is None or act.shape != (g.n,):
        raise KeyError("Memory graph missing node feature 'activation' with shape (n,)")

    updated_activation = act.copy()
    updated_activation[idx] = mem
    g.set_node_feat("activation", updated_activation)

    return MemoryState(gs=g, length=strip_length, sensory_n_nodes=ms.sensory_n_nodes)

def update_memory(ms: MemoryState) -> MemoryState:
    """One step shift along edges: act' = act @ adj."""
    g = ms.gs
    act = g.node_features["activation"].astype(np.float32, copy=False)  # (n,)
    A = g.adj.astype(np.float32, copy=False)                            # (n,n)
    g.set_node_feat("activation", act @ A)
    return MemoryState(gs=g, length=ms.length, sensory_n_nodes=ms.sensory_n_nodes)

def memory_view_at_global_timestep(ms: MemoryState, k: int) -> MemoryState:
    """
    Re-index the memory so that the ORIGINAL column k becomes local t=0, with zero padding after the end.

    Definitions
    ----------
    Let L = ms.length and mem be the S×L matrix view of the current activation, where:
      - mem[:, 0] is the oldest timestep,
      - mem[:, L-1] is the newest timestep.

    Semantics
    ---------
    The returned state's activation (call it view) satisfies:
      view[:, 0] = mem[:, k]
      view[:, 1] = mem[:, k+1]
      ...
      view[:, L-1] = 0  (for any column that would require mem[:, t] with t ≥ L)

    Therefore:
      - k = 0   ⇒ view == mem  (no shift; original memory unchanged)
      - k = L-1 ⇒ view has only the original last column at t=0; columns 1..L-1 are zeros
      - k = L   ⇒ invalid (out of range)

    Constraints
    -----------
    - Valid k range: 0 ≤ k ≤ L-1.
    - The function is intended to be NON-DESTRUCTIVE: it must NOT overwrite ms.gs features.
      It should build a fresh Graph (same topology, copied features) with the reindexed activation
      and return a new MemoryState bound to that graph.

    Returns
    -------
    MemoryState
        A snapshot whose activation encodes the reindexed-and-padded view described above.
    """
    g_src = ms.gs
    L = int(ms.length)
    n_nodes_total = g_src.n
    if n_nodes_total % L != 0:
        raise ValueError(
            f"memory graph inconsistent: g.n={n_nodes_total} not divisible by length={L}"
        )
    S = n_nodes_total // L

    if not (0 <= k < L):
        raise ValueError(f"k={k} out of range [0,{L-1}]")

    # read source activation and reshape
    act_src_flat = g_src.node_features["activation"].astype(np.float32, copy=False)
    mem_matrix = act_src_flat.reshape(S, L)

    # build shifted+zero view
    out_matrix = np.zeros((S, L), dtype=np.float32)
    remaining_cols = L - k
    if remaining_cols > 0:
        out_matrix[:, :remaining_cols] = mem_matrix[:, k:]

    # clone graph structure without sharing activation array
    g_new = Graph(directed=True)

    # register same node features
    g_new.register_node_feature("activation", dim=1)

    # replicate nodes
    for _ in range(g_src.n):
        g_new.add_node()

    # replicate edges
    # g_src.adj is adjacency (n,n). Add edges where weight != 0
    A = g_src.adj
    nz_u, nz_v = np.nonzero(A)
    for u, v in zip(nz_u.tolist(), nz_v.tolist()):
        g_new.add_edge(u, v, weight=A[u, v])

    # set activation for the cloned graph
    g_new.set_node_feat("activation", out_matrix.reshape(S * L))

    return MemoryState(gs=g_new, length=L, sensory_n_nodes=ms.sensory_n_nodes)


import numpy as np
from typing import List

def memory_view_at_global_timestep_external_mem(
    ms: MemoryState,
    k: int,
    mem_vec: List[float] | np.ndarray,
) -> MemoryState:
    """
    Re-index like memory_view_at_global_timestep, but fill columns that would be zeros
    (i.e., columns that would require mem[:, t] with t >= L) from an external vector.

    Inputs
    ------
    ms: MemoryState with graph size n = S*L and node feature 'activation' length S*L
    k : int, 0 <= k < L
    mem_vec : array-like with either shape (S, k) or flat length S*k.
              Column 0 fills the first unknown column (new view's column L-k),
              column 1 fills the next, etc. Extra columns are ignored; missing
              columns are zero-padded.

    Returns
    -------
    MemoryState bound to a fresh graph clone with reindexed 'activation'.
    """
    g_src = ms.gs
    L = int(ms.length)
    n_nodes_total = g_src.n
    if n_nodes_total % L != 0:
        raise ValueError(
            f"memory graph inconsistent: g.n={n_nodes_total} not divisible by length={L}"
        )
    S = n_nodes_total // L

    if not (0 <= k < L):
        raise ValueError(f"k={k} out of range [0,{L-1}]")

    # source memory as S×L
    act_src_flat = g_src.node_features["activation"].astype(np.float32, copy=False)
    mem_matrix = act_src_flat.reshape(S, L)

    # base view: shift mem[:, k:] to columns 0..L-k-1
    out_matrix = np.zeros((S, L), dtype=np.float32)
    keep_cols = L - k
    if keep_cols > 0:
        out_matrix[:, :keep_cols] = mem_matrix[:, k:]

    # normalize mem_vec to shape (S, ?)
    mv = np.asarray(mem_vec, dtype=np.float32)
    if mv.ndim == 1:
        if mv.size % S != 0:
            raise ValueError(
                f"mem_vec length {mv.size} not divisible by S={S}; "
                "expected S*k elements (flat) or (S,k) matrix"
            )
        mv = mv.reshape(S, mv.size // S)
    elif mv.ndim == 2:
        if mv.shape[0] != S:
            raise ValueError(f"mem_vec first dim {mv.shape[0]} != S={S}")
    else:
        raise ValueError("mem_vec must be 1D (flat) or 2D (S, k)")

    # number of unknown columns to fill = k
    provide = min(k, mv.shape[1])
    if provide > 0:
        out_matrix[:, keep_cols:keep_cols + provide] = mv[:, :provide]
    # any shortfall remains zeros by construction

    # clone graph topology; register activation independently
    g_new = Graph(directed=True)
    g_new.register_node_feature("activation", dim=1)

    for _ in range(g_src.n):
        g_new.add_node()

    A = g_src.adj
    nz_u, nz_v = np.nonzero(A)
    for u, v in zip(nz_u.tolist(), nz_v.tolist()):
        g_new.add_edge(u, v, weight=A[u, v])

    g_new.set_node_feat("activation", out_matrix.reshape(S * L))

    return MemoryState(gs=g_new, length=L, sensory_n_nodes=ms.sensory_n_nodes)
