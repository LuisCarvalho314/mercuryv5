# mercury/memory/state.py
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

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
        return self.gs.node_features["activations"]

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


def init_mem(ss: SensoryState, length: int = 5) -> MemoryState:
    """Create S strips of length L with edges t->t-1. S = ss.gs.n."""
    L = int(length)
    if L <= 0:
        raise ValueError("length must be >= 1")

    g = Graph(directed=True)
    g.register_node_feature("activation", dim=1)

    S = int(ss.gs.n)  # sensory node count
    for _ in range(S):
        base = g.add_node()
        for _ in range(1, L):
            g.add_node()
        for t in range(1, L):
            g.add_edge(base + t, base + t - 1, weight=1.0)

    return MemoryState(gs=g, length=L, sensory_n_nodes=S)


def add_memory(ms: MemoryState, memory: Array) -> MemoryState:
    """Write activations into the last node of each strip. memory.shape == (S,)."""
    g, L = ms.gs, int(ms.length)
    S = g.n // L
    mem = np.ravel(np.asarray(memory, dtype=np.float32))
    if mem.shape[0] != S:
        raise ValueError(f"len(memory)={mem.shape[0]} must equal S={S} (gs.n//length)")

    idx = np.arange(S, dtype=np.int32) * L + (L - 1)
    act = g.node_features.get("activation")
    if act is None or act.shape != (g.n,):
        raise KeyError("Memory graph missing node feature 'activation' with shape (n,)")

    new_act = act.copy()
    new_act[idx] = mem
    g.set_node_feat("activation", new_act)
    return MemoryState(gs=g, length=L, sensory_n_nodes=ms.sensory_n_nodes)


def update_memory(ms: MemoryState) -> MemoryState:
    """One step shift along edges: act' = act @ adj."""
    g = ms.gs
    act = g.node_features["activation"].astype(np.float32, copy=False)  # (n,)
    A = g.adj.astype(np.float32, copy=False)                            # (n,n)
    g.set_node_feat("activation", act @ A)
    return MemoryState(gs=g, length=ms.length, sensory_n_nodes=ms.sensory_n_nodes)

