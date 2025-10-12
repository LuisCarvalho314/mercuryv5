from dataclasses import dataclass, replace

import jax.numpy as jnp

from mercury.graph.core import GraphState, gs_empty, gs_add_node, gs_add_edge, \
    gs_register_node_feature, gs_set_node_feat_full
from mercury.sensory.state import SensoryState

Array = jnp.ndarray


@dataclass(frozen=True)
class MemoryState:
    gs: GraphState
    length: int


def mem_id(s: int, t: int, length: int) -> int:
    return s * length + t

def activations_at_t(ms: MemoryState, t: int) -> jnp.ndarray:
    L = ms.length
    if not (0 <= t < L):
        raise ValueError(f"t={t} out of range [0,{L-1}]")
    n = ms.gs.n
    S = n // L
    idx = jnp.arange(S, dtype=jnp.int32) * L + t
    act = ms.gs.node_features["activation"]  # shape (S*L,)
    return act[idx]                           # shape (S,)


def init_mem(ss: SensoryState, length: int = 5) -> MemoryState:
    gs = gs_empty()
    gs = gs_register_node_feature(gs, "activation", 1)
    S, L = ss.gs.n, int(length)
    for _ in range(S):
        base = gs.n
        for _ in range(L):
            gs = gs_add_node(gs)
        for t in range(1, L):
            gs = gs_add_edge(gs, base + t, base + t - 1)  # (t)->(t-1)
    return MemoryState(gs, L)

def add_memory(ms: MemoryState, memory: Array) -> MemoryState:
    """Set only the last node of each strip. memory.shape == (S,), S = gs.n // length."""
    gs, L = ms.gs, ms.length
    S = gs.n // L
    mem = jnp.ravel(memory)
    if mem.shape[0] != S:
        raise ValueError(f"len(memory)={mem.shape[0]} must equal S={S} (gs.n//length)")
    idx = jnp.arange(S, dtype=jnp.int32) * L + (L - 1)
    act = gs.node_features.get("activation", jnp.zeros((gs.n,), gs.adj.dtype))
    new_act = act.at[idx].set(mem)
    return MemoryState(gs_set_node_feat_full(gs, "activation", new_act), L)

def update_memory(ms: MemoryState) -> MemoryState:
    act = ms.gs.node_features["activation"]
    A = ms.gs.adj
    return MemoryState(gs_set_node_feat_full(ms.gs, "activation", act @ A), ms.length)
