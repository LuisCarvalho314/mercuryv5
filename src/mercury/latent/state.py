# latent/state.py

from dataclasses import dataclass, replace
from typing import Any, Tuple

import numpy as np

from ..action_map.adapter import ActionMap
from mercury.graph.core import Graph, Array
from mercury.graph.maintenance import age_maintenance
from .params import LatentParams
from ..memory.state import MemoryState, mem_id
from ..sensory.state import SensoryState

Array = np.ndarray


@dataclass
class LatentState:
    g: Graph
    mapping: Array
    prev_bmu: int | None = None
    step_idx: int = 0

    @property
    def mem_adj(self) -> Array:
        return self.g.node_features["mem_adj"]

def _register_features(g: Graph, mem_len: int) -> Graph:
    """
    Register required features.

    Node features
    -------------
    activation  : (n,)
    mem_adj     : (n, mem_len)

    Edge features
    -------------
    age         : (n, n, 1)
    action      : (n, n, 1)
    """

    g.register_node_feature("activation", 1, init_value=0)
    g.register_node_feature("mem_adj", mem_len, init_value=0)
    g.register_edge_feature("age", 1, dtype=np.int32, init_value=0)
    g.register_edge_feature("action", 1, dtype=np.int32, init_value=0)
    return g


def build_initial_graph(ms: MemoryState) -> Graph:
    g = Graph(directed=True)
    _register_features(g, ms.gs.n)
    for n in np.arange(ms.sensory_n_nodes):
        g = _add_node(g, ms, n)


    return g

def init_state(ms: MemoryState) -> LatentState:
    g = build_initial_graph(ms)
    return LatentState(g, mapping = np.arange(g.n))



def _add_node(g: Graph, ms: MemoryState, u) -> Graph:
    mem_len = g.node_features["mem_adj"].shape[1]
    new_node = g.add_node(
        node_feat={
            "activation": np.array(0.0, np.float32),
            "mem_adj": np.zeros(mem_len, np.float32),
        },
        edge_defaults_for_new_node={
            "age": np.array([0], np.int32),
            "action": np.array([0], np.int32),
        },
    )
    g = _add_temporal_edge(g, u, new_node, 0, ms.length)
    return g


def _add_temporal_edge(g: Graph, u: int, v: int, t: int, L: int) -> Graph:
    """Record link from memory node (u at lag t) into latent node v."""
    n_lat = g.n
    if not (0 <= v < n_lat):
        raise ValueError(f"v={v} out of range [0,{n_lat-1}]")
    if not (0 <= t < L):
        raise ValueError(f"t={t} out of range [0,{L-1}]")

    mem_len = g.node_features["mem_adj"].shape[1]  # == ms.gs.n
    idx = mem_id(u, t, L)                          # flatten (u,t) into memory graph index
    if not (0 <= idx < mem_len):
        raise ValueError(f"mem index {idx} out of range [0,{mem_len-1}]")

    g.node_features["mem_adj"][v, idx] = np.float32(1.0)
    return g


def compute_bmu(ls: LatentState, ms: MemoryState) -> int:
    """
    Compute latent activations from memory, apply contender mask
    (t0 required + consecutive 0..k timesteps), write 'activation'
    to the graph, and return BMU index.
    """
    g = ls.g
    mem_adj = g.node_features["mem_adj"]                  # (n_latent, S*L)
    n_lat = g.n
    mem_len = mem_adj.shape[1]

    L = getattr(ms, "length", mem_len)
    if mem_len % L != 0:
        raise ValueError(f"mem_len={mem_len} not multiple of L={L}")

    act_mem = ms.activations.astype(np.float32)           # (S*L,)
    if act_mem.shape[0] != mem_len:
        raise ValueError(f"ms.activations len={act_mem.shape[0]} != mem_len={mem_len}")

    # Raw latent scores
    act = act_mem @ mem_adj.T                             # (n_lat,)

    # Per-timestep contributions summed over sensors
    contrib = np.empty((n_lat, L), dtype=np.float32)
    for t in range(L):
        idx = np.arange(t, mem_len, L, dtype=np.int64)    # s*L + t
        contrib[:, t] = (mem_adj[:, idx] * act_mem[idx]).sum(axis=1)

    # Contender rule: must include t=0 and be a contiguous prefix 0..k
    has_t0 = contrib[:, 0] > 0
    M = contrib > 0
    has_any = M.any(axis=1)

    last_from_end = np.argmax(M[:, ::-1], axis=1)         # 0 if all False
    last_idx = (L - 1) - last_from_end
    pos = np.arange(L)
    within_prefix = pos <= last_idx[:, None]
    prefix_all_true = np.all(~within_prefix | M, axis=1)

    contenders = has_t0 & has_any & prefix_all_true

    if contenders.any():
        masked = act.copy()
        masked[~contenders] = -np.float32(np.inf)
        bmu = int(np.argmax(masked))
    else:
        bmu = int(np.argmax(act))  # fallback

    g.set_node_feat("activation", act.astype(np.float32))
    return bmu



