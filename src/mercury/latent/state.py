# latent/state.py
from copy import deepcopy, copy
from dataclasses import dataclass, replace, field
from typing import Any, Tuple, List

import numpy as np
import scipy.stats as stats

from ..action_map.adapter import ActionMap
from mercury.graph.core import Graph
from mercury.graph.maintenance import age_maintenance
from .params import LatentParams
from ..memory.state import (MemoryState, mem_id, activations_at_t,
                            memory_view_at_global_timestep, init_mem,
                            add_memory, update_memory)
from ..sensory.state import SensoryState

type Array = np.ndarray

@dataclass
class _MemoryView:
    length: int
    activations: np.ndarray  # shape (S*L,)
    gs: Any                  # must have .n so compute_bmu doesn't break
    sensory_n_nodes: int




@dataclass
class LatentState:
    g: Graph
    mapping: Array
    prev_bmu: int | None = None
    step_idx: int = 0
    preds: List = field(default_factory=list)
    prev_activations : np.ndarray | None = None

    @property
    def mem_adj(self) -> Array:
        return self.g.node_features["mem_adj"]

def _register_features(g: Graph, mem_len: int, n_actions: int) -> Graph:
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
    g.register_node_feature("ambiguity_score", n_actions, init_value=0)
    g.register_edge_feature("age", 1, dtype=np.int32, init_value=0)
    g.register_edge_feature("action", 1, dtype=np.int32, init_value=0)
    return g


def build_initial_graph(ms: MemoryState, n_actions: int = 1) -> Graph:
    g = Graph(directed=True)
    _register_features(g, ms.gs.n, n_actions)
    for n in np.arange(ms.sensory_n_nodes):
        g = _add_node(g, ms, n)


    return g

def init_latent_state(ms: MemoryState, n_actions: int = 1) -> LatentState:
    g = build_initial_graph(ms, n_actions=n_actions)
    return LatentState(g, mapping = np.arange(g.n))



def _add_node(
    g: Graph,
    ms: MemoryState,
    u: int,
    t: int = 0,
    mem_adj: Array | None = None,
) -> Graph:
    """
    Add a new latent node corresponding to memory unit u at timestep t.

    If mem_adj is provided it is used to initialise the node's mem_adj feature.
    Otherwise mem_adj is initialised to zeros.

    g: Graph
    ms: MemoryState
    u: index of memory unit in [0, ms.gs.n//ms.length)
    t: timestep index in [0, ms.length)
    mem_adj: optional (mem_len,) float32 array defining adjacency-to-memory encoding
    """
    mem_len = g.node_features["mem_adj"].shape[1]

    if mem_adj is None:
        mem_adj_init = np.zeros(mem_len, dtype=np.float32)
    else:
        mem_adj = np.asarray(mem_adj, dtype=np.float32)
        if mem_adj.shape != (mem_len,):
            raise ValueError(
                f"mem_adj shape {mem_adj.shape} does not match expected {(mem_len,)}"
            )
        mem_adj_init = mem_adj

    new_node = g.add_node(
        node_feat={
            "activation": np.array(0.0, dtype=np.float32),
            "mem_adj": mem_adj_init,
            "ambiguity_score": (
                np.zeros((g.node_features["ambiguity_score"].shape[1],), dtype=np.float32)
                if g.node_features["ambiguity_score"].ndim > 1
                else np.array(0.0, dtype=np.float32)
            ),
        },
        edge_defaults_for_new_node={
            "age": np.array([0], dtype=np.int32),
            "action": np.array([0], dtype=np.int32),
        },
    )

    g = _add_temporal_edge(g, u, new_node, t, ms.length)
    return g



def _add_temporal_edge(g: Graph, u: int, v: int, t: int, L: int) -> Graph:
    n_lat = g.n
    if not (0 <= v < n_lat):
        raise ValueError(f"v={v} out of range [0,{n_lat-1}]")
    if not (0 <= t < L):
        raise ValueError(f"t={t} out of range [0,{L-1}]")

    mem_len = g.node_features["mem_adj"].shape[1]
    if mem_len % L != 0:                                   # NEW
        raise ValueError(f"mem_len={mem_len} not divisible by L={L}")  # NEW

    idx = mem_id(u, t, L)
    if not (0 <= idx < mem_len):
        raise ValueError(f"mem index {idx} out of range [0,{mem_len-1}]")
    g.node_features["mem_adj"][v, idx] = np.float32(1.0)
    return g

# def compute_bmu(
#     state: LatentState,
#     ms: MemoryState,
# ) -> int:
#     g = state.g
#     mem_adj = g.node_features["mem_adj"]
#     act_mem = ms.activations
#
#     act = act_mem @ mem_adj.T
def compute_bmu_current_timestep_only(state: LatentState, ms: MemoryState,
                                 action_bmu:int, cfg : LatentParams) \
        -> (
        int):
    """
    Compute latent activations from memory, apply contender mask
    (t0 required + consecutive 0..k timesteps), write 'activation'
    to the graph, and return BMU index.
    """
    trace_decay = cfg.trace_decay
    mixture_alpha = cfg.mixture_alpha
    mixture_beta = cfg.mixture_beta


    g = state.g
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
    # contenders = has_t0

    # if contenders.any():
    masked = act.copy()
    masked[~contenders] = 0
    # bmu = int(np.argmax(masked))

    # --- use trace_{t-1} to bias BMU selection (diffusion) ---

    prev_trace = state.prev_activations
    if prev_trace is None:
        prev_trace = np.zeros((g.n,), dtype=np.float32)
    else:
        prev_trace = np.asarray(prev_trace, dtype=np.float32).reshape(-1)

    # ensure shape matches current graph
    if prev_trace.shape[0] < g.n:
        prev_trace = np.pad(prev_trace, (0, g.n - prev_trace.shape[0]),
                            mode="constant")
    elif prev_trace.shape[0] > g.n:
        prev_trace = prev_trace[: g.n]

    A_action_conditioned = _transition_for_action_strength(g, action_bmu)
    A_tilde_conditioned = A_action_conditioned + np.identity(g.n, dtype=np.float32)
    degrees = np.sum(A_tilde_conditioned, axis=1).astype(np.float32)
    degrees_inv_sqrt = 1.0 / np.sqrt(np.maximum(degrees, 1e-12))
    A_action_norm = ((A_tilde_conditioned * degrees_inv_sqrt[None, :]) *
               degrees_inv_sqrt[:, None])

    A_tilde = g.adj.astype(np.float32) + np.identity(g.n, dtype=np.float32)
    degrees = np.sum(A_tilde, axis=1).astype(np.float32)
    degrees_inv_sqrt = 1.0 / np.sqrt(np.maximum(degrees, 1e-12))
    A_norm = (A_tilde * degrees_inv_sqrt[None, :]) * degrees_inv_sqrt[:, None]

    action_diffusion_from_trace = A_action_norm @ prev_trace
    diffusion_from_trace = A_norm @ prev_trace  # (n,)



    t_now = 0
    idx_now = t_now + np.arange(ms.sensory_n_nodes,
                                dtype=np.int64) * ms.length  # (S,)

    a_now = np.maximum(act_mem[idx_now], 0.0).astype(np.float32)  # (S,)
    mem_adj_now = mem_adj[:, idx_now]

    active_sensors = (a_now != 0)  # (S,)

    # Candidate if it connects to all active sensors (when there are any)
    if np.any(active_sensors):
        candidate_mask = np.all(mem_adj_now[:, active_sensors] != 0,
                                axis=1)  # (N,)
    else:
        candidate_mask = np.ones(mem_adj_now.shape[0],
                                 dtype=bool)  # no evidence -> allow all

    masked[~candidate_mask] = 0
    diffusion_from_trace[~candidate_mask] = 0
    action_diffusion_from_trace[~candidate_mask] = 0

    # print(
    #     f"mem_score {np.array2string(masked, precision=3, floatmode='fixed')}")
    # print(
    #     f"diffusion_from_trace {np.array2string(diffusion_from_trace, precision=3, floatmode='fixed')}")
    # print(
    #     f"action_diffusion_from_trace {np.array2string(action_diffusion_from_trace, precision=3, floatmode='fixed')}")

    # Normalise
    masked = norm(masked)
    diffusion_from_trace = norm(diffusion_from_trace)
    action_diffusion_from_trace = norm(action_diffusion_from_trace)

    # print(
    #     f"mem_score {np.array2string(masked, precision=3, floatmode='fixed')}")
    # print(
    #     f"diffusion_from_trace {np.array2string(diffusion_from_trace, precision=3, floatmode='fixed')}")
    # print(
    #     f"action_diffusion_from_trace {np.array2string(action_diffusion_from_trace, precision=3, floatmode='fixed')}")



    w_mem, w_trace, w_action = _convex_mixture_weights(mixture_alpha, mixture_beta)
    scored = (w_mem * masked) + (w_trace * diffusion_from_trace) + (w_action * action_diffusion_from_trace)


    print(
        f"scored {np.array2string(scored, precision=3, floatmode='fixed')}")

    memory_bmu = int(np.argmax(masked))

    bmu = int(np.argmax(scored))

    print(f"memory_bmu {memory_bmu}")
    print(f"bmu {bmu}")

    # if memory_bmu != bmu:
    #     print("memory bmu", memory_bmu)
    #     print("bmu", bmu)
    # #     print(f"memory_bmu mem {state.g.node_features["mem_adj"][memory_bmu]}")
    # #     print(f"bmu mem {state.g.node_features["mem_adj"][bmu]})")
    #     print(50*"-")
    # dist_w = np.zeros((g.n), dtype=np.float32)
    # for i in range(g.n):
    #     a = wasserstein_per_sensor(mem_adj[i], act_mem, ms.sensory_n_nodes,
    #                             ms.length)
    #     dist_w[i] = sum(a)
    # print(dist_w)
    #
    # bmu = np.argmin(dist_w)
    #
    # # --- your selection + scoring ---
    # t_now = 0
    # idx_now = t_now + np.arange(ms.sensory_n_nodes,
    #                             dtype=np.int64) * ms.length  # (S,)
    #
    # a_now = np.maximum(act_mem[idx_now], 0.0).astype(np.float32)
    # mem_adj_now = mem_adj[:, idx_now]  # (N, S)
    #
    # candidate_mask = (mem_adj_now != 0) & (a_now[None, :] != 0)
    # candidate_nodes = np.flatnonzero(candidate_mask.any(axis=1))
    #
    # # precompute kernel once
    # K = time_varying_gaussian_kernel(
    #     L=ms.length,
    #     sigma_min=0.5,  # tune
    #     sigma_max=4.0,  # tune
    #     power=1,  # tune
    # )
    #
    # observed_mass_sl = np.maximum(
    #     act_mem.reshape(ms.sensory_n_nodes, ms.length), 0.0
    # ).astype(np.float32)
    #
    # # distances only for candidates
    # dist_cand = np.empty(candidate_nodes.size, dtype=np.float32)
    # for j, node_index in enumerate(candidate_nodes):
    #     node_edge_weights_sl = np.abs(mem_adj[node_index]).reshape(
    #         ms.sensory_n_nodes, ms.length
    #     ).astype(np.float32)
    #     dist_cand[j] = kernel_distance_with_evidence(
    #         observed_mass_sl=observed_mass_sl,
    #         node_edge_weights_sl=node_edge_weights_sl,
    #         K=K,
    #         kappa=5.0,
    #     )
    #
    # # convert to similarity (local, independent)
    # tau = 1.0  # temperature; tune
    # sim_cand = softmax_neg_distance(dist_cand, tau)  # in (0,1]
    #
    # # write back to full vector (non-candidates get 0 similarity)
    # sim_gauss = np.zeros(g.n, dtype=np.float32)
    #
    # sim_gauss[candidate_nodes] = sim_cand
    #
    # scored = sim_gauss + (trace_influence * diffusion_from_trace) + (
    #     action_trace_influence * action_diffusion_from_trace)
    # bmu = int(np.argmax(scored))




    # bmu = np.argmin(dist_gaus)


    # --- now update trace_t using current masked evidence ---
    state.prev_activations = _update_exponential_trace(
        prev_trace=prev_trace,
        current_signal=masked.astype(np.float32, copy=False),
        trace_decay=trace_decay,
        clip_min=0.0,
    )

    return bmu

def compute_bmu_current_timestep_only(
    state: LatentState,
    memory_state: MemoryState,
    action_bmu: int,
    cfg: LatentParams,
    timestep_now: int = 0,
) -> int:
    trace_decay = cfg.trace_decay
    mixture_alpha = cfg.mixture_alpha
    mixture_beta = cfg.mixture_beta

    graph = state.g
    mem_adj = graph.node_features["mem_adj"]  # (n_latent, S*L)
    n_latent = graph.n
    mem_len = mem_adj.shape[1]

    length = getattr(memory_state, "length", mem_len)
    if mem_len % length != 0:
        raise ValueError(f"mem_len={mem_len} not multiple of length={length}")

    memory_activations = memory_state.activations.astype(np.float32)
    if memory_activations.shape[0] != mem_len:
        raise ValueError(
            f"memory_state.activations len={memory_activations.shape[0]} != mem_len={mem_len}"
        )

    if not (0 <= timestep_now < length):
        raise ValueError(f"timestep_now={timestep_now} out of range [0, {length})")

    # --- memory score uses ONLY current timestep evidence ---
    indices_now = timestep_now + np.arange(
        memory_state.sensory_n_nodes, dtype=np.int64
    ) * length  # (S,)

    activations_now = np.maximum(memory_activations[indices_now], 0.0).astype(np.float32)  # (S,)
    mem_adj_now = mem_adj[:, indices_now].astype(np.float32, copy=False)  # (N, S)

    memory_score = (mem_adj_now * activations_now[None, :]).sum(axis=1)  # (N,)

    # Sensor-consistency gating (based on active sensors at timestep_now)
    active_sensors = activations_now != 0.0
    if np.any(active_sensors):
        candidate_mask = np.all(mem_adj_now[:, active_sensors] != 0, axis=1)
    else:
        candidate_mask = np.ones((n_latent,), dtype=bool)

    memory_score[~candidate_mask] = 0.0

    # --- trace diffusion terms (same as full version) ---
    prev_trace = state.prev_activations
    if prev_trace is None:
        prev_trace = np.zeros((n_latent,), dtype=np.float32)
    else:
        prev_trace = np.asarray(prev_trace, dtype=np.float32).reshape(-1)

    if prev_trace.shape[0] < n_latent:
        prev_trace = np.pad(prev_trace, (0, n_latent - prev_trace.shape[0]), mode="constant")
    elif prev_trace.shape[0] > n_latent:
        prev_trace = prev_trace[:n_latent]

    base_norm = _normalized_adjacency_with_self_loops(graph.adj)
    action_adjacency = _transition_for_action_strength(graph, action_bmu)
    action_norm = _normalized_adjacency_with_self_loops(action_adjacency)

    diffusion_from_trace = base_norm @ prev_trace
    action_diffusion_from_trace = action_norm @ prev_trace

    diffusion_from_trace[~candidate_mask] = 0.0
    action_diffusion_from_trace[~candidate_mask] = 0.0

    # Combine
    memory_score = norm(memory_score)
    diffusion_from_trace = norm(diffusion_from_trace)
    action_diffusion_from_trace = norm(action_diffusion_from_trace)

    w_mem, w_trace, w_action = _convex_mixture_weights(mixture_alpha, mixture_beta)
    combined_score = (
        (w_mem * memory_score)
        + (w_trace * diffusion_from_trace)
        + (w_action * action_diffusion_from_trace)
    )

    bmu = int(np.argmax(combined_score))

    # Update trace using masked memory evidence
    state.prev_activations = _update_exponential_trace(
        prev_trace=prev_trace,
        current_signal=memory_score.astype(np.float32, copy=False),
        trace_decay=trace_decay,
        clip_min=0.0,
    )
    return bmu


def _normalized_undirected_adjacency_with_self_loops(
        adjacency: np.ndarray) -> np.ndarray:
    adjacency = adjacency.astype(np.float32, copy=False)

    # Convert directed -> undirected without doubling reciprocal edges.
    undirected = np.maximum(adjacency, adjacency.T)

    adjacency_tilde = undirected + np.identity(undirected.shape[0],
                                               dtype=np.float32)
    degrees = np.sum(adjacency_tilde, axis=1).astype(np.float32)
    degrees_inv_sqrt = 1.0 / np.sqrt(np.maximum(degrees, 1e-12))
    return (adjacency_tilde * degrees_inv_sqrt[None, :]) * degrees_inv_sqrt[
        :, None]


def _normalized_adjacency_with_self_loops(adjacency: np.ndarray) -> np.ndarray:
    adjacency = adjacency.astype(np.float32, copy=False)
    adjacency_tilde = adjacency + np.identity(adjacency.shape[0], dtype=np.float32)
    degrees = np.sum(adjacency_tilde, axis=1).astype(np.float32)
    degrees_inv_sqrt = 1.0 / np.sqrt(np.maximum(degrees, 1e-12))
    return (adjacency_tilde * degrees_inv_sqrt[None, :]) * degrees_inv_sqrt[:, None]


def _convex_mixture_weights(p: float, q: float) -> Tuple[float, float, float]:
    p = float(np.clip(p, 0.0, 1.0))
    q = float(np.clip(q, 0.0, 1.0))
    w_mem = p
    w_action = (1.0 - p) * q
    w_trace = (1.0 - p) * (1.0 - q)
    return w_mem, w_trace, w_action



def compute_bmu(state: LatentState, ms: MemoryState, action_bmu :int, cfg : LatentParams) \
        -> (
        int):
    """
    Compute latent activations from memory, apply contender mask
    (t0 required + consecutive 0..k timesteps), write 'activation'
    to the graph, and return BMU index.
    """
    trace_decay = cfg.trace_decay
    mixture_alpha = cfg.mixture_alpha
    mixture_beta = cfg.mixture_beta


    g = state.g
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
    # contenders = has_t0

    # if contenders.any():
    masked = act.copy()
    masked[~contenders] = 0
    # bmu = int(np.argmax(masked))

    # --- use trace_{t-1} to bias BMU selection (diffusion) ---

    prev_trace = state.prev_activations
    if prev_trace is None:
        prev_trace = np.zeros((g.n,), dtype=np.float32)
    else:
        prev_trace = np.asarray(prev_trace, dtype=np.float32).reshape(-1)

    # ensure shape matches current graph
    if prev_trace.shape[0] < g.n:
        prev_trace = np.pad(prev_trace, (0, g.n - prev_trace.shape[0]),
                            mode="constant")
    elif prev_trace.shape[0] > g.n:
        prev_trace = prev_trace[: g.n]

    A_action_conditioned = _transition_for_action_strength(g, action_bmu)
    A_tilde_conditioned = A_action_conditioned + np.identity(g.n, dtype=np.float32)
    degrees = np.sum(A_tilde_conditioned, axis=1).astype(np.float32)
    degrees_inv_sqrt = 1.0 / np.sqrt(np.maximum(degrees, 1e-12))
    A_action_norm = ((A_tilde_conditioned * degrees_inv_sqrt[None, :]) *
               degrees_inv_sqrt[:, None])

    A_tilde = g.adj.astype(np.float32) + np.identity(g.n, dtype=np.float32)
    degrees = np.sum(A_tilde, axis=1).astype(np.float32)
    degrees_inv_sqrt = 1.0 / np.sqrt(np.maximum(degrees, 1e-12))
    A_norm = (A_tilde * degrees_inv_sqrt[None, :]) * degrees_inv_sqrt[:, None]

    action_diffusion_from_trace = A_action_norm @ prev_trace
    diffusion_from_trace = A_norm @ prev_trace  # (n,)



    t_now = 0
    idx_now = t_now + np.arange(ms.sensory_n_nodes,
                                dtype=np.int64) * ms.length  # (S,)

    a_now = np.maximum(act_mem[idx_now], 0.0).astype(np.float32)  # (S,)
    mem_adj_now = mem_adj[:, idx_now]

    active_sensors = (a_now != 0)  # (S,)

    # Candidate if it connects to all active sensors (when there are any)
    if np.any(active_sensors):
        candidate_mask = np.all(mem_adj_now[:, active_sensors] != 0,
                                axis=1)  # (N,)
    else:
        candidate_mask = np.ones(mem_adj_now.shape[0],
                                 dtype=bool)  # no evidence -> allow all

    masked[~candidate_mask] = 0
    diffusion_from_trace[~candidate_mask] = 0
    action_diffusion_from_trace[~candidate_mask] = 0

    # print(
    #     f"mem_score {np.array2string(masked, precision=3, floatmode='fixed')}")
    # print(
    #     f"diffusion_from_trace {np.array2string(diffusion_from_trace, precision=3, floatmode='fixed')}")
    # print(
    #     f"action_diffusion_from_trace {np.array2string(action_diffusion_from_trace, precision=3, floatmode='fixed')}")

    # Normalise
    masked = norm(masked)
    diffusion_from_trace = norm(diffusion_from_trace)
    action_diffusion_from_trace = norm(action_diffusion_from_trace)

    # print(
    #     f"mem_score {np.array2string(masked, precision=3, floatmode='fixed')}")
    # print(
    #     f"diffusion_from_trace {np.array2string(diffusion_from_trace, precision=3, floatmode='fixed')}")
    # print(
    #     f"action_diffusion_from_trace {np.array2string(action_diffusion_from_trace, precision=3, floatmode='fixed')}")



    w_mem, w_trace, w_action = _convex_mixture_weights(mixture_alpha, mixture_beta)
    scored = (w_mem * masked) + (w_trace * diffusion_from_trace) + (w_action * action_diffusion_from_trace)

    # print(
    #     f"scored {np.array2string(scored, precision=3, floatmode='fixed')}")

    memory_bmu = int(np.argmax(masked))

    bmu = int(np.argmax(scored))

    # if memory_bmu != bmu:
    #     print("memory bmu", memory_bmu)
    #     print("bmu", bmu)
    #     print(f"memory_bmu mem {state.g.node_features["mem_adj"][memory_bmu]}")
    #     print(f"bmu mem {state.g.node_features["mem_adj"][bmu]})")
    #     print(50*"-")
    # dist_w = np.zeros((g.n), dtype=np.float32)
    # for i in range(g.n):
    #     a = wasserstein_per_sensor(mem_adj[i], act_mem, ms.sensory_n_nodes,
    #                             ms.length)
    #     dist_w[i] = sum(a)
    # print(dist_w)
    #
    # bmu = np.argmin(dist_w)
    #
    # # --- your selection + scoring ---
    # t_now = 0
    # idx_now = t_now + np.arange(ms.sensory_n_nodes,
    #                             dtype=np.int64) * ms.length  # (S,)
    #
    # a_now = np.maximum(act_mem[idx_now], 0.0).astype(np.float32)
    # mem_adj_now = mem_adj[:, idx_now]  # (N, S)
    #
    # candidate_mask = (mem_adj_now != 0) & (a_now[None, :] != 0)
    # candidate_nodes = np.flatnonzero(candidate_mask.any(axis=1))
    #
    # # precompute kernel once
    # K = time_varying_gaussian_kernel(
    #     L=ms.length,
    #     sigma_min=0.5,  # tune
    #     sigma_max=4.0,  # tune
    #     power=1,  # tune
    # )
    #
    # observed_mass_sl = np.maximum(
    #     act_mem.reshape(ms.sensory_n_nodes, ms.length), 0.0
    # ).astype(np.float32)
    #
    # # distances only for candidates
    # dist_cand = np.empty(candidate_nodes.size, dtype=np.float32)
    # for j, node_index in enumerate(candidate_nodes):
    #     node_edge_weights_sl = np.abs(mem_adj[node_index]).reshape(
    #         ms.sensory_n_nodes, ms.length
    #     ).astype(np.float32)
    #     dist_cand[j] = kernel_distance_with_evidence(
    #         observed_mass_sl=observed_mass_sl,
    #         node_edge_weights_sl=node_edge_weights_sl,
    #         K=K,
    #         kappa=5.0,
    #     )
    #
    # # convert to similarity (local, independent)
    # tau = 1.0  # temperature; tune
    # sim_cand = softmax_neg_distance(dist_cand, tau)  # in (0,1]
    #
    # # write back to full vector (non-candidates get 0 similarity)
    # sim_gauss = np.zeros(g.n, dtype=np.float32)
    #
    # sim_gauss[candidate_nodes] = sim_cand
    #
    # scored = sim_gauss + (trace_influence * diffusion_from_trace) + (
    #     action_trace_influence * action_diffusion_from_trace)
    # bmu = int(np.argmax(scored))




    # bmu = np.argmin(dist_gaus)


    # --- now update trace_t using current masked evidence ---
    state.prev_activations = _update_exponential_trace(
        prev_trace=prev_trace,
        current_signal=masked.astype(np.float32, copy=False),
        trace_decay=trace_decay,
        clip_min=0.0,
    )

    return bmu

def compute_bmu(
    state: LatentState,
    memory_state: MemoryState,
    action_bmu: int,
    cfg: LatentParams,
    timestep_for_gating: int = 0,
) -> int:
    trace_decay = cfg.trace_decay
    mixture_alpha = cfg.mixture_alpha
    mixture_beta = cfg.mixture_beta

    graph = state.g
    mem_adj = graph.node_features["mem_adj"]  # (n_latent, S*L)
    n_latent = graph.n
    mem_len = mem_adj.shape[1]

    length = getattr(memory_state, "length", mem_len)
    if mem_len % length != 0:
        raise ValueError(f"mem_len={mem_len} not multiple of length={length}")

    memory_activations = memory_state.activations.astype(np.float32)
    if memory_activations.shape[0] != mem_len:
        raise ValueError(
            f"memory_state.activations len={memory_activations.shape[0]} != mem_len={mem_len}"
        )

    # --- memory score uses ALL timesteps ---
    raw_scores = memory_activations @ mem_adj.T  # (N,)

    # Per-timestep contributions (summed over sensors)
    timestep_contrib = np.empty((n_latent, length), dtype=np.float32)
    for t in range(length):
        indices_t = np.arange(t, mem_len, length, dtype=np.int64)
        timestep_contrib[:, t] = (mem_adj[:, indices_t] * memory_activations[indices_t]).sum(axis=1)

    # Contender: must include t=0 and be contiguous prefix 0..k
    has_t0 = timestep_contrib[:, 0] > 0
    positive_mask = timestep_contrib > 0
    has_any_positive = positive_mask.any(axis=1)

    last_true_from_end = np.argmax(positive_mask[:, ::-1], axis=1)
    last_true_index = (length - 1) - last_true_from_end
    positions = np.arange(length)
    within_prefix = positions <= last_true_index[:, None]
    prefix_all_true = np.all(~within_prefix | positive_mask, axis=1)

    contenders = has_t0 & has_any_positive & prefix_all_true

    memory_score = raw_scores.astype(np.float32, copy=True)
    memory_score[~contenders] = 0.0

    # Optional: gate using a single timestep (default t=0)
    if not (0 <= timestep_for_gating < length):
        raise ValueError(f"timestep_for_gating={timestep_for_gating} out of range [0, {length})")

    indices_gate = timestep_for_gating + np.arange(
        memory_state.sensory_n_nodes, dtype=np.int64
    ) * length

    activations_gate = np.maximum(memory_activations[indices_gate], 0.0).astype(np.float32)
    mem_adj_gate = mem_adj[:, indices_gate]
    active_sensors = activations_gate != 0.0

    if np.any(active_sensors):
        candidate_mask = np.all(mem_adj_gate[:, active_sensors] != 0, axis=1)
    else:
        candidate_mask = np.ones((n_latent,), dtype=bool)

    memory_score[~candidate_mask] = 0.0

    # --- trace diffusion terms ---
    prev_trace = state.prev_activations
    if prev_trace is None:
        prev_trace = np.zeros((n_latent,), dtype=np.float32)
    else:
        prev_trace = np.asarray(prev_trace, dtype=np.float32).reshape(-1)

    if prev_trace.shape[0] < n_latent:
        prev_trace = np.pad(prev_trace, (0, n_latent - prev_trace.shape[0]), mode="constant")
    elif prev_trace.shape[0] > n_latent:
        prev_trace = prev_trace[:n_latent]

    undirected_base_norm = _normalized_undirected_adjacency_with_self_loops(graph.adj)
    base_norm = _normalized_adjacency_with_self_loops(graph.adj)
    action_adjacency = _transition_for_action_strength(graph, action_bmu)
    action_norm = _normalized_adjacency_with_self_loops(action_adjacency)

    undirected_diffusion_from_trace = undirected_base_norm @ prev_trace
    diffusion_from_trace = base_norm @ prev_trace
    # action_diffusion_from_trace = action_norm @ prev_trace

    undirected_diffusion_from_trace[~candidate_mask] = 0.0
    diffusion_from_trace[~candidate_mask] = 0.0
    # action_diffusion_from_trace[~candidate_mask] = 0.0

    # Combine
    # memory_score = norm(memory_score)
    # diffusion_from_trace = norm(diffusion_from_trace)
    # action_diffusion_from_trace = norm(action_diffusion_from_trace)

    w_mem, w_trace, w_action = _convex_mixture_weights(mixture_alpha, mixture_beta)
    combined_score = (
        (w_mem * memory_score)
        + (w_trace * undirected_diffusion_from_trace)
        + (w_action * diffusion_from_trace)
    )

    bmu = int(np.argmax(combined_score))

    state.prev_activations = _update_exponential_trace(
        prev_trace=prev_trace,
        current_signal=memory_score.astype(np.float32, copy=False),
        trace_decay=trace_decay,
        clip_min=0.0,
    )
    return bmu

def compute_bmu(
    state: LatentState,
    memory_state: MemoryState,
    action_bmu: int,
    cfg: LatentParams,
    timestep_for_gating: int = 0,
) -> int:
    """
    Select the winning latent unit by minimizing a constrained neural energy.

    Neural interpretation
    ---------------------
    Each latent node is treated as a neural unit receiving four forms of drive:

        1. memory drive
        2. undirected recurrent drive
        3. baseline directed transition drive
        4. action-conditioned transition drive

    For latent unit i, the instantaneous neural support is

        h_i =
            w_memory      * memory_drive_i
          + w_undirected  * recurrent_drive_i
          + w_base        * baseline_transition_drive_i
          + w_action      * action_transition_drive_i

    Structural admissibility is enforced by binary gating. Units that fail the
    temporal-prefix gate or the sensory gate are excluded from competition.

    The neural energy is

        E_i = -h_i    for admissible units
        E_i = +inf    for gated-out units

    The inferred latent state is the winner-take-all unit with minimum energy.

    Notes
    -----
    - Temporal-prefix gating requires positive support beginning at t=0 and
      continuing as a contiguous prefix.
    - Sensory gating requires all currently active gated sensory channels to be
      represented by the latent unit.
    - The trace update at the end is a separate persistence dynamic and not part
      of the instantaneous energy itself.
    """
    trace_decay = cfg.trace_decay

    latent_graph = state.g
    latent_memory_templates = latent_graph.node_features["mem_adj"]  # shape: (n_latent, S * L)
    n_latent_units = latent_graph.n
    flattened_memory_length = latent_memory_templates.shape[1]

    temporal_window_length = getattr(memory_state, "length", flattened_memory_length)
    if flattened_memory_length % temporal_window_length != 0:
        raise ValueError(
            f"flattened_memory_length={flattened_memory_length} is not a multiple "
            f"of temporal_window_length={temporal_window_length}"
        )

    current_memory_activity = memory_state.activations.astype(np.float32)
    if current_memory_activity.shape[0] != flattened_memory_length:
        raise ValueError(
            f"memory_state.activations has length {current_memory_activity.shape[0]} "
            f"but expected {flattened_memory_length}"
        )

    # ------------------------------------------------------------------
    # 1. Memory drive: direct overlap between current memory activity and
    #    each latent unit's stored sensory-temporal template.
    # ------------------------------------------------------------------
    raw_memory_drive = current_memory_activity @ latent_memory_templates.T  # shape: (n_latent,)

    per_timestep_memory_drive = np.empty(
        (n_latent_units, temporal_window_length),
        dtype=np.float32,
    )
    for timestep_index in range(temporal_window_length):
        timestep_flat_indices = np.arange(
            timestep_index,
            flattened_memory_length,
            temporal_window_length,
            dtype=np.int64,
        )
        per_timestep_memory_drive[:, timestep_index] = (
            latent_memory_templates[:, timestep_flat_indices]
            * current_memory_activity[timestep_flat_indices]
        ).sum(axis=1)

    # ------------------------------------------------------------------
    # 2. Temporal admissibility gate:
    #    the latent unit must show positive support from t=0 over a
    #    contiguous prefix of the temporal memory strip.
    # ------------------------------------------------------------------
    has_positive_initial_drive = per_timestep_memory_drive[:, 0] > 0
    positive_timestep_mask = per_timestep_memory_drive > 0
    has_any_positive_timestep = positive_timestep_mask.any(axis=1)

    last_positive_from_end = np.argmax(positive_timestep_mask[:, ::-1], axis=1)
    last_positive_timestep = (temporal_window_length - 1) - last_positive_from_end

    timestep_positions = np.arange(temporal_window_length)
    within_positive_prefix = timestep_positions <= last_positive_timestep[:, None]
    prefix_is_contiguous = np.all(
        ~within_positive_prefix | positive_timestep_mask,
        axis=1,
    )

    passes_temporal_prefix_gate = (
        has_positive_initial_drive
        & has_any_positive_timestep
        & prefix_is_contiguous
    )

    memory_drive = raw_memory_drive.astype(np.float32, copy=True)
    memory_drive[~passes_temporal_prefix_gate] = 0.0

    # ------------------------------------------------------------------
    # 3. Sensory admissibility gate:
    #    all active sensory channels at the chosen gating timestep must be
    #    represented by the latent unit.
    # ------------------------------------------------------------------
    if not (0 <= timestep_for_gating < temporal_window_length):
        raise ValueError(
            f"timestep_for_gating={timestep_for_gating} out of range "
            f"[0, {temporal_window_length})"
        )

    gated_flat_indices = timestep_for_gating + np.arange(
        memory_state.sensory_n_nodes,
        dtype=np.int64,
    ) * temporal_window_length

    gated_sensory_activity = np.maximum(
        current_memory_activity[gated_flat_indices],
        0.0,
    ).astype(np.float32)

    gated_latent_templates = latent_memory_templates[:, gated_flat_indices]
    active_gated_channels = gated_sensory_activity != 0.0

    if np.any(active_gated_channels):
        passes_sensory_gate = np.all(
            gated_latent_templates[:, active_gated_channels] != 0,
            axis=1,
        )
    else:
        passes_sensory_gate = np.ones((n_latent_units,), dtype=bool)

    memory_drive[~passes_sensory_gate] = 0.0

    admissibility_gate = passes_temporal_prefix_gate & passes_sensory_gate

    # ------------------------------------------------------------------
    # 4. Persistent latent trace from the previous step.
    # ------------------------------------------------------------------
    previous_latent_trace = state.prev_activations
    if previous_latent_trace is None:
        previous_latent_trace = np.zeros((n_latent_units,), dtype=np.float32)
    else:
        previous_latent_trace = np.asarray(previous_latent_trace, dtype=np.float32).reshape(-1)

    if previous_latent_trace.shape[0] < n_latent_units:
        previous_latent_trace = np.pad(
            previous_latent_trace,
            (0, n_latent_units - previous_latent_trace.shape[0]),
            mode="constant",
        )
    elif previous_latent_trace.shape[0] > n_latent_units:
        previous_latent_trace = previous_latent_trace[:n_latent_units]

    # ------------------------------------------------------------------
    # 5. Recurrent and transition drives.
    # ------------------------------------------------------------------
    undirected_recurrent_graph = np.maximum(latent_graph.adj,
                                            latent_graph.adj.T).astype(
        np.float32,
        copy=False,
    )
    baseline_transition_graph = latent_graph.adj.astype(np.float32, copy=False)

    action_conditioned_transition_graph = _transition_for_action_strength(
        latent_graph,
        action_bmu,
    ).astype(np.float32, copy=False)

    if cfg.allow_self_loops:
        identity = np.identity(latent_graph.n, dtype=np.float32)
        undirected_recurrent_graph = undirected_recurrent_graph + identity
        baseline_transition_graph = baseline_transition_graph + identity
        action_conditioned_transition_graph = action_conditioned_transition_graph + identity

    undirected_recurrent_drive = (undirected_recurrent_graph @
                                  previous_latent_trace)
    baseline_transition_drive = (baseline_transition_graph @
                                 previous_latent_trace)
    action_transition_drive = (action_conditioned_transition_graph @
                               previous_latent_trace)

    undirected_recurrent_drive[~admissibility_gate] = 0.0
    baseline_transition_drive[~admissibility_gate] = 0.0
    action_transition_drive[~admissibility_gate] = 0.0

    # ------------------------------------------------------------------
    # 6. Weighted neural support.
    # ------------------------------------------------------------------
    (
        weight_memory,
        weight_undirected,
        weight_base,
        weight_action,
    ) = cfg.normalized_energy_weights()

    latent_neural_support = (
        (weight_memory * memory_drive)
        + (weight_undirected * undirected_recurrent_drive)
        + (weight_base * baseline_transition_drive)
        + (weight_action * action_transition_drive)
        - (cfg.lambda_trace * previous_latent_trace)
    )

    # ------------------------------------------------------------------
    # 7. Constrained neural energy and winner-take-all selection.
    # ------------------------------------------------------------------
    neural_energy = np.full((n_latent_units,), np.inf, dtype=np.float32)
    neural_energy[admissibility_gate] = -latent_neural_support[admissibility_gate]

    winning_latent_unit = int(np.argmin(neural_energy))

    # ------------------------------------------------------------------
    # 8. Trace persistence dynamics.
    # ------------------------------------------------------------------
    bounded_memory_drive = np.tanh(memory_drive)

    state.prev_activations = _update_exponential_trace(
        prev_trace=previous_latent_trace,
        current_signal=bounded_memory_drive.astype(np.float32, copy=False),
        trace_decay=trace_decay,
        clip_min=0.0,
    )

    return winning_latent_unit



def softmax_neg_distance(distances: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    x = -distances / temperature
    x = x - np.max(x)                      # stability
    e = np.exp(x)
    return e / np.sum(e)

def norm(x):
    if np.sum(x) == 0:
        return x
    return x / np.sum(x)

def time_varying_gaussian_kernel(L: int, sigma_min: float, sigma_max: float, power: float = 1.0, eps: float = 1e-12) -> np.ndarray:
    t = np.arange(L, dtype=np.float32)
    frac = t / max(L - 1, 1)
    sigma = sigma_min + (sigma_max - sigma_min) * (frac ** power)
    sigma2 = sigma * sigma
    dt = t[:, None] - t[None, :]
    denom = 2.0 * (sigma2[:, None] + sigma2[None, :] + eps)
    return np.exp(-(dt * dt) / denom)

def kernel_distance_with_evidence(
    observed_mass_sl: np.ndarray,     # (S,L) nonnegative
    node_edge_weights_sl: np.ndarray, # (S,L) nonnegative (abs(mem_adj[i]) reshaped)
    K: np.ndarray,                    # (L,L)
    kappa: float = 5.0,
    eps: float = 1e-12,
) -> float:
    # predicted mass = evidence gate * observed mass
    predicted_mass_sl = node_edge_weights_sl * observed_mass_sl

    # kernel distance per sensor
    aKa = np.sum(observed_mass_sl * (observed_mass_sl @ K.T), axis=1)
    bKb = np.sum(predicted_mass_sl * (predicted_mass_sl @ K.T), axis=1)
    aKb = np.sum(observed_mass_sl * (predicted_mass_sl @ K.T), axis=1)
    dist_per_sensor = np.sqrt(np.maximum(aKa + bKb - 2.0 * aKb, 0.0))  # (S,)

    # evidence weight from total predicted mass (saturating)
    mass_pred = np.sum(predicted_mass_sl, axis=1)                      # (S,)
    w = mass_pred / (mass_pred + kappa + eps)                          # (S,) in [0,1)

    # aggregate: mismatch penalized less when evidence is higher
    return float(np.mean(dist_per_sensor / (w + eps)))

def wasserstein_per_sensor(
    flat_a: np.ndarray,
    flat_b: np.ndarray,
    sensory_count: int,
    strip_length: int,
    epsilon: float = 1e-12,
    normalise: bool = True,          # True -> proper probability per sensor
) -> np.ndarray:
    A = flat_a.reshape(sensory_count, strip_length)
    B = flat_b.reshape(sensory_count, strip_length)

    # calculate relevant timesteps
    # relevant_mask = (A != 0).any(axis=0)
    # relevant_timesteps = np.flatnonzero(relevant_mask)
    # A = A[:,relevant_timesteps]
    # B = B[:,relevant_timesteps]


    if normalise:
        A = A / (A.sum(axis=1, keepdims=True) + epsilon)
        B = B / (B.sum(axis=1, keepdims=True) + epsilon)

    cdf_a = np.cumsum(A, axis=1)
    cdf_b = np.cumsum(B, axis=1)

    w1_per_sensor = np.sum(np.abs(cdf_a - cdf_b), axis=1)  # shape (S,)
    return w1_per_sensor

def softmax(x: np.ndarray) -> np.ndarray:
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def _transition_for_action_strength(graph: Graph,
                                    action_bmu: int) -> np.ndarray:
    strengths = graph.adj.astype(np.float32, copy=False)
    actions = graph.edge_features["action"][:, :, 0].astype(np.int32,
                                                            copy=False)

    mask = (strengths != 0.0) & (actions == int(action_bmu))
    transition = np.zeros_like(strengths, dtype=np.float32)
    transition[mask] = strengths[mask]
    return transition

    #
    # if state.prev_activations is not None:
    #     A_tilde = state.g.adj + np.identity(state.g.n,
    #                                         dtype=np.float32)  # (n, n)
    #     d = np.sum(A_tilde, axis=1).astype(np.float32)  # (n,)
    #     d_inv_sqrt = 1.0 / np.sqrt(np.maximum(d, 1e-12))  # (n,)
    #
    #     x = np.asarray(state.prev_activations, dtype=np.float32).reshape(
    #         -1)  # (n,)
    #     print(x)
    #     # A_norm = D^-1/2 A D^-1/2 without building diag matrices
    #     A_norm = (A_tilde * d_inv_sqrt[None, :]) * d_inv_sqrt[:, None]  # (n, n)
    #
    #     convo = A_norm @ x  # (n,)
    #     print(convo)
    #
    #     graph_conditioned_mask = masked + (1-0.9)*convo
    #     bmu = int(np.argmax(graph_conditioned_mask))

    # else:
    #     bmu = int(np.argmax(act))  # fallback
    #     print("FALLBACK BMU")
    # bmu = int(np.argmax(act))  # fallback

    # g.set_node_feat("activation", act.astype(np.float32))

    # state.prev_activations = masked
    # return bmu

def _align_prev_activations_to_graph(state: LatentState) -> None:
    if state.prev_activations is None:
        return
    n = int(state.g.n)
    x = np.asarray(state.prev_activations, dtype=np.float32).reshape(-1)
    if x.shape[0] < n:
        x = np.pad(x, (0, n - x.shape[0]), mode="constant", constant_values=0.0)
    elif x.shape[0] > n:
        x = x[:n]
    state.prev_activations = x

def compute_bmu_array(state: LatentState, ms: MemoryState) -> int:
    """
    Compute latent activations from memory, apply contender mask
    (t0 required + consecutive 0..k timesteps), write 'activation'
    to the graph, and return BMU index.
    """
    g = state.g
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

    # if contenders.any():
    masked = act.copy()
    masked[~contenders] = -np.float32(np.inf)
    bmu = int(np.argmax(masked))
    # else:
    #     bmu = int(np.argmax(act))  # fallback
    #     print("FALLBACK BMU")
    # bmu = int(np.argmax(act))  # fallback

    # g.set_node_feat("activation", act.astype(np.float32))
    return masked


def _calc_activation(d: Array | float, gaussian_shape: float) -> Array | float:
    d = np.asarray(d, dtype=np.float32)
    sigma = float(gaussian_shape)
    if sigma <= 0:
        out = np.zeros_like(d, dtype=np.float32)
    else:
        out = np.exp(-(d * d) / (2.0 * sigma * sigma)).astype(np.float32)
    return float(out) if out.ndim == 0 else out


def _remove_aliased_connections(g: Graph,
                                bmu_prev: int,
                                predicted_nodes: Array) -> Graph:
    dst = np.asarray(predicted_nodes, dtype=np.int32).ravel()
    if dst.size == 0:
        return g
    src = np.full(dst.size, bmu_prev, dtype=np.int32)
    g.remove_edges(src, dst)  # vectorized
    return g


def _update_edge(
    g: Graph,
    u: int, v: int,
    prior_row: Array,
    observed_row: Array,
    action_bmu: int,
    beta: float,
    gaussian_shape: float,
) -> Graph:
    dist = float(np.linalg.norm(prior_row - observed_row))
    sim = float(_calc_activation(dist, gaussian_shape))
    current = float(g.adj[u, v])
    new_w = current + float(beta) * (sim - current)

    g.add_edge(
        u, v, weight=new_w,
        edge_feat={"action": np.array([int(action_bmu)], dtype=np.int32)},
    )
    return g

def _edge_action_row(g: Graph, u: int, v: int, *, am: ActionMap) -> Array:
    dim = int(am.params.dim)
    if (u < 0) or (v < 0) or (u >= g.n) or (v >= g.n):
        return np.zeros((dim,), np.float32)
    if float(g.adj[u, v]) == 0.0:
        return np.zeros((dim,), np.float32)
    if "action" not in g.edge_features:
        return np.zeros((dim,), np.float32)

    lbl = np.asarray(g.edge_features["action"][u, v], dtype=np.int32).reshape(())
    n_codes = am.state.codebook.shape[0]
    lbl = int(np.clip(lbl, 0, n_codes - 1))
    return am.state.codebook[lbl].astype(np.float32, copy=False)

# def _remap_prev_activations(prev_activations: np.ndarray | None,
#                             mapping: np.ndarray,
#                             new_n: int) -> np.ndarray | None:
#     if prev_activations is None:
#         return None
#
#     old = np.asarray(prev_activations, dtype=np.float32).reshape(-1)
#     out = np.zeros((new_n,), dtype=np.float32)
#
#     # mapping is old_index -> new_index (or -1 if removed)
#     old_n = min(mapping.shape[0], old.shape[0])
#     for old_idx in range(old_n):
#         new_idx = int(mapping[old_idx])
#         if new_idx >= 0:
#             out[new_idx] = old[old_idx]
#
#     return out

def _remap_prev_activations(
    prev_activations: np.ndarray | None,
    mapping: np.ndarray,
    new_n: int,
) -> np.ndarray | None:
    if prev_activations is None:
        return None

    old = np.asarray(prev_activations, dtype=np.float32).reshape(-1)
    remapped = np.zeros((new_n,), dtype=np.float32)

    old_n = min(mapping.shape[0], old.shape[0])
    for old_idx in range(old_n):
        new_idx = int(mapping[old_idx])
        if new_idx >= 0:
            remapped[new_idx] = old[old_idx]

    return remapped

def _update_exponential_trace(
    prev_trace: np.ndarray,
    current_signal: np.ndarray,
    trace_decay: float,
    *,
    clip_min: float = 0.0,
) -> np.ndarray:
    if not (0.0 <= trace_decay <= 1.0):
        raise ValueError(f"trace_decay must be in [0, 1], got {trace_decay}")

    updated = (trace_decay * prev_trace) + ((1.0 - trace_decay) * current_signal)
    if clip_min is not None:
        updated = np.maximum(updated, np.float32(clip_min))
    return updated.astype(np.float32, copy=False)


def _set_activation(g: Graph, bmu: int) -> Graph:
    act = np.zeros((g.n,), np.float32)
    act[bmu] = 1.0
    g.set_node_feat("activation", act)
    return g

def _update_edges_with_actions(g: Graph,
                               prev_bmu: int,
                               bmu: int,
                               action_bmu: int,
                               action_map: ActionMap,
                               gaussian_shape: int,
                               beta : float) -> Graph:
    observed_row = np.asarray(action_map.state.codebook[action_bmu],
                              np.float32)
    prior_row = _edge_action_row(g, prev_bmu, bmu, am=action_map)
    g = _update_edge(
        g, prev_bmu, bmu, prior_row, observed_row,
        action_bmu=action_bmu, beta=beta, gaussian_shape=gaussian_shape,
    )
    return g

def _predict(g: Graph, bmu: int, action_bmu: int) -> Array:
    """
    Return destination node indices j such that:
    - there is an edge bmu -> j
    - the 'action' feature on that edge equals action_bmu

    Output:
        np.ndarray[int32] of shape (k,)
        k == number of matching neighbours
    """
    actions_out = g.edge_features["action"][bmu, :, 0]      # (n,)
    edge_exists = g.adj[bmu, :] != 0                        # (n,) bool
    matches_action = actions_out == action_bmu              # (n,) bool

    mask = edge_exists & matches_action                      # (n,) bool
    idx = np.flatnonzero(mask).astype(np.int32, copy=False) # node indices
    # mask1 = idx!=bmu
    # idx1 = np.flatnonzero(mask1).astype(np.int32, copy=False)
    # if np.array_equal(idx, np.array([3, 7])):
    #     print("ayo")
    # print(f"prev {bmu} -> {action_bmu} -> preds: {idx}")
    return idx



def _is_aliased(g: Graph, action_bmu_out: Array) -> bool:
    return action_bmu_out.shape[0] > 1


def _find_matching_node(g: Graph, row: np.ndarray) -> int | None:
    M = g.node_features["mem_adj"]
    hit = np.flatnonzero(np.all(M == row[None, :], axis=1))
    return int(hit[0]) if hit.size else None

def _debug_print_latent_action_adj(g: Graph):
    for u in range(g.n):
        for v in range(g.n):
            if g.adj[u, v] != 0.0:
                action = g.edge_features["action"][u, v]
                print(f"u:{u} v:{v} action:{action}")


def _debug_print_latent_mem_adj(g: Graph, ms: MemoryState, node_idx: int) -> None:
    L = int(ms.length)
    mem_len = int(ms.gs.n)
    S = mem_len // L

    row = g.node_features["mem_adj"][node_idx].astype(np.float32, copy=False)

    pairs = []
    for u in range(S):
        for t in range(L):
            idx = u * L + t
            if row[idx] > 0:
                pairs.append((t, u))  # (t first for sorting)

    pairs.sort(key=lambda x: x[0])  # sort by timestep

    desc = ", ".join([f"(sensory {u}, t={t})" for (t, u) in pairs])
    print(f"latent {node_idx} encodes {desc}")

def rollback_memory_state(ms: MemoryState, mem_vec: Array, length) -> MemoryState:
    for i in range(length):
        ms = update_memory(ms)
        ms = add_memory(ms, mem_vec[i])
    return ms

def rollback_k_memory_state(ms: MemoryState, mem_vec: Array, k: int, L: int) -> MemoryState:
    length = 2*L
    mem_vec = np.asarray(mem_vec[-length:])
    replay_mem = init_mem(ms.sensory_n_nodes, ms.length)


    replay_mem = rollback_memory_state(replay_mem, mem_vec, length-k)
    return replay_mem

def _resolve_prev_with_memory_replay(
    state: LatentState,
    ms: MemoryState,
    mem_vec: list,
    prev_bmu: int,
    bmu: int,
    action_mem: list[int],
    action_map: ActionMap,
    gaussian_shape: int,
    max_replay: int,
    cfg: LatentParams,
    preds: Array,
) -> tuple[Graph, int, int]:





    replay_state = deepcopy(state)

    L = int(ms.length)
    if state.step_idx < 2*L:
        return replay_state.g, int(prev_bmu), bmu

    am = list(action_mem)[-L:]              # oldest → newest
    mem_vec = np.asarray(mem_vec[-2*L:])
    replay_mem = init_mem(ms.sensory_n_nodes, ms.length)

    replay_mem = rollback_memory_state(replay_mem, mem_vec, L)

    if len(am) < L:
        raise ValueError(f"action_mem length {len(am)} < L={L}")


    for i in range(L):
        action_bmu = int(am[i])
        if cfg.memory_disambiguation:
            bmu = int(compute_bmu(replay_state, replay_mem, action_bmu, cfg))
        else:
            bmu = int(
                compute_bmu_current_timestep_only(replay_state, replay_mem,
                                                  action_bmu, cfg))
        if bmu == prev_bmu:
        # if preds.size >1:
        #     nov = novelty_for_prev(replay_mem, replay_state.g, int(prev_bmu))
            # if nov >= 0.35:
            if True:
                g_new, bmu_refined, created = add_temporal_edge_at_first_gap_clone(
                    replay_state.g, replay_mem, int(prev_bmu), preds
                )
                # cfg.ambiguity_threshold += 1
                replay_state.g = g_new
                _align_prev_activations_to_graph(replay_state)
            # # refine by spawning a clone extended at the first gap
            # g_new, bmu_refined, created = add_temporal_edge_at_first_gap_clone(
            #     replay_state.g, replay_mem, bmu, preds
            # )
            # replay_state.g = g_new
            # _align_prev_activations_to_graph(replay_state)

            # _debug_print_latent_mem_adj(g_new, ms, bmu_refined)
            # print(f"created {created}")
            # replay_state.g = g_new

                # bmu = int(compute_bmu(replay_state, replay_mem, action_bmu,
                #                       cfg))
                # bmu = int(
                #     compute_bmu_current_timestep_only(replay_state, replay_mem,
                #                                       action_bmu, cfg))

                # if bmu == bmu_refined:
                #     print("bmu created properly")

        replay_mem = update_memory(replay_mem)
        replay_mem = add_memory(replay_mem, mem_vec[L+i])


    seen_bmus = set()
    bmus = []
    for _ in range(int(max_replay)):
        bmu = None
        replay_mem = rollback_memory_state(replay_mem, mem_vec, L)
        # oldest → newest frames
        for j in range(L):
            replay_mem = update_memory(replay_mem)
            replay_mem = add_memory(replay_mem, mem_vec[L+j])
            prev_bmu = bmu
            mem_view = memory_view_at_global_timestep(ms, L-j-1)
            action_bmu = int(am[j])

            if cfg.memory_replay:
                bmu = int(compute_bmu(replay_state, replay_mem, action_bmu, cfg))
            else:
                bmu = int(compute_bmu_current_timestep_only(replay_state, replay_mem, action_bmu, cfg))

            # if bmu in seen_bmus:
            #     break
            seen_bmus.add(bmu)
            bmus.append(bmu)
            # if bmu == bmu_refined:
            #     print(f"prev {prev_bmu} -> {action_bmu} -> bmu: {bmu}")
            #
            # elif prev_bmu == bmu_refined:
            #     print(f"prev {prev_bmu} -> {action_bmu} -> bmu: {bmu}")
            seen_bmus.add(bmu)
            #

            if prev_bmu is not None:
                replay_state.g = _update_edges_with_actions(
                    replay_state.g, prev_bmu=prev_bmu, bmu=bmu,
                    action_bmu=action_bmu, action_map=action_map,
                    gaussian_shape=cfg.gaussian_shape, beta=cfg.action_lr
                )
                # print(bmus)

    # if bmu_refined not in seen_bmus:
    #     raise EnvironmentError("added bmu not seen in replay")
    return replay_state.g, int(prev_bmu), bmu




def _still_aliased(g: Graph, node: int | None, action_bmu: int) -> bool:
    if node is None:
        return False
    return _is_aliased(g, _predict(g, int(node), int(action_bmu)))


def opposite_actions(action_map: ActionMap, action_mem: List[int] ) -> bool:
    if len(action_mem) < 2:
        return False
    curr_action_vec = action_map.state.codebook[action_mem[-1]]
    prev_action_vec = action_map.state.codebook[action_mem[-2]]

    reference_vectors = [np.array([0,1,0,1]), np.array([1,0,1,0])]

    combined_vec = curr_action_vec + prev_action_vec

    for ref_vec in reference_vectors:
        if np.allclose(ref_vec, combined_vec):
            return True
    return False


def add_temporal_edge_at_first_gap_clone(g: Graph, ms: MemoryState, v: int,
                                         preds: Array
                                         ) -> tuple[Graph, int, bool]:
    """
    Oldest-first. Create a NEW node that extends v's mem_adj at the first zero column.
    Returns:
        g_out, new_idx, created
        - If an identical node already exists, returns its index and created=False.
        - If no gap or no unique active sensor, returns (g, v, False).
    """
    L = int(ms.length)
    v = int(v)

    # find first gap for v
    t0 = first_zero_col_node(g, ms, v)
    if t0 is None:
        return g, v, False  # no gap to fill

    # pred_roots = []
    # for pred in preds:
    #     mem_adj_pred = g.node_features["mem_adj"][pred]
    #
    #     act = mem_adj_pred[0:ms.sensory_n_nodes]
    #     pred_root = np.argmax(act)
    #     pred_roots.append(pred_root)
    #
    # pred_roots = np.array(pred_roots)
    # # pick the active sensor at t0 (exactly one required)
    act = activations_at_t(ms, t0)                  # (S,)
    # act[pred_roots] = 0
    sensors = np.flatnonzero(act > 0)
    if sensors.size != 1:
        return g, v, False  # 0 or >1 actives -> skip (or raise if you prefer)
    u = int(sensors[0])

    # build updated mem_adj row (copy from v, then set (u,t0))
    M = g.node_features["mem_adj"]
    mem_len = int(M.shape[1])
    if mem_len % L != 0:
        raise ValueError(f"mem_len={mem_len} not divisible by L={L}")

    row = M[v].astype(np.float32, copy=True)
    idx = mem_id(u, t0, L)
    row[idx] = 1.0

    # deduplicate
    reuse = _find_matching_node(g, row)
    if reuse is not None:
        return g, int(reuse), False

    # spawn new node with this mem_adj and set the temporal bit again via API
    new_idx = g.n
    g = _add_node(g, ms, u=u, t=t0, mem_adj=row)    # _add_node will also set (u,t0)
    return g, int(new_idx), True


def first_zero_col_node(g: Graph, ms: MemoryState, node_idx: int) -> int | None:
    """
    Return the smallest t in [0, L-1] where mem_adj for `node_idx` is all zeros.
    Returns None if every column has at least one nonzero.

    Assumes g.node_features["mem_adj"] has shape (g.n, S*L).
    """
    M = g.node_features.get("mem_adj")
    if M is None:
        raise KeyError("mem_adj feature missing")
    node_idx = int(node_idx)
    if not (0 <= node_idx < g.n):
        raise IndexError("node_idx out of range")

    L = int(ms.length)
    SL = int(M.shape[1])
    if SL % L != 0:
        raise ValueError(f"mem_adj width {SL} not divisible by L={L}")

    row = M[node_idx]
    S = SL // L
    mat = row.reshape(S, L)                 # (S, L)
    zeros = np.flatnonzero(~np.any(mat != 0, axis=0))
    return int(zeros[0]) if zeros.size else None

def novelty_for_prev(ms: MemoryState, g: Graph, prev_bmu: int) -> float:
    mem_adj_row = g.node_features["mem_adj"][prev_bmu].astype(np.float32, copy=False)
    act_mem = ms.activations.astype(np.float32, copy=False)

    # Expected indices for this node:
    expected_mask = mem_adj_row > 0.0  # bool mask over S*L

    # Overlap between current activation and this node's expected pattern
    overlap = (act_mem * expected_mask).sum()
    total = act_mem.sum() + 1e-6

    # 0 → fully explained by prev_bmu; 1 → completely novel
    return 1.0 - float(overlap / total)

def _should_resolve_alias(
    ms: MemoryState,
    g: Graph,
    prev_bmu: int,
    novelty_history: List[float],
    cfg: LatentParams,
) -> bool:
    # Compute current novelty
    nov = novelty_for_prev(ms, g, prev_bmu)
    novelty_history.append(nov)

    # keep only last K values
    K = getattr(cfg, "alias_novelty_window", 5)
    if len(novelty_history) > K:
        del novelty_history[:-K]

    # Require sustained novelty above threshold
    thr = getattr(cfg, "alias_novelty_threshold", 0.3)
    min_count = getattr(cfg, "alias_min_novel_steps", 3)

    high = [x for x in novelty_history if x >= thr]
    return len(high) >= min_count

def update_bmu_memory_adj(
    ms: MemoryState,
    g: Graph ,
    bmu: int,):

    diff = g.node_features["mem_adj"][bmu] - ms.activations.astype(np.float32)
    print(diff)

def latent_step(
    ms: MemoryState,
    mem_vec: List[int],
    state: LatentState,
    action_bmu: int,
    cfg: LatentParams,
    action_map: ActionMap,
    action_mem: List[int],
    state_mem: List[int]
) -> tuple[LatentState, int, List[int]]:
    g = state.g

    bmu_now = compute_bmu(state, ms, action_bmu, cfg=cfg)


    # print("bmu_now", bmu_now)
    mapping = np.arange(g.n, dtype=np.int32)

    # assert np.array_equal(ms.activations, memory_view_at_global_timestep(ms,
    #                                                          4).activations),\
    #     (f"{ms.activations} vs"
    #                                                      f" "
    #      f"{memory_view_at_global_timestep(ms,4).activations}")



    if bmu_now >= state.g.n or mapping[bmu_now] == -1:
        bmu_now = compute_bmu(state, ms, action_bmu, cfg=cfg)
    else:
        bmu_now = mapping[bmu_now]

    if state.prev_bmu is None:
        # no previous state yet
        next_prev_bmu = int(bmu_now)
    else:
        resolved_prev = state.prev_bmu

        preds = _predict(state.g, int(state.prev_bmu), action_bmu)


        is_current_aliased = False
        for a in range(action_map.state.codebook.shape[0]):
            current_preds = _predict(state.g, bmu_now, a)
            if current_preds.size > 1:
                is_current_aliased = True
                # state.average_not_surprise = max(state.average_not_surprise -
                #                                  1 ,0)
                # # state.average_not_surprise -= 1
            else:
                is_current_aliased = False
                # state.average_not_surprise = min(state.average_not_surprise +
                #                                  1 ,6)

        # if is_current_aliased:
        #     print(True)

        ambiguity_score = state.g.node_features["ambiguity_score"][state.prev_bmu, action_bmu]
        ambiguity_score = float(cfg.ambiguity_decay) * float(ambiguity_score)
        if preds.size > 1:
            ambiguity_score += 1.0
        state.g.node_features["ambiguity_score"][state.prev_bmu, action_bmu] = ambiguity_score

        if ambiguity_score > cfg.ambiguity_threshold:
            # print(f"ALIASED {state.prev_bmu}")
            state.preds.append(preds)
            # print(f"seen preds: {state.preds}")
            # resolve aliasing of the previous state chain using replay
            # """ remove all connections"""
            # for i in range(g.n):
            #     removed = range(g.n)
            #     g = _remove_aliased_connections(g, i, removed)

            """ remove all connections to and from bmu_prev"""
            g = _remove_aliased_connections(g, resolved_prev,range(g.n))
            for i in range(g.n):
                g = _remove_aliased_connections(g, i, resolved_prev)

            state.g.node_features["ambiguity_score"][state.prev_bmu, action_bmu] = 0

            # g = _remove_aliased_connections(
            #     g,
            #     bmu_prev=int(resolved_prev),
            #     predicted_nodes=preds,
            # )


            state.g = g


            g, resolved_prev, bmu_now = _resolve_prev_with_memory_replay(
                state=state,
                ms=ms,
                mem_vec=mem_vec,
                prev_bmu=state.prev_bmu,
                bmu=bmu_now,
                action_mem=action_mem,
                action_map=action_map,
                gaussian_shape=cfg.gaussian_shape,
                max_replay=2,
                cfg=cfg,
                preds=preds,
            )

            mapping = np.arange(g.n, dtype=np.int32)

            # for i in range(state.g.n):
            #     _debug_print_latent_mem_adj(g,ms,i)
            # _debug_print_latent_action_adj(g)

            state_mem[-1] = resolved_prev

        # learn live edge from resolved_prev -> bmu_now under current action_bmu
        g = _update_edges_with_actions(
            g,
            prev_bmu=resolved_prev,
            bmu=bmu_now,
            action_bmu=action_bmu,
            action_map=action_map,
            gaussian_shape=cfg.gaussian_shape,
            beta = cfg.action_lr
        )
        #
        if state.step_idx % 1 == 0:
            g, mapping = age_maintenance(resolved_prev,bmu_now,g,cfg,
                                         exclude=np.arange(
                                             ms.sensory_n_nodes))


            bmu_now = mapping[bmu_now]

            state.prev_activations = _remap_prev_activations(
                state.prev_activations, mapping, g.n)

    # print(f"STEP {state.step_idx} | {state.prev_bmu} -> {action_bmu} ->"
    #       f" {bmu_now}")

    # mark active for plotting
    g = _set_activation(g, bmu_now)




    # commit
    state_mem.append(bmu_now)
    state.g = g

    if bmu_now >= state.g.n or mapping[bmu_now] == -1:
        bmu_now = compute_bmu(state, ms, action_bmu, cfg)
    else:
        bmu_now = mapping[bmu_now]

    # update_bmu_memory_adj()

    state.prev_bmu = int(bmu_now)
    state.step_idx += 1
    state.mapping = mapping

    # if state.step_idx % 10 ==0:
        # for i in np.arange(state.g.n-1):
    #         for j in np.arange(i+1, state.g.n):
    #             mem_adj_i = state.g.node_features["mem_adj"][i]
    #             mem_adj_j = state.g.node_features["mem_adj"][j]
    #             print(i, j, stats.wasserstein_distance(mem_adj_i, mem_adj_j),
    #                   stats.wasserstein_distance(state.g.adj[i], state.g.adj[
    #                       j]))
    #     for i in range(state.g.n):
    #         _debug_print_latent_mem_adj(g,ms,i)
    #     _debug_print_latent_action_adj(g)
    # #





    # print(f"{g.n} nodes @ step {state.step_idx}")
    return state, int(bmu_now), state_mem

def latent_step_frozen(
    ms: MemoryState,
    mem_vec: List[int],
    state: LatentState,
    action_bmu: int,
    cfg: LatentParams,
    action_map: ActionMap,
    action_mem: List[int],
    state_mem: List[int]
) -> tuple[LatentState, int, List[int]]:
    g = state.g

    bmu_now = compute_bmu(state, ms, action_bmu, cfg=cfg)


    # print("bmu_now", bmu_now)
    mapping = np.arange(g.n, dtype=np.int32)

    # assert np.array_equal(ms.activations, memory_view_at_global_timestep(ms,
    #                                                          4).activations),\
    #     (f"{ms.activations} vs"
    #                                                      f" "
    #      f"{memory_view_at_global_timestep(ms,4).activations}")

    g = _set_activation(g, bmu_now)

    # commit
    state_mem.append(bmu_now)
    state.g = g

    if bmu_now >= state.g.n or mapping[bmu_now] == -1:
        bmu_now = compute_bmu(state, ms, action_bmu, cfg)
    else:
        bmu_now = mapping[bmu_now]

    # update_bmu_memory_adj()

    state.prev_bmu = int(bmu_now)
    state.step_idx += 1
    state.mapping = mapping

    # print(f"{g.n} nodes @ step {state.step_idx}")
    return state, int(bmu_now), state_mem


def latent_step_predict_only(
    ms: MemoryState,
    state: LatentState,
    action_bmu: int,
    cfg: LatentParams,
    state_mem: List[int],
) -> tuple[LatentState, int, List[int]]:
    bmu_now = compute_bmu(state, ms, action_bmu, cfg=cfg)
    mapping = np.arange(state.g.n, dtype=np.int32)
    state_mem.append(bmu_now)
    if bmu_now >= state.g.n or mapping[bmu_now] == -1:
        bmu_now = compute_bmu(state, ms, action_bmu, cfg)
    else:
        bmu_now = mapping[bmu_now]
    state.prev_bmu = int(bmu_now)
    state.step_idx += 1
    state.mapping = mapping
    return state, int(bmu_now), state_mem
