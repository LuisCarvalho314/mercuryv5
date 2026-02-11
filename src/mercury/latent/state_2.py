# latent/state_2.py
from copy import deepcopy, copy
from dataclasses import dataclass, replace, field
from typing import Any, Tuple, List

import numpy as np
import scipy.stats as stats

from ..action_map.adapter import ActionMap
from mercury.graph.core import Graph
from mercury.graph.maintenance import age_maintenance
from .params import LatentParams
from ..memory.state import (
    MemoryState,
    mem_id,
    activations_at_t,
    memory_view_at_global_timestep,
    init_mem,
    add_memory,
    update_memory,
)
from ..sensory.state import SensoryState

type Array = np.ndarray


# ============================================================
# mem_adj K-sets helpers (stored flattened in graph node_features)
# ============================================================

def _mem_len_from_ms(ms: MemoryState) -> int:
    # In your code, mem_len == ms.gs.n == S*L
    return int(ms.gs.n)


def _max_sets_from_graph(g: Graph, mem_len: int) -> int:
    flat_dim = int(g.node_features["mem_adj"].shape[1])
    if flat_dim % mem_len != 0:
        raise ValueError(f"mem_adj flat dim {flat_dim} not divisible by mem_len {mem_len}")
    return flat_dim // mem_len


def mem_adj_all_view(g: Graph, *, max_mem_adj_sets: int, mem_len: int) -> np.ndarray:
    # (n, K, mem_len)
    return g.node_features["mem_adj"].reshape(g.n, max_mem_adj_sets, mem_len)


def mem_adj_matrix(g: Graph, *, max_mem_adj_sets: int, mem_len: int, set_index: int = 0) -> np.ndarray:
    # (n, mem_len) for a chosen set
    return mem_adj_all_view(g, max_mem_adj_sets=max_mem_adj_sets, mem_len=mem_len)[:, int(set_index), :]


def mem_adj_row(
    g: Graph,
    node_index: int,
    *,
    max_mem_adj_sets: int,
    mem_len: int,
    set_index: int = 0,
) -> np.ndarray:
    return mem_adj_all_view(g, max_mem_adj_sets=max_mem_adj_sets, mem_len=mem_len)[int(node_index), int(set_index), :]


def set_mem_adj_row(
    g: Graph,
    node_index: int,
    row: np.ndarray,
    *,
    max_mem_adj_sets: int,
    mem_len: int,
    set_index: int,
) -> None:
    row = np.asarray(row, dtype=np.float32).reshape(mem_len)
    view = mem_adj_all_view(g, max_mem_adj_sets=max_mem_adj_sets, mem_len=mem_len)
    view[int(node_index), int(set_index), :] = row


def mem_adj_used_sets(g: Graph) -> np.ndarray:
    return g.node_features["mem_adj_used_sets"].astype(np.int32).reshape(-1)


def allocate_mem_adj_set(
    g: Graph,
    node_index: int,
    *,
    max_mem_adj_sets: int,
) -> int:
    used = int(g.node_features["mem_adj_used_sets"][int(node_index)])
    if used >= int(max_mem_adj_sets):
        raise ValueError(f"node {node_index} exceeded max_mem_adj_sets={max_mem_adj_sets}")
    g.node_features["mem_adj_used_sets"][int(node_index)] = np.int32(used + 1)
    return used


# ============================================================
# State containers
# ============================================================

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
    ambiguity_threshold = 10
    prev_activations: np.ndarray | None = None

    # K sets per node; compute_bmu selects the best set each step
    max_mem_adj_sets: int = 8
    last_mem_adj_set: int = 0


# ============================================================
# Graph feature registration
# ============================================================

def _register_features(g: Graph, mem_len: int, max_mem_adj_sets: int) -> Graph:
    """
    Register required features.

    Node features
    -------------
    activation         : (n,)
    mem_adj            : (n, max_mem_adj_sets * mem_len)  flattened
    mem_adj_used_sets  : (n,) int32
    ambiguity_score    : (n,)

    Edge features
    -------------
    age                : (n, n, 1)
    action             : (n, n, 1)
    """
    g.register_node_feature("activation", 1, init_value=0)
    g.register_node_feature("mem_adj", max_mem_adj_sets * mem_len, init_value=0)
    g.register_node_feature("mem_adj_used_sets", 1, dtype=np.int32, init_value=0)
    g.register_node_feature("ambiguity_score", 1, init_value=0)

    g.register_edge_feature("age", 1, dtype=np.int32, init_value=0)
    g.register_edge_feature("action", 1, dtype=np.int32, init_value=0)
    return g


def build_initial_graph(ms: MemoryState, max_mem_adj_sets: int) -> Graph:
    g = Graph(directed=True)
    mem_len = _mem_len_from_ms(ms)
    _register_features(g, mem_len, max_mem_adj_sets)

    for n in np.arange(ms.sensory_n_nodes):
        g = _add_node(g, ms, u=int(n), t=0, mem_adj=None, max_mem_adj_sets=max_mem_adj_sets, set_index=0)

    return g


def init_latent_state(ms: MemoryState, max_mem_adj_sets: int = 8) -> LatentState:
    g = build_initial_graph(ms, max_mem_adj_sets=max_mem_adj_sets)
    return LatentState(g, mapping=np.arange(g.n), max_mem_adj_sets=max_mem_adj_sets, last_mem_adj_set=0)


# ============================================================
# Node/edge construction
# ============================================================

def _add_node(
    g: Graph,
    ms: MemoryState,
    u: int,
    t: int = 0,
    mem_adj: Array | None = None,
    *,
    max_mem_adj_sets: int,
    set_index: int = 0,
) -> Graph:
    """
    Add a new latent node corresponding to memory unit u at timestep t.

    Stored mem_adj is flattened over sets:
        mem_adj_flat: (max_mem_adj_sets * mem_len,)

    This initialises ONLY one set (set_index) for the node (typically 0),
    and sets mem_adj_used_sets = 1.
    """
    mem_len = _mem_len_from_ms(ms)

    if mem_adj is None:
        mem_adj_init = np.zeros(mem_len, dtype=np.float32)
    else:
        mem_adj_arr = np.asarray(mem_adj, dtype=np.float32)
        if mem_adj_arr.shape != (mem_len,):
            raise ValueError(f"mem_adj shape {mem_adj_arr.shape} does not match expected {(mem_len,)}")
        mem_adj_init = mem_adj_arr

    flat = np.zeros(max_mem_adj_sets * mem_len, dtype=np.float32)
    si = int(set_index)
    flat[si * mem_len : (si + 1) * mem_len] = mem_adj_init

    new_node = g.add_node(
        node_feat={
            "activation": np.array(0.0, dtype=np.float32),
            "mem_adj": flat,
            "mem_adj_used_sets": np.array(1, dtype=np.int32),
        },
        edge_defaults_for_new_node={
            "age": np.array([0], dtype=np.int32),
            "action": np.array([0], dtype=np.int32),
        },
    )

    g = _add_temporal_edge(
        g,
        u=int(u),
        v=int(new_node),
        t=int(t),
        L=int(ms.length),
        max_mem_adj_sets=max_mem_adj_sets,
        mem_len=mem_len,
        set_index=si,
    )
    return g


def _add_temporal_edge(
    g: Graph,
    u: int,
    v: int,
    t: int,
    L: int,
    *,
    max_mem_adj_sets: int,
    mem_len: int,
    set_index: int = 0,
) -> Graph:
    n_lat = g.n
    if not (0 <= v < n_lat):
        raise ValueError(f"v={v} out of range [0,{n_lat-1}]")
    if not (0 <= t < L):
        raise ValueError(f"t={t} out of range [0,{L-1}]")
    if mem_len % L != 0:
        raise ValueError(f"mem_len={mem_len} not divisible by L={L}")

    idx = mem_id(int(u), int(t), int(L))
    if not (0 <= idx < mem_len):
        raise ValueError(f"mem index {idx} out of range [0,{mem_len-1}]")

    offset = int(set_index) * mem_len + idx
    g.node_features["mem_adj"][int(v), offset] = np.float32(1.0)
    return g


# ============================================================
# BMU selection
# ============================================================

def compute_bmu(state: LatentState, ms: MemoryState, action_bmu: int) -> int:
    """
    Compute latent activations from memory for EACH mem_adj set, apply contender/candidate masks,
    combine with trace diffusion, and choose the (set,node) with maximum final score.
    Returns the node index (BMU). The chosen set is stored in state.last_mem_adj_set.
    """
    g = state.g
    n_lat = int(g.n)

    mem_len = _mem_len_from_ms(ms)
    L = int(ms.length)
    if mem_len % L != 0:
        raise ValueError(f"mem_len={mem_len} not divisible by L={L}")

    K = int(state.max_mem_adj_sets)
    # Defensive: if graph was created with a different K, adapt to graph K
    K_graph = _max_sets_from_graph(g, mem_len)
    if K_graph != K:
        K = K_graph
        state.max_mem_adj_sets = K

    mem_adj_all = mem_adj_all_view(g, max_mem_adj_sets=K, mem_len=mem_len)  # (n, K, mem_len)

    act_mem = ms.activations.astype(np.float32, copy=False)  # (mem_len,)
    if act_mem.shape[0] != mem_len:
        raise ValueError(f"ms.activations len={act_mem.shape[0]} != mem_len={mem_len}")

    # --- trace diffusion terms (depend only on graph + prev_trace) ---
    trace_decay = 0.99
    trace_influence = 0.2
    action_trace_influence = 0.2

    prev_trace = state.prev_activations
    if prev_trace is None:
        prev_trace = np.zeros((n_lat,), dtype=np.float32)
    else:
        prev_trace = np.asarray(prev_trace, dtype=np.float32).reshape(-1)

    if prev_trace.shape[0] < n_lat:
        prev_trace = np.pad(prev_trace, (0, n_lat - prev_trace.shape[0]), mode="constant")
    elif prev_trace.shape[0] > n_lat:
        prev_trace = prev_trace[:n_lat]

    A_action_conditioned = _transition_for_action_strength(g, action_bmu)
    A_tilde = g.adj.astype(np.float32) + np.identity(n_lat, dtype=np.float32)
    degrees = np.sum(A_tilde, axis=1).astype(np.float32)
    degrees_inv_sqrt = 1.0 / np.sqrt(np.maximum(degrees, 1e-12))
    A_norm = (A_tilde * degrees_inv_sqrt[None, :]) * degrees_inv_sqrt[:, None]

    action_diffusion_from_trace_base = A_action_conditioned @ prev_trace
    diffusion_from_trace_base = A_norm @ prev_trace

    # --- candidate evidence mask indices at t=0 (your current rule) ---
    t_now = 0
    idx_now = t_now + np.arange(ms.sensory_n_nodes, dtype=np.int64) * L  # (S,)
    a_now = np.maximum(act_mem[idx_now], 0.0).astype(np.float32)  # (S,)
    active_sensors = (a_now != 0)  # (S,)

    best_score = -np.float32(np.inf)
    best_node = 0
    best_set = 0
    best_masked_for_trace = np.zeros((n_lat,), dtype=np.float32)

    # Evaluate all sets
    for set_index in range(K):
        mem_adj = mem_adj_all[:, set_index, :]  # (n, mem_len)

        # Raw latent scores
        act = act_mem @ mem_adj.T  # (n,)

        # Per-timestep contributions summed over sensors
        contrib = np.empty((n_lat, L), dtype=np.float32)
        for t in range(L):
            idx = np.arange(t, mem_len, L, dtype=np.int64)
            contrib[:, t] = (mem_adj[:, idx] * act_mem[idx]).sum(axis=1)

        # Contender rule: must include t=0 and be a contiguous prefix 0..k
        has_t0 = contrib[:, 0] > 0
        M = contrib > 0
        has_any = M.any(axis=1)

        last_from_end = np.argmax(M[:, ::-1], axis=1)  # 0 if all False
        last_idx = (L - 1) - last_from_end
        pos = np.arange(L)
        within_prefix = pos <= last_idx[:, None]
        prefix_all_true = np.all(~within_prefix | M, axis=1)

        contenders = has_t0 & has_any & prefix_all_true

        masked = act.copy()
        masked[~contenders] = 0.0

        # Candidate mask: connect to all active sensors at t=0 (if any)
        mem_adj_now = mem_adj[:, idx_now]  # (n, S)
        if np.any(active_sensors):
            candidate_mask = np.all(mem_adj_now[:, active_sensors] != 0, axis=1)  # (n,)
        else:
            candidate_mask = np.ones((n_lat,), dtype=bool)

        masked[~candidate_mask] = 0.0

        diffusion_from_trace = diffusion_from_trace_base.copy()
        action_diffusion_from_trace = action_diffusion_from_trace_base.copy()
        diffusion_from_trace[~candidate_mask] = 0.0
        action_diffusion_from_trace[~candidate_mask] = 0.0

        # Normalise
        masked_n = norm(masked)
        diffusion_n = norm(diffusion_from_trace)
        action_diffusion_n = norm(action_diffusion_from_trace)

        w_mem, w_trace, w_action = _convex_mixture_weights(trace_influence, action_trace_influence)
        scored = (w_mem * masked_n) + (w_trace * diffusion_n) + (w_action * action_diffusion_n)

        node = int(np.argmax(scored))
        score = float(scored[node])

        if score > float(best_score):
            best_score = np.float32(score)
            best_node = node
            best_set = set_index
            best_masked_for_trace = masked_n.astype(np.float32, copy=False)

    # Store the set that won
    state.last_mem_adj_set = int(best_set)

    # Debug prints (optional): comment out if too verbose
    # print(f"chosen_set {best_set} | bmu {best_node} | score {best_score:.3f}")

    # Update trace using the winning set's masked evidence
    state.prev_activations = _update_exponential_trace(
        prev_trace=prev_trace,
        current_signal=best_masked_for_trace,
        trace_decay=trace_decay,
        clip_min=0.0,
    )

    return int(best_node)


# ============================================================
# Misc utilities (your originals)
# ============================================================

def softmax_neg_distance(distances: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    x = -distances / temperature
    x = x - np.max(x)  # stability
    e = np.exp(x)
    return e / np.sum(e)


def norm(x):
    s = float(np.sum(x))
    if s == 0.0:
        return x
    return x / s


def _convex_mixture_weights(alpha: float, beta: float) -> Tuple[float, float, float]:
    a = float(np.clip(alpha, 0.0, 1.0))
    b = float(np.clip(beta, 0.0, 1.0))
    w_mem = 1.0 - a
    w_trace = a * (1.0 - b)
    w_action = a * b
    return w_mem, w_trace, w_action


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
    predicted_mass_sl = node_edge_weights_sl * observed_mass_sl

    aKa = np.sum(observed_mass_sl * (observed_mass_sl @ K.T), axis=1)
    bKb = np.sum(predicted_mass_sl * (predicted_mass_sl @ K.T), axis=1)
    aKb = np.sum(observed_mass_sl * (predicted_mass_sl @ K.T), axis=1)
    dist_per_sensor = np.sqrt(np.maximum(aKa + bKb - 2.0 * aKb, 0.0))  # (S,)

    mass_pred = np.sum(predicted_mass_sl, axis=1)                      # (S,)
    w = mass_pred / (mass_pred + kappa + eps)                          # (S,) in [0,1)

    return float(np.mean(dist_per_sensor / (w + eps)))


def wasserstein_per_sensor(
    flat_a: np.ndarray,
    flat_b: np.ndarray,
    sensory_count: int,
    strip_length: int,
    epsilon: float = 1e-12,
    normalise: bool = True,
) -> np.ndarray:
    A = flat_a.reshape(sensory_count, strip_length)
    B = flat_b.reshape(sensory_count, strip_length)

    if normalise:
        A = A / (A.sum(axis=1, keepdims=True) + epsilon)
        B = B / (B.sum(axis=1, keepdims=True) + epsilon)

    cdf_a = np.cumsum(A, axis=1)
    cdf_b = np.cumsum(B, axis=1)

    w1_per_sensor = np.sum(np.abs(cdf_a - cdf_b), axis=1)  # shape (S,)
    return w1_per_sensor


def softmax(x: np.ndarray) -> np.ndarray:
    return np.exp(x) / np.sum(np.exp(x), axis=0)


def _transition_for_action_strength(graph: Graph, action_bmu: int) -> np.ndarray:
    strengths = graph.adj.astype(np.float32, copy=False)
    actions = graph.edge_features["action"][:, :, 0].astype(np.int32, copy=False)

    mask = (strengths != 0.0) & (actions == int(action_bmu))
    transition = np.zeros_like(strengths, dtype=np.float32)
    transition[mask] = strengths[mask]
    return transition


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
    Legacy helper: returns masked activations for the SINGLE best set (same selection logic as compute_bmu,
    but returns the masked vector instead of the index).
    """
    # Choose BMU (also updates state.last_mem_adj_set / trace)
    bmu = compute_bmu(state, ms, action_bmu=0)

    mem_len = _mem_len_from_ms(ms)
    K = int(state.max_mem_adj_sets)
    set_index = int(state.last_mem_adj_set)

    mem_adj = mem_adj_matrix(state.g, max_mem_adj_sets=K, mem_len=mem_len, set_index=set_index)
    act_mem = ms.activations.astype(np.float32, copy=False)
    act = act_mem @ mem_adj.T

    # A simple mask: return act (or replicate your contender logic if needed)
    masked = act.copy()
    return masked


def _calc_activation(d: Array | float, gaussian_shape: float) -> Array | float:
    d = np.asarray(d, dtype=np.float32)
    sigma = float(gaussian_shape)
    if sigma <= 0:
        out = np.zeros_like(d, dtype=np.float32)
    else:
        out = np.exp(-(d * d) / (2.0 * sigma * sigma)).astype(np.float32)
    return float(out) if out.ndim == 0 else out


def _remove_aliased_connections(g: Graph, bmu_prev: int, predicted_nodes: Array) -> Graph:
    dst = np.asarray(predicted_nodes, dtype=np.int32).ravel()
    if dst.size == 0:
        return g
    src = np.full(dst.size, int(bmu_prev), dtype=np.int32)
    g.remove_edges(src, dst)
    return g


def _update_edge(
    g: Graph,
    u: int,
    v: int,
    prior_row: Array,
    observed_row: Array,
    action_bmu: int,
    beta: float,
    gaussian_shape: float,
) -> Graph:
    dist = float(np.linalg.norm(prior_row - observed_row))
    sim = float(_calc_activation(dist, gaussian_shape))
    current = float(g.adj[int(u), int(v)])
    new_w = current + float(beta) * (sim - current)

    g.add_edge(
        int(u), int(v), weight=new_w,
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
    act[int(bmu)] = 1.0
    g.set_node_feat("activation", act)
    return g


def _update_edges_with_actions(
    g: Graph,
    prev_bmu: int,
    bmu: int,
    action_bmu: int,
    action_map: ActionMap,
    gaussian_shape: int,
) -> Graph:
    observed_row = np.asarray(action_map.state.codebook[action_bmu], np.float32)
    prior_row = _edge_action_row(g, int(prev_bmu), int(bmu), am=action_map)
    g = _update_edge(
        g, int(prev_bmu), int(bmu), prior_row, observed_row,
        action_bmu=int(action_bmu), beta=0.1, gaussian_shape=float(gaussian_shape),
    )
    return g


def _predict(g: Graph, bmu: int, action_bmu: int) -> Array:
    actions_out = g.edge_features["action"][int(bmu), :, 0]  # (n,)
    edge_exists = g.adj[int(bmu), :] != 0                    # (n,) bool
    matches_action = actions_out == int(action_bmu)          # (n,) bool

    mask = edge_exists & matches_action
    idx = np.flatnonzero(mask).astype(np.int32, copy=False)
    print(f"prev {bmu} -> {action_bmu} -> preds: {idx}")
    return idx


def _is_aliased(g: Graph, action_bmu_out: Array) -> bool:
    return action_bmu_out.shape[0] > 1


def _find_matching_node(g: Graph, ms: MemoryState, row: np.ndarray, *, set_index: int = 0) -> int | None:
    mem_len = _mem_len_from_ms(ms)
    K = _max_sets_from_graph(g, mem_len)
    M = mem_adj_matrix(g, max_mem_adj_sets=K, mem_len=mem_len, set_index=set_index)  # (n, mem_len)
    row = np.asarray(row, dtype=np.float32).reshape(mem_len)
    hit = np.flatnonzero(np.all(M == row[None, :], axis=1))
    return int(hit[0]) if hit.size else None


def _debug_print_latent_action_adj(g: Graph):
    for u in range(g.n):
        for v in range(g.n):
            if g.adj[u, v] != 0.0:
                action = g.edge_features["action"][u, v]
                print(f"u:{u} v:{v} action:{action}")


def _debug_print_latent_mem_adj(g: Graph, ms: MemoryState, node_idx: int, *, set_index: int = 0) -> None:
    L = int(ms.length)
    mem_len = _mem_len_from_ms(ms)
    S = mem_len // L
    K = _max_sets_from_graph(g, mem_len)

    row = mem_adj_row(g, int(node_idx), max_mem_adj_sets=K, mem_len=mem_len, set_index=set_index).astype(np.float32, copy=False)

    pairs = []
    for u in range(S):
        for t in range(L):
            idx = u * L + t
            if row[idx] > 0:
                pairs.append((t, u))

    pairs.sort(key=lambda x: x[0])
    desc = ", ".join([f"(sensory {u}, t={t})" for (t, u) in pairs])
    print(f"latent {node_idx} [set {set_index}] encodes {desc}")


def rollback_memory_state(ms: MemoryState, mem_vec: Array, length) -> MemoryState:
    for i in range(length):
        ms = update_memory(ms)
        ms = add_memory(ms, mem_vec[i])
    return ms


def rollback_k_memory_state(ms: MemoryState, mem_vec: Array, k: int, L: int) -> MemoryState:
    length = 2 * L
    mem_vec = np.asarray(mem_vec[-length:])
    replay_mem = init_mem(ms.sensory_n_nodes, ms.length)
    replay_mem = rollback_memory_state(replay_mem, mem_vec, length - k)
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
    if state.step_idx < 2 * L:
        return replay_state.g, int(prev_bmu), bmu

    am = list(action_mem)[-L:]  # oldest -> newest
    mem_vec = np.asarray(mem_vec[-2 * L:])
    replay_mem = init_mem(ms.sensory_n_nodes, ms.length)

    replay_mem = rollback_memory_state(replay_mem, mem_vec, L)

    if len(am) < L:
        raise ValueError(f"action_mem length {len(am)} < L={L}")

    for i in range(L):
        action_bmu_i = int(am[i])
        bmu = int(compute_bmu(replay_state, replay_mem, action_bmu_i))

        if bmu == prev_bmu:
            nov = novelty_for_prev(replay_mem, replay_state.g, int(prev_bmu))
            if True:
                g_new, bmu_refined, created = add_temporal_edge_at_first_gap_clone(
                    replay_state.g, replay_mem, int(prev_bmu), preds
                )
                state.ambiguity_threshold += 1
                replay_state.g = g_new
                _align_prev_activations_to_graph(replay_state)

                bmu = int(compute_bmu(replay_state, replay_mem, action_bmu_i))
                if bmu == bmu_refined:
                    print("bmu created properly")

        replay_mem = update_memory(replay_mem)
        replay_mem = add_memory(replay_mem, mem_vec[L + i])

    seen_bmus = set()
    bmus = []
    for _ in range(int(max_replay)):
        bmu = None
        replay_mem = rollback_memory_state(replay_mem, mem_vec, L)
        for j in range(L):
            replay_mem = update_memory(replay_mem)
            replay_mem = add_memory(replay_mem, mem_vec[L + j])
            prev_bmu = bmu
            _ = memory_view_at_global_timestep(ms, L - j - 1)
            action_bmu_j = int(am[j])

            bmu = int(compute_bmu(replay_state, replay_mem, action_bmu_j))
            bmus.append(bmu)
            seen_bmus.add(bmu)

            if prev_bmu is not None:
                replay_state.g = _update_edges_with_actions(
                    replay_state.g,
                    prev_bmu=int(prev_bmu),
                    bmu=int(bmu),
                    action_bmu=int(action_bmu_j),
                    action_map=action_map,
                    gaussian_shape=cfg.gaussian_shape,
                )

    return replay_state.g, int(prev_bmu), bmu


def _still_aliased(g: Graph, node: int | None, action_bmu: int) -> bool:
    if node is None:
        return False
    return _is_aliased(g, _predict(g, int(node), int(action_bmu)))


def opposite_actions(action_map: ActionMap, action_mem: List[int]) -> bool:
    if len(action_mem) < 2:
        return False
    curr_action_vec = action_map.state.codebook[action_mem[-1]]
    prev_action_vec = action_map.state.codebook[action_mem[-2]]

    reference_vectors = [np.array([0, 1, 0, 1]), np.array([1, 0, 1, 0])]
    combined_vec = curr_action_vec + prev_action_vec

    for ref_vec in reference_vectors:
        if np.allclose(ref_vec, combined_vec):
            return True
    return False


def add_temporal_edge_at_first_gap_clone(
    g: Graph,
    ms: MemoryState,
    v: int,
    preds: Array,
) -> tuple[Graph, int, bool]:
    """
    Prototype behavior with K sets:
    - Prefer allocating a NEW mem_adj set on the SAME node v (if capacity remains),
      copying set 0 then filling the first gap bit.
    - Falls back to creating a NEW node only if no set slots remain.

    Returns:
        (g_out, node_index, created)
        node_index is the node whose representation was refined (often v).
    """
    L = int(ms.length)
    v = int(v)

    mem_len = _mem_len_from_ms(ms)
    K = _max_sets_from_graph(g, mem_len)

    # Find first gap in base set 0 for v
    t0 = first_zero_col_node(g, ms, v, set_index=0)
    if t0 is None:
        return g, v, False

    act = activations_at_t(ms, t0)  # (S,)
    sensors = np.flatnonzero(act > 0)
    if sensors.size != 1:
        return g, v, False
    u = int(sensors[0])

    # Build updated row from base set 0
    base_row = mem_adj_row(g, v, max_mem_adj_sets=K, mem_len=mem_len, set_index=0).astype(np.float32, copy=True)
    idx = mem_id(u, t0, L)
    base_row[idx] = 1.0

    used = int(g.node_features["mem_adj_used_sets"][v])
    if used < K:
        # Allocate a new set on the SAME node v
        new_set = allocate_mem_adj_set(g, v, max_mem_adj_sets=K)
        set_mem_adj_row(g, v, base_row, max_mem_adj_sets=K, mem_len=mem_len, set_index=new_set)
        return g, v, True

    # No set capacity left: fallback to old behavior (spawn a new node)
    reuse = _find_matching_node(g, ms, base_row, set_index=0)
    if reuse is not None:
        return g, int(reuse), False

    new_idx = g.n
    g = _add_node(g, ms, u=u, t=t0, mem_adj=base_row, max_mem_adj_sets=K, set_index=0)
    return g, int(new_idx), True


def first_zero_col_node(g: Graph, ms: MemoryState, node_idx: int, *, set_index: int = 0) -> int | None:
    """
    Return the smallest t in [0, L-1] where mem_adj for `node_idx` is all zeros (for a given set).
    Returns None if every column has at least one nonzero.
    """
    if "mem_adj" not in g.node_features:
        raise KeyError("mem_adj feature missing")
    node_idx = int(node_idx)
    if not (0 <= node_idx < g.n):
        raise IndexError("node_idx out of range")

    L = int(ms.length)
    mem_len = _mem_len_from_ms(ms)
    if mem_len % L != 0:
        raise ValueError(f"mem_len {mem_len} not divisible by L={L}")
    S = mem_len // L
    K = _max_sets_from_graph(g, mem_len)

    row = mem_adj_row(g, node_idx, max_mem_adj_sets=K, mem_len=mem_len, set_index=set_index)
    mat = row.reshape(S, L)
    zeros = np.flatnonzero(~np.any(mat != 0, axis=0))
    return int(zeros[0]) if zeros.size else None


def novelty_for_prev(ms: MemoryState, g: Graph, prev_bmu: int, *, set_index: int = 0) -> float:
    mem_len = _mem_len_from_ms(ms)
    K = _max_sets_from_graph(g, mem_len)

    mem_adj_row_prev = mem_adj_row(g, int(prev_bmu), max_mem_adj_sets=K, mem_len=mem_len, set_index=set_index).astype(np.float32, copy=False)
    act_mem = ms.activations.astype(np.float32, copy=False)

    expected_mask = mem_adj_row_prev > 0.0
    overlap = float((act_mem * expected_mask).sum())
    total = float(act_mem.sum()) + 1e-6

    return 1.0 - float(overlap / total)


def _should_resolve_alias(
    ms: MemoryState,
    g: Graph,
    prev_bmu: int,
    novelty_history: List[float],
    cfg: LatentParams,
) -> bool:
    nov = novelty_for_prev(ms, g, prev_bmu)
    novelty_history.append(nov)

    K = getattr(cfg, "alias_novelty_window", 5)
    if len(novelty_history) > K:
        del novelty_history[:-K]

    thr = getattr(cfg, "alias_novelty_threshold", 0.3)
    min_count = getattr(cfg, "alias_min_novel_steps", 3)

    high = [x for x in novelty_history if x >= thr]
    return len(high) >= min_count


def update_bmu_memory_adj(ms: MemoryState, g: Graph, bmu: int, *, set_index: int = 0):
    mem_len = _mem_len_from_ms(ms)
    K = _max_sets_from_graph(g, mem_len)
    row = mem_adj_row(g, int(bmu), max_mem_adj_sets=K, mem_len=mem_len, set_index=set_index)
    diff = row - ms.activations.astype(np.float32)
    print(diff)


def latent_step(
    ms: MemoryState,
    mem_vec: List[int],
    state: LatentState,
    action_bmu: int,
    cfg: LatentParams,
    action_map: ActionMap,
    action_mem: List[int],
    state_mem: List[int],
) -> tuple[LatentState, int, List[int]]:
    g = state.g

    bmu_now = compute_bmu(state, ms, action_bmu)

    mapping = np.arange(g.n, dtype=np.int32)

    if bmu_now >= state.g.n or mapping[bmu_now] == -1:
        bmu_now = compute_bmu(state, ms, action_bmu)
    else:
        bmu_now = mapping[bmu_now]

    if state.prev_bmu is None:
        next_prev_bmu = int(bmu_now)
    else:
        resolved_prev = state.prev_bmu

        preds = _predict(state.g, int(state.prev_bmu), action_bmu)

        is_current_aliased = False
        for a in range(action_map.state.codebook.shape[0]):
            current_preds = _predict(state.g, bmu_now, a)
            if current_preds.size > 1:
                is_current_aliased = True
            else:
                is_current_aliased = False

        if is_current_aliased:
            print(True)

        if preds.size > 1:
            state.g.node_features["ambiguity_score"][state.prev_bmu] += 1

        if state.g.node_features["ambiguity_score"][state.prev_bmu] > state.ambiguity_threshold:
            print(f"ALIASED {state.prev_bmu}")
            state.preds.append(preds)

            # remove all connections to and from bmu_prev
            g = _remove_aliased_connections(g, resolved_prev, range(g.n))
            for i in range(g.n):
                g = _remove_aliased_connections(g, i, resolved_prev)

            state.g.node_features["ambiguity_score"][state.prev_bmu] = 0
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

            for i in range(state.g.n):
                _debug_print_latent_mem_adj(g, ms, i, set_index=state.last_mem_adj_set)
            _debug_print_latent_action_adj(g)

            state_mem[-1] = resolved_prev

        g = _update_edges_with_actions(
            g,
            prev_bmu=resolved_prev,
            bmu=bmu_now,
            action_bmu=action_bmu,
            action_map=action_map,
            gaussian_shape=cfg.gaussian_shape,
        )

        if state.step_idx % 1 == 0:
            g, mapping = age_maintenance(
                resolved_prev,
                bmu_now,
                g,
                cfg,
                exclude=np.arange(ms.sensory_n_nodes),
            )

            bmu_now = mapping[bmu_now]
            state.prev_activations = _remap_prev_activations(state.prev_activations, mapping, g.n)

    print(f"STEP {state.step_idx} | {state.prev_bmu} -> {action_bmu} -> {bmu_now}")

    g = _set_activation(g, bmu_now)

    state_mem.append(bmu_now)
    state.g = g

    if bmu_now >= state.g.n or mapping[bmu_now] == -1:
        bmu_now = compute_bmu(state, ms, action_bmu)
    else:
        bmu_now = mapping[bmu_now]

    state.prev_bmu = int(bmu_now)
    state.step_idx += 1
    state.mapping = mapping

    return state, int(bmu_now), state_mem
