# src/vector_symbolic_architectures/BSC_SOM.py
from __future__ import annotations

from dataclasses import replace
from typing import Tuple

import numpy as np

from mercury.action_map.adapter import ActionMap
from mercury.graph.core import Graph
from mercury.graph.maintenance import age_maintenance
from mercury.sensory.params import SensoryParams
from mercury.sensory.state import (
    SensoryState,
    _add_node,
    _edge_action_row,
    _set_activation,
    _should_create_node,
    _update_edge,
    _update_node,
)

Array = np.ndarray


# =============================================================================
# Utilities
# =============================================================================

def _as_bits01_vector(vector: np.ndarray) -> np.ndarray:
    arr = np.asarray(vector)
    if arr.dtype == np.uint8:
        return arr
    if arr.dtype == np.bool_:
        return arr.astype(np.uint8, copy=False)
    return arr.astype(np.uint8, copy=False)


def bits01_to_bipolar(bits01: np.ndarray) -> np.ndarray:
    b = _as_bits01_vector(bits01)
    return (b.astype(np.int8) * 2 - 1).astype(np.int8)


def _hash_u32(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.uint32, copy=False)
    x ^= x >> np.uint32(16)
    x *= np.uint32(0x7feb352d)
    x ^= x >> np.uint32(15)
    x *= np.uint32(0x846ca68b)
    x ^= x >> np.uint32(16)
    return x


def deterministic_tie_break_bits01(node_index: int, bit_indices: np.ndarray, salt: int) -> np.ndarray:
    node_u32 = np.uint32(node_index)
    bit_u32 = bit_indices.astype(np.uint32, copy=False)
    salt_u32 = np.uint32(salt)
    # wraparound is intended; silence overflow warning
    with np.errstate(over="ignore"):
        x = bit_u32 ^ (node_u32 * np.uint32(0x9e3779b9)) ^ salt_u32
    h = _hash_u32(x)
    return (h & np.uint32(1)).astype(np.uint8)


def votes_to_bits01(votes: np.ndarray, *, tie_node_index: int, tie_salt: int) -> np.ndarray:
    v = np.asarray(votes, dtype=np.float32)
    out = np.empty(v.shape[0], dtype=np.uint8)
    out[v > 0] = 1
    out[v < 0] = 0
    tie = v == 0
    if np.any(tie):
        tie_idx = np.nonzero(tie)[0].astype(np.int32, copy=False)
        out[tie] = deterministic_tie_break_bits01(
            node_index=int(tie_node_index),
            bit_indices=tie_idx,
            salt=int(tie_salt),
        )
    return out


def _vector_length_from_graph(g: Graph) -> int:
    w = g.node_features["weight"]
    w = np.asarray(w)
    if w.ndim == 2:
        return int(w.shape[1])
    # if your Graph ever stores object arrays, this would be the fallback:
    return int(np.asarray(g.node_features["weight"][0]).shape[0])


def hamming_distance_per_node(node_feature: np.ndarray, vector_bits01: np.ndarray) -> np.ndarray:
    x = _as_bits01_vector(vector_bits01).astype(np.bool_, copy=False)
    W = _as_bits01_vector(node_feature).astype(np.bool_, copy=False)  # (N,D)
    return np.count_nonzero(np.logical_xor(W, x), axis=1).astype(np.int32)


# =============================================================================
# Roles / deterministic HV-from-id (expected uint32 wrap)
# =============================================================================

def hv_from_id(vector_length: int, identifier: int, salt: int = 0x12345678) -> np.ndarray:
    bit_indices = np.arange(vector_length, dtype=np.uint32)
    with np.errstate(over="ignore"):
        x = bit_indices ^ (np.uint32(identifier) * np.uint32(0x9e3779b9)) ^ np.uint32(salt)
    h = _hash_u32(x)
    return (h & np.uint32(1)).astype(np.uint8)


def roles(vector_length: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r_obs = hv_from_id(vector_length, 1, salt=0xA11CE001)
    r_ctx = hv_from_id(vector_length, 2, salt=0xA11CE001)
    r_act = hv_from_id(vector_length, 3, salt=0xA11CE001)
    return r_obs, r_ctx, r_act


def make_state_hv(observation_bits01: np.ndarray, global_context_bits01: np.ndarray) -> np.ndarray:
    observation_bits01 = _as_bits01_vector(observation_bits01)
    global_context_bits01 = _as_bits01_vector(global_context_bits01)
    d = int(observation_bits01.size)
    r_obs, r_ctx, _ = roles(d)
    return (r_obs ^ observation_bits01 ^ r_ctx ^ global_context_bits01).astype(np.uint8)


def transition_token(observation_bits01: np.ndarray, action_bmu: int) -> np.ndarray:
    observation_bits01 = _as_bits01_vector(observation_bits01)
    d = int(observation_bits01.size)
    r_obs, _, r_act = roles(d)
    action_hv = hv_from_id(d, int(action_bmu), salt=0xAC710)
    return (r_act ^ action_hv ^ r_obs ^ observation_bits01).astype(np.uint8, copy=False)


# =============================================================================
# Register + reconcile extra node features (critical for age_maintenance)
# =============================================================================

def ensure_scalar_features_exist(g: Graph) -> Graph:
    # Register so padding/removal stays consistent with g.n
    if "habituation" not in g.node_features:
        g.register_node_feature("habituation", 1, init_value=1.0)
    if "error_accum" not in g.node_features:
        g.register_node_feature("error_accum", 1, init_value=0.0)
    return g


def ensure_vote_features_exist(g: Graph) -> Graph:
    d = _vector_length_from_graph(g)

    # Register vote features so Graph maintains shapes on add/remove
    if "weight_votes" not in g.node_features:
        g.register_node_feature("weight_votes", d, init_value=0.0)
    if "context_votes" not in g.node_features:
        g.register_node_feature("context_votes", d, init_value=0.0)

    # Initialize votes from current prototypes for existing nodes
    W = _as_bits01_vector(g.node_features["weight"]).astype(np.uint8, copy=False)   # (N,D)
    C = _as_bits01_vector(g.node_features["context"]).astype(np.uint8, copy=False) # (N,D)
    g.node_features["weight_votes"] = bits01_to_bipolar(W).astype(np.float32, copy=False)
    g.node_features["context_votes"] = bits01_to_bipolar(C).astype(np.float32, copy=False)
    return g


def reconcile_registered_feature_shapes(g: Graph) -> Graph:
    """
    Safety: after maintenance/remaps, guarantee every extra feature has first dim == g.n.
    Should be redundant if features are registered, but prevents edge cases.
    """
    n = int(g.n)
    for name in ("weight_votes", "context_votes", "habituation", "error_accum"):
        if name not in g.node_features:
            continue
        mat = np.asarray(g.node_features[name])
        if mat.shape[0] == n:
            continue
        if mat.shape[0] > n:
            g.node_features[name] = mat[:n].copy()
        else:
            # pad
            pad_rows = n - mat.shape[0]
            if mat.ndim == 1:
                g.node_features[name] = np.concatenate([mat, np.zeros((pad_rows,), mat.dtype)])
            else:
                g.node_features[name] = np.vstack([mat, np.zeros((pad_rows, mat.shape[1]), mat.dtype)])
    return g


# =============================================================================
# Prototype learning (vote-space EMA)
# =============================================================================

def update_votes_ema(current_votes: np.ndarray, target_bits01: np.ndarray, lr: float) -> np.ndarray:
    target_bipolar = bits01_to_bipolar(target_bits01).astype(np.float32, copy=False)
    return (1.0 - float(lr)) * current_votes + float(lr) * target_bipolar


def update_node_from_votes(g: Graph, node_index: int) -> Graph:
    wv = np.asarray(g.node_features["weight_votes"], dtype=np.float32)
    cv = np.asarray(g.node_features["context_votes"], dtype=np.float32)

    w_bits = votes_to_bits01(wv[node_index], tie_node_index=node_index, tie_salt=0xA5B35705)
    c_bits = votes_to_bits01(cv[node_index], tie_node_index=node_index, tie_salt=0xC3E11A77)
    return _update_node(g, int(node_index), w_bits, c_bits)


# =============================================================================
# Leaky (decaying) global context trace + surprise modulation
# =============================================================================

def _ensure_context_trace(state: SensoryState, d: int) -> None:
    if not hasattr(state, "context_votes") or getattr(state, "context_votes") is None:
        setattr(state, "context_votes", np.zeros((d,), dtype=np.float32))
    if not hasattr(state, "surprise_ema"):
        setattr(state, "surprise_ema", 0.0)


def update_global_context_leaky(
    state: SensoryState,
    *,
    observation_bits01: np.ndarray,
    action_bmu: int,
    gamma_min: float,
    gamma_max: float,
    surprise_ema_lr: float,
    current_surprise: float,
) -> SensoryState:
    observation_bits01 = _as_bits01_vector(observation_bits01)
    d = int(observation_bits01.size)
    _ensure_context_trace(state, d)

    prev_s = float(getattr(state, "surprise_ema"))
    s_ema = (1.0 - float(surprise_ema_lr)) * prev_s + float(surprise_ema_lr) * float(current_surprise)
    setattr(state, "surprise_ema", float(s_ema))

    gamma = float(gamma_min) + (float(gamma_max) - float(gamma_min)) * float(np.clip(s_ema, 0.0, 1.0))

    token = transition_token(observation_bits01, int(action_bmu))
    token_bip = bits01_to_bipolar(token).astype(np.float32, copy=False)

    cv = np.asarray(getattr(state, "context_votes"), dtype=np.float32)
    cv = gamma * cv + token_bip
    setattr(state, "context_votes", cv)

    gc_bits01 = votes_to_bits01(cv, tie_node_index=0, tie_salt=0xBADC0DE)
    return replace(state, global_context=gc_bits01.astype(np.uint8, copy=False))


# =============================================================================
# Main BSC step (maintenance-safe)
# =============================================================================

def sensory_step_BSC(
    observation: Array,      # (D,) bits01
    action_bmu: int,
    state: SensoryState,
    cfg: SensoryParams,
    action_map: ActionMap,
    *,
    # context trace
    gamma_min: float = 0.90,
    gamma_max: float = 0.995,
    surprise_ema_lr: float = 0.05,
    # growth control
    growth_activation_threshold: float | None = None,
    habituation_tau: float = 0.01,
    habituation_kappa: float = 0.05,
    growth_habituation_threshold: float = 0.25,
    # soft updates
    top_k: int = 3,
    rank_lambda: float = 1.0,
    # edges
    edge_activation_threshold: float = 0.92,
    do_maintenance: bool = True,
) -> SensoryState:
    g = state.gs

    g = ensure_vote_features_exist(g)
    g = ensure_scalar_features_exist(g)

    observation_bits01 = _as_bits01_vector(observation)
    global_context_bits01 = _as_bits01_vector(state.global_context)

    # identity includes context
    state_hv = make_state_hv(observation_bits01, global_context_bits01)

    # distances
    W = _as_bits01_vector(g.node_features["weight"])
    C = _as_bits01_vector(g.node_features["context"])
    d_w = hamming_distance_per_node(W, state_hv).astype(np.float32)
    d_c = hamming_distance_per_node(C, global_context_bits01).astype(np.float32)
    dn = float(cfg.sensory_weighting) * d_w + (1.0 - float(cfg.sensory_weighting)) * d_c

    bmu = int(np.argmin(dn))
    D = int(state_hv.size)
    activation = 1.0 - float(d_w[bmu]) / float(D)
    surprise = 1.0 - activation

    # habituation dynamics
    hab = np.asarray(g.node_features["habituation"], dtype=np.float32).reshape(-1)
    hab = np.clip(hab + float(habituation_tau) * (1.0 - hab), 0.0, 1.0)
    hab[bmu] = float(np.clip(hab[bmu] - float(habituation_kappa) * hab[bmu], 0.0, 1.0))
    g.node_features["habituation"] = hab.reshape(-1, 1)

    # update global context trace now
    state.gs = g
    state = update_global_context_leaky(
        state,
        observation_bits01=observation_bits01,
        action_bmu=int(action_bmu),
        gamma_min=gamma_min,
        gamma_max=gamma_max,
        surprise_ema_lr=surprise_ema_lr,
        current_surprise=surprise,
    )
    global_context_bits01 = _as_bits01_vector(state.global_context)

    # growth decision
    act_thr = float(cfg.activation_threshold) if growth_activation_threshold is None else float(growth_activation_threshold)
    grow_gate = (activation < act_thr) and (hab[bmu] < float(growth_habituation_threshold))

    wants_growth = grow_gate and _should_create_node(
        activation,
        activation_threshold=act_thr,
        n_neurons=g.n,
        max_neurons=cfg.max_neurons,
    )

    if wants_growth:
        # _add_node pads all registered node features (including our vote/scalars)
        g = _add_node(g, state_hv, global_context_bits01)
        bmu = g.n - 1

        # set new node’s votes into the already-padded registered arrays
        wv = np.asarray(g.node_features["weight_votes"], dtype=np.float32)
        cv = np.asarray(g.node_features["context_votes"], dtype=np.float32)
        wv[bmu] = bits01_to_bipolar(state_hv).astype(np.float32, copy=False)
        cv[bmu] = bits01_to_bipolar(global_context_bits01).astype(np.float32, copy=False)
        g.node_features["weight_votes"] = wv
        g.node_features["context_votes"] = cv

        # initialize scalars for new node
        hab2 = np.asarray(g.node_features["habituation"], dtype=np.float32).reshape(-1)
        err2 = np.asarray(g.node_features["error_accum"], dtype=np.float32).reshape(-1)
        hab2[bmu] = 1.0
        err2[bmu] = 0.0
        g.node_features["habituation"] = hab2.reshape(-1, 1)
        g.node_features["error_accum"] = err2.reshape(-1, 1)

        g = update_node_from_votes(g, bmu)

    else:
        # soft top-k updates
        k = int(max(1, top_k))
        top_idx = np.argsort(dn)[:k]

        wv = np.asarray(g.node_features["weight_votes"], dtype=np.float32)
        cv = np.asarray(g.node_features["context_votes"], dtype=np.float32)

        for rank, j in enumerate(top_idx.tolist()):
            eta = float(cfg.winning_node_lr) * float(np.exp(-float(rank) / float(max(1e-6, rank_lambda))))
            if eta <= 0.0:
                continue
            wv[j] = update_votes_ema(wv[j], state_hv, lr=eta)
            cv[j] = update_votes_ema(cv[j], global_context_bits01, lr=eta)

        g.node_features["weight_votes"] = wv
        g.node_features["context_votes"] = cv

        g = update_node_from_votes(g, bmu)

    mapping = np.arange(g.n, dtype=np.int32)

    # edge learning (activation-gated) + maintenance
    if state.prev_bmu is not None and activation >= float(edge_activation_threshold):
        observed_row = np.asarray(action_map.state.codebook[action_bmu], np.float32)
        prior_row = _edge_action_row(g, state.prev_bmu, bmu, am=action_map)

        g = _update_edge(
            g,
            state.prev_bmu,
            bmu,
            prior_row,
            observed_row,
            action_bmu=action_bmu,
            beta=0.5,
            gaussian_shape=cfg.gaussian_shape,
        )

        if do_maintenance:
            g, mapping = age_maintenance(u=state.prev_bmu, v=bmu, g=g, p=cfg)
            # after maintenance, ensure our extra features still match g.n
            g = reconcile_registered_feature_shapes(g)
            bmu = int(mapping[bmu])

    g = _set_activation(g, bmu)

    state.gs = g
    state.prev_bmu = int(bmu)
    state.step_idx += 1
    state.mapping = mapping
    return state
