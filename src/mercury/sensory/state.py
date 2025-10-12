# sensory/state.py
from __future__ import annotations
from dataclasses import dataclass, replace
from typing import Any, Tuple

import numpy as np

from ..action_map.adapter import ActionMap
from mercury.graph.core import Graph
from mercury.graph.maintenance import age_maintenance
from .params import SensoryParams

Array = np.ndarray


# =====================================================================
# State container
# =====================================================================

@dataclass
class SensoryState:
    """
    Mutable container for model state.

    Attributes
    ----------
    gs : Graph
        Graph with node/edge features.
    global_context : Array
        Global context vector, shape (D,).
    mapping : Array
        Last compaction mapping, shape (n,), int32.
    prev_bmu : int | None
        Previous BMU index.
    step_idx : int
        Step counter.
    """
    gs: Graph
    global_context: Array
    mapping: Array
    prev_bmu: int | None = None
    step_idx: int = 0


# =====================================================================
# Graph scaffolding
# =====================================================================

def _register_features(g: Graph, data_dim: int) -> Graph:
    """
    Register required features.

    Node features
    -------------
    weight      : (n, data_dim)
    context     : (n, data_dim)
    activation  : (n,)

    Edge features
    -------------
    age         : (n, n, 1)
    action      : (n, n, 1)
    """
    g.register_node_feature("weight", data_dim)
    g.register_node_feature("context", data_dim)
    g.register_node_feature("activation", 1)
    g.register_edge_feature("age", 1, dtype=np.int32, init_value=0)
    g.register_edge_feature("action", 1, dtype=np.int32, init_value=0)
    return g


def init_global_context(data_dim: int) -> Array:
    return np.zeros((data_dim,), dtype=np.float32)


def build_initial_graph(data_dim: int, n: int = 2) -> Graph:
    """
    Create empty graph, register features, add `n` nodes.
    """
    g = Graph(directed=True)
    _register_features(g, data_dim)
    for _ in range(n):
        g.add_node(
            node_feat={
                "weight": np.zeros((data_dim,), np.float32),
                "context": np.zeros((data_dim,), np.float32),
                "activation": np.array(0.0, np.float32),
            },
            edge_defaults_for_new_node={"age": np.array([0], np.int32),
                                        "action": np.array([0], np.int32)},
        )
    g.assert_invariants()
    return g


def init_state(data_dim: int, n: int = 2) -> SensoryState:
    """
    Initialize SensoryState with scaffolded graph.
    """
    g = build_initial_graph(data_dim, n)
    return SensoryState(
        gs=g,
        global_context=init_global_context(data_dim),
        mapping=np.arange(g.n, dtype=np.int32),
    )


# =====================================================================
# Internal helpers (pure)
# =====================================================================

def _compute_dn(
    sensory_weighting: float,
    observation: Array,        # (F,)
    global_context: Array,     # (F,)
    node_weights: Array,       # (N,F)
    node_contexts: Array,      # (N,F)
) -> tuple[Array, Array, Array]:
    diff_weights = node_weights - observation
    weight_term = np.sum(diff_weights ** 2, axis=1)
    context_term = np.sum((node_contexts - global_context) ** 2, axis=1)
    dn = sensory_weighting * weight_term + (1.0 - sensory_weighting) * context_term
    return dn.astype(np.float32), weight_term.astype(np.float32), diff_weights.astype(np.float32)


def _pick_bmu(
    weights: Array, contexts: Array, observation: Array,
    global_context: Array, sensory_weighting: float
) -> tuple[int, Array, Array]:
    d, d_weight, diff_weights = _compute_dn(
        sensory_weighting, observation, global_context, weights, contexts
    )
    bmu = int(np.argmin(d))
    return bmu, d_weight, diff_weights


def _calc_activation(d: Array | float, gaussian_shape: float) -> Array | float:
    d = np.asarray(d, dtype=np.float32)
    sigma = float(gaussian_shape)
    if sigma <= 0:
        out = np.zeros_like(d, dtype=np.float32)
    else:
        out = np.exp(-(d * d) / (2.0 * sigma * sigma)).astype(np.float32)
    return float(out) if out.ndim == 0 else out


def _meets_threshold(x: Array | float, threshold: float) -> Array:
    # Note: original code used <=; keep semantics.
    return np.asarray(x, dtype=np.float32) <= float(threshold)


def _should_create_node(
    activation: float,
    activation_threshold: float,
    n_neurons: int,
    max_neurons: int,
) -> bool:
    return bool(_meets_threshold(activation, activation_threshold)) and (n_neurons < max_neurons)


def _update_node(g: Graph, node: int, updated_weight: Array, updated_context: Array) -> Graph:
    w = g.node_features["weight"].copy()
    c = g.node_features["context"].copy()
    w[node] = updated_weight.astype(w.dtype, copy=False)
    c[node] = updated_context.astype(c.dtype, copy=False)
    g.set_node_feat("weight", w)
    g.set_node_feat("context", c)
    return g


def _update_winning_node(g: Graph, bmu: int, observation: Array, global_context: Array, lr: float) -> Graph:
    weights = g.node_features["weight"]
    contexts = g.node_features["context"]
    w_new = weights[bmu] + float(lr) * (observation - weights[bmu])
    c_new = contexts[bmu] + float(lr) * (global_context - contexts[bmu])
    return _update_node(g, bmu, w_new, c_new)


def _topological_similarity(dist: Array, distance_threshold: float, gaussian_shape: float) -> Array:
    d = np.atleast_1d(np.asarray(dist, dtype=np.float32))
    if gaussian_shape <= 0.0:
        sim = np.zeros_like(d, dtype=np.float32)
    else:
        sig = np.float32(max(gaussian_shape, 1e-12))
        sim = np.exp(-0.5 * (d / sig) ** 2).astype(np.float32)
    mask = _meets_threshold(d, float(distance_threshold))
    return np.where(mask, sim, 0.0).astype(np.float32)


def _update_neighbours(
    g: Graph,
    bmu: int,
    observation: Array,
    diff_weights: Array,
    gaussian_shape: float,
    distance_threshold: float,
    lr: float,
) -> Graph:
    neigh = g.neighbors(bmu, mode="any")
    neigh = neigh[neigh != bmu]
    if neigh.size == 0:
        return g

    weights = g.node_features["weight"]    # (N,D)
    contexts = g.node_features["context"]  # (N,D)

    w_nb = weights[neigh]                  # (K,D)
    c_nb = contexts[neigh]                 # (K,D)
    diff_nb = diff_weights[neigh]          # (K,D)

    dist = np.linalg.norm(diff_nb, axis=1)  # (K,)
    topo = _topological_similarity(dist, distance_threshold, gaussian_shape)[:, None]  # (K,1)

    w_nb_new = w_nb + float(lr) * diff_nb * topo
    c_nb_new = c_nb + float(lr) * (observation - c_nb) * topo

    weights2 = weights.copy()
    contexts2 = contexts.copy()
    weights2[neigh] = w_nb_new.astype(weights2.dtype, copy=False)
    contexts2[neigh] = c_nb_new.astype(contexts2.dtype, copy=False)

    g.set_node_feat("weight", weights2)
    g.set_node_feat("context", contexts2)
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


def _update_global_context(state: SensoryState, weight: Array, context: Array, lr: float) -> SensoryState:
    new_gc = float(lr) * weight + (1.0 - float(lr)) * context
    return replace(state, global_context=new_gc.astype(np.float32, copy=False))


def _add_node(g: Graph, weight: Array, global_context: Array) -> Graph:
    g.add_node(
        node_feat={
            "weight": np.asarray(weight, np.float32),
            "context": np.asarray(global_context, np.float32),
            "activation": np.array(0.0, np.float32),
        },
        edge_defaults_for_new_node={"age": np.array([0], np.int32),
                                    "action": np.array([0], np.int32)},
    )
    return g


def _set_activation(g: Graph, bmu: int) -> Graph:
    act = np.zeros((g.n,), np.float32)
    act[bmu] = 1.0
    g.set_node_feat("activation", act)
    return g


# =====================================================================
# Pure step & driver
# =====================================================================

def sensory_step(
    observation: Array,
    action_bmu: int,
    state: SensoryState,
    cfg: SensoryParams,
    action_map: ActionMap,
) -> SensoryState:
    g = state.gs

    # 1) BMU selection
    weights = g.node_features["weight"]
    contexts = g.node_features["context"]
    bmu, d_weight, diff_weights = _pick_bmu(
        weights, contexts, np.asarray(observation, np.float32),
        state.global_context, cfg.sensory_weighting
    )

    activation = float(_calc_activation(d_weight[bmu], cfg.gaussian_shape))

    if _should_create_node(
        activation,
        activation_threshold=cfg.activation_threshold,
        n_neurons=g.n,
        max_neurons=cfg.max_neurons,
    ):
        g = _add_node(g, observation, state.global_context)
        bmu = g.n - 1
    else:
        g = _update_winning_node(g, bmu, np.asarray(observation, np.float32), state.global_context, cfg.winning_node_lr)
        g = _update_neighbours(
            g, bmu, np.asarray(observation, np.float32), diff_weights,
            gaussian_shape=cfg.gaussian_shape,
            distance_threshold=cfg.topological_neighbourhood_threshold,
            lr=cfg.topological_neighbourhood_lr,
        )

    state = _update_global_context(state, g.node_features["weight"][bmu], g.node_features["context"][bmu], cfg.global_context_lr)
    mapping = np.arange(g.n, dtype=np.int32)

    # 3) Temporal link + maintenance
    if state.prev_bmu is not None:
        observed_row = np.asarray(action_map.state.codebook[action_bmu], np.float32)
        prior_row = _edge_action_row(g, state.prev_bmu, bmu, am=action_map)
        g = _update_edge(
            g, state.prev_bmu, bmu, prior_row, observed_row,
            action_bmu=action_bmu, beta=0.5, gaussian_shape=cfg.gaussian_shape,
        )
        g, mapping = age_maintenance(u=state.prev_bmu, v=bmu, g=g, p=cfg)  # numpy version
        bmu = int(mapping[bmu])

    g = _set_activation(g, bmu)

    state.gs = g
    state.prev_bmu = bmu
    state.step_idx += 1
    state.mapping = mapping
    return state


def sensory_step_frozen(
    observation: Array,
    action_bmu: int,
    state: SensoryState,
    cfg: SensoryParams,
    action_map: ActionMap,
) -> SensoryState:
    g = state.gs

    # 1) BMU selection
    weights = g.node_features["weight"]
    contexts = g.node_features["context"]
    bmu, d_weight, diff_weights = _pick_bmu(
        weights, contexts, np.asarray(observation, np.float32),
        state.global_context, cfg.sensory_weighting
    )

    activation = float(_calc_activation(d_weight[bmu], cfg.gaussian_shape))

    if _should_create_node(
        activation,
        activation_threshold=cfg.activation_threshold,
        n_neurons=g.n,
        max_neurons=cfg.max_neurons,
    ):
        g = _add_node(g, observation, state.global_context)
        bmu = g.n - 1
    else:
        g = _update_winning_node(g, bmu, np.asarray(observation, np.float32), state.global_context, cfg.winning_node_lr)
        g = _update_neighbours(
            g, bmu, np.asarray(observation, np.float32), diff_weights,
            gaussian_shape=cfg.gaussian_shape,
            distance_threshold=cfg.topological_neighbourhood_threshold,
            lr=cfg.topological_neighbourhood_lr,
        )

    state = _update_global_context(state, g.node_features["weight"][bmu], g.node_features["context"][bmu], cfg.global_context_lr)
    mapping = np.arange(g.n, dtype=np.int32)

    # 3) Temporal link + maintenance
    if state.prev_bmu is not None:
        observed_row = np.asarray(action_map.state.codebook[action_bmu], np.float32)
        prior_row = _edge_action_row(g, state.prev_bmu, bmu, am=action_map)
        g = _update_edge(
            g, state.prev_bmu, bmu, prior_row, observed_row,
            action_bmu=action_bmu, beta=cfg.action_lr,
            gaussian_shape=cfg.gaussian_shape,
        )
        bmu = int(mapping[bmu])

    g = _set_activation(g, bmu)

    state.gs = g
    state.prev_bmu = bmu
    state.step_idx += 1
    state.mapping = mapping
    return state