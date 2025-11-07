# latent/state.py
from copy import deepcopy
from dataclasses import dataclass, replace, field
from typing import Any, Tuple, List

import numpy as np

from ..action_map.adapter import ActionMap
from mercury.graph.core import Graph, Array
from mercury.graph.maintenance import age_maintenance
from .params import LatentParams
from ..memory.state import MemoryState, mem_id, activations_at_t, memory_view_at_global_timestep
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

def init_latent_state(ms: MemoryState) -> LatentState:
    g = build_initial_graph(ms)
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

def compute_bmu(state: LatentState, ms: MemoryState) -> int:
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
    return bmu


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
                               gaussian_shape: int) -> Graph:
    observed_row = np.asarray(action_map.state.codebook[action_bmu],
                              np.float32)
    prior_row = _edge_action_row(g, prev_bmu, bmu, am=action_map)
    g = _update_edge(
        g, prev_bmu, bmu, prior_row, observed_row,
        action_bmu=action_bmu, beta=0.5, gaussian_shape=gaussian_shape,
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
    print(f"prev {bmu} -> {action_bmu} -> preds: {idx}")
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


def _resolve_prev_with_memory_replay(
    state: LatentState,
    ms: MemoryState,
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
    if state.step_idx < L:
        return replay_state.g, int(prev_bmu), bmu

    am = list(action_mem)[-L:]              # oldest → newest
    if len(am) < L:
        raise ValueError(f"action_mem length {len(am)} < L={L}")


    for i in range(L):
        mem_view = memory_view_at_global_timestep(ms, L-i-1)
        # action_bmu = int(am[i])
        bmu = int(compute_bmu(replay_state, mem_view))

        if bmu == prev_bmu:
            # refine by spawning a clone extended at the first gap
            g_new, bmu_refined, created = add_temporal_edge_at_first_gap_clone(
                replay_state.g, mem_view, bmu, preds
            )
            _debug_print_latent_mem_adj(g_new, ms, bmu_refined)
            print(f"created {created}")
            replay_state.g = g_new

            bmu = int(compute_bmu(replay_state, mem_view))
            if bmu == bmu_refined:
                print("bmu created properly")

            # break


    """ 
    TODO
    When bmu 10 is created it is not activated in the replay hinting at some 
    flaw in the logic
    """
    seen_bmus = set()
    for _ in range(int(max_replay)):
        bmu = None
        # oldest → newest frames
        for j in range(L):
            prev_bmu = bmu
            mem_view = memory_view_at_global_timestep(ms, L-j-1)
            action_bmu = int(am[j])

            bmu = int(compute_bmu(replay_state, mem_view))
            # if bmu == bmu_refined:
            #     print(f"prev {prev_bmu} -> {action_bmu} -> bmu: {bmu}")
            #
            # elif prev_bmu == bmu_refined:
            #     print(f"prev {prev_bmu} -> {action_bmu} -> bmu: {bmu}")
            seen_bmus.add(bmu)
            #
            # optional learning of replay chain edges
            # keep only if needed for your dynamics
            # if prev_bmu is not None:
            #     replay_state.g = _update_edges_with_actions(
            #         replay_state.g, prev_bmu=prev_bmu, bmu=bmu,
            #         action_bmu=action_bmu, action_map=action_map,
            #         gaussian_shape=cfg.gaussian_shape,
            #     )

    if bmu_refined not in seen_bmus:
        raise EnvironmentError("added bmu not seen in replay")
    return replay_state.g, int(prev_bmu), bmu




def _still_aliased(g: Graph, node: int | None, action_bmu: int) -> bool:
    if node is None:
        return False
    return _is_aliased(g, _predict(g, int(node), int(action_bmu)))

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




def latent_step(
    ms: MemoryState,
    state: LatentState,
    action_bmu: int,
    cfg: LatentParams,
    action_map: ActionMap,
    action_mem: list[int],
) -> tuple[LatentState, int]:
    g = state.g

    # BMU for the current timestep (true present)
    bmu_now = compute_bmu(state, ms)
    # print("bmu_now", bmu_now)
    mapping = np.arange(g.n, dtype=np.int32)

    # assert np.array_equal(ms.activations, memory_view_at_global_timestep(ms,
    #                                                          4).activations),\
    #     (f"{ms.activations} vs"
    #                                                      f" "
    #      f"{memory_view_at_global_timestep(ms,4).activations}")


    if state.prev_bmu is None:
        # no previous state yet
        next_prev_bmu = int(bmu_now)
    else:
        resolved_prev = state.prev_bmu

        preds = _predict(state.g, int(state.prev_bmu), action_bmu)


        if preds.size > 1:
            print(f"ALIASED {state.prev_bmu}")
            state.preds.append(preds)
            print(f"seen preds: {state.preds}")
            # resolve aliasing of the previous state chain using replay
            # """ remove all connections"""
            # for i in range(g.n):
            #     removed = range(g.n)
            #     g = _remove_aliased_connections(g, i, removed)

            """ remove all connections to and from bmu_prev"""
            g = _remove_aliased_connections(g, resolved_prev,range(g.n))
            for i in range(g.n):
                g = _remove_aliased_connections(g, i, resolved_prev)


            # g = _remove_aliased_connections(
            #     g,
            #     bmu_prev=int(resolved_prev),
            #     predicted_nodes=preds,
            # )


            state.g = g


            g, resolved_prev, bmu_now = _resolve_prev_with_memory_replay(
                state=state,
                ms=ms,
                prev_bmu=state.prev_bmu,
                bmu=bmu_now,
                action_mem=action_mem,
                action_map=action_map,
                gaussian_shape=cfg.gaussian_shape,
                max_replay=2,
                cfg=cfg,
                preds=preds,
            )

            bmu_now = mapping[bmu_now]

            # # mark active for plotting
            # g = _set_activation(g, bmu_now)
            #
            # # # commit
            # state.g = g
            # state.prev_bmu = int(bmu_now)
            # state.step_idx += 1
            # state.mapping = mapping
            #
            # if state.step_idx % 10 == 0:
            #     for i in range(state.g.n):
            #         _debug_print_latent_mem_adj(g, ms, i)
            #
            # # print(f"{g.n} nodes @ step {state.step_idx}")
            # return state, int(bmu_now)

        # learn live edge from resolved_prev -> bmu_now under current action_bmu
        g = _update_edges_with_actions(
            g,
            prev_bmu=resolved_prev,
            bmu=bmu_now,
            action_bmu=action_bmu,
            action_map=action_map,
            gaussian_shape=cfg.gaussian_shape,
        )
        #
        # if state.step_idx % 5 == 0:
        #     g, mapping = age_maintenance(resolved_prev,bmu_now,g,cfg,
        #                                  exclude=np.arange(
        #                                      ms.sensory_n_nodes))


    bmu_now = mapping[bmu_now]

    print(f"STEP {state.step_idx} | {state.prev_bmu} -> {action_bmu} ->"
          f" {bmu_now}")

    # mark active for plotting
    g = _set_activation(g, bmu_now)

    # commit
    state.g = g
    state.prev_bmu = int(bmu_now)
    state.step_idx += 1
    state.mapping = mapping

    if state.step_idx % 10 ==0:
        for i in range(state.g.n):
            _debug_print_latent_mem_adj(g,ms,i)
        _debug_print_latent_action_adj(g)

    # print(f"{g.n} nodes @ step {state.step_idx}")
    return state, int(bmu_now)

