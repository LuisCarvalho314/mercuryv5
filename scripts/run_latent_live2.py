from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

from data_helper.csv_loader import CSVConfig, load_level_csv, iter_sequence
from mercury.action_map.adapter import ActionMap
from mercury.sensory.params import SensoryParams
from mercury.sensory.state import sensory_step, init_state, sensory_step_frozen
from mercury.memory.state import init_mem, update_memory, add_memory, MemoryState
from mercury.latent.state import LatentState, latent_step, init_latent_state
from mercury.latent.params import LatentParams
from mercury.graph.plot_graph import (
    draw_graph_on_axes,
    draw_memory_grid_on_axes,
    Positioner,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASETS_DIR = PROJECT_ROOT / "datasets"

# ----------------------------
# edge diff + plotting helpers
# ----------------------------

def _edge_presence_mask(adj: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Boolean mask of present edges, excluding self-loops."""
    mask = np.abs(adj) > eps
    np.fill_diagonal(mask, False)
    return mask


def _pad_mask(target_shape: tuple[int, int], previous: Optional[np.ndarray]) -> np.ndarray:
    """Pad previous mask to target shape with False."""
    if previous is None:
        return np.zeros(target_shape, dtype=bool)
    if previous.shape == target_shape:
        return previous
    out = np.zeros(target_shape, dtype=bool)
    r = min(target_shape[0], previous.shape[0])
    c = min(target_shape[1], previous.shape[1])
    out[:r, :c] = previous[:r, :c]
    return out


def _edge_add_remove(
    prev_mask: Optional[np.ndarray],
    cur_adj: np.ndarray,
    eps: float = 1e-8,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    """
    Returns:
      cur_mask : bool (n,n) current presence
      added    : int (k,2) edges now present that were absent
      removed  : int (k,2) edges now absent that were present
      changed  : bool any add/remove
    Notes:
      - Ignores weight-only changes.
      - Self-loops excluded.
      - Handles n changes via padding.
    """
    cur_mask = _edge_presence_mask(cur_adj, eps)
    prev_aligned = _pad_mask(cur_mask.shape, prev_mask)
    added = np.argwhere(cur_mask & (~prev_aligned))
    removed = np.argwhere((~cur_mask) & prev_aligned)
    changed = (added.size + removed.size) > 0
    return cur_mask, added, removed, changed


def _summarize_change(
    prev_mask: Optional[np.ndarray],
    cur_adj: np.ndarray,
    prev_bmu: Optional[int],
    cur_bmu: Optional[int],
    eps: float = 1e-8,
) -> Tuple[str, np.ndarray]:
    """
    Human-readable summary string for the live plot.
    Returns (summary_text, cur_mask_for_next_iter).
    """
    cur_mask, added, removed, _ = _edge_add_remove(prev_mask, cur_adj, eps=eps)
    add_cnt = int(added.shape[0])
    rem_cnt = int(removed.shape[0])
    add_examples = [f"{int(u)}→{int(v)}" for u, v in added[:3]]
    rem_examples = [f"{int(u)}→{int(v)}" for u, v in removed[:3]]

    parts: list[str] = []
    if add_cnt or rem_cnt:
        ae = f" ({', '.join(add_examples)})" if add_examples else ""
        re = f" ({', '.join(rem_examples)})" if rem_examples else ""
        parts.append(f"edges +{add_cnt}{ae}, -{rem_cnt}{re}")
    else:
        parts.append("edges 0")

    if prev_bmu is None:
        parts.append(f"BMU {cur_bmu}")
    else:
        delta_flag = " (Δ)" if cur_bmu != prev_bmu else " (=)"
        parts.append(f"BMU {prev_bmu}→{cur_bmu}{delta_flag}")

    return " | ".join(parts), cur_mask


def _should_plot(
    trigger: str,
    *,
    prev_mask: Optional[np.ndarray],
    cur_adj: np.ndarray,
    prev_bmu: Optional[int],
    cur_bmu: Optional[int],
    eps: float = 1e-8,
) -> bool:
    """
    trigger:
      - "always"    -> always True
      - "bmu"       -> True iff BMU changed
      - "structure" -> True iff edge set changed (add/remove only, self-loops excluded)
    """
    t = (trigger or "structure").lower()
    if t == "always":
        return True
    if t == "bmu":
        return prev_bmu is None or cur_bmu != prev_bmu
    # "structure"
    _, _, _, changed = _edge_add_remove(prev_mask, cur_adj, eps=eps)
    return changed


# ----------------------------
# Script body
# ----------------------------

# config
train_plot_trigger = "structure"   # "always" | "structure" | "bmu"
replay_plot_trigger = "bmu"        # "always" | "structure" | "bmu"
train_pause_s = 0.2
replay_pause_s = 0.005
mem_length = 10
edge_curvature = 0.18
layout_for_cache = "spring_layout"

# ----- data -----
data_cfg = CSVConfig(root=str(DATASETS_DIR))
obs, act = load_level_csv(data_cfg)
data_dim = int(obs.shape[1])
action_dim = int(act.shape[1]) if act.ndim == 2 else 1

# ----- sensory model + action map -----
state = init_state(data_dim)
cfg = SensoryParams(activation_threshold=0.95, max_age=100)
am = ActionMap.random(n_codebook=4, dim=action_dim, lr=0.5, sigma=0.0, key=0)

# cache for node layout so the graph is visually stable
positioner = Positioner(layout=layout_for_cache)

# ----------------------------
# Phase 1. Train sensory over full dataset (live plot optional)
# ----------------------------

plt.ion()
fig_train, ax_train = plt.subplots(1, 1, figsize=(7, 5), dpi=120)

prev_mask_train: Optional[np.ndarray] = None
prev_bmu_train: Optional[int] = None

for observation, action in iter_sequence(obs, act):
    action_vec = np.atleast_1d(action).astype(np.float32)
    action_bmu, _ = am.step(action_vec)

    # update sensory graph with plasticity
    state = sensory_step(
        observation.astype(np.float32),
        int(action_bmu),
        state,
        cfg,
        am,
    )

    # live training plot
    cur_adj = state.gs.adj
    cur_bmu = state.prev_bmu

#     if _should_plot(
#         train_plot_trigger,
#         prev_mask=prev_mask_train,
#         cur_adj=cur_adj,
#         prev_bmu=prev_bmu_train,
#         cur_bmu=cur_bmu,
#     ):
#         ax_train.clear()
#         draw_graph_on_axes(
#             ax_train,
#             state.gs,
#             layout="kamada_kawai",
#             node_color_key="activation",
#             edge_width_key="weight",
#             edge_color_key="action",
#             with_labels=False,
#             arrows=True,
#             alpha=0.9,
#             positioner=positioner,
#             edge_curvature=edge_curvature,
#         )
#         summary, cur_mask_now = _summarize_change(
#             prev_mask_train,
#             cur_adj,
#             prev_bmu_train,
#             cur_bmu,
#         )
#         ax_train.set_title(f"Training • step {state.step_idx}")
#         ax_train.text(
#             0.02,
#             0.98,
#             summary,
#             transform=ax_train.transAxes,
#             va="top",
#             ha="left",
#             fontsize=8,
#             bbox=dict(boxstyle="round", fc="white", alpha=0.85),
#         )
#         ax_train.figure.tight_layout()
#         plt.pause(train_pause_s)
#         prev_mask_train = cur_mask_now
#     else:
#         prev_mask_train = _edge_presence_mask(cur_adj)
#
#     prev_bmu_train = cur_bmu
#
# plt.ioff()
# plt.show()

# ----------------------------
# Phase 2. Freeze sensory. Run memory + latent. Live plot sensory+memory.
# ----------------------------

# init memory state using final sensory graph
mem: MemoryState = init_mem(state, mem_length)

# init latent from that memory
latent: LatentState = init_latent_state(mem)
latent_cfg = LatentParams()

plt.ion()
fig_replay, (ax_latent, ax_mem) = plt.subplots(1, 2, figsize=(12, 5), dpi=120)

prev_mask_latent: Optional[np.ndarray] = None
prev_bmu_latent: Optional[int] = None
action_mem = []
steps = 0
for observation, action in iter_sequence(obs, act):
    steps += 1

    action_vec = np.atleast_1d(action).astype(np.float32)
    action_bmu, _ = am.step(action_vec)

    # freeze sensory structure (sensory_step_frozen does not grow graph)
    state = sensory_step_frozen(
        observation.astype(np.float32),
        int(action_bmu),
        state,
        cfg,
        am,
    )

    sensory_node_count = state.gs.n
    # if memory shape no longer matches (S * L) recreate memory ring buffer
    if mem is None or mem.gs.n != sensory_node_count * mem_length:
        mem = init_mem(state, length=mem_length)

    # update memory and latent whenever we moved to a new BMU in sensory
    if prev_bmu_latent is None or state.prev_bmu != prev_bmu_latent:
        sensory_activation = np.asarray(
            state.gs.node_features["activation"],
            dtype=np.float32,
        )
        mem = update_memory(mem)
        mem = add_memory(mem, sensory_activation)
        action_mem.append(action_bmu)
        latent, bmu = latent_step(mem, latent, action_bmu, latent_cfg, am,
                                  action_mem)

    # plotting current latent graph + memory grid
    cur_adj_latent = latent.g.adj
    cur_bmu_latent = latent.prev_bmu

    if _should_plot(
        replay_plot_trigger,
        prev_mask=prev_mask_latent,
        cur_adj=cur_adj_latent,
        prev_bmu=prev_bmu_latent,
        cur_bmu=cur_bmu_latent,
    ):
        # left: latent graph
        ax_latent.clear()
        draw_graph_on_axes(
            ax_latent,
            latent.g,
            layout="kamada_kawai",
            node_color_key="activation",
            edge_width_key="weight",
            edge_color_key="action",
            with_labels=True,
            arrows=True,
            alpha=0.9,
            positioner=positioner,
            edge_curvature=edge_curvature,
        )
        summary_latent, cur_mask_latent_now = _summarize_change(
            prev_mask_latent,
            cur_adj_latent,
            prev_bmu_latent,
            cur_bmu_latent,
        )
        ax_latent.set_title(f"Latent • step {latent.step_idx}")
        ax_latent.text(
            0.02,
            0.98,
            summary_latent,
            transform=ax_latent.transAxes,
            va="top",
            ha="left",
            fontsize=8,
            bbox=dict(boxstyle="round", fc="white", alpha=0.85),
        )

        # right: memory grid (S x L occupancy / activation)
        ax_mem.clear()
        draw_memory_grid_on_axes(
            ax_mem,
            mem_state=mem,
            S=sensory_node_count,
            L=mem_length,
            dx=1.0,
            dy=1.0,
            node_color_key="activation",
            arrows=True,
            alpha=0.9,
        )
        ax_mem.set_title(f"Memory grid S×L = {sensory_node_count}×{mem_length}")
        ax_mem.set_aspect("equal", adjustable="box")

        fig_replay.tight_layout()
        plt.pause(replay_pause_s)

        prev_mask_latent = cur_mask_latent_now
    else:
        prev_mask_latent = _edge_presence_mask(cur_adj_latent)

    prev_bmu_latent = state.prev_bmu

plt.ioff()
plt.show()

print(f"n states in latent | {latent.g.n}")
