# scripts/run_sensory_numpy_live.py
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

from data_helper.csv_loader import CSVConfig, load_level_csv, iter_sequence
from mercury.action_map.adapter import ActionMap
from mercury.sensory.params import SensoryParams
from mercury.sensory.state import sensory_step, sensory_step_frozen, init_state
from mercury.graph.plot_graph import (
    draw_graph_on_axes,
    draw_memory_grid_on_axes,
    Positioner,
)
from mercury.memory.state import init_mem, update_memory, add_memory, MemoryState

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASETS_DIR = PROJECT_ROOT / "datasets"

# ----------------------------
# topology helpers
# ----------------------------

def _edge_presence_mask(adj: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Boolean mask of present edges, excluding self-loops."""
    m = np.abs(adj) > eps
    np.fill_diagonal(m, False)
    return m

def _pad_mask(target_shape: tuple[int, int], prev: Optional[np.ndarray]) -> np.ndarray:
    """Pad previous mask to target shape with False."""
    if prev is None:
        return np.zeros(target_shape, dtype=bool)
    if prev.shape == target_shape:
        return prev
    out = np.zeros(target_shape, dtype=bool)
    r = min(target_shape[0], prev.shape[0])
    c = min(target_shape[1], prev.shape[1])
    out[:r, :c] = prev[:r, :c]
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
    # structure
    _, _, _, changed = _edge_add_remove(prev_mask, cur_adj, eps=eps)
    return changed

def _summarize_change(
    prev_mask: Optional[np.ndarray],
    cur_adj: np.ndarray,
    prev_bmu: Optional[int],
    cur_bmu: Optional[int],
    eps: float = 1e-8,
) -> Tuple[str, np.ndarray]:
    cur_mask, added, removed, _ = _edge_add_remove(prev_mask, cur_adj, eps=eps)
    add_cnt = int(added.shape[0])
    rem_cnt = int(removed.shape[0])
    add_examples = [f"{int(u)}→{int(v)}" for u, v in added[:3]]
    rem_examples = [f"{int(u)}→{int(v)}" for u, v in removed[:3]]

    parts = []
    if add_cnt or rem_cnt:
        ae = f" ({', '.join(add_examples)})" if add_examples else ""
        re = f" ({', '.join(rem_examples)})" if rem_examples else ""
        parts.append(f"edges +{add_cnt}{ae}, -{rem_cnt}{re}")
    else:
        parts.append("edges 0")

    if prev_bmu is None:
        parts.append(f"BMU {cur_bmu}")
    else:
        parts.append(f"BMU {prev_bmu}→{cur_bmu}" + (" (Δ)" if cur_bmu != prev_bmu else " (=)"))
    return " | ".join(parts), cur_mask

# ----------------------------
# training pass (optional live plot)
# ----------------------------

def train_sensory_full_dataset(
    obs: np.ndarray,
    act: np.ndarray,
    state,
    cfg: SensoryParams,
    am: ActionMap,
    *,
    live_ax: Optional[plt.Axes] = None,
    pause_s: float = 0.3,
    plot_trigger: str = "structure",  # "always" | "structure" | "bmu"
    positioner: Optional[Positioner] = None,
    edge_curvature: float = 0.15,
) -> Tuple[object, ActionMap]:
    prev_mask: Optional[np.ndarray] = None
    prev_bmu: Optional[int] = None

    for observation, action in iter_sequence(obs, act):
        a = np.atleast_1d(action).astype(np.float32)
        action_bmu, _ = am.step(a)

        state = sensory_step(
            observation.astype(np.float32),
            int(action_bmu),
            state,
            cfg,
            am,
        )

        if live_ax is not None:
            cur_adj = state.gs.adj
            cur_bmu = state.prev_bmu

            if _should_plot(
                plot_trigger,
                prev_mask=prev_mask,
                cur_adj=cur_adj,
                prev_bmu=prev_bmu,
                cur_bmu=cur_bmu,
            ):
                live_ax.clear()
                draw_graph_on_axes(
                    live_ax,
                    state.gs,
                    layout="kamada_kawai",           # good crossing reduction
                    node_color_key="activation",
                    edge_width_key="weight",
                    edge_color_key="action",
                    with_labels=False,
                    arrows=True,
                    alpha=0.9,
                    positioner=positioner,           # reuse positions; recompute on node add
                    edge_curvature=edge_curvature,   # curved antiparallel edges
                )
                summary, cur_mask = _summarize_change(prev_mask, cur_adj, prev_bmu, cur_bmu)
                live_ax.set_title(f"Training • step {state.step_idx}")
                live_ax.text(
                    0.02, 0.98, summary, transform=live_ax.transAxes,
                    va="top", ha="left", fontsize=8,
                    bbox=dict(boxstyle="round", fc="white", alpha=0.85),
                )
                live_ax.figure.tight_layout()
                plt.pause(pause_s)
                prev_mask = cur_mask
            else:
                prev_mask = _edge_presence_mask(cur_adj)

            prev_bmu = cur_bmu

    return state, am

# ----------------------------
# replay with frozen sensory + memory + live plot
# ----------------------------

def replay_with_memory(
    obs: np.ndarray,
    act: np.ndarray,
    state,
    cfg: SensoryParams,
    am: ActionMap,
    *,
    max_steps: int = 100,
    pause_s: float = 0.5,
    mem_length: int = 5,
    plot_trigger: str = "bmu",   # "always" | "structure" | "bmu"
    positioner: Optional[Positioner] = None,
    edge_curvature: float = 0.15,
) -> None:
    mem: MemoryState | None = None

    plt.ion()
    fig, (ax_s, ax_m) = plt.subplots(1, 2, figsize=(12, 5), dpi=120)

    prev_mask: Optional[np.ndarray] = None
    prev_bmu: Optional[int] = None

    steps = 0
    for observation, action in iter_sequence(obs, act):
        if steps >= max_steps:
            break
        steps += 1

        a = np.atleast_1d(action).astype(np.float32)
        action_bmu, _ = am.step(a)

        state = sensory_step_frozen(
            observation.astype(np.float32),
            int(action_bmu),
            state,
            cfg,
            am,
        )

        S = state.gs.n
        if mem is None or mem.gs.n != S * mem_length:
            mem = init_mem(state, length=mem_length)

        if prev_bmu is None or state.prev_bmu != prev_bmu:
            s_act = np.asarray(state.gs.node_features["activation"], dtype=np.float32)
            mem = update_memory(mem)
            mem = add_memory(mem, s_act)

        cur_adj = state.gs.adj
        cur_bmu = state.prev_bmu

        if _should_plot(
            plot_trigger,
            prev_mask=prev_mask,
            cur_adj=cur_adj,
            prev_bmu=prev_bmu,
            cur_bmu=cur_bmu,
        ):
            # sensory
            ax_s.clear()
            draw_graph_on_axes(
                ax_s,
                state.gs,
                layout="kamada_kawai",
                node_color_key="activation",
                edge_width_key="weight",
                edge_color_key="action",
                with_labels=False,
                arrows=True,
                alpha=0.9,
                positioner=positioner,
                edge_curvature=edge_curvature,
            )
            summary, cur_mask = _summarize_change(prev_mask, cur_adj, prev_bmu, cur_bmu)
            ax_s.set_title(f"Sensory • step {state.step_idx}")
            ax_s.text(
                0.02, 0.98, summary, transform=ax_s.transAxes,
                va="top", ha="left", fontsize=8,
                bbox=dict(boxstyle="round", fc="white", alpha=0.85),
            )

            # memory grid
            ax_m.clear()
            draw_memory_grid_on_axes(
                ax_m,
                mem_state=mem,
                S=S,
                L=mem_length,
                dx=1.0,
                dy=1.0,
                node_color_key="activation",
                arrows=True,
                alpha=0.9,
            )
            ax_m.set_title(f"Memory grid S×L = {S}×{mem_length}")
            ax_m.set_aspect("equal", adjustable="box")

            fig.tight_layout()
            plt.pause(pause_s)
            prev_mask = cur_mask
        else:
            prev_mask = _edge_presence_mask(cur_adj)

        prev_bmu = cur_bmu

    plt.ioff()
    plt.show()

# ----------------------------
# main
# ----------------------------

def main(
    *,
    max_steps: int = 100,
    pause_s: float = 0.5,
    mem_length: int = 5,
    train_live: bool = True,
    train_pause_s: float = 0.2,
    train_plot_trigger: str = "structure",  # "always" | "structure" | "bmu"
    replay_plot_trigger: str = "bmu",       # "always" | "structure" | "bmu"
    layout_for_cache: str = "kamada_kawai",
    edge_curvature: float = 0.15,
) -> None:
    cfg_csv = CSVConfig(root=str(DATASETS_DIR))
    obs, act = load_level_csv(cfg_csv)
    data_dim = int(obs.shape[1])
    action_dim = int(act.shape[1]) if act.ndim == 2 else 1

    state = init_state(data_dim)
    cfg = SensoryParams(activation_threshold=0.95, max_age=100)
    am = ActionMap.random(n_codebook=4, dim=action_dim, lr=0.5, sigma=0.0, key=0)

    # Position cache: recompute only when node count increases
    positioner = Positioner(layout=layout_for_cache)

    if train_live:
        plt.ion()
        fig_train, ax_train = plt.subplots(1, 1, figsize=(7, 5), dpi=120)
        state, am = train_sensory_full_dataset(
            obs,
            act,
            state,
            cfg,
            am,
            live_ax=ax_train,
            pause_s=train_pause_s,
            plot_trigger=train_plot_trigger,
            positioner=positioner,
            edge_curvature=edge_curvature,
        )
        plt.ioff()
        plt.show()
    else:
        state, am = train_sensory_full_dataset(obs, act, state, cfg, am)

    replay_with_memory(
        obs,
        act,
        state,
        cfg,
        am,
        max_steps=max_steps,
        pause_s=pause_s,
        mem_length=mem_length,
        plot_trigger=replay_plot_trigger,
        positioner=positioner,
        edge_curvature=edge_curvature,
    )

if __name__ == "__main__":
    main(
        max_steps=100,
        pause_s=0.05,
        mem_length=5,
        train_live=True,
        train_pause_s=0.2,
        train_plot_trigger="structure",
        replay_plot_trigger="bmu",
        layout_for_cache="kamada_kawai",
        edge_curvature=0.18,
    )
