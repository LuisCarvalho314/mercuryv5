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
# topology + plotting helpers
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
        action_vec = np.atleast_1d(action).astype(np.float32)
        action_bmu, _ = am.step(action_vec)

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
                live_ax.set_title(f"Training • step {state.step_idx}")
                live_ax.text(
                    0.02,
                    0.98,
                    summary,
                    transform=live_ax.transAxes,
                    va="top",
                    ha="left",
                    fontsize=8,
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
# replay with frozen sensory + memory + latent + live plot
# ----------------------------

def replay_with_memory_and_latent(
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
    latent: Optional[LatentState] = None,
    latent_cfg: Optional[LatentParams] = None,
) -> Tuple[Optional[LatentState], MemoryState]:
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

        action_vec = np.atleast_1d(action).astype(np.float32)
        action_bmu, _ = am.step(action_vec)

        state = sensory_step_frozen(
            observation.astype(np.float32),
            int(action_bmu),
            state,
            cfg,
            am,
        )

        sensory_node_count = state.gs.n
        if mem is None or mem.gs.n != sensory_node_count * mem_length:
            mem = init_mem(state, length=mem_length)

        if prev_bmu is None or state.prev_bmu != prev_bmu:
            sensory_activation = np.asarray(state.gs.node_features["activation"], dtype=np.float32)
            mem = update_memory(mem)
            mem = add_memory(mem, sensory_activation)
            if latent is not None and latent_cfg is not None:
                latent, bmu = latent_step(mem, latent, action_bmu, latent_cfg,
                                         am)

        cur_adj = latent.g.adj
        cur_bmu = latent.prev_bmu

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
                latent.g,
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
            ax_s.set_title(f"Sensory • step {latent.step_idx}")
            ax_s.text(
                0.02,
                0.98,
                summary,
                transform=ax_s.transAxes,
                va="top",
                ha="left",
                fontsize=8,
                bbox=dict(boxstyle="round", fc="white", alpha=0.85),
            )

            # memory grid
            ax_m.clear()
            draw_memory_grid_on_axes(
                ax_m,
                mem_state=mem,
                S=sensory_node_count,
                L=mem_length,
                dx=1.0,
                dy=1.0,
                node_color_key="activation",
                arrows=True,
                alpha=0.9,
            )
            ax_m.set_title(f"Memory grid S×L = {sensory_node_count}×{mem_length}")
            ax_m.set_aspect("equal", adjustable="box")

            fig.tight_layout()
            plt.pause(pause_s)
            prev_mask = cur_mask
        else:
            prev_mask = _edge_presence_mask(cur_adj)

        prev_bmu = cur_bmu

    plt.ioff()
    plt.show()

    assert mem is not None  # for type checkers
    return latent, mem


# ----------------------------
# main
# ----------------------------

def main(
    *,
    train_live: bool = True,
    train_pause_s: float = 0.2,
    train_plot_trigger: str = "structure",  # "always" | "structure" | "bmu"
    replay_max_steps: int = 100,
    replay_pause_s: float = 0.05,
    mem_length: int = 5,
    layout_for_cache: str = "kamada_kawai",
    edge_curvature: float = 0.18,
) -> None:
    # ----- data -----
    data_cfg = CSVConfig(root=str(DATASETS_DIR))
    obs, act = load_level_csv(data_cfg)
    data_dim = int(obs.shape[1])
    action_dim = int(act.shape[1]) if act.ndim == 2 else 1

    # ----- models -----
    state = init_state(data_dim)
    cfg = SensoryParams(activation_threshold=0.95, max_age=100)
    am = ActionMap.random(n_codebook=4, dim=action_dim, lr=0.5, sigma=0.0, key=0)

    # latent containers (created post-training)
    latent: Optional[LatentState] = None
    latent_cfg: Optional[LatentParams] = None

    # Position cache: recompute only when node count increases
    positioner = Positioner(layout=layout_for_cache)

    # ----- training (optional live plot) -----
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

    # ----- init memory + latent -----
    mem_state = init_mem(state, mem_length)
    latent = init_latent_state(mem_state)
    latent_cfg = LatentParams()

    # ----- replay with frozen sensory + memory + latent, live plot -----
    latent, _ = replay_with_memory_and_latent(
        obs,
        act,
        state,
        cfg,
        am,
        max_steps=replay_max_steps,
        pause_s=replay_pause_s,
        mem_length=mem_length,
        plot_trigger="bmu",
        positioner=positioner,
        edge_curvature=edge_curvature,
        latent=latent,
        latent_cfg=latent_cfg,
    )

    print(f"n states in latent | {latent.g.n}")


if __name__ == "__main__":
    main(train_live=False, train_plot_trigger="always",train_pause_s=2,
         replay_max_steps=200)
    print("done")

