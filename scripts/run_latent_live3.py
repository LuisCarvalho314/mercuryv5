from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.patches import FancyArrowPatch
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import os
import imageio

from data_helper.csv_loader import CSVConfig, load_level_csv, iter_sequence
from mercury.action_map.adapter import ActionMap
from mercury.sensory.params import SensoryParams
from mercury.sensory.state import sensory_step, init_state, sensory_step_frozen
from mercury.memory.state import init_mem, update_memory, add_memory, MemoryState, mem_id
from mercury.latent.state import LatentState, latent_step, init_latent_state
from mercury.latent.params import LatentParams

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASETS_DIR = PROJECT_ROOT / "datasets"


# -------------------------------------------------
# original helpers from your script (unchanged API)
# -------------------------------------------------

def _edge_presence_mask(adj: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    mask = np.abs(adj) > eps
    np.fill_diagonal(mask, False)
    return mask


def _pad_mask(target_shape: tuple[int, int], previous: Optional[np.ndarray]) -> np.ndarray:
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
    t = (trigger or "structure").lower()
    if t == "always":
        return True
    if t == "bmu":
        return prev_bmu is None or cur_bmu != prev_bmu
    _, _, _, changed = _edge_add_remove(prev_mask, cur_adj, eps=eps)
    return changed


# -------------------------------------------------
# extended draw() with mem_adj edge coloring
# -------------------------------------------------

def draw(
    G,
    fig=None,
    ax=None,
    frame_num=None,
    frame_dir="frames",
    save_frames=False,
):
    if fig is None or ax is None:
        fig, ax = plt.subplots(1, 1)
        plt.ion()

    ax.clear()

    # layout is already embedded in G as node["pos"]
    pos = {node: G.nodes[node]["pos"] for node in G.nodes}

    # color nodes by y-position so memory rows vs latent clumps stand out
    y_vals = [pos[node][1] for node in G.nodes]
    if len(y_vals) == 0:
        vmin, vmax = 0.0, 1.0
    else:
        vmin, vmax = min(y_vals), max(y_vals)
        if vmin == vmax:
            vmax = vmin + 1.0
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap = cm.jet
    node_colors = [cmap(norm(pos[node][1])) for node in G.nodes]

    # action -> color and curvature for latent->latent edges
    action_to_color = {
        0: "#00FF00",  # lime
        1: "#FFC107",  # amber
        2: "#00FFFF",  # cyan
        3: "#9C27B0",  # purple
    }
    action_to_rad = {
        0: -0.3,
        1: -0.1,
        2: 0.15,
        3: 0.35,
    }

    # draw nodes
    nx.draw_networkx_nodes(
        G,
        pos=pos,
        ax=ax,
        node_color=node_colors,
        node_size=100,
    )

    # outline active nodes (activation == 1)
    active_nodes = [n for n, d in G.nodes(data=True)
                    if d.get("activation") == 1]
    if active_nodes:
        nx.draw_networkx_nodes(
            G,
            pos=pos,
            ax=ax,
            nodelist=active_nodes,
            node_color=[node_colors[list(G.nodes).index(n)] for n in active_nodes],
            node_size=100,
            edgecolors="black",
            linewidths=2.5,
        )

    # draw edges manually for curvature + custom colors
    for (u, v, data) in G.edges(data=True):
        action_val = data.get("action", None)
        edge_type = data.get("edge_type", None)

        # default style
        rad = action_to_rad.get(action_val, 0.0)
        color = action_to_color.get(action_val, "gray")

        # override for mem_adj links
        # this is the new special case
        if edge_type == "mem_adj":
            color = "magenta"
            rad = 0.0  # straight line

        # optional legacy types
        if edge_type == "sensory":
            color = "red"
        elif edge_type == "inhibitory":
            color = "blue"

        start = pos[u]
        end = pos[v]

        arrow = FancyArrowPatch(
            start,
            end,
            connectionstyle=f"arc3,rad={rad}",
            color=color,
            arrowstyle="-|>",
            mutation_scale=10,
            lw=1,
            zorder=0,
        )
        ax.add_patch(arrow)

    # labels (node id text)
    nx.draw_networkx_labels(
        G,
        pos=pos,
        labels={n: str(n) for n in G.nodes},
        font_size=6,
        ax=ax,
    )

    ax.set_aspect("equal", adjustable="box")
    ax.set_axis_off()

    # frame dump option
    if save_frames and frame_num is not None:
        os.makedirs(frame_dir, exist_ok=True)
        filename = os.path.join(frame_dir, f"frame_{frame_num:04d}.png")
        fig.savefig(filename, dpi=150)

    plt.pause(0.001)
    fig.canvas.draw()
    fig.canvas.flush_events()
    return fig, ax


def create_video_from_frames(frame_dir="frames", output_file="output.mp4", fps=1):
    frame_files = sorted([
        os.path.join(frame_dir, f)
        for f in os.listdir(frame_dir)
        if f.endswith(".png")
    ])
    with imageio.get_writer(output_file, fps=fps) as writer:
        for filename in frame_files:
            image = imageio.imread(filename)
            writer.append_data(image)
    print(f"Video saved to {output_file}")


# -------------------------------------------------
# build a merged (memory + latent) graph for replay
# -------------------------------------------------

def build_latent_memory_graph(
    mem_state: MemoryState,
    latent_state: LatentState,
    *,
    S: int,
    L: int,
) -> nx.DiGraph:
    """
    Nodes in the output graph:
      - One node per memory slot (s,t) named f"M{s}_{t}"
      - One node per latent index j named f"L{j}"
    Edges:
      - memory -> latent (edge_type="mem_adj") for each (s,t) that latent j attends to
      - latent -> latent for each nonzero latent transition
    Positions:
      - memory grid laid out on integer lattice (t, -s)
      - each latent node placed near the centroid of its supporting memory cells,
        slightly shifted to the right (+x_offset) so arrows point inward
    Node attrs:
      - "activation" from:
          mem_state.activations[idx] for memory nodes
          latent_state.g.node_features["activation"][j] for latent nodes
      - "pos" 2-tuple for draw()
    Edge attrs:
      - "action" from latent_state.g.edge_features["action"][u,v,0] for latent->latent
      - "edge_type" == "mem_adj" for memory->latent links
    """
    G = nx.DiGraph()

    # convenience
    mem_adj_all = np.asarray(latent_state.g.node_features["mem_adj"], dtype=np.float32)
    latent_act = np.asarray(latent_state.g.node_features["activation"], dtype=np.float32)
    mem_act_flat = np.asarray(mem_state.activations, dtype=np.float32)  # shape S*L
    adj_latent = np.asarray(latent_state.g.adj, dtype=np.float32)
    action_feat = latent_state.g.edge_features.get("action", None)

    # 1. add memory nodes
    # position memory node (s,t) at (t, -s)
    mem_positions = {}
    for s in range(S):
        for t in range(L):
            idx = mem_id(s, t, L)  # linear index s*L + t
            node_name = f"M{s}_{t}"
            x = float(t)
            y = float(-s)
            mem_positions[idx] = (x, y)
            G.add_node(
                node_name,
                pos=(x, y),
                activation=float(mem_act_flat[idx] > 0),
            )

    # 2. latent node provisional placement
    # for each latent j take all memory cells w/ mem_adj[j, idx] > 0
    # compute centroid of those memory coords, then push it slightly right
    latent_positions = {}
    n_latent = mem_adj_all.shape[0]
    for j in range(n_latent):
        row = mem_adj_all[j] > 0
        idxs = np.flatnonzero(row)
        if idxs.size == 0:
            # fallback position: to the right of time axis, stack vertically
            x = float(L + 0.5)
            y = float(-j * (S / max(n_latent, 1)))
        else:
            coords = np.array([mem_positions[idx] for idx in idxs], dtype=np.float32)
            cx, cy = coords.mean(axis=0)
            x = float(cx + 0.5)  # shift right so arrows slope rightward
            y = float(cy)
        latent_positions[j] = (x, y)
        G.add_node(
            f"L{j}",
            pos=(x, y),
            activation=float(latent_act[j] > 0),
        )

    # 3. edges memory -> latent (visualize mem_adj)
    # color handled in draw() by edge_type == "mem_adj"
    for j in range(n_latent):
        row = mem_adj_all[j] > 0
        idxs = np.flatnonzero(row)
        for idx in idxs:
            s = idx // L
            t = idx % L
            mem_node = f"M{s}_{t}"
            lat_node = f"L{j}"
            G.add_edge(
                mem_node,
                lat_node,
                edge_type="mem_adj",
            )

    # 4. edges latent -> latent (action edges, curved)
    if action_feat is not None:
        for u in range(n_latent):
            for v in range(n_latent):
                if u == v:
                    continue
                w = float(adj_latent[u, v])
                if w == 0.0:
                    continue
                act_lbl = int(action_feat[u, v, 0])
                G.add_edge(
                    f"L{u}",
                    f"L{v}",
                    action=act_lbl,
                    edge_type="latent",
                )

    return G


# -------------------------------------------------
# main script
# -------------------------------------------------

# config
train_plot_trigger = "structure"   # "always" | "structure" | "bmu"
replay_plot_trigger = "bmu"        # "always" | "structure" | "bmu"
train_pause_s = 0.2
replay_pause_s = 0.05
mem_length = 5

# ----- data -----
data_cfg = CSVConfig(root=str(DATASETS_DIR))
obs, act = load_level_csv(data_cfg)
data_dim = int(obs.shape[1])
action_dim = int(act.shape[1]) if act.ndim == 2 else 1

# ----- sensory model + action map -----
state = init_state(data_dim)
cfg = SensoryParams(activation_threshold=0.95, max_age=100)
am = ActionMap.random(n_codebook=4, dim=action_dim, lr=0.5, sigma=0.0, key=0)

# ----------------------------
# Phase 1. Train sensory over dataset (same as before, plotting optional)
# ----------------------------

plt.ion()
fig_train, ax_train = plt.subplots(1, 1, figsize=(7, 5), dpi=120)

prev_mask_train: Optional[np.ndarray] = None
prev_bmu_train: Optional[int] = None

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

    # if you want live sensory plotting you can adapt draw() similarly

plt.ioff()
plt.show()

# ----------------------------
# Phase 2. Freeze sensory. Memory + latent. Live joint plot.
# ----------------------------

mem: MemoryState = init_mem(state, mem_length)
latent: LatentState = init_latent_state(mem)
latent_cfg = LatentParams()

plt.ion()
fig_replay, ax_replay = plt.subplots(1, 1, figsize=(8, 5), dpi=120)

prev_mask_latent: Optional[np.ndarray] = None
prev_bmu_latent: Optional[int] = None
action_mem = []
frame_i = 0

for observation, action in iter_sequence(obs, act):
    action_vec = np.atleast_1d(action).astype(np.float32)
    action_bmu, _ = am.step(action_vec)

    # sensory update without growing graph
    state = sensory_step_frozen(
        observation.astype(np.float32),
        int(action_bmu),
        state,
        cfg,
        am,
    )

    sensory_node_count = state.gs.n  # S
    if mem is None or mem.gs.n != sensory_node_count * mem_length:
        mem = init_mem(state, length=mem_length)

    # update memory + latent only when BMU changes
    if prev_bmu_latent is None or state.prev_bmu != prev_bmu_latent:
        sensory_activation = np.asarray(
            state.gs.node_features["activation"],
            dtype=np.float32,
        )
        mem = update_memory(mem)
        mem = add_memory(mem, sensory_activation)
        action_mem.append(action_bmu)

        latent, _ = latent_step(
            mem,
            latent,
            action_bmu,
            latent_cfg,
            am,
            action_mem,
        )

    cur_adj_latent = latent.g.adj
    cur_bmu_latent = latent.prev_bmu

    if _should_plot(
        replay_plot_trigger,
        prev_mask=prev_mask_latent,
        cur_adj=cur_adj_latent,
        prev_bmu=prev_bmu_latent,
        cur_bmu=cur_bmu_latent,
    ):
        # build merged graph of memory cells + latent states
        G_vis = build_latent_memory_graph(
            mem_state=mem,
            latent_state=latent,
            S=sensory_node_count,
            L=mem_length,
        )

        # annotate figure title/summary
        summary_latent, cur_mask_latent_now = _summarize_change(
            prev_mask_latent,
            cur_adj_latent,
            prev_bmu_latent,
            cur_bmu_latent,
        )
        ax_replay.set_title(
            f"Latent + Memory  |  step {latent.step_idx}  |  {summary_latent}",
            fontsize=8,
        )

        # draw with special magenta mem_adj edges
        fig_replay, ax_replay = draw(
            G_vis,
            fig=fig_replay,
            ax=ax_replay,
            frame_num=frame_i,
            save_frames=False,
        )
        frame_i += 1

        prev_mask_latent = cur_mask_latent_now
    else:
        prev_mask_latent = _edge_presence_mask(cur_adj_latent)

    prev_bmu_latent = state.prev_bmu

plt.ioff()
plt.show()

print(f"n states in latent | {latent.g.n}")
