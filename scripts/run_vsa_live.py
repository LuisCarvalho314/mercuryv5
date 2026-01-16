# scripts/run_vsa_live.py
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

from data_helper.csv_loader import CSVConfig, load_level_csv, iter_sequence
from mercury.action_map.adapter import ActionMap
from mercury.sensory.params import SensoryParams
from mercury.sensory.state import sensory_step, init_state, sensory_step_frozen
from vector_symbolic_architectures.binary_splatter_code import (
    random_hv,
    encode_labeled_directed_graph,
)
from vector_symbolic_architectures.BSC_SOM import (
    sensory_step_BSC,
    hv_from_id,
)
from mercury.graph.plot_graph import draw_graph_on_axes, Positioner

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASETS_DIR = PROJECT_ROOT / "datasets"


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


# ----------------------------
# Config
# ----------------------------
train_plot_trigger = "always"
replay_plot_trigger = "bmu"
replay_pause_s = 0.005
edge_curvature = 0.18

# BSC latent settings
DIM = 10000

# Context trace (leaky)
CTX_GAMMA_MIN = 0.90
CTX_GAMMA_MAX = 0.995
SURPRISE_EMA_LR = 0.05

# Growth control
GROWTH_ACTIVATION_THRESHOLD = 0.985   # tighten to reduce redundant nodes
GROWTH_HABITUATION_THRESHOLD = 0.25   # GWR-style gate

# Soft updates
TOP_K = 5
RANK_LAMBDA = 1.0

# Edge gating
EDGE_ACTIVATION_THRESHOLD = 0.94

# ----------------------------
# Data
# ----------------------------
data_cfg = CSVConfig(root=str(DATASETS_DIR), level=17)
obs, act, col = load_level_csv(data_cfg)

data_dim = int(obs.shape[1])
action_dim = int(act.shape[1]) if act.ndim == 2 else 1

# ----------------------------
# Train sensory (real-valued SOM) - Phase 1
# ----------------------------
state = init_state(data_dim)
cfg = SensoryParams(activation_threshold=0.95, max_age=17, sensory_weighting=1.0)
am = ActionMap.random(n_codebook=4, dim=action_dim, lr=0.5, sigma=0.0, key=0)

for observation, action, collision in iter_sequence(obs, act, col):
    action_vec = np.atleast_1d(action).astype(np.float32)
    action_bmu, _ = am.step(action_vec)
    state = sensory_step(observation.astype(np.float32), int(action_bmu), state, cfg, am)

# ----------------------------
# Latent BSC-SOM over transition HVs - Phase 2
# ----------------------------
rng = np.random.default_rng(0)

latent = init_state(DIM)
latent.global_context = hv_from_id(DIM, identifier=999, salt=0xBADC0DE).astype(np.uint8)

R_SRC = random_hv(DIM, rng)
R_REL = random_hv(DIM, rng)
R_DST = random_hv(DIM, rng)

node_im = {n: random_hv(DIM, rng) for n in range(state.gs.n)}
action_im = {a: random_hv(DIM, rng) for a in range(am.state.codebook.shape[0])}

cfg_latent = SensoryParams(
    activation_threshold=0.99,
    max_age=50,
    sensory_weighting=0.5,
    topological_neighbourhood_threshold=0.9,
)

# plotting
plt.ion()
fig_replay, ax_latent = plt.subplots(1, 1, figsize=(7, 5), dpi=120)
fig_replay.tight_layout()
latent_positioner = Positioner(layout="spring_layout")

prev_mask_latent: Optional[np.ndarray] = None
prev_bmu_latent: Optional[int] = None
prev_bmu_sensory: Optional[int] = None

for observation, action, collision in iter_sequence(obs, act, col):
    action_vec = np.atleast_1d(action).astype(np.float32)
    action_bmu, _ = am.step(action_vec)

    # save sensory BMU BEFORE stepping (to encode transition)
    prev_bmu_sensory = state.prev_bmu

    # frozen sensory (no new nodes)
    state = sensory_step_frozen(observation.astype(np.float32), int(action_bmu), state, cfg, am)
    cur_bmu_sensory = state.prev_bmu

    moved = (
        (prev_bmu_sensory is not None)
        and (cur_bmu_sensory is not None)
        and (cur_bmu_sensory != prev_bmu_sensory)
    )

    if moved and (not collision):
        edges = [(int(prev_bmu_sensory), int(action_bmu), int(cur_bmu_sensory))]
        latent_obs = encode_labeled_directed_graph(
            edges, node_im, action_im, R_SRC, R_REL, R_DST, dst_shift=1
        )

        latent = sensory_step_BSC(
            latent_obs.astype(np.uint8),
            int(action_bmu),
            latent,
            cfg_latent,
            am,
            gamma_min=CTX_GAMMA_MIN,
            gamma_max=CTX_GAMMA_MAX,
            surprise_ema_lr=SURPRISE_EMA_LR,
            growth_activation_threshold=GROWTH_ACTIVATION_THRESHOLD,
            growth_habituation_threshold=GROWTH_HABITUATION_THRESHOLD,
            top_k=TOP_K,
            rank_lambda=RANK_LAMBDA,
            edge_activation_threshold=EDGE_ACTIVATION_THRESHOLD,
            do_maintenance=False,
        )

    # plot latent
    cur_adj_latent = latent.gs.adj
    cur_bmu_latent = latent.prev_bmu

    if _should_plot(
        replay_plot_trigger,
        prev_mask=prev_mask_latent,
        cur_adj=cur_adj_latent,
        prev_bmu=prev_bmu_latent,
        cur_bmu=cur_bmu_latent,
    ):
        ax_latent.clear()
        draw_graph_on_axes(
            ax_latent,
            latent.gs,
            layout="spring_layout",
            positioner=latent_positioner,
            node_color_key="activation",
            edge_width_key="weight",
            edge_color_key="action",
            with_labels=True,
            arrows=True,
            alpha=0.9,
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
            0.02, 0.98, summary_latent,
            transform=ax_latent.transAxes,
            va="top", ha="left",
            fontsize=8,
            bbox=dict(boxstyle="round", fc="white", alpha=0.85),
        )

        plt.pause(replay_pause_s)
        prev_mask_latent = cur_mask_latent_now
    else:
        prev_mask_latent = _edge_presence_mask(cur_adj_latent)

    prev_bmu_latent = cur_bmu_latent

plt.ioff()
plt.show()

print(f"n states in latent | {latent.gs.n}")
