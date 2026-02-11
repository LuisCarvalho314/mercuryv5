# scripts/latent_live_2.py
from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from data_helper.csv_loader import iter_sequence
from mercury.action_map.adapter import ActionMap
from mercury.graph.plot_graph import (
    Positioner,
    draw_graph_on_axes,
    draw_memory_grid_on_axes,
)
from mercury.latent.params import LatentParams
from mercury.latent.state import LatentState, init_latent_state, latent_step
from mercury.memory.state import MemoryState, add_memory, init_mem, update_memory
from mercury.sensory.params import SensoryParams
from mercury.sensory.state import init_state, sensory_step, sensory_step_frozen

from mercury_runs.io_parquet import ParquetConfig, load_level_parquet


# ----------------------------
# edge diff + plotting helpers
# ----------------------------

def edge_presence_mask(adjacency: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Boolean mask of present edges, excluding self-loops."""
    mask = np.abs(adjacency) > eps
    np.fill_diagonal(mask, False)
    return mask


def pad_mask(target_shape: tuple[int, int], previous_mask: Optional[np.ndarray]) -> np.ndarray:
    """Pad previous mask to target shape with False."""
    if previous_mask is None:
        return np.zeros(target_shape, dtype=bool)
    if previous_mask.shape == target_shape:
        return previous_mask
    padded = np.zeros(target_shape, dtype=bool)
    row_count = min(target_shape[0], previous_mask.shape[0])
    col_count = min(target_shape[1], previous_mask.shape[1])
    padded[:row_count, :col_count] = previous_mask[:row_count, :col_count]
    return padded


def edge_add_remove(
    previous_mask: Optional[np.ndarray],
    current_adjacency: np.ndarray,
    eps: float = 1e-8,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    """
    Returns:
      current_mask : bool (n,n) current presence
      added        : int (k,2) edges now present that were absent
      removed      : int (k,2) edges now absent that were present
      changed      : bool any add/remove
    """
    current_mask = edge_presence_mask(current_adjacency, eps)
    previous_aligned = pad_mask(current_mask.shape, previous_mask)
    added = np.argwhere(current_mask & (~previous_aligned))
    removed = np.argwhere((~current_mask) & previous_aligned)
    changed = (added.size + removed.size) > 0
    return current_mask, added, removed, changed


def summarize_change(
    previous_mask: Optional[np.ndarray],
    current_adjacency: np.ndarray,
    previous_bmu: Optional[int],
    current_bmu: Optional[int],
    eps: float = 1e-8,
) -> Tuple[str, np.ndarray]:
    current_mask, added, removed, _ = edge_add_remove(previous_mask, current_adjacency, eps=eps)

    added_count = int(added.shape[0])
    removed_count = int(removed.shape[0])

    added_examples = [f"{int(u)}→{int(v)}" for u, v in added[:3]]
    removed_examples = [f"{int(u)}→{int(v)}" for u, v in removed[:3]]

    edge_part = "edges 0"
    if added_count or removed_count:
        added_suffix = f" ({', '.join(added_examples)})" if added_examples else ""
        removed_suffix = f" ({', '.join(removed_examples)})" if removed_examples else ""
        edge_part = f"edges +{added_count}{added_suffix}, -{removed_count}{removed_suffix}"

    if previous_bmu is None:
        bmu_part = f"BMU {current_bmu}"
    else:
        changed_flag = " (Δ)" if current_bmu != previous_bmu else " (=)"
        bmu_part = f"BMU {previous_bmu}→{current_bmu}{changed_flag}"

    return f"{edge_part} | {bmu_part}", current_mask


def should_plot(
    trigger: str,
    *,
    previous_mask: Optional[np.ndarray],
    current_adjacency: np.ndarray,
    previous_bmu: Optional[int],
    current_bmu: Optional[int],
    eps: float = 1e-8,
) -> bool:
    """
    trigger:
      - "always"    -> always True
      - "bmu"       -> True iff BMU changed
      - "structure" -> True iff edge set changed (add/remove, self-loops excluded)
    """
    mode = (trigger or "structure").lower()
    if mode == "always":
        return True
    if mode == "bmu":
        return previous_bmu is None or current_bmu != previous_bmu
    _, _, _, changed = edge_add_remove(previous_mask, current_adjacency, eps=eps)
    return changed


# ----------------------------
# core pipeline steps
# ----------------------------

def build_action_map(action_dim: int, *, n_codebook: int, lr: float, sigma: float, key: int) -> ActionMap:
    return ActionMap.random(
        n_codebook=int(n_codebook),
        dim=int(action_dim),
        lr=float(lr),
        sigma=float(sigma),
        key=int(key),
    )


def train_sensory_over_dataset(
    observations: np.ndarray,
    actions: np.ndarray,
    collisions: np.ndarray,
    *,
    sensory_params: SensoryParams,
    action_map: ActionMap,
    training_plot_trigger: str,
    training_pause_s: float,
    edge_curvature: float,
    positioner: Positioner,
    enable_training_plot: bool,
) -> tuple[Any, list[int]]:
    observation_dim = int(observations.shape[1])
    sensory_state = init_state(observation_dim)

    training_bmus: list[int] = []
    previous_mask: Optional[np.ndarray] = None
    previous_bmu: Optional[int] = None

    if enable_training_plot:
        plt.ion()
        figure, axis = plt.subplots(1, 1, figsize=(7, 5), dpi=120)
    else:
        figure = None
        axis = None

    for observation, action, collision in iter_sequence(observations, actions, collisions):
        action_vector = np.atleast_1d(action).astype(np.float32)
        action_bmu, _ = action_map.step(action_vector)

        sensory_state = sensory_step(
            observation.astype(np.float32),
            int(action_bmu),
            sensory_state,
            sensory_params,
            action_map,
        )

        training_bmus.append(int(sensory_state.prev_bmu))

        if not enable_training_plot:
            continue

        current_adjacency = sensory_state.gs.adj
        current_bmu = int(sensory_state.prev_bmu)

        if should_plot(
            training_plot_trigger,
            previous_mask=previous_mask,
            current_adjacency=current_adjacency,
            previous_bmu=previous_bmu,
            current_bmu=current_bmu,
        ):
            axis.clear()
            draw_graph_on_axes(
                axis,
                sensory_state.gs,
                layout="spring_layout",
                node_color_key="activation",
                edge_width_key="weight",
                edge_color_key="action",
                with_labels=False,
                arrows=True,
                alpha=0.9,
                positioner=positioner,
                edge_curvature=edge_curvature,
            )
            summary_text, current_mask = summarize_change(previous_mask, current_adjacency, previous_bmu, current_bmu)
            axis.set_title(f"Sensory training • step {getattr(sensory_state, 'step_idx', '?')}")
            axis.text(
                0.02,
                0.98,
                summary_text,
                transform=axis.transAxes,
                va="top",
                ha="left",
                fontsize=8,
                bbox=dict(boxstyle="round", fc="white", alpha=0.85),
            )
            figure.tight_layout()
            plt.pause(training_pause_s)
            previous_mask = current_mask
        else:
            previous_mask = edge_presence_mask(current_adjacency)

        previous_bmu = current_bmu

    return sensory_state, training_bmus


def replay_latent_with_live_plot(
    observations: np.ndarray,
    actions: np.ndarray,
    collisions: np.ndarray,
    *,
    sensory_state: Any,
    sensory_params: SensoryParams,
    latent_params: LatentParams,
    action_map: ActionMap,
    memory_length: int,
    replay_plot_trigger: str,
    replay_pause_s: float,
    edge_curvature: float,
) -> tuple[LatentState, MemoryState]:
    memory_state: MemoryState = init_mem(sensory_state.gs.n, memory_length)
    latent_state: LatentState = init_latent_state(memory_state)

    plt.ion()
    figure, (latent_axis, memory_axis) = plt.subplots(1, 2, figsize=(12, 5), dpi=120)

    previous_mask: Optional[np.ndarray] = None
    previous_latent_bmu: Optional[int] = None

    memory_vectors: list[np.ndarray] = []
    action_memory: list[int] = []
    state_memory: list[int] = []

    for observation, action, collision in iter_sequence(observations, actions, collisions):
        action_vector = np.atleast_1d(action).astype(np.float32)
        action_bmu, _ = action_map.step(action_vector)

        sensory_state = sensory_step_frozen(
            observation.astype(np.float32),
            int(action_bmu),
            sensory_state,
            sensory_params,
            action_map,
        )

        sensory_node_count = int(sensory_state.gs.n)
        if memory_state.gs.n != sensory_node_count * memory_length:
            memory_state = init_mem(sensory_node_count, length=memory_length)

        if previous_latent_bmu is None or not bool(collision):
            sensory_activation = np.asarray(sensory_state.gs.node_features["activation"], dtype=np.float32)
            memory_state = update_memory(memory_state)
            memory_state = add_memory(memory_state, sensory_activation)

            memory_vectors.append(sensory_activation)
            action_memory.append(int(action_bmu))

            latent_state, _, state_memory = latent_step(
                memory_state,
                memory_vectors,
                latent_state,
                int(action_bmu),
                latent_params,
                action_map,
                action_memory,
                state_memory,
            )

        current_adjacency = latent_state.g.adj
        current_latent_bmu = int(latent_state.prev_bmu)

        if should_plot(
            replay_plot_trigger,
            previous_mask=previous_mask,
            current_adjacency=current_adjacency,
            previous_bmu=previous_latent_bmu,
            current_bmu=current_latent_bmu,
        ):
            latent_axis.clear()
            draw_graph_on_axes(
                latent_axis,
                latent_state.g,
                layout="kamada_kawai",
                node_color_key="activation",
                edge_width_key="weight",
                edge_color_key="action",
                with_labels=True,
                arrows=True,
                alpha=0.9,
                edge_curvature=edge_curvature,
            )
            summary_text, current_mask = summarize_change(previous_mask, current_adjacency, previous_latent_bmu, current_latent_bmu)
            latent_axis.set_title(f"Latent • step {getattr(latent_state, 'step_idx', '?')}")
            latent_axis.text(
                0.02,
                0.98,
                summary_text,
                transform=latent_axis.transAxes,
                va="top",
                ha="left",
                fontsize=8,
                bbox=dict(boxstyle="round", fc="white", alpha=0.85),
            )

            memory_axis.clear()
            draw_memory_grid_on_axes(
                memory_axis,
                mem_state=memory_state,
                S=sensory_node_count,
                L=memory_length,
                dx=1.0,
                dy=1.0,
                node_color_key="activation",
                arrows=True,
                alpha=0.9,
            )
            memory_axis.set_title(f"Memory grid S×L = {sensory_node_count}×{memory_length}")
            memory_axis.set_aspect("equal", adjustable="box")

            figure.tight_layout()
            plt.pause(replay_pause_s)

            previous_mask = current_mask
        else:
            previous_mask = edge_presence_mask(current_adjacency)

        previous_latent_bmu = current_latent_bmu

    plt.ioff()
    plt.show()
    return latent_state, memory_state


# ----------------------------
# CLI / Entry
# ----------------------------

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--level", type=int, required=True)
    parser.add_argument(
        "--sensor",
        type=str,
        default="cardinal distance",
        choices=["cartesian", "cardinal distance"],
        help='Dataset sensor (quote if it contains spaces): --sensor "cardinal distance"',
    )
    parser.add_argument("--sensor_range", type=int, default=1)

    parser.add_argument("--datasets_root", type=str, default="datasets")
    parser.add_argument("--select", type=str, default="latest", choices=["latest", "run_id"])
    parser.add_argument("--run_id", type=str, default=None)

    parser.add_argument("--memory_length", type=int, default=10)

    # plotting controls
    parser.add_argument("--enable_training_plot", action="store_true")
    parser.add_argument("--training_plot_trigger", type=str, default="structure", choices=["always", "structure", "bmu"])
    parser.add_argument("--replay_plot_trigger", type=str, default="bmu", choices=["always", "structure", "bmu"])
    parser.add_argument("--training_pause_s", type=float, default=0.0002)
    parser.add_argument("--replay_pause_s", type=float, default=0.0005)
    parser.add_argument("--edge_curvature", type=float, default=0.18)

    # action map params (same AM used for both phases)
    parser.add_argument("--am_n_codebook", type=int, default=4)
    parser.add_argument("--am_lr", type=float, default=0.5)
    parser.add_argument("--am_sigma", type=float, default=0.0)
    parser.add_argument("--am_key", type=int, default=0)

    # sensory params (minimal knobs for live script)
    parser.add_argument("--activation_threshold", type=float, default=0.95)
    parser.add_argument("--sensory_weighting", type=float, default=0.6)
    parser.add_argument("--sensory_max_age", type=int, default=17)

    # latent params (minimal knobs)
    parser.add_argument("--latent_max_age", type=int, default=50)
    parser.add_argument("--latent_action_lr", type=float, default=0.1)

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    datasets_root = Path(args.datasets_root)
    loaded = load_level_parquet(
        ParquetConfig(
            root=datasets_root,
            level=args.level,
            sensor=args.sensor,
            sensor_range=args.sensor_range,
            select=args.select,
            run_id=args.run_id,
        )
    )

    observations = loaded.observations
    actions = loaded.actions
    collisions = loaded.collisions

    action_dim = int(actions.shape[1]) if actions.ndim == 2 else 1
    action_map = build_action_map(
        action_dim,
        n_codebook=args.am_n_codebook,
        lr=args.am_lr,
        sigma=args.am_sigma,
        key=args.am_key,
    )

    sensory_params = SensoryParams(
        activation_threshold=args.activation_threshold,
        sensory_weighting=args.sensory_weighting,
        max_age=args.sensory_max_age,
    )
    latent_params = LatentParams(
        max_age=args.latent_max_age,
        action_lr=args.latent_action_lr,
    )

    positioner = Positioner(layout="spring_layout")

    sensory_state, _ = train_sensory_over_dataset(
        observations=observations,
        actions=actions,
        collisions=collisions,
        sensory_params=sensory_params,
        action_map=action_map,
        training_plot_trigger=args.training_plot_trigger,
        training_pause_s=args.training_pause_s,
        edge_curvature=args.edge_curvature,
        positioner=positioner,
        enable_training_plot=bool(args.enable_training_plot),
    )

    latent_state, _ = replay_latent_with_live_plot(
        observations=observations,
        actions=actions,
        collisions=collisions,
        sensory_state=sensory_state,
        sensory_params=sensory_params,
        latent_params=latent_params,
        action_map=action_map,
        memory_length=args.memory_length,
        replay_plot_trigger=args.replay_plot_trigger,
        replay_pause_s=args.replay_pause_s,
        edge_curvature=args.edge_curvature,
    )

    print(f"n states in latent | {int(latent_state.g.n)}")


if __name__ == "__main__":
    main()
