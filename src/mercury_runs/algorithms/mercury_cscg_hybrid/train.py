from __future__ import annotations

from typing import Any

import numpy as np

from .prepare import HybridAssignment
from ..cscg.structural_init import (
    apply_observation_tensor_to_clone_counts,
    clone_edges_from_graph,
    clone_edges_from_oracle_topology,
    normalize_structural_options,
    observation_edges_from_empirical_ground_truth,
    observation_edges_from_graph,
    observation_edges_from_oracle_topology,
)


def initialize_cscg_from_mercury_graph(
    *,
    chmm_mod: Any,
    latent_graph: Any,
    assignment: HybridAssignment,
    obs_idx: np.ndarray,
    act_idx: np.ndarray,
    pseudocount: float,
    seed: int,
    transfer_level: str = "clone_state",
    init_mode: str = "binary",
    structural_source: str = "source_graph",
    level: int | None = None,
    ground_truth_states: np.ndarray | None = None,
) -> Any:
    transfer_level, init_mode, structural_source, _ = normalize_structural_options(
        transfer_level=transfer_level,
        init_mode=init_mode,
        structural_source=structural_source,
        capacity_mode="source",
    )
    n_obs = int(assignment.n_clones.shape[0])
    n_actions = int(np.max(act_idx)) + 1
    seed_obs = np.asarray([0, min(n_obs - 1, int(np.max(obs_idx))), 0], dtype=np.int64)
    seed_actions = np.asarray([0, max(0, n_actions - 1), 0], dtype=np.int64)
    model = chmm_mod.CHMM(
        n_clones=assignment.n_clones.astype(np.int64, copy=False),
        x=seed_obs,
        a=seed_actions,
        pseudocount=float(pseudocount),
        dtype=np.float32,
        seed=int(seed),
    )
    if init_mode == "none" or structural_source == "none":
        counts = np.zeros_like(model.C, dtype=np.float32)
    elif structural_source == "oracle_maze_topology" and transfer_level == "clone_state":
        if level is None or ground_truth_states is None:
            raise ValueError("Oracle clone/state hybrid transfer requires level and aligned ground-truth states.")
        counts, _ = clone_edges_from_oracle_topology(
            level=int(level),
            obs_idx=obs_idx,
            action_idx=act_idx,
            ground_truth_states=np.asarray(ground_truth_states),
            n_obs=n_obs,
            n_actions=n_actions,
            clones_per_state=1,
            mode=init_mode,
        )
    elif structural_source in {"oracle_maze_topology", "oracle_empirical_ground_truth"}:
        if level is None or ground_truth_states is None:
            raise ValueError("Oracle first-order hybrid transfer requires level and aligned ground-truth states.")
        if structural_source == "oracle_maze_topology":
            obs_tensor = observation_edges_from_oracle_topology(
                level=int(level),
                obs_idx=obs_idx,
                action_idx=act_idx,
                ground_truth_states=np.asarray(ground_truth_states),
                n_obs=n_obs,
                n_actions=n_actions,
                mode=init_mode,
            )
        else:
            obs_tensor = observation_edges_from_empirical_ground_truth(
                obs_idx=obs_idx,
                action_idx=act_idx,
                ground_truth_states=np.asarray(ground_truth_states),
                n_obs=n_obs,
                n_actions=n_actions,
                mode=init_mode,
            )
        counts = apply_observation_tensor_to_clone_counts(observation_tensor=obs_tensor, n_clones=assignment.n_clones)
    elif transfer_level == "first_order_observation":
        obs_tensor = observation_edges_from_graph(
            graph=latent_graph,
            node_to_observation=assignment.node_to_observation,
            n_obs=n_obs,
            n_actions=n_actions,
            mode=init_mode,
        )
        counts = apply_observation_tensor_to_clone_counts(observation_tensor=obs_tensor, n_clones=assignment.n_clones)
    else:
        counts = clone_edges_from_graph(
            graph=latent_graph,
            node_to_state=assignment.node_to_state,
            n_states=int(assignment.n_clones.sum()),
            n_actions=n_actions,
            mode=init_mode,
        )
    if counts.shape != model.C.shape:
        raise ValueError(f"Hybrid structural counts shape {counts.shape} does not match model.C shape {model.C.shape}.")
    model.C = counts.astype(model.dtype, copy=False)
    model.update_T()
    return model
