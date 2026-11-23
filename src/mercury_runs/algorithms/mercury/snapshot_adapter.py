"""Mercury snapshot evaluation adapter.

Decodes evaluation walks using frozen winner-take-all inference.
Strip memory and transient inference state are reset per walk.
No graph updates (node templates, edge weights, etc.) are performed.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ...infrastructure.paper_precision import (
    action_ids_from_action_vectors,
    build_action_conditioned_tensor_from_graph,
    infer_num_actions_from_valid_transitions,
    exact_valid_sensorimotor_transitions,
)


def _infer_memory_length(snapshot: dict[str, Any]) -> int:
    meta = snapshot.get("meta", {})
    pp = meta.get("paper_precision", {})
    if "memory_length" in meta:
        return int(meta["memory_length"])
    if "memory_length" in pp:
        return int(pp["memory_length"])

    sensory_graph = snapshot["sensory_graph"]
    latent_graph = snapshot["latent_graph"]
    mem_adj = getattr(latent_graph, "node_features", {}).get("mem_adj")
    if mem_adj is not None:
        width = int(np.asarray(mem_adj).shape[1])
        n_sensory = int(sensory_graph.n)
        if n_sensory > 0 and width % n_sensory == 0:
            return int(width // n_sensory)

    return 10


def _decode_mercury_walk_frozen(
    *,
    walk_observations: np.ndarray,
    walk_actions: np.ndarray,
    walk_collisions: np.ndarray,
    sensory_graph: Any,
    latent_graph: Any,
    action_map: Any,
    sensory_params: Any,
    latent_params: Any,
    memory_length: int,
    obs_uniques: np.ndarray | None = None,
) -> np.ndarray:
    """Run frozen Mercury inference on one walk. Returns latent BMU per step."""
    from mercury.latent.state import LatentState, init_latent_state, latent_step_predict_only
    from mercury.memory.state import add_memory, init_mem, update_memory
    from mercury.sensory.state import SensoryState, init_global_context, sensory_step_predict_only
    from ...infrastructure.observation_encoding import encode_observations_with_vocabulary

    observations = np.asarray(walk_observations)
    if obs_uniques is not None:
        encoded = encode_observations_with_vocabulary(
            observations=observations, obs_uniques=obs_uniques
        )
        observations = encoded.observations

    observation_dim = int(observations.shape[1])
    sensory_state = SensoryState(
        gs=sensory_graph,
        global_context=init_global_context(observation_dim),
        mapping=np.arange(sensory_graph.n, dtype=np.int32),
    )
    mem = init_mem(sensory_graph.n, int(memory_length))
    try:
        latent_state = init_latent_state(mem, n_actions=int(action_map.state.codebook.shape[0]))
    except (TypeError, AttributeError):
        latent_state = init_latent_state(mem)
    latent_state.g = latent_graph
    latent_state.mapping = np.arange(latent_graph.n, dtype=np.int32)

    state_memory: list[int] = []
    latent_bmus: list[int] = []

    for observation, action, collision in zip(observations, walk_actions, walk_collisions, strict=False):
        action_vector = np.atleast_1d(action).astype(np.float32)
        action_bmu = int(action_map.predict(action=action_vector))
        sensory_state, activation_vector = sensory_step_predict_only(
            np.asarray(observation, dtype=np.float32),
            sensory_state,
            sensory_params,
        )
        sensory_graph_size = int(sensory_state.gs.n)
        if mem.gs.n != sensory_graph_size * int(memory_length):
            mem = init_mem(sensory_graph_size, length=int(memory_length))
        if not bool(collision):
            mem = update_memory(mem)
            mem = add_memory(mem, activation_vector)
            latent_state, _, state_memory = latent_step_predict_only(
                mem, latent_state, int(action_bmu), latent_params, state_memory
            )
        current_bmu = latent_state.prev_bmu
        latent_bmus.append(int(current_bmu) if current_bmu is not None else 0)

    return np.asarray(latent_bmus, dtype=np.int64)


class MercurySnapshotAdapter:
    """Load Mercury snapshots and decode evaluation walks with frozen WTA inference."""

    def load_snapshot(self, snapshot_path: Path) -> dict[str, Any]:
        from ...infrastructure.paper_precision_snapshots import load_mercury_paper_precision_snapshot
        return load_mercury_paper_precision_snapshot(snapshot_path)

    def decode_walks(
        self,
        snapshot: dict[str, Any],
        walk_data: dict[str, Any],
        *,
        level: int,
    ) -> dict[str, Any]:
        """Decode cached evaluation walks using frozen Mercury WTA inference."""
        from mercury.latent.params import LatentParams
        from mercury.sensory.params import SensoryParams

        meta = snapshot.get("meta", {})
        sensory_graph = snapshot["sensory_graph"]
        latent_graph = snapshot["latent_graph"]
        action_map = snapshot["action_map"]
        obs_uniques = snapshot.get("obs_uniques")

        # Retrieve params from snapshot config if available. Older handoff
        # snapshots omit memory_length, so infer it from latent mem_adj.
        memory_length = _infer_memory_length(snapshot)

        # SensoryParams and LatentParams: retrieve from config if stored, else use defaults.
        sensory_config = meta.get("sensory", {})
        latent_config = meta.get("latent", {})
        sensory_params = SensoryParams(**sensory_config) if sensory_config else SensoryParams()
        latent_params = LatentParams(**latent_config) if latent_config else LatentParams()

        decoded_walks: list[np.ndarray] = []
        gt_exact_walks: list[np.ndarray] = []
        gt_action_walks: list[np.ndarray] = []
        skipped = 0

        for filtered_obs, filtered_actions, walk_collisions, gt_exact, gt_action in zip(
            walk_data["filtered_obs_per_walk"],
            walk_data["filtered_actions_per_walk"],
            walk_data.get("collisions_per_walk", [None] * walk_data["n_walks"]),
            walk_data["gt_exact_per_walk"],
            walk_data["gt_action_per_walk"],
        ):
            if filtered_obs is None:
                skipped += 1
                continue

            if walk_collisions is None:
                walk_collisions = np.zeros(filtered_obs.shape[0], dtype=bool)

            try:
                latent_bmus = _decode_mercury_walk_frozen(
                    walk_observations=filtered_obs,
                    walk_actions=filtered_actions,
                    walk_collisions=walk_collisions,
                    sensory_graph=sensory_graph,
                    latent_graph=latent_graph,
                    action_map=action_map,
                    sensory_params=sensory_params,
                    latent_params=latent_params,
                    memory_length=memory_length,
                    obs_uniques=obs_uniques,
                )
            except Exception:
                skipped += 1
                continue

            length = min(int(latent_bmus.shape[0]), int(gt_exact.shape[0]))
            if length == 0:
                skipped += 1
                continue

            decoded_walks.append(latent_bmus[:length].astype(np.int64, copy=False))
            gt_exact_walks.append(gt_exact[:length])
            gt_action_walks.append(gt_action[:length])

        return {
            "decoded_walks": decoded_walks,
            "gt_exact_walks": gt_exact_walks,
            "gt_action_walks": gt_action_walks,
            "skipped_short_walks": skipped,
            "unsupported_walks": 0,
            "requested_walks": int(walk_data["n_walks"]),
            "decoder": "frozen_wta",
            "effective_batch_size": None,
            "parameter_updates": False,
            "state_reset": "per_walk",
            "initial_timesteps": "retained",
        }

    def transition_tensor(self, snapshot: dict[str, Any], *, n_actions: int) -> np.ndarray:
        latent_graph = snapshot["latent_graph"]
        return build_action_conditioned_tensor_from_graph(latent_graph, n_actions=n_actions)

    def n_learned_states(self, snapshot: dict[str, Any]) -> int:
        return int(snapshot["latent_graph"].n)

    def step(self, snapshot: dict[str, Any]) -> int:
        return int(snapshot.get("meta", {}).get("step", 0))

    def observed_samples(self, snapshot: dict[str, Any]) -> int:
        return int(snapshot.get("meta", {}).get("observed_samples", self.step(snapshot)))

    def inference_metadata(self, snapshot: dict[str, Any]) -> dict[str, Any]:
        return {
            "decoder": "frozen_wta",
            "effective_batch_size": None,
            "parameter_updates": False,
            "state_reset": "per_walk",
            "initial_timestep_policy": "include_all",
            "burn_in_steps": 0,
        }
