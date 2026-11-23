"""Sensory (TMGWR) snapshot evaluation adapter.

Decodes evaluation walks using frozen winner-take-all inference over the
sensory graph. Global context is the only state carried across a walk and it
is reset per walk. No graph updates (node templates, edge weights, ages) are
performed, so the snapshot is left untouched.

Used for standalone `sensory` runs and for the TMGWR bootstrap phase of
`tmgwr_cscg_hybrid` runs, whose snapshot lives under
`bundles/snapshots/sensory/paper_precision`.
"""
from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

from ...infrastructure.paper_precision import build_action_conditioned_tensor_from_graph


def _sensory_params_for_snapshot(snapshot: dict[str, Any]) -> Any:
    """Resolve the SensoryParams the run trained with.

    The snapshot metadata does not carry the sensory block, so it is read back
    from the owning run's run_config.json. Falling back to library defaults
    would silently decode with different BMU weighting than training used.
    """
    from mercury.sensory.params import SensoryParams

    config = snapshot.get("sensory_config")
    return SensoryParams(**config) if config else SensoryParams()


def _run_config_sensory_block(snapshot_path: Path) -> dict[str, Any] | None:
    # <run_root>/[bundles/]snapshots/sensory/paper_precision/step=N
    for candidate in list(snapshot_path.parents)[:5]:
        config_path = candidate / "run_config.json"
        if not config_path.exists():
            continue
        try:
            payload = json.loads(config_path.read_text(encoding="utf-8"))
        except Exception:
            return None
        block = payload.get("run_payload", {}).get("sensory")
        return dict(block) if isinstance(block, dict) else None
    return None


class SensorySnapshotAdapter:
    """Load sensory snapshots and decode evaluation walks with frozen WTA inference."""

    def load_snapshot(self, snapshot_path: Path) -> dict[str, Any]:
        from ...infrastructure.paper_precision_snapshots import load_sensory_paper_precision_snapshot
        snapshot = load_sensory_paper_precision_snapshot(snapshot_path)
        snapshot["sensory_config"] = _run_config_sensory_block(Path(snapshot_path))
        return snapshot

    def decode_walks(
        self,
        snapshot: dict[str, Any],
        walk_data: dict[str, Any],
        *,
        level: int,
    ) -> dict[str, Any]:
        """Decode cached evaluation walks using frozen sensory WTA inference."""
        from .api import _run_sensory_on_walk

        sensory_graph = snapshot["sensory_graph"]
        action_map = snapshot["action_map"]
        obs_uniques = snapshot.get("obs_uniques")
        sensory_params = _sensory_params_for_snapshot(snapshot)

        decoded_walks: list[np.ndarray] = []
        gt_exact_walks: list[np.ndarray] = []
        gt_action_walks: list[np.ndarray] = []
        skipped = 0

        for filtered_obs, filtered_actions, gt_exact, gt_action in zip(
            walk_data["filtered_obs_per_walk"],
            walk_data["filtered_actions_per_walk"],
            walk_data["gt_exact_per_walk"],
            walk_data["gt_action_per_walk"],
        ):
            if filtered_obs is None:
                skipped += 1
                continue

            try:
                sensory_bmus = _run_sensory_on_walk(
                    observations=np.asarray(filtered_obs),
                    actions=np.asarray(filtered_actions),
                    sensory_graph=deepcopy(sensory_graph),
                    action_map=deepcopy(action_map),
                    sensory_params=sensory_params,
                    obs_uniques=obs_uniques,
                )
            except Exception:
                skipped += 1
                continue

            length = min(int(sensory_bmus.shape[0]), int(gt_exact.shape[0]))
            if length == 0:
                skipped += 1
                continue

            decoded_walks.append(sensory_bmus[:length].astype(np.int64, copy=False))
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
        return build_action_conditioned_tensor_from_graph(snapshot["sensory_graph"], n_actions=n_actions)

    def n_learned_states(self, snapshot: dict[str, Any]) -> int:
        return int(snapshot["sensory_graph"].n)

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
