"""CSCG snapshot evaluation adapter.

Loads frozen CSCG snapshots and assigns latent states on evaluation walks using
the same max-product MAP decoder used for CSCG run artifacts.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .prepare import (
    action_to_indices,
    align_post_step_actions_to_cscg,
    filter_valid_cscg_rows,
    import_cscg_module,
    map_to_indices,
)
from ...infrastructure.paper_precision import map_values_with_reference, action_ids_from_action_vectors


def _build_cscg_model_from_snapshot(
    *,
    chmm_mod: Any,
    model_npz: Any,
    n_clones: np.ndarray,
    obs_idx_seed: np.ndarray,
    act_idx_seed: np.ndarray,
    pseudocount: float = 2e-3,
    seed: int = 0,
) -> Any:
    model = chmm_mod.CHMM(
        n_clones=n_clones,
        x=obs_idx_seed,
        a=act_idx_seed,
        pseudocount=float(pseudocount),
        dtype=np.float32,
        seed=int(seed),
    )
    model.T = np.asarray(model_npz["T"], dtype=np.float32)
    return model


def decode_walk_frozen_online_batch(
    *,
    model: Any,
    obs_idx: np.ndarray,
    act_idx: np.ndarray,
    batch_size: int,
    chmm_mod: Any,
) -> np.ndarray:
    """Frozen recursive online batch inference for one walk.

    Process observations in batches of `batch_size`, propagating the terminal
    forward message as the initial distribution for the next batch.
    No M-step is performed; model parameters remain frozen.

    Returns: (length,) int64 hard decoded state assignments.
    """
    n_total = int(np.asarray(model.n_clones, dtype=np.int64).sum())
    T_t = model.T.transpose(0, 2, 1)
    Pi_x = np.asarray(model.Pi_x, dtype=np.float64)

    all_states: list[int] = []
    seq_len = int(obs_idx.shape[0])
    cscg_act_idx = align_post_step_actions_to_cscg(act_idx)

    for start in range(0, seq_len, batch_size):
        end = min(start + batch_size, seq_len)
        batch_obs = obs_idx[start:end]
        batch_act = cscg_act_idx[start:end]
        if batch_obs.shape[0] == 0:
            continue

        try:
            _log2_lik, mess_fwd = chmm_mod.forward(
                T_t,
                Pi_x,
                model.n_clones,
                batch_obs,
                batch_act,
                store_messages=True,
            )
        except Exception:
            # If forward fails for this batch, assign -1 for each step.
            all_states.extend([-1] * int(batch_obs.shape[0]))
            continue

        mess_fwd = np.asarray(mess_fwd, dtype=np.float64)
        # Hard state: argmax at each timestep.
        batch_states = np.argmax(mess_fwd, axis=-1).tolist()
        all_states.extend(batch_states)

        # Propagate terminal forward message as initial distribution.
        Pi_x = mess_fwd[-1].copy()
        Pi_sum = float(Pi_x.sum())
        if Pi_sum > 0:
            Pi_x /= Pi_sum
        else:
            Pi_x = np.ones(n_total, dtype=np.float64) / max(1, n_total)

    return np.asarray(all_states, dtype=np.int64)


class CSCGSnapshotAdapter:
    """Load CSCG snapshots and decode evaluation walks with frozen model parameters."""

    def __init__(self, *, pseudocount: float = 2e-3, seed: int = 0) -> None:
        self._pseudocount = float(pseudocount)
        self._seed = int(seed)
        self._chmm_mod: Any = None
        self._indexed_walk_cache_key: tuple[str, str] | None = None
        self._indexed_walk_cache: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None] | None = None

    def _get_chmm_mod(self) -> Any:
        if self._chmm_mod is None:
            self._chmm_mod = import_cscg_module(Path.cwd())
        return self._chmm_mod

    def load_snapshot(self, snapshot_path: Path) -> dict[str, Any]:
        from ...infrastructure.paper_precision_snapshots import load_cscg_paper_precision_snapshot
        return load_cscg_paper_precision_snapshot(snapshot_path)

    @staticmethod
    def _array_fingerprint(values: np.ndarray) -> str:
        arr = np.ascontiguousarray(values)
        digest = hashlib.sha1()
        digest.update(str(arr.shape).encode("utf-8"))
        digest.update(str(arr.dtype).encode("utf-8"))
        digest.update(arr.tobytes())
        return digest.hexdigest()

    def _indexed_walks_for_vocab(
        self,
        *,
        walk_data: dict[str, Any],
        obs_uniques: np.ndarray,
        action_uniques: np.ndarray,
    ) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None]:
        cache_key = (
            self._array_fingerprint(obs_uniques),
            self._array_fingerprint(action_uniques),
        )
        if self._indexed_walk_cache_key == cache_key and self._indexed_walk_cache is not None:
            return self._indexed_walk_cache

        indexed_walks: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None] = []
        for filtered_obs, filtered_actions, gt_exact, gt_action in zip(
            walk_data["filtered_obs_per_walk"],
            walk_data["filtered_actions_per_walk"],
            walk_data["gt_exact_per_walk"],
            walk_data["gt_action_per_walk"],
        ):
            if filtered_obs is None:
                indexed_walks.append(None)
                continue
            try:
                obs_idx = map_values_with_reference(
                    reference_uniques=obs_uniques, values=filtered_obs
                ).astype(np.int64, copy=False)
                act_idx = map_to_indices(action_uniques, filtered_actions).astype(np.int64, copy=False)
            except (ValueError, KeyError):
                indexed_walks.append(None)
                continue
            indexed_walks.append((obs_idx, act_idx, gt_exact, gt_action))

        self._indexed_walk_cache_key = cache_key
        self._indexed_walk_cache = indexed_walks
        return indexed_walks

    def decode_walks(
        self,
        snapshot: dict[str, Any],
        walk_data: dict[str, Any],
        *,
        batch_size: int | None = None,
    ) -> dict[str, Any]:
        """Decode cached evaluation walks using frozen max-product MAP inference.

        Returns a dict with decoded_walks and coverage statistics.
        """
        chmm_mod = self._get_chmm_mod()
        transition_tensor = np.asarray(snapshot.get("T", snapshot["model_npz"]["T"]), dtype=np.float32)
        n_clones = np.asarray(snapshot["n_clones"], dtype=np.int64)
        obs_uniques = np.asarray(snapshot["obs_uniques"])
        action_uniques = np.asarray(snapshot["action_uniques"])

        # Effective batch size from snapshot config, or default to full walk.
        meta = snapshot.get("meta", {})
        if batch_size is None:
            batch_size = int(meta.get("batch_size", int(meta.get("paper_precision", {}).get("walk_length", 1024))))

        decoded_walks: list[np.ndarray] = []
        gt_exact_walks: list[np.ndarray] = []
        gt_action_walks: list[np.ndarray] = []
        skipped = 0
        unsupported = 0
        indexed_walks = self._indexed_walks_for_vocab(
            walk_data=walk_data,
            obs_uniques=obs_uniques,
            action_uniques=action_uniques,
        )

        total_walks = len(walk_data["filtered_obs_per_walk"])
        for walk_index, indexed_walk in enumerate(indexed_walks, start=1):
            if indexed_walk is None:
                skipped += 1
                continue
            if walk_index == 1 or walk_index == total_walks or walk_index % 10 == 0:
                print(f"      decode walk {walk_index}/{total_walks}", flush=True)

            obs_idx, act_idx, gt_exact, gt_action = indexed_walk

            if obs_idx.shape[0] == 0 or act_idx.shape[0] == 0:
                skipped += 1
                continue

            try:
                model = chmm_mod.CHMM(
                    n_clones=n_clones,
                    x=obs_idx,
                    a=act_idx,
                    pseudocount=float(self._pseudocount),
                    dtype=np.float32,
                    seed=int(self._seed),
                )
                model.T = transition_tensor
                _, clone_states = model.decode(obs_idx, align_post_step_actions_to_cscg(act_idx))
                states = np.asarray(clone_states, dtype=np.int64)
            except Exception:
                unsupported += 1
                continue

            length = min(int(states.shape[0]), int(gt_exact.shape[0]))
            if length == 0:
                skipped += 1
                continue

            decoded_walks.append(states[:length].astype(np.int64, copy=False))
            gt_exact_walks.append(gt_exact[:length])
            gt_action_walks.append(gt_action[:length])

        return {
            "decoded_walks": decoded_walks,
            "gt_exact_walks": gt_exact_walks,
            "gt_action_walks": gt_action_walks,
            "skipped_short_walks": skipped,
            "unsupported_walks": unsupported,
            "requested_walks": int(walk_data["n_walks"]),
            "decoder": "max_product_map_frozen",
            "parameter_updates": False,
        }

    def negative_log2_likelihood(
        self,
        snapshot: dict[str, Any],
        walk_data: dict[str, Any],
    ) -> dict[str, Any]:
        """Score frozen CSCG snapshot on cached evaluation walks.

        This is the evaluation analogue of the EM training objective:
        `-mean(log2 p(x_t | x_<t, a_<t, T))`. No messages are reused between
        walks and no parameters are updated.
        """
        chmm_mod = self._get_chmm_mod()
        transition_tensor = np.asarray(snapshot.get("T", snapshot["model_npz"]["T"]), dtype=np.float32)
        n_clones = np.asarray(snapshot["n_clones"], dtype=np.int64)
        obs_uniques = np.asarray(snapshot["obs_uniques"])
        action_uniques = np.asarray(snapshot["action_uniques"])
        indexed_walks = self._indexed_walks_for_vocab(
            walk_data=walk_data,
            obs_uniques=obs_uniques,
            action_uniques=action_uniques,
        )

        total = 0.0
        count = 0
        skipped = 0
        unsupported = 0
        for indexed_walk in indexed_walks:
            if indexed_walk is None:
                skipped += 1
                continue
            obs_idx, act_idx, _, _ = indexed_walk
            if obs_idx.shape[0] == 0 or act_idx.shape[0] == 0:
                skipped += 1
                continue
            try:
                model = chmm_mod.CHMM(
                    n_clones=n_clones,
                    x=obs_idx,
                    a=act_idx,
                    pseudocount=float(self._pseudocount),
                    dtype=np.float32,
                    seed=int(self._seed),
                )
                model.T = transition_tensor
                cscg_actions = align_post_step_actions_to_cscg(act_idx)
                log2_lik, _ = chmm_mod.forward(
                    model.T.transpose(0, 2, 1),
                    model.Pi_x,
                    model.n_clones,
                    obs_idx,
                    cscg_actions,
                    store_messages=False,
                )
            except Exception:
                unsupported += 1
                continue
            values = -np.asarray(log2_lik, dtype=np.float64)
            total += float(values.sum())
            count += int(values.shape[0])

        return {
            "negative_log2_likelihood_sum": total,
            "negative_log2_likelihood_mean": (total / float(count)) if count > 0 else None,
            "bits_per_symbol": (total / float(count)) if count > 0 else None,
            "scored_symbols": int(count),
            "skipped_walks": int(skipped),
            "unsupported_walks": int(unsupported),
            "parameter_updates": False,
            "state_reset": "per_walk",
        }

    def transition_tensor(self, snapshot: dict[str, Any]) -> np.ndarray:
        return np.asarray(snapshot["T"], dtype=np.float64)

    def n_learned_states(self, snapshot: dict[str, Any]) -> int:
        return int(np.asarray(snapshot["n_clones"], dtype=np.int64).sum())

    def step(self, snapshot: dict[str, Any]) -> int:
        return int(snapshot.get("meta", {}).get("step", 0))

    def observed_samples(self, snapshot: dict[str, Any]) -> int:
        return int(snapshot.get("meta", {}).get("observed_samples", self.step(snapshot)))

    def inference_metadata(self, snapshot: dict[str, Any], *, batch_size: int) -> dict[str, Any]:
        meta = snapshot.get("meta", {})
        return {
            "decoder": "max_product_map_frozen",
            "effective_batch_size": batch_size,
            "parameter_updates": False,
            "state_reset": "per_walk",
            "initial_timestep_policy": "include_all",
            "burn_in_steps": 0,
            "training_mode": str(meta.get("training_mode", "online_em")),
        }
