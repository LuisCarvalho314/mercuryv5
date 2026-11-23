from __future__ import annotations

import importlib
import importlib.util
import json
import sys
import types
from pathlib import Path
from typing import Any, Iterator, Optional

import numpy as np
import polars as pl

from ...infrastructure.state_columns import load_ground_truth_proxy_states

from ...io_parquet import ParquetConfig, dataset_directory, load_level_parquet, select_parquet_path


def import_cscg_module(repo_root: Path) -> Any:
    cscg_path = repo_root / "external" / "naturecomm_cscg"
    if not cscg_path.exists():
        raise FileNotFoundError(
            f"CSCG repo not found at {cscg_path}. Clone it with: "
            f"git clone https://github.com/vicariousinc/naturecomm_cscg {cscg_path}"
        )
    if str(cscg_path) not in sys.path:
        sys.path.insert(0, str(cscg_path))
    loaded = sys.modules.get("chmm_actions")
    loaded_file = getattr(loaded, "__file__", "") if loaded is not None else ""
    if loaded is not None and loaded_file and str(cscg_path) not in str(loaded_file):
        del sys.modules["chmm_actions"]
    if importlib.util.find_spec("numba") is None and "numba" not in sys.modules:
        shim = types.ModuleType("numba")
        shim.njit = lambda fn: fn
        sys.modules["numba"] = shim
    try:
        return importlib.import_module("chmm_actions")
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "Failed to import CSCG modules. Ensure dependencies are installed "
            "(e.g. `pip install -r external/naturecomm_cscg/requirements.txt`)."
        ) from exc


def load_dataset_arrays(
    *,
    datasets_root: Path,
    level: int,
    sensor: str,
    sensor_range: Optional[int],
    dataset_select: str,
    dataset_run_id: Optional[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any], str]:
    loaded = load_level_parquet(
        ParquetConfig(
            root=datasets_root,
            level=level,
            sensor=sensor,
            sensor_range=sensor_range,
            select=dataset_select,
            run_id=dataset_run_id,
        )
    )
    obs = np.asarray(loaded.observations)
    act = np.asarray(loaded.actions)
    collisions = np.asarray(loaded.collisions, dtype=bool)
    if act.ndim > 1 and act.shape[1] == 1:
        act = act.squeeze(1)
    return obs, act, collisions, (loaded.source_metadata or {}), loaded.parquet_path.name


def resolve_dataset_path_and_metadata(
    *,
    datasets_root: Path,
    level: int,
    sensor: str,
    sensor_range: Optional[int],
    dataset_select: str,
    dataset_run_id: Optional[str],
) -> tuple[Path, dict[str, Any]]:
    config = ParquetConfig(
        root=datasets_root,
        level=level,
        sensor=sensor,
        sensor_range=sensor_range,
        select=dataset_select,
        run_id=dataset_run_id,
    )
    parquet_path = select_parquet_path(dataset_directory(config), config.select, config.run_id)
    metadata_path = parquet_path.with_suffix(".metadata.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    return parquet_path, (metadata if isinstance(metadata, dict) else {})


def load_ground_truth_states(path: Path) -> np.ndarray:
    return load_ground_truth_proxy_states(path)


def to_indices(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if values.ndim == 1:
        uniques, inverse = np.unique(values, return_inverse=True)
        return inverse.astype(np.int64, copy=False), np.asarray(uniques)
    uniques, inverse = np.unique(values, axis=0, return_inverse=True)
    return inverse.astype(np.int64, copy=False), np.asarray(uniques)


def _is_one_hot_actions(values: np.ndarray) -> bool:
    action_array = np.asarray(values)
    if action_array.ndim != 2 or action_array.shape[1] == 0:
        return False
    return bool(
        np.allclose(action_array.sum(axis=1), 1.0)
        and np.all(np.count_nonzero(action_array > 0.5, axis=1) == 1)
    )


def action_to_indices(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    action_array = np.asarray(values)
    if _is_one_hot_actions(action_array):
        action_width = int(action_array.shape[1])
        return (
            np.argmax(action_array, axis=1).astype(np.int64, copy=False),
            np.eye(action_width, dtype=action_array.dtype),
        )
    return to_indices(action_array)


def prepare_cscg_transition_inputs(
    observations: np.ndarray,
    post_step_actions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert persisted post-step samples into CSCG transition tuples.

    Persisted sample ``t`` contains the observation reached by
    ``post_step_actions[t]``.  Therefore the transition between two persisted
    observations is ``(observation[t], post_step_actions[t + 1],
    observation[t + 1])``.
    """
    observation_array = np.asarray(observations)
    action_array = np.asarray(post_step_actions)
    if observation_array.ndim == 0 or action_array.ndim == 0:
        raise ValueError("CSCG observations and actions must have a leading timestep dimension.")
    if observation_array.shape[0] != action_array.shape[0]:
        raise ValueError("Persisted CSCG observations and actions must have matching lengths.")
    return observation_array[:-1], action_array[1:], observation_array[1:]


def align_post_step_actions_to_cscg(actions: np.ndarray) -> np.ndarray:
    """Convert environment post-step actions to CSCG transition indexing.

    Maze trajectories store ``action[t]`` beside the observation produced by
    that action.  Consequently, the transition ``observation[t] ->
    observation[t + 1]`` was caused by ``action[t + 1]``.  The vendored CSCG
    implementation instead consumes ``a[t]`` for that transition.

    CSCG expects an action array with the same length as the observation array,
    although its final element is unused.  Shift left and repeat the final
    action to preserve that shape without inventing a new action identifier.
    """
    action_array = np.asarray(actions)
    if action_array.ndim == 0:
        raise ValueError("CSCG actions must have a leading timestep dimension.")
    aligned = np.array(action_array, copy=True)
    if aligned.shape[0] > 1:
        placeholder_observations = np.empty((aligned.shape[0], 0), dtype=np.uint8)
        _, transition_actions, _ = prepare_cscg_transition_inputs(
            placeholder_observations,
            action_array,
        )
        aligned[:-1] = transition_actions
    return aligned


def map_to_indices(reference_uniques: np.ndarray, values: np.ndarray) -> np.ndarray:
    if reference_uniques.ndim == 1:
        lookup = {value.item() if hasattr(value, "item") else value: idx for idx, value in enumerate(reference_uniques.tolist())}
        return np.asarray([lookup[row.item() if hasattr(row, "item") else row] for row in np.asarray(values).tolist()], dtype=np.int64)
    if _is_one_hot_actions(np.asarray(values)) and np.asarray(reference_uniques).shape == (
        np.asarray(values).shape[1],
        np.asarray(values).shape[1],
    ) and np.allclose(reference_uniques, np.eye(np.asarray(values).shape[1], dtype=np.asarray(reference_uniques).dtype)):
        return np.argmax(np.asarray(values), axis=1).astype(np.int64, copy=False)
    lookup = {tuple(np.asarray(row).tolist()): idx for idx, row in enumerate(reference_uniques)}
    return np.asarray([lookup[tuple(np.asarray(row).tolist())] for row in np.asarray(values)], dtype=np.int64)


def filter_valid_cscg_rows(
    *,
    observations: np.ndarray,
    actions: np.ndarray,
    collisions: np.ndarray,
    ground_truth_bmu: np.ndarray | None = None,
    valid_trajectories_only: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, dict[str, int]]:
    obs = np.asarray(observations)
    act = np.asarray(actions)
    coll = np.asarray(collisions, dtype=bool)
    if obs.shape[0] != act.shape[0] or obs.shape[0] != coll.shape[0]:
        raise ValueError("observations, actions, and collisions must have matching row counts for CSCG filtering.")
    if not bool(valid_trajectories_only):
        filtered_gt = None if ground_truth_bmu is None else np.asarray(ground_truth_bmu)
        return (
            obs,
            act,
            filtered_gt,
            {
                "raw_steps": int(obs.shape[0]),
                "filtered_steps": int(obs.shape[0]),
                "dropped_collision_steps": 0,
            },
        )
    keep_mask = ~coll
    filtered_gt = None if ground_truth_bmu is None else np.asarray(ground_truth_bmu)[keep_mask]
    return (
        obs[keep_mask],
        act[keep_mask],
        filtered_gt,
        {
            "raw_steps": int(obs.shape[0]),
            "filtered_steps": int(keep_mask.sum()),
            "dropped_collision_steps": int(coll.sum()),
        },
    )


def iter_cscg_training_batches(
    *,
    parquet_path: Path,
    batch_size: int,
    include_prev: bool,
    valid_trajectories_only: bool,
) -> Iterator[dict[str, Any]]:
    source = pl.scan_parquet(parquet_path)
    schema = source.collect_schema()
    observation_columns = [name for name in schema.names() if name.startswith("observation_")]
    action_columns = [name for name in schema.names() if name.startswith("action_")]
    if not observation_columns:
        raise ValueError(f"No observation_ columns found in {parquet_path.name}")
    if not action_columns:
        raise ValueError(f"No action_ columns found in {parquet_path.name}")
    if "collision" not in schema:
        raise ValueError(f"Missing required column 'collision' in {parquet_path.name}")

    previous_valid_obs: np.ndarray | None = None
    previous_valid_act: np.ndarray | None = None
    offset = 0
    while True:
        frame = source.slice(offset, batch_size).collect()
        if frame.height == 0:
            break
        offset += frame.height
        observations = frame.select(observation_columns).to_numpy()
        actions = frame.select(action_columns).to_numpy()
        collisions = frame.get_column("collision").to_numpy().astype(bool, copy=False)
        filtered_obs, filtered_actions, _, stats = filter_valid_cscg_rows(
            observations=observations,
            actions=actions,
            collisions=collisions,
            valid_trajectories_only=bool(valid_trajectories_only),
        )
        prepended_previous_step = False
        payload_obs = filtered_obs
        payload_actions = filtered_actions
        if include_prev and filtered_obs.shape[0] > 0 and previous_valid_obs is not None and previous_valid_act is not None:
            payload_obs = np.concatenate([previous_valid_obs[None, ...], filtered_obs], axis=0)
            payload_actions = np.concatenate([previous_valid_act[None, ...], filtered_actions], axis=0)
            prepended_previous_step = True
        if filtered_obs.shape[0] > 0:
            previous_valid_obs = np.array(filtered_obs[-1], copy=True)
            previous_valid_act = np.array(filtered_actions[-1], copy=True)
        yield {
            "observations": payload_obs,
            "actions": payload_actions,
            "raw_steps": int(observations.shape[0]),
            "filtered_steps": int(filtered_obs.shape[0]),
            "dropped_collision_steps": int(stats["dropped_collision_steps"]),
            "prepended_previous_step": prepended_previous_step,
        }


def collect_streaming_cscg_metadata(
    *,
    parquet_path: Path,
    batch_size: int,
    valid_trajectories_only: bool,
) -> tuple[np.ndarray, np.ndarray, dict[str, int], int]:
    stats = {"raw_steps": 0, "filtered_steps": 0, "dropped_collision_steps": 0}
    obs_unique_rows: set[tuple[Any, ...] | Any] = set()
    action_unique_rows: set[tuple[Any, ...] | Any] = set()
    one_hot_action_width: int | None = None
    obs_width: int | None = None
    action_width: int | None = None
    non_empty_batch_count = 0
    for batch in iter_cscg_training_batches(
        parquet_path=parquet_path,
        batch_size=batch_size,
        include_prev=False,
        valid_trajectories_only=bool(valid_trajectories_only),
    ):
        stats["raw_steps"] += int(batch["raw_steps"])
        stats["filtered_steps"] += int(batch["filtered_steps"])
        stats["dropped_collision_steps"] += int(batch["dropped_collision_steps"])
        if int(batch["filtered_steps"]) == 0:
            continue
        non_empty_batch_count += 1
        observations = np.asarray(batch["observations"])
        actions = np.asarray(batch["actions"])
        obs_width = int(observations.shape[1]) if observations.ndim > 1 else 1
        action_width = int(actions.shape[1]) if actions.ndim > 1 else 1
        if _is_one_hot_actions(actions):
            one_hot_action_width = max(int(one_hot_action_width or 0), int(actions.shape[1]))
        if observations.ndim == 1:
            obs_unique_rows.update(observations.tolist())
        else:
            obs_unique_rows.update(tuple(row.tolist()) for row in observations)
        if actions.ndim == 1:
            action_unique_rows.update(actions.tolist())
        else:
            action_unique_rows.update(tuple(row.tolist()) for row in actions)
    if not obs_unique_rows or not action_unique_rows or obs_width is None or action_width is None:
        raise ValueError("CSCG baseline could not find any non-collision training rows in the dataset.")
    ordered_obs = sorted(obs_unique_rows)
    ordered_actions = sorted(action_unique_rows)
    if obs_width == 1:
        obs_uniques = np.asarray(ordered_obs)
    else:
        obs_uniques = np.asarray(ordered_obs).reshape(-1, obs_width)
    if one_hot_action_width is not None:
        action_uniques = np.eye(one_hot_action_width, dtype=np.asarray(ordered_actions).dtype)
    elif action_width == 1:
        action_uniques = np.asarray(ordered_actions)
    else:
        action_uniques = np.asarray(ordered_actions).reshape(-1, action_width)
    return obs_uniques, action_uniques, stats, non_empty_batch_count


def fit_state_to_gt_map(state_indices: np.ndarray, gt_indices: np.ndarray) -> np.ndarray:
    n_state = int(state_indices.max()) + 1
    n_gt = int(gt_indices.max()) + 1
    counts = np.zeros((n_state, n_gt), dtype=np.int64)
    for state_i, gt_i in zip(state_indices.tolist(), gt_indices.tolist()):
        counts[int(state_i), int(gt_i)] += 1
    return counts.argmax(axis=1).astype(np.int32, copy=False)
