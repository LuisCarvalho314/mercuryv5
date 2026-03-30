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


def map_to_indices(reference_uniques: np.ndarray, values: np.ndarray) -> np.ndarray:
    if reference_uniques.ndim == 1:
        lookup = {value.item() if hasattr(value, "item") else value: idx for idx, value in enumerate(reference_uniques.tolist())}
        return np.asarray([lookup[row.item() if hasattr(row, "item") else row] for row in np.asarray(values).tolist()], dtype=np.int64)
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
    if action_width == 1:
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
