from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np
import polars as pl

from ...infrastructure.state_columns import load_ground_truth_proxy_states


def import_pocml_modules(repo_root: Path) -> tuple[Any, Any]:
    pocml_path = repo_root / "external" / "POCML"
    if not pocml_path.exists():
        raise FileNotFoundError(
            f"POCML repo not found at {pocml_path}. Clone it with: "
            f"git clone https://github.com/calvin4770/POCML {pocml_path}"
        )
    if str(pocml_path) not in sys.path:
        sys.path.insert(0, str(pocml_path))
    for module_name in ("model", "trainer", "dataloader"):
        loaded = sys.modules.get(module_name)
        loaded_file = getattr(loaded, "__file__", "") if loaded is not None else ""
        if loaded is not None and loaded_file and str(pocml_path) not in str(loaded_file):
            del sys.modules[module_name]
    try:
        model_mod = importlib.import_module("model")
        trainer_mod = importlib.import_module("trainer")
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "Failed to import POCML modules. Ensure POCML dependencies are installed (e.g. torch)."
        ) from exc
    return model_mod, trainer_mod


def load_dataset_arrays(
    *,
    datasets_root: Path,
    level: int,
    sensor: str,
    sensor_range: Optional[int],
    dataset_select: str,
    dataset_run_id: Optional[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any], str]:
    from ...io_parquet import ParquetConfig, load_level_parquet

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


def to_indices(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if values.ndim == 1:
        uniques, inverse = np.unique(values, return_inverse=True)
        return inverse.astype(np.int64, copy=False), np.asarray(uniques)
    uniques, inverse = np.unique(values, axis=0, return_inverse=True)
    return inverse.astype(np.int64, copy=False), np.asarray(uniques)


def filter_valid_pocml_rows(
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
        raise ValueError("observations, actions, and collisions must have matching row counts for POCML filtering.")
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


def load_ground_truth_states(path: Path) -> np.ndarray:
    return load_ground_truth_proxy_states(path)


def fit_obs_to_gt_map(obs_indices: np.ndarray, gt_indices: np.ndarray) -> np.ndarray:
    n_obs = int(obs_indices.max()) + 1
    n_gt = int(gt_indices.max()) + 1
    counts = np.zeros((n_obs, n_gt), dtype=np.int64)
    for obs_i, gt_i in zip(obs_indices.tolist(), gt_indices.tolist()):
        counts[int(obs_i), int(gt_i)] += 1
    return counts.argmax(axis=1).astype(np.int32, copy=False)


def fit_state_to_gt_map(state_indices: np.ndarray, gt_indices: np.ndarray) -> np.ndarray:
    n_state = int(state_indices.max()) + 1
    n_gt = int(gt_indices.max()) + 1
    counts = np.zeros((n_state, n_gt), dtype=np.int64)
    for state_i, gt_i in zip(state_indices.tolist(), gt_indices.tolist()):
        counts[int(state_i), int(gt_i)] += 1
    return counts.argmax(axis=1).astype(np.int32, copy=False)


def build_train_trajectories(
    *,
    o_pre: np.ndarray,
    a: np.ndarray,
    o_next: np.ndarray,
    node_pre: np.ndarray,
    node_next: np.ndarray,
    trajectory_length: int,
    max_trajectories: Optional[int],
    torch_module: Any,
) -> list[Any]:
    total_steps = int(o_pre.shape[0])
    chunk_len = max(1, int(trajectory_length))
    chunks: list[Any] = []
    if total_steps < chunk_len:
        block = np.stack([o_pre, a, o_next, node_pre, node_next], axis=1).astype(np.int64, copy=False)
        chunks.append(torch_module.tensor(block, dtype=torch_module.long))
        return chunks
    start = 0
    while start + chunk_len <= total_steps:
        end = start + chunk_len
        block = np.stack(
            [o_pre[start:end], a[start:end], o_next[start:end], node_pre[start:end], node_next[start:end]],
            axis=1,
        ).astype(np.int64, copy=False)
        chunks.append(torch_module.tensor(block, dtype=torch_module.long))
        start = end
        if max_trajectories is not None and len(chunks) >= int(max_trajectories):
            break
    return chunks
