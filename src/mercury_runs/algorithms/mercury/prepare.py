from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ...io_parquet import LoadedDataset


def generate_mercury_datasets(*, config: Any) -> Any:
    from filelock import FileLock
    from src.maze_environment.generate_data import generate_data

    datasets_root = Path(config.datasets_root)
    lock_path = datasets_root / ".dataset_generation.lock"
    with FileLock(str(lock_path)):
        return generate_data(
            level=config.level,
            seed=config.seed,
            rand_prob=config.rand_prob,
            num_steps=config.num_steps,
            valid_trajectories_only=bool(config.valid_trajectories_only),
            output_root=datasets_root,
            reuse_existing=bool(config.reuse_existing_dataset),
        )


def filter_mercury_dataset(
    *,
    dataset: LoadedDataset,
    valid_trajectories_only: bool,
) -> LoadedDataset:
    if not bool(valid_trajectories_only):
        return dataset
    collisions = np.asarray(dataset.collisions, dtype=bool)
    keep_mask = ~collisions
    return LoadedDataset(
        observations=np.asarray(dataset.observations)[keep_mask],
        actions=np.asarray(dataset.actions)[keep_mask],
        collisions=np.asarray(dataset.collisions)[keep_mask],
        source_metadata=dict(dataset.source_metadata),
        parquet_path=dataset.parquet_path,
    )
