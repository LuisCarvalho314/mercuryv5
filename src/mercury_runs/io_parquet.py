# src/mercury_runs/io_parquet.py
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import polars as pl


@dataclass(frozen=True)
class ParquetConfig:
    root: Path
    level: int
    sensor: str
    sensor_range: Optional[int] = None

    select: str = "latest"  # "latest" or "run_id"
    run_id: Optional[str] = None


@dataclass(frozen=True)
class LoadedDataset:
    observations: np.ndarray  # (T, D_obs)
    actions: np.ndarray       # (T, D_act)
    collisions: np.ndarray    # (T,)
    source_metadata: dict
    parquet_path: Path


def dataset_directory(config: ParquetConfig) -> Path:
    if config.sensor == "cardinal distance":
        return config.root / f"level={config.level}" / f"sensor={config.sensor}" / f"range={config.sensor_range}"
    return config.root / f"level={config.level}" / f"sensor={config.sensor}"


def select_parquet_path(directory: Path, select: str, run_id: Optional[str]) -> Path:
    if not directory.exists():
        raise FileNotFoundError(f"Dataset directory not found: {directory}")

    if select == "run_id":
        if not run_id:
            raise ValueError("select='run_id' requires run_id.")
        parquet_path = directory / f"{run_id}.parquet"
        if not parquet_path.exists():
            raise FileNotFoundError(f"Parquet file not found: {parquet_path}")
        return parquet_path

    parquet_candidates = sorted(directory.glob("*.parquet"))
    if not parquet_candidates:
        raise FileNotFoundError(f"No parquet files found in: {directory}")
    return parquet_candidates[-1]


def load_level_parquet(config: ParquetConfig) -> LoadedDataset:
    directory = dataset_directory(config)
    parquet_path = select_parquet_path(directory, config.select, config.run_id)

    metadata_path = parquet_path.with_suffix(".metadata.json")
    source_metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}

    frame = pl.read_parquet(parquet_path)

    observation_columns = [c for c in frame.columns if c.startswith("observation_")]
    action_columns = [c for c in frame.columns if c.startswith("action_")]

    if not observation_columns:
        raise ValueError(f"No observation_ columns found in {parquet_path.name}")
    if not action_columns:
        raise ValueError(f"No action_ columns found in {parquet_path.name}")
    if "collision" not in frame.columns:
        raise ValueError(f"Missing required column 'collision' in {parquet_path.name}")

    observations = frame.select(observation_columns).to_numpy()
    actions = frame.select(action_columns).to_numpy()
    collisions = frame.get_column("collision").to_numpy()

    return LoadedDataset(
        observations=observations,
        actions=actions,
        collisions=collisions,
        source_metadata=source_metadata,
        parquet_path=parquet_path,
    )
