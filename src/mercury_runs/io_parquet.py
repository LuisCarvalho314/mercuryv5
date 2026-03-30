# src/mercury_runs/io_parquet.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal, Optional

import numpy as np
import polars as pl
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class ParquetConfig(BaseModel):
    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    root: Path
    level: int = Field(ge=0)
    sensor: str = Field(min_length=1)
    sensor_range: Optional[int] = Field(default=None, ge=0)

    select: Literal["latest", "run_id"] = "latest"
    run_id: Optional[str] = None

    @model_validator(mode="after")
    def _validate_selection_and_sensor(self) -> "ParquetConfig":
        if self.select == "run_id" and not self.run_id:
            raise ValueError("select='run_id' requires run_id.")
        if self.sensor == "cardinal distance" and self.sensor_range is None:
            raise ValueError("sensor='cardinal distance' requires sensor_range.")
        return self


class LoadedDataset(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    observations: np.ndarray  # (T, D_obs)
    actions: np.ndarray       # (T, D_act)
    collisions: np.ndarray    # (T,)
    source_metadata: dict[str, Any]
    parquet_path: Path

    @field_validator("observations", "actions", "collisions", mode="before")
    @classmethod
    def _coerce_array(cls, value: Any) -> np.ndarray:
        if isinstance(value, np.ndarray):
            return value
        return np.asarray(value)

    @model_validator(mode="after")
    def _validate_shapes(self) -> "LoadedDataset":
        if self.observations.ndim != 2:
            raise ValueError("observations must be a 2D array.")
        if self.actions.ndim != 2:
            raise ValueError("actions must be a 2D array.")
        if self.collisions.ndim != 1:
            raise ValueError("collisions must be a 1D array.")

        n_rows = int(self.observations.shape[0])
        if int(self.actions.shape[0]) != n_rows or int(self.collisions.shape[0]) != n_rows:
            raise ValueError("observations, actions, and collisions must have matching row counts.")
        return self


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
    if not isinstance(source_metadata, dict):
        raise ValueError(f"Expected metadata JSON object in {metadata_path.name}, got {type(source_metadata).__name__}.")

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
