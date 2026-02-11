# src/mercury_runs/schemas_results.py
from __future__ import annotations

from pydantic import BaseModel
from typing import Any, Dict, Optional


class SourceDatasetRef(BaseModel):
    level: int
    sensor: str
    sensor_range: Optional[int] = None

    # how the dataset was selected
    select: str = "latest"  # "latest" or "run_id"
    dataset_run_id: Optional[str] = None

    # useful for debugging/provenance
    dataset_parquet_name: Optional[str] = None


class ResultBundleMeta(BaseModel):
    run_id: str
    timestamp_utc: str
    source: SourceDatasetRef

    # Hyperparameters needed to reproduce
    sensory_params: Dict[str, Any]
    latent_params: Optional[Dict[str, Any]] = None
    action_map_params: Dict[str, Any]
    memory_length: Optional[int] = None

    # optional, free-form notes
    notes: Optional[str] = None
