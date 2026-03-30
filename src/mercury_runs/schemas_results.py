# src/mercury_runs/schemas_results.py
from __future__ import annotations

from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class SourceDatasetRef(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    level: int = Field(ge=0)
    sensor: str = Field(min_length=1)
    sensor_range: Optional[int] = Field(default=None, ge=0)

    # how the dataset was selected
    select: Literal["latest", "run_id"] = "latest"
    dataset_run_id: Optional[str] = None

    # useful for debugging/provenance
    dataset_parquet_name: Optional[str] = None


class ResultBundleMeta(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    run_id: str
    timestamp_utc: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
    source: SourceDatasetRef

    # Hyperparameters needed to reproduce
    sensory_params: Dict[str, Any]
    latent_params: Optional[Dict[str, Any]] = None
    action_map_params: Dict[str, Any]
    memory_length: Optional[int] = None
    run_parameters: Dict[str, Any] = Field(default_factory=dict)
    source_dataset_metadata: Dict[str, Any] = Field(default_factory=dict)
    ground_truth_dataset_metadata: Dict[str, Any] = Field(default_factory=dict)

    # optional, free-form notes
    notes: Optional[str] = None
