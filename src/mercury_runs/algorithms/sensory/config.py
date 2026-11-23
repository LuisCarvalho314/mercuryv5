from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


class SensoryConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    datasets_root: Path
    output_root: Path
    run_id: str = Field(min_length=1)
    level: int = Field(ge=0)
    sensor: str = Field(min_length=1)
    sensor_range: Optional[int] = Field(default=None, ge=0)
    seed: int = Field(ge=0)
    rand_prob: float = Field(ge=0.0, le=1.0)
    num_steps: int = Field(gt=0)
    window_length: int = Field(gt=0)
    valid_trajectories_only: bool = False
    observation_mode: str = "raw"
    dataset_select: str = "latest"
    dataset_run_id: Optional[str] = None
    random_start: bool = True
    reuse_existing_dataset: bool = True
    sensory: dict[str, Any]
    action_map: dict[str, Any]
    paper_precision_enabled: bool = False
    paper_precision_mode: str = "off"
    paper_precision_eval_interval: int = Field(default=1, gt=0)
    paper_precision_eval_checkpoints: Optional[list[int]] = None
    paper_precision_num_points: Optional[int] = Field(default=None, gt=0)
    paper_precision_num_walks: int = Field(default=100, gt=0)
    paper_precision_schedule: str = "log"
    paper_precision_walk_length: int = Field(default=10_000, gt=0)
    structure_metrics_enabled: bool = True
    structure_metrics_epsilon: float = Field(default=1e-6, ge=0.0)
    structure_metrics_epsilons: Optional[list[float]] = None
    structure_metrics_ignore_self_loops: bool = True
    structure_metrics_scope: str = "exact_level"
    structure_metrics_ground_truth_source: str = "empirical_walks"
    run_parameters: dict[str, Any] = Field(default_factory=dict)
    embed_metadata_in_parquet: bool = True
    notes: Optional[str] = None

    @model_validator(mode="after")
    def _validate_observation_mode_sensor(self) -> "SensoryConfig":
        if self.sensor == "cardinal distance" and self.sensor_range is None:
            raise ValueError("sensor='cardinal distance' requires sensor_range.")
        if str(self.observation_mode).strip().lower() == "cscg_one_hot":
            if self.sensor != "cardinal distance":
                raise ValueError("observation_mode='cscg_one_hot' requires sensor='cardinal distance'.")
            if self.sensor_range is None:
                raise ValueError("observation_mode='cscg_one_hot' requires sensor_range.")
        return self
