from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field


class CSCGConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    datasets_root: Path
    output_root: Path
    run_id: str = Field(min_length=1)
    level: int = Field(ge=0)
    sensor: str = Field(min_length=1)
    sensor_range: Optional[int] = Field(default=None, ge=0)
    ground_truth_states_parquet: Path
    window_length: int = Field(default=100, gt=0)
    valid_trajectories_only: bool = False
    dataset_select: str = "latest"
    dataset_run_id: Optional[str] = None
    clones_per_obs: int = Field(default=10, gt=0)
    n_iter: int = Field(default=1000, gt=0)
    pseudocount: float = Field(default=2e-3, ge=0.0)
    term_early: bool = True
    train_algorithm: str = Field(default="em")
    training_mode: str = Field(default="full")
    batch_size: int = Field(default=1024, gt=0)
    online_lambda: float = Field(default=1.0, gt=0.0, le=1.0)
    seed: int = Field(default=42, ge=0)
    paper_precision_enabled: bool = False
    paper_precision_mode: str = "off"
    paper_precision_eval_interval: int = Field(default=1, gt=0)
    paper_precision_num_points: Optional[int] = Field(default=None, gt=0)
    paper_precision_num_walks: int = Field(default=100, gt=0)
    paper_precision_walk_length: int = Field(default=10_000, gt=0)
    paper_precision_seed: int = Field(default=0, ge=0)
    paper_precision_rand_prob: float = Field(default=0.3, ge=0.0, le=1.0)
    paper_precision_sensory_params: dict[str, Any] = Field(default_factory=dict)
    paper_precision_action_map_params: dict[str, Any] = Field(default_factory=dict)
    structure_metrics_enabled: bool = True
    structure_metrics_epsilon: float = Field(default=1e-6, ge=0.0)
    structure_metrics_ignore_self_loops: bool = True
    structure_metrics_scope: str = "exact_level"
    structure_metrics_ground_truth_source: str = "empirical_walks"
