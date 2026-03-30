from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field


class POCMLConfig(BaseModel):
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
    state_dim: int = Field(default=50, gt=0)
    random_feature_dim: int = Field(default=500, gt=0)
    n_states: Optional[int] = Field(default=None, gt=0)
    precision_capacities: Optional[list[int]] = None
    alpha: float = Field(default=2.0, gt=0)
    epochs: int = Field(default=30, gt=0)
    lr_q: float = Field(default=0.1, gt=0)
    lr_v: float = Field(default=0.08, gt=0)
    lr_all: float = Field(default=0.05, gt=0)
    lr_m: float = Field(default=1.0, gt=0)
    reg_m: float = Field(default=0.0, ge=0)
    max_iter_m: int = Field(default=1, gt=0)
    eps_m: float = Field(default=1e-3, gt=0)
    trajectory_length: int = Field(default=15, gt=1)
    max_trajectories: Optional[int] = Field(default=None, gt=0)
    use_ground_truth_node_ids: bool = False
    batch_size: int = Field(default=64, gt=0)
    memory_bias: bool = True
    paper_precision_enabled: bool = False
    paper_precision_mode: str = "off"
    paper_precision_eval_interval: int = Field(default=1, gt=0)
    paper_precision_num_points: Optional[int] = Field(default=None, gt=0)
    paper_precision_num_walks: int = Field(default=100, gt=0)
    paper_precision_walk_length: int = Field(default=10_000, gt=0)
    paper_precision_seed: int = Field(default=0, ge=0)
    paper_precision_rand_prob: float = Field(default=0.3, ge=0.0, le=1.0)
    paper_precision_n_obs: int = Field(default=1, gt=0)
    paper_precision_n_actions: int = Field(default=1, gt=0)
    paper_precision_sensory_params: dict[str, Any] = Field(default_factory=dict)
    paper_precision_action_map_params: dict[str, Any] = Field(default_factory=dict)
    structure_metrics_enabled: bool = True
    structure_metrics_epsilon: float = Field(default=1e-6, ge=0.0)
    structure_metrics_ignore_self_loops: bool = True
    structure_metrics_scope: str = "exact_level"
    structure_metrics_ground_truth_source: str = "empirical_walks"
