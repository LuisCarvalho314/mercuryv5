from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field


class MercuryCSCGHybridConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    datasets_root: Path
    output_root: Path
    run_id: str = Field(min_length=1)
    level: int = Field(ge=0)
    sensor: str = Field(min_length=1)
    sensor_range: Optional[int] = Field(default=None, ge=0)
    ground_truth_states_parquet: Path
    memory_length: int = Field(gt=0)
    window_length: int = Field(gt=0)
    seed: int = Field(ge=0)
    rand_prob: float = Field(ge=0.0, le=1.0)
    num_steps: int = Field(gt=0)
    valid_trajectories_only: bool = False
    mercury_valid_trajectories_only: bool = False
    mercury_split_sensory_raw_latent_valid: bool = False
    reuse_existing_dataset: bool = True
    reuse_existing_run: bool = True
    mercury_input_mode: str = "pretrained_raw"
    sensory: dict[str, Any]
    latent: dict[str, Any]
    action_map: dict[str, Any]
    dataset_select: str = "latest"
    dataset_run_id: Optional[str] = None
    n_iter: int = Field(default=1000, gt=0)
    pseudocount: float = Field(default=2e-3, ge=0.0)
    term_early: bool = True
    train_algorithm: str = Field(default="em")
    training_mode: str = Field(default="full")
    batch_size: int = Field(default=1024, gt=0)
    online_lambda: float = Field(default=1.0, gt=0.0, le=1.0)
    mercury_prior_weight: float = Field(default=1.0, ge=0.0)
    method_name: str = "mercury_cscg_hybrid"
    source_method: str = "mercury"
    structural_transfer_level: str = "clone_state"
    structural_init_mode: str = "binary"
    structural_source: str = "source_graph"
    clone_capacity_mode: str = "source"
    eval_viterbi: bool = False
    paper_precision_enabled: bool = False
    paper_precision_mode: str = "off"
    paper_precision_eval_interval: int = Field(default=1, gt=0)
    paper_precision_eval_checkpoints: Optional[list[int]] = None
    paper_precision_num_points: Optional[int] = Field(default=None, gt=0)
    paper_precision_schedule: str = "log"
    paper_precision_num_walks: int = Field(default=100, gt=0)
    paper_precision_walk_length: int = Field(default=10_000, gt=0)
    paper_precision_seed: int = Field(default=0, ge=0)
    paper_precision_rand_prob: float = Field(default=0.3, ge=0.0, le=1.0)
    paper_precision_sensory_params: dict[str, Any] = Field(default_factory=dict)
    paper_precision_action_map_params: dict[str, Any] = Field(default_factory=dict)
    structure_metrics_enabled: bool = True
    structure_metrics_epsilon: float = Field(default=1e-6, ge=0.0)
    structure_metrics_epsilons: Optional[list[float]] = None
    structure_metrics_ignore_self_loops: bool = True
    structure_metrics_scope: str = "exact_level"
    structure_metrics_ground_truth_source: str = "empirical_walks"
    random_start: bool = True
    embed_metadata_in_parquet: bool = True
    notes: Optional[str] = None
