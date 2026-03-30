from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


PIPELINE_SCHEMA_VERSION = 2


class ArtifactLayout(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mercury_states_parquet: Path
    mercury_paper_precision_json: Path
    mercury_computational_json: Path
    mercury_latent_graph_npz: Path
    mercury_internal_graph_png: Path
    pocml_states_parquet: Path
    pocml_paper_precision_json: Path
    pocml_computational_json: Path
    pocml_embeddings_npz: Path
    pocml_model_checkpoint_pt: Path
    pocml_internal_graph_png: Path
    cscg_states_parquet: Path
    cscg_paper_precision_json: Path
    cscg_computational_json: Path
    cscg_model_npz: Path
    cscg_internal_graph_png: Path
    internal_graphs_comparison_png: Path
    comparison_summary_json: Path
    run_config_json: Path
    run_status_json: Path


class RunIdentity(BaseModel):
    model_config = ConfigDict(extra="forbid")

    requested_run_id: str
    run_id: str
    run_root: Path
    is_reused: bool


class SubsystemStatus(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    status: str
    artifacts: Dict[str, Optional[str]] = Field(default_factory=dict)
    native_metrics: Dict[str, Any] = Field(default_factory=dict)
    config: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None


class RunStatusModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int = PIPELINE_SCHEMA_VERSION
    run_id: str
    requested_run_id: str
    status: str
    started_at_utc: str
    finished_at_utc: Optional[str] = None
    study: Dict[str, Any]
    config: Dict[str, Any]
    execution: Dict[str, Any]
    paths: Dict[str, Any]
    artifacts: Dict[str, Optional[str]] = Field(default_factory=dict)
    subsystems: Dict[str, SubsystemStatus] = Field(default_factory=dict)
    error: Optional[str] = None


class RunConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int = PIPELINE_SCHEMA_VERSION
    run_id: str
    requested_run_id: str
    saved_at_utc: str
    cli_args: Dict[str, Any]
    run_payload: Dict[str, Any]
    execution: Dict[str, Any]
    paths: Dict[str, Any]


@dataclass(frozen=True)
class StudyDefaults:
    study_root: str = "results/current/studies"
    study_name: str = "level16_cd_grid"
    python_executable: str = "python3"
    main_path: str = "main.py"


class StudyMethodConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    grid: Dict[str, Any] = Field(default_factory=dict)


class StudyMethodsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mercury: StudyMethodConfig = Field(default_factory=lambda: StudyMethodConfig(enabled=True))
    pocml: StudyMethodConfig = Field(default_factory=StudyMethodConfig)
    cscg: StudyMethodConfig = Field(default_factory=StudyMethodConfig)


class HierarchicalStudyConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    environment: Dict[str, Any] = Field(default_factory=dict)
    evaluation: Dict[str, Any] = Field(default_factory=dict)
    methods: StudyMethodsConfig


class StudyGridConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    method: Literal["mercury", "pocml", "cscg"] = "mercury"
    level: int = Field(ge=0)
    sensor: str = Field(min_length=1)
    sensor_range: Optional[int] = Field(default=None, ge=0)
    seed: int = Field(ge=0)
    rand_prob: float = Field(ge=0.0, le=1.0)
    memory_length: int = Field(gt=0)
    mercury_valid_trajectories_only: bool = False
    mercury_split_sensory_raw_latent_valid: bool = False
    pocml_valid_trajectories_only: bool = False
    cscg_valid_trajectories_only: bool = False
    activation_threshold: float = Field(ge=0.0, le=1.0)
    ambiguity_threshold: int = Field(ge=0)
    ambiguity_decay: float = Field(default=0.99, ge=0.0, le=1.0)
    am_lr: float = Field(ge=0.0)
    am_key: int = Field(ge=0)
    window_length: int = Field(gt=0)
    weight_memory: float = Field(ge=0.0)
    weight_undirected: float = Field(ge=0.0)
    weight_base: float = Field(ge=0.0)
    weight_action: float = Field(ge=0.0)
    memory_replay: bool
    memory_disambiguation: bool
    study_root: str = Field(min_length=1)
    study_name: str = Field(min_length=1)
    reuse_existing_run: bool = False
