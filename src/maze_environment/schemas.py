from pydantic import BaseModel
from typing import List, Optional
import numpy as np


class MazeTrajectory(BaseModel):
    observations: np.ndarray
    actions: np.ndarray
    collisions: np.ndarray

    class Config:
        arbitrary_types_allowed = True


class AgentConfig(BaseModel):
    sensor: str
    range: Optional[int] = None


class MazeDataset(BaseModel):
    # Minimal reproducibility + traceability
    run_id: str               # unique id for this saved artifact/run
    timestamp_utc: str        # ISO-8601 UTC, e.g. "2026-02-02T14:03:11Z"

    # What was run
    maze_id: int
    agent_config: AgentConfig

    # How it was generated
    random_seed: int
    random_prob: float
    num_steps: int            # number of environment steps collected

    # Data
    trajectories: List[MazeTrajectory]
