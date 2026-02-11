# generate_data.py

import datetime
import json
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from .maze_environment_v3 import MazeEnvironment
from .LEVEL import levels
from .schemas import AgentConfig, MazeTrajectory, MazeDataset

import json
from pathlib import Path
from typing import Optional


def _find_existing_dataset_run_id(
    output_dir: Path,
    *,
    level: int,
    agent_sensors: dict,
    seed: int,
    rand_prob: float,
    num_steps: int,
) -> Optional[str]:
    """
    Returns an existing run_id if a dataset with identical params exists in output_dir.
    Assumes metadata json schema contains:
      run_id, maze_id, agent_config{sensor, range}, random_seed, random_prob, num_steps
    """
    if not output_dir.exists():
        return None

    metadata_files = sorted(output_dir.glob("*.metadata.json"))
    if not metadata_files:
        return None

    desired_sensor = agent_sensors["sensor"]
    desired_range = agent_sensors.get("range", None)

    for metadata_path in reversed(metadata_files):  # newest first
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except Exception:
            continue

        agent_config = metadata.get("agent_config", {}) or {}
        if (
            metadata.get("maze_id") == level
            and agent_config.get("sensor") == desired_sensor
            and agent_config.get("range", None) == desired_range
            and metadata.get("random_seed") == seed
            and float(metadata.get("random_prob")) == float(rand_prob)
            and int(metadata.get("num_steps")) == int(num_steps)
        ):
            existing_run_id = metadata.get("run_id")
            if existing_run_id:
                parquet_path = output_dir / f"{existing_run_id}.parquet"
                if parquet_path.exists():
                    return existing_run_id

    return None

def _utc_now() -> tuple[str, str]:
    """
    Returns:
      - timestamp_utc_iso: ISO-8601 UTC (e.g., "2026-02-02T14:03:11Z")
      - timestamp_utc_compact: filesystem-friendly (e.g., "20260202T140311Z")
    """
    now = datetime.datetime.utcnow().replace(microsecond=0)
    timestamp_utc_iso = now.isoformat() + "Z"
    timestamp_utc_compact = now.strftime("%Y%m%dT%H%M%SZ")
    return timestamp_utc_iso, timestamp_utc_compact


def _to_2d_array(values) -> np.ndarray:
    array = np.asarray(values)
    if array.dtype == object:
        array = np.vstack(array)
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    return array


def _build_output_directory(output_root: Path, level: int, agent_sensors: dict) -> Path:
    sensor_name = agent_sensors["sensor"]
    if sensor_name == "cardinal distance":
        sensor_range = agent_sensors.get("range", None)
        return output_root / f"level={level}" / f"sensor={sensor_name}" / f"range={sensor_range}"
    return output_root / f"level={level}" / f"sensor={sensor_name}"


def _build_run_id(
    level: int,
    agent_sensors: dict,
    seed: int,
    rand_prob: float,
    num_steps: int,
    timestamp_compact: str,
) -> str:
    sensor_name = agent_sensors["sensor"]
    sensor_range = agent_sensors.get("range", None)

    if sensor_name == "cardinal distance":
        return f"level{level}_{sensor_name}_range{sensor_range}_seed{seed}_p{rand_prob}_steps{num_steps}_{timestamp_compact}"

    return f"level{level}_{sensor_name}_seed{seed}_p{rand_prob}_steps{num_steps}_{timestamp_compact}"


def _infer_action_column_names(action_width: int) -> list[str]:
    default = ["action_north", "action_east", "action_south", "action_west"]
    if action_width <= len(default):
        return default[:action_width]
    return [f"action_{i}" for i in range(action_width)]


def _dataset_to_dict(dataset: MazeDataset) -> dict[str, Any]:
    # Pydantic v2: model_dump; v1: dict
    if hasattr(dataset, "model_dump"):
        return dataset.model_dump()
    return dataset.dict()


def _rollout(env: MazeEnvironment, num_steps: int, rand_prob: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    observations = []
    actions = []
    collisions = []

    action = env.random_action()
    collision = True

    for _ in range(num_steps):
        action = env.random_policy(action, collision, rand_prob)
        observation, action_vec, collision = env.step(action)

        observations.append(observation)
        actions.append(action_vec)
        collisions.append(bool(collision))

    observations_array = _to_2d_array(observations)
    actions_array = _to_2d_array(actions).astype(np.int8, copy=False)
    collisions_array = np.asarray(collisions, dtype=bool)

    return observations_array, actions_array, collisions_array


def _build_frame(observations: np.ndarray, actions: np.ndarray, collisions: np.ndarray) -> pl.DataFrame:
    num_steps = int(collisions.shape[0])
    iteration = np.arange(num_steps, dtype=np.int32)

    observation_column_names = [f"observation_{i}" for i in range(observations.shape[1])]
    action_column_names = _infer_action_column_names(actions.shape[1])

    columns: dict[str, Any] = {"iteration": iteration, "collision": collisions}

    for i, name in enumerate(observation_column_names):
        columns[name] = observations[:, i]

    for i, name in enumerate(action_column_names):
        columns[name] = actions[:, i]

    return pl.DataFrame(columns)


def _json_safe(value):
    import numpy as np

    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.ndarray,)):
        # never serialize full arrays into JSON metadata
        return {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }
    return value


def _write_artifacts(output_dir, run_id, frame, dataset, compression="zstd"):
    import json

    output_dir.mkdir(parents=True, exist_ok=True)

    parquet_path = output_dir / f"{run_id}.parquet"
    metadata_path = output_dir / f"{run_id}.metadata.json"

    # Write Parquet (data)
    frame.write_parquet(
        parquet_path,
        compression=compression,
        statistics=True,
        row_group_size=100_000,
    )

    # Build JSON metadata WITHOUT arrays
    metadata = {
        "run_id": dataset.run_id,
        "timestamp_utc": dataset.timestamp_utc,
        "maze_id": dataset.maze_id,
        "agent_config": dataset.agent_config.model_dump() if hasattr(dataset.agent_config, "model_dump") else dataset.agent_config.dict(),
        "random_seed": dataset.random_seed,
        "random_prob": dataset.random_prob,
        "num_steps": dataset.num_steps,
        "artifact": {"parquet": parquet_path.name},
        "table": {
            "num_rows": frame.height,
            "columns": frame.columns,
            "dtypes": {name: str(dtype) for name, dtype in zip(frame.columns, frame.dtypes)},
        },
    }

    # Optionally record trajectory array shapes/dtypes (no data)
    if dataset.trajectories:
        traj = dataset.trajectories[0]
        metadata["trajectory"] = {
            "observations": _json_safe(traj.observations),
            "actions": _json_safe(traj.actions),
            "collisions": _json_safe(traj.collisions),
        }

    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    print(f"Saved: {parquet_path}")
    print(f"Saved: {metadata_path}")



def generate_data(
    level: int,
    seed: int = 0,
    rand_prob: float = 0.3,
    num_steps: int = 20_000,
    output_root: str | Path = "../../datasets",
    reuse_existing: bool = True,
) -> dict[str, str]:
    """
    Returns a mapping {sensor_key: run_id} for each generated/reused dataset.
    sensor_key examples:
      - "cartesian"
      - "cardinal distance:range=1"
    """
    agent_sensors_list = [
        {"sensor": "cardinal distance", "range": 1},
        {"sensor": "cartesian"},
    ]

    output_root_path = Path(output_root)
    run_ids: dict[str, str] = {}

    for agent_sensors in agent_sensors_list:
        timestamp_utc_iso, timestamp_utc_compact = _utc_now()
        output_dir = _build_output_directory(output_root_path, level, agent_sensors)

        sensor_key = agent_sensors["sensor"]
        if agent_sensors["sensor"] == "cardinal distance":
            sensor_key = f"{sensor_key}:range={agent_sensors.get('range', None)}"

        if reuse_existing:
            existing_run_id = _find_existing_dataset_run_id(
                output_dir,
                level=level,
                agent_sensors=agent_sensors,
                seed=seed,
                rand_prob=rand_prob,
                num_steps=num_steps,
            )
            if existing_run_id is not None:
                run_ids[sensor_key] = existing_run_id
                print(f"Reusing existing dataset: {output_dir / (existing_run_id + '.parquet')}")
                continue

        run_id = _build_run_id(
            level=level,
            agent_sensors=agent_sensors,
            seed=seed,
            rand_prob=rand_prob,
            num_steps=num_steps,
            timestamp_compact=timestamp_utc_compact,
        )

        env = MazeEnvironment(
            level=levels[level],
            plotting=False,
            agent_sensors=agent_sensors,
            seed=seed,
        )

        observations, actions, collisions = _rollout(env, num_steps=num_steps, rand_prob=rand_prob)

        trajectory = MazeTrajectory(observations=observations, actions=actions, collisions=collisions)
        agent_config = AgentConfig(sensor=agent_sensors["sensor"], range=agent_sensors.get("range", None))

        dataset = MazeDataset(
            run_id=run_id,
            timestamp_utc=timestamp_utc_iso,
            maze_id=level,
            agent_config=agent_config,
            random_seed=seed,
            random_prob=rand_prob,
            num_steps=num_steps,
            trajectories=[trajectory],
        )

        frame = _build_frame(observations, actions, collisions)
        _write_artifacts(output_dir=output_dir, run_id=run_id, frame=frame, dataset=dataset)

        run_ids[sensor_key] = run_id

    return run_ids
