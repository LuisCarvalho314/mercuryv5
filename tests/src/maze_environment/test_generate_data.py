from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import polars as pl

from maze_environment.generate_data import generate_data


class FakeAgent:
    def __init__(self) -> None:
        self.position = (0, 0)

    def make_observation(self, maze):
        return self.position


class FakeMazeEnvironment:
    def __init__(self, *, level, plotting=False, agent_sensors=None, seed=0):
        self.agent_sensors = agent_sensors or {"sensor": "cartesian"}
        self.maze = np.zeros((2, 3), dtype=np.uint8)
        self.agent_position = (0, 0)
        self.agent = FakeAgent()
        self.agent.position = self.agent_position
        self.rng = random.Random(seed)
        self._steps = iter(
            [
                ((0, 0), [1, 0, 0, 0], True),
                ((0, 1), [0, 1, 0, 0], False),
                ((0, 1), [0, 0, 1, 0], True),
                ((1, 1), [0, 0, 0, 1], False),
                ((1, 2), [1, 0, 0, 0], False),
            ]
        )

    def random_action(self) -> str:
        return "north"

    def random_policy(self, previous_action: str, collision: bool, rand_prob: float) -> str:
        return previous_action

    def step(self, action: str):
        observation, action_vec, collision = next(self._steps)
        sensor_name = self.agent_sensors["sensor"]
        if sensor_name == "cardinal distance":
            return (9, 8, 7, 6), action_vec, collision
        return observation, action_vec, collision


def _metadata_path(root: Path, *, level: int, sensor: str, run_id: str, sensor_range: int | None = None) -> Path:
    base = root / f"level={level}" / f"sensor={sensor}"
    if sensor == "cardinal distance":
        base = base / f"range={sensor_range}"
    return base / f"{run_id}.metadata.json"


def _parquet_path(root: Path, *, level: int, sensor: str, run_id: str, sensor_range: int | None = None) -> Path:
    base = root / f"level={level}" / f"sensor={sensor}"
    if sensor == "cardinal distance":
        base = base / f"range={sensor_range}"
    return base / f"{run_id}.parquet"


def test_generate_data_default_mode_preserves_collisions(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("maze_environment.generate_data.MazeEnvironment", FakeMazeEnvironment)

    run_ids = generate_data(level=0, seed=1, rand_prob=0.2, num_steps=3, output_root=tmp_path, reuse_existing=False)
    cart_run_id = run_ids["cartesian"]
    metadata = json.loads(_metadata_path(tmp_path, level=0, sensor="cartesian", run_id=cart_run_id).read_text(encoding="utf-8"))
    frame = pl.read_parquet(_parquet_path(tmp_path, level=0, sensor="cartesian", run_id=cart_run_id))

    assert metadata["valid_trajectories_only"] is False
    assert metadata["generation_stats"] == {
        "requested_steps": 3,
        "saved_steps": 3,
        "attempted_steps": 3,
        "dropped_collision_steps": 2,
    }
    assert frame.get_column("collision").to_list() == [True, False, True]


def test_generate_data_valid_only_saves_requested_number_of_clean_rows(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("maze_environment.generate_data.MazeEnvironment", FakeMazeEnvironment)

    run_ids = generate_data(
        level=0,
        seed=1,
        rand_prob=0.2,
        num_steps=3,
        valid_trajectories_only=True,
        output_root=tmp_path,
        reuse_existing=False,
    )
    cart_run_id = run_ids["cartesian"]
    metadata = json.loads(_metadata_path(tmp_path, level=0, sensor="cartesian", run_id=cart_run_id).read_text(encoding="utf-8"))
    frame = pl.read_parquet(_parquet_path(tmp_path, level=0, sensor="cartesian", run_id=cart_run_id))

    assert metadata["valid_trajectories_only"] is True
    assert metadata["generation_stats"] == {
        "requested_steps": 3,
        "saved_steps": 3,
        "attempted_steps": 5,
        "dropped_collision_steps": 2,
    }
    assert frame.height == 3
    assert frame.get_column("collision").to_list() == [False, False, False]


def test_generate_data_does_not_reuse_default_dataset_for_valid_only_requests(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("maze_environment.generate_data.MazeEnvironment", FakeMazeEnvironment)

    default_run_ids = generate_data(level=0, seed=1, rand_prob=0.2, num_steps=3, output_root=tmp_path, reuse_existing=True)
    strict_run_ids = generate_data(
        level=0,
        seed=1,
        rand_prob=0.2,
        num_steps=3,
        valid_trajectories_only=True,
        output_root=tmp_path,
        reuse_existing=True,
    )

    assert default_run_ids["cartesian"] != strict_run_ids["cartesian"]
    strict_metadata = json.loads(
        _metadata_path(tmp_path, level=0, sensor="cartesian", run_id=strict_run_ids["cartesian"]).read_text(encoding="utf-8")
    )
    assert strict_metadata["valid_trajectories_only"] is True


def test_generate_data_random_start_records_seed_controlled_initial_position(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("maze_environment.generate_data.MazeEnvironment", FakeMazeEnvironment)

    run_ids = generate_data(
        level=0,
        seed=5,
        rand_prob=0.2,
        num_steps=3,
        random_start=True,
        output_root=tmp_path,
        reuse_existing=False,
    )

    expected_positions = [(row, col) for row in range(2) for col in range(3)]
    expected_start = list(random.Random(5).choice(expected_positions))

    cart_metadata = json.loads(
        _metadata_path(tmp_path, level=0, sensor="cartesian", run_id=run_ids["cartesian"]).read_text(encoding="utf-8")
    )
    sensory_metadata = json.loads(
        _metadata_path(
            tmp_path,
            level=0,
            sensor="cardinal distance",
            sensor_range=1,
            run_id=run_ids["cardinal distance:range=1"],
        ).read_text(encoding="utf-8")
    )

    assert cart_metadata["random_start"] is True
    assert cart_metadata["initial_agent_position"] == expected_start
    assert cart_metadata["initial_agent_position"] != [0, 0]
    assert sensory_metadata["initial_agent_position"] == expected_start


def test_generate_data_does_not_reuse_fixed_start_dataset_for_random_start_requests(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("maze_environment.generate_data.MazeEnvironment", FakeMazeEnvironment)

    fixed_run_ids = generate_data(
        level=0,
        seed=1,
        rand_prob=0.2,
        num_steps=3,
        random_start=False,
        output_root=tmp_path,
        reuse_existing=True,
    )
    random_run_ids = generate_data(
        level=0,
        seed=1,
        rand_prob=0.2,
        num_steps=3,
        random_start=True,
        output_root=tmp_path,
        reuse_existing=True,
    )

    assert fixed_run_ids["cartesian"] != random_run_ids["cartesian"]
    fixed_metadata = json.loads(
        _metadata_path(tmp_path, level=0, sensor="cartesian", run_id=fixed_run_ids["cartesian"]).read_text(encoding="utf-8")
    )
    random_metadata = json.loads(
        _metadata_path(tmp_path, level=0, sensor="cartesian", run_id=random_run_ids["cartesian"]).read_text(encoding="utf-8")
    )
    assert fixed_metadata["random_start"] is False
    assert random_metadata["random_start"] is True
