from __future__ import annotations

from pathlib import Path

import numpy as np

from mercury.latent.params import LatentParams
from mercury.sensory.params import SensoryParams

from mercury_runs.io_parquet import LoadedDataset
from mercury_runs.pipelines import _build_action_map, run_all_bundled, run_latent


class _FakeGraph:
    def to_npz(self, path: str) -> None:
        Path(path).write_bytes(b"graph")


def _dataset(path: Path) -> LoadedDataset:
    return LoadedDataset(
        observations=np.asarray([[0.0], [1.0]], dtype=np.float32),
        actions=np.asarray([[0.0], [1.0]], dtype=np.float32),
        collisions=np.asarray([False, True]),
        source_metadata={"source": "test"},
        parquet_path=path,
    )


def test_run_all_bundled_accepts_mercury_valid_trajectory_flag(monkeypatch, tmp_path: Path) -> None:
    latent_dataset = _dataset(tmp_path / "latent.parquet")
    cartesian_dataset = _dataset(tmp_path / "cartesian.parquet")
    captured: dict = {"filters": 0}

    def _fake_load_level_parquet(config):
        if config.sensor == "cartesian":
            return cartesian_dataset
        return latent_dataset

    def _fake_filter(dataset, valid_trajectories_only: bool):
        captured["filters"] += 1
        captured["valid_trajectories_only"] = valid_trajectories_only
        return dataset

    def _fake_ground_truth(*args, **kwargs):
        return np.asarray([0, 1], dtype=np.int32)

    def _fake_run_latent(*args, **kwargs):
        return (
            np.asarray([0, 1], dtype=np.int32),
            np.asarray([0, 1], dtype=np.int32),
            np.asarray([1, 2], dtype=np.int32),
            _FakeGraph(),
        )

    def _fake_write_bundle_parquet(**kwargs):
        captured["bundle_columns"] = kwargs["columns"]
        return tmp_path / "states.parquet"

    monkeypatch.setattr("mercury_runs.pipelines.load_level_parquet", _fake_load_level_parquet)
    monkeypatch.setattr("mercury_runs.pipelines.filter_mercury_dataset", _fake_filter)
    monkeypatch.setattr("mercury_runs.pipelines.run_ground_truth_cartesian", _fake_ground_truth)
    monkeypatch.setattr("mercury_runs.pipelines.run_latent", _fake_run_latent)
    monkeypatch.setattr("mercury_runs.pipelines.write_bundle_parquet", _fake_write_bundle_parquet)

    result = run_all_bundled(
        datasets_root=tmp_path / "datasets",
        results_root=tmp_path / "results",
        level=21,
        sensor="cardinal distance",
        sensor_range=1,
        mercury_valid_trajectories_only=True,
        memory_length=10,
        sensory_params=SensoryParams(),
        latent_params=LatentParams(),
        action_map_params={"n_codebook": 4, "lr": 0.5, "sigma": 0.0, "key": 0},
        run_parameters={},
        run_id="run-1",
    )

    assert result == tmp_path / "states.parquet"
    assert captured["filters"] == 2
    assert captured["valid_trajectories_only"] is True
    assert list(captured["bundle_columns"]) == [
        "cartesian_proxy_bmu",
        "sensory_bmu",
        "latent_bmu",
        "latent_node_count",
    ]


def test_run_all_bundled_can_split_raw_sensory_from_valid_latent(monkeypatch, tmp_path: Path) -> None:
    latent_dataset = LoadedDataset(
        observations=np.asarray([[10.0], [11.0], [12.0]], dtype=np.float32),
        actions=np.asarray([[0.0], [1.0], [2.0]], dtype=np.float32),
        collisions=np.asarray([False, True, False]),
        source_metadata={"source": "test"},
        parquet_path=tmp_path / "latent.parquet",
    )
    cartesian_dataset = LoadedDataset(
        observations=np.asarray([[20.0], [21.0], [22.0]], dtype=np.float32),
        actions=np.asarray([[0.0], [1.0], [2.0]], dtype=np.float32),
        collisions=np.asarray([False, True, False]),
        source_metadata={"source": "test"},
        parquet_path=tmp_path / "cartesian.parquet",
    )
    captured: dict = {}

    def _fake_load_level_parquet(config):
        if config.sensor == "cartesian":
            return cartesian_dataset
        return latent_dataset

    def _fake_filter(dataset, valid_trajectories_only: bool):
        raise AssertionError("split mode should not pre-filter datasets")

    def _fake_ground_truth(dataset, *args, **kwargs):
        captured["ground_truth_rows"] = int(dataset.observations.shape[0])
        return np.asarray([100, 101, 102], dtype=np.int32)

    def _fake_run_latent(dataset_latent, *args, dataset_sensory=None, **kwargs):
        captured["latent_rows"] = int(dataset_latent.observations.shape[0])
        captured["sensory_rows"] = int(dataset_sensory.observations.shape[0]) if dataset_sensory is not None else None
        captured["latent_valid_only"] = bool(kwargs.get("latent_valid_trajectories_only", False))
        return (
            np.asarray([200, 201, 202], dtype=np.int32),
            np.asarray([300, 300, 302], dtype=np.int32),
            np.asarray([1, 1, 2], dtype=np.int32),
            _FakeGraph(),
        )

    def _fake_write_bundle_parquet(**kwargs):
        captured["bundle_rows"] = {key: value.tolist() for key, value in kwargs["columns"].items()}
        return tmp_path / "states.parquet"

    monkeypatch.setattr("mercury_runs.pipelines.load_level_parquet", _fake_load_level_parquet)
    monkeypatch.setattr("mercury_runs.pipelines.filter_mercury_dataset", _fake_filter)
    monkeypatch.setattr("mercury_runs.pipelines.run_ground_truth_cartesian", _fake_ground_truth)
    monkeypatch.setattr("mercury_runs.pipelines.run_latent", _fake_run_latent)
    monkeypatch.setattr("mercury_runs.pipelines.write_bundle_parquet", _fake_write_bundle_parquet)

    result = run_all_bundled(
        datasets_root=tmp_path / "datasets",
        results_root=tmp_path / "results",
        level=21,
        sensor="cardinal distance",
        sensor_range=1,
        mercury_split_sensory_raw_latent_valid=True,
        memory_length=10,
        sensory_params=SensoryParams(),
        latent_params=LatentParams(),
        action_map_params={"n_codebook": 4, "lr": 0.5, "sigma": 0.0, "key": 0},
        run_parameters={},
        run_id="run-1",
    )

    assert result == tmp_path / "states.parquet"
    assert captured["ground_truth_rows"] == 3
    assert captured["sensory_rows"] == 3
    assert captured["latent_rows"] == 3
    assert captured["latent_valid_only"] is True
    assert captured["bundle_rows"] == {
        "cartesian_proxy_bmu": [100, 101, 102],
        "sensory_bmu": [200, 201, 202],
        "latent_bmu": [300, 300, 302],
        "latent_node_count": [1, 1, 2],
    }


def test_run_latent_returns_frozen_sensory_bmus(monkeypatch) -> None:
    class _SensoryGraph:
        def __init__(self) -> None:
            self.n = 2
            self.node_features = {"activation": np.array([1.0, 0.0], dtype=np.float32)}

    class _SensoryState:
        def __init__(self) -> None:
            self.gs = _SensoryGraph()
            self.prev_bmu = None

    class _ActionMap:
        def __init__(self) -> None:
            self.step_calls = 0
            self.predict_calls = 0

        def step(self, action):
            self.step_calls += 1
            return 0, action

        def predict(self, *, action=None, actions=None):
            self.predict_calls += 1
            return 0

    class _Mem:
        def __init__(self, n: int, length: int) -> None:
            self.gs = type("MG", (), {"n": n * length})()
            self.length = length
            self.sensory_n_nodes = n

    class _LatentState:
        def __init__(self) -> None:
            self.g = type("LG", (), {"n": 2})()
            self.prev_bmu = 0

    training_iter = iter([2, 0, 2])
    frozen_iter = iter([0, 1, 0])

    def _fake_sensory_step(observation, action_bmu, sensory_state, sensory_params, action_map):
        sensory_state.prev_bmu = next(training_iter)
        return sensory_state

    def _fake_sensory_step_frozen(observation, action_bmu, sensory_state, sensory_params, action_map):
        sensory_state.prev_bmu = next(frozen_iter)
        return sensory_state

    latent_counter = {"value": 0}

    def _fake_latent_step(mem, memory_vectors, latent_state, action_bmu, latent_params, action_map, action_memory, state_memory):
        latent_state.prev_bmu = latent_counter["value"]
        latent_counter["value"] += 1
        return latent_state, 0, state_memory

    monkeypatch.setattr("mercury_runs.pipelines.init_state", lambda observation_dim: _SensoryState())
    action_map = _ActionMap()
    monkeypatch.setattr("mercury_runs.pipelines._build_action_map", lambda action_dim, action_map_params: action_map)
    monkeypatch.setattr("mercury_runs.pipelines.sensory_step", _fake_sensory_step)
    monkeypatch.setattr("mercury_runs.pipelines.sensory_step_frozen", _fake_sensory_step_frozen)
    monkeypatch.setattr("mercury_runs.pipelines.init_mem", lambda n, length=5: _Mem(n, length))
    monkeypatch.setattr("mercury_runs.pipelines.init_latent_state", lambda mem: _LatentState())
    monkeypatch.setattr("mercury_runs.pipelines.update_memory", lambda mem: mem)
    monkeypatch.setattr("mercury_runs.pipelines.add_memory", lambda mem, activation_vector: mem)
    monkeypatch.setattr("mercury_runs.pipelines.latent_step", _fake_latent_step)

    dataset = LoadedDataset(
        observations=np.asarray([[0.0], [1.0], [0.0]], dtype=np.float32),
        actions=np.asarray([[0.0], [1.0], [0.0]], dtype=np.float32),
        collisions=np.asarray([False, False, False]),
        source_metadata={"source": "test"},
        parquet_path=Path("dummy.parquet"),
    )

    sensory_bmu, latent_bmu, latent_node_count, latent_graph = run_latent(
        dataset,
        sensory_params=object(),
        latent_params=object(),
        action_map_params={},
        memory_length=2,
        show_progress=False,
    )

    assert sensory_bmu.tolist() == [0, 1, 0]
    assert latent_bmu.tolist() == [0, 1, 2]
    assert latent_node_count.tolist() == [2, 2, 2]
    assert action_map.step_calls == 3
    assert action_map.predict_calls == 3


def test_build_action_map_identity_for_one_hot() -> None:
    action_map = _build_action_map(
        4,
        {
            "n_codebook": 4,
            "lr": 0.5,
            "sigma": 0.0,
            "key": 0,
            "identity_for_one_hot": True,
        },
    )

    actions = np.eye(4, dtype=np.float32)
    predicted = action_map.predict(actions=actions)
    stepped = [int(action_map.step(action)[0]) for action in actions]

    assert predicted.tolist() == [0, 1, 2, 3]
    assert stepped == [0, 1, 2, 3]
    assert np.array_equal(action_map.state.codebook, np.eye(4, dtype=np.float32))
