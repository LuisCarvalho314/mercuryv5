from __future__ import annotations

from pathlib import Path

import numpy as np

from mercury.action_map.adapter import ActionMap
from mercury.graph.core import Graph
from mercury_runs.infrastructure.paper_precision import resolve_structure_metric_epsilons, structure_epsilon_key
from mercury_runs.infrastructure.paper_precision_snapshots import (
    list_pocml_paper_precision_snapshots,
    load_cscg_paper_precision_snapshot,
    load_mercury_paper_precision_snapshot,
    load_pocml_paper_precision_snapshot,
    write_cscg_paper_precision_snapshot,
    write_mercury_paper_precision_snapshot,
    write_pocml_paper_precision_snapshot,
)


class DummyMercuryConfig:
    def __init__(self, results_root: Path):
        self.results_root = results_root
        self.level = 16
        self.sensor = "cartesian"
        self.sensor_range = None


class DummyCSCGConfig:
    def __init__(self, output_root: Path):
        self.output_root = output_root
        self.level = 16
        self.sensor = "cartesian"
        self.sensor_range = None


class DummyPOCMLConfig:
    def __init__(self, output_root: Path):
        self.output_root = output_root
        self.n_states = None
        self.level = 16
        self.sensor = "cartesian"
        self.sensor_range = None
        self.state_dim = 50
        self.random_feature_dim = 500
        self.alpha = 2.0
        self.memory_bias = False


class DummyTorch:
    @staticmethod
    def save(state_dict, path: str) -> None:
        Path(path).write_text(str(state_dict), encoding="utf-8")


def test_resolve_structure_metric_epsilons_preserves_order_and_deduplicates() -> None:
    class Config:
        structure_metrics_epsilon = 1e-6
        structure_metrics_epsilons = [1e-3, 1e-6, 1e-3]

    assert resolve_structure_metric_epsilons(Config()) == [1e-3, 1e-6]
    assert structure_epsilon_key(1e-6) == "1e-06"


def test_mercury_snapshot_round_trip(tmp_path: Path) -> None:
    graph = Graph()
    graph.add_node()
    graph.add_node()
    graph.add_edge(0, 1, weight=1.0)
    action_map = ActionMap.identity(4)
    config = DummyMercuryConfig(tmp_path / "run" / "bundles" / "mercury")

    path = write_mercury_paper_precision_snapshot(
        config=config,
        step=10,
        observed_samples=10,
        sensory_graph=graph,
        latent_graph=graph,
        action_map=action_map,
    )

    loaded = load_mercury_paper_precision_snapshot(path)
    assert loaded["meta"]["step"] == 10
    assert np.array_equal(loaded["sensory_graph"].adj, graph.adj)
    assert np.array_equal(loaded["latent_graph"].adj, graph.adj)
    assert loaded["action_map"].identity_mapping is True


def test_cscg_snapshot_round_trip(tmp_path: Path) -> None:
    config = DummyCSCGConfig(tmp_path / "run" / "bundles" / "cscg")
    model_npz = {
        "T": np.ones((2, 3, 3), dtype=np.float32),
        "n_clones": np.asarray([1, 2], dtype=np.int64),
    }

    path = write_cscg_paper_precision_snapshot(
        config=config,
        stage="em",
        iteration=7,
        step=70,
        observed_samples=70,
        model_npz=model_npz,
        obs_uniques=np.asarray([1, 2]),
        action_uniques=np.asarray([0, 1]),
    )

    loaded = load_cscg_paper_precision_snapshot(path)
    assert loaded["meta"]["iteration"] == 7
    assert np.array_equal(loaded["model_npz"]["T"], model_npz["T"])
    assert np.array_equal(loaded["n_clones"], model_npz["n_clones"])


def test_cscg_snapshot_loads_vocabulary_sidecar(tmp_path: Path) -> None:
    snapshot_dir = tmp_path / "snapshots" / "mercury_cscg_hybrid" / "paper_precision" / "step=5"
    snapshot_dir.mkdir(parents=True)
    np.savez_compressed(
        snapshot_dir / "model.npz",
        T=np.ones((2, 3, 3), dtype=np.float32),
        n_clones=np.asarray([1, 2], dtype=np.int64),
    )
    np.savez_compressed(
        snapshot_dir / "observation_vocabulary.npz",
        obs_uniques=np.asarray([[1, 0], [2, 0]]),
        action_uniques=np.asarray([[1, 0], [0, 1]]),
    )
    (snapshot_dir / "snapshot.json").write_text(
        '{"method": "mercury_cscg_hybrid", "step": 5, "observed_samples": 50}',
        encoding="utf-8",
    )

    loaded = load_cscg_paper_precision_snapshot(snapshot_dir)

    assert loaded["meta"]["step"] == 5
    assert np.array_equal(loaded["obs_uniques"], np.asarray([[1, 0], [2, 0]]))
    assert np.array_equal(loaded["action_uniques"], np.asarray([[1, 0], [0, 1]]))


def test_pocml_snapshot_uses_resolved_capacity_for_directory_and_metadata(tmp_path: Path) -> None:
    config = DummyPOCMLConfig(tmp_path / "run" / "bundles" / "pocml")

    path = write_pocml_paper_precision_snapshot(
        config=config,
        step=5,
        observed_samples=50,
        obs_uniques=np.asarray([1, 2]),
        action_uniques=np.asarray([0, 1]),
        model_state_dict={"weights": [1]},
        torch_module=DummyTorch,
        n_states=7,
    )

    loaded = load_pocml_paper_precision_snapshot(path)
    assert path.name == "step=5"
    assert path.parent.name == "capacity=7"
    assert loaded["meta"]["model_arch"]["n_states"] == 7
    assert list_pocml_paper_precision_snapshots(tmp_path / "run") == {7: [path]}
