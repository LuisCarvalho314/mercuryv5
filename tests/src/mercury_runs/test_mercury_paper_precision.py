from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mercury.memory.state import init_mem
from mercury.latent.state import init_latent_state
from mercury.sensory.params import SensoryParams
from mercury.sensory.state import init_state, sensory_step_predict_only
from mercury_runs.algorithms.mercury.api import run_mercury
from mercury_runs.algorithms.mercury.config import MercuryConfig
from mercury_runs.algorithms.mercury.evaluate import _run_mercury_snapshot_on_walk, compute_mercury_paper_precision_metrics
from mercury_runs.infrastructure.paper_precision import generate_random_start_walks
from mercury_runs.pipelines import run_ground_truth_cartesian, run_latent


def _base_config(tmp_path: Path) -> MercuryConfig:
    return MercuryConfig(
        datasets_root=tmp_path / "datasets",
        results_root=tmp_path / "out",
        run_id="run-mercury",
        level=0,
        sensor="cartesian",
        memory_length=3,
        window_length=5,
        seed=0,
        rand_prob=0.3,
        num_steps=4,
        sensory={"activation_threshold": 0.95, "topological_neighbourhood_threshold": 0.6, "max_neurons": 10, "sensory_weighting": 0.8, "winning_node_lr": 0.55, "topological_neighbourhood_lr": 0.9, "action_lr": 0.5, "global_context_lr": 0.9, "max_age": 18, "gaussian_shape": 2, "allow_self_loops": False},
        latent={"allow_self_loops": True, "max_neurons": 20, "action_lr": 0.1, "gaussian_shape": 2, "max_age": 18, "ambiguity_threshold": 10, "trace_decay": 0.99, "weight_memory": 0.4, "weight_undirected": 0.2, "weight_base": 0.2, "weight_action": 0.2, "memory_replay": True, "memory_disambiguation": True},
        action_map={"n_codebook": 4, "lr": 0.5, "sigma": 0.0, "key": 0},
        paper_precision_enabled=True,
        paper_precision_mode="per_iteration",
        paper_precision_eval_interval=2,
    )


def _advance_sensory_state(state):
    state.prev_bmu = 0
    state.step_idx += 1
    return state


def _advance_latent_state(state):
    state.prev_bmu = 0
    state.step_idx += 1
    return state


def test_run_mercury_per_iteration_writes_history(monkeypatch, tmp_path: Path) -> None:
    written: dict = {}
    metrics_calls: list[int] = []

    monkeypatch.setattr("mercury_runs.algorithms.mercury.api.generate_mercury_datasets", lambda **kwargs: {})

    def _fake_run_bundle(*, config, dataset_run_ids, latent_step_callback=None, show_progress=True):
        if latent_step_callback is not None:
            latent_step_callback(step=2, sensory_graph=object(), latent_graph=object(), action_map=object())
            latent_step_callback(step=4, sensory_graph=object(), latent_graph=object(), action_map=object())
        return tmp_path / "states.parquet"

    monkeypatch.setattr("mercury_runs.algorithms.mercury.api.run_mercury_bundle", _fake_run_bundle)
    monkeypatch.setattr("mercury_runs.algorithms.mercury.api._compute_mercury_precision", lambda **kwargs: None)

    def _fake_metrics(*, config, sensory_graph=None, latent_graph=None, action_map=None, show_progress=False, progress_desc=None):
        metrics_calls.append(len(metrics_calls) + 1)
        return {"sensory_precision": 0.5, "latent_precision": 0.75}

    monkeypatch.setattr("mercury_runs.algorithms.mercury.api.compute_mercury_paper_precision_metrics", _fake_metrics)
    monkeypatch.setattr(
        "mercury_runs.algorithms.mercury.api.write_mercury_paper_precision_payload",
        lambda **kwargs: written.update(kwargs),
    )

    result = run_mercury(_base_config(tmp_path))

    assert result == tmp_path / "states.parquet"
    assert metrics_calls == [1, 2]
    assert written["schedule_unit"] == "training_step"
    assert written["history"] == [
        {"step": 2, "observed_samples": 2, "sensory_precision": 0.5, "latent_precision": 0.75},
        {"step": 4, "observed_samples": 4, "sensory_precision": 0.5, "latent_precision": 0.75},
    ]
    assert written["metrics"] == {"sensory_precision": 0.5, "latent_precision": 0.75}


def test_run_mercury_per_iteration_uses_num_points_schedule(monkeypatch, tmp_path: Path) -> None:
    written: dict = {}

    monkeypatch.setattr("mercury_runs.algorithms.mercury.api.generate_mercury_datasets", lambda **kwargs: {})

    def _fake_run_bundle(*, config, dataset_run_ids, latent_step_callback=None, show_progress=True):
        if latent_step_callback is not None:
            for step in range(1, 11):
                latent_step_callback(step=step, sensory_graph=object(), latent_graph=object(), action_map=object())
        return tmp_path / "states.parquet"

    monkeypatch.setattr("mercury_runs.algorithms.mercury.api.run_mercury_bundle", _fake_run_bundle)
    monkeypatch.setattr("mercury_runs.algorithms.mercury.api._compute_mercury_precision", lambda **kwargs: None)
    monkeypatch.setattr(
        "mercury_runs.algorithms.mercury.api.compute_mercury_paper_precision_metrics",
        lambda **kwargs: {"sensory_precision": 0.5, "latent_precision": 0.75},
    )
    monkeypatch.setattr(
        "mercury_runs.algorithms.mercury.api.write_mercury_paper_precision_payload",
        lambda **kwargs: written.update(kwargs),
    )

    config = _base_config(tmp_path).model_copy(update={"num_steps": 10, "paper_precision_num_points": 3, "paper_precision_eval_interval": 1000})
    run_mercury(config)

    assert [item["step"] for item in written["history"]] == [1, 5, 10]


def test_run_mercury_final_uses_frozen_snapshot(monkeypatch, tmp_path: Path) -> None:
    written: dict = {}
    received: dict = {}
    config = _base_config(tmp_path).model_copy(update={"paper_precision_mode": "final"})

    monkeypatch.setattr("mercury_runs.algorithms.mercury.api.generate_mercury_datasets", lambda **kwargs: {})

    def _fake_run_bundle(*, config, dataset_run_ids, latent_step_callback=None, show_progress=True):
        if latent_step_callback is not None:
            latent_step_callback(step=4, sensory_graph="sensory-graph", latent_graph="latent-graph", action_map="action-map")
        return tmp_path / "states.parquet"

    monkeypatch.setattr("mercury_runs.algorithms.mercury.api.run_mercury_bundle", _fake_run_bundle)
    monkeypatch.setattr("mercury_runs.algorithms.mercury.api._compute_mercury_precision", lambda **kwargs: None)

    def _fake_metrics(*, config, sensory_graph=None, latent_graph=None, action_map=None, show_progress=False, progress_desc=None):
        received.update(
            sensory_graph=sensory_graph,
            latent_graph=latent_graph,
            action_map=action_map,
            show_progress=show_progress,
            progress_desc=progress_desc,
        )
        return {"sensory_precision": 0.6, "latent_precision": 0.8}

    monkeypatch.setattr("mercury_runs.algorithms.mercury.api.compute_mercury_paper_precision_metrics", _fake_metrics)
    monkeypatch.setattr(
        "mercury_runs.algorithms.mercury.api.write_mercury_paper_precision_payload",
        lambda **kwargs: written.update(kwargs),
    )

    result = run_mercury(config)

    assert result == tmp_path / "states.parquet"
    assert received == {
        "sensory_graph": "sensory-graph",
        "latent_graph": "latent-graph",
        "action_map": "action-map",
        "show_progress": True,
        "progress_desc": "Mercury Paper Precision",
    }
    assert written["schedule_unit"] == "final_only"
    assert written["history"] == []
    assert written["metrics"] == {"sensory_precision": 0.6, "latent_precision": 0.8}


def test_mercury_pipeline_progress_uses_tqdm(monkeypatch) -> None:
    calls: list[dict] = []

    class _FakeBar(list):
        def set_postfix(self, **kwargs):
            calls.append({"postfix": kwargs})

    monkeypatch.setattr("mercury_runs.pipelines.tqdm", lambda iterable, **kwargs: calls.append(kwargs) or _FakeBar(iterable))
    monkeypatch.setattr("mercury_runs.pipelines.init_state", lambda observation_dim: type("S", (), {"gs": type("G", (), {"n": 2, "node_features": {"activation": np.array([1.0, 0.0], dtype=np.float32)}})(), "prev_bmu": None})())
    monkeypatch.setattr("mercury_runs.pipelines._build_action_map", lambda action_dim, action_map_params: type("A", (), {"step": lambda self, action: (0, action)})())
    monkeypatch.setattr("mercury_runs.pipelines.sensory_step", lambda observation, action_bmu, sensory_state, sensory_params, action_map: setattr(sensory_state, "prev_bmu", 0) or sensory_state)
    monkeypatch.setattr("mercury_runs.pipelines.sensory_step_frozen", lambda observation, action_bmu, sensory_state, sensory_params, action_map: setattr(sensory_state, "prev_bmu", 0) or sensory_state)
    monkeypatch.setattr("mercury_runs.pipelines.init_mem", lambda n, length=5: type("M", (), {"gs": type("MG", (), {"n": n * length})(), "length": length, "sensory_n_nodes": n})())
    monkeypatch.setattr("mercury_runs.pipelines.init_latent_state", lambda mem: type("L", (), {"g": type("LG", (), {"n": 2})(), "prev_bmu": 0})())
    monkeypatch.setattr("mercury_runs.pipelines.update_memory", lambda mem: mem)
    monkeypatch.setattr("mercury_runs.pipelines.add_memory", lambda mem, activation_vector: mem)
    monkeypatch.setattr("mercury_runs.pipelines.latent_step", lambda mem, memory_vectors, latent_state, action_bmu, latent_params, action_map, action_memory, state_memory: (latent_state, 0, state_memory))

    dataset = type(
        "Dataset",
        (),
        {
            "observations": np.array([[0.0], [1.0]], dtype=np.float32),
            "actions": np.array([[1.0], [0.0]], dtype=np.float32),
            "collisions": np.array([False, False]),
        },
    )()

    run_ground_truth_cartesian(dataset, sensory_params=object(), action_map_params={}, show_progress=True, progress_desc="Mercury GT")
    run_latent(dataset, sensory_params=object(), latent_params=object(), action_map_params={}, memory_length=2, show_progress=True, progress_desc="Mercury Latent")

    assert any(call.get("desc") == "Mercury GT" for call in calls if "desc" in call)
    assert any(call.get("desc") == "Mercury Latent Sensory" for call in calls if "desc" in call)
    assert any(call.get("desc") == "Mercury Latent" for call in calls if "desc" in call)


def test_mercury_paper_precision_progress_uses_tqdm(monkeypatch, tmp_path: Path) -> None:
    calls: list[dict] = []

    class _Graph:
        adj = np.zeros((2, 2), dtype=np.float32)
        edge_features = {"action": np.zeros((2, 2, 1), dtype=np.int32)}

    monkeypatch.setattr("mercury_runs.algorithms.mercury.evaluate.tqdm", lambda iterable, **kwargs: calls.append(kwargs) or iterable)
    monkeypatch.setattr(
        "mercury_runs.algorithms.mercury.evaluate.generate_random_start_walks",
        lambda **kwargs: [
            type("Walk", (), {"observations": np.array([[0.0]], dtype=np.float32), "cartesian_observations": np.array([[0.0]], dtype=np.float32), "actions": np.array([[1.0]], dtype=np.float32), "collisions": np.array([False])})()
        ],
    )
    monkeypatch.setattr(
        "mercury_runs.algorithms.mercury.evaluate.cartesian_state_ids",
        lambda positions: np.array([1], dtype=np.int32),
    )
    monkeypatch.setattr(
        "mercury_runs.algorithms.mercury.evaluate.exact_cartesian_state_ids_for_level",
        lambda **kwargs: np.array([1], dtype=np.int32),
    )
    monkeypatch.setattr(
        "mercury_runs.algorithms.mercury.evaluate.exact_cartesian_reference_positions",
        lambda level_index: np.array([[0, 0], [0, 1]], dtype=np.int64),
    )
    monkeypatch.setattr(
        "mercury_runs.algorithms.mercury.evaluate._run_mercury_snapshot_on_walk",
        lambda *args, **kwargs: (
            np.array([1], dtype=np.int32),
            np.array([1], dtype=np.int32),
        ),
        )

    compute_mercury_paper_precision_metrics(
        config=_base_config(tmp_path),
        sensory_graph=_Graph(),
        latent_graph=_Graph(),
        action_map=object(),
        show_progress=True,
        progress_desc="Mercury Paper Precision",
    )

    assert any(call.get("desc") == "Mercury Paper Precision" for call in calls if "desc" in call)


def test_mercury_paper_precision_passes_valid_trajectory_flag(monkeypatch, tmp_path: Path) -> None:
    captured: dict = {}

    class _Graph:
        adj = np.zeros((2, 2), dtype=np.float32)
        edge_features = {"action": np.zeros((2, 2, 1), dtype=np.int32)}

    def _fake_walks(**kwargs):
        captured.update(kwargs)
        return [
            type(
                "Walk",
                (),
                {
                    "observations": np.array([[0.0]], dtype=np.float32),
                    "cartesian_observations": np.array([[0.0, 0.0]], dtype=np.float32),
                    "actions": np.array([[1.0]], dtype=np.float32),
                    "collisions": np.array([False]),
                },
            )()
        ]

    monkeypatch.setattr("mercury_runs.algorithms.mercury.evaluate.generate_random_start_walks", _fake_walks)
    monkeypatch.setattr("mercury_runs.algorithms.mercury.evaluate.cartesian_state_ids", lambda positions: np.array([0], dtype=np.int32))
    monkeypatch.setattr(
        "mercury_runs.algorithms.mercury.evaluate.exact_cartesian_state_ids_for_level",
        lambda **kwargs: np.array([0], dtype=np.int32),
    )
    monkeypatch.setattr(
        "mercury_runs.algorithms.mercury.evaluate.exact_cartesian_reference_positions",
        lambda level_index: np.array([[0, 0], [0, 1]], dtype=np.int64),
    )
    monkeypatch.setattr(
        "mercury_runs.algorithms.mercury.evaluate._run_mercury_snapshot_on_walk",
        lambda *args, **kwargs: (np.array([0], dtype=np.int32), np.array([0], dtype=np.int32)),
    )

    compute_mercury_paper_precision_metrics(
        config=_base_config(tmp_path).model_copy(update={"mercury_valid_trajectories_only": True}),
        sensory_graph=_Graph(),
        latent_graph=_Graph(),
        action_map=object(),
    )

    assert captured["valid_trajectories_only"] is True


def test_mercury_paper_precision_base_valid_flag_passes_walk_filter(monkeypatch, tmp_path: Path) -> None:
    captured: dict = {}

    class _Graph:
        adj = np.zeros((2, 2), dtype=np.float32)
        edge_features = {"action": np.zeros((2, 2, 1), dtype=np.int32)}

    def _fake_walks(**kwargs):
        captured.update(kwargs)
        return [
            type(
                "Walk",
                (),
                {
                    "observations": np.array([[0.0]], dtype=np.float32),
                    "cartesian_observations": np.array([[0.0, 0.0]], dtype=np.float32),
                    "actions": np.array([[1.0]], dtype=np.float32),
                    "collisions": np.array([False]),
                },
            )()
        ]

    monkeypatch.setattr("mercury_runs.algorithms.mercury.evaluate.generate_random_start_walks", _fake_walks)
    monkeypatch.setattr("mercury_runs.algorithms.mercury.evaluate.cartesian_state_ids", lambda positions: np.array([0], dtype=np.int32))
    monkeypatch.setattr(
        "mercury_runs.algorithms.mercury.evaluate.exact_cartesian_state_ids_for_level",
        lambda **kwargs: np.array([0], dtype=np.int32),
    )
    monkeypatch.setattr(
        "mercury_runs.algorithms.mercury.evaluate.exact_cartesian_reference_positions",
        lambda level_index: np.array([[0, 0], [0, 1]], dtype=np.int64),
    )
    monkeypatch.setattr(
        "mercury_runs.algorithms.mercury.evaluate._run_mercury_snapshot_on_walk",
        lambda *args, **kwargs: (np.array([0], dtype=np.int32), np.array([0], dtype=np.int32)),
    )

    compute_mercury_paper_precision_metrics(
        config=_base_config(tmp_path).model_copy(update={"valid_trajectories_only": True}),
        sensory_graph=_Graph(),
        latent_graph=_Graph(),
        action_map=object(),
    )

    assert captured["valid_trajectories_only"] is True


def test_mercury_snapshot_replay_skips_latent_updates_on_collision_when_requested(monkeypatch) -> None:
    walk = type(
        "Walk",
        (),
        {
            "observations": np.array([[0.0], [1.0], [2.0]], dtype=np.float32),
            "actions": np.array([[1.0], [0.0], [1.0]], dtype=np.float32),
            "collisions": np.array([False, True, False]),
        },
    )()
    update_calls = {"update_memory": 0, "add_memory": 0, "latent_step": 0}

    monkeypatch.setattr(
        "mercury_runs.algorithms.mercury.evaluate.sensory_step_predict_only",
        lambda observation, state, params: (
            setattr(state, "prev_bmu", int(float(np.asarray(observation)[0]))),
            np.array([1.0], dtype=np.float32),
        )[1:] and (state, np.array([1.0], dtype=np.float32)),
    )
    monkeypatch.setattr("mercury_runs.algorithms.mercury.evaluate.init_global_context", lambda observation_dim: np.zeros((observation_dim,), dtype=np.float32))
    monkeypatch.setattr("mercury_runs.algorithms.mercury.evaluate.init_mem", lambda n, length: type("M", (), {"gs": type("G", (), {"n": n * length})()})())
    monkeypatch.setattr(
        "mercury_runs.algorithms.mercury.evaluate._init_latent_state_compat",
        lambda mem, n_actions: type("L", (), {"g": type("LG", (), {"n": 2})(), "mapping": None, "prev_bmu": 7})(),
    )
    monkeypatch.setattr(
        "mercury_runs.algorithms.mercury.evaluate.update_memory",
        lambda mem: update_calls.__setitem__("update_memory", update_calls["update_memory"] + 1) or mem,
    )
    monkeypatch.setattr(
        "mercury_runs.algorithms.mercury.evaluate.add_memory",
        lambda mem, activation_vector: update_calls.__setitem__("add_memory", update_calls["add_memory"] + 1) or mem,
    )

    def _fake_latent_step(mem, latent_state, action_bmu, latent_params, state_memory):
        update_calls["latent_step"] += 1
        latent_state.prev_bmu += 1
        return latent_state, None, state_memory

    monkeypatch.setattr("mercury_runs.algorithms.mercury.evaluate.latent_step_predict_only", _fake_latent_step)

    class _ActionMap:
        def predict(self, *, action=None):
            return 0

    sensory_bmu, latent_bmu = _run_mercury_snapshot_on_walk(
        walk=walk,
        sensory_graph=type("SG", (), {"n": 1})(),
        latent_graph=type("LG", (), {"n": 2})(),
        action_map=_ActionMap(),
        sensory_params=object(),
        latent_params=object(),
        memory_length=2,
        latent_valid_trajectories_only=True,
    )

    assert sensory_bmu.tolist() == [0, 1, 2]
    assert latent_bmu.tolist() == [8, 8, 9]
    assert update_calls == {"update_memory": 2, "add_memory": 2, "latent_step": 2}


def test_mercury_paper_precision_reports_purity_and_link_error_for_identity_actions(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        "mercury_runs.algorithms.mercury.evaluate.generate_random_start_walks",
        lambda **kwargs: [
            type(
                "Walk",
                (),
                {
                    "observations": np.array([[0.0], [1.0]], dtype=np.float32),
                    "cartesian_observations": np.array([[0.0, 0.0], [0.0, 1.0]], dtype=np.float32),
                    "actions": np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float32),
                    "collisions": np.array([False, False]),
                },
            )()
        ],
    )
    monkeypatch.setattr("mercury_runs.algorithms.mercury.evaluate.cartesian_state_ids", lambda positions: np.array([0, 1], dtype=np.int32))
    monkeypatch.setattr(
        "mercury_runs.algorithms.mercury.evaluate.exact_cartesian_state_ids_for_level",
        lambda **kwargs: np.array([0, 1], dtype=np.int32),
    )
    monkeypatch.setattr(
        "mercury_runs.algorithms.mercury.evaluate._run_mercury_snapshot_on_walk",
        lambda *args, **kwargs: (
            np.array([0, 1], dtype=np.int32),
            np.array([0, 1], dtype=np.int32),
        ),
    )
    monkeypatch.setattr(
        "mercury_runs.algorithms.mercury.evaluate.exact_cartesian_reference_positions",
        lambda level_index: np.array([[0, 0], [0, 1]], dtype=np.int64),
    )
    monkeypatch.setattr(
        "mercury_runs.algorithms.mercury.evaluate.exact_valid_sensorimotor_transitions",
        lambda **kwargs: {(0, 0, 1)},
    )

    class _Graph:
        def __init__(self):
            self.adj = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=np.float32)
            self.edge_features = {"action": np.array([[[0], [0]], [[0], [0]]], dtype=np.int32)}

    metrics = compute_mercury_paper_precision_metrics(
        config=_base_config(tmp_path).model_copy(update={"action_map": {"n_codebook": 4, "lr": 0.5, "sigma": 0.0, "key": 0, "identity_for_one_hot": True}}),
        sensory_graph=_Graph(),
        latent_graph=_Graph(),
        action_map=type("IdentityMap", (), {"identity_mapping": True})(),
    )

    assert metrics["sensory_purity"] == pytest.approx(100.0)
    assert metrics["latent_purity"] == pytest.approx(100.0)
    assert metrics["sensory_mean_total_variation"] == pytest.approx(0.0)
    assert metrics["latent_mean_total_variation"] == pytest.approx(0.0)
    assert metrics["sensory_edge_precision"] == pytest.approx(1.0)
    assert metrics["sensory_edge_recall"] == pytest.approx(1.0)
    assert metrics["sensory_edge_f1"] == pytest.approx(1.0)
    assert metrics["sensory_action_conditioned_mean_total_variation"] == pytest.approx(0.0)
    assert metrics["sensory_action_conditioned_edge_precision"] == pytest.approx(1.0)
    assert metrics["sensory_action_conditioned_edge_recall"] == pytest.approx(1.0)
    assert metrics["sensory_action_conditioned_edge_f1"] == pytest.approx(1.0)
    assert metrics["latent_edge_precision"] == pytest.approx(1.0)
    assert metrics["latent_edge_recall"] == pytest.approx(1.0)
    assert metrics["latent_edge_f1"] == pytest.approx(1.0)
    assert metrics["latent_action_conditioned_mean_total_variation"] == pytest.approx(0.0)
    assert metrics["latent_action_conditioned_edge_precision"] == pytest.approx(1.0)
    assert metrics["latent_action_conditioned_edge_recall"] == pytest.approx(1.0)
    assert metrics["latent_action_conditioned_edge_f1"] == pytest.approx(1.0)
    assert metrics["sensory_sensorimotor_link_error"] == pytest.approx((0.0 + 1e-4) / (1.0 + 1e-4))
    assert metrics["latent_sensorimotor_link_error"] == pytest.approx((0.0 + 1e-4) / (1.0 + 1e-4))


def test_mercury_paper_precision_marks_link_error_unsupported_without_identity_actions(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        "mercury_runs.algorithms.mercury.evaluate.generate_random_start_walks",
        lambda **kwargs: [
            type(
                "Walk",
                (),
                {
                    "observations": np.array([[0.0], [1.0]], dtype=np.float32),
                    "cartesian_observations": np.array([[0.0, 0.0], [0.0, 1.0]], dtype=np.float32),
                    "actions": np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float32),
                    "collisions": np.array([False, False]),
                },
            )()
        ],
    )
    monkeypatch.setattr("mercury_runs.algorithms.mercury.evaluate.cartesian_state_ids", lambda positions: np.array([0, 1], dtype=np.int32))
    monkeypatch.setattr(
        "mercury_runs.algorithms.mercury.evaluate.exact_cartesian_state_ids_for_level",
        lambda **kwargs: np.array([0, 1], dtype=np.int32),
    )
    monkeypatch.setattr(
        "mercury_runs.algorithms.mercury.evaluate._run_mercury_snapshot_on_walk",
        lambda *args, **kwargs: (
            np.array([0, 1], dtype=np.int32),
            np.array([0, 1], dtype=np.int32),
        ),
    )
    monkeypatch.setattr(
        "mercury_runs.algorithms.mercury.evaluate.exact_cartesian_reference_positions",
        lambda level_index: np.array([[0, 0], [0, 1]], dtype=np.int64),
    )

    class _Graph:
        def __init__(self):
            self.adj = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=np.float32)
            self.edge_features = {"action": np.array([[[0], [0]], [[0], [0]]], dtype=np.int32)}

    metrics = compute_mercury_paper_precision_metrics(
        config=_base_config(tmp_path),
        sensory_graph=_Graph(),
        latent_graph=_Graph(),
        action_map=type("LearnedMap", (), {"identity_mapping": False})(),
    )

    assert metrics["sensory_purity"] == pytest.approx(100.0)
    assert metrics["latent_purity"] == pytest.approx(100.0)
    assert metrics["sensory_mean_total_variation"] == pytest.approx(0.0)
    assert metrics["latent_mean_total_variation"] == pytest.approx(0.0)
    assert metrics["sensory_edge_f1"] == pytest.approx(1.0)
    assert metrics["latent_edge_f1"] == pytest.approx(1.0)
    assert metrics["sensory_action_conditioned_mean_total_variation"] is None
    assert metrics["latent_action_conditioned_mean_total_variation"] is None
    assert metrics["sensory_action_conditioned_edge_f1"] is None
    assert metrics["latent_action_conditioned_edge_f1"] is None
    assert metrics["sensory_action_conditioned_edge_f1_reason"] == "action_map_does_not_preserve_true_action_ids"
    assert metrics["latent_action_conditioned_edge_f1_reason"] == "action_map_does_not_preserve_true_action_ids"
    assert metrics["sensory_sensorimotor_link_error"] is None
    assert metrics["latent_sensorimotor_link_error"] is None
    assert metrics["sensory_sensorimotor_link_error_reason"] == "action_map_does_not_preserve_true_action_ids"
    assert metrics["latent_sensorimotor_link_error_reason"] == "action_map_does_not_preserve_true_action_ids"


def test_generate_random_start_walks_valid_only_filters_collisions(monkeypatch) -> None:
    class _FakeEnv:
        def __init__(self):
            self.agent = type("AgentState", (), {"position": (0, 0), "make_observation": lambda self, maze: None})()
            self.agent_position = (0, 0)
            self.maze = np.zeros((2, 2), dtype=np.uint8)
            self._steps = iter(
                [
                    ((0, 0), np.array([1.0], dtype=np.float32), True, (0, 0)),
                    ((0, 1), np.array([2.0], dtype=np.float32), False, (0, 1)),
                    ((0, 1), np.array([3.0], dtype=np.float32), True, (0, 1)),
                    ((1, 1), np.array([4.0], dtype=np.float32), False, (1, 1)),
                ]
            )

        def random_action(self) -> str:
            return "north"

        def random_policy(self, previous_action: str, collision: bool, rand_prob: float) -> str:
            return previous_action

        def step(self, action: str):
            observation, action_vec, collision, position = next(self._steps)
            self.agent.position = position
            self.agent_position = position
            return observation, action_vec, collision

    monkeypatch.setattr("mercury_runs.infrastructure.paper_precision._open_positions", lambda level_index: [(0, 0)])
    monkeypatch.setattr("mercury_runs.infrastructure.paper_precision._init_env", lambda **kwargs: _FakeEnv())
    monkeypatch.setattr("mercury_runs.infrastructure.paper_precision._set_start_position", lambda env, position: None)

    walk = generate_random_start_walks(
        level=0,
        sensor="cartesian",
        sensor_range=None,
        rand_prob=0.3,
        num_walks=1,
        walk_length=2,
        base_seed=0,
        valid_trajectories_only=True,
    )[0]

    assert walk.observations.tolist() == [[0, 1], [1, 1]]
    assert walk.actions[:, 0].tolist() == [2.0, 4.0]
    assert walk.collisions.tolist() == [False, False]
    assert walk.cartesian_observations.tolist() == [[0, 1], [1, 1]]


def test_sensory_step_predict_only_keeps_graph_fixed() -> None:
    sensory_state = init_state(2)
    before_weights = sensory_state.gs.node_features["weight"].copy()
    before_context = sensory_state.gs.node_features["context"].copy()
    before_activation = sensory_state.gs.node_features["activation"].copy()
    before_adj = sensory_state.gs.adj.copy()
    sensory_params = SensoryParams(**_base_config(Path("/tmp")).sensory)

    next_state, activation = sensory_step_predict_only(
        np.array([0.0, 0.0], dtype=np.float32),
        sensory_state,
        sensory_params,
    )

    assert np.array_equal(sensory_state.gs.node_features["weight"], before_weights)
    assert np.array_equal(sensory_state.gs.node_features["context"], before_context)
    assert np.array_equal(sensory_state.gs.node_features["activation"], before_activation)
    assert np.array_equal(sensory_state.gs.adj, before_adj)
    assert int(np.argmax(activation)) == int(next_state.prev_bmu)
    assert next_state.step_idx == 1


def test_mercury_snapshot_replay_uses_predict_only_action_map(monkeypatch) -> None:
    sensory_graph = init_state(1).gs
    latent_graph = init_latent_state(init_mem(sensory_graph.n, 2)).g
    original_sensory_activation = sensory_graph.node_features["activation"].copy()
    original_latent_activation = latent_graph.node_features["activation"].copy()
    walk = type(
        "Walk",
        (),
        {
            "observations": np.array([[0.0], [1.0]], dtype=np.float32),
            "actions": np.array([[1.0], [0.0]], dtype=np.float32),
            "collisions": np.array([False, False]),
        },
    )()
    calls = {"predict": 0}

    class _ActionMap:
        def predict(self, *, action=None, actions=None):
            assert action is not None
            calls["predict"] += 1
            return 0

        def step(self, action):
            raise AssertionError("step() should not be used during frozen replay")

    monkeypatch.setattr(
        "mercury_runs.algorithms.mercury.evaluate.sensory_step_predict_only",
        lambda observation, state, cfg: (_advance_sensory_state(state), np.array([1.0, 0.0], dtype=np.float32)),
    )
    monkeypatch.setattr(
        "mercury_runs.algorithms.mercury.evaluate.latent_step_predict_only",
        lambda ms, state, action_bmu, cfg, state_mem: (_advance_latent_state(state), 0, state_mem),
    )

    _run_mercury_snapshot_on_walk(
        walk=walk,
        sensory_graph=sensory_graph,
        latent_graph=latent_graph,
        action_map=_ActionMap(),
        sensory_params=object(),
        latent_params=object(),
        memory_length=2,
        latent_valid_trajectories_only=False,
    )

    assert calls["predict"] == 2
    assert np.array_equal(sensory_graph.node_features["activation"], original_sensory_activation)
    assert np.array_equal(latent_graph.node_features["activation"], original_latent_activation)
