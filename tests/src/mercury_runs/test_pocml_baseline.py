from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from mercury_runs.algorithms.pocml.api import run_pocml
from mercury_runs.algorithms.pocml.config import POCMLConfig
from mercury_runs.algorithms.pocml.evaluate import (
    compute_ground_truth_observation_prediction_accuracy,
    evaluate_pocml_sequence,
    load_pocml_checkpoint,
)
from mercury_runs.algorithms.pocml.prepare import filter_valid_pocml_rows


class _ArrayBox:
    def __init__(self, arr):
        self.arr = np.asarray(arr)

    def to(self, dtype):
        return self

    def item(self):
        return self.arr.item()

    def unsqueeze(self, dim):
        return _ArrayBox(np.expand_dims(self.arr, axis=dim))

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return np.asarray(self.arr)

    def __getitem__(self, idx):
        out = self.arr[idx]
        return _ArrayBox(out) if isinstance(out, np.ndarray) else out


class _FakeTorch:
    float32 = "float32"
    long = "long"

    @staticmethod
    def tensor(values, dtype=None):
        return _ArrayBox(values)

    @staticmethod
    def argmax(values, dim=None):
        arr = values.arr if isinstance(values, _ArrayBox) else np.asarray(values)
        return _ArrayBox(np.argmax(arr, axis=dim))

    @staticmethod
    def device(name):
        return name


class _FakeDataLoader:
    def __init__(self, dataset, batch_size, shuffle):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        return iter(self.dataset)


class _FakeFunctional:
    @staticmethod
    def one_hot(values, num_classes):
        arr = values.arr if isinstance(values, _ArrayBox) else np.asarray(values)
        arr = np.asarray(arr, dtype=np.int64).reshape(-1)
        out = np.zeros((arr.shape[0], num_classes), dtype=np.float32)
        out[np.arange(arr.shape[0]), arr] = 1.0
        return _ArrayBox(out)


_FakeTorch.nn = types.SimpleNamespace(functional=types.SimpleNamespace(one_hot=_FakeFunctional.one_hot))


class _EvalModel:
    def __init__(self):
        self.u = _ArrayBox([[0.2, 0.8]])
        self.predicted_states = [_ArrayBox([[0.1, 0.9]]), _ArrayBox([[0.7, 0.3]])]
        self.predicted_obs = [_ArrayBox([[0.2, 0.8]]), _ArrayBox([[0.9, 0.1]])]
        self.corrected_states = [_ArrayBox([[0.05, 0.95]]), _ArrayBox([[0.6, 0.4]])]
        self.calls: list[str] = []
        self._step = 0

    def init_time(self):
        self.calls.append("init_time")

    def init_state(self, obs=None, state_idx=None, mask=None):
        self.calls.append("init_state")
        self.u = _ArrayBox([[0.2, 0.8]])

    def update_state(self, oh_a):
        self.calls.append(f"update_state:{self._step}")
        return _ArrayBox([[1.0, 0.0]])

    def get_expected_state(self, hd_state):
        self.calls.append(f"get_expected_state:{self._step}")
        return self.predicted_states[self._step]

    def get_obs_from_memory(self, oh_u_next):
        self.calls.append(f"get_obs_from_memory:{self._step}")
        return self.predicted_obs[self._step]

    def update_state_given_obs(self, oh_o_next):
        self.calls.append(f"update_state_given_obs:{self._step}")
        self.u = self.corrected_states[self._step]

    def clean_up(self):
        self.calls.append(f"clean_up:{self._step}")

    def inc_time(self):
        self.calls.append(f"inc_time:{self._step}")
        self._step += 1


def test_evaluate_native_pocml_matches_repo_accuracy_flow() -> None:
    model = _EvalModel()

    result = evaluate_pocml_sequence(
        model=model,
        obs_idx=np.array([1, 1, 0], dtype=np.int64),
        action_idx=np.array([0, 1], dtype=np.int64),
        n_obs=2,
        n_actions=2,
        torch_module=_FakeTorch,
        functional_module=_FakeFunctional,
    )

    assert result["direct_observation_prediction_accuracy"] == {"n1": 1.0}
    assert result["next_obs_steps"] == 2
    assert result["predicted_obs"].tolist() == [1, 1, 0]
    assert result["latent_proxy_state_ids"].tolist() == [1, 1, 0]
    assert result["sensory_proxy_state_ids"].tolist() == [1, 1, 0]
    assert "init_memory" not in model.calls
    assert all(not call.startswith("update_memory") for call in model.calls)


class _RolloutEvalModel:
    def __init__(self):
        self.u = _ArrayBox([[1.0, 0.0]])
        self._step = 0
        self._actions: list[int] = []

    def init_time(self):
        self._step = 0
        self._actions = []

    def init_state(self, obs=None, state_idx=None, mask=None):
        self.u = _ArrayBox([[1.0, 0.0]])

    def update_state(self, oh_a):
        action = int(np.argmax(oh_a.arr))
        self._actions.append(action)
        return _ArrayBox([[1.0, 0.0]])

    def get_expected_state(self, hd_state):
        return _ArrayBox([[1.0, 0.0]])

    def get_obs_from_memory(self, oh_u_next):
        pred_obs = 1 if sum(self._actions) == 0 else 0
        out = np.zeros((1, 2), dtype=np.float32)
        out[0, pred_obs] = 1.0
        return _ArrayBox(out)

    def clean_up(self):
        self._step += 1

    def inc_time(self):
        self._step += 1


def test_compute_ground_truth_observation_prediction_accuracy_maps_predicted_obs_to_gt() -> None:
    result = compute_ground_truth_observation_prediction_accuracy(
        model=_RolloutEvalModel(),
        obs_idx=np.array([0, 0, 1], dtype=np.int64),
        action_idx=np.array([0, 1], dtype=np.int64),
        ground_truth_bmu=np.array([8, 7, 9], dtype=np.int32),
        obs_to_gt=np.array([9, 7], dtype=np.int32),
        horizons=[1, 2],
        n_obs=2,
        n_actions=2,
        torch_module=_FakeTorch,
    )

    assert result == {"n1": 1.0, "n2": 1.0}


def test_evaluate_pocml_sequence_uses_progress_when_requested(monkeypatch) -> None:
    calls: list[dict] = []
    monkeypatch.setattr(
        "mercury_runs.algorithms.pocml.evaluate.tqdm",
        lambda iterable, **kwargs: calls.append(kwargs) or iterable,
    )

    evaluate_pocml_sequence(
        model=_EvalModel(),
        obs_idx=np.array([1, 1, 0], dtype=np.int64),
        action_idx=np.array([0, 1], dtype=np.int64),
        n_obs=2,
        n_actions=2,
        torch_module=_FakeTorch,
        functional_module=_FakeFunctional,
        show_progress=True,
        progress_desc="POCML Native Eval",
    )

    assert any(call.get("desc") == "POCML Native Eval" for call in calls if "desc" in call)


class _FakeParam:
    def __init__(self, arr):
        self.arr = np.asarray(arr, dtype=np.float32)

    def __rmul__(self, other):
        return _ArrayBox(other * self.arr)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.arr


class _TrainModel:
    def __init__(self, q_value: float):
        self.Q = _FakeParam([[q_value, q_value]])
        self.V = _FakeParam([[q_value]])
        self.M = _FakeParam([[[1.0, 0.0], [0.0, 1.0]]])


class _FakePOCMLModule:
    class POCML(_TrainModel):
        def __init__(self, **kwargs):
            super().__init__(q_value=1.0)


class _FakeTrainer:
    def __init__(self, model, **kwargs):
        self.model = model

    def train(self, epochs=10):
        return np.array([[3.0, 2.0]], dtype=np.float32), _TrainModel(q_value=9.0)


def _fake_torch_modules():
    torch_mod = types.ModuleType("torch")
    torch_mod.float32 = "float32"
    torch_mod.long = "long"
    torch_mod.tensor = lambda values, dtype=None: _ArrayBox(values)
    torch_mod.device = lambda name: name
    torch_mod.softmax = lambda values, dim: _ArrayBox(
        np.exp(values.arr - np.max(values.arr, axis=dim, keepdims=True))
        / np.exp(values.arr - np.max(values.arr, axis=dim, keepdims=True)).sum(axis=dim, keepdims=True)
    )
    nn_mod = types.ModuleType("torch.nn")
    functional_mod = types.ModuleType("torch.nn.functional")
    functional_mod.one_hot = lambda values, num_classes: _FakeFunctional.one_hot(values, num_classes)
    nn_mod.functional = functional_mod
    torch_mod.nn = nn_mod
    utils_mod = types.ModuleType("torch.utils")
    data_mod = types.ModuleType("torch.utils.data")
    data_mod.DataLoader = _FakeDataLoader
    utils_mod.data = data_mod
    torch_mod.utils = utils_mod
    return torch_mod, nn_mod, functional_mod, utils_mod, data_mod


def test_run_pocml_baseline_uses_best_model_and_posterior_proxy(monkeypatch, tmp_path: Path) -> None:
    written: dict = {}
    saved: dict = {}
    paper_precision: dict = {}
    torch_mod, nn_mod, functional_mod, utils_mod, data_mod = _fake_torch_modules()
    monkeypatch.setitem(sys.modules, "torch", torch_mod)
    monkeypatch.setitem(sys.modules, "torch.nn", nn_mod)
    monkeypatch.setitem(sys.modules, "torch.nn.functional", functional_mod)
    monkeypatch.setitem(sys.modules, "torch.utils", utils_mod)
    monkeypatch.setitem(sys.modules, "torch.utils.data", data_mod)

    monkeypatch.setattr(
        "mercury_runs.algorithms.pocml.train.import_pocml_modules",
        lambda repo_root: (_FakePOCMLModule, types.SimpleNamespace(POCMLTrainer=_FakeTrainer)),
    )
    monkeypatch.setattr(
        "mercury_runs.algorithms.pocml.train.load_dataset_arrays",
        lambda **kwargs: (
            np.array([10, 11, 12, 13], dtype=np.int64),
            np.array([0, 1, 1, 0], dtype=np.int64),
            np.array([False, True, False, False]),
            {"seed": 5},
            "dataset.parquet",
        ),
    )
    monkeypatch.setattr(
        "mercury_runs.algorithms.pocml.train.load_ground_truth_states",
        lambda path: np.array([2, 3, 4, 5], dtype=np.int32),
    )
    monkeypatch.setattr(
        "mercury_runs.algorithms.pocml.train.build_train_trajectories",
        lambda **kwargs: ["trajectory"],
    )
    monkeypatch.setattr(
        "mercury_runs.algorithms.pocml.api.evaluate_pocml_sequence",
        lambda **kwargs: {
            "predicted_obs": np.array([0, 1, 0], dtype=np.int64),
            "latent_proxy_state_ids": np.array([1, 1, 0], dtype=np.int32),
            "sensory_proxy_state_ids": np.array([1, 0, 0], dtype=np.int32),
            "direct_observation_prediction_accuracy": {"n1": 0.5},
            "next_obs_confidence_mean": 0.6,
            "next_obs_confidence_std": 0.1,
            "trajectory_log_likelihood_sum": -3.0,
            "trajectory_log_likelihood_mean": -1.5,
            "next_obs_steps": 2,
        },
    )
    monkeypatch.setattr(
        "mercury_runs.algorithms.pocml.api.compute_direct_observation_prediction_accuracy",
        lambda **kwargs: {"n3": 0.0, "n5": 0.0},
    )
    monkeypatch.setattr(
        "mercury_runs.algorithms.pocml.api.compute_ground_truth_observation_prediction_accuracy",
        lambda **kwargs: {"n1": 1.0, "n3": 0.5, "n5": 0.0},
    )

    def _capture_embeddings(**kwargs):
        saved.update(kwargs)
        return tmp_path / "embeddings.npz"

    def _capture_checkpoint(**kwargs):
        saved["checkpoint"] = kwargs
        return tmp_path / "model.pt"

    def _capture_write_bundle_parquet(**kwargs):
        written.update(kwargs)
        return tmp_path / "states.parquet"

    monkeypatch.setattr("mercury_runs.algorithms.pocml.api.save_pocml_embeddings", _capture_embeddings)
    monkeypatch.setattr("mercury_runs.algorithms.pocml.api.save_pocml_checkpoint", _capture_checkpoint)
    monkeypatch.setattr("mercury_runs.algorithms.pocml.artifacts.write_bundle_parquet", _capture_write_bundle_parquet)
    monkeypatch.setattr("mercury_runs.algorithms.pocml.api.compute_pocml_precision", lambda **kwargs: None)
    monkeypatch.setattr(
        "mercury_runs.algorithms.pocml.api.compute_pocml_paper_precision_metrics_from_model",
        lambda **kwargs: {"capacity_precision": 0.5},
    )
    monkeypatch.setattr(
        "mercury_runs.algorithms.pocml.api.write_pocml_paper_precision_payload",
        lambda **kwargs: paper_precision.update(kwargs),
    )

    config = POCMLConfig(
        datasets_root=tmp_path,
        output_root=tmp_path / "out",
        run_id="run-pocml",
        level=0,
        sensor="cartesian",
        ground_truth_states_parquet=tmp_path / "ground_truth.parquet",
        valid_trajectories_only=True,
        epochs=2,
        paper_precision_enabled=True,
    )
    result = run_pocml(config)

    assert result == tmp_path / "states.parquet"
    assert saved["q_matrix"].tolist() == [[9.0, 9.0]]
    assert written["columns"]["sensory_proxy_state_id"].tolist() == [1, 0, 0]
    assert written["columns"]["latent_proxy_state_id"].tolist() == [1, 1, 0]
    assert written["meta"].run_parameters["training_data"]["model_selection"] == "best_model_by_mean_epoch_loss"
    assert written["meta"].run_parameters["training_data"]["valid_transition_filter"] is True
    assert written["meta"].run_parameters["training_data"]["raw_steps"] == 4
    assert written["meta"].run_parameters["training_data"]["filtered_steps"] == 3
    assert written["meta"].run_parameters["training_data"]["dropped_collision_steps"] == 1
    assert written["meta"].run_parameters["training_data"]["batch_size"] == 1
    assert written["meta"].run_parameters["training_data"]["memory_bias"] is True
    assert written["meta"].run_parameters["bmu_proxy"]["method"] == "posterior_argmax_state"
    assert written["meta"].run_parameters["pocml_eval"]["n_step_observation_prediction_accuracy"] == {
        "n1": 1.0,
        "n3": 0.5,
        "n5": 0.0,
    }
    assert written["meta"].run_parameters["pocml_eval"]["direct_observation_prediction_accuracy"] == {
        "n1": 0.5,
        "n3": 0.0,
        "n5": 0.0,
    }
    assert written["meta"].run_parameters["pocml_eval"]["paper_precision_metric_name"] == "paper_precision_by_capacity"
    assert written["meta"].run_parameters["pocml_eval"]["valid_transition_filter"] is True
    assert written["meta"].run_parameters["pocml_eval"]["raw_steps"] == 4
    assert written["meta"].run_parameters["pocml_eval"]["filtered_steps"] == 3
    assert written["meta"].run_parameters["pocml_eval"]["dropped_collision_steps"] == 1
    assert written["meta"].run_parameters["pocml_eval"]["precision_state_series"] == "latent_proxy_state_id"
    assert written["meta"].run_parameters["pocml_eval"]["primary_precision_capacity"] == 3
    assert written["meta"].run_parameters["pocml_eval"]["precision_capacities"] == [3]
    assert paper_precision["resolved_capacities"] == [3]
    assert paper_precision["metrics_by_capacity"] == {"3": {"capacity_precision": 0.5}}


def test_run_pocml_per_iteration_uses_num_points_schedule(monkeypatch, tmp_path: Path) -> None:
    written: dict = {}
    captured_payload: dict = {}
    torch_mod, nn_mod, functional_mod, utils_mod, data_mod = _fake_torch_modules()
    monkeypatch.setitem(sys.modules, "torch", torch_mod)
    monkeypatch.setitem(sys.modules, "torch.nn", nn_mod)
    monkeypatch.setitem(sys.modules, "torch.nn.functional", functional_mod)
    monkeypatch.setitem(sys.modules, "torch.utils", utils_mod)
    monkeypatch.setitem(sys.modules, "torch.utils.data", data_mod)
    monkeypatch.setattr(
        "mercury_runs.algorithms.pocml.train.import_pocml_modules",
        lambda repo_root: (_FakePOCMLModule, types.SimpleNamespace(POCMLTrainer=_FakeTrainer)),
    )
    monkeypatch.setattr(
        "mercury_runs.algorithms.pocml.train.load_dataset_arrays",
        lambda **kwargs: (
            np.array([10, 11, 12, 13], dtype=np.int64),
            np.array([0, 1, 1, 0], dtype=np.int64),
            np.array([False, True, False, False]),
            {"seed": 5},
            "dataset.parquet",
        ),
    )
    monkeypatch.setattr(
        "mercury_runs.algorithms.pocml.train.load_ground_truth_states",
        lambda path: np.array([2, 3, 4, 5], dtype=np.int32),
    )
    monkeypatch.setattr(
        "mercury_runs.algorithms.pocml.train.build_train_trajectories",
        lambda **kwargs: ["trajectory"],
    )
    monkeypatch.setattr(
        "mercury_runs.algorithms.pocml.api.evaluate_pocml_sequence",
        lambda **kwargs: {
            "predicted_obs": np.array([0, 1, 0], dtype=np.int64),
            "latent_proxy_state_ids": np.array([1, 1, 0], dtype=np.int32),
            "sensory_proxy_state_ids": np.array([1, 0, 0], dtype=np.int32),
            "direct_observation_prediction_accuracy": {"n1": 0.5},
            "next_obs_confidence_mean": 0.6,
            "next_obs_confidence_std": 0.1,
            "trajectory_log_likelihood_sum": -3.0,
            "trajectory_log_likelihood_mean": -1.5,
            "next_obs_steps": 2,
        },
    )
    monkeypatch.setattr("mercury_runs.algorithms.pocml.api.compute_direct_observation_prediction_accuracy", lambda **kwargs: {"n3": 0.0, "n5": 0.0})
    monkeypatch.setattr("mercury_runs.algorithms.pocml.api.compute_ground_truth_observation_prediction_accuracy", lambda **kwargs: {"n1": 1.0, "n3": 0.5, "n5": 0.0})
    monkeypatch.setattr("mercury_runs.algorithms.pocml.api.save_pocml_embeddings", lambda **kwargs: tmp_path / "embeddings.npz")
    monkeypatch.setattr("mercury_runs.algorithms.pocml.api.save_pocml_checkpoint", lambda **kwargs: tmp_path / "model.pt")
    monkeypatch.setattr("mercury_runs.algorithms.pocml.artifacts.write_bundle_parquet", lambda **kwargs: written.update(kwargs) or (tmp_path / "states.parquet"))
    monkeypatch.setattr("mercury_runs.algorithms.pocml.api.compute_pocml_precision", lambda **kwargs: None)
    monkeypatch.setattr(
        "mercury_runs.algorithms.pocml.api.compute_pocml_paper_precision_metrics_from_model",
        lambda **kwargs: {"capacity_precision": 0.5},
    )
    monkeypatch.setattr(
        "mercury_runs.algorithms.pocml.api.write_pocml_paper_precision_payload",
        lambda **kwargs: captured_payload.update(kwargs),
    )

    config = POCMLConfig(
        datasets_root=tmp_path,
        output_root=tmp_path / "out",
        run_id="run-pocml-points",
        level=0,
        sensor="cartesian",
        ground_truth_states_parquet=tmp_path / "ground_truth.parquet",
        valid_trajectories_only=True,
        epochs=6,
        paper_precision_enabled=True,
        paper_precision_mode="per_iteration",
        paper_precision_eval_interval=1000,
        paper_precision_num_points=3,
    )
    run_pocml(config)

    assert [item["step"] for item in captured_payload["history_by_capacity"]["3"]] == [1, 3, 6]
    assert [item["observed_samples"] for item in captured_payload["history_by_capacity"]["3"]] == [10, 30, 60]
    assert captured_payload["metrics_by_capacity"] == {"3": {"capacity_precision": 0.5}}


def test_run_pocml_writes_multi_capacity_paper_precision_payload(monkeypatch, tmp_path: Path) -> None:
    written: dict = {}
    captured_payload: dict = {}

    def _fake_single_capacity(*, config, precision_capacities):
        capacity = int(config.n_states)
        if capacity == 3:
            written["primary_precision_capacities"] = precision_capacities
        return {
            "states_path": tmp_path / f"{config.run_id}_states.parquet",
            "resolved_capacity": capacity,
            "paper_precision_metrics": {"capacity_precision": capacity / 10.0},
            "paper_precision_history": [{"step": 1, "observed_samples": 15, "capacity_precision": capacity / 10.0}],
            "checkpoint_path": tmp_path / f"{config.run_id}.pt",
            "embeddings_path": tmp_path / f"{config.run_id}.npz",
        }

    monkeypatch.setattr("mercury_runs.algorithms.pocml.api._run_single_pocml_capacity", _fake_single_capacity)
    monkeypatch.setattr(
        "mercury_runs.algorithms.pocml.api.write_pocml_paper_precision_payload",
        lambda **kwargs: captured_payload.update(kwargs),
    )

    result = run_pocml(
        POCMLConfig(
            datasets_root=tmp_path,
            output_root=tmp_path / "out",
            run_id="run-pocml-multi",
            level=0,
            sensor="cartesian",
            ground_truth_states_parquet=tmp_path / "ground_truth.parquet",
            precision_capacities=[3, 5],
            paper_precision_enabled=True,
            n_states=99,
        )
    )

    assert result == tmp_path / "run-pocml-multi_states.parquet"
    assert written["primary_precision_capacities"] == [3, 5]
    assert captured_payload["resolved_capacities"] == [3, 5]
    assert captured_payload["history_by_capacity"]["3"][0]["observed_samples"] == 15
    assert captured_payload["metrics_by_capacity"] == {
        "3": {"capacity_precision": 0.3},
        "5": {"capacity_precision": 0.5},
    }


def test_load_pocml_checkpoint_prepares_model_for_single_sequence_eval(monkeypatch, tmp_path: Path) -> None:
    class _CheckpointModel:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.loaded = None

        def load_state_dict(self, state_dict):
            self.loaded = state_dict

    class _CheckpointModule:
        POCML = _CheckpointModel

    class _FakeTorchModule:
        @staticmethod
        def load(path, map_location="cpu", weights_only=True):
            return {"weights": 1}

    prepared = {}

    monkeypatch.setattr(
        "mercury_runs.algorithms.pocml.evaluate.import_pocml_modules",
        lambda repo_root: (_CheckpointModule, object()),
    )
    monkeypatch.setitem(sys.modules, "torch", _FakeTorchModule)
    monkeypatch.setattr(
        "mercury_runs.algorithms.pocml.evaluate.prepare_model_for_single_sequence_eval",
        lambda model: prepared.setdefault("model", model) or model,
    )

    config = POCMLConfig(
        datasets_root=tmp_path,
        output_root=tmp_path / "out",
        run_id="run-pocml",
        level=0,
        sensor="cartesian",
        ground_truth_states_parquet=tmp_path / "ground_truth.parquet",
    )

    model, model_mod, torch_mod = load_pocml_checkpoint(config=config, n_obs=3, n_actions=2)

    assert model is prepared["model"]
    assert isinstance(model, _CheckpointModel)
    assert model.loaded == {"weights": 1}
    assert model_mod is _CheckpointModule
    assert torch_mod is _FakeTorchModule


def test_train_pocml_model_rejects_memory_bias_when_capacity_exceeds_observations(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        "mercury_runs.algorithms.pocml.train.import_pocml_modules",
        lambda repo_root: (_FakePOCMLModule, types.SimpleNamespace(POCMLTrainer=_FakeTrainer)),
    )
    monkeypatch.setattr(
        "mercury_runs.algorithms.pocml.train.load_dataset_arrays",
        lambda **kwargs: (
            np.array([10, 11, 12, 13], dtype=np.int64),
            np.array([0, 1, 1, 0], dtype=np.int64),
            np.array([False, True, False, False]),
            {"seed": 5},
            "dataset.parquet",
        ),
    )
    monkeypatch.setattr(
        "mercury_runs.algorithms.pocml.train.load_ground_truth_states",
        lambda path: np.array([2, 3, 4, 5], dtype=np.int32),
    )

    with pytest.raises(ValueError, match="memory_bias=True is incompatible with n_states > n_obs"):
        from mercury_runs.algorithms.pocml.train import train_pocml_model

        train_pocml_model(
            config=POCMLConfig(
                datasets_root=tmp_path,
                output_root=tmp_path / "out",
                run_id="run-pocml-invalid",
                level=0,
                sensor="cartesian",
                ground_truth_states_parquet=tmp_path / "ground_truth.parquet",
                n_states=5,
                memory_bias=True,
            )
        )


def test_filter_valid_pocml_rows_drops_collision_steps() -> None:
    obs, actions, gt, stats = filter_valid_pocml_rows(
        observations=np.array([[1], [2], [3]], dtype=np.int64),
        actions=np.array([[0], [1], [2]], dtype=np.int64),
        collisions=np.array([False, True, False]),
        ground_truth_bmu=np.array([5, 6, 7], dtype=np.int32),
    )

    assert obs.tolist() == [[1], [3]]
    assert actions.tolist() == [[0], [2]]
    assert gt.tolist() == [5, 7]
    assert stats == {"raw_steps": 3, "filtered_steps": 2, "dropped_collision_steps": 1}


def test_filter_valid_pocml_rows_can_preserve_collision_steps() -> None:
    obs, actions, gt, stats = filter_valid_pocml_rows(
        observations=np.array([[1], [2], [3]], dtype=np.int64),
        actions=np.array([[0], [1], [2]], dtype=np.int64),
        collisions=np.array([False, True, False]),
        ground_truth_bmu=np.array([5, 6, 7], dtype=np.int32),
        valid_trajectories_only=False,
    )

    assert obs.tolist() == [[1], [2], [3]]
    assert actions.tolist() == [[0], [1], [2]]
    assert gt.tolist() == [5, 6, 7]
    assert stats == {"raw_steps": 3, "filtered_steps": 3, "dropped_collision_steps": 0}
