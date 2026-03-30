from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from mercury_runs.algorithms.cscg.api import run_cscg
from mercury_runs.algorithms.cscg.config import CSCGConfig
from mercury_runs.algorithms.cscg.prepare import filter_valid_cscg_rows


class _FakeCHMM:
    last_instance: "_FakeCHMM | None" = None

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.pseudocount = kwargs["pseudocount"]
        self.dtype = kwargs.get("dtype", np.float32)
        self.n_clones = np.asarray(kwargs["n_clones"], dtype=np.int64)
        n_actions = int(np.asarray(kwargs["a"]).max()) + 1
        n_states = int(self.n_clones.sum())
        self.C = np.zeros((n_actions, n_states, n_states), dtype=np.float32)
        self.T = np.zeros((2, 4, 4), dtype=np.float32)
        self.Pi_x = np.ones(n_states, dtype=np.float32) / max(1, n_states)
        self.train_calls: list[str] = []
        self.score_calls: list[str] = []
        _FakeCHMM.last_instance = self

    def update_T(self):
        self.T = self.C + self.pseudocount
        norm = self.T.sum(2, keepdims=True)
        norm[norm == 0] = 1
        self.T = self.T / norm

    def learn_em_T(self, x, a, n_iter=100, term_early=True):
        self.train_calls.append(f"em:{n_iter}:{term_early}:{self.pseudocount}")
        return [1.0, 0.5]

    def learn_viterbi_T(self, x, a, n_iter=100):
        self.train_calls.append(f"viterbi:{n_iter}:{self.pseudocount}")
        return [2.0]

    def bps(self, x, a):
        self.score_calls.append("bps")
        return np.array([1.0, 2.0, 3.0], dtype=np.float64)

    def bpsV(self, x, a):
        self.score_calls.append("bpsV")
        return np.array([4.0, 5.0, 6.0], dtype=np.float64)

    def decode(self, x, a):
        self.score_calls.append("decode")
        return np.array([0.25, 0.5, 0.75], dtype=np.float64), np.array([0, 1, 1], dtype=np.int64)


def _fake_forward(T_tr, Pi_x, n_clones, x, a, store_messages=False):
    log2_lik = -np.ones(len(x), dtype=np.float32)
    messages = np.ones(int(np.asarray(n_clones)[np.asarray(x)].sum()), dtype=np.float32)
    return log2_lik, messages


def _fake_backward(T, n_clones, x, a):
    return np.ones(int(np.asarray(n_clones)[np.asarray(x)].sum()), dtype=np.float32)


def _fake_updateC(C, T, n_clones, mess_fwd, mess_bwd, x, a):
    C[:] = 0
    C[0, 0, 0] = float(len(x) - 1)


def _fake_module():
    return type(
        "FakeModule",
        (),
        {
            "CHMM": _FakeCHMM,
            "forward": staticmethod(_fake_forward),
            "backward": staticmethod(_fake_backward),
            "updateC": staticmethod(_fake_updateC),
        },
    )


def _base_config(tmp_path: Path, train_algorithm: str) -> CSCGConfig:
    return CSCGConfig(
        datasets_root=tmp_path,
        output_root=tmp_path / "out",
        run_id=f"run-{train_algorithm}",
        level=0,
        sensor="cartesian",
        ground_truth_states_parquet=tmp_path / "ground_truth.parquet",
        valid_trajectories_only=True,
        train_algorithm=train_algorithm,
        n_iter=3,
        paper_precision_enabled=True,
    )


def _patch_common(
    monkeypatch,
    tmp_path: Path,
    written: dict,
    precision: dict | None = None,
    paper_precision: dict | None = None,
) -> None:
    monkeypatch.setattr(
        "mercury_runs.algorithms.cscg.api.import_cscg_module",
        lambda repo_root: _fake_module(),
    )
    monkeypatch.setattr(
        "mercury_runs.algorithms.cscg.api.load_dataset_arrays",
        lambda **kwargs: (
            np.array([10, 11, 12, 11], dtype=np.int64),
            np.array([0, 1, 0, 1], dtype=np.int64),
            np.array([False, True, False, False]),
            {"seed": 7},
            "dataset.parquet",
        ),
    )
    monkeypatch.setattr(
        "mercury_runs.algorithms.cscg.api.load_ground_truth_states",
        lambda path: np.array([2, 3, 4, 3], dtype=np.int32),
    )
    monkeypatch.setattr(
        pl,
        "read_parquet",
        lambda path: pl.DataFrame({"ground_truth_bmu": [2, 3, 4, 3]}),
    )

    def _capture_write_bundle_parquet(**kwargs):
        written.update(kwargs)
        return tmp_path / "states.parquet"

    monkeypatch.setattr("mercury_runs.algorithms.cscg.artifacts.write_bundle_parquet", _capture_write_bundle_parquet)
    if precision is not None:
        monkeypatch.setattr("mercury_runs.algorithms.cscg.api.compute_cscg_precision", lambda **kwargs: precision.update(kwargs))
    if paper_precision is not None:
        monkeypatch.setattr("mercury_runs.algorithms.cscg.api.compute_cscg_paper_precision", lambda **kwargs: paper_precision.update(kwargs))


def test_run_cscg_baseline_uses_forward_objective_for_em(monkeypatch, tmp_path: Path) -> None:
    written: dict = {}
    precision: dict = {}
    paper_precision: dict = {}
    _patch_common(monkeypatch, tmp_path, written, precision, paper_precision)

    result = run_cscg(_base_config(tmp_path, "em"))

    assert result == tmp_path / "states.parquet"
    assert _FakeCHMM.last_instance is not None
    assert _FakeCHMM.last_instance.train_calls == ["em:3:True:0.002"]
    assert _FakeCHMM.last_instance.score_calls == ["bps", "decode"]
    cscg_eval = written["meta"].run_parameters["cscg_eval"]
    assert cscg_eval["train_algorithm"] == "em"
    assert cscg_eval["objective_metric"] == "forward_bits_per_symbol"
    assert cscg_eval["objective_bits_per_symbol_mean"] == 2.0
    assert cscg_eval["objective_negative_log2_likelihood_sum"] == 6.0
    assert cscg_eval["state_decode_algorithm"] == "viterbi"
    assert cscg_eval["state_decode_negative_log2_likelihood_sum"] == 1.5
    assert cscg_eval["state_decode_negative_log2_likelihood_mean"] == 0.5
    assert cscg_eval["num_clone_states"] == 30
    assert cscg_eval["num_observations"] == 3
    assert cscg_eval["num_actions"] == 2
    assert cscg_eval["valid_transition_filter"] is True
    assert cscg_eval["raw_steps"] == 4
    assert cscg_eval["filtered_steps"] == 3
    assert cscg_eval["dropped_collision_steps"] == 1
    assert cscg_eval["convergence_steps"] == 2
    assert cscg_eval["convergence_steps_by_stage"] == {"em": 2}
    assert cscg_eval["n_step_observation_prediction_accuracy"] == {"n1": 0.5, "n3": 0.0, "n5": 0.0}
    assert cscg_eval["trajectory_log_likelihood_sum"] == pytest.approx(-1.3862943641198908)
    assert cscg_eval["trajectory_log_likelihood_mean"] == pytest.approx(-0.6931471820599454)
    assert cscg_eval["steps"] == 3
    assert written["meta"].run_parameters["training_stages"] == ["em"]
    assert written["meta"].run_parameters["training_data"]["valid_transition_filter"] is True
    assert written["meta"].run_parameters["training_data"]["raw_steps"] == 4
    assert written["meta"].run_parameters["training_data"]["filtered_steps"] == 3
    assert written["meta"].run_parameters["training_data"]["dropped_collision_steps"] == 1
    assert precision["config"].run_id == "run-em"
    assert paper_precision["config"].run_id == "run-em"


def test_run_cscg_baseline_uses_viterbi_objective_for_viterbi(monkeypatch, tmp_path: Path) -> None:
    written: dict = {}
    precision: dict = {}
    paper_precision: dict = {}
    _patch_common(monkeypatch, tmp_path, written, precision, paper_precision)

    run_cscg(_base_config(tmp_path, "viterbi"))

    assert _FakeCHMM.last_instance is not None
    assert _FakeCHMM.last_instance.train_calls == ["em:3:True:0.002", "viterbi:3:0.0"]
    assert _FakeCHMM.last_instance.score_calls == ["bpsV", "decode"]
    assert written["meta"].run_parameters["cscg_eval"]["objective_metric"] == "viterbi_bits_per_symbol"
    assert written["meta"].run_parameters["cscg_eval"]["objective_bits_per_symbol_mean"] == 5.0
    assert written["meta"].run_parameters["cscg_eval"]["n_step_observation_prediction_accuracy"] == {
        "n1": 0.5,
        "n3": 0.0,
        "n5": 0.0,
    }
    assert written["meta"].run_parameters["training_stages"] == ["em", "viterbi"]
    assert precision["config"].run_id == "run-viterbi"
    assert paper_precision["config"].run_id == "run-viterbi"


def test_run_cscg_per_iteration_history_skips_viterbi_stage(monkeypatch, tmp_path: Path) -> None:
    written: dict = {}
    precision: dict = {}
    captured_payload: dict = {}
    _patch_common(monkeypatch, tmp_path, written, precision, paper_precision=None)

    monkeypatch.setattr(
        "mercury_runs.algorithms.cscg.api.compute_cscg_paper_precision_metrics_from_model",
        lambda **kwargs: {"latent_precision": 0.5},
    )
    monkeypatch.setattr(
        "mercury_runs.algorithms.cscg.api.write_cscg_paper_precision_payload",
        lambda **kwargs: captured_payload.update(kwargs),
    )

    config = _base_config(tmp_path, "viterbi").model_copy(
        update={"paper_precision_mode": "per_iteration", "paper_precision_eval_interval": 1}
    )
    run_cscg(config)

    assert captured_payload["mode"] == "per_iteration"
    assert captured_payload["history"]
    stages = [item["stage"] for item in captured_payload["history"]]
    assert all(stage == "em" for stage in stages)
    assert "viterbi" not in stages
    assert [item["step"] for item in captured_payload["history"]] == [3, 6]
    assert [item["observed_samples"] for item in captured_payload["history"]] == [3, 6]
    assert captured_payload["metrics"] == {"latent_precision": 0.5}


def test_run_cscg_per_iteration_uses_training_step_interval(monkeypatch, tmp_path: Path) -> None:
    written: dict = {}
    precision: dict = {}
    captured_payload: dict = {}
    _patch_common(monkeypatch, tmp_path, written, precision, paper_precision=None)

    monkeypatch.setattr(
        "mercury_runs.algorithms.cscg.api.compute_cscg_paper_precision_metrics_from_model",
        lambda **kwargs: {"latent_precision": 0.25},
    )
    monkeypatch.setattr(
        "mercury_runs.algorithms.cscg.api.write_cscg_paper_precision_payload",
        lambda **kwargs: captured_payload.update(kwargs),
    )

    config = _base_config(tmp_path, "em").model_copy(
        update={"paper_precision_mode": "per_iteration", "paper_precision_eval_interval": 4}
    )
    run_cscg(config)

    assert [item["step"] for item in captured_payload["history"]] == [6]
    assert [item["iteration"] for item in captured_payload["history"]] == [2]


def test_run_cscg_per_iteration_records_final_em_point_when_interval_not_reached(monkeypatch, tmp_path: Path) -> None:
    written: dict = {}
    precision: dict = {}
    captured_payload: dict = {}
    _patch_common(monkeypatch, tmp_path, written, precision, paper_precision=None)

    monkeypatch.setattr(
        "mercury_runs.algorithms.cscg.api.compute_cscg_paper_precision_metrics_from_model",
        lambda **kwargs: {"latent_precision": 0.75},
    )
    monkeypatch.setattr(
        "mercury_runs.algorithms.cscg.api.write_cscg_paper_precision_payload",
        lambda **kwargs: captured_payload.update(kwargs),
    )

    config = _base_config(tmp_path, "em").model_copy(
        update={"paper_precision_mode": "per_iteration", "paper_precision_eval_interval": 100}
    )
    run_cscg(config)

    assert [item["step"] for item in captured_payload["history"]] == [6]
    assert [item["iteration"] for item in captured_payload["history"]] == [2]
    assert captured_payload["metrics"] == {"latent_precision": 0.75}


def test_run_cscg_online_em_uses_streamed_batches(monkeypatch, tmp_path: Path) -> None:
    written: dict = {}
    precision: dict = {}
    paper_precision: dict = {}
    _patch_common(monkeypatch, tmp_path, written, precision, paper_precision)

    monkeypatch.setattr(
        "mercury_runs.algorithms.cscg.api.resolve_dataset_path_and_metadata",
        lambda **kwargs: (tmp_path / "dataset.parquet", {"seed": 7}),
    )
    monkeypatch.setattr(
        "mercury_runs.algorithms.cscg.api.collect_streaming_cscg_metadata",
        lambda **kwargs: (
            np.array([10, 11, 12], dtype=np.int64),
            np.array([0, 1], dtype=np.int64),
            {"raw_steps": 4, "filtered_steps": 3, "dropped_collision_steps": 1},
            2,
        ),
    )
    monkeypatch.setattr(
        "mercury_runs.algorithms.cscg.api.iter_cscg_training_batches",
        lambda **kwargs: iter(
            [
                {
                    "observations": np.array([10, 12], dtype=np.int64),
                    "actions": np.array([0, 0], dtype=np.int64),
                    "raw_steps": 2,
                    "filtered_steps": 2,
                    "dropped_collision_steps": 0,
                    "prepended_previous_step": False,
                },
                {
                    "observations": np.array([12, 11], dtype=np.int64),
                    "actions": np.array([0, 1], dtype=np.int64),
                    "raw_steps": 2,
                    "filtered_steps": 1,
                    "dropped_collision_steps": 1,
                    "prepended_previous_step": True,
                },
            ]
        ),
    )

    config = _base_config(tmp_path, "em").model_copy(
        update={"training_mode": "online_em", "batch_size": 2, "online_lambda": 1.0}
    )
    run_cscg(config)

    assert _FakeCHMM.last_instance is not None
    assert _FakeCHMM.last_instance.train_calls == []
    assert written["meta"].run_parameters["cscg_eval"]["training_mode"] == "online_em"
    assert written["meta"].run_parameters["cscg_eval"]["batch_size"] == 2
    assert written["meta"].run_parameters["cscg_eval"]["online_lambda"] == 1.0
    assert written["meta"].run_parameters["cscg_eval"]["num_batches"] == 2
    assert written["meta"].run_parameters["training_data"]["training_mode"] == "online_em"
    assert precision["config"].run_id == "run-em"
    assert paper_precision["config"].run_id == "run-em"


def test_run_cscg_per_iteration_uses_num_points_schedule(monkeypatch, tmp_path: Path) -> None:
    written: dict = {}
    precision: dict = {}
    captured_payload: dict = {}
    _patch_common(monkeypatch, tmp_path, written, precision, paper_precision=None)

    monkeypatch.setattr(
        "mercury_runs.algorithms.cscg.api.compute_cscg_paper_precision_metrics_from_model",
        lambda **kwargs: {"latent_precision": 0.4},
    )
    monkeypatch.setattr(
        "mercury_runs.algorithms.cscg.api.write_cscg_paper_precision_payload",
        lambda **kwargs: captured_payload.update(kwargs),
    )

    config = _base_config(tmp_path, "em").model_copy(
        update={"n_iter": 4, "paper_precision_mode": "per_iteration", "paper_precision_eval_interval": 1000, "paper_precision_num_points": 2}
    )
    run_cscg(config)

    assert [item["iteration"] for item in captured_payload["history"]] == [1, 2]
    assert [item["step"] for item in captured_payload["history"]] == [3, 6]


def test_run_cscg_online_em_num_points_uses_available_batch_count(monkeypatch, tmp_path: Path) -> None:
    written: dict = {}
    precision: dict = {}
    captured_payload: dict = {}
    _patch_common(monkeypatch, tmp_path, written, precision, paper_precision=None)

    monkeypatch.setattr(
        "mercury_runs.algorithms.cscg.api.resolve_dataset_path_and_metadata",
        lambda **kwargs: (tmp_path / "dataset.parquet", {"seed": 7}),
    )
    monkeypatch.setattr(
        "mercury_runs.algorithms.cscg.api.collect_streaming_cscg_metadata",
        lambda **kwargs: (
            np.array([10, 11, 12], dtype=np.int64),
            np.array([0, 1], dtype=np.int64),
            {"raw_steps": 6, "filtered_steps": 6, "dropped_collision_steps": 0},
            3,
        ),
    )
    monkeypatch.setattr(
        "mercury_runs.algorithms.cscg.api.iter_cscg_training_batches",
        lambda **kwargs: iter(
            [
                {
                    "observations": np.array([10, 11], dtype=np.int64),
                    "actions": np.array([0, 0], dtype=np.int64),
                    "raw_steps": 2,
                    "filtered_steps": 2,
                    "dropped_collision_steps": 0,
                    "prepended_previous_step": False,
                },
                {
                    "observations": np.array([11, 12], dtype=np.int64),
                    "actions": np.array([0, 1], dtype=np.int64),
                    "raw_steps": 2,
                    "filtered_steps": 2,
                    "dropped_collision_steps": 0,
                    "prepended_previous_step": True,
                },
                {
                    "observations": np.array([12, 10], dtype=np.int64),
                    "actions": np.array([1, 0], dtype=np.int64),
                    "raw_steps": 2,
                    "filtered_steps": 2,
                    "dropped_collision_steps": 0,
                    "prepended_previous_step": True,
                },
            ]
        ),
    )
    monkeypatch.setattr(
        "mercury_runs.algorithms.cscg.api.compute_cscg_paper_precision_metrics_from_model",
        lambda **kwargs: {"latent_precision": 0.4},
    )
    monkeypatch.setattr(
        "mercury_runs.algorithms.cscg.api.write_cscg_paper_precision_payload",
        lambda **kwargs: captured_payload.update(kwargs),
    )

    config = _base_config(tmp_path, "em").model_copy(
        update={
            "training_mode": "online_em",
            "batch_size": 2,
            "n_iter": 1000,
            "term_early": False,
            "paper_precision_mode": "per_iteration",
            "paper_precision_eval_interval": 1000,
            "paper_precision_num_points": 3,
        }
    )
    run_cscg(config)

    assert [item["iteration"] for item in captured_payload["history"]] == [1, 2, 3]


def test_run_cscg_online_em_with_viterbi_keeps_final_full_refinement(monkeypatch, tmp_path: Path) -> None:
    written: dict = {}
    precision: dict = {}
    paper_precision: dict = {}
    _patch_common(monkeypatch, tmp_path, written, precision, paper_precision)

    monkeypatch.setattr(
        "mercury_runs.algorithms.cscg.api.resolve_dataset_path_and_metadata",
        lambda **kwargs: (tmp_path / "dataset.parquet", {"seed": 7}),
    )
    monkeypatch.setattr(
        "mercury_runs.algorithms.cscg.api.collect_streaming_cscg_metadata",
        lambda **kwargs: (
            np.array([10, 11, 12], dtype=np.int64),
            np.array([0, 1], dtype=np.int64),
            {"raw_steps": 4, "filtered_steps": 3, "dropped_collision_steps": 1},
            1,
        ),
    )
    monkeypatch.setattr(
        "mercury_runs.algorithms.cscg.api.iter_cscg_training_batches",
        lambda **kwargs: iter(
            [
                {
                    "observations": np.array([10, 12], dtype=np.int64),
                    "actions": np.array([0, 0], dtype=np.int64),
                    "raw_steps": 2,
                    "filtered_steps": 2,
                    "dropped_collision_steps": 0,
                    "prepended_previous_step": False,
                }
            ]
        ),
    )

    config = _base_config(tmp_path, "viterbi").model_copy(
        update={"training_mode": "online_em", "batch_size": 2, "online_lambda": 0.9}
    )
    run_cscg(config)

    assert _FakeCHMM.last_instance is not None
    assert _FakeCHMM.last_instance.train_calls == ["viterbi:3:0.0"]
    assert written["meta"].run_parameters["training_stages"] == ["em", "viterbi"]
    assert written["meta"].run_parameters["cscg_eval"]["final_viterbi_refinement"] is True


def test_filter_valid_cscg_rows_drops_collision_steps() -> None:
    obs, actions, gt, stats = filter_valid_cscg_rows(
        observations=np.array([[1], [2], [3]], dtype=np.int64),
        actions=np.array([[0], [1], [2]], dtype=np.int64),
        collisions=np.array([False, True, False]),
        ground_truth_bmu=np.array([5, 6, 7], dtype=np.int32),
    )

    assert obs.tolist() == [[1], [3]]
    assert actions.tolist() == [[0], [2]]
    assert gt.tolist() == [5, 7]
    assert stats == {"raw_steps": 3, "filtered_steps": 2, "dropped_collision_steps": 1}


def test_filter_valid_cscg_rows_can_preserve_collision_steps() -> None:
    obs, actions, gt, stats = filter_valid_cscg_rows(
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
