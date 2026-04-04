from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mercury_runs.algorithms.cscg.config import CSCGConfig
from mercury_runs.algorithms.cscg.evaluate import compute_cscg_paper_precision_metrics_from_model
from mercury_runs.infrastructure.paper_precision import EvaluationWalk


class _DecodeCHMM:
    decode_calls = 0

    def __init__(self, **kwargs) -> None:
        self.T = np.zeros((1, 2, 2), dtype=np.float32)

    def decode(self, x, a):
        type(self).decode_calls += 1
        if type(self).decode_calls == 1:
            raise AssertionError("p_obs > 0")
        return np.array([0.0, 0.0], dtype=np.float64), np.array([0, 1], dtype=np.int64)


class _DecodeModule:
    CHMM = _DecodeCHMM


def _base_config(tmp_path: Path) -> CSCGConfig:
    return CSCGConfig(
        datasets_root=tmp_path,
        output_root=tmp_path / "out",
        run_id="paper-precision",
        level=0,
        sensor="cartesian",
        ground_truth_states_parquet=tmp_path / "ground_truth.parquet",
        paper_precision_num_walks=2,
        paper_precision_walk_length=2,
    )


def test_cscg_paper_precision_skips_unsupported_walks(monkeypatch, tmp_path: Path) -> None:
    _DecodeCHMM.decode_calls = 0
    monkeypatch.setattr(
        "mercury_runs.algorithms.cscg.evaluate.generate_random_start_walks",
        lambda **kwargs: [
            EvaluationWalk(
                observations=np.array([[0, 0], [1, 0]], dtype=np.int64),
                cartesian_observations=np.array([[0, 0], [0, 1]], dtype=np.int64),
                actions=np.array([[1], [1]], dtype=np.int64),
                collisions=np.array([False, False], dtype=bool),
                start_position=(0, 0),
                seed=0,
            ),
            EvaluationWalk(
                observations=np.array([[0, 0], [1, 0]], dtype=np.int64),
                cartesian_observations=np.array([[0, 0], [0, 1]], dtype=np.int64),
                actions=np.array([[1], [1]], dtype=np.int64),
                collisions=np.array([False, False], dtype=bool),
                start_position=(0, 0),
                seed=1,
            ),
        ],
    )
    monkeypatch.setattr(
        "mercury_runs.algorithms.cscg.evaluate.exact_cartesian_state_ids_for_level",
        lambda **kwargs: np.array([0, 1], dtype=np.int64),
    )
    monkeypatch.setattr(
        "mercury_runs.algorithms.cscg.evaluate.exact_cartesian_reference_positions",
        lambda level_index: np.array([[0, 0], [0, 1]], dtype=np.int64),
    )
    monkeypatch.setattr(
        "mercury_runs.algorithms.cscg.evaluate.exact_valid_sensorimotor_transitions",
        lambda **kwargs: {(0, 0, 0), (0, 0, 1), (1, 0, 0), (1, 0, 1)},
    )

    metrics = compute_cscg_paper_precision_metrics_from_model(
        model_npz={"T": np.ones((1, 2, 2), dtype=np.float32)},
        n_clones=np.array([1, 1], dtype=np.int64),
        config=_base_config(tmp_path),
        obs_uniques=np.array([[0, 0], [1, 0]], dtype=np.int64),
        action_uniques=np.array([[1]], dtype=np.int64),
        chmm_mod=_DecodeModule(),
    )

    assert metrics["requested_walks"] == 2
    assert metrics["attempted_walks"] == 2
    assert metrics["decoded_walks"] == 1
    assert metrics["unsupported_walks"] == 1
    assert metrics["skipped_short_walks"] == 0
    assert metrics["decode_coverage"] == 0.5
    assert metrics["evaluation_steps"] == 2
    assert metrics["latent_precision"] == 1.0


def test_cscg_paper_precision_reports_purity_and_link_error(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        "mercury_runs.algorithms.cscg.evaluate.generate_random_start_walks",
        lambda **kwargs: [
            EvaluationWalk(
                observations=np.array([[0, 0], [1, 0]], dtype=np.int64),
                cartesian_observations=np.array([[0, 0], [0, 1]], dtype=np.int64),
                actions=np.array([[1], [1]], dtype=np.int64),
                collisions=np.array([False, False], dtype=bool),
                start_position=(0, 0),
                seed=0,
            )
        ],
    )
    monkeypatch.setattr(
        "mercury_runs.algorithms.cscg.evaluate.exact_cartesian_state_ids_for_level",
        lambda **kwargs: np.array([0, 1], dtype=np.int64),
    )
    monkeypatch.setattr(
        "mercury_runs.algorithms.cscg.evaluate.exact_cartesian_reference_positions",
        lambda level_index: np.array([[0, 0], [0, 1]], dtype=np.int64),
    )
    monkeypatch.setattr(
        "mercury_runs.algorithms.cscg.evaluate.exact_valid_sensorimotor_transitions",
        lambda **kwargs: {(0, 0, 1)},
    )

    class _PerfectDecodeCHMM:
        def __init__(self, **kwargs) -> None:
            self.T = np.zeros((1, 2, 2), dtype=np.float32)

        def decode(self, x, a):
            return np.array([0.0, 0.0], dtype=np.float64), np.array([0, 1], dtype=np.int64)

    class _PerfectDecodeModule:
        CHMM = _PerfectDecodeCHMM

    metrics = compute_cscg_paper_precision_metrics_from_model(
        model_npz={"T": np.array([[[0.0, 1.0], [0.0, 0.0]]], dtype=np.float32)},
        n_clones=np.array([1, 1], dtype=np.int64),
        config=_base_config(tmp_path),
        obs_uniques=np.array([[0, 0], [1, 0]], dtype=np.int64),
        action_uniques=np.array([[1]], dtype=np.int64),
        chmm_mod=_PerfectDecodeModule(),
    )

    assert metrics["latent_precision"] == 1.0
    assert metrics["latent_purity"] == 100.0
    assert metrics["latent_mean_total_variation"] == pytest.approx(0.0)
    assert metrics["latent_edge_precision"] == pytest.approx(1.0)
    assert metrics["latent_edge_recall"] == pytest.approx(1.0)
    assert metrics["latent_edge_f1"] == pytest.approx(1.0)
    assert metrics["latent_action_conditioned_mean_total_variation"] == pytest.approx(0.0)
    assert metrics["latent_action_conditioned_edge_precision"] == pytest.approx(1.0)
    assert metrics["latent_action_conditioned_edge_recall"] == pytest.approx(1.0)
    assert metrics["latent_action_conditioned_edge_f1"] == pytest.approx(1.0)
    assert metrics["latent_sensorimotor_link_error"] == pytest.approx((0.0 + 1e-4) / (1.0 + 1e-4))


def test_cscg_paper_precision_passes_valid_trajectory_flag(monkeypatch, tmp_path: Path) -> None:
    captured: dict = {}

    def _fake_walks(**kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr("mercury_runs.algorithms.cscg.evaluate.generate_random_start_walks", _fake_walks)
    monkeypatch.setattr(
        "mercury_runs.algorithms.cscg.evaluate.exact_cartesian_reference_positions",
        lambda level_index: np.array([[0, 0]], dtype=np.int64),
    )
    monkeypatch.setattr(
        "mercury_runs.algorithms.cscg.evaluate.exact_valid_sensorimotor_transitions",
        lambda **kwargs: set(),
    )

    compute_cscg_paper_precision_metrics_from_model(
        model_npz={"T": np.ones((1, 1, 1), dtype=np.float32)},
        n_clones=np.array([1], dtype=np.int64),
        config=_base_config(tmp_path).model_copy(
            update={"valid_trajectories_only": True, "paper_precision_num_walks": 1, "structure_metrics_enabled": False}
        ),
        obs_uniques=np.array([[0]], dtype=np.int64),
        action_uniques=np.array([[1]], dtype=np.int64),
        chmm_mod=_DecodeModule(),
    )

    assert captured["valid_trajectories_only"] is True
