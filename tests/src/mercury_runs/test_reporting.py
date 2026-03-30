from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from mercury_runs.infrastructure.paper_precision import summarize_paper_precision_artifact
from mercury_runs.infrastructure.paper_precision import (
    aggregate_aligned_weighted_adjacency,
    build_weighted_adjacency_from_walks,
    collapse_action_transition_tensor_to_adjacency,
    compute_alignment,
    compute_contingency_matrix,
    compute_edge_metrics,
    compute_mean_tv,
    compute_occupancy,
    compute_weighted_structure_metrics,
    row_normalize,
)
from mercury_runs.infrastructure.reporting import (
    rollout_n_step_observation_prediction_metrics,
    summarize_precision_metrics,
    summarize_mercury_native_metrics,
    write_method_comparison_report,
    write_study_summary,
)
from mercury_runs.infrastructure.paper_precision import cartesian_state_ids
from mercury_runs.domain.models import RunConfigModel, RunStatusModel, SubsystemStatus
from mercury_runs.schemas_results import ResultBundleMeta, SourceDatasetRef


def test_rollout_n_step_observation_prediction_metrics_uses_unified_accuracy_key() -> None:
    metrics = rollout_n_step_observation_prediction_metrics(
        state_series=np.array([0, 1, 1], dtype=np.int32),
        gt_series=np.array([2, 3, 3], dtype=np.int32),
        action_series=np.array([0, 1, 1], dtype=np.int32),
        horizons=[1, 3, 5],
    )

    assert metrics["n_step_observation_prediction_accuracy"] == {"n1": 1.0, "n3": 0.0, "n5": 0.0}
    assert metrics["trajectory_log_likelihood_sum"] == pytest.approx(-2.9999998063819027e-09)
    assert metrics["trajectory_log_likelihood_mean"] == pytest.approx(-1.4999999031909513e-09)
    assert metrics["steps"] == 3


def test_summarize_mercury_native_metrics_zero_case_uses_unified_accuracy_key(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        "mercury_runs.infrastructure.reporting.resolve_dataset_parquet_from_states",
        lambda states_path: tmp_path / "dataset.parquet",
    )
    monkeypatch.setattr(
        "mercury_runs.infrastructure.reporting.load_action_series",
        lambda dataset_path: np.array([0], dtype=np.int32),
    )
    monkeypatch.setattr(
        pl,
        "read_parquet",
        lambda path: pl.DataFrame(
            {
                "ground_truth_bmu": [1],
                "sensory_bmu": [1],
                "latent_bmu": [1],
            }
        ),
    )

    summary = summarize_mercury_native_metrics(tmp_path / "states.parquet")

    assert summary == {
        "sensory": {
            "n_step_observation_prediction_accuracy": {"n1": 0.0, "n3": 0.0, "n5": 0.0},
            "trajectory_log_likelihood_sum": 0.0,
            "trajectory_log_likelihood_mean": 0.0,
            "steps": 1,
        },
        "latent": {
            "n_step_observation_prediction_accuracy": {"n1": 0.0, "n3": 0.0, "n5": 0.0},
            "trajectory_log_likelihood_sum": 0.0,
            "trajectory_log_likelihood_mean": 0.0,
            "steps": 1,
        },
    }


def test_summarize_mercury_native_metrics_reads_cartesian_proxy_bmu(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        "mercury_runs.infrastructure.reporting.resolve_dataset_parquet_from_states",
        lambda states_path: tmp_path / "dataset.parquet",
    )
    monkeypatch.setattr(
        "mercury_runs.infrastructure.reporting.load_action_series",
        lambda dataset_path: np.array([0, 1], dtype=np.int32),
    )
    monkeypatch.setattr(
        pl,
        "read_parquet",
        lambda path: pl.DataFrame(
            {
                "cartesian_proxy_bmu": [1, 2],
                "sensory_bmu": [1, 2],
                "latent_bmu": [1, 2],
            }
        ),
    )

    summary = summarize_mercury_native_metrics(tmp_path / "states.parquet")

    assert summary["sensory"]["steps"] == 2
    assert summary["latent"]["steps"] == 2


def test_summarize_precision_metrics_supports_single_precision_column(tmp_path: Path) -> None:
    metrics_path = tmp_path / "precision.parquet"
    pl.DataFrame(
        {
            "iteration": [1, 2],
            "capacity_precision": [0.25, 0.5],
            "latent_node_count": [4, 4],
        }
    ).write_parquet(metrics_path)

    summary = summarize_precision_metrics(metrics_path)

    assert summary == {
        "rows": 2,
        "capacity_precision_mean": 0.375,
        "capacity_precision_min": 0.25,
        "capacity_precision_max": 0.5,
        "latent_node_count_mean": 4.0,
        "latent_node_count_min": 4,
        "latent_node_count_max": 4,
    }


def test_summarize_paper_precision_artifact_returns_scalar_metrics(tmp_path: Path) -> None:
    metrics_path = tmp_path / "paper_precision.json"
    metrics_path.write_text(
        (
            "{"
            "\"method\": \"cscg\", "
            "\"protocol\": {\"num_walks\": 100, \"walk_length\": 10000}, "
            "\"metrics\": {"
            "\"latent_precision\": 0.75, "
            "\"evaluation_steps\": 1000000, "
            "\"inferred_state_count\": 12, "
            "\"ground_truth_state_count\": 9, "
            "\"cooccurrence_matrix\": [[1, 2]]"
            "}"
            "}"
        ),
        encoding="utf-8",
    )

    summary = summarize_paper_precision_artifact(metrics_path)

    assert summary == {
        "latent_precision": 0.75,
        "evaluation_steps": 1000000,
        "inferred_state_count": 12,
        "ground_truth_state_count": 9,
        "num_walks": 100,
        "walk_length": 10000,
    }


def test_summarize_paper_precision_artifact_supports_multi_capacity_payload(tmp_path: Path) -> None:
    metrics_path = tmp_path / "paper_precision_multi.json"
    metrics_path.write_text(
        (
            "{"
            "\"method\": \"pocml\", "
            "\"protocol\": {\"resolved_capacities\": [3, 5], \"primary_capacity\": 3}, "
            "\"metrics_by_capacity\": {"
            "\"3\": {\"capacity_precision\": 0.25, \"evaluation_steps\": 100}, "
            "\"5\": {\"capacity_precision\": 0.5, \"evaluation_steps\": 120}"
            "}"
            "}"
        ),
        encoding="utf-8",
    )

    summary = summarize_paper_precision_artifact(metrics_path)

    assert summary == {
        "resolved_capacities": [3, 5],
        "primary_capacity": 3,
        "metrics_by_capacity": {
            "3": {"capacity_precision": 0.25, "evaluation_steps": 100},
            "5": {"capacity_precision": 0.5, "evaluation_steps": 120},
        },
    }


def test_weighted_structure_helpers_match_reference_formula() -> None:
    decoded_states = np.array([0, 0, 1, 2], dtype=np.int64)
    true_states = np.array([1, 1, 0, 1], dtype=np.int64)
    W_hat = np.array(
        [
            [0.0, 2.0, 1.0],
            [3.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )

    contingency = compute_contingency_matrix(decoded_states, true_states, n_learned=3, n_true=2)
    alignment = compute_alignment(contingency)
    occupancy = compute_occupancy(decoded_states, n_learned=3)
    aligned = aggregate_aligned_weighted_adjacency(W_hat, alignment, occupancy, n_true=2)

    assert contingency.tolist() == [[0, 2], [1, 0], [0, 1]]
    assert alignment.tolist() == [1, 0, 1]
    assert occupancy.tolist() == [2.0, 1.0, 1.0]
    assert np.allclose(aligned, np.array([[0.0, 3.0], [4.0 / 3.0, 1.0]], dtype=np.float64))


def test_row_normalize_mean_tv_and_edge_metrics_follow_spec() -> None:
    W_true = np.array([[0.0, 0.0], [2.0, 2.0]], dtype=np.float64)
    W_aligned = np.array([[0.0, 0.0], [0.0, 4.0]], dtype=np.float64)

    T_true = row_normalize(W_true)
    T_aligned = row_normalize(W_aligned)

    assert np.allclose(T_true, np.array([[0.0, 0.0], [0.5, 0.5]], dtype=np.float64))
    assert np.allclose(T_aligned, np.array([[0.0, 0.0], [0.0, 1.0]], dtype=np.float64))
    assert compute_mean_tv(T_true, T_aligned) == pytest.approx(0.25)
    assert compute_edge_metrics(W_true, W_aligned) == {
        "edge_precision": pytest.approx(1.0),
        "edge_recall": pytest.approx(0.5),
        "edge_f1": pytest.approx(2.0 / 3.0),
    }


def test_weighted_structure_metrics_respect_walk_boundaries() -> None:
    walks = [
        np.array([0, 1], dtype=np.int64),
        np.array([1, 0], dtype=np.int64),
    ]

    adjacency = build_weighted_adjacency_from_walks(walks, n_states=2)

    assert np.allclose(adjacency, np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64))


def test_compute_weighted_structure_metrics_perfect_alignment() -> None:
    metrics = compute_weighted_structure_metrics(
        decoded_walks=[np.array([0, 1, 0], dtype=np.int64)],
        true_walks=[np.array([0, 1, 0], dtype=np.int64)],
        W_hat=np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64),
        n_true=2,
    )

    assert metrics == {
        "mean_total_variation": pytest.approx(0.0),
        "edge_precision": pytest.approx(1.0),
        "edge_recall": pytest.approx(1.0),
        "edge_f1": pytest.approx(1.0),
    }


def test_collapse_action_transition_tensor_to_adjacency_sums_over_actions() -> None:
    transition_tensor = np.array(
        [
            [[0.0, 1.0], [0.0, 0.0]],
            [[0.0, 2.0], [3.0, 0.0]],
        ],
        dtype=np.float64,
    )

    assert np.allclose(
        collapse_action_transition_tensor_to_adjacency(transition_tensor),
        np.array([[0.0, 3.0], [3.0, 0.0]], dtype=np.float64),
    )


def test_cartesian_state_ids_level20_corridor_has_five_states() -> None:
    positions = np.asarray([[1, 1], [1, 2], [1, 3], [1, 4], [1, 5], [1, 1]], dtype=np.int64)
    state_ids = cartesian_state_ids(positions)

    assert len(np.unique(state_ids)) == 5


def test_rollout_n_step_observation_prediction_metrics_uses_progress_when_requested(monkeypatch) -> None:
    calls: list[dict] = []

    monkeypatch.setattr(
        "mercury_runs.infrastructure.reporting.tqdm",
        lambda iterable, **kwargs: calls.append(kwargs) or iterable,
    )

    rollout_n_step_observation_prediction_metrics(
        state_series=np.array([0, 1, 1], dtype=np.int32),
        gt_series=np.array([2, 3, 3], dtype=np.int32),
        action_series=np.array([0, 1, 1], dtype=np.int32),
        horizons=[1],
        show_progress=True,
        progress_desc="CSCG Native Eval",
    )

    assert any(call.get("desc") == "CSCG Native Eval" for call in calls if "desc" in call)


def test_write_study_summary_uses_target_method_column(monkeypatch, tmp_path: Path) -> None:
    study_root = tmp_path / "studies"
    study_dir = study_root / "ablation"
    run_dir = study_dir / "run-1"
    run_dir.mkdir(parents=True)
    states_path = run_dir / "pocml_states.parquet"
    states_path.write_text("states", encoding="utf-8")
    paper_path = run_dir / "pocml_paper_precision.json"
    paper_path.write_text("{}", encoding="utf-8")
    computational_path = run_dir / "pocml_computational.json"
    computational_path.write_text(
        (
            "{"
            "\"metrics\": {"
            "\"train_wall_time_seconds\": 2.0, "
            "\"eval_wall_time_seconds\": 1.0, "
            "\"total_wall_time_seconds\": 3.0, "
            "\"train_steps_per_second\": 15.0, "
            "\"eval_steps_per_second\": 20.0, "
            "\"peak_rss_mb\": 128.0, "
            "\"deployable_artifact_bytes\": 1048576, "
            "\"total_artifact_bytes\": 2097152"
            "}"
            "}"
        ),
        encoding="utf-8",
    )

    run_status = RunStatusModel(
        run_id="run-1",
        requested_run_id="run-1",
        status="completed",
        started_at_utc="2026-03-25T16:00:00Z",
        finished_at_utc="2026-03-25T16:10:00Z",
        study={"study_root": str(study_root), "study_name": "ablation"},
        config={"method": "pocml"},
        execution={},
        paths={},
        subsystems={
            "mercury": SubsystemStatus(name="mercury", status="completed", artifacts={}),
            "pocml": SubsystemStatus(
                name="pocml",
                status="completed",
                artifacts={
                    "states_parquet": str(states_path),
                    "paper_precision_json": str(paper_path),
                    "computational_json": str(computational_path),
                },
                native_metrics={
                    "n_step_observation_prediction_accuracy": {"n1": 0.5, "n3": 0.0, "n5": 0.0},
                    "trajectory_log_likelihood_mean": -0.7,
                },
            ),
            "cscg": SubsystemStatus(name="cscg", status="skipped", artifacts={}),
        },
    )
    run_config = RunConfigModel(
        run_id="run-1",
        requested_run_id="run-1",
        saved_at_utc="2026-03-25T16:00:00Z",
        cli_args={},
        run_payload={
            "method": "pocml",
            "level": 21,
            "sensor": "cardinal distance",
            "sensor_range": 1,
            "seed": 0,
            "rand_prob": 0.3,
            "num_steps": 100,
            "window_length": 100,
            "memory_length": 10,
            "baselines": {
                "pocml": {"enabled": True, "epochs": 30, "trajectory_length": 15, "alpha": 2.0, "batch_size": 64},
                "cscg": {"enabled": False},
            },
            "latent": {
                "lambda_trace": 0.05,
                "weight_memory": 0.4,
                "weight_undirected": 0.2,
                "weight_base": 0.2,
                "weight_action": 0.2,
                "memory_replay": True,
                "memory_disambiguation": True,
            },
            "action_map": {"n_codebook": 4, "lr": 0.5, "sigma": 0.0, "key": 0},
        },
        execution={},
        paths={},
    )
    meta = ResultBundleMeta(
        run_id="run-1",
        timestamp_utc="2026-03-25T16:00:00Z",
        source=SourceDatasetRef(
            level=21,
            sensor="cardinal distance",
            sensor_range=1,
            select="latest",
            dataset_run_id=None,
            dataset_parquet_name="dataset.parquet",
        ),
        sensory_params={},
        latent_params={},
        action_map_params={"n_codebook": 4, "lr": 0.5, "sigma": 0.0, "key": 0},
        memory_length=None,
        run_parameters={"run_payload": run_config.run_payload},
    )

    monkeypatch.setattr("mercury_runs.infrastructure.reporting.load_run_status_model", lambda path: run_status)
    monkeypatch.setattr("mercury_runs.infrastructure.reporting.load_run_config_model", lambda path: run_config)
    monkeypatch.setattr("mercury_runs.infrastructure.reporting.read_bundle_meta", lambda path: meta)
    monkeypatch.setattr("mercury_runs.infrastructure.reporting.summarize_paper_precision_artifact", lambda path: {"capacity_precision": 0.5})

    summary_path = write_study_summary(study_root=study_root, study_name="ablation")
    frame = pl.read_parquet(summary_path)

    assert frame.get_column("method").to_list() == ["pocml"]
    assert frame.get_column("target_n1_accuracy").to_list() == [0.5]
    assert frame.get_column("paper_capacity_precision").to_list() == [0.5]
    assert frame.get_column("comp_total_wall_time_seconds").to_list() == [3.0]
    assert frame.get_column("comp_precision_per_total_second").to_list() == [pytest.approx(1.0 / 6.0)]
    assert frame.get_column("lambda_trace").to_list() == [0.05]


def test_write_method_comparison_report_includes_computational_sections(monkeypatch, tmp_path: Path) -> None:
    run_root = tmp_path / "run-1"
    run_root.mkdir()
    mercury_states = run_root / "mercury_states.parquet"
    mercury_states.write_text("states", encoding="utf-8")
    mercury_computational = run_root / "mercury_computational.json"
    mercury_computational.write_text(
        (
            "{"
            "\"notes\": [\"artifact scope definition\"], "
            "\"metrics\": {"
            "\"train_wall_time_seconds\": 4.0, "
            "\"eval_wall_time_seconds\": 2.0, "
            "\"total_wall_time_seconds\": 6.0, "
            "\"peak_rss_mb\": 64.0, "
            "\"deployable_artifact_bytes\": 2048, "
            "\"total_artifact_bytes\": 4096"
            "}"
            "}"
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "mercury_runs.infrastructure.reporting.summarize_mercury_native_metrics",
        lambda path: {
            "sensory": {"n_step_observation_prediction_accuracy": {"n1": 0.25}, "trajectory_log_likelihood_mean": -1.0},
            "latent": {"n_step_observation_prediction_accuracy": {"n1": 0.5}, "trajectory_log_likelihood_mean": -0.5},
        },
    )

    report_path = write_method_comparison_report(
        run_root=run_root,
        run_id="run-1",
        run_payload={"level": 1, "sensor": "cartesian", "sensor_range": None, "baselines": {}},
        subsystems={
            "mercury": {
                "status": "completed",
                "artifacts": {
                    "states_parquet": str(mercury_states),
                    "computational_json": str(mercury_computational),
                },
            },
            "pocml": {"status": "skipped", "artifacts": {}},
            "cscg": {"status": "skipped", "artifacts": {}},
        },
    )

    payload = __import__("json").loads(report_path.read_text(encoding="utf-8"))
    assert payload["mercury_computational"]["raw"]["total_wall_time_seconds"] == 6.0
    assert payload["mercury_computational"]["derived"]["precision_per_total_second"] == pytest.approx(0.5 / 6.0)
