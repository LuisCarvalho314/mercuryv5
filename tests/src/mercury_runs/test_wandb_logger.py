import json
from pathlib import Path

import polars as pl

from mercury_runs.infrastructure.wandb_logger import (
    _log_paper_precision_to_wandb,
    _log_parquet_history_to_wandb,
    _log_states_history_to_wandb,
)


class _FakeRun:
    def __init__(self) -> None:
        self.logged: list[dict] = []
        self.defined_metrics: list[tuple[tuple, dict]] = []
        self.summary: dict = {}

    def define_metric(self, *args, **kwargs) -> None:
        self.defined_metrics.append((args, kwargs))

    def log(self, payload: dict) -> None:
        self.logged.append(payload)


def test_log_states_history_to_wandb_logs_metric_columns_only(tmp_path: Path) -> None:
    states_path = tmp_path / "states.parquet"
    pl.DataFrame(
        {
            "latent_bmu": [0, 1],
            "cartesian_proxy_bmu": [3, 4],
            "bmu_memory_depth": [0, 2],
            "node_age_distribution": [[0, 1], [1, 2]],
        }
    ).write_parquet(states_path)
    run = _FakeRun()

    _log_states_history_to_wandb(run, states_path, namespace="mercury/states")

    assert run.logged == [
        {"mercury/states/step": 1, "mercury/states/bmu_memory_depth": 0},
        {"mercury/states/step": 2, "mercury/states/bmu_memory_depth": 2},
    ]
    assert (("mercury/states/step",), {}) in run.defined_metrics
    assert (("mercury/states/*",), {"step_metric": "mercury/states/step"}) in run.defined_metrics


def test_log_parquet_history_to_wandb_uses_existing_step_column(tmp_path: Path) -> None:
    attribution_path = tmp_path / "attribution.parquet"
    pl.DataFrame(
        {
            "step": [5, 10],
            "latent_bmu": [0, 1],
            "selected_memory_drive_weighted": [0.25, 0.75],
        }
    ).write_parquet(attribution_path)
    run = _FakeRun()

    _log_parquet_history_to_wandb(
        run,
        attribution_path,
        namespace="mercury/attribution",
        exclude_columns=frozenset({"latent_bmu"}),
    )

    assert run.logged == [
        {
            "mercury/attribution/step": 5,
            "mercury/attribution/selected_memory_drive_weighted": 0.25,
        },
        {
            "mercury/attribution/step": 10,
            "mercury/attribution/selected_memory_drive_weighted": 0.75,
        },
    ]


def test_log_paper_precision_to_wandb_logs_final_metrics_in_paper_precision_namespace(tmp_path: Path) -> None:
    paper_path = tmp_path / "run_paper_precision.json"
    paper_path.write_text(
        json.dumps(
            {
                "metrics": {
                    "latent_precision": 0.5,
                    "latent_recall": 0.4,
                    "latent_purity": 0.75,
                    "latent_edge_precision": 0.8,
                    "latent_edge_recall": 0.6,
                    "latent_edge_f1": 0.685,
                    "latent_mean_total_variation": 0.2,
                    "latent_precision_reason": "ok",
                    "latent_action_conditioned_edge_precision": None,
                    "latent_cooccurrence_matrix": [[1, 0], [0, 1]],
                },
                "protocol": {
                    "num_walks": 2,
                    "walk_length": 50,
                    "mode": "final",
                },
            }
        ),
        encoding="utf-8",
    )
    run = _FakeRun()

    _log_paper_precision_to_wandb(run, paper_path, subsystem_name="mercury", final_observed_samples=10000)

    assert run.summary["mercury/paper_precision/latent_precision"] == 0.5
    assert run.summary["mercury/paper_precision/latent_purity"] == 0.75
    assert run.summary["mercury/paper_precision/latent_precision_reason"] == "ok"
    assert run.summary["mercury/paper_precision/latent_action_conditioned_edge_precision"] == "null"
    assert run.summary["mercury/paper_precision/latent_cooccurrence_matrix_json"] == "[[1, 0], [0, 1]]"
    assert run.summary["mercury/paper_precision/mode"] == "final"
    assert run.logged == [
        {
            "mercury/paper_precision/observed_samples": 10000,
            "mercury/paper_precision/latent_precision": 0.5,
            "mercury/paper_precision/latent_recall": 0.4,
            "mercury/paper_precision/latent_edge_precision": 0.8,
            "mercury/paper_precision/latent_edge_recall": 0.6,
            "mercury/paper_precision/latent_edge_f1": 0.685,
            "mercury/paper_precision/latent_edge_fscore": 0.685,
            "mercury/paper_precision/latent_mean_total_variation": 0.2,
        }
    ]


def test_log_paper_precision_to_wandb_logs_per_iteration_history(tmp_path: Path) -> None:
    paper_path = tmp_path / "run_paper_precision.json"
    paper_path.write_text(
        json.dumps(
            {
                "metrics": {
                    "latent_precision": 0.5,
                    "latent_recall": 0.4,
                    "latent_cooccurrence_matrix": [[1]],
                },
                "protocol": {
                    "num_walks": 2,
                    "walk_length": 50,
                },
                "history": [
                    {
                        "step": 10,
                        "observed_samples": 10,
                        "latent_precision": 0.25,
                        "latent_recall": 0.2,
                        "latent_edge_precision": 0.6,
                        "latent_edge_recall": 0.5,
                        "latent_edge_f1": 0.545,
                        "latent_mean_total_variation": 0.3,
                        "latent_action_conditioned_edge_precision": 0.7,
                        "latent_action_conditioned_edge_recall": 0.4,
                        "latent_action_conditioned_edge_f1": 0.509,
                        "latent_action_conditioned_mean_total_variation": 0.35,
                        "latent_precision_reason": "ok",
                        "latent_cooccurrence_matrix": [[1]],
                    },
                    {
                        "step": 20,
                        "observed_samples": 20,
                        "latent_precision": 0.5,
                        "latent_recall": 0.4,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    run = _FakeRun()

    _log_paper_precision_to_wandb(run, paper_path, subsystem_name="mercury")

    assert (("mercury/paper_precision/observed_samples",), {}) in run.defined_metrics
    assert (
        ("mercury/paper_precision/*",),
        {"step_metric": "mercury/paper_precision/observed_samples"},
    ) in run.defined_metrics
    assert run.summary["mercury/paper_precision/latent_precision"] == 0.5
    assert run.summary["mercury/paper_precision/latent_recall"] == 0.4
    assert run.summary["mercury/paper_precision/latent_cooccurrence_matrix_json"] == "[[1]]"
    assert run.logged == [
        {
            "mercury/paper_precision/observed_samples": 10,
            "mercury/paper_precision/latent_precision": 0.25,
            "mercury/paper_precision/latent_recall": 0.2,
            "mercury/paper_precision/latent_edge_precision": 0.6,
            "mercury/paper_precision/latent_edge_recall": 0.5,
            "mercury/paper_precision/latent_edge_f1": 0.545,
            "mercury/paper_precision/latent_edge_fscore": 0.545,
            "mercury/paper_precision/latent_mean_total_variation": 0.3,
            "mercury/paper_precision/latent_action_conditioned_edge_precision": 0.7,
            "mercury/paper_precision/latent_action_conditioned_edge_recall": 0.4,
            "mercury/paper_precision/latent_action_conditioned_edge_f1": 0.509,
            "mercury/paper_precision/latent_action_conditioned_edge_fscore": 0.509,
            "mercury/paper_precision/latent_action_conditioned_mean_total_variation": 0.35,
        },
        {
            "mercury/paper_precision/observed_samples": 20,
            "mercury/paper_precision/latent_precision": 0.5,
            "mercury/paper_precision/latent_recall": 0.4,
        },
    ]
