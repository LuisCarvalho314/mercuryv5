from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.mercury_runs.interfaces.cli import parse_arguments
from src.mercury_runs.application.run_service import _baseline_dataset_selection
from src.mercury_runs.domain.models import RunConfigModel, RunStatusModel, SubsystemStatus
from src.mercury_runs.infrastructure.runtime import atomic_write_json, stable_hash
from src.mercury_runs.schemas_results import ResultBundleMeta, SourceDatasetRef
from src.mercury_runs.infrastructure.storage import (
    artifact_layout,
    build_run_payload,
    collect_run_artifacts,
    resolve_run_identity,
    validate_reusable_run,
)


def _run_payload(*, with_pocml: bool = False, with_cscg: bool = False) -> dict:
    return {
        "method": "mercury",
        "level": 13,
        "sensor": "cardinal distance",
        "sensor_range": 1,
        "seed": 0,
        "rand_prob": 0.3,
        "num_steps": 100,
        "valid_trajectories_only": False,
        "mercury_valid_trajectories_only": False,
        "mercury_split_sensory_raw_latent_valid": False,
        "memory_length": 10,
        "window_length": 20,
        "evaluation": {
            "paper_precision": {
                "enabled": False,
                "mode": "off",
                "eval_interval": 1,
                "num_points": None,
                "num_walks": 20,
                "walk_length": 1000,
                "structure_metrics": {
                    "enabled": True,
                    "epsilon": 1e-6,
                    "scope": "exact_level",
                    "ground_truth_source": "empirical_walks",
                },
            },
            "computational": {"enabled": False, "profile": "full"},
        },
        "sensory": {},
        "latent": {},
        "action_map": {},
        "baselines": {
            "pocml": {"enabled": with_pocml, "valid_trajectories_only": False},
            "cscg": {"enabled": with_cscg, "valid_trajectories_only": False},
        },
    }


def _write_valid_run(tmp_path: Path, run_id: str, payload: dict) -> Path:
    run_root = tmp_path / run_id
    run_root.mkdir(parents=True, exist_ok=True)
    layout = artifact_layout(run_root=run_root, run_id=run_id, run_payload=payload)
    layout.mercury_states_parquet.parent.mkdir(parents=True, exist_ok=True)
    layout.mercury_states_parquet.write_text("states", encoding="utf-8")
    layout.mercury_paper_precision_json.parent.mkdir(parents=True, exist_ok=True)
    layout.mercury_paper_precision_json.write_text("{}", encoding="utf-8")
    layout.mercury_computational_json.write_text("{}", encoding="utf-8")
    layout.mercury_latent_graph_npz.write_text("graph", encoding="utf-8")
    layout.mercury_internal_graph_png.parent.mkdir(parents=True, exist_ok=True)
    layout.mercury_internal_graph_png.write_text("plot", encoding="utf-8")
    layout.internal_graphs_comparison_png.write_text("plot", encoding="utf-8")
    if payload["baselines"]["pocml"]["enabled"]:
        layout.pocml_states_parquet.parent.mkdir(parents=True, exist_ok=True)
        layout.pocml_states_parquet.write_text("pocml", encoding="utf-8")
        layout.pocml_paper_precision_json.parent.mkdir(parents=True, exist_ok=True)
        layout.pocml_paper_precision_json.write_text("{}", encoding="utf-8")
        layout.pocml_computational_json.write_text("{}", encoding="utf-8")
        layout.pocml_embeddings_npz.write_text("emb", encoding="utf-8")
        layout.pocml_model_checkpoint_pt.write_text("model", encoding="utf-8")
        layout.pocml_internal_graph_png.write_text("plot", encoding="utf-8")
        layout.comparison_summary_json.write_text("{}", encoding="utf-8")
    if payload["baselines"]["cscg"]["enabled"]:
        layout.cscg_states_parquet.parent.mkdir(parents=True, exist_ok=True)
        layout.cscg_states_parquet.write_text("cscg", encoding="utf-8")
        layout.cscg_paper_precision_json.parent.mkdir(parents=True, exist_ok=True)
        layout.cscg_paper_precision_json.write_text("{}", encoding="utf-8")
        layout.cscg_computational_json.write_text("{}", encoding="utf-8")
        layout.cscg_model_npz.write_text("model", encoding="utf-8")
        layout.cscg_internal_graph_png.write_text("plot", encoding="utf-8")
        layout.comparison_summary_json.write_text("{}", encoding="utf-8")

    run_config = RunConfigModel(
        run_id=run_id,
        requested_run_id=run_id,
        saved_at_utc="2026-03-13T10:00:00Z",
        cli_args={},
        run_payload=payload,
        execution={},
        paths={"run_root": str(run_root), "bundles_root": str(run_root / "bundles"), "metrics_root": str(run_root / "metrics")},
    )
    run_status = RunStatusModel(
        run_id=run_id,
        requested_run_id=run_id,
        status="completed",
        started_at_utc="2026-03-13T10:00:00Z",
        finished_at_utc="2026-03-13T10:01:00Z",
        study={"study_root": str(tmp_path), "study_name": "default"},
        config=payload,
        execution={},
        paths={"run_root": str(run_root), "bundles_root": str(run_root / "bundles"), "metrics_root": str(run_root / "metrics")},
        subsystems={
            "mercury": SubsystemStatus(
                name="mercury",
                status="completed",
                artifacts={
                    "states_parquet": str(layout.mercury_states_parquet),
                    "latent_graph_npz": str(layout.mercury_latent_graph_npz),
                },
            ),
            "pocml": SubsystemStatus(
                name="pocml",
                status="completed" if payload["baselines"]["pocml"]["enabled"] else "skipped",
                artifacts={
                    "states_parquet": str(layout.pocml_states_parquet),
                    "embeddings_npz": str(layout.pocml_embeddings_npz),
                } if payload["baselines"]["pocml"]["enabled"] else {},
            ),
            "cscg": SubsystemStatus(
                name="cscg",
                status="completed" if payload["baselines"]["cscg"]["enabled"] else "skipped",
                artifacts={
                    "states_parquet": str(layout.cscg_states_parquet),
                    "model_npz": str(layout.cscg_model_npz),
                } if payload["baselines"]["cscg"]["enabled"] else {},
            ),
        },
        artifacts=collect_run_artifacts(layout=layout, run_payload=payload),
    )
    atomic_write_json(layout.run_config_json, run_config.model_dump(mode="json"))
    atomic_write_json(layout.run_status_json, run_status.model_dump(mode="json"))
    return run_root


def test_validate_reusable_run_accepts_baseline_metrics_artifacts(tmp_path: Path) -> None:
    payload = _run_payload(with_pocml=True)
    run_root = _write_valid_run(tmp_path, "abc123", payload)

    reusable, reason = validate_reusable_run(run_root=run_root, run_payload=payload, run_id="abc123")

    assert reusable is True
    assert reason == "ok"


def test_resolve_run_identity_uses_fresh_directory_for_invalid_auto_run(tmp_path: Path) -> None:
    payload = _run_payload()
    requested_run_id = stable_hash(payload)
    study_dir = tmp_path / "default"
    study_dir.mkdir(parents=True, exist_ok=True)
    existing = study_dir / requested_run_id
    existing.mkdir()
    (existing / "run_status.json").write_text("{}", encoding="utf-8")

    args = argparse.Namespace(
        run_id=None,
        study_root=str(tmp_path),
        study_name="default",
    )

    identity = resolve_run_identity(args=args, run_payload=payload, reuse_existing_run=True)

    assert identity.requested_run_id == requested_run_id
    assert identity.run_id.startswith(f"{requested_run_id}-")
    assert identity.run_root.name == identity.run_id
    assert identity.is_reused is False


def test_collect_run_artifacts_includes_baseline_precision_artifacts(tmp_path: Path) -> None:
    payload = _run_payload(with_pocml=True, with_cscg=True)
    run_root = tmp_path / "run"
    layout = artifact_layout(run_root=run_root, run_id="run", run_payload=payload)
    layout.run_config_json.parent.mkdir(parents=True, exist_ok=True)
    layout.run_config_json.write_text("{}", encoding="utf-8")
    layout.run_status_json.write_text("{}", encoding="utf-8")
    layout.mercury_states_parquet.parent.mkdir(parents=True, exist_ok=True)
    layout.mercury_states_parquet.write_text("states", encoding="utf-8")
    layout.mercury_paper_precision_json.parent.mkdir(parents=True, exist_ok=True)
    layout.mercury_paper_precision_json.write_text("{}", encoding="utf-8")
    layout.mercury_computational_json.write_text("{}", encoding="utf-8")
    layout.mercury_latent_graph_npz.write_text("graph", encoding="utf-8")
    layout.mercury_internal_graph_png.parent.mkdir(parents=True, exist_ok=True)
    layout.mercury_internal_graph_png.write_text("plot", encoding="utf-8")
    layout.internal_graphs_comparison_png.write_text("plot", encoding="utf-8")
    layout.pocml_states_parquet.parent.mkdir(parents=True, exist_ok=True)
    layout.pocml_states_parquet.write_text("pocml", encoding="utf-8")
    layout.pocml_paper_precision_json.parent.mkdir(parents=True, exist_ok=True)
    layout.pocml_paper_precision_json.write_text("{}", encoding="utf-8")
    layout.pocml_computational_json.write_text("{}", encoding="utf-8")
    layout.pocml_embeddings_npz.write_text("emb", encoding="utf-8")
    layout.pocml_model_checkpoint_pt.write_text("model", encoding="utf-8")
    layout.pocml_internal_graph_png.write_text("plot", encoding="utf-8")
    layout.cscg_states_parquet.parent.mkdir(parents=True, exist_ok=True)
    layout.cscg_states_parquet.write_text("cscg", encoding="utf-8")
    layout.cscg_paper_precision_json.parent.mkdir(parents=True, exist_ok=True)
    layout.cscg_paper_precision_json.write_text("{}", encoding="utf-8")
    layout.cscg_computational_json.write_text("{}", encoding="utf-8")
    layout.cscg_model_npz.write_text("model", encoding="utf-8")
    layout.cscg_internal_graph_png.write_text("plot", encoding="utf-8")
    layout.comparison_summary_json.write_text("{}", encoding="utf-8")

    artifacts = collect_run_artifacts(layout=layout, run_payload=payload)

    assert artifacts["pocml_paper_precision_json"] == str(layout.pocml_paper_precision_json)
    assert artifacts["pocml_computational_json"] == str(layout.pocml_computational_json)
    assert artifacts["cscg_paper_precision_json"] == str(layout.cscg_paper_precision_json)
    assert artifacts["cscg_computational_json"] == str(layout.cscg_computational_json)
    assert artifacts["mercury_paper_precision_json"] == str(layout.mercury_paper_precision_json)
    assert artifacts["mercury_computational_json"] == str(layout.mercury_computational_json)
    assert artifacts["mercury_latent_graph_npz"] == str(layout.mercury_latent_graph_npz)
    assert artifacts["mercury_internal_graph_png"] == str(layout.mercury_internal_graph_png)
    assert artifacts["internal_graphs_comparison_png"] == str(layout.internal_graphs_comparison_png)
    assert artifacts["pocml_embeddings_npz"] == str(layout.pocml_embeddings_npz)
    assert artifacts["pocml_model_checkpoint_pt"] == str(layout.pocml_model_checkpoint_pt)
    assert artifacts["pocml_internal_graph_png"] == str(layout.pocml_internal_graph_png)
    assert artifacts["cscg_model_npz"] == str(layout.cscg_model_npz)
    assert artifacts["cscg_internal_graph_png"] == str(layout.cscg_internal_graph_png)


def test_baseline_dataset_selection_uses_mercury_source_dataset_run_id(monkeypatch, tmp_path: Path) -> None:
    meta = ResultBundleMeta(
        run_id="run-1",
        timestamp_utc="2026-03-17T12:00:00Z",
        source=SourceDatasetRef(
            level=20,
            sensor="cardinal distance",
            sensor_range=1,
            select="run_id",
            dataset_run_id="level20_cardinal_distance_seed0_steps60000",
            dataset_parquet_name="dataset.parquet",
        ),
        sensory_params={},
        latent_params={},
        action_map_params={},
    )
    monkeypatch.setattr("src.mercury_runs.application.run_service.read_bundle_meta", lambda path: meta)

    dataset_select, dataset_run_id = _baseline_dataset_selection(tmp_path / "states.parquet")

    assert dataset_select == "run_id"
    assert dataset_run_id == "level20_cardinal_distance_seed0_steps60000"


def test_build_run_payload_preserves_paper_precision_walk_settings() -> None:
    args = argparse.Namespace(
        level=13,
        sensor="cardinal distance",
        sensor_range=1,
        seed=0,
        rand_prob=0.3,
        num_steps=100,
        valid_trajectories_only=True,
        mercury_valid_trajectories_only=False,
        mercury_split_sensory_raw_latent_valid=True,
        memory_length=10,
        window_length=20,
        paper_precision=False,
        paper_precision_mode="final",
        paper_precision_eval_interval=3,
        paper_precision_num_points=None,
        paper_precision_num_walks=7,
        paper_precision_walk_length=250,
        structure_metrics=False,
        structure_metrics_epsilon=1e-4,
        structure_metrics_scope="walk_local",
        structure_metrics_ground_truth_source="maze_topology",
        computational_eval=True,
        computational_eval_profile="full",
        sensory_allow_self_loops=False,
        activation_threshold=0.95,
        topological_neighbourhood_threshold=0.6,
        sensory_max_neurons=50,
        sensory_weighting=0.8,
        winning_node_lr=0.55,
        topological_neighbourhood_lr=0.9,
        sensory_action_lr=0.5,
        global_context_lr=0.9,
        sensory_max_age=18,
        sensory_gaussian_shape=2,
        latent_allow_self_loops=True,
        latent_max_neurons=300,
        latent_action_lr=0.1,
        latent_gaussian_shape=2,
        latent_max_age=18,
        ambiguity_threshold=10,
        ambiguity_decay=0.99,
        trace_decay=0.99,
        lambda_trace=0.05,
        weight_memory=0.4,
        weight_undirected=0.2,
        weight_base=0.2,
        weight_action=0.2,
        memory_replay=True,
        memory_disambiguation=True,
        am_n_codebook=4,
        am_lr=0.5,
        am_sigma=0.0,
        am_key=0,
        baseline_pocml=False,
        pocml_valid_trajectories_only=False,
        baseline_cscg=False,
        cscg_valid_trajectories_only=False,
        baseline_all=False,
        pocml_epochs=30,
        pocml_trajectory_length=15,
        pocml_max_trajectories=None,
        pocml_state_dim=50,
        pocml_random_feature_dim=500,
        pocml_n_states=None,
        pocml_precision_capacities=None,
        pocml_alpha=2.0,
        pocml_lr_q=0.1,
        pocml_lr_v=0.08,
        pocml_lr_all=0.05,
        pocml_lr_m=1.0,
        pocml_reg_m=0.0,
        pocml_max_iter_m=1,
        pocml_eps_m=1e-3,
        pocml_batch_size=64,
        pocml_memory_bias=True,
        pocml_use_ground_truth_node_ids=False,
        cscg_clones_per_obs=10,
        cscg_n_iter=1000,
        cscg_pseudocount=2e-3,
        cscg_term_early=True,
        cscg_train_algorithm="em",
        cscg_training_mode="full",
        cscg_batch_size=1024,
        cscg_online_lambda=1.0,
        cscg_seed=None,
        method=None,
    )

    payload = build_run_payload(args)

    assert payload["method"] == "mercury"
    assert payload["evaluation"]["paper_precision"] == {
        "enabled": True,
        "mode": "final",
        "eval_interval": 3,
        "num_points": None,
        "num_walks": 7,
        "walk_length": 250,
        "structure_metrics": {
            "enabled": False,
            "epsilon": 1e-4,
            "scope": "walk_local",
            "ground_truth_source": "maze_topology",
        },
    }
    assert payload["evaluation"]["computational"] == {"enabled": True, "profile": "full"}
    assert payload["valid_trajectories_only"] is True
    assert payload["mercury_valid_trajectories_only"] is False
    assert payload["mercury_split_sensory_raw_latent_valid"] is True
    assert payload["baselines"]["pocml"]["valid_trajectories_only"] is False
    assert payload["baselines"]["cscg"]["valid_trajectories_only"] is False
    assert payload["latent"]["weight_memory"] == 0.4
    assert payload["latent"]["weight_undirected"] == 0.2
    assert payload["latent"]["weight_base"] == 0.2
    assert payload["latent"]["weight_action"] == 0.2
    assert payload["latent"]["ambiguity_decay"] == 0.99
    assert payload["latent"]["lambda_trace"] == 0.05
    assert "mixture_alpha" not in payload["latent"]
    assert "mixture_beta" not in payload["latent"]


def test_validate_reusable_run_requires_computational_artifact_when_enabled(tmp_path: Path) -> None:
    payload = _run_payload()
    payload["evaluation"]["computational"]["enabled"] = True
    run_root = _write_valid_run(tmp_path, "comp123", payload)
    layout = artifact_layout(run_root=run_root, run_id="comp123", run_payload=payload)
    layout.mercury_computational_json.unlink()

    reusable, reason = validate_reusable_run(run_root=run_root, run_payload=payload, run_id="comp123")

    assert reusable is False
    assert reason == "missing Mercury computational artifact"


def test_parse_arguments_accepts_valid_trajectories_only_flag(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["main.py", "--level", "13", "--valid_trajectories_only"])

    args = parse_arguments()

    assert args.level == 13
    assert args.valid_trajectories_only is True


def test_parse_arguments_accepts_algorithm_specific_valid_trajectory_flags(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "main.py",
            "--level",
            "13",
            "--mercury_valid_trajectories_only",
            "True",
            "--mercury_split_sensory_raw_latent_valid",
            "True",
            "--pocml_valid_trajectories_only",
            "False",
            "--cscg_valid_trajectories_only",
            "False",
        ],
    )

    args = parse_arguments()

    assert args.mercury_valid_trajectories_only is True
    assert args.mercury_split_sensory_raw_latent_valid is True
    assert args.pocml_valid_trajectories_only is False
    assert args.cscg_valid_trajectories_only is False


def test_parse_arguments_accepts_structure_metric_flags(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "main.py",
            "--level",
            "13",
            "--structure_metrics",
            "False",
            "--structure_metrics_epsilon",
            "0.0001",
            "--structure_metrics_scope",
            "walk_local",
            "--structure_metrics_ground_truth_source",
            "maze_topology",
        ],
    )

    args = parse_arguments()

    assert args.structure_metrics is False
    assert args.structure_metrics_epsilon == 1e-4
    assert args.structure_metrics_scope == "walk_local"
    assert args.structure_metrics_ground_truth_source == "maze_topology"


def test_parse_arguments_accepts_lambda_trace(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "main.py",
            "--level",
            "13",
            "--lambda_trace",
            "0.125",
        ],
    )

    args = parse_arguments()

    assert args.lambda_trace == 0.125
