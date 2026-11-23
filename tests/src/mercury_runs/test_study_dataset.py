from __future__ import annotations

import json
from pathlib import Path
import sys

import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mercury_runs.analysis.study_dataset import load_mercury_study_dataset
from mercury_runs.save_results import write_bundle_parquet
from mercury_runs.schemas_results import ResultBundleMeta, SourceDatasetRef


def _bundle_meta(run_id: str, *, level: int, dataset_name: str = "dataset.parquet") -> ResultBundleMeta:
    return ResultBundleMeta(
        run_id=run_id,
        timestamp_utc="2026-04-07T12:00:00Z",
        source=SourceDatasetRef(
            level=level,
            sensor="cardinal distance",
            sensor_range=1,
            select="latest",
            dataset_parquet_name=dataset_name,
        ),
        sensory_params={"activation_threshold": 0.95, "sensory_weighting": 1.0},
        latent_params={
            "weight_memory": 1.0,
            "weight_action": 0.0,
            "weight_undirected": 0.0,
            "weight_base": 0.0,
            "ambiguity_threshold": 5,
            "ambiguity_decay": 0.95,
            "trace_decay": 0.99,
            "lambda_trace": 0.0,
            "memory_replay": True,
            "memory_disambiguation": True,
        },
        action_map_params={"n_codebook": 4, "lr": 0.5, "sigma": 0.0, "key": 0},
        memory_length=40,
        run_parameters={},
    )


def _write_run(study_root: Path, *, run_id: str, level: int, include_summary: bool = True) -> None:
    run_root = study_root / run_id
    bundle_root = run_root / "bundles" / "mercury" / f"level={level}" / "sensor=cardinal distance" / "range=1"
    bundle_root.mkdir(parents=True, exist_ok=True)

    states_path = write_bundle_parquet(
        output_dir=bundle_root,
        run_id=run_id,
        bundle_name="states",
        columns={
            "cartesian_proxy_bmu": [0, 1, 1],
            "sensory_bmu": [0, 1, 1],
            "latent_bmu": [0, 1, 2],
            "latent_node_count": [2, 2, 3],
        },
        meta=_bundle_meta(run_id, level=level),
    )
    attribution_path = write_bundle_parquet(
        output_dir=bundle_root,
        run_id=run_id,
        bundle_name="attribution",
        columns={
            "step": [1, 2, 3],
            "latent_bmu": [0, 1, 2],
            "latent_node_count": [2, 2, 3],
            "selected_memory_drive_share": [1.0, 0.8, 0.7],
        },
        meta=_bundle_meta(run_id, level=level),
    )
    paper_root = run_root / "metrics" / "mercury"
    paper_root.mkdir(parents=True, exist_ok=True)
    paper_path = paper_root / f"{run_id}_paper_precision.json"
    paper_path.write_text(
        json.dumps(
            {
                "method": "mercury",
                "history": [
                    {
                        "step": 1,
                        "observed_samples": 1,
                        "latent_precision": 0.9,
                        "sensory_precision": 0.8,
                        "latent_inferred_state_count": 2,
                        "latent_ground_truth_state_count": 3,
                        "sensory_inferred_state_count": 2,
                        "sensory_ground_truth_state_count": 3,
                        "latent_cooccurrence_matrix": [[1, 2]],
                        "latent_exact_state_cooccurrence_matrix": [[1, 0, 2], [0, 3, 1]],
                    },
                    {
                        "step": 2,
                        "observed_samples": 2,
                        "latent_precision": 0.95,
                        "sensory_precision": 0.85,
                        "latent_inferred_state_count": 3,
                        "latent_ground_truth_state_count": 3,
                        "sensory_inferred_state_count": 2,
                        "sensory_ground_truth_state_count": 3,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    latent_graph = bundle_root / f"{run_id}_latent_graph.npz"
    latent_graph.write_text("npz", encoding="utf-8")
    plots_root = run_root / "plots"
    plots_root.mkdir(parents=True, exist_ok=True)
    for name in (
        "mercury_internal_graph.png",
        "mercury_bmu_attribution_vs_training_step.png",
        "mercury_bmu_attribution_vs_edge_f1.png",
        "mercury_bmu_attribution_vs_node_ratio.png",
        "internal_graphs_comparison.png",
    ):
        (plots_root / name).write_text("png", encoding="utf-8")

    run_payload = {
        "method": "mercury",
        "level": level,
        "sensor": "cardinal distance",
        "sensor_range": 1,
        "seed": 7,
        "rand_prob": 0.3,
        "num_steps": 3,
        "window_length": 100,
        "memory_length": 40,
        "sensory": {"activation_threshold": 0.95, "sensory_weighting": 1.0},
        "latent": {
            "ambiguity_threshold": 5,
            "ambiguity_decay": 0.95,
            "trace_decay": 0.99,
            "lambda_trace": 0.0,
            "weight_memory": 1.0,
            "weight_undirected": 0.0,
            "weight_base": 0.0,
            "weight_action": 0.0,
            "memory_replay": True,
            "memory_disambiguation": True,
        },
        "action_map": {"n_codebook": 4, "lr": 0.5, "sigma": 0.0, "key": 0},
    }
    (run_root / "run_config.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "run_id": run_id,
                "requested_run_id": run_id,
                "saved_at_utc": "2026-04-07T12:00:00Z",
                "cli_args": {
                    "level": level,
                    "seed": 7,
                    "rand_prob": 0.3,
                    "window_length": 100,
                    "latent_max_age": 18,
                    "topological_neighbourhood_lr": 0.9,
                },
                "run_payload": run_payload,
                "execution": {},
                "paths": {},
            }
        ),
        encoding="utf-8",
    )
    (run_root / "run_status.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "run_id": run_id,
                "requested_run_id": run_id,
                "status": "completed",
                "started_at_utc": "2026-04-07T12:00:00Z",
                "finished_at_utc": "2026-04-07T12:10:00Z",
                "study": {},
                "config": {},
                "execution": {},
                "paths": {},
                "artifacts": {
                    "run_config_json": str(run_root / "run_config.json"),
                    "run_status_json": str(run_root / "run_status.json"),
                    "mercury_states_parquet": str(states_path),
                    "mercury_attribution_parquet": str(attribution_path),
                    "mercury_paper_precision_json": str(paper_path),
                    "mercury_latent_graph_npz": str(latent_graph),
                    "mercury_internal_graph_png": str(plots_root / "mercury_internal_graph.png"),
                    "mercury_bmu_attribution_step_png": str(plots_root / "mercury_bmu_attribution_vs_training_step.png"),
                    "mercury_bmu_attribution_edge_f1_png": str(plots_root / "mercury_bmu_attribution_vs_edge_f1.png"),
                    "mercury_bmu_attribution_node_ratio_png": str(plots_root / "mercury_bmu_attribution_vs_node_ratio.png"),
                    "internal_graphs_comparison_png": str(plots_root / "internal_graphs_comparison.png"),
                },
                "subsystems": {},
                "error": None,
            }
        ),
        encoding="utf-8",
    )

    if include_summary:
        pl.DataFrame(
            {
                "run_id": [run_id],
                "method": ["mercury"],
                "level": [level + 100],
                "seed": [7],
                "rand_prob": [0.3],
                "weight_memory": [1.0],
                "weight_action": [0.0],
                "paper_latent_precision": [0.95],
            }
        ).write_parquet(study_root / "study_summary.parquet")


def test_load_mercury_study_dataset_builds_frame_bundle(tmp_path: Path) -> None:
    study_root = tmp_path / "study"
    _write_run(study_root, run_id="run-a", level=20)

    dataset = load_mercury_study_dataset(study_root)

    assert dataset.runs.height == 1
    assert dataset.history.height == 2
    assert dataset.states.height == 3
    assert dataset.attribution.height == 3
    assert dataset.artifacts.height == 1
    assert dataset.runs.get_column("level").to_list() == [20]
    assert dataset.runs.get_column("paper_latent_precision").to_list() == [0.95]
    assert dataset.history.get_column("run_id").to_list() == ["run-a", "run-a"]
    assert "latent_cooccurrence_matrix" in dataset.history.columns
    assert "latent_exact_state_cooccurrence_matrix" in dataset.history.columns
    assert dataset.history.get_column("latent_cooccurrence_matrix").to_list() == [[[1, 2]], None]
    assert dataset.history.get_column("latent_exact_state_cooccurrence_matrix").to_list() == [
        [[1, 0, 2], [0, 3, 1]],
        None,
    ]
    assert "latent_max_age" not in dataset.history.columns
    assert "topological_neighbourhood_lr" not in dataset.history.columns
    assert dataset.states.get_column("step").to_list() == [1, 2, 3]
    assert "level" not in dataset.states.columns
    assert "weight_memory" not in dataset.attribution.columns
    assert dataset.artifacts.get_column("has_mercury_states_parquet").to_list() == [True]
    history_with_metadata = dataset.history_with_run_metadata(["latent_max_age", "topological_neighbourhood_lr"])
    assert history_with_metadata.get_column("latent_max_age").to_list() == [18, 18]
    assert history_with_metadata.get_column("topological_neighbourhood_lr").to_list() == [0.9, 0.9]
    states_with_metadata = dataset.states_with_run_metadata(["level"])
    assert states_with_metadata.get_column("level").to_list() == [20, 20, 20]
    attribution_with_metadata = dataset.attribution_with_run_metadata(["weight_memory"])
    assert attribution_with_metadata.get_column("weight_memory").to_list() == [1.0, 1.0, 1.0]


def test_load_mercury_study_dataset_falls_back_without_summary(tmp_path: Path) -> None:
    study_root = tmp_path / "study"
    _write_run(study_root, run_id="run-b", level=21, include_summary=False)

    dataset = load_mercury_study_dataset(study_root)

    assert dataset.runs.height == 1
    assert dataset.runs.get_column("level").to_list() == [21]
    assert dataset.runs.get_column("method").to_list() == ["mercury"]
    assert dataset.runs.get_column("seed").to_list() == [7]
