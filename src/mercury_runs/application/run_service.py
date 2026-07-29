from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
from tqdm import tqdm

from ..algorithms.cscg import CSCGConfig, run_cscg
from ..algorithms.cscg.evaluate import compute_cscg_paper_precision_metrics_from_model, write_cscg_paper_precision_payload
from ..algorithms.cscg.prepare import import_cscg_module
from ..algorithms.mercury import MercuryConfig, run_mercury
from ..algorithms.mercury.evaluate import compute_mercury_paper_precision_metrics, write_mercury_paper_precision_payload
from ..algorithms.mercury.api import summarize_mercury_run
from ..algorithms.mercury_cscg_hybrid import MercuryCSCGHybridConfig, run_mercury_cscg_hybrid, write_hybrid_paper_precision_payload
from ..algorithms.pocml import POCMLConfig, run_pocml
from ..algorithms.pocml.evaluate import compute_pocml_paper_precision_metrics_from_model, write_pocml_paper_precision_payload
from ..algorithms.pocml.prepare import import_pocml_modules
from ..algorithms.pocml.train import prepare_model_for_single_sequence_eval
from ..algorithms.sensory import (
    SensoryConfig,
    compute_sensory_paper_precision_metrics,
    run_sensory,
    write_sensory_paper_precision_payload,
)
from ..analysis import generate_bmu_attribution_plots, generate_method_graph_plots
from ..domain.models import ArtifactLayout, RunStatusModel
from ..infrastructure.paper_precision import exact_cartesian_state_ids_for_level
from ..infrastructure.paper_precision_snapshots import (
    list_paper_precision_snapshots,
    list_pocml_paper_precision_snapshots,
    load_cscg_paper_precision_snapshot,
    load_mercury_paper_precision_snapshot,
    load_pocml_paper_precision_snapshot,
    load_sensory_paper_precision_snapshot,
)
from ..infrastructure import (
    args_from_config,
    artifact_layout,
    atomic_write_json,
    base_runtime_env,
    baseline_enabled,
    build_computational_payload,
    build_run_config_payload,
    build_run_payload,
    collect_run_artifacts,
    computational_eval_enabled,
    computational_notes,
    PhaseTimer,
    load_run_config_model,
    load_run_status_model,
    maybe_log_wandb,
    peak_rss_mb,
    read_native_metrics_from_bundle,
    resolve_run_identity,
    safe_artifact_bytes,
    write_computational_artifact,
    utc_now,
)
from ..io_parquet import ParquetConfig, load_level_parquet
from ..infrastructure.progress import write_progress
from ..infrastructure.reporting import write_method_comparison_report
from ..save_results import write_bundle_parquet
from ..save_results import read_bundle_meta
from ..schemas_results import ResultBundleMeta, SourceDatasetRef
from .status_service import (
    build_run_status_model,
    mark_run_completed,
    mark_run_failed,
    persist_run_status,
)


def _sensor_dataset_key(sensor: str, sensor_range: int | None) -> str:
    if sensor == "cardinal distance":
        return f"{sensor}:range={sensor_range}"
    return sensor


def _run_bundles_root(run_root: Path, run_payload: dict) -> Path:
    method = str(run_payload.get("method", "mercury"))
    return run_root / "bundles" / method


def _baseline_dataset_selection(states_parquet: Path) -> tuple[str, str | None]:
    meta = read_bundle_meta(states_parquet)
    if meta is None:
        return "latest", None
    source = meta.source
    if source.select == "run_id" and source.dataset_run_id:
        return "run_id", str(source.dataset_run_id)
    return "latest", None


def _generate_standalone_dataset_run_ids(
    *,
    args: argparse.Namespace,
    reuse_existing_dataset: bool,
    num_steps: int | None = None,
) -> dict[str, str]:
    from src.maze_environment.generate_data import generate_data

    return generate_data(
        level=args.level,
        seed=args.seed,
        rand_prob=args.rand_prob,
        num_steps=(int(num_steps) if num_steps is not None else args.num_steps),
        valid_trajectories_only=bool(args.valid_trajectories_only),
        output_root=Path(args.datasets_root),
        reuse_existing=bool(reuse_existing_dataset),
        random_start=bool(args.random_start),
    )


def _hybrid_total_cscg_baseline_steps(args: argparse.Namespace) -> int:
    return int(args.num_steps)


def _hybrid_online_train_steps(args: argparse.Namespace) -> int:
    return int(args.num_steps) + int(args.cscg_n_iter)


def _baseline_cscg_n_iter_for_budget(args: argparse.Namespace) -> int:
    return int(args.cscg_n_iter)


def _write_exact_ground_truth_states(
    *,
    args: argparse.Namespace,
    run_id: str,
    run_payload: dict,
    layout: ArtifactLayout,
    dataset_run_ids: dict[str, str],
) -> Path:
    cartesian_run_id = dataset_run_ids.get("cartesian")
    cartesian_dataset = load_level_parquet(
        ParquetConfig(
            root=Path(args.datasets_root),
            level=args.level,
            sensor="cartesian",
            sensor_range=None,
            select=("run_id" if cartesian_run_id else "latest"),
            run_id=cartesian_run_id,
        )
    )
    ground_truth_state_ids = np.asarray(
        exact_cartesian_state_ids_for_level(
            level_index=int(args.level),
            positions=np.asarray(cartesian_dataset.observations),
        ),
        dtype=np.int32,
    )
    meta = ResultBundleMeta(
        run_id=run_id,
        timestamp_utc=utc_now(),
        source=SourceDatasetRef(
            level=int(args.level),
            sensor="cartesian",
            sensor_range=None,
            select=("run_id" if cartesian_run_id else "latest"),
            dataset_run_id=cartesian_run_id,
            dataset_parquet_name=cartesian_dataset.parquet_path.name,
        ),
        sensory_params=dict(run_payload.get("sensory") or {}),
        latent_params=None,
        action_map_params=dict(run_payload.get("action_map") or {}),
        memory_length=None,
        run_parameters={
            "ground_truth": {
                "series_name": "ground_truth_state_id",
                "source": "exact_cartesian_state_ids_for_level",
            },
            "dataset_runs": dataset_run_ids,
        },
        source_dataset_metadata=dict(cartesian_dataset.source_metadata),
        ground_truth_dataset_metadata=dict(cartesian_dataset.source_metadata),
        notes=f"standalone_ground_truth study={args.study_name}",
    )
    return write_bundle_parquet(
        output_dir=layout.ground_truth_states_parquet.parent,
        run_id=run_id,
        bundle_name="ground_truth_states",
        columns={"ground_truth_state_id": ground_truth_state_ids},
        meta=meta,
    )


def _computational_profile(run_payload: dict) -> str:
    return str((((run_payload.get("evaluation") or {}).get("computational") or {}).get("profile", "full")))


def _computational_deployable_paths(method: str, layout: ArtifactLayout) -> list[Path]:
    if method == "mercury":
        return [layout.mercury_states_parquet, layout.mercury_latent_graph_npz]
    if method == "pocml":
        return [layout.pocml_states_parquet, layout.pocml_embeddings_npz, layout.pocml_model_checkpoint_pt]
    if method == "cscg":
        return [layout.cscg_states_parquet, layout.cscg_model_npz]
    if method == "sensory":
        return [layout.sensory_states_parquet, layout.sensory_graph_npz, layout.sensory_action_map_npz]
    return []


def _computational_total_paths(method: str, layout: ArtifactLayout) -> list[Path]:
    if method == "mercury":
        return [
            layout.mercury_states_parquet,
            layout.mercury_latent_graph_npz,
            layout.mercury_internal_graph_png,
            layout.mercury_paper_precision_json,
        ]
    if method == "pocml":
        return [
            layout.pocml_states_parquet,
            layout.pocml_embeddings_npz,
            layout.pocml_model_checkpoint_pt,
            layout.pocml_internal_graph_png,
            layout.pocml_paper_precision_json,
        ]
    if method == "cscg":
        return [
            layout.cscg_states_parquet,
            layout.cscg_model_npz,
            layout.cscg_internal_graph_png,
            layout.cscg_paper_precision_json,
        ]
    if method == "mercury_cscg_hybrid":
        return [
            layout.mercury_cscg_hybrid_states_parquet,
            layout.mercury_cscg_hybrid_model_npz,
            layout.mercury_cscg_hybrid_paper_precision_json,
        ]
    if method == "sensory":
        return [
            layout.sensory_states_parquet,
            layout.sensory_graph_npz,
            layout.sensory_action_map_npz,
            layout.sensory_paper_precision_json,
        ]
    return []


def _computational_artifact_path(method: str, layout: ArtifactLayout) -> Path:
    if method == "mercury":
        return layout.mercury_computational_json
    if method == "pocml":
        return layout.pocml_computational_json
    if method == "cscg":
        return layout.cscg_computational_json
    if method == "sensory":
        return layout.sensory_computational_json
    if method == "mercury_cscg_hybrid":
        return layout.mercury_cscg_hybrid_computational_json
    raise ValueError(f"Unsupported method for computational artifact: {method}")


def _track_for_stage(method: str, stage: str) -> str:
    if stage.startswith("paper_precision"):
        return "paper_precision"
    if stage.startswith("jit_warmup") or stage.startswith("steady_state") or stage.startswith("computational_"):
        return "computational"
    if method == "mercury":
        return "eval" if stage == "native_eval" else "train"
    if method == "pocml":
        return "train" if stage == "train" else "eval"
    if method in {"cscg", "mercury_cscg_hybrid"}:
        return "train" if stage.startswith("train_") else "eval"
    if method == "sensory":
        return "eval" if stage == "native_eval" else "train"
    return "lifecycle"


def _resolve_eval_steps(method: str, native_metrics: Dict[str, Any]) -> int | None:
    if method == "mercury":
        return int(((native_metrics.get("latent") or {}).get("steps"))) if ((native_metrics.get("latent") or {}).get("steps")) is not None else None
    if method == "pocml":
        return int(native_metrics.get("next_obs_steps")) if native_metrics.get("next_obs_steps") is not None else None
    if method in {"cscg", "mercury_cscg_hybrid"}:
        return int(native_metrics.get("steps")) if native_metrics.get("steps") is not None else None
    if method == "sensory":
        return int(native_metrics.get("steps")) if native_metrics.get("steps") is not None else None
    return None


def _write_computational_metrics(
    *,
    method: str,
    layout: ArtifactLayout,
    run_payload: dict,
    run_status: RunStatusModel,
    timing: Dict[str, float],
    train_steps: int | None,
    native_metrics: Dict[str, Any],
) -> None:
    if not computational_eval_enabled(run_payload):
        return
    eval_steps = _resolve_eval_steps(method, native_metrics)
    peak_rss = peak_rss_mb()
    raw_metrics: Dict[str, Any] = {
        "train_wall_time_seconds": float(timing.get("train", 0.0)),
        "eval_wall_time_seconds": float(timing.get("eval", 0.0)),
        "total_wall_time_seconds": float(timing.get("train", 0.0) + timing.get("eval", 0.0)),
        "train_steps_per_second": (float(train_steps) / float(timing.get("train", 0.0))) if train_steps and timing.get("train", 0.0) > 0 else None,
        "eval_steps_per_second": (float(eval_steps) / float(timing.get("eval", 0.0))) if eval_steps and timing.get("eval", 0.0) > 0 else None,
        "peak_rss_mb": peak_rss,
        "deployable_artifact_bytes": safe_artifact_bytes(_computational_deployable_paths(method, layout)),
        "total_artifact_bytes": safe_artifact_bytes(_computational_total_paths(method, layout)),
        "n_train_steps_measured": train_steps,
        "n_eval_steps_measured": eval_steps,
    }
    include_jit_note = method in {"cscg", "mercury_cscg_hybrid"}
    if method in {"cscg", "mercury_cscg_hybrid"}:
        raw_metrics["jit_warmup_wall_time_seconds"] = float(timing.get("jit_warmup", 0.0))
        raw_metrics["steady_state_train_wall_time_seconds"] = float(timing.get("train", 0.0))
        raw_metrics["steady_state_eval_wall_time_seconds"] = float(timing.get("eval", 0.0))
    payload = build_computational_payload(
        method=method,
        measurement_profile=_computational_profile(run_payload),
        raw_metrics=raw_metrics,
        n_train_steps_measured=train_steps,
        n_eval_steps_measured=eval_steps,
        notes=computational_notes(peak_rss=peak_rss, include_jit_note=include_jit_note),
    )
    write_computational_artifact(_computational_artifact_path(method, layout), payload)
    run_status.subsystems[method].artifacts["computational_json"] = str(_computational_artifact_path(method, layout))
    run_status.subsystems[method].native_metrics["computational"] = raw_metrics


def _run_mercury_stage(
    *,
    args: argparse.Namespace,
    bundles_root: Path,
    run_id: str,
    run_payload: dict,
    reuse_existing_run: bool,
    reuse_existing_dataset: bool,
    run_status: RunStatusModel,
    layout: ArtifactLayout,
) -> Path:
    timer = PhaseTimer()
    eval_started = False
    if computational_eval_enabled(run_payload):
        write_progress(
            run_root=Path(layout.run_status_json).parent,
            run_id=run_id,
            method="mercury",
            track="computational",
            stage="computational_start",
            current=0,
            total=1,
            message="Starting computational measurement",
        )

    def _progress_callback(*, stage: str, current: int | None = None, total: int | None = None, message: str | None = None, extra: dict | None = None) -> None:
        nonlocal eval_started
        if stage == "native_eval" and not eval_started:
            timer.switch("eval")
            eval_started = True
        track = _track_for_stage("mercury", stage)
        write_progress(
            run_root=Path(layout.run_status_json).parent,
            run_id=run_id,
            method="mercury",
            track=track,
            stage=stage,
            current=current,
            total=total,
            message=message,
            extra=extra,
        )

    _progress_callback(stage="mercury.start", message="Starting Mercury")
    states_parquet = run_mercury(
        MercuryConfig(
            datasets_root=Path(args.datasets_root),
            results_root=bundles_root,
            run_id=run_id,
            level=args.level,
            sensor=args.sensor,
            sensor_range=args.sensor_range,
            memory_length=args.memory_length,
            window_length=args.window_length,
            seed=args.seed,
            rand_prob=args.rand_prob,
            num_steps=args.num_steps,
            valid_trajectories_only=bool(args.valid_trajectories_only),
            mercury_valid_trajectories_only=bool(args.mercury_valid_trajectories_only),
            mercury_split_sensory_raw_latent_valid=bool(args.mercury_split_sensory_raw_latent_valid),
            mercury_input_mode=str(run_payload.get("mercury_input_mode", getattr(args, "mercury_input_mode", "pretrained_raw"))),
            random_start=bool(args.random_start),
            reuse_existing_dataset=reuse_existing_dataset,
            reuse_existing_run=reuse_existing_run,
            sensory=run_payload["sensory"],
            latent=run_payload["latent"],
            action_map=run_payload["action_map"],
            paper_precision_enabled=bool((run_payload.get("evaluation") or {}).get("paper_precision", {}).get("enabled", False)),
            paper_precision_mode=str((run_payload.get("evaluation") or {}).get("paper_precision", {}).get("mode", "off")),
            paper_precision_eval_interval=int((run_payload.get("evaluation") or {}).get("paper_precision", {}).get("eval_interval", 1)),
            paper_precision_eval_checkpoints=(run_payload.get("evaluation") or {}).get("paper_precision", {}).get("eval_checkpoints"),
            paper_precision_num_points=(run_payload.get("evaluation") or {}).get("paper_precision", {}).get("num_points"),
            paper_precision_num_walks=int((run_payload.get("evaluation") or {}).get("paper_precision", {}).get("num_walks", 100)),
            paper_precision_walk_length=int((run_payload.get("evaluation") or {}).get("paper_precision", {}).get("walk_length", 10_000)),
            structure_metrics_enabled=bool(((((run_payload.get("evaluation") or {}).get("paper_precision") or {}).get("structure_metrics") or {}).get("enabled", True))),
            structure_metrics_epsilon=float(((((run_payload.get("evaluation") or {}).get("paper_precision") or {}).get("structure_metrics") or {}).get("epsilon", 1e-6))),
            structure_metrics_epsilons=(((((run_payload.get("evaluation") or {}).get("paper_precision") or {}).get("structure_metrics") or {}).get("thresholds"))),
            structure_metrics_scope=str(((((run_payload.get("evaluation") or {}).get("paper_precision") or {}).get("structure_metrics") or {}).get("scope", "exact_level"))),
            structure_metrics_ground_truth_source=str(((((run_payload.get("evaluation") or {}).get("paper_precision") or {}).get("structure_metrics") or {}).get("ground_truth_source", "empirical_walks"))),
            run_parameters={
                "run_payload": run_payload,
                "study": {"study_root": args.study_root, "study_name": args.study_name},
            },
            notes=f"study={args.study_name}",
        ),
        progress_callback=_progress_callback,
    )
    run_status.subsystems["mercury"].status = "completed"
    run_status.subsystems["mercury"].artifacts = {
        "states_parquet": str(layout.mercury_states_parquet),
        "attribution_parquet": str(layout.mercury_attribution_parquet),
        "paper_precision_json": str(layout.mercury_paper_precision_json),
        "paper_precision_snapshots_dir": str(Path(layout.run_status_json).parent / "snapshots" / "mercury" / "paper_precision"),
        "latent_graph_npz": str(layout.mercury_latent_graph_npz),
    }
    run_status.subsystems["mercury"].native_metrics = summarize_mercury_run(states_parquet, datasets_root=Path(args.datasets_root))
    _write_computational_metrics(
        method="mercury",
        layout=layout,
        run_payload=run_payload,
        run_status=run_status,
        timing=timer.finish(),
        train_steps=int(args.num_steps),
        native_metrics=run_status.subsystems["mercury"].native_metrics,
    )
    if computational_eval_enabled(run_payload):
        write_progress(
            run_root=Path(layout.run_status_json).parent,
            run_id=run_id,
            method="mercury",
            track="computational",
            stage="computational_end",
            current=1,
            total=1,
            message="Completed computational measurement",
        )
    return states_parquet


def _run_sensory_stage(
    *,
    args: argparse.Namespace,
    bundles_root: Path,
    run_id: str,
    run_payload: dict,
    run_status: RunStatusModel,
    layout: ArtifactLayout,
    dataset_run_id: str | None,
) -> Path:
    timer = PhaseTimer()
    if computational_eval_enabled(run_payload):
        write_progress(
            run_root=Path(layout.run_status_json).parent,
            run_id=run_id,
            method="sensory",
            track="computational",
            stage="computational_start",
            current=0,
            total=1,
            message="Starting computational measurement",
        )

    def _progress_callback(*, stage: str, current: int | None = None, total: int | None = None, message: str | None = None, extra: dict | None = None) -> None:
        write_progress(
            run_root=Path(layout.run_status_json).parent,
            run_id=run_id,
            method="sensory",
            track=_track_for_stage("sensory", stage),
            stage=stage,
            current=current,
            total=total,
            message=message,
            extra=extra,
        )

    _progress_callback(stage="sensory.start", message="Starting Sensory")
    states_parquet = run_sensory(
        SensoryConfig(
            datasets_root=Path(args.datasets_root),
            output_root=bundles_root,
            run_id=run_id,
            level=args.level,
            sensor=args.sensor,
            sensor_range=args.sensor_range,
            seed=args.seed,
            rand_prob=args.rand_prob,
            num_steps=args.num_steps,
            window_length=args.window_length,
            valid_trajectories_only=bool(args.valid_trajectories_only),
            observation_mode=str(run_payload.get("sensory_observation_mode", getattr(args, "sensory_observation_mode", "raw"))),
            dataset_select=("run_id" if dataset_run_id else "latest"),
            dataset_run_id=dataset_run_id,
            random_start=bool(args.random_start),
            reuse_existing_dataset=True,
            sensory=run_payload["sensory"],
            action_map=run_payload["action_map"],
            paper_precision_enabled=bool((run_payload.get("evaluation") or {}).get("paper_precision", {}).get("enabled", False)),
            paper_precision_mode=str((run_payload.get("evaluation") or {}).get("paper_precision", {}).get("mode", "off")),
            paper_precision_eval_interval=int((run_payload.get("evaluation") or {}).get("paper_precision", {}).get("eval_interval", 1)),
            paper_precision_eval_checkpoints=(run_payload.get("evaluation") or {}).get("paper_precision", {}).get("eval_checkpoints"),
            paper_precision_num_points=(run_payload.get("evaluation") or {}).get("paper_precision", {}).get("num_points"),
            paper_precision_num_walks=int((run_payload.get("evaluation") or {}).get("paper_precision", {}).get("num_walks", 100)),
            paper_precision_walk_length=int((run_payload.get("evaluation") or {}).get("paper_precision", {}).get("walk_length", 10_000)),
            structure_metrics_enabled=bool(((((run_payload.get("evaluation") or {}).get("paper_precision") or {}).get("structure_metrics") or {}).get("enabled", True))),
            structure_metrics_epsilon=float(((((run_payload.get("evaluation") or {}).get("paper_precision") or {}).get("structure_metrics") or {}).get("epsilon", 1e-6))),
            structure_metrics_epsilons=(((((run_payload.get("evaluation") or {}).get("paper_precision") or {}).get("structure_metrics") or {}).get("thresholds"))),
            structure_metrics_scope=str(((((run_payload.get("evaluation") or {}).get("paper_precision") or {}).get("structure_metrics") or {}).get("scope", "exact_level"))),
            structure_metrics_ground_truth_source=str(((((run_payload.get("evaluation") or {}).get("paper_precision") or {}).get("structure_metrics") or {}).get("ground_truth_source", "empirical_walks"))),
            run_parameters={
                "run_payload": run_payload,
                "study": {"study_root": args.study_root, "study_name": args.study_name},
            },
            notes=f"study={args.study_name}",
        ),
        progress_callback=_progress_callback,
    )
    run_status.subsystems["sensory"].status = "completed"
    run_status.subsystems["sensory"].artifacts = {
        "states_parquet": str(layout.sensory_states_parquet),
        "paper_precision_json": str(layout.sensory_paper_precision_json),
        "paper_precision_snapshots_dir": str(Path(layout.run_status_json).parent / "snapshots" / "sensory" / "paper_precision"),
        "sensory_graph_npz": str(layout.sensory_graph_npz),
        "action_map_npz": str(layout.sensory_action_map_npz),
        "observation_vocabulary_npz": str(layout.sensory_observation_vocabulary_npz),
    }
    run_status.subsystems["sensory"].native_metrics = read_native_metrics_from_bundle(Path(states_parquet), "sensory_eval")
    _write_computational_metrics(
        method="sensory",
        layout=layout,
        run_payload=run_payload,
        run_status=run_status,
        timing=timer.finish(),
        train_steps=int(args.num_steps),
        native_metrics=run_status.subsystems["sensory"].native_metrics,
    )
    if computational_eval_enabled(run_payload):
        write_progress(
            run_root=Path(layout.run_status_json).parent,
            run_id=run_id,
            method="sensory",
            track="computational",
            stage="computational_end",
            current=1,
            total=1,
            message="Completed computational measurement",
        )
    return states_parquet


def _run_mercury_cscg_hybrid_stage(
    *,
    args: argparse.Namespace,
    run_root: Path,
    run_id: str,
    run_payload: dict,
    run_status: RunStatusModel,
    layout: ArtifactLayout,
    dataset_select: str,
    dataset_run_id: str | None,
    reuse_existing_run: bool,
    reuse_existing_dataset: bool,
) -> Path:
    timer = PhaseTimer()
    steady_state_started = False
    eval_started = False
    if computational_eval_enabled(run_payload):
        write_progress(
            run_root=run_root,
            run_id=run_id,
            method="mercury_cscg_hybrid",
            track="computational",
            stage="computational_start",
            current=0,
            total=1,
            message="Starting computational measurement",
        )

    def _progress_callback(*, stage: str, current: int | None = None, total: int | None = None, message: str | None = None, extra: dict | None = None) -> None:
        nonlocal steady_state_started, eval_started
        if stage == "jit_warmup_start":
            timer.switch("jit_warmup")
        elif stage == "jit_warmup_end" and not steady_state_started:
            timer.switch("train")
            steady_state_started = True
        elif stage == "native_eval" and not eval_started:
            timer.switch("eval")
            eval_started = True
        write_progress(
            run_root=run_root,
            run_id=run_id,
            method="mercury_cscg_hybrid",
            track=_track_for_stage("mercury_cscg_hybrid", stage),
            stage=stage,
            current=current,
            total=total,
            message=message,
            extra=extra,
        )

    states_path = run_mercury_cscg_hybrid(
        MercuryCSCGHybridConfig(
            datasets_root=Path(args.datasets_root),
            output_root=run_root / "bundles" / "mercury_cscg_hybrid",
            run_id=run_id,
            level=args.level,
            sensor=args.sensor,
            sensor_range=args.sensor_range,
            ground_truth_states_parquet=layout.ground_truth_states_parquet,
            memory_length=args.memory_length,
            window_length=args.window_length,
            seed=args.seed,
            rand_prob=args.rand_prob,
            num_steps=args.num_steps,
            valid_trajectories_only=bool(args.valid_trajectories_only),
            mercury_valid_trajectories_only=bool(args.mercury_valid_trajectories_only),
            mercury_split_sensory_raw_latent_valid=bool(args.mercury_split_sensory_raw_latent_valid),
            mercury_input_mode=str(run_payload.get("mercury_input_mode", getattr(args, "mercury_input_mode", "pretrained_raw"))),
            random_start=bool(args.random_start),
            reuse_existing_dataset=reuse_existing_dataset,
            reuse_existing_run=reuse_existing_run,
            sensory=run_payload["sensory"],
            latent=run_payload["latent"],
            action_map=run_payload["action_map"],
            dataset_select=dataset_select,
            dataset_run_id=dataset_run_id,
            n_iter=args.cscg_n_iter,
            pseudocount=args.cscg_pseudocount,
            term_early=args.cscg_term_early,
            train_algorithm=args.cscg_train_algorithm,
            training_mode=args.cscg_training_mode,
            batch_size=args.cscg_batch_size,
            online_lambda=args.cscg_online_lambda,
            mercury_prior_weight=float(getattr(args, "mercury_cscg_prior_weight", 1.0)),
            eval_viterbi=bool(getattr(args, "cscg_eval_viterbi", False)),
            paper_precision_enabled=bool((run_payload.get("evaluation") or {}).get("paper_precision", {}).get("enabled", False)),
            paper_precision_mode=str((run_payload.get("evaluation") or {}).get("paper_precision", {}).get("mode", "off")),
            paper_precision_eval_interval=int((run_payload.get("evaluation") or {}).get("paper_precision", {}).get("eval_interval", 1)),
            paper_precision_eval_checkpoints=(run_payload.get("evaluation") or {}).get("paper_precision", {}).get("eval_checkpoints"),
            paper_precision_num_points=(run_payload.get("evaluation") or {}).get("paper_precision", {}).get("num_points"),
            paper_precision_num_walks=int((run_payload.get("evaluation") or {}).get("paper_precision", {}).get("num_walks", 100)),
            paper_precision_walk_length=int((run_payload.get("evaluation") or {}).get("paper_precision", {}).get("walk_length", 10_000)),
            paper_precision_seed=args.seed,
            paper_precision_rand_prob=args.rand_prob,
            paper_precision_sensory_params=run_payload["sensory"],
            paper_precision_action_map_params=run_payload["action_map"],
            structure_metrics_enabled=bool(((((run_payload.get("evaluation") or {}).get("paper_precision") or {}).get("structure_metrics") or {}).get("enabled", True))),
            structure_metrics_epsilon=float(((((run_payload.get("evaluation") or {}).get("paper_precision") or {}).get("structure_metrics") or {}).get("epsilon", 1e-6))),
            structure_metrics_epsilons=(((((run_payload.get("evaluation") or {}).get("paper_precision") or {}).get("structure_metrics") or {}).get("thresholds"))),
            structure_metrics_scope=str(((((run_payload.get("evaluation") or {}).get("paper_precision") or {}).get("structure_metrics") or {}).get("scope", "exact_level"))),
            structure_metrics_ground_truth_source=str(((((run_payload.get("evaluation") or {}).get("paper_precision") or {}).get("structure_metrics") or {}).get("ground_truth_source", "empirical_walks"))),
            notes=f"study={args.study_name}",
        ),
        progress_callback=_progress_callback,
    )
    run_status.subsystems["mercury_cscg_hybrid"].status = "completed"
    run_status.subsystems["mercury_cscg_hybrid"].artifacts = {
        "states_parquet": str(states_path),
        "paper_precision_json": str(layout.mercury_cscg_hybrid_paper_precision_json),
        "paper_precision_snapshots_dir": str(run_root / "snapshots" / "mercury_cscg_hybrid" / "paper_precision"),
        "model_npz": str(layout.mercury_cscg_hybrid_model_npz),
    }
    run_status.subsystems["mercury_cscg_hybrid"].native_metrics = read_native_metrics_from_bundle(Path(states_path), "cscg_eval")
    _write_computational_metrics(
        method="mercury_cscg_hybrid",
        layout=layout,
        run_payload=run_payload,
        run_status=run_status,
        timing=timer.finish(),
        train_steps=(
            _hybrid_online_train_steps(args)
            if str(args.cscg_training_mode).strip().lower() == "online_em"
            else int(args.num_steps) + int(args.cscg_n_iter)
        ),
        native_metrics=run_status.subsystems["mercury_cscg_hybrid"].native_metrics,
    )
    if computational_eval_enabled(run_payload):
        write_progress(
            run_root=run_root,
            run_id=run_id,
            method="mercury_cscg_hybrid",
            track="computational",
            stage="computational_end",
            current=1,
            total=1,
            message="Completed computational measurement",
        )
    return states_path


def _run_baseline_stages(
    *,
    args: argparse.Namespace,
    run_root: Path,
    run_id: str,
    states_parquet: Path,
    run_payload: dict,
    run_status: RunStatusModel,
    layout: ArtifactLayout,
    baseline_dataset_select: str | None = None,
    baseline_dataset_run_id: str | None = None,
    baseline_cscg_n_iter: int | None = None,
    write_comparison: bool = True,
) -> None:
    if baseline_dataset_select is None:
        baseline_dataset_select, baseline_dataset_run_id = _baseline_dataset_selection(states_parquet)

    if baseline_enabled(run_payload, "pocml"):
        pocml_timer = PhaseTimer()
        pocml_eval_started = False
        if computational_eval_enabled(run_payload):
            write_progress(
                run_root=run_root,
                run_id=run_id,
                method="pocml",
                track="computational",
                stage="computational_start",
                current=0,
                total=1,
                message="Starting computational measurement",
            )

        def _pocml_progress_callback(*, stage: str, current: int | None = None, total: int | None = None, message: str | None = None, extra: dict | None = None) -> None:
            nonlocal pocml_eval_started
            if stage in {"native_eval", "direct_eval", "gt_eval"} and not pocml_eval_started:
                pocml_timer.switch("eval")
                pocml_eval_started = True
            track = _track_for_stage("pocml", stage)
            write_progress(
                run_root=run_root,
                run_id=run_id,
                method="pocml",
                track=track,
                stage=stage,
                current=current,
                total=total,
                message=message,
                extra=extra,
            )

        pocml_states_path = run_pocml(
            POCMLConfig(
                datasets_root=Path(args.datasets_root),
                output_root=run_root / "bundles" / "pocml",
                run_id=run_id,
                level=args.level,
                sensor=args.sensor,
                sensor_range=args.sensor_range,
                ground_truth_states_parquet=states_parquet,
                window_length=args.window_length,
                valid_trajectories_only=bool(args.pocml_valid_trajectories_only),
                dataset_select=baseline_dataset_select,
                dataset_run_id=baseline_dataset_run_id,
                trajectory_length=args.pocml_trajectory_length,
                max_trajectories=args.pocml_max_trajectories,
                state_dim=args.pocml_state_dim,
                random_feature_dim=args.pocml_random_feature_dim,
                n_states=args.pocml_n_states,
                precision_capacities=args.pocml_precision_capacities,
                alpha=args.pocml_alpha,
                epochs=args.pocml_epochs,
                lr_q=args.pocml_lr_q,
                lr_v=args.pocml_lr_v,
                lr_all=args.pocml_lr_all,
                lr_m=args.pocml_lr_m,
                reg_m=args.pocml_reg_m,
                max_iter_m=args.pocml_max_iter_m,
                eps_m=args.pocml_eps_m,
                batch_size=args.pocml_batch_size,
                memory_bias=args.pocml_memory_bias,
                use_ground_truth_node_ids=args.pocml_use_ground_truth_node_ids,
                paper_precision_enabled=bool((run_payload.get("evaluation") or {}).get("paper_precision", {}).get("enabled", False)),
                paper_precision_mode=str((run_payload.get("evaluation") or {}).get("paper_precision", {}).get("mode", "off")),
                paper_precision_eval_interval=int((run_payload.get("evaluation") or {}).get("paper_precision", {}).get("eval_interval", 1)),
                paper_precision_eval_checkpoints=(run_payload.get("evaluation") or {}).get("paper_precision", {}).get("eval_checkpoints"),
                paper_precision_num_points=(run_payload.get("evaluation") or {}).get("paper_precision", {}).get("num_points"),
                paper_precision_num_walks=int((run_payload.get("evaluation") or {}).get("paper_precision", {}).get("num_walks", 100)),
                paper_precision_walk_length=int((run_payload.get("evaluation") or {}).get("paper_precision", {}).get("walk_length", 10_000)),
                paper_precision_seed=args.seed,
                paper_precision_rand_prob=args.rand_prob,
                paper_precision_sensory_params=run_payload["sensory"],
                paper_precision_action_map_params=run_payload["action_map"],
                structure_metrics_enabled=bool(((((run_payload.get("evaluation") or {}).get("paper_precision") or {}).get("structure_metrics") or {}).get("enabled", True))),
                structure_metrics_epsilon=float(((((run_payload.get("evaluation") or {}).get("paper_precision") or {}).get("structure_metrics") or {}).get("epsilon", 1e-6))),
                structure_metrics_epsilons=(((((run_payload.get("evaluation") or {}).get("paper_precision") or {}).get("structure_metrics") or {}).get("thresholds"))),
                structure_metrics_scope=str(((((run_payload.get("evaluation") or {}).get("paper_precision") or {}).get("structure_metrics") or {}).get("scope", "exact_level"))),
                structure_metrics_ground_truth_source=str(((((run_payload.get("evaluation") or {}).get("paper_precision") or {}).get("structure_metrics") or {}).get("ground_truth_source", "empirical_walks"))),
            ),
            progress_callback=_pocml_progress_callback,
        )
        run_status.subsystems["pocml"].status = "completed"
        run_status.subsystems["pocml"].artifacts = {
            "states_parquet": str(pocml_states_path),
            "paper_precision_json": str(layout.pocml_paper_precision_json),
            "embeddings_npz": str(layout.pocml_embeddings_npz),
            "model_checkpoint_pt": str(layout.pocml_model_checkpoint_pt),
        }
        run_status.subsystems["pocml"].native_metrics = read_native_metrics_from_bundle(Path(pocml_states_path), "pocml_eval")
        _write_computational_metrics(
            method="pocml",
            layout=layout,
            run_payload=run_payload,
            run_status=run_status,
            timing=pocml_timer.finish(),
            train_steps=int(args.pocml_epochs),
            native_metrics=run_status.subsystems["pocml"].native_metrics,
        )
        if computational_eval_enabled(run_payload):
            write_progress(
                run_root=run_root,
                run_id=run_id,
                method="pocml",
                track="computational",
                stage="computational_end",
                current=1,
                total=1,
                message="Completed computational measurement",
            )

    if baseline_enabled(run_payload, "cscg"):
        cscg_timer = PhaseTimer()
        steady_state_started = False
        eval_started = False
        if computational_eval_enabled(run_payload):
            write_progress(
                run_root=run_root,
                run_id=run_id,
                method="cscg",
                track="computational",
                stage="computational_start",
                current=0,
                total=1,
                message="Starting computational measurement",
            )

        def _cscg_progress_callback(*, stage: str, current: int | None = None, total: int | None = None, message: str | None = None, extra: dict | None = None) -> None:
            nonlocal steady_state_started, eval_started
            if stage == "jit_warmup_start":
                cscg_timer.switch("jit_warmup")
            elif stage == "jit_warmup_end" and not steady_state_started:
                cscg_timer.switch("train")
                steady_state_started = True
            elif stage == "native_eval" and not eval_started:
                cscg_timer.switch("eval")
                eval_started = True
            write_progress(
                run_root=run_root,
                run_id=run_id,
                method="cscg",
                track=_track_for_stage("cscg", stage),
                stage=stage,
                current=current,
                total=total,
                message=message,
                extra=extra,
            )

        cscg_states_path = run_cscg(
            CSCGConfig(
                datasets_root=Path(args.datasets_root),
                output_root=run_root / "bundles" / "cscg",
                run_id=run_id,
                level=args.level,
                sensor=args.sensor,
                sensor_range=args.sensor_range,
                ground_truth_states_parquet=states_parquet,
                window_length=args.window_length,
                valid_trajectories_only=bool(args.cscg_valid_trajectories_only),
                dataset_select=baseline_dataset_select,
                dataset_run_id=baseline_dataset_run_id,
                clones_per_obs=args.cscg_clones_per_obs,
                clone_allocation=str(getattr(args, "cscg_clone_allocation", "uniform")),
                clones_per_state=int(getattr(args, "cscg_clones_per_state", 1)),
                n_iter=(int(baseline_cscg_n_iter) if baseline_cscg_n_iter is not None else args.cscg_n_iter),
                pseudocount=args.cscg_pseudocount,
                term_early=args.cscg_term_early,
                train_algorithm=args.cscg_train_algorithm,
                training_mode=args.cscg_training_mode,
                batch_size=args.cscg_batch_size,
                online_lambda=args.cscg_online_lambda,
                eval_viterbi=bool(getattr(args, "cscg_eval_viterbi", False)),
                seed=(args.cscg_seed if args.cscg_seed is not None else args.seed),
                paper_precision_enabled=bool((run_payload.get("evaluation") or {}).get("paper_precision", {}).get("enabled", False)),
                paper_precision_mode=str((run_payload.get("evaluation") or {}).get("paper_precision", {}).get("mode", "off")),
                paper_precision_eval_interval=int((run_payload.get("evaluation") or {}).get("paper_precision", {}).get("eval_interval", 1)),
                paper_precision_eval_checkpoints=(run_payload.get("evaluation") or {}).get("paper_precision", {}).get("eval_checkpoints"),
                paper_precision_num_points=(run_payload.get("evaluation") or {}).get("paper_precision", {}).get("num_points"),
                paper_precision_num_walks=int((run_payload.get("evaluation") or {}).get("paper_precision", {}).get("num_walks", 100)),
                paper_precision_walk_length=int((run_payload.get("evaluation") or {}).get("paper_precision", {}).get("walk_length", 10_000)),
                paper_precision_seed=args.seed,
                paper_precision_rand_prob=args.rand_prob,
                paper_precision_sensory_params=run_payload["sensory"],
                paper_precision_action_map_params=run_payload["action_map"],
                structure_metrics_enabled=bool(((((run_payload.get("evaluation") or {}).get("paper_precision") or {}).get("structure_metrics") or {}).get("enabled", True))),
                structure_metrics_epsilon=float(((((run_payload.get("evaluation") or {}).get("paper_precision") or {}).get("structure_metrics") or {}).get("epsilon", 1e-6))),
                structure_metrics_epsilons=(((((run_payload.get("evaluation") or {}).get("paper_precision") or {}).get("structure_metrics") or {}).get("thresholds"))),
                structure_metrics_scope=str(((((run_payload.get("evaluation") or {}).get("paper_precision") or {}).get("structure_metrics") or {}).get("scope", "exact_level"))),
                structure_metrics_ground_truth_source=str(((((run_payload.get("evaluation") or {}).get("paper_precision") or {}).get("structure_metrics") or {}).get("ground_truth_source", "empirical_walks"))),
            ),
            progress_callback=_cscg_progress_callback,
        )
        run_status.subsystems["cscg"].status = "completed"
        run_status.subsystems["cscg"].artifacts = {
            "states_parquet": str(cscg_states_path),
            "paper_precision_json": str(layout.cscg_paper_precision_json),
            "paper_precision_snapshots_dir": str(run_root / "snapshots" / "cscg" / "paper_precision"),
            "model_npz": str(layout.cscg_model_npz),
        }
        run_status.subsystems["cscg"].native_metrics = read_native_metrics_from_bundle(Path(cscg_states_path), "cscg_eval")
        _write_computational_metrics(
            method="cscg",
            layout=layout,
            run_payload=run_payload,
            run_status=run_status,
            timing=cscg_timer.finish(),
            train_steps=(int(baseline_cscg_n_iter) if baseline_cscg_n_iter is not None else int(args.cscg_n_iter)),
            native_metrics=run_status.subsystems["cscg"].native_metrics,
        )
        if computational_eval_enabled(run_payload):
            write_progress(
                run_root=run_root,
                run_id=run_id,
                method="cscg",
                track="computational",
                stage="computational_end",
                current=1,
                total=1,
                message="Completed computational measurement",
            )

    if write_comparison and (baseline_enabled(run_payload, "pocml") or baseline_enabled(run_payload, "cscg")):
        write_method_comparison_report(
            run_root=run_root,
            run_id=run_id,
            run_payload=run_payload,
            subsystems={name: payload.model_dump(mode="json") for name, payload in run_status.subsystems.items()},
        )


def _generate_plot_artifacts(*, args: argparse.Namespace, run_root: Path, layout: ArtifactLayout, run_payload: dict) -> None:
    artifacts = collect_run_artifacts(layout=layout, run_payload=run_payload)
    generate_method_graph_plots(
        run_root=run_root,
        datasets_root=Path(args.datasets_root),
        artifacts={key: value for key, value in artifacts.items() if isinstance(value, str) and value},
    )
    generate_bmu_attribution_plots(layout=layout)


def execute_pipeline_run(
    *,
    args: argparse.Namespace,
    run_id: str,
    requested_run_id: str,
    run_root: Path,
    run_payload: dict,
    reuse_existing_run: bool,
    reuse_existing_dataset: bool,
) -> RunStatusModel:
    bundles_root = _run_bundles_root(run_root, run_payload)
    metrics_root = run_root / "metrics"
    layout = artifact_layout(run_root=run_root, run_id=run_id, run_payload=run_payload)
    run_status = build_run_status_model(
        args=args,
        run_id=run_id,
        requested_run_id=requested_run_id,
        run_payload=run_payload,
        run_root=run_root,
        bundles_root=bundles_root,
        metrics_root=metrics_root,
        reuse_existing_run=reuse_existing_run,
        reuse_existing_dataset=reuse_existing_dataset,
        status="running",
    )
    write_progress(
        run_root=run_root,
        run_id=run_id,
        method=str(run_payload.get("method", "mercury")),
        track="lifecycle",
        stage="queued",
        current=0,
        total=1,
        message="Queued",
    )
    persist_run_status(run_status, layout.run_status_json)
    try:
        method = str(run_payload.get("method", "mercury"))
        if method == "mercury":
            states_parquet = _run_mercury_stage(
                args=args,
                bundles_root=bundles_root,
                run_id=run_id,
                run_payload=run_payload,
                reuse_existing_run=reuse_existing_run,
                reuse_existing_dataset=reuse_existing_dataset,
                run_status=run_status,
                layout=layout,
            )
            _run_baseline_stages(
                args=args,
                run_root=run_root,
                run_id=run_id,
                states_parquet=states_parquet,
                run_payload=run_payload,
                run_status=run_status,
                layout=layout,
            )
            _generate_plot_artifacts(args=args, run_root=run_root, layout=layout, run_payload=run_payload)
        elif method == "sensory":
            dataset_run_ids = _generate_standalone_dataset_run_ids(
                args=args,
                reuse_existing_dataset=reuse_existing_dataset,
            )
            dataset_run_id = dataset_run_ids.get(_sensor_dataset_key(args.sensor, args.sensor_range))
            _run_sensory_stage(
                args=args,
                bundles_root=bundles_root,
                run_id=run_id,
                run_payload=run_payload,
                run_status=run_status,
                layout=layout,
                dataset_run_id=dataset_run_id,
            )
        elif method in {"pocml", "cscg", "mercury_cscg_hybrid"}:
            dataset_run_ids = _generate_standalone_dataset_run_ids(
                args=args,
                reuse_existing_dataset=reuse_existing_dataset,
            )
            ground_truth_states_parquet = _write_exact_ground_truth_states(
                args=args,
                run_id=run_id,
                run_payload=run_payload,
                layout=layout,
                dataset_run_ids=dataset_run_ids,
            )
            dataset_run_id = dataset_run_ids.get(_sensor_dataset_key(args.sensor, args.sensor_range))
            if method == "mercury_cscg_hybrid":
                _run_mercury_cscg_hybrid_stage(
                    args=args,
                    run_root=run_root,
                    run_id=run_id,
                    run_payload=run_payload,
                    run_status=run_status,
                    layout=layout,
                    dataset_select=("run_id" if dataset_run_id else "latest"),
                    dataset_run_id=dataset_run_id,
                    reuse_existing_run=reuse_existing_run,
                    reuse_existing_dataset=reuse_existing_dataset,
                )
                baseline_states_parquet = ground_truth_states_parquet
                baseline_select = "run_id" if dataset_run_id else "latest"
                baseline_run_id = dataset_run_id
                baseline_cscg_n_iter = None
                if baseline_enabled(run_payload, "cscg"):
                    baseline_total_steps = _hybrid_total_cscg_baseline_steps(args)
                    baseline_dataset_run_ids = _generate_standalone_dataset_run_ids(
                        args=args,
                        reuse_existing_dataset=reuse_existing_dataset,
                        num_steps=baseline_total_steps,
                    )
                    baseline_run_id = baseline_dataset_run_ids.get(_sensor_dataset_key(args.sensor, args.sensor_range))
                    baseline_select = "run_id" if baseline_run_id else "latest"
                    baseline_layout = layout.model_copy(
                        update={
                            "ground_truth_states_parquet": run_root
                            / "ground_truth"
                            / "baseline_cscg"
                            / f"{run_id}_ground_truth_states.parquet"
                        }
                    )
                    baseline_states_parquet = _write_exact_ground_truth_states(
                        args=args,
                        run_id=run_id,
                        run_payload=run_payload,
                        layout=baseline_layout,
                        dataset_run_ids=baseline_dataset_run_ids,
                    )
                    baseline_cscg_n_iter = _baseline_cscg_n_iter_for_budget(args)
                _run_baseline_stages(
                    args=args,
                    run_root=run_root,
                    run_id=run_id,
                    states_parquet=baseline_states_parquet,
                    run_payload=run_payload,
                    run_status=run_status,
                    layout=layout,
                    baseline_dataset_select=baseline_select,
                    baseline_dataset_run_id=baseline_run_id,
                    baseline_cscg_n_iter=baseline_cscg_n_iter,
                    write_comparison=baseline_enabled(run_payload, "cscg"),
                )
            else:
                _run_baseline_stages(
                    args=args,
                    run_root=run_root,
                    run_id=run_id,
                    states_parquet=ground_truth_states_parquet,
                    run_payload=run_payload,
                    run_status=run_status,
                    layout=layout,
                    baseline_dataset_select=("run_id" if dataset_run_id else "latest"),
                    baseline_dataset_run_id=dataset_run_id,
                    write_comparison=False,
                )
        else:
            raise ValueError(f"Unsupported method: {method}")
        write_progress(
            run_root=run_root,
            run_id=run_id,
            method=str(run_payload.get("method", "mercury")),
            track="lifecycle",
            stage="completed",
            current=1,
            total=1,
            message="Completed",
        )
        return mark_run_completed(
            run_status=run_status,
            status_path=layout.run_status_json,
            layout=layout,
            run_payload=run_payload,
        )
    except Exception as exc:
        write_progress(
            run_root=run_root,
            run_id=run_id,
            method=str(run_payload.get("method", "mercury")),
            track="lifecycle",
            stage="failed",
            current=1,
            total=1,
            message=repr(exc),
        )
        mark_run_failed(
            run_status=run_status,
            status_path=layout.run_status_json,
            layout=layout,
            run_payload=run_payload,
            error=exc,
        )
        raise


def _run_worker_subprocess(*, args: argparse.Namespace, run_id: str) -> None:
    worker_config = dict(vars(args))
    worker_config["run_id"] = run_id
    worker_config["_worker_run"] = True
    subprocess.run(
        [sys.executable, "main.py"] + args_from_config(worker_config),
        check=True,
        env=base_runtime_env(),
    )


def _paper_precision_config(args: argparse.Namespace, run_payload: dict) -> dict[str, Any]:
    paper_precision = (run_payload.get("evaluation") or {}).get("paper_precision") or {}
    structure_metrics = paper_precision.get("structure_metrics") or {}
    thresholds = (
        [float(value) for value in args.structure_metrics_epsilons]
        if getattr(args, "structure_metrics_epsilons", None) is not None
        else structure_metrics.get("thresholds")
    )
    return {
        "paper_precision_enabled": True,
        "paper_precision_mode": str(paper_precision.get("mode", "per_iteration")),
        "paper_precision_eval_interval": int(paper_precision.get("eval_interval", 1)),
        "paper_precision_eval_checkpoints": paper_precision.get("eval_checkpoints"),
        "paper_precision_num_points": paper_precision.get("num_points"),
        "paper_precision_schedule": str(paper_precision.get("schedule", getattr(args, "paper_precision_schedule", "log"))),
        "paper_precision_num_walks": int(paper_precision.get("num_walks", 100)),
        "paper_precision_walk_length": int(paper_precision.get("walk_length", 10_000)),
        "structure_metrics_enabled": bool(structure_metrics.get("enabled", True)),
        "structure_metrics_epsilon": float(structure_metrics.get("epsilon", 1e-6)),
        "structure_metrics_epsilons": thresholds,
        "structure_metrics_ignore_self_loops": bool(structure_metrics.get("ignore_self_loops", True)),
        "structure_metrics_scope": str(structure_metrics.get("scope", "exact_level")),
        "structure_metrics_ground_truth_source": str(structure_metrics.get("ground_truth_source", "empirical_walks")),
    }


def _hybrid_snapshot_eval_config(
    *,
    args: argparse.Namespace,
    run_root: Path,
    run_id: str,
    run_payload: dict,
    pp_config: dict[str, Any],
    datasets_root: str | Path,
) -> MercuryCSCGHybridConfig:
    cscg_payload = ((run_payload.get("baselines") or {}).get("cscg") or {})
    return MercuryCSCGHybridConfig(
        datasets_root=Path(datasets_root),
        output_root=run_root / "bundles" / "mercury_cscg_hybrid",
        run_id=run_id,
        level=int(run_payload["level"]),
        sensor=str(run_payload["sensor"]),
        sensor_range=run_payload.get("sensor_range"),
        ground_truth_states_parquet=run_root / "ground_truth" / f"{run_id}_ground_truth_states.parquet",
        memory_length=int(run_payload["memory_length"]),
        window_length=int(run_payload["window_length"]),
        seed=int(cscg_payload.get("seed", run_payload["seed"])),
        rand_prob=float(run_payload["rand_prob"]),
        num_steps=int(run_payload["num_steps"]),
        valid_trajectories_only=bool(run_payload.get("valid_trajectories_only", False)),
        mercury_valid_trajectories_only=bool(run_payload.get("mercury_valid_trajectories_only", False)),
        mercury_split_sensory_raw_latent_valid=bool(run_payload.get("mercury_split_sensory_raw_latent_valid", False)),
        mercury_input_mode=str(run_payload.get("mercury_input_mode", "pretrained_raw")),
        sensory=dict(run_payload["sensory"]),
        latent=dict(run_payload["latent"]),
        action_map=dict(run_payload["action_map"]),
        n_iter=int(cscg_payload.get("n_iter", run_payload.get("cscg_n_iter", 1000))),
        pseudocount=float(cscg_payload.get("pseudocount", run_payload.get("cscg_pseudocount", 2e-3))),
        term_early=bool(cscg_payload.get("term_early", run_payload.get("cscg_term_early", True))),
        train_algorithm=str(cscg_payload.get("train_algorithm", run_payload.get("cscg_train_algorithm", "em"))),
        training_mode=str(cscg_payload.get("training_mode", run_payload.get("cscg_training_mode", "full"))),
        batch_size=int(cscg_payload.get("batch_size", run_payload.get("cscg_batch_size", 1024))),
        online_lambda=float(cscg_payload.get("online_lambda", run_payload.get("cscg_online_lambda", 1.0))),
        mercury_prior_weight=float(cscg_payload.get("mercury_cscg_prior_weight", run_payload.get("mercury_cscg_prior_weight", 1.0))),
        eval_viterbi=bool(cscg_payload.get("eval_viterbi", run_payload.get("cscg_eval_viterbi", False))),
        paper_precision_seed=int(run_payload["seed"]),
        paper_precision_rand_prob=float(run_payload["rand_prob"]),
        paper_precision_sensory_params=dict(run_payload["sensory"]),
        paper_precision_action_map_params=dict(run_payload["action_map"]),
        random_start=bool(run_payload.get("random_start", True)),
        **pp_config,
    )


def _normalize_one_hot_sensor_args(args: argparse.Namespace) -> None:
    mercury_input_mode = str(getattr(args, "mercury_input_mode", "pretrained_raw")).strip().lower()
    sensory_observation_mode = str(getattr(args, "sensory_observation_mode", "raw")).strip().lower()
    needs_cardinal_observation = (
        mercury_input_mode in {"pretrained_cscg_one_hot", "direct_cscg_one_hot"}
        or sensory_observation_mode == "cscg_one_hot"
    )
    if not needs_cardinal_observation:
        return
    if str(getattr(args, "sensor", "cartesian")) == "cartesian" and getattr(args, "sensor_range", None) is None:
        args.sensor = "cardinal distance"
        args.sensor_range = 1
        return
    if str(getattr(args, "sensor", "")) != "cardinal distance":
        raise ValueError("CSCG one-hot observations require --sensor 'cardinal distance'.")
    if getattr(args, "sensor_range", None) is None:
        args.sensor_range = 1


def reevaluate_paper_precision_snapshots(args: argparse.Namespace) -> None:
    source_run_id = args.snapshot_source_run_id or args.run_id
    if not source_run_id:
        raise ValueError("--snapshot_source_run_id or --run_id is required for snapshot re-evaluation")
    run_root = Path(args.study_root) / args.study_name / str(source_run_id)
    run_config = load_run_config_model(run_root / "run_config.json")
    if run_config is None:
        raise FileNotFoundError(f"Missing valid run_config.json for {run_root}")
    run_payload = dict(run_config.run_payload)
    method = str(args.method or run_payload.get("method", "mercury"))
    run_id = str(run_config.run_id)
    snapshots = [] if method == "pocml" else list_paper_precision_snapshots(run_root, method)
    if method != "pocml" and not snapshots:
        raise FileNotFoundError(f"No {method} paper-precision snapshots found under {run_root / 'snapshots'}")

    pp_config = _paper_precision_config(args, run_payload)
    if method == "mercury":
        config = MercuryConfig(
            datasets_root=Path((run_config.cli_args or {}).get("datasets_root", args.datasets_root)),
            results_root=run_root / "bundles" / "mercury",
            run_id=run_id,
            level=int(run_payload["level"]),
            sensor=str(run_payload["sensor"]),
            sensor_range=run_payload.get("sensor_range"),
            memory_length=int(run_payload["memory_length"]),
            window_length=int(run_payload["window_length"]),
            seed=int(run_payload["seed"]),
            rand_prob=float(run_payload["rand_prob"]),
            num_steps=int(run_payload["num_steps"]),
            valid_trajectories_only=bool(run_payload.get("valid_trajectories_only", False)),
            mercury_valid_trajectories_only=bool(run_payload.get("mercury_valid_trajectories_only", False)),
            mercury_split_sensory_raw_latent_valid=bool(run_payload.get("mercury_split_sensory_raw_latent_valid", False)),
            random_start=bool(run_payload.get("random_start", True)),
            sensory=dict(run_payload["sensory"]),
            latent=dict(run_payload["latent"]),
            action_map=dict(run_payload["action_map"]),
            **pp_config,
        )
        history: list[dict[str, Any]] = []
        for snapshot_path in snapshots:
            snapshot = load_mercury_paper_precision_snapshot(snapshot_path)
            meta = dict(snapshot["meta"])
            metrics = compute_mercury_paper_precision_metrics(
                config=config,
                sensory_graph=snapshot["sensory_graph"],
                latent_graph=snapshot["latent_graph"],
                action_map=snapshot["action_map"],
                obs_uniques=snapshot.get("obs_uniques"),
            )
            history.append(
                {
                    "step": int(meta.get("step", 0)),
                    "observed_samples": int(meta.get("observed_samples", meta.get("step", 0))),
                    **metrics,
                }
            )
        final_metrics = {key: value for key, value in history[-1].items() if key not in {"step", "observed_samples"}}
        write_mercury_paper_precision_payload(
            config=config,
            metrics=final_metrics,
            history=history,
            schedule_unit="training_step",
        )
        return

    if method == "cscg":
        cscg_payload = ((run_payload.get("baselines") or {}).get("cscg") or {})
        config = CSCGConfig(
            datasets_root=Path((run_config.cli_args or {}).get("datasets_root", args.datasets_root)),
            output_root=run_root / "bundles" / "cscg",
            run_id=run_id,
            level=int(run_payload["level"]),
            sensor=str(run_payload["sensor"]),
            sensor_range=run_payload.get("sensor_range"),
            ground_truth_states_parquet=run_root / "ground_truth" / f"{run_id}_ground_truth_states.parquet",
            window_length=int(run_payload["window_length"]),
            valid_trajectories_only=bool(cscg_payload.get("valid_trajectories_only", False)),
            clones_per_obs=int(cscg_payload.get("clones_per_obs", 10)),
            n_iter=int(cscg_payload.get("n_iter", 1000)),
            pseudocount=float(cscg_payload.get("pseudocount", 2e-3)),
            term_early=bool(cscg_payload.get("term_early", True)),
            train_algorithm=str(cscg_payload.get("train_algorithm", "em")),
            training_mode=str(cscg_payload.get("training_mode", "full")),
            batch_size=int(cscg_payload.get("batch_size", 1024)),
            online_lambda=float(cscg_payload.get("online_lambda", 1.0)),
            seed=int(cscg_payload.get("seed", run_payload["seed"])),
            paper_precision_seed=int(run_payload["seed"]),
            paper_precision_rand_prob=float(run_payload["rand_prob"]),
            paper_precision_sensory_params=dict(run_payload["sensory"]),
            paper_precision_action_map_params=dict(run_payload["action_map"]),
            **pp_config,
        )
        chmm_mod = import_cscg_module(Path.cwd())
        history = []
        for snapshot_path in snapshots:
            snapshot = load_cscg_paper_precision_snapshot(snapshot_path)
            meta = dict(snapshot["meta"])
            metrics = compute_cscg_paper_precision_metrics_from_model(
                model_npz=snapshot["model_npz"],
                n_clones=snapshot["n_clones"],
                config=config,
                obs_uniques=snapshot["obs_uniques"],
                action_uniques=snapshot["action_uniques"],
                chmm_mod=chmm_mod,
            )
            history.append(
                {
                    "stage": str(meta.get("stage", "em")),
                    "iteration": int(meta.get("iteration", meta.get("step", 0))),
                    "step": int(meta.get("step", 0)),
                    "observed_samples": int(meta.get("observed_samples", meta.get("step", 0))),
                    **metrics,
                }
            )
        final_metrics = {
            key: value
            for key, value in history[-1].items()
            if key not in {"stage", "iteration", "step", "observed_samples"}
        }
        write_cscg_paper_precision_payload(
            config=config,
            metrics=final_metrics,
            history=history,
            mode=str(config.paper_precision_mode),
        )
        return

    if method == "sensory":
        config = SensoryConfig(
            datasets_root=Path((run_config.cli_args or {}).get("datasets_root", args.datasets_root)),
            output_root=run_root / "bundles" / "sensory",
            run_id=run_id,
            level=int(run_payload["level"]),
            sensor=str(run_payload["sensor"]),
            sensor_range=run_payload.get("sensor_range"),
            seed=int(run_payload["seed"]),
            rand_prob=float(run_payload["rand_prob"]),
            num_steps=int(run_payload["num_steps"]),
            window_length=int(run_payload["window_length"]),
            valid_trajectories_only=bool(run_payload.get("valid_trajectories_only", False)),
            observation_mode=str(run_payload.get("sensory_observation_mode", "raw")),
            sensory=dict(run_payload["sensory"]),
            action_map=dict(run_payload["action_map"]),
            **pp_config,
        )
        history: list[dict[str, Any]] = []
        for snapshot_path in snapshots:
            snapshot = load_sensory_paper_precision_snapshot(snapshot_path)
            meta = dict(snapshot["meta"])
            metrics = compute_sensory_paper_precision_metrics(
                config=config,
                sensory_graph=snapshot["sensory_graph"],
                action_map=snapshot["action_map"],
                obs_uniques=snapshot.get("obs_uniques"),
            )
            history.append(
                {
                    "step": int(meta.get("step", 0)),
                    "observed_samples": int(meta.get("observed_samples", meta.get("step", 0))),
                    **metrics,
                }
            )
        final_metrics = {key: value for key, value in history[-1].items() if key not in {"step", "observed_samples"}}
        write_sensory_paper_precision_payload(
            config=config,
            metrics=final_metrics,
            history=history,
            schedule_unit="training_step",
        )
        return

    if method == "mercury_cscg_hybrid":
        snapshots = list_paper_precision_snapshots(run_root, "mercury_cscg_hybrid")
        if not snapshots:
            raise FileNotFoundError(f"No mercury_cscg_hybrid paper-precision snapshots found under {run_root / 'snapshots'}")
        config = _hybrid_snapshot_eval_config(
            args=args,
            run_root=run_root,
            run_id=run_id,
            run_payload=run_payload,
            pp_config=pp_config,
            datasets_root=(run_config.cli_args or {}).get("datasets_root", args.datasets_root),
        )
        chmm_mod = import_cscg_module(Path.cwd())
        history = []
        for snapshot_path in snapshots:
            snapshot = load_cscg_paper_precision_snapshot(snapshot_path)
            meta = dict(snapshot["meta"])
            metrics = compute_cscg_paper_precision_metrics_from_model(
                model_npz=snapshot["model_npz"],
                n_clones=snapshot["n_clones"],
                config=config,
                obs_uniques=snapshot["obs_uniques"],
                action_uniques=snapshot["action_uniques"],
                chmm_mod=chmm_mod,
            )
            step = int(meta.get("step", 0))
            history.append(
                {
                    "method": "mercury_cscg_hybrid_cscg",
                    "series_label": "Hybrid CSCG",
                    "phase": "cscg",
                    "stage": str(meta.get("stage", "em")),
                    "iteration": int(meta.get("iteration", step)),
                    "step": step,
                    "observed_samples": int(meta.get("observed_samples", step)),
                    "phase_step": step,
                    "cumulative_step": int(run_payload["num_steps"]) + step,
                    "handoff_step": int(run_payload["num_steps"]),
                    **metrics,
                }
            )
        final_metrics = {
            key: value
            for key, value in history[-1].items()
            if key
            not in {
                "method",
                "series_label",
                "phase",
                "stage",
                "iteration",
                "step",
                "observed_samples",
                "phase_step",
                "cumulative_step",
                "handoff_step",
            }
        }
        write_hybrid_paper_precision_payload(
            config=config,
            metrics=final_metrics,
            history=history,
            mode=str(config.paper_precision_mode),
        )
        return

    if method == "pocml":
        pocml_payload = ((run_payload.get("baselines") or {}).get("pocml") or {})
        config = POCMLConfig(
            datasets_root=Path((run_config.cli_args or {}).get("datasets_root", args.datasets_root)),
            output_root=run_root / "bundles" / "pocml",
            run_id=run_id,
            level=int(run_payload["level"]),
            sensor=str(run_payload["sensor"]),
            sensor_range=run_payload.get("sensor_range"),
            ground_truth_states_parquet=run_root / "ground_truth" / f"{run_id}_ground_truth_states.parquet",
            window_length=int(run_payload.get("window_length", 1)),
            valid_trajectories_only=bool(pocml_payload.get("valid_trajectories_only", run_payload.get("valid_trajectories_only", False))),
            n_states=pocml_payload.get("n_states", run_payload.get("pocml_n_states")),
            state_dim=int(pocml_payload.get("state_dim", run_payload.get("pocml_state_dim", 64))),
            random_feature_dim=int(pocml_payload.get("random_feature_dim", run_payload.get("pocml_random_feature_dim", 256))),
            alpha=float(pocml_payload.get("alpha", run_payload.get("pocml_alpha", 0.2))),
            memory_bias=bool(pocml_payload.get("memory_bias", run_payload.get("pocml_memory_bias", True))),
            paper_precision_seed=int(run_payload["seed"]),
            paper_precision_rand_prob=float(run_payload["rand_prob"]),
            paper_precision_sensory_params=dict(run_payload["sensory"]),
            paper_precision_action_map_params=dict(run_payload["action_map"]),
            **pp_config,
        )
        snapshots_by_capacity = list_pocml_paper_precision_snapshots(run_root)
        if not snapshots_by_capacity:
            raise FileNotFoundError(f"No pocml paper-precision snapshots found under {run_root / 'snapshots'}")
        model_mod, _ = import_pocml_modules(Path.cwd())
        import torch as _torch
        history_by_capacity: dict[str, list[dict[str, Any]]] = {}
        metrics_by_capacity: dict[str, dict[str, Any]] = {}
        resolved_capacities = sorted(snapshots_by_capacity.keys())
        primary_capacity = int(resolved_capacities[0])
        for n_states, step_dirs in snapshots_by_capacity.items():
            capacity_key = str(int(n_states))
            history_by_capacity[capacity_key] = []
            for snapshot_path in step_dirs:
                snapshot = load_pocml_paper_precision_snapshot(snapshot_path)
                meta = dict(snapshot["meta"])
                model_arch = dict(meta.get("model_arch") or {})
                model = model_mod.POCML(
                    n_obs=int(model_arch.get("n_obs", snapshot["obs_uniques"].shape[0])),
                    n_states=int(model_arch.get("n_states", n_states)),
                    n_actions=int(model_arch.get("n_actions", snapshot["action_uniques"].shape[0])),
                    state_dim=int(model_arch.get("state_dim", int(config.state_dim))),
                    random_feature_dim=int(model_arch.get("random_feature_dim", int(config.random_feature_dim))),
                    alpha=float(model_arch.get("alpha", float(config.alpha))),
                    batch_size=1,
                    memory_bias=bool(model_arch.get("memory_bias", bool(config.memory_bias))),
                )
                state_dict = _torch.load(str(snapshot["model_state_dict_path"]), map_location="cpu", weights_only=True)
                model.load_state_dict(state_dict)
                model = prepare_model_for_single_sequence_eval(model)
                metrics = compute_pocml_paper_precision_metrics_from_model(
                    model=model,
                    config=config,
                    obs_uniques=snapshot["obs_uniques"],
                    action_uniques=snapshot["action_uniques"],
                    torch_module=_torch,
                )
                history_by_capacity[capacity_key].append(
                    {
                        "step": int(meta.get("step", 0)),
                        "observed_samples": int(meta.get("observed_samples", meta.get("step", 0))),
                        **metrics,
                    }
                )
            metrics_by_capacity[capacity_key] = {
                key: value
                for key, value in history_by_capacity[capacity_key][-1].items()
                if key not in {"step", "observed_samples"}
            }
        write_pocml_paper_precision_payload(
            config=config,
            metrics_by_capacity=metrics_by_capacity,
            history_by_capacity=history_by_capacity,
            mode=str(config.paper_precision_mode),
            resolved_capacities=resolved_capacities,
            primary_capacity=primary_capacity,
        )
        return

    raise ValueError(f"Snapshot re-evaluation is not supported for method {method!r}")


def evaluate_study_snapshots(args: argparse.Namespace) -> None:
    """Post-training evaluation for all snapshot-based methods at study level.

    Iterates over every run directory in the study, finds paper-precision
    snapshots for the relevant method, and evaluates each snapshot.
    Writes the canonical payload (history + final metrics) to each run's metrics directory.
    """
    study_root = Path(args.study_root) / args.study_name
    if not study_root.exists():
        raise FileNotFoundError(f"Study directory not found: {study_root}")

    run_dirs = sorted(
        [d for d in study_root.iterdir() if d.is_dir() and (d / "run_config.json").exists()]
    )
    if not run_dirs:
        raise FileNotFoundError(f"No run directories found under {study_root}")

    evaluated = 0
    skipped = 0
    for run_root in tqdm(run_dirs, desc=f"eval {args.method} lvl={args.level}", unit="run", dynamic_ncols=True):
        run_config = load_run_config_model(run_root / "run_config.json")
        if run_config is None:
            skipped += 1
            continue

        run_payload = dict(run_config.run_payload)
        run_id = str(run_config.run_id)
        run_method = str(run_payload.get("method", "mercury"))
        target_method = str(args.method or run_method)
        datasets_root = str((run_config.cli_args or {}).get("datasets_root", getattr(args, "datasets_root", "datasets")))
        pp_config = _paper_precision_config(args, run_payload)

        if target_method == "mercury":
            snapshots = list_paper_precision_snapshots(run_root, "mercury")
            if not snapshots:
                skipped += 1
                continue
            config = MercuryConfig(
                datasets_root=Path(datasets_root),
                results_root=run_root / "bundles" / "mercury",
                run_id=run_id,
                level=int(run_payload["level"]),
                sensor=str(run_payload["sensor"]),
                sensor_range=run_payload.get("sensor_range"),
                memory_length=int(run_payload["memory_length"]),
                window_length=int(run_payload["window_length"]),
                seed=int(run_payload["seed"]),
                rand_prob=float(run_payload["rand_prob"]),
                num_steps=int(run_payload["num_steps"]),
                valid_trajectories_only=bool(run_payload.get("valid_trajectories_only", False)),
                mercury_valid_trajectories_only=bool(run_payload.get("mercury_valid_trajectories_only", False)),
                mercury_split_sensory_raw_latent_valid=bool(run_payload.get("mercury_split_sensory_raw_latent_valid", False)),
                random_start=bool(run_payload.get("random_start", True)),
                sensory=dict(run_payload["sensory"]),
                latent=dict(run_payload["latent"]),
                action_map=dict(run_payload["action_map"]),
                **pp_config,
            )
            history: list[dict[str, Any]] = []
            for snapshot_path in snapshots:
                snapshot = load_mercury_paper_precision_snapshot(snapshot_path)
                meta = dict(snapshot["meta"])
                metrics = compute_mercury_paper_precision_metrics(
                    config=config,
                    sensory_graph=snapshot["sensory_graph"],
                    latent_graph=snapshot["latent_graph"],
                    action_map=snapshot["action_map"],
                    obs_uniques=snapshot.get("obs_uniques"),
                )
                history.append(
                    {
                        "step": int(meta.get("step", 0)),
                        "observed_samples": int(meta.get("observed_samples", meta.get("step", 0))),
                        **metrics,
                    }
                )
            final_metrics = {key: value for key, value in history[-1].items() if key not in {"step", "observed_samples"}}
            write_mercury_paper_precision_payload(
                config=config,
                metrics=final_metrics,
                history=history,
                schedule_unit="training_step",
            )
            evaluated += 1

        elif target_method == "sensory":
            snapshots = list_paper_precision_snapshots(run_root, "sensory")
            if not snapshots:
                skipped += 1
                continue
            config = SensoryConfig(
                datasets_root=Path(datasets_root),
                output_root=run_root / "bundles" / "sensory",
                run_id=run_id,
                level=int(run_payload["level"]),
                sensor=str(run_payload["sensor"]),
                sensor_range=run_payload.get("sensor_range"),
                seed=int(run_payload["seed"]),
                rand_prob=float(run_payload["rand_prob"]),
                num_steps=int(run_payload["num_steps"]),
                window_length=int(run_payload["window_length"]),
                valid_trajectories_only=bool(run_payload.get("valid_trajectories_only", False)),
                observation_mode=str(run_payload.get("sensory_observation_mode", "raw")),
                sensory=dict(run_payload["sensory"]),
                action_map=dict(run_payload["action_map"]),
                **pp_config,
            )
            history: list[dict[str, Any]] = []
            for snapshot_path in snapshots:
                snapshot = load_sensory_paper_precision_snapshot(snapshot_path)
                meta = dict(snapshot["meta"])
                metrics = compute_sensory_paper_precision_metrics(
                    config=config,
                    sensory_graph=snapshot["sensory_graph"],
                    action_map=snapshot["action_map"],
                    obs_uniques=snapshot.get("obs_uniques"),
                )
                history.append(
                    {
                        "step": int(meta.get("step", 0)),
                        "observed_samples": int(meta.get("observed_samples", meta.get("step", 0))),
                        **metrics,
                    }
                )
            final_metrics = {key: value for key, value in history[-1].items() if key not in {"step", "observed_samples"}}
            write_sensory_paper_precision_payload(
                config=config,
                metrics=final_metrics,
                history=history,
                schedule_unit="training_step",
            )
            evaluated += 1

        elif target_method == "cscg":
            snapshots = list_paper_precision_snapshots(run_root, "cscg")
            if not snapshots:
                skipped += 1
                continue
            cscg_payload = ((run_payload.get("baselines") or {}).get("cscg") or {})
            config = CSCGConfig(
                datasets_root=Path(datasets_root),
                output_root=run_root / "bundles" / "cscg",
                run_id=run_id,
                level=int(run_payload["level"]),
                sensor=str(run_payload["sensor"]),
                sensor_range=run_payload.get("sensor_range"),
                ground_truth_states_parquet=run_root / "ground_truth" / f"{run_id}_ground_truth_states.parquet",
                window_length=int(run_payload["window_length"]),
                valid_trajectories_only=bool(cscg_payload.get("valid_trajectories_only", run_payload.get("valid_trajectories_only", False))),
                clones_per_obs=int(cscg_payload.get("clones_per_obs", run_payload.get("cscg_clones_per_obs", 10))),
                n_iter=int(cscg_payload.get("n_iter", run_payload.get("cscg_n_iter", 1000))),
                pseudocount=float(cscg_payload.get("pseudocount", run_payload.get("cscg_pseudocount", 2e-3))),
                term_early=bool(cscg_payload.get("term_early", run_payload.get("cscg_term_early", True))),
                train_algorithm=str(cscg_payload.get("train_algorithm", run_payload.get("cscg_train_algorithm", "em"))),
                training_mode=str(cscg_payload.get("training_mode", run_payload.get("cscg_training_mode", "full"))),
                batch_size=int(cscg_payload.get("batch_size", run_payload.get("cscg_batch_size", 1024))),
                online_lambda=float(cscg_payload.get("online_lambda", run_payload.get("cscg_online_lambda", 1.0))),
                seed=int(cscg_payload.get("seed", run_payload["seed"])),
                paper_precision_seed=int(run_payload["seed"]),
                paper_precision_rand_prob=float(run_payload["rand_prob"]),
                paper_precision_sensory_params=dict(run_payload["sensory"]),
                paper_precision_action_map_params=dict(run_payload["action_map"]),
                **pp_config,
            )
            chmm_mod = import_cscg_module(Path.cwd())
            history = []
            for snapshot_path in snapshots:
                snapshot = load_cscg_paper_precision_snapshot(snapshot_path)
                meta = dict(snapshot["meta"])
                metrics = compute_cscg_paper_precision_metrics_from_model(
                    model_npz=snapshot["model_npz"],
                    n_clones=snapshot["n_clones"],
                    config=config,
                    obs_uniques=snapshot["obs_uniques"],
                    action_uniques=snapshot["action_uniques"],
                    chmm_mod=chmm_mod,
                )
                history.append(
                    {
                        "stage": str(meta.get("stage", "em")),
                        "iteration": int(meta.get("iteration", meta.get("step", 0))),
                        "step": int(meta.get("step", 0)),
                        "observed_samples": int(meta.get("observed_samples", meta.get("step", 0))),
                        **metrics,
                    }
                )
            final_metrics = {
                key: value
                for key, value in history[-1].items()
                if key not in {"stage", "iteration", "step", "observed_samples"}
            }
            write_cscg_paper_precision_payload(
                config=config,
                metrics=final_metrics,
                history=history,
                mode=str(config.paper_precision_mode),
            )
            evaluated += 1

        elif target_method == "mercury_cscg_hybrid":
            snapshots = list_paper_precision_snapshots(run_root, "mercury_cscg_hybrid")
            if not snapshots:
                skipped += 1
                continue
            config = _hybrid_snapshot_eval_config(
                args=args,
                run_root=run_root,
                run_id=run_id,
                run_payload=run_payload,
                pp_config=pp_config,
                datasets_root=datasets_root,
            )
            chmm_mod = import_cscg_module(Path.cwd())
            history = []
            for snapshot_path in snapshots:
                snapshot = load_cscg_paper_precision_snapshot(snapshot_path)
                meta = dict(snapshot["meta"])
                metrics = compute_cscg_paper_precision_metrics_from_model(
                    model_npz=snapshot["model_npz"],
                    n_clones=snapshot["n_clones"],
                    config=config,
                    obs_uniques=snapshot["obs_uniques"],
                    action_uniques=snapshot["action_uniques"],
                    chmm_mod=chmm_mod,
                )
                step = int(meta.get("step", 0))
                history.append(
                    {
                        "method": "mercury_cscg_hybrid_cscg",
                        "series_label": "Hybrid CSCG",
                        "phase": "cscg",
                        "stage": str(meta.get("stage", "em")),
                        "iteration": int(meta.get("iteration", step)),
                        "step": step,
                        "observed_samples": int(meta.get("observed_samples", step)),
                        "phase_step": step,
                        "cumulative_step": int(run_payload["num_steps"]) + step,
                        "handoff_step": int(run_payload["num_steps"]),
                        **metrics,
                    }
                )
            final_metrics = {
                key: value
                for key, value in history[-1].items()
                if key
                not in {
                    "method",
                    "series_label",
                    "phase",
                    "stage",
                    "iteration",
                    "step",
                    "observed_samples",
                    "phase_step",
                    "cumulative_step",
                    "handoff_step",
                }
            }
            write_hybrid_paper_precision_payload(
                config=config,
                metrics=final_metrics,
                history=history,
                mode=str(config.paper_precision_mode),
            )
            evaluated += 1

        elif target_method == "pocml":
            snapshots_by_capacity = list_pocml_paper_precision_snapshots(run_root)
            if not snapshots_by_capacity:
                skipped += 1
                continue
            pocml_payload = ((run_payload.get("baselines") or {}).get("pocml") or {})
            config = POCMLConfig(
                datasets_root=Path(datasets_root),
                output_root=run_root / "bundles" / "pocml",
                run_id=run_id,
                level=int(run_payload["level"]),
                sensor=str(run_payload["sensor"]),
                sensor_range=run_payload.get("sensor_range"),
                ground_truth_states_parquet=run_root / "ground_truth" / f"{run_id}_ground_truth_states.parquet",
                window_length=int(run_payload.get("window_length", 1)),
                valid_trajectories_only=bool(pocml_payload.get("valid_trajectories_only", run_payload.get("valid_trajectories_only", False))),
                n_states=pocml_payload.get("n_states", run_payload.get("pocml_n_states")),
                state_dim=int(pocml_payload.get("state_dim", run_payload.get("pocml_state_dim", 64))),
                random_feature_dim=int(pocml_payload.get("random_feature_dim", run_payload.get("pocml_random_feature_dim", 256))),
                alpha=float(pocml_payload.get("alpha", run_payload.get("pocml_alpha", 0.2))),
                memory_bias=bool(pocml_payload.get("memory_bias", run_payload.get("pocml_memory_bias", True))),
                paper_precision_seed=int(run_payload["seed"]),
                paper_precision_rand_prob=float(run_payload["rand_prob"]),
                paper_precision_sensory_params=dict(run_payload["sensory"]),
                paper_precision_action_map_params=dict(run_payload["action_map"]),
                **pp_config,
            )
            model_mod, _ = import_pocml_modules(Path.cwd())
            import torch as _torch
            history_by_capacity: dict[str, list[dict[str, Any]]] = {}
            metrics_by_capacity: dict[str, dict[str, Any]] = {}
            resolved_capacities = sorted(snapshots_by_capacity.keys())
            primary_capacity = int(resolved_capacities[0])
            for n_states, step_dirs in snapshots_by_capacity.items():
                capacity_key = str(int(n_states))
                history_by_capacity[capacity_key] = []
                for snapshot_path in step_dirs:
                    snapshot = load_pocml_paper_precision_snapshot(snapshot_path)
                    meta = dict(snapshot["meta"])
                    model_arch = dict(meta.get("model_arch") or {})
                    model = model_mod.POCML(
                        n_obs=int(model_arch.get("n_obs", snapshot["obs_uniques"].shape[0])),
                        n_states=int(model_arch.get("n_states", n_states)),
                        n_actions=int(model_arch.get("n_actions", snapshot["action_uniques"].shape[0])),
                        state_dim=int(model_arch.get("state_dim", int(config.state_dim))),
                        random_feature_dim=int(model_arch.get("random_feature_dim", int(config.random_feature_dim))),
                        alpha=float(model_arch.get("alpha", float(config.alpha))),
                        batch_size=1,
                        memory_bias=bool(model_arch.get("memory_bias", bool(config.memory_bias))),
                    )
                    state_dict = _torch.load(str(snapshot["model_state_dict_path"]), map_location="cpu", weights_only=True)
                    model.load_state_dict(state_dict)
                    model = prepare_model_for_single_sequence_eval(model)
                    metrics = compute_pocml_paper_precision_metrics_from_model(
                        model=model,
                        config=config,
                        obs_uniques=snapshot["obs_uniques"],
                        action_uniques=snapshot["action_uniques"],
                        torch_module=_torch,
                    )
                    history_by_capacity[capacity_key].append(
                        {
                            "step": int(meta.get("step", 0)),
                            "observed_samples": int(meta.get("observed_samples", meta.get("step", 0))),
                            **metrics,
                        }
                    )
                metrics_by_capacity[capacity_key] = {
                    key: value
                    for key, value in history_by_capacity[capacity_key][-1].items()
                    if key not in {"step", "observed_samples"}
                }
            write_pocml_paper_precision_payload(
                config=config,
                metrics_by_capacity=metrics_by_capacity,
                history_by_capacity=history_by_capacity,
                mode=str(config.paper_precision_mode),
                resolved_capacities=resolved_capacities,
                primary_capacity=primary_capacity,
            )
            evaluated += 1

        else:
            skipped += 1

    print(f"evaluate_study_snapshots: evaluated {evaluated} runs, skipped {skipped}")


def run_single(args: argparse.Namespace) -> None:
    _normalize_one_hot_sensor_args(args)
    if bool(getattr(args, "reevaluate_paper_precision_snapshots", False)):
        reevaluate_paper_precision_snapshots(args)
        return

    if args.level is None:
        raise ValueError("--level is required for single-run mode")

    reuse_existing_run = bool(args.reuse_existing_run) and (not bool(args.no_reuse_existing_run))
    reuse_existing_dataset = not bool(args.no_reuse_existing_dataset)
    base_run_payload = build_run_payload(args)

    if args._worker_run:
        run_id = str(args.run_id)
        execute_pipeline_run(
            args=args,
            run_id=run_id,
            requested_run_id=run_id,
            run_root=Path(args.study_root) / args.study_name / run_id,
            run_payload=base_run_payload,
            reuse_existing_run=reuse_existing_run,
            reuse_existing_dataset=reuse_existing_dataset,
        )
        return

    identity = resolve_run_identity(args=args, run_payload=base_run_payload, reuse_existing_run=reuse_existing_run)
    run_payload = dict(base_run_payload)
    run_payload["requested_run_id"] = identity.requested_run_id
    run_root = identity.run_root
    bundles_root = _run_bundles_root(run_root, run_payload)
    metrics_root = run_root / "metrics"
    layout = artifact_layout(run_root=run_root, run_id=identity.run_id, run_payload=run_payload)
    run_root.mkdir(parents=True, exist_ok=True)

    run_config = build_run_config_payload(
        args=args,
        run_id=identity.run_id,
        run_payload=run_payload,
        reuse_existing_run=reuse_existing_run,
        reuse_existing_dataset=reuse_existing_dataset,
        run_root=run_root,
        bundles_root=bundles_root,
        metrics_root=metrics_root,
    )

    if identity.is_reused:
        existing_status = load_run_status_model(layout.run_status_json)
        if existing_status is None:
            raise RuntimeError(f"Reusable run {identity.run_id} is missing a valid run_status.json")
        existing_status.status = "reused"
        existing_status.finished_at_utc = utc_now()
        atomic_write_json(layout.run_status_json, existing_status.model_dump(mode="json"))
        method = str(run_payload.get("method", "mercury"))
        print(f"Reused {method} run: {identity.run_id}")
        print(f"Run status: {layout.run_status_json}")
        maybe_log_wandb(
            args=args,
            run_id=identity.run_id,
            run_status=existing_status.model_dump(mode="json"),
            run_config=run_config.model_dump(mode="json"),
        )
        return

    initial_status = build_run_status_model(
        args=args,
        run_id=identity.run_id,
        requested_run_id=identity.requested_run_id,
        run_payload=run_payload,
        run_root=run_root,
        bundles_root=bundles_root,
        metrics_root=metrics_root,
        reuse_existing_run=reuse_existing_run,
        reuse_existing_dataset=reuse_existing_dataset,
        status="started",
    )
    atomic_write_json(layout.run_config_json, run_config.model_dump(mode="json"))
    atomic_write_json(layout.run_status_json, initial_status.model_dump(mode="json"))

    try:
        _run_worker_subprocess(args=args, run_id=identity.run_id)
    except Exception as exc:
        failed_status = load_run_status_model(layout.run_status_json) or initial_status
        failed_status.status = "failed"
        failed_status.finished_at_utc = utc_now()
        failed_status.error = repr(exc)
        atomic_write_json(layout.run_status_json, failed_status.model_dump(mode="json"))
        maybe_log_wandb(
            args=args,
            run_id=identity.run_id,
            run_status=failed_status.model_dump(mode="json"),
            run_config=run_config.model_dump(mode="json"),
        )
        raise

    final_status = load_run_status_model(layout.run_status_json)
    if final_status is None:
        raise RuntimeError(f"Worker run {identity.run_id} completed without a valid run_status.json")
    method = str(run_payload.get("method", "mercury"))
    print(f"Completed {method} run: {identity.run_id}")
    print(f"Run status: {layout.run_status_json}")
    subsystem = final_status.subsystems.get(method)
    if subsystem is not None:
        states_path = subsystem.artifacts.get("states_parquet")
        metrics_path = subsystem.artifacts.get("paper_precision_json")
        snapshots_path = subsystem.artifacts.get("paper_precision_snapshots_dir")
        if states_path:
            print(f"States: {states_path}")
        if metrics_path:
            print(f"Paper precision: {metrics_path}")
        if snapshots_path:
            print(f"Snapshots: {snapshots_path}")
    maybe_log_wandb(
        args=args,
        run_id=identity.run_id,
        run_status=final_status.model_dump(mode="json"),
        run_config=run_config.model_dump(mode="json"),
    )
