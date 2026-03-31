from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

from ..algorithms.cscg import CSCGConfig, run_cscg
from ..algorithms.mercury import MercuryConfig, run_mercury
from ..algorithms.mercury.api import summarize_mercury_run
from ..algorithms.pocml import POCMLConfig, run_pocml
from ..analysis import generate_bmu_attribution_plots, generate_method_graph_plots
from ..domain.models import ArtifactLayout, RunStatusModel
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
    load_run_status_model,
    maybe_log_wandb,
    peak_rss_mb,
    read_native_metrics_from_bundle,
    resolve_run_identity,
    safe_artifact_bytes,
    write_computational_artifact,
    utc_now,
)
from ..infrastructure.progress import write_progress
from ..infrastructure.reporting import write_method_comparison_report
from ..save_results import read_bundle_meta
from .status_service import (
    build_run_status_model,
    mark_run_completed,
    mark_run_failed,
    persist_run_status,
)


def _baseline_dataset_selection(states_parquet: Path) -> tuple[str, str | None]:
    meta = read_bundle_meta(states_parquet)
    if meta is None:
        return "latest", None
    source = meta.source
    if source.select == "run_id" and source.dataset_run_id:
        return "run_id", str(source.dataset_run_id)
    return "latest", None


def _computational_profile(run_payload: dict) -> str:
    return str((((run_payload.get("evaluation") or {}).get("computational") or {}).get("profile", "full")))


def _computational_deployable_paths(method: str, layout: ArtifactLayout) -> list[Path]:
    if method == "mercury":
        return [layout.mercury_states_parquet, layout.mercury_latent_graph_npz]
    if method == "pocml":
        return [layout.pocml_states_parquet, layout.pocml_embeddings_npz, layout.pocml_model_checkpoint_pt]
    if method == "cscg":
        return [layout.cscg_states_parquet, layout.cscg_model_npz]
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
    return []


def _computational_artifact_path(method: str, layout: ArtifactLayout) -> Path:
    if method == "mercury":
        return layout.mercury_computational_json
    if method == "pocml":
        return layout.pocml_computational_json
    if method == "cscg":
        return layout.cscg_computational_json
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
    if method == "cscg":
        return "train" if stage.startswith("train_") else "eval"
    return "lifecycle"


def _resolve_eval_steps(method: str, native_metrics: Dict[str, Any]) -> int | None:
    if method == "mercury":
        return int(((native_metrics.get("latent") or {}).get("steps"))) if ((native_metrics.get("latent") or {}).get("steps")) is not None else None
    if method == "pocml":
        return int(native_metrics.get("next_obs_steps")) if native_metrics.get("next_obs_steps") is not None else None
    if method == "cscg":
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
    include_jit_note = method == "cscg"
    if method == "cscg":
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
            reuse_existing_dataset=reuse_existing_dataset,
            reuse_existing_run=reuse_existing_run,
            sensory=run_payload["sensory"],
            latent=run_payload["latent"],
            action_map=run_payload["action_map"],
            paper_precision_enabled=bool((run_payload.get("evaluation") or {}).get("paper_precision", {}).get("enabled", False)),
            paper_precision_mode=str((run_payload.get("evaluation") or {}).get("paper_precision", {}).get("mode", "off")),
            paper_precision_eval_interval=int((run_payload.get("evaluation") or {}).get("paper_precision", {}).get("eval_interval", 1)),
            paper_precision_num_points=(run_payload.get("evaluation") or {}).get("paper_precision", {}).get("num_points"),
            paper_precision_num_walks=int((run_payload.get("evaluation") or {}).get("paper_precision", {}).get("num_walks", 100)),
            paper_precision_walk_length=int((run_payload.get("evaluation") or {}).get("paper_precision", {}).get("walk_length", 10_000)),
            structure_metrics_enabled=bool(((((run_payload.get("evaluation") or {}).get("paper_precision") or {}).get("structure_metrics") or {}).get("enabled", True))),
            structure_metrics_epsilon=float(((((run_payload.get("evaluation") or {}).get("paper_precision") or {}).get("structure_metrics") or {}).get("epsilon", 1e-6))),
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
        "latent_graph_npz": str(layout.mercury_latent_graph_npz),
    }
    run_status.subsystems["mercury"].native_metrics = summarize_mercury_run(states_parquet)
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


def _run_baseline_stages(
    *,
    args: argparse.Namespace,
    run_root: Path,
    run_id: str,
    states_parquet: Path,
    run_payload: dict,
    run_status: RunStatusModel,
    layout: ArtifactLayout,
) -> None:
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
                paper_precision_num_points=(run_payload.get("evaluation") or {}).get("paper_precision", {}).get("num_points"),
                paper_precision_num_walks=int((run_payload.get("evaluation") or {}).get("paper_precision", {}).get("num_walks", 100)),
                paper_precision_walk_length=int((run_payload.get("evaluation") or {}).get("paper_precision", {}).get("walk_length", 10_000)),
                paper_precision_seed=args.seed,
                paper_precision_rand_prob=args.rand_prob,
                paper_precision_sensory_params=run_payload["sensory"],
                paper_precision_action_map_params=run_payload["action_map"],
                structure_metrics_enabled=bool(((((run_payload.get("evaluation") or {}).get("paper_precision") or {}).get("structure_metrics") or {}).get("enabled", True))),
                structure_metrics_epsilon=float(((((run_payload.get("evaluation") or {}).get("paper_precision") or {}).get("structure_metrics") or {}).get("epsilon", 1e-6))),
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
                n_iter=args.cscg_n_iter,
                pseudocount=args.cscg_pseudocount,
                term_early=args.cscg_term_early,
                train_algorithm=args.cscg_train_algorithm,
                training_mode=args.cscg_training_mode,
                batch_size=args.cscg_batch_size,
                online_lambda=args.cscg_online_lambda,
                seed=(args.cscg_seed if args.cscg_seed is not None else args.seed),
                paper_precision_enabled=bool((run_payload.get("evaluation") or {}).get("paper_precision", {}).get("enabled", False)),
                paper_precision_mode=str((run_payload.get("evaluation") or {}).get("paper_precision", {}).get("mode", "off")),
                paper_precision_eval_interval=int((run_payload.get("evaluation") or {}).get("paper_precision", {}).get("eval_interval", 1)),
                paper_precision_num_points=(run_payload.get("evaluation") or {}).get("paper_precision", {}).get("num_points"),
                paper_precision_num_walks=int((run_payload.get("evaluation") or {}).get("paper_precision", {}).get("num_walks", 100)),
                paper_precision_walk_length=int((run_payload.get("evaluation") or {}).get("paper_precision", {}).get("walk_length", 10_000)),
                paper_precision_seed=args.seed,
                paper_precision_rand_prob=args.rand_prob,
                paper_precision_sensory_params=run_payload["sensory"],
                paper_precision_action_map_params=run_payload["action_map"],
                structure_metrics_enabled=bool(((((run_payload.get("evaluation") or {}).get("paper_precision") or {}).get("structure_metrics") or {}).get("enabled", True))),
                structure_metrics_epsilon=float(((((run_payload.get("evaluation") or {}).get("paper_precision") or {}).get("structure_metrics") or {}).get("epsilon", 1e-6))),
                structure_metrics_scope=str(((((run_payload.get("evaluation") or {}).get("paper_precision") or {}).get("structure_metrics") or {}).get("scope", "exact_level"))),
                structure_metrics_ground_truth_source=str(((((run_payload.get("evaluation") or {}).get("paper_precision") or {}).get("structure_metrics") or {}).get("ground_truth_source", "empirical_walks"))),
            ),
            progress_callback=_cscg_progress_callback,
        )
        run_status.subsystems["cscg"].status = "completed"
        run_status.subsystems["cscg"].artifacts = {
            "states_parquet": str(cscg_states_path),
            "paper_precision_json": str(layout.cscg_paper_precision_json),
            "model_npz": str(layout.cscg_model_npz),
        }
        run_status.subsystems["cscg"].native_metrics = read_native_metrics_from_bundle(Path(cscg_states_path), "cscg_eval")
        _write_computational_metrics(
            method="cscg",
            layout=layout,
            run_payload=run_payload,
            run_status=run_status,
            timing=cscg_timer.finish(),
            train_steps=int(args.cscg_n_iter),
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

    if baseline_enabled(run_payload, "pocml") or baseline_enabled(run_payload, "cscg"):
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
    bundles_root = run_root / "bundles" / "mercury"
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


def run_single(args: argparse.Namespace) -> None:
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
    bundles_root = run_root / "bundles" / "mercury"
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
    maybe_log_wandb(
        args=args,
        run_id=identity.run_id,
        run_status=final_status.model_dump(mode="json"),
        run_config=run_config.model_dump(mode="json"),
    )
