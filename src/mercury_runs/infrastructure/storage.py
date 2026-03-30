from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from ..domain.models import (
    ArtifactLayout,
    PIPELINE_SCHEMA_VERSION,
    RunConfigModel,
    RunIdentity,
    RunStatusModel,
    SubsystemStatus,
)
from ..save_results import read_bundle_meta
from .runtime import args_from_config, compact_utc_now, load_json, utc_now, stable_hash


def prepare_study_logs(study_directory: Path, *, keep_existing: bool) -> tuple[Path, Path]:
    manifest_path = study_directory / "study_manifest.jsonl"
    errors_path = study_directory / "study_errors.jsonl"
    if not keep_existing:
        if manifest_path.exists():
            manifest_path.unlink()
        if errors_path.exists():
            errors_path.unlink()
    return manifest_path, errors_path


def load_run_status_model(path: Path) -> Optional[RunStatusModel]:
    payload = load_json(path)
    if payload is None:
        return None
    try:
        return RunStatusModel.model_validate(payload)
    except Exception:
        return None


def load_run_config_model(path: Path) -> Optional[RunConfigModel]:
    payload = load_json(path)
    if payload is None:
        return None
    try:
        return RunConfigModel.model_validate(payload)
    except Exception:
        return None


def method_output_dir(root: Path, level: int, sensor: str, sensor_range: Optional[int]) -> Path:
    base = root / f"level={level}" / f"sensor={sensor}"
    if sensor == "cardinal distance":
        return base / f"range={sensor_range}"
    return base


def artifact_layout(*, run_root: Path, run_id: str, run_payload: Dict[str, Any]) -> ArtifactLayout:
    level = int(run_payload["level"])
    sensor = str(run_payload["sensor"])
    sensor_range = run_payload.get("sensor_range")
    mercury_dir = method_output_dir(run_root / "bundles" / "mercury", level, sensor, sensor_range)
    pocml_dir = method_output_dir(run_root / "bundles" / "pocml", level, sensor, sensor_range)
    cscg_dir = method_output_dir(run_root / "bundles" / "cscg", level, sensor, sensor_range)
    return ArtifactLayout(
        mercury_states_parquet=mercury_dir / f"{run_id}_states.parquet",
        mercury_paper_precision_json=(run_root / "metrics" / "mercury" / f"{run_id}_paper_precision.json"),
        mercury_computational_json=(run_root / "metrics" / "mercury" / f"{run_id}_computational.json"),
        mercury_latent_graph_npz=mercury_dir / f"{run_id}_latent_graph.npz",
        mercury_internal_graph_png=run_root / "plots" / "mercury_internal_graph.png",
        pocml_states_parquet=pocml_dir / f"{run_id}_states.parquet",
        pocml_paper_precision_json=(run_root / "metrics" / "pocml" / f"{run_id}_paper_precision.json"),
        pocml_computational_json=(run_root / "metrics" / "pocml" / f"{run_id}_computational.json"),
        pocml_embeddings_npz=pocml_dir / f"{run_id}_embeddings.npz",
        pocml_model_checkpoint_pt=pocml_dir / f"{run_id}_pocml_model.pt",
        pocml_internal_graph_png=run_root / "plots" / "pocml_internal_graph.png",
        cscg_states_parquet=cscg_dir / f"{run_id}_states.parquet",
        cscg_paper_precision_json=(run_root / "metrics" / "cscg" / f"{run_id}_paper_precision.json"),
        cscg_computational_json=(run_root / "metrics" / "cscg" / f"{run_id}_computational.json"),
        cscg_model_npz=cscg_dir / f"{run_id}_cscg_model.npz",
        cscg_internal_graph_png=run_root / "plots" / "cscg_internal_graph.png",
        internal_graphs_comparison_png=run_root / "plots" / "internal_graphs_comparison.png",
        comparison_summary_json=run_root / "comparison_summary.json",
        run_config_json=run_root / "run_config.json",
        run_status_json=run_root / "run_status.json",
    )


def path_if_exists(path: Path) -> Optional[str]:
    return str(path) if path.exists() else None


def baseline_enabled(run_payload: Dict[str, Any], baseline_name: str) -> bool:
    return bool(((run_payload.get("baselines") or {}).get(baseline_name) or {}).get("enabled", False))


def paper_precision_enabled(run_payload: Dict[str, Any]) -> bool:
    return bool(((run_payload.get("evaluation") or {}).get("paper_precision") or {}).get("enabled", False))


def paper_precision_mode(run_payload: Dict[str, Any]) -> str:
    return str((((run_payload.get("evaluation") or {}).get("paper_precision") or {}).get("mode", "off")))


def computational_eval_enabled(run_payload: Dict[str, Any]) -> bool:
    return bool(((run_payload.get("evaluation") or {}).get("computational") or {}).get("enabled", False))


def subsystem_status(
    *,
    name: str,
    status: str,
    artifacts: Optional[Dict[str, Optional[str]]] = None,
    native_metrics: Optional[Dict[str, Any]] = None,
    config: Optional[Dict[str, Any]] = None,
    error: Optional[str] = None,
) -> SubsystemStatus:
    return SubsystemStatus(
        name=name,
        status=status,
        artifacts=artifacts or {},
        native_metrics=native_metrics or {},
        config=config or {},
        error=error,
    )


def collect_run_artifacts(
    *,
    layout: ArtifactLayout,
    run_payload: Dict[str, Any],
    include_baseline_artifacts: bool = True,
) -> Dict[str, Optional[str]]:
    artifacts: Dict[str, Optional[str]] = {
        "run_config_json": path_if_exists(layout.run_config_json),
        "run_status_json": path_if_exists(layout.run_status_json),
        "comparison_summary_json": path_if_exists(layout.comparison_summary_json),
        "mercury_states_parquet": path_if_exists(layout.mercury_states_parquet),
        "mercury_paper_precision_json": path_if_exists(layout.mercury_paper_precision_json),
        "mercury_computational_json": path_if_exists(layout.mercury_computational_json),
        "mercury_latent_graph_npz": path_if_exists(layout.mercury_latent_graph_npz),
        "mercury_internal_graph_png": path_if_exists(layout.mercury_internal_graph_png),
        "internal_graphs_comparison_png": path_if_exists(layout.internal_graphs_comparison_png),
    }
    if include_baseline_artifacts and baseline_enabled(run_payload, "pocml"):
        artifacts["pocml_states_parquet"] = path_if_exists(layout.pocml_states_parquet)
        artifacts["pocml_paper_precision_json"] = path_if_exists(layout.pocml_paper_precision_json)
        artifacts["pocml_computational_json"] = path_if_exists(layout.pocml_computational_json)
        artifacts["pocml_embeddings_npz"] = path_if_exists(layout.pocml_embeddings_npz)
        artifacts["pocml_model_checkpoint_pt"] = path_if_exists(layout.pocml_model_checkpoint_pt)
        artifacts["pocml_internal_graph_png"] = path_if_exists(layout.pocml_internal_graph_png)
    if include_baseline_artifacts and baseline_enabled(run_payload, "cscg"):
        artifacts["cscg_states_parquet"] = path_if_exists(layout.cscg_states_parquet)
        artifacts["cscg_paper_precision_json"] = path_if_exists(layout.cscg_paper_precision_json)
        artifacts["cscg_computational_json"] = path_if_exists(layout.cscg_computational_json)
        artifacts["cscg_model_npz"] = path_if_exists(layout.cscg_model_npz)
        artifacts["cscg_internal_graph_png"] = path_if_exists(layout.cscg_internal_graph_png)
    return artifacts


def read_native_metrics_from_bundle(states_path: Optional[Path], run_parameters_key: str) -> Dict[str, Any]:
    if states_path is None or not states_path.exists():
        return {}
    meta = read_bundle_meta(states_path)
    if meta is None:
        return {}
    return ((meta.run_parameters or {}).get(run_parameters_key) or {})


def validate_reusable_run(
    *,
    run_root: Path,
    run_payload: Dict[str, Any],
    run_id: str,
) -> tuple[bool, str]:
    layout = artifact_layout(run_root=run_root, run_id=run_id, run_payload=run_payload)
    run_status = load_run_status_model(layout.run_status_json)
    run_config = load_run_config_model(layout.run_config_json)
    if run_status is None or run_config is None:
        return False, "missing run_config.json or run_status.json"
    if int(run_status.schema_version) != PIPELINE_SCHEMA_VERSION:
        return False, "run_status schema version mismatch"
    if int(run_config.schema_version) != PIPELINE_SCHEMA_VERSION:
        return False, "run_config schema version mismatch"
    if run_status.status not in {"completed", "reused"}:
        return False, f"run status is {run_status.status!r}"
    if (
        not layout.mercury_states_parquet.exists()
        or not layout.mercury_latent_graph_npz.exists()
        or not layout.mercury_internal_graph_png.exists()
        or not layout.internal_graphs_comparison_png.exists()
    ):
        return False, "missing Mercury artifacts"
    if paper_precision_enabled(run_payload) and not layout.mercury_paper_precision_json.exists():
        return False, "missing Mercury paper precision artifact"
    if computational_eval_enabled(run_payload) and not layout.mercury_computational_json.exists():
        return False, "missing Mercury computational artifact"
    if baseline_enabled(run_payload, "pocml"):
        if (
            not layout.pocml_states_parquet.exists()
            or not layout.pocml_embeddings_npz.exists()
            or not layout.pocml_model_checkpoint_pt.exists()
            or not layout.pocml_internal_graph_png.exists()
        ):
            return False, "missing POCML artifacts"
        if paper_precision_enabled(run_payload) and not layout.pocml_paper_precision_json.exists():
            return False, "missing POCML paper precision artifact"
        if computational_eval_enabled(run_payload) and not layout.pocml_computational_json.exists():
            return False, "missing POCML computational artifact"
    if baseline_enabled(run_payload, "cscg"):
        if (
            not layout.cscg_states_parquet.exists()
            or not layout.cscg_model_npz.exists()
            or not layout.cscg_internal_graph_png.exists()
        ):
            return False, "missing CSCG artifacts"
        if paper_precision_enabled(run_payload) and not layout.cscg_paper_precision_json.exists():
            return False, "missing CSCG paper precision artifact"
        if computational_eval_enabled(run_payload) and not layout.cscg_computational_json.exists():
            return False, "missing CSCG computational artifact"
    if (baseline_enabled(run_payload, "pocml") or baseline_enabled(run_payload, "cscg")) and not layout.comparison_summary_json.exists():
        return False, "missing comparison summary"
    return True, "ok"


def resolve_run_identity(
    *,
    args: argparse.Namespace,
    run_payload: Dict[str, Any],
    reuse_existing_run: bool,
) -> RunIdentity:
    requested_run_id = args.run_id or stable_hash(run_payload)
    study_directory = Path(args.study_root) / args.study_name
    candidate_root = study_directory / requested_run_id
    explicit_run_id = args.run_id is not None
    if not candidate_root.exists():
        return RunIdentity(
            requested_run_id=requested_run_id,
            run_id=requested_run_id,
            run_root=candidate_root,
            is_reused=False,
        )
    reusable, reason = validate_reusable_run(run_root=candidate_root, run_payload=run_payload, run_id=requested_run_id)
    if reuse_existing_run and reusable:
        return RunIdentity(
            requested_run_id=requested_run_id,
            run_id=requested_run_id,
            run_root=candidate_root,
            is_reused=True,
        )
    if explicit_run_id:
        raise RuntimeError(
            f"Run directory already exists for explicit --run_id {requested_run_id!r} but is not reusable: {reason}"
        )
    fresh_run_id = f"{requested_run_id}-{compact_utc_now()}"
    return RunIdentity(
        requested_run_id=requested_run_id,
        run_id=fresh_run_id,
        run_root=study_directory / fresh_run_id,
        is_reused=False,
    )


def build_run_config_payload(
    *,
    args: argparse.Namespace,
    run_id: str,
    run_payload: Dict[str, Any],
    reuse_existing_run: bool,
    reuse_existing_dataset: bool,
    run_root: Path,
    bundles_root: Path,
    metrics_root: Path,
) -> RunConfigModel:
    worker_config = dict(vars(args))
    worker_config["run_id"] = run_id
    worker_config["_worker_run"] = True
    reproduction_command = [sys.executable, "main.py"] + args_from_config(worker_config)
    return RunConfigModel(
        run_id=run_id,
        requested_run_id=str(run_payload.get("requested_run_id", run_id)),
        saved_at_utc=utc_now(),
        cli_args=dict(vars(args)),
        run_payload=run_payload,
        execution={
            "reuse_existing_run": reuse_existing_run,
            "reuse_existing_dataset": reuse_existing_dataset,
            "dataset_select": "latest",
            "dataset_run_id": None,
            "worker_command": reproduction_command,
        },
        paths={
            "run_root": str(run_root),
            "bundles_root": str(bundles_root),
            "metrics_root": str(metrics_root),
        },
    )


def build_run_payload(args: argparse.Namespace) -> Dict[str, Any]:
    requested_mode = str(getattr(args, "paper_precision_mode", "off")).strip().lower()
    enabled = bool(getattr(args, "paper_precision", False)) or requested_mode != "off"
    resolved_mode = requested_mode if requested_mode != "off" else ("final" if bool(getattr(args, "paper_precision", False)) else "off")
    target_method = getattr(args, "method", None)
    if target_method is None:
        pocml_enabled = bool(getattr(args, "baseline_pocml", False)) or bool(getattr(args, "baseline_all", False))
        cscg_enabled = bool(getattr(args, "baseline_cscg", False)) or bool(getattr(args, "baseline_all", False))
        if pocml_enabled and not cscg_enabled:
            target_method = "pocml"
        elif cscg_enabled and not pocml_enabled:
            target_method = "cscg"
        else:
            target_method = "mercury"
    return {
        "method": str(target_method),
        "level": args.level,
        "sensor": args.sensor,
        "sensor_range": args.sensor_range,
        "seed": args.seed,
        "rand_prob": args.rand_prob,
        "num_steps": args.num_steps,
        "valid_trajectories_only": bool(getattr(args, "valid_trajectories_only", False)),
        "mercury_valid_trajectories_only": bool(getattr(args, "mercury_valid_trajectories_only", False)),
        "mercury_split_sensory_raw_latent_valid": bool(getattr(args, "mercury_split_sensory_raw_latent_valid", False)),
        "memory_length": args.memory_length,
        "window_length": args.window_length,
        "evaluation": {
            "paper_precision": {
                "enabled": enabled,
                "mode": resolved_mode,
                "eval_interval": int(getattr(args, "paper_precision_eval_interval", 1)),
                "num_points": getattr(args, "paper_precision_num_points", None),
                "num_walks": int(getattr(args, "paper_precision_num_walks", 100)),
                "walk_length": int(getattr(args, "paper_precision_walk_length", 10_000)),
                "structure_metrics": {
                    "enabled": bool(getattr(args, "structure_metrics", True)),
                    "epsilon": float(getattr(args, "structure_metrics_epsilon", 1e-6)),
                    "ignore_self_loops": bool(getattr(args, "structure_metrics_ignore_self_loops", True)),
                    "scope": str(getattr(args, "structure_metrics_scope", "exact_level")),
                    "ground_truth_source": str(getattr(args, "structure_metrics_ground_truth_source", "empirical_walks")),
                },
            },
            "computational": {
                "enabled": bool(getattr(args, "computational_eval", False)),
                "profile": str(getattr(args, "computational_eval_profile", "full")),
            },
        },
        "sensory": {
            "allow_self_loops": args.sensory_allow_self_loops,
            "activation_threshold": args.activation_threshold,
            "topological_neighbourhood_threshold": args.topological_neighbourhood_threshold,
            "max_neurons": args.sensory_max_neurons,
            "sensory_weighting": args.sensory_weighting,
            "winning_node_lr": args.winning_node_lr,
            "topological_neighbourhood_lr": args.topological_neighbourhood_lr,
            "action_lr": args.sensory_action_lr,
            "global_context_lr": args.global_context_lr,
            "max_age": args.sensory_max_age,
            "gaussian_shape": args.sensory_gaussian_shape,
        },
        "latent": {
            "allow_self_loops": args.latent_allow_self_loops,
            "max_neurons": args.latent_max_neurons,
            "action_lr": args.latent_action_lr,
            "gaussian_shape": args.latent_gaussian_shape,
            "max_age": args.latent_max_age,
            "ambiguity_threshold": args.ambiguity_threshold,
            "ambiguity_decay": args.ambiguity_decay,
            "trace_decay": args.trace_decay,
            "lambda_trace": args.lambda_trace,
            "weight_memory": args.weight_memory,
            "weight_undirected": args.weight_undirected,
            "weight_base": args.weight_base,
            "weight_action": args.weight_action,
            "memory_replay": args.memory_replay,
            "memory_disambiguation": args.memory_disambiguation,
        },
        "action_map": {
            "n_codebook": args.am_n_codebook,
            "lr": args.am_lr,
            "sigma": args.am_sigma,
            "key": args.am_key,
            "identity_for_one_hot": bool(getattr(args, "am_identity_for_one_hot", False)),
        },
        "baselines": {
            "pocml": {
                "enabled": bool(getattr(args, "baseline_pocml", False)) or bool(getattr(args, "baseline_all", False)),
                "valid_trajectories_only": bool(getattr(args, "pocml_valid_trajectories_only", False)),
                "epochs": int(getattr(args, "pocml_epochs", 30)),
                "trajectory_length": int(getattr(args, "pocml_trajectory_length", 15)),
                "max_trajectories": getattr(args, "pocml_max_trajectories", None),
                "state_dim": int(getattr(args, "pocml_state_dim", 50)),
                "random_feature_dim": int(getattr(args, "pocml_random_feature_dim", 500)),
                "n_states": getattr(args, "pocml_n_states", None),
                "precision_capacities": getattr(args, "pocml_precision_capacities", None),
                "alpha": float(getattr(args, "pocml_alpha", 2.0)),
                "lr_q": float(getattr(args, "pocml_lr_q", 0.1)),
                "lr_v": float(getattr(args, "pocml_lr_v", 0.08)),
                "lr_all": float(getattr(args, "pocml_lr_all", 0.05)),
                "lr_m": float(getattr(args, "pocml_lr_m", 1.0)),
                "reg_m": float(getattr(args, "pocml_reg_m", 0.0)),
                "max_iter_m": int(getattr(args, "pocml_max_iter_m", 1)),
                "eps_m": float(getattr(args, "pocml_eps_m", 1e-3)),
                "batch_size": int(getattr(args, "pocml_batch_size", 64)),
                "memory_bias": bool(getattr(args, "pocml_memory_bias", True)),
                "use_ground_truth_node_ids": bool(getattr(args, "pocml_use_ground_truth_node_ids", False)),
            },
            "cscg": {
                "enabled": bool(getattr(args, "baseline_cscg", False)) or bool(getattr(args, "baseline_all", False)),
                "valid_trajectories_only": bool(getattr(args, "cscg_valid_trajectories_only", False)),
                "clones_per_obs": int(getattr(args, "cscg_clones_per_obs", 10)),
                "n_iter": int(getattr(args, "cscg_n_iter", 100)),
                "pseudocount": float(getattr(args, "cscg_pseudocount", 2e-3)),
                "term_early": bool(getattr(args, "cscg_term_early", True)),
                "train_algorithm": str(getattr(args, "cscg_train_algorithm", "em")),
                "training_mode": str(getattr(args, "cscg_training_mode", "full")),
                "batch_size": int(getattr(args, "cscg_batch_size", 1024)),
                "online_lambda": float(getattr(args, "cscg_online_lambda", 1.0)),
                "seed": int(args.seed if getattr(args, "cscg_seed", None) is None else args.cscg_seed),
            },
        },
    }


def run_complete(run_root: Path) -> bool:
    run_config = load_run_config_model(run_root / "run_config.json")
    run_status = load_run_status_model(run_root / "run_status.json")
    if run_config is None or run_status is None:
        return False
    reusable, _ = validate_reusable_run(
        run_root=run_root,
        run_payload=run_config.run_payload,
        run_id=run_status.run_id,
    )
    return reusable
