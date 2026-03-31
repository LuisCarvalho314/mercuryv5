from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

from ..analysis.study_history_plots import generate_study_history_plots
from ..domain.models import HierarchicalStudyConfig, StudyDefaults, StudyGridConfig
from ..infrastructure import (
    append_jsonl,
    args_from_config,
    base_runtime_env,
    build_run_payload,
    grid,
    maybe_log_study_wandb,
    prepare_study_logs,
    resolve_run_identity,
    run_complete,
    run_subprocess_command,
)
from ..infrastructure.reporting import write_study_summary
from ..infrastructure.progress import load_progress, load_progress_tracks


WEIGHT_PRESETS: Dict[str, Dict[str, float]] = {
    "baseline": {
        "weight_memory": 0.4,
        "weight_undirected": 0.2,
        "weight_base": 0.2,
        "weight_action": 0.2,
    },
    "no_memory": {
        "weight_memory": 0.0,
        "weight_undirected": 0.333333,
        "weight_base": 0.333333,
        "weight_action": 0.333334,
    },
    "no_undirected": {
        "weight_memory": 0.5,
        "weight_undirected": 0.0,
        "weight_base": 0.25,
        "weight_action": 0.25,
    },
    "no_base": {
        "weight_memory": 0.5,
        "weight_undirected": 0.25,
        "weight_base": 0.0,
        "weight_action": 0.25,
    },
    "no_action": {
        "weight_memory": 0.5,
        "weight_undirected": 0.25,
        "weight_base": 0.25,
        "weight_action": 0.0,
    },
    "memory_dominant": {
        "weight_memory": 0.7,
        "weight_undirected": 0.1,
        "weight_base": 0.1,
        "weight_action": 0.1,
    },
    "undirected_dominant": {
        "weight_memory": 0.1,
        "weight_undirected": 0.7,
        "weight_base": 0.1,
        "weight_action": 0.1,
    },
    "base_dominant": {
        "weight_memory": 0.1,
        "weight_undirected": 0.1,
        "weight_base": 0.7,
        "weight_action": 0.1,
    },
    "action_dominant": {
        "weight_memory": 0.1,
        "weight_undirected": 0.1,
        "weight_base": 0.1,
        "weight_action": 0.7,
    },
}


def default_parameter_space(defaults: StudyDefaults) -> Dict[str, Any]:
    return {
        "environment": {
            "level": [16],
            "sensor": ["cardinal distance"],
            "sensor_range": [1],
            "seed": [0],
            "rand_prob": [0.3],
            "mercury_valid_trajectories_only": [False],
            "mercury_split_sensory_raw_latent_valid": [False],
            "pocml_valid_trajectories_only": [False],
            "cscg_valid_trajectories_only": [False],
            "window_length": [100],
            "reuse_existing_run": [False],
        },
        "evaluation": {
            "structure_metrics": [True],
            "structure_metrics_epsilon": [1e-6],
            "structure_metrics_scope": ["exact_level"],
            "structure_metrics_ground_truth_source": ["empirical_walks"],
        },
        "methods": {
            "mercury": {
                "enabled": True,
                "grid": {
                    "memory_length": [40],
                    "activation_threshold": [0.95],
                    "ambiguity_threshold": [10],
                    "ambiguity_decay": [0.99],
                    "am_lr": [0.001],
                    "am_key": [0],
                    "weight_memory": [0.4],
                    "weight_undirected": [0.2],
                    "weight_base": [0.2],
                    "weight_action": [0.2],
                    "memory_replay": [False, True],
                    "memory_disambiguation": [False, True],
                },
            },
            "pocml": {"enabled": False, "grid": {}},
            "cscg": {"enabled": False, "grid": {}},
        },
    }


def normalize_parameter_space(raw_space: Dict[str, Any]) -> Dict[str, List[Any]]:
    normalized: Dict[str, List[Any]] = {}
    for key, value in raw_space.items():
        normalized[key] = value if isinstance(value, list) else [value]
        if not normalized[key]:
            raise ValueError(f"Parameter '{key}' has an empty list.")
    return normalized


def apply_weight_preset(run_configuration: Dict[str, Any]) -> Dict[str, Any]:
    preset_name = run_configuration.get("weight_preset")
    if preset_name is None:
        return dict(run_configuration)
    resolved_name = str(preset_name)
    if resolved_name not in WEIGHT_PRESETS:
        available = ", ".join(sorted(WEIGHT_PRESETS))
        raise ValueError(f"Unknown weight_preset '{resolved_name}'. Expected one of: {available}")
    expanded = dict(run_configuration)
    expanded.pop("weight_preset", None)
    expanded.update(WEIGHT_PRESETS[resolved_name])
    expanded["weight_preset"] = resolved_name
    return expanded


def _normalize_section(raw: Dict[str, Any]) -> Dict[str, List[Any]]:
    if not isinstance(raw, dict):
        raise ValueError("Study config sections must be JSON objects.")
    return normalize_parameter_space(raw)


def _merge_sections(*sections: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
    merged: Dict[str, List[Any]] = {}
    for section in sections:
        merged.update(section)
    return merged


def _method_flags(method: str) -> Dict[str, Any]:
    if method == "mercury":
        return {"method": "mercury", "baseline_all": False, "baseline_pocml": False, "baseline_cscg": False}
    if method == "pocml":
        return {"method": "pocml", "baseline_all": False, "baseline_pocml": True, "baseline_cscg": False}
    if method == "cscg":
        return {"method": "cscg", "baseline_all": False, "baseline_pocml": False, "baseline_cscg": True}
    raise ValueError(f"Unknown method '{method}'")


def _expand_method_runs(
    *,
    method_name: str,
    method_grid: Dict[str, Any],
    common_space: Dict[str, List[Any]],
    defaults: StudyDefaults,
) -> List[Dict[str, Any]]:
    parameter_space = _merge_sections(common_space, _normalize_section(method_grid))
    parameter_space["study_root"] = [defaults.study_root]
    parameter_space["study_name"] = [defaults.study_name]
    method_flags = _method_flags(method_name)
    for key, value in method_flags.items():
        parameter_space[key] = [value]
    parameter_space.setdefault("reuse_existing_run", [False])

    runs: List[Dict[str, Any]] = []
    for run_configuration in grid(parameter_space):
        runs.append(apply_weight_preset(run_configuration) if method_name == "mercury" else dict(run_configuration))
    return runs


def _load_hierarchical_study_config(config_path: Path, defaults: StudyDefaults) -> List[Dict[str, Any]]:
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    study_config = HierarchicalStudyConfig.model_validate(raw)
    environment_space = _normalize_section(study_config.environment)
    evaluation_space = _normalize_section(study_config.evaluation)
    common_space = _merge_sections(environment_space, evaluation_space)

    run_configurations: List[Dict[str, Any]] = []
    for method_name in ("mercury", "pocml", "cscg"):
        method_model = getattr(study_config.methods, method_name)
        if not bool(method_model.enabled):
            continue
        run_configurations.extend(
            _expand_method_runs(
                method_name=method_name,
                method_grid=dict(method_model.grid),
                common_space=common_space,
                defaults=defaults,
            )
        )
    if not run_configurations:
        raise ValueError(f"Study config must enable at least one method: {config_path}")
    return run_configurations


def resolve_run_configurations(args: argparse.Namespace, defaults: StudyDefaults) -> List[Dict[str, Any]]:
    if args.study_config is None:
        default_config = default_parameter_space(defaults)
        study_config = HierarchicalStudyConfig.model_validate(default_config)
        environment_space = _normalize_section(study_config.environment)
        evaluation_space = _normalize_section(study_config.evaluation)
        common_space = _merge_sections(environment_space, evaluation_space)
        run_configurations: List[Dict[str, Any]] = []
        for method_name in ("mercury", "pocml", "cscg"):
            method_model = getattr(study_config.methods, method_name)
            if not bool(method_model.enabled):
                continue
            run_configurations.extend(
                _expand_method_runs(
                    method_name=method_name,
                    method_grid=dict(method_model.grid),
                    common_space=common_space,
                    defaults=defaults,
                )
            )
        return run_configurations
    config_path = Path(args.study_config)
    if not config_path.exists():
        raise FileNotFoundError(f"Study config not found: {config_path}")
    return _load_hierarchical_study_config(config_path, defaults)


def _prune_command_config(config: Dict[str, Any]) -> Dict[str, Any]:
    method = str(config.get("method", "mercury"))
    pruned = dict(config)
    for key in {
        "study",
        "study_config",
        "max_workers",
        "no_parallel",
        "resume",
        "continue_on_error",
        "retry_failed",
        "retry_incomplete",
        "_worker_run",
    }:
        pruned.pop(key, None)
    pruned.pop("weight_preset", None)
    if method == "mercury":
        for key in list(pruned):
            if key.startswith("pocml_") or key.startswith("cscg_"):
                pruned.pop(key, None)
    elif method == "pocml":
        for key in list(pruned):
            if key.startswith("cscg_"):
                pruned.pop(key, None)
    elif method == "cscg":
        for key in list(pruned):
            if key.startswith("pocml_"):
                pruned.pop(key, None)
    return pruned


def load_retry_configs(errors_path: Path) -> List[Dict[str, Any]]:
    if not errors_path.exists():
        raise FileNotFoundError(f"Missing error log: {errors_path}")
    retry_configs: List[Dict[str, Any]] = []
    for line in errors_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        config = record.get("config")
        if config:
            retry_config = dict(config)
            retry_config.pop("run_id", None)
            retry_config.pop("requested_run_id", None)
            retry_configs.append(retry_config)
    if not retry_configs:
        raise RuntimeError(f"No configs found in {errors_path}")
    return retry_configs


def load_incomplete_configs(manifest_path: Path, study_directory: Path) -> List[Dict[str, Any]]:
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")
    configs_by_run_id: Dict[str, Dict[str, Any]] = {}
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        config = record.get("config")
        run_id = (config or {}).get("requested_run_id") or (config or {}).get("run_id")
        if not config or not run_id:
            continue
        configs_by_run_id[str(run_id)] = dict(config)
    incomplete_configs: List[Dict[str, Any]] = []
    for run_id, config in sorted(configs_by_run_id.items()):
        if run_complete(study_directory / run_id):
            continue
        cfg = dict(config)
        cfg.pop("run_id", None)
        cfg.pop("requested_run_id", None)
        incomplete_configs.append(cfg)
    if not incomplete_configs:
        raise RuntimeError(f"No incomplete configs found from manifest: {manifest_path}")
    return incomplete_configs


def write_incomplete_report(study_directory: Path, manifest_path: Path) -> Path:
    report_path = study_directory / "study_incomplete.jsonl"
    manifest_run_ids: set[str] = set()
    if manifest_path.exists():
        for line in manifest_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            config = record.get("config") or {}
            run_id = config.get("requested_run_id") or config.get("run_id")
            if run_id:
                manifest_run_ids.add(str(run_id))
    rows = [
        {
            "run_id": run_dir.name,
            "has_run_status": (run_dir / "run_status.json").exists(),
            "has_run_config": (run_dir / "run_config.json").exists(),
        }
        for run_dir in sorted([p for p in study_directory.iterdir() if p.is_dir()])
        if run_dir.name in manifest_run_ids
        if not run_complete(run_dir)
    ]
    if rows:
        with report_path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")
    elif report_path.exists():
        report_path.unlink()
    return report_path


def build_study_run_records(
    *,
    args: argparse.Namespace,
    run_configurations: List[Dict[str, Any]],
    study_directory: Path,
    manifest_path: Path,
    python_executable: str,
    main_path: str,
) -> List[Dict[str, Any]]:
    run_records: List[Dict[str, Any]] = []
    inherited_keys = {
        "num_steps", "datasets_root", "window_length", "sensor", "sensor_range", "study_root", "study_name",
        "wandb", "wandb_project", "wandb_entity", "wandb_group", "wandb_tags", "wandb_mode",
        "structure_metrics", "structure_metrics_epsilon", "structure_metrics_scope", "structure_metrics_ground_truth_source",
        "wandb_job_type", "wandb_log_artifacts", "wandb_run_name", "baseline_pocml", "baseline_all",
        "pocml_epochs", "pocml_trajectory_length", "pocml_max_trajectories", "pocml_state_dim",
        "pocml_random_feature_dim", "pocml_n_states", "pocml_precision_capacities", "pocml_alpha", "pocml_lr_q", "pocml_lr_v",
        "pocml_lr_all", "pocml_lr_m", "pocml_reg_m", "pocml_max_iter_m", "pocml_eps_m",
        "pocml_use_ground_truth_node_ids", "pocml_batch_size", "pocml_memory_bias", "baseline_cscg",
        "cscg_clones_per_obs", "cscg_n_iter", "cscg_pseudocount", "cscg_term_early", "cscg_train_algorithm", "cscg_training_mode",
        "cscg_batch_size", "cscg_online_lambda", "cscg_seed", "method",
    }
    for run_index, run_configuration in enumerate(run_configurations):
        inherited = {key: getattr(args, key) for key in inherited_keys if hasattr(args, key) and getattr(args, key) is not None}
        merged_config = {**inherited, **run_configuration}
        merged_with_defaults = {**vars(args), **merged_config}
        validated_configuration = StudyGridConfig.model_validate(merged_with_defaults).model_dump()
        merged_config = {**merged_with_defaults, **validated_configuration}
        merged_args = argparse.Namespace(**merged_config)
        identity = resolve_run_identity(
            args=merged_args,
            run_payload=build_run_payload(merged_args),
            reuse_existing_run=bool(getattr(merged_args, "reuse_existing_run", False)),
        )
        if args.resume and run_complete(identity.run_root):
            continue
        configuration_for_command = _prune_command_config(merged_config)
        configuration_for_manifest = dict(merged_config)
        configuration_for_manifest["requested_run_id"] = identity.requested_run_id
        command = [python_executable, main_path] + args_from_config(configuration_for_command)
        append_jsonl(manifest_path, {"run_index": run_index, "command": command, "config": configuration_for_manifest})
        run_records.append({"run_index": run_index, "config": configuration_for_manifest, "command": command, "run_id": identity.requested_run_id})
    return run_records


def _format_progress_row(*, record: Dict[str, Any], track_name: str, progress: Dict[str, Any], show_identity: bool) -> str:
    stage = str(progress.get("stage") or "pending")
    current = progress.get("current")
    total = progress.get("total")
    message = str(progress.get("message") or "")
    extra = progress.get("extra") or {}
    percent = ""
    bar = ""
    if isinstance(current, int) and isinstance(total, int) and total > 0:
        ratio = max(0.0, min(1.0, float(current) / float(total)))
        width = 12
        filled = int(width * ratio)
        bar = "[" + ("#" * filled) + ("-" * (width - filled)) + "]"
        percent = f"{int(ratio * 100):3d}%"
    nodes = f" nodes={extra.get('nodes')}" if isinstance(extra, dict) and extra.get("nodes") is not None else ""
    method = str(record["config"].get("method", "mercury"))
    identity = f"{record['run_id']} {method:<7}" if show_identity else " " * (len(record["run_id"]) + 8)
    return f"{identity} {track_name:<15} {stage:<20} {bar:<14} {percent:<4} {message}{nodes}".rstrip()


def _is_terminal_stage(stage: str | None) -> bool:
    return str(stage or "").strip().lower() in {"completed", "failed"}


def _track_sort_key(name: str) -> tuple[int, str]:
    track_order = ["lifecycle", "train", "eval", "paper_precision"]
    if name in track_order:
        return (track_order.index(name), name)
    return (len(track_order), name)


def _visible_progress_tracks(tracks: Dict[str, Dict[str, Any]]) -> List[tuple[str, Dict[str, Any]]]:
    lifecycle = dict(tracks.get("lifecycle") or {})
    lifecycle_stage = str(lifecycle.get("stage") or "")
    if _is_terminal_stage(lifecycle_stage):
        collapsed_name = "done" if lifecycle_stage == "completed" else "failed"
        return [(collapsed_name, lifecycle or {"stage": lifecycle_stage})]

    visible: List[tuple[str, Dict[str, Any]]] = []
    if lifecycle:
        visible.append(("lifecycle", lifecycle))
    for name in sorted((track_name for track_name in tracks if track_name != "lifecycle"), key=lambda item: _track_sort_key(item)):
        progress = dict(tracks[name])
        if _is_terminal_stage(str(progress.get("stage") or "")):
            continue
        visible.append((name, progress))
    return visible


def _record_activity_rank(visible_tracks: List[tuple[str, Dict[str, Any]]]) -> int:
    non_lifecycle_active = any((name != "lifecycle") and (not _is_terminal_stage(str(progress.get("stage") or ""))) for name, progress in visible_tracks)
    if non_lifecycle_active:
        return 0
    lifecycle = next((progress for name, progress in visible_tracks if name in {"lifecycle", "done", "failed"}), {})
    if not _is_terminal_stage(str(lifecycle.get("stage") or "")):
        return 1
    return 2


def _record_timestamp(visible_tracks: List[tuple[str, Dict[str, Any]]]) -> str:
    timestamps = [str(progress.get("updated_at_utc") or "") for _, progress in visible_tracks]
    return max(timestamps) if timestamps else ""


def _render_study_dashboard(*, run_records: List[Dict[str, Any]], completed: int, total: int) -> str:
    width = shutil.get_terminal_size((120, 20)).columns
    header = f"Study progress: {completed}/{total} completed"
    rows = [header[:width], "-" * min(width, max(len(header), 40))]
    records_by_method: Dict[str, List[tuple[Dict[str, Any], List[tuple[str, Dict[str, Any]]]]]] = {"mercury": [], "pocml": [], "cscg": []}
    for record in run_records:
        run_root = Path(record["config"]["study_root"]) / record["config"]["study_name"] / record["run_id"]
        tracks = load_progress_tracks(run_root)
        if not tracks:
            continue
        method = str(record["config"].get("method", "mercury"))
        visible_tracks = _visible_progress_tracks(tracks)
        records_by_method.setdefault(method, []).append((record, visible_tracks))
    for method in ["mercury", "pocml", "cscg"]:
        method_records = records_by_method.get(method) or []
        if not method_records:
            continue
        method_records.sort(key=lambda item: (_record_activity_rank(item[1]), _record_timestamp(item[1]), item[0]["run_id"]), reverse=False)
        method_records.sort(key=lambda item: _record_timestamp(item[1]), reverse=True)
        method_records.sort(key=lambda item: _record_activity_rank(item[1]))
        rows.append(f"{method.upper()}")
        for record, visible_tracks in method_records:
            for index, (track_name, progress) in enumerate(visible_tracks):
                rows.append(
                    _format_progress_row(
                        record=record,
                        track_name=track_name,
                        progress=progress,
                        show_identity=(index == 0),
                    )[:width]
                )
    return "\n".join(rows)


def _print_study_dashboard(*, run_records: List[Dict[str, Any]], completed: int, total: int, first_render: bool) -> None:
    dashboard = _render_study_dashboard(run_records=run_records, completed=completed, total=total)
    if sys.stdout.isatty():
        prefix = "" if first_render else "\x1b[H\x1b[J"
        print(prefix + dashboard, end="", flush=True)
    else:
        print(dashboard, flush=True)


def _write_study_artifacts(*, study_root: Path, study_name: str) -> Path:
    summary_path = write_study_summary(study_root=study_root, study_name=study_name)
    study_directory = study_root / study_name
    try:
        outputs = generate_study_history_plots(study_root=study_directory)
    except Exception as exc:
        print(f"Study history plot generation failed: {exc}")
    else:
        if outputs:
            print(f"Wrote study history plots: {study_directory / 'plots' / 'study_history'}")
    return summary_path


def run_study(args: argparse.Namespace) -> None:
    defaults = StudyDefaults(study_root=args.study_root, study_name=args.study_name)
    study_root = Path(defaults.study_root)
    study_directory = study_root / defaults.study_name
    study_directory.mkdir(parents=True, exist_ok=True)
    manifest_path, errors_path = prepare_study_logs(
        study_directory,
        keep_existing=bool(args.resume or args.retry_failed or args.retry_incomplete),
    )
    run_configurations = resolve_run_configurations(args, defaults)
    if args.retry_failed:
        run_configurations = load_retry_configs(errors_path)
    elif args.retry_incomplete:
        run_configurations = load_incomplete_configs(manifest_path, study_directory)
    base_env = base_runtime_env()
    if (args.max_workers > 1) and (not args.no_parallel):
        base_env["MERCURY_PROGRESS"] = "0"
    run_records = build_study_run_records(
        args=args,
        run_configurations=run_configurations,
        study_directory=study_directory,
        manifest_path=manifest_path,
        python_executable=defaults.python_executable,
        main_path=defaults.main_path,
    )
    print(f"Study runs scheduled: {len(run_records)}")
    if (args.max_workers > 1) and (not args.no_parallel):
        from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
        with ProcessPoolExecutor(max_workers=int(args.max_workers)) as pool:
            future_map = {pool.submit(run_subprocess_command, record["command"], base_env): record for record in run_records}
            pending = set(future_map)
            completed_count = 0
            last_render_at = 0.0
            first_render = True
            while pending:
                done, pending = wait(pending, timeout=0.5, return_when=FIRST_COMPLETED)
                now = time.monotonic()
                if first_render or (now - last_render_at >= 0.5):
                    _print_study_dashboard(
                        run_records=run_records,
                        completed=completed_count,
                        total=len(run_records),
                        first_render=first_render,
                    )
                    first_render = False
                    last_render_at = now
                for future in done:
                    record = future_map[future]
                    try:
                        future.result()
                        completed_count += 1
                    except Exception as exc:
                        append_jsonl(errors_path, {"run_index": record["run_index"], "run_id": record["run_id"], "command": record["command"], "config": record["config"], "error": repr(exc)})
                        if not args.continue_on_error:
                            if sys.stdout.isatty():
                                print()
                            raise
                if done:
                    _print_study_dashboard(
                        run_records=run_records,
                        completed=completed_count,
                        total=len(run_records),
                        first_render=first_render,
                    )
                    first_render = False
                    last_render_at = time.monotonic()
            if sys.stdout.isatty():
                print()
    else:
        for record in run_records:
            try:
                subprocess.run(record["command"], check=True, env=base_env)
            except Exception as exc:
                append_jsonl(errors_path, {"run_index": record["run_index"], "run_id": record["run_id"], "command": record["command"], "config": record["config"], "error": repr(exc)})
                if not args.continue_on_error:
                    raise
    summary_path = _write_study_artifacts(study_root=study_root, study_name=defaults.study_name)
    print(f"Wrote: {summary_path}")
    incomplete_report = write_incomplete_report(study_directory, manifest_path)
    if incomplete_report.exists():
        print(f"Incomplete runs report: {incomplete_report}")
    maybe_log_study_wandb(
        args=args,
        run_records=run_records,
        study_directory=study_directory,
        summary_path=summary_path,
        errors_path=errors_path,
        incomplete_report=incomplete_report,
    )


__all__ = [
    "run_study",
    "default_parameter_space",
    "normalize_parameter_space",
    "apply_weight_preset",
    "resolve_run_configurations",
    "load_retry_configs",
    "load_incomplete_configs",
    "write_incomplete_report",
    "build_study_run_records",
]
