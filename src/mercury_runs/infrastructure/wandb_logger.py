from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..domain.models import PIPELINE_SCHEMA_VERSION
from .paper_precision import summarize_paper_precision_artifact
from .runtime import stable_hash, utc_now
from .storage import baseline_enabled


def maybe_log_wandb(
    *,
    args: argparse.Namespace,
    run_id: str,
    run_status: Dict[str, Any],
    run_config: Dict[str, Any],
    job_type: Optional[str] = None,
    name: Optional[str] = None,
    group: Optional[str] = None,
) -> None:
    if not bool(getattr(args, "wandb", False)):
        return
    if str(getattr(args, "wandb_mode", "online")).lower() == "disabled":
        return
    try:
        import wandb
    except ImportError:
        print("W&B logging requested but `wandb` is not installed. Skipping W&B logging.")
        return
    tags_raw = getattr(args, "wandb_tags", None)
    tags = [token.strip() for token in str(tags_raw).split(",") if token.strip()] if tags_raw else []
    tags.extend([f"study:{args.study_name}", f"sensor:{args.sensor}", f"level:{args.level}" if args.level is not None else "study-mode"])
    if baseline_enabled(run_config.get("run_payload", {}), "pocml"):
        tags.append("baseline:pocml")
    if baseline_enabled(run_config.get("run_payload", {}), "cscg"):
        tags.append("baseline:cscg")
    init_kwargs: Dict[str, Any] = {
        "project": getattr(args, "wandb_project", "mercuryv5"),
        "name": name or getattr(args, "wandb_run_name", None) or run_id,
        "id": run_id,
        "resume": "allow",
        "job_type": job_type or getattr(args, "wandb_job_type", "run"),
        "mode": getattr(args, "wandb_mode", "online"),
        "config": run_config,
        "tags": sorted(set(tags)),
    }
    if getattr(args, "wandb_entity", None):
        init_kwargs["entity"] = args.wandb_entity
    resolved_group = group or getattr(args, "wandb_group", None) or args.study_name
    if resolved_group:
        init_kwargs["group"] = resolved_group
    run = wandb.init(**init_kwargs)
    if run is None:
        return
    try:
        run.summary["schema_version"] = run_status.get("schema_version")
        run.summary["status"] = run_status.get("status")
        run.summary["started_at_utc"] = run_status.get("started_at_utc")
        run.summary["finished_at_utc"] = run_status.get("finished_at_utc")
        run.summary["run_root"] = ((run_status.get("paths") or {}).get("run_root"))
        if run_status.get("error") is not None:
            run.summary["error"] = run_status.get("error")
        subsystems = run_status.get("subsystems") or {}
        for subsystem_name, payload in subsystems.items():
            run.summary[f"{subsystem_name}/status"] = payload.get("status")
            if payload.get("error") is not None:
                run.summary[f"{subsystem_name}/error"] = payload.get("error")
            for metric_name, metric_value in (payload.get("native_metrics") or {}).items():
                if isinstance(metric_value, (int, float, str, bool)) or metric_value is None:
                    run.summary[f"{subsystem_name}/{metric_name}"] = metric_value
        for subsystem_name in ("mercury", "pocml", "cscg"):
            paper_precision_path = ((subsystems.get(subsystem_name) or {}).get("artifacts") or {}).get("paper_precision_json")
            if paper_precision_path:
                run.summary.update(
                    {f"{subsystem_name}/paper_{k}": v for k, v in summarize_paper_precision_artifact(Path(paper_precision_path)).items()}
                )
        if bool(getattr(args, "wandb_log_artifacts", True)):
            artifact = wandb.Artifact(name=f"{run_id}-artifacts", type="run-artifacts")
            artifact_paths: List[str] = []
            for value in (run_status.get("artifacts") or {}).values():
                if value:
                    artifact_paths.append(str(value))
            for payload in subsystems.values():
                for value in (payload.get("artifacts") or {}).values():
                    if value:
                        artifact_paths.append(str(value))
            for path_str in sorted(set(artifact_paths)):
                path_obj = Path(path_str)
                if path_obj.exists():
                    artifact.add_file(str(path_obj), name=path_obj.name)
            run.log_artifact(artifact)
    finally:
        run.finish()


def maybe_log_study_wandb(
    *,
    args: argparse.Namespace,
    run_records: List[Dict[str, Any]],
    study_directory: Path,
    summary_path: Optional[Path],
    errors_path: Path,
    incomplete_report: Path,
) -> None:
    if not bool(getattr(args, "wandb", False)):
        return
    if str(getattr(args, "wandb_mode", "online")).lower() == "disabled":
        return
    run_status = {
        "schema_version": PIPELINE_SCHEMA_VERSION,
        "status": "completed" if not errors_path.exists() else "completed_with_errors",
        "started_at_utc": None,
        "finished_at_utc": utc_now(),
        "paths": {"run_root": str(study_directory)},
        "artifacts": {
            "study_summary_parquet": str(summary_path) if summary_path and summary_path.exists() else None,
            "study_errors_jsonl": str(errors_path) if errors_path.exists() else None,
            "study_incomplete_jsonl": str(incomplete_report) if incomplete_report.exists() else None,
        },
        "subsystems": {
            "study": {
                "status": "completed",
                "native_metrics": {
                    "scheduled_runs": len(run_records),
                    "error_count": sum(1 for _ in errors_path.open("r", encoding="utf-8")) if errors_path.exists() else 0,
                },
                "artifacts": {},
                "config": {},
                "error": None,
            }
        },
    }
    run_config = {
        "schema_version": PIPELINE_SCHEMA_VERSION,
        "run_id": f"study-{args.study_name}",
        "requested_run_id": f"study-{args.study_name}",
        "saved_at_utc": utc_now(),
        "cli_args": dict(vars(args)),
        "run_payload": {"study": {"study_root": args.study_root, "study_name": args.study_name, "max_workers": args.max_workers}},
        "execution": {"mode": "study", "parallel": (args.max_workers > 1) and (not args.no_parallel)},
        "paths": {"run_root": str(study_directory), "bundles_root": None, "metrics_root": None},
    }
    maybe_log_wandb(
        args=args,
        run_id=f"study-{stable_hash({'study_root': args.study_root, 'study_name': args.study_name})}",
        run_status=run_status,
        run_config=run_config,
        job_type="study",
        name=getattr(args, "wandb_run_name", None) or f"{args.study_name}-study",
        group=args.study_name,
    )
