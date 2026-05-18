from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import polars as pl

from ..domain.models import PIPELINE_SCHEMA_VERSION
from .paper_precision import summarize_paper_precision_artifact
from .runtime import stable_hash, utc_now
from .storage import baseline_enabled


MAX_WANDB_HISTORY_ROWS = 2000
RAW_STATE_COLUMNS = frozenset(
    {
        "cartesian_proxy_bmu",
        "ground_truth_bmu",
        "sensory_bmu",
        "latent_bmu",
    }
)
PAPER_PRECISION_HISTORY_COLUMNS = frozenset(
    {
        "sensory_precision",
        "sensory_recall",
        "latent_precision",
        "latent_recall",
        "sensory_edge_precision",
        "sensory_edge_recall",
        "sensory_edge_f1",
        "latent_edge_precision",
        "latent_edge_recall",
        "latent_edge_f1",
        "sensory_mean_total_variation",
        "latent_mean_total_variation",
        "sensory_action_conditioned_edge_precision",
        "sensory_action_conditioned_edge_recall",
        "sensory_action_conditioned_edge_f1",
        "latent_action_conditioned_edge_precision",
        "latent_action_conditioned_edge_recall",
        "latent_action_conditioned_edge_f1",
        "sensory_action_conditioned_mean_total_variation",
        "latent_action_conditioned_mean_total_variation",
    }
)


def _numeric_columns(frame: pl.DataFrame, *, exclude: set[str] | frozenset[str] | None = None) -> list[str]:
    exclude = exclude or frozenset()
    numeric_types = {
        pl.Int8,
        pl.Int16,
        pl.Int32,
        pl.Int64,
        pl.UInt8,
        pl.UInt16,
        pl.UInt32,
        pl.UInt64,
        pl.Float32,
        pl.Float64,
    }
    return [
        column
        for column, dtype in frame.schema.items()
        if dtype in numeric_types and column not in exclude
    ]


def _log_parquet_history_to_wandb(
    run: Any,
    parquet_path: Path,
    *,
    namespace: str,
    exclude_columns: set[str] | frozenset[str] | None = None,
) -> None:
    if not parquet_path.exists():
        return
    frame = pl.read_parquet(parquet_path)
    if frame.is_empty():
        return
    has_step_column = "step" in frame.columns
    metric_columns = _numeric_columns(
        frame,
        exclude=frozenset({"step"}) | frozenset(exclude_columns or ()),
    )
    if not metric_columns:
        return
    step_metric_name = f"{namespace}/step"
    if hasattr(run, "define_metric"):
        run.define_metric(step_metric_name)
        run.define_metric(f"{namespace}/*", step_metric=step_metric_name)

    stride = max(1, int(frame.height // MAX_WANDB_HISTORY_ROWS))
    sampled = frame if has_step_column else frame.with_row_index("step", offset=1)
    if stride > 1:
        sampled = sampled.filter(((pl.col("step") - 1) % stride) == 0)

    for row in sampled.select(["step", *metric_columns]).iter_rows(named=True):
        step = int(row.pop("step"))
        payload = {step_metric_name: step}
        payload.update(
            {
                f"{namespace}/{key}": value
                for key, value in row.items()
                if isinstance(value, (int, float)) and value is not None
            }
        )
        if payload:
            run.log(payload)


def _log_states_history_to_wandb(run: Any, states_path: Path, *, namespace: str) -> None:
    _log_parquet_history_to_wandb(
        run,
        states_path,
        namespace=namespace,
        exclude_columns=RAW_STATE_COLUMNS,
    )


def _wandb_scalar_value(value: Any) -> Any:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float, str)):
        return value
    return None


def _paper_precision_summary_payload(payload: dict[str, Any], *, namespace: str) -> dict[str, Any]:
    metrics = dict(payload.get("metrics") or {})
    summary: dict[str, Any] = {}
    for key, value in metrics.items():
        scalar_value = _wandb_scalar_value(value)
        if scalar_value is not None:
            summary[f"{namespace}/{key}"] = scalar_value
        elif isinstance(value, (list, dict)):
            summary[f"{namespace}/{key}_json"] = json.dumps(value)
    protocol = dict(payload.get("protocol") or {})
    for key in ("num_walks", "walk_length", "mode", "eval_interval", "num_points", "schedule_unit", "x_axis_unit"):
        if key in protocol:
            summary[f"{namespace}/{key}"] = protocol[key]
    return summary


def _paper_precision_history_payload(row: dict[str, Any], *, namespace: str) -> dict[str, Any]:
    logged_row: dict[str, Any] = {}
    for key in sorted(PAPER_PRECISION_HISTORY_COLUMNS):
        value = row.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        logged_row[f"{namespace}/{key}"] = value
        if key.endswith("_edge_f1"):
            logged_row[f"{namespace}/{key.removesuffix('_f1')}_fscore"] = value
    return logged_row


def _log_paper_precision_to_wandb(
    run: Any,
    paper_precision_path: Path,
    *,
    subsystem_name: str,
    final_observed_samples: int | None = None,
) -> None:
    if not paper_precision_path.exists():
        return
    payload = json.loads(paper_precision_path.read_text(encoding="utf-8"))
    namespace = f"{subsystem_name}/paper_precision"
    run.summary.update(_paper_precision_summary_payload(payload, namespace=namespace))

    scalar_summary = {
        key: value
        for key, value in summarize_paper_precision_artifact(paper_precision_path).items()
        if isinstance(value, (int, float)) and value is not None
    }
    history = payload.get("history") or []
    if not history and scalar_summary:
        try:
            observed_samples = int(final_observed_samples or 0)
        except (TypeError, ValueError):
            observed_samples = 0
        history = [{"observed_samples": observed_samples, **scalar_summary}]

    step_metric_name = f"{namespace}/observed_samples"
    if hasattr(run, "define_metric"):
        run.define_metric(step_metric_name)
        run.define_metric(f"{namespace}/*", step_metric=step_metric_name)

    stride = max(1, int(len(history) // MAX_WANDB_HISTORY_ROWS))
    for index, row in enumerate(history):
        if index % stride != 0:
            continue
        if not isinstance(row, dict):
            continue
        observed_samples = row.get("observed_samples", row.get("step", index + 1))
        if not isinstance(observed_samples, (int, float)):
            continue
        logged_row = {step_metric_name: observed_samples}
        logged_row.update(_paper_precision_history_payload(row, namespace=namespace))
        if len(logged_row) > 1:
            run.log(logged_row)


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
                _log_paper_precision_to_wandb(
                    run,
                    Path(paper_precision_path),
                    subsystem_name=subsystem_name,
                    final_observed_samples=(run_config.get("run_payload") or {}).get("num_steps"),
                )
        mercury_states_path = ((subsystems.get("mercury") or {}).get("artifacts") or {}).get("states_parquet")
        if mercury_states_path:
            _log_states_history_to_wandb(
                run,
                Path(mercury_states_path),
                namespace="mercury/states",
            )
        mercury_attribution_path = ((subsystems.get("mercury") or {}).get("artifacts") or {}).get("attribution_parquet")
        if mercury_attribution_path:
            _log_parquet_history_to_wandb(
                run,
                Path(mercury_attribution_path),
                namespace="mercury/attribution",
                exclude_columns=frozenset({"latent_bmu"}),
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
