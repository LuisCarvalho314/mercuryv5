from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

from ..domain.models import RunStatusModel
from ..infrastructure import (
    baseline_enabled,
    collect_run_artifacts,
    subsystem_status,
    utc_now,
)
from ..infrastructure.runtime import atomic_write_json


def build_run_status_model(
    *,
    args: argparse.Namespace,
    run_id: str,
    requested_run_id: str,
    run_payload: Dict[str, Any],
    run_root: Path,
    bundles_root: Path,
    metrics_root: Path,
    reuse_existing_run: bool,
    reuse_existing_dataset: bool,
    status: str,
) -> RunStatusModel:
    return RunStatusModel(
        run_id=run_id,
        requested_run_id=requested_run_id,
        status=status,
        started_at_utc=utc_now(),
        study={"study_root": args.study_root, "study_name": args.study_name},
        config=run_payload,
        execution={
            "role": "worker" if args._worker_run else "coordinator",
            "reuse_existing_run": reuse_existing_run,
            "reuse_existing_dataset": reuse_existing_dataset,
            "dataset_select": "latest",
            "dataset_run_id": None,
        },
        paths={"run_root": str(run_root), "bundles_root": str(bundles_root), "metrics_root": str(metrics_root)},
        subsystems={
            "mercury": subsystem_status(
                name="mercury",
                status="pending" if str(run_payload.get("method", "mercury")) == "mercury" else "skipped",
            ),
            "sensory": subsystem_status(
                name="sensory",
                status="pending" if str(run_payload.get("method", "mercury")) == "sensory" else "skipped",
                config={"observation_mode": str(run_payload.get("sensory_observation_mode", "raw"))},
            ),
            "pocml": subsystem_status(
                name="pocml",
                status="pending" if baseline_enabled(run_payload, "pocml") else "skipped",
                config=((run_payload.get("baselines") or {}).get("pocml") or {}),
            ),
            "cscg": subsystem_status(
                name="cscg",
                status="pending" if baseline_enabled(run_payload, "cscg") else "skipped",
                config=((run_payload.get("baselines") or {}).get("cscg") or {}),
            ),
            "mercury_cscg_hybrid": subsystem_status(
                name="mercury_cscg_hybrid",
                status="pending" if str(run_payload.get("method", "mercury")) == "mercury_cscg_hybrid" else "skipped",
                config={
                    "cscg": ((run_payload.get("baselines") or {}).get("cscg") or {}),
                    "mercury_input_mode": str(run_payload.get("mercury_input_mode", "pretrained_raw")),
                },
            ),
            "tmgwr_cscg_hybrid": subsystem_status(
                name="tmgwr_cscg_hybrid",
                status="pending" if str(run_payload.get("method", "mercury")) == "tmgwr_cscg_hybrid" else "skipped",
                config={
                    "cscg": ((run_payload.get("baselines") or {}).get("cscg") or {}),
                    "source_method": "tmgwr",
                },
            ),
        },
    )


def persist_run_status(status: RunStatusModel, status_path: Path) -> None:
    atomic_write_json(status_path, status.model_dump(mode="json"))


def mark_run_completed(
    *,
    run_status: RunStatusModel,
    status_path: Path,
    layout: Any,
    run_payload: Dict[str, Any],
) -> RunStatusModel:
    run_status.status = "completed"
    run_status.finished_at_utc = utc_now()
    run_status.artifacts = collect_run_artifacts(layout=layout, run_payload=run_payload)
    persist_run_status(run_status, status_path)
    return run_status


def mark_run_failed(
    *,
    run_status: RunStatusModel,
    status_path: Path,
    layout: Any,
    run_payload: Dict[str, Any],
    error: Exception,
) -> RunStatusModel:
    method = str(run_payload.get("method", "mercury"))
    if method in run_status.subsystems and run_status.subsystems[method].status == "pending":
        run_status.subsystems[method] = subsystem_status(
            name=method,
            status="failed",
            error=repr(error),
        )
    run_status.status = "failed"
    run_status.finished_at_utc = utc_now()
    run_status.error = repr(error)
    run_status.artifacts = collect_run_artifacts(layout=layout, run_payload=run_payload)
    persist_run_status(run_status, status_path)
    return run_status
