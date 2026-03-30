# scripts/analysis/run_health_report.py
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional


def _find_states_parquet(run_root: Path) -> Optional[Path]:
    bundles_root = run_root / "bundles" / "mercury"
    if not bundles_root.exists():
        return None
    candidates = sorted(bundles_root.rglob("*_states.parquet"))
    return candidates[-1] if candidates else None


def _run_complete(run_root: Path) -> bool:
    return _find_states_parquet(run_root) is not None


def _load_status(run_root: Path) -> Dict[str, Any]:
    status_path = run_root / "run_status.json"
    if not status_path.exists():
        return {"status": "missing_status"}
    try:
        payload = json.loads(status_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"status": "corrupt_status", "error": repr(exc)}
    if not isinstance(payload, dict):
        return {"status": "corrupt_status", "error": "status payload is not an object"}
    return payload


def _study_runs(study_dir: Path) -> list[Path]:
    return sorted([p for p in study_dir.iterdir() if p.is_dir()])


def _summarize_study(study_dir: Path) -> Dict[str, Any]:
    runs = _study_runs(study_dir)
    counts = {
        "runs_total": len(runs),
        "complete": 0,
        "incomplete": 0,
        "status_completed": 0,
        "status_failed": 0,
        "status_reused": 0,
        "status_started": 0,
        "status_missing": 0,
        "status_corrupt": 0,
    }
    failed_runs: list[str] = []

    for run_dir in runs:
        is_complete = _run_complete(run_dir)
        status_payload = _load_status(run_dir)
        status = str(status_payload.get("status", "missing_status"))

        if is_complete:
            counts["complete"] += 1
        else:
            counts["incomplete"] += 1

        if status == "completed":
            counts["status_completed"] += 1
        elif status == "failed":
            counts["status_failed"] += 1
            failed_runs.append(run_dir.name)
        elif status == "reused":
            counts["status_reused"] += 1
        elif status == "started":
            counts["status_started"] += 1
        elif status == "missing_status":
            counts["status_missing"] += 1
        elif status == "corrupt_status":
            counts["status_corrupt"] += 1

    return {
        "study_name": study_dir.name,
        **counts,
        "failed_run_ids": failed_runs,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize study/run artifact health.")
    parser.add_argument("--study-root", type=str, default="results/current/studies")
    parser.add_argument("--study-name", type=str, default=None, help="Optional single study to inspect")
    parser.add_argument("--write-json", type=str, default=None, help="Optional output JSON report path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    study_root = Path(args.study_root)
    if not study_root.exists():
        raise FileNotFoundError(f"Study root not found: {study_root}")

    if args.study_name:
        study_dirs = [study_root / args.study_name]
    else:
        study_dirs = sorted([p for p in study_root.iterdir() if p.is_dir()])

    rows: list[Dict[str, Any]] = []
    for study_dir in study_dirs:
        if not study_dir.exists():
            print(f"[WARN] Missing study dir: {study_dir}")
            continue
        row = _summarize_study(study_dir)
        rows.append(row)

    if not rows:
        print("No studies found.")
        return

    overall = {
        "studies_total": len(rows),
        "runs_total": sum(int(r["runs_total"]) for r in rows),
        "complete": sum(int(r["complete"]) for r in rows),
        "incomplete": sum(int(r["incomplete"]) for r in rows),
        "status_completed": sum(int(r["status_completed"]) for r in rows),
        "status_failed": sum(int(r["status_failed"]) for r in rows),
        "status_reused": sum(int(r["status_reused"]) for r in rows),
        "status_started": sum(int(r["status_started"]) for r in rows),
        "status_missing": sum(int(r["status_missing"]) for r in rows),
        "status_corrupt": sum(int(r["status_corrupt"]) for r in rows),
    }

    print("Study Health Report")
    print("=" * 72)
    for row in rows:
        print(
            f"{row['study_name']}: runs={row['runs_total']} complete={row['complete']} "
            f"incomplete={row['incomplete']} failed_status={row['status_failed']} "
            f"missing_status={row['status_missing']}"
        )
    print("-" * 72)
    print(
        f"overall: studies={overall['studies_total']} runs={overall['runs_total']} "
        f"complete={overall['complete']} incomplete={overall['incomplete']} "
        f"failed_status={overall['status_failed']}"
    )

    if args.write_json:
        output_path = Path(args.write_json)
        output_payload = {"overall": overall, "studies": rows}
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(output_payload, indent=2), encoding="utf-8")
        print(f"Wrote: {output_path}")


if __name__ == "__main__":
    main()
