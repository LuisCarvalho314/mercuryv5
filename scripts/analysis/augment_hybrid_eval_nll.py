from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for path in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from mercury_runs.algorithms.cscg.snapshot_adapter import CSCGSnapshotAdapter
from mercury_runs.analysis.evaluation_records import (
    list_evaluation_steps,
    load_evaluation_step,
    load_study_manifest,
)
from mercury_runs.application.evaluation_protocol import protocol_from_dict
from mercury_runs.application.evaluation_service import (
    _load_walk_cache,
    _walk_cache_dir,
    _walk_cache_filename,
    discover_snapshots,
    snapshot_step_hint,
)
from mercury_runs.infrastructure.runtime import atomic_write_json


def _record_path(run_root: Path, protocol_id: str, step: int, method: str) -> Path | None:
    candidates = [
        run_root / "evaluations" / protocol_id / "snapshots" / method / f"step={step:08d}.json",
        run_root / "evaluations" / protocol_id / "snapshots" / f"step={step:08d}.json",
    ]
    return next((path for path in candidates if path.exists()), None)


def augment_cscg_snapshot_nll(
    *,
    study_root: Path,
    protocol_id: str,
    protocol_ids_by_rand: dict[float, str],
    batch_size: int,
    step: int | None,
) -> int:
    manifest = load_study_manifest(study_root)
    adapter = CSCGSnapshotAdapter()
    updated = 0
    walk_cache: dict[tuple[str, int], dict] = {}

    def sort_key(item: tuple[tuple, Path]) -> tuple:
        key, _run_root = item
        return tuple("" if value is None else value for value in key)

    for (method, level, _seed, rand_prob, bs, preset), run_root in sorted(manifest.items(), key=sort_key):
        if method == "mercury_cscg_hybrid" and preset is not None:
            continue
        if method not in {"cscg", "mercury_cscg_hybrid", "tmgwr_cscg_hybrid"}:
            continue
        if int(bs) != int(batch_size):
            continue
        row_protocol_id = protocol_ids_by_rand.get(float(rand_prob), protocol_id)
        record_steps = list_evaluation_steps(run_root, row_protocol_id, method=str(method))
        method_for_path: str | None = str(method)
        if not record_steps:
            record_steps = list_evaluation_steps(run_root, row_protocol_id)
            method_for_path = None
        if not record_steps:
            continue
        target_step = record_steps[-1] if step is None else int(step)
        record = load_evaluation_step(run_root, row_protocol_id, target_step, method=str(method))
        if record is None:
            continue
        snapshot_path = next(
            (
                path for path in discover_snapshots(run_root, str(method))
                if snapshot_step_hint(path) == int(target_step)
            ),
            None,
        )
        if snapshot_path is None:
            continue
        if method_for_path is None:
            record_path = _record_path(run_root, row_protocol_id, int(target_step), str(method))
        else:
            record_path = (
                run_root
                / "evaluations"
                / row_protocol_id
                / "snapshots"
                / method_for_path
                / f"step={int(target_step):08d}.json"
            )
        if record_path is None:
            continue
        protocol = protocol_from_dict(record["protocol"])
        cache_key = (row_protocol_id, int(level))
        if cache_key not in walk_cache:
            walk_cache[cache_key] = _load_walk_cache(
                _walk_cache_dir(study_root, protocol),
                _walk_cache_filename(protocol, int(level)),
            )
        snapshot = adapter.load_snapshot(snapshot_path)
        record["model_fit"] = adapter.negative_log2_likelihood(snapshot, walk_cache[cache_key])
        atomic_write_json(record_path, record)
        updated += 1
    return updated


def _parse_protocol_ids_by_rand(raw: str | None) -> dict[float, str]:
    if not raw:
        return {}
    result: dict[float, str] = {}
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        rand_text, protocol_id = item.split("=", 1)
        result[float(rand_text)] = protocol_id.strip()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Add CSCG-snapshot eval NLL to existing canonical evaluation records.")
    parser.add_argument("--study_root", type=Path, required=True)
    parser.add_argument("--protocol_id", type=str, required=True)
    parser.add_argument("--protocol_ids_by_rand", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=40)
    parser.add_argument("--step", type=str, default="2000")
    args = parser.parse_args()
    step = None if args.step.lower() == "final" else int(args.step)
    count = augment_cscg_snapshot_nll(
        study_root=args.study_root,
        protocol_id=args.protocol_id,
        protocol_ids_by_rand=_parse_protocol_ids_by_rand(args.protocol_ids_by_rand),
        batch_size=int(args.batch_size),
        step=step,
    )
    print(f"updated {count} evaluation record(s)")


if __name__ == "__main__":
    main()
