"""Version canonical records after correcting conditional no-self-loop metrics.

The stored projected distributions are sufficient statistics for this metric
revision, so no model decoding or training is repeated.  Source protocol
directories remain untouched; corrected records are written under a new
protocol hash.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from mercury_runs.infrastructure.paper_precision import (
    _action_tv_by_pair,
    compute_occupancy_weighted_action_tv,
    condition_non_selfloop_action_distributions,
)
from mercury_runs.infrastructure.runtime import atomic_write_json


def _protocol_id(protocol: dict[str, Any]) -> str:
    canonical = json.dumps(protocol, sort_keys=True, ensure_ascii=True)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
    return f"{protocol['protocol_name']}-{digest}"


def _correct_record(
    record: dict[str, Any],
    *,
    source_protocol_id: str,
    target_schema_version: str,
    target_protocol_name: str,
) -> tuple[dict[str, Any], str, dict[str, float]]:
    if str(record.get("protocol_id")) != source_protocol_id:
        raise ValueError(f"Unexpected source protocol: {record.get('protocol_id')!r}")
    protocol = dict(record["protocol"])
    protocol["schema_version"] = target_schema_version
    protocol["protocol_name"] = target_protocol_name
    target_protocol_id = _protocol_id(protocol)

    action = record["action_conditioned"]
    old_action_similarity = float(action["tv_non_self_loop_similarity_occupancy"])
    P_true = np.asarray(action["P_true_full"], dtype=np.float64)
    P_hat = np.asarray(action["P_hat_full"], dtype=np.float64)
    valid_rows = np.asarray(action["valid_rows"], dtype=bool)
    N_ai = np.asarray(action["source_action_counts"], dtype=np.float64)
    P_hat_ns, P_true_ns, valid_move = condition_non_selfloop_action_distributions(
        P_hat, P_true, valid_rows
    )
    tv_pair = _action_tv_by_pair(P_hat_ns, P_true_ns, valid_move)
    tv_occ, tv_macro, _, tv_state_occ, tv_state_macro = compute_occupancy_weighted_action_tv(
        tv_pair, valid_move, N_ai
    )
    action.update(
        {
            "valid_non_self_loop_rows": valid_move.tolist(),
            "tv_by_pair_non_self_loop": tv_pair.tolist(),
            "tv_non_self_loop_occupancy": float(tv_occ),
            "tv_non_self_loop_similarity_occupancy": float(1.0 - tv_occ),
            "tv_non_self_loop_macro": float(tv_macro),
            "tv_non_self_loop_by_state_occupancy": tv_state_occ.tolist(),
            "tv_non_self_loop_by_state_macro": tv_state_macro.tolist(),
        }
    )

    policy = record["policy_collapsed"]
    old_collapsed_similarity = float(policy["tv_non_self_loop_similarity_occupancy"])
    P_true_coll = np.asarray(policy["P_true"], dtype=np.float64)
    P_hat_coll = np.asarray(policy["P_hat"], dtype=np.float64)
    valid_states = valid_rows.any(axis=0)
    P_hat_coll_ns, P_true_coll_ns, valid_move_states_2d = condition_non_selfloop_action_distributions(
        P_hat_coll[np.newaxis, :, :],
        P_true_coll[np.newaxis, :, :],
        valid_states[np.newaxis, :],
    )
    valid_move_states = valid_move_states_2d[0]
    tv_state = _action_tv_by_pair(
        P_hat_coll_ns, P_true_coll_ns, valid_move_states_2d
    )[0]
    state_counts = N_ai.sum(axis=0)
    move_total = float(state_counts[valid_move_states].sum())
    move_count = int(valid_move_states.sum())
    if move_total > 0.0:
        move_weights = np.where(valid_move_states, state_counts / move_total, 0.0)
    else:
        move_weights = np.where(
            valid_move_states,
            1.0 / move_count if move_count else 0.0,
            0.0,
        )
    collapsed_tv_occ = float((tv_state * move_weights).sum())
    collapsed_tv_macro = float(tv_state[valid_move_states].mean()) if move_count else 0.0
    policy.update(
        {
            "valid_non_self_loop_rows": valid_move_states.tolist(),
            "tv_non_self_loop_by_state": tv_state.tolist(),
            "tv_non_self_loop_occupancy": collapsed_tv_occ,
            "tv_non_self_loop_similarity_occupancy": float(1.0 - collapsed_tv_occ),
            "tv_non_self_loop_macro": collapsed_tv_macro,
        }
    )

    record["protocol"] = protocol
    record["protocol_id"] = target_protocol_id
    record["metric_revision"] = {
        "name": "conditional_nonself_v2",
        "source_protocol_id": source_protocol_id,
        "validity_population": "ground_truth_rows_with_positive_nonself_mass",
        "plot_conditioning": "remove_diagonal_then_renormalise_including_sink",
    }
    differences = {
        "old_action_no_self_similarity": old_action_similarity,
        "new_action_no_self_similarity": float(1.0 - tv_occ),
        "action_no_self_similarity_delta": float(1.0 - tv_occ - old_action_similarity),
        "old_collapsed_no_self_similarity": old_collapsed_similarity,
        "new_collapsed_no_self_similarity": float(1.0 - collapsed_tv_occ),
        "collapsed_no_self_similarity_delta": float(
            1.0 - collapsed_tv_occ - old_collapsed_similarity
        ),
    }
    return record, target_protocol_id, differences


def migrate_study(
    *,
    study_root: Path,
    source_protocol_id: str,
    target_schema_version: str,
    target_protocol_name: str,
) -> dict[str, Any]:
    source_records = sorted(
        study_root.glob(f"*/evaluations/{source_protocol_id}/snapshots/**/*.json")
    )
    if not source_records:
        raise RuntimeError(f"No records found for {source_protocol_id} under {study_root}")

    corrected: list[tuple[Path, dict[str, Any], str]] = []
    difference_rows: list[dict[str, Any]] = []
    target_ids: set[str] = set()
    for source_path in source_records:
        record = json.loads(source_path.read_text(encoding="utf-8"))
        record, target_id, differences = _correct_record(
            record,
            source_protocol_id=source_protocol_id,
            target_schema_version=target_schema_version,
            target_protocol_name=target_protocol_name,
        )
        target_ids.add(target_id)
        corrected.append((source_path, record, target_id))
        difference_rows.append(
            {
                "run_id": source_path.relative_to(study_root).parts[0],
                "method": record.get("method"),
                "level": record.get("level"),
                "snapshot_step": record.get("snapshot_step"),
                **differences,
            }
        )
    if len(target_ids) != 1:
        raise RuntimeError(f"Source records produced multiple target protocol IDs: {target_ids}")
    target_id = next(iter(target_ids))

    source_eval_dirs = sorted(
        {path.parents[2] for path, _, _ in corrected}, key=lambda path: str(path)
    )
    for source_eval_dir in source_eval_dirs:
        target_eval_dir = source_eval_dir.with_name(target_id)
        shutil.copytree(source_eval_dir, target_eval_dir, dirs_exist_ok=True)
        for status_path in target_eval_dir.glob("*/status.json"):
            status = json.loads(status_path.read_text(encoding="utf-8"))
            status["protocol_id"] = target_id
            atomic_write_json(status_path, status)

    for source_path, record, _ in corrected:
        parts = list(source_path.parts)
        protocol_index = parts.index(source_protocol_id)
        parts[protocol_index] = target_id
        atomic_write_json(Path(*parts), record)

    reports_source = study_root / "reports" / source_protocol_id
    reports_target = study_root / "reports" / target_id
    if reports_source.exists():
        shutil.copytree(reports_source, reports_target, dirs_exist_ok=True)
        for report_path in reports_target.rglob("*"):
            if report_path.is_file() and report_path.suffix in {".json", ".csv", ".txt"}:
                text = report_path.read_text(encoding="utf-8")
                report_path.write_text(text.replace(source_protocol_id, target_id), encoding="utf-8")

    protocol_path = study_root / "protocol.json"
    if protocol_path.exists():
        payload = json.loads(protocol_path.read_text(encoding="utf-8"))
        if payload.get("protocol_name") == corrected[0][1]["protocol"]["protocol_name"] or _protocol_id(payload) == source_protocol_id:
            atomic_write_json(protocol_path, corrected[0][1]["protocol"])

    summary = {
        "study_root": str(study_root),
        "source_protocol_id": source_protocol_id,
        "target_protocol_id": target_id,
        "records_migrated": len(corrected),
        "source_records_preserved": True,
    }
    report_dir = study_root / "reports" / target_id
    report_dir.mkdir(parents=True, exist_ok=True)
    differences_path = report_dir / "conditional_nonself_differences.csv"
    with differences_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(difference_rows[0]))
        writer.writeheader()
        writer.writerows(difference_rows)

    def _difference_summary(prefix: str) -> dict[str, float | int]:
        old = np.asarray([row[f"old_{prefix}"] for row in difference_rows], dtype=np.float64)
        new = np.asarray([row[f"new_{prefix}"] for row in difference_rows], dtype=np.float64)
        delta = new - old
        return {
            "old_mean": float(old.mean()),
            "new_mean": float(new.mean()),
            "mean_delta": float(delta.mean()),
            "median_delta": float(np.median(delta)),
            "min_delta": float(delta.min()),
            "max_delta": float(delta.max()),
            "changed_records": int(np.count_nonzero(np.abs(delta) > 1e-12)),
        }

    summary["action_no_self_similarity"] = _difference_summary(
        "action_no_self_similarity"
    )
    summary["collapsed_no_self_similarity"] = _difference_summary(
        "collapsed_no_self_similarity"
    )
    summary["differences_csv"] = str(differences_path)
    atomic_write_json(report_dir / "conditional_nonself_migration.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", type=Path, required=True)
    parser.add_argument("--source-protocol-id", required=True)
    parser.add_argument("--target-schema-version", required=True)
    parser.add_argument("--target-protocol-name", required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            migrate_study(
                study_root=args.study_root,
                source_protocol_id=args.source_protocol_id,
                target_schema_version=args.target_schema_version,
                target_protocol_name=args.target_protocol_name,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
