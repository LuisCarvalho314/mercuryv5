#!/usr/bin/env python3
"""Evaluate CSCG initialisations before the first CSCG training batch.

This is intentionally a standalone analysis workflow.  It reads a completed
study, reconstructs the transition counts present at the CSCG handoff, applies
the CSCG pseudocount to every method, and evaluates the resulting frozen model
with the canonical-v4 machinery.  Nothing is written beneath the source study.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for import_root in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

import numpy as np

from mercury.graph.core import Graph
from mercury_runs.algorithms.cscg.prepare import load_ground_truth_states
from mercury_runs.algorithms.cscg.snapshot_adapter import CSCGSnapshotAdapter
from mercury_runs.algorithms.cscg.structural_init import (
    apply_observation_tensor_to_clone_counts,
    clone_edges_from_graph,
    clone_edges_from_oracle_topology,
    observation_edges_from_graph,
    observation_edges_from_oracle_topology,
)
from mercury_runs.algorithms.mercury_cscg_hybrid.prepare import (
    build_hybrid_assignment,
    build_source_graph_observation_assignment,
    load_mercury_latent_graph,
)
from mercury_runs.application.evaluation_protocol import EvaluationProtocol, protocol_from_dict
from mercury_runs.application.evaluation_service import (
    _load_walk_cache,
    _walk_cache_dir,
    _walk_cache_filename,
    build_empirical_ground_truth,
    generate_and_cache_walks,
    save_evaluation_result,
    save_evaluation_status,
    serialise_evaluation_result,
)
from mercury_runs.infrastructure.paper_precision import (
    compute_fixed_map_precision_details,
    compute_transition_statistics,
    concatenate_walk_series,
    evaluate_transition_structure_canonical,
    exact_cartesian_reference_positions,
    exact_valid_sensorimotor_transitions,
    infer_num_actions_from_valid_transitions,
)
from mercury_runs.infrastructure.paper_precision_snapshots import load_cscg_paper_precision_snapshot
from mercury_runs.infrastructure.runtime import atomic_write_json


SUPPORTED_METHODS = ("mercury_cscg_hybrid", "tmgwr_cscg_hybrid", "cscg")
CSCG_PRESETS = (
    "vanilla_cscg",
    "oracle_capacity",
    "oracle_capacity_first_order",
    "oracle_capacity_structural",
)
METHOD_ORDER = {
    "mercury_cscg_hybrid": 0,
    "tmgwr_cscg_hybrid": 1,
    "cscg": 2,
}


@dataclass(frozen=True)
class EvaluationTask:
    source_run: Path
    output_run: Path
    config: dict[str, Any]
    reference_step: int


_WORKER_CONTEXT: dict[str, Any] = {}


def _json_checksum(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    return hashlib.sha256(array.tobytes()).hexdigest()[:16]


def counts_to_transition_tensor(counts: np.ndarray, *, pseudocount: float) -> np.ndarray:
    """Apply one common pseudocount rule and row-normalise action transitions."""
    if float(pseudocount) < 0.0:
        raise ValueError("CSCG pseudocount must be non-negative.")
    transition = np.asarray(counts, dtype=np.float64) + float(pseudocount)
    row_sums = transition.sum(axis=2, keepdims=True)
    if np.any(row_sums <= 0.0):
        raise ValueError(
            "Initial transition tensor has an empty row after pseudocount application; "
            "use a positive pseudocount for empty transition counts."
        )
    transition /= row_sums
    return transition.astype(np.float32, copy=False)


def coverage_failed(row: dict[str, Any]) -> bool:
    if str(row.get("status")) != "complete":
        return True
    requested = int(row.get("requested_walks", 0))
    return bool(
        requested <= 0
        or int(row.get("decoded_walks", 0)) != requested
        or int(row.get("decode_skipped_walks", 0)) != 0
        or int(row.get("decode_unsupported_walks", 0)) != 0
        or int(row.get("nll_skipped_walks", 0)) != 0
        or int(row.get("nll_unsupported_walks", 0)) != 0
    )


def _read_manifest(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for raw in handle:
            raw = raw.strip()
            if not raw:
                continue
            payload = json.loads(raw)
            config = dict(payload.get("config") or {})
            if config:
                rows.append(config)
    return rows


def _parse_seed_selection(raw: str | None) -> set[int] | None:
    if raw is None or not str(raw).strip():
        return None
    selected: set[int] = set()
    for part in str(raw).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, stop = part.split("-", 1)
            selected.update(range(int(start), int(stop) + 1))
        else:
            selected.add(int(part))
    return selected


def _select_configs(
    configs: Iterable[dict[str, Any]],
    *,
    levels: set[int] | None,
    seeds: set[int] | None,
    methods: set[str],
    rand_prob: float,
    batch_size: int,
) -> list[dict[str, Any]]:
    selected = []
    for config in configs:
        method = str(config.get("method"))
        if method not in methods:
            continue
        if levels is not None and int(config.get("level", -1)) not in levels:
            continue
        if seeds is not None and int(config.get("seed", -1)) not in seeds:
            continue
        if float(config.get("rand_prob", "nan")) != float(rand_prob):
            continue
        if int(config.get("cscg_batch_size", -1)) != int(batch_size):
            continue
        if method == "cscg" and str(config.get("cscg_structural_preset")) not in CSCG_PRESETS:
            continue
        selected.append(config)
    return sorted(
        selected,
        key=lambda item: (
            int(item.get("level", -1)),
            METHOD_ORDER.get(str(item.get("method")), 99),
            str(item.get("cscg_structural_preset") or ""),
            int(item.get("seed", -1)),
        ),
    )


def _copy_shadow_metadata(*, source_study: Path, output_study: Path, configs: list[dict[str, Any]]) -> None:
    output_study.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_study / "study_manifest.jsonl", output_study / "study_manifest.jsonl")
    atomic_write_json(
        output_study / "source_study.json",
        {
            "source_study": str(source_study.resolve()),
            "purpose": "CSCG initial structures before CSCG training",
            "source_is_read_only": True,
        },
    )
    for config in configs:
        run_id = str(config["requested_run_id"])
        source_run = source_study / run_id
        output_run = output_study / run_id
        output_run.mkdir(parents=True, exist_ok=True)
        for name in ("run_config.json", "run_status.json"):
            source_path = source_run / name
            if not source_path.exists():
                raise FileNotFoundError(f"Missing source run metadata: {source_path}")
            shutil.copy2(source_path, output_run / name)


def _reference_snapshot(task: EvaluationTask) -> dict[str, Any]:
    method = str(task.config["method"])
    path = (
        task.source_run
        / "snapshots"
        / method
        / "paper_precision"
        / f"step={task.reference_step}"
    )
    if not path.exists():
        raise FileNotFoundError(f"Missing reference CSCG snapshot: {path}")
    return load_cscg_paper_precision_snapshot(path)


def _state_trace_observation_ids(*, task: EvaluationTask, n_clones: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    run_id = task.source_run.name
    states_paths = list((task.source_run / "bundles" / "cscg").rglob(f"{run_id}_states.parquet"))
    ground_truth_paths = list((task.source_run / "ground_truth").glob(f"{run_id}_ground_truth_states.parquet"))
    if len(states_paths) != 1 or len(ground_truth_paths) != 1:
        raise FileNotFoundError(
            f"Expected one CSCG state trace and one ground-truth trace for {run_id}; "
            f"found {len(states_paths)} and {len(ground_truth_paths)}."
        )

    import polars as pl

    frame = pl.read_parquet(states_paths[0], columns=["cscg_state_id"])
    learned_states = frame.get_column("cscg_state_id").to_numpy().astype(np.int64, copy=False)
    ground_truth = np.asarray(load_ground_truth_states(ground_truth_paths[0]), dtype=np.int64)
    length = min(int(learned_states.shape[0]), int(ground_truth.shape[0]))
    learned_states = learned_states[:length]
    ground_truth = ground_truth[:length]
    boundaries = np.cumsum(np.asarray(n_clones, dtype=np.int64))
    if length < 1 or np.any(learned_states < 0) or np.any(learned_states >= int(boundaries[-1])):
        raise ValueError(f"Invalid CSCG state trace for observation reconstruction: {states_paths[0]}")
    # CSCG emissions are fixed blocks: every clone state belongs to exactly one
    # observation.  Recovering that block does not use the learned transition
    # probabilities, even though the persisted state trace was written later.
    observation_ids = np.searchsorted(boundaries, learned_states, side="right").astype(np.int64, copy=False)
    return observation_ids, ground_truth


def reconstruct_initial_counts(
    *,
    task: EvaluationTask,
    snapshot: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Recreate exactly the method-specific count tensor before CSCG training."""
    config = task.config
    method = str(config["method"])
    n_clones = np.asarray(snapshot["n_clones"], dtype=np.int64)
    obs_values = np.asarray(snapshot["obs_uniques"])
    n_obs = int(n_clones.shape[0])
    n_actions = int(np.asarray(snapshot["action_uniques"]).shape[0])
    n_states = int(n_clones.sum())
    init_mode = str(config.get("cscg_structural_init_mode", "none"))
    transfer_level = str(config.get("cscg_structural_transfer_level", "first_order_observation"))
    structural_source = str(config.get("cscg_structural_source", "none"))
    counts = np.zeros((n_actions, n_states, n_states), dtype=np.float32)
    source_graph_path: Path | None = None

    if method in {"mercury_cscg_hybrid", "tmgwr_cscg_hybrid"}:
        if structural_source != "source_graph":
            raise ValueError(f"Unsupported hybrid structural source for isolated evaluation: {structural_source}")
        if method == "mercury_cscg_hybrid":
            source_graph_path = (
                task.source_run / "bundles" / "snapshots" / "mercury" / "paper_precision" / "step=2000" / "latent_graph.npz"
            )
            graph = load_mercury_latent_graph(source_graph_path)
            assignment = build_hybrid_assignment(
                latent_graph=graph,
                obs_idx=np.arange(n_obs, dtype=np.int64),
                n_obs=n_obs,
                memory_length=int(config.get("memory_length", 10)),
            )
        else:
            source_graph_path = (
                task.source_run / "bundles" / "snapshots" / "sensory" / "paper_precision" / "step=2000" / "sensory_graph.npz"
            )
            if not source_graph_path.exists():
                raise FileNotFoundError(f"Missing TMGWR bootstrap graph: {source_graph_path}")
            graph = Graph.from_npz(str(source_graph_path))
            assignment = build_source_graph_observation_assignment(
                graph=graph,
                obs_values=obs_values,
                n_obs=n_obs,
                clone_capacity_mode=str(config.get("cscg_clone_capacity_mode", "source")),
                clones_per_obs=1,
                memory_length=None,
            )
        if not np.array_equal(np.asarray(assignment.n_clones, dtype=np.int64), n_clones):
            raise ValueError(
                f"Reconstructed hybrid clone allocation differs from the saved model for run {task.source_run.name}."
            )
        if transfer_level == "clone_state":
            counts = clone_edges_from_graph(
                graph=graph,
                node_to_state=assignment.node_to_state,
                n_states=n_states,
                n_actions=n_actions,
                mode=init_mode,
            )
        elif transfer_level == "first_order_observation":
            observation_counts = observation_edges_from_graph(
                graph=graph,
                node_to_observation=assignment.node_to_observation,
                n_obs=n_obs,
                n_actions=n_actions,
                mode=init_mode,
            )
            counts = apply_observation_tensor_to_clone_counts(
                observation_tensor=observation_counts,
                n_clones=n_clones,
            )
        else:
            raise ValueError(f"Unsupported hybrid structural transfer level: {transfer_level}")
    elif method == "cscg" and init_mode != "none" and structural_source != "none":
        observation_ids, ground_truth = _state_trace_observation_ids(task=task, n_clones=n_clones)
        dummy_actions = np.zeros_like(observation_ids, dtype=np.int64)
        if transfer_level == "clone_state":
            counts, reconstructed_n_clones = clone_edges_from_oracle_topology(
                level=int(config["level"]),
                obs_idx=observation_ids,
                action_idx=dummy_actions,
                ground_truth_states=ground_truth,
                n_obs=n_obs,
                n_actions=n_actions,
                clones_per_state=int(config.get("cscg_clones_per_state", 1)),
                mode=init_mode,
            )
            if not np.array_equal(np.asarray(reconstructed_n_clones, dtype=np.int64), n_clones):
                raise ValueError(
                    f"Reconstructed oracle clone allocation differs from the saved model for run {task.source_run.name}."
                )
        elif transfer_level == "first_order_observation":
            observation_counts = observation_edges_from_oracle_topology(
                level=int(config["level"]),
                obs_idx=observation_ids,
                action_idx=dummy_actions,
                ground_truth_states=ground_truth,
                n_obs=n_obs,
                n_actions=n_actions,
                mode=init_mode,
            )
            counts = apply_observation_tensor_to_clone_counts(
                observation_tensor=observation_counts,
                n_clones=n_clones,
            )
        else:
            raise ValueError(f"Unsupported oracle structural transfer level: {transfer_level}")

    if counts.shape != (n_actions, n_states, n_states):
        raise ValueError(
            f"Initial counts shape {counts.shape} does not match expected {(n_actions, n_states, n_states)}."
        )
    details = {
        "structural_preset": config.get("cscg_structural_preset"),
        "structural_source": structural_source,
        "transfer_level": transfer_level,
        "init_mode": init_mode,
        "clone_capacity_mode": config.get("cscg_clone_capacity_mode"),
        "source_graph_path": None if source_graph_path is None else str(source_graph_path),
        "count_nonzero": int(np.count_nonzero(counts)),
        "count_sum": float(np.asarray(counts, dtype=np.float64).sum()),
        "count_checksum": _json_checksum(counts),
    }
    return np.asarray(counts, dtype=np.float32), details


def _evaluate_initial_snapshot(
    *,
    task: EvaluationTask,
    protocol: EvaluationProtocol,
    walk_data: dict[str, Any],
    gt: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    reference = _reference_snapshot(task)
    counts, initialization = reconstruct_initial_counts(task=task, snapshot=reference)
    pseudocount = float(task.config.get("cscg_pseudocount", 2e-3))
    transition = counts_to_transition_tensor(counts, pseudocount=pseudocount)
    if np.any(transition <= 0.0):
        raise ValueError(
            f"Pseudocount application left non-positive transitions for run {task.source_run.name}."
        )
    row_sums = np.asarray(transition, dtype=np.float64).sum(axis=2)
    if not np.allclose(row_sums, 1.0, atol=1e-6):
        raise ValueError(f"Initial transition rows are not normalised for run {task.source_run.name}.")

    method = str(task.config["method"])
    batch_size = int(task.config.get("cscg_batch_size", 40))
    snapshot = {
        **reference,
        "T": transition,
        "model_npz": {"T": transition, "n_clones": np.asarray(reference["n_clones"], dtype=np.int64)},
        "meta": {
            **dict(reference.get("meta") or {}),
            "stage": "initial",
            "step": 0,
            "observed_samples": 0,
            "batch_size": batch_size,
        },
    }
    adapter = CSCGSnapshotAdapter(pseudocount=pseudocount, seed=0)
    decoded = adapter.decode_walks(snapshot, walk_data, batch_size=batch_size)
    model_fit = adapter.negative_log2_likelihood(snapshot, walk_data)
    decoded_walks = decoded["decoded_walks"]
    if not decoded_walks:
        raise RuntimeError("No evaluation walks could be decoded.")

    n_learned = adapter.n_learned_states(snapshot)
    C, H, N_ai = compute_transition_statistics(
        decoded_walks=decoded_walks,
        true_walks=decoded["gt_exact_walks"],
        action_walks=decoded["gt_action_walks"],
        n_learned=n_learned,
        n_true=int(gt["n_true"]),
        n_actions=int(gt["n_actions"]),
    )
    inferred_series = concatenate_walk_series(decoded_walks)
    gt_series = concatenate_walk_series(decoded["gt_exact_walks"])
    precision = compute_fixed_map_precision_details(
        inferred_states=inferred_series,
        ground_truth_states=gt_series,
        metric_name="latent_precision",
        inferred_state_count=n_learned,
        ground_truth_state_count=int(gt["n_true"]),
    )
    evaluation = evaluate_transition_structure_canonical(
        T_hat=adapter.transition_tensor(snapshot),
        T_true=np.asarray(gt["T_GT"], dtype=np.float64),
        cooccurrence=C,
        action_source_cooccurrence=H,
        source_action_counts=N_ai,
        edge_thresholds=tuple(float(value) for value in protocol.edge_thresholds),
    )
    inference = adapter.inference_metadata(snapshot, batch_size=batch_size)
    inference.update(
        {
            "requested_walks": int(decoded["requested_walks"]),
            "decoded_walks": int(len(decoded_walks)),
            "skipped_walks": int(decoded["skipped_short_walks"]),
            "unsupported_walks": int(decoded["unsupported_walks"]),
        }
    )
    record = serialise_evaluation_result(
        protocol=protocol,
        level=int(task.config["level"]),
        method=method,
        step=0,
        observed_samples=0,
        walk_cache_id=protocol.walk_cache_key(),
        gt=gt,
        result=evaluation,
        inference_meta=inference,
        latent_precision=float(precision["latent_precision"]),
        node_recall=float(precision["latent_recall"]),
        n_learned=n_learned,
        n_true=int(gt["n_true"]),
        n_actions=int(gt["n_actions"]),
        model_fit=model_fit,
    )
    initialization.update(
        {
            "phase": "before_cscg_training",
            "cscg_observed_samples": 0,
            "bootstrap_observed_samples": 2000 if method in {"mercury_cscg_hybrid", "tmgwr_cscg_hybrid"} else 0,
            "pseudocount": pseudocount,
            "pseudocount_applied_to_all_entries": True,
            "transition_min": float(transition.min()),
            "transition_max": float(transition.max()),
            "transition_row_sum_max_error": float(np.max(np.abs(row_sums - 1.0))),
            "transition_checksum": _json_checksum(transition),
            "reference_snapshot_step": int(task.reference_step),
        }
    )
    record["initialization"] = initialization
    coverage = {
        "run_id": task.source_run.name,
        "method": method,
        "preset": task.config.get("cscg_structural_preset"),
        "level": int(task.config["level"]),
        "seed": int(task.config["seed"]),
        "rand_prob": float(task.config["rand_prob"]),
        "status": "complete",
        "requested_walks": int(decoded["requested_walks"]),
        "decoded_walks": int(len(decoded_walks)),
        "decode_skipped_walks": int(decoded["skipped_short_walks"]),
        "decode_unsupported_walks": int(decoded["unsupported_walks"]),
        "nll_skipped_walks": int(model_fit["skipped_walks"]),
        "nll_unsupported_walks": int(model_fit["unsupported_walks"]),
        "scored_symbols": int(model_fit["scored_symbols"]),
        "pseudocount": pseudocount,
        "count_nonzero": int(initialization["count_nonzero"]),
        "error": None,
    }
    return record, coverage


def _record_path(task: EvaluationTask, protocol: EvaluationProtocol) -> Path:
    return (
        task.output_run
        / "evaluations"
        / protocol.protocol_id()
        / "snapshots"
        / str(task.config["method"])
        / "step=00000000.json"
    )


def _coverage_from_record(record: dict[str, Any], task: EvaluationTask) -> dict[str, Any]:
    inference = dict(record.get("inference") or {})
    model_fit = dict(record.get("model_fit") or {})
    initialization = dict(record.get("initialization") or {})
    return {
        "run_id": task.source_run.name,
        "method": str(task.config["method"]),
        "preset": task.config.get("cscg_structural_preset"),
        "level": int(task.config["level"]),
        "seed": int(task.config["seed"]),
        "rand_prob": float(task.config["rand_prob"]),
        "status": "complete",
        "requested_walks": int(inference.get("requested_walks", 0)),
        "decoded_walks": int(inference.get("decoded_walks", 0)),
        "decode_skipped_walks": int(inference.get("skipped_walks", 0)),
        "decode_unsupported_walks": int(inference.get("unsupported_walks", 0)),
        "nll_skipped_walks": int(model_fit.get("skipped_walks", 0)),
        "nll_unsupported_walks": int(model_fit.get("unsupported_walks", 0)),
        "scored_symbols": int(model_fit.get("scored_symbols", 0)),
        "pseudocount": float(initialization.get("pseudocount", 0.0)),
        "count_nonzero": int(initialization.get("count_nonzero", 0)),
        "error": None,
    }


def _evaluate_task(task: EvaluationTask, *, force: bool = False) -> dict[str, Any]:
    protocol: EvaluationProtocol = _WORKER_CONTEXT["protocol"]
    record_path = _record_path(task, protocol)
    if record_path.exists() and not force:
        return _coverage_from_record(json.loads(record_path.read_text(encoding="utf-8")), task)

    log_dir = Path(_WORKER_CONTEXT["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"run-{task.source_run.name}.log"
    try:
        with log_path.open("w", encoding="utf-8") as log_handle, redirect_stdout(log_handle), redirect_stderr(log_handle):
            record, coverage = _evaluate_initial_snapshot(
                task=task,
                protocol=protocol,
                walk_data=_WORKER_CONTEXT["walk_data"],
                gt=_WORKER_CONTEXT["gt"],
            )
            save_evaluation_result(
                run_root=task.output_run,
                protocol=protocol,
                step=0,
                record=record,
                method=str(task.config["method"]),
            )
            status = "failed" if coverage_failed(coverage) else "complete"
            save_evaluation_status(
                run_root=task.output_run,
                protocol=protocol,
                status=status,
                evaluated_steps=[0],
                method=str(task.config["method"]),
            )
        return coverage
    except Exception as exc:
        try:
            with log_path.open("a", encoding="utf-8") as log_handle:
                import traceback

                traceback.print_exc(file=log_handle)
        except Exception:
            pass
        save_evaluation_status(
            run_root=task.output_run,
            protocol=protocol,
            status="failed",
            evaluated_steps=[],
            method=str(task.config["method"]),
        )
        return {
            "run_id": task.source_run.name,
            "method": str(task.config["method"]),
            "preset": task.config.get("cscg_structural_preset"),
            "level": int(task.config["level"]),
            "seed": int(task.config["seed"]),
            "rand_prob": float(task.config["rand_prob"]),
            "status": "error",
            "requested_walks": int(_WORKER_CONTEXT["walk_data"].get("n_walks", 0)),
            "decoded_walks": 0,
            "decode_skipped_walks": 0,
            "decode_unsupported_walks": 0,
            "nll_skipped_walks": 0,
            "nll_unsupported_walks": 0,
            "scored_symbols": 0,
            "pseudocount": float(task.config.get("cscg_pseudocount", 2e-3)),
            "count_nonzero": 0,
            "error": f"{type(exc).__name__}: {exc}",
        }


def _initialise_worker(
    protocol_dict: dict[str, Any],
    walk_data: dict[str, Any],
    gt: dict[str, Any],
    log_dir: str,
) -> None:
    _WORKER_CONTEXT.clear()
    _WORKER_CONTEXT.update(
        {
            "protocol": protocol_from_dict(protocol_dict),
            "walk_data": walk_data,
            "gt": gt,
            "log_dir": log_dir,
        }
    )


def _load_walks_read_only_or_generate_isolated(
    *,
    source_study: Path,
    output_study: Path,
    protocol: EvaluationProtocol,
    level: int,
) -> tuple[dict[str, Any], str]:
    source_dir = _walk_cache_dir(source_study, protocol)
    filename = _walk_cache_filename(protocol, level)
    if (source_dir / filename).exists():
        return _load_walk_cache(source_dir, filename), "source-study-cache"
    return (
        generate_and_cache_walks(study_root=output_study, protocol=protocol, level=level),
        "isolated-generated-cache",
    )


def _write_coverage_report(output_study: Path, protocol: EvaluationProtocol, rows: list[dict[str, Any]]) -> None:
    report_dir = output_study / "reports" / protocol.protocol_id()
    report_dir.mkdir(parents=True, exist_ok=True)
    ordered = sorted(
        rows,
        key=lambda row: (
            int(row.get("level", -1)),
            METHOD_ORDER.get(str(row.get("method")), 99),
            str(row.get("preset") or ""),
            int(row.get("seed", -1)),
        ),
    )
    failures = [row for row in ordered if coverage_failed(row)]
    complete_rows = [row for row in ordered if str(row.get("status")) == "complete"]
    atomic_write_json(
        report_dir / "coverage.json",
        {
            "protocol_id": protocol.protocol_id(),
            "runs": ordered,
            "summary": {
                "total_runs": len(ordered),
                "complete_runs": len(ordered) - len(failures),
                "failed_or_incomplete_runs": len(failures),
                "all_methods_pseudocounted": bool(complete_rows)
                and all(float(row.get("pseudocount", 0.0)) > 0.0 for row in complete_rows),
            },
        },
    )
    fieldnames = [
        "run_id",
        "method",
        "preset",
        "level",
        "seed",
        "rand_prob",
        "status",
        "requested_walks",
        "decoded_walks",
        "decode_skipped_walks",
        "decode_unsupported_walks",
        "nll_skipped_walks",
        "nll_unsupported_walks",
        "scored_symbols",
        "pseudocount",
        "count_nonzero",
        "error",
    ]
    with (report_dir / "coverage.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in fieldnames} for row in ordered)


def _plot_results(
    *,
    output_study: Path,
    protocol: EvaluationProtocol,
    output_dir: Path,
    levels: tuple[int, ...],
    seeds: tuple[int, ...],
    rand_prob: float,
    batch_size: int,
) -> list[Path]:
    """Render correctly titled initial-structure plots from isolated records."""
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
    import matplotlib

    matplotlib.use("Agg", force=True)
    import numpy as np
    import polars as pl

    from mercury_runs.analysis.evaluation_records import load_study_manifest
    from mercury_runs.analysis.plots import publication_style as pub
    from mercury_runs.analysis.plots import batch_summary_forest as forest
    from mercury_runs.analysis.plots import batch_tv_similarity as tv
    from mercury_runs.analysis.plots import selected_residual_maps as residual
    from mercury_runs.analysis.residual_map_rendering import save_selected_figure

    output_dir.mkdir(parents=True, exist_ok=True)
    forest.LEVEL_ORDER = list(levels)
    tv.LEVEL_ORDER = list(levels)
    frame = forest._build_frame(
        output_study,
        protocol.protocol_id(),
        batch_size=int(batch_size),
        levels=list(levels),
    )
    source = frame.filter(
        (pl.col("comparison") == "final")
        & pl.col("rand_prob").eq(float(rand_prob))
        & pl.col("seed").is_in(list(seeds))
    )
    if source.is_empty():
        raise RuntimeError("No isolated step-0 records were available for plotting.")
    series_order = forest._available_series_order(source, list(forest.HYBRID_SERIES_ORDER))
    labels = forest._series_labels_for_comparison("final")
    colors = forest._series_colors_for_comparison("final")
    title = f"Before CSCG training: initial structures (batch {int(batch_size)})"
    metric_summary_dir = output_dir / "metric_summary"
    metric_summary_dir.mkdir(parents=True, exist_ok=True)
    common = {
        "series_order": series_order,
        "title": title,
        "plot_dir": metric_summary_dir,
        "series_labels": labels,
        "series_colors": colors,
    }
    paths = [
        forest.plot_forest(source, output_name="initial_structure_forest.png", **common),
        forest.plot_metric_matrix(
            source,
            output_name="initial_structure_metric_matrix.png",
            gain_label="Initialisation gain",
            **common,
        ),
        forest.plot_metric_profile(
            source,
            output_name="initial_structure_metric_profile.png",
            metric_keys=forest.PROFILE_FOCUSED_METRICS,
            **common,
        ),
    ]

    for metric_spec in tv.METRIC_SPECS:
        path = (
            output_dir
            / "tv_similarity"
            / f"initial_{metric_spec['family']}_{metric_spec['self_loop_policy']}.png"
        )
        paths.append(
            tv._plot_one(
                source,
                output_path=path,
                rand_prob=float(rand_prob),
                comparison="final",
                comparison_label="Before CSCG training",
                metric=str(metric_spec["column"]),
                metric_label=str(metric_spec["label"]),
                self_loop_policy=str(metric_spec["self_loop_policy"]),
                series_order=series_order,
                primary_series="hybrid",
            )
        )

    manifest = load_study_manifest(output_study)
    scenario = {
        "key": "all_methods_initial",
        "title": "Matched-step transition residuals before CSCG training",
        "targets": (
            (pub.method_label("hybrid"), "hybrid_final"),
            (pub.method_label("tmgwr_hybrid"), "tmgwr_hybrid_final"),
        ),
        "cscg_kind": "cscg_final",
        "output_name": "transition_residual_initial.png",
    }
    for include_self_loops in (False, True):
        for action_conditioned in (False, True):
            payload = residual._build_payload(
                manifest,
                scenario,
                protocol.protocol_id(),
                rand_prob=float(rand_prob),
                include_self_loops=include_self_loops,
                action_conditioned=action_conditioned,
                levels=tuple(levels),
                seeds=tuple(seeds),
            )
            panels = [panel for panel in payload["panels"].values() if panel.get("R")]
            if not panels:
                continue
            scale = max(
                residual.VIS_THRESHOLD,
                max(float(np.max(np.abs(np.asarray(panel["R"], dtype=np.float64)))) for panel in panels),
            )
            suffix = "action_" if action_conditioned else "collapsed_"
            suffix += "with_self_loops" if include_self_loops else "without_self_loops"
            path = output_dir / "residual_maps" / f"initial_structure_{suffix}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            payload["output_name"] = path.name
            payload["visual_encoding"]["global_residual_scale"] = scale
            payload["figure"] = str(path)
            payload["json"] = str(path.with_suffix(".json"))
            path.with_suffix(".json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
            save_selected_figure(
                payload=payload,
                output_path=path,
                vis_threshold=residual.VIS_THRESHOLD,
                max_residual_override=scale,
                font_scale=float(payload.get("font_scale", 1.60)),
                header_font_scale=float(payload.get("header_font_scale", 1.60)),
                omit_leading_zero=bool(payload.get("omit_leading_zero", False)),
            )
            paths.append(path)

    atomic_write_json(
        output_dir / "plot_manifest.json",
        {
            "title": "Before CSCG training",
            "protocol_id": protocol.protocol_id(),
            "figures": [str(path) for path in paths],
        },
    )
    return paths


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", type=Path, default=Path("results/current/studies"))
    parser.add_argument("--study-name", default="main_study")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("results/current/initial_structure_evaluation/main_study"),
    )
    parser.add_argument("--plot-dir", type=Path, default=None)
    parser.add_argument("--levels", type=int, nargs="+", default=[13, 20, 21, 22])
    parser.add_argument("--seeds", default="0-29")
    parser.add_argument("--methods", nargs="+", choices=SUPPORTED_METHODS, default=list(SUPPORTED_METHODS))
    parser.add_argument("--rand-prob", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=40)
    parser.add_argument("--reference-step", type=int, default=4000)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--eval-num-walks", type=int, default=100)
    parser.add_argument("--eval-walk-length", type=int, default=10_000)
    parser.add_argument("--eval-seed", type=int, default=0)
    parser.add_argument("--eval-edge-thresholds", type=float, nargs="+", default=[1e-6])
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    source_study = Path(args.study_root) / str(args.study_name)
    output_study = Path(args.output_root)
    manifest_path = source_study / "study_manifest.jsonl"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Study manifest not found: {manifest_path}")
    levels = set(int(value) for value in args.levels)
    seeds = _parse_seed_selection(args.seeds)
    configs = _select_configs(
        _read_manifest(manifest_path),
        levels=levels,
        seeds=seeds,
        methods=set(str(value) for value in args.methods),
        rand_prob=float(args.rand_prob),
        batch_size=int(args.batch_size),
    )
    if not configs:
        raise RuntimeError("No main-study runs matched the requested initial-structure filters.")
    _copy_shadow_metadata(source_study=source_study, output_study=output_study, configs=configs)

    protocol = EvaluationProtocol(
        sensor="cardinal distance",
        sensor_range=1,
        rand_prob=float(args.rand_prob),
        evaluation_seed=int(args.eval_seed),
        number_of_walks=int(args.eval_num_walks),
        walk_length=int(args.eval_walk_length),
        edge_thresholds=[float(value) for value in args.eval_edge_thresholds],
    )
    atomic_write_json(output_study / "protocol.json", protocol.to_dict())
    print(f"Source study : {source_study}")
    print(f"Output study : {output_study}")
    print(f"Protocol     : {protocol.protocol_id()}")
    print(f"Selected runs: {len(configs)}")

    all_coverage: list[dict[str, Any]] = []
    if not bool(args.skip_eval):
        configs_by_level: dict[int, list[dict[str, Any]]] = {}
        for config in configs:
            configs_by_level.setdefault(int(config["level"]), []).append(config)
        completed = 0
        for level in sorted(configs_by_level):
            walk_data, cache_source = _load_walks_read_only_or_generate_isolated(
                source_study=source_study,
                output_study=output_study,
                protocol=protocol,
                level=level,
            )
            n_true = int(exact_cartesian_reference_positions(level).shape[0])
            n_actions = infer_num_actions_from_valid_transitions(
                exact_valid_sensorimotor_transitions(level_index=level)
            )
            gt = build_empirical_ground_truth(
                walk_data=walk_data,
                n_true=n_true,
                n_actions=n_actions,
                level=level,
            )
            tasks = [
                EvaluationTask(
                    source_run=source_study / str(config["requested_run_id"]),
                    output_run=output_study / str(config["requested_run_id"]),
                    config=config,
                    reference_step=int(args.reference_step),
                )
                for config in configs_by_level[level]
            ]
            log_dir = output_study / "evaluation_logs" / protocol.protocol_id() / f"level={level}"
            print(f"Level {level}: {len(tasks)} run(s), walks={cache_source}")
            if int(args.max_workers) <= 1:
                _initialise_worker(protocol.to_dict(), walk_data, gt, str(log_dir))
                for task in tasks:
                    row = _evaluate_task(task, force=bool(args.force))
                    all_coverage.append(row)
                    completed += 1
                    print(f"  [{completed}/{len(configs)}] {row['run_id']} {row['status']}")
            else:
                try:
                    executor = ProcessPoolExecutor(
                        max_workers=int(args.max_workers),
                        initializer=_initialise_worker,
                        initargs=(protocol.to_dict(), walk_data, gt, str(log_dir)),
                    )
                except (OSError, PermissionError) as exc:
                    print(f"  Parallel workers unavailable ({exc}); using one worker.")
                    _initialise_worker(protocol.to_dict(), walk_data, gt, str(log_dir))
                    for task in tasks:
                        row = _evaluate_task(task, force=bool(args.force))
                        all_coverage.append(row)
                        completed += 1
                        print(f"  [{completed}/{len(configs)}] {row['run_id']} {row['status']}")
                else:
                    with executor:
                        future_to_task = {
                            executor.submit(_evaluate_task, task, force=bool(args.force)): task for task in tasks
                        }
                        for future in as_completed(future_to_task):
                            row = future.result()
                            all_coverage.append(row)
                            completed += 1
                            print(f"  [{completed}/{len(configs)}] {row['run_id']} {row['status']}")
    else:
        _initialise_worker(protocol.to_dict(), {"n_walks": int(args.eval_num_walks)}, {}, str(output_study / "evaluation_logs"))
        for config in configs:
            task = EvaluationTask(
                source_run=source_study / str(config["requested_run_id"]),
                output_run=output_study / str(config["requested_run_id"]),
                config=config,
                reference_step=int(args.reference_step),
            )
            path = _record_path(task, protocol)
            if path.exists():
                all_coverage.append(_coverage_from_record(json.loads(path.read_text(encoding="utf-8")), task))
            else:
                all_coverage.append(
                    {
                        "run_id": task.source_run.name,
                        "method": str(config["method"]),
                        "preset": config.get("cscg_structural_preset"),
                        "level": int(config["level"]),
                        "seed": int(config["seed"]),
                        "rand_prob": float(config["rand_prob"]),
                        "status": "missing",
                        "requested_walks": int(args.eval_num_walks),
                        "decoded_walks": 0,
                        "error": "step-0 record does not exist",
                    }
                )

    _write_coverage_report(output_study, protocol, all_coverage)
    failures = [row for row in all_coverage if coverage_failed(row)]

    if not bool(args.skip_plots):
        available_seeds = tuple(sorted({int(config["seed"]) for config in configs}))
        plot_dir = Path(args.plot_dir) if args.plot_dir is not None else output_study / "plots"
        try:
            paths = _plot_results(
                output_study=output_study,
                protocol=protocol,
                output_dir=plot_dir,
                levels=tuple(sorted(levels)),
                seeds=available_seeds,
                rand_prob=float(args.rand_prob),
                batch_size=int(args.batch_size),
            )
            print(f"Plots written : {len(paths)} file(s) under {plot_dir}")
        except Exception as exc:
            print(f"Plot generation failed: {type(exc).__name__}: {exc}", file=sys.stderr)
            failures.append({"run_id": "<plots>", "error": str(exc)})

    if failures:
        print(f"Initial-structure evaluation found {len(failures)} failed or incomplete item(s):", file=sys.stderr)
        for row in failures[:25]:
            print(
                f"  {row.get('run_id')} method={row.get('method')} preset={row.get('preset')} "
                f"decoded={row.get('decoded_walks')}/{row.get('requested_walks')} error={row.get('error')}",
                file=sys.stderr,
            )
        if len(failures) > 25:
            print(f"  ... and {len(failures) - 25} more; see the coverage report.", file=sys.stderr)
        return 2

    print(f"All {len(all_coverage)} initial structures decoded and scored every requested walk.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
