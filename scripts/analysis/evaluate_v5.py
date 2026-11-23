#!/usr/bin/env python3
"""Isolated evaluation-v5 workflow for frozen, training-matched CSCG inference.

Source studies and canonical-v4 records are read-only.  All v5 metadata,
evaluation records, logs, walk caches, and coverage reports are written to a
shadow study below ``results/current/evaluation_v5/studies`` by default.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for import_root in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

import numpy as np

from scripts.analysis.evaluate_initial_cscg_structures import (
    EvaluationTask as InitialEvaluationTask,
    counts_to_transition_tensor,
    reconstruct_initial_counts,
)
from mercury_runs.algorithms.cscg.prepare import (
    align_post_step_actions_to_cscg,
    import_cscg_module,
    map_to_indices,
)
from mercury_runs.algorithms.mercury.snapshot_adapter import MercurySnapshotAdapter
from mercury_runs.algorithms.sensory.snapshot_adapter import SensorySnapshotAdapter
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
    map_values_with_reference,
)
from mercury_runs.infrastructure.paper_precision_snapshots import (
    load_cscg_paper_precision_snapshot,
)
from mercury_runs.infrastructure.runtime import atomic_write_json


CSCG_METHODS = {"cscg", "mercury_cscg_hybrid", "tmgwr_cscg_hybrid"}
MAIN_STUDY_PRESETS = {
    "vanilla_cscg",
    "oracle_capacity",
    "oracle_capacity_first_order",
    "oracle_capacity_structural",
    "source_graph_clone_state",
    "source_graph_first_order",
}
DEFAULT_STUDIES = ("main_study", "cscg_batch_sweep")
DEFAULT_CHECKPOINTS = (0, 2_000, 4_000, 2_000_000)
_WORKER_CONTEXT: dict[str, Any] = {}


@dataclass
class EvaluationV5Protocol(EvaluationProtocol):
    """Protocol identity for frozen inference matching CSCG online-EM batches."""

    schema_version: str = "evaluation_v5_conditional_nonself_v2"
    protocol_name: str = "evaluation-v5-nonself-v2"
    decoder: str = "posterior_marginal_argmax"
    decoder_configuration_source: str = "run_config"
    online_inference_mode: str = "frozen_training_batch_forward_backward"
    inference_state_initialisation: str = "frozen_model_pi_x_per_batch"
    inference_state_reset_policy: str = "reset_per_training_batch"
    batch_size_source: str = "run_config"
    batch_overlap_samples: int = 1
    batch_construction: str = "prepend_previous_sample"
    posterior_inference: str = "sum_product_forward_times_backward"
    overlap_metric_policy: str = "context_only_excluded_from_state_and_nll"
    parameter_updates: bool = False


@dataclass
class EvaluationV5MatchedBudgetProtocol(EvaluationV5Protocol):
    """V5 inference with a total-sample-matched hybrid final checkpoint."""

    schema_version: str = "evaluation_v5_conditional_nonself_matched_budget_v1"
    protocol_name: str = "evaluation-v5-nonself-matched-budget-v1"
    checkpoint_accounting: str = "hybrid_final_source_plus_cscg_phase"


def evaluation_v5_protocol_from_dict(payload: dict[str, Any]) -> EvaluationV5Protocol:
    protocol_type = (
        EvaluationV5MatchedBudgetProtocol
        if payload.get("schema_version") == EvaluationV5MatchedBudgetProtocol.schema_version
        else EvaluationV5Protocol
    )
    known = set(protocol_type.__dataclass_fields__)
    return protocol_type(**{key: value for key, value in payload.items() if key in known})


def protocol_type_for_study(configs: Iterable[dict[str, Any]]) -> type[EvaluationV5Protocol]:
    """Select the revised identity only for consistently matched hybrid runs."""
    budget_modes: set[str] = set()
    final_step = max(DEFAULT_CHECKPOINTS)
    for config in configs:
        if str(config.get("method")) not in {"mercury_cscg_hybrid", "tmgwr_cscg_hybrid"}:
            continue
        phase_budget = int(config.get("cscg_n_iter", 0))
        source_budget = int(config.get("num_steps", 0))
        if phase_budget < final_step and phase_budget + source_budget == final_step:
            budget_modes.add("matched_total")
        else:
            budget_modes.add("phase")
    if len(budget_modes) > 1:
        raise ValueError("A study cannot mix matched-total and phase-only hybrid budgets.")
    return EvaluationV5MatchedBudgetProtocol if "matched_total" in budget_modes else EvaluationV5Protocol


@dataclass(frozen=True)
class FrozenBatch:
    observations: np.ndarray
    actions: np.ndarray
    overlap: int
    new_samples: int


@dataclass(frozen=True)
class EvaluationTask:
    study_name: str
    source_run: Path
    output_run: Path
    config: dict[str, Any]
    checkpoints: tuple[int, ...]
    reference_step: int = 4_000


def iter_training_matched_batches(
    observations: np.ndarray,
    actions: np.ndarray,
    *,
    batch_size: int,
) -> Iterator[FrozenBatch]:
    """Yield training-equivalent batches for an already-filtered walk.

    Main study and batch sweep retain collision rows, so chunking cached walks
    into ``batch_size`` new rows and prepending one predecessor exactly matches
    ``iter_cscg_training_batches(..., include_prev=True)``.
    """
    obs = np.asarray(observations)
    act = np.asarray(actions)
    if obs.shape[0] != act.shape[0]:
        raise ValueError("Observations and actions must have equal lengths.")
    if int(batch_size) <= 0:
        raise ValueError("batch_size must be positive.")
    for start in range(0, int(obs.shape[0]), int(batch_size)):
        stop = min(start + int(batch_size), int(obs.shape[0]))
        overlap = 0 if start == 0 else 1
        context_start = start - overlap
        yield FrozenBatch(
            observations=obs[context_start:stop],
            actions=act[context_start:stop],
            overlap=overlap,
            new_samples=stop - start,
        )


def posterior_marginal_states(
    *,
    n_clones: np.ndarray,
    observations: np.ndarray,
    forward_messages: np.ndarray,
    backward_messages: np.ndarray,
) -> np.ndarray:
    """Return global clone IDs from per-timestep smoothed marginal argmaxes."""
    clones = np.asarray(n_clones, dtype=np.int64)
    obs = np.asarray(observations, dtype=np.int64)
    if obs.ndim != 1:
        raise ValueError("Indexed observations must be one-dimensional.")
    state_locations = np.hstack((np.asarray([0], dtype=np.int64), clones)).cumsum()
    message_locations = np.hstack(
        (np.asarray([0], dtype=np.int64), clones[obs])
    ).cumsum()
    fwd = np.asarray(forward_messages, dtype=np.float64)
    bwd = np.asarray(backward_messages, dtype=np.float64)
    if fwd.ndim != 1 or bwd.ndim != 1 or fwd.shape != bwd.shape:
        raise ValueError("CSCG forward/backward messages must be aligned packed vectors.")
    if int(fwd.size) != int(message_locations[-1]):
        raise ValueError("Packed message length does not match the observation sequence.")

    states = np.empty(obs.shape[0], dtype=np.int64)
    for timestep, observation in enumerate(obs.tolist()):
        start, stop = message_locations[timestep : timestep + 2]
        posterior = fwd[int(start) : int(stop)] * bwd[int(start) : int(stop)]
        normaliser = float(posterior.sum())
        if not np.isfinite(normaliser) or normaliser <= 0.0:
            raise ValueError(f"Invalid posterior marginal at timestep {timestep}.")
        posterior = posterior / normaliser
        # numpy argmax deterministically selects the smallest local clone on ties.
        states[timestep] = int(state_locations[int(observation)]) + int(np.argmax(posterior))
    return states


def _model_invariants(model: Any) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for name in ("T", "C", "Pi_x", "Pi_a", "n_clones"):
        if hasattr(model, name):
            values[name] = np.asarray(getattr(model, name)).copy()
    for name in ("pseudocount", "dtype"):
        if hasattr(model, name):
            values[name] = getattr(model, name)
    return values


def assert_model_invariants(model: Any, before: dict[str, Any]) -> None:
    for name, expected in before.items():
        actual = getattr(model, name)
        if isinstance(expected, np.ndarray):
            if not np.array_equal(np.asarray(actual), expected, equal_nan=True):
                raise RuntimeError(f"Frozen inference mutated model.{name}.")
        elif actual != expected:
            raise RuntimeError(f"Frozen inference mutated model.{name}.")


def _array_checksum(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    return hashlib.sha256(array.tobytes()).hexdigest()[:16]


def _build_frozen_model(
    *,
    snapshot: dict[str, Any],
    pseudocount: float,
    chmm_mod: Any,
    seed: int,
) -> Any:
    transition = np.asarray(snapshot["T"], dtype=np.float32)
    n_clones = np.asarray(snapshot["n_clones"], dtype=np.int64)
    n_actions = int(transition.shape[0])
    n_observations = int(n_clones.shape[0])
    seed_obs = np.asarray([0, min(1, n_observations - 1)], dtype=np.int64)
    seed_actions = np.asarray([0, max(0, n_actions - 1)], dtype=np.int64)
    model = chmm_mod.CHMM(
        n_clones=n_clones,
        x=seed_obs,
        a=seed_actions,
        pseudocount=float(pseudocount),
        dtype=np.float32,
        seed=int(seed),
    )
    model.T = transition.copy()
    return model


def evaluate_frozen_cscg_snapshot(
    *,
    snapshot: dict[str, Any],
    walk_data: dict[str, Any],
    batch_size: int,
    pseudocount: float,
    seed: int = 0,
    chmm_mod: Any | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Decode and score a frozen CSCG with training-matched batch inference."""
    module = chmm_mod if chmm_mod is not None else import_cscg_module(PROJECT_ROOT)
    n_clones = np.asarray(snapshot["n_clones"], dtype=np.int64)
    obs_uniques = np.asarray(snapshot["obs_uniques"])
    action_uniques = np.asarray(snapshot["action_uniques"])
    model = _build_frozen_model(
        snapshot=snapshot,
        pseudocount=float(pseudocount),
        chmm_mod=module,
        seed=int(seed),
    )
    invariants = _model_invariants(model)

    decoded_walks: list[np.ndarray] = []
    gt_exact_walks: list[np.ndarray] = []
    gt_action_walks: list[np.ndarray] = []
    failures: list[dict[str, Any]] = []
    total_nll = 0.0
    scored_symbols = 0
    overlap_terms_excluded = 0

    walks = zip(
        walk_data["filtered_obs_per_walk"],
        walk_data["filtered_actions_per_walk"],
        walk_data["gt_exact_per_walk"],
        walk_data["gt_action_per_walk"],
    )
    for walk_index, (raw_obs, raw_actions, gt_exact, gt_action) in enumerate(walks):
        if raw_obs is None:
            failures.append({"walk": walk_index, "stage": "mapping", "error": "invalid cached walk"})
            continue
        try:
            obs_idx = map_values_with_reference(
                reference_uniques=obs_uniques,
                values=np.asarray(raw_obs),
            ).astype(np.int64, copy=False)
            act_idx = map_to_indices(action_uniques, np.asarray(raw_actions)).astype(
                np.int64, copy=False
            )
            if obs_idx.shape[0] == 0 or act_idx.shape[0] == 0:
                raise ValueError("empty indexed walk")

            state_parts: list[np.ndarray] = []
            nll_parts: list[np.ndarray] = []
            for batch in iter_training_matched_batches(
                obs_idx,
                act_idx,
                batch_size=int(batch_size),
            ):
                cscg_actions = align_post_step_actions_to_cscg(
                    np.asarray(batch.actions, dtype=np.int64)
                )
                log2_likelihood, forward_messages = module.forward(
                    model.T.transpose(0, 2, 1),
                    model.Pi_x,
                    model.n_clones,
                    np.asarray(batch.observations, dtype=np.int64),
                    cscg_actions,
                    store_messages=True,
                )
                backward_messages = module.backward(
                    model.T,
                    model.n_clones,
                    np.asarray(batch.observations, dtype=np.int64),
                    cscg_actions,
                )
                batch_states = posterior_marginal_states(
                    n_clones=model.n_clones,
                    observations=batch.observations,
                    forward_messages=forward_messages,
                    backward_messages=backward_messages,
                )
                state_parts.append(batch_states[int(batch.overlap) :])
                nll_parts.append(
                    -np.asarray(log2_likelihood, dtype=np.float64)[int(batch.overlap) :]
                )
                overlap_terms_excluded += int(batch.overlap)

            states = np.concatenate(state_parts)
            nll = np.concatenate(nll_parts)
            expected = int(obs_idx.shape[0])
            if int(states.shape[0]) != expected:
                raise AssertionError(f"decoded {states.shape[0]} states for {expected} observations")
            if int(nll.shape[0]) != expected:
                raise AssertionError(f"scored {nll.shape[0]} NLL terms for {expected} observations")
            if int(np.asarray(gt_exact).shape[0]) != expected:
                raise AssertionError("ground-truth state length differs from observation length")
            if int(np.asarray(gt_action).shape[0]) != expected:
                raise AssertionError("ground-truth action length differs from observation length")
            if not np.all(np.isfinite(nll)):
                raise ValueError("non-finite NLL terms")
        except Exception as exc:
            failures.append(
                {
                    "walk": walk_index,
                    "stage": "inference",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            continue

        decoded_walks.append(states)
        gt_exact_walks.append(np.asarray(gt_exact, dtype=np.int64))
        gt_action_walks.append(np.asarray(gt_action, dtype=np.int64))
        total_nll += float(nll.sum())
        scored_symbols += int(nll.shape[0])

    assert_model_invariants(model, invariants)
    requested = int(walk_data["n_walks"])
    skipped = sum(item["stage"] == "mapping" for item in failures)
    unsupported = len(failures) - skipped
    decoded = {
        "decoded_walks": decoded_walks,
        "gt_exact_walks": gt_exact_walks,
        "gt_action_walks": gt_action_walks,
        "skipped_short_walks": int(skipped),
        "unsupported_walks": int(unsupported),
        "requested_walks": requested,
        "decoder": "posterior_marginal_argmax",
        "parameter_updates": False,
        "failures": failures,
    }
    model_fit = {
        "negative_log2_likelihood_sum": float(total_nll),
        "negative_log2_likelihood_mean": (
            float(total_nll) / float(scored_symbols) if scored_symbols else None
        ),
        "bits_per_symbol": float(total_nll) / float(scored_symbols) if scored_symbols else None,
        "scored_symbols": int(scored_symbols),
        "skipped_walks": int(skipped),
        "unsupported_walks": int(unsupported),
        "parameter_updates": False,
        "state_reset": "per_training_batch",
        "overlap_terms_excluded": int(overlap_terms_excluded),
    }
    inference = {
        "decoder": "posterior_marginal_argmax",
        "posterior": "sum_product_forward_times_backward",
        "effective_batch_size": int(batch_size),
        "batch_new_samples": int(batch_size),
        "batch_overlap_samples": 1,
        "batch_construction": "prepend_previous_sample",
        "within_batch_smoothing": True,
        "parameter_updates": False,
        "state_reset": "frozen_model_pi_x_per_training_batch",
        "initial_timestep_policy": "include_once",
        "burn_in_steps": 0,
        "requested_walks": requested,
        "decoded_walks": int(len(decoded_walks)),
        "skipped_walks": int(skipped),
        "unsupported_walks": int(unsupported),
        "model_invariants_checked": sorted(invariants),
        "transition_checksum": _array_checksum(np.asarray(model.T)),
    }
    return decoded, model_fit, inference


def _evaluate_metrics(
    *,
    protocol: EvaluationV5Protocol,
    level: int,
    method: str,
    step: int,
    observed_samples: int,
    walk_cache_id: str,
    gt: dict[str, Any],
    decoded: dict[str, Any],
    transition: np.ndarray,
    n_learned: int,
    inference: dict[str, Any],
    model_fit: dict[str, Any] | None,
) -> dict[str, Any]:
    decoded_walks = decoded["decoded_walks"]
    if not decoded_walks:
        raise RuntimeError("No evaluation walks could be decoded.")
    C, H, N_ai = compute_transition_statistics(
        decoded_walks=decoded_walks,
        true_walks=decoded["gt_exact_walks"],
        action_walks=decoded["gt_action_walks"],
        n_learned=int(n_learned),
        n_true=int(gt["n_true"]),
        n_actions=int(gt["n_actions"]),
    )
    precision = compute_fixed_map_precision_details(
        inferred_states=concatenate_walk_series(decoded_walks),
        ground_truth_states=concatenate_walk_series(decoded["gt_exact_walks"]),
        metric_name="latent_precision",
        inferred_state_count=int(n_learned),
        ground_truth_state_count=int(gt["n_true"]),
    )
    result = evaluate_transition_structure_canonical(
        T_hat=np.asarray(transition, dtype=np.float64),
        T_true=np.asarray(gt["T_GT"], dtype=np.float64),
        cooccurrence=C,
        action_source_cooccurrence=H,
        source_action_counts=N_ai,
        edge_thresholds=tuple(float(value) for value in protocol.edge_thresholds),
    )
    return serialise_evaluation_result(
        protocol=protocol,
        level=int(level),
        method=str(method),
        step=int(step),
        observed_samples=int(observed_samples),
        walk_cache_id=str(walk_cache_id),
        gt=gt,
        result=result,
        inference_meta=inference,
        latent_precision=float(precision["latent_precision"]),
        node_recall=float(precision["latent_recall"]),
        n_learned=int(n_learned),
        n_true=int(gt["n_true"]),
        n_actions=int(gt["n_actions"]),
        model_fit=model_fit,
    )


def _coverage_row(
    *,
    task: EvaluationTask,
    method: str,
    step: int,
    record: dict[str, Any] | None,
    error: str | None = None,
) -> dict[str, Any]:
    inference = dict((record or {}).get("inference") or {})
    model_fit = dict((record or {}).get("model_fit") or {})
    initialization = dict((record or {}).get("initialization") or {})
    requested = int(_WORKER_CONTEXT.get("walk_data", {}).get("n_walks", 0))
    source_phase = str(method).endswith(("_mercury", "_tmgwr"))
    return {
        "study": task.study_name,
        "run_id": task.source_run.name,
        "method": method,
        "preset": task.config.get("cscg_structural_preset"),
        "level": int(task.config["level"]),
        "seed": int(task.config["seed"]),
        "batch_size": int(task.config.get("cscg_batch_size", 40)),
        "step": int(step),
        "status": "complete" if record is not None and error is None else "error",
        "requested_walks": int(inference.get("requested_walks", requested)),
        "decoded_walks": int(inference.get("decoded_walks", 0)),
        "decode_skipped_walks": int(inference.get("skipped_walks", 0)),
        "decode_unsupported_walks": int(inference.get("unsupported_walks", 0)),
        "nll_skipped_walks": int(model_fit.get("skipped_walks", 0)),
        "nll_unsupported_walks": int(model_fit.get("unsupported_walks", 0)),
        "scored_symbols": int(model_fit.get("scored_symbols", 0)),
        "pseudocount": (
            None
            if source_phase
            else float(initialization.get("pseudocount", task.config.get("cscg_pseudocount", 2e-3)))
        ),
        "pseudocount_policy": initialization.get("pseudocount_policy"),
        "error": error,
    }


def coverage_failed(row: dict[str, Any]) -> bool:
    if str(row.get("status")) != "complete":
        return True
    requested = int(row.get("requested_walks", 0))
    source_phase = str(row.get("method", "")).endswith(("_mercury", "_tmgwr"))
    return bool(
        requested <= 0
        or int(row.get("decoded_walks", 0)) != requested
        or int(row.get("decode_skipped_walks", 0)) != 0
        or int(row.get("decode_unsupported_walks", 0)) != 0
        or (
            not source_phase
            and (
                int(row.get("nll_skipped_walks", 0)) != 0
                or int(row.get("nll_unsupported_walks", 0)) != 0
            )
        )
    )


def _record_path(output_run: Path, protocol: EvaluationV5Protocol, method: str, step: int) -> Path:
    return (
        output_run
        / "evaluations"
        / protocol.protocol_id()
        / "snapshots"
        / method
        / f"step={int(step):08d}.json"
    )


def _load_existing_record(
    *, output_run: Path, protocol: EvaluationV5Protocol, method: str, step: int
) -> dict[str, Any] | None:
    path = _record_path(output_run, protocol, method, step)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _initial_snapshot(task: EvaluationTask) -> tuple[dict[str, Any], dict[str, Any]]:
    method = str(task.config["method"])
    reference_step = int(task.reference_step)
    if (
        method in {"mercury_cscg_hybrid", "tmgwr_cscg_hybrid"}
        and int(task.config.get("cscg_n_iter", 0)) < max(DEFAULT_CHECKPOINTS)
        and int(task.config.get("num_steps", 0)) + int(task.config.get("cscg_n_iter", 0))
        == max(DEFAULT_CHECKPOINTS)
    ):
        saved_steps = sorted(int(step) for step in task.config.get("paper_precision_eval_checkpoints", []))
        if reference_step not in saved_steps:
            if not saved_steps:
                raise ValueError("Matched-budget hybrid has no saved CSCG reference snapshots.")
            reference_step = saved_steps[0]
    reference_path = (
        task.source_run
        / "snapshots"
        / method
        / "paper_precision"
        / f"step={reference_step}"
    )
    reference = load_cscg_paper_precision_snapshot(reference_path)
    initial_task = InitialEvaluationTask(
        source_run=task.source_run,
        output_run=task.output_run,
        config=task.config,
        reference_step=reference_step,
    )
    counts, metadata = reconstruct_initial_counts(task=initial_task, snapshot=reference)
    pseudocount = float(task.config.get("cscg_pseudocount", 2e-3))
    transition = counts_to_transition_tensor(counts, pseudocount=pseudocount)
    if np.any(transition <= 0.0):
        raise ValueError("Initial pseudocount failed to produce strictly positive transitions.")
    metadata.update(
        {
            "pseudocount": pseudocount,
            "pseudocount_policy": "applied_once_to_reconstructed_counts",
            "transition_checksum": _array_checksum(transition),
            "reference_snapshot_step": reference_step,
        }
    )
    snapshot = {
        **reference,
        "T": transition,
        "model_npz": {"T": transition, "n_clones": reference["n_clones"]},
        "meta": {
            **dict(reference.get("meta") or {}),
            "stage": "initial",
            "step": 0,
            "observed_samples": 0,
        },
    }
    return snapshot, metadata


def _trained_snapshot(task: EvaluationTask, step: int) -> tuple[dict[str, Any], dict[str, Any]]:
    method = str(task.config["method"])
    phase_step = int(step)
    if method in {"mercury_cscg_hybrid", "tmgwr_cscg_hybrid"} and int(step) == max(DEFAULT_CHECKPOINTS):
        phase_budget = int(task.config.get("cscg_n_iter", 0))
        source_budget = int(task.config.get("num_steps", 0))
        if phase_budget < int(step) and phase_budget + source_budget == int(step):
            phase_step = phase_budget
    path = task.source_run / "snapshots" / method / "paper_precision" / f"step={phase_step}"
    snapshot = load_cscg_paper_precision_snapshot(path)
    if phase_step != int(step):
        meta = dict(snapshot.get("meta") or {})
        if int(meta.get("step", -1)) != phase_step or int(meta.get("observed_samples", -1)) != phase_step:
            raise ValueError(f"Hybrid final snapshot at {path} does not contain {phase_step} CSCG samples.")
    transition = np.asarray(snapshot["T"], dtype=np.float64)
    row_sums = transition.sum(axis=2)
    pseudocount = float(task.config.get("cscg_pseudocount", 2e-3))
    if pseudocount <= 0.0:
        raise ValueError("All evaluation-v5 CSCG runs require a positive configured pseudocount.")
    if np.any(~np.isfinite(transition)) or np.any(transition <= 0.0):
        raise ValueError(f"Saved step {step} contains non-positive or non-finite transitions.")
    if not np.allclose(row_sums, 1.0, atol=1e-6):
        raise ValueError(f"Saved step {step} transition rows are not normalized.")
    return snapshot, {
        "pseudocount": pseudocount,
        "pseudocount_policy": "already_applied_during_training_not_reapplied",
        "transition_checksum": _array_checksum(transition),
        "snapshot_phase_step": phase_step,
    }


def _evaluate_cscg_step(
    *,
    task: EvaluationTask,
    protocol: EvaluationV5Protocol,
    step: int,
) -> dict[str, Any]:
    if int(step) == 0:
        snapshot, initialization = _initial_snapshot(task)
    else:
        snapshot, initialization = _trained_snapshot(task, int(step))
    decoded, model_fit, inference = evaluate_frozen_cscg_snapshot(
        snapshot=snapshot,
        walk_data=_WORKER_CONTEXT["walk_data"],
        batch_size=int(task.config.get("cscg_batch_size", 40)),
        pseudocount=float(task.config.get("cscg_pseudocount", 2e-3)),
        seed=0,
    )
    record = _evaluate_metrics(
        protocol=protocol,
        level=int(task.config["level"]),
        method=str(task.config["method"]),
        step=int(step),
        observed_samples=int(snapshot.get("meta", {}).get("observed_samples", step)),
        walk_cache_id=protocol.walk_cache_key(),
        gt=_WORKER_CONTEXT["gt"],
        decoded=decoded,
        transition=np.asarray(snapshot["T"]),
        n_learned=int(np.asarray(snapshot["n_clones"]).sum()),
        inference=inference,
        model_fit=model_fit,
    )
    record["initialization"] = initialization
    if isinstance(protocol, EvaluationV5MatchedBudgetProtocol):
        phase_samples = int(snapshot.get("meta", {}).get("observed_samples", step))
        source_samples = (
            int(task.config.get("num_steps", 0))
            if str(task.config["method"]) in {"mercury_cscg_hybrid", "tmgwr_cscg_hybrid"}
            else 0
        )
        cumulative_samples = source_samples + phase_samples
        if int(step) == max(DEFAULT_CHECKPOINTS) and cumulative_samples != int(step):
            raise ValueError(
                f"Final checkpoint {step} has {cumulative_samples} total samples "
                f"({source_samples} source + {phase_samples} CSCG)."
            )
        record["training_budget"] = {
            "source_samples": source_samples,
            "cscg_phase_samples": phase_samples,
            "cumulative_samples": cumulative_samples,
        }
    if decoded.get("failures"):
        record["walk_failures"] = decoded["failures"]
    save_evaluation_result(
        run_root=task.output_run,
        protocol=protocol,
        step=int(step),
        record=record,
        method=str(task.config["method"]),
    )
    return record


def _evaluate_source_phase(
    *,
    task: EvaluationTask,
    protocol: EvaluationV5Protocol,
) -> tuple[str, int, dict[str, Any]] | None:
    method = str(task.config["method"])
    if method == "mercury_cscg_hybrid":
        namespace = "mercury_cscg_hybrid_mercury"
        snapshot_path = task.source_run / "bundles" / "snapshots" / "mercury" / "paper_precision" / "step=2000"
        adapter: Any = MercurySnapshotAdapter()
    elif method == "tmgwr_cscg_hybrid":
        namespace = "tmgwr_cscg_hybrid_tmgwr"
        snapshot_path = task.source_run / "bundles" / "snapshots" / "sensory" / "paper_precision" / "step=2000"
        adapter = SensorySnapshotAdapter()
    else:
        return None
    snapshot = adapter.load_snapshot(snapshot_path)
    decoded = adapter.decode_walks(snapshot, _WORKER_CONTEXT["walk_data"], level=int(task.config["level"]))
    inference = adapter.inference_metadata(snapshot)
    inference.update(
        {
            "requested_walks": int(decoded["requested_walks"]),
            "decoded_walks": int(len(decoded["decoded_walks"])),
            "skipped_walks": int(decoded["skipped_short_walks"]),
            "unsupported_walks": int(decoded["unsupported_walks"]),
            "evaluation_v5_cscg_change_applicable": False,
        }
    )
    step = int(adapter.step(snapshot))
    record = _evaluate_metrics(
        protocol=protocol,
        level=int(task.config["level"]),
        method=namespace,
        step=step,
        observed_samples=int(adapter.observed_samples(snapshot)),
        walk_cache_id=protocol.walk_cache_key(),
        gt=_WORKER_CONTEXT["gt"],
        decoded=decoded,
        transition=adapter.transition_tensor(snapshot, n_actions=int(_WORKER_CONTEXT["gt"]["n_actions"])),
        n_learned=int(adapter.n_learned_states(snapshot)),
        inference=inference,
        model_fit=None,
    )
    save_evaluation_result(
        run_root=task.output_run,
        protocol=protocol,
        step=step,
        record=record,
        method=namespace,
    )
    return namespace, step, record


def _cscg_checkpoints_for_task(
    task: EvaluationTask, protocol: EvaluationV5Protocol
) -> tuple[int, ...]:
    checkpoints = tuple(int(step) for step in task.checkpoints)
    if (
        isinstance(protocol, EvaluationV5MatchedBudgetProtocol)
        and str(task.config["method"]) in {"mercury_cscg_hybrid", "tmgwr_cscg_hybrid"}
    ):
        # The hybrid source consumes the first 2,000 samples. Its CSCG phase
        # step 2,000 represents the total-sample 4,000 checkpoint; phase step
        # 4,000 would be an unrequested 6,000-sample checkpoint.
        return tuple(step for step in checkpoints if step != 4_000)
    return checkpoints


def _evaluate_task(task: EvaluationTask, *, force: bool) -> list[dict[str, Any]]:
    protocol: EvaluationV5Protocol = _WORKER_CONTEXT["protocol"]
    log_dir = Path(_WORKER_CONTEXT["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"run-{task.source_run.name}.log"
    coverage: list[dict[str, Any]] = []
    evaluated_by_method: dict[str, list[int]] = {}
    with log_path.open("w", encoding="utf-8") as log_handle:
        with redirect_stdout(log_handle), redirect_stderr(log_handle):
            if task.study_name == "main_study" and bool(_WORKER_CONTEXT["source_phases"]):
                source_namespace = (
                    "mercury_cscg_hybrid_mercury"
                    if str(task.config["method"]) == "mercury_cscg_hybrid"
                    else "tmgwr_cscg_hybrid_tmgwr"
                    if str(task.config["method"]) == "tmgwr_cscg_hybrid"
                    else None
                )
                if source_namespace is not None:
                    existing = _load_existing_record(
                        output_run=task.output_run,
                        protocol=protocol,
                        method=source_namespace,
                        step=2_000,
                    )
                    try:
                        if existing is None or force:
                            result = _evaluate_source_phase(task=task, protocol=protocol)
                            assert result is not None
                            namespace, phase_step, record = result
                        else:
                            namespace, phase_step, record = source_namespace, 2_000, existing
                        coverage.append(
                            _coverage_row(task=task, method=namespace, step=phase_step, record=record)
                        )
                        evaluated_by_method.setdefault(namespace, []).append(phase_step)
                    except Exception as exc:
                        traceback.print_exc()
                        coverage.append(
                            _coverage_row(
                                task=task,
                                method=source_namespace,
                                step=2_000,
                                record=None,
                                error=f"{type(exc).__name__}: {exc}",
                            )
                        )

            method = str(task.config["method"])
            for step in _cscg_checkpoints_for_task(task, protocol):
                existing = _load_existing_record(
                    output_run=task.output_run,
                    protocol=protocol,
                    method=method,
                    step=int(step),
                )
                try:
                    record = (
                        _evaluate_cscg_step(task=task, protocol=protocol, step=int(step))
                        if existing is None or force
                        else existing
                    )
                    coverage.append(
                        _coverage_row(task=task, method=method, step=int(step), record=record)
                    )
                    evaluated_by_method.setdefault(method, []).append(int(step))
                except Exception as exc:
                    traceback.print_exc()
                    coverage.append(
                        _coverage_row(
                            task=task,
                            method=method,
                            step=int(step),
                            record=None,
                            error=f"{type(exc).__name__}: {exc}",
                        )
                    )

    for method in sorted({str(row["method"]) for row in coverage}):
        method_rows = [row for row in coverage if str(row["method"]) == method]
        save_evaluation_status(
            run_root=task.output_run,
            protocol=protocol,
            status="failed" if any(coverage_failed(row) for row in method_rows) else "complete",
            evaluated_steps=sorted(set(evaluated_by_method.get(method, []))),
            method=method,
        )
    return coverage


def _initialise_worker(
    protocol_payload: dict[str, Any],
    walk_data: dict[str, Any],
    gt: dict[str, Any],
    log_dir: str,
    source_phases: bool,
) -> None:
    _WORKER_CONTEXT.clear()
    _WORKER_CONTEXT.update(
        {
            "protocol": evaluation_v5_protocol_from_dict(protocol_payload),
            "walk_data": walk_data,
            "gt": gt,
            "log_dir": log_dir,
            "source_phases": bool(source_phases),
        }
    )


def _read_manifest(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for raw in handle:
            if raw.strip():
                rows.append(dict(json.loads(raw).get("config") or {}))
    return [row for row in rows if row]


def _parse_int_selection(raw: str) -> set[int]:
    selected: set[int] = set()
    for item in str(raw).split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            start, stop = item.split("-", 1)
            selected.update(range(int(start), int(stop) + 1))
        else:
            selected.add(int(item))
    return selected


def _select_configs(
    configs: Iterable[dict[str, Any]],
    *,
    study_name: str,
    levels: set[int],
    seeds: set[int],
    rand_prob: float,
    methods: set[str] | None = None,
) -> list[dict[str, Any]]:
    selected = []
    for config in configs:
        if str(config.get("method")) not in CSCG_METHODS:
            continue
        if methods is not None and str(config.get("method")) not in methods:
            continue
        if int(config.get("level", -1)) not in levels or int(config.get("seed", -1)) not in seeds:
            continue
        if not np.isclose(float(config.get("rand_prob", -1.0)), float(rand_prob)):
            continue
        preset = str(config.get("cscg_structural_preset"))
        if study_name == "main_study" and preset not in MAIN_STUDY_PRESETS:
            continue
        if study_name == "cscg_batch_sweep" and (
            str(config.get("method")) != "cscg" or preset != "vanilla_cscg"
        ):
            continue
        if bool(config.get("cscg_valid_trajectories_only", False)):
            raise ValueError(
                f"evaluation-v5 exact cached-array batching currently requires retained collision rows: {config.get('requested_run_id')}"
            )
        selected.append(config)
    return sorted(
        selected,
        key=lambda row: (
            int(row["level"]),
            int(row.get("cscg_batch_size", 40)),
            str(row["method"]),
            str(row.get("cscg_structural_preset")),
            int(row["seed"]),
        ),
    )


def _copy_shadow_metadata(
    *, source_study: Path, output_study: Path, configs: list[dict[str, Any]]
) -> None:
    output_study.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_study / "study_manifest.jsonl", output_study / "study_manifest.jsonl")
    atomic_write_json(
        output_study / "source_study.json",
        {
            "source_study": str(source_study.resolve()),
            "purpose": "evaluation-v5 frozen training-matched inference",
            "source_is_read_only": True,
        },
    )
    for config in configs:
        run_id = str(config.get("_resolved_run_id", config["requested_run_id"]))
        source_run = source_study / run_id
        output_run = output_study / run_id
        output_run.mkdir(parents=True, exist_ok=True)
        for filename in ("run_config.json", "run_status.json"):
            source_path = source_run / filename
            if not source_path.exists():
                raise FileNotFoundError(source_path)
            shutil.copy2(source_path, output_run / filename)


def _run_identity(config: dict[str, Any]) -> tuple[Any, ...]:
    return (
        str(config.get("method")),
        int(config.get("level", -1)),
        int(config.get("seed", -1)),
        float(config.get("rand_prob", -1.0)),
        int(config.get("cscg_batch_size", -1)),
        str(config.get("cscg_structural_preset")),
    )


def _resolve_physical_run_ids(
    *, source_study: Path, configs: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Resolve reused runs whose physical directory differs from the manifest ID."""
    candidates: dict[tuple[Any, ...], list[tuple[str, str | None]]] = {}
    for config_path in source_study.glob("*/run_config.json"):
        try:
            payload = json.loads(config_path.read_text(encoding="utf-8"))
            run_config = dict(payload.get("cli_args") or payload.get("run_payload") or {})
        except (OSError, json.JSONDecodeError, TypeError):
            continue
        status: str | None = None
        status_path = config_path.parent / "run_status.json"
        try:
            status_payload = json.loads(status_path.read_text(encoding="utf-8"))
            status = str(status_payload.get("status"))
        except (OSError, json.JSONDecodeError, TypeError, AttributeError):
            pass
        candidates.setdefault(_run_identity(run_config), []).append(
            (config_path.parent.name, status)
        )

    resolved: list[dict[str, Any]] = []
    for original in configs:
        config = dict(original)
        requested = str(config["requested_run_id"])
        if (source_study / requested / "run_config.json").exists():
            config["_resolved_run_id"] = requested
        else:
            matches = candidates.get(_run_identity(config), [])
            successful = [
                run_id for run_id, status in matches if status in {"completed", "reused"}
            ]
            if len(successful) == 1:
                config["_resolved_run_id"] = successful[0]
            elif len(matches) == 1:
                config["_resolved_run_id"] = matches[0][0]
            else:
                raise FileNotFoundError(
                    f"Could not uniquely resolve physical run for requested ID {requested}: "
                    f"{[run_id for run_id, _ in matches]}"
                )
        resolved.append(config)
    return resolved


def _load_matching_walk_cache(
    *, source_study: Path, output_study: Path, protocol: EvaluationV5Protocol, level: int
) -> tuple[dict[str, Any], str]:
    filename = _walk_cache_filename(protocol, int(level))
    for protocol_path in sorted((source_study / "evaluation_caches").glob("*/protocol.json")):
        try:
            candidate = protocol_from_dict(json.loads(protocol_path.read_text(encoding="utf-8")))
        except Exception:
            continue
        path = protocol_path.parent / "walks" / filename
        if candidate.walk_cache_key() == protocol.walk_cache_key() and path.exists():
            return _load_walk_cache(path.parent, path.name), str(path)
    output_path = _walk_cache_dir(output_study, protocol) / filename
    if output_path.exists():
        return _load_walk_cache(output_path.parent, output_path.name), str(output_path)
    return (
        generate_and_cache_walks(study_root=output_study, protocol=protocol, level=int(level)),
        "isolated-generated-cache",
    )


def _write_coverage(
    *, output_study: Path, protocol: EvaluationV5Protocol, rows: list[dict[str, Any]]
) -> None:
    report_dir = output_study / "reports" / protocol.protocol_id()
    report_dir.mkdir(parents=True, exist_ok=True)
    ordered = sorted(
        rows,
        key=lambda row: (
            int(row.get("level", -1)),
            int(row.get("batch_size", -1)),
            str(row.get("method")),
            str(row.get("preset")),
            int(row.get("seed", -1)),
            int(row.get("step", -1)),
        ),
    )
    failures = [row for row in ordered if coverage_failed(row)]
    cscg_rows = [
        row
        for row in ordered
        if not str(row.get("method", "")).endswith(("_mercury", "_tmgwr"))
    ]
    initial_rows = [row for row in cscg_rows if int(row.get("step", -1)) == 0]
    atomic_write_json(
        report_dir / "coverage.json",
        {
            "protocol_id": protocol.protocol_id(),
            "rows": ordered,
            "summary": {
                "total_records": len(ordered),
                "complete_records": len(ordered) - len(failures),
                "failed_or_incomplete_records": len(failures),
                "all_cscg_records_positive_pseudocount": bool(cscg_rows)
                and all(float(row["pseudocount"]) > 0.0 for row in cscg_rows),
                "all_initial_structures_pseudocounted_once": bool(initial_rows)
                and all(
                    float(row["pseudocount"]) > 0.0
                    and row.get("pseudocount_policy")
                    == "applied_once_to_reconstructed_counts"
                    for row in initial_rows
                ),
            },
        },
    )
    fieldnames = list(ordered[0]) if ordered else []
    with (report_dir / "coverage.csv").open("w", encoding="utf-8", newline="") as handle:
        if fieldnames:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(ordered)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-studies-root", type=Path, default=Path("results/current/studies"))
    parser.add_argument(
        "--output-studies-root",
        type=Path,
        default=Path("results/current/evaluation_v5/studies"),
    )
    parser.add_argument("--studies", nargs="+", default=list(DEFAULT_STUDIES))
    parser.add_argument("--levels", type=int, nargs="+", default=[13, 20, 21, 22])
    parser.add_argument("--seeds", default="0-29")
    parser.add_argument("--methods", nargs="+", choices=sorted(CSCG_METHODS))
    parser.add_argument("--checkpoints", type=int, nargs="+", default=list(DEFAULT_CHECKPOINTS))
    parser.add_argument("--reference-step", type=int, default=4_000)
    parser.add_argument("--rand-prob", type=float, default=0.1)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--eval-num-walks", type=int, default=100)
    parser.add_argument("--eval-walk-length", type=int, default=10_000)
    parser.add_argument("--eval-seed", type=int, default=0)
    parser.add_argument("--eval-edge-thresholds", type=float, nargs="+", default=[1e-6])
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-source-phases", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    levels = set(int(value) for value in args.levels)
    seeds = _parse_int_selection(args.seeds)
    methods = set(args.methods) if args.methods else None
    checkpoints = tuple(dict.fromkeys(int(value) for value in args.checkpoints))
    if set(checkpoints) != set(DEFAULT_CHECKPOINTS):
        print(f"Using non-default v5 checkpoints: {checkpoints}")
    all_failures: list[dict[str, Any]] = []

    for study_name in args.studies:
        source_study = Path(args.source_studies_root) / str(study_name)
        output_study = Path(args.output_studies_root) / str(study_name)
        manifest_configs = _read_manifest(source_study / "study_manifest.jsonl")
        protocol_type = protocol_type_for_study(manifest_configs)
        protocol = protocol_type(
            sensor="cardinal distance",
            sensor_range=1,
            rand_prob=float(args.rand_prob),
            evaluation_seed=int(args.eval_seed),
            number_of_walks=int(args.eval_num_walks),
            walk_length=int(args.eval_walk_length),
            edge_thresholds=[float(value) for value in args.eval_edge_thresholds],
        )
        print(f"{study_name} protocol: {protocol.protocol_id()}")
        configs = _select_configs(
            manifest_configs,
            study_name=str(study_name),
            levels=levels,
            seeds=seeds,
            rand_prob=float(args.rand_prob),
            methods=methods,
        )
        if not configs:
            raise RuntimeError(f"No compatible runs selected for {study_name}.")
        configs = _resolve_physical_run_ids(source_study=source_study, configs=configs)
        _copy_shadow_metadata(source_study=source_study, output_study=output_study, configs=configs)
        atomic_write_json(output_study / "protocol.json", asdict(protocol))
        print(f"{study_name}: {len(configs)} runs -> {output_study}")

        study_coverage: list[dict[str, Any]] = []
        completed = 0
        for level in sorted(levels):
            level_configs = [config for config in configs if int(config["level"]) == level]
            if not level_configs:
                continue
            walk_data, cache_source = _load_matching_walk_cache(
                source_study=source_study,
                output_study=output_study,
                protocol=protocol,
                level=level,
            )
            gt = build_empirical_ground_truth(
                walk_data=walk_data,
                n_true=int(exact_cartesian_reference_positions(level).shape[0]),
                n_actions=infer_num_actions_from_valid_transitions(
                    exact_valid_sensorimotor_transitions(level_index=level)
                ),
                level=level,
            )
            tasks = [
                EvaluationTask(
                    study_name=str(study_name),
                    source_run=source_study / str(config["_resolved_run_id"]),
                    output_run=output_study / str(config["_resolved_run_id"]),
                    config=config,
                    checkpoints=checkpoints,
                    reference_step=int(args.reference_step),
                )
                for config in level_configs
            ]
            log_dir = output_study / "evaluation_logs" / protocol.protocol_id() / f"level={level}"
            print(f"  level={level}: {len(tasks)} runs; walks={cache_source}")
            worker_args = (
                asdict(protocol),
                walk_data,
                gt,
                str(log_dir),
                not bool(args.skip_source_phases),
            )
            if int(args.max_workers) <= 1:
                _initialise_worker(*worker_args)
                for task in tasks:
                    rows = _evaluate_task(task, force=bool(args.force))
                    study_coverage.extend(rows)
                    completed += 1
                    print(f"    [{completed}/{len(configs)}] {task.source_run.name}")
            else:
                with ProcessPoolExecutor(
                    max_workers=int(args.max_workers),
                    initializer=_initialise_worker,
                    initargs=worker_args,
                ) as executor:
                    futures = {
                        executor.submit(_evaluate_task, task, force=bool(args.force)): task
                        for task in tasks
                    }
                    for future in as_completed(futures):
                        task = futures[future]
                        study_coverage.extend(future.result())
                        completed += 1
                        print(f"    [{completed}/{len(configs)}] {task.source_run.name}")
        _write_coverage(output_study=output_study, protocol=protocol, rows=study_coverage)
        failures = [row for row in study_coverage if coverage_failed(row)]
        all_failures.extend(failures)
        print(f"  records={len(study_coverage)} failures={len(failures)}")

    if all_failures:
        print(f"evaluation-v5 found {len(all_failures)} failed/incomplete records:", file=sys.stderr)
        for row in all_failures[:30]:
            print(
                f"  {row['study']}/{row['run_id']} {row['method']} step={row['step']} "
                f"decoded={row['decoded_walks']}/{row['requested_walks']} error={row['error']}",
                file=sys.stderr,
            )
        if len(all_failures) > 30:
            print(f"  ... {len(all_failures) - 30} more; see coverage reports", file=sys.stderr)
        return 2
    print("evaluation-v5 completed with full walk coverage.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
