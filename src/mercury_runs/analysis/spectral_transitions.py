"""Spectral diagnostics of frozen, GT-projected evaluation records.

The directed matched-target block retains leakage to unmatched targets. No
stationary reweighting, symmetrisation, or row renormalisation is performed.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.linalg import schur, subspace_angles
from scipy.optimize import linear_sum_assignment

from mercury_runs.analysis.evaluation_records import (
    action_matrices,
    list_evaluation_steps,
    load_evaluation_step,
)


def projected_operator(record: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Collapse stored projected tensors using the GT evaluation action policy.

    Return truth, prediction (both n×n), and per-source unmatched probability.
    Reject unsupported GT sources rather than inventing dynamics for them.
    """
    truth, learned, valid = action_matrices(record, include_sink=True)
    if truth.shape != learned.shape or truth.shape != (*valid.shape, valid.shape[1] + 1):
        raise ValueError("Projected tensor shape differs from GT state/action support")
    counts = np.asarray(record["action_conditioned"]["source_action_counts"], dtype=float)
    if counts.shape != valid.shape or not np.isfinite(counts).all() or np.any(counts < 0):
        raise ValueError("Invalid GT action counts")
    if np.any(counts[~valid] != 0) or np.any(counts.sum(axis=0) <= 0):
        raise ValueError("Incomplete GT source support or inconsistent valid rows")
    policy = counts / counts.sum(axis=0, keepdims=True)
    stored_policy = np.asarray(record["policy_collapsed"]["action_weights_evaluation_policy"])
    if not np.allclose(policy, stored_policy, atol=1e-12, rtol=0):
        raise ValueError("Stored policy differs from GT action counts")
    for tensor in (truth, learned):
        if not np.isfinite(tensor).all() or np.any(tensor < 0):
            raise ValueError("Invalid projected probabilities")
        if not np.allclose(tensor.sum(axis=-1), valid.astype(float), atol=1e-10, rtol=0):
            raise ValueError("Projected rows must include all unmatched-target mass")
    if np.any(truth[..., -1] != 0):
        raise ValueError("GT must not leak into the unmatched target")
    collapsed = [np.einsum("ai,aij->ij", policy, tensor) for tensor in (truth, learned)]
    for key, matrix in zip(("P_true", "P_hat"), collapsed):
        if not np.allclose(matrix, record["policy_collapsed"][key], atol=1e-12, rtol=0):
            raise ValueError("Stored collapsed matrix differs from projected tensor")
    return collapsed[0][:, :-1], collapsed[1][:, :-1], collapsed[1][:, -1]


def dominant_subspace(operator: np.ndarray, rank: int, *, tolerance: float = 1e-7) -> np.ndarray | None:
    """Orthonormal right invariant basis; undefined at a tied modulus boundary.

    A complex Schur basis handles nonnormal/defective operators. A boundary tie
    includes conjugate pairs and repeated eigenvalues, so these are never split.
    The dominant (Perron) mode is included in the requested rank.
    """
    n = len(operator)
    if not 1 <= rank < n:
        raise ValueError("Subspace rank must be between 1 and n_states - 1")
    moduli = np.sort(np.abs(np.linalg.eigvals(operator)))[::-1]
    if moduli[rank - 1] - moduli[rank] <= tolerance * max(1.0, moduli[0]):
        return None
    cutoff = (moduli[rank - 1] + moduli[rank]) / 2
    _, basis, selected = schur(operator, output="complex", sort=lambda value: abs(value) > cutoff)
    return basis[:, :rank] if selected == rank else None


def subspace_error(left: np.ndarray | None, right: np.ndarray | None) -> float:
    """Largest sine of principal angles, invariant to basis signs and rotations."""
    if left is None or right is None:
        return float("nan")
    return float(np.sin(subspace_angles(left, right)).max())


def spectral_comparison(
    truth: np.ndarray, learned: np.ndarray, *, rank: int = 2,
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    """Match complex eigenvalue multisets and compare separated slow subspaces."""
    truth = np.asarray(truth, dtype=float)
    learned = np.asarray(learned, dtype=float)
    if truth.ndim != 2 or truth.shape[0] != truth.shape[1] or learned.shape != truth.shape:
        raise ValueError("Operators must be square with matching GT coordinates")
    if not np.isfinite(truth).all() or not np.isfinite(learned).all():
        raise ValueError("Operators must be finite")
    reference = np.linalg.eigvals(truth).astype(complex)
    # Sorting is for output only. Matching uses the complete complex spectrum.
    reference = reference[np.lexsort((reference.imag, reference.real, -np.abs(reference)))]
    predicted = np.linalg.eigvals(learned).astype(complex)
    rows, columns = linear_sum_assignment(np.abs(reference[:, None] - predicted[None, :]))
    matched = predicted[columns[np.argsort(rows)]]
    magnitude = np.sort(np.abs(predicted))[::-1]
    gt_magnitude = np.sort(np.abs(reference))[::-1]
    basis = dominant_subspace(learned, rank)
    gt_basis = dominant_subspace(truth, rank)
    error = np.abs(reference - matched)
    return {
        "eigenvalue_rmse": float(np.sqrt(np.mean(error**2))),
        "eigenvalue_max_error": float(error.max()),
        "spectral_radius": float(magnitude[0]),
        "unit_second_gap": float(1 - magnitude[1]),
        "modulus_gap": float(magnitude[0] - magnitude[1]),
        "gt_unit_second_gap": float(1 - gt_magnitude[1]),
        "gt_modulus_gap": float(gt_magnitude[0] - gt_magnitude[1]),
        "gap_error": float(abs((magnitude[0] - magnitude[1]) - (gt_magnitude[0] - gt_magnitude[1]))),
        "subspace_sin_max": subspace_error(gt_basis, basis),
        "subspace_defined": bool(basis is not None and gt_basis is not None),
        "operator_frobenius_error": float(np.linalg.norm(truth - learned)),
    }, reference, matched


def read_capacity_records(
    study_root: Path, protocol_id: str, level: int,
    presets: tuple[str, ...] = ("oracle_capacity", "vanilla_cscg"),
) -> list[dict[str, Any]]:
    """Read exact CSCG namespaces, checking common GT and evaluation evidence."""
    samples = []
    expected = None
    identities = set()
    for path in sorted(Path(study_root).glob("*/run_config.json")):
        config = json.loads(path.read_text())["cli_args"]
        preset = config.get("cscg_structural_preset")
        if config.get("method") != "cscg" or config.get("level") != level or preset not in presets:
            continue
        steps = list_evaluation_steps(path.parent, protocol_id, "cscg")
        if not steps:
            raise ValueError(f"No records for selected run: {path.parent}")
        identity = (preset, int(config["seed"]))
        if identity in identities:
            raise ValueError(f"Duplicate preset/seed (possibly different training settings): {identity}")
        identities.add(identity)
        for step in steps:
            record = load_evaluation_step(path.parent, protocol_id, step, "cscg")
            if record["protocol_id"] != protocol_id or record["method"] != "cscg" or record["level"] != level or record["snapshot_step"] != step:
                raise ValueError(f"Record identity mismatch: {path.parent}, {step}")
            truth, learned, leakage = projected_operator(record)
            signature = json.dumps({
                "protocol": record["protocol"], "walk_cache_id": record["walk_cache_id"],
                "ground_truth": record["ground_truth"],
                "counts": record["action_conditioned"]["source_action_counts"],
                "policy": record["policy_collapsed"]["action_weights_evaluation_policy"],
                "gt_projection": record["action_conditioned"]["P_true_full"],
                "valid": record["action_conditioned"]["valid_rows"],
                "training": {key: config.get(key) for key in (
                    "num_steps", "cscg_n_iter", "cscg_batch_size", "cscg_online_lambda",
                    "cscg_pseudocount", "cscg_training_mode", "cscg_train_algorithm",
                    "cscg_term_early", "cscg_structural_init_mode", "cscg_structural_source",
                )},
            }, sort_keys=True)
            if expected is not None and signature != expected:
                raise ValueError(f"Incompatible GT, policy, protocol, or training settings: {path.parent}")
            expected = signature
            samples.append({"record": record, "config": config, "truth": truth,
                            "operator": learned, "leakage": leakage, "run_id": path.parent.name,
                            "record_path": str(path.parent / "evaluations" / protocol_id / "snapshots" / "cscg" / f"step={step:08d}.json")})
    if not samples or {sample["config"]["cscg_structural_preset"] for sample in samples} != set(presets):
        raise ValueError(f"Missing requested capacity conditions at level {level}")
    coverage = {}
    for sample in samples:
        config, record = sample["config"], sample["record"]
        coverage.setdefault(config["cscg_structural_preset"], {}).setdefault(config["seed"], []).append(record["snapshot_step"])
    populations = list(coverage.values())
    if any(population != populations[0] for population in populations[1:]):
        raise ValueError("Capacity conditions do not share paired seeds and checkpoints")
    return samples


def read_method_records(study_root: Path, protocol_id: str, level: int) -> list[dict[str, Any]]:
    """Load pure CSCG, source models and hybrid continuations without conflating stages.

    Source models use their own exact namespace, never the CSCG handoff record.
    Hybrid cumulative clocks come from the saved training budget, not file names.
    """
    samples = []
    expected = None
    coverage = {}
    for path in sorted(Path(study_root).glob("*/run_config.json")):
        config = json.loads(path.read_text())["cli_args"]
        if config.get("level") != level:
            continue
        method = config.get("method")
        if method == "cscg" and config.get("cscg_structural_preset") in ("oracle_capacity", "vanilla_cscg"):
            namespaces = [(method, config["cscg_structural_preset"], "cscg")]
        elif method == "mercury_cscg_hybrid":
            namespaces = [("mercury_cscg_hybrid_mercury", "mercury", "source"), (method, "mercury", "hybrid")]
        elif method == "tmgwr_cscg_hybrid":
            namespaces = [("tmgwr_cscg_hybrid_tmgwr", "tmgwr", "source"), (method, "tmgwr", "hybrid")]
        else:
            continue
        for namespace, condition, phase in namespaces:
            identity = (condition, phase, int(config["seed"]))
            if identity in coverage:
                raise ValueError(f"Duplicate method/seed: {identity}")
            steps = list_evaluation_steps(path.parent, protocol_id, namespace)
            if not steps:
                raise ValueError(f"Missing exact source/hybrid namespace: {path.parent}/{namespace}")
            coverage[identity] = []
            for step in steps:
                record = load_evaluation_step(path.parent, protocol_id, step, namespace)
                if record["protocol_id"] != protocol_id or record["method"] != namespace or record["level"] != level or record["snapshot_step"] != step:
                    raise ValueError(f"Record identity mismatch: {path.parent}/{namespace}/{step}")
                truth, learned, leakage = projected_operator(record)
                signature = json.dumps({
                    "protocol": record["protocol"], "walk_cache_id": record["walk_cache_id"],
                    "ground_truth": record["ground_truth"],
                    "counts": record["action_conditioned"]["source_action_counts"],
                    "gt_projection": record["action_conditioned"]["P_true_full"],
                    "valid": record["action_conditioned"]["valid_rows"],
                }, sort_keys=True)
                if expected is not None and signature != expected:
                    raise ValueError(f"Incompatible GT, policy evidence or protocol: {path.parent}/{namespace}")
                expected = signature
                if phase == "hybrid":
                    budget = record["training_budget"]
                    source = int(budget["source_samples"])
                    phase_samples = int(budget["cscg_phase_samples"])
                    cumulative = int(budget["cumulative_samples"])
                    if source != int(config["num_steps"]) or source + phase_samples != cumulative or phase_samples != int(record["observed_samples"]):
                        raise ValueError("Inconsistent hybrid cumulative training budget")
                else:
                    cumulative = int(record["observed_samples"])
                    if cumulative != step:
                        raise ValueError("Source/pure CSCG sample clock differs from saved checkpoint")
                coverage[identity].append(cumulative)
                samples.append({"record": record, "config": config, "truth": truth, "operator": learned,
                                "leakage": leakage, "run_id": path.parent.name, "condition": condition,
                                "phase": "handoff" if phase == "hybrid" and phase_samples == 0 else phase,
                                "checkpoint_step": cumulative,
                                "record_path": str(path.parent / "evaluations" / protocol_id / "snapshots" / namespace / f"step={step:08d}.json")})
    conditions = (("oracle_capacity", "cscg"), ("vanilla_cscg", "cscg"), ("mercury", "source"),
                  ("tmgwr", "source"), ("mercury", "hybrid"), ("tmgwr", "hybrid"))
    populations = [{seed for (name, phase, seed) in coverage if (name, phase) == condition} for condition in conditions]
    if not populations[0] or any(population != populations[0] for population in populations[1:]):
        raise ValueError("Methods do not share the same paired seed population")
    for condition in conditions:
        clocks = [steps for (name, phase, _), steps in coverage.items() if (name, phase) == condition]
        if any(steps != clocks[0] for steps in clocks[1:]):
            raise ValueError(f"Inconsistent checkpoints within {condition}")
    return samples
