"""Utilities for loading and querying versioned canonical evaluation records.

Evaluation records are produced by evaluation_service.evaluate_study() and stored at:
    <run_root>/evaluations/<protocol_id>/snapshots/step=XXXXXXXX.json

Plotting modules should import from here rather than loading model snapshots directly.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np


def load_evaluation_record(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _evaluation_snapshot_dir(run_root: Path, protocol_id: str, method: str | None = None) -> Path:
    snap_dir = Path(run_root) / "evaluations" / protocol_id / "snapshots"
    return snap_dir / str(method) if method else snap_dir


def list_evaluation_steps(run_root: Path, protocol_id: str, method: str | None = None) -> list[int]:
    """Return sorted step numbers for which evaluation records exist."""
    snap_dir = _evaluation_snapshot_dir(run_root, protocol_id, method)
    if not snap_dir.exists():
        return []
    steps: list[int] = []
    for p in snap_dir.iterdir():
        name = p.name
        if name.startswith("step=") and name.endswith(".json"):
            try:
                steps.append(int(name[5:-5]))
            except ValueError:
                pass
    return sorted(steps)


def load_evaluation_step(
    run_root: Path,
    protocol_id: str,
    step: int | None = None,
    method: str | None = None,
) -> dict[str, Any] | None:
    """Load an exact evaluation record for a method and step.

    If ``method`` is provided, records from another method namespace are
    never substituted. If ``step`` is None, the final available step within
    that exact namespace is loaded.
    """
    steps = list_evaluation_steps(run_root, protocol_id, method)

    if not steps:
        return None

    target = steps[-1] if step is None else int(step)

    if target not in steps:
        return None

    path = (
        _evaluation_snapshot_dir(run_root, protocol_id, method)
        / f"step={target:08d}.json"
    )
    return load_evaluation_record(path)


HYBRID_METHODS = ("mercury_cscg_hybrid", "tmgwr_cscg_hybrid")


def load_study_manifest(study_root: Path) -> dict[tuple[Any, ...], Path]:
    """Load study_manifest.jsonl, returning {(method, level, seed, rand_prob, batch_size, preset): run_root}.

    The preset key is the structural preset when present, otherwise the clone
    preset. This keeps old clone-preset studies and new structural-preset
    studies addressable through the same plotting utilities.

    Hybrid runs are additionally registered under a preset-agnostic key whose
    preset slot is None, so callers that do not care which CSCG configuration a
    hybrid used can address it directly. That collapsed key is only registered
    when exactly one run claims it. When several hybrid runs differ solely in
    their CSCG preset they all collapse onto the same key, and picking one of
    them would silently attribute one run's records to another — so the key is
    omitted and a warning names the candidates. Those runs stay reachable
    through their preset-qualified keys.
    """
    manifest_path = Path(study_root) / "study_manifest.jsonl"
    entries: dict[tuple[Any, ...], Path] = {}
    if not manifest_path.exists():
        return entries
    collapsed_candidates: dict[tuple[Any, ...], list[tuple[str, Any]]] = {}
    with manifest_path.open(encoding="utf-8") as fh:
        for raw in fh:
            raw = raw.strip()
            if not raw:
                continue
            config = dict(json.loads(raw)["config"])
            run_id = str(config["requested_run_id"])
            run_root = Path(study_root) / run_id
            if not (run_root / "run_status.json").exists():
                continue
            preset_key = config.get("cscg_structural_preset") or config.get("cscg_clone_preset")
            identity: tuple[Any, ...] = (
                int(config.get("level")),
                int(config.get("seed")),
                float(config.get("rand_prob")),
                int(config.get("cscg_batch_size", 0)),
            )
            key: tuple[Any, ...] = (config.get("method"), *identity, preset_key)
            entries[key] = run_root
            if config.get("method") in HYBRID_METHODS:
                collapsed_key: tuple[Any, ...] = (config.get("method"), *identity, None)
                collapsed_candidates.setdefault(collapsed_key, []).append((run_id, preset_key))
            if config.get("method") == "mercury_cscg_hybrid" and bool(config.get("baseline_cscg", False)):
                if preset_key is not None:
                    cscg_key: tuple[Any, ...] = ("cscg", *identity, preset_key)
                    entries[cscg_key] = run_root

    ambiguous: list[tuple[tuple[Any, ...], list[tuple[str, Any]]]] = []
    for collapsed_key, candidates in collapsed_candidates.items():
        if len(candidates) == 1:
            entries[collapsed_key] = Path(study_root) / candidates[0][0]
        else:
            # A preset-less hybrid run writes its preset-qualified key straight
            # onto the collapsed key; drop it so no arbitrary winner survives.
            entries.pop(collapsed_key, None)
            ambiguous.append((collapsed_key, candidates))
    if ambiguous:
        detail = "\n".join(
            f"  {key}: " + ", ".join(f"{run_id} (preset={preset!r})" for run_id, preset in sorted(candidates))
            for key, candidates in sorted(ambiguous, key=lambda item: str(item[0]))
        )
        warnings.warn(
            f"{len(ambiguous)} preset-agnostic hybrid manifest key(s) in {study_root} "
            f"are claimed by more than one run and were dropped rather than resolved "
            f"arbitrarily. Address these runs by their preset-qualified key instead:\n"
            f"{detail}",
            stacklevel=2,
        )
    return entries


def n_true_from_record(record: dict[str, Any]) -> int:
    return int(record["state_representation"]["n_true_states"])


def n_actions_from_record(record: dict[str, Any]) -> int:
    return int(record["state_representation"]["n_actions"])


def collapsed_matrices(
    record: dict[str, Any], *, include_sink: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """Return stored collapsed matrices, optionally retaining the ⊥ column."""
    n = n_true_from_record(record)
    coll = record["policy_collapsed"]
    targets = n + 1 if include_sink else n
    P_true = np.asarray(coll["P_true"], dtype=np.float64)[:, :targets]
    P_hat = np.asarray(coll["P_hat"], dtype=np.float64)[:, :targets]
    if include_sink and P_true.shape[1] == n:
        P_true = np.pad(P_true, ((0, 0), (0, 1)))
        P_hat = np.pad(P_hat, ((0, 0), (0, 1)))
    return P_true, P_hat


def action_matrices(
    record: dict[str, Any], *, include_sink: bool = False
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return action matrices and valid rows, optionally retaining ⊥.

    Shapes: (n_actions, n_true, n_true), (n_actions, n_true, n_true), (n_actions, n_true).
    """
    n = n_true_from_record(record)
    ac = record["action_conditioned"]
    targets = n + 1 if include_sink else n
    P_true = np.asarray(ac["P_true_full"], dtype=np.float64)[:, :n, :targets]
    P_hat = np.asarray(ac["P_hat_full"], dtype=np.float64)[:, :n, :targets]
    if include_sink and P_true.shape[2] == n:
        P_true = np.pad(P_true, ((0, 0), (0, 0), (0, 1)))
        P_hat = np.pad(P_hat, ((0, 0), (0, 0), (0, 1)))
    valid = np.asarray(ac["valid_rows"], dtype=bool)
    return P_true, P_hat, valid


def edge_metrics_at_threshold(
    record: dict[str, Any],
    threshold: float,
    *,
    action_conditioned: bool = False,
    include_self_loops: bool = True,
) -> dict[str, float | int]:
    """Compute edge precision/recall/F1 from stored P_true and P_hat at an arbitrary threshold."""
    n = n_true_from_record(record)
    if action_conditioned:
        P_true, P_hat, valid = action_matrices(record)
        mask = valid[:, :, np.newaxis]
        true_edges = (P_true > float(threshold)) & mask
        pred_edges = (P_hat > float(threshold)) & mask
        if not include_self_loops:
            diag = np.zeros((n, n), dtype=bool)
            np.fill_diagonal(diag, True)
            self_loop = diag[np.newaxis, :, :]
            true_edges = true_edges & ~self_loop
            pred_edges = pred_edges & ~self_loop
    else:
        P_true, P_hat = collapsed_matrices(record)
        true_edges = P_true > float(threshold)
        pred_edges = P_hat > float(threshold)
        if not include_self_loops:
            diag = np.zeros((n, n), dtype=bool)
            np.fill_diagonal(diag, True)
            true_edges = true_edges & ~diag
            pred_edges = pred_edges & ~diag

    tp = int(np.count_nonzero(true_edges & pred_edges))
    p_count = int(np.count_nonzero(pred_edges))
    t_count = int(np.count_nonzero(true_edges))
    precision = float(tp / p_count) if p_count else 0.0
    recall = float(tp / t_count) if t_count else 0.0
    denom = precision + recall
    f1 = float(2.0 * precision * recall / denom) if denom > 0.0 else 0.0
    return {
        "edge_precision": precision,
        "edge_recall": recall,
        "edge_f1": f1,
        "true_positive_count": tp,
        "predicted_edge_count": p_count,
        "true_edge_count": t_count,
    }


def _similarity_metric(section: dict[str, Any], tv_key: str, similarity_key: str) -> float:
    if similarity_key in section:
        return float(section[similarity_key])
    return float(1.0 - float(section[tv_key]))


def validate_conditional_nonself_record(
    record: dict[str, Any], *, atol: float = 1e-12
) -> None:
    """Verify that stored v2 no-self scalars match their stored distributions."""
    schema_version = str(record.get("protocol", {}).get("schema_version", ""))
    if "conditional_nonself" not in schema_version:
        return
    from mercury_runs.infrastructure.paper_precision import (
        _action_tv_by_pair,
        compute_occupancy_weighted_action_tv,
        condition_non_selfloop_action_distributions,
    )

    action = record["action_conditioned"]
    P_true, P_hat, valid_rows = action_matrices(record, include_sink=True)
    N_ai = np.asarray(action["source_action_counts"], dtype=np.float64)
    P_hat_ns, P_true_ns, valid_move = condition_non_selfloop_action_distributions(
        P_hat, P_true, valid_rows
    )
    tv_pair = _action_tv_by_pair(P_hat_ns, P_true_ns, valid_move)
    tv_occ, _, _, _, _ = compute_occupancy_weighted_action_tv(
        tv_pair, valid_move, N_ai
    )
    stored_action_tv = float(action["tv_non_self_loop_occupancy"])
    if not np.isclose(tv_occ, stored_action_tv, atol=atol, rtol=0.0):
        raise ValueError(
            "Action-conditioned no-self record inconsistency: "
            f"recomputed TV={tv_occ:.17g}, stored TV={stored_action_tv:.17g}."
        )

    policy = record["policy_collapsed"]
    P_true_coll, P_hat_coll = collapsed_matrices(record, include_sink=True)
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
    total = float(state_counts[valid_move_states].sum())
    count = int(valid_move_states.sum())
    if total > 0.0:
        weights = np.where(valid_move_states, state_counts / total, 0.0)
    else:
        weights = np.where(valid_move_states, 1.0 / count if count else 0.0, 0.0)
    collapsed_tv = float((tv_state * weights).sum())
    stored_collapsed_tv = float(policy["tv_non_self_loop_occupancy"])
    if not np.isclose(collapsed_tv, stored_collapsed_tv, atol=atol, rtol=0.0):
        raise ValueError(
            "Collapsed no-self record inconsistency: "
            f"recomputed TV={collapsed_tv:.17g}, stored TV={stored_collapsed_tv:.17g}."
        )


def diagnostics_from_record(
    record: dict[str, Any],
    *,
    method: str,
    action_conditioned: bool = False,
    ignore_self_loops: bool = False,
    epsilon: float = 1e-6,
) -> "TransitionResidualDiagnostics":
    """Build TransitionResidualDiagnostics from a saved evaluation record.

    Replaces all model loading: P_true and P_hat come directly from the record.
    The ⊥ (unmatched-target-mass) column is excluded from the square matrices
    but its effect is already baked into P_hat rows not summing to 1.
    """
    if ignore_self_loops:
        validate_conditional_nonself_record(record)
    from mercury_runs.analysis.transition_residual_maps import (
        TransitionResidualDiagnostics,
        _collapse_valid_action_rows,
        _collapse_valid_action_values,
        _mean_valid_pair_values,
    )
    from mercury_runs.infrastructure.paper_precision import (
        condition_non_selfloop_action_distributions,
    )

    n = n_true_from_record(record)
    cooccurrence = np.asarray(record["alignment"]["cooccurrence"], dtype=np.int64)
    alignment = np.asarray(record["alignment"]["learned_to_true"], dtype=np.int64)
    occupancy = cooccurrence.sum(axis=1).astype(np.float64)
    projected_source_occupancy = np.zeros(n, dtype=np.float64)
    for k, i in enumerate(alignment.tolist()):
        if 0 <= int(i) < n:
            projected_source_occupancy[int(i)] += float(occupancy[k])

    if action_conditioned:
        P_true_full, P_hat_full, valid_rows = action_matrices(
            record, include_sink=ignore_self_loops
        )
        action = record["action_conditioned"]
        source_action_counts = np.asarray(
            action.get("source_action_counts", valid_rows.astype(np.float64)),
            dtype=np.float64,
        )
        if ignore_self_loops:
            P_hat_full, P_true_full, diagnostic_valid_rows = condition_non_selfloop_action_distributions(
                P_hat_full, P_true_full, valid_rows
            )
        else:
            diagnostic_valid_rows = valid_rows

        residual_by_action = P_hat_full - P_true_full
        transition_true, transition_predicted, residual, residual_magnitude = _collapse_valid_action_rows(
            transition_true_by_action=P_true_full,
            transition_predicted_by_action=P_hat_full,
            residual_by_action=residual_by_action,
            valid_pairs=diagnostic_valid_rows,
            action_weights=source_action_counts,
        )
        excess_mass = _collapse_valid_action_values(
            np.maximum(residual_by_action, 0.0),
            diagnostic_valid_rows,
            action_weights=source_action_counts,
        )
        missing_mass = _collapse_valid_action_values(
            np.maximum(-residual_by_action, 0.0),
            diagnostic_valid_rows,
            action_weights=source_action_counts,
        )
        transition_true = transition_true[:, :n]
        transition_predicted = transition_predicted[:, :n]
        residual = residual[:, :n]
        residual_magnitude = residual_magnitude[:, :n]
        excess_mass = excess_mass[:, :n]
        missing_mass = missing_mass[:, :n]

        if ignore_self_loops:
            tv_by_pair = np.asarray(
                action.get("tv_by_pair_non_self_loop", action["tv_by_pair_full"]),
                dtype=np.float64,
            )
            tv_by_state = np.asarray(
                action.get("tv_non_self_loop_by_state_occupancy", _mean_valid_pair_values(tv_by_pair, valid_rows)),
                dtype=np.float64,
            )
            tv_sim = _similarity_metric(
                action,
                "tv_non_self_loop_occupancy",
                "tv_non_self_loop_similarity_occupancy",
            )
        else:
            tv_by_pair = np.asarray(action["tv_by_pair_full"], dtype=np.float64)
            tv_by_state = np.asarray(
                action.get("tv_by_state_occupancy", _mean_valid_pair_values(tv_by_pair, valid_rows)),
                dtype=np.float64,
            )
            tv_sim = _similarity_metric(action, "tv_occupancy", "tv_similarity_occupancy")

        predicted_row_mass = P_hat_full.sum(axis=2)  # (n_actions, n_true)
        uncovered_rows = (projected_source_occupancy <= epsilon) | (
            np.any(diagnostic_valid_rows, axis=0)
            & ~np.any((predicted_row_mass > epsilon) & diagnostic_valid_rows, axis=0)
        )
    else:
        P_true, P_hat = collapsed_matrices(record, include_sink=ignore_self_loops)
        if ignore_self_loops:
            valid_collapsed = np.asarray(
                record["policy_collapsed"].get(
                    "valid_rows",
                    np.asarray(P_true[:, :n], dtype=np.float64).sum(axis=1) > epsilon,
                ),
                dtype=bool,
            )
            P_hat_3d, P_true_3d, _ = condition_non_selfloop_action_distributions(
                P_hat[np.newaxis, :, :],
                P_true[np.newaxis, :, :],
                valid_collapsed[np.newaxis, :],
            )
            P_hat, P_true = P_hat_3d[0], P_true_3d[0]

        policy = record["policy_collapsed"]
        if ignore_self_loops:
            tv_by_state = np.asarray(
                policy.get("tv_non_self_loop_by_state", policy["tv_by_state"]),
                dtype=np.float64,
            )
            tv_sim = _similarity_metric(
                policy,
                "tv_non_self_loop_occupancy",
                "tv_non_self_loop_similarity_occupancy",
            )
        else:
            tv_by_state = np.asarray(policy["tv_by_state"], dtype=np.float64)
            tv_sim = _similarity_metric(policy, "tv_occupancy", "tv_similarity_occupancy")

        transition_true = P_true[:, :n]
        transition_predicted = P_hat[:, :n]
        residual = transition_predicted - transition_true
        residual_magnitude = None
        excess_mass = np.maximum(residual, 0.0)
        missing_mass = np.maximum(-residual, 0.0)

        truth_row_mass = P_true.sum(axis=1)
        predicted_row_mass = P_hat.sum(axis=1)
        uncovered_rows = (projected_source_occupancy <= epsilon) | (
            (truth_row_mass > epsilon) & (predicted_row_mass <= epsilon)
        )

    similarity_by_state = np.clip(1.0 - tv_by_state, 0.0, 1.0)
    similarity_by_state[uncovered_rows] = 0.0

    return TransitionResidualDiagnostics(
        method=str(method),
        transition_true=transition_true,
        transition_predicted=transition_predicted,
        residual=residual,
        excess_mass=excess_mass,
        missing_mass=missing_mass,
        matched_mass=np.minimum(transition_true, transition_predicted),
        tv_by_state=tv_by_state,
        similarity_by_state=similarity_by_state,
        uncovered_rows=uncovered_rows.astype(bool),
        tv_sim=tv_sim,
        residual_magnitude=residual_magnitude,
    )
