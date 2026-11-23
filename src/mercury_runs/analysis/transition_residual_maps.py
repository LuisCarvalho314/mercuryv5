from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import TwoSlopeNorm
from matplotlib.cm import ScalarMappable

from mercury.graph.core import Graph
from mercury_runs.infrastructure.paper_precision import (
    action_ids_from_action_vectors,
    aggregate_aligned_action_transition_tensor,
    aggregate_aligned_weighted_adjacency,
    build_action_conditioned_tensor_from_graph,
    build_action_conditioned_tensor_from_valid_transitions,
    build_action_conditioned_tensor_from_walks,
    build_weighted_adjacency_from_valid_transitions,
    build_weighted_adjacency_from_walks,
    collapse_action_transition_tensor_to_adjacency,
    compute_alignment,
    exact_cartesian_reference_positions,
    exact_cartesian_state_ids_for_level,
    exact_valid_sensorimotor_transitions,
    generate_random_start_walks,
    infer_num_actions_from_valid_transitions,
    normalize_action_rows,
    remap_position_walks_to_global_ids,
    row_normalize,
    zero_action_tensor_diagonal,
    zero_diagonal,
)


SUPPORTED_METHODS = (
    "mercury_latent",
    "mercury_sensory",
    "sensory",
    "cscg",
    "mercury_cscg_hybrid",
)


@dataclass(frozen=True)
class TransitionResidualDiagnostics:
    method: str
    transition_true: np.ndarray
    transition_predicted: np.ndarray
    residual: np.ndarray
    excess_mass: np.ndarray
    missing_mass: np.ndarray
    matched_mass: np.ndarray
    tv_by_state: np.ndarray
    similarity_by_state: np.ndarray
    uncovered_rows: np.ndarray
    tv_sim: float
    residual_magnitude: np.ndarray | None = None

    def to_json_dict(self) -> dict[str, Any]:
        payload = {
            "method": self.method,
            "tv_sim": float(self.tv_sim),
            "tv_by_state": self.tv_by_state.tolist(),
            "similarity_by_state": self.similarity_by_state.tolist(),
            "uncovered_rows": self.uncovered_rows.astype(bool).tolist(),
            "P_gt": self.transition_true.tolist(),
            "P_hat": self.transition_predicted.tolist(),
            "R": self.residual.tolist(),
            "E_plus": self.excess_mass.tolist(),
            "E_minus": self.missing_mass.tolist(),
            "M": self.matched_mass.tolist(),
        }
        if self.residual_magnitude is not None:
            payload["mean_abs_R"] = self.residual_magnitude.tolist()
        return payload


def compute_transition_residual_diagnostics(
    *,
    method: str,
    W_true: np.ndarray,
    W_hat: np.ndarray,
    cooccurrence: np.ndarray | list[list[int]],
    ignore_self_loops: bool = False,
    epsilon: float = 1e-6,
) -> TransitionResidualDiagnostics:
    truth_weights = np.asarray(W_true, dtype=np.float64)
    learned_weights = np.asarray(W_hat, dtype=np.float64)
    contingency = np.asarray(cooccurrence, dtype=np.int64)
    if truth_weights.ndim != 2 or truth_weights.shape[0] != truth_weights.shape[1]:
        raise ValueError(f"Expected square ground-truth adjacency, got {truth_weights.shape!r}")
    if learned_weights.ndim != 2 or learned_weights.shape[0] != learned_weights.shape[1]:
        raise ValueError(f"Expected square learned adjacency, got {learned_weights.shape!r}")
    if contingency.ndim != 2:
        raise ValueError(f"Expected cooccurrence matrix with shape (n_learned, n_true), got {contingency.shape!r}")

    n_learned = int(learned_weights.shape[0])
    n_true = int(truth_weights.shape[0])
    if contingency.shape[0] < n_learned or contingency.shape[1] < n_true:
        padded = np.zeros((max(n_learned, contingency.shape[0]), max(n_true, contingency.shape[1])), dtype=np.int64)
        padded[: contingency.shape[0], : contingency.shape[1]] = contingency
        contingency = padded
    contingency = contingency[:n_learned, :n_true]

    if ignore_self_loops:
        truth_weights = zero_diagonal(truth_weights)
        # Do NOT zero learned_weights before projection: two learned clones that
        # map to the same true state produce post-projection self-loops that must
        # be removed after aggregation, not before.

    alignment = compute_alignment(contingency)
    occupancy = contingency.sum(axis=1).astype(np.float64, copy=False)
    learned_transitions = row_normalize(learned_weights)
    aligned_weights = aggregate_aligned_weighted_adjacency(
        W_hat=learned_transitions,
        A=alignment,
        mu=occupancy,
        n_true=n_true,
    )
    if ignore_self_loops:
        aligned_weights = zero_diagonal(aligned_weights)
    projected_source_occupancy = np.zeros(n_true, dtype=np.float64)
    for true_state in range(n_true):
        projected_source_occupancy[true_state] = float(occupancy[np.where(alignment == true_state)[0]].sum())

    transition_true = row_normalize(truth_weights)
    transition_predicted = row_normalize(aligned_weights)
    residual = transition_predicted - transition_true
    excess_mass = np.maximum(residual, 0.0)
    missing_mass = np.maximum(-residual, 0.0)
    matched_mass = np.minimum(transition_true, transition_predicted)

    truth_row_mass = truth_weights.sum(axis=1)
    predicted_row_mass = aligned_weights.sum(axis=1)
    uncovered_rows = (projected_source_occupancy <= float(epsilon)) | (
        (truth_row_mass > float(epsilon)) & (predicted_row_mass <= float(epsilon))
    )

    tv_by_state = np.zeros(n_true, dtype=np.float64)
    for state_idx in range(n_true):
        if bool(uncovered_rows[state_idx]):
            tv_by_state[state_idx] = 1.0
        elif float(truth_row_mass[state_idx]) <= float(epsilon) and float(predicted_row_mass[state_idx]) <= float(epsilon):
            tv_by_state[state_idx] = 0.0
        else:
            tv_by_state[state_idx] = float(0.5 * np.abs(residual[state_idx]).sum())
    similarity = np.clip(1.0 - tv_by_state, 0.0, 1.0)
    similarity[uncovered_rows] = 0.0

    return TransitionResidualDiagnostics(
        method=str(method),
        transition_true=transition_true,
        transition_predicted=transition_predicted,
        residual=residual,
        excess_mass=excess_mass,
        missing_mass=missing_mass,
        matched_mass=matched_mass,
        tv_by_state=tv_by_state,
        similarity_by_state=similarity,
        uncovered_rows=uncovered_rows.astype(bool, copy=False),
        tv_sim=float(similarity.mean()) if similarity.size else 0.0,
    )


def compute_action_conditioned_transition_residual_diagnostics(
    *,
    method: str,
    T_true: np.ndarray,
    T_hat: np.ndarray,
    cooccurrence: np.ndarray | list[list[int]],
    ignore_self_loops: bool = False,
    epsilon: float = 1e-6,
) -> TransitionResidualDiagnostics:
    truth_weights = np.asarray(T_true, dtype=np.float64)
    learned_weights = np.asarray(T_hat, dtype=np.float64)
    contingency = np.asarray(cooccurrence, dtype=np.int64)
    if truth_weights.ndim != 3 or truth_weights.shape[1] != truth_weights.shape[2]:
        raise ValueError(f"Expected ground-truth action tensor with shape (a, n, n), got {truth_weights.shape!r}")
    if learned_weights.ndim != 3 or learned_weights.shape[1] != learned_weights.shape[2]:
        raise ValueError(f"Expected learned action tensor with shape (a, n, n), got {learned_weights.shape!r}")
    if contingency.ndim != 2:
        raise ValueError(f"Expected cooccurrence matrix with shape (n_learned, n_true), got {contingency.shape!r}")

    n_actions = max(int(truth_weights.shape[0]), int(learned_weights.shape[0]))
    n_true = int(truth_weights.shape[1])
    n_learned = int(learned_weights.shape[1])
    truth_weights = _pad_action_tensor(truth_weights, n_actions=n_actions, n_states=n_true)
    learned_weights = _pad_action_tensor(learned_weights, n_actions=n_actions, n_states=n_learned)
    if contingency.shape[0] < n_learned or contingency.shape[1] < n_true:
        padded = np.zeros((max(n_learned, contingency.shape[0]), max(n_true, contingency.shape[1])), dtype=np.int64)
        padded[: contingency.shape[0], : contingency.shape[1]] = contingency
        contingency = padded
    contingency = contingency[:n_learned, :n_true]

    alignment = compute_alignment(contingency)
    occupancy = contingency.sum(axis=1).astype(np.float64, copy=False)
    learned_distribution = normalize_action_rows(learned_weights)
    aligned_weights = aggregate_aligned_action_transition_tensor(
        learned_distribution,
        alignment,
        occupancy,
        n_true=n_true,
    )
    if ignore_self_loops:
        truth_weights = zero_action_tensor_diagonal(truth_weights)
        aligned_weights = zero_action_tensor_diagonal(aligned_weights)

    transition_true_by_action = normalize_action_rows(truth_weights)
    transition_predicted_by_action = normalize_action_rows(aligned_weights)
    residual_by_action = transition_predicted_by_action - transition_true_by_action
    truth_row_mass = truth_weights.sum(axis=2)
    predicted_row_mass = aligned_weights.sum(axis=2)
    valid_pairs = truth_row_mass > float(epsilon)

    projected_source_occupancy = np.zeros(n_true, dtype=np.float64)
    for true_state in range(n_true):
        projected_source_occupancy[true_state] = float(occupancy[np.where(alignment == true_state)[0]].sum())

    tv_by_pair = np.zeros((n_actions, n_true), dtype=np.float64)
    for action_idx, state_idx in np.argwhere(valid_pairs).tolist():
        if float(predicted_row_mass[int(action_idx), int(state_idx)]) <= float(epsilon):
            tv_by_pair[int(action_idx), int(state_idx)] = 1.0
        else:
            tv_by_pair[int(action_idx), int(state_idx)] = float(
                0.5 * np.abs(residual_by_action[int(action_idx), int(state_idx)]).sum()
            )
    similarity_by_pair = np.clip(1.0 - tv_by_pair, 0.0, 1.0)

    transition_true, transition_predicted, residual, residual_magnitude = _collapse_valid_action_rows(
        transition_true_by_action=transition_true_by_action,
        transition_predicted_by_action=transition_predicted_by_action,
        residual_by_action=residual_by_action,
        valid_pairs=valid_pairs,
    )
    tv_by_state = _mean_valid_pair_values(tv_by_pair, valid_pairs)
    similarity_by_state = _mean_valid_pair_values(similarity_by_pair, valid_pairs)
    uncovered_rows = (projected_source_occupancy <= float(epsilon)) | (
        np.any(valid_pairs, axis=0) & ~np.any((predicted_row_mass > float(epsilon)) & valid_pairs, axis=0)
    )
    similarity_by_state[uncovered_rows] = 0.0
    if np.any(valid_pairs):
        tv_sim = float(np.mean(similarity_by_pair[valid_pairs]))
    else:
        tv_sim = 0.0

    return TransitionResidualDiagnostics(
        method=str(method),
        transition_true=transition_true,
        transition_predicted=transition_predicted,
        residual=residual,
        excess_mass=np.maximum(residual, 0.0),
        missing_mass=np.maximum(-residual, 0.0),
        matched_mass=np.minimum(transition_true, transition_predicted),
        tv_by_state=tv_by_state,
        similarity_by_state=similarity_by_state,
        uncovered_rows=uncovered_rows.astype(bool, copy=False),
        tv_sim=tv_sim,
        residual_magnitude=residual_magnitude,
    )


def _pad_action_tensor(tensor: np.ndarray, *, n_actions: int, n_states: int) -> np.ndarray:
    values = np.asarray(tensor, dtype=np.float64)
    out = np.zeros((int(n_actions), int(n_states), int(n_states)), dtype=np.float64)
    out[: min(int(n_actions), values.shape[0]), : min(int(n_states), values.shape[1]), : min(int(n_states), values.shape[2])] = values[
        : min(int(n_actions), values.shape[0]),
        : min(int(n_states), values.shape[1]),
        : min(int(n_states), values.shape[2]),
    ]
    return out


def _collapse_valid_action_rows(
    *,
    transition_true_by_action: np.ndarray,
    transition_predicted_by_action: np.ndarray,
    residual_by_action: np.ndarray,
    valid_pairs: np.ndarray,
    action_weights: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    transition_true = _collapse_valid_action_values(
        transition_true_by_action, valid_pairs, action_weights=action_weights
    )
    transition_predicted = _collapse_valid_action_values(
        transition_predicted_by_action, valid_pairs, action_weights=action_weights
    )
    residual = _collapse_valid_action_values(
        residual_by_action, valid_pairs, action_weights=action_weights
    )
    residual_magnitude = _collapse_valid_action_values(
        np.abs(residual_by_action), valid_pairs, action_weights=action_weights
    )
    return transition_true, transition_predicted, residual, residual_magnitude


def _collapse_valid_action_values(
    values_by_action: np.ndarray,
    valid_pairs: np.ndarray,
    *,
    action_weights: np.ndarray | None = None,
) -> np.ndarray:
    """Collapse action rows using explicit per-state action weights.

    If weights are omitted, valid actions are averaged uniformly for backwards
    compatibility with legacy diagnostics.  Provided weights are renormalised
    over ``valid_pairs`` independently for each source state.
    """
    values = np.asarray(values_by_action, dtype=np.float64)
    valid = np.asarray(valid_pairs, dtype=bool)
    n_actions, n_states, n_targets = values.shape
    if valid.shape != (n_actions, n_states):
        raise ValueError("valid_pairs must match the action/source dimensions.")
    weights = None if action_weights is None else np.asarray(action_weights, dtype=np.float64)
    if weights is not None and weights.shape != valid.shape:
        raise ValueError("action_weights must match valid_pairs.")
    collapsed = np.zeros((n_states, n_targets), dtype=np.float64)
    for state_idx in range(n_states):
        valid_actions = np.flatnonzero(valid_pairs[:, state_idx])
        if valid_actions.size == 0:
            continue
        if weights is None:
            state_weights = np.full(valid_actions.size, 1.0 / valid_actions.size)
        else:
            state_weights = weights[valid_actions, state_idx].astype(np.float64)
            total = float(state_weights.sum())
            state_weights = (
                state_weights / total
                if total > 0.0
                else np.full(valid_actions.size, 1.0 / valid_actions.size)
            )
        collapsed[state_idx] = np.einsum(
            "a,aj->j", state_weights, values[valid_actions, state_idx, :]
        )
    return collapsed


def _mean_valid_pair_values(values: np.ndarray, valid_pairs: np.ndarray) -> np.ndarray:
    n_states = int(values.shape[1])
    out = np.zeros(n_states, dtype=np.float64)
    for state_idx in range(n_states):
        mask = valid_pairs[:, state_idx]
        if np.any(mask):
            out[state_idx] = float(np.mean(values[mask, state_idx]))
    return out


def generate_transition_residual_maps(
    *,
    run_root: Path,
    methods: Iterable[str] | None = None,
    output_path: Path | None = None,
    vis_threshold: float = 1e-3,
    show_edge_labels: bool = False,
    separate_files: bool = False,
) -> dict[str, Path]:
    run_root = Path(run_root)
    artifacts, run_config = _load_run_context(run_root)
    selected_methods = [str(method) for method in (methods or SUPPORTED_METHODS)]
    W_true, positions, settings = _ground_truth_structure(run_config)

    diagnostics: list[TransitionResidualDiagnostics] = []
    for method in selected_methods:
        maybe = _diagnostics_for_method(
            method=method,
            artifacts=artifacts,
            run_root=run_root,
            run_config=run_config,
            W_true=W_true,
            ignore_self_loops=bool(settings["ignore_self_loops"]),
        )
        if maybe is not None:
            diagnostics.append(maybe)

    if not diagnostics:
        raise RuntimeError(f"No supported metric-backed method artifacts found for {run_root}")

    resolved_output = output_path or (run_root / "plots" / "transition_residual_maps.png")
    outputs: dict[str, Path] = {}
    outputs["combined_plot"] = _save_residual_figure(
        diagnostics=diagnostics,
        positions=positions,
        output_path=resolved_output,
        vis_threshold=float(vis_threshold),
        show_edge_labels=bool(show_edge_labels),
    )
    sidecar = resolved_output.with_suffix(".json")
    outputs["diagnostics_json"] = _write_diagnostics_json(
        diagnostics=diagnostics,
        output_path=sidecar,
        vis_threshold=float(vis_threshold),
        settings=settings,
    )

    if separate_files:
        for diagnostic in diagnostics:
            method_path = resolved_output.with_name(f"{resolved_output.stem}_{diagnostic.method}{resolved_output.suffix}")
            outputs[f"{diagnostic.method}_plot"] = _save_residual_figure(
                diagnostics=[diagnostic],
                positions=positions,
                output_path=method_path,
                vis_threshold=float(vis_threshold),
                show_edge_labels=bool(show_edge_labels),
            )
    return outputs


def _load_run_context(run_root: Path) -> tuple[dict[str, str], dict[str, Any]]:
    status_path = Path(run_root) / "run_status.json"
    if not status_path.exists():
        raise FileNotFoundError(f"Missing run status: {status_path}")
    payload = json.loads(status_path.read_text(encoding="utf-8"))
    artifacts: dict[str, str] = {
        str(key): str(value)
        for key, value in dict(payload.get("artifacts") or {}).items()
        if isinstance(value, str) and value
    }
    for method, subsystem in dict(payload.get("subsystems") or {}).items():
        for key, value in dict((subsystem or {}).get("artifacts") or {}).items():
            if isinstance(value, str) and value:
                artifacts.setdefault(f"{method}_{key}", value)
    return artifacts, dict(payload.get("config") or {})


def _paper_precision_payload(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _latest_metric_row(payload: dict[str, Any], *, phase: str | None = None) -> dict[str, Any]:
    history = payload.get("history")
    if isinstance(history, list) and history:
        for row in reversed(history):
            if not isinstance(row, dict):
                continue
            if phase is not None and str(row.get("phase", phase)) != phase:
                continue
            return row
    metrics = payload.get("metrics")
    if isinstance(metrics, dict):
        return metrics
    return payload if isinstance(payload, dict) else {}


def _cooccurrence_from_payload(payload_path: Path | None, key: str, *, phase: str | None = None) -> np.ndarray | None:
    row = _latest_metric_row(_paper_precision_payload(payload_path), phase=phase)
    matrix = row.get(key)
    if matrix is None:
        return None
    return np.asarray(matrix, dtype=np.int64)


def _diagnostics_for_method(
    *,
    method: str,
    artifacts: dict[str, str],
    run_root: Path,
    run_config: dict[str, Any],
    W_true: np.ndarray,
    ignore_self_loops: bool,
) -> TransitionResidualDiagnostics | None:
    if method == "mercury_latent":
        graph_path = _artifact_path(artifacts, "mercury_latent_graph_npz", "mercury_latent_graph_npz")
        paper_path = _artifact_path(artifacts, "mercury_paper_precision_json", "mercury_paper_precision_json")
        cooccurrence = _cooccurrence_from_payload(paper_path, "latent_exact_state_cooccurrence_matrix")
        if graph_path is None or cooccurrence is None:
            return None
        W_hat = np.asarray(Graph.from_npz(str(graph_path)).adj, dtype=np.float64)
        return compute_transition_residual_diagnostics(
            method=method,
            W_true=W_true,
            W_hat=W_hat,
            cooccurrence=cooccurrence,
            ignore_self_loops=ignore_self_loops,
        )

    if method in {"mercury_sensory", "sensory"}:
        graph_path = _artifact_path(artifacts, "sensory_graph_npz", "sensory_sensory_graph_npz")
        if graph_path is None and method == "mercury_sensory":
            graph_path = _latest_snapshot_file(run_root / "snapshots" / "mercury" / "paper_precision", "sensory_graph.npz")
        paper_path = _artifact_path(
            artifacts,
            "sensory_paper_precision_json" if method == "sensory" else "mercury_paper_precision_json",
            "sensory_paper_precision_json" if method == "sensory" else "mercury_paper_precision_json",
        )
        cooccurrence = _cooccurrence_from_payload(
            paper_path,
            "latent_exact_state_cooccurrence_matrix" if method == "sensory" else "sensory_exact_state_cooccurrence_matrix",
        )
        if graph_path is None or cooccurrence is None:
            return None
        W_hat = np.asarray(Graph.from_npz(str(graph_path)).adj, dtype=np.float64)
        return compute_transition_residual_diagnostics(
            method=method,
            W_true=W_true,
            W_hat=W_hat,
            cooccurrence=cooccurrence,
            ignore_self_loops=ignore_self_loops,
        )

    if method in {"cscg", "mercury_cscg_hybrid"}:
        model_key = "cscg_model_npz" if method == "cscg" else "mercury_cscg_hybrid_model_npz"
        paper_key = "cscg_paper_precision_json" if method == "cscg" else "mercury_cscg_hybrid_paper_precision_json"
        model_path = _artifact_path(artifacts, model_key, model_key)
        paper_path = _artifact_path(artifacts, paper_key, paper_key)
        cooccurrence = _cooccurrence_from_payload(
            paper_path,
            "latent_exact_state_cooccurrence_matrix",
            phase=("cscg" if method == "mercury_cscg_hybrid" else None),
        )
        if model_path is None or cooccurrence is None:
            return None
        model = np.load(model_path, allow_pickle=False)
        W_hat = collapse_action_transition_tensor_to_adjacency(np.asarray(model["T"], dtype=np.float64))
        return compute_transition_residual_diagnostics(
            method=method,
            W_true=W_true,
            W_hat=W_hat,
            cooccurrence=cooccurrence,
            ignore_self_loops=ignore_self_loops,
        )

    return None


def _artifact_path(artifacts: dict[str, str], *keys: str) -> Path | None:
    for key in keys:
        raw = artifacts.get(key)
        if raw:
            path = Path(raw)
            if path.exists():
                return path
    return None


def _latest_snapshot_file(snapshot_root: Path, filename: str) -> Path | None:
    if not snapshot_root.exists():
        return None
    candidates = [
        path / filename
        for path in snapshot_root.iterdir()
        if path.is_dir() and path.name.startswith("step=") and (path / filename).exists()
    ]
    if not candidates:
        return None
    return sorted(candidates, key=lambda path: int(path.parent.name.split("=", 1)[1]))[-1]


def _ground_truth_structure(run_config: dict[str, Any]) -> tuple[np.ndarray, dict[int, np.ndarray], dict[str, Any]]:
    level = int(run_config.get("level", 0))
    sensor = str(run_config.get("sensor", "cartesian"))
    sensor_range = run_config.get("sensor_range")
    paper = dict(((run_config.get("evaluation") or {}).get("paper_precision") or {}))
    structure = dict((paper.get("structure_metrics") or {}))
    scope = str(structure.get("scope", "exact_level")).strip().lower()
    source = str(structure.get("ground_truth_source", "empirical_walks")).strip().lower()
    ignore_self_loops = bool(structure.get("ignore_self_loops", True))

    if scope == "walk_local":
        walks = _evaluation_walks(run_config)
        position_walks = [np.asarray(walk.cartesian_observations) for walk in walks]
        true_walks = remap_position_walks_to_global_ids(position_walks)
        n_true = int(len(np.unique(np.concatenate(true_walks)))) if true_walks else 0
        if source == "maze_topology":
            exact_walks = [
                exact_cartesian_state_ids_for_level(level_index=level, positions=np.asarray(walk.cartesian_observations))
                for walk in walks
            ]
            exact_series = np.concatenate(exact_walks) if exact_walks else np.asarray([], dtype=np.int64)
            local_series = np.concatenate(true_walks) if true_walks else np.asarray([], dtype=np.int64)
            mapping = {
                int(exact_state): int(local_state)
                for exact_state, local_state in zip(exact_series.tolist(), local_series.tolist(), strict=False)
            }
            W_true = build_weighted_adjacency_from_valid_transitions(
                exact_valid_sensorimotor_transitions(level_index=level),
                n_states=n_true,
                state_mapping=mapping,
            )
        else:
            W_true = build_weighted_adjacency_from_walks(true_walks, n_states=n_true)
        positions = _layout_from_size(n_true)
    else:
        reference_positions = exact_cartesian_reference_positions(level)
        n_true = int(reference_positions.shape[0])
        if source == "maze_topology":
            W_true = build_weighted_adjacency_from_valid_transitions(
                exact_valid_sensorimotor_transitions(level_index=level),
                n_states=n_true,
            )
        else:
            walks = _evaluation_walks(run_config)
            exact_walks = [
                exact_cartesian_state_ids_for_level(level_index=level, positions=np.asarray(walk.cartesian_observations))
                for walk in walks
            ]
            W_true = build_weighted_adjacency_from_walks(exact_walks, n_states=n_true)
        positions = {
            int(idx): np.asarray([float(col), -float(row)], dtype=np.float64)
            for idx, (row, col) in enumerate(reference_positions.tolist())
        }

    return W_true, positions, {"scope": scope, "ground_truth_source": source, "ignore_self_loops": ignore_self_loops}


def _ground_truth_action_structure(run_config: dict[str, Any], *, n_actions: int | None = None) -> tuple[np.ndarray, dict[str, Any]]:
    level = int(run_config.get("level", 0))
    paper = dict(((run_config.get("evaluation") or {}).get("paper_precision") or {}))
    structure = dict((paper.get("structure_metrics") or {}))
    scope = str(structure.get("scope", "exact_level")).strip().lower()
    source = str(structure.get("ground_truth_source", "empirical_walks")).strip().lower()
    ignore_self_loops = bool(structure.get("ignore_self_loops", True))
    valid_transitions = exact_valid_sensorimotor_transitions(level_index=level)
    resolved_n_actions = max(
        infer_num_actions_from_valid_transitions(valid_transitions),
        int(n_actions) if n_actions is not None else 0,
    )

    if scope == "walk_local":
        walks = _evaluation_walks(run_config)
        position_walks = [np.asarray(walk.cartesian_observations) for walk in walks]
        true_walks = remap_position_walks_to_global_ids(position_walks)
        n_true = int(len(np.unique(np.concatenate(true_walks)))) if true_walks else 0
        if source == "maze_topology":
            exact_walks = [
                exact_cartesian_state_ids_for_level(level_index=level, positions=np.asarray(walk.cartesian_observations))
                for walk in walks
            ]
            exact_series = np.concatenate(exact_walks) if exact_walks else np.asarray([], dtype=np.int64)
            local_series = np.concatenate(true_walks) if true_walks else np.asarray([], dtype=np.int64)
            mapping = {
                int(exact_state): int(local_state)
                for exact_state, local_state in zip(exact_series.tolist(), local_series.tolist(), strict=False)
            }
            T_true = build_action_conditioned_tensor_from_valid_transitions(
                valid_transitions,
                n_states=n_true,
                n_actions=resolved_n_actions,
                state_mapping=mapping,
            )
        else:
            action_walks = [action_ids_from_action_vectors(np.asarray(walk.actions)) for walk in walks]
            T_true = build_action_conditioned_tensor_from_walks(
                state_walks=true_walks,
                action_walks=action_walks,
                n_states=n_true,
                n_actions=resolved_n_actions,
            )
    else:
        reference_positions = exact_cartesian_reference_positions(level)
        n_true = int(reference_positions.shape[0])
        if source == "maze_topology":
            T_true = build_action_conditioned_tensor_from_valid_transitions(
                valid_transitions,
                n_states=n_true,
                n_actions=resolved_n_actions,
            )
        else:
            walks = _evaluation_walks(run_config)
            exact_walks = [
                exact_cartesian_state_ids_for_level(level_index=level, positions=np.asarray(walk.cartesian_observations))
                for walk in walks
            ]
            action_walks = [action_ids_from_action_vectors(np.asarray(walk.actions)) for walk in walks]
            T_true = build_action_conditioned_tensor_from_walks(
                state_walks=exact_walks,
                action_walks=action_walks,
                n_states=n_true,
                n_actions=resolved_n_actions,
            )

    return T_true, {"scope": scope, "ground_truth_source": source, "ignore_self_loops": ignore_self_loops}


def _evaluation_walks(run_config: dict[str, Any]) -> list[Any]:
    paper = dict(((run_config.get("evaluation") or {}).get("paper_precision") or {}))
    rand_prob = paper.get("rand_prob")
    if rand_prob is None:
        rand_prob = run_config.get("rand_prob", 0.3)
    base_seed = paper.get("base_seed")
    if base_seed is None:
        base_seed = run_config.get("seed", 0)
    return generate_random_start_walks(
        level=int(run_config.get("level", 0)),
        sensor=str(run_config.get("sensor", "cartesian")),
        sensor_range=run_config.get("sensor_range"),
        rand_prob=float(rand_prob),
        num_walks=int(paper.get("num_walks", 100)),
        walk_length=int(paper.get("walk_length", 10_000)),
        base_seed=int(base_seed),
        valid_trajectories_only=bool(run_config.get("valid_trajectories_only", False)),
    )


def _layout_from_size(n_nodes: int) -> dict[int, np.ndarray]:
    if int(n_nodes) <= 0:
        return {}
    angles = np.linspace(0.0, 2.0 * np.pi, num=int(n_nodes), endpoint=False)
    return {
        int(idx): np.asarray([float(np.cos(angle)), float(np.sin(angle))], dtype=np.float64)
        for idx, angle in enumerate(angles.tolist())
    }


def _save_residual_figure(
    *,
    diagnostics: list[TransitionResidualDiagnostics],
    positions: dict[int, np.ndarray],
    output_path: Path,
    vis_threshold: float,
    show_edge_labels: bool,
) -> Path:
    matplotlib.use("Agg", force=True)
    columns = max(1, len(diagnostics))
    fig_width = 8.0 if columns == 1 else (6.2 * columns + 2.0)
    fig_height = 6.4

    fig, axes = plt.subplots(
        1,
        columns,
        figsize=(fig_width, fig_height),
        dpi=220,
    )
    fig.subplots_adjust(left=0.025, right=0.80, bottom=0.16, top=0.91, wspace=0.035)
    axes_list = [axes] if columns == 1 else list(np.ravel(axes))
    max_residual = max(
        [float(np.max(np.abs(item.residual))) for item in diagnostics if item.residual.size] or [1.0]
    )
    max_residual = max(max_residual, float(vis_threshold), 1e-9)
    node_collection = None
    for ax, diagnostic in zip(axes_list, diagnostics, strict=False):
        node_collection = _draw_residual_map(
            ax=ax,
            diagnostic=diagnostic,
            positions=positions,
            vis_threshold=float(vis_threshold),
            max_residual=max_residual,
            show_edge_labels=bool(show_edge_labels),
        )
    for ax in axes_list[len(diagnostics) :]:
        ax.set_axis_off()
    residual_cax = fig.add_axes([0.835, 0.16, 0.022, 0.75])
    node_cax = fig.add_axes([0.915, 0.16, 0.022, 0.75])
    if node_collection is not None:
        node_colorbar = fig.colorbar(node_collection, cax=node_cax, label=r"$S_i = 1 - TV(i)$")
        node_colorbar.ax.tick_params(labelsize=8, pad=2)
    residual_cmap = _residual_colormap()
    residual_norm = TwoSlopeNorm(vmin=-max_residual, vcenter=0.0, vmax=max_residual)
    residual_mappable = ScalarMappable(norm=residual_norm, cmap=residual_cmap)
    residual_mappable.set_array([])
    residual_colorbar = fig.colorbar(
        residual_mappable,
        cax=residual_cax,
        label=r"$R_{ij}=\hat{P}(j|i)-P^{GT}(j|i)$",
    )
    residual_colorbar.ax.yaxis.set_ticks_position("left")
    residual_colorbar.ax.yaxis.set_label_position("left")
    residual_colorbar.ax.tick_params(labelsize=8, pad=2)
    width_values = _linewidth_reference_values(max_residual)
    fig.legend(
        handles=(
            [
                Line2D(
                    [0],
                    [0],
                    color="#4b5563",
                    lw=_residual_width(value, max_residual),
                    label=fr"$|R_{{ij}}|={value:.2g}$",
                )
                for value in width_values
            ]
        ),
        loc="lower center",
        bbox_to_anchor=(0.42, 0.025),
        ncol=min(3, len(width_values)),
        fontsize=8,
        frameon=True,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def _draw_residual_map(
    *,
    ax: plt.Axes,
    diagnostic: TransitionResidualDiagnostics,
    positions: dict[int, np.ndarray],
    vis_threshold: float,
    max_residual: float,
    show_edge_labels: bool,
    font_scale: float = 1.0,
):
    n_nodes = int(diagnostic.transition_true.shape[0])
    graph = nx.DiGraph()
    graph.add_nodes_from(range(n_nodes))
    pos = {node: np.asarray(positions.get(node, [float(node), 0.0]), dtype=np.float64) for node in range(n_nodes)}

    nx.draw_networkx_edges(
        graph,
        pos=pos,
        edgelist=[],
        ax=ax,
    )
    labels = {node: str(node) for node in range(n_nodes)}
    node_colors = diagnostic.similarity_by_state.tolist()
    linewidths = [1.8 if bool(diagnostic.uncovered_rows[node]) else 0.55 for node in range(n_nodes)]
    edgecolors = ["#111827" if bool(diagnostic.uncovered_rows[node]) else "#374151" for node in range(n_nodes)]
    gt_edges = [tuple(edge) for edge in np.argwhere(diagnostic.transition_true > 0.0).tolist()]
    if gt_edges:
        nx.draw_networkx_edges(
            graph,
            pos=pos,
            edgelist=gt_edges,
            edge_color="#d1d5db",
            width=1.05,
            alpha=1.0,
            arrows=True,
            arrowstyle="-|>",
            arrowsize=9,
            connectionstyle="arc3,rad=0.0",
            ax=ax,
        )
    nodes = nx.draw_networkx_nodes(
        graph,
        pos=pos,
        node_color=node_colors,
        cmap=plt.get_cmap("viridis"),
        vmin=0.0,
        vmax=1.0,
        node_size=410,
        linewidths=linewidths,
        edgecolors=edgecolors,
        ax=ax,
    )
    nx.draw_networkx_labels(graph, pos=pos, labels=labels, font_size=7.0 * float(font_scale), font_color="white", ax=ax)

    residual_edges: list[tuple[int, int]] = []
    residual_widths: list[float] = []
    residual_colors: list[tuple[float, float, float, float]] = []
    residual_styles: list[str] = []
    edge_labels: dict[tuple[int, int], str] = {}
    residual_norm = TwoSlopeNorm(vmin=-max_residual, vcenter=0.0, vmax=max_residual)
    residual_cmap = _residual_colormap()
    edge_magnitude = (
        np.asarray(diagnostic.residual_magnitude, dtype=np.float64)
        if diagnostic.residual_magnitude is not None
        else np.abs(diagnostic.residual)
    )
    if float(vis_threshold) <= 0.0:
        residual_edge_mask = edge_magnitude > 0.0
    else:
        residual_edge_mask = edge_magnitude > float(vis_threshold)
    for source, target in np.argwhere(residual_edge_mask).tolist():
        signed_value = float(diagnostic.residual[int(source), int(target)])
        magnitude = float(edge_magnitude[int(source), int(target)])
        residual_edges.append((int(source), int(target)))
        residual_widths.append(_residual_width(magnitude, max_residual))
        residual_colors.append(residual_cmap(residual_norm(signed_value)))
        residual_styles.append("-")
        if show_edge_labels:
            edge_labels[(int(source), int(target))] = f"{signed_value:+.2f}"

    for edge, color, width, style in zip(residual_edges, residual_colors, residual_widths, residual_styles, strict=False):
        nx.draw_networkx_edges(
            graph,
            pos=pos,
            edgelist=[edge],
            edge_color=[color],
            width=[width],
            style=style,
            alpha=1.0,
            arrows=True,
            arrowstyle="-|>",
            arrowsize=15,
            connectionstyle="arc3,rad=0.22",
            ax=ax,
        )
    if show_edge_labels and edge_labels:
        nx.draw_networkx_edge_labels(graph, pos=pos, edge_labels=edge_labels, font_size=7.0 * float(font_scale), ax=ax)

    ax.set_title(f"{diagnostic.method}\nTV_sim={diagnostic.tv_sim:.3f}", fontsize=11.5 * float(font_scale))
    ax.set_axis_off()
    ax.set_aspect("equal", adjustable="box")
    _pad_axis_limits(ax, pos)
    return nodes


def _residual_width(value: float, max_residual: float) -> float:
    return 0.35 + 4.95 * float(value) / max(float(max_residual), 1e-9)


def _residual_colormap():
    return plt.get_cmap("RdBu_r")


def _linewidth_reference_values(max_residual: float) -> list[float]:
    candidates = [0.1, 0.25, 0.5]
    values = [value for value in candidates if value <= float(max_residual) * 1.05]
    if values:
        return values
    return [float(max_residual)]


def padded_axis_limits(pos: dict[int, np.ndarray]) -> tuple[tuple[float, float], tuple[float, float]]:
    """Return the padded (xlim, ylim) window used to display a node layout."""
    coords = np.stack([np.asarray(value, dtype=np.float64) for value in pos.values()], axis=0)
    x_min, y_min = coords.min(axis=0)
    x_max, y_max = coords.max(axis=0)
    x_span = max(float(x_max - x_min), 1.0)
    y_span = max(float(y_max - y_min), 1.0)
    visible_y_span = max(y_span, 0.38 * x_span)
    y_center = 0.5 * float(y_min + y_max)
    return (
        (float(x_min - 0.20 * x_span), float(x_max + 0.20 * x_span)),
        (float(y_center - 0.76 * visible_y_span), float(y_center + 0.76 * visible_y_span)),
    )


def _pad_axis_limits(ax: plt.Axes, pos: dict[int, np.ndarray]) -> None:
    if not pos:
        return
    xlim, ylim = padded_axis_limits(pos)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)


def _write_diagnostics_json(
    *,
    diagnostics: list[TransitionResidualDiagnostics],
    output_path: Path,
    vis_threshold: float,
    settings: dict[str, Any],
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "transition_residual_diagnostics_v1",
        "vis_threshold": float(vis_threshold),
        "visual_encoding": {
            "node_fill": "S_i = 1 - TV(i)",
            "edge_colour": "signed residual R_ij = P_hat(j|i) - P_GT(j|i), shared empirical diverging scale",
            "edge_width": "|R_ij| scaled on the same shared empirical residual range",
            "ground_truth_edges": "thin light-grey directed support edges are drawn before residual edges",
        },
        "settings": settings,
        "methods": {item.method: item.to_json_dict() for item in diagnostics},
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path
