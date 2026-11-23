from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
for import_root in (PROJECT_ROOT, SRC_ROOT):
    import_root_string = str(import_root)
    if import_root_string not in sys.path:
        sys.path.insert(0, import_root_string)

from mercury_runs.infrastructure.paper_precision import (
    action_ids_from_action_vectors,
    aggregate_aligned_action_transition_tensor,
    aggregate_aligned_weighted_adjacency,
    build_action_conditioned_tensor_from_walks,
    build_weighted_adjacency_from_walks,
    collapse_action_transition_tensor_to_adjacency,
    compute_action_conditioned_edge_metrics,
    compute_action_conditioned_mean_tv,
    compute_alignment,
    compute_edge_metrics,
    compute_mean_tv,
    exact_cartesian_reference_positions,
    exact_cartesian_state_ids_for_level,
    exact_valid_sensorimotor_transitions,
    generate_random_start_walks,
    infer_num_actions_from_valid_transitions,
    normalize_action_rows,
    row_normalize,
)


DEFAULT_STUDY = PROJECT_ROOT / (
    "results/current/studies/"
    "mercury_cscg_hybrid_vs_cscg_batch_size_30_seed_level_23_self_loops_5000_steps_5000_eval"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / (
    "results/current/notebook_plots/"
    "mercury_cscg_hybrid_vs_cscg_batch_size_30_seed_level_23_with_self_loops_5000_steps_5000_eval"
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_path(raw_path: str | None) -> Path | None:
    if not raw_path:
        return None
    path = Path(raw_path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _latest_metric_row(payload: dict[str, Any], *, phase: str | None = None) -> dict[str, Any]:
    history = payload.get("history")
    if isinstance(history, list) and history:
        for row in reversed(history):
            if not isinstance(row, dict):
                continue
            if phase is not None and row.get("phase") != phase:
                continue
            return row
    metrics = payload.get("metrics")
    if isinstance(metrics, dict):
        return metrics
    return payload


def _paper_precision_payload_for_run(run_status: dict[str, Any]) -> tuple[Path, dict[str, Any], str | None]:
    method = str(run_status["config"]["method"])
    artifacts = dict(run_status.get("artifacts") or {})
    if method == "mercury_cscg_hybrid":
        paper_path = _resolve_path(artifacts.get("mercury_cscg_hybrid_paper_precision_json"))
        phase = "cscg"
    elif method == "cscg":
        paper_path = _resolve_path(artifacts.get("cscg_paper_precision_json"))
        phase = None
    else:
        raise ValueError(f"Unsupported method: {method}")
    if paper_path is None or not paper_path.exists():
        raise FileNotFoundError(f"Missing paper precision JSON for {run_status.get('run_id')}: {paper_path}")
    return paper_path, _read_json(paper_path), phase


def _model_tensor_for_run(run_status: dict[str, Any]) -> np.ndarray:
    method = str(run_status["config"]["method"])
    artifacts = dict(run_status.get("artifacts") or {})
    artifact_key = "mercury_cscg_hybrid_model_npz" if method == "mercury_cscg_hybrid" else "cscg_model_npz"
    model_path = _resolve_path(artifacts.get(artifact_key))
    if model_path is None or not model_path.exists():
        raise FileNotFoundError(f"Missing model artifact {artifact_key} for {run_status.get('run_id')}: {model_path}")
    model = np.load(model_path, allow_pickle=False)
    return np.asarray(model["T"], dtype=np.float64)


def _true_eval_walks(*, config: dict[str, Any], protocol: dict[str, Any]) -> tuple[list[np.ndarray], list[np.ndarray]]:
    level = int(config["level"])
    walks = generate_random_start_walks(
        level=level,
        sensor=str(config["sensor"]),
        sensor_range=config.get("sensor_range"),
        rand_prob=float(protocol.get("rand_prob", config.get("rand_prob", 0.0))),
        num_walks=int(protocol["num_walks"]),
        walk_length=int(protocol["walk_length"]),
        base_seed=int(protocol.get("base_seed", config.get("seed", 0))),
        valid_trajectories_only=bool(protocol.get("valid_trajectories_only", False)),
    )
    state_walks = [
        exact_cartesian_state_ids_for_level(level_index=level, positions=np.asarray(walk.cartesian_observations))
        .astype(np.int64, copy=False)
        for walk in walks
    ]
    action_walks = [
        action_ids_from_action_vectors(np.asarray(walk.actions)).astype(np.int64, copy=False)
        for walk in walks
    ]
    return state_walks, action_walks


def _pad_contingency_rows(contingency: np.ndarray, n_rows: int) -> np.ndarray:
    matrix = np.asarray(contingency, dtype=np.int64)
    if matrix.shape[0] == int(n_rows):
        return matrix
    if matrix.shape[0] > int(n_rows):
        return matrix[: int(n_rows)]
    padded = np.zeros((int(n_rows), matrix.shape[1]), dtype=np.int64)
    padded[: matrix.shape[0], : matrix.shape[1]] = matrix
    return padded


def _recalculate_metrics(
    *,
    run_status: dict[str, Any],
    paper_payload: dict[str, Any],
    metric_row: dict[str, Any],
    phase: str | None,
) -> dict[str, float]:
    config = dict(run_status["config"])
    protocol = dict(paper_payload["protocol"])
    level = int(config["level"])
    n_true = int(exact_cartesian_reference_positions(level).shape[0])
    structure = dict((((config.get("evaluation") or {}).get("paper_precision") or {}).get("structure_metrics") or {}))
    epsilon = float(structure.get("epsilon", 1e-6))

    T_hat = _model_tensor_for_run(run_status)
    W_hat = collapse_action_transition_tensor_to_adjacency(T_hat)
    contingency = _pad_contingency_rows(
        np.asarray(metric_row["latent_exact_state_cooccurrence_matrix"], dtype=np.int64),
        int(W_hat.shape[0]),
    )
    alignment = compute_alignment(contingency)
    occupancy = contingency.sum(axis=1).astype(np.float64, copy=False)

    true_state_walks, true_action_walks = _true_eval_walks(config=config, protocol=protocol)
    W_true = build_weighted_adjacency_from_walks(true_state_walks, n_states=n_true)
    W_hat_projected = aggregate_aligned_weighted_adjacency(
        W_hat=row_normalize(W_hat),
        A=alignment,
        mu=occupancy,
        n_true=n_true,
    )

    valid_transitions = exact_valid_sensorimotor_transitions(level_index=level)
    n_actions = max(
        infer_num_actions_from_valid_transitions(valid_transitions),
        int(T_hat.shape[0]),
        int(max((actions.max(initial=0) for actions in true_action_walks), default=0)) + 1,
    )
    T_true = build_action_conditioned_tensor_from_walks(
        state_walks=true_state_walks,
        action_walks=true_action_walks,
        n_states=n_true,
        n_actions=n_actions,
    )
    if T_hat.shape[0] < n_actions:
        padded = np.zeros((n_actions, T_hat.shape[1], T_hat.shape[2]), dtype=np.float64)
        padded[: T_hat.shape[0]] = T_hat
        T_hat = padded
    elif T_hat.shape[0] > n_actions:
        T_hat = T_hat[:n_actions]
    T_hat_projected = aggregate_aligned_action_transition_tensor(
        normalize_action_rows(T_hat),
        alignment,
        occupancy,
        n_true=n_true,
    )

    result: dict[str, float] = {
        "recalculated_latent_mean_total_variation": compute_mean_tv(
            row_normalize(W_true),
            row_normalize(W_hat_projected),
        ),
        "recalculated_latent_action_conditioned_mean_total_variation": compute_action_conditioned_mean_tv(
            T_true,
            T_hat_projected,
            epsilon=epsilon,
        ),
    }
    result.update(
        {
            f"recalculated_latent_{key}": value
            for key, value in compute_edge_metrics(W_true, W_hat_projected, epsilon=epsilon).items()
        }
    )
    result.update(
        {
            f"recalculated_latent_action_conditioned_{key}": value
            for key, value in compute_action_conditioned_edge_metrics(T_true, T_hat_projected, epsilon=epsilon).items()
        }
    )
    if phase is not None:
        result["paper_precision_phase"] = phase  # type: ignore[assignment]
    return result


def _row_for_run(run_root: Path) -> dict[str, Any]:
    run_status = _read_json(run_root / "run_status.json")
    if run_status.get("status") != "completed":
        raise RuntimeError(f"Run is not completed: {run_root}")

    paper_path, paper_payload, phase = _paper_precision_payload_for_run(run_status)
    metric_row = _latest_metric_row(paper_payload, phase=phase)
    recalculated = _recalculate_metrics(
        run_status=run_status,
        paper_payload=paper_payload,
        metric_row=metric_row,
        phase=phase,
    )

    config = dict(run_status["config"])
    row: dict[str, Any] = {
        "run_id": run_status.get("run_id", run_root.name),
        "method": config.get("method"),
        "level": config.get("level"),
        "seed": config.get("seed"),
        "rand_prob": config.get("rand_prob"),
        "num_steps": config.get("num_steps"),
        "paper_precision_json": str(paper_path.relative_to(PROJECT_ROOT)),
        "paper_precision_phase": phase or "",
    }
    for key in (
        "latent_mean_total_variation",
        "latent_action_conditioned_mean_total_variation",
        "latent_edge_precision",
        "latent_edge_recall",
        "latent_edge_f1",
        "latent_action_conditioned_edge_precision",
        "latent_action_conditioned_edge_recall",
        "latent_action_conditioned_edge_f1",
    ):
        if key in metric_row:
            row[f"existing_{key}"] = metric_row[key]
    row.update(recalculated)
    for key, value in list(row.items()):
        if not key.startswith("recalculated_"):
            continue
        existing_key = "existing_" + key[len("recalculated_") :]
        if existing_key in row:
            row["delta_" + key[len("recalculated_") :]] = float(value) - float(row[existing_key])
    return row


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    metric_keys = (
        "recalculated_latent_mean_total_variation",
        "recalculated_latent_action_conditioned_mean_total_variation",
        "delta_latent_mean_total_variation",
        "delta_latent_action_conditioned_mean_total_variation",
    )
    groups = sorted({str(row["method"]) for row in rows})
    payload: dict[str, Any] = {"run_count": len(rows), "by_method": {}}
    for method in groups:
        group_rows = [row for row in rows if str(row["method"]) == method]
        method_payload: dict[str, Any] = {"run_count": len(group_rows)}
        for key in metric_keys:
            values = np.asarray([float(row[key]) for row in group_rows if key in row], dtype=np.float64)
            if values.size == 0:
                continue
            method_payload[key] = {
                "mean": float(values.mean()),
                "std": float(values.std(ddof=0)),
                "min": float(values.min()),
                "max": float(values.max()),
            }
        payload["by_method"][method] = method_payload
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Recalculate level-23 self-loop TV metrics.")
    parser.add_argument("--study", type=Path, default=DEFAULT_STUDY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    study = args.study.resolve()
    output_dir = args.output_dir.resolve()
    run_roots = sorted(path for path in study.iterdir() if (path / "run_status.json").exists())
    if not run_roots:
        raise SystemExit(f"No runs found under {study}")

    rows = [_row_for_run(run_root) for run_root in run_roots]
    if len(rows) != 20:
        raise RuntimeError(f"Expected 20 runs, found {len(rows)}")

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "recalculated_tv_metrics_level_23_self_loops.csv"
    summary_path = output_dir / "recalculated_tv_metrics_level_23_self_loops_summary.json"
    fieldnames = sorted({key for row in rows for key in row})
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = _summary(rows)
    summary.update(
        {
            "study": str(study.relative_to(PROJECT_ROOT)),
            "csv": str(csv_path.relative_to(PROJECT_ROOT)),
            "assumptions": {
                "tv_metric": "latent_mean_total_variation",
                "action_conditioned_tv_metric": "latent_action_conditioned_mean_total_variation",
                "self_loops": "retained; study config has structure_metrics.ignore_self_loops=false",
                "ground_truth_source": "empirical transition frequencies from each run's paper_precision protocol",
                "output_mode": "separate artifact; original run JSONs are not modified",
            },
        }
    )
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {len(rows)} rows to {csv_path}")
    print(f"Wrote summary to {summary_path}")
    for method, method_summary in summary["by_method"].items():
        tv = method_summary["recalculated_latent_mean_total_variation"]["mean"]
        atv = method_summary["recalculated_latent_action_conditioned_mean_total_variation"]["mean"]
        print(f"{method}: TV={tv:.6f}, action-conditioned TV={atv:.6f}")


if __name__ == "__main__":
    main()
