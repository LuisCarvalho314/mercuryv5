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

from mercury.graph.core import Graph
from mercury_runs.infrastructure.paper_precision import (
    action_ids_from_action_vectors,
    aggregate_aligned_action_transition_tensor,
    aggregate_aligned_weighted_adjacency,
    build_action_conditioned_tensor_from_graph,
    build_action_conditioned_tensor_from_walks,
    build_weighted_adjacency_from_walks,
    compute_action_conditioned_edge_metrics,
    compute_action_conditioned_mean_tv,
    compute_alignment,
    compute_edge_metrics,
    compute_mean_tv,
    exact_cartesian_reference_positions,
    exact_cartesian_state_ids_for_level,
    generate_random_start_walks,
    infer_num_actions_from_valid_transitions,
    exact_valid_sensorimotor_transitions,
    row_normalize,
)


DEFAULT_STUDY = (
    PROJECT_ROOT
    / "results/current/studies/levels_13_21_22_memory_study_fixed_eval_1"
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_artifact(run_root: Path, raw_path: str | None) -> Path | None:
    if not raw_path:
        return None
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def _pad_contingency_rows(contingency: np.ndarray, n_rows: int) -> np.ndarray:
    matrix = np.asarray(contingency, dtype=np.int64)
    if matrix.shape[0] == int(n_rows):
        return matrix
    if matrix.shape[0] > int(n_rows):
        return matrix[: int(n_rows)]
    padded = np.zeros((int(n_rows), matrix.shape[1]), dtype=np.int64)
    padded[: matrix.shape[0], : matrix.shape[1]] = matrix
    return padded


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
        valid_trajectories_only=bool(protocol.get("walk_valid_trajectories_only", False)),
    )
    state_walks = [
        exact_cartesian_state_ids_for_level(
            level_index=level,
            positions=walk.cartesian_observations,
        ).astype(np.int64, copy=False)
        for walk in walks
    ]
    action_walks = [
        action_ids_from_action_vectors(np.asarray(walk.actions)).astype(np.int64, copy=False)
        for walk in walks
    ]
    return state_walks, action_walks


def _recompute_latent_metrics(
    *,
    run_status: dict[str, Any],
    paper_payload: dict[str, Any],
    graph_path: Path,
) -> dict[str, float]:
    config = run_status["config"]
    protocol = paper_payload["protocol"]
    metrics = paper_payload["metrics"]
    level = int(config["level"])
    epsilon = float(
        config.get("evaluation", {})
        .get("paper_precision", {})
        .get("structure_metrics", {})
        .get("epsilon", 1e-6)
    )

    graph = Graph.from_npz(str(graph_path))
    n_true = int(exact_cartesian_reference_positions(level).shape[0])
    state_walks, action_walks = _true_eval_walks(config=config, protocol=protocol)

    contingency = _pad_contingency_rows(
        np.asarray(metrics["latent_exact_state_cooccurrence_matrix"], dtype=np.int64),
        graph.n,
    )
    alignment = compute_alignment(contingency)
    occupancy = contingency.sum(axis=1).astype(np.float64, copy=False)

    true_adjacency = build_weighted_adjacency_from_walks(state_walks, n_states=n_true)
    aligned_adjacency = aggregate_aligned_weighted_adjacency(
        W_hat=np.asarray(graph.adj, dtype=np.float64),
        A=alignment,
        mu=occupancy,
        n_true=n_true,
    )

    result = {
        "latent_mean_total_variation_false": compute_mean_tv(
            row_normalize(true_adjacency),
            row_normalize(aligned_adjacency),
        ),
        **{
            f"latent_{key}_false": value
            for key, value in compute_edge_metrics(
                true_adjacency,
                aligned_adjacency,
                epsilon=epsilon,
            ).items()
        },
    }

    if "action" in graph.edge_features:
        valid_transitions = exact_valid_sensorimotor_transitions(level_index=level)
        n_actions = max(
            infer_num_actions_from_valid_transitions(valid_transitions),
            int(np.max([actions.max(initial=0) for actions in action_walks]) + 1),
        )
        true_tensor = build_action_conditioned_tensor_from_walks(
            state_walks=state_walks,
            action_walks=action_walks,
            n_states=n_true,
            n_actions=n_actions,
        )
        learned_tensor = build_action_conditioned_tensor_from_graph(
            graph,
            n_actions=n_actions,
            epsilon=epsilon,
        )
        aligned_tensor = aggregate_aligned_action_transition_tensor(
            learned_tensor,
            alignment,
            occupancy,
            n_true=n_true,
        )
        result["latent_action_conditioned_mean_total_variation_false"] = (
            compute_action_conditioned_mean_tv(
                true_tensor,
                aligned_tensor,
                epsilon=epsilon,
            )
        )
        result.update(
            {
                f"latent_action_conditioned_{key}_false": value
                for key, value in compute_action_conditioned_edge_metrics(
                    true_tensor,
                    aligned_tensor,
                    epsilon=epsilon,
                ).items()
            }
        )

    return result


def _row_for_run(run_root: Path) -> dict[str, Any] | None:
    status_path = run_root / "run_status.json"
    if not status_path.exists():
        return None
    run_status = _read_json(status_path)
    if run_status.get("status") != "completed":
        return None

    artifacts = run_status.get("artifacts", {})
    paper_path = _resolve_artifact(run_root, artifacts.get("mercury_paper_precision_json"))
    graph_path = _resolve_artifact(run_root, artifacts.get("mercury_latent_graph_npz"))
    if paper_path is None or graph_path is None or not paper_path.exists() or not graph_path.exists():
        return None

    paper_payload = _read_json(paper_path)
    existing = paper_payload.get("metrics", {})
    recomputed = _recompute_latent_metrics(
        run_status=run_status,
        paper_payload=paper_payload,
        graph_path=graph_path,
    )

    config = run_status["config"]
    row: dict[str, Any] = {
        "run_id": run_status["run_id"],
        "level": config.get("level"),
        "seed": config.get("seed"),
        "rand_prob": config.get("rand_prob"),
        "weight_memory": config.get("latent", {}).get("weight_memory"),
        "memory_replay": config.get("latent", {}).get("memory_replay"),
        "memory_disambiguation": config.get("latent", {}).get("memory_disambiguation"),
    }
    for key in (
        "latent_mean_total_variation",
        "latent_edge_precision",
        "latent_edge_recall",
        "latent_edge_f1",
        "latent_action_conditioned_mean_total_variation",
        "latent_action_conditioned_edge_precision",
        "latent_action_conditioned_edge_recall",
        "latent_action_conditioned_edge_f1",
    ):
        if key in existing:
            row[f"{key}_true"] = existing[key]
    row.update(recomputed)
    for key in list(row):
        if key.endswith("_false"):
            true_key = key[: -len("_false")] + "_true"
            if true_key in row:
                row[key[: -len("_false")] + "_delta_false_minus_true"] = (
                    float(row[key]) - float(row[true_key])
                )
    return row


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare final Mercury latent structure metrics with self-loops included."
    )
    parser.add_argument("--study", type=Path, default=DEFAULT_STUDY)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    study = args.study.resolve()
    output = args.output or (study / "final_latent_self_loop_effect.csv")
    rows = [row for run_root in sorted(study.iterdir()) if run_root.is_dir() if (row := _row_for_run(run_root))]
    if not rows:
        raise SystemExit(f"No recomputable Mercury runs found under {study}")

    fieldnames = sorted({key for row in rows for key in row})
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    delta_keys = [key for key in fieldnames if key.endswith("_delta_false_minus_true")]
    print(f"Wrote {len(rows)} rows to {output}")
    for key in delta_keys:
        values = np.asarray([float(row.get(key, 0.0)) for row in rows], dtype=np.float64)
        changed = int(np.count_nonzero(np.abs(values) > 1e-12))
        print(
            f"{key}: changed {changed}/{len(rows)}, "
            f"mean_delta={values.mean():.6g}, max_abs_delta={np.max(np.abs(values)):.6g}"
        )


if __name__ == "__main__":
    main()
