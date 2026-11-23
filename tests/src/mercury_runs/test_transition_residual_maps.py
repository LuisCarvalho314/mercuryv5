from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np
import pytest

from mercury.graph.core import Graph
from mercury_runs.analysis.evaluation_records import (
    diagnostics_from_record,
    validate_conditional_nonself_record,
)
from mercury_runs.analysis.plots.selected_residual_maps import ALL_CSCG_PRESETS
from mercury_runs.analysis.residual_map_rendering import (
    _compact_method_label,
    _display_method,
    positions_for_level,
    save_selected_figure_compact,
)
from mercury_runs.analysis.transition_residual_maps import (
    compute_transition_residual_diagnostics,
    generate_transition_residual_maps,
)

matplotlib.use("Agg", force=True)


def test_residual_maps_use_standard_cscg_for_new_and_saved_payloads() -> None:
    assert ALL_CSCG_PRESETS[:4] == (
        ("Standard CSCG", "vanilla_cscg"),
        ("Oracle capacity", "oracle_capacity"),
        ("Oracle capacity + first-order structure", "oracle_capacity_first_order"),
        ("Oracle capacity + structure", "oracle_capacity_structural"),
    )
    assert _display_method("CSCG - vanilla") == "Standard CSCG"
    assert _display_method("Vanilla CSCG") == "Standard CSCG"
    assert _display_method("MERCURY-CSCG Hybrid") == "MERCURY initialised"
    assert _compact_method_label("MERCURY initialised") == "MERCURY"
    assert _compact_method_label("Standard CSCG") == "CSCG"
    assert _compact_method_label("Oracle capacity + structure") == "Oracle-C+Struct"


def test_compact_residual_renderer_is_additive_and_writes_png_and_pdf(tmp_path: Path) -> None:
    methods = (
        "MERCURY initialised",
        "TMGWR initialised",
        "Standard CSCG",
        "Oracle capacity",
        "Oracle capacity + first-order structure",
        "Oracle capacity + structure",
    )
    panels: dict[str, dict[str, object]] = {}
    for level in (13, 21, 20, 22):
        n_states = len(positions_for_level(level))
        p_true = np.zeros((n_states, n_states), dtype=float)
        residual = np.zeros_like(p_true)
        for state in range(n_states):
            p_true[state, (state + 1) % n_states] = 1.0
        residual[0, 1] = -0.2
        residual[0, 2] = 0.2
        for method in methods:
            panels[f"L{level}/{method}"] = {
                "method": method,
                "tv_sim": 0.8,
                "tv_by_state": [0.2] * n_states,
                "similarity_by_state": [0.8] * n_states,
                "uncovered_rows": [False] * n_states,
                "P_gt": p_true.tolist(),
                "P_hat": (p_true + residual).tolist(),
                "R": residual.tolist(),
                "mean_abs_R": np.abs(residual).tolist(),
            }
    payload = {
        "scenario": "Matched-step action-conditioned transition residuals: final comparison",
        "panels": panels,
        "visual_encoding": {"action_conditioned": True},
    }
    original = tmp_path / "existing.png"
    original.write_bytes(b"unchanged")
    compact = tmp_path / "figure_compact.png"
    supplementary = tmp_path / "figure_compact_supplementary.png"

    save_selected_figure_compact(
        payload=payload,
        output_path=compact,
        vis_threshold=0.02,
        supplementary=False,
        top_k=2,
    )
    save_selected_figure_compact(
        payload=payload,
        output_path=supplementary,
        vis_threshold=0.02,
        supplementary=True,
        top_k=2,
    )

    assert original.read_bytes() == b"unchanged"
    assert compact.exists() and compact.with_suffix(".pdf").exists()
    assert supplementary.exists() and supplementary.with_suffix(".pdf").exists()


def test_transition_residual_decomposition_matches_tv_formula() -> None:
    diagnostic = compute_transition_residual_diagnostics(
        method="example",
        W_true=np.array([[0.0, 2.0], [1.0, 1.0]], dtype=np.float64),
        W_hat=np.array([[1.0, 1.0], [0.0, 3.0]], dtype=np.float64),
        cooccurrence=np.array([[5, 0], [0, 5]], dtype=np.int64),
    )

    assert diagnostic.transition_true.tolist() == [[0.0, 1.0], [0.5, 0.5]]
    assert diagnostic.transition_predicted.tolist() == [[0.5, 0.5], [0.0, 1.0]]
    assert diagnostic.tv_by_state.tolist() == pytest.approx([0.5, 0.5])
    assert diagnostic.similarity_by_state.tolist() == pytest.approx([0.5, 0.5])
    assert diagnostic.tv_sim == pytest.approx(0.5)
    assert diagnostic.excess_mass.sum(axis=1).tolist() == pytest.approx(diagnostic.tv_by_state.tolist())
    assert diagnostic.missing_mass.sum(axis=1).tolist() == pytest.approx(diagnostic.tv_by_state.tolist())
    assert diagnostic.excess_mass[0, 0] == pytest.approx(0.5)
    assert diagnostic.missing_mass[0, 1] == pytest.approx(0.5)


def test_transition_residual_marks_uncovered_rows_as_zero_similarity() -> None:
    diagnostic = compute_transition_residual_diagnostics(
        method="example",
        W_true=np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64),
        W_hat=np.array([[0.0, 1.0], [0.0, 0.0]], dtype=np.float64),
        cooccurrence=np.array([[4, 0], [0, 1]], dtype=np.int64),
    )

    assert diagnostic.uncovered_rows.tolist() == [False, True]
    assert diagnostic.tv_by_state.tolist() == pytest.approx([0.0, 1.0])
    assert diagnostic.similarity_by_state.tolist() == pytest.approx([1.0, 0.0])
    assert diagnostic.tv_sim == pytest.approx(0.5)


def test_record_diagnostics_use_matching_tv_similarity_fields() -> None:
    record = {
        "state_representation": {"n_true_states": 2, "n_actions": 2},
        "alignment": {
            "cooccurrence": [[5, 0], [0, 5]],
            "learned_to_true": [0, 1],
        },
        "action_conditioned": {
            "P_true_full": [
                [[0.7, 0.3], [0.2, 0.8]],
                [[0.6, 0.4], [0.1, 0.9]],
            ],
            "P_hat_full": [
                [[0.6, 0.4], [0.3, 0.7]],
                [[0.5, 0.5], [0.2, 0.8]],
            ],
            "valid_rows": [[True, True], [True, True]],
            "tv_by_pair_full": [[0.10, 0.20], [0.30, 0.40]],
            "tv_by_pair_non_self_loop": [[0.15, 0.25], [0.35, 0.45]],
            "tv_by_state_occupancy": [0.20, 0.30],
            "tv_non_self_loop_by_state_occupancy": [0.25, 0.35],
            "tv_occupancy": 0.30,
            "tv_similarity_occupancy": 0.70,
            "tv_non_self_loop_occupancy": 0.40,
            "tv_non_self_loop_similarity_occupancy": 0.60,
        },
        "policy_collapsed": {
            "P_true": [[0.7, 0.3], [0.2, 0.8]],
            "P_hat": [[0.6, 0.4], [0.3, 0.7]],
            "tv_by_state": [0.10, 0.20],
            "tv_occupancy": 0.20,
            "tv_similarity_occupancy": 0.80,
            "tv_non_self_loop_by_state": [0.30, 0.40],
            "tv_non_self_loop_occupancy": 0.50,
            "tv_non_self_loop_similarity_occupancy": 0.50,
        },
    }

    action = diagnostics_from_record(record, method="hybrid", action_conditioned=True)
    action_no_self = diagnostics_from_record(
        record,
        method="hybrid",
        action_conditioned=True,
        ignore_self_loops=True,
    )
    collapsed_no_self = diagnostics_from_record(record, method="hybrid", ignore_self_loops=True)

    assert action.tv_sim == pytest.approx(0.70)
    assert action.similarity_by_state.tolist() == pytest.approx([0.80, 0.70])
    assert action_no_self.tv_sim == pytest.approx(0.60)
    assert action_no_self.similarity_by_state.tolist() == pytest.approx([0.75, 0.65])
    assert collapsed_no_self.tv_sim == pytest.approx(0.50)
    assert collapsed_no_self.similarity_by_state.tolist() == pytest.approx([0.70, 0.60])


def test_record_action_no_self_diagnostics_condition_and_use_evaluation_weights() -> None:
    record = {
        "state_representation": {"n_true_states": 3, "n_actions": 2},
        "alignment": {
            "cooccurrence": [[10, 0, 0], [0, 10, 0], [0, 0, 10]],
            "learned_to_true": [0, 1, 2],
        },
        "action_conditioned": {
            "P_true_full": [
                [[0.8, 0.2, 0.0, 0.0], [0, 0, 0, 0], [0, 0, 0, 0]],
                [[0.8, 0.0, 0.2, 0.0], [0, 0, 0, 0], [0, 0, 0, 0]],
            ],
            "P_hat_full": [
                [[0.8, 0.1, 0.1, 0.0], [0, 0, 0, 0], [0, 0, 0, 0]],
                [[0.8, 0.1, 0.1, 0.0], [0, 0, 0, 0], [0, 0, 0, 0]],
            ],
            "valid_rows": [[True, False, False], [True, False, False]],
            "source_action_counts": [[9, 0, 0], [1, 0, 0]],
            "tv_by_pair_full": [[0.1, 0, 0], [0.1, 0, 0]],
            "tv_by_pair_non_self_loop": [[0.5, 0, 0], [0.5, 0, 0]],
            "tv_non_self_loop_by_state_occupancy": [0.5, 0, 0],
            "tv_non_self_loop_occupancy": 0.5,
            "tv_non_self_loop_similarity_occupancy": 0.5,
        },
        "policy_collapsed": {
            "P_true": [[0.8, 0.18, 0.02, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
            "P_hat": [[0.8, 0.1, 0.1, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
            "tv_by_state": [0.08, 0, 0],
            "tv_occupancy": 0.08,
            "tv_similarity_occupancy": 0.92,
            "tv_non_self_loop_by_state": [0.4, 0, 0],
            "tv_non_self_loop_occupancy": 0.4,
            "tv_non_self_loop_similarity_occupancy": 0.6,
        },
    }

    diagnostic = diagnostics_from_record(
        record,
        method="weighted",
        action_conditioned=True,
        ignore_self_loops=True,
    )

    assert diagnostic.transition_true[0].tolist() == pytest.approx([0.0, 0.9, 0.1])
    assert diagnostic.transition_predicted[0].tolist() == pytest.approx([0.0, 0.5, 0.5])
    assert diagnostic.residual[0].tolist() == pytest.approx([0.0, -0.4, 0.4])
    assert diagnostic.residual_magnitude[0].tolist() == pytest.approx([0.0, 0.5, 0.5])
    assert diagnostic.missing_mass[0].tolist() == pytest.approx([0.0, 0.45, 0.05])
    assert diagnostic.excess_mass[0].tolist() == pytest.approx([0.0, 0.05, 0.45])
    assert diagnostic.tv_sim == pytest.approx(0.5)


def test_versioned_record_validation_rejects_scalar_matrix_mismatch() -> None:
    record = {
        "protocol": {"schema_version": "canonical_v5_conditional_nonself"},
        "state_representation": {"n_true_states": 2, "n_actions": 1},
        "action_conditioned": {
            "P_true_full": [[[0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]],
            "P_hat_full": [[[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]],
            "valid_rows": [[True, False]],
            "source_action_counts": [[10, 0]],
            "tv_non_self_loop_occupancy": 0.0,
        },
        "policy_collapsed": {
            "P_true": [[0.0, 1.0, 0.0], [0.0, 0.0, 0.0]],
            "P_hat": [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            "tv_non_self_loop_occupancy": 1.0,
        },
    }

    with pytest.raises(ValueError, match="Action-conditioned no-self record inconsistency"):
        validate_conditional_nonself_record(record)


def _write_graph(path: Path, adjacency: np.ndarray) -> None:
    graph = Graph(directed=True)
    for _ in range(adjacency.shape[0]):
        graph.add_node()
    for source, target in np.argwhere(adjacency > 0.0).tolist():
        graph.add_edge(int(source), int(target), weight=float(adjacency[int(source), int(target)]))
    graph.to_npz(str(path))


def test_generate_transition_residual_maps_writes_png_and_json(tmp_path: Path, monkeypatch) -> None:
    run_root = tmp_path / "run-1"
    graph_path = run_root / "bundles" / "mercury" / "run-1_latent_graph.npz"
    paper_path = run_root / "metrics" / "mercury" / "run-1_paper_precision.json"
    graph_path.parent.mkdir(parents=True, exist_ok=True)
    paper_path.parent.mkdir(parents=True, exist_ok=True)
    _write_graph(graph_path, np.array([[0.0, 1.0], [0.0, 1.0]], dtype=np.float64))
    paper_path.write_text(
        json.dumps(
            {
                "metrics": {
                    "latent_exact_state_cooccurrence_matrix": [[5, 0], [0, 5]],
                    "latent_mean_total_variation": 0.5,
                }
            }
        ),
        encoding="utf-8",
    )
    (run_root / "run_status.json").write_text(
        json.dumps(
            {
                "config": {"level": 0, "evaluation": {"paper_precision": {"structure_metrics": {"ignore_self_loops": False}}}},
                "artifacts": {
                    "mercury_latent_graph_npz": str(graph_path),
                    "mercury_paper_precision_json": str(paper_path),
                },
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "mercury_runs.analysis.transition_residual_maps._ground_truth_structure",
        lambda _config: (
            np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64),
            {0: np.array([0.0, 0.0]), 1: np.array([1.0, 0.0])},
            {"scope": "test", "ground_truth_source": "test", "ignore_self_loops": False},
        ),
    )

    outputs = generate_transition_residual_maps(
        run_root=run_root,
        methods=["mercury_latent"],
        vis_threshold=0.8,
    )

    assert outputs["combined_plot"].exists()
    assert outputs["diagnostics_json"].exists()
    payload = json.loads(outputs["diagnostics_json"].read_text(encoding="utf-8"))
    assert payload["vis_threshold"] == pytest.approx(0.8)
    assert payload["methods"]["mercury_latent"]["tv_sim"] == pytest.approx(0.5)
    assert payload["methods"]["mercury_latent"]["E_plus"] == [[0.0, 0.0], [0.0, 1.0]]
