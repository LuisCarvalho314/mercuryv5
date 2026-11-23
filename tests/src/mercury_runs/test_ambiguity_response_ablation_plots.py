from __future__ import annotations

from pathlib import Path

import pytest

from mercury_runs.analysis.plots import ambiguity_response_ablation as plots_module
from mercury_runs.analysis.plots.ambiguity_response_ablation import (
    MODE_ORDER,
    _frame,
    _threshold_rows,
)


def _record(offset: float) -> dict:
    predicted = [[0.0, 1.0], [1.0 - offset, offset]]
    true = [[0.0, 1.0], [1.0, 0.0]]
    return {
        "state_representation": {
            "latent_precision": 0.8,
            "latent_recall": 0.9,
            "node_recall": 0.9,
            "n_true_states": 2,
            "n_actions": 2,
        },
        "policy_collapsed": {
            "P_true": true,
            "P_hat": predicted,
            "tv_occupancy": offset,
            "tv_non_self_loop_occupancy": offset,
        },
        "action_conditioned": {
            "P_true_full": [true, true],
            "P_hat_full": [predicted, predicted],
            "valid_rows": [[True, True], [True, True]],
            "tv_occupancy": offset,
            "tv_non_self_loop_occupancy": offset,
        },
    }


def _rows(tmp_path: Path) -> list[dict]:
    return [
        {
            "run_id": f"run-{index}",
            "run_root": tmp_path / f"run-{index}",
            "level": 13,
            "seed": index,
            "rand_prob": 0.3,
            "mode": mode,
            "record": _record(0.05 * index),
        }
        for index, mode in enumerate(MODE_ORDER)
    ]


def test_ablation_frame_uses_canonical_metric_columns(tmp_path: Path) -> None:
    frame = _frame(_rows(tmp_path))

    assert frame["plot_series"].to_list() == list(MODE_ORDER)
    assert frame["latent_precision"].to_list() == [0.8] * 4
    assert frame["latent_mean_total_variation_similarity"].to_list() == pytest.approx(
        [1.0, 0.95, 0.9, 0.85]
    )


def test_summary_suite_includes_transposed_focused_profile(tmp_path: Path, monkeypatch) -> None:
    output_names: list[str] = []

    def fake_plot(*args, output_name: str, plot_dir: Path, **kwargs) -> Path:
        output_names.append(output_name)
        return plot_dir / output_name

    monkeypatch.setattr(plots_module.forest, "plot_forest", fake_plot)
    monkeypatch.setattr(plots_module.forest, "plot_metric_matrix", fake_plot)
    monkeypatch.setattr(plots_module.forest, "plot_metric_profile", fake_plot)
    monkeypatch.setattr(plots_module.forest, "plot_metric_profile_transposed", fake_plot)
    monkeypatch.setattr(plots_module.forest, "plot_metric_profile_transposed_compact", fake_plot)

    outputs = plots_module._summary_plots(_frame(_rows(tmp_path)), tmp_path / "plots")

    assert len(outputs) == 6
    assert "ambiguity_response_metric_profile_focused_transposed.png" in output_names
    assert "ambiguity_response_metric_profile_focused_transposed_compact.png" in output_names


def test_threshold_rows_use_canonical_summary_schema(tmp_path: Path) -> None:
    rows = _threshold_rows(
        _rows(tmp_path),
        thresholds=(0.2,),
        action_conditioned=False,
        include_self_loops=False,
    )

    assert len(rows) == 4
    assert {row["method"] for row in rows} == {
        "Full MERCURY",
        "Refinement without reconstruction",
        "Local repair only",
        "No ambiguity response",
    }
    assert all(row["metric_scope"] == "collapsed" for row in rows)
    assert all("edge_precision" in row and "edge_recall" in row and "edge_f1" in row for row in rows)
