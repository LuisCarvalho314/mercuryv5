from __future__ import annotations

from pathlib import Path

import pytest

from mercury_runs.analysis.plots import age_ambiguity_ablation as plots_module
from mercury_runs.analysis.plots.age_ambiguity_ablation import (
    DEFAULT_THRESHOLDS,
    PARAMETER_ORDER,
    SERIES_ORDER,
    _frame,
    _series_key,
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
            "ambiguity_threshold": threshold,
            "ambiguity_decay": decay,
            "series": _series_key(threshold, decay),
            "record": _record(0.02 * index),
        }
        for index, (threshold, decay) in enumerate(PARAMETER_ORDER)
    ]


def test_parameter_order_contains_every_threshold_decay_pair() -> None:
    assert PARAMETER_ORDER == (
        (10, 0.99),
        (10, 1.0),
        (20, 0.99),
        (20, 1.0),
        (30, 0.99),
        (30, 1.0),
        (40, 0.99),
        (40, 1.0),
    )
    assert len(SERIES_ORDER) == len(set(SERIES_ORDER)) == 8


def test_threshold_sweep_matches_reference_study() -> None:
    assert DEFAULT_THRESHOLDS == (
        0.0,
        1e-6,
        1e-5,
        1e-4,
        1e-3,
        2e-3,
        5e-3,
        1e-2,
        2e-2,
        5e-2,
        0.1,
        0.15,
        0.2,
        0.25,
        0.3,
        0.4,
        0.5,
    )


def test_ablation_frame_uses_all_series_and_canonical_metrics(tmp_path: Path) -> None:
    frame = _frame(_rows(tmp_path))

    assert frame["plot_series"].to_list() == list(SERIES_ORDER)
    assert frame["ambiguity_threshold"].to_list() == [10, 10, 20, 20, 30, 30, 40, 40]
    assert frame["ambiguity_decay"].to_list() == [0.99, 1.0] * 4
    assert frame["latent_precision"].to_list() == [0.8] * 8
    assert frame["latent_mean_total_variation_similarity"].to_list() == pytest.approx(
        [1.0 - 0.02 * index for index in range(8)]
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

    outputs = plots_module._summary_plots(_frame(_rows(tmp_path)), tmp_path / "plots")

    assert len(outputs) == 7
    assert (tmp_path / "plots" / "age_ambiguity_parameter_response_faceted.png") in outputs
    assert (tmp_path / "plots" / "age_ambiguity_parameter_response_combined.png") in outputs
    assert "age_ambiguity_metric_profile_focused_transposed.png" in output_names


@pytest.mark.parametrize("action_conditioned", [False, True])
@pytest.mark.parametrize("include_self_loops", [False, True])
def test_threshold_rows_cover_all_series_scopes_and_loop_policies(
    tmp_path: Path,
    action_conditioned: bool,
    include_self_loops: bool,
) -> None:
    thresholds = (0.0, 1e-6, 0.5)
    rows = _threshold_rows(
        _rows(tmp_path),
        thresholds=thresholds,
        action_conditioned=action_conditioned,
        include_self_loops=include_self_loops,
    )

    assert len(rows) == 8 * len(thresholds)
    assert {row["threshold"] for row in rows} == set(thresholds)
    assert {row["run_kind"] for row in rows} == set(SERIES_ORDER)
    expected_scope = "action_conditioned" if action_conditioned else "collapsed"
    assert all(row["metric_scope"] == expected_scope for row in rows)
    assert all("edge_precision" in row and "edge_recall" in row and "edge_f1" in row for row in rows)
