from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = PROJECT_ROOT / "scripts" / "analysis"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import plot_age_ambiguity_structural_trajectories as analysis


def _trajectory_rows() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for seed in (0, 1):
        common = {
            "series": "threshold_40_decay_0p99",
            "series_label": "threshold=40, decay=0.99",
            "ambiguity_threshold": 40,
            "ambiguity_decay": 0.99,
            "level": 22,
            "level_label": "3x4",
            "run_id": f"run-{seed}",
            "seed": seed,
            "n_true_states": 4,
        }
        for step in range(1, 101):
            clones = float((step >= 25) + (step >= 75)) if seed == 0 else 0.0
            deletions = float(step >= 80) if seed == 0 else 0.0
            node_ratio = (4.0 + clones - deletions) / 4.0
            rows.extend(
                [
                    {**common, "metric": "clone_created_cumulative", "step": step, "value": clones},
                    {**common, "metric": "node_prune_cumulative", "step": step, "value": deletions},
                    {**common, "metric": "node_ratio", "step": step, "value": node_ratio},
                ]
            )
    return pd.DataFrame(rows)


def test_seed_phase_computes_local_flux_before_aggregation_and_keeps_quiescent_seed() -> None:
    phase = analysis.construct_seed_phase_trajectories(
        _trajectory_rows(), window=50, sample_interval=50, final_step=100
    )
    active = phase[phase["seed"] == 0].set_index("time")
    quiescent = phase[phase["seed"] == 1].set_index("time")

    assert active.loc[0, "node_ratio"] == 1.0
    assert active.loc[50, "node_ratio"] == 1.25
    assert active.loc[50, "birth_flux"] == 1 / 50 / 4
    assert active.loc[50, "death_flux"] == 0.0
    assert active.loc[100, "birth_flux"] == 1 / 50 / 4
    assert active.loc[100, "death_flux"] == 1 / 50 / 4
    assert active.loc[100, "net_flux"] == 0.0
    assert active.loc[100, "gross_flux"] == 2 / 50 / 4
    assert active.loc[100, "churn_flux"] == 2 / 50 / 4
    assert active.loc[50, "gross_flux"] == 1 / 50 / 4
    assert active.loc[50, "churn_flux"] == 1 / 50 / 4
    assert active.loc[50, "cumulative_churn_per_gt"] == 1 / 4
    assert active.loc[100, "cumulative_churn_per_gt"] == 3 / 4
    assert np.allclose(phase["churn_flux"], phase["gross_flux"])
    assert np.allclose(
        phase["churn_flux"], phase["birth_flux"] + phase["death_flux"]
    )
    assert not quiescent["structurally_active"].any()
    assert np.allclose(quiescent["net_flux"], 0.0)
    assert np.allclose(quiescent["node_ratio"], 1.0)


def test_phase_medoids_are_observed_and_split_clone_active_from_quiescent_runs() -> None:
    phase = analysis.construct_seed_phase_trajectories(
        _trajectory_rows(), window=50, sample_interval=50, final_step=100
    )
    medoids = analysis.select_phase_medoids(phase)
    endpoints = medoids[medoids["time"] == 100].set_index("mode")

    assert set(endpoints.index) == {"active", "inactive"}
    assert endpoints.loc["active", "run_id"] == "run-0"
    assert endpoints.loc["inactive", "run_id"] == "run-1"
    assert endpoints.loc["active", "mode_n"] == 1
    assert endpoints.loc["inactive", "mode_n"] == 1
    assert endpoints.loc["active", "n_clone_positive"] == 1
    assert endpoints.loc["active", "n_quiescent"] == 1
    assert endpoints.loc["active", "node_ratio"] == 1.25
    assert endpoints.loc["inactive", "node_ratio"] == 1.0
    assert endpoints.loc["inactive", "net_flux"] == 0.0

    joint = analysis.select_phase_medoids(
        phase, feature_columns=("node_ratio", "cumulative_churn_per_gt")
    )
    assert set(joint["medoid_feature_space"]) == {
        "node_ratio+cumulative_churn_per_gt"
    }
    assert joint.groupby("mode")["medoid_run_id"].nunique().eq(1).all()


def test_flux_phase_summary_uses_seedwise_coordinates_and_preserves_inactive_runs() -> None:
    phase = analysis.construct_seed_phase_trajectories(
        _trajectory_rows(), window=50, sample_interval=50, final_step=100
    )
    summary = analysis.summarise_structural_flux_phase(phase)
    all_seeds = summary[summary["scope"] == "all_seeds"].set_index("time")
    active = summary[summary["scope"] == "active_conditional"].set_index("time")

    assert all_seeds.loc[50, "n_seeds"] == 2
    assert all_seeds.loc[50, "net_flux_50"] == 0.125
    assert all_seeds.loc[50, "gross_turnover_50"] == 0.125
    assert all_seeds.loc[100, "net_flux_50"] == 0.0
    assert all_seeds.loc[100, "gross_turnover_50"] == 0.25
    assert active.loc[50, "n_seeds"] == 1
    assert active.loc[50, "net_flux_50"] == 0.25
    assert active.loc[100, "gross_turnover_50"] == 0.5
    assert np.all(summary["gross_turnover_50"] >= np.abs(summary["net_flux_50"]))


def test_window_sensitivity_compares_aligned_seed_first_phase_paths() -> None:
    rows = _trajectory_rows()
    reference = analysis.select_phase_medoids(
        analysis.construct_seed_phase_trajectories(rows, window=50, sample_interval=50, final_step=100)
    )
    alternate = analysis.select_phase_medoids(
        analysis.construct_seed_phase_trajectories(rows, window=25, sample_interval=50, final_step=100)
    )
    sensitivity = analysis.window_sensitivity(reference, [alternate])

    assert set(sensitivity["mode"]) == {"active", "inactive"}
    assert sensitivity["reference_window"].eq(50).all()
    assert sensitivity["alternate_window"].eq(25).all()
    assert sensitivity["max_abs_inventory_difference"].eq(0.0).all()
    assert sensitivity["net_flux_sign_agreement"].between(0.0, 1.0).all()
    assert np.isfinite(sensitivity["net_flux_spearman"]).all()


def test_phase_portrait_writes_vector_and_raster_outputs(tmp_path: Path) -> None:
    summary_rows: list[dict[str, object]] = []
    seed_rows: list[dict[str, object]] = []
    times = (0, 500, 2000)
    for level in (13, 21, 20, 22):
        for threshold in (10, 20, 30, 40):
            for decay in (0.99, 1.0):
                modes = ("active", "inactive") if (level, threshold, decay) == (22, 40, 0.99) else ("all",)
                for mode in modes:
                    for time in times:
                        inactive = mode == "inactive"
                        common = {
                            "level": level,
                            "level_label": analysis.LEVEL_LABELS[level],
                            "ambiguity_threshold": threshold,
                            "ambiguity_decay": decay,
                            "time": time,
                        }
                        summary_rows.append(
                            {
                                **common,
                                "mode": mode,
                                "node_ratio": 1.0 if inactive else 1.0 + threshold / 100.0 * time / 2000.0,
                                "net_flux": 0.0 if inactive or time != 500 else 0.002,
                                "churn_flux": 0.0 if inactive or time != 500 else 0.003,
                                "n_seeds": 30,
                                "n_clone_positive": 14 if (level, threshold, decay) == (22, 40, 0.99) else 30,
                                "n_quiescent": 16 if (level, threshold, decay) == (22, 40, 0.99) else 0,
                                "mode_n": 16 if inactive else (14 if mode == "active" else 30),
                                "activation_time_q25": np.nan if inactive else 400.0,
                                "activation_time_q75": np.nan if inactive else 700.0,
                            }
                        )

    seed_rows.append({"placeholder": True})

    outputs = analysis.plot_structural_phase_portrait(
        pd.DataFrame(seed_rows), pd.DataFrame(summary_rows), tmp_path
    )

    assert [path.suffix for path in outputs] == [".pdf", ".png"]
    assert all(path.exists() and path.stat().st_size > 0 for path in outputs)
