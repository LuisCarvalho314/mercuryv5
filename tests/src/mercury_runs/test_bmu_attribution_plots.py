from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from mercury_runs.analysis import generate_bmu_attribution_plots
from mercury_runs.infrastructure.storage import artifact_layout


def _write_attribution_parquet(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "step": [1, 2, 3, 4],
            "latent_bmu": [0, 0, 1, 1],
            "latent_node_count": [2, 2, 3, 3],
            "admissible_node_count": [2, 2, 3, 3],
            "selected_memory_drive_raw": [1.0, 0.9, 0.8, 0.7],
            "selected_undirected_drive_raw": [0.0, 0.1, 0.2, 0.3],
            "selected_baseline_drive_raw": [0.1, 0.1, 0.1, 0.1],
            "selected_action_drive_raw": [0.2, 0.2, 0.2, 0.2],
            "selected_memory_drive_weighted": [0.5, 0.45, 0.4, 0.35],
            "selected_undirected_drive_weighted": [0.0, 0.05, 0.1, 0.15],
            "selected_baseline_drive_weighted": [0.05, 0.05, 0.05, 0.05],
            "selected_action_drive_weighted": [0.1, 0.1, 0.1, 0.1],
            "selected_memory_drive_share": [0.7692308, 0.6923077, 0.6153846, 0.5384615],
            "selected_undirected_drive_share": [0.0, 0.0769231, 0.1538462, 0.2307692],
            "selected_baseline_drive_share": [0.0769231, 0.0769231, 0.0769231, 0.0769231],
            "selected_action_drive_share": [0.1538462, 0.1538462, 0.1538462, 0.1538462],
            "selected_trace_penalty": [0.0, 0.0, 0.0, 0.0],
            "selected_total_support_pre_trace": [0.65, 0.65, 0.65, 0.65],
            "selected_total_support_post_trace": [0.65, 0.65, 0.65, 0.65],
        }
    ).write_parquet(path)


def test_generate_bmu_attribution_plots_writes_step_and_relation_plots(tmp_path: Path) -> None:
    run_payload = {
        "level": 13,
        "sensor": "cardinal distance",
        "sensor_range": 1,
        "baselines": {"pocml": {"enabled": False}, "cscg": {"enabled": False}},
    }
    layout = artifact_layout(run_root=tmp_path, run_id="run-1", run_payload=run_payload)
    _write_attribution_parquet(layout.mercury_attribution_parquet)
    layout.mercury_paper_precision_json.parent.mkdir(parents=True, exist_ok=True)
    layout.mercury_paper_precision_json.write_text(
        json.dumps(
            {
                "history": [
                    {"step": 2, "latent_edge_f1": 0.3, "latent_inferred_state_count": 4, "latent_ground_truth_state_count": 2},
                    {"step": 4, "latent_edge_f1": 0.6, "latent_inferred_state_count": 6, "latent_ground_truth_state_count": 3},
                ]
            }
        ),
        encoding="utf-8",
    )

    generated_paths = generate_bmu_attribution_plots(layout=layout)

    assert layout.mercury_bmu_attribution_step_png in generated_paths
    assert layout.mercury_bmu_attribution_edge_f1_png in generated_paths
    assert layout.mercury_bmu_attribution_node_ratio_png in generated_paths
    assert layout.mercury_bmu_attribution_step_png.exists()
    assert layout.mercury_bmu_attribution_edge_f1_png.exists()
    assert layout.mercury_bmu_attribution_node_ratio_png.exists()
