from __future__ import annotations

import json
from pathlib import Path
import sys

import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mercury_runs.analysis.study_history_plots import (
    PLOTTED_METRICS,
    RELATION_PLOTS,
    aggregate_study_history,
    generate_study_history_plots,
    load_study_history_frame,
)


def _write_run(
    study_root: Path,
    *,
    run_id: str,
    offset: float,
    latent_max_age: int = 50,
) -> None:
    run_root = study_root / run_id
    metrics_root = run_root / "metrics" / "mercury"
    metrics_root.mkdir(parents=True, exist_ok=True)
    history_path = metrics_root / f"{run_id}_paper_precision.json"
    history_path.write_text(
        json.dumps(
            {
                "method": "mercury",
                "history": [
                    {
                        "step": 1,
                        "sensory_precision": 0.2 + offset,
                        "latent_precision": 0.4 + offset,
                        "sensory_inferred_state_count": 3,
                        "sensory_ground_truth_state_count": 4,
                        "latent_inferred_state_count": 3,
                        "latent_ground_truth_state_count": 4,
                        "sensory_edge_precision": 0.1 + offset,
                        "latent_edge_precision": 0.2 + offset,
                        "sensory_edge_recall": 0.3 + offset,
                        "latent_edge_recall": 0.5 + offset,
                        "sensory_edge_f1": 0.4 + offset,
                        "latent_edge_f1": 0.6 + offset,
                        "sensory_mean_total_variation": 0.5 + offset,
                        "latent_mean_total_variation": 0.7 + offset,
                        "sensory_action_conditioned_edge_precision": 0.15 + offset,
                        "latent_action_conditioned_edge_precision": 0.25 + offset,
                        "sensory_action_conditioned_edge_recall": 0.35 + offset,
                        "latent_action_conditioned_edge_recall": 0.45 + offset,
                        "sensory_action_conditioned_edge_f1": 0.55 + offset,
                        "latent_action_conditioned_edge_f1": 0.65 + offset,
                        "sensory_action_conditioned_mean_total_variation": 0.75 + offset,
                        "latent_action_conditioned_mean_total_variation": 0.85 + offset,
                    },
                    {
                        "step": 2,
                        "sensory_precision": 0.25 + offset,
                        "latent_precision": 0.45 + offset,
                        "sensory_inferred_state_count": 3,
                        "sensory_ground_truth_state_count": 4,
                        "latent_inferred_state_count": 3,
                        "latent_ground_truth_state_count": 4,
                        "sensory_edge_precision": 0.12 + offset,
                        "latent_edge_precision": 0.22 + offset,
                        "sensory_edge_recall": 0.32 + offset,
                        "latent_edge_recall": 0.52 + offset,
                        "sensory_edge_f1": 0.42 + offset,
                        "latent_edge_f1": 0.62 + offset,
                        "sensory_mean_total_variation": 0.52 + offset,
                        "latent_mean_total_variation": 0.72 + offset,
                        "sensory_action_conditioned_edge_precision": 0.17 + offset,
                        "latent_action_conditioned_edge_precision": 0.27 + offset,
                        "sensory_action_conditioned_edge_recall": 0.37 + offset,
                        "latent_action_conditioned_edge_recall": 0.47 + offset,
                        "sensory_action_conditioned_edge_f1": 0.57 + offset,
                        "latent_action_conditioned_edge_f1": 0.67 + offset,
                        "sensory_action_conditioned_mean_total_variation": 0.77 + offset,
                        "latent_action_conditioned_mean_total_variation": 0.87 + offset,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    (run_root / "run_status.json").write_text(
        json.dumps({"artifacts": {"mercury_paper_precision_json": str(history_path)}}),
        encoding="utf-8",
    )
    (run_root / "run_config.json").write_text(
        json.dumps({"cli_args": {"latent_max_age": latent_max_age}}),
        encoding="utf-8",
    )


def test_load_study_history_frame_reads_run_histories(tmp_path: Path) -> None:
    study_root = tmp_path / "study"
    _write_run(study_root, run_id="run-a", offset=0.0)
    pl.DataFrame(
        {
            "run_id": ["run-a"],
            "method": ["mercury"],
            "level": [13],
            "seed": [0],
            "rand_prob": [0.0],
            "weight_undirected": [0.05],
            "lambda_trace": [0.9],
        }
    ).write_parquet(study_root / "study_summary.parquet")

    frame = load_study_history_frame(study_root)

    assert frame.shape == (2, 30)
    assert frame.get_column("step").to_list() == [1, 2]
    assert frame.get_column("latent_max_age").to_list() == [50.0, 50.0]
    assert frame.get_column("sensory_node_count").to_list() == [3.0, 3.0]
    assert frame.get_column("latent_node_count").to_list() == [3.0, 3.0]
    assert frame.get_column("sensory_state_count_ratio").to_list() == [0.75, 0.75]
    assert frame.get_column("latent_state_count_ratio").to_list() == [0.75, 0.75]
    for metric_key, _ in PLOTTED_METRICS:
        assert f"sensory_{metric_key}" in frame.columns
        assert f"latent_{metric_key}" in frame.columns


def test_aggregate_study_history_computes_mean_and_std_across_seeds() -> None:
    frame = pl.DataFrame(
        {
            "level": [13, 13, 13, 13],
            "step": [1, 1, 2, 2],
            "seed": [0, 1, 0, 1],
            "rand_prob": [0.0, 0.0, 0.0, 0.0],
            "weight_undirected": [0.05, 0.05, 0.05, 0.05],
            "lambda_trace": [0.9, 0.9, 0.9, 0.9],
            "sensory_precision": [0.2, 0.4, 0.3, 0.5],
            "latent_precision": [0.1, 0.3, 0.2, 0.4],
        }
    )

    aggregated = aggregate_study_history(frame)
    record = aggregated.filter(pl.col("step") == 1).to_dicts()[0]

    assert record["seed_count"] == 2
    assert round(record["sensory_precision_mean"], 6) == 0.3
    assert round(record["sensory_precision_std"], 6) == round(0.14142135623730953, 6)
    assert round(record["latent_precision_mean"], 6) == 0.2


def test_generate_study_history_plots_writes_one_plot_per_level(tmp_path: Path) -> None:
    study_root = tmp_path / "study"
    rows: list[dict[str, object]] = []
    for level in (13, 21):
        for seed, offset in ((0, 0.0), (1, 0.1)):
            for rand_prob in (0.0, 0.3):
                for weight_undirected in (0.0, 0.05):
                    for lambda_trace in (0.8, 0.9):
                        for latent_max_age in (18, 50):
                            run_id = f"run-{level}-{seed}-{rand_prob}-{weight_undirected}-{lambda_trace}-{latent_max_age}"
                            _write_run(study_root, run_id=run_id, offset=offset, latent_max_age=latent_max_age)
                            rows.append(
                                {
                                    "run_id": run_id,
                                    "method": "mercury",
                                    "level": level,
                                    "seed": seed,
                                    "rand_prob": rand_prob,
                                    "weight_undirected": weight_undirected,
                                    "lambda_trace": lambda_trace,
                                }
                            )
    pl.DataFrame(rows).write_parquet(study_root / "study_summary.parquet")

    outputs = generate_study_history_plots(study_root=study_root)

    expected_per_level = (len(PLOTTED_METRICS) + len(RELATION_PLOTS)) * (1 + 2 + 2 + 2 + 2)
    assert len(outputs) == expected_per_level * 2
    assert outputs["level=13|metric=precision|view=all"] == study_root / "plots" / "study_history" / "level_13" / "precision" / "all_vary" / "plot.png"
    assert outputs["level=13|metric=precision|fixed=rand_prob|value=0"] == study_root / "plots" / "study_history" / "level_13" / "precision" / "fixed_rand_prob" / "value_0p0.png"
    assert outputs["level=13|metric=precision|fixed=weight_undirected|value=0.05"] == study_root / "plots" / "study_history" / "level_13" / "precision" / "fixed_weight_undirected" / "value_0p05.png"
    assert outputs["level=13|metric=precision|fixed=lambda_trace|value=0.9"] == study_root / "plots" / "study_history" / "level_13" / "precision" / "fixed_lambda_trace" / "value_0p9.png"
    assert outputs["level=13|metric=precision|fixed=latent_max_age|value=18"] == study_root / "plots" / "study_history" / "level_13" / "precision" / "fixed_latent_max_age" / "value_18p0.png"
    assert outputs["level=13|metric=edge_f1_vs_state_count_ratio_pareto|view=all"] == study_root / "plots" / "study_history" / "level_13" / "edge_f1_vs_state_count_ratio_pareto" / "all_vary" / "plot.png"
    assert outputs["level=13|metric=precision|view=all"].exists()
    assert outputs["level=13|metric=precision|fixed=rand_prob|value=0"].exists()
