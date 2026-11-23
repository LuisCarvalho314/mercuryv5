from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from scripts.entropy_curves import (
    EMPIRICAL_RANDOM_PROBABILITIES,
    LEVEL_INDICES,
    _probability_weighted_integral_records_by_level,
    _structural_unweighted_integral_records_by_level,
    bit_threshold_to_state_normalized_threshold,
    build_empirical_integral_records,
    build_empirical_near_zero_area_records,
    build_level_entropy_model,
    compute_empirical_condition_outputs,
    compute_empirical_history_entropy_summaries,
    compute_empirical_next_state_condition_outputs,
    compute_empirical_next_state_history_entropy_summaries,
    compute_empirical_next_state_sensorimotor_history_entropy_summaries,
    compute_empirical_sensorimotor_history_entropy_summaries,
    compute_history_conditioned_state_entropy_summaries,
    compute_history_conditioned_next_state_entropy_summaries,
    dataset_artifact_paths,
    generate_empirical_entropy_run_ids,
    integrate_entropy_heatmap,
    integrate_entropy_heatmap_area_below_threshold,
    plot_empirical_integral_records,
    plot_empirical_integral_records_by_level,
    plot_empirical_near_zero_area_records,
    plot_empirical_near_zero_area_records_by_level,
    plot_empirical_structural_integral_records,
    plot_empirical_structural_integral_records_by_level,
    plot_empirical_structural_near_zero_area_records,
    plot_empirical_structural_near_zero_area_records_by_level,
    run_empirical_entropy_outputs_for_level,
    run_empirical_entropy_outputs_for_levels,
    run_empirical_normalized_entropy_outputs_for_level,
    run_empirical_normalized_entropy_outputs_for_levels,
    run_empirical_normalized_next_state_entropy_outputs_for_level,
    run_empirical_next_state_entropy_outputs_for_level,
    state_count_entropy_log_base,
)
from src.maze_environment.LEVEL import levels
from src.maze_environment.agent import Agent


def _maze_for_level(level_index: int) -> np.ndarray:
    level = levels[level_index]
    maze = np.zeros((len(level), len(level[0])), dtype=np.uint8)
    for row_index, row in enumerate(level):
        for col_index, value in enumerate(row):
            if value == "X":
                maze[row_index, col_index] = 1
    return maze


def test_build_level_entropy_model_uses_level_accessible_positions() -> None:
    transitions, observations = build_level_entropy_model(
        level_index=19,
        sensor="cardinal distance",
        sensor_range=None,
    )

    expected_positions = {(1, col_index) for col_index in range(1, 7)}

    assert set(transitions) == expected_positions
    assert set(observations) == expected_positions


def test_build_level_entropy_model_keeps_boundary_moves_as_self_loops() -> None:
    transitions, _ = build_level_entropy_model(
        level_index=19,
        sensor="cardinal distance",
        sensor_range=None,
    )

    assert transitions[(1, 1)]["west"] == {(1, 1): 1.0}
    assert transitions[(1, 1)]["north"] == {(1, 1): 1.0}
    assert transitions[(1, 1)]["south"] == {(1, 1): 1.0}
    assert transitions[(1, 6)]["east"] == {(1, 6): 1.0}


def test_build_level_entropy_model_can_filter_collision_moves() -> None:
    transitions, _ = build_level_entropy_model(
        level_index=19,
        sensor="cardinal distance",
        sensor_range=None,
        collision_filtered=True,
    )

    assert "west" not in transitions[(1, 1)]
    assert "north" not in transitions[(1, 1)]
    assert "south" not in transitions[(1, 1)]
    assert "east" not in transitions[(1, 6)]
    assert transitions[(1, 1)]["east"] == {(1, 2): 1.0}
    assert transitions[(1, 6)]["west"] == {(1, 5): 1.0}


def test_build_level_entropy_model_moves_between_open_cells() -> None:
    transitions, _ = build_level_entropy_model(
        level_index=19,
        sensor="cardinal distance",
        sensor_range=None,
    )

    assert transitions[(1, 1)]["east"] == {(1, 2): 1.0}
    assert transitions[(1, 6)]["west"] == {(1, 5): 1.0}


def test_build_level_entropy_model_matches_agent_cardinal_distance_observation() -> None:
    _, observations = build_level_entropy_model(
        level_index=19,
        sensor="cardinal distance",
        sensor_range=None,
    )
    maze = _maze_for_level(19)
    agent = Agent(
        initial_position=(1, 1),
        maze=maze,
        sensors={"sensor": "cardinal distance", "range": None},
    )

    assert observations[(1, 1)] == agent.make_observation(maze)


def test_generated_level_model_can_compute_k0_entropy_summaries() -> None:
    transitions, observations = build_level_entropy_model(
        level_index=19,
        sensor="cardinal distance",
        sensor_range=None,
    )

    summaries = compute_history_conditioned_state_entropy_summaries(
        transition_probabilities=transitions,
        observation_by_state=observations,
        max_history_length=0,
        weight_by_history_mass=True,
        restrict_to_aliased_histories=False,
    )

    assert len(summaries["observation_only"]["mean"]) == 1
    assert len(summaries["sensorimotor"]["mean"]) == 1


def test_entropy_summaries_include_one_percent_quantiles() -> None:
    transitions, observations = build_level_entropy_model(
        level_index=19,
        sensor="cardinal distance",
        sensor_range=None,
    )

    summaries = compute_history_conditioned_state_entropy_summaries(
        transition_probabilities=transitions,
        observation_by_state=observations,
        max_history_length=0,
        weight_by_history_mass=True,
        restrict_to_aliased_histories=False,
    )

    quantile_keys = {
        key for key in summaries["observation_only"] if key.startswith("q")
    }

    assert len(quantile_keys) == 101
    assert "q01" in quantile_keys
    assert "q99" in quantile_keys


def test_integrate_entropy_heatmap_integrates_fraction_and_history_axes() -> None:
    summary_family = {"mean": [1.0, 3.0]}
    for percentage in range(101):
        summary_family[f"q{percentage:02d}"] = [1.0, 3.0]

    assert np.isclose(integrate_entropy_heatmap(summary_family), 2.0)


def test_integrate_entropy_heatmap_area_below_threshold_counts_below_values() -> None:
    summary_family = {"mean": [0.0, 0.02]}
    for percentage in range(101):
        summary_family[f"q{percentage:02d}"] = [0.0, 0.02]

    assert np.isclose(
        integrate_entropy_heatmap_area_below_threshold(
            summary_family,
            threshold=1e-2,
        ),
        0.5,
    )


def test_integrate_entropy_heatmap_area_below_threshold_is_strict() -> None:
    summary_family = {"mean": [1e-2, 1e-2]}
    for percentage in range(101):
        summary_family[f"q{percentage:02d}"] = [1e-2, 1e-2]

    assert np.isclose(
        integrate_entropy_heatmap_area_below_threshold(
            summary_family,
            threshold=1e-2,
        ),
        0.0,
    )


def test_integrate_entropy_heatmap_area_below_threshold_full_domain() -> None:
    summary_family = {"mean": [0.0, 0.0]}
    for percentage in range(101):
        summary_family[f"q{percentage:02d}"] = [0.0, 0.0]

    assert np.isclose(
        integrate_entropy_heatmap_area_below_threshold(
            summary_family,
            threshold=1e-2,
        ),
        1.0,
    )


def _write_dataset(
    root: Path,
    *,
    level_index: int,
    sensor_key: str,
    run_id: str,
    observations: list[tuple[int, ...]],
    actions: list[tuple[int, ...]] | None = None,
    seed: int,
    rand_prob: float,
    valid_only: bool,
    start_position: tuple[int, int],
    collisions: list[bool] | None = None,
) -> None:
    parquet_path, metadata_path = dataset_artifact_paths(
        datasets_root=root,
        level_index=level_index,
        sensor_key=sensor_key,
        run_id=run_id,
    )
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    width = len(observations[0])
    actions = actions or [(1, 0, 0, 0)] * len(observations)
    action_width = len(actions[0])
    columns = {
        "iteration": list(range(len(observations))),
        "collision": collisions or [False] * len(observations),
    }
    for index in range(width):
        columns[f"observation_{index}"] = [row[index] for row in observations]
    for index in range(action_width):
        columns[f"action_{index}"] = [row[index] for row in actions]
    pl.DataFrame(columns).write_parquet(parquet_path)
    metadata_path.write_text(
        (
            "{"
            f'"run_id": "{run_id}", '
            f'"random_seed": {seed}, '
            f'"random_prob": {rand_prob}, '
            f'"valid_trajectories_only": {str(valid_only).lower()}, '
            '"random_start": true, '
            f'"initial_agent_position": [{start_position[0]}, {start_position[1]}], '
            f'"generation_stats": {{"saved_steps": {len(observations)}, "attempted_steps": {len(observations) + 1}, "dropped_collision_steps": 1}}'
            "}"
        ),
        encoding="utf-8",
    )


def test_generate_empirical_entropy_run_ids_keeps_conditions_separate(monkeypatch, tmp_path: Path) -> None:
    calls = []

    def _fake_generate_data(**kwargs):
        calls.append(kwargs)
        suffix = f"s{kwargs['seed']}_p{kwargs['rand_prob']}_v{kwargs['valid_trajectories_only']}"
        return {
            "cardinal distance:range=1": f"sensory_{suffix}",
            "cartesian": f"cartesian_{suffix}",
        }

    monkeypatch.setattr("scripts.entropy_curves.generate_data", _fake_generate_data)

    runs = generate_empirical_entropy_run_ids(
        level_index=20,
        datasets_root=tmp_path,
        seeds=(0, 1),
        random_probabilities=(0.0, 0.3),
        valid_only_options=(False, True),
        num_steps=7,
        random_start=True,
    )

    assert set(runs) == {(0.0, False), (0.0, True), (0.3, False), (0.3, True)}
    assert [run["seed"] for run in runs[(0.3, True)]] == [0, 1]
    assert all(call["random_start"] is True for call in calls)
    assert all(call["num_steps"] == 7 for call in calls)


def test_empirical_random_probabilities_are_decimal_sweep() -> None:
    assert list(EMPIRICAL_RANDOM_PROBABILITIES) == [0.0, 0.3]


def test_default_empirical_level_indices_match_existing_outputs() -> None:
    assert LEVEL_INDICES == (13, 20, 21)


def test_state_count_log_base_uses_accessible_level_positions() -> None:
    assert state_count_entropy_log_base(19) == 6.0


def test_bit_threshold_converts_to_state_normalized_entropy_units() -> None:
    assert np.isclose(
        bit_threshold_to_state_normalized_threshold(
            threshold_bits=0.01,
            level_index=19,
        ),
        0.01 / np.log2(6),
    )


def test_log_base_num_states_normalizes_uniform_state_entropy_to_one() -> None:
    summaries = compute_empirical_history_entropy_summaries(
        sensory_histories_by_seed=[
            (
                [(0,), (0,), (0,), (0,)],
                [(0,), (1,), (2,), (3,)],
            )
        ],
        max_history_length=0,
        log_base=4.0,
        weight_by_history_mass=True,
    )

    assert np.isclose(summaries["mean"][0], 1.0)


def _constant_empirical_summaries(value: float = 1.0) -> dict[str, dict[str, list[float]]]:
    summary_family = {"mean": [value, value]}
    for percentage in range(101):
        summary_family[f"q{percentage:02d}"] = [value, value]
    return {
        "observation_only": summary_family,
        "sensorimotor": summary_family,
    }


def test_per_level_empirical_runner_passes_level_to_generation_and_compute(
    monkeypatch,
    tmp_path: Path,
) -> None:
    generate_calls = []
    compute_calls = []
    summaries = _constant_empirical_summaries()

    def _fake_generate_empirical_entropy_run_ids(**kwargs):
        generate_calls.append(kwargs)
        return {
            (0.0, False): [
                {
                    "seed": 0,
                    "run_ids": {
                        "cardinal distance:range=1": "sensory",
                        "cartesian": "cartesian",
                    },
                }
            ]
        }

    def _fake_compute_empirical_condition_outputs(**kwargs):
        compute_calls.append(kwargs)
        return summaries, summaries, {
            "num_seeds": 1,
            "saved_rows": 1,
            "attempted_rows": 1,
            "dropped_collision_rows": 0,
            "start_positions": [(1, 1)],
        }

    monkeypatch.setattr(
        "scripts.entropy_curves.generate_empirical_entropy_run_ids",
        _fake_generate_empirical_entropy_run_ids,
    )
    monkeypatch.setattr(
        "scripts.entropy_curves.compute_empirical_condition_outputs",
        _fake_compute_empirical_condition_outputs,
    )
    monkeypatch.setattr("scripts.entropy_curves.OUTPUT_DIRECTORY", tmp_path)
    monkeypatch.setattr(
        "scripts.entropy_curves.print_empirical_metadata_totals",
        lambda **kwargs: None,
    )
    monkeypatch.setattr(
        "scripts.entropy_curves.print_summary_table",
        lambda **kwargs: None,
    )
    monkeypatch.setattr(
        "scripts.entropy_curves.print_heatmap_integrals",
        lambda **kwargs: None,
    )
    monkeypatch.setattr(
        "scripts.entropy_curves.plot_observation_and_sensorimotor_entropy_summaries",
        lambda **kwargs: None,
    )
    monkeypatch.setattr(
        "scripts.entropy_curves.plot_empirical_integral_records",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "scripts.entropy_curves.plot_empirical_structural_integral_records",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "scripts.entropy_curves.plot_empirical_near_zero_area_records",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "scripts.entropy_curves.plot_empirical_structural_near_zero_area_records",
        lambda *args, **kwargs: None,
    )

    run_empirical_entropy_outputs_for_level(13)

    assert generate_calls[0]["level_index"] == 13
    assert compute_calls[0]["level_index"] == 13


def test_per_level_empirical_runner_returns_level_annotated_records(
    monkeypatch,
    tmp_path: Path,
) -> None:
    summaries = _constant_empirical_summaries()

    def _fake_generate_empirical_entropy_run_ids(**kwargs):
        return {
            (0.0, False): [
                {
                    "seed": 0,
                    "run_ids": {
                        "cardinal distance:range=1": "sensory",
                        "cartesian": "cartesian",
                    },
                }
            ]
        }

    def _fake_compute_empirical_condition_outputs(**kwargs):
        return summaries, summaries, {
            "num_seeds": 1,
            "saved_rows": 1,
            "attempted_rows": 1,
            "dropped_collision_rows": 0,
            "start_positions": [(1, 1)],
        }

    monkeypatch.setattr(
        "scripts.entropy_curves.generate_empirical_entropy_run_ids",
        _fake_generate_empirical_entropy_run_ids,
    )
    monkeypatch.setattr(
        "scripts.entropy_curves.compute_empirical_condition_outputs",
        _fake_compute_empirical_condition_outputs,
    )
    monkeypatch.setattr("scripts.entropy_curves.OUTPUT_DIRECTORY", tmp_path)
    monkeypatch.setattr(
        "scripts.entropy_curves.print_empirical_metadata_totals",
        lambda **kwargs: None,
    )
    monkeypatch.setattr(
        "scripts.entropy_curves.print_summary_table",
        lambda **kwargs: None,
    )
    monkeypatch.setattr(
        "scripts.entropy_curves.print_heatmap_integrals",
        lambda **kwargs: None,
    )
    monkeypatch.setattr(
        "scripts.entropy_curves.plot_observation_and_sensorimotor_entropy_summaries",
        lambda **kwargs: None,
    )
    monkeypatch.setattr(
        "scripts.entropy_curves.plot_empirical_integral_records",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "scripts.entropy_curves.plot_empirical_structural_integral_records",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "scripts.entropy_curves.plot_empirical_near_zero_area_records",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "scripts.entropy_curves.plot_empirical_structural_near_zero_area_records",
        lambda *args, **kwargs: None,
    )

    records = run_empirical_entropy_outputs_for_level(21)

    assert records
    assert {record["level_index"] for record in records["integral"]} == {21}
    assert {record["level_index"] for record in records["near_zero_area"]} == {21}
    assert {
        record["summary_mode"]
        for record in records["integral"]
    } == {"structural_unweighted", "probability_weighted"}
    assert {
        record["summary_mode"]
        for record in records["near_zero_area"]
    } == {"structural_unweighted", "probability_weighted"}


def test_multi_level_empirical_runner_sends_all_records_to_cross_level_plot(
    monkeypatch,
    tmp_path: Path,
) -> None:
    probability_plotted_records = []
    structural_plotted_records = []
    near_zero_plotted_records = []
    structural_near_zero_plotted_records = []
    plotted_paths = []

    def _fake_run_level(level_index: int):
        base_records = [
            {
                "level_index": level_index,
                "rand_prob": 0.0,
                "valid_trajectories_only": False,
                "summary_mode": "probability_weighted",
                "history_type": "observation_only",
            },
            {
                "level_index": level_index,
                "rand_prob": 0.0,
                "valid_trajectories_only": False,
                "summary_mode": "structural_unweighted",
                "history_type": "observation_only",
            },
        ]
        return {
            "integral": [
                {**record, "integral": float(level_index) + index}
                for index, record in enumerate(base_records)
            ],
            "near_zero_area": [
                {**record, "area": float(level_index) + index}
                for index, record in enumerate(base_records)
            ],
        }

    def _fake_plot_by_level(records, *, output_path, title):
        probability_plotted_records.extend(records)
        plotted_paths.append(Path(output_path))

    def _fake_plot_structural_by_level(records, *, output_path, title):
        structural_plotted_records.extend(records)
        plotted_paths.append(Path(output_path))

    def _fake_plot_near_zero_by_level(records, *, output_path, title):
        near_zero_plotted_records.extend(records)
        plotted_paths.append(Path(output_path))

    def _fake_plot_structural_near_zero_by_level(records, *, output_path, title):
        structural_near_zero_plotted_records.extend(records)
        plotted_paths.append(Path(output_path))

    monkeypatch.setattr("scripts.entropy_curves.OUTPUT_DIRECTORY", tmp_path)
    monkeypatch.setattr(
        "scripts.entropy_curves.run_empirical_entropy_outputs_for_level",
        _fake_run_level,
    )
    monkeypatch.setattr(
        "scripts.entropy_curves.plot_empirical_integral_records_by_level",
        _fake_plot_by_level,
    )
    monkeypatch.setattr(
        "scripts.entropy_curves.plot_empirical_structural_integral_records_by_level",
        _fake_plot_structural_by_level,
    )
    monkeypatch.setattr(
        "scripts.entropy_curves.plot_empirical_near_zero_area_records_by_level",
        _fake_plot_near_zero_by_level,
    )
    monkeypatch.setattr(
        "scripts.entropy_curves.plot_empirical_structural_near_zero_area_records_by_level",
        _fake_plot_structural_near_zero_by_level,
    )

    records = run_empirical_entropy_outputs_for_levels((13, 20, 21))

    assert {record["level_index"] for record in records["integral"]} == {13, 20, 21}
    assert {record["level_index"] for record in records["near_zero_area"]} == {13, 20, 21}
    assert probability_plotted_records == []
    assert structural_plotted_records == []
    assert near_zero_plotted_records == []
    assert structural_near_zero_plotted_records == []
    assert plotted_paths == []


def test_multi_level_normalized_empirical_runner_uses_distinct_cross_level_output_names(
    monkeypatch,
    tmp_path: Path,
) -> None:
    plotted_paths = []

    def _fake_run_level(level_index: int):
        return {
            "integral": [
                {
                    "level_index": level_index,
                    "rand_prob": 0.0,
                    "valid_trajectories_only": False,
                    "summary_mode": "probability_weighted",
                    "history_type": "observation_only",
                    "integral": float(level_index),
                }
            ],
            "near_zero_area": [
                {
                    "level_index": level_index,
                    "rand_prob": 0.0,
                    "valid_trajectories_only": False,
                    "summary_mode": "probability_weighted",
                    "history_type": "observation_only",
                    "area": float(level_index),
                }
            ],
        }

    def _fake_plot(records, *, output_path, title, **kwargs):
        plotted_paths.append(Path(output_path))

    monkeypatch.setattr("scripts.entropy_curves.OUTPUT_DIRECTORY", tmp_path)
    monkeypatch.setattr(
        "scripts.entropy_curves.run_empirical_normalized_entropy_outputs_for_level",
        _fake_run_level,
    )
    monkeypatch.setattr(
        "scripts.entropy_curves.plot_empirical_integral_records_by_level",
        _fake_plot,
    )
    monkeypatch.setattr(
        "scripts.entropy_curves.plot_empirical_structural_integral_records_by_level",
        _fake_plot,
    )
    monkeypatch.setattr(
        "scripts.entropy_curves.plot_empirical_near_zero_area_records_by_level",
        _fake_plot,
    )
    monkeypatch.setattr(
        "scripts.entropy_curves.plot_empirical_structural_near_zero_area_records_by_level",
        _fake_plot,
    )

    run_empirical_normalized_entropy_outputs_for_levels((13, 20))

    assert plotted_paths == []


def test_per_level_empirical_runner_uses_level_specific_output_names(
    monkeypatch,
    tmp_path: Path,
) -> None:
    summaries = _constant_empirical_summaries()
    condition_filename_stems = []
    integral_output_paths = []

    def _fake_generate_empirical_entropy_run_ids(**kwargs):
        return {
            (0.0, False): [
                {
                    "seed": 0,
                    "run_ids": {
                        "cardinal distance:range=1": "sensory",
                        "cartesian": "cartesian",
                    },
                }
            ]
        }

    def _fake_compute_empirical_condition_outputs(**kwargs):
        return summaries, summaries, {
            "num_seeds": 1,
            "saved_rows": 1,
            "attempted_rows": 1,
            "dropped_collision_rows": 0,
            "start_positions": [(1, 1)],
        }

    def _fake_plot_condition(**kwargs):
        condition_filename_stems.append(kwargs["filename_stem"])

    def _fake_plot_integrals(records, *, output_path, title, **kwargs):
        integral_output_paths.append(Path(output_path))

    monkeypatch.setattr(
        "scripts.entropy_curves.generate_empirical_entropy_run_ids",
        _fake_generate_empirical_entropy_run_ids,
    )
    monkeypatch.setattr(
        "scripts.entropy_curves.compute_empirical_condition_outputs",
        _fake_compute_empirical_condition_outputs,
    )
    monkeypatch.setattr("scripts.entropy_curves.OUTPUT_DIRECTORY", tmp_path)
    monkeypatch.setattr(
        "scripts.entropy_curves.print_empirical_metadata_totals",
        lambda **kwargs: None,
    )
    monkeypatch.setattr(
        "scripts.entropy_curves.print_summary_table",
        lambda **kwargs: None,
    )
    monkeypatch.setattr(
        "scripts.entropy_curves.print_heatmap_integrals",
        lambda **kwargs: None,
    )
    monkeypatch.setattr(
        "scripts.entropy_curves.plot_observation_and_sensorimotor_entropy_summaries",
        _fake_plot_condition,
    )
    monkeypatch.setattr(
        "scripts.entropy_curves.plot_empirical_integral_records",
        _fake_plot_integrals,
    )
    monkeypatch.setattr(
        "scripts.entropy_curves.plot_empirical_structural_integral_records",
        _fake_plot_integrals,
    )
    monkeypatch.setattr(
        "scripts.entropy_curves.plot_empirical_near_zero_area_records",
        _fake_plot_integrals,
    )
    monkeypatch.setattr(
        "scripts.entropy_curves.plot_empirical_structural_near_zero_area_records",
        _fake_plot_integrals,
    )

    run_empirical_entropy_outputs_for_level(13)
    run_empirical_entropy_outputs_for_level(20)

    assert "level_13_cardinal_distance_range_1_empirical_p0_allsteps_randomstart_structural_unweighted" in condition_filename_stems
    assert "level_13_cardinal_distance_range_1_empirical_p0_allsteps_randomstart_probability_weighted" in condition_filename_stems
    assert "level_20_cardinal_distance_range_1_empirical_p0_allsteps_randomstart_structural_unweighted" in condition_filename_stems
    assert "level_20_cardinal_distance_range_1_empirical_p0_allsteps_randomstart_probability_weighted" in condition_filename_stems
    assert integral_output_paths == []
    assert len(set(condition_filename_stems)) == len(condition_filename_stems)
    assert len(set(integral_output_paths)) == len(integral_output_paths)


def test_per_level_normalized_runner_uses_state_count_log_base_and_converted_threshold(
    monkeypatch,
    tmp_path: Path,
) -> None:
    summaries = _constant_empirical_summaries()
    compute_calls = []
    near_zero_thresholds = []
    condition_filename_stems = []

    def _fake_generate_empirical_entropy_run_ids(**kwargs):
        return {
            (0.0, False): [
                {
                    "seed": 0,
                    "run_ids": {
                        "cardinal distance:range=1": "sensory",
                        "cartesian": "cartesian",
                    },
                }
            ]
        }

    def _fake_compute_empirical_condition_outputs(**kwargs):
        compute_calls.append(kwargs)
        return summaries, summaries, {
            "num_seeds": 1,
            "saved_rows": 1,
            "attempted_rows": 1,
            "dropped_collision_rows": 0,
            "start_positions": [(1, 1)],
        }

    def _fake_near_zero_records(**kwargs):
        near_zero_thresholds.append(kwargs["threshold"])
        return []

    def _fake_plot_condition(**kwargs):
        condition_filename_stems.append(kwargs["filename_stem"])

    monkeypatch.setattr(
        "scripts.entropy_curves.generate_empirical_entropy_run_ids",
        _fake_generate_empirical_entropy_run_ids,
    )
    monkeypatch.setattr(
        "scripts.entropy_curves.compute_empirical_condition_outputs",
        _fake_compute_empirical_condition_outputs,
    )
    monkeypatch.setattr(
        "scripts.entropy_curves.build_empirical_near_zero_area_records",
        _fake_near_zero_records,
    )
    monkeypatch.setattr("scripts.entropy_curves.OUTPUT_DIRECTORY", tmp_path)
    monkeypatch.setattr(
        "scripts.entropy_curves.print_empirical_metadata_totals",
        lambda **kwargs: None,
    )
    monkeypatch.setattr(
        "scripts.entropy_curves.print_summary_table",
        lambda **kwargs: None,
    )
    monkeypatch.setattr(
        "scripts.entropy_curves.print_heatmap_integrals",
        lambda **kwargs: None,
    )
    monkeypatch.setattr(
        "scripts.entropy_curves.plot_observation_and_sensorimotor_entropy_summaries",
        _fake_plot_condition,
    )
    monkeypatch.setattr(
        "scripts.entropy_curves.plot_empirical_integral_records",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "scripts.entropy_curves.plot_empirical_structural_integral_records",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "scripts.entropy_curves.plot_empirical_near_zero_area_records",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "scripts.entropy_curves.plot_empirical_structural_near_zero_area_records",
        lambda *args, **kwargs: None,
    )

    run_empirical_normalized_entropy_outputs_for_level(19)

    assert compute_calls[0]["log_base"] == 6.0
    assert np.isclose(near_zero_thresholds[0], 0.01 / np.log2(6))
    assert condition_filename_stems == [
        "level_19_cardinal_distance_range_1_normalized_by_num_states_empirical_p0_allsteps_randomstart_structural_unweighted",
        "level_19_cardinal_distance_range_1_normalized_by_num_states_empirical_p0_allsteps_randomstart_probability_weighted",
    ]


def test_per_level_normalized_next_state_runner_uses_state_count_log_base(
    monkeypatch,
    tmp_path: Path,
) -> None:
    summaries = _constant_empirical_summaries()
    compute_calls = []

    def _fake_generate_empirical_entropy_run_ids(**kwargs):
        return {
            (0.0, False): [
                {
                    "seed": 0,
                    "run_ids": {
                        "cardinal distance:range=1": "sensory",
                        "cartesian": "cartesian",
                    },
                }
            ]
        }

    def _fake_compute_empirical_next_state_condition_outputs(**kwargs):
        compute_calls.append(kwargs)
        return summaries, summaries, {
            "num_seeds": 1,
            "saved_rows": 1,
            "attempted_rows": 1,
            "dropped_collision_rows": 0,
            "start_positions": [(1, 1)],
        }

    monkeypatch.setattr(
        "scripts.entropy_curves.generate_empirical_entropy_run_ids",
        _fake_generate_empirical_entropy_run_ids,
    )
    monkeypatch.setattr(
        "scripts.entropy_curves.compute_empirical_next_state_condition_outputs",
        _fake_compute_empirical_next_state_condition_outputs,
    )
    monkeypatch.setattr("scripts.entropy_curves.OUTPUT_DIRECTORY", tmp_path)
    monkeypatch.setattr(
        "scripts.entropy_curves.print_empirical_metadata_totals",
        lambda **kwargs: None,
    )
    monkeypatch.setattr(
        "scripts.entropy_curves.print_summary_table",
        lambda **kwargs: None,
    )
    monkeypatch.setattr(
        "scripts.entropy_curves.print_heatmap_integrals",
        lambda **kwargs: None,
    )
    monkeypatch.setattr(
        "scripts.entropy_curves.plot_observation_and_sensorimotor_entropy_summaries",
        lambda **kwargs: None,
    )
    monkeypatch.setattr(
        "scripts.entropy_curves.plot_empirical_integral_records",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "scripts.entropy_curves.plot_empirical_structural_integral_records",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "scripts.entropy_curves.plot_empirical_near_zero_area_records",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "scripts.entropy_curves.plot_empirical_structural_near_zero_area_records",
        lambda *args, **kwargs: None,
    )

    run_empirical_normalized_next_state_entropy_outputs_for_level(19)

    assert compute_calls[0]["log_base"] == 6.0


def test_empirical_condition_outputs_load_generated_paired_run_ids(tmp_path: Path) -> None:
    sensory_run_id = "sensory_selected"
    cartesian_run_id = "cartesian_selected"
    _write_dataset(
        tmp_path,
        level_index=20,
        sensor_key="cardinal distance:range=1",
        run_id=sensory_run_id,
        observations=[(0,), (0,), (0,), (1,)],
        seed=0,
        rand_prob=0.3,
        valid_only=False,
        start_position=(1, 1),
    )
    _write_dataset(
        tmp_path,
        level_index=20,
        sensor_key="cartesian",
        run_id=cartesian_run_id,
        observations=[(0, 0), (0, 0), (0, 1), (1, 0)],
        seed=0,
        rand_prob=0.3,
        valid_only=False,
        start_position=(1, 1),
    )

    unweighted, weighted, totals = compute_empirical_condition_outputs(
        runs=[
            {
                "seed": 0,
                "run_ids": {
                    "cardinal distance:range=1": sensory_run_id,
                    "cartesian": cartesian_run_id,
                },
            }
        ],
        datasets_root=tmp_path,
        level_index=20,
        max_history_length=0,
    )

    expected_h0 = -((2 / 3) * np.log2(2 / 3) + (1 / 3) * np.log2(1 / 3))
    assert np.isclose(unweighted["observation_only"]["mean"][0], expected_h0 / 2)
    assert np.isclose(weighted["observation_only"]["mean"][0], expected_h0 * 3 / 4)
    assert totals["num_seeds"] == 1
    assert totals["saved_rows"] == 4
    assert totals["attempted_rows"] == 5
    assert totals["dropped_collision_rows"] == 1
    assert totals["start_positions"] == [(1, 1)]


def test_empirical_condition_outputs_threads_log_base_to_summaries(tmp_path: Path) -> None:
    sensory_run_id = "sensory_selected"
    cartesian_run_id = "cartesian_selected"
    _write_dataset(
        tmp_path,
        level_index=20,
        sensor_key="cardinal distance:range=1",
        run_id=sensory_run_id,
        observations=[(0,), (0,), (0,), (0,)],
        seed=0,
        rand_prob=0.3,
        valid_only=False,
        start_position=(1, 1),
    )
    _write_dataset(
        tmp_path,
        level_index=20,
        sensor_key="cartesian",
        run_id=cartesian_run_id,
        observations=[(0,), (1,), (2,), (3,)],
        seed=0,
        rand_prob=0.3,
        valid_only=False,
        start_position=(1, 1),
    )

    _, weighted, _ = compute_empirical_condition_outputs(
        runs=[
            {
                "seed": 0,
                "run_ids": {
                    "cardinal distance:range=1": sensory_run_id,
                    "cartesian": cartesian_run_id,
                },
            }
        ],
        datasets_root=tmp_path,
        level_index=20,
        max_history_length=0,
        log_base=4.0,
    )

    assert np.isclose(weighted["observation_only"]["mean"][0], 1.0)


def test_empirical_k0_entropy_matches_hand_computed_counts() -> None:
    summaries = compute_empirical_history_entropy_summaries(
        sensory_histories_by_seed=[
            (
                [(0,), (0,), (0,), (1,)],
                [(0, 0), (0, 0), (0, 1), (1, 0)],
            )
        ],
        max_history_length=0,
        weight_by_history_mass=False,
    )

    expected_h0 = -((2 / 3) * np.log2(2 / 3) + (1 / 3) * np.log2(1 / 3))
    assert np.isclose(summaries["mean"][0], expected_h0 / 2)


def test_empirical_next_state_k0_entropy_uses_following_state() -> None:
    summaries = compute_empirical_next_state_history_entropy_summaries(
        sensory_histories_by_seed=[
            (
                [(0,), (1,), (0,), (1,)],
                [(0, 0), (0, 1), (1, 0), (0, 1)],
            )
        ],
        max_history_length=0,
        weight_by_history_mass=False,
    )

    assert np.isclose(summaries["mean"][0], 0.0)


def test_empirical_sensorimotor_k0_uses_action_columns_and_can_differ() -> None:
    observation_summaries = compute_empirical_history_entropy_summaries(
        sensory_histories_by_seed=[
            (
                [(0,), (0,), (0,), (1,)],
                [(0, 0), (0, 0), (0, 1), (1, 0)],
            )
        ],
        max_history_length=0,
        weight_by_history_mass=False,
    )
    sensorimotor_summaries = compute_empirical_sensorimotor_history_entropy_summaries(
        sensory_histories_by_seed=[
            (
                [(0,), (0,), (0,), (1,)],
                [(1, 0), (1, 0), (0, 1), (1, 0)],
                [(0, 0), (0, 0), (0, 1), (1, 0)],
            )
        ],
        max_history_length=0,
        weight_by_history_mass=False,
    )

    assert observation_summaries["mean"][0] > 0.0
    assert np.isclose(sensorimotor_summaries["mean"][0], 0.0)


def test_empirical_next_state_sensorimotor_k0_uses_current_action_columns() -> None:
    observation_summaries = compute_empirical_next_state_history_entropy_summaries(
        sensory_histories_by_seed=[
            (
                [(0,), (0,), (0,)],
                [(0, 0), (0, 1), (1, 0)],
            )
        ],
        max_history_length=0,
        weight_by_history_mass=False,
    )
    sensorimotor_summaries = compute_empirical_next_state_sensorimotor_history_entropy_summaries(
        sensory_histories_by_seed=[
            (
                [(0,), (0,), (0,)],
                [(1, 0), (0, 1), (1, 0)],
                [(0, 0), (0, 1), (1, 0)],
            )
        ],
        max_history_length=0,
        weight_by_history_mass=False,
    )

    assert observation_summaries["mean"][0] > 0.0
    assert np.isclose(sensorimotor_summaries["mean"][0], 0.0)


def test_exact_next_state_sensorimotor_k0_conditions_on_current_action() -> None:
    transitions = {
        "left": {"east": {"right": 1.0}},
        "right": {"west": {"left": 1.0}},
    }
    observations = {
        "left": "same",
        "right": "same",
    }

    summaries = compute_history_conditioned_next_state_entropy_summaries(
        transition_probabilities=transitions,
        observation_by_state=observations,
        max_history_length=0,
        restrict_to_aliased_histories=False,
    )

    assert np.isclose(summaries["observation_only"]["mean"][0], 1.0)
    assert np.isclose(summaries["sensorimotor"]["mean"][0], 0.0)


def test_empirical_next_state_condition_outputs_load_generated_paired_run_ids(tmp_path: Path) -> None:
    sensory_run_id = "sensory_selected"
    cartesian_run_id = "cartesian_selected"
    _write_dataset(
        tmp_path,
        level_index=20,
        sensor_key="cardinal distance:range=1",
        run_id=sensory_run_id,
        observations=[(0,), (1,), (0,), (1,)],
        seed=0,
        rand_prob=0.3,
        valid_only=False,
        start_position=(1, 1),
    )
    _write_dataset(
        tmp_path,
        level_index=20,
        sensor_key="cartesian",
        run_id=cartesian_run_id,
        observations=[(0, 0), (0, 1), (1, 0), (0, 1)],
        seed=0,
        rand_prob=0.3,
        valid_only=False,
        start_position=(1, 1),
    )

    unweighted, weighted, totals = compute_empirical_next_state_condition_outputs(
        runs=[
            {
                "seed": 0,
                "run_ids": {
                    "cardinal distance:range=1": sensory_run_id,
                    "cartesian": cartesian_run_id,
                },
            }
        ],
        datasets_root=tmp_path,
        level_index=20,
        max_history_length=0,
    )

    assert np.isclose(unweighted["observation_only"]["mean"][0], 0.0)
    assert np.isclose(weighted["observation_only"]["mean"][0], 0.0)
    assert totals["num_seeds"] == 1
    assert totals["saved_rows"] == 4


def test_empirical_integral_records_cover_summary_and_history_combinations() -> None:
    summary_family = {"mean": [1.0, 1.0]}
    for percentage in range(101):
        summary_family[f"q{percentage:02d}"] = [1.0, 1.0]
    summaries = {
        "observation_only": summary_family,
        "sensorimotor": summary_family,
    }

    records = build_empirical_integral_records(
        rand_prob=0.4,
        valid_trajectories_only=True,
        unweighted_summaries=summaries,
        weighted_summaries=summaries,
    )

    assert {
        (record["rand_prob"], record["valid_trajectories_only"], record["summary_mode"], record["history_type"])
        for record in records
    } == {
        (0.4, True, "structural_unweighted", "observation_only"),
        (0.4, True, "structural_unweighted", "sensorimotor"),
        (0.4, True, "probability_weighted", "observation_only"),
        (0.4, True, "probability_weighted", "sensorimotor"),
    }
    assert all(np.isclose(record["integral"], 1.0) for record in records)


def test_empirical_near_zero_area_records_cover_summary_and_history_combinations() -> None:
    summary_family = {"mean": [0.0, 0.0]}
    for percentage in range(101):
        summary_family[f"q{percentage:02d}"] = [0.0, 0.0]
    summaries = {
        "observation_only": summary_family,
        "sensorimotor": summary_family,
    }

    records = build_empirical_near_zero_area_records(
        rand_prob=0.4,
        valid_trajectories_only=True,
        unweighted_summaries=summaries,
        weighted_summaries=summaries,
        threshold=1e-2,
    )

    assert {
        (record["rand_prob"], record["valid_trajectories_only"], record["summary_mode"], record["history_type"])
        for record in records
    } == {
        (0.4, True, "structural_unweighted", "observation_only"),
        (0.4, True, "structural_unweighted", "sensorimotor"),
        (0.4, True, "probability_weighted", "observation_only"),
        (0.4, True, "probability_weighted", "sensorimotor"),
    }
    assert all(np.isclose(record["area"], 1.0) for record in records)


def test_empirical_integral_line_plot_writes_png(tmp_path: Path) -> None:
    records = []
    for rand_prob in (0.0, 0.1):
        for valid_only in (False, True):
            for summary_mode in ("structural_unweighted", "probability_weighted"):
                for history_type in ("observation_only", "sensorimotor"):
                    records.append(
                        {
                            "rand_prob": rand_prob,
                            "valid_trajectories_only": valid_only,
                            "summary_mode": summary_mode,
                            "history_type": history_type,
                            "integral": rand_prob + int(valid_only),
                        }
                    )

    output_path = tmp_path / "integrals.png"
    plot_empirical_integral_records(
        records,
        output_path=output_path,
        title="Synthetic integrals",
    )

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_empirical_structural_integral_line_plot_writes_png(tmp_path: Path) -> None:
    records = []
    for rand_prob in (0.0, 0.1):
        for valid_only in (False, True):
            for history_type in ("observation_only", "sensorimotor"):
                records.append(
                    {
                        "rand_prob": rand_prob,
                        "valid_trajectories_only": valid_only,
                        "summary_mode": "structural_unweighted",
                        "history_type": history_type,
                        "integral": rand_prob + int(valid_only),
                    }
                )

    output_path = tmp_path / "structural_integrals.png"
    plot_empirical_structural_integral_records(
        records,
        output_path=output_path,
        title="Synthetic structural integrals",
    )

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_empirical_near_zero_area_line_plot_writes_png(tmp_path: Path) -> None:
    records = []
    for rand_prob in (0.0, 0.1):
        for valid_only in (False, True):
            for history_type in ("observation_only", "sensorimotor"):
                records.append(
                    {
                        "rand_prob": rand_prob,
                        "valid_trajectories_only": valid_only,
                        "summary_mode": "probability_weighted",
                        "history_type": history_type,
                        "area": rand_prob + int(valid_only),
                    }
                )

    output_path = tmp_path / "near_zero_area.png"
    plot_empirical_near_zero_area_records(
        records,
        output_path=output_path,
        title="Synthetic near-zero area",
    )

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_empirical_structural_near_zero_area_line_plot_writes_png(tmp_path: Path) -> None:
    records = []
    for rand_prob in (0.0, 0.1):
        for valid_only in (False, True):
            for history_type in ("observation_only", "sensorimotor"):
                records.append(
                    {
                        "rand_prob": rand_prob,
                        "valid_trajectories_only": valid_only,
                        "summary_mode": "structural_unweighted",
                        "history_type": history_type,
                        "area": rand_prob + int(valid_only),
                    }
                )

    output_path = tmp_path / "structural_near_zero_area.png"
    plot_empirical_structural_near_zero_area_records(
        records,
        output_path=output_path,
        title="Synthetic structural near-zero area",
    )

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_cross_level_empirical_integral_line_plot_writes_png(tmp_path: Path) -> None:
    records = []
    for level_index in (13, 20, 21):
        for rand_prob in (0.0, 0.1):
            for valid_only in (False, True):
                for history_type in ("observation_only", "sensorimotor"):
                    records.append(
                        {
                            "level_index": level_index,
                            "rand_prob": rand_prob,
                            "valid_trajectories_only": valid_only,
                            "summary_mode": "probability_weighted",
                            "history_type": history_type,
                            "integral": level_index + rand_prob + int(valid_only),
                        }
                    )

    output_path = tmp_path / "integrals_by_level.png"
    plot_empirical_integral_records_by_level(
        records,
        output_path=output_path,
        title="Synthetic integrals by level",
    )

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_cross_level_structural_integral_line_plot_writes_png(tmp_path: Path) -> None:
    records = []
    for level_index in (13, 20, 21):
        for rand_prob in (0.0, 0.1):
            for valid_only in (False, True):
                for history_type in ("observation_only", "sensorimotor"):
                    records.append(
                        {
                            "level_index": level_index,
                            "rand_prob": rand_prob,
                            "valid_trajectories_only": valid_only,
                            "summary_mode": "structural_unweighted",
                            "history_type": history_type,
                            "integral": level_index + rand_prob + int(valid_only),
                        }
                    )

    output_path = tmp_path / "structural_integrals_by_level.png"
    plot_empirical_structural_integral_records_by_level(
        records,
        output_path=output_path,
        title="Synthetic structural integrals by level",
    )

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_cross_level_near_zero_area_line_plot_writes_png(tmp_path: Path) -> None:
    records = []
    for level_index in (13, 20, 21):
        for rand_prob in (0.0, 0.1):
            for valid_only in (False, True):
                for history_type in ("observation_only", "sensorimotor"):
                    records.append(
                        {
                            "level_index": level_index,
                            "rand_prob": rand_prob,
                            "valid_trajectories_only": valid_only,
                            "summary_mode": "probability_weighted",
                            "history_type": history_type,
                            "area": level_index + rand_prob + int(valid_only),
                        }
                    )

    output_path = tmp_path / "near_zero_area_by_level.png"
    plot_empirical_near_zero_area_records_by_level(
        records,
        output_path=output_path,
        title="Synthetic near-zero area by level",
    )

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_cross_level_structural_near_zero_area_line_plot_writes_png(tmp_path: Path) -> None:
    records = []
    for level_index in (13, 20, 21):
        for rand_prob in (0.0, 0.1):
            for valid_only in (False, True):
                for history_type in ("observation_only", "sensorimotor"):
                    records.append(
                        {
                            "level_index": level_index,
                            "rand_prob": rand_prob,
                            "valid_trajectories_only": valid_only,
                            "summary_mode": "structural_unweighted",
                            "history_type": history_type,
                            "area": level_index + rand_prob + int(valid_only),
                        }
                    )

    output_path = tmp_path / "structural_near_zero_area_by_level.png"
    plot_empirical_structural_near_zero_area_records_by_level(
        records,
        output_path=output_path,
        title="Synthetic structural near-zero area by level",
    )

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_cross_level_empirical_integral_plot_groups_probability_weighted_records_only() -> None:
    records = [
        {
            "level_index": 13,
            "rand_prob": 0.0,
            "valid_trajectories_only": False,
            "summary_mode": "structural_unweighted",
            "history_type": "observation_only",
            "integral": 10.0,
        },
        {
            "level_index": 13,
            "rand_prob": 0.0,
            "valid_trajectories_only": False,
            "summary_mode": "probability_weighted",
            "history_type": "observation_only",
            "integral": 1.0,
        },
    ]

    grouped_records = _probability_weighted_integral_records_by_level(records)

    assert set(grouped_records) == {(13, False, "observation_only")}
    assert grouped_records[(13, False, "observation_only")][0]["integral"] == 1.0


def test_cross_level_structural_integral_plot_groups_structural_records_only() -> None:
    records = [
        {
            "level_index": 13,
            "rand_prob": 0.0,
            "valid_trajectories_only": False,
            "summary_mode": "probability_weighted",
            "history_type": "observation_only",
            "integral": 10.0,
        },
        {
            "level_index": 13,
            "rand_prob": 0.0,
            "valid_trajectories_only": False,
            "summary_mode": "structural_unweighted",
            "history_type": "observation_only",
            "integral": 1.0,
        },
    ]

    grouped_records = _structural_unweighted_integral_records_by_level(records)

    assert set(grouped_records) == {(13, False, "observation_only")}
    assert grouped_records[(13, False, "observation_only")][0]["integral"] == 1.0
