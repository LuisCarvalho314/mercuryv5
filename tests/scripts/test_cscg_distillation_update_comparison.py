"""Checks for the paired summaries used in the two/five-update experiment."""
import pandas as pd
import pytest
import numpy as np

from scripts.run_cscg_distillation_experiment import SUMMARY_METRICS, checkpoint_schedule, ground_truth_clone_allocation, summarise_rows


def test_longer_horizon_retains_checkpoint_count_and_scales_spacing():
    original = checkpoint_schedule(2000)
    extended = checkpoint_schedule(20000)
    assert original == [0, 40, *range(100, 2001, 100)]
    assert len(original) == len(extended) == 22
    assert extended == [10 * step for step in original]
    assert extended[-1] == 20000


def test_summary_integrates_on_observation_axis_and_keeps_seed_pairing():
    rows = []
    for seed in [0, 1]:
        for step in [0, 40, 100, 2000]:
            row = dict(level=21, seed=seed, method="predictive", step=step,
                       train_seconds=step / 1000, work_entries=step * 20)
            row.update({metric: seed + step / 2000 for metric in SUMMARY_METRICS})
            rows.append(row)
    summary = summarise_rows(pd.DataFrame(rows).sample(frac=1, random_state=1)).set_index("seed")
    for seed in [0, 1]:
        for metric in SUMMARY_METRICS:
            assert summary.loc[seed, "final_" + metric] == pytest.approx(seed + 1)
            assert summary.loc[seed, "auc_" + metric] == pytest.approx(seed + .5)
        assert summary.loc[seed, "train_seconds"] == 2
        assert summary.loc[seed, "work_entries"] == 40000


@pytest.mark.parametrize("steps", [[0, 40, 40, 2000], [40, 100, 2000], [0]])
def test_summary_rejects_duplicate_or_incomplete_checkpoint_grids(steps):
    rows = []
    for step in steps:
        row = dict(level=21, seed=0, method="long", step=step, train_seconds=1., work_entries=1)
        row.update({metric: .5 for metric in SUMMARY_METRICS})
        rows.append(row)
    with pytest.raises(ValueError, match="checkpoints"):
        summarise_rows(pd.DataFrame(rows))


@pytest.mark.parametrize("level, expected", [(21, [1, 1, 2]), (22, [1, 1, 2, 1, 1, 1, 2, 1, 2]),
                                             (23, [1, 1, 6, 1, 3, 1, 6, 3, 18])])
def test_ground_truth_allocation_counts_states_per_symbol_and_respects_vocabulary_order(level, expected):
    from mercury_runs.infrastructure.paper_precision import _init_env, _set_start_position, exact_cartesian_reference_positions
    positions = exact_cartesian_reference_positions(level)
    environment = _init_env(level_index=level, sensor="cardinal distance", sensor_range=1, seed=0)
    observations = []
    for position in positions:
        _set_start_position(environment, tuple(map(int, position)))
        observations.append(np.asarray(environment.agent.observation).copy())
    vocabulary = np.unique(observations, axis=0)
    allocation = ground_truth_clone_allocation(level, vocabulary)
    np.testing.assert_array_equal(allocation, expected)
    assert allocation.sum() == len(positions)
    np.testing.assert_array_equal(ground_truth_clone_allocation(level, vocabulary[::-1]), allocation[::-1])
