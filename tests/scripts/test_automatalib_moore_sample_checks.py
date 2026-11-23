from __future__ import annotations

import numpy as np
import pytest

from scripts.analysis.automatalib_moore_audit.check_samples import (
    check_dataset_arrays,
    check_evaluation_walk,
)


ORACLE = {
    "physical_state_count": 1,
    "outputs": [[0, 0, 0, 0]],
    "transitions": [[0, action, 0] for action in range(4)],
}


def test_dataset_coverage_requires_all_actions():
    actions = np.eye(4, dtype=np.int64)
    shared = dict(initial_state=0, observations=np.zeros((4, 4), dtype=np.int64),
                  collisions=np.ones(4, dtype=bool), positions=[(1, 1)], oracle=ORACLE)
    full = check_dataset_arrays(vectors=actions, **shared)
    assert full["complete"]
    assert full["_sample_machine"] == ORACLE
    partial = check_dataset_arrays(vectors=actions[[0, 0, 1, 2]], **shared)
    assert not partial["complete"]
    assert partial["_sample_machine"] is None
    assert partial["missing_state_action_pairs"] == [[0, 3]]


def test_dataset_replay_rejects_wrong_observation():
    observations = np.zeros((4, 4), dtype=np.int64)
    observations[2, 0] = 1
    with pytest.raises(ValueError, match="observations disagree"):
        check_dataset_arrays(
            initial_state=0, observations=observations,
            vectors=np.eye(4, dtype=np.int64), collisions=np.ones(4, dtype=bool),
            positions=[(1, 1)], oracle=ORACLE,
        )


def test_evaluation_walk_checks_all_rows_but_omits_unknown_first_transition():
    count = 10_000
    actions = np.arange(count) % 4
    prefix = "walk_00000"
    cached = {
        f"{prefix}_valid": np.array(True),
        f"{prefix}_obs": np.zeros((count, 4), dtype=np.int64),
        f"{prefix}_act": np.eye(4, dtype=np.int64)[actions],
        f"{prefix}_gt_exact": np.zeros(count, dtype=np.int64),
        f"{prefix}_gt_action": actions,
        f"{prefix}_collisions": np.ones(count, dtype=bool),
    }
    cached[f"{prefix}_collisions"][0] = False  # no cached source for row zero
    result = check_evaluation_walk(cached, 0, ORACLE)
    assert result["complete"]
    assert result["_sample_machine"] == ORACLE
    cached[f"{prefix}_collisions"][1] = False
    with pytest.raises(ValueError, match="collision flags"):
        check_evaluation_walk(cached, 0, ORACLE)
