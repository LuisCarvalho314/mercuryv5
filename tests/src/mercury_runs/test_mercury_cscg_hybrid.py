from __future__ import annotations

from mercury_runs.algorithms.mercury_cscg_hybrid.api import _consume_crossed_checkpoints


def test_online_checkpoint_scheduler_consumes_multiple_crossed_targets_once() -> None:
    checkpoints = [10, 25, 50, 75]

    next_index, should_record = _consume_crossed_checkpoints(checkpoints, 0, 40)

    assert should_record is True
    assert next_index == 2


def test_online_checkpoint_scheduler_waits_until_next_target_is_crossed() -> None:
    checkpoints = [10, 25, 50, 75]

    next_index, should_record = _consume_crossed_checkpoints(checkpoints, 2, 40)

    assert should_record is False
    assert next_index == 2
