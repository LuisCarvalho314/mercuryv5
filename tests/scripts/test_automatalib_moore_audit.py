from __future__ import annotations

import numpy as np
import pytest

from scripts.analysis.automatalib_moore_audit import run as audit


def _cache(tmp_path, *, actions=(0, 1, 2, 3), outputs=None, collisions=None):
    walks = tmp_path / "walks"
    walks.mkdir()
    action_ids = np.asarray((0, *actions), dtype=np.int64)
    vectors = np.eye(4, dtype=np.int64)[action_ids]
    if outputs is None:
        outputs = np.zeros((len(action_ids), 4), dtype=np.int64)
    if collisions is None:
        collisions = np.ones(len(action_ids), dtype=bool)
    values = {"n_walks": np.array(3, dtype=np.int64)}
    for walk in range(3):
        prefix = f"walk_{walk:05d}"
        values.update({
            f"{prefix}_valid": np.array(True),
            f"{prefix}_obs": outputs,
            f"{prefix}_act": vectors,
            f"{prefix}_gt_exact": np.zeros(len(action_ids), dtype=np.int64),
            f"{prefix}_gt_action": action_ids,
            f"{prefix}_collisions": collisions,
        })
    np.savez_compressed(
        walks / "level=13_sensor=cardinal distance_range=1_rand=0.3_eval-seed=0.npz",
        **values,
    )


def test_extract_accepts_complete_deterministic_collision_table(tmp_path, monkeypatch):
    monkeypatch.setattr(audit, "_reference_positions", lambda level: [(1, 1)])
    _cache(tmp_path)
    machine = audit.extract_machine(13, tmp_path)
    assert machine["physical_state_count"] == 1
    assert machine["state_action_pair_count"] == 4
    assert machine["transitions"] == [[0, action, 0] for action in range(4)]


def test_extract_rejects_missing_state_action_pair(tmp_path, monkeypatch):
    monkeypatch.setattr(audit, "_reference_positions", lambda level: [(1, 1)])
    _cache(tmp_path, actions=(0, 1, 2))
    with pytest.raises(ValueError, match="Missing state-action pairs"):
        audit.extract_machine(13, tmp_path)


def test_extract_rejects_conflicting_state_output(tmp_path, monkeypatch):
    monkeypatch.setattr(audit, "_reference_positions", lambda level: [(1, 1)])
    outputs = np.zeros((5, 4), dtype=np.int64)
    outputs[-1, 0] = 1
    _cache(tmp_path, outputs=outputs)
    with pytest.raises(ValueError, match="Conflicting observations"):
        audit.extract_machine(13, tmp_path)


def test_extract_rejects_collision_alignment_error(tmp_path, monkeypatch):
    monkeypatch.setattr(audit, "_reference_positions", lambda level: [(1, 1)])
    collisions = np.ones(5, dtype=bool)
    collisions[2] = False
    _cache(tmp_path, collisions=collisions)
    with pytest.raises(ValueError, match="Collision/transition mismatch"):
        audit.extract_machine(13, tmp_path)


def test_characterizing_set_replay_handles_empty_word_and_equivalent_states():
    machine = {
        "physical_state_count": 4,
        "outputs": [[0], [1], [1], [0]],
        "transitions": [[s, a, (1, 1, 2, 3)[s]] for s in range(4) for a in range(4)],
    }
    java = "\n".join([
        "P\t4\t3", "B\t0", "B\t1,2", "B\t3",
        "W\t0\t3\t0\t0;1\t0;0",
        "C\t1\t0", "C\t0\t",  # epsilon is a real word, including its trailing tab
    ]) + "\n"
    result = audit._parse_java(java, machine)
    assert result["characterizing_set"] == [
        {"actions": [], "length": 0}, {"actions": [0], "length": 1},
    ]
    assert not result["characterizing_set_distinguishes_all_distinct_physical_states"]
    assert next(pair for pair in result["characterizing_set_pair_coverage"]
                if pair["states"] == [1, 2])["distinguishing_word_indices"] == []


def test_characterizing_set_rejects_missing_alias_witness():
    machine = {
        "physical_state_count": 4,
        "outputs": [[0], [1], [1], [0]],
        "transitions": [[s, a, (1, 1, 2, 3)[s]] for s in range(4) for a in range(4)],
    }
    java = "\n".join([
        "P\t4\t3", "B\t0", "B\t1,2", "B\t3",
        "W\t0\t3\t0\t0;1\t0;0", "C\t0\t",
    ]) + "\n"
    with pytest.raises(ValueError, match="Characterizing set disagrees"):
        audit._parse_java(java, machine)
