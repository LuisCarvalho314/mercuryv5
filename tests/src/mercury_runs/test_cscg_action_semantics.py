from __future__ import annotations

import numpy as np
import pytest

from src.mercury_runs.algorithms.cscg.prepare import (
    action_to_indices,
    align_post_step_actions_to_cscg,
    prepare_cscg_transition_inputs,
    to_indices,
)
from src.mercury_runs.application.evaluation_protocol import EvaluationProtocol
from src.mercury_runs.application.evaluation_service import generate_and_cache_walks
from src.mercury_runs.infrastructure import paper_precision
from src.mercury_runs.infrastructure.paper_precision import (
    action_id_from_cartesian_displacement,
    build_action_conditioned_tensor_from_walks,
    exact_cartesian_reference_positions,
    generate_random_start_walks,
    validate_evaluation_walk_action_alignment,
)
from src.mercury_runs.infrastructure.reporting import fit_transition_probs


SCRIPTED_ACTIONS = [
    "east",
    "north",
    "east",
    "south",
    "west",
    "north",
    "west",
    "south",
]


def _install_non_sticky_policy(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    commands = iter(SCRIPTED_ACTIONS)
    calls: list[str] = []

    def scripted_policy(self, previous_action: str, collision: bool, rand_prob: float) -> str:
        del self, previous_action, collision, rand_prob
        action = next(commands)
        calls.append(action)
        return action

    open_positions = paper_precision._open_positions(3)
    start_index = open_positions.index((2, 1))
    monkeypatch.setattr(
        paper_precision.random.Random,
        "randrange",
        lambda self, stop: start_index,
    )
    monkeypatch.setattr(paper_precision.MazeEnvironment, "random_policy", scripted_policy)
    return calls


def _generate_scripted_walk(monkeypatch: pytest.MonkeyPatch):
    _install_non_sticky_policy(monkeypatch)
    return generate_random_start_walks(
        level=3,
        sensor="cartesian",
        sensor_range=None,
        rand_prob=1.0,
        num_walks=1,
        walk_length=len(SCRIPTED_ACTIONS),
        base_seed=0,
        valid_trajectories_only=False,
    )[0]


def _realised_actions(positions: np.ndarray) -> np.ndarray:
    return np.asarray(
        [
            action_id_from_cartesian_displacement(positions[t], positions[t + 1])
            for t in range(positions.shape[0] - 1)
        ],
        dtype=np.int64,
    )


def test_generator_actions_are_post_step_aligned_to_realised_displacements(monkeypatch) -> None:
    walk = _generate_scripted_walk(monkeypatch)
    positions = np.asarray(walk.cartesian_observations)
    stored_actions, _ = action_to_indices(np.asarray(walk.actions))
    expected = _realised_actions(positions)

    assert not np.any(walk.collisions)
    assert validate_evaluation_walk_action_alignment(
        positions=positions,
        actions=walk.actions,
    ) == len(expected)

    # The persisted action at t + 1 caused position[t] -> position[t + 1].
    np.testing.assert_array_equal(stored_actions[1:], expected)

    # Sentinel for the old CSCG convention: this adversarial trajectory makes
    # every source-aligned action disagree with the independently derived oracle.
    assert float(np.mean(stored_actions[:-1] == expected)) == 0.0

    corrupted = np.array(walk.actions, copy=True)
    corrupted[1] = corrupted[0]
    with pytest.raises(ValueError, match="action alignment invariant failed"):
        validate_evaluation_walk_action_alignment(positions=positions, actions=corrupted)


def test_mercury_native_transition_metrics_use_the_same_post_step_semantics(monkeypatch) -> None:
    walk = _generate_scripted_walk(monkeypatch)
    stored_actions, _ = action_to_indices(np.asarray(walk.actions))
    states = np.arange(stored_actions.shape[0], dtype=np.int64)

    fitted = fit_transition_probs(states, stored_actions)

    for timestep in range(states.shape[0] - 1):
        expected_key = (int(states[timestep]), int(stored_actions[timestep + 1]))
        assert expected_key in fitted
        next_states, probabilities = fitted[expected_key]
        np.testing.assert_array_equal(next_states, [states[timestep + 1]])
        np.testing.assert_array_equal(probabilities, [1.0])

    assert (int(states[0]), int(stored_actions[0])) not in fitted


def test_mercury_and_tmgwr_evaluation_tensors_use_post_step_actions(monkeypatch) -> None:
    walk = _generate_scripted_walk(monkeypatch)
    stored_actions, _ = action_to_indices(np.asarray(walk.actions))
    states = np.arange(stored_actions.shape[0], dtype=np.int64)

    tensor = build_action_conditioned_tensor_from_walks(
        state_walks=[states],
        action_walks=[stored_actions],
        n_states=states.shape[0],
        n_actions=4,
    )

    for timestep in range(states.shape[0] - 1):
        correct_action = int(stored_actions[timestep + 1])
        old_wrong_action = int(stored_actions[timestep])
        assert tensor[correct_action, timestep, timestep + 1] == 1.0
        assert tensor[old_wrong_action, timestep, timestep + 1] == 0.0


def test_cscg_adapter_actions_match_realised_transitions(monkeypatch) -> None:
    walk = _generate_scripted_walk(monkeypatch)
    positions = np.asarray(walk.cartesian_observations)
    observation_ids, _ = to_indices(np.asarray(walk.observations))
    stored_actions, _ = action_to_indices(np.asarray(walk.actions))

    source_x, transition_actions, target_x = prepare_cscg_transition_inputs(
        observation_ids,
        stored_actions,
    )
    expected = _realised_actions(positions)

    assert len(transition_actions) == len(observation_ids) - 1
    np.testing.assert_array_equal(source_x, observation_ids[:-1])
    np.testing.assert_array_equal(target_x, observation_ids[1:])
    np.testing.assert_array_equal(transition_actions, expected)

    # This is the same-length representation passed to the vendored CSCG API;
    # CSCG ignores its final action entry.
    cscg_actions = align_post_step_actions_to_cscg(stored_actions)
    np.testing.assert_array_equal(cscg_actions[:-1], expected)

    purity_by_persisted_offset = {
        +1: float(np.mean(stored_actions[1:] == expected)),
        0: float(np.mean(stored_actions[:-1] == expected)),
        -1: float(np.mean(stored_actions[:-2] == expected[1:])),
    }
    assert purity_by_persisted_offset[+1] == 1.0
    assert purity_by_persisted_offset[0] < 1.0
    assert purity_by_persisted_offset[-1] < 1.0


def test_generator_cache_loader_and_cscg_adapter_preserve_action_semantics(
    monkeypatch,
    tmp_path,
) -> None:
    policy_calls = _install_non_sticky_policy(monkeypatch)
    protocol = EvaluationProtocol(
        sensor="cartesian",
        sensor_range=None,
        rand_prob=1.0,
        evaluation_seed=0,
        number_of_walks=1,
        walk_length=len(SCRIPTED_ACTIONS),
        valid_trajectories_only=False,
    )

    generated = generate_and_cache_walks(
        study_root=tmp_path,
        protocol=protocol,
        level=3,
    )
    assert policy_calls == SCRIPTED_ACTIONS

    # A second call must traverse the production cache loader, not regenerate.
    loaded = generate_and_cache_walks(
        study_root=tmp_path,
        protocol=protocol,
        level=3,
    )
    assert policy_calls == SCRIPTED_ACTIONS
    np.testing.assert_array_equal(
        loaded["filtered_actions_per_walk"][0],
        generated["filtered_actions_per_walk"][0],
    )

    observations = np.asarray(loaded["filtered_obs_per_walk"][0])
    stored_action_vectors = np.asarray(loaded["filtered_actions_per_walk"][0])
    stored_actions, _ = action_to_indices(stored_action_vectors)
    positions = exact_cartesian_reference_positions(3)[loaded["gt_exact_per_walk"][0]]
    observation_ids, _ = to_indices(observations)
    _, cscg_transition_actions, _ = prepare_cscg_transition_inputs(
        observation_ids,
        stored_actions,
    )
    expected = _realised_actions(positions)

    np.testing.assert_array_equal(cscg_transition_actions, expected)
    np.testing.assert_array_equal(
        align_post_step_actions_to_cscg(stored_actions)[:-1],
        expected,
    )
    assert not np.array_equal(stored_actions[:-1], expected)
