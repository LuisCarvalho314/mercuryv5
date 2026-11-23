"""Tests for the canonical transition structure evaluator.

Covers acceptance criteria 22-24 from the First-Class Snapshot Evaluation
Architecture and the mathematical primitives that support it.
"""
from __future__ import annotations

import numpy as np
import pytest

from mercury_runs.infrastructure.paper_precision import (
    CanonicalTransitionEvaluationResult,
    compute_action_projection_weights,
    compute_collapsed_distributions_canonical,
    compute_hard_alignment,
    compute_non_selfloop_action_tv,
    compute_occupancy_weighted_action_tv,
    compute_transition_statistics,
    evaluate_transition_structure_canonical,
    project_transition_tensor_canonical,
    complete_and_normalize_projected_rows,
    _action_tv_by_pair,
)


# ---------------------------------------------------------------------------
# compute_hard_alignment
# ---------------------------------------------------------------------------


def test_hard_alignment_returns_minus_one_for_zero_rows() -> None:
    C = np.array([[0, 0], [3, 1], [0, 0]], dtype=np.int64)
    alignment = compute_hard_alignment(C)
    assert int(alignment[0]) == -1
    assert int(alignment[1]) == 0  # argmax([3, 1]) = 0
    assert int(alignment[2]) == -1


def test_hard_alignment_non_zero_row_uses_argmax() -> None:
    C = np.array([[2, 5, 1]], dtype=np.int64)
    alignment = compute_hard_alignment(C)
    assert int(alignment[0]) == 1  # argmax([2, 5, 1]) = 1


def test_hard_alignment_empty_true_states_returns_all_minus_one() -> None:
    C = np.zeros((3, 0), dtype=np.int64)
    alignment = compute_hard_alignment(C)
    assert (alignment == -1).all()


# ---------------------------------------------------------------------------
# compute_transition_statistics
# ---------------------------------------------------------------------------


def test_compute_transition_statistics_total_cooccurrence() -> None:
    # Walk: decoded [0, 1, 0], true [0, 0, 1], actions [0, 1, 0]
    C, H, N_ai = compute_transition_statistics(
        decoded_walks=[np.array([0, 1, 0], dtype=np.int64)],
        true_walks=[np.array([0, 0, 1], dtype=np.int64)],
        action_walks=[np.array([0, 1, 0], dtype=np.int64)],
        n_learned=2,
        n_true=2,
        n_actions=2,
    )
    # t=0: decoded=0, true=0 → C[0,0]++
    # t=1: decoded=1, true=0 → C[1,0]++
    # t=2: decoded=0, true=1 → C[0,1]++
    assert int(C[0, 0]) == 1
    assert int(C[1, 0]) == 1
    assert int(C[0, 1]) == 1


def test_compute_transition_statistics_action_source_cooccurrence() -> None:
    # Walk of length 3, post-step action convention.
    # States: decoded=[0,0,1], true=[0,1,0], actions=[_, 0, 1] (post-step)
    # Transition t=0→1: source=(decoded=0,true=0), action=actions[1]=0
    # Transition t=1→2: source=(decoded=0,true=1), action=actions[2]=1
    _, H, N_ai = compute_transition_statistics(
        decoded_walks=[np.array([0, 0, 1], dtype=np.int64)],
        true_walks=[np.array([0, 1, 0], dtype=np.int64)],
        action_walks=[np.array([99, 0, 1], dtype=np.int64)],  # [0] ignored
        n_learned=2,
        n_true=2,
        n_actions=2,
    )
    # H[0, 0, 0] should be 1 (action=0, learned_src=0, true_src=0)
    assert int(H[0, 0, 0]) == 1
    # H[1, 0, 1] should be 1 (action=1, learned_src=0, true_src=1)
    assert int(H[1, 0, 1]) == 1
    # N_ai[0, 0] = 1, N_ai[1, 1] = 1
    assert int(N_ai[0, 0]) == 1
    assert int(N_ai[1, 1]) == 1


# ---------------------------------------------------------------------------
# Acceptance criterion 22: occupancy weights cannot be influenced by the model
# ---------------------------------------------------------------------------


def test_acceptance_22_occupancy_weights_from_true_counts_not_model() -> None:
    """Occupancy weights must be derived from N_ai (true evaluation counts).

    Two models with different decoded states but the same true state-action
    counts must produce identical occupancy weights.
    """
    # Ground truth: 2-state, 2-action system.
    T_true = np.array(
        [
            [[0.0, 1.0], [1.0, 0.0]],  # action 0
            [[1.0, 0.0], [0.0, 1.0]],  # action 1
        ],
        dtype=np.float64,
    )
    n_true = 2

    # N_ai: fixed evaluation counts (same for both models)
    N_ai = np.array([[10, 5], [5, 10]], dtype=np.int64)

    # Model A: 2 learned states, perfectly aligned.
    T_hat_A = T_true.copy()
    C_A = np.eye(2, dtype=np.int64) * 10
    H_A = np.zeros((2, 2, 2), dtype=np.int64)
    H_A[0, 0, 0] = 10
    H_A[0, 1, 1] = 5
    H_A[1, 0, 0] = 5
    H_A[1, 1, 1] = 10

    result_A = evaluate_transition_structure_canonical(
        T_hat=T_hat_A,
        T_true=T_true,
        cooccurrence=C_A,
        action_source_cooccurrence=H_A,
        source_action_counts=N_ai,
    )

    # Model B: 4 learned states (more clones), different decoded distribution.
    T_hat_B = np.zeros((2, 4, 4), dtype=np.float64)
    # Clone 0 and 1 both align to true state 0.
    # Clone 2 and 3 both align to true state 1.
    T_hat_B[0, 0, 2] = 1.0  # clone 0, action 0 → clone 2 (true state 1)
    T_hat_B[0, 1, 2] = 1.0  # clone 1, action 0 → clone 2 (true state 1)
    T_hat_B[0, 2, 0] = 1.0  # clone 2, action 0 → clone 0 (true state 0)
    T_hat_B[0, 3, 0] = 1.0  # clone 3, action 0 → clone 0 (true state 0)
    T_hat_B[1, 0, 0] = 1.0  # clone 0, action 1 → clone 0
    T_hat_B[1, 1, 0] = 1.0
    T_hat_B[1, 2, 2] = 1.0
    T_hat_B[1, 3, 2] = 1.0
    C_B = np.zeros((4, 2), dtype=np.int64)
    C_B[0, 0] = 5
    C_B[1, 0] = 5
    C_B[2, 1] = 5
    C_B[3, 1] = 5
    H_B = np.zeros((2, 4, 2), dtype=np.int64)
    H_B[0, 0, 0] = 5
    H_B[0, 1, 0] = 5
    H_B[0, 2, 1] = 3
    H_B[0, 3, 1] = 2
    H_B[1, 0, 0] = 3
    H_B[1, 1, 0] = 2
    H_B[1, 2, 1] = 5
    H_B[1, 3, 1] = 5

    result_B = evaluate_transition_structure_canonical(
        T_hat=T_hat_B,
        T_true=T_true,
        cooccurrence=C_B,
        action_source_cooccurrence=H_B,
        source_action_counts=N_ai,
    )

    # Key assertion: both models see the same occupancy weights because
    # N_ai is fixed and independent of model output.
    np.testing.assert_allclose(
        result_A.evaluation_occupancy_weights,
        result_B.evaluation_occupancy_weights,
        atol=1e-12,
        err_msg="Occupancy weights must depend only on N_ai, not on decoded states",
    )


# ---------------------------------------------------------------------------
# Acceptance criterion 23: occupancy-weighted and macro TV differ on imbalanced walks
# ---------------------------------------------------------------------------


def test_acceptance_23_occupancy_weighted_and_macro_tv_differ_on_imbalanced_walks() -> None:
    """Occupancy-weighted and macro TV must differ when one row dominates.

    Construct a case where:
    - Row (a=0, i=0) is visited 99 times and has TV = 0.0 (perfect prediction).
    - Row (a=0, i=1) is visited 1 time  and has TV = 1.0 (complete mismatch).

    Macro TV = mean([0.0, 1.0]) = 0.5.
    Occupancy-weighted TV = (99/100)*0.0 + (1/100)*1.0 = 0.01.
    """
    T_true = np.array(
        [[[1.0, 0.0], [0.0, 1.0]]],
        dtype=np.float64,
    )
    T_hat = np.array(
        [[[1.0, 0.0], [1.0, 0.0]]],  # row i=1 is completely wrong
        dtype=np.float64,
    )
    n_true = 2
    n_learned = 2
    n_actions = 1
    C = np.eye(2, dtype=np.int64) * 10
    H = np.zeros((n_actions, n_learned, n_true), dtype=np.int64)
    H[0, 0, 0] = 99
    H[0, 1, 1] = 1
    N_ai = np.array([[99, 1]], dtype=np.int64)

    result = evaluate_transition_structure_canonical(
        T_hat=T_hat,
        T_true=T_true,
        cooccurrence=C,
        action_source_cooccurrence=H,
        source_action_counts=N_ai,
    )

    assert result.tv_occupancy == pytest.approx(0.01, abs=1e-9)
    assert result.tv_macro == pytest.approx(0.5, abs=1e-9)
    assert result.tv_occupancy < result.tv_macro


# ---------------------------------------------------------------------------
# Acceptance criterion 24: action-conditioned TV >= policy-collapsed TV
# ---------------------------------------------------------------------------


def test_acceptance_24_action_conditioned_tv_no_smaller_than_collapsed_tv() -> None:
    """TV(sum_a q_a P_a, sum_a q_a P_hat_a) <= sum_a q_a TV(P_a, P_hat_a).

    Construct a cancellation case:
    - Action 0: predicted distributes mass to the RIGHT  (P_hat != P_GT).
    - Action 1: predicted distributes mass to the LEFT   (P_hat != P_GT, opposite error).
    - Uniform action prior: q(0|0) = q(1|0) = 0.5.
    - After marginalising, errors cancel → collapsed TV = 0.
    - But action-conditioned TV = mean of per-action TVs > 0.
    """
    # 2 true states (i=0 source, j=0..1 targets), 2 learned states, 2 actions.
    # T_true: source=0 transitions uniformly to both targets under each action.
    n_true = 2
    n_learned = 2
    n_actions = 2
    T_true = np.array(
        [
            [[0.5, 0.5], [0.0, 0.0]],   # action 0: source 0 uniform; source 1 unused
            [[0.5, 0.5], [0.0, 0.0]],   # action 1: source 0 uniform; source 1 unused
        ],
        dtype=np.float64,
    )
    # Predicted: action 0 always goes right (j=1), action 1 always goes left (j=0).
    # Perfect alignment: learned state k maps to true state i=k.
    T_hat = np.array(
        [
            [[0.0, 1.0], [0.0, 0.0]],   # action 0
            [[1.0, 0.0], [0.0, 0.0]],   # action 1
        ],
        dtype=np.float64,
    )
    # Alignment: k=0 -> i=0, k=1 -> i=1 (both aligned).
    C = np.array([[20, 0], [0, 20]], dtype=np.int64)
    H = np.zeros((n_actions, n_learned, n_true), dtype=np.int64)
    H[0, 0, 0] = 10  # action 0: learned k=0, true i=0
    H[1, 0, 0] = 10  # action 1: learned k=0, true i=0
    N_ai = np.array([[10, 0], [10, 0]], dtype=np.int64)

    result = evaluate_transition_structure_canonical(
        T_hat=T_hat,
        T_true=T_true,
        cooccurrence=C,
        action_source_cooccurrence=H,
        source_action_counts=N_ai,
    )

    # After uniform marginalisation, collapsed predicted = 0.5 * T_hat[0] + 0.5 * T_hat[1]
    # = 0.5 * [0, 1] + 0.5 * [1, 0] = [0.5, 0.5] == GT.  Collapsed TV = 0.
    assert result.collapsed_unif_tv_occupancy == pytest.approx(0.0, abs=1e-9)

    # Action-conditioned: TV(a=0, i=0) = 0.5 * (|0-0.5| + |1-0.5|) = 0.5
    #                     TV(a=1, i=0) = 0.5 * (|1-0.5| + |0-0.5|) = 0.5
    # Occupancy-weighted action TV = 0.5 (uniform N_ai).
    assert result.tv_occupancy == pytest.approx(0.5, abs=1e-9)
    assert result.tv_occupancy > result.collapsed_unif_tv_occupancy


# ---------------------------------------------------------------------------
# Additional canonical evaluator tests
# ---------------------------------------------------------------------------


def test_perfect_prediction_gives_zero_tv() -> None:
    T_true = np.array([[[0.0, 1.0], [1.0, 0.0]]], dtype=np.float64)
    T_hat = T_true.copy()
    C = np.eye(2, dtype=np.int64) * 5
    H = np.zeros((1, 2, 2), dtype=np.int64)
    H[0, 0, 0] = 5
    H[0, 1, 1] = 5
    N_ai = np.array([[5, 5]], dtype=np.int64)

    result = evaluate_transition_structure_canonical(
        T_hat=T_hat,
        T_true=T_true,
        cooccurrence=C,
        action_source_cooccurrence=H,
        source_action_counts=N_ai,
    )

    assert result.tv_occupancy == pytest.approx(0.0, abs=1e-9)
    assert result.tv_macro == pytest.approx(0.0, abs=1e-9)
    assert result.total_unmatched_target_mass == pytest.approx(0.0, abs=1e-9)


def test_unaligned_learned_states_are_recorded_with_minus_one() -> None:
    # 3 learned states: only k=1 has evaluation evidence.
    C = np.array([[0, 0], [5, 0], [0, 0]], dtype=np.int64)
    alignment = compute_hard_alignment(C)
    assert int(alignment[0]) == -1
    assert int(alignment[1]) == 0
    assert int(alignment[2]) == -1


def test_unmatched_target_mass_when_learned_target_is_unaligned() -> None:
    # 2 learned states: k=0 aligned to true 0, k=1 unaligned.
    # T_hat: k=0 sends all mass to k=1 (unaligned) → all goes to ⊥.
    # n_true=2: state 0 is the valid source, state 1 unused as source.
    T_true = np.array([[[1.0, 0.0], [0.0, 0.0]]], dtype=np.float64)   # (1, 2, 2)
    T_hat = np.array([[[0.0, 1.0], [0.0, 0.0]]], dtype=np.float64)    # (1, 2, 2)
    # C: k=0 → true 0 (aligned), k=1 never observed (unaligned=-1).
    C = np.array([[5, 0], [0, 0]], dtype=np.int64)
    H = np.zeros((1, 2, 2), dtype=np.int64)
    H[0, 0, 0] = 5
    N_ai = np.array([[5, 0]], dtype=np.int64)

    result = evaluate_transition_structure_canonical(
        T_hat=T_hat,
        T_true=T_true,
        cooccurrence=C,
        action_source_cooccurrence=H,
        source_action_counts=N_ai,
    )

    # All predicted mass goes to ⊥: TV should be 1.
    assert result.tv_occupancy == pytest.approx(1.0, abs=1e-9)
    assert result.total_unmatched_target_mass == pytest.approx(1.0, abs=1e-9)


def test_row_with_no_predicted_mass_gets_all_to_sink() -> None:
    # True row is valid but model has no learned state aligned to this true state.
    T_true = np.array([[[0.0, 1.0], [1.0, 0.0]]], dtype=np.float64)
    T_hat = np.array([[[0.0, 1.0], [1.0, 0.0]]], dtype=np.float64)
    # Only k=0 aligned to true 0; no clone aligned to true state 1.
    C = np.array([[5, 0]], dtype=np.int64)
    H = np.zeros((1, 1, 2), dtype=np.int64)
    H[0, 0, 0] = 5
    N_ai = np.array([[5, 5]], dtype=np.int64)

    result = evaluate_transition_structure_canonical(
        T_hat=T_hat,
        T_true=T_true,
        cooccurrence=C,
        action_source_cooccurrence=H,
        source_action_counts=N_ai,
    )

    # True state 1 has no aligned learned clone → omega=0 → projected row is all zero
    # → row completion assigns all to ⊥ → TV = 1 for that row.
    assert result.tv_by_pair_full[0, 1] == pytest.approx(1.0, abs=1e-9)
    assert result.P_hat_full[0, 1, -1] == pytest.approx(1.0, abs=1e-9)


def test_non_self_loop_tv_is_zero_for_deterministic_non_self_loop_prediction() -> None:
    # T_true: state 0 → state 1 (no self-loop in GT).
    # T_hat:  state 0 → state 1 (perfect match).
    T_true = np.array([[[0.0, 1.0], [1.0, 0.0]]], dtype=np.float64)
    T_hat = T_true.copy()
    C = np.eye(2, dtype=np.int64) * 5
    H = np.zeros((1, 2, 2), dtype=np.int64)
    H[0, 0, 0] = 5
    H[0, 1, 1] = 5
    N_ai = np.array([[5, 5]], dtype=np.int64)

    result = evaluate_transition_structure_canonical(
        T_hat=T_hat,
        T_true=T_true,
        cooccurrence=C,
        action_source_cooccurrence=H,
        source_action_counts=N_ai,
    )

    assert result.tv_by_pair_non_self_loop[0, 0] == pytest.approx(0.0, abs=1e-9)
    assert result.tv_non_self_loop_occupancy == pytest.approx(0.0, abs=1e-9)
    assert result.tv_non_self_loop_macro == pytest.approx(0.0, abs=1e-9)
    assert result.collapsed_eval_tv_non_self_loop_by_state[0] == pytest.approx(0.0, abs=1e-9)
    assert result.collapsed_eval_tv_non_self_loop_occupancy == pytest.approx(0.0, abs=1e-9)
    assert result.collapsed_eval_tv_non_self_loop_macro == pytest.approx(0.0, abs=1e-9)


def test_non_self_loop_tv_is_one_when_all_predicted_mass_is_self_loop() -> None:
    # GT: state 0 goes to state 1 (no self-loop).
    # Predicted: state 0 goes to itself (complete self-loop).
    T_true = np.array([[[0.0, 1.0], [1.0, 0.0]]], dtype=np.float64)
    T_hat = np.array([[[1.0, 0.0], [1.0, 0.0]]], dtype=np.float64)
    C = np.eye(2, dtype=np.int64) * 5
    H = np.zeros((1, 2, 2), dtype=np.int64)
    H[0, 0, 0] = 5
    H[0, 1, 1] = 5
    N_ai = np.array([[5, 5]], dtype=np.int64)

    result = evaluate_transition_structure_canonical(
        T_hat=T_hat,
        T_true=T_true,
        cooccurrence=C,
        action_source_cooccurrence=H,
        source_action_counts=N_ai,
    )

    # Non-self-loop row for i=0: predicted self-loop only → all to ⊥ → TV=1.
    assert result.tv_by_pair_non_self_loop[0, 0] == pytest.approx(1.0, abs=1e-9)
    assert result.tv_non_self_loop_occupancy == pytest.approx(0.5, abs=1e-9)
    assert result.tv_non_self_loop_macro == pytest.approx(0.5, abs=1e-9)
    assert result.collapsed_eval_tv_non_self_loop_by_state[0] == pytest.approx(1.0, abs=1e-9)
    assert result.collapsed_eval_tv_non_self_loop_occupancy == pytest.approx(0.5, abs=1e-9)
    assert result.collapsed_eval_tv_non_self_loop_macro == pytest.approx(0.5, abs=1e-9)


def test_non_self_loop_aggregation_excludes_self_loop_only_rows() -> None:
    """Wall-collision occupancy must not dilute conditional movement error."""
    T_true = np.zeros((2, 2, 2), dtype=np.float64)
    T_hat = np.zeros_like(T_true)
    T_true[0, 0, 1] = 1.0       # the only movement-valid row
    T_hat[0, 0, 0] = 1.0        # completely wrong: predicts only self
    T_true[1, 0, 0] = 1.0       # high-occupancy wall collision
    T_hat[1, 0, 0] = 1.0
    C = np.eye(2, dtype=np.int64)
    H = np.zeros((2, 2, 2), dtype=np.int64)
    H[0, 0, 0] = 1
    H[1, 0, 0] = 99
    N_ai = np.array([[1, 0], [99, 0]], dtype=np.int64)

    result = evaluate_transition_structure_canonical(
        T_hat=T_hat,
        T_true=T_true,
        cooccurrence=C,
        action_source_cooccurrence=H,
        source_action_counts=N_ai,
    )

    assert result.valid_non_self_loop_rows.tolist() == [[True, False], [False, False]]
    assert result.tv_by_pair_non_self_loop[0, 0] == pytest.approx(1.0)
    assert result.tv_non_self_loop_occupancy == pytest.approx(1.0)
    assert result.tv_non_self_loop_macro == pytest.approx(1.0)
    assert result.tv_non_self_loop_by_state_occupancy[0] == pytest.approx(1.0)
    assert result.collapsed_eval_tv_non_self_loop_occupancy == pytest.approx(1.0)
    assert result.collapsed_eval_tv_non_self_loop_macro == pytest.approx(1.0)


def test_collapsed_eval_policy_uses_same_prior_for_truth_and_prediction() -> None:
    """Truth and prediction must use the same q(a|i) prior when collapsed."""
    T_true = np.array(
        [
            [[0.0, 1.0], [1.0, 0.0]],  # action 0
            [[1.0, 0.0], [0.0, 1.0]],  # action 1
        ],
        dtype=np.float64,
    )
    # Perfect prediction.
    T_hat = T_true.copy()
    C = np.eye(2, dtype=np.int64) * 10
    H = np.zeros((2, 2, 2), dtype=np.int64)
    H[0, 0, 0] = 8
    H[0, 1, 1] = 8
    H[1, 0, 0] = 2
    H[1, 1, 1] = 2
    N_ai = np.array([[8, 8], [2, 2]], dtype=np.int64)

    result = evaluate_transition_structure_canonical(
        T_hat=T_hat,
        T_true=T_true,
        cooccurrence=C,
        action_source_cooccurrence=H,
        source_action_counts=N_ai,
    )

    # Perfect prediction → collapsed TV should be zero regardless of prior.
    assert result.collapsed_eval_tv_occupancy == pytest.approx(0.0, abs=1e-9)
    assert result.collapsed_unif_tv_occupancy == pytest.approx(0.0, abs=1e-9)

    # Verify evaluation policy weights match N_ai proportions.
    # q_eval(a=0 | i=0) = N_ai[0,0] / (N_ai[0,0] + N_ai[1,0]) = 8/10 = 0.8
    assert result.action_weights_evaluation_policy[0, 0] == pytest.approx(0.8, abs=1e-9)
    assert result.action_weights_evaluation_policy[1, 0] == pytest.approx(0.2, abs=1e-9)


def test_primary_tv_includes_self_loops() -> None:
    """Self-loops must be penalised in the primary TV (not removed before comparison)."""
    # GT: state 0 can only self-loop under action 0.
    # Predicted: state 0 goes to state 1 under action 0 (no self-loop).
    T_true = np.array([[[1.0, 0.0], [0.0, 1.0]]], dtype=np.float64)
    T_hat = np.array([[[0.0, 1.0], [0.0, 1.0]]], dtype=np.float64)
    C = np.eye(2, dtype=np.int64) * 5
    H = np.zeros((1, 2, 2), dtype=np.int64)
    H[0, 0, 0] = 5
    H[0, 1, 1] = 5
    N_ai = np.array([[5, 5]], dtype=np.int64)

    result = evaluate_transition_structure_canonical(
        T_hat=T_hat,
        T_true=T_true,
        cooccurrence=C,
        action_source_cooccurrence=H,
        source_action_counts=N_ai,
    )

    # TV(a=0, i=0): |0.0-1.0| + |1.0-0.0| = 2.0 → TV = 1.0.
    assert result.tv_by_pair_full[0, 0] == pytest.approx(1.0, abs=1e-9)
    # TV(a=0, i=1): |0.0-0.0| + |1.0-1.0| = 0.0 → TV = 0.0.
    assert result.tv_by_pair_full[0, 1] == pytest.approx(0.0, abs=1e-9)
