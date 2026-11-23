"""Unit tests for the First-Class Snapshot Evaluation Architecture.

Tests acceptance criteria 32-40 from the architecture document.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.mercury_runs.application.evaluation_protocol import (
    EvaluationProtocol,
    default_canonical_v4_protocol,
    protocol_from_dict,
)
from src.mercury_runs.application.evaluation_service import (
    build_empirical_ground_truth,
    discover_snapshots,
    filter_evaluation_runs,
    generate_and_cache_walks,
    is_evaluated,
    manifest_run_ids_for_filters,
    save_evaluation_result,
    serialise_evaluation_result,
    snapshot_step_hint,
)
from src.mercury_runs.analysis.evaluation_records import load_evaluation_step
from src.mercury_runs.infrastructure.paper_precision import (
    CanonicalTransitionEvaluationResult,
    compute_hard_alignment,
    compute_occupancy_weighted_action_tv,
    compute_transition_statistics,
    evaluate_transition_structure_canonical,
    normalize_action_rows,
    _action_tv_by_pair,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_simple_walks(n_walks=3, length=10, n_true=4, n_actions=2, n_learned=6, seed=0):
    rng = np.random.default_rng(seed)
    decoded_walks = [rng.integers(0, n_learned, size=length) for _ in range(n_walks)]
    true_walks = [rng.integers(0, n_true, size=length) for _ in range(n_walks)]
    action_walks = [rng.integers(0, n_actions, size=length) for _ in range(n_walks)]
    return decoded_walks, true_walks, action_walks


def _make_protocol(level: int = 13) -> EvaluationProtocol:
    return EvaluationProtocol(
        sensor="cartesian",
        sensor_range=None,
        rand_prob=0.3,
        evaluation_seed=0,
        number_of_walks=5,
        walk_length=20,
        edge_thresholds=[1e-6],
    )


def _make_transition_tensor(n_actions, n_states, rng):
    T = rng.random((n_actions, n_states, n_states)).astype(np.float64)
    row_sums = T.sum(axis=2, keepdims=True)
    row_sums = np.where(row_sums > 0, row_sums, 1.0)
    return T / row_sums


# ---------------------------------------------------------------------------
# Criterion 32: Evaluation never mutates a snapshot
# ---------------------------------------------------------------------------

class TestSnapshotImmutability:
    def test_cscg_snapshot_not_mutated(self):
        """CSCG adapter decode_walks must not modify T, n_clones, or any model params."""
        from src.mercury_runs.algorithms.cscg.snapshot_adapter import CSCGSnapshotAdapter

        rng = np.random.default_rng(42)
        n_obs, n_actions, n_learned = 8, 2, 10
        T_original = _make_transition_tensor(n_actions, n_learned, rng).astype(np.float32)
        T_copy = T_original.copy()
        n_clones = np.ones(n_obs, dtype=np.int64) + 1  # 2 clones per obs => n_learned=16 but we use 10

        # Use a mock so we don't need the real CHMM module.
        mock_npz = {"T": T_original, "n_clones": np.ones(n_obs, dtype=np.int64)}
        snapshot = {
            "model_npz": mock_npz,
            "n_clones": np.ones(n_obs, dtype=np.int64),
            "obs_uniques": np.arange(n_obs, dtype=np.int64).reshape(-1, 1),
            "action_uniques": np.arange(n_actions, dtype=np.int64).reshape(-1, 1),
            "meta": {},
        }

        adapter = CSCGSnapshotAdapter()

        # Build minimal walk data.
        walk_data = {
            "filtered_obs_per_walk": [None],
            "filtered_actions_per_walk": [None],
            "gt_exact_per_walk": [None],
            "gt_action_per_walk": [None],
            "collisions_per_walk": [None],
            "n_walks": 1,
        }

        # decode_walks will skip None walks — the snapshot should not be touched.
        decoded = adapter.decode_walks(snapshot, walk_data, batch_size=10)

        # T must be unchanged.
        np.testing.assert_array_equal(mock_npz["T"], T_copy)
        assert decoded["parameter_updates"] is False

    def test_mercury_snapshot_not_mutated(self):
        """Mercury adapter must not mutate graph or action_map objects."""
        # Without the full Mercury environment we just verify the adapter's API.
        from src.mercury_runs.algorithms.mercury.snapshot_adapter import MercurySnapshotAdapter

        adapter = MercurySnapshotAdapter()

        mock_graph = MagicMock()
        mock_graph.n = 5
        mock_action_map = MagicMock()
        snapshot = {
            "meta": {},
            "sensory_graph": mock_graph,
            "latent_graph": mock_graph,
            "action_map": mock_action_map,
            "obs_uniques": None,
        }

        walk_data = {
            "filtered_obs_per_walk": [None],
            "filtered_actions_per_walk": [None],
            "gt_exact_per_walk": [None],
            "gt_action_per_walk": [None],
            "collisions_per_walk": [None],
            "n_walks": 1,
        }

        decoded = adapter.decode_walks(snapshot, walk_data, level=13)
        assert decoded["parameter_updates"] is False


# ---------------------------------------------------------------------------
# Criterion 33: Recursive CSCG state propagation across batches and reset per walk
# ---------------------------------------------------------------------------

class TestCSCGRecursiveBatchInference:
    def test_batch_propagation_uses_terminal_state(self):
        """Simulated recursive batch: terminal forward message feeds next batch."""
        from src.mercury_runs.algorithms.cscg.snapshot_adapter import decode_walk_frozen_online_batch

        n_clones_arr = np.array([2, 2, 2], dtype=np.int64)
        n_total = int(n_clones_arr.sum())

        # Build a minimal mock CHMM model.
        mock_model = MagicMock()
        mock_model.n_clones = n_clones_arr
        mock_model.T = np.ones((2, n_total, n_total), dtype=np.float32) / n_total
        mock_model.Pi_x = np.ones(n_total, dtype=np.float64) / n_total

        call_count = [0]
        initial_pis: list[np.ndarray] = []

        def fake_forward(T_t, Pi_x, n_clones, obs, act, store_messages=False):
            call_count[0] += 1
            initial_pis.append(Pi_x.copy())
            length = obs.shape[0]
            # Return uniform forward messages.
            mess_fwd = np.ones((length, n_total), dtype=np.float64) / n_total
            log2_lik = np.zeros(length, dtype=np.float64)
            return log2_lik, mess_fwd

        mock_chmm = MagicMock()
        mock_chmm.forward = fake_forward

        obs_idx = np.arange(6, dtype=np.int64) % 3
        act_idx = np.zeros(6, dtype=np.int64)
        batch_size = 2

        states = decode_walk_frozen_online_batch(
            model=mock_model,
            obs_idx=obs_idx,
            act_idx=act_idx,
            batch_size=batch_size,
            chmm_mod=mock_chmm,
        )

        # Should have called forward 3 times (6 / 2 = 3 batches).
        assert call_count[0] == 3, f"Expected 3 forward calls, got {call_count[0]}"

        # Each batch should use the terminal message of the previous batch.
        # First call: initial uniform Pi_x.
        np.testing.assert_allclose(initial_pis[0], np.ones(n_total) / n_total)
        # Second and third calls: terminal message from previous batch (still uniform).
        np.testing.assert_allclose(initial_pis[1], np.ones(n_total) / n_total)

        # Should produce one state per timestep.
        assert states.shape[0] == 6

    def test_reset_between_walks(self):
        """Each walk starts with a fresh initial distribution, not the terminal of the previous walk."""
        from src.mercury_runs.algorithms.cscg.snapshot_adapter import decode_walk_frozen_online_batch

        n_total = 4
        n_clones_arr = np.array([4], dtype=np.int64)

        # Pi_x starts non-uniform.
        initial_pi = np.array([0.7, 0.1, 0.1, 0.1], dtype=np.float64)

        mock_model = MagicMock()
        mock_model.n_clones = n_clones_arr
        mock_model.T = np.eye(n_total, dtype=np.float32).reshape(1, n_total, n_total)
        mock_model.Pi_x = initial_pi.copy()

        observed_pis: list[np.ndarray] = []

        def fake_forward(T_t, Pi_x, n_clones, obs, act, store_messages=False):
            observed_pis.append(Pi_x.copy())
            mess_fwd = np.tile(Pi_x, (obs.shape[0], 1))
            return np.zeros(obs.shape[0]), mess_fwd

        mock_chmm = MagicMock()
        mock_chmm.forward = fake_forward

        obs = np.zeros(3, dtype=np.int64)
        act = np.zeros(3, dtype=np.int64)

        # First walk: starts with model.Pi_x.
        decode_walk_frozen_online_batch(model=mock_model, obs_idx=obs, act_idx=act, batch_size=10, chmm_mod=mock_chmm)
        # Second walk: reset — also starts with model.Pi_x (not carry-over from walk 1).
        mock_model.Pi_x = initial_pi.copy()
        decode_walk_frozen_online_batch(model=mock_model, obs_idx=obs, act_idx=act, batch_size=10, chmm_mod=mock_chmm)

        # Both walks should start with the same initial Pi_x.
        np.testing.assert_allclose(observed_pis[0], initial_pi)
        np.testing.assert_allclose(observed_pis[1], initial_pi)


# ---------------------------------------------------------------------------
# Criterion 34: All initial timesteps included (burn_in_steps = 0)
# ---------------------------------------------------------------------------

class TestInitialTimestepsIncluded:
    def test_all_timesteps_retained_in_walk_data(self):
        """Walk data must include all timesteps; no burn-in masking applied."""
        rng = np.random.default_rng(0)
        n_walks, length, n_true, n_actions, n_learned = 2, 8, 4, 2, 6
        decoded_walks, true_walks, action_walks = _make_simple_walks(n_walks, length, n_true, n_actions, n_learned)

        C, H, N_ai = compute_transition_statistics(
            decoded_walks=decoded_walks,
            true_walks=true_walks,
            action_walks=action_walks,
            n_learned=n_learned,
            n_true=n_true,
            n_actions=n_actions,
        )

        # C total should equal sum of all walk lengths (no steps excluded).
        total_steps = sum(w.shape[0] for w in decoded_walks)
        assert int(C.sum()) == total_steps, (
            f"Expected C.sum()={total_steps} (all timesteps), got {int(C.sum())}"
        )

    def test_protocol_burn_in_is_zero(self):
        protocol = EvaluationProtocol()
        assert protocol.burn_in_steps == 0
        assert protocol.initial_timestep_policy == "include_all"


# ---------------------------------------------------------------------------
# Criterion 35: Empirical GT counts match true state/action sequences exactly
# ---------------------------------------------------------------------------

class TestEmpiricalGroundTruth:
    def test_gt_counts_match_true_sequences(self):
        """T_GT[a, s_src, s_tgt] must exactly count transitions from true walks."""
        n_true, n_actions = 3, 2

        # Construct a deterministic walk: state cycle 0->1->2->0->1->2, action alternates.
        true_states = np.array([0, 1, 2, 0, 1, 2], dtype=np.int64)
        actions = np.array([0, 1, 0, 1, 0, 1], dtype=np.int64)

        walk_data = {
            "gt_exact_per_walk": [true_states],
            "gt_action_per_walk": [actions],
            "n_walks": 1,
        }

        gt = build_empirical_ground_truth(
            walk_data=walk_data, n_true=n_true, n_actions=n_actions, level=0,
        )
        T_GT = gt["T_GT"]

        # Transition at t=0: state[0]=0 -> state[1]=1, action=actions[1]=1.
        # Transition at t=1: state[1]=1 -> state[2]=2, action=actions[2]=0.
        # Transition at t=2: state[2]=2 -> state[3]=0, action=actions[3]=1.
        # Transition at t=3: state[3]=0 -> state[4]=1, action=actions[4]=0.
        # Transition at t=4: state[4]=1 -> state[5]=2, action=actions[5]=1.
        # Trace with post-step action convention (action[t+1] causes move t->t+1):
        # t=0: src=0, tgt=1, a=actions[1]=1 => T_GT[1,0,1]+=1
        # t=1: src=1, tgt=2, a=actions[2]=0 => T_GT[0,1,2]+=1
        # t=2: src=2, tgt=0, a=actions[3]=1 => T_GT[1,2,0]+=1
        # t=3: src=0, tgt=1, a=actions[4]=0 => T_GT[0,0,1]+=1
        # t=4: src=1, tgt=2, a=actions[5]=1 => T_GT[1,1,2]+=1
        assert T_GT[1, 0, 1] == 1, f"Expected T_GT[1,0,1]=1, got {T_GT[1,0,1]}"
        assert T_GT[0, 1, 2] == 1, f"Expected T_GT[0,1,2]=1, got {T_GT[0,1,2]}"
        assert T_GT[1, 2, 0] == 1, f"Expected T_GT[1,2,0]=1, got {T_GT[1,2,0]}"
        assert T_GT[0, 0, 1] == 1, f"Expected T_GT[0,0,1]=1, got {T_GT[0,0,1]}"
        assert T_GT[1, 1, 2] == 1, f"Expected T_GT[1,1,2]=1, got {T_GT[1,1,2]}"
        # Total transitions = 5.
        assert int(T_GT.sum()) == 5, f"Expected 5 total transitions, got {int(T_GT.sum())}"

    def test_valid_rows_match_observed_state_action_pairs(self):
        """Valid rows must exactly be the (a, i) pairs observed in held-out walks."""
        true_states = np.array([0, 1, 0], dtype=np.int64)
        actions = np.array([0, 1, 0], dtype=np.int64)
        walk_data = {
            "gt_exact_per_walk": [true_states],
            "gt_action_per_walk": [actions],
            "n_walks": 1,
        }
        gt = build_empirical_ground_truth(walk_data=walk_data, n_true=3, n_actions=2, level=0)
        # Transitions: t=0: 0->1 under a=1; t=1: 1->0 under a=0
        assert gt["valid_action_rows"][1, 0], "State 0, action 1 must be valid"
        assert gt["valid_action_rows"][0, 1], "State 1, action 0 must be valid"
        # State 2 never appears as source.
        assert not gt["valid_action_rows"][:, 2].any(), "State 2 must not be a valid source"


# ---------------------------------------------------------------------------
# Criterion 36: Occupancy weights from true evaluation counts only
# ---------------------------------------------------------------------------

class TestOccupancyWeights:
    def test_occupancy_weights_from_true_counts_not_model(self):
        """Occupancy weights must be derived from N_ai (true counts), not from decoded model output."""
        n_true, n_actions, n_learned = 3, 2, 4
        rng = np.random.default_rng(7)

        # True counts: state 0 visited 100 times under action 0, state 1 once.
        N_ai = np.zeros((n_actions, n_true), dtype=np.int64)
        N_ai[0, 0] = 100
        N_ai[0, 1] = 1
        valid_rows = N_ai > 0

        tv_by_pair = np.ones((n_actions, n_true), dtype=np.float64) * 0.5  # equal TV everywhere
        tv_by_pair[~valid_rows] = 0.0

        tv_occ, tv_macro, occ_weights, _, _ = compute_occupancy_weighted_action_tv(
            tv_by_pair, valid_rows, N_ai
        )

        # Occupancy weight of (a=0, i=0) must be 100/101.
        expected_weight_00 = 100.0 / 101.0
        assert abs(float(occ_weights[0, 0]) - expected_weight_00) < 1e-6, (
            f"Expected weight[0,0]≈{expected_weight_00}, got {float(occ_weights[0,0])}"
        )

        # TV_occ = weight * TV everywhere; since TV=0.5 for all valid pairs:
        expected_tv_occ = 0.5  # all valid pairs have TV=0.5
        assert abs(tv_occ - expected_tv_occ) < 1e-6


# ---------------------------------------------------------------------------
# Criterion 37: Occupancy-weighted and macro TV differ on imbalanced walks
# ---------------------------------------------------------------------------

class TestTVAggregation:
    def test_occ_macro_differ_on_imbalanced_walks(self):
        """Occupancy-weighted TV and macro TV must differ when walks are imbalanced."""
        n_true, n_actions = 3, 1

        # State 0 visited 1000 times (high TV=0.9), state 1 visited 1 time (TV=0.1).
        N_ai = np.zeros((n_actions, n_true), dtype=np.int64)
        N_ai[0, 0] = 1000
        N_ai[0, 1] = 1
        valid_rows = N_ai > 0

        tv_by_pair = np.zeros((n_actions, n_true), dtype=np.float64)
        tv_by_pair[0, 0] = 0.9
        tv_by_pair[0, 1] = 0.1

        tv_occ, tv_macro, _, _, _ = compute_occupancy_weighted_action_tv(
            tv_by_pair, valid_rows, N_ai
        )

        # Occ-weighted: dominated by state 0 (high TV).
        # Macro: average of 0.9 and 0.1 = 0.5.
        assert tv_occ > 0.8, f"Expected tv_occ > 0.8 (state 0 dominant), got {tv_occ}"
        assert abs(tv_macro - 0.5) < 1e-6, f"Expected tv_macro=0.5, got {tv_macro}"
        assert tv_occ != tv_macro


# ---------------------------------------------------------------------------
# Criterion 38: Action-conditioned TV >= policy-collapsed TV (Jensen's inequality)
# ---------------------------------------------------------------------------

class TestActionVsCollapsedTV:
    def test_action_tv_no_smaller_than_collapsed_in_cancellation(self):
        """In constructed cancellation case, action TV >= collapsed TV."""
        # Two actions with opposite errors that cancel after marginalisation.
        # Action 0: P_hat gives mass to state j=1 instead of j=0.
        # Action 1: P_hat gives mass to state j=0 instead of j=1.
        # Equal action marginal => collapsed TV cancels to ~0.

        n_true, n_actions = 2, 2
        n_learned = n_true

        # GT: action 0 => go to state 0; action 1 => go to state 1.
        T_true = np.zeros((n_actions, n_true, n_true), dtype=np.float64)
        T_true[0, 0, 0] = 1.0  # state 0, action 0 -> stay
        T_true[1, 0, 1] = 1.0  # state 0, action 1 -> go right
        # Only state 0 is a source (to keep it simple).

        # Learned: swapped — action 0 -> right, action 1 -> stay.
        T_hat = np.zeros((n_actions, n_learned, n_learned), dtype=np.float64)
        T_hat[0, 0, 1] = 1.0
        T_hat[1, 0, 0] = 1.0

        # Perfect alignment: learned state k=0 maps to true state i=0.
        C = np.zeros((n_learned, n_true), dtype=np.int64)
        C[0, 0] = 100

        H = np.zeros((n_actions, n_learned, n_true), dtype=np.int64)
        H[0, 0, 0] = 50
        H[1, 0, 0] = 50

        N_ai = np.zeros((n_actions, n_true), dtype=np.int64)
        N_ai[0, 0] = 50
        N_ai[1, 0] = 50

        result = evaluate_transition_structure_canonical(
            T_hat=T_hat,
            T_true=T_true,
            cooccurrence=C,
            action_source_cooccurrence=H,
            source_action_counts=N_ai,
            edge_thresholds=(1e-6,),
        )

        # Action TV: for each (a, i=0), TV = 1.0 (completely wrong).
        # Collapsed TV: action prior is 50/50, collapsed P_hat = 0.5*P_0 + 0.5*P_1.
        # P_hat_a0 = [0, 1], P_hat_a1 = [1, 0] => collapsed_hat = [0.5, 0.5].
        # P_true_a0 = [1, 0], P_true_a1 = [0, 1] => collapsed_true = [0.5, 0.5].
        # Collapsed TV = 0; Action TV = 1.

        assert result.tv_occupancy > result.collapsed_eval_tv_occupancy, (
            f"Action TV ({result.tv_occupancy}) must exceed collapsed TV "
            f"({result.collapsed_eval_tv_occupancy}) in this cancellation case"
        )


# ---------------------------------------------------------------------------
# Criterion 39: Projected mass assigned to aligned target or ⊥, no silent loss
# ---------------------------------------------------------------------------

class TestMassPreservation:
    def test_projected_rows_sum_to_one(self):
        """Every valid projected row must sum to 1.0 (aligned mass + ⊥ mass)."""
        n_true, n_actions, n_learned = 4, 2, 6
        rng = np.random.default_rng(123)

        T_hat = _make_transition_tensor(n_actions, n_learned, rng)
        T_true = _make_transition_tensor(n_actions, n_true, rng)

        # Random cooccurrence with some unaligned learned states.
        C = np.zeros((n_learned, n_true), dtype=np.int64)
        for k in range(n_learned - 1):
            C[k, rng.integers(0, n_true)] = rng.integers(10, 100)
        # Leave learned state n_learned-1 with no evidence (unaligned).

        H = np.zeros((n_actions, n_learned, n_true), dtype=np.int64)
        N_ai = np.zeros((n_actions, n_true), dtype=np.int64)
        for a in range(n_actions):
            for k in range(n_learned - 1):
                for i in range(n_true):
                    H[a, k, i] = C[k, i]
            N_ai[a] = C.sum(axis=0)

        result = evaluate_transition_structure_canonical(
            T_hat=T_hat,
            T_true=T_true,
            cooccurrence=C,
            action_source_cooccurrence=H,
            source_action_counts=N_ai,
            edge_thresholds=(1e-6,),
        )

        P_hat = result.P_hat_full  # (n_actions, n_true, n_true+1)
        valid = result.valid_action_rows

        for a in range(n_actions):
            for i in range(n_true):
                if valid[a, i]:
                    row_sum = float(P_hat[a, i].sum())
                    assert abs(row_sum - 1.0) < 1e-8, (
                        f"Valid row (a={a}, i={i}) sums to {row_sum}, expected 1.0"
                    )

    def test_unaligned_learned_states_map_to_sink(self):
        """Mass routed through unaligned learned target states must go to ⊥."""
        n_true, n_actions, n_learned = 2, 1, 3
        # Learned state 2 is unaligned (no evidence in C).
        C = np.array([[10, 0], [0, 10], [0, 0]], dtype=np.int64)  # (3, 2)
        # Alignment: k=0->i=0, k=1->i=1, k=2->-1 (unaligned).
        alignment = compute_hard_alignment(C)
        assert alignment[2] == -1, f"Learned state 2 should be unaligned, got {alignment[2]}"

        # Learned T: state 0 always transitions to learned state 2 (unaligned).
        T_hat = np.zeros((n_actions, n_learned, n_learned), dtype=np.float64)
        T_hat[0, 0, 2] = 1.0  # all mass to unaligned target

        T_true = np.zeros((n_actions, n_true, n_true), dtype=np.float64)
        T_true[0, 0, 1] = 1.0  # GT: state 0 -> state 1

        H = np.zeros((n_actions, n_learned, n_true), dtype=np.int64)
        H[0, 0, 0] = 10  # source: learned 0 = true 0, action 0

        N_ai = np.zeros((n_actions, n_true), dtype=np.int64)
        N_ai[0, 0] = 10

        result = evaluate_transition_structure_canonical(
            T_hat=T_hat,
            T_true=T_true,
            cooccurrence=C,
            action_source_cooccurrence=H,
            source_action_counts=N_ai,
            edge_thresholds=(1e-6,),
        )

        # All predicted mass goes to ⊥ => P_hat[0, 0, n_true] should be 1.0.
        sink_mass = float(result.P_hat_full[0, 0, n_true])
        assert abs(sink_mass - 1.0) < 1e-8, (
            f"Expected all mass in ⊥ (index {n_true}), got {sink_mass}"
        )
        # TV should be 1.0 (GT has no mass at ⊥).
        tv = float(result.tv_by_pair_full[0, 0])
        assert abs(tv - 1.0) < 1e-8, f"Expected TV=1.0 for fully unmatched row, got {tv}"


# ---------------------------------------------------------------------------
# Criterion 40: Study summaries aggregate across seeds, not transitions/walks
# ---------------------------------------------------------------------------

class TestStudyAggregation:
    def test_seed_is_independent_replication_unit(self):
        """Each seed produces one scalar metric; aggregation is over seeds."""
        # This is a structural test: verify that the evaluation result contains
        # exactly one scalar per snapshot (not one per walk or transition).
        n_true, n_actions, n_learned = 4, 2, 6
        rng = np.random.default_rng(99)

        T_hat = _make_transition_tensor(n_actions, n_learned, rng)
        T_true = _make_transition_tensor(n_actions, n_true, rng)
        C = np.zeros((n_learned, n_true), dtype=np.int64)
        for k in range(n_learned):
            C[k, k % n_true] = 10
        H = np.zeros((n_actions, n_learned, n_true), dtype=np.int64)
        N_ai = np.zeros((n_actions, n_true), dtype=np.int64)
        for a in range(n_actions):
            for k in range(n_learned):
                H[a, k, k % n_true] = 5
            N_ai[a] = C.sum(axis=0)

        result = evaluate_transition_structure_canonical(
            T_hat=T_hat,
            T_true=T_true,
            cooccurrence=C,
            action_source_cooccurrence=H,
            source_action_counts=N_ai,
            edge_thresholds=(1e-6,),
        )

        # Each of these is a single scalar — one value per snapshot/seed.
        assert isinstance(result.tv_occupancy, float)
        assert isinstance(result.tv_macro, float)
        assert isinstance(result.collapsed_eval_tv_occupancy, float)


# ---------------------------------------------------------------------------
# Protocol hashing
# ---------------------------------------------------------------------------

class TestProtocolHashing:
    def test_same_protocol_same_hash(self):
        p1 = EvaluationProtocol()
        p2 = EvaluationProtocol()
        assert p1.protocol_hash() == p2.protocol_hash()

    def test_different_seed_different_hash(self):
        p1 = EvaluationProtocol(evaluation_seed=0)
        p2 = EvaluationProtocol(evaluation_seed=1)
        assert p1.protocol_hash() != p2.protocol_hash()

    def test_different_walk_length_different_hash(self):
        p1 = EvaluationProtocol(walk_length=1000)
        p2 = EvaluationProtocol(walk_length=2000)
        assert p1.protocol_hash() != p2.protocol_hash()

    def test_protocol_roundtrip(self):
        p = EvaluationProtocol(evaluation_seed=42, rand_prob=0.5)
        d = p.to_dict()
        p2 = protocol_from_dict(d)
        assert p.protocol_hash() == p2.protocol_hash()

    def test_protocol_id_format(self):
        p = EvaluationProtocol()
        pid = p.protocol_id()
        assert pid.startswith("canonical-v5-")


# ---------------------------------------------------------------------------
# Walk cache persistence
# ---------------------------------------------------------------------------

class TestWalkCachePersistence:
    def test_walk_cache_idempotent(self, tmp_path):
        """Loading a saved walk cache must return identical data."""
        from src.mercury_runs.application.evaluation_service import (
            _save_walk_cache, _load_walk_cache, _walk_cache_dir, _walk_cache_filename,
        )

        protocol = EvaluationProtocol(evaluation_seed=0, number_of_walks=3, walk_length=5)
        level = 13

        rng = np.random.default_rng(0)
        walk_data = {
            "n_walks": 3,
            "filtered_obs_per_walk": [rng.integers(0, 5, (5, 3)).astype(np.float32), None, rng.integers(0, 5, (5, 3)).astype(np.float32)],
            "filtered_actions_per_walk": [rng.integers(0, 2, (5, 4)).astype(np.float32), None, rng.integers(0, 2, (5, 4)).astype(np.float32)],
            "gt_exact_per_walk": [rng.integers(0, 10, 5).astype(np.int64), None, rng.integers(0, 10, 5).astype(np.int64)],
            "gt_cartesian_per_walk": [rng.integers(0, 10, 5).astype(np.int64), None, None],
            "gt_action_per_walk": [rng.integers(0, 2, 5).astype(np.int64), None, rng.integers(0, 2, 5).astype(np.int64)],
            "collisions_per_walk": [np.zeros(5, dtype=bool), None, None],
        }

        cache_dir = _walk_cache_dir(tmp_path, protocol)
        filename = _walk_cache_filename(protocol, level)
        _save_walk_cache(cache_dir, filename, walk_data)

        loaded = _load_walk_cache(cache_dir, filename)
        assert loaded["n_walks"] == 3
        assert loaded["filtered_obs_per_walk"][1] is None  # second walk was None
        np.testing.assert_array_equal(
            loaded["filtered_obs_per_walk"][0],
            walk_data["filtered_obs_per_walk"][0],
        )
        np.testing.assert_array_equal(
            loaded["gt_exact_per_walk"][0],
            walk_data["gt_exact_per_walk"][0],
        )


# ---------------------------------------------------------------------------
# is_evaluated / save_evaluation_result
# ---------------------------------------------------------------------------

class TestEvaluationPersistence:
    def test_is_evaluated_false_initially(self, tmp_path):
        protocol = EvaluationProtocol()
        assert not is_evaluated(run_root=tmp_path, protocol=protocol, step=100)

    def test_is_evaluated_true_after_save(self, tmp_path):
        protocol = EvaluationProtocol()
        record = {"schema": "test", "step": 100}
        save_evaluation_result(run_root=tmp_path, protocol=protocol, step=100, record=record)
        assert is_evaluated(run_root=tmp_path, protocol=protocol, step=100)

    def test_saved_record_is_valid_json(self, tmp_path):
        protocol = EvaluationProtocol()
        record = {"schema": "transition_structure_evaluation_v4", "step": 50, "tv": 0.3}
        path = save_evaluation_result(run_root=tmp_path, protocol=protocol, step=50, record=record)
        loaded = json.loads(path.read_text())
        assert loaded["schema"] == "transition_structure_evaluation_v4"
        assert loaded["step"] == 50

    def test_method_namespaces_prevent_step_collision(self, tmp_path):
        protocol = EvaluationProtocol()
        save_evaluation_result(
            run_root=tmp_path,
            protocol=protocol,
            step=2000,
            method="mercury_cscg_hybrid",
            record={"schema": "test", "source": "hybrid_cscg"},
        )
        save_evaluation_result(
            run_root=tmp_path,
            protocol=protocol,
            step=2000,
            method="mercury_cscg_hybrid_mercury",
            record={"schema": "test", "source": "mercury"},
        )

        protocol_id = protocol.protocol_id()
        hybrid_record = load_evaluation_step(tmp_path, protocol_id, 2000, method="mercury_cscg_hybrid")
        mercury_record = load_evaluation_step(tmp_path, protocol_id, 2000, method="mercury_cscg_hybrid_mercury")

        assert hybrid_record is not None
        assert mercury_record is not None
        assert hybrid_record["source"] == "hybrid_cscg"
        assert mercury_record["source"] == "mercury"

    def test_tmgwr_phase_namespace_prevents_step_collision(self, tmp_path):
        # The CSCG phase restarts its step counter, so the TMGWR bootstrap at
        # cumulative 2000 and the CSCG phase step 2000 (cumulative 4000) would
        # otherwise overwrite each other.
        protocol = EvaluationProtocol()
        save_evaluation_result(
            run_root=tmp_path, protocol=protocol, step=2000,
            method="tmgwr_cscg_hybrid", record={"schema": "test", "source": "hybrid_cscg"},
        )
        save_evaluation_result(
            run_root=tmp_path, protocol=protocol, step=2000,
            method="tmgwr_cscg_hybrid_tmgwr", record={"schema": "test", "source": "tmgwr"},
        )

        protocol_id = protocol.protocol_id()
        hybrid_record = load_evaluation_step(tmp_path, protocol_id, 2000, method="tmgwr_cscg_hybrid")
        tmgwr_record = load_evaluation_step(tmp_path, protocol_id, 2000, method="tmgwr_cscg_hybrid_tmgwr")

        assert hybrid_record["source"] == "hybrid_cscg"
        assert tmgwr_record["source"] == "tmgwr"


class TestBootstrapSnapshotDiscovery:
    """Bootstrap-phase snapshots live under bundles/, not the run's own snapshots/."""

    @staticmethod
    def _make_snapshot(run_root: Path, stage: str, step: int) -> Path:
        path = run_root / "bundles" / "snapshots" / stage / "paper_precision" / f"step={step}"
        path.mkdir(parents=True)
        return path

    def test_tmgwr_phase_discovers_sensory_snapshots(self, tmp_path):
        self._make_snapshot(tmp_path, "sensory", 2000)
        self._make_snapshot(tmp_path, "sensory", 4000)
        found = discover_snapshots(tmp_path, "tmgwr_cscg_hybrid_tmgwr")
        assert [path.name for path in found] == ["step=2000", "step=4000"]

    def test_mercury_phase_ignores_sensory_snapshots(self, tmp_path):
        self._make_snapshot(tmp_path, "sensory", 2000)
        assert discover_snapshots(tmp_path, "mercury_cscg_hybrid_mercury") == []

    def test_tmgwr_phase_without_snapshots_is_empty(self, tmp_path):
        assert discover_snapshots(tmp_path, "tmgwr_cscg_hybrid_tmgwr") == []


# ---------------------------------------------------------------------------
# Run filtering for parallel evaluation
# ---------------------------------------------------------------------------

class TestEvaluationRunFiltering:
    @staticmethod
    def _make_run(tmp_path: Path, name: str, level: int) -> Path:
        run_root = tmp_path / name
        run_root.mkdir()
        (run_root / "run_config.json").write_text(
            json.dumps({"run_payload": {"level": level, "rand_prob": 0.3}}),
            encoding="utf-8",
        )
        return run_root

    def test_snapshot_step_hint_parses_step_directory(self) -> None:
        assert snapshot_step_hint(Path("step=002500")) == 2500
        assert snapshot_step_hint(Path("snapshot-final")) is None

    def test_manifest_filter_resolves_reused_run_identity(self, tmp_path: Path) -> None:
        actual_id = "actual-reused-id"
        requested_id = "new-requested-id"
        config = {
            "method": "cscg",
            "level": 13,
            "seed": 0,
            "rand_prob": 0.1,
            "cscg_batch_size": 10,
            "requested_run_id": requested_id,
        }
        (tmp_path / "study_manifest.jsonl").write_text(
            json.dumps({"config": config}) + "\n",
            encoding="utf-8",
        )
        run_root = tmp_path / actual_id
        run_root.mkdir()
        (run_root / "run_status.json").write_text('{"status": "completed"}', encoding="utf-8")
        (run_root / "run_config.json").write_text(
            json.dumps({"cli_args": {**config, "requested_run_id": actual_id}}),
            encoding="utf-8",
        )

        selected = manifest_run_ids_for_filters(
            study_root=tmp_path,
            method="cscg",
            level=13,
            rand_prob=0.1,
            cscg_batch_size=10,
            cscg_clone_preset=None,
        )

        assert selected == {requested_id, actual_id}

    def test_filter_evaluation_runs_keeps_only_matching_runs_with_snapshots(self, tmp_path, monkeypatch):
        protocol = EvaluationProtocol()
        keep = self._make_run(tmp_path, "keep", 13)
        wrong_level = self._make_run(tmp_path, "wrong_level", 20)
        no_snapshots = self._make_run(tmp_path, "no_snapshots", 13)

        def fake_discover_snapshots(run_root: Path, method: str) -> list[Path]:
            assert method == "cscg"
            if run_root == keep:
                return [run_root / "step=000010"]
            if run_root == wrong_level:
                return [run_root / "step=000010"]
            return []

        monkeypatch.setattr(
            "src.mercury_runs.application.evaluation_service.discover_snapshots",
            fake_discover_snapshots,
        )

        selected = filter_evaluation_runs(
            runs=[keep, wrong_level, no_snapshots],
            method="cscg",
            level=13,
            protocol=protocol,
            resume=True,
            force=False,
        )
        assert selected == [keep]

    def test_filter_evaluation_runs_can_filter_by_rand_prob(self, tmp_path, monkeypatch):
        protocol = EvaluationProtocol()
        keep = self._make_run(tmp_path, "keep", 13)
        other_rand = self._make_run(tmp_path, "other_rand", 13)
        (other_rand / "run_config.json").write_text(
            json.dumps({"run_payload": {"level": 13, "rand_prob": 0.0}}),
            encoding="utf-8",
        )

        def fake_discover_snapshots(run_root: Path, method: str) -> list[Path]:
            return [run_root / "step=000010"]

        monkeypatch.setattr(
            "src.mercury_runs.application.evaluation_service.discover_snapshots",
            fake_discover_snapshots,
        )

        selected = filter_evaluation_runs(
            runs=[keep, other_rand],
            method="cscg",
            level=13,
            protocol=protocol,
            rand_prob=0.3,
            resume=True,
            force=False,
        )
        assert selected == [keep]

    def test_filter_evaluation_runs_skips_fully_evaluated_runs_when_resuming(self, tmp_path, monkeypatch):
        protocol = EvaluationProtocol()
        complete = self._make_run(tmp_path, "complete", 13)
        incomplete = self._make_run(tmp_path, "incomplete", 13)
        save_evaluation_result(run_root=complete, protocol=protocol, step=10, record={"schema": "test"})
        save_evaluation_result(run_root=complete, protocol=protocol, step=20, record={"schema": "test"})
        save_evaluation_result(run_root=incomplete, protocol=protocol, step=10, record={"schema": "test"})

        def fake_discover_snapshots(run_root: Path, method: str) -> list[Path]:
            return [run_root / "step=000010", run_root / "step=000020"]

        monkeypatch.setattr(
            "src.mercury_runs.application.evaluation_service.discover_snapshots",
            fake_discover_snapshots,
        )

        selected = filter_evaluation_runs(
            runs=[complete, incomplete],
            method="cscg",
            level=13,
            protocol=protocol,
            resume=True,
            force=False,
        )
        assert selected == [incomplete]

    def test_filter_evaluation_runs_includes_complete_runs_when_forced(self, tmp_path, monkeypatch):
        protocol = EvaluationProtocol()
        complete = self._make_run(tmp_path, "complete", 13)
        save_evaluation_result(run_root=complete, protocol=protocol, step=10, record={"schema": "test"})

        monkeypatch.setattr(
            "src.mercury_runs.application.evaluation_service.discover_snapshots",
            lambda run_root, method: [run_root / "step=000010"],
        )

        selected = filter_evaluation_runs(
            runs=[complete],
            method="cscg",
            level=13,
            protocol=protocol,
            resume=True,
            force=True,
        )
        assert selected == [complete]
