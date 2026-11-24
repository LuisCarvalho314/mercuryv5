"""The training-matched decoder reproduces the online training inference exactly."""
from __future__ import annotations

import numpy as np
import pytest

from scripts.analysis.cscg_distillation_matched_eval import (
    OUTPUT, decoder_lag, fixed_lag_decode, output_name, protocol_for,
)
from src.mercury_runs.algorithms.cscg.distillation import CarriedFilter, clone_groups, row_normalize, smooth_window


def problem(seed=3, n=60):
    rng = np.random.default_rng(seed)
    clones = np.array([3, 1, 4, 2])
    T = row_normalize(rng.uniform(.05, 1., (3, clones.sum(), clones.sum())))
    x = rng.integers(0, len(clones), n)
    incoming = rng.integers(0, 3, n)
    return T, clones, x, incoming


def decode(T, clones, x, incoming, lag):
    starts = np.r_[0, np.cumsum(clones)[:-1]].astype(np.int64)
    return fixed_lag_decode(T, starts, clones.astype(np.int64), x.astype(np.int64), incoming.astype(np.int64), lag)


def filtered(T, clones, x, incoming):
    f = CarriedFilter(clones)
    _, beliefs, losses = f.batch(T, x, incoming)
    return beliefs, losses


def test_lag_zero_is_the_carried_filter_and_nll_is_causal():
    T, clones, x, a = problem()
    states, nll = decode(T, clones, x, a, 0)
    beliefs, losses = filtered(T, clones, x, a)
    np.testing.assert_array_equal(states, beliefs.argmax(axis=1))
    np.testing.assert_allclose(nll, losses / np.log(2), rtol=1e-12)


@pytest.mark.parametrize("lag", [1, 5, 39])
def test_fixed_lag_equals_window_smoothing_from_the_filtered_boundary(lag):
    """Lag 1 = smoothed source P(z_u | x_<=u+1); lag W-1 = long-EM window-start marginal."""
    T, clones, x, a = problem()
    groups = clone_groups(clones)
    beliefs, _ = filtered(T, clones, x, a)
    states, _ = decode(T, clones, x, a, lag)
    for t in range(len(x)):
        end = min(t + lag, len(x) - 1)
        if end == t:
            expected = beliefs[t].argmax()
        else:
            q, _ = smooth_window(T, groups, x[t:end + 1], a[t + 1:end + 1], beliefs[t])
            expected = q[0].argmax()
        assert states[t] == expected


def test_decoder_never_reads_beyond_the_lag():
    T, clones, x, a = problem()
    lag = 4
    base, nll = decode(T, clones, x, a, lag)
    for t in range(0, len(x) - lag - 1, 7):
        y = x.copy(); y[t + lag + 1:] = (y[t + lag + 1:] + 1) % len(clones)
        other, other_nll = decode(T, clones, y, a, lag)
        np.testing.assert_array_equal(base[:t + 1], other[:t + 1])
        np.testing.assert_array_equal(nll[:t + lag + 1], other_nll[:t + lag + 1])


def test_lags_match_training_inference():
    assert decoder_lag("short", 40) == decoder_lag("predictive", 40) == decoder_lag("nll_control", 100) == 1
    assert decoder_lag("long", 40) == 39 and decoder_lag("long", 100) == 99


def test_protocol_uses_the_canonical_walk_set():
    reference = protocol_for(13, 1)
    assert (reference.sensor, reference.sensor_range, reference.rand_prob, reference.evaluation_seed,
            reference.number_of_walks, reference.walk_length) == ("cardinal distance", 1, 0.1, 0, 100, 10_000)
    assert reference.walk_cache_key() == protocol_for(13, 39).walk_cache_key()
    assert reference.protocol_id() != protocol_for(13, 39).protocol_id()


def test_reduced_budget_changes_only_walk_count_and_length():
    canonical, reduced = protocol_for(21, 1), protocol_for(21, 1, 10, 500)
    assert (reduced.number_of_walks, reduced.walk_length) == (10, 500)
    differing = {k for k, v in vars(canonical).items() if vars(reduced)[k] != v}
    assert differing == {"number_of_walks", "walk_length"}
    assert reduced.walk_cache_key() != canonical.walk_cache_key()
    assert output_name() == output_name(100, 10_000) == OUTPUT
    assert output_name(10, 500) == f"{OUTPUT}_walks10x500"
    assert output_name(10, 500, .8) == f"{OUTPUT}_walks10x500_p0.8"
    assert output_name(rand_prob=.8) == f"{OUTPUT}_p0.8"
    other = protocol_for(21, 1, 10, 500, .8)
    assert {k for k, v in vars(reduced).items() if vars(other)[k] != v} == {"rand_prob"}
    assert other.walk_cache_key() != reduced.walk_cache_key()


def test_teacher_is_decoded_with_its_windowed_inference():
    assert decoder_lag("teacher", 100) == 99 and decoder_lag("teacher", 40) == 39
