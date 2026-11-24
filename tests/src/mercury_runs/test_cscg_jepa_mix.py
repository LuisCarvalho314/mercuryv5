"""Explicit predictive JEPA-CSCG objective: (1 - rho) EM / sum(B) + rho L_JEPA."""
from __future__ import annotations

import math
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from src.mercury_runs.algorithms.cscg import distillation as module
from src.mercury_runs.algorithms.cscg.distillation import (
    CarriedFilter, ExperimentConfig, ZeroPredictiveSupportError, clone_groups, legacy_jepa_mix,
    mixed_counts, mm_update, pair_counts, replay_terms, row_normalize, smooth_window, train_experiment,
)

MIXES = [0., .25, .5, .75, 1.]
FIXTURES = Path(__file__).resolve().parent / "fixtures"


def tiny_problem():
    rng = np.random.default_rng(7)
    return row_normalize(rng.uniform(.1, 1., (2, 5, 5))), np.array([2, 3])


def ambiguous_stream():
    """Random symbols: the same (symbol, action) has several successors."""
    return np.array([0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1]), np.array([0, 1, 0, 1, 1, 0, 1, 0, 0, 1])


def alternating_stream(length=12, seed=3):
    """0 -> 1 -> 0 -> ...: every (symbol, action) has a single successor symbol,
    so pure JEPA never meets a successor it has not already supported."""
    x = np.arange(length) % 2
    return x, np.random.default_rng(seed).integers(0, 2, length - 1)


def stream_for(rho):
    return alternating_stream() if rho == 1 else ambiguous_stream()


def mixed(rho, **kwargs):
    return ExperimentConfig(jepa_mix=rho, **{"window": 3, "local_updates": 3, **kwargs})


def assert_row_stochastic(T):
    assert np.all(np.isfinite(T)) and np.all(T >= 0)
    np.testing.assert_allclose(T.sum(-1), 1, rtol=0, atol=1e-12)


def direct_reference(T0, clones, x, a, rho, config):
    """Independent loop implementation of the per-arrival MM recursion.

    D is accumulated entry by entry from r^u_ij q_{u+1}(j) / n, not via replay_terms.
    """
    groups = clone_groups(clones)
    T, teacher, S = T0.copy(), T0.copy(), np.zeros_like(T0)
    tau = 2 ** (-1 / config.teacher_half_life)
    f, h = CarriedFilter(clones), CarriedFilter(clones)
    fs, hs, source = [], [], {}
    for t, z in enumerate(x):
        previous = None if t == 0 else f.belief.copy()
        fs.append(f.step(T, z, None if t == 0 else a[t - 1])[1])
        if rho > 0:
            hs.append(h.step(teacher, z, None if t == 0 else a[t - 1])[1])
        if t == 0:
            continue
        if rho > 0 and config.student_inference == "smooth":
            source[t - 1] = smooth_window(T, groups, x[t - 1:t + 1], a[t - 1:t], previous)[0][0]
        s = max(0, t - config.window + 1)
        if rho > 0:
            q = smooth_window(teacher, groups, x[s:t + 1], a[s:t], hs[s])[0]
        for _ in range(config.local_updates):
            M = np.zeros_like(T)
            if rho < 1:
                candidate = S + pair_counts(T, previous, a[t - 1], groups[z])
                B = candidate + config.pseudocount
                M += (1 - rho) * B / B.sum()
            if rho > 0:
                D, n = np.zeros_like(T), t - s
                for u in range(s, t):
                    b = source[u] if config.student_inference == "smooth" else fs[u]
                    action, J = a[u], groups[x[u + 1]]
                    for jj, j in enumerate(J):
                        p = sum(b[i] * T[action, i, j] for i in range(T.shape[1]))
                        for i in range(T.shape[1]):
                            D[action, i, j] += b[i] * T[action, i, j] / p * q[u + 1 - s, j] / n
                M += rho * D
            rows = M.sum(-1, keepdims=True)
            T = np.where(rows > 0, M / np.where(rows > 0, rows, 1), T)
        if rho < 1:
            S = candidate
        if rho > 0:
            teacher = tau * teacher + (1 - tau) * T
    return T, teacher, (S if rho < 1 else None)


# --- configuration contract -------------------------------------------------

@pytest.mark.parametrize("value", [-1e-9, 1 + 1e-9, math.nan, math.inf, -math.inf, True, False, "0.5", [0.5]])
def test_invalid_jepa_mix_is_rejected_not_clipped(value):
    with pytest.raises(ValueError, match="jepa_mix"):
        ExperimentConfig(jepa_mix=value)


@pytest.mark.parametrize("value", [0, 0., .5, 1, 1., np.float64(.25)])
def test_closed_interval_is_accepted_and_recorded(value):
    config = ExperimentConfig(jepa_mix=value)
    assert asdict(config)["jepa_mix"] == value
    assert asdict(ExperimentConfig())["jepa_mix"] is None


def test_legacy_parameters_are_not_reinterpreted():
    with pytest.raises(ValueError, match="penalty_weight"):
        ExperimentConfig(jepa_mix=.5, penalty_weight=1.)
    with pytest.raises(ValueError, match="coupled_auxiliary"):
        ExperimentConfig(jepa_mix=.5, local_updates=5, coupled_auxiliary=True)
    assert ExperimentConfig().legacy_penalty_weight == 1.
    T, clones = tiny_problem(); x, a = ambiguous_stream()
    for method in ["short", "long", "posterior", "nll_control"]:
        with pytest.raises(ValueError, match="predictive"):
            train_experiment(T, clones, x, a, method, mixed(.5), [])


# --- endpoint and hybrid equivalence -----------------------------------------

@pytest.mark.parametrize("forgetting", [1., .99])
@pytest.mark.parametrize("inference", ["filter", "smooth"])
@pytest.mark.parametrize("updates", [2, 5])
def test_pure_em_is_exactly_short_em(forgetting, inference, updates):
    T, clones = tiny_problem(); x, a = ambiguous_stream()
    kwargs = dict(window=3, forgetting=forgetting, local_updates=updates, student_inference=inference)
    short = train_experiment(T, clones, x, a, "short", ExperimentConfig(**kwargs), range(len(x) + 1))
    pure = train_experiment(T, clones, x, a, "predictive", ExperimentConfig(jepa_mix=0, **kwargs), range(len(x) + 1))
    for step in short["snapshots"]:
        np.testing.assert_array_equal(pure["snapshots"][step], short["snapshots"][step])
    np.testing.assert_array_equal(pure["counts"], short["counts"])
    assert pure["work"] == short["work"]


def test_hybrid_matches_legacy_coupled_update_at_the_exact_weight_correspondence():
    """One arrival: rho = w / (<B,1> + w), w = eta <S~,1>, <B,1> = <S~,1> + kappa |T|."""
    T, clones = tiny_problem(); x, a = np.array([0, 1]), np.array([1])
    for eta, inference in [(1., "filter"), (.3, "smooth"), (4., "smooth")]:
        legacy = ExperimentConfig(window=3, local_updates=5, coupled_auxiliary=True, penalty_weight=eta,
                                  student_inference=inference, forgetting=1.)
        sum_S = 1.   # lambda = 1, S_1 = 0, <xi, 1> = 1
        rho = eta * sum_S / (sum_S + legacy.pseudocount * T.size + eta * sum_S)
        assert rho != pytest.approx(eta / (1 + eta))
        new = ExperimentConfig(window=3, local_updates=5, jepa_mix=rho, student_inference=inference, forgetting=1.)
        old_result = train_experiment(T, clones, x, a, "predictive", legacy, [2])
        new_result = train_experiment(T, clones, x, a, "predictive", new, [2])
        np.testing.assert_allclose(new_result["final_T"], old_result["final_T"], rtol=0, atol=1e-15)
        np.testing.assert_allclose(new_result["teacher_T"], old_result["teacher_T"], rtol=0, atol=1e-15)
        np.testing.assert_allclose(new_result["counts"], old_result["counts"], rtol=0, atol=1e-15)


def test_mixed_counts_reproduce_every_legacy_count_scale():
    rng = np.random.default_rng(1)
    T = row_normalize(rng.uniform(.1, 1, (2, 5, 5)))
    D = rng.uniform(0, 1, T.shape); D /= D.sum()
    for mass in [1e-3, 1., 37., 2e4]:
        B = mass * rng.dirichlet(np.ones(T.size)).reshape(T.shape) + .002
        weight = 1.3 * (B.sum() - .002 * T.size)
        np.testing.assert_allclose(mm_update(T, mixed_counts(B, D, legacy_jepa_mix(weight, B))),
                                   row_normalize(B + weight * D), rtol=0, atol=1e-15)


@pytest.mark.parametrize("rho", [.5, 1.])
@pytest.mark.parametrize("inference", ["filter", "smooth"])
def test_matches_direct_teacher_derived_mm_implementation(rho, inference):
    T, clones = tiny_problem(); x, a = stream_for(rho)
    config = mixed(rho, student_inference=inference, forgetting=1.)
    result = train_experiment(T, clones, x, a, "predictive", config, [len(x)])
    reference_T, reference_teacher, reference_S = direct_reference(T, clones, x, a, rho, config)
    np.testing.assert_allclose(result["final_T"], reference_T, rtol=0, atol=1e-13)
    np.testing.assert_allclose(result["teacher_T"], reference_teacher, rtol=0, atol=1e-13)
    if rho == 1:
        assert result["counts"] is None
    else:
        np.testing.assert_allclose(result["counts"], reference_S, rtol=0, atol=1e-13)


def test_pure_jepa_update_is_rownorm_of_teacher_counts_without_pseudocounts():
    T, clones = tiny_problem(); x, a = alternating_stream()
    base = train_experiment(T, clones, x, a, "predictive", mixed(1.), [len(x)])
    for changed in [dict(pseudocount=5.), dict(forgetting=.5), dict(forgetting=1.)]:
        other = train_experiment(T, clones, x, a, "predictive", mixed(1., **changed), [len(x)])
        np.testing.assert_array_equal(other["final_T"], base["final_T"])


# --- endpoint isolation ------------------------------------------------------

def count_calls(monkeypatch):
    calls = dict(pair_counts=0, replay_terms=0, smooth_window=0)
    originals = {name: getattr(module, name) for name in ["pair_counts", "replay_terms", "smooth_window"]}
    def pair(*args):
        calls["pair_counts"] += 1; return originals["pair_counts"](*args)
    def replay(*args):
        calls["replay_terms"] += 1; return originals["replay_terms"](*args)
    def smooth(T, groups, observations, actions, boundary):
        calls["smooth_window"] += 1
        return originals["smooth_window"](T, groups, observations, actions, boundary)
    monkeypatch.setattr(module, "pair_counts", pair)
    monkeypatch.setattr(module, "replay_terms", replay)
    monkeypatch.setattr(module, "smooth_window", smooth)
    return calls


@pytest.mark.parametrize("inference", ["filter", "smooth"])
def test_pure_em_never_computes_teacher_or_auxiliary_terms(monkeypatch, inference):
    T, clones = tiny_problem(); x, a = ambiguous_stream()
    calls = count_calls(monkeypatch)
    result = train_experiment(T, clones, x, a, "predictive", mixed(0., student_inference=inference), [len(x)])
    assert calls["replay_terms"] == 0
    assert calls["smooth_window"] == 0   # neither teacher targets nor JEPA source smoothing
    np.testing.assert_array_equal(result["teacher_T"], T)
    work = result["work"]
    assert work["teacher_updates"] == work["penalty_updates"] == 0
    assert work["statistics_commits"] == work["transitions"] == len(a)
    assert work["em_passes"] == 3 * len(a)
    assert result["diagnostics"] == []


def test_pure_jepa_never_computes_or_commits_em_counts(monkeypatch):
    T, clones = tiny_problem(); x, a = alternating_stream()
    calls = count_calls(monkeypatch)
    result = train_experiment(T, clones, x, a, "predictive", mixed(1.), range(len(x) + 1))
    assert calls["pair_counts"] == 0
    assert result["counts"] is None
    work = result["work"]
    assert work["statistics_commits"] == work["em_passes"] == 0
    # No EM warm-up: every local iteration is a JEPA iteration.
    assert work["penalty_updates"] == 3 * len(a)
    assert work["teacher_updates"] == len(a)
    for arrival in result["diagnostics"]:
        assert arrival["jepa_mix"] == 1.
        assert all("running_count_mass" not in it and it["jepa_mix"] == 1. for it in arrival["local_iterations"])


@pytest.mark.parametrize("rho", [.25, .75])
def test_hybrid_commits_only_data_counts(rho):
    T, clones = tiny_problem(); x, a = ambiguous_stream()
    config = mixed(rho, forgetting=1.)
    hybrid = train_experiment(T, clones, x, a, "predictive", config, [len(x)])
    assert hybrid["counts"].sum() == pytest.approx(len(a))   # unit mass per transition, no teacher mass
    work = hybrid["work"]
    assert work["em_passes"] == work["penalty_updates"] == 3 * len(a)
    assert work["statistics_commits"] == work["teacher_updates"] == len(a)
    # The committed statistics equal those of the direct reference, which never
    # adds D to S (test_matches_direct_teacher_derived_mm_implementation).
    np.testing.assert_allclose(hybrid["counts"], direct_reference(T, clones, x, a, rho, config)[2],
                               rtol=0, atol=1e-13)


# --- MM descent ----------------------------------------------------------------

@pytest.mark.parametrize("rho", MIXES)
def test_each_local_iteration_descends_its_fixed_objective(rho):
    """L_rho^(k)(T) = (1-rho) <-B^(k), log T>/<B^(k),1> + rho L_JEPA(T), B^(k) frozen."""
    T, clones = tiny_problem(); x, a = stream_for(rho)
    result = train_experiment(T, clones, x, a, "predictive", mixed(rho, local_updates=5), range(len(x) + 1))
    if rho == 0:
        assert result["diagnostics"] == []
        return
    assert len(result["diagnostics"]) == len(a)
    for arrival in result["diagnostics"]:
        assert len(arrival["local_iterations"]) == 5
        for iteration in arrival["local_iterations"]:
            assert iteration["objective_after"] <= iteration["objective_before"] + 1e-12


@pytest.mark.parametrize("rho", [0., .3, .7, 1.])
def test_mm_iterations_descend_the_fixed_arrival_objective(rho):
    """F_rho(T) = (1-rho) G_EM(T) / <B,1> + rho L(T) with the marginal G_EM is
    fixed over the arrival (<B,1> does not depend on k) and never increases."""
    T = tiny_problem()[0]
    # Frozen JEPA window: two transitions with fixed sources and teacher targets.
    beliefs = np.array([[.3, .7, 0, 0, 0], [0, 0, .2, .5, .3]])
    actions, ids = [0, 1], [np.array([2, 3, 4]), np.array([0, 1])]
    q = np.array([[0, 0, .1, .6, .3], [.8, .2, 0, 0, 0]])
    prior = np.random.default_rng(8).uniform(.01, .1, T.shape)
    f, action, dest = np.array([0, 0, .2, .5, .3]), 1, np.array([0, 1])
    mass = prior.sum() + 1.
    def F(tensor):
        jepa = replay_terms(tensor, beliefs, actions, ids, q)[0]
        if rho == 1:   # pure JEPA iterates may contain exact zeros outside the support
            return jepa
        em = -np.sum(prior * np.log(tensor)) - np.log((f @ tensor[action])[dest].sum())
        return (1 - rho) * em / mass + rho * jepa
    values = [F(T)]
    for _ in range(20):
        B = prior + pair_counts(T, f, action, dest)
        assert B.sum() == pytest.approx(mass)
        D = None if rho == 0 else replay_terms(T, beliefs, actions, ids, q)[3]
        T = mm_update(T, mixed_counts(B, D, rho))
        values.append(F(T))
    assert np.all(np.diff(values) <= 1e-12)
    assert values[-1] < values[0]


# --- support policies --------------------------------------------------------

def test_unsupported_rows_keep_the_current_distribution():
    T, clones = tiny_problem()
    counts = np.zeros_like(T)
    counts[0, 2, 3] = 2.; counts[0, 2, 4] = 2.
    updated = mm_update(T, counts)
    np.testing.assert_array_equal(updated[0, 2], [0, 0, 0, .5, .5])
    mask = np.ones(T.shape[:2], bool); mask[0, 2] = False
    np.testing.assert_array_equal(updated[mask], T[mask])
    # In training, pure JEPA leaves every row outside the window's source support untouched.
    x, a = np.array([0, 1]), np.array([1])
    result = train_experiment(T, clones, x, a, "predictive", mixed(1.), [2])
    touched = np.zeros(T.shape[:2], bool); touched[1, :2] = True
    np.testing.assert_array_equal(result["final_T"][~touched], T[~touched])
    assert np.all(result["final_T"][1, :2, :2] == 0) and np.all(result["final_T"][1, :2, 2:] > 0)


@pytest.mark.parametrize("rho", MIXES)
def test_episode_starts_and_empty_windows_make_no_update(rho):
    T, clones = tiny_problem(); x, a = ambiguous_stream()
    result = train_experiment(T, clones, x, a, "predictive", mixed(rho), range(len(x) + 1),
                              episode_starts=range(len(x)))
    for step in range(len(x) + 1):
        np.testing.assert_array_equal(result["snapshots"][step], T)
    np.testing.assert_array_equal(result["teacher_T"], T)
    assert all(value == 0 for key, value in result["work"].items())
    single = train_experiment(T, clones, x[:1], a[:0], "predictive", mixed(rho), [1])
    np.testing.assert_array_equal(single["final_T"], T)


def test_zero_predictive_support_is_raised_not_clipped():
    T, clones = tiny_problem()
    x, a = np.array([0, 1]), np.array([1])
    partial = T.copy(); partial[1, :2, 2] = 0; partial[1, :2] /= partial[1, :2].sum(-1, keepdims=True)
    # The teacher target is positive on clone 2, the student prediction is zero there.
    for rho in [.5, 1.]:
        with pytest.raises(ZeroPredictiveSupportError, match=r"arrival 1, iteration 0: .*1 destination clone"):
            train_experiment(partial, clones, x, a, "predictive", mixed(rho), [2])
    total = T.copy(); total[1, :2, 2:] = 0; total[1, :2] /= total[1, :2].sum(-1, keepdims=True)
    with pytest.raises(ZeroPredictiveSupportError, match="arrival 1: student filter"):
        train_experiment(total, clones, x, a, "predictive", mixed(1.), [2])
    assert issubclass(ZeroPredictiveSupportError, ValueError)


def test_pure_jepa_loses_support_for_successors_outside_the_window():
    """Analytic consequence of the unmodified objective, not a numerical artefact:
    rowNorm(D) places zero mass on every destination absent from the window."""
    T, clones = row_normalize(np.random.default_rng(0).uniform(.1, 1, (1, 3, 3))), np.array([1, 1, 1])
    x, a = np.array([0, 1, 0, 2]), np.array([0, 0, 0])
    with pytest.raises(ZeroPredictiveSupportError, match="arrival 3: student filter"):
        train_experiment(T, clones, x, a, "predictive", mixed(1., window=2), [4])
    # Every rho < 1 keeps the pseudocount and so full support.
    for rho in [0., .5, .99]:
        result = train_experiment(T, clones, x, a, "predictive", mixed(rho, window=2), [4])
        assert np.all(result["final_T"] > 0)


@pytest.mark.parametrize("rho", MIXES)
@pytest.mark.parametrize("inference", ["filter", "smooth"])
def test_every_snapshot_is_row_stochastic(rho, inference):
    T, clones = tiny_problem(); x, a = stream_for(rho)
    result = train_experiment(T, clones, x, a, "predictive", mixed(rho, student_inference=inference),
                              range(len(x) + 1), episode_starts=[5])
    for snapshot in result["snapshots"].values():
        assert_row_stochastic(snapshot)
        if rho < 1:
            assert np.all(snapshot > 0)
    assert_row_stochastic(result["teacher_T"])


# --- lifecycle -----------------------------------------------------------------

@pytest.mark.parametrize("rho", [.5, 1.])
def test_exactly_one_ema_update_per_arrival(rho):
    T, clones = tiny_problem(); x, a = stream_for(rho)
    config = mixed(rho, local_updates=5)
    result = train_experiment(T, clones, x, a, "predictive", config, range(len(x) + 1), episode_starts=[4])
    teacher, tau = T.copy(), 2 ** (-1 / config.teacher_half_life)
    for step in range(2, len(x) + 1):
        if step - 1 != 4:   # no transition arrives at the episode start
            teacher = tau * teacher + (1 - tau) * result["snapshots"][step]
    np.testing.assert_allclose(result["teacher_T"], teacher, rtol=0, atol=1e-15)
    assert result["work"]["teacher_updates"] == result["work"]["transitions"] == len(a) - 1


@pytest.mark.parametrize("rho", [.5, 1.])
@pytest.mark.parametrize("inference", ["filter", "smooth"])
def test_targets_and_sources_are_frozen_across_local_iterations(monkeypatch, rho, inference):
    T, clones = tiny_problem(); x, a = stream_for(rho)
    calls, original = [], module.replay_terms
    def capture(tensor, beliefs, actions, ids, targets):
        calls.append(([b.copy() for b in beliefs], np.array(targets).copy(), tensor.copy()))
        return original(tensor, beliefs, actions, ids, targets)
    monkeypatch.setattr(module, "replay_terms", capture)
    teacher_calls, smooth = [], module.smooth_window
    def capture_smoothing(tensor, groups, observations, actions, boundary):
        teacher_calls.append(len(observations))
        return smooth(tensor, groups, observations, actions, boundary)
    monkeypatch.setattr(module, "smooth_window", capture_smoothing)
    K = 4
    train_experiment(T, clones, x, a, "predictive", mixed(rho, local_updates=K, student_inference=inference), [])
    assert len(calls) == K * len(a)   # every iteration evaluates the JEPA term: no EM warm-up
    smoothing_per_arrival = 2 if inference == "smooth" else 1
    assert len(teacher_calls) == smoothing_per_arrival * len(a)   # targets built once per arrival
    for arrival in range(len(a)):
        group = calls[arrival * K:(arrival + 1) * K]
        for beliefs, targets, _ in group[1:]:
            np.testing.assert_array_equal(targets, group[0][1])
            for b, b0 in zip(beliefs, group[0][0]):
                np.testing.assert_array_equal(b, b0)
        # The student tensor does move between iterations.
        assert any(not np.array_equal(group[0][2], tensor) for _, _, tensor in group[1:])


@pytest.mark.parametrize("rho", [.5, 1.])
def test_future_observations_cannot_change_earlier_updates(rho):
    T, clones = tiny_problem(); x, a = stream_for(rho)
    config = mixed(rho, local_updates=4, student_inference="smooth")
    full = train_experiment(T, clones, x, a, "predictive", config, range(len(x) + 1))
    for k in range(2, len(x)):
        changed_a = a.copy(); changed_a[k - 1:] = 1 - changed_a[k - 1:]
        changed_x = x.copy()
        if rho < 1:
            changed_x[k:] = 1 - changed_x[k:]
        other = train_experiment(T, clones, changed_x, changed_a, "predictive", config, range(len(x) + 1))
        for step in range(k + 1):
            np.testing.assert_array_equal(other["snapshots"][step], full["snapshots"][step])
        prefix = train_experiment(T, clones, x[:k], a[:k - 1], "predictive", config, [k])
        np.testing.assert_array_equal(prefix["final_T"], full["snapshots"][k])


@pytest.mark.parametrize("rho", [.5, 1.])
def test_no_window_crosses_an_episode_reset(monkeypatch, rho):
    T, clones = tiny_problem(); x, a = stream_for(rho)
    reset, window, teacher_window = 5, 3, 4
    lengths, original = [], module.replay_terms
    def capture(tensor, beliefs, actions, ids, targets):
        lengths.append(len(actions))
        return original(tensor, beliefs, actions, ids, targets)
    monkeypatch.setattr(module, "replay_terms", capture)
    teacher_lengths, smooth = [], module.smooth_window
    def capture_smoothing(tensor, groups, observations, actions, boundary):
        teacher_lengths.append(len(observations))
        return smooth(tensor, groups, observations, actions, boundary)
    monkeypatch.setattr(module, "smooth_window", capture_smoothing)
    config = mixed(rho, local_updates=2, window=window, teacher_window=teacher_window)
    train_experiment(T, clones, x, a, "predictive", config, [], episode_starts=[reset])
    arrivals = [t for t in range(1, len(x)) if t != reset]
    start = lambda t: 0 if t < reset else reset
    assert lengths == [t - max(start(t), t - window + 1) for t in arrivals for _ in range(2)]
    assert teacher_lengths == [t - max(start(t), t - teacher_window + 1) + 1 for t in arrivals]


# --- legacy reproducibility ----------------------------------------------------

def test_legacy_configurations_reproduce_the_pre_change_golden_trajectories():
    import sys
    sys.path.insert(0, str(FIXTURES))
    try:
        from make_legacy_distillation_fixture import OUTPUT, cases, problem
    finally:
        sys.path.remove(str(FIXTURES))
    T, clones, x, a = problem()
    with np.load(OUTPUT) as gold:
        names = list(cases())
        assert len(gold.files) == 3 * len(names)
        for name, method, kwargs in names:
            result = train_experiment(T, clones, x, a, method, ExperimentConfig(**kwargs), range(len(x) + 1),
                                      episode_starts=[30])
            trajectory = np.stack([result["snapshots"][s] for s in range(len(x) + 1)])
            # Bit-identical except where rowNorm(B + w D) is now formed as the
            # equivalent rho_t mixture: those differ by floating-point rounding only.
            tolerance = 0 if method in {"short", "long", "posterior"} else 1e-14
            np.testing.assert_allclose(trajectory, gold[name + "|T"], rtol=0, atol=tolerance, err_msg=name)
            np.testing.assert_allclose(result["teacher_T"], gold[name + "|teacher"], rtol=0, atol=tolerance, err_msg=name)
            np.testing.assert_allclose(result["counts"], gold[name + "|counts"], rtol=0, atol=tolerance, err_msg=name)
