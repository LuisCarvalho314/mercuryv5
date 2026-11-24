"""Accumulated predictive JEPA-CSCG: new counts C = (1 - rho) xi + rho D enter the
running statistics; T = rowNorm(lambda S + (1 - lambda) C + kappa)."""
from __future__ import annotations

import math
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from src.mercury_runs.algorithms.cscg import distillation as module
from src.mercury_runs.algorithms.cscg.distillation import (
    CarriedFilter, ExperimentConfig, clone_groups, data_nll, expected_counts, pair_counts, replay_terms,
    row_normalize, smooth_window, train_experiment,
)

MIXES = [0., .25, .5, .99, 1.]
FIXTURES = Path(__file__).resolve().parent / "fixtures"


def tiny_problem():
    rng = np.random.default_rng(7)
    return row_normalize(rng.uniform(.1, 1., (2, 5, 5))), np.array([2, 3])


def ambiguous_stream():
    """Random symbols: the same (symbol, action) has several successors."""
    return np.array([0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1]), np.array([0, 1, 0, 1, 1, 0, 1, 0, 0, 1])


def mixed(rho, **kwargs):
    return ExperimentConfig(jepa_mix=rho, **{"window": 3, "local_updates": 3, **kwargs})


def assert_row_stochastic(T):
    assert np.all(np.isfinite(T)) and np.all(T >= 0)
    np.testing.assert_allclose(T.sum(-1), 1, rtol=0, atol=1e-12)


def direct_reference(T0, clones, x, a, rho, config):
    """Independent loop implementation of the accumulated per-arrival MM recursion.

    D is accumulated entry by entry from r^u_ij q_{u+1}(j) / n, not via replay_terms.
    """
    groups = clone_groups(clones)
    T, teacher, S = T0.copy(), T0.copy(), np.zeros_like(T0)
    lam, kappa = config.forgetting, config.pseudocount
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
            C = np.zeros_like(T)
            if rho < 1:
                C += (1 - rho) * pair_counts(T, previous, a[t - 1], groups[z])
            if rho > 0:
                D, n = np.zeros_like(T), t - s
                for u in range(s, t):
                    b = source[u] if config.student_inference == "smooth" else fs[u]
                    action, J = a[u], groups[x[u + 1]]
                    for j in J:
                        p = sum(b[i] * T[action, i, j] for i in range(T.shape[1]))
                        for i in range(T.shape[1]):
                            D[action, i, j] += b[i] * T[action, i, j] / p * q[u + 1 - s, j] / n
                C += rho * D
            candidate = S + C if lam == 1 else lam * S + (1 - lam) * C
            B = candidate + kappa
            T = B / B.sum(-1, keepdims=True)
        S = candidate
        if rho > 0:
            teacher = tau * teacher + (1 - tau) * T
    return T, teacher, S


def capture_final_counts(monkeypatch):
    """Per call: (xi or None, D or None) as formed inside train_experiment."""
    calls, original = [], module.expected_counts
    def capture(X, D, rho):
        calls.append((None if X is None else X.copy(), None if D is None else D.copy()))
        return original(X, D, rho)
    monkeypatch.setattr(module, "expected_counts", capture)
    return calls


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


# --- count mass and accumulation ---------------------------------------------

def test_conventional_and_teacher_counts_have_unit_mass():
    T, clones = tiny_problem(); groups = clone_groups(clones)
    rng = np.random.default_rng(3)
    f = np.zeros(5); f[2:] = rng.dirichlet(np.ones(3))
    assert pair_counts(T, f, 1, groups[0]).sum() == pytest.approx(1, abs=1e-15)
    beliefs = [f, np.r_[rng.dirichlet(np.ones(2)), 0, 0, 0]]
    ids = [groups[0], groups[1]]
    targets = [np.r_[rng.dirichlet(np.ones(2)), 0, 0, 0], np.r_[0, 0, rng.dirichlet(np.ones(3))]]
    D = replay_terms(T, beliefs, [1, 0], ids, targets)[3]
    assert D.sum() == pytest.approx(1, abs=1e-15) and np.all(D >= 0)
    for rho in MIXES:
        X = pair_counts(T, f, 1, groups[0])
        assert expected_counts(X, D, rho).sum() == pytest.approx(1, abs=1e-15)
    assert expected_counts(X, None, 0.) is X and expected_counts(None, D, 1.) is D


@pytest.mark.parametrize("forgetting", [1., .9])
@pytest.mark.parametrize("inference", ["filter", "smooth"])
def test_pure_em_is_exactly_short_em(forgetting, inference):
    T, clones = tiny_problem(); x, a = ambiguous_stream()
    for updates in [2, 5]:
        kwargs = dict(window=3, forgetting=forgetting, local_updates=updates, student_inference=inference)
        short = train_experiment(T, clones, x, a, "short", ExperimentConfig(**kwargs), range(len(x) + 1))
        pure = train_experiment(T, clones, x, a, "predictive", ExperimentConfig(jepa_mix=0, **kwargs),
                                range(len(x) + 1))
        for step in short["snapshots"]:
            np.testing.assert_array_equal(pure["snapshots"][step], short["snapshots"][step])
        np.testing.assert_array_equal(pure["counts"], short["counts"])
        assert pure["work"] == short["work"]


@pytest.mark.parametrize("rho", [.5, .99, 1.])
@pytest.mark.parametrize("inference", ["filter", "smooth"])
@pytest.mark.parametrize("forgetting", [1., .9])
def test_matches_direct_accumulated_mm_implementation(rho, inference, forgetting):
    T, clones = tiny_problem(); x, a = ambiguous_stream()
    config = mixed(rho, student_inference=inference, forgetting=forgetting)
    result = train_experiment(T, clones, x, a, "predictive", config, [len(x)])
    reference_T, reference_teacher, reference_S = direct_reference(T, clones, x, a, rho, config)
    np.testing.assert_allclose(result["final_T"], reference_T, rtol=0, atol=1e-13)
    np.testing.assert_allclose(result["teacher_T"], reference_teacher, rtol=0, atol=1e-13)
    np.testing.assert_allclose(result["counts"], reference_S, rtol=0, atol=1e-13)


@pytest.mark.parametrize("rho", [.25, .5, .99, 1.])
def test_statistics_accumulate_the_final_iteration_mixture_once_per_arrival(monkeypatch, rho):
    """lambda = 1: S_n = sum_t [(1-rho) xi_t^(K) + rho D_t^(K)], one term per arrival."""
    T, clones = tiny_problem(); x, a = ambiguous_stream()
    K = 3
    calls = capture_final_counts(monkeypatch)
    result = train_experiment(T, clones, x, a, "predictive", mixed(rho, forgetting=1., local_updates=K), [])
    assert len(calls) == K * len(a)
    expected = np.zeros_like(T)
    for arrival in range(len(a)):
        X, D = calls[(arrival + 1) * K - 1]   # the final local iteration of this arrival
        assert (X is None) == (rho == 1) and D is not None
        expected += (0 if X is None else (1 - rho) * X) + rho * D
    np.testing.assert_allclose(result["counts"], expected, rtol=0, atol=1e-13)
    assert result["counts"].sum() == pytest.approx(len(a), abs=1e-12)   # unit mass per arrival, not K
    work = result["work"]
    assert work["statistics_commits"] == work["teacher_updates"] == work["transitions"] == len(a)


def test_pure_jepa_accumulates_only_teacher_counts(monkeypatch):
    T, clones = tiny_problem(); x, a = ambiguous_stream()
    original, xi_calls = module.pair_counts, []
    def pair(*args):
        xi_calls.append(1); return original(*args)
    monkeypatch.setattr(module, "pair_counts", pair)
    result = train_experiment(T, clones, x, a, "predictive", mixed(1.), range(len(x) + 1))
    assert xi_calls == []   # no observation-derived counts are formed
    work = result["work"]
    assert work["em_passes"] == 0 and work["penalty_updates"] == 3 * len(a)
    for arrival in result["diagnostics"]:
        assert np.sum(arrival["em_row_mass"]) == 0
        assert all(it["data_nll_after"] is None for it in arrival["local_iterations"])


@pytest.mark.parametrize("rho", [.5, 1.])
def test_forgetting_and_pseudocount_remain_active(rho):
    T, clones = tiny_problem(); x, a = ambiguous_stream()
    base = train_experiment(T, clones, x, a, "predictive", mixed(rho, forgetting=1.), [len(x)])
    forgetful = train_experiment(T, clones, x, a, "predictive", mixed(rho, forgetting=.9), [len(x)])
    prior = train_experiment(T, clones, x, a, "predictive", mixed(rho, forgetting=1., pseudocount=.5), [len(x)])
    assert not np.allclose(forgetful["final_T"], base["final_T"])
    assert not np.allclose(prior["final_T"], base["final_T"])
    # lambda < 1: mass recursion m <- lambda m + (1 - lambda), so m_n = 1 - lambda^n.
    assert forgetful["counts"].sum() == pytest.approx(1 - .9 ** len(a), abs=1e-12)


# --- support ---------------------------------------------------------------------

@pytest.mark.parametrize("rho", MIXES)
def test_pseudocount_preserves_strictly_positive_support(rho):
    """The stream that drove transient pure JEPA to zero support (a successor absent
    from the window) is now learned with T >= kappa / (row mass + |Z| kappa) > 0."""
    T, clones = row_normalize(np.random.default_rng(0).uniform(.1, 1, (1, 3, 3))), np.array([1, 1, 1])
    x, a = np.array([0, 1, 0, 2, 0, 1, 0, 2]), np.zeros(7, dtype=int)
    config = mixed(rho, window=2)
    result = train_experiment(T, clones, x, a, "predictive", config, range(1, len(x) + 1))
    S = result["counts"]
    bound = config.pseudocount / (S.sum(-1, keepdims=True) + T.shape[-1] * config.pseudocount)
    assert np.all(result["final_T"] >= bound * (1 - 1e-12))
    for snapshot in result["snapshots"].values():
        assert np.all(snapshot > 0)


@pytest.mark.parametrize("rho", MIXES)
@pytest.mark.parametrize("inference", ["filter", "smooth"])
def test_every_snapshot_is_finite_nonnegative_and_row_stochastic(rho, inference):
    T, clones = tiny_problem(); x, a = ambiguous_stream()
    result = train_experiment(T, clones, x, a, "predictive", mixed(rho, student_inference=inference),
                              range(len(x) + 1), episode_starts=[5])
    for snapshot in result["snapshots"].values():
        assert_row_stochastic(snapshot)
        assert np.all(snapshot > 0)
    assert_row_stochastic(result["teacher_T"])
    assert np.all(np.isfinite(result["counts"])) and np.all(result["counts"] >= 0)


# --- MM descent ----------------------------------------------------------------

@pytest.mark.parametrize("rho", MIXES[1:])
@pytest.mark.parametrize("forgetting", [1., .9])
def test_each_local_iteration_descends_the_per_arrival_objective(rho, forgetting):
    """F(T) = -<prior + kappa, log T> + inc [(1-rho) NLL_t(T) + rho L_JEPA(T)], same prior at every k."""
    T, clones = tiny_problem(); x, a = ambiguous_stream()
    result = train_experiment(T, clones, x, a, "predictive", mixed(rho, local_updates=5, forgetting=forgetting),
                              range(len(x) + 1))
    assert len(result["diagnostics"]) == len(a)
    for arrival in result["diagnostics"]:
        iterations = arrival["local_iterations"]
        assert len(iterations) == 5
        for previous, it in zip(iterations, iterations[1:]):
            assert it["objective_before"] == pytest.approx(previous["objective_after"], abs=1e-12)
        for it in iterations:
            assert it["objective_after"] <= it["objective_before"] + 1e-12


@pytest.mark.parametrize("rho", [0., .3, .99, 1.])
def test_mm_iterations_on_a_frozen_arrival_descend_monotonically(rho):
    T = tiny_problem()[0]
    beliefs = [np.array([.3, .7, 0, 0, 0]), np.array([0, 0, .2, .5, .3])]
    actions, ids = [0, 1], [np.array([2, 3, 4]), np.array([0, 1])]
    q = [np.array([0, 0, .1, .6, .3]), np.array([.8, .2, 0, 0, 0])]
    prior = np.random.default_rng(8).uniform(.01, .1, T.shape)   # historical statistics + pseudocount
    f, action, dest = np.array([0, 0, .2, .5, .3]), 1, np.array([0, 1])
    def F(tensor):
        return (-np.sum(prior * np.log(tensor)) + (1 - rho) * data_nll(tensor, f, action, dest)
                + rho * replay_terms(tensor, beliefs, actions, ids, q)[0])
    values = [F(T)]
    for _ in range(20):
        X = None if rho == 1 else pair_counts(T, f, action, dest)
        D = None if rho == 0 else replay_terms(T, beliefs, actions, ids, q)[3]
        T = row_normalize(prior + expected_counts(X, D, rho))
        values.append(F(T))
    assert np.all(np.diff(values) <= 1e-12)
    assert values[-1] < values[0]


# --- endpoint isolation and lifecycle -----------------------------------------------

@pytest.mark.parametrize("inference", ["filter", "smooth"])
def test_pure_em_never_computes_teacher_or_auxiliary_terms(monkeypatch, inference):
    T, clones = tiny_problem(); x, a = ambiguous_stream()
    calls = dict(replay=0, smooth=0)
    replay, smooth = module.replay_terms, module.smooth_window
    monkeypatch.setattr(module, "replay_terms", lambda *args: calls.__setitem__("replay", calls["replay"] + 1) or replay(*args))
    monkeypatch.setattr(module, "smooth_window", lambda *args: calls.__setitem__("smooth", calls["smooth"] + 1) or smooth(*args))
    result = train_experiment(T, clones, x, a, "predictive", mixed(0., student_inference=inference), [len(x)])
    assert calls == dict(replay=0, smooth=0)
    np.testing.assert_array_equal(result["teacher_T"], T)
    assert result["work"]["teacher_updates"] == result["work"]["penalty_updates"] == 0
    assert result["diagnostics"] == []


@pytest.mark.parametrize("rho", [.5, 1.])
def test_teacher_is_exclusively_ema_updated_once_per_arrival(rho):
    T, clones = tiny_problem(); x, a = ambiguous_stream()
    config = mixed(rho, local_updates=5)
    result = train_experiment(T, clones, x, a, "predictive", config, range(len(x) + 1), episode_starts=[4])
    teacher, tau = T.copy(), 2 ** (-1 / config.teacher_half_life)
    for step in range(2, len(x) + 1):
        if step - 1 != 4:   # no transition arrives at the episode start
            teacher = tau * teacher + (1 - tau) * result["snapshots"][step]
    np.testing.assert_allclose(result["teacher_T"], teacher, rtol=0, atol=1e-15)
    assert result["work"]["teacher_updates"] == result["work"]["transitions"] == len(a) - 1


@pytest.mark.parametrize("rho", MIXES)
def test_episode_starts_and_empty_windows_make_no_update(rho):
    T, clones = tiny_problem(); x, a = ambiguous_stream()
    result = train_experiment(T, clones, x, a, "predictive", mixed(rho), range(len(x) + 1),
                              episode_starts=range(len(x)))
    for step in range(len(x) + 1):
        np.testing.assert_array_equal(result["snapshots"][step], T)
    np.testing.assert_array_equal(result["teacher_T"], T)
    assert all(value == 0 for value in result["work"].values())
    assert not result["counts"].any()


@pytest.mark.parametrize("rho", [.5, 1.])
@pytest.mark.parametrize("inference", ["filter", "smooth"])
def test_targets_and_sources_are_frozen_across_local_iterations(monkeypatch, rho, inference):
    T, clones = tiny_problem(); x, a = ambiguous_stream()
    calls, original = [], module.replay_terms
    def capture(tensor, beliefs, actions, ids, targets):
        calls.append(([b.copy() for b in beliefs], np.array(targets).copy(), tensor.copy()))
        return original(tensor, beliefs, actions, ids, targets)
    monkeypatch.setattr(module, "replay_terms", capture)
    smoothing, smooth = [], module.smooth_window
    def capture_smoothing(tensor, groups, observations, actions, boundary):
        smoothing.append(len(observations))
        return smooth(tensor, groups, observations, actions, boundary)
    monkeypatch.setattr(module, "smooth_window", capture_smoothing)
    K = 4
    train_experiment(T, clones, x, a, "predictive", mixed(rho, local_updates=K, student_inference=inference), [])
    assert len(calls) == K * len(a)
    assert len(smoothing) == (2 if inference == "smooth" else 1) * len(a)   # targets built once per arrival
    for arrival in range(len(a)):
        group = calls[arrival * K:(arrival + 1) * K]
        for beliefs, targets, _ in group[1:]:
            np.testing.assert_array_equal(targets, group[0][1])
            for b, b0 in zip(beliefs, group[0][0]):
                np.testing.assert_array_equal(b, b0)
        assert any(not np.array_equal(group[0][2], tensor) for _, _, tensor in group[1:])


@pytest.mark.parametrize("rho", [.5, 1.])
def test_future_observations_cannot_change_earlier_updates(rho):
    T, clones = tiny_problem(); x, a = ambiguous_stream()
    config = mixed(rho, local_updates=4, student_inference="smooth")
    full = train_experiment(T, clones, x, a, "predictive", config, range(len(x) + 1))
    for k in range(2, len(x)):
        changed_a = a.copy(); changed_a[k - 1:] = 1 - changed_a[k - 1:]
        changed_x = x.copy(); changed_x[k:] = 1 - changed_x[k:]
        other = train_experiment(T, clones, changed_x, changed_a, "predictive", config, range(len(x) + 1))
        for step in range(k + 1):
            np.testing.assert_array_equal(other["snapshots"][step], full["snapshots"][step])
        prefix = train_experiment(T, clones, x[:k], a[:k - 1], "predictive", config, [k])
        np.testing.assert_array_equal(prefix["final_T"], full["snapshots"][k])


@pytest.mark.parametrize("rho", [.5, 1.])
def test_no_window_crosses_an_episode_reset(monkeypatch, rho):
    T, clones = tiny_problem(); x, a = ambiguous_stream()
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
            np.testing.assert_array_equal(trajectory, gold[name + "|T"], err_msg=name)
            np.testing.assert_array_equal(result["teacher_T"], gold[name + "|teacher"], err_msg=name)
            np.testing.assert_array_equal(result["counts"], gold[name + "|counts"], err_msg=name)


# --- recording-only diagnostics ------------------------------------------------

@pytest.mark.parametrize("rho", [.5, .99, 1.])
def test_diagnostics_decompose_the_arrival_objective(rho):
    T, clones = tiny_problem(); x, a = ambiguous_stream()
    metadata = {}
    config = mixed(rho, student_inference="smooth", forgetting=1.)
    result = train_experiment(T, clones, x, a, "predictive", config, range(len(x) + 1),
                              callback=lambda step, tensor, meta: metadata.update({step: meta}))
    for arrival in result["diagnostics"]:
        assert np.sum(arrival["em_row_mass"]) == pytest.approx(1 - rho, abs=1e-12)
        assert np.sum(arrival["jepa_row_mass"]) == pytest.approx(rho, abs=1e-12)
        assert np.sum(arrival["history_row_mass"]) == pytest.approx(
            arrival["step"] - 2 + T.size * config.pseudocount, abs=1e-12)
        assert arrival["min_window_student_prediction"] > 0 and arrival["min_window_teacher_target"] > 0
        for it in arrival["local_iterations"]:
            nll = 0. if it["data_nll_after"] is None else it["data_nll_after"]
            assert it["prior_term_after"] + (1 - rho) * nll + rho * it["jepa_term_after"] == pytest.approx(
                it["objective_after"], abs=1e-10)
        assert arrival["local_iterations"][-1]["jepa_term_after"] == pytest.approx(arrival["predictive_kl"], abs=1e-15)
    for step, meta in metadata.items():
        assert meta["min_T"] == pytest.approx(result["snapshots"][step].min())
        assert (meta["min_student_evidence"] is None) == (step <= 1)
    assert metadata[len(x)]["min_teacher_T"] == pytest.approx(result["teacher_T"].min())
