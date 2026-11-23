"""Filtered versus locally smoothed student source beliefs in online JEPA-CSCG."""
from __future__ import annotations

import itertools

import numpy as np
import pytest

from src.mercury_runs.algorithms.cscg import distillation as module
from src.mercury_runs.algorithms.cscg.distillation import (
    CarriedFilter, ExperimentConfig, clone_groups, pair_counts, replay_terms, row_normalize,
    smooth_window, softmax_rows, train_experiment,
)

METHODS = ["short", "long", "predictive", "posterior", "nll_control"]
WORK_KEYS = ["transitions", "base_msteps", "refinements", "em_passes", "penalty_updates",
             "statistics_commits", "teacher_updates"]


def tiny_problem():
    rng = np.random.default_rng(7)
    return row_normalize(rng.uniform(.1, 1., (2, 5, 5))), np.array([2, 3])


def stream():
    return np.array([0, 1, 1, 0, 1, 0, 0, 1]), np.array([0, 1, 0, 1, 1, 0, 1])


def capture_replay(monkeypatch):
    """Record (beliefs, actions, ids, targets) passed to every replay evaluation."""
    calls, original = [], module.replay_terms
    def wrapper(T, beliefs, actions, ids, targets):
        calls.append(dict(T=T.copy(), beliefs=[np.array(b) for b in beliefs], actions=list(actions),
                          ids=[np.array(i) for i in ids], targets=np.array(targets)))
        return original(T, beliefs, actions, ids, targets)
    monkeypatch.setattr(module, "replay_terms", wrapper)
    return calls


def test_invalid_student_inference_is_rejected():
    with pytest.raises(ValueError, match="student_inference"):
        ExperimentConfig(student_inference="smoothed")


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("schedule", [(2, False), (5, True)])
def test_filter_setting_is_the_unchanged_default(method, schedule):
    T, clones = tiny_problem(); x, a = stream()
    kwargs = dict(window=3, local_updates=schedule[0], coupled_auxiliary=schedule[1])
    default = train_experiment(T, clones, x, a, method, ExperimentConfig(**kwargs), range(9))
    explicit = train_experiment(T, clones, x, a, method, ExperimentConfig(student_inference="filter", **kwargs), range(9))
    for step in range(9):
        np.testing.assert_array_equal(default["snapshots"][step], explicit["snapshots"][step])
    np.testing.assert_array_equal(default["teacher_T"], explicit["teacher_T"])
    assert default["work"] == explicit["work"]
    assert default["diagnostics"] == explicit["diagnostics"]


def test_two_observation_smoothing_equals_the_short_data_term():
    """The data E-step already uses the smoothed pair posterior of x_{t-1:t}.

    So student smoothing is a no-op without a replay penalty (the 2x2 'without
    JEPA' row), and the smoothed source equals the source marginal of Xi_fast.
    """
    T, clones = tiny_problem(); groups = clone_groups(clones)
    f = CarriedFilter(clones); f.step(T, 0)
    q, counts = smooth_window(T, groups, np.array([0, 1]), np.array([1]), f.belief)
    fast = pair_counts(T, f.belief, 1, groups[1])
    np.testing.assert_allclose(counts, fast, atol=1e-15)
    np.testing.assert_allclose(q[0], fast[1].sum(axis=1), atol=1e-15)


@pytest.mark.parametrize("method", ["short", "long"])
@pytest.mark.parametrize("schedule", [(2, False), (5, True)])
def test_smoothing_has_no_effect_without_replay_penalty(method, schedule):
    T, clones = tiny_problem(); x, a = stream()
    kwargs = dict(window=3, local_updates=schedule[0], coupled_auxiliary=schedule[1])
    filt = train_experiment(T, clones, x, a, method, ExperimentConfig(**kwargs), [8])
    smooth = train_experiment(T, clones, x, a, method, ExperimentConfig(student_inference="smooth", **kwargs), [8])
    np.testing.assert_array_equal(filt["final_T"], smooth["final_T"])
    assert filt["work"] == smooth["work"]


def test_smoothed_source_conditions_on_the_newest_observation(monkeypatch):
    T, clones = tiny_problem(); groups = clone_groups(clones)
    x, a = stream()
    calls = capture_replay(monkeypatch)
    train_experiment(T, clones, x[:2], a[:1], "predictive", ExperimentConfig(student_inference="smooth"), [2])
    # Arrival 1 is the first transition: one replayed source, under the initial T.
    belief = calls[0]["beliefs"][0]
    labels = np.repeat(np.arange(len(clones)), clones)
    exact = np.zeros(T.shape[1])
    for z0, z1 in itertools.product(range(T.shape[1]), repeat=2):
        if labels[z0] == x[0] and labels[z1] == x[1]:
            exact[z0] += T[a[0], z0, z1] / T.shape[1]
    np.testing.assert_allclose(belief, exact / exact.sum(), atol=1e-15)
    filtered = np.zeros(T.shape[1]); filtered[groups[x[0]]] = 1 / clones[x[0]]
    assert np.abs(belief - filtered).sum() > 1e-3


def aliased_problem():
    """Symbol 0 has two clones; clone 0 leads to symbol 1, clone 1 to symbol 2."""
    clones = np.array([2, 1, 1])
    T = np.full((1, 4, 4), 1e-3)
    T[0, 0, 2] = T[0, 1, 3] = 1.
    T[0, 2, [0, 1]] = T[0, 3, [0, 1]] = .5
    return row_normalize(T), clones


def test_aliased_example_separates_filtered_and_smoothed_sources(monkeypatch):
    T, clones = aliased_problem()
    # First transition: the M-step rebuilds T from counts, so test before any update.
    x, a = np.array([0, 1]), np.array([0])
    beliefs = {}
    for mode in ["filter", "smooth"]:
        calls = capture_replay(monkeypatch)
        train_experiment(T, clones, x, a, "predictive", ExperimentConfig(student_inference=mode), [2])
        beliefs[mode] = calls[-1]["beliefs"][0]   # aliased source symbol 0
        monkeypatch.undo()
    np.testing.assert_allclose(beliefs["filter"][:2], [.5, .5], atol=1e-12)
    assert beliefs["smooth"][0] > .99    # x_2 = 1 reveals clone 0
    smooth_prediction = beliefs["smooth"] @ T[0]
    filter_prediction = beliefs["filter"] @ T[0]
    assert np.abs(smooth_prediction - filter_prediction).sum() / 2 > .45


def test_smoothed_student_never_reads_future_observations(monkeypatch):
    T, clones = tiny_problem(); x, a = stream()
    config = ExperimentConfig(window=3, student_inference="smooth", local_updates=5, coupled_auxiliary=True)
    full = train_experiment(T, clones, x, a, "predictive", config, range(9))
    for k in range(2, 8):
        changed = x.copy(); changed[k:] = 1 - changed[k:]
        other = train_experiment(T, clones, changed, a, "predictive", config, range(9))
        np.testing.assert_array_equal(other["snapshots"][k], full["snapshots"][k])
        prefix = train_experiment(T, clones, x[:k], a[:k - 1], "predictive", config, [k])
        np.testing.assert_array_equal(prefix["final_T"], full["snapshots"][k])
    # Every student smoothing call covers exactly x_{t-1:t} at arrival t.
    windows, original = [], module.smooth_window
    def spy(tensor, groups, observations, actions, boundary):
        windows.append(np.array(observations))
        return original(tensor, groups, observations, actions, boundary)
    monkeypatch.setattr(module, "smooth_window", spy)
    train_experiment(T, clones, x, a, "predictive", config, [8])
    # Two-node calls: student then teacher at t=1 (teacher window still short), then student only.
    two_node = [list(w) for w in windows if len(w) == 2]
    assert two_node == [list(x[0:2]), list(x[0:2])] + [list(x[t - 1:t + 1]) for t in range(2, 8)]


@pytest.mark.parametrize("mode", ["filter", "smooth"])
def test_student_and_teacher_predictive_distributions_are_normalised(monkeypatch, mode):
    T, clones = tiny_problem(); x, a = stream(); groups = clone_groups(clones)
    calls = capture_replay(monkeypatch)
    train_experiment(T, clones, x, a, "predictive", ExperimentConfig(window=4, student_inference=mode), [8])
    assert calls
    for call in calls:
        for b, action, ids, target in zip(call["beliefs"], call["actions"], call["ids"], call["targets"]):
            assert b.sum() == pytest.approx(1, abs=1e-12) and np.all(b >= 0)
            np.testing.assert_allclose((b @ call["T"][action]).sum(), 1, atol=1e-12)
            assert target.sum() == pytest.approx(1, abs=1e-12)
            outside = np.setdiff1d(np.arange(T.shape[1]), ids)
            assert np.all(target[outside] == 0)


def smoothed_replay_problem(monkeypatch):
    T, clones = tiny_problem(); x, a = stream()
    calls = capture_replay(monkeypatch)
    train_experiment(T, clones, x, a, "predictive", ExperimentConfig(window=4, student_inference="smooth"), [8])
    call = calls[-1]
    monkeypatch.undo()
    return call["T"], call["beliefs"], call["actions"], call["ids"], call["targets"]


def test_predictive_kl_value_gradient_and_mm_descent_with_smoothed_sources(monkeypatch):
    T, beliefs, actions, ids, q = smoothed_replay_problem(monkeypatch)
    pred, _, _, D, _ = replay_terms(T, beliefs, actions, ids, q)
    direct = np.mean([np.sum(t[i][t[i] > 0] * np.log(t[i][t[i] > 0] / (b @ T[act])[i][t[i] > 0]))
                      for b, act, i, t in zip(beliefs, actions, ids, q)])
    assert pred == pytest.approx(direct, abs=1e-13)
    # Detached-belief logit gradient N T - D against central differences.
    W = np.log(T); gradient = D.sum(-1, keepdims=True) * T - D; h = 1e-6
    for index in [(0, 0, 1), (1, 3, 2), (0, 4, 0), (1, 1, 4)]:
        plus, minus = W.copy(), W.copy(); plus[index] += h; minus[index] -= h
        numeric = (replay_terms(softmax_rows(plus), beliefs, actions, ids, q)[0]
                   - replay_terms(softmax_rows(minus), beliefs, actions, ids, q)[0]) / (2 * h)
        assert numeric == pytest.approx(gradient[index], abs=1e-8)
    B = np.random.default_rng(3).uniform(.01, .1, T.shape); weight = 2.3
    objective = lambda tensor: -np.sum(B * np.log(tensor)) + weight * replay_terms(tensor, beliefs, actions, ids, q)[0]
    start = row_normalize(B)
    D0 = replay_terms(start, beliefs, actions, ids, q)[3]
    assert objective(row_normalize(B + weight * D0)) < objective(start)


def test_coupled_smoothed_training_matches_explicit_reference():
    T, clones = tiny_problem(); x, aa = stream()
    initial, teacher, S = T.copy(), T.copy(), np.zeros_like(T)
    groups = clone_groups(clones)
    config = ExperimentConfig(window=4, forgetting=.99, local_updates=5, coupled_auxiliary=True,
                              student_inference="smooth")
    tau = 2 ** (-1 / config.teacher_half_life)
    f, h = CarriedFilter(clones), CarriedFilter(clones)
    hs, sources = [], {}
    for t, z in enumerate(x):
        previous = None if f.belief is None else f.belief.copy()
        f.step(T, z, None if t == 0 else aa[t - 1])
        hs.append(h.step(teacher, z, None if t == 0 else aa[t - 1])[1])
        if not t:
            continue
        # Independent smoothed source: f_{t-1} times the likelihood of x_t, pre-arrival T.
        likelihood = T[aa[t - 1]][:, groups[z]].sum(axis=1)
        sources[t - 1] = previous * likelihood / np.sum(previous * likelihood)
        s = max(0, t - config.window + 1)
        q, _ = smooth_window(teacher, groups, x[s:t + 1], aa[s:t], hs[s])
        prior = config.forgetting * S
        for _ in range(5):
            X = pair_counts(T, previous, aa[t - 1], groups[z])
            candidate = prior + (1 - config.forgetting) * X
            B, beta = candidate + config.pseudocount, candidate.sum()
            D = replay_terms(T, [sources[u] for u in range(s, t)], aa[s:t],
                             [groups[v] for v in x[s + 1:t + 1]], q[1:])[3]
            T = row_normalize(B + beta * D)
        S = candidate
        teacher = tau * teacher + (1 - tau) * T
    result = train_experiment(initial, clones, x, aa, "predictive", config, [8])
    np.testing.assert_allclose(result["final_T"], T, atol=1e-13, rtol=0)
    np.testing.assert_allclose(result["counts"], S, atol=1e-13, rtol=0)
    np.testing.assert_allclose(result["teacher_T"], teacher, atol=1e-13, rtol=0)
    assert S.sum() == pytest.approx(1 - config.forgetting ** len(aa))


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("schedule", [(2, False), (5, True)])
@pytest.mark.parametrize("forgetting", [1., .99])
def test_update_budgets_ema_and_count_decay_match_between_settings(method, schedule, forgetting):
    T, clones = tiny_problem(); x, a = stream()
    kwargs = dict(window=3, forgetting=forgetting, local_updates=schedule[0], coupled_auxiliary=schedule[1])
    results = {mode: train_experiment(T, clones, x, a, method, ExperimentConfig(student_inference=mode, **kwargs),
                                      range(9), episode_starts=[5]) for mode in ["filter", "smooth"]}
    for key in WORK_KEYS:
        assert results["filter"]["work"][key] == results["smooth"]["work"][key]
    tau = 2 ** (-1 / 40)
    expected_mass = 6 if forgetting == 1 else 1 - forgetting ** 6   # 7 transitions, one reset
    for result in results.values():
        assert result["counts"].sum() == pytest.approx(expected_mass)
        if method in {"predictive", "posterior", "nll_control"}:
            teacher = T.copy()
            for step in [2, 3, 4, 5, 7, 8]:   # arrivals with a transition
                teacher = tau * teacher + (1 - tau) * result["snapshots"][step]
            np.testing.assert_allclose(result["teacher_T"], teacher, atol=1e-14)
    if method in {"predictive", "posterior", "nll_control"}:
        assert np.abs(results["filter"]["final_T"] - results["smooth"]["final_T"]).max() > 1e-6
