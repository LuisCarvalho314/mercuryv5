from __future__ import annotations

import itertools

import numpy as np
import pytest

from src.mercury_runs.algorithms.cscg.distillation import (
    CarriedFilter, ExperimentConfig, clone_groups, pair_counts, posterior_gradient,
    posterior_refine, replay_terms, row_normalize, smooth_window, softmax_rows,
    train_experiment,
)


def tiny_problem():
    rng = np.random.default_rng(7)
    return row_normalize(rng.uniform(.1, 1., (2, 5, 5))), np.array([2, 3])


@pytest.mark.parametrize("chunk", [1, 2, 3, 7, 40])
def test_filter_chunk_boundaries_equal_uninterrupted_and_enumeration(chunk):
    T, clones = tiny_problem()
    x = np.array([0, 1, 0, 1, 1, 0, 1])
    incoming = np.array([0, 1, 0, 0, 1, 1, 0])
    uninterrupted = CarriedFilter(clones).batch(T, x, incoming)
    stream = CarriedFilter(clones)
    batches = [stream.batch(T, x[s:s + chunk], incoming[s:s + chunk]) for s in range(0, len(x), chunk)]
    for k in range(3):
        np.testing.assert_allclose(np.concatenate([b[k] for b in batches]), uninterrupted[k], atol=1e-14)
    labels = np.repeat(np.arange(len(clones)), clones)
    for end in range(1, len(x) + 1):
        marginal = np.zeros(T.shape[1])
        for path in itertools.product(range(T.shape[1]), repeat=end):
            if np.any(labels[list(path)] != x[:end]):
                continue
            probability = 1 / T.shape[1]
            for u in range(1, end):
                probability *= T[incoming[u], path[u - 1], path[u]]
            marginal[path[-1]] += probability
        np.testing.assert_allclose(uninterrupted[1][end - 1], marginal / marginal.sum(), atol=1e-14)


def test_smoothing_and_auxiliary_counts_equal_enumerated_pairs():
    T, clones = tiny_problem()
    groups = clone_groups(clones)
    x = np.array([0, 1, 0, 1]); actions = np.array([1, 0, 1])
    _, f, _ = CarriedFilter(clones).batch(T, x, np.r_[0, actions])
    q, pairs = smooth_window(T, groups, x, actions, f[0])
    D = replay_terms(T, f[:-1], actions, [groups[z] for z in x[1:]], q[1:])[3]
    np.testing.assert_allclose(D, pairs, atol=1e-14)
    labels = np.repeat(np.arange(len(clones)), clones)
    exact = np.zeros_like(q)
    exact_pairs = np.zeros_like(T)
    total = 0.
    for path in itertools.product(range(T.shape[1]), repeat=len(x)):
        if np.any(labels[list(path)] != x):
            continue
        p = 1 / T.shape[1]
        for u in range(len(actions)):
            p *= T[actions[u], path[u], path[u + 1]]
        total += p
        for u, z in enumerate(path):
            exact[u, z] += p
        for u, a in enumerate(actions):
            exact_pairs[a, path[u], path[u + 1]] += p / len(actions)
    np.testing.assert_allclose(q, exact / total, atol=1e-14)
    np.testing.assert_allclose(pairs, exact_pairs / total, atol=1e-14)


def replay_problem():
    T, clones = tiny_problem()
    beliefs = np.array([[.3, .7, 0, 0, 0], [0, 0, .2, .5, .3]])
    actions = [0, 1]
    ids = [np.array([2, 3, 4]), np.array([0, 1])]
    q = np.array([[0, 0, .1, .6, .3], [.8, .2, 0, 0, 0]])
    return T, beliefs, actions, ids, q


def test_predictive_posterior_nll_identity_and_gradients():
    T, beliefs, actions, ids, q = replay_problem()
    pred, post, obs, D, O = replay_terms(T, beliefs, actions, ids, q)
    assert pred == pytest.approx(post + obs)
    pred_grad = D.sum(-1, keepdims=True) * T - D
    obs_grad = O.sum(-1, keepdims=True) * T - O
    post_grad = posterior_gradient(T, D, O)
    np.testing.assert_allclose(pred_grad, post_grad + obs_grad, atol=1e-14)
    eps = 1e-6
    numerical = np.zeros_like(T)
    for loss_index, analytic in [(0, pred_grad), (1, post_grad), (2, obs_grad)]:
        for idx in np.ndindex(T.shape):
            delta = np.zeros_like(T); delta[idx] = eps
            plus = replay_terms(softmax_rows(np.log(T) + delta), beliefs, actions, ids, q)[loss_index]
            minus = replay_terms(softmax_rows(np.log(T) - delta), beliefs, actions, ids, q)[loss_index]
            numerical[idx] = (plus - minus) / (2 * eps)
        np.testing.assert_allclose(analytic, numerical, atol=5e-9)
    assert D.sum() == pytest.approx(1.)
    assert O.sum() == pytest.approx(1.)


def test_predictive_auxiliary_ascent_and_posterior_gradient_descent():
    T, beliefs, actions, ids, q = replay_problem()
    B = np.random.default_rng(2).uniform(.01, .1, T.shape)
    base = row_normalize(B)
    weight = 1.7
    before = replay_terms(base, beliefs, actions, ids, q)
    updated = row_normalize(B + weight * before[3])
    objective = lambda tensor: -np.sum(B * np.log(tensor)) + weight * replay_terms(tensor, beliefs, actions, ids, q)[0]
    assert objective(updated) < objective(base)
    posterior, trials, old_loss, new_loss = posterior_refine(base, B, weight, beliefs, actions, ids, q)
    assert new_loss < old_loss
    assert 1 <= trials <= 8
    np.testing.assert_allclose(posterior.sum(-1), 1)


@pytest.mark.parametrize("forgetting", [1., .99])
@pytest.mark.parametrize("local_updates", [2, 5])
def test_original_short_expected_count_recurrence_and_zero_penalty_recovery(forgetting, local_updates):
    T, clones = tiny_problem()
    x = np.array([0, 1, 1, 0, 1]); actions = np.array([0, 1, 0, 1])
    config = ExperimentConfig(forgetting=forgetting, penalty_weight=0, local_updates=local_updates)
    result = train_experiment(T, clones, x, actions, "short", config, range(6))
    current = T.copy(); S = np.zeros_like(T); filt = CarriedFilter(clones)
    for t, z in enumerate(x):
        previous = filt.belief
        filt.step(current, z, None if t == 0 else actions[t - 1])
        if t:
            X = pair_counts(current, previous, actions[t - 1], clone_groups(clones)[z])
            prior = S.copy() if forgetting == 1 else forgetting * S
            scale = 1. if forgetting == 1 else 1 - forgetting
            local = current.copy()
            for _ in range(local_updates - 1):
                X = pair_counts(local, previous, actions[t - 1], clone_groups(clones)[z])
                local = row_normalize(prior + scale * X + config.pseudocount)
            X = pair_counts(local, previous, actions[t - 1], clone_groups(clones)[z])
            S = prior + scale * X
            current = row_normalize(S + config.pseudocount)
        np.testing.assert_allclose(result["snapshots"][t + 1], current)
    for method in ["predictive", "posterior", "nll_control"]:
        other = train_experiment(T, clones, x, actions, method, config, range(6))
        np.testing.assert_allclose(other["final_T"], result["final_T"], atol=1e-14)
    expected_mass = len(actions) if forgetting == 1 else 1 - forgetting ** len(actions)
    assert S.sum() == pytest.approx(expected_mass)


@pytest.mark.parametrize("local_updates", [2, 5])
def test_teacher_is_exclusively_ema_and_checkpoint_is_causal(local_updates):
    T, clones = tiny_problem()
    x = np.array([0, 1, 1, 0, 1, 0]); actions = np.array([0, 1, 0, 1, 0])
    config = ExperimentConfig(window=3, local_updates=local_updates)
    result = train_experiment(T, clones, x, actions, "posterior", config, range(7))
    teacher = T.copy(); tau = 2 ** (-1 / config.teacher_half_life)
    for step in range(2, 7):
        teacher = tau * teacher + (1 - tau) * result["snapshots"][step]
    np.testing.assert_allclose(result["teacher_T"], teacher, atol=1e-14)
    prefix = train_experiment(T, clones, x[:4], actions[:3], "posterior", config, [4])
    np.testing.assert_array_equal(prefix["final_T"], result["snapshots"][4])


@pytest.mark.parametrize("local_updates", [2, 5])
def test_reset_omits_cross_episode_transition_and_all_arms_keep_count_mass(local_updates):
    T, clones = tiny_problem()
    x = np.array([0, 1, 0, 1]); actions = np.array([0, 1, 0])
    for method in ["short", "long", "predictive", "posterior", "nll_control"]:
        result = train_experiment(T, clones, x, actions, method,
                                  ExperimentConfig(forgetting=1., local_updates=local_updates), [4], episode_starts=[2])
        assert result["counts"].sum() == pytest.approx(2.)
        assert result["work"]["transitions"] == 2
        work = result["work"]
        extra = method in {"predictive", "posterior", "nll_control"}
        assert work["statistics_commits"] == 2
        assert work["em_passes"] == (local_updates - int(extra)) * 2
        assert work["penalty_updates"] == int(extra) * 2
        assert work["teacher_updates"] == int(extra) * 2
        assert work["base_msteps"] + work["refinements"] == local_updates * 2
    filt = CarriedFilter(clones)
    filt.step(T, 0); filt.step(T, 1, 0); filt.reset()
    np.testing.assert_allclose(filt.step(T, 0)[1], [.5, .5, 0, 0, 0])


def test_local_refinements_improve_frozen_observed_likelihood_objectives():
    T, beliefs, actions, ids, q = replay_problem()
    # The new transition's expected counts are reestimated at the first local
    # M-step. The second step must improve the actual marginal objective, not
    # merely an arbitrary count objective.
    prior = np.random.default_rng(8).uniform(.01, .1, T.shape)
    b, action, dest = beliefs[0], actions[0], ids[0]
    fast = pair_counts(T, b, action, dest)
    base = row_normalize(prior + .3 * fast)
    refined_fast = pair_counts(base, b, action, dest)
    B = prior + .3 * refined_fast
    pred, post, obs, D, O = replay_terms(base, beliefs, actions, ids, q)
    def objective(tensor, kind):
        original = -np.sum(prior * np.log(tensor)) - .3 * np.log((b @ tensor[action])[dest].sum())
        return original + .8 * replay_terms(tensor, beliefs, actions, ids, q)[kind]
    for kind, counts in [(0, D), (2, O)]:
        trial = row_normalize(B + .8 * counts)
        assert objective(trial, kind) <= objective(base, kind) + 1e-12
    trial, _, _, _ = posterior_refine(base, B, .8, beliefs, actions, ids, q)
    assert objective(trial, 1) <= objective(base, 1) + 1e-12


@pytest.mark.parametrize("invalid", [0, 1, -1, 2.5, True])
def test_invalid_local_update_budget_is_rejected(invalid):
    with pytest.raises(ValueError, match="local_updates"):
        ExperimentConfig(local_updates=invalid)


@pytest.mark.parametrize("method", ["predictive", "posterior", "nll_control"])
@pytest.mark.parametrize("coupled", [False, True])
def test_five_update_teacher_targets_are_computed_once_before_local_updates(monkeypatch, method, coupled):
    from src.mercury_runs.algorithms.cscg import distillation as module
    T, clones = tiny_problem()
    target_calls, normalizations = [], []
    original_smooth, original_normalize = module.smooth_window, module.row_normalize
    def capture_target(tensor, *args):
        target_calls.append((len(normalizations), tensor.copy()))
        return original_smooth(tensor, *args)
    def capture_normalization(counts):
        normalizations.append(counts.copy())
        return original_normalize(counts)
    monkeypatch.setattr(module, "smooth_window", capture_target)
    monkeypatch.setattr(module, "row_normalize", capture_normalization)
    result = train_experiment(T, clones, np.array([0, 1, 0]), np.array([0, 1]), method,
                              ExperimentConfig(local_updates=5, forgetting=1., coupled_auxiliary=coupled), [1, 2, 3])
    assert len(target_calls) == 2
    count_per_arrival = (0 if coupled else 4) if method == "posterior" else 5
    assert [call[0] for call in target_calls] == [0, count_per_arrival]
    np.testing.assert_array_equal(target_calls[0][1], T)
    tau = 2 ** (-1 / 40)
    expected_teacher = tau * T + (1 - tau) * result["snapshots"][2]
    np.testing.assert_array_equal(target_calls[1][1], expected_teacher)
    assert result["work"]["em_passes"] == (10 if coupled else 8)
    assert result["work"]["penalty_updates"] == (10 if coupled else 2)
    assert result["work"]["statistics_commits"] == result["work"]["teacher_updates"] == 2


@pytest.mark.parametrize("method", ["predictive", "posterior", "nll_control"])
def test_coupled_iterations_match_explicit_latest_parameter_and_fixed_history_reference(method):
    T, clones = tiny_problem()
    initial = T.copy(); teacher = T.copy(); S = np.zeros_like(T)
    groups = clone_groups(clones)
    x, aa = np.array([0, 1, 0]), np.array([0, 1])
    config = ExperimentConfig(forgetting=1., local_updates=5, coupled_auxiliary=True)
    f, h = CarriedFilter(clones), CarriedFilter(clones)
    fs, hs = [], []
    tau = 2 ** (-1 / config.teacher_half_life)
    for t, z in enumerate(x):
        previous = f.belief
        fs.append(f.step(T, z, None if t == 0 else aa[t - 1])[1])
        hs.append(h.step(teacher, z, None if t == 0 else aa[t - 1])[1])
        if not t:
            continue
        q, _ = smooth_window(teacher, groups, x[:t + 1], aa[:t], hs[0])
        prior = S.copy()
        ids = [groups[v] for v in x[1:t + 1]]
        for _ in range(5):
            X = pair_counts(T, previous, aa[t - 1], groups[z])
            candidate = prior + X
            B, beta = candidate + config.pseudocount, candidate.sum()
            terms = replay_terms(T, fs[:-1], aa[:t], ids, q[1:])
            kind = {"predictive": 0, "posterior": 1, "nll_control": 2}[method]
            def actual_objective(tensor):
                likelihood = -np.log((previous @ tensor[aa[t - 1]])[groups[z]].sum())
                historical = -np.sum((prior + config.pseudocount) * np.log(tensor))
                auxiliary = replay_terms(tensor, fs[:-1], aa[:t], ids, q[1:])[kind]
                return historical + likelihood + beta * auxiliary
            before = actual_objective(T)
            if method == "posterior":
                T = posterior_refine(T, B, beta, fs[:-1], aa[:t], ids, q[1:])[0]
            else:
                T = row_normalize(B + beta * terms[3 if method == "predictive" else 4])
            assert actual_objective(T) <= before + 1e-10
        S = candidate
        teacher = tau * teacher + (1 - tau) * T
    result = train_experiment(initial, clones, x, aa, method, config, [1, 2, 3])
    np.testing.assert_array_equal(result["final_T"], T)
    np.testing.assert_array_equal(result["counts"], S)
    np.testing.assert_array_equal(result["teacher_T"], teacher)
    assert S.sum() == pytest.approx(2.)
    for arrival in result["diagnostics"]:
        assert len(arrival["local_iterations"]) == 5
        for iteration in arrival["local_iterations"]:
            assert iteration["running_count_mass"] == pytest.approx(arrival["step"] - 1)
            assert iteration["objective_after"] <= iteration["objective_before"] + 1e-10


@pytest.mark.parametrize("method", ["short", "long", "predictive", "posterior", "nll_control"])
def test_coupling_zero_weight_recovers_five_standard_em_iterations(method):
    T, clones = tiny_problem()
    x, actions = np.array([0, 1, 0, 1]), np.array([0, 1, 0])
    config = ExperimentConfig(forgetting=.99, local_updates=5, penalty_weight=0, coupled_auxiliary=True)
    reference_method = "long" if method == "long" else "short"
    baseline = train_experiment(T, clones, x, actions, reference_method, config, [4])
    result = train_experiment(T, clones, x, actions, method, config, [4])
    np.testing.assert_array_equal(result["final_T"], baseline["final_T"])
    assert result["work"]["em_passes"] == 15
    assert result["work"]["penalty_updates"] == 0
    assert result["work"]["statistics_commits"] == 3
