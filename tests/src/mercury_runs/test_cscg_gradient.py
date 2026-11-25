"""Gradient-based CSCG: analytic gradients, simplex projection and the controlled long-EM objective.

See docs/gradient_cscg_formulation.md.
"""
from __future__ import annotations

import itertools

import numpy as np
import pytest

from src.mercury_runs.algorithms.cscg import distillation as module
from src.mercury_runs.algorithms.cscg.distillation import (
    CarriedFilter, ExperimentConfig, NumericalInstabilityError, clone_groups, project_rows_to_simplex,
    projected_gradient_step, row_normalize, smooth_window, train_experiment, window_nll, window_nll_gradient,
)


def problem(seed=3):
    rng = np.random.default_rng(seed)
    clones = np.array([2, 3, 2])
    T = row_normalize(rng.uniform(.05, 1., (2, 7, 7)))
    x = np.array([0, 1, 2, 1, 0, 1, 1, 2])
    a = np.array([0, 1, 1, 0, 1, 0, 1])
    boundary = CarriedFilter(clones).batch(T, x[:1], np.zeros(1, int))[1][0]
    return T, clones, clone_groups(clones), x, a, boundary


def feasible_direction(T, rng):
    """Row sums zero; nonnegative where T = 0, so T + eps d stays on the simplex for small eps > 0."""
    d = rng.normal(size=T.shape)
    d[T == 0] = np.abs(d[T == 0])
    return d - d.sum(-1, keepdims=True) * np.where(T > 0, 1, 0) / np.maximum((T > 0).sum(-1, keepdims=True), 1)


def test_window_nll_equals_enumerated_conditional_likelihood():
    T, clones, groups, x, a, h = problem()
    labels = np.repeat(np.arange(len(clones)), clones)
    total = 0.
    for path in itertools.product(range(T.shape[1]), repeat=len(x)):
        if np.any(labels[list(path)] != x):
            continue
        p = h[path[0]]
        for u in range(len(a)):
            p *= T[a[u], path[u], path[u + 1]]
        total += p
    nll, _, _ = window_nll_gradient(T, groups, x, a, h)
    np.testing.assert_allclose(nll, -np.log(total) / (len(x) - 1), rtol=1e-12)
    np.testing.assert_allclose(window_nll(T, groups, x, a, h), nll, rtol=1e-14)


def test_gradient_count_identity_and_shared_forward_backward():
    """T * grad(window NLL) = -X: the gradient is the E-step count divided by T."""
    T, _, groups, x, a, h = problem()
    _, X, gradient = window_nll_gradient(T, groups, x, a, h)
    _, counts = smooth_window(T, groups, x, a, h)
    np.testing.assert_allclose(X, counts, atol=1e-15)
    np.testing.assert_allclose(T * gradient, -X, atol=1e-15)
    assert np.isclose(X.sum(), 1)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_window_gradient_matches_central_differences_along_feasible_directions(seed):
    T, _, groups, x, a, h = problem(seed)
    _, _, gradient = window_nll_gradient(T, groups, x, a, h)
    rng = np.random.default_rng(seed)
    for _ in range(5):
        d = feasible_direction(T, rng)
        np.testing.assert_allclose(d.sum(-1), 0, atol=1e-14)
        eps = 1e-6
        numeric = (window_nll(T + eps * d, groups, x, a, h) - window_nll(T - eps * d, groups, x, a, h)) / (2 * eps)
        np.testing.assert_allclose(np.sum(gradient * d), numeric, rtol=1e-6, atol=1e-9)


def test_window_gradient_is_finite_and_correct_at_exact_zeros():
    """Zero entries off the observed paths: one-sided differences into the simplex."""
    T, _, groups, x, a, h = problem(4)
    T = T.copy()
    T[0, 0, 2] = T[1, 3, 0] = T[1, 5, 6] = 0.
    T = row_normalize(T)
    _, X, gradient = window_nll_gradient(T, groups, x, a, h)
    assert np.all(np.isfinite(gradient))
    assert np.all(X[T == 0] == 0)
    assert np.any(gradient[T == 0] != 0)   # zero counts, nonzero gradient: X / T would be 0/0
    rng = np.random.default_rng(4)
    for _ in range(5):
        d = feasible_direction(T, rng)
        eps = 1e-7
        numeric = (window_nll(T + eps * d, groups, x, a, h) - window_nll(T, groups, x, a, h)) / eps
        np.testing.assert_allclose(np.sum(gradient * d), numeric, rtol=1e-4, atol=1e-7)


def test_simplex_projection_is_exact_idempotent_and_keeps_zeros():
    rng = np.random.default_rng(0)
    V = rng.normal(size=(3, 4, 6)) * 2
    P = project_rows_to_simplex(V)
    np.testing.assert_allclose(P.sum(-1), 1, atol=1e-12)
    assert P.min() >= 0 and np.any(P == 0)
    np.testing.assert_allclose(project_rows_to_simplex(P), P, atol=1e-15)
    # KKT: v - p = theta on the support, v <= theta off it.
    for v, p in zip(V.reshape(-1, 6), P.reshape(-1, 6)):
        theta = (v - p)[p > 0]
        np.testing.assert_allclose(theta, theta[0], atol=1e-12)
        assert np.all(v[p == 0] <= theta[0] + 1e-12)
    np.testing.assert_allclose(project_rows_to_simplex(np.array([[.2, .3, .5]])), [[.2, .3, .5]])
    np.testing.assert_array_equal(project_rows_to_simplex(np.array([[3., 0., 0.]])), [[1., 0., 0.]])


def arrival(T, groups, x, a, h, kappa=.002, seed=0):
    """Long-EM per-arrival objective with synthetic history S (same as train_experiment)."""
    S = np.random.default_rng(seed).uniform(0, 5, T.shape)
    history = S + kappa
    mass = history.sum() + 1
    def value(tensor):   # +inf where infeasible, as in train_experiment
        try:
            with np.errstate(divide="ignore"):
                return (-np.sum(history * np.log(tensor)) + window_nll(tensor, groups, x, a, h)) / mass
        except module.ZeroPredictiveSupportError:
            return np.inf
    nll, X, g = window_nll_gradient(T, groups, x, a, h)
    return value, (g - history / T) / mass, S + X + kappa


def test_arrival_gradient_matches_finite_differences_and_em_relation():
    T, _, groups, x, a, h = problem(5)
    value, gradient, B = arrival(T, groups, x, a, h)
    np.testing.assert_allclose(gradient, -B / T / (B.sum()), rtol=1e-12)   # grad G = -B / (M T)
    rng = np.random.default_rng(5)
    for _ in range(5):
        d = feasible_direction(T, rng)
        eps = 1e-6
        numeric = (value(T + eps * d) - value(T - eps * d)) / (2 * eps)
        np.testing.assert_allclose(np.sum(gradient * d), numeric, rtol=1e-6, atol=1e-10)


def test_projected_gradient_step_satisfies_armijo_and_plain_step_detects_instability():
    T, _, groups, x, a, h = problem(6)
    value, gradient, _ = arrival(T, groups, x, a, h)
    new, info = projected_gradient_step(T, value(T), gradient, value, learning_rate=100., max_backtracks=30)
    assert info["accepted"] and value(new) < value(T)
    np.testing.assert_allclose(new.sum(-1), 1, atol=1e-12)
    # A huge plain step drives entries to exactly zero; the log-barrier gradient is then infinite.
    plain, _ = projected_gradient_step(T, value(T), gradient, value, learning_rate=1e4, max_backtracks=0)
    assert np.any(plain == 0)
    infeasible = value(plain)
    assert infeasible == np.inf
    with pytest.raises(NumericalInstabilityError):
        projected_gradient_step(plain, infeasible, gradient, value, 1., 0)


def stream(seed=1, n=60):
    rng = np.random.default_rng(seed)
    clones = np.array([2, 3, 2])
    return (row_normalize(rng.uniform(.05, 1., (2, 7, 7))), clones,
            rng.integers(0, 3, n), rng.integers(0, 2, n - 1))


def test_training_gradient_arm_descends_the_shared_arrival_objective():
    T, clones, x, a = stream()
    checkpoints = [0, 20, 40, 60]
    config = ExperimentConfig(window=6, forgetting=1.0, local_updates=5, optimizer="projected_gradient",
                              gradient_learning_rate=10.)
    result = train_experiment(T, clones, x, a, "long", config, checkpoints)
    work = result["work"]
    assert work["gradient_steps"] == 5 * work["transitions"] and work["em_passes"] == 5 * work["transitions"]
    assert work["line_search_evaluations"] == work["gradient_steps"] + work["backtracks"]
    assert result["counts"].sum() == pytest.approx(work["transitions"])
    assert [d["step"] for d in result["objective_diagnostics"]] == checkpoints[1:]
    for record in result["objective_diagnostics"]:
        for iteration in record["local_iterations"]:
            assert iteration["objective_after"] <= iteration["objective_before"] + 1e-15
    final = result["final_T"]
    np.testing.assert_allclose(final.sum(-1), 1, atol=1e-12)
    assert final.min() > 0   # the pseudocount log-barrier keeps every entry positive


def test_em_arm_is_unchanged_and_records_the_same_objective():
    T, clones, x, a = stream()
    config = ExperimentConfig(window=6, forgetting=1.0, local_updates=5)
    result = train_experiment(T, clones, x, a, "long", config, [0, 30, 60])
    for record in result["objective_diagnostics"]:
        assert record["optimizer"] == "em"
        for iteration in record["local_iterations"]:   # EM is an MM method: monotone on G
            assert iteration["objective_after"] <= iteration["objective_before"] + 1e-15
    assert result["work"]["gradient_steps"] == 0
    # Reference: the explicit EM recursion with the unrefactored smoothing arithmetic.
    groups, S, Tref, filt = clone_groups(clones), np.zeros_like(T), T.copy(), CarriedFilter(clones)
    beliefs = []
    for t, z in enumerate(x):
        beliefs.append(filt.step(Tref, int(z), None if t == 0 else int(a[t - 1]))[1])
        if t == 0:
            continue
        s = max(0, t - 5)
        for _ in range(5):
            _, X = smooth_window(Tref, groups, x[s:t + 1], a[s:t], beliefs[s])
            Tref = row_normalize(S + X + .002)
        S = S + X
    np.testing.assert_array_equal(result["final_T"], Tref)


def test_gradient_optimizer_is_restricted_to_long_and_validated():
    T, clones, x, a = stream()
    with pytest.raises(ValueError, match="long"):
        train_experiment(T, clones, x, a, "short", ExperimentConfig(optimizer="projected_gradient"), [0])
    for bad in [dict(optimizer="sgd"), dict(optimizer="projected_gradient", gradient_learning_rate=0),
                dict(optimizer="projected_gradient", gradient_backtracks=-1),
                dict(optimizer="projected_gradient", jepa_mix=.5)]:
        with pytest.raises(ValueError):
            ExperimentConfig(**bad)


# --- PyTorch projected Adam (primary gradient arm) -------------------------------------------

torch = pytest.importorskip("torch")
from src.mercury_runs.algorithms.cscg import torch_gradient as tg  # noqa: E402


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_torch_inference_matches_numpy_forward_backward(seed):
    T, clones, groups, x, a, h = problem(seed)
    P = torch.tensor(T, requires_grad=True)
    nll = tg.window_nll(P, tg.clone_slices(clones), x, a, torch.tensor(h))
    nll.backward()
    reference, X, gradient = window_nll_gradient(T, groups, x, a, h)
    np.testing.assert_allclose(nll.item(), reference, rtol=1e-13)
    np.testing.assert_allclose(P.grad.numpy(), gradient, rtol=1e-11, atol=1e-14)   # autodiff == analytic
    np.testing.assert_allclose(-(T * P.grad.numpy()), smooth_window(T, groups, x, a, h)[1], atol=1e-14)


def test_torch_evaluate_matches_numpy_objective_and_counts():
    T, clones, groups, x, a, h = problem(5)
    S = np.random.default_rng(0).uniform(0, 5, T.shape)
    history = S + .002
    mass = history.sum() + 1
    arm = tg.ProjectedAdamCSCG(T, clones, learning_rate=1e-3)
    value, nll, X = arm.evaluate(history, 1., mass, x, a, h)
    np.testing.assert_allclose(value, (-np.sum(history * np.log(T)) + window_nll(T, groups, x, a, h)) / mass, rtol=1e-13)
    np.testing.assert_allclose(X, smooth_window(T, groups, x, a, h)[1], atol=1e-12)
    _, _, gradient = window_nll_gradient(T, groups, x, a, h)
    full = (gradient - history / T) / mass
    np.testing.assert_allclose(arm.P.grad.numpy(), full - full.mean(-1, keepdims=True), rtol=1e-9, atol=1e-15)


def test_torch_projection_matches_numpy():
    V = np.random.default_rng(1).normal(size=(3, 4, 6)) * 2
    np.testing.assert_allclose(tg.project_rows_to_simplex(torch.tensor(V)).numpy(), project_rows_to_simplex(V),
                               atol=1e-15)


def test_adam_step_is_the_conventional_adam_update_then_projection():
    T, clones, groups, x, a, h = problem(6)
    history = np.full(T.shape, .002)
    arm = tg.ProjectedAdamCSCG(T, clones, learning_rate=1e-4, epsilon=1e-8)
    arm.evaluate(history, 1., history.sum() + 1, x, a, h)
    g = arm.P.grad.numpy().copy()
    new, info = arm.step()
    m, v = .1 * g, .001 * g ** 2
    expected = project_rows_to_simplex(T - 1e-4 * (m / .1) / (np.sqrt(v / .001) + 1e-8))
    np.testing.assert_allclose(new, expected, atol=1e-15)
    assert info == dict(row_halvings=0, rejected_rows=0)


def test_adam_halves_only_rows_whose_projection_hits_a_barrier_zero():
    T, clones, groups, x, a, h = problem(7)
    history = np.full(T.shape, .002)
    arm = tg.ProjectedAdamCSCG(T, clones, learning_rate=.5)   # first Adam step moves every entry by 0.5
    arm.evaluate(history, 1., history.sum() + 1, x, a, h)
    new, info = arm.step()
    assert info["row_halvings"] > 0 and new.min() > 0
    np.testing.assert_allclose(new.sum(-1), 1, atol=1e-12)


def test_training_adam_arm_runs_and_commits_unit_mass_counts():
    T, clones, x, a = stream()
    config = ExperimentConfig(window=6, forgetting=1.0, local_updates=5, optimizer="projected_adam",
                              gradient_learning_rate=1e-3)
    result = train_experiment(T, clones, x, a, "long", config, [0, 30, 60])
    work = result["work"]
    assert work["gradient_steps"] == 5 * work["transitions"] and work["line_search_evaluations"] == 0
    assert result["counts"].sum() == pytest.approx(work["transitions"], rel=1e-9)
    final = result["final_T"]
    np.testing.assert_allclose(final.sum(-1), 1, atol=1e-12)
    assert final.min() > 0 and not np.array_equal(final, T)
    assert [d["optimizer"] for d in result["objective_diagnostics"]] == ["projected_adam"] * 2


# --- Reference-style gradCSCG (logits + softmax + log-space forward + Adam), primary arm --------

def test_reference_batched_forward_matches_numpy_on_padded_sequences():
    T, clones, groups, x, a, h = problem(8)
    windows = [(x, a), (x[:4], a[:3]), (x[2:7], a[2:6])]
    beliefs = [CarriedFilter(clones).batch(T, w[:1], np.zeros(1, int))[1][0] for w, _ in windows]
    state_obs = torch.from_numpy(np.repeat(np.arange(len(clones)), clones))
    width = max(len(w) for w, _ in windows)
    O = torch.zeros((3, width), dtype=torch.int64); A = torch.zeros((3, width), dtype=torch.int64)
    for b, (w, act) in enumerate(windows):
        O[b, :len(w)] = torch.as_tensor(w); A[b, :len(act)] = torch.as_tensor(act)
    L = torch.tensor([len(w) for w, _ in windows])
    log_alpha0 = torch.stack([tg.boundary_log_belief(h_b, state_obs, w[0]) for h_b, (w, _) in zip(beliefs, windows)])
    nll = tg.batch_neg_log_likelihood(torch.log(torch.tensor(T)), log_alpha0, O, A, L, state_obs)
    for b, ((w, act), h_b) in enumerate(zip(windows, beliefs)):
        np.testing.assert_allclose(nll[b].item(), window_nll(T, groups, w, act, h_b) * (len(w) - 1), rtol=1e-12)


def test_gradcscg_objective_counts_and_logit_gradient_match_numpy():
    T, clones, groups, x, a, h = problem(9)
    S = np.random.default_rng(1).uniform(0, 5, T.shape)
    history = S + .002
    mass = history.sum() + 1
    arm = tg.GradCSCG(T, clones, learning_rate=1e-2)
    np.testing.assert_allclose(torch.softmax(arm.W, -1).detach().numpy(), T, atol=1e-15)   # same initial T
    value, nll, X = arm.evaluate(history, 1., mass, x, a, h)
    np.testing.assert_allclose(value, (-np.sum(history * np.log(T)) + window_nll(T, groups, x, a, h)) / mass, rtol=1e-12)
    np.testing.assert_allclose(X, smooth_window(T, groups, x, a, h)[1], atol=1e-12)
    _, _, g_nll = window_nll_gradient(T, groups, x, a, h)
    g_T = (g_nll - history / T) / mass                                   # dG/dT (analytic)
    g_W = T * (g_T - np.sum(T * g_T, axis=-1, keepdims=True))            # softmax chain rule
    np.testing.assert_allclose(arm.W.grad.numpy(), g_W, rtol=1e-9, atol=1e-15)


def test_gradcscg_step_is_standard_keras_default_adam_on_logits():
    T, clones, groups, x, a, h = problem(10)
    history = np.full(T.shape, .002)
    arm = tg.GradCSCG(T, clones, learning_rate=1e-2)
    arm.evaluate(history, 1., history.sum() + 1, x, a, h)
    g = arm.W.grad.numpy().copy()
    new, info = arm.step()
    W1 = np.log(T) - 1e-2 * (.1 * g / .1) / (np.sqrt(.001 * g ** 2 / .001) + 1e-7)
    expected = np.exp(W1 - W1.max(-1, keepdims=True)); expected /= expected.sum(-1, keepdims=True)
    np.testing.assert_allclose(new, expected, atol=1e-14)
    assert info == {}


def test_training_gradcscg_arm_runs_with_shared_controls():
    T, clones, x, a = stream()
    config = ExperimentConfig(window=6, forgetting=1.0, local_updates=5, optimizer="gradcscg",
                              gradient_learning_rate=1e-2)
    result = train_experiment(T, clones, x, a, "long", config, [0, 30, 60])
    work = result["work"]
    assert work["gradient_steps"] == work["em_passes"] == 5 * work["transitions"]
    assert work["backtracks"] == work["rejected_steps"] == 0
    assert result["counts"].sum() == pytest.approx(work["transitions"], rel=1e-9)
    np.testing.assert_array_equal(result["snapshots"][0], T)
    final = result["final_T"]
    np.testing.assert_allclose(final.sum(-1), 1, atol=1e-12)
    assert final.min() > 0 and not np.array_equal(final, T)
    assert [d["optimizer"] for d in result["objective_diagnostics"]] == ["gradcscg"] * 2
