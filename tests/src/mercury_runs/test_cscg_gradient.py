"""Reference-style gradCSCG (PyTorch, Adam) on the shared long-EM objective, validated against NumPy.

See docs/gradient_cscg_formulation.md.
"""
from __future__ import annotations

import itertools

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.mercury_runs.algorithms.cscg import torch_gradient as tg  # noqa: E402
from src.mercury_runs.algorithms.cscg.distillation import (  # noqa: E402
    CarriedFilter, ExperimentConfig, clone_groups, data_nll, expected_counts, pair_counts, replay_terms,
    row_normalize, smooth_window, train_experiment, window_nll,
)


def problem(seed=3):
    rng = np.random.default_rng(seed)
    clones = np.array([2, 3, 2])
    T = row_normalize(rng.uniform(.05, 1., (2, 7, 7)))
    x = np.array([0, 1, 2, 1, 0, 1, 1, 2])
    a = np.array([0, 1, 1, 0, 1, 0, 1])
    boundary = CarriedFilter(clones).batch(T, x[:1], np.zeros(1, int))[1][0]
    return T, clones, clone_groups(clones), x, a, boundary


def stream(seed=1, n=60):
    rng = np.random.default_rng(seed)
    clones = np.array([2, 3, 2])
    return (row_normalize(rng.uniform(.05, 1., (2, 7, 7))), clones,
            rng.integers(0, 3, n), rng.integers(0, 2, n - 1))


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
    np.testing.assert_allclose(window_nll(T, groups, x, a, h), -np.log(total) / (len(x) - 1), rtol=1e-12)


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


def arrival(seed):
    T, clones, groups, x, a, h = problem(seed)
    history = np.random.default_rng(seed).uniform(0, 5, T.shape) + .002
    return T, clones, groups, x, a, h, history, history.sum() + 1


def numpy_objective(T, groups, x, a, h, history, mass):
    return (-np.sum(history * np.log(T)) + window_nll(T, groups, x, a, h)) / mass


@pytest.mark.parametrize("seed", [0, 9])
def test_gradcscg_objective_and_counts_match_numpy(seed):
    T, clones, groups, x, a, h, history, mass = arrival(seed)
    arm = tg.GradCSCG(T, clones, learning_rate=1e-2)
    np.testing.assert_allclose(torch.softmax(arm.W, -1).detach().numpy(), T, atol=1e-15)   # same initial T
    value, nll, X = arm.evaluate(history, 1., mass, arm.window_term(x, a, h))
    np.testing.assert_allclose(value, numpy_objective(T, groups, x, a, h, history, mass), rtol=1e-12)
    np.testing.assert_allclose(nll, window_nll(T, groups, x, a, h), rtol=1e-12)
    # E-step counts from the log-space gradient equal the NumPy forward-backward counts.
    np.testing.assert_allclose(X, smooth_window(T, groups, x, a, h)[1], atol=1e-12)


def test_gradcscg_logit_gradient_matches_central_differences():
    T, clones, groups, x, a, h, history, mass = arrival(5)
    arm = tg.GradCSCG(T, clones, learning_rate=1e-2)
    arm.evaluate(history, 1., mass, arm.window_term(x, a, h))
    gradient = arm.W.grad.numpy().copy()
    W = np.log(T)
    def G(logits):
        P = np.exp(logits - logits.max(-1, keepdims=True)); P /= P.sum(-1, keepdims=True)
        return numpy_objective(P, groups, x, a, h, history, mass)
    rng = np.random.default_rng(5)
    for _ in range(5):
        d = rng.normal(size=W.shape)
        eps = 1e-6
        numeric = (G(W + eps * d) - G(W - eps * d)) / (2 * eps)
        np.testing.assert_allclose(np.sum(gradient * d), numeric, rtol=1e-6, atol=1e-11)


def test_gradcscg_step_is_standard_keras_default_adam_on_logits():
    T, clones, groups, x, a, h = problem(10)
    history = np.full(T.shape, .002)
    arm = tg.GradCSCG(T, clones, learning_rate=1e-2)
    arm.evaluate(history, 1., history.sum() + 1, arm.window_term(x, a, h))
    g = arm.W.grad.numpy().copy()
    new = arm.step()
    W1 = np.log(T) - 1e-2 * (.1 * g / .1) / (np.sqrt(.001 * g ** 2 / .001) + 1e-7)
    expected = np.exp(W1 - W1.max(-1, keepdims=True)); expected /= expected.sum(-1, keepdims=True)
    np.testing.assert_allclose(new, expected, atol=1e-14)


def test_training_gradcscg_arm_runs_with_shared_controls():
    T, clones, x, a = stream()
    config = ExperimentConfig(window=6, forgetting=1.0, local_updates=5, optimizer="gradcscg",
                              gradient_learning_rate=1e-2)
    result = train_experiment(T, clones, x, a, "long", config, [0, 30, 60])
    work = result["work"]
    assert work["gradient_steps"] == work["em_passes"] == 5 * work["transitions"]
    assert result["counts"].sum() == pytest.approx(work["transitions"], rel=1e-9)
    np.testing.assert_array_equal(result["snapshots"][0], T)
    final = result["final_T"]
    np.testing.assert_allclose(final.sum(-1), 1, atol=1e-12)
    assert final.min() > 0 and not np.array_equal(final, T)
    assert [d["optimizer"] for d in result["objective_diagnostics"]] == ["gradcscg"] * 2


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
        train_experiment(T, clones, x, a, "short", ExperimentConfig(optimizer="gradcscg"), [0])
    for bad in [dict(optimizer="projected_gradient"), dict(optimizer="gradcscg", gradient_learning_rate=0),
                dict(optimizer="gradcscg", jepa_mix=0.), dict(optimizer="gradcscg", adam_beta1=1.)]:
        with pytest.raises(ValueError):
            ExperimentConfig(**bad)


@pytest.mark.parametrize("rho", [.5, 1.])
def test_gradcscg_jepa_term_and_counts_match_numpy_jepa_mix(rho):
    """Torch JEPA data term = (1 - rho) data_nll + rho replay predictive KL; counts = (1 - rho) xi + rho D."""
    T, clones, groups, x, a, h, history, mass = arrival(11)
    _, f, _ = CarriedFilter(clones).batch(T, x, np.r_[0, a])
    teacher = row_normalize(T + np.random.default_rng(2).uniform(0, .5, T.shape))
    targets = smooth_window(teacher, groups, x, a, f[0])[0][1:]
    beliefs, ids = f[:-1], [groups[int(z)] for z in x[1:]]
    previous, incoming, newest = f[-2], int(a[-1]), int(x[-1])
    arm = tg.GradCSCG(T, clones, learning_rate=1e-2)
    value, data, C = arm.evaluate(history, .7, mass, arm.jepa_term(rho, previous, incoming, newest, beliefs, a, targets))
    predictive, _, _, D, _ = replay_terms(T, beliefs, a, ids, targets)
    nll = data_nll(T, previous, incoming, groups[newest])
    np.testing.assert_allclose(data, (1 - rho) * nll + rho * predictive, rtol=1e-11)
    np.testing.assert_allclose(value, (-np.sum(history * np.log(T)) + .7 * data) / mass, rtol=1e-12)
    xi = None if rho == 1 else pair_counts(T, previous, incoming, groups[newest])
    np.testing.assert_allclose(C, expected_counts(xi, D, rho), atol=1e-11)


def test_training_gradcscg_jepa_arm_runs_with_teacher_and_shared_controls():
    T, clones, x, a = stream()
    common = dict(window=6, teacher_window=6, forgetting=1.0, local_updates=5, jepa_mix=.5,
                  student_inference="smooth")
    em = train_experiment(T, clones, x, a, "predictive", ExperimentConfig(**common), [0, 30, 60])
    grad = train_experiment(T, clones, x, a, "predictive",
                            ExperimentConfig(**common, optimizer="gradcscg", gradient_learning_rate=1e-2), [0, 30, 60])
    work = grad["work"]
    assert work["gradient_steps"] == work["penalty_updates"] == 5 * work["transitions"]
    assert work["teacher_updates"] == em["work"]["teacher_updates"] == work["transitions"]
    assert grad["counts"].sum() == pytest.approx(work["transitions"], rel=1e-9)
    np.testing.assert_array_equal(grad["snapshots"][0], em["snapshots"][0])
    assert not np.allclose(grad["final_T"], em["final_T"])   # only the optimiser differs
    np.testing.assert_allclose(grad["teacher_T"].sum(-1), 1, atol=1e-12)
    assert [d["step"] for d in grad["diagnostics"]] == [30, 60]


def test_gradcscg_jepa_training_does_not_depend_on_checkpoint_diagnostics():
    """Counts come from the gradient; NumPy counts are formed only at checkpoints, for diagnostics."""
    T, clones, x, a = stream()
    config = ExperimentConfig(window=6, teacher_window=6, forgetting=1.0, local_updates=5, jepa_mix=.99,
                              student_inference="smooth", optimizer="gradcscg", gradient_learning_rate=1e-2)
    sparse = train_experiment(T, clones, x, a, "predictive", config, [0, len(x)])
    dense = train_experiment(T, clones, x, a, "predictive", config, list(range(len(x) + 1)))
    for key in ("final_T", "counts", "teacher_T"):
        np.testing.assert_array_equal(sparse[key], dense[key])
    assert sparse["work"]["penalty_updates"] == dense["work"]["penalty_updates"] == 5 * sparse["work"]["transitions"]


def test_gini_regulariser_value_gradient_and_unchanged_counts():
    T, clones, groups, x, a, h, history, mass = arrival(12)
    lam = .1
    plain = tg.GradCSCG(T, clones, learning_rate=1e-2)
    reg = tg.GradCSCG(T, clones, learning_rate=1e-2, gini_weight=lam)
    v0, _, C0 = plain.evaluate(history, 1., mass, plain.window_term(x, a, h))
    v1, _, C1 = reg.evaluate(history, 1., mass, reg.window_term(x, a, h))
    gini = np.mean(1 - (T ** 2).sum(-1))                       # mean over the |A| K rows
    np.testing.assert_allclose(v1, v0 + lam * gini, rtol=1e-12)
    np.testing.assert_allclose(C1, C0, atol=1e-13)             # counts are the shared objective's
    gradient = reg.W.grad.numpy().copy()
    W = np.log(T)
    def total(logits):
        P = np.exp(logits - logits.max(-1, keepdims=True)); P /= P.sum(-1, keepdims=True)
        return numpy_objective(P, groups, x, a, h, history, mass) + lam * np.mean(1 - (P ** 2).sum(-1))
    rng = np.random.default_rng(12)
    for _ in range(5):
        d = rng.normal(size=W.shape)
        eps = 1e-6
        numeric = (total(W + eps * d) - total(W - eps * d)) / (2 * eps)
        np.testing.assert_allclose(np.sum(gradient * d), numeric, rtol=1e-6, atol=1e-11)


def test_gini_weight_zero_is_the_current_implementation_and_em_rejects_it():
    T, clones, x, a = stream()
    common = dict(window=6, forgetting=1.0, local_updates=5, optimizer="gradcscg", gradient_learning_rate=1e-2)
    default = train_experiment(T, clones, x, a, "long", ExperimentConfig(**common), [0, len(x)])
    zero = train_experiment(T, clones, x, a, "long", ExperimentConfig(**common, gini_weight=0.), [0, len(x)])
    np.testing.assert_array_equal(default["final_T"], zero["final_T"])
    sparse = train_experiment(T, clones, x, a, "long", ExperimentConfig(**common, gini_weight=1.), [0, len(x)])
    gini = lambda M: np.mean(1 - (M ** 2).sum(-1))
    assert gini(sparse["final_T"]) < gini(default["final_T"])   # the regulariser sparsifies T
    for bad in [dict(gini_weight=.1), dict(optimizer="gradcscg", gini_weight=-1.)]:
        with pytest.raises(ValueError):
            ExperimentConfig(**bad)
