"""Numerical checks for the offline JEPA-CSCG experiment (brute-force path references)."""
from __future__ import annotations

import itertools

import numpy as np
import pytest

from src.mercury_runs.algorithms.cscg import offline_jepa as oj
from src.mercury_runs.algorithms.cscg.distillation import clone_groups, row_normalize

CLONES = np.array([2, 3])
LABELS = np.repeat(np.arange(len(CLONES)), CLONES)
GROUPS = clone_groups(CLONES)


def random_T(seed, normalise=True):
    T = np.random.default_rng(seed).uniform(.1, 1., (2, 5, 5))
    return row_normalize(T) if normalise else T


def joint(T, x, actions):
    """Unnormalised P(z_{1:n}, x) for every label-consistent path (uniform prior)."""
    paths, probs = [], []
    for path in itertools.product(range(T.shape[1]), repeat=len(x)):
        if np.any(LABELS[list(path)] != x):
            continue
        p = 1 / T.shape[1]
        for u, a in enumerate(actions):
            p *= T[a, path[u], path[u + 1]]
        paths.append(path); probs.append(p)
    return np.array(paths), np.array(probs)


def smoothed(T, x, actions):
    paths, probs = joint(T, x, actions)
    q = np.zeros((len(x), T.shape[1]))
    for u in range(len(x)):
        np.add.at(q[u], paths[:, u], probs)
    return q / probs.sum(), probs.sum()


def reference_loss(T, x, actions, targets, c):
    """-log P(x) + c * sum_t KL(target_t || q_t A[a_t]) by enumeration."""
    q, likelihood = smoothed(T, x, actions)
    kl = 0.0
    for t, a in enumerate(actions):
        r = q[t] @ T[a]
        pos = targets[t] > 0
        kl += np.sum(targets[t][pos] * np.log(targets[t][pos] / r[pos]))
    return -np.log(likelihood) + c * kl, kl


def random_targets(seed, m):
    return np.random.default_rng(seed).dirichlet(np.ones(5), size=m)


X = np.array([0, 1, 1, 0, 1])
A = np.array([1, 0, 1, 1])
TERM_WEIGHTS = [0.0, 0.1, 0.25]   # c * 4 transitions <= 1; 0.25 is the boundary case


def test_teacher_targets_use_full_sequence_smoothing():
    T = random_T(1)
    q, _ = smoothed(T, X, A)
    expected = np.stack([q[t] @ T[a] for t, a in enumerate(A)])
    np.testing.assert_allclose(oj.teacher_targets(T, GROUPS, X, A), expected, atol=1e-14)
    np.testing.assert_allclose(expected.sum(1), 1, atol=1e-14)


@pytest.mark.parametrize("c", TERM_WEIGHTS)
def test_patch_loss_counts_one_kl_term_per_owned_transition(c):
    T = random_T(2)
    targets = random_targets(3, len(A))
    _, values = oj.jepa_patch_statistics(T, GROUPS, X, A, targets, c)
    loss, kl = reference_loss(T, X, A, targets, c)
    assert values["transitions"] == len(A)
    assert values["kl_sum"] == pytest.approx(kl, abs=1e-12)
    assert values["loss"] == pytest.approx(loss, abs=1e-12)
    with pytest.raises(ValueError):   # an outgoing action would double count a transition
        oj.jepa_patch_statistics(T, GROUPS, X, np.r_[A, 0], random_targets(3, 5), c)


@pytest.mark.parametrize("c", TERM_WEIGHTS)
def test_em_counts_equal_fisher_gradient_of_full_objective(c):
    """N_ij = T_ij * d(-L)/dT_ij, including the theta-dependence of student smoothing."""
    T = random_T(4, normalise=False)
    targets = random_targets(5, len(A))
    counts, _ = oj.jepa_patch_statistics(T, GROUPS, X, A, targets, c)
    gradient = np.zeros_like(T)
    h = 1e-6
    for index in np.ndindex(T.shape):
        up, down = T.copy(), T.copy()
        up[index] += h; down[index] -= h
        gradient[index] = -(reference_loss(up, X, A, targets, c)[0] - reference_loss(down, X, A, targets, c)[0]) / (2 * h)
    np.testing.assert_allclose(counts, T * gradient, atol=1e-7)
    assert counts.sum() == pytest.approx(len(A) * (1 + c), abs=1e-12)


def test_em_update_monotonically_decreases_tiling_objective():
    """A full patch and a short boundary patch share one per-transition weight."""
    kappa, lam, L = 0.01, 1.0, 5
    c = lam / (L - 1)
    patches = [(X, A, random_targets(6, 4)), (np.array([1, 0]), np.array([0]), random_targets(7, 1))]
    T = random_T(8)

    def objective(T):
        return sum(reference_loss(T, x, a, r, c)[0] for x, a, r in patches) - kappa * np.log(T).sum()

    values = [objective(T)]
    for _ in range(15):
        counts = sum(oj.jepa_patch_statistics(T, GROUPS, x, a, r, c)[0] for x, a, r in patches)
        T = row_normalize(counts + kappa)
        values.append(objective(T))
    assert np.all(np.diff(values) <= 1e-10)
    assert values[-1] < values[0]


def test_invalid_weights_are_rejected():
    with pytest.raises(ValueError, match="jepa_weight"):
        oj.OfflineJepaConfig(jepa_weight=1.5)
    with pytest.raises(ValueError, match="Exact EM"):
        oj.jepa_patch_statistics(random_T(0), GROUPS, X, A, random_targets(0, 4), 0.3)
    assert oj.OfflineJepaConfig(patch_length=40, jepa_weight=.5).kl_term_weight == pytest.approx(.5 / 39)


def test_zero_weight_recovers_patch_em_and_matched_teacher_gives_zero_kl():
    T = random_T(9)
    counts, _ = oj.jepa_patch_statistics(T, GROUPS, X, A, random_targets(15, 4), 0.0)
    np.testing.assert_allclose(counts, oj.sequence_counts(T, GROUPS, X, A), atol=1e-14)
    _, values = oj.jepa_patch_statistics(T, GROUPS, X, A, oj.teacher_targets(T, GROUPS, X, A), .2)
    assert values["kl_sum"] == pytest.approx(0.0, abs=1e-13)


def test_student_uses_only_patch_context_while_teacher_sees_outside():
    T = random_T(10)
    left, right = np.array([1, 0, 1, 0, 0, 1, 1]), np.array([0, 0, 1, 0, 0, 1, 0])
    actions = np.array([0, 1, 1, 0, 1, 0])
    s, e = 2, 5
    targets = random_targets(11, e - s - 1)
    a = [oj.jepa_patch_statistics(T, GROUPS, z[s:e], actions[s:e - 1], targets, .5) for z in (left, right)]
    np.testing.assert_allclose(a[0][0], a[1][0], atol=1e-15)
    assert a[0][1] == a[1][1]
    paths, probs = joint(T, left[s:e], actions[s:e - 1])
    pairs = np.zeros_like(T)
    for path, p in zip(paths, probs):
        for u, act in enumerate(actions[s:e - 1]):
            pairs[act, path[u], path[u + 1]] += p
    np.testing.assert_allclose(oj.sequence_counts(T, GROUPS, left[s:e], actions[s:e - 1]), pairs / probs.sum(), atol=1e-14)
    tl, tr = (oj.teacher_targets(T, GROUPS, z, actions)[s:e - 1] for z in (left, right))
    assert np.abs(tl - tr).max() > 1e-5


@pytest.mark.parametrize("lengths, L", [([9, 6], 4), ([9, 6, 2], 5), ([3], 10), ([40, 41], 40)])
def test_tiling_covers_every_transition_exactly_once(lengths, L):
    for p in range(20):
        tiling = oj.tile_transitions(lengths, L, np.random.default_rng([0, p]))
        for k, n in enumerate(lengths):
            owned = [t for kk, s, e in tiling if kk == k for t in range(s, e - 1)]
            assert sorted(owned) == list(range(n - 1))
            patches = sorted((s, e) for kk, s, e in tiling if kk == k)
            assert all(0 <= s < e <= n and 2 <= e - s <= L for s, e in patches)
            assert all(a[1] - 1 == b[0] for a, b in zip(patches, patches[1:]))  # shared boundary observation
    offsets = {oj.tile_transitions([50], 10, np.random.default_rng([0, p]))[0][2] for p in range(30)}
    assert len(offsets) > 1   # the partition offset is randomised between passes


SYNTHETIC = [(np.array([0, 1, 1, 0, 1, 0, 0, 1, 1]), np.array([1, 0, 1, 1, 0, 0, 1, 0])),
             (np.array([1, 0, 0, 1, 1, 0]), np.array([0, 1, 1, 0, 1]))]   # 8 + 5 = 13 transitions


def run(method, monkeypatch=None, calls=None, **kwargs):
    config = oj.OfflineJepaConfig(**{"patch_length": 4, "passes": 3, "coupled_iterations_per_pass": 2,
                                     "ema": .6, "jepa_weight": .9, **kwargs})
    if monkeypatch is not None:
        for name in ("sequence_counts", "jepa_patch_statistics", "teacher_targets", "row_normalize"):
            original = getattr(oj, name)

            def spy(*args, _name=name, _original=original):
                calls.append((_name, args))
                return _original(*args)
            monkeypatch.setattr(oj, name, spy)
    return oj.train_offline(random_T(13), CLONES, SYNTHETIC, method, config, seed=0,
                            checkpoints=set(range(config.passes + 1)))


@pytest.mark.parametrize("method", oj.METHODS)
def test_exact_budget_counts_on_synthetic_trajectories(method, monkeypatch):
    calls = []
    result = run(method, monkeypatch, calls)
    E, J, transitions = 3, 2, 13
    cost = result["cost"]
    assert cost["passes"] == E and cost["student_em_iterations"] == cost["msteps"] == E * J
    assert cost["transition_presentations"] == E * J * transitions
    assert cost["tilings"] == (0 if method == "offline" else E)
    assert cost["teacher_updates"] == (E * J if method in ("ema_control", "jepa") else 0)
    assert cost["teacher_refreshes"] == (E * J if method == "jepa" else 0)
    assert cost["teacher_inference_passes"] == (E * J * 2 if method == "jepa" else 0)
    assert cost["kl_terms"] == (E * J * transitions if method == "jepa" else 0)
    estep = "jepa_patch_statistics" if method == "jepa" else "sequence_counts"
    names = [n for n, _ in calls]
    assert names.count("row_normalize") == E * J
    assert cost["student_patch_calls"] == names.count(estep)
    # Between consecutive M-steps: (JEPA) one refresh per trajectory, then E-steps only,
    # all under the same parameters, which equal the previous M-step's output.
    current = random_T(13)
    segment = []
    for name, args in calls:
        if name != "row_normalize":
            segment.append((name, args)); continue
        if method == "jepa":
            assert [n for n, _ in segment[:2]] == ["teacher_targets"] * 2
            segment = segment[2:]
        assert segment and all(n == estep for n, _ in segment)
        for _, estep_args in segment:
            np.testing.assert_array_equal(estep_args[0], current)   # recomputed after each update, none between patches
        current = row_normalize(args[0])   # the M-step output (unpatched reference)
        segment = []
    assert not segment


def test_patch_arms_share_tilings_and_lambda_zero_reproduces_patch_cscg(monkeypatch):
    seen = {}
    for method in ("patch", "ema_control", "jepa"):
        calls = []
        run(method, monkeypatch, calls)
        monkeypatch.undo()
        estep = "jepa_patch_statistics" if method == "jepa" else "sequence_counts"
        seen[method] = [tuple(args[2]) for n, args in calls if n == estep]   # patch observation sequences
    assert seen["patch"] == seen["ema_control"] == seen["jepa"]
    patch, control, jepa0 = run("patch"), run("ema_control"), run("jepa", jepa_weight=0.0)
    for index in range(4):
        np.testing.assert_array_equal(control["snapshots"][index], patch["snapshots"][index])
        np.testing.assert_array_equal(jepa0["snapshots"][index], patch["snapshots"][index])


def test_ema_teacher_is_updated_once_per_coupled_iteration():
    """With J = 1 every snapshot is an M-step output, so the EMA recursion is checkable."""
    control = run("ema_control", coupled_iterations_per_pass=1)
    teacher = control["snapshots"][0]
    for index in range(1, 4):
        teacher = .6 * teacher + .4 * control["snapshots"][index]
        np.testing.assert_allclose(control["teachers"][index], teacher, atol=1e-15)
    np.testing.assert_array_equal(run("patch")["teachers"][3], random_T(13))


def test_targets_come_from_the_current_teacher_each_iteration(monkeypatch):
    calls = []
    result = run("jepa", monkeypatch, calls, coupled_iterations_per_pass=1)
    refreshes = [args[0] for n, args in calls if n == "teacher_targets"]
    assert len(refreshes) == 3 * 2   # E * J iterations x 2 trajectories
    for index in range(3):
        np.testing.assert_array_equal(refreshes[2 * index], result["teachers"][index])


def test_whole_trajectory_tiling_reproduces_offline_em(monkeypatch):
    config = oj.OfflineJepaConfig(patch_length=50, passes=2, coupled_iterations_per_pass=2)
    offline = oj.train_offline(random_T(14), CLONES, SYNTHETIC, "offline", config, 0, {2})
    monkeypatch.setattr(oj, "tile_transitions", lambda lengths, L, rng: [(k, 0, n) for k, n in enumerate(lengths)])
    patch = oj.train_offline(random_T(14), CLONES, SYNTHETIC, "patch", config, 0, {2})
    np.testing.assert_allclose(patch["snapshots"][2], offline["snapshots"][2], atol=1e-15)
    assert patch["cost"]["msteps"] == offline["cost"]["msteps"] == 4
