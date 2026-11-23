"""Research-only offline JEPA-CSCG: full-sequence EMA teacher, patch-smoothed student.

Actions are source aligned: actions[t] causes observations[t + 1]. Emissions are
deterministic (fixed clone allocation) and every smoothing pass starts from the
uniform clone prior, as in the vendored CHMM (Pi_x = 1 / n_states).

Training units
--------------
* Patch: contiguous observations x[s:e+1] owning transitions s..e-1.
* Tiling / data pass: a random-offset partition of every trajectory's transitions
  into blocks of at most L-1; each transition belongs to exactly one patch and
  adjacent patches share their boundary observation.
* Student EM iteration: E-step over every patch of the tiling, one global M-step.
* Coupled iteration: refresh teacher targets from the current EMA teacher, one
  student EM iteration, one EMA teacher update. J coupled iterations per pass.

Objective for one tiling (c = lam / (L - 1), one KL term per owned transition):

    L = sum_p [-log P_theta(X_p)] + c * sum_t KL(sg[r^T_{t+1}] || r^S_{t+1})
    r^T_{t+1} = P_teacher(z_t | X_{1:T}) A_teacher[a_t]   (full-trajectory smoothing)
    r^S_{t+1} = P_theta(z_t | X_p) A_theta[a_t]            (smoothing of t's patch only)

Because log r^S(j) = log P(X_p, z'=j) - log P(X_p) for an auxiliary branch
z' ~ A[a_t](z_t, .), patch p (m_p transitions) contributes, up to a constant,

    -(1 - c m_p) log P(X_p) - c * sum_t E_{j ~ r^T_t} log P(X_p, z'_t = j).

This is a nonnegative mixture of latent-variable log-likelihoods iff c m_p <= 1,
which holds for every patch when 0 <= lam <= 1 (m_p <= L - 1). Ordinary EM then
decreases the objective exactly, including the theta-dependence of the student's
smoothed beliefs. For lam > 1 the bound fails on full patches; it is rejected.
Note that the per-transition KL weight c depends on L at fixed lam.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
import time

import numpy as np

from .distillation import clone_groups, row_normalize, smooth_window

METHODS = ("offline", "patch", "ema_control", "jepa")


@dataclass(frozen=True)
class OfflineJepaConfig:
    patch_length: int = 40
    jepa_weight: float = 0.5      # lambda; per-transition KL weight is lambda / (L - 1)
    ema: float = 0.9              # beta, applied once per coupled iteration
    pseudocount: float = 0.002
    passes: int = 100             # E: distinct tilings / data passes
    coupled_iterations_per_pass: int = 1  # J

    def __post_init__(self):
        if self.patch_length < 2 or self.passes < 1 or self.pseudocount <= 0:
            raise ValueError("Require patch_length >= 2, passes >= 1 and a positive pseudocount")
        if self.coupled_iterations_per_pass < 1:
            raise ValueError("coupled_iterations_per_pass must be positive")
        if not 0 <= self.jepa_weight <= 1:
            raise ValueError("jepa_weight must lie in [0, 1]: above 1 a full patch gains "
                             "+(lambda-1) log P(X) and the EM update is no longer a descent step")
        if not 0 <= self.ema < 1:
            raise ValueError("ema must lie in [0, 1)")

    @property
    def kl_term_weight(self) -> float:
        return self.jepa_weight / (self.patch_length - 1)


def uniform_prior(n_states: int) -> np.ndarray:
    return np.full(n_states, 1 / n_states)


def sequence_counts(T, groups, x, actions):
    """Conventional E-step: summed pair posteriors (mass len(x) - 1), uniform prior."""
    _, mean_counts = smooth_window(T, groups, x, actions, uniform_prior(T.shape[1]))
    return mean_counts * (len(x) - 1)


def teacher_targets(teacher, groups, x, actions):
    """r^T_{t+1} = P(z_t | X_{1:T}) A[a_t] for every transition t of one trajectory."""
    q, _ = smooth_window(teacher, groups, x, actions, uniform_prior(teacher.shape[1]))
    targets = np.empty((len(actions), teacher.shape[1]))
    for t, a in enumerate(actions):
        ids = groups[int(x[t])]
        targets[t] = q[t, ids] @ teacher[int(a)][ids, :]
    return targets


def _forward_backward(T, groups, x, actions):
    ids = [groups[int(z)] for z in x]
    blocks = [T[int(a)][np.ix_(ids[u], ids[u + 1])] for u, a in enumerate(actions)]
    first = np.full(len(ids[0]), 1 / T.shape[1])
    log_likelihood = float(np.log(first.sum()))
    alpha = [first / first.sum()]
    for block in blocks:
        nxt = alpha[-1] @ block
        log_likelihood += float(np.log(nxt.sum()))
        alpha.append(nxt / nxt.sum())
    beta = [np.ones(len(i)) for i in ids]
    for u in reversed(range(len(blocks))):
        beta[u] = blocks[u] @ beta[u + 1]
        beta[u] /= beta[u].sum()
    return ids, blocks, alpha, beta, log_likelihood


def jepa_patch_statistics(T, groups, x, actions, targets, term_weight):
    """Exact EM counts and loss for one patch owning transitions 0..n-2.

    Loss: -log P(x) + term_weight * sum_t KL(targets[t] || q^S_t A[a_t]), one term
    per owned transition (the patch's last observation has no term: its outgoing
    transition belongs to the next patch). Requires term_weight * (n - 1) <= 1.

    For term t the E-step is the patch posterior reweighted at node t by
    w_t(i) = sum_j A[a_t]_{ij} r^T_t(j) / r^S_t(j); sum_i q^S_t(i) w_t(i) = 1, so
    every reweighted pair marginal is already normalised. Branch counts are
    D_t(i, j) = q^S_t(i) A_ij r^T_t(j) / r^S_t(j).
    """
    n, m = len(x), len(actions)
    if n < 2 or m != n - 1 or len(targets) != m:
        raise ValueError("Need n >= 2 observations, n - 1 actions and one target per action")
    if term_weight < 0 or term_weight * m > 1 + 1e-12:
        raise ValueError("Exact EM requires 0 <= term_weight * transitions <= 1")
    ids, blocks, alpha, beta, log_likelihood = _forward_backward(T, groups, x, actions)
    q = [a * b / np.sum(a * b) for a, b in zip(alpha, beta)]
    pairs = []
    for u, block in enumerate(blocks):
        pair = alpha[u][:, None] * block * beta[u + 1][None, :]
        pairs.append((1 - term_weight * m) * pair / pair.sum())
    counts = np.zeros_like(T)
    kl_sum = 0.0
    for t in range(m):
        a = int(actions[t])
        A = T[a][ids[t], :]
        prediction = q[t] @ A
        target = targets[t]
        positive = target > 0
        kl_sum += float(np.sum(target[positive] * (np.log(target[positive]) - np.log(prediction[positive]))))
        if term_weight == 0:
            continue
        ratio = np.where(positive, target / prediction, 0.0)
        counts[a][ids[t], :] += term_weight * q[t][:, None] * A * ratio[None, :]
        w = A @ ratio
        forward = alpha[t] * w
        for u in range(t, m):  # pairs at or after the reweighted node
            pair = forward[:, None] * blocks[u] * beta[u + 1][None, :]
            pairs[u] += term_weight * pair / pair.sum()
            forward = forward @ blocks[u]
            forward /= forward.sum()
        backward = beta[t] * w
        for u in reversed(range(t)):  # pairs before the reweighted node
            pair = alpha[u][:, None] * blocks[u] * backward[None, :]
            pairs[u] += term_weight * pair / pair.sum()
            backward = blocks[u] @ backward
            backward /= backward.sum()
    for u, a in enumerate(actions):
        counts[int(a)][np.ix_(ids[u], ids[u + 1])] += pairs[u]
    return counts, dict(nll=-log_likelihood, kl_sum=kl_sum, transitions=m,
                        loss=-log_likelihood + term_weight * kl_sum)


def tile_transitions(lengths, patch_length, rng):
    """Random-offset tiling: (trajectory, start, stop) observation slices.

    Transitions of each trajectory are cut at offset + i (L - 1), offset uniform in
    [0, L - 2]; every transition lies in exactly one patch and adjacent patches
    share their boundary observation. Patches never cross trajectories.
    """
    block = patch_length - 1
    tiling = []
    for k, n in enumerate(lengths):
        offset = int(rng.integers(block))
        cuts = sorted({0, n - 1, *range(offset, n - 1, block)})
        tiling += [(k, s, e + 1) for s, e in zip(cuts[:-1], cuts[1:])]
    return tiling


def train_offline(initial_T, n_clones, trajectories, method, config, seed, checkpoints,
                  callback: Callable | None = None):
    """E passes x J coupled iterations; one global M-step per iteration.

    Pass p uses tiling rng([seed, p]) for every patch arm (the offline arm treats
    each whole trajectory as its patch). Within a coupled iteration all patches are
    smoothed under the same student parameters; JEPA targets are recomputed from
    the current EMA teacher before every student E-step. Checkpoints are pass
    indices (0 = initial parameters).
    """
    if method not in METHODS:
        raise ValueError(f"Unknown method: {method}")
    for x, a in trajectories:
        if len(x) < 2 or len(a) != len(x) - 1:
            raise ValueError("Each trajectory needs n >= 2 observations and n - 1 actions")
    groups = clone_groups(n_clones)
    T = np.asarray(initial_T, dtype=float).copy()
    teacher = T.copy()
    lengths = [len(x) for x, _ in trajectories]
    c = config.kl_term_weight if method == "jepa" else 0.0
    cost = dict(passes=0, tilings=0, student_em_iterations=0, msteps=0, teacher_updates=0,
                teacher_refreshes=0, teacher_inference_passes=0, transition_presentations=0,
                kl_terms=0, student_patch_calls=0, student_seconds=0.0, teacher_seconds=0.0,
                student_block_entries=0, teacher_block_entries=0)
    snapshots, teachers, diagnostics = {}, {}, []

    def entries(x):
        return sum(len(groups[int(x[u])]) * len(groups[int(x[u + 1])]) for u in range(len(x) - 1))

    def save(index):
        if index in checkpoints:
            snapshots[index], teachers[index] = T.copy(), teacher.copy()
            if callback:
                callback(index, T.copy(), dict(cost))

    save(0)
    for p in range(config.passes):
        if method == "offline":
            tiling = [(k, 0, n) for k, n in enumerate(lengths)]
        else:
            tiling = tile_transitions(lengths, config.patch_length, np.random.default_rng([seed, p]))
            cost["tilings"] += 1
        for j in range(config.coupled_iterations_per_pass):
            targets = None
            if method == "jepa" and c > 0:
                started = time.perf_counter()
                targets = [teacher_targets(teacher, groups, x, a) for x, a in trajectories]
                cost["teacher_seconds"] += time.perf_counter() - started
                cost["teacher_refreshes"] += 1
                cost["teacher_inference_passes"] += len(trajectories)
                cost["teacher_block_entries"] += sum(
                    3 * entries(x) + sum(len(groups[int(z)]) for z in x[:-1]) * T.shape[1] for x, _ in trajectories)
            started = time.perf_counter()
            counts = np.zeros_like(T)
            nll = kl = 0.0
            for k, s, e in tiling:
                x, a = trajectories[k][0][s:e], trajectories[k][1][s:e - 1]
                if targets is None:
                    counts += sequence_counts(T, groups, x, a)
                    cost["student_block_entries"] += 3 * entries(x)
                else:
                    patch_counts, values = jepa_patch_statistics(T, groups, x, a, targets[k][s:e - 1], c)
                    counts += patch_counts
                    nll += values["nll"]; kl += values["kl_sum"]
                    cost["kl_terms"] += values["transitions"]
                    cost["student_block_entries"] += (3 + 2 * len(a)) * entries(x)
                cost["student_patch_calls"] += 1
                cost["transition_presentations"] += e - s - 1
            T = row_normalize(counts + config.pseudocount)
            cost["student_em_iterations"] += 1
            cost["msteps"] += 1
            cost["student_seconds"] += time.perf_counter() - started
            if method in ("ema_control", "jepa"):
                teacher = config.ema * teacher + (1 - config.ema) * T
                cost["teacher_updates"] += 1
            if targets is not None:
                transitions = sum(n - 1 for n in lengths)
                diagnostics.append(dict(pass_index=p + 1, iteration=j + 1,
                                        nll_per_transition=nll / transitions,
                                        mean_predictive_kl=kl / transitions,
                                        teacher_student_tv=float(np.abs(teacher - T).sum(-1).mean() / 2)))
        cost["passes"] += 1
        save(p + 1)
    return dict(snapshots=snapshots, teachers=teachers, diagnostics=diagnostics,
                final_T=T, teacher_T=teacher, cost=cost)
