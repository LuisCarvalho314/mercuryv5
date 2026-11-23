"""Research-only online CSCG: carried filters and frozen-window KL penalties.

Actions here are source aligned: actions[u] causes observations[u + 1].
The production online runner and the external CSCG implementation are untouched.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
import time

import numpy as np


def row_normalize(counts: np.ndarray) -> np.ndarray:
    totals = counts.sum(axis=-1, keepdims=True)
    if np.any(totals <= 0) or np.any(counts < 0):
        raise ValueError("Transition rows require positive mass and nonnegative counts")
    return counts / totals


def clone_groups(n_clones: np.ndarray) -> list[np.ndarray]:
    if np.any(np.asarray(n_clones) <= 0):
        raise ValueError("Each observation must have at least one clone")
    offsets = np.r_[0, np.cumsum(n_clones)]
    return [np.arange(offsets[i], offsets[i + 1]) for i in range(len(n_clones))]


class CarriedFilter:
    """Chunk boundaries do not reset beliefs or omit the crossing transition."""

    def __init__(self, n_clones: np.ndarray, prior: np.ndarray | None = None):
        self.groups = clone_groups(n_clones)
        self.n_states = int(np.sum(n_clones))
        self.prior = (np.full(self.n_states, 1 / self.n_states) if prior is None
                      else np.asarray(prior, dtype=float).copy())
        if self.prior.shape != (self.n_states,) or np.any(self.prior < 0) or self.prior.sum() <= 0:
            raise ValueError("Invalid initial state prior")
        self.prior /= self.prior.sum()
        self.belief: np.ndarray | None = None

    def reset(self) -> None:
        self.belief = None

    def step(self, T: np.ndarray, observation: int, incoming_action: int | None = None):
        if self.belief is None:
            prediction = self.prior.copy()
        else:
            if incoming_action is None:
                raise ValueError("A continuing filter requires the incoming action")
            active = np.flatnonzero(self.belief)
            prediction = self.belief[active] @ T[incoming_action, active, :]
        ids = self.groups[int(observation)]
        mass = float(prediction[ids].sum())
        if mass <= 0 or not np.isfinite(mass):
            raise ValueError("Observation has zero or invalid predictive probability")
        belief = np.zeros(self.n_states)
        belief[ids] = prediction[ids] / mass
        self.belief = belief
        return prediction, belief.copy(), -np.log(mass)

    def batch(self, T: np.ndarray, observations: np.ndarray, incoming_actions: np.ndarray):
        """One incoming action per observation; the first is ignored after reset."""
        if len(observations) != len(incoming_actions):
            raise ValueError("Observations and incoming actions must have equal length")
        values = [self.step(T, int(x), int(a)) for x, a in zip(observations, incoming_actions)]
        if not values:
            return np.empty((0, self.n_states)), np.empty((0, self.n_states)), np.empty(0)
        p, f, losses = zip(*values)
        return np.asarray(p), np.asarray(f), np.asarray(losses)


def pair_counts(T, source_belief, action, destination_ids):
    sources = np.flatnonzero(source_belief)
    block = source_belief[sources, None] * T[action][np.ix_(sources, destination_ids)]
    total = block.sum()
    if total <= 0:
        raise ValueError("Transition has zero predictive probability")
    result = np.zeros_like(T)
    result[action][np.ix_(sources, destination_ids)] = block / total
    return result


def smooth_window(T, groups, observations, actions, boundary):
    """Exact within-window smoothing, conditional on a historical filtered boundary.

    Returns full-state marginals and the mean pair count (unit total mass).
    """
    n = len(observations)
    if n < 2 or len(actions) != n - 1:
        raise ValueError("Smoothing requires n observations and n-1 actions")
    ids = [groups[int(x)] for x in observations]
    blocks = [T[int(a)][np.ix_(ids[u], ids[u + 1])] for u, a in enumerate(actions)]
    alpha = [boundary[ids[0]].copy()]
    if alpha[0].sum() <= 0:
        raise ValueError("Boundary belief does not support the first observation")
    alpha[0] /= alpha[0].sum()
    for block in blocks:
        nxt = alpha[-1] @ block
        alpha.append(nxt / nxt.sum())
    beta = [np.ones(len(i)) for i in ids]
    for u in reversed(range(n - 1)):
        beta[u] = blocks[u] @ beta[u + 1]
        beta[u] /= beta[u].sum()
    q = np.zeros((n, T.shape[1]))
    counts = np.zeros_like(T)
    for u in range(n):
        marginal = alpha[u] * beta[u]
        q[u, ids[u]] = marginal / marginal.sum()
        if u < n - 1:
            pair = alpha[u][:, None] * blocks[u] * beta[u + 1][None, :]
            counts[int(actions[u])][np.ix_(ids[u], ids[u + 1])] += pair / pair.sum() / (n - 1)
    return q, counts


def replay_terms(T, beliefs, actions, destination_ids, targets):
    """Return mean predictive/posterior KL, observation NLL, and auxiliary counts.

    Teacher targets and historical source beliefs are constants. Responsibilities
    are recomputed under T; no posterior outer-product approximation is used.
    """
    D = np.zeros_like(T)
    O = np.zeros_like(T)
    predictive, observation = 0.0, 0.0
    n = len(actions)
    for b, a, ids, target in zip(beliefs, actions, destination_ids, targets):
        sources = np.flatnonzero(b)
        block = b[sources, None] * T[int(a)][np.ix_(sources, ids)]
        p = block.sum(axis=0)
        if np.any(p <= 0):
            raise ValueError("Replay target is outside positive student support")
        q = target[ids]
        positive = q > 0
        predictive += float(np.sum(q[positive] * (np.log(q[positive]) - np.log(p[positive])))) / n
        observation -= float(np.log(p.sum())) / n
        D[int(a)][np.ix_(sources, ids)] += block * (q / p)[None, :] / n
        O[int(a)][np.ix_(sources, ids)] += block / p.sum() / n
    return predictive, predictive - observation, observation, D, O


def posterior_gradient(T, D, O):
    """Exact row-logit gradient of KL(q || masked(b T) / Z)."""
    difference = D - O
    return difference.sum(axis=-1, keepdims=True) * T - difference


def softmax_rows(W):
    values = np.exp(W - W.max(axis=-1, keepdims=True))
    return values / values.sum(axis=-1, keepdims=True)


def posterior_refine(T, B, weight, beliefs, actions, ids, targets, learning_rate=1.0):
    """One analytic-gradient step with bounded backtracking on the frozen loss.

    B contains original online EM counts plus the unchanged pseudocount.
    Per-row scaling is a positive diagonal preconditioner. This is a generalized
    M-step with descent, not an exact solution of the marginal argmin.
    """
    values = replay_terms(T, beliefs, actions, ids, targets)
    gradient = B.sum(-1, keepdims=True) * T - B + weight * posterior_gradient(T, values[3], values[4])
    scale = B.sum(-1, keepdims=True) + weight * (values[3] + values[4]).sum(-1, keepdims=True)
    direction = gradient / np.maximum(scale, 1e-15)
    before = -float(np.sum(B * np.log(T))) + weight * values[1]
    logits = np.log(T)
    for backtrack in range(8):
        trial = softmax_rows(logits - learning_rate * 0.5**backtrack * direction)
        trial_values = replay_terms(trial, beliefs, actions, ids, targets)
        after = -float(np.sum(B * np.log(trial))) + weight * trial_values[1]
        if after <= before + 1e-12:
            return trial, backtrack + 1, before, after
    return T.copy(), 8, before, before


@dataclass(frozen=True)
class ExperimentConfig:
    window: int = 40
    forgetting: float = 0.99
    pseudocount: float = 0.002
    penalty_weight: float = 1.0
    teacher_half_life: float = 40.0
    posterior_learning_rate: float = 1.0
    local_updates: int = 2
    coupled_auxiliary: bool = False
    # Source belief b_u in the replay penalty. "filter": f_u = P(z_u | x_{<=u}).
    # "smooth": P(z_u | x_{<=u}, x_{u+1}), forward-backward over the student's
    # short window x_{u:u+1} from the carried boundary f_u. Both are detached.
    student_inference: str = "filter"
    # Teacher smoothing context. None = `window` (historical behaviour). A longer
    # teacher window smooths the teacher over x_{t-W_T+1:t}, from its cached
    # boundary belief at t-W_T+1, but only the targets for the student's replay
    # window x_{s+1:t} are used; the replay window and long EM keep `window`.
    teacher_window: int | None = None

    def __post_init__(self):
        if self.window < 2 or not 0 < self.forgetting <= 1:
            raise ValueError("Invalid window or forgetting factor")
        if self.pseudocount <= 0 or self.penalty_weight < 0 or self.teacher_half_life <= 0:
            raise ValueError("Require positive pseudocount/half-life and nonnegative penalty")
        if self.posterior_learning_rate <= 0:
            raise ValueError("Learning rate must be positive")
        if isinstance(self.local_updates, bool) or not isinstance(self.local_updates, int) or self.local_updates < 2:
            raise ValueError("local_updates must be an integer of at least two")
        if self.student_inference not in STUDENT_INFERENCE:
            raise ValueError(f"student_inference must be one of {STUDENT_INFERENCE}")
        if self.teacher_window is not None and (isinstance(self.teacher_window, bool) or not isinstance(self.teacher_window, int)
                                                or self.teacher_window < self.window):
            raise ValueError("teacher_window must be an integer no smaller than window")


STUDENT_INFERENCE = ("filter", "smooth")


METHODS = ("short", "long", "predictive", "posterior", "nll_control")


def train_experiment(initial_T, n_clones, observations, actions, method, config,
                     checkpoints, callback: Callable | None = None, episode_starts=None):
    """Single causal pass. No future targets or replay is used after a checkpoint.

    All methods keep S <- lambda S + (1-lambda) X (or cumulative S for
    lambda=1). Long X is a window mean rescaled to the short arm's unit mass.
    The penalty applies to the current window, not an invented teacher M-step.
    Its coefficient is eta * sum(S), matching base-count scale over time.
    """
    if method not in METHODS:
        raise ValueError(f"Unknown experimental method: {method}")
    if len(actions) != len(observations) - 1:
        raise ValueError("Actions must be source aligned with length n-1")
    T = np.asarray(initial_T, dtype=float).copy()
    teacher = T.copy()
    groups = clone_groups(n_clones)
    student_filter, teacher_filter = CarriedFilter(n_clones), CarriedFilter(n_clones)
    S = np.zeros_like(T)
    cached_student, cached_teacher = [], []
    resets = set(episode_starts or ()) | {0}
    tau = 2 ** (-1 / config.teacher_half_life)
    teacher_window = config.window if config.teacher_window is None else config.teacher_window
    extra = method in {"predictive", "posterior", "nll_control"}
    smoothed_sources = extra and config.penalty_weight > 0 and config.student_inference == "smooth"
    # cached_source[u]: smoothed source belief of transition u -> u+1, computed once
    # at arrival u+1 under that arrival's pre-update student tensor (smooth mode only).
    cached_source = {}
    snapshots, diagnostics = {}, []
    work = dict(filter_block_entries=0, smoothing_block_entries=0,
                replay_block_entries=0, tensor_entries=0, base_msteps=0,
                refinements=0, loss_evaluations=0, transitions=0,
                em_passes=0, penalty_updates=0, statistics_commits=0, teacher_updates=0)
    elapsed = 0.0
    if 0 in checkpoints:
        snapshots[0] = T.copy()
        if callback:
            callback(0, T.copy(), dict(train_seconds=0., **work))
    episode_start = 0
    for t, x in enumerate(observations):
        started = time.perf_counter()
        if t in resets:
            student_filter.reset(); teacher_filter.reset()
            episode_start = t
        previous = None if student_filter.belief is None else student_filter.belief.copy()
        incoming = None if previous is None else int(actions[t - 1])
        _, belief, causal_nll = student_filter.step(T, int(x), incoming)
        cached_student.append(belief)
        if extra:
            _, h, _ = teacher_filter.step(teacher, int(x), incoming)
            cached_teacher.append(h)
        else:
            cached_teacher.append(belief)
        if previous is not None:
            work["transitions"] += 1
            active = len(groups[int(observations[t - 1])])
            work["filter_block_entries"] += active * T.shape[-1] * (2 if extra else 1)
            fast = pair_counts(T, previous, incoming, groups[int(x)])
            if smoothed_sources:
                # Smoothing the window's last node equals filtering, so smooth the
                # SOURCE node t-1 with the newest observation x_t (never beyond t).
                q_student, _ = smooth_window(T, groups, observations[t - 1:t + 1], actions[t - 1:t], previous)
                cached_source[t - 1] = q_student[0]
                work["smoothing_block_entries"] += 3 * active * len(groups[int(x)])
            s = max(episode_start, t - config.window + 1)
            xs, aa = observations[s:t + 1], actions[s:t]
            window_entries = sum(len(groups[int(xs[u])]) * len(groups[int(xs[u + 1])])
                                 for u in range(len(aa)))
            # Freeze the teacher target before any local student update. Neither
            # the teacher tensor nor its boundary belief changes inside the loop.
            if extra and config.penalty_weight > 0:
                ts = max(episode_start, t - teacher_window + 1)   # ts <= s
                q, _ = smooth_window(teacher, groups, observations[ts:t + 1], actions[ts:t], cached_teacher[ts])
                q = q[s - ts:]   # targets aligned with the student's replay window x_{s:t}
                work["smoothing_block_entries"] += 3 * sum(
                    len(groups[int(observations[u])]) * len(groups[int(observations[u + 1])]) for u in range(ts, t))
            prior_counts = S if config.forgetting == 1 else config.forgetting * S
            increment = 1. if config.forgetting == 1 else 1 - config.forgetting
            # Each E-step uses the latest T and the same pre-arrival statistics.
            # Coupled mode incorporates the auxiliary objective at EVERY M-step;
            # historical mode reserves only the final update for that objective.
            local_diagnostics = []
            for local_iteration in range(config.local_updates):
                if method == "long":
                    _, X = smooth_window(T, groups, xs, aa, cached_student[s])
                    work["smoothing_block_entries"] += 3 * window_entries
                else:
                    X = fast if local_iteration == 0 else pair_counts(T, previous, incoming, groups[int(x)])
                candidate_counts = prior_counts + increment * X
                B = candidate_counts + config.pseudocount
                final_iteration = local_iteration + 1 == config.local_updates
                auxiliary_step = extra and config.penalty_weight > 0 and (config.coupled_auxiliary or final_iteration)
                work["base_msteps" if local_iteration == 0 else "refinements"] += 1
                work["tensor_entries"] += T.size
                if config.coupled_auxiliary or not auxiliary_step:
                    work["em_passes"] += 1
                if auxiliary_step:
                    beliefs = ([cached_source[u] for u in range(s, t)] if smoothed_sources
                               else cached_student[s:t])
                    ids = [groups[int(z)] for z in xs[1:]]
                    targets = q[1:]
                    weight = config.penalty_weight * float(candidate_counts.sum())
                    before = replay_terms(T, beliefs, aa, ids, targets)
                    work["penalty_updates"] += 1
                    work["replay_block_entries"] += window_entries
                    objective_index = {"predictive": 0, "posterior": 1, "nll_control": 2}[method]
                    if (t + 1) in checkpoints:
                        objective_before = -float(np.sum(B * np.log(T))) + weight * before[objective_index]
                    if method in {"predictive", "nll_control"}:
                        counts = before[3] if method == "predictive" else before[4]
                        T = row_normalize(B + weight * counts)
                        trials = 0
                    else:
                        T, trials, _, _ = posterior_refine(
                            T, B, weight, beliefs, aa, ids, targets, config.posterior_learning_rate)
                        work["replay_block_entries"] += (trials + 1) * window_entries
                    work["loss_evaluations"] += trials
                    if (t + 1) in checkpoints:
                        after = replay_terms(T, beliefs, aa, ids, targets)
                        objective_after = -float(np.sum(B * np.log(T))) + weight * after[objective_index]
                        local_diagnostics.append(dict(iteration=local_iteration + 1,
                                                      objective_before=objective_before, objective_after=objective_after,
                                                      running_count_mass=float(candidate_counts.sum())))
                else:
                    T = row_normalize(B)
            # A single commit after the final iteration; the statistics contain
            # data expected counts only, never repeated auxiliary pseudocounts.
            S = candidate_counts
            work["statistics_commits"] += 1
            if extra and config.penalty_weight > 0 and (t + 1) in checkpoints:
                diagnostics.append(dict(step=t + 1, predictive_kl=after[0], posterior_kl=after[1],
                                        replay_nll=after[2], penalty_scale=weight,
                                        teacher_target_entropy=float(np.mean([
                                            -np.sum(z[z > 0] * np.log(z[z > 0])) for z in targets])),
                                        teacher_student_tv=float(np.abs(teacher-T).sum(-1).mean()/2),
                                        local_iterations=local_diagnostics,
                                        **({"source_smoothing_tv": float(np.mean([
                                            np.abs(cached_source[u] - cached_student[u]).sum() / 2
                                            for u in range(s, t)]))} if smoothed_sources else {})))
            if extra:
                teacher = tau * teacher + (1 - tau) * T
                work["teacher_updates"] += 1
                work["tensor_entries"] += T.size
        elapsed += time.perf_counter() - started
        if (t + 1) in checkpoints:
            snapshots[t + 1] = T.copy()
            if callback:
                callback(t + 1, T.copy(), dict(train_seconds=elapsed, **work))
    return dict(snapshots=snapshots, diagnostics=diagnostics, counts=S,
                final_T=T, teacher_T=teacher, train_seconds=elapsed, work=work)
