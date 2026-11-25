"""Research-only online CSCG: carried filters and frozen-window KL objectives.

Actions here are source aligned: actions[u] causes observations[u + 1].
The production online runner and the external CSCG implementation are untouched.
"""
from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from typing import Callable
import math
import time

import numpy as np


class ZeroPredictiveSupportError(ValueError):
    """The student assigns zero probability where the data or a target has mass.

    Raised, never repaired: the predictive objective is infinite there and the MM
    responsibilities are undefined. See docs/online_jepa_cscg_formulation.md, Section 5.11.
    """


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
            raise ZeroPredictiveSupportError("Observation has zero or invalid predictive probability")
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


def _window_messages(T, groups, observations, actions, boundary, backward=True):
    """Scaled forward-backward over x_{s:t} from a boundary belief on z_s.

    Returns clone ids, transition blocks, normalised alpha (and beta), and the window
    log-evidence log p(x_{s+1:t} | h, a; T) = sum_u log(alpha_u^T T_{a_u} e_{u+1}).
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
    log_evidence = 0.
    for u, block in enumerate(blocks):
        nxt = alpha[-1] @ block
        mass = nxt.sum()
        if not mass > 0 or not np.isfinite(mass):
            raise ZeroPredictiveSupportError(f"Window observation {u + 1} has zero or invalid predictive probability")
        log_evidence += math.log(mass)
        alpha.append(nxt / mass)
    beta = None
    if backward:
        beta = [np.ones(len(i)) for i in ids]
        for u in reversed(range(n - 1)):
            beta[u] = blocks[u] @ beta[u + 1]
            beta[u] /= beta[u].sum()
    return ids, blocks, alpha, beta, log_evidence


def smooth_window(T, groups, observations, actions, boundary):
    """Exact within-window smoothing, conditional on a historical filtered boundary.

    Returns full-state marginals and the mean pair count (unit total mass).
    """
    n = len(observations)
    ids, blocks, alpha, beta, _ = _window_messages(T, groups, observations, actions, boundary)
    q = np.zeros((n, T.shape[1]))
    counts = np.zeros_like(T)
    for u in range(n):
        marginal = alpha[u] * beta[u]
        q[u, ids[u]] = marginal / marginal.sum()
        if u < n - 1:
            pair = alpha[u][:, None] * blocks[u] * beta[u + 1][None, :]
            counts[int(actions[u])][np.ix_(ids[u], ids[u + 1])] += pair / pair.sum() / (n - 1)
    return q, counts


def window_nll(T, groups, observations, actions, boundary):
    """Mean per-transition window NLL -log p(x_{s+1:t} | h, a; T) / (n-1): forward pass only."""
    return -_window_messages(T, groups, observations, actions, boundary, backward=False)[4] / (len(observations) - 1)


class NumericalInstabilityError(ValueError):
    """A gradient step produced a nonfinite objective or gradient. Raised, never repaired."""


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
        q = target[ids]
        if np.any(p <= 0):
            raise ZeroPredictiveSupportError(
                "Replay target is outside positive student support: "
                f"{int(np.sum(p <= 0))} destination clone(s) with p_u(j; T) = 0, "
                f"target mass {float(q[p <= 0].sum()):.3g}")
        positive = q > 0
        predictive += float(np.sum(q[positive] * (np.log(q[positive]) - np.log(p[positive])))) / n
        observation -= float(np.log(p.sum())) / n
        D[int(a)][np.ix_(sources, ids)] += block * (q / p)[None, :] / n
        O[int(a)][np.ix_(sources, ids)] += block / p.sum() / n
    return predictive, predictive - observation, observation, D, O


def expected_counts(X, D, rho):
    """C = (1 - rho) xi + rho D, the newly accumulated evidence of one arrival.

    xi (short pair counts) and D (mean teacher-supervised window counts) both have
    unit total mass, so C does too. Each endpoint returns its own tensor exactly
    (the unused one may be None): rho = 0 is short EM, rho = 1 teacher counts only.
    """
    if rho == 0:
        return X
    if rho == 1:
        return D
    return (1 - rho) * X + rho * D


def data_nll(T, source_belief, action, destination_ids):
    """-log p(x_t | x_<t; T) of the short arrival term, under a fixed source belief."""
    return -float(np.log((source_belief @ T[action])[destination_ids].sum()))


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
    # Legacy penalty scale eta: the auxiliary weight is w = eta * sum(S~), with the
    # historical (or coupled_auxiliary) schedule. None means the legacy default 1.0.
    # Mutually exclusive with jepa_mix.
    penalty_weight: float | None = None
    teacher_half_life: float = 40.0
    posterior_learning_rate: float = 1.0
    local_updates: int = 2
    coupled_auxiliary: bool = False
    # Source belief b_u in the replay penalty. "filter": f_u = P(z_u | x_{<=u}).
    # "smooth": P(z_u | x_{<=u}, x_{u+1}), forward-backward over the student's
    # short window x_{u:u+1} from the carried boundary f_u. Both are detached.
    student_inference: str = "filter"
    # Smooth mode only: historical context H_S of the student's forward-backward pass.
    # b_u is the node-u marginal of smooth_window over x_{u-H_S+1:u+1} (smoothing lag one),
    # started from the cached filter f_{u-H_S+1} and recomputed under the current student
    # tensor -- the same inference the teacher runs over its longer window. H_S = 1 is the
    # historical one-transition smoothing from f_u.
    student_context: int = 1
    # Teacher smoothing context. None = `window` (historical behaviour). A longer
    # teacher window smooths the teacher over x_{t-W_T+1:t}, from its cached
    # boundary belief at t-W_T+1, but only the targets for the student's replay
    # window x_{s+1:t} are used; the replay window and long EM keep `window`.
    teacher_window: int | None = None
    # Accumulated predictive JEPA (method "predictive" only). Each arrival's new
    # expected counts are C = (1 - rho) xi + rho D (short pair counts xi, teacher-
    # supervised window counts D), accumulated into the usual running statistics
    # with the usual forgetting and pseudocount: T = rowNorm(lambda S + (1-lambda) C + kappa).
    # 0 = short EM, 1 = teacher-derived evidence only (statistics, forgetting and
    # pseudocount retained). None = legacy penalty parameterisation via penalty_weight.
    jepa_mix: float | None = None
    # Optimiser of the per-arrival objective. "em": T <- rowNorm(B) (or its jepa_mix analogue) at
    # every local iteration. "gradcscg" (method "long", or "predictive" with jepa_mix > 0): the same
    # objective, normalised by its constant count mass, minimised by reference gradCSCG -- logits,
    # row-softmax, log-space forward, autodiff and standard Adam (adam_* coefficients, learning
    # rate gradient_learning_rate); cscg/torch_gradient.py, docs/gradient_cscg_formulation.md.
    optimizer: str = "em"
    gradient_learning_rate: float = 1e-3
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-7   # Keras Adam default, as used by the reference

    @property
    def legacy_penalty_weight(self) -> float:
        return 1.0 if self.penalty_weight is None else float(self.penalty_weight)

    def __post_init__(self):
        if self.window < 2 or not 0 < self.forgetting <= 1:
            raise ValueError("Invalid window or forgetting factor")
        if self.pseudocount <= 0 or self.legacy_penalty_weight < 0 or self.teacher_half_life <= 0:
            raise ValueError("Require positive pseudocount/half-life and nonnegative penalty")
        if self.jepa_mix is not None:
            if (isinstance(self.jepa_mix, bool) or not isinstance(self.jepa_mix, Real)
                    or not math.isfinite(self.jepa_mix) or not 0 <= self.jepa_mix <= 1):
                raise ValueError("jepa_mix must be a finite real number in the closed interval [0, 1]")
            if self.penalty_weight is not None:
                raise ValueError("jepa_mix replaces the legacy penalty_weight; set only one of them")
            if self.coupled_auxiliary:
                raise ValueError("coupled_auxiliary is a legacy schedule flag; jepa_mix already "
                                 "applies its objective at every local iteration")
        if self.posterior_learning_rate <= 0:
            raise ValueError("Learning rate must be positive")
        if isinstance(self.local_updates, bool) or not isinstance(self.local_updates, int) or self.local_updates < 2:
            raise ValueError("local_updates must be an integer of at least two")
        if self.student_inference not in STUDENT_INFERENCE:
            raise ValueError(f"student_inference must be one of {STUDENT_INFERENCE}")
        if (isinstance(self.student_context, bool) or not isinstance(self.student_context, int)
                or self.student_context < 1):
            raise ValueError("student_context must be a positive integer")
        if self.student_context > 1 and self.student_inference != "smooth":
            raise ValueError("student_context > 1 requires student_inference='smooth'")
        if self.teacher_window is not None and (isinstance(self.teacher_window, bool) or not isinstance(self.teacher_window, int)
                                                or self.teacher_window < self.window):
            raise ValueError("teacher_window must be an integer no smaller than window")
        if self.optimizer not in OPTIMIZERS:
            raise ValueError(f"optimizer must be one of {OPTIMIZERS}")
        if not math.isfinite(self.gradient_learning_rate) or self.gradient_learning_rate <= 0:
            raise ValueError("gradient_learning_rate must be positive and finite")
        if self.optimizer != "em" and self.jepa_mix == 0:
            raise ValueError("gradcscg with jepa_mix requires jepa_mix > 0 (jepa_mix = 0 is short EM)")
        if not (0 <= self.adam_beta1 < 1 and 0 <= self.adam_beta2 < 1 and self.adam_epsilon > 0):
            raise ValueError("Adam requires beta1, beta2 in [0, 1) and positive epsilon")


STUDENT_INFERENCE = ("filter", "smooth")
OPTIMIZERS = ("em", "gradcscg")


METHODS = ("short", "long", "predictive", "posterior", "nll_control")


def train_experiment(initial_T, n_clones, observations, actions, method, config,
                     checkpoints, callback: Callable | None = None, episode_starts=None):
    """Single causal pass. No future targets or replay is used after a checkpoint.

    Every arm with an EM component keeps S <- lambda S + (1-lambda) X (or
    cumulative S for lambda=1). Long X is a window mean rescaled to the short
    arm's unit mass. The JEPA term applies to the current window only.

    Objective selection for "predictive":
      * config.jepa_mix = rho: every local iteration k forms
        C^(k) = (1-rho) xi^(k) + rho D^(k) at the current T and sets
        T <- rowNorm(prior + inc * C^(k) + kappa) from the SAME pre-arrival
        statistics; prior + inc * C is committed once, after the final iteration.
        rho = 0 skips the teacher entirely; rho = 1 forms no xi.
      * config.jepa_mix = None (legacy): rowNorm(B + w D) with w = eta * sum(S~),
        D transient, applied only at the final iteration (historical schedule) or
        at every iteration (coupled_auxiliary).
    """
    if method not in METHODS:
        raise ValueError(f"Unknown experimental method: {method}")
    if len(actions) != len(observations) - 1:
        raise ValueError("Actions must be source aligned with length n-1")
    mixed = config.jepa_mix is not None
    if mixed and method != "predictive":
        raise ValueError("jepa_mix defines the objective of method 'predictive' only")
    gradcscg = config.optimizer == "gradcscg"
    if gradcscg and not (method == "long" or (method == "predictive" and mixed)):
        raise ValueError("optimizer 'gradcscg' is implemented for method 'long' and for "
                         "method 'predictive' with jepa_mix")
    extra = method in {"predictive", "posterior", "nll_control"}
    if mixed:
        rho = float(config.jepa_mix)
        # em_active: the observation-derived counts xi are formed (rho < 1).
        em_active, jepa_active, teacher_active = rho < 1, rho > 0, rho > 0
    else:
        eta = config.legacy_penalty_weight
        # Historical behaviour: the teacher runs for every extra arm, even at eta = 0.
        em_active, jepa_active, teacher_active = True, extra and eta > 0, extra
    T = np.asarray(initial_T, dtype=float).copy()
    teacher = T.copy()
    if gradcscg:
        from .torch_gradient import GradCSCG
        gradient_arm = GradCSCG(T, n_clones, config.gradient_learning_rate,
                                (config.adam_beta1, config.adam_beta2), config.adam_epsilon)
    groups = clone_groups(n_clones)
    student_filter, teacher_filter = CarriedFilter(n_clones), CarriedFilter(n_clones)
    # Every arm keeps running sufficient statistics, including rho = 1.
    S = np.zeros_like(T)
    cached_student, cached_teacher = [], []
    resets = set(episode_starts or ()) | {0}
    tau = 2 ** (-1 / config.teacher_half_life)
    teacher_window = config.window if config.teacher_window is None else config.teacher_window
    smoothed_sources = jepa_active and config.student_inference == "smooth"
    # cached_source[u]: smoothed source belief of transition u -> u+1, computed once
    # at arrival u+1 under that arrival's pre-update student tensor (smooth mode only).
    cached_source = {}
    snapshots, diagnostics = {}, []
    objective_diagnostics = []   # method "long" only: per-iteration arrival objective at checkpoints
    teacher_snapshots = {}   # recording only: the EMA teacher at each checkpoint (empty without a teacher)
    work = dict(filter_block_entries=0, smoothing_block_entries=0,
                replay_block_entries=0, tensor_entries=0, base_msteps=0,
                refinements=0, loss_evaluations=0, transitions=0,
                em_passes=0, penalty_updates=0, statistics_commits=0, teacher_updates=0,
                gradient_steps=0)
    elapsed = 0.0
    # Recording only (never read by the learner): support margins since the last checkpoint.
    def support_record():
        return dict(min_T=float(T.min()), min_teacher_T=float(teacher.min()) if teacher_active else None,
                    min_student_evidence=support["student"], min_teacher_evidence=support["teacher"])
    support = dict(student=None, teacher=None)
    def track(key, nll):
        value = float(np.exp(-nll))
        support[key] = value if support[key] is None else min(support[key], value)
    if 0 in checkpoints:
        snapshots[0] = T.copy()
        if teacher_active:
            teacher_snapshots[0] = teacher.copy()
        if callback:
            callback(0, T.copy(), dict(train_seconds=0., **work, **support_record()))
    episode_start = 0
    for t, x in enumerate(observations):
        started = time.perf_counter()
        if t in resets:
            student_filter.reset(); teacher_filter.reset()
            episode_start = t
        previous = None if student_filter.belief is None else student_filter.belief.copy()
        incoming = None if previous is None else int(actions[t - 1])
        try:
            _, belief, causal_nll = student_filter.step(T, int(x), incoming)
        except ZeroPredictiveSupportError as error:
            raise ZeroPredictiveSupportError(f"arrival {t}: student filter: {error}") from error
        if previous is not None:
            track("student", causal_nll)
        cached_student.append(belief)
        if teacher_active:
            try:
                _, h, teacher_nll = teacher_filter.step(teacher, int(x), incoming)
            except ZeroPredictiveSupportError as error:
                raise ZeroPredictiveSupportError(f"arrival {t}: teacher filter: {error}") from error
            if previous is not None:
                track("teacher", teacher_nll)
            cached_teacher.append(h)
        else:
            cached_teacher.append(belief)
        # An episode start carries no transition: no targets, no update, no commit.
        if previous is not None:
            work["transitions"] += 1
            active = len(groups[int(observations[t - 1])])
            work["filter_block_entries"] += active * T.shape[-1] * (2 if teacher_active else 1)
            # gradCSCG-JEPA takes its new counts from the gradient; NumPy counts are then formed
            # only at checkpoints, for the recorded diagnostics.
            gradient_jepa = gradcscg and mixed
            numpy_counts = not gradient_jepa or (t + 1) in checkpoints
            if em_active and numpy_counts:
                fast = pair_counts(T, previous, incoming, groups[int(x)])
            if smoothed_sources:
                # Source node u = t-1, smoothed with lag one (newest observation x_t,
                # never beyond t) over the historical context x_{u-H_S+1:t}, from the
                # cached filter at the window start, under the current tensor T.
                # H_S = 1: window x_{t-1:t} from f_{t-1}, the historical computation.
                a0 = max(episode_start, t - config.student_context)
                q_student, _ = smooth_window(T, groups, observations[a0:t + 1], actions[a0:t], cached_student[a0])
                cached_source[t - 1] = q_student[-2]
                work["smoothing_block_entries"] += 3 * sum(
                    len(groups[int(observations[u])]) * len(groups[int(observations[u + 1])]) for u in range(a0, t))
            s = max(episode_start, t - config.window + 1)
            xs, aa = observations[s:t + 1], actions[s:t]
            window_entries = sum(len(groups[int(xs[u])]) * len(groups[int(xs[u + 1])])
                                 for u in range(len(aa)))
            # Freeze sources, teacher targets and the teacher tensor before any
            # local student update. None of them changes inside the loop.
            if jepa_active:
                ts = max(episode_start, t - teacher_window + 1)   # ts <= s
                q, _ = smooth_window(teacher, groups, observations[ts:t + 1], actions[ts:t], cached_teacher[ts])
                q = q[s - ts:]   # targets aligned with the student's replay window x_{s:t}
                work["smoothing_block_entries"] += 3 * sum(
                    len(groups[int(observations[u])]) * len(groups[int(observations[u + 1])]) for u in range(ts, t))
                beliefs = ([cached_source[u] for u in range(s, t)] if smoothed_sources
                           else cached_student[s:t])
                ids = [groups[int(z)] for z in xs[1:]]
                targets = q[1:]
            prior_counts = S if config.forgetting == 1 else config.forgetting * S
            increment = 1. if config.forgetting == 1 else 1 - config.forgetting
            record_objective = method == "long" and (t + 1) in checkpoints
            if gradcscg or record_objective:
                # Long-EM per-arrival objective, normalised by its T-independent count mass:
                #   G(T) = [-<lambda S + kappa, log T> + gamma * window_nll(T)] / M,
                # the objective whose majoriser rowNorm(B) minimises exactly (EM).
                history = prior_counts + config.pseudocount
                mass = float(history.sum()) + increment
                def arrival_objective(tensor):
                    with np.errstate(divide="ignore"):
                        prior_term = -float(np.sum(history * np.log(tensor)))
                    if not np.isfinite(prior_term):
                        return math.inf
                    try:
                        nll = window_nll(tensor, groups, xs, aa, cached_student[s])
                    except ZeroPredictiveSupportError:
                        return math.inf
                    return (prior_term + increment * nll) / mass
            if gradient_jepa:
                # Accumulated JEPA data term in torch, from the frozen sources and teacher targets.
                jepa_data = gradient_arm.jepa_term(rho, previous, incoming, x, beliefs, aa, targets)
            # Each E-step uses the latest T and the same pre-arrival statistics.
            local_diagnostics = []
            for local_iteration in range(config.local_updates):
                X = None
                if em_active:
                    if method == "long" and gradcscg:
                        # Autodiff through the torch forward pass; X follows from T * grad = -X.
                        value, nll, X = gradient_arm.evaluate(
                            history, increment, mass, gradient_arm.window_term(xs, aa, cached_student[s]))
                        work["smoothing_block_entries"] += 3 * window_entries
                    elif method == "long":
                        _, X = smooth_window(T, groups, xs, aa, cached_student[s])
                        work["smoothing_block_entries"] += 3 * window_entries
                    elif numpy_counts:
                        X = fast if local_iteration == 0 else pair_counts(T, previous, incoming, groups[int(x)])
                final_iteration = local_iteration + 1 == config.local_updates
                # jepa_mix: teacher counts at every iteration. Legacy: the coupled
                # flag, or else only the final iteration (EM warm-up first).
                auxiliary_step = jepa_active and (mixed or config.coupled_auxiliary or final_iteration)
                work["base_msteps" if local_iteration == 0 else "refinements"] += 1
                work["tensor_entries"] += T.size
                if (em_active if mixed else (config.coupled_auxiliary or not auxiliary_step)):
                    work["em_passes"] += 1
                if auxiliary_step:
                    work["penalty_updates"] += 1
                    work["replay_block_entries"] += window_entries
                    if numpy_counts:
                        try:
                            before = replay_terms(T, beliefs, aa, ids, targets)
                        except ZeroPredictiveSupportError as error:
                            raise ZeroPredictiveSupportError(f"arrival {t}, iteration {local_iteration}: {error}") from error
                # New evidence of this arrival: data counts xi (EM and legacy arms),
                # or the accumulated mixture C = (1-rho) xi + rho D (jepa_mix).
                if gradient_jepa:
                    # One forward/backward pass of the true objective at the current logits: its
                    # gradient drives the Adam step below and gives C = (1-rho) xi + rho D.
                    _, _, C = gradient_arm.evaluate(history, increment, mass, jepa_data)
                else:
                    C = expected_counts(X, before[3] if auxiliary_step else None, rho) if mixed else X
                candidate_counts = prior_counts + increment * C
                B = candidate_counts + config.pseudocount
                if mixed and auxiliary_step:
                    record = (t + 1) in checkpoints
                    if record:
                        # Per-arrival objective F(T) = -<prior + kappa, log T>
                        #   + inc * [(1-rho) (-log p(x_t | x_<t; T)) + rho L_JEPA(T)];
                        # rowNorm(B^(k)) minimises its MM majoriser at T^(k).
                        def arrival_terms(tensor, jepa):
                            prior_term = -float(np.sum((prior_counts + config.pseudocount) * np.log(tensor)))
                            nll = data_nll(tensor, previous, incoming, groups[int(x)]) if em_active else None
                            value = prior_term + increment * ((1 - rho) * (nll or 0.) + rho * jepa)
                            return dict(objective=value, prior_term=prior_term, data_nll=nll, jepa_term=float(jepa))
                        terms_before = arrival_terms(T, before[0])
                        if final_iteration:
                            # Per-row masses of this arrival's new evidence, by source,
                            # and of the historical statistics it is added to.
                            row_masses = dict(
                                em_row_mass=(np.zeros(T.shape[:2]) if X is None
                                             else increment * (1 - rho) * X.sum(-1)).tolist(),
                                jepa_row_mass=(increment * rho * before[3].sum(-1)).tolist(),
                                history_row_mass=(prior_counts + config.pseudocount).sum(-1).tolist())
                    if gradcscg:
                        # Same objective F; one Adam step on the logits replaces the MM update rowNorm(B).
                        T = gradient_arm.step()
                        work["gradient_steps"] += 1
                    else:
                        T = row_normalize(B)
                    if record:
                        after = replay_terms(T, beliefs, aa, ids, targets)
                        local_diagnostics.append(dict(
                            iteration=local_iteration + 1, jepa_mix=rho,
                            **{f"{k}_before": v for k, v in terms_before.items()},
                            **{f"{k}_after": v for k, v in arrival_terms(T, after[0]).items()},
                            running_count_mass=float(candidate_counts.sum())))
                elif auxiliary_step:
                    # Legacy penalty, as at 852cf3b: D is transient and never committed.
                    weight = eta * float(candidate_counts.sum())
                    objective_index = {"predictive": 0, "posterior": 1, "nll_control": 2}[method]
                    def objective(tensor, auxiliary):
                        return -float(np.sum(B * np.log(tensor))) + weight * auxiliary
                    if (t + 1) in checkpoints:
                        objective_before = objective(T, before[objective_index])
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
                        local_diagnostics.append(dict(
                            iteration=local_iteration + 1, objective_before=objective_before,
                            objective_after=objective(T, after[objective_index]),
                            running_count_mass=float(candidate_counts.sum())))
                elif gradcscg and method == "long":
                    T = gradient_arm.step()
                    work["gradient_steps"] += 1
                    if record_objective:
                        local_diagnostics.append(dict(iteration=local_iteration + 1, objective_before=value,
                                                      objective_after=arrival_objective(T)))
                else:
                    if record_objective:
                        objective_before = arrival_objective(T)
                    T = row_normalize(B)
                    if record_objective:
                        local_diagnostics.append(dict(iteration=local_iteration + 1, objective_before=objective_before,
                                                      objective_after=arrival_objective(T)))
            if record_objective:
                objective_diagnostics.append(dict(step=t + 1, optimizer=config.optimizer,
                                                  local_iterations=local_diagnostics))
            # A single commit after the final iteration. EM and legacy arms commit
            # data counts only; jepa_mix commits the accumulated mixture C.
            S = candidate_counts
            work["statistics_commits"] += 1
            if jepa_active and (t + 1) in checkpoints:
                diagnostics.append(dict(step=t + 1, predictive_kl=after[0], posterior_kl=after[1],
                                        replay_nll=after[2],
                                        **(dict(jepa_mix=rho) if mixed else dict(penalty_scale=weight)),
                                        teacher_target_entropy=float(np.mean([
                                            -np.sum(z[z > 0] * np.log(z[z > 0])) for z in targets])),
                                        teacher_student_tv=float(np.abs(teacher-T).sum(-1).mean()/2),
                                        local_iterations=local_diagnostics,
                                        **(row_masses if mixed else {}),
                                        min_window_student_prediction=float(min(
                                            (b[b > 0] @ T[int(act)][np.ix_(np.flatnonzero(b), j)]).min()
                                            for b, act, j in zip(beliefs, aa, ids))),
                                        min_window_teacher_target=float(min(z[j].min() for z, j in zip(targets, ids))),
                                        **({"source_smoothing_tv": float(np.mean([
                                            np.abs(cached_source[u] - cached_student[u]).sum() / 2
                                            for u in range(s, t)]))} if smoothed_sources else {})))
            # One EMA step per arrival, after the committed student update.
            if teacher_active:
                teacher = tau * teacher + (1 - tau) * T
                work["teacher_updates"] += 1
                work["tensor_entries"] += T.size
        elapsed += time.perf_counter() - started
        if (t + 1) in checkpoints:
            snapshots[t + 1] = T.copy()
            if teacher_active:
                teacher_snapshots[t + 1] = teacher.copy()
            if callback:
                callback(t + 1, T.copy(), dict(train_seconds=elapsed, **work, **support_record()))
            support = dict(student=None, teacher=None)
    return dict(snapshots=snapshots, teacher_snapshots=teacher_snapshots, diagnostics=diagnostics,
                objective_diagnostics=objective_diagnostics, counts=S,
                final_T=T, teacher_T=teacher, train_seconds=elapsed, work=work)
