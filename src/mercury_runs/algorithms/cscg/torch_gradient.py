"""Gradient-based CSCG in PyTorch on the long-EM per-arrival objective.

Primary arm, GradCSCG: the reference gradCSCG formulation (tristanstoeber/gradCSCG,
models/gradient_hmm.py): trainable transition logits, row-wise softmax, the batched log-space
forward recursion of GradientHMM._batch_neg_log_likelihood, automatic differentiation and standard
Adam (Keras defaults, as in the reference). The sink state and trainable initial distribution are
not reproduced: the initial log-belief is the carried boundary belief, and the loss is the shared
EM objective (history counts + pseudocount + window NLL).

Secondary arm, ProjectedAdamCSCG: Adam on the probabilities themselves with row-wise simplex
projection; an adaptation, not the reference parameterisation. See docs/gradient_cscg_formulation.md.
"""
from __future__ import annotations

import numpy as np
import torch

from .distillation import NumericalInstabilityError


NEG_INF = -1.0e10   # the reference's finite log-space mask value


def batch_neg_log_likelihood(log_T, log_alpha0, O, A, L, state_obs):
    """Reference batched forward pass, per-sequence NLL [B] (no legacy final_alpha * 10).

    log_T [A, N, N] log transition probabilities; log_alpha0 [B, N] initial log-belief (the
    reference's log_pi); O, A [B, T] padded observations and actions (A[:, t] leads from t to
    t + 1); L [B] true lengths; state_obs [N] the observation of each clone.
    """
    def mask(obs):
        return torch.where(state_obs[None, :] == obs[:, None], 0., NEG_INF).to(log_T.dtype)
    alpha = log_alpha0 + mask(O[:, 0])
    for t in range(O.shape[1] - 1):
        s = alpha[:, :, None] + log_T[A[:, t]] + mask(O[:, t + 1])[:, None, :]
        alpha_next = torch.logsumexp(s, dim=1)
        alpha = torch.where((t < L - 1)[:, None], alpha_next, alpha)
    return -torch.logsumexp(alpha, dim=1)


def boundary_log_belief(boundary, state_obs, first_observation):
    """log h restricted to C(x_s) and renormalised there (as the NumPy forward pass does)."""
    h = torch.as_tensor(boundary) * (state_obs == int(first_observation))
    return torch.log(h / h.sum()).clamp(min=NEG_INF)


class GradCSCG:
    """Reference-style gradCSCG: logits W with T = softmax(W), persistent standard Adam.

    W is initialised to log T0 so softmax(W) = T0, EM's initial tensor. evaluate() forms the
    shared objective G from log T = log_softmax(W) and back-propagates once. Because
    dG/dlog T = -(history + increment * X) / mass (log-space form of T * grad = -X), the E-step
    counts X follow from the gradient retained on log T. step() is one Adam step on W.
    """

    def __init__(self, T, n_clones, learning_rate, betas=(0.9, 0.999), epsilon=1e-7):
        torch.set_num_threads(1)   # tiny per-step tensors: avoid thread contention between arms
        self.W = torch.tensor(np.log(np.asarray(T, dtype=np.float64)), requires_grad=True)
        self.adam = torch.optim.Adam([self.W], lr=learning_rate, betas=betas, eps=epsilon)
        self.state_obs = torch.from_numpy(np.repeat(np.arange(len(n_clones)), n_clones))

    def objective(self, history, increment, mass, observations, actions, boundary):
        log_T = torch.log_softmax(self.W, dim=-1)
        O = torch.as_tensor(np.asarray(observations, dtype=np.int64))[None]
        A = torch.as_tensor(np.r_[np.asarray(actions, dtype=np.int64), 0])[None]   # last entry unused
        L = torch.tensor([O.shape[1]])
        h = boundary_log_belief(boundary, self.state_obs, observations[0])[None]
        nll = batch_neg_log_likelihood(log_T, h, O, A, L, self.state_obs)[0] / (O.shape[1] - 1)
        return (-(torch.from_numpy(history) * log_T).sum() + increment * nll) / mass, nll, log_T

    def evaluate(self, history, increment, mass, observations, actions, boundary):
        self.adam.zero_grad()
        value, nll, log_T = self.objective(history, increment, mass, observations, actions, boundary)
        if not torch.isfinite(value):
            raise NumericalInstabilityError(f"Nonfinite objective {value.item()}")
        log_T.retain_grad()
        value.backward()
        if not torch.all(torch.isfinite(self.W.grad)):
            raise NumericalInstabilityError("Nonfinite gradient")
        with torch.no_grad():
            X = torch.clamp(-(log_T.grad * mass + torch.from_numpy(history)) / increment, min=0)
        return value.item(), nll.item(), X.numpy().copy()

    def step(self):
        self.adam.step()
        with torch.no_grad():
            T = torch.softmax(self.W, dim=-1)
        return T.numpy().copy(), {}


def clone_slices(n_clones):
    offsets = np.r_[0, np.cumsum(n_clones)].astype(int)
    return [slice(int(offsets[i]), int(offsets[i + 1])) for i in range(len(n_clones))]


def window_nll(P, slices, observations, actions, boundary):
    """Mean per-transition window NLL -log p(x_{s+1:t} | h, a; P) / (n-1), differentiable in P.

    Same scaled recursion as the NumPy forward pass: alpha_{u+1} = alpha_u P_{a_u}[C(x_u), C(x_{u+1})] / c_u,
    log Z = sum_u log c_u. A zero c_u gives +inf (an impossible window), detected by the caller.
    """
    if len(observations) < 2 or len(actions) != len(observations) - 1:
        raise ValueError("A window requires n observations and n-1 actions")
    alpha = boundary[slices[int(observations[0])]]
    alpha = alpha / alpha.sum()
    log_evidence = P.new_zeros(())
    for u, a in enumerate(actions):
        nxt = alpha @ P[int(a), slices[int(observations[u])], slices[int(observations[u + 1])]]
        mass = nxt.sum()
        log_evidence = log_evidence + torch.log(mass)
        alpha = nxt / mass
    return -log_evidence / len(actions)


def project_rows_to_simplex(V):
    """Row-wise Euclidean projection onto the simplex (sort-based, exact); may give exact zeros."""
    flat = V.reshape(-1, V.shape[-1])
    ordered = torch.sort(flat, dim=1, descending=True).values
    cumulative = torch.cumsum(ordered, dim=1) - 1
    index = torch.arange(1, flat.shape[1] + 1, dtype=V.dtype)
    support = (ordered - cumulative / index > 0).sum(dim=1)
    theta = cumulative.gather(1, (support - 1)[:, None]) / support[:, None].to(V.dtype)
    return torch.clamp(flat - theta, min=0).reshape(V.shape)


class ProjectedAdamCSCG:
    """Persistent parameter P and torch.optim.Adam state across arrivals and local iterations.

    evaluate() forms the objective at the current P and back-propagates once (P.grad is then the
    tangent-space gradient); step() applies Adam and the projection. Feasibility: a row whose projected proposal sets an entry with positive
    objective weight to exactly zero (G = +inf under the pseudocount log-barrier) has only its own
    Adam step halved, at most max_halvings times, else that row is kept.
    """

    def __init__(self, T, n_clones, learning_rate, max_halvings=20, betas=(0.9, 0.999), epsilon=1e-8):
        torch.set_num_threads(1)   # tiny per-step tensors: avoid thread contention between arms
        self.P = torch.tensor(np.asarray(T, dtype=np.float64), requires_grad=True)
        self.adam = torch.optim.Adam([self.P], lr=learning_rate, betas=betas, eps=epsilon)
        self.slices = clone_slices(n_clones)
        self.max_halvings = int(max_halvings)
        self.history = None

    def objective(self, history, increment, mass, observations, actions, boundary):
        """G(P) = [-<history, log P> + increment * window_nll(P)] / mass (torch scalar)."""
        prior = -(torch.from_numpy(history) * torch.log(self.P)).sum()
        nll = window_nll(self.P, self.slices, observations, actions, torch.from_numpy(boundary))
        return (prior + increment * nll) / mass, nll

    def evaluate(self, history, increment, mass, observations, actions, boundary):
        """Objective, window NLL and E-step counts X at the current P, from one backward pass.

        grad G = (-history / P + increment * grad nll) / mass and P * grad nll = -X, hence
        X = -(P * grad G * mass + history) / increment (rounding clamped at zero).
        """
        self.adam.zero_grad()
        value, nll = self.objective(history, increment, mass, observations, actions, boundary)
        if not torch.isfinite(value):
            raise NumericalInstabilityError(f"Nonfinite objective {value.item()}")
        value.backward()
        grad = self.P.grad
        if not torch.all(torch.isfinite(grad)):
            raise NumericalInstabilityError("Nonfinite gradient")
        with torch.no_grad():
            X = torch.clamp(-(self.P * grad * mass + torch.from_numpy(history)) / increment, min=0)
            # Adam receives the gradient in the simplex tangent space (row mean removed). The row
            # constant is the multiplier of sum_j T_ij = 1: it does not change Pi(T - eta g), but it
            # would dominate Adam's per-coordinate normalisation (every entry of grad G is negative,
            # so m_hat / sqrt(v_hat) ~ -1 everywhere and the projection cancels the step).
            grad -= grad.mean(dim=-1, keepdim=True)
        self.history = history
        return value.item(), nll.item(), X.numpy().copy()

    def step(self):
        """Adam update, then row-wise projection under no_grad; returns T (NumPy) and step info."""
        positive = torch.from_numpy(self.history > 0).reshape(-1, self.P.shape[-1])
        with torch.no_grad():
            before = self.P.detach().clone()
            self.adam.step()
            raw = (self.P - before).reshape(-1, self.P.shape[-1])
            start = before.reshape(-1, self.P.shape[-1])
            proposal = project_rows_to_simplex(start + raw)
            bad = ((proposal == 0) & positive).any(dim=1)
            halvings = 0
            for halving in range(1, self.max_halvings + 1):
                if not bad.any():
                    break
                halvings += int(bad.sum())
                proposal[bad] = project_rows_to_simplex(start[bad] + 0.5 ** halving * raw[bad])
                bad = bad & ((proposal == 0) & positive).any(dim=1)
            rejected = int(bad.sum())
            proposal[bad] = start[bad]
            self.P.copy_(proposal.reshape(self.P.shape))
        return self.P.detach().numpy().copy(), dict(row_halvings=halvings, rejected_rows=rejected)
