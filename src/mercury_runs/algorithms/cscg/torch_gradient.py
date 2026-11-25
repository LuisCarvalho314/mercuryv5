"""Gradient-based CSCG in PyTorch on the long-EM per-arrival objective.

GradCSCG follows the reference gradCSCG formulation (tristanstoeber/gradCSCG,
models/gradient_hmm.py): trainable transition logits, row-wise softmax, the batched log-space
forward recursion of GradientHMM._batch_neg_log_likelihood, automatic differentiation and standard
Adam (Keras defaults, as in the reference). The sink state and trainable initial distribution are
not reproduced: the initial log-belief is the carried boundary belief, and the loss is the shared
EM objective (history counts + pseudocount + window NLL).
See docs/gradient_cscg_formulation.md.
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
        """One Adam step on W; returns T = softmax(W) as NumPy."""
        self.adam.step()
        with torch.no_grad():
            return torch.softmax(self.W, dim=-1).numpy().copy()

