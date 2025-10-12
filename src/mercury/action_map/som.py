# action_map/som.py
"""
Minimal NumPy Self-Organizing Map (SOM).

Design
------
- Stateless functions + small @dataclass state.
- 1D codebook (N units), vectors of dim D.
- Winner-only update when sigma <= 0.
- Gaussian neighborhood over unit indices when sigma > 0.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple
import numpy as np

Array = np.ndarray


@dataclass(frozen=True)
class SOMParams:
    """
    Hyperparameters for a 1D SOM.

    Attributes
    ----------
    n_codebook : int
        Number of prototype vectors (units).
    dim : int
        Dimensionality of each prototype/action vector.
    lr : float, default 0.5
        Learning rate for updates.
    sigma : float, default 0.0
        Neighborhood width. <=0 means winner-only updates.
    """
    n_codebook: int
    dim: int
    lr: float = 0.5
    sigma: float = 0.0


@dataclass(frozen=True)
class SOMState:
    """
    Runtime SOM state.

    Attributes
    ----------
    codebook : Array
        Prototype matrix (n_codebook, dim), float32.
    bmu : int
        Index of the last Best-Matching Unit.
    """
    codebook: Array
    bmu: int = 0


def som_init_zeros(p: SOMParams) -> SOMState:
    """
    Zero-initialize the codebook.

    Parameters
    ----------
    p : SOMParams

    Returns
    -------
    SOMState
        codebook=zeros, bmu=0.
    """
    return SOMState(codebook=np.zeros((p.n_codebook, p.dim), np.float32), bmu=0)


def som_init(p: SOMParams, key: int | np.random.Generator) -> SOMState:
    """
    Random-normal initialize the codebook (std ~ 1e-3).

    Parameters
    ----------
    p : SOMParams
    key : int | numpy.random.Generator

    Returns
    -------
    SOMState
    """
    rng = key if isinstance(key, np.random.Generator) else np.random.default_rng(int(key))
    cb = (1e-3 * rng.standard_normal((p.n_codebook, p.dim))).astype(np.float32)
    return SOMState(codebook=cb, bmu=0)


def som_predict(p: SOMParams, s: SOMState, x: Array) -> int:
    """
    Best-Matching Unit for a single vector.

    Parameters
    ----------
    p : SOMParams
    s : SOMState
    x : Array
        Shape (p.dim,).

    Returns
    -------
    int
        BMU index.

    Raises
    ------
    ValueError
        If x has wrong shape.
    """
    if x.shape != (p.dim,):
        raise ValueError(f"action shape ({p.dim},) required, got {x.shape}")
    d2 = np.sum((s.codebook - x) ** 2, axis=1)  # (N,)
    return int(np.argmin(d2))


def som_predict_batch(p: SOMParams, s: SOMState, X: Array) -> Array:
    """
    BMUs for a batch of vectors.

    Parameters
    ----------
    p : SOMParams
    s : SOMState
    X : Array
        Shape (B, p.dim).

    Returns
    -------
    Array
        BMU indices, shape (B,), int32.

    Raises
    ------
    ValueError
        If X has wrong shape.
    """
    if X.ndim != 2 or X.shape[1] != p.dim:
        raise ValueError(f"(B, {p.dim}) required, got {X.shape}")
    # d^2 = ||x||^2 + ||w||^2 - 2 x·w
    a2 = np.sum(X * X, axis=1, keepdims=True)              # (B,1)
    w2 = np.sum(s.codebook * s.codebook, axis=1)[None, :]  # (1,N)
    dot = X @ s.codebook.T                                 # (B,N)
    d2 = a2 + w2 - 2.0 * dot
    return np.argmin(d2, axis=1).astype(np.int32)


def som_update_one(p: SOMParams, s: SOMState, x: Array) -> Tuple[SOMState, int]:
    """
    Single online update on one example.

    Steps
    -----
    1) Find BMU.
    2) Update codebook:
       - winner-only if sigma<=0
       - Gaussian neighborhood on unit indices if sigma>0

    Parameters
    ----------
    p : SOMParams
    s : SOMState
    x : Array
        Shape (p.dim,).

    Returns
    -------
    (SOMState, int)
        New state and BMU index.

    Raises
    ------
    ValueError
        If x has wrong shape.
    """
    if x.shape != (p.dim,):
        raise ValueError(f"action shape ({p.dim},) required, got {x.shape}")

    b = som_predict(p, s, x)
    cb = s.codebook
    lr = float(p.lr)

    if p.sigma <= 0.0:
        cb_new = cb.copy()
        cb_new[b] += lr * (x - cb[b])
    else:
        idx = np.arange(p.n_codebook, dtype=np.float32)         # (N,)
        h = np.exp(-((idx - float(b)) ** 2) / (2.0 * (p.sigma ** 2)))[:, None]  # (N,1)
        cb_new = cb + lr * h * (x[None, :] - cb)

    return SOMState(codebook=cb_new.astype(np.float32, copy=False), bmu=b), b


def som_epoch(p: SOMParams, s: SOMState, X: Array) -> SOMState:
    """
    Apply `som_update_one` over a batch (online training).

    Parameters
    ----------
    p : SOMParams
    s : SOMState
    X : Array
        Shape (B, p.dim).

    Returns
    -------
    SOMState

    Raises
    ------
    ValueError
        If X has wrong shape.
    """
    if X.ndim != 2 or X.shape[1] != p.dim:
        raise ValueError(f"(B, {p.dim}) required, got {X.shape}")
    cur = s
    for i in range(X.shape[0]):
        cur, _ = som_update_one(p, cur, X[i])
    return cur
