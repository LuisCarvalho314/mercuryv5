"""Symmetric spatial diagnostics, separate from canonical directed evaluation."""
from __future__ import annotations

import numpy as np
from scipy.linalg import orthogonal_procrustes, subspace_angles
from scipy.spatial.distance import cdist


def spatial_laplacian(transition: np.ndarray) -> np.ndarray:
    """Symmetric normalised killed Laplacian on projected GT states.

    W=(P+P.T)/2, leakage=1-P@1, D=diag(W@1+leakage),
    L=I-D^(-1/2) W D^(-1/2). Including leakage in D prevents
    normalisation from redistributing unmatched probability to matched targets.
    Self-transitions remain in W. L is positive semidefinite.
    """
    p = np.asarray(transition, dtype=float)
    if p.ndim != 2 or p.shape[0] != p.shape[1] or not np.isfinite(p).all() or np.any(p < 0):
        raise ValueError("Expected finite nonnegative square transition matrix")
    leakage = 1 - p.sum(axis=1)
    if np.any(leakage < -1e-10):
        raise ValueError("Matched transition rows must be substochastic")
    w = (p + p.T) / 2
    degree = w.sum(axis=1) + np.maximum(leakage, 0)
    if np.any(degree <= 0):
        raise ValueError("Spatial operator has a zero total degree")
    inverse = 1 / np.sqrt(degree)
    return np.eye(len(p)) - inverse[:, None] * w * inverse[None, :]


def spatial_representation(transition: np.ndarray, *, modes: int = 2, time: float = 3., tolerance: float = 1e-7) -> dict:
    """Leading nonprincipal modes and full nonprincipal diffusion distances.

    Exclude the lowest (principal) eigenmode. In a closed connected graph this
    is the stationary mode; with leakage it is a survival/coverage mode. A tied
    principal boundary is unresolved. Do not split a tied upper mode boundary.
    Distances use ALL remaining modes with exp(-time*lambda) weighting, so an
    internal repeated eigenspace can rotate without changing distances.
    """
    operator = spatial_laplacian(transition)
    n = len(operator)
    if not 1 <= modes < n or time <= 0:
        raise ValueError("Require 1 <= modes < n_states and positive diffusion time")
    values, vectors = np.linalg.eigh(operator)
    principal_defined = values[1] - values[0] > tolerance
    upper_defined = modes == n - 1 or values[modes + 1] - values[modes] > tolerance
    basis = vectors[:, 1:modes + 1] if principal_defined and upper_defined else None
    distances = None
    if principal_defined:
        embedding = vectors[:, 1:] * np.exp(-time * values[1:])[None, :]
        distances = cdist(embedding, embedding)
    return {"operator": operator, "eigenvalues": values, "basis": basis, "distances": distances,
            "principal_defined": bool(principal_defined), "modes_defined": basis is not None}


def align_spatial_modes(basis: np.ndarray | None, reference: np.ndarray | None) -> np.ndarray | None:
    """Align the complete displayed subspace, allowing signs and rotations.

    Columns after Procrustes are GT-aligned coordinates of the eigenspace,
    rather than individually identified eigenvectors when modes are mixed.
    """
    if basis is None or reference is None:
        return None
    rotation, _ = orthogonal_procrustes(basis, reference)
    return basis @ rotation


def compare_spatial(reference: dict, learned: dict) -> dict[str, float]:
    a, b = reference["basis"], learned["basis"]
    gt_distance, distance = reference["distances"], learned["distances"]
    mask = np.triu_indices(len(reference["operator"]), k=1)
    return {
        "symmetric_eigenvalue_rmse": float(np.sqrt(np.mean((reference["eigenvalues"] - learned["eigenvalues"]) ** 2))),
        "symmetric_subspace_error": float(np.sin(subspace_angles(a, b)).max()) if a is not None and b is not None else np.nan,
        "spectral_distance_rmse": float(np.sqrt(np.mean((gt_distance[mask] - distance[mask]) ** 2))) if gt_distance is not None and distance is not None else np.nan,
    }
