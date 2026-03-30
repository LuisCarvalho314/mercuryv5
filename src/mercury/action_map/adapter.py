# action_map/adapter.py
"""
Minimal stateful wrapper over the functional SOM.

Purpose
-------
- Hold params and state.
- Provide simple constructors.
- Expose step() and predict() methods for single or batched usage.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np

from .som import (
    SOMParams,
    SOMState,
    som_init,
    som_init_zeros,
    som_update_one,
    som_predict,
    som_predict_batch,
)

Array = np.ndarray


@dataclass
class ActionMap:
    """
    Self-Organizing Map for action vectors.

    Attributes
    ----------
    params : SOMParams
        Hyperparameters.
    state : SOMState
        Current codebook and last BMU.
    """
    params: SOMParams
    state: SOMState
    identity_mapping: bool = False

    # ---------- constructors ----------

    @classmethod
    def random(
        cls,
        n_codebook: int,
        dim: int,
        *,
        lr: float = 0.5,
        sigma: float = 0.0,
        key: Optional[int | np.random.Generator] = None,
    ) -> "ActionMap":
        """
        Build an ActionMap with random or zero initialization.

        Parameters
        ----------
        n_codebook : int
            Number of prototypes.
        dim : int
            Vector dimensionality.
        lr : float, default 0.5
            Learning rate.
        sigma : float, default 0.0
            Neighborhood width. <=0 winner-only.
        key : int | numpy.random.Generator | None
            RNG spec. None => zeros.

        Returns
        -------
        ActionMap
        """
        p = SOMParams(n_codebook=n_codebook, dim=dim, lr=lr, sigma=sigma)
        s = som_init(p, key) if key is not None else som_init_zeros(p)
        return cls(p, s)

    @classmethod
    def identity(cls, dim: int) -> "ActionMap":
        """
        Build a fixed identity action map for one-hot actions.

        BMU i corresponds to the i-th action basis vector.
        """
        p = SOMParams(n_codebook=dim, dim=dim, lr=0.0, sigma=0.0)
        codebook = np.eye(dim, dtype=np.float32)
        return cls(p, SOMState(codebook=codebook, bmu=0), identity_mapping=True)

    # ---------- API ----------

    def step(self, action: Array) -> Tuple[int, Array]:
        """
        Update with one action and return (bmu, prototype).

        Parameters
        ----------
        action : Array
            Shape (params.dim,).

        Returns
        -------
        (int, Array)
            BMU index and its updated prototype, shape (params.dim,).
        """
        if self.identity_mapping:
            action = np.asarray(action, dtype=np.float32)
            if action.shape != (self.params.dim,):
                raise ValueError(f"action shape ({self.params.dim},) required, got {action.shape}")
            if not np.isclose(float(action.sum()), 1.0) or np.count_nonzero(action > 0.5) != 1:
                raise ValueError("identity action map requires one-hot actions")
            bmu = int(np.argmax(action))
            self.state = SOMState(codebook=self.state.codebook, bmu=bmu)
            return bmu, self.state.codebook[bmu].copy()
        self.state, bmu = som_update_one(self.params, self.state, action)
        return bmu, self.state.codebook[bmu].copy()

    def predict(self, *, action: Array | None = None, actions: Array | None = None):
        """
        Predict BMU(s) without state updates.

        Parameters
        ----------
        action : Array | None
            Single vector (params.dim,). Mutually exclusive with `actions`.
        actions : Array | None
            Batch (B, params.dim). Mutually exclusive with `action`.

        Returns
        -------
        int | Array
            BMU index for single input, or (B,) int32 for batch.

        Raises
        ------
        ValueError
            If both or neither arguments are provided.
        """
        if (action is None) == (actions is None):
            raise ValueError("pass exactly one of action= or actions=")
        if self.identity_mapping:
            if action is not None:
                action = np.asarray(action, dtype=np.float32)
                if action.shape != (self.params.dim,):
                    raise ValueError(f"action shape ({self.params.dim},) required, got {action.shape}")
                if not np.isclose(float(action.sum()), 1.0) or np.count_nonzero(action > 0.5) != 1:
                    raise ValueError("identity action map requires one-hot actions")
                return int(np.argmax(action))
            actions = np.asarray(actions, dtype=np.float32)
            if actions.ndim != 2 or actions.shape[1] != self.params.dim:
                raise ValueError(f"(B, {self.params.dim}) required, got {actions.shape}")
            if not np.allclose(actions.sum(axis=1), 1.0) or not np.all(np.count_nonzero(actions > 0.5, axis=1) == 1):
                raise ValueError("identity action map requires one-hot actions")
            return np.argmax(actions, axis=1).astype(np.int32, copy=False)
        return som_predict(self.params, self.state, action) if action is not None \
            else som_predict_batch(self.params, self.state, actions)
