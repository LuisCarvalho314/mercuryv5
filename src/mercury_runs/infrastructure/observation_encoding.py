from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .paper_precision import map_values_with_reference


@dataclass(frozen=True)
class CSCGOneHotEncoding:
    obs_uniques: np.ndarray
    observations: np.ndarray
    indices: np.ndarray
    unsupported_rows: int = 0


def cscg_observation_uniques(observations: np.ndarray) -> np.ndarray:
    values = np.asarray(observations)
    if values.ndim == 1:
        return np.unique(values)
    return np.unique(values, axis=0)


def encode_observations_with_vocabulary(
    *,
    observations: np.ndarray,
    obs_uniques: np.ndarray,
    skip_unsupported: bool = False,
) -> CSCGOneHotEncoding:
    values = np.asarray(observations)
    vocab = np.asarray(obs_uniques)
    if not skip_unsupported:
        indices = map_values_with_reference(reference_uniques=vocab, values=values)
        return CSCGOneHotEncoding(
            obs_uniques=vocab,
            observations=np.eye(int(vocab.shape[0]), dtype=np.float32)[indices],
            indices=indices.astype(np.int64, copy=False),
        )

    encoded_rows: list[np.ndarray] = []
    mapped_indices: list[int] = []
    unsupported = 0
    for row in values:
        row_array = np.asarray(row)
        try:
            idx = int(
                map_values_with_reference(
                    reference_uniques=vocab,
                    values=row_array.reshape(1, -1) if values.ndim > 1 else np.asarray([row_array.item()]),
                )[0]
            )
        except ValueError:
            unsupported += 1
            continue
        mapped_indices.append(idx)
        encoded_rows.append(np.eye(int(vocab.shape[0]), dtype=np.float32)[idx])
    observations_one_hot = (
        np.vstack(encoded_rows).astype(np.float32, copy=False)
        if encoded_rows
        else np.zeros((0, int(vocab.shape[0])), dtype=np.float32)
    )
    return CSCGOneHotEncoding(
        obs_uniques=vocab,
        observations=observations_one_hot,
        indices=np.asarray(mapped_indices, dtype=np.int64),
        unsupported_rows=int(unsupported),
    )


def fit_cscg_one_hot_observations(observations: np.ndarray) -> CSCGOneHotEncoding:
    obs_uniques = cscg_observation_uniques(observations)
    return encode_observations_with_vocabulary(observations=observations, obs_uniques=obs_uniques)


def vocabulary_json(value: np.ndarray | None) -> list[Any] | None:
    if value is None:
        return None
    return np.asarray(value).tolist()
