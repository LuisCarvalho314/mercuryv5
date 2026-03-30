from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

GROUND_TRUTH_PROXY_COLUMN = "cartesian_proxy_bmu"
LEGACY_GROUND_TRUTH_PROXY_COLUMN = "ground_truth_bmu"


def resolve_ground_truth_proxy_column(columns: list[str]) -> str:
    if GROUND_TRUTH_PROXY_COLUMN in columns:
        return GROUND_TRUTH_PROXY_COLUMN
    if LEGACY_GROUND_TRUTH_PROXY_COLUMN in columns:
        return LEGACY_GROUND_TRUTH_PROXY_COLUMN
    raise RuntimeError(
        f"Missing proxy ground-truth column. Expected one of: {GROUND_TRUTH_PROXY_COLUMN}, {LEGACY_GROUND_TRUTH_PROXY_COLUMN}."
    )


def load_ground_truth_proxy_states(path: Path) -> np.ndarray:
    frame = pl.read_parquet(path)
    column_name = resolve_ground_truth_proxy_column(frame.columns)
    return frame.get_column(column_name).to_numpy().astype(np.int32, copy=False)
