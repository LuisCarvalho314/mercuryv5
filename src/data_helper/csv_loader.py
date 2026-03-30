# /data/csv_loader.py
from __future__ import annotations
from pathlib import Path
from typing import Any, Iterator, Tuple
import re
import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator

Array = np.ndarray


class CSVConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True, str_strip_whitespace=True)

    root: Path = Path(__file__).resolve().parent.parent.parent / "datasets"
    level: int = Field(default=13, ge=0)
    coords: str = Field(default="cardinal distance/range_1", min_length=1)
    timestamp: str | None = Field(default=None, pattern=r"^\d{14}$")
    delimiter: str = Field(default=",", min_length=1)
    dtype: Any = np.float32

    @field_validator("dtype")
    @classmethod
    def _validate_dtype(cls, value: Any) -> np.dtype:
        return np.dtype(value)


def _latest_timestamp(dirpath: Path, *, include_collisions: bool = False) -> str:
    """Return max timestamp with a complete set of required files."""
    if not dirpath.exists():
        raise FileNotFoundError(f"Data directory missing: {dirpath}")

    required_suffixes = {"observations", "actions"}
    if include_collisions:
        required_suffixes.add("collisions")

    by_timestamp: dict[str, set[str]] = {}
    for p in dirpath.iterdir():
        m = re.match(r"(\d{14})_(observations|actions|collisions)\.csv$", p.name)
        if m:
            by_timestamp.setdefault(m.group(1), set()).add(m.group(2))

    candidates = [ts for ts, found in by_timestamp.items() if required_suffixes.issubset(found)]
    if not candidates:
        required = ", ".join(sorted(required_suffixes))
        raise FileNotFoundError(f"No complete timestamped dataset ({required}) in {dirpath}")
    return max(candidates)


def _load_csv(path: Path, delim: str, dtype: np.dtype) -> Array:
    """Load a delimited file into a NumPy array."""
    return np.loadtxt(path, delimiter=delim, dtype=dtype)


def load_level_csv(cfg: CSVConfig, *, include_collisions: bool = False) -> Tuple[Array, Array] | Tuple[Array, Array, Array]:
    """
    Load observation and action CSVs for a level/coords/timestamp.

    Returns
    -------
    (obs, act)
        obs: (T, D)
        act: (T,) or (T, A)
    """
    base = cfg.root / f"level{cfg.level}" / cfg.coords
    ts = cfg.timestamp or _latest_timestamp(base, include_collisions=include_collisions)
    obs_p = base / f"{ts}_observations.csv"
    act_p = base / f"{ts}_actions.csv"
    if not obs_p.exists() or not act_p.exists():
        raise FileNotFoundError(f"Missing observations/actions files for timestamp {ts} under {base}")

    obs = _load_csv(obs_p, cfg.delimiter, cfg.dtype)
    act = _load_csv(act_p, cfg.delimiter, cfg.dtype)
    if act.ndim > 1 and act.shape[1] == 1:
        act = act.squeeze(1)
    if obs.shape[0] != act.shape[0]:
        raise ValueError(f"Observations/actions length mismatch: {obs.shape[0]} vs {act.shape[0]}")

    if not include_collisions:
        return obs, act

    col_p = base / f"{ts}_collisions.csv"
    if not col_p.exists():
        raise FileNotFoundError(f"Missing collisions file for timestamp {ts} under {base}")
    col = _load_csv(col_p, cfg.delimiter, cfg.dtype)
    if col.ndim > 1 and col.shape[1] == 1:
        col = col.squeeze(1)
    if col.shape[0] != obs.shape[0]:
        raise ValueError(f"Observations/collisions length mismatch: {obs.shape[0]} vs {col.shape[0]}")
    return obs, act, col


def iter_sequence(obs: Array, act: Array, col: Array | None = None) -> Iterator[Tuple[Array, ...]]:
    """Yield per-step rows as (obs, act) or (obs, act, col)."""
    if obs.shape[0] != act.shape[0]:
        raise ValueError(f"Observations/actions length mismatch: {obs.shape[0]} vs {act.shape[0]}")
    if col is not None and col.shape[0] != obs.shape[0]:
        raise ValueError(f"Observations/collisions length mismatch: {obs.shape[0]} vs {col.shape[0]}")

    for i in range(obs.shape[0]):
        if col is None:
            yield obs[i], act[i]
        else:
            yield obs[i], act[i], col[i]

# def iter_sequence(obs: Array, act: Array) -> Iterator[
#     Tuple[Array,
#     Array, Array]]:
#     """Yield (obs_t, act_t) pairs in sequence order."""
#     for i in range(obs.shape[0]):
#         yield obs[i], act[i]
