# /data/csv_loader.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Tuple
import re
import numpy as np

Array = np.ndarray


@dataclass(frozen=True)
class CSVConfig:
    root: str = f"{Path(__file__).resolve().parent.parent.parent}/datasets"
    level: int = 13
    coords: str = "cartesian"
    timestamp: str | None = None        # "YYYYMMDDHHMMSS" or None => latest
    delimiter: str = ","
    dtype: np.dtype = np.float32


def _latest_timestamp(dirpath: Path) -> str:
    """Return max 14-digit timestamp found in filenames under dirpath."""
    if not dirpath.exists():
        raise FileNotFoundError(f"Data directory missing: {dirpath}")
    candidates = []
    for p in dirpath.iterdir():
        m = re.search(r"(\d{14})", p.name)
        if m:
            candidates.append(m.group(1))
    if not candidates:
        raise FileNotFoundError(f"No timestamped files in {dirpath}")
    return max(candidates)


def _load_csv(path: Path, delim: str, dtype: np.dtype) -> Array:
    """Load a delimited file into a NumPy array."""
    return np.loadtxt(path, delimiter=delim, dtype=dtype)


def load_level_csv(cfg: CSVConfig) -> Tuple[Array, Array]:
    """
    Load observation and action CSVs for a level/coords/timestamp.

    Returns
    -------
    (obs, act)
        obs: (T, D)
        act: (T,) or (T, A)
    """
    base = Path(cfg.root) / f"level{cfg.level}" / cfg.coords
    ts = cfg.timestamp or _latest_timestamp(base)
    obs_p = base / f"{ts}_observations.csv"
    act_p = base / f"{ts}_actions.csv"
    if not obs_p.exists() or not act_p.exists():
        raise FileNotFoundError(f"Missing files for timestamp {ts} under {base}")
    obs = _load_csv(obs_p, cfg.delimiter, cfg.dtype)
    act = _load_csv(act_p, cfg.delimiter, cfg.dtype)
    if act.ndim > 1 and act.shape[1] == 1:
        act = act.squeeze(1)
    return obs, act


def iter_sequence(obs: Array, act: Array) -> Iterator[Tuple[Array, Array]]:
    """Yield (obs_t, act_t) pairs in sequence order."""
    for i in range(obs.shape[0]):
        yield obs[i], act[i]
