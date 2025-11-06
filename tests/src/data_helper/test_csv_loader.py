# tests/test_csv_loader.py
from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest

# adjust import to your package layout
from data_helper.csv_loader import (
    CSVConfig, load_level_csv, iter_sequence, _latest_timestamp, _load_csv
)

Array = np.ndarray


def _write_csv(path: Path, arr: np.ndarray, delim: str = ","):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, arr, delimiter=delim)


def test_latest_timestamp_picks_max(tmp_path: Path):
    base = tmp_path / "datasets" / "level13" / "cartesian"
    _write_csv(base / "notes.txt", np.array([1]))
    _write_csv(base / "20230101000000_misc.csv", np.array([1]))
    _write_csv(base / "20240102030405_observations.csv", np.zeros((1, 2)))
    _write_csv(base / "20240102030405_actions.csv", np.zeros((1,)))
    _write_csv(base / "20250101000000_observations.csv", np.zeros((1, 2)))
    _write_csv(base / "20250101000000_actions.csv", np.zeros((1,)))

    ts = _latest_timestamp(base)
    assert ts == "20250101000000"


def test_load_level_csv_specific_timestamp_and_shapes(tmp_path: Path):
    base = tmp_path / "datasets" / "level5" / "cartesian"
    ts = "20241231235959"
    obs = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)  # (T=2, D=2)
    act = np.array([7.0, 8.0], dtype=np.float32)                # (T,)
    _write_csv(base / f"{ts}_observations.csv", obs)
    _write_csv(base / f"{ts}_actions.csv", act)

    cfg = CSVConfig(root=str(tmp_path / "datasets"), level=5, coords="cartesian", timestamp=ts)
    o, a = load_level_csv(cfg)

    assert np.array_equal(o, obs)
    assert np.array_equal(a, act)
    assert o.shape == (2, 2)
    assert a.shape == (2,)


def test_load_level_csv_uses_latest_when_timestamp_none(tmp_path: Path):
    base = tmp_path / "datasets" / "level1" / "polar"
    old_ts, new_ts = "20240101000000", "20240202000000"

    _write_csv(base / f"{old_ts}_observations.csv", np.ones((2, 3)))
    _write_csv(base / f"{old_ts}_actions.csv",     np.ones((2,)))
    _write_csv(base / f"{new_ts}_observations.csv", 2 * np.ones((2, 3)))
    _write_csv(base / f"{new_ts}_actions.csv",       2 * np.ones((2,)))

    cfg = CSVConfig(root=str(tmp_path / "datasets"), level=1, coords="polar")
    o, a = load_level_csv(cfg)
    assert float(o[0, 0]) == 2.0 and float(a[0]) == 2.0


def test_missing_files_raise(tmp_path: Path):
    base = tmp_path / "datasets" / "level2" / "cartesian"
    ts = "20240101000000"
    _write_csv(base / f"{ts}_observations.csv", np.zeros((2, 2)))
    cfg = CSVConfig(root=str(tmp_path / "datasets"), level=2, coords="cartesian", timestamp=ts)
    with pytest.raises(FileNotFoundError):
        load_level_csv(cfg)


def test_no_timestamped_files_raise(tmp_path: Path):
    base = tmp_path / "datasets" / "level3" / "cartesian"
    base.mkdir(parents=True, exist_ok=True)
    (base / "readme.md").write_text("no timestamps here")
    cfg = CSVConfig(root=str(tmp_path / "datasets"), level=3, coords="cartesian", timestamp=None)
    with pytest.raises(FileNotFoundError):
        load_level_csv(cfg)


def test_dtype_and_delimiter_respected(tmp_path: Path):
    base = tmp_path / "datasets" / "level4" / "cartesian"
    ts = "20240303030303"
    _write_csv(base / f"{ts}_observations.csv", np.array([[1, 2], [3, 4]], dtype=np.float64), delim=";")
    _write_csv(base / f"{ts}_actions.csv", np.array([5, 6], dtype=np.float32), delim=";")

    cfg = CSVConfig(root=str(tmp_path / "datasets"), level=4, coords="cartesian",
                    timestamp=ts, delimiter=";", dtype=np.float32)
    o, a = load_level_csv(cfg)
    assert o.dtype == np.float32
    assert a.dtype == np.float32


def test_actions_single_column_end_up_1d(tmp_path: Path):
    base = tmp_path / "datasets" / "level6" / "cartesian"
    ts = "20240505050505"
    _write_csv(base / f"{ts}_observations.csv", np.arange(6, dtype=float).reshape(3, 2))
    _write_csv(base / f"{ts}_actions.csv", np.array([1, 2, 3], dtype=float).reshape(3,))  # 1D

    cfg = CSVConfig(root=str(tmp_path / "datasets"), level=6, coords="cartesian", timestamp=ts)
    _, a = load_level_csv(cfg)
    assert a.ndim == 1 and a.shape == (3,)


def test_iter_sequence_yields_in_order_and_types(tmp_path: Path):
    base = tmp_path / "datasets" / "level7" / "cartesian"
    ts = "20240606060606"
    obs = np.array([[10, 11], [12, 13], [14, 15]], dtype=np.float32)
    act = np.array([0, 1, 2], dtype=np.float32)
    _write_csv(base / f"{ts}_observations.csv", obs)
    _write_csv(base / f"{ts}_actions.csv", act)

    cfg = CSVConfig(root=str(tmp_path / "datasets"), level=7, coords="cartesian", timestamp=ts)
    o, a = load_level_csv(cfg)

    seq = list(iter_sequence(o, a))
    assert len(seq) == 3
    for i, (oi, ai) in enumerate(seq):
        assert np.array_equal(oi, o[i])
        assert float(ai) == float(a[i])
        assert isinstance(oi, np.ndarray)
        assert isinstance(ai, (np.ndarray, np.floating, float))


def test__load_csv_returns_numpy_array(tmp_path: Path):
    p = tmp_path / "x.csv"
    _write_csv(p, np.array([[1, 2], [3, 4]], dtype=np.float32))
    arr = _load_csv(p, ",", np.float32)
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (2, 2)


def test_ignores_other_files_and_picks_pair(tmp_path: Path):
    base = tmp_path / "datasets" / "level8" / "cartesian"
    ts = "20240707070707"
    _write_csv(base / f"{ts}_observations.csv", np.ones((2, 2)))
    _write_csv(base / f"{ts}_actions.csv", np.zeros((2,)))
    _write_csv(base / f"{ts}_misc.csv", np.array([123]))

    cfg = CSVConfig(root=str(tmp_path / "datasets"), level=8, coords="cartesian", timestamp=None)
    o, a = load_level_csv(cfg)
    assert o.shape == (2, 2)
    assert a.shape == (2,)
