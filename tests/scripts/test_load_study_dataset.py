from __future__ import annotations

from pathlib import Path
import sys

import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from scripts.analysis.load_study_dataset import main


def test_load_study_dataset_main_writes_parquet_bundle(tmp_path: Path, monkeypatch, capsys) -> None:
    study_root = tmp_path / "study"
    study_root.mkdir()
    output_dir = tmp_path / "out"

    class _Dataset:
        runs = pl.DataFrame({"run_id": ["a"]})
        history = pl.DataFrame({"run_id": ["a"], "step": [1]})
        states = pl.DataFrame({"run_id": ["a"], "step": [1]})
        attribution = pl.DataFrame({"run_id": ["a"], "step": [1]})
        artifacts = pl.DataFrame({"run_id": ["a"]})

    monkeypatch.setattr("scripts.analysis.load_study_dataset.load_mercury_study_dataset", lambda path: _Dataset())
    monkeypatch.setattr(sys, "argv", ["load_study_dataset.py", "--study-root", str(study_root), "--write-dir", str(output_dir)])

    main()

    captured = capsys.readouterr()
    assert "runs: rows=1 cols=1" in captured.out
    assert (output_dir / "runs.parquet").exists()
    assert (output_dir / "history.parquet").exists()
