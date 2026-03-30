from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from mercury_runs.application.study_service import (
    build_study_run_records,
    resolve_run_configurations,
    write_incomplete_report,
    _render_study_dashboard,
)
from mercury_runs.domain.models import StudyDefaults
from mercury_runs.infrastructure.progress import write_progress
from mercury_runs.interfaces.cli import parse_arguments


def test_resolve_run_configurations_expands_weight_preset(tmp_path) -> None:
    config_path = tmp_path / "weight_ablation.json"
    config_path.write_text(
        json.dumps(
            {
                "environment": {
                    "level": [16],
                    "sensor": ["cardinal distance"],
                    "sensor_range": [1],
                    "seed": [0],
                    "rand_prob": [0.3],
                    "window_length": [100],
                },
                "evaluation": {},
                "methods": {
                    "mercury": {
                        "enabled": True,
                        "grid": {
                            "memory_length": [40],
                            "activation_threshold": [0.95],
                            "ambiguity_threshold": [10],
                            "am_lr": [0.001],
                            "am_key": [0],
                            "weight_preset": ["baseline"],
                            "memory_replay": [True],
                            "memory_disambiguation": [True],
                        },
                    },
                    "pocml": {"enabled": False, "grid": {}},
                    "cscg": {"enabled": False, "grid": {}},
                },
            }
        ),
        encoding="utf-8",
    )
    args = argparse.Namespace(study_config=str(config_path))

    runs = resolve_run_configurations(args, StudyDefaults(study_root="results/current/studies", study_name="weight_ablation"))

    assert len(runs) == 1
    assert runs[0]["study_root"] == "results/current/studies"
    assert runs[0]["study_name"] == "weight_ablation"
    assert runs[0]["reuse_existing_run"] is False
    assert runs[0]["method"] == "mercury"
    assert runs[0]["baseline_pocml"] is False
    assert runs[0]["baseline_cscg"] is False
    assert runs[0]["weight_memory"] == 0.4
    assert runs[0]["weight_undirected"] == 0.2
    assert runs[0]["weight_base"] == 0.2
    assert runs[0]["weight_action"] == 0.2
    assert runs[0]["weight_preset"] == "baseline"


def test_build_study_run_records_prunes_irrelevant_method_args(tmp_path, monkeypatch) -> None:
    manifest_path = tmp_path / "study_manifest.jsonl"
    study_directory = tmp_path / "study"
    study_directory.mkdir()
    monkeypatch.setattr(
        sys,
        "argv",
        ["main.py", "--study", "--study_root", str(tmp_path), "--study_name", "study", "--wandb_mode", "disabled"],
    )
    args = parse_arguments()

    run_records = build_study_run_records(
        args=args,
        run_configurations=[
            {
                "method": "mercury",
                "level": 16,
                "sensor": "cardinal distance",
                "sensor_range": 1,
                "seed": 0,
                "rand_prob": 0.3,
                "memory_length": 10,
                "activation_threshold": 0.95,
                "ambiguity_threshold": 10,
                "am_lr": 0.001,
                "am_key": 0,
                "window_length": 100,
                "weight_memory": 0.4,
                "weight_undirected": 0.2,
                "weight_base": 0.2,
                "weight_action": 0.2,
                "memory_replay": True,
                "memory_disambiguation": True,
                "study_root": str(tmp_path),
                "study_name": "study",
                "reuse_existing_run": False,
                "pocml_epochs": 99,
                "cscg_n_iter": 999,
            }
        ],
        study_directory=study_directory,
        manifest_path=manifest_path,
        python_executable="python3",
        main_path="main.py",
    )

    command = run_records[0]["command"]
    assert "--method" in command
    assert "mercury" in command
    assert "--pocml_epochs" not in command
    assert "--cscg_n_iter" not in command


def test_render_study_dashboard_groups_active_first_and_collapses_completed(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("shutil.get_terminal_size", lambda fallback=(120, 20): __import__("os").terminal_size((120, 20)))
    study_root = tmp_path / "studies"
    study_name = "demo"
    records = [
        {"run_id": "mactive01", "config": {"study_root": str(study_root), "study_name": study_name, "method": "mercury"}},
        {"run_id": "mdone001", "config": {"study_root": str(study_root), "study_name": study_name, "method": "mercury"}},
        {"run_id": "pactive01", "config": {"study_root": str(study_root), "study_name": study_name, "method": "pocml"}},
        {"run_id": "cdone001", "config": {"study_root": str(study_root), "study_name": study_name, "method": "cscg"}},
    ]

    mercury_active_root = study_root / study_name / "mactive01"
    write_progress(run_root=mercury_active_root, run_id="mactive01", method="mercury", track="lifecycle", stage="queued", current=0, total=1, message="Queued")
    write_progress(run_root=mercury_active_root, run_id="mactive01", method="mercury", track="train", stage="latent", current=5, total=10, message="training")
    write_progress(run_root=mercury_active_root, run_id="mactive01", method="mercury", track="eval", stage="native_eval", current=1, total=4, message="evaluating")
    write_progress(run_root=mercury_active_root, run_id="mactive01", method="mercury", track="paper_precision", stage="completed", current=1, total=1, message="done")

    mercury_done_root = study_root / study_name / "mdone001"
    write_progress(run_root=mercury_done_root, run_id="mdone001", method="mercury", track="lifecycle", stage="completed", current=1, total=1, message="Completed")
    write_progress(run_root=mercury_done_root, run_id="mdone001", method="mercury", track="train", stage="latent", current=10, total=10, message="training")

    pocml_active_root = study_root / study_name / "pactive01"
    write_progress(run_root=pocml_active_root, run_id="pactive01", method="pocml", track="lifecycle", stage="queued", current=0, total=1, message="Queued")
    write_progress(run_root=pocml_active_root, run_id="pactive01", method="pocml", track="train", stage="train", current=3, total=8, message="epoch 3/8")

    cscg_done_root = study_root / study_name / "cdone001"
    write_progress(run_root=cscg_done_root, run_id="cdone001", method="cscg", track="lifecycle", stage="failed", current=1, total=1, message="boom")

    dashboard = _render_study_dashboard(run_records=records, completed=2, total=4)
    lines = dashboard.splitlines()

    mercury_header = lines.index("MERCURY")
    pocml_header = lines.index("POCML")
    cscg_header = lines.index("CSCG")
    assert mercury_header < pocml_header < cscg_header
    assert any("mactive01 mercury" in line and "lifecycle" in line for line in lines)
    assert any("train" in line and "training" in line for line in lines)
    assert any("eval" in line and "evaluating" in line for line in lines)
    assert not any("paper_precision" in line and "mactive01" in line for line in lines)
    mercury_rows = [line for line in lines[mercury_header + 1:pocml_header] if line.strip()]
    assert "mactive01 mercury" in mercury_rows[0]
    assert any("mdone001 mercury" in line and "done" in line for line in mercury_rows)
    assert any("cdone001 cscg" in line and "failed" in line for line in lines[cscg_header + 1:])


def test_write_incomplete_report_ignores_stale_runs_outside_manifest(tmp_path, monkeypatch) -> None:
    study_directory = tmp_path / "weight_ablation3"
    study_directory.mkdir()
    manifest_path = study_directory / "study_manifest.jsonl"

    current_run_ids = ["current01", "current02"]
    manifest_path.write_text(
        "\n".join(
            json.dumps({"config": {"requested_run_id": run_id, "run_id": run_id}})
            for run_id in current_run_ids
        )
        + "\n",
        encoding="utf-8",
    )

    for run_id in current_run_ids:
        (study_directory / run_id).mkdir()

    stale_run_root = study_directory / "stale01"
    stale_run_root.mkdir()
    (stale_run_root / "run_config.json").write_text(
        json.dumps({"schema_version": 2, "run_id": "stale01", "requested_run_id": "stale01", "run_payload": {}, "cli_args": {}, "execution": {}, "paths": {}}),
        encoding="utf-8",
    )
    (stale_run_root / "run_status.json").write_text(
        json.dumps({"schema_version": 2, "run_id": "stale01", "status": "running", "paths": {"run_root": str(stale_run_root)}}),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "mercury_runs.application.study_service.run_complete",
        lambda run_root: run_root.name in current_run_ids,
    )

    report_path = write_incomplete_report(study_directory, manifest_path)

    assert report_path.exists() is False
