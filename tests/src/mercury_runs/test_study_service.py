from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from mercury_runs.application.study_service import (
    apply_ambiguity_response_preset,
    apply_cscg_structural_preset,
    build_study_run_records,
    resolve_run_configurations,
    write_incomplete_report,
    _write_study_artifacts,
    _render_study_dashboard,
)
from mercury_runs.domain.models import StudyDefaults
from mercury_runs.infrastructure.progress import write_progress
from mercury_runs.interfaces.cli import parse_arguments


def test_ambiguity_response_presets_define_four_orthogonal_conditions() -> None:
    expected = {
        "no_ambiguity_response": (False, False, False),
        "local_repair_only": (True, False, True),
        "refinement_without_reconstruction": (True, True, False),
        "full_mercury": (True, True, True),
    }

    for mode, flags in expected.items():
        run = apply_ambiguity_response_preset({"ambiguity_response_mode": mode})
        assert (
            run["ambiguity_response_enabled"],
            run["ambiguity_refinement_enabled"],
            run["transition_reconstruction_enabled"],
        ) == flags


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


def test_resolve_run_configurations_expands_weight_preset_for_hybrid(tmp_path) -> None:
    config_path = tmp_path / "hybrid_weight_preset.json"
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
                    "mercury": {"enabled": False, "grid": {}},
                    "pocml": {"enabled": False, "grid": {}},
                    "cscg": {"enabled": False, "grid": {}},
                    "mercury_cscg_hybrid": {
                        "enabled": True,
                        "grid": {
                            "memory_length": [10],
                            "activation_threshold": [0.95],
                            "ambiguity_threshold": [30],
                            "am_lr": [0.5],
                            "am_key": [0],
                            "weight_preset": ["only_memory"],
                            "memory_replay": [True],
                            "memory_disambiguation": [True],
                        },
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    runs = resolve_run_configurations(
        argparse.Namespace(study_config=str(config_path)),
        StudyDefaults(study_root="results/current/studies", study_name="hybrid_weight_preset"),
    )

    assert len(runs) == 1
    assert runs[0]["method"] == "mercury_cscg_hybrid"
    assert runs[0]["weight_memory"] == 1.0
    assert runs[0]["weight_undirected"] == 0.0
    assert runs[0]["weight_base"] == 0.0
    assert runs[0]["weight_action"] == 0.0
    assert runs[0]["weight_preset"] == "only_memory"


def test_resolve_run_configurations_expands_cscg_clone_preset(tmp_path) -> None:
    config_path = tmp_path / "clone_preset.json"
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
                    "mercury": {"enabled": False, "grid": {}},
                    "pocml": {"enabled": False, "grid": {}},
                    "cscg": {
                        "enabled": True,
                        "grid": {
                            "cscg_clone_preset": ["ground_truth_state_2", "uniform_obs_4"],
                            "cscg_n_iter": [1000],
                            "cscg_pseudocount": [0.002],
                            "cscg_term_early": [False],
                            "cscg_train_algorithm": ["em"],
                            "cscg_training_mode": ["full"],
                            "cscg_batch_size": [2000],
                            "cscg_online_lambda": [1.0],
                        },
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    runs = resolve_run_configurations(
        argparse.Namespace(study_config=str(config_path)),
        StudyDefaults(study_root="results/current/studies", study_name="clone_preset"),
    )

    assert len(runs) == 2
    assert {
        (
            run["cscg_clone_preset"],
            run["cscg_clone_allocation"],
            run["cscg_clones_per_state"],
            run["cscg_clones_per_obs"],
        )
        for run in runs
    } == {
        ("ground_truth_state_2", "ground_truth", 2, 1),
        ("uniform_obs_4", "uniform", 1, 4),
    }


def test_resolve_run_configurations_expands_clone_preset_for_hybrid(tmp_path) -> None:
    config_path = tmp_path / "hybrid_clone_preset.json"
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
                    "mercury": {"enabled": False, "grid": {}},
                    "pocml": {"enabled": False, "grid": {}},
                    "cscg": {"enabled": False, "grid": {}},
                    "mercury_cscg_hybrid": {
                        "enabled": True,
                        "grid": {
                            "memory_length": [10],
                            "activation_threshold": [0.95],
                            "ambiguity_threshold": [30],
                            "am_lr": [0.5],
                            "am_key": [0],
                            "weight_preset": ["only_memory"],
                            "memory_replay": [True],
                            "memory_disambiguation": [True],
                            "cscg_clone_preset": ["ground_truth_state_2", "uniform_obs_4"],
                        },
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    runs = resolve_run_configurations(
        argparse.Namespace(study_config=str(config_path)),
        StudyDefaults(study_root="results/current/studies", study_name="hybrid_clone_preset"),
    )

    assert len(runs) == 2
    by_preset = {run["cscg_clone_preset"]: run for run in runs}
    assert by_preset["ground_truth_state_2"]["cscg_clone_allocation"] == "ground_truth"
    assert by_preset["ground_truth_state_2"]["cscg_clones_per_state"] == 2
    assert by_preset["uniform_obs_4"]["cscg_clone_allocation"] == "uniform"
    assert by_preset["uniform_obs_4"]["cscg_clones_per_obs"] == 4


def test_apply_cscg_structural_preset_expands_default_variants() -> None:
    vanilla = apply_cscg_structural_preset({"cscg_structural_preset": "vanilla_cscg"})
    assert vanilla["cscg_structural_init_mode"] == "none"
    assert vanilla["cscg_structural_source"] == "none"
    assert vanilla["cscg_clone_capacity_mode"] == "fixed"
    assert vanilla["cscg_clone_allocation"] == "uniform"

    capacity = apply_cscg_structural_preset({"cscg_structural_preset": "oracle_capacity"})
    assert capacity["cscg_structural_init_mode"] == "none"
    assert capacity["cscg_clone_capacity_mode"] == "ground_truth"
    assert capacity["cscg_clone_allocation"] == "ground_truth"

    structural = apply_cscg_structural_preset({"cscg_structural_preset": "oracle_capacity_structural"})
    assert structural["cscg_structural_transfer_level"] == "clone_state"
    assert structural["cscg_structural_init_mode"] == "binary"
    assert structural["cscg_structural_source"] == "oracle_maze_topology"

    first_order = apply_cscg_structural_preset({"cscg_structural_preset": "oracle_capacity_first_order"})
    assert first_order["cscg_structural_transfer_level"] == "first_order_observation"
    assert first_order["cscg_structural_init_mode"] == "binary"
    assert first_order["cscg_structural_source"] == "oracle_maze_topology"


def test_resolve_run_configurations_expands_structural_preset_for_hybrids(tmp_path) -> None:
    config_path = tmp_path / "hybrid_structural_preset.json"
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
                    "mercury": {"enabled": False, "grid": {}},
                    "pocml": {"enabled": False, "grid": {}},
                    "cscg": {"enabled": False, "grid": {}},
                    "mercury_cscg_hybrid": {
                        "enabled": True,
                        "grid": {
                            "memory_length": [10],
                            "activation_threshold": [0.95],
                            "ambiguity_threshold": [30],
                            "am_lr": [0.5],
                            "am_key": [0],
                            "weight_preset": ["only_memory"],
                            "memory_replay": [True],
                            "memory_disambiguation": [True],
                            "cscg_structural_preset": ["source_graph_clone_state"],
                        },
                    },
                    "tmgwr_cscg_hybrid": {
                        "enabled": True,
                        "grid": {
                            "memory_length": [10],
                            "activation_threshold": [0.95],
                            "ambiguity_threshold": [30],
                            "am_lr": [0.5],
                            "am_key": [0],
                            "memory_replay": [True],
                            "memory_disambiguation": [True],
                            "cscg_structural_preset": ["source_graph_first_order"],
                        },
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    runs = resolve_run_configurations(
        argparse.Namespace(study_config=str(config_path)),
        StudyDefaults(study_root="results/current/studies", study_name="hybrid_structural_preset"),
    )

    by_method = {run["method"]: run for run in runs}
    mercury_hybrid = by_method["mercury_cscg_hybrid"]
    assert mercury_hybrid["cscg_structural_preset"] == "source_graph_clone_state"
    assert mercury_hybrid["cscg_structural_source"] == "source_graph"
    assert mercury_hybrid["cscg_structural_transfer_level"] == "clone_state"
    assert mercury_hybrid["cscg_clone_capacity_mode"] == "source"

    tmgwr_hybrid = by_method["tmgwr_cscg_hybrid"]
    assert tmgwr_hybrid["cscg_structural_preset"] == "source_graph_first_order"
    assert tmgwr_hybrid["cscg_structural_source"] == "source_graph"
    assert tmgwr_hybrid["cscg_structural_transfer_level"] == "first_order_observation"
    assert tmgwr_hybrid["cscg_clone_capacity_mode"] == "source"


def test_default_study_json_schedules_standalone_default_matrix() -> None:
    config_path = ROOT / "configs" / "studies" / "study.json"
    runs = resolve_run_configurations(
        argparse.Namespace(study_config=str(config_path)),
        StudyDefaults(study_root="results/current/studies", study_name="default_matrix"),
    )

    assert len(runs) == 6
    assert [run["method"] for run in runs].count("cscg") == 4
    assert [run["method"] for run in runs].count("mercury_cscg_hybrid") == 1
    assert [run["method"] for run in runs].count("tmgwr_cscg_hybrid") == 1
    assert all(run["baseline_pocml"] is False for run in runs)
    assert all(run["baseline_cscg"] is False for run in runs)
    assert all(run["baseline_all"] is False for run in runs)
    assert {
        run["cscg_structural_preset"]
        for run in runs
    } == {
        "vanilla_cscg",
        "oracle_capacity",
        "oracle_capacity_structural",
        "oracle_capacity_first_order",
        "source_graph_clone_state",
        "source_graph_first_order",
    }


def test_structure_metrics_epsilons_do_not_expand_study_grid(tmp_path) -> None:
    config_path = tmp_path / "thresholds.json"
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
                "evaluation": {"structure_metrics_epsilons": [1e-6, 1e-4, 1e-3]},
                "methods": {
                    "mercury": {
                        "enabled": True,
                        "grid": {
                            "memory_length": [40],
                            "activation_threshold": [0.95],
                            "ambiguity_threshold": [10],
                            "am_lr": [0.001],
                            "am_key": [0],
                            "weight_memory": [0.4],
                            "weight_undirected": [0.2],
                            "weight_base": [0.2],
                            "weight_action": [0.2],
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

    runs = resolve_run_configurations(
        argparse.Namespace(study_config=str(config_path)),
        StudyDefaults(study_root="results/current/studies", study_name="thresholds"),
    )

    assert len(runs) == 1
    assert runs[0]["structure_metrics_epsilons"] == [1e-6, 1e-4, 1e-3]


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


def test_default_matrix_smoke_config_is_lean_and_expands(tmp_path) -> None:
    config_path = ROOT / "configs" / "studies" / "default_matrix_smoke_3_seed.json"

    runs = resolve_run_configurations(
        argparse.Namespace(study_config=str(config_path)),
        StudyDefaults(study_root=str(tmp_path), study_name="smoke"),
    )

    assert len(runs) == 72
    assert sorted({run["seed"] for run in runs}) == [0, 1, 2]
    assert sorted({run["cscg_batch_size"] for run in runs}) == [1, 10, 40, 100]
    tmgwr = [run for run in runs if run["method"] == "tmgwr_cscg_hybrid"]
    assert len(tmgwr) == 12
    assert all("memory_length" not in run for run in tmgwr)
    assert all("latent_max_neurons" not in run for run in tmgwr)
    assert {run["cscg_structural_preset"] for run in tmgwr} == {"source_graph_first_order"}


def test_render_study_dashboard_shows_active_runs_only(tmp_path, monkeypatch) -> None:
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

    assert lines[0] == "Study progress: 2/4 completed"
    mercury_header = lines.index("MERCURY")
    pocml_header = lines.index("POCML")
    assert mercury_header < pocml_header
    assert "CSCG" not in lines
    assert any("mactive01 mercury" in line and "lifecycle" in line for line in lines)
    assert any("train" in line and "training" in line for line in lines)
    assert any("eval" in line and "evaluating" in line for line in lines)
    assert not any("paper_precision" in line and "mactive01" in line for line in lines)
    mercury_rows = [line for line in lines[mercury_header + 1:pocml_header] if line.strip()]
    assert "mactive01 mercury" in mercury_rows[0]
    assert not any("mdone001" in line or "cdone001" in line for line in lines)


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


def test_write_study_artifacts_generates_summary_and_plots(tmp_path, monkeypatch, capsys) -> None:
    summary_path = tmp_path / "studies" / "demo" / "study_summary.parquet"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text("summary", encoding="utf-8")
    captured: dict[str, object] = {}

    def _fake_write_study_summary(*, study_root: Path, study_name: str) -> Path:
        captured["summary_args"] = (study_root, study_name)
        return summary_path

    def _fake_generate_study_history_plots(*, study_root: Path) -> dict[str, Path]:
        captured["plot_study_root"] = study_root
        return {"level=13|metric=precision|view=all": study_root / "plots" / "study_history" / "level_13" / "precision" / "all_vary" / "plot.png"}

    monkeypatch.setattr("mercury_runs.application.study_service.write_study_summary", _fake_write_study_summary)
    monkeypatch.setattr("mercury_runs.application.study_service.generate_study_history_plots", _fake_generate_study_history_plots)

    output = _write_study_artifacts(study_root=tmp_path / "studies", study_name="demo")

    assert output == summary_path
    assert captured["summary_args"] == (tmp_path / "studies", "demo")
    assert captured["plot_study_root"] == tmp_path / "studies" / "demo"
    assert "Wrote study history plots" in capsys.readouterr().out


def test_write_study_artifacts_keeps_summary_when_plots_fail(tmp_path, monkeypatch, capsys) -> None:
    summary_path = tmp_path / "studies" / "demo" / "study_summary.parquet"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text("summary", encoding="utf-8")

    monkeypatch.setattr(
        "mercury_runs.application.study_service.write_study_summary",
        lambda *, study_root, study_name: summary_path,
    )

    def _raise(*, study_root: Path) -> dict[str, Path]:
        raise RuntimeError("boom")

    monkeypatch.setattr("mercury_runs.application.study_service.generate_study_history_plots", _raise)

    output = _write_study_artifacts(study_root=tmp_path / "studies", study_name="demo")

    assert output == summary_path
    assert "Study history plot generation failed: boom" in capsys.readouterr().out
