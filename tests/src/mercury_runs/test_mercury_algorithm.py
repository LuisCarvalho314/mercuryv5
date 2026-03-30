from __future__ import annotations

from pathlib import Path

from mercury_runs.algorithms.mercury.config import MercuryConfig
from mercury_runs.algorithms.mercury.prepare import generate_mercury_datasets
from mercury_runs.algorithms.mercury.train import run_mercury_bundle


def _config(tmp_path: Path, sensor: str = "cardinal distance", sensor_range: int | None = 1) -> MercuryConfig:
    return MercuryConfig(
        datasets_root=tmp_path / "datasets",
        results_root=tmp_path / "results",
        run_id="run-1",
        level=13,
        sensor=sensor,
        sensor_range=sensor_range,
        memory_length=10,
        window_length=20,
        seed=0,
        rand_prob=0.3,
        num_steps=100,
        sensory={},
        latent={},
        action_map={"n_codebook": 4, "lr": 0.5, "sigma": 0.0, "key": 0},
    )


def test_run_mercury_bundle_pins_generated_dataset_pair(monkeypatch, tmp_path: Path) -> None:
    captured: dict = {}

    def _capture(**kwargs):
        captured.update(kwargs)
        return tmp_path / "states.parquet"

    monkeypatch.setattr("mercury_runs.algorithms.mercury.train.run_all_bundled", _capture)

    result = run_mercury_bundle(
        config=_config(tmp_path),
        dataset_run_ids={
            "cardinal distance:range=1": "sensor-run",
            "cartesian": "cart-run",
        },
    )

    assert result == tmp_path / "states.parquet"
    assert captured["dataset_select"] == "run_id"
    assert captured["dataset_run_id"] == "sensor-run"
    assert captured["ground_truth_dataset_run_id"] == "cart-run"
    assert captured["run_parameters"]["execution"]["dataset_select"] == "run_id"
    assert captured["run_parameters"]["execution"]["dataset_run_id"] == "sensor-run"
    assert captured["run_parameters"]["execution"]["ground_truth_dataset_run_id"] == "cart-run"


def test_run_mercury_bundle_falls_back_to_latest_without_generated_ids(monkeypatch, tmp_path: Path) -> None:
    captured: dict = {}

    def _capture(**kwargs):
        captured.update(kwargs)
        return tmp_path / "states.parquet"

    monkeypatch.setattr("mercury_runs.algorithms.mercury.train.run_all_bundled", _capture)

    run_mercury_bundle(config=_config(tmp_path, sensor="cartesian", sensor_range=None), dataset_run_ids={})

    assert captured["dataset_select"] == "latest"
    assert captured["dataset_run_id"] is None
    assert captured["ground_truth_dataset_run_id"] is None


def test_run_mercury_bundle_passes_algorithm_valid_trajectory_flag(monkeypatch, tmp_path: Path) -> None:
    captured: dict = {}

    def _capture(**kwargs):
        captured.update(kwargs)
        return tmp_path / "states.parquet"

    monkeypatch.setattr("mercury_runs.algorithms.mercury.train.run_all_bundled", _capture)

    run_mercury_bundle(
        config=_config(tmp_path).model_copy(update={"mercury_valid_trajectories_only": True}),
        dataset_run_ids={},
    )

    assert captured["mercury_valid_trajectories_only"] is True
    assert captured["run_parameters"]["execution"]["mercury_valid_trajectories_only"] is True


def test_run_mercury_bundle_passes_split_trajectory_flag(monkeypatch, tmp_path: Path) -> None:
    captured: dict = {}

    def _capture(**kwargs):
        captured.update(kwargs)
        return tmp_path / "states.parquet"

    monkeypatch.setattr("mercury_runs.algorithms.mercury.train.run_all_bundled", _capture)

    run_mercury_bundle(
        config=_config(tmp_path).model_copy(update={"mercury_split_sensory_raw_latent_valid": True}),
        dataset_run_ids={},
    )

    assert captured["mercury_split_sensory_raw_latent_valid"] is True
    assert captured["run_parameters"]["execution"]["mercury_split_sensory_raw_latent_valid"] is True


def test_generate_mercury_datasets_passes_valid_trajectory_flag(monkeypatch, tmp_path: Path) -> None:
    captured: dict = {}

    def _capture(**kwargs):
        captured.update(kwargs)
        return {"cartesian": "cart-run"}

    monkeypatch.setattr("src.maze_environment.generate_data.generate_data", _capture)

    result = generate_mercury_datasets(config=_config(tmp_path).model_copy(update={"valid_trajectories_only": True}))

    assert result == {"cartesian": "cart-run"}
    assert captured["valid_trajectories_only"] is True
