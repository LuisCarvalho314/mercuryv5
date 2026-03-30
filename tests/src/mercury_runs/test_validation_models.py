from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from mercury_runs.domain.models import StudyGridConfig
from mercury_runs.io_parquet import LoadedDataset, ParquetConfig
from mercury_runs.schemas_results import ResultBundleMeta, SourceDatasetRef


def test_parquet_config_requires_run_id_for_run_selection() -> None:
    with pytest.raises(ValidationError):
        ParquetConfig(
            root=Path("datasets"),
            level=17,
            sensor="cartesian",
            select="run_id",
            run_id=None,
        )


def test_parquet_config_requires_sensor_range_for_cardinal_distance() -> None:
    with pytest.raises(ValidationError):
        ParquetConfig(
            root=Path("datasets"),
            level=17,
            sensor="cardinal distance",
            sensor_range=None,
            select="latest",
        )


def test_loaded_dataset_validates_array_shapes() -> None:
    with pytest.raises(ValidationError):
        LoadedDataset(
            observations=np.zeros((3, 2), dtype=np.float32),
            actions=np.zeros((3,), dtype=np.float32),
            collisions=np.zeros((3,), dtype=np.float32),
            source_metadata={},
            parquet_path=Path("x.parquet"),
        )


def test_loaded_dataset_validates_row_counts() -> None:
    with pytest.raises(ValidationError):
        LoadedDataset(
            observations=np.zeros((3, 2), dtype=np.float32),
            actions=np.zeros((2, 1), dtype=np.float32),
            collisions=np.zeros((3,), dtype=np.float32),
            source_metadata={},
            parquet_path=Path("x.parquet"),
        )


def test_result_bundle_meta_timestamp_format_is_validated() -> None:
    source = SourceDatasetRef(level=16, sensor="cartesian", select="latest")
    with pytest.raises(ValidationError):
        ResultBundleMeta(
            run_id="abc123",
            timestamp_utc="2026-02-02 12:00:00",
            source=source,
            sensory_params={},
            latent_params={},
            action_map_params={},
        )


def test_result_bundle_meta_accepts_full_run_parameters() -> None:
    source = SourceDatasetRef(level=16, sensor="cartesian", select="latest")
    meta = ResultBundleMeta(
        run_id="abc123",
        timestamp_utc="2026-02-26T12:00:00Z",
        source=source,
        sensory_params={"activation_threshold": 0.95},
        latent_params={"ambiguity_threshold": 10},
        action_map_params={"n_codebook": 4},
        memory_length=10,
        run_parameters={"run_payload": {"seed": 0, "num_steps": 1000}},
        source_dataset_metadata={"random_seed": 0, "num_steps": 1000},
        ground_truth_dataset_metadata={"random_seed": 0, "num_steps": 1000},
    )
    assert meta.run_parameters["run_payload"]["seed"] == 0


def test_study_grid_config_defaults_algorithm_valid_trajectory_flags() -> None:
    config = StudyGridConfig(
        level=16,
        sensor="cartesian",
        sensor_range=None,
        seed=0,
        rand_prob=0.3,
        memory_length=10,
        activation_threshold=0.95,
        ambiguity_threshold=10,
        am_lr=0.5,
        am_key=0,
        window_length=20,
        weight_memory=0.4,
        weight_undirected=0.2,
        weight_base=0.2,
        weight_action=0.2,
        memory_replay=True,
        memory_disambiguation=True,
        study_root="results/current/studies",
        study_name="default",
    )

    assert config.mercury_valid_trajectories_only is False
    assert config.mercury_split_sensory_raw_latent_valid is False
    assert config.pocml_valid_trajectories_only is False
    assert config.cscg_valid_trajectories_only is False
