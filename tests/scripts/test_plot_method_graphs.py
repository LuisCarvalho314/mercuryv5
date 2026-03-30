from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np
import polars as pl

from mercury.graph.core import Graph
from mercury_runs.algorithms.pocml.artifacts import capacity_run_id
from mercury_runs.analysis.graph_plots import _build_edge_summary_from_graph
from mercury_runs.save_results import write_bundle_parquet
from mercury_runs.schemas_results import ResultBundleMeta, SourceDatasetRef
from mercury_runs.analysis.graph_plots import _grounded_positions, generate_method_graph_plots

matplotlib.use("Agg", force=True)


def _write_dataset_parquet(datasets_root: Path) -> Path:
    dataset_dir = datasets_root / "level=0" / "sensor=cartesian"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = dataset_dir / "dataset.parquet"
    pl.DataFrame(
        {
            "observation_0": [1.0, 0.0, 1.0, 0.0],
            "observation_1": [0.0, 1.0, 0.0, 1.0],
            "action_left": [1, 0, 1, 0],
            "action_right": [0, 1, 0, 1],
        }
    ).write_parquet(dataset_path)
    return dataset_path


def _write_mercury_states(run_root: Path, dataset_name: str) -> Path:
    mercury_dir = run_root / "bundles" / "mercury" / "level=0" / "sensor=cartesian"
    meta = ResultBundleMeta(
        run_id="run-1",
        timestamp_utc="2026-03-13T10:00:00Z",
        source=SourceDatasetRef(
            level=0,
            sensor="cartesian",
            sensor_range=None,
            select="latest",
            dataset_parquet_name=dataset_name,
        ),
        sensory_params={},
        latent_params={},
        action_map_params={},
        memory_length=1,
        run_parameters={},
        source_dataset_metadata={},
        ground_truth_dataset_metadata={},
    )
    return write_bundle_parquet(
        output_dir=mercury_dir,
        run_id="run-1",
        bundle_name="states",
        columns={
            "cartesian_proxy_bmu": np.array([0, 1, 0, 1], dtype=np.int32),
            "sensory_bmu": np.array([0, 1, 0, 1], dtype=np.int32),
            "latent_bmu": np.array([0, 1, 0, 1], dtype=np.int32),
            "latent_node_count": np.array([2, 2, 2, 2], dtype=np.int32),
        },
        meta=meta,
        embed_metadata_in_parquet=True,
    )


def _write_mercury_graph(run_root: Path) -> Path:
    graph_dir = run_root / "bundles" / "mercury" / "level=0" / "sensor=cartesian"
    graph_dir.mkdir(parents=True, exist_ok=True)
    graph_path = graph_dir / "run-1_latent_graph.npz"
    graph = Graph(directed=True)
    graph.register_node_feature("activation", 1, init_value=0.0)
    graph.register_node_feature("mem_adj", 2, init_value=0.0)
    graph.register_edge_feature("action", 1, dtype=np.int32, init_value=0)
    graph.add_node(node_feat={"activation": np.array(1.0, dtype=np.float32), "mem_adj": np.array([1.0, 0.0], dtype=np.float32)})
    graph.add_node(node_feat={"activation": np.array(0.0, dtype=np.float32), "mem_adj": np.array([0.0, 1.0], dtype=np.float32)})
    graph.add_edge(0, 1, weight=1.0, edge_feat={"action": np.array([0], dtype=np.int32)})
    graph.add_edge(1, 0, weight=1.0, edge_feat={"action": np.array([1], dtype=np.int32)})
    graph.to_npz(str(graph_path))
    return graph_path


def _write_pocml_embeddings(run_root: Path) -> Path:
    pocml_dir = run_root / "bundles" / "pocml" / "level=0" / "sensor=cartesian"
    pocml_dir.mkdir(parents=True, exist_ok=True)
    out_path = pocml_dir / "run-1_embeddings.npz"
    np.savez_compressed(
        out_path,
        Q=np.array([[1.0, 0.0, -1.0], [0.0, 1.0, 0.5]], dtype=np.float32),
        V=np.array([[0.4, -0.2], [0.1, 0.3]], dtype=np.float32),
        state_sequence=np.array([0, 1, 2, 1], dtype=np.int32),
        action_sequence=np.array([0, 1, 0], dtype=np.int32),
        state_obs_from_memory=np.array([0, 1, 1], dtype=np.int32),
    )
    return out_path


def _write_pocml_states(run_root: Path, dataset_name: str, *, capacities: list[int], primary_capacity: int) -> Path:
    pocml_dir = run_root / "bundles" / "pocml" / "level=0" / "sensor=cartesian"
    meta = ResultBundleMeta(
        run_id="run-1",
        timestamp_utc="2026-03-13T10:00:00Z",
        source=SourceDatasetRef(
            level=0,
            sensor="cartesian",
            sensor_range=None,
            select="latest",
            dataset_parquet_name=dataset_name,
        ),
        sensory_params={},
        latent_params={},
        action_map_params={},
        memory_length=None,
        run_parameters={
            "pocml_eval": {
                "primary_precision_capacity": int(primary_capacity),
                "precision_capacities": [int(value) for value in capacities],
            }
        },
        source_dataset_metadata={},
        ground_truth_dataset_metadata={},
    )
    return write_bundle_parquet(
        output_dir=pocml_dir,
        run_id="run-1",
        bundle_name="states",
        columns={
            "cartesian_proxy_bmu": np.array([0, 1, 0, 1], dtype=np.int32),
            "sensory_bmu": np.array([0, 1, 0, 1], dtype=np.int32),
            "latent_bmu": np.array([0, 1, 0, 1], dtype=np.int32),
            "latent_node_count": np.array([int(primary_capacity)] * 4, dtype=np.int32),
            "sensory_proxy_state_id": np.array([0, 1, 0, 1], dtype=np.int32),
            "latent_proxy_state_id": np.array([0, 1, 0, 1], dtype=np.int32),
        },
        meta=meta,
        embed_metadata_in_parquet=True,
    )


def _write_pocml_embeddings_for_capacity(run_root: Path, *, capacity: int, primary_capacity: int) -> Path:
    pocml_dir = run_root / "bundles" / "pocml" / "level=0" / "sensor=cartesian"
    pocml_dir.mkdir(parents=True, exist_ok=True)
    run_id = capacity_run_id(base_run_id="run-1", capacity=int(capacity), primary_capacity=int(primary_capacity))
    out_path = pocml_dir / f"{run_id}_embeddings.npz"
    q = np.vstack(
        [
            np.linspace(1.0, -1.0, num=int(capacity), dtype=np.float32),
            np.linspace(0.0, 1.0, num=int(capacity), dtype=np.float32),
        ]
    )
    v = np.array([[0.4, -0.2], [0.1, 0.3]], dtype=np.float32)
    state_sequence = np.arange(4, dtype=np.int32) % int(capacity)
    np.savez_compressed(
        out_path,
        Q=q,
        V=v,
        state_sequence=state_sequence,
        action_sequence=np.array([0, 1, 0], dtype=np.int32),
        state_obs_from_memory=(np.arange(int(capacity), dtype=np.int32) % 2),
    )
    return out_path


def _write_cscg_artifacts(run_root: Path) -> tuple[Path, Path]:
    cscg_dir = run_root / "bundles" / "cscg" / "level=0" / "sensor=cartesian"
    cscg_dir.mkdir(parents=True, exist_ok=True)
    states_path = cscg_dir / "run-1_states.parquet"
    model_path = cscg_dir / "run-1_cscg_model.npz"
    pl.DataFrame(
        {
            "cartesian_proxy_bmu": [0, 1, 2, 1],
            "cscg_state_id": [0, 1, 2, 1],
        }
    ).write_parquet(states_path)
    np.savez_compressed(
        model_path,
        T=np.array(
            [
                [[0.0, 0.6, 0.0], [0.0, 0.0, 0.5], [0.4, 0.0, 0.0]],
                [[0.0, 0.0, 0.2], [0.7, 0.0, 0.0], [0.0, 0.3, 0.0]],
            ],
            dtype=np.float32,
        ),
        n_clones=np.array([2, 1], dtype=np.int32),
    )
    return states_path, model_path


def _write_run_status(run_root: Path, artifacts: dict[str, str]) -> None:
    (run_root / "run_status.json").write_text(json.dumps({"artifacts": artifacts}), encoding="utf-8")


def test_generate_method_graph_plots_all_methods(tmp_path: Path) -> None:
    datasets_root = tmp_path / "datasets"
    run_root = tmp_path / "study" / "run-1"
    dataset_path = _write_dataset_parquet(datasets_root)
    mercury_states = _write_mercury_states(run_root, dataset_path.name)
    mercury_graph = _write_mercury_graph(run_root)
    pocml_embeddings = _write_pocml_embeddings(run_root)
    cscg_states, cscg_model = _write_cscg_artifacts(run_root)
    _write_run_status(
        run_root,
        {
            "mercury_states_parquet": str(mercury_states),
            "mercury_latent_graph_npz": str(mercury_graph),
            "pocml_embeddings_npz": str(pocml_embeddings),
            "cscg_states_parquet": str(cscg_states),
            "cscg_model_npz": str(cscg_model),
        },
    )

    outputs = generate_method_graph_plots(run_root=run_root, datasets_root=datasets_root)

    assert outputs["mercury_plot"].exists()
    assert outputs["pocml_plot"].exists()
    assert outputs["cscg_plot"].exists()
    assert outputs["combined_plot"].exists()


def test_generate_method_graph_plots_writes_pocml_capacity_plots(tmp_path: Path) -> None:
    datasets_root = tmp_path / "datasets"
    run_root = tmp_path / "study" / "run-1"
    dataset_path = _write_dataset_parquet(datasets_root)
    mercury_states = _write_mercury_states(run_root, dataset_path.name)
    mercury_graph = _write_mercury_graph(run_root)
    pocml_states = _write_pocml_states(run_root, dataset_path.name, capacities=[3, 5], primary_capacity=3)
    pocml_embeddings = _write_pocml_embeddings_for_capacity(run_root, capacity=3, primary_capacity=3)
    _write_pocml_embeddings_for_capacity(run_root, capacity=5, primary_capacity=3)
    _write_run_status(
        run_root,
        {
            "mercury_states_parquet": str(mercury_states),
            "mercury_latent_graph_npz": str(mercury_graph),
            "pocml_states_parquet": str(pocml_states),
            "pocml_embeddings_npz": str(pocml_embeddings),
        },
    )

    outputs = generate_method_graph_plots(run_root=run_root, datasets_root=datasets_root)

    assert outputs["pocml_plot"].exists()
    assert outputs["pocml_plot_capacity=3"].exists()
    assert outputs["pocml_plot_capacity=5"].exists()


def test_grounded_positions_keep_cardinal_edges_in_expected_quadrants() -> None:
    positions = _grounded_positions(
        nodes=[10, 11, 12, 13],
        node_series=np.array([10, 11, 12, 13], dtype=np.int32),
        gt_series=np.array([0, 1, 2, 3], dtype=np.int32),
        action_series=np.array([0, 1, 2], dtype=np.int32),
        action_names=["north", "east", "south", "west"],
        node_obs_modes={10: 0, 11: 1, 12: 0, 13: 1},
        edge_summary={
            (10, 11): {"action_mode": 0.0, "weight": 1.0},
            (11, 12): {"action_mode": 1.0, "weight": 1.0},
            (12, 13): {"action_mode": 2.0, "weight": 1.0},
        },
    )

    north_vec = positions[11] - positions[10]
    east_vec = positions[12] - positions[11]
    south_vec = positions[13] - positions[12]

    assert north_vec[1] > 0.0
    assert east_vec[0] > 0.0
    assert south_vec[1] < 0.0


def test_grounded_positions_separate_different_observation_nodes() -> None:
    positions = _grounded_positions(
        nodes=[20, 21, 22, 23],
        node_series=np.array([20, 21, 22, 23], dtype=np.int32),
        gt_series=np.array([0, 0, 1, 1], dtype=np.int32),
        action_series=np.array([1, 0, 1], dtype=np.int32),
        action_names=["north", "east", "south", "west"],
        node_obs_modes={20: 0, 21: 1, 22: 0, 23: 1},
        edge_summary={
            (20, 22): {"action_mode": 1.0, "weight": 1.0},
            (21, 23): {"action_mode": 1.0, "weight": 1.0},
        },
    )

    same_anchor_dist = float(np.linalg.norm(positions[21] - positions[20]))
    second_anchor_dist = float(np.linalg.norm(positions[23] - positions[22]))

    assert same_anchor_dist > 0.25
    assert second_anchor_dist > 0.25


def test_generate_method_graph_plots_mercury_only(tmp_path: Path) -> None:
    datasets_root = tmp_path / "datasets"
    run_root = tmp_path / "study" / "run-1"
    dataset_path = _write_dataset_parquet(datasets_root)
    mercury_states = _write_mercury_states(run_root, dataset_path.name)
    mercury_graph = _write_mercury_graph(run_root)
    _write_run_status(
        run_root,
        {
            "mercury_states_parquet": str(mercury_states),
            "mercury_latent_graph_npz": str(mercury_graph),
        },
    )

    outputs = generate_method_graph_plots(run_root=run_root, datasets_root=datasets_root)

    assert set(outputs.keys()) == {"mercury_plot", "combined_plot"}
    assert outputs["mercury_plot"].exists()
    assert outputs["combined_plot"].exists()


def test_build_edge_summary_from_graph_uses_saved_graph_edges_not_rollout_modes() -> None:
    graph = Graph(directed=True)
    graph.register_node_feature("activation", 1, init_value=0.0)
    graph.register_node_feature("mem_adj", 2, init_value=0.0)
    graph.register_edge_feature("action", 1, dtype=np.int32, init_value=0)
    graph.add_node(node_feat={"activation": np.array(1.0, dtype=np.float32), "mem_adj": np.array([1.0, 0.0], dtype=np.float32)})
    graph.add_node(node_feat={"activation": np.array(0.0, dtype=np.float32), "mem_adj": np.array([0.0, 1.0], dtype=np.float32)})
    graph.add_edge(0, 1, weight=0.8, edge_feat={"action": np.array([1], dtype=np.int32)})
    graph.add_edge(1, 0, weight=0.3, edge_feat={"action": np.array([0], dtype=np.int32)})

    summary = _build_edge_summary_from_graph(graph)

    assert summary == {
        (0, 1): {"action_mode": 1.0, "count": 0.800000011920929, "weight": 0.800000011920929},
        (1, 0): {"action_mode": 0.0, "count": 0.30000001192092896, "weight": 0.30000001192092896},
    }
