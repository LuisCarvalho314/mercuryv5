from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib
import numpy as np
import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mercury_runs.analysis.plotting import (
    AlignmentHeatmapResult,
    AlignmentHistoryResult,
    CanonicalTitleParameterSpec,
    CooccurrenceHistorySurfaceResult,
    LearnedStateIdentityGraphResult,
    NotebookPlot,
    NotebookSeries,
    NotebookTitleSpec,
    ObservationHistoryResult,
    TitleParameterSpec,
    dominant_observation_ids_by_learned_state,
    load_cooccurrence_history_tensor_from_paper_precision_json,
    load_cooccurrence_matrix_from_paper_precision_json,
    observation_history_ids_by_checkpoint,
    render_alignment_history_from_paper_precision_json,
    render_cooccurrence_history_surface_from_paper_precision_json,
    render_learned_state_identity_graph,
    render_observation_history_from_paper_precision_json,
    render_alignment_contingency_heatmap,
    render_alignment_contingency_heatmap_from_paper_precision_json,
    prefilled_title_spec_from_canonical_parameters,
    notebook_plot_to_plot_request,
    prefilled_title_spec_for_varied_parameters,
    render_notebook_plot,
    varied_parameter_names,
)

matplotlib.use("Agg", force=True)


def _results_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "run_id": ["run-a", "run-b", "run-a", "run-b"],
            "step": [1, 1, 2, 2],
            "level": [21, 21, 21, 21],
            "rand_prob": [0.3, 0.3, 0.3, 0.3],
            "weight_memory": [1.0, 0.8, 1.0, 0.8],
            "mercury_split_sensory_raw_latent_valid": [True, False, True, False],
            "latent_mean_total_variation": [0.7, 0.0, 0.8, 0.0],
            "sensory_precision": [0.0, 0.6, 0.0, 0.65],
        }
    )


def test_notebook_plot_to_plot_request_merges_shared_and_series_filters() -> None:
    base_plot = NotebookPlot(
        shared_filters={"level": 21, "rand_prob": 0.3},
        series=[
            NotebookSeries(
                name="latent",
                metric_column_name="latent_mean_total_variation",
                label="Latent transition precision",
                filter_values={"weight_memory": 1.0, "mercury_split_sensory_raw_latent_valid": True},
            ),
            NotebookSeries(
                name="baseline",
                metric_column_name="sensory_precision",
                label="Sensory precision",
                filter_values={"weight_memory": 0.8, "mercury_split_sensory_raw_latent_valid": False},
                line_style="--",
                show_standard_deviation_band=False,
                is_baseline=True,
            ),
        ],
        y_axis_label="Precision",
        y_limits=(0.0, 1.0),
        use_grid=True,
    )
    plot = NotebookPlot(
        shared_filters=base_plot.shared_filters,
        series=base_plot.series,
        title_spec=prefilled_title_spec_for_varied_parameters(
            base_plot,
            [
                TitleParameterSpec(parameter_name="level", label="Level"),
                TitleParameterSpec(
                    parameter_name="rand_prob",
                    label="Persistence",
                    transform=lambda value: 1 - float(value),
                    formatter=lambda value: f"{value:.1f}",
                ),
                TitleParameterSpec(
                    parameter_name="weight_memory",
                    label="$w_{mem}$",
                    template="{label}={value}",
                    source_series_name="latent",
                    formatter=lambda value: f"{float(value):.1f}",
                ),
                TitleParameterSpec(
                    parameter_name="mercury_split_sensory_raw_latent_valid",
                    label="Valid-only",
                    template="{label} {value}",
                    formatter=lambda value: "yes" if value else "no",
                    source_series_name="latent",
                ),
            ],
        ),
        y_axis_label=base_plot.y_axis_label,
        y_limits=base_plot.y_limits,
        use_grid=base_plot.use_grid,
    )

    request = notebook_plot_to_plot_request(plot)

    assert request.plot_spec.title == "Latent transition precision | $w_{mem}$=1.0 | Valid-only yes"
    assert request.axis_spec.y_axis_label == "Precision"
    assert request.axis_spec.y_limits == (0.0, 1.0)
    assert request.plot_spec.use_grid is True
    assert request.series_specs[0].filter_spec.parameter_values == {
        "level": 21,
        "rand_prob": 0.3,
        "weight_memory": 1.0,
        "mercury_split_sensory_raw_latent_valid": True,
    }
    assert request.series_specs[1].filter_spec.parameter_values == {
        "level": 21,
        "rand_prob": 0.3,
        "weight_memory": 0.8,
        "mercury_split_sensory_raw_latent_valid": False,
    }
    assert request.series_specs[1].style.line_style == "--"
    assert request.series_specs[1].style.show_standard_deviation_band is False
    assert request.series_specs[1].is_baseline is True


def test_prefilled_title_spec_from_canonical_parameters_uses_registry() -> None:
    plot = NotebookPlot(
        shared_filters={"level": 21, "rand_prob": 0.3},
        series=[
            NotebookSeries(
                name="latent",
                metric_column_name="latent_mean_total_variation",
                label="Latent transition precision",
                filter_values={"weight_memory": 1.0, "mercury_split_sensory_raw_latent_valid": True},
            ),
            NotebookSeries(
                name="baseline",
                metric_column_name="sensory_precision",
                label="Sensory precision",
                filter_values={"weight_memory": 0.8, "mercury_split_sensory_raw_latent_valid": False},
            ),
        ],
    )

    title_spec = prefilled_title_spec_from_canonical_parameters(
        plot,
        canonical_parameter_specs={
            "level": CanonicalTitleParameterSpec(parameter_name="level", label="Level"),
            "rand_prob": CanonicalTitleParameterSpec(
                parameter_name="rand_prob",
                label="Persistence",
                transform=lambda value: 1 - float(value),
                formatter=lambda value: f"{value:.1f}",
            ),
            "weight_memory": CanonicalTitleParameterSpec(
                parameter_name="weight_memory",
                label="$w_{mem}$",
                template="{label}={value}",
                formatter=lambda value: f"{float(value):.1f}",
            ),
        },
        parameter_order=["level", "rand_prob", "weight_memory"],
        default_source_series_name="latent",
    )

    assert [spec.parameter_name for spec in title_spec.parameter_specs] == ["weight_memory"]
    assert title_spec.parameter_specs[0].source_series_name == "latent"


def test_varied_parameter_names_only_returns_parameters_that_change_between_series() -> None:
    plot = NotebookPlot(
        shared_filters={"level": 21, "rand_prob": 0.3},
        series=[
            NotebookSeries(
                name="latent",
                metric_column_name="latent_mean_total_variation",
                label="Latent transition precision",
                filter_values={"weight_memory": 1.0, "mercury_split_sensory_raw_latent_valid": True},
            ),
            NotebookSeries(
                name="baseline",
                metric_column_name="sensory_precision",
                label="Sensory precision",
                filter_values={"weight_memory": 0.8, "mercury_split_sensory_raw_latent_valid": False},
            ),
        ],
    )

    assert varied_parameter_names(plot) == ["weight_memory", "mercury_split_sensory_raw_latent_valid"]


def test_prefilled_title_spec_rebinds_source_series_names_for_new_plots() -> None:
    plot = NotebookPlot(
        shared_filters={"level": 21, "rand_prob": 0.3, "mercury_split_sensory_raw_latent_valid": True},
        series=[
            NotebookSeries(
                name="memory_08",
                metric_column_name="latent_precision",
                label="Latent precision",
                filter_values={"weight_memory": 0.8},
            ),
            NotebookSeries(
                name="memory_10",
                metric_column_name="latent_precision",
                label="Latent precision",
                filter_values={"weight_memory": 1.0},
            ),
        ],
    )

    title_spec = prefilled_title_spec_for_varied_parameters(
        plot,
        [
            TitleParameterSpec(
                parameter_name="weight_memory",
                label="$w_{mem}$",
                template="{label}={value}",
                source_series_name="latent_precision_main",
                formatter=lambda value: f"{float(value):.1f}",
            ),
            TitleParameterSpec(
                parameter_name="level",
                label="Level",
            ),
        ],
    )

    assert len(title_spec.parameter_specs) == 1
    assert title_spec.parameter_specs[0].parameter_name == "weight_memory"
    assert title_spec.parameter_specs[0].source_series_name == "memory_08"


def test_explicit_title_overrides_generated_title() -> None:
    plot = NotebookPlot(
        shared_filters={"level": 21},
        series=[
            NotebookSeries(
                name="latent",
                metric_column_name="latent_mean_total_variation",
                label="Latent transition precision",
            ),
        ],
        title="Manual title",
        title_spec=NotebookTitleSpec(
            metric_label_mode="first_series",
            parameter_specs=[TitleParameterSpec(parameter_name="level", label="Level")],
        ),
    )

    request = notebook_plot_to_plot_request(plot)

    assert request.plot_spec.title == "Manual title"


def test_render_notebook_plot_writes_png_and_returns_summary(tmp_path: Path) -> None:
    plot = NotebookPlot(
        shared_filters={"level": 21, "rand_prob": 0.3},
        series=[
            NotebookSeries(
                name="latent",
                metric_column_name="latent_mean_total_variation",
                label="Latent transition precision",
                filter_values={"weight_memory": 1.0, "mercury_split_sensory_raw_latent_valid": True},
            ),
            NotebookSeries(
                name="baseline",
                metric_column_name="sensory_precision",
                label="Sensory precision",
                filter_values={"weight_memory": 0.8, "mercury_split_sensory_raw_latent_valid": False},
                line_style="--",
                show_standard_deviation_band=False,
                is_baseline=True,
            ),
        ],
        output_path=tmp_path / "plots" / "comparison.png",
        y_axis_label="Precision",
    )

    bundle = render_notebook_plot(_results_frame(), plot, show=False)

    assert plot.output_path is not None
    output_path = Path(plot.output_path)
    assert output_path.exists()
    assert "latent" in bundle.summary_frames_by_series_name
    assert "baseline" in bundle.summary_frames_by_series_name
    assert bundle.summary_frames_by_series_name["latent"]["step"].to_list() == [1, 2]
    assert bundle.summary_frames_by_series_name["baseline"]["step"].to_list() == [1, 2]


def test_render_alignment_contingency_heatmap_writes_png_and_returns_alignment(tmp_path: Path) -> None:
    result = render_alignment_contingency_heatmap(
        decoded_states=[0, 0, 1, 2],
        true_states=[1, 1, 0, 1],
        n_learned=3,
        n_true=2,
        output_path=tmp_path / "alignment_heatmap.png",
        show=False,
    )

    assert isinstance(result, AlignmentHeatmapResult)
    assert result.output_path.exists()
    assert result.contingency.tolist() == [[0, 2], [1, 0], [0, 1]]
    assert result.alignment.tolist() == [1, 0, 1]


def test_load_cooccurrence_matrix_from_paper_precision_json_reads_exact_matrix(tmp_path: Path) -> None:
    paper_precision_json = tmp_path / "run_paper_precision.json"
    paper_precision_json.write_text(
        json.dumps(
            {
                "metrics": {
                    "latent_exact_state_cooccurrence_matrix": [[5, 1], [0, 4], [2, 0]],
                    "sensory_exact_state_cooccurrence_matrix": [[3, 3]],
                }
            }
        ),
        encoding="utf-8",
    )

    matrix = load_cooccurrence_matrix_from_paper_precision_json(paper_precision_json)

    assert matrix.tolist() == [[5, 1], [0, 4], [2, 0]]


def test_render_alignment_contingency_heatmap_from_paper_precision_json_writes_png(tmp_path: Path) -> None:
    paper_precision_json = tmp_path / "run_paper_precision.json"
    paper_precision_json.write_text(
        json.dumps(
            {
                "metrics": {
                    "latent_exact_state_cooccurrence_matrix": [[5, 1], [0, 4], [2, 0]],
                }
            }
        ),
        encoding="utf-8",
    )

    result = render_alignment_contingency_heatmap_from_paper_precision_json(
        paper_precision_json_path=paper_precision_json,
        output_path=tmp_path / "exact_alignment_heatmap.png",
        show=False,
    )

    assert result.output_path.exists()
    assert result.contingency.tolist() == [[5, 1], [0, 4], [2, 0]]
    assert result.alignment.tolist() == [0, 1, 0]


def test_load_cooccurrence_history_tensor_from_paper_precision_json_pads_varying_shapes(tmp_path: Path) -> None:
    paper_precision_json = tmp_path / "run_paper_precision.json"
    paper_precision_json.write_text(
        json.dumps(
            {
                "history": [
                    {"step": 10, "latent_exact_state_cooccurrence_matrix": [[1, 2], [3, 4]]},
                    {"step": 20, "latent_exact_state_cooccurrence_matrix": [[5, 6], [7, 8], [9, 10]]},
                ]
            }
        ),
        encoding="utf-8",
    )

    steps, tensor = load_cooccurrence_history_tensor_from_paper_precision_json(paper_precision_json)

    assert steps.tolist() == [10, 20]
    assert tensor.shape == (2, 3, 2)
    assert tensor[0].tolist() == [[1, 2], [3, 4], [0, 0]]
    assert tensor[1].tolist() == [[5, 6], [7, 8], [9, 10]]


def test_render_cooccurrence_history_surface_from_paper_precision_json_writes_png(tmp_path: Path) -> None:
    paper_precision_json = tmp_path / "run_paper_precision.json"
    paper_precision_json.write_text(
        json.dumps(
            {
                "history": [
                    {"step": 10, "latent_exact_state_cooccurrence_matrix": [[1, 2], [3, 4]]},
                    {"step": 20, "latent_exact_state_cooccurrence_matrix": [[2, 3], [4, 5], [6, 7]]},
                ]
            }
        ),
        encoding="utf-8",
    )

    result = render_cooccurrence_history_surface_from_paper_precision_json(
        paper_precision_json_path=paper_precision_json,
        learned_state_index=1,
        output_path=tmp_path / "cooccurrence_surface.png",
        show=False,
    )

    assert isinstance(result, CooccurrenceHistorySurfaceResult)
    assert result.output_path.exists()
    assert result.steps.tolist() == [10, 20]
    assert result.tensor.shape == (2, 3, 2)
    assert result.learned_state_index == 1


def test_render_alignment_history_from_paper_precision_json_writes_png(tmp_path: Path) -> None:
    paper_precision_json = tmp_path / "run_paper_precision.json"
    paper_precision_json.write_text(
        json.dumps(
            {
                "history": [
                    {"step": 10, "latent_exact_state_cooccurrence_matrix": [[1, 2], [3, 1]]},
                    {"step": 20, "latent_exact_state_cooccurrence_matrix": [[4, 0], [1, 5], [2, 1]]},
                ]
            }
        ),
        encoding="utf-8",
    )

    result = render_alignment_history_from_paper_precision_json(
        paper_precision_json_path=paper_precision_json,
        output_path=tmp_path / "alignment_history.png",
        show=False,
    )

    assert isinstance(result, AlignmentHistoryResult)
    assert result.output_path.exists()
    assert result.steps.tolist() == [10, 20]
    assert result.assignments.shape == (2, 3)
    assert result.assignments.tolist() == [[1, 0, -1], [0, 1, 0]]
    assert result.confidence.shape == (2, 3)
    assert result.active_mask.shape == (2, 3)
    assert result.active_mask.tolist() == [[True, True, False], [True, True, True]]


def test_dominant_observation_ids_by_learned_state_returns_mode_ids(tmp_path: Path) -> None:
    states_path = tmp_path / "states.parquet"
    pl.DataFrame(
        {
            "latent_bmu": [0, 0, 0, 1, 1, 2],
            "sensory_bmu": [3, 3, 4, 5, 5, 1],
        }
    ).write_parquet(states_path)

    observation_ids = dominant_observation_ids_by_learned_state(states_path)

    assert observation_ids == {0: 3, 1: 5, 2: 1}


def test_observation_history_ids_by_checkpoint_is_checkpoint_local(tmp_path: Path) -> None:
    states_path = tmp_path / "states.parquet"
    pl.DataFrame(
        {
            "latent_bmu": [0, 0, 1, 1, 0, 0],
            "sensory_bmu": [3, 3, 4, 4, 5, 5],
        }
    ).write_parquet(states_path)

    history = observation_history_ids_by_checkpoint(
        states_path,
        np.asarray([2, 6]),
        max_learned_state_count=2,
    )

    assert history.tolist() == [[3, -1], [3, 4]]


def test_render_learned_state_identity_graph_writes_png(tmp_path: Path) -> None:
    from mercury.graph.core import Graph

    states_path = tmp_path / "states.parquet"
    pl.DataFrame(
        {
            "latent_bmu": [0, 0, 1, 1, 2, 2, 2],
            "sensory_bmu": [3, 3, 4, 4, 1, 1, 2],
        }
    ).write_parquet(states_path)

    paper_precision_json = tmp_path / "run_paper_precision.json"
    paper_precision_json.write_text(
        json.dumps(
            {
                "metrics": {
                    "latent_exact_state_cooccurrence_matrix": [[5, 1], [0, 4], [2, 0]],
                }
            }
        ),
        encoding="utf-8",
    )

    graph = Graph(directed=True)
    graph.register_edge_feature(name="action", dim=1, init_value=0.0)
    for _ in range(3):
        graph.add_node()
    graph.add_edge(0, 1, weight=2.0, edge_feat={"action": np.asarray([0.0], dtype=np.float32)})
    graph.add_edge(1, 2, weight=1.0, edge_feat={"action": np.asarray([0.0], dtype=np.float32)})
    graph_path = tmp_path / "latent_graph.npz"
    graph.to_npz(str(graph_path))

    result = render_learned_state_identity_graph(
        latent_graph_npz_path=graph_path,
        states_parquet_path=states_path,
        paper_precision_json_path=paper_precision_json,
        output_path=tmp_path / "identity_graph.png",
        show=False,
    )

    assert isinstance(result, LearnedStateIdentityGraphResult)
    assert result.output_path.exists()
    assert result.dominant_observation_ids == {0: 3, 1: 4, 2: 1}
    assert result.compressed_ground_truth_assignments == {0: 0, 1: 1, 2: 0}


def test_render_observation_history_from_paper_precision_json_writes_png(tmp_path: Path) -> None:
    states_path = tmp_path / "states.parquet"
    pl.DataFrame(
        {
            "latent_bmu": [0, 0, 1, 1, 2, 2, 2],
            "sensory_bmu": [3, 3, 4, 4, 1, 1, 2],
        }
    ).write_parquet(states_path)

    paper_precision_json = tmp_path / "run_paper_precision.json"
    paper_precision_json.write_text(
        json.dumps(
            {
                "history": [
                    {"step": 10, "latent_exact_state_cooccurrence_matrix": [[1, 2], [3, 1]]},
                    {"step": 20, "latent_exact_state_cooccurrence_matrix": [[4, 0], [1, 5], [2, 1]]},
                ]
            }
        ),
        encoding="utf-8",
    )

    result = render_observation_history_from_paper_precision_json(
        paper_precision_json_path=paper_precision_json,
        states_parquet_path=states_path,
        output_path=tmp_path / "observation_history.png",
        show=False,
    )

    assert isinstance(result, ObservationHistoryResult)
    assert result.output_path.exists()
    assert result.steps.tolist() == [10, 20]
    assert result.observation_ids.tolist() == [[3, 4, 1], [3, 4, 1]]
    assert result.active_mask.tolist() == [[True, True, False], [True, True, True]]
