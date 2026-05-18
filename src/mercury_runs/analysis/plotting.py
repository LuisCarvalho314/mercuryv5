from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path
import re
from typing import Any, Callable, Literal

import networkx as nx
import numpy as np
import polars as pl
from matplotlib.colors import BoundaryNorm, ListedColormap

from mercury_runs.domain.plotting_schema import (
    AggregationSpec,
    AxisSpec,
    FilterSpec,
    MetricSpec,
    PlotRequest,
    PlotSpec,
    SeriesSpec,
    SeriesStyle,
    SmoothingSpec,
    SummaryFrameBundle,
    plt,
)


@dataclass(frozen=True)
class NotebookSeries:
    name: str
    metric_column_name: str
    label: str
    filter_values: dict[str, Any] = field(default_factory=dict)
    line_style: str = "-"
    line_width: float = 2.0
    band_alpha: float = 0.2
    show_standard_deviation_band: bool = True
    show_individual_runs: bool = False
    individual_run_alpha: float = 0.1
    is_baseline: bool = False


@dataclass(frozen=True)
class CanonicalTitleParameterSpec:
    parameter_name: str
    label: str
    template: str = "{label} {value}"
    transform: Callable[[Any], Any] | None = None
    formatter: Callable[[Any], str] | None = None


@dataclass(frozen=True)
class TitleParameterSpec:
    parameter_name: str
    label: str
    template: str = "{label} {value}"
    source_series_name: str | None = None
    transform: Callable[[Any], Any] | None = None
    formatter: Callable[[Any], str] | None = None


@dataclass(frozen=True)
class NotebookTitleSpec:
    parameter_specs: list[TitleParameterSpec] = field(default_factory=list)
    metric_label_mode: Literal["none", "first_series", "series_name", "custom"] = "none"
    metric_label_series_name: str | None = None
    metric_label_text: str | None = None
    separator: str = " | "


@dataclass(frozen=True)
class NotebookPlot:
    series: list[NotebookSeries]
    shared_filters: dict[str, Any] = field(default_factory=dict)
    title: str | None = None
    title_spec: NotebookTitleSpec | None = None
    figure_size: tuple[float, float] = (9, 5)
    x_column_name: str = "step"
    run_column_name: str = "run_id"
    x_axis_label: str = "Step"
    y_axis_label: str = "Metric value"
    y_limits: tuple[float, float] | None = None
    x_limits: tuple[float, float] | None = None
    show_legend: bool = True
    legend_location: str = "best"
    use_grid: bool = False
    aggregation_mode: str = "mean_std"
    mean_alias_suffix: str = "mean"
    std_alias_suffix: str = "std"
    fill_null_standard_deviation_with: float = 0.0
    smoothing_mode: str = "none"
    smoothing_window_size: int = 1
    smoothing_min_periods: int = 1
    output_path: str | Path | None = None


@dataclass(frozen=True)
class AlignmentHeatmapResult:
    contingency: np.ndarray
    alignment: np.ndarray
    output_path: Path


@dataclass(frozen=True)
class CooccurrenceHistorySurfaceResult:
    steps: np.ndarray
    tensor: np.ndarray
    learned_state_index: int
    output_path: Path


@dataclass(frozen=True)
class AlignmentHistoryResult:
    steps: np.ndarray
    assignments: np.ndarray
    confidence: np.ndarray
    active_mask: np.ndarray
    output_path: Path


@dataclass(frozen=True)
class LearnedStateIdentityGraphResult:
    dominant_observation_ids: dict[int, int]
    compressed_ground_truth_assignments: dict[int, int]
    occupancy: dict[int, int]
    output_path: Path


@dataclass(frozen=True)
class ObservationHistoryResult:
    steps: np.ndarray
    observation_ids: np.ndarray
    active_mask: np.ndarray
    output_path: Path


CANONICAL_TITLE_PARAMETERS: dict[str, CanonicalTitleParameterSpec] = {
    "mean_total_variation": CanonicalTitleParameterSpec(
        parameter_name="mean_total_variation",
        label="TV_{mean}",
    ),
    "level": CanonicalTitleParameterSpec(
        parameter_name="level",
        label="Level",
    ),
    "rand_prob": CanonicalTitleParameterSpec(
        parameter_name="rand_prob",
        label="Persistence",
        transform=lambda value: 1 - float(value),
        formatter=lambda value: f"{value:.1f}",
    ),
    "weight_memory": CanonicalTitleParameterSpec(
        parameter_name="weight_memory",
        label="$w_{mem}$",
        template="{label} {value}",
        formatter=lambda value: f"{float(value):.1f}",
    ),
    "mercury_split_sensory_raw_latent_valid": CanonicalTitleParameterSpec(
        parameter_name="mercury_split_sensory_raw_latent_valid",
        label="Collision Conditioned",
        formatter=lambda value: "True" if value else "False",
    ),
}

CANONICAL_TITLE_PARAMETER_ORDER: list[str] = [
    "level",
    "rand_prob",
    "weight_memory",
    "mercury_split_sensory_raw_latent_valid",
]

DEFAULT_NOTEBOOK_PLOT_OUTPUT_DIR = Path("results/current/notebook_plots")


def apply_light_mode_plot_style() -> None:
    plt.style.use("default")
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "figure.edgecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.edgecolor": "white",
            "savefig.transparent": False,
            "text.color": "black",
            "axes.titlecolor": "black",
            "axes.labelcolor": "black",
            "xtick.color": "black",
            "ytick.color": "black",
            "axes.edgecolor": "black",
            "grid.color": "#d0d0d0",
            "legend.facecolor": "white",
            "legend.edgecolor": "black",
            "legend.framealpha": 1.0,
        }
    )


def _apply_light_mode_to_axes(figure, axes) -> None:
    figure.patch.set_facecolor("white")
    axes.set_facecolor("white")
    axes.tick_params(colors="black")

    for spine in axes.spines.values():
        spine.set_color("black")

    legend = axes.get_legend()
    if legend is not None:
        legend.get_frame().set_facecolor("white")
        legend.get_frame().set_edgecolor("black")
        legend.get_frame().set_alpha(1.0)
        for text in legend.get_texts():
            text.set_color("black")


def _format_title_value(value: Any) -> str:
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def _slugify_plot_name(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower()).strip("_")
    return slug or "plot"


def notebook_plot_output_path(
    plot: NotebookPlot,
    *,
    output_dir: str | Path = DEFAULT_NOTEBOOK_PLOT_OUTPUT_DIR,
) -> Path:
    if plot.output_path is not None:
        return Path(plot.output_path)
    title = generate_notebook_plot_title(plot)
    if title:
        file_stem = _slugify_plot_name(title)
    elif plot.series:
        file_stem = _slugify_plot_name(plot.series[0].name)
    else:
        file_stem = "plot"
    return Path(output_dir) / f"{file_stem}.png"


def alignment_heatmap_output_path(
    name: str = "alignment_contingency_heatmap",
    *,
    output_dir: str | Path = DEFAULT_NOTEBOOK_PLOT_OUTPUT_DIR,
) -> Path:
    return Path(output_dir) / f"{_slugify_plot_name(name)}.png"


def _find_notebook_series(
    plot: NotebookPlot,
    series_name: str,
) -> NotebookSeries:
    for series in plot.series:
        if series.name == series_name:
            return series
    raise ValueError(f"Notebook series not found for title generation: {series_name}")


def _resolve_title_parameter_value(
    plot: NotebookPlot,
    parameter_spec: TitleParameterSpec,
) -> Any:
    if parameter_spec.source_series_name is None:
        filter_values = plot.shared_filters
    else:
        series = _find_notebook_series(plot, parameter_spec.source_series_name)
        filter_values = {**plot.shared_filters, **series.filter_values}

    if parameter_spec.parameter_name not in filter_values:
        raise ValueError(
            f"Title parameter {parameter_spec.parameter_name!r} was not found in the selected filter source",
        )

    value = filter_values[parameter_spec.parameter_name]
    if parameter_spec.transform is not None:
        value = parameter_spec.transform(value)
    return value


def _merged_plot_filter_values(series: NotebookSeries, shared_filters: dict[str, Any]) -> dict[str, Any]:
    return {**shared_filters, **series.filter_values}


def _resolve_title_parameter_source_series_name(
    plot: NotebookPlot,
    parameter_spec: TitleParameterSpec,
) -> str | None:
    if parameter_spec.source_series_name is not None:
        for series in plot.series:
            if series.name == parameter_spec.source_series_name:
                return parameter_spec.source_series_name

    for series in plot.series:
        merged_filters = _merged_plot_filter_values(series, plot.shared_filters)
        if parameter_spec.parameter_name in merged_filters:
            return series.name

    return None


def build_title_parameter_spec(
    canonical_spec: CanonicalTitleParameterSpec,
    *,
    source_series_name: str | None = None,
) -> TitleParameterSpec:
    return TitleParameterSpec(
        parameter_name=canonical_spec.parameter_name,
        label=canonical_spec.label,
        template=canonical_spec.template,
        source_series_name=source_series_name,
        transform=canonical_spec.transform,
        formatter=canonical_spec.formatter,
    )


def varied_parameter_names(plot: NotebookPlot) -> list[str]:
    if not plot.series:
        return []

    ordered_parameter_names: list[str] = []
    parameter_values_by_name: dict[str, list[Any]] = {}

    for series in plot.series:
        merged_filters = _merged_plot_filter_values(series, plot.shared_filters)
        for parameter_name, value in merged_filters.items():
            if parameter_name not in parameter_values_by_name:
                parameter_values_by_name[parameter_name] = []
                ordered_parameter_names.append(parameter_name)
            parameter_values_by_name[parameter_name].append(value)

    varied: list[str] = []
    for parameter_name in ordered_parameter_names:
        values = parameter_values_by_name[parameter_name]
        if len(values) != len(plot.series):
            varied.append(parameter_name)
            continue
        if any(value != values[0] for value in values[1:]):
            varied.append(parameter_name)
    return varied


def prefilled_title_spec_for_varied_parameters(
    plot: NotebookPlot,
    parameter_specs: list[TitleParameterSpec],
    *,
    metric_label_mode: Literal["none", "first_series", "series_name", "custom"] = "first_series",
    metric_label_series_name: str | None = None,
    metric_label_text: str | None = None,
    separator: str = " | ",
) -> NotebookTitleSpec:
    varied_names = set(varied_parameter_names(plot))
    included_parameter_specs: list[TitleParameterSpec] = []
    for parameter_spec in parameter_specs:
        if parameter_spec.parameter_name not in varied_names:
            continue
        source_series_name = _resolve_title_parameter_source_series_name(plot, parameter_spec)
        included_parameter_specs.append(
            replace(parameter_spec, source_series_name=source_series_name),
        )
    return NotebookTitleSpec(
        parameter_specs=included_parameter_specs,
        metric_label_mode=metric_label_mode,
        metric_label_series_name=metric_label_series_name,
        metric_label_text=metric_label_text,
        separator=separator,
    )


def prefilled_title_spec_from_canonical_parameters(
    plot: NotebookPlot,
    canonical_parameter_specs: dict[str, CanonicalTitleParameterSpec],
    *,
    parameter_order: list[str] | None = None,
    default_source_series_name: str | None = None,
    metric_label_mode: Literal["none", "first_series", "series_name", "custom"] = "first_series",
    metric_label_series_name: str | None = None,
    metric_label_text: str | None = None,
    separator: str = " | ",
) -> NotebookTitleSpec:
    ordered_parameter_names = parameter_order if parameter_order is not None else list(canonical_parameter_specs.keys())
    parameter_specs = [
        build_title_parameter_spec(
            canonical_parameter_specs[parameter_name],
            source_series_name=default_source_series_name,
        )
        for parameter_name in ordered_parameter_names
        if parameter_name in canonical_parameter_specs
    ]
    return prefilled_title_spec_for_varied_parameters(
        plot,
        parameter_specs,
        metric_label_mode=metric_label_mode,
        metric_label_series_name=metric_label_series_name,
        metric_label_text=metric_label_text,
        separator=separator,
    )


def _render_title_parameter(
    plot: NotebookPlot,
    parameter_spec: TitleParameterSpec,
) -> str:
    value = _resolve_title_parameter_value(plot, parameter_spec)
    formatted_value = (
        parameter_spec.formatter(value)
        if parameter_spec.formatter is not None
        else _format_title_value(value)
    )
    return parameter_spec.template.format(label=parameter_spec.label, value=formatted_value)


def generate_notebook_plot_title(plot: NotebookPlot) -> str | None:
    if plot.title is not None:
        return plot.title

    if plot.title_spec is None:
        return None

    parts: list[str] = []
    title_spec = plot.title_spec

    if title_spec.metric_label_mode == "first_series":
        if not plot.series:
            raise ValueError("Cannot build a title from the first series when no series are defined")
        parts.append(plot.series[0].label)
    elif title_spec.metric_label_mode == "series_name":
        if title_spec.metric_label_series_name is None:
            raise ValueError("metric_label_series_name is required when metric_label_mode='series_name'")
        parts.append(_find_notebook_series(plot, title_spec.metric_label_series_name).label)
    elif title_spec.metric_label_mode == "custom":
        if title_spec.metric_label_text is None:
            raise ValueError("metric_label_text is required when metric_label_mode='custom'")
        parts.append(title_spec.metric_label_text)

    parts.extend(_render_title_parameter(plot, parameter_spec) for parameter_spec in title_spec.parameter_specs)
    return title_spec.separator.join(parts) if parts else None


def apply_filter(
    results_frame: pl.DataFrame,
    filter_spec: FilterSpec,
) -> pl.DataFrame:
    filter_expressions = filter_spec.to_polars_expressions()
    if not filter_expressions:
        return results_frame
    return results_frame.filter(*filter_expressions)


def select_series_columns(
    filtered_results_frame: pl.DataFrame,
    x_column_name: str,
    run_column_name: str,
    metric_column_name: str,
) -> pl.DataFrame:
    return filtered_results_frame.select(
        run_column_name,
        x_column_name,
        metric_column_name,
    )


def build_series_summary(
    series_frame: pl.DataFrame,
    metric_column_name: str,
    x_column_name: str,
    aggregation_spec: AggregationSpec,
) -> pl.DataFrame:
    if aggregation_spec.mode != "mean_std":
        raise ValueError(f"Unsupported aggregation mode: {aggregation_spec.mode}")

    return (
        series_frame
        .group_by(x_column_name)
        .agg(
            pl.col(metric_column_name)
            .mean()
            .alias(aggregation_spec.mean_column_name(metric_column_name)),
            pl.col(metric_column_name)
            .std()
            .alias(aggregation_spec.std_column_name(metric_column_name)),
        )
        .sort(x_column_name)
    )


def apply_smoothing_to_summary(
    summary_frame: pl.DataFrame,
    metric_column_name: str,
    x_column_name: str,
    aggregation_spec: AggregationSpec,
    smoothing_spec: SmoothingSpec,
) -> pl.DataFrame:
    if not smoothing_spec.enabled():
        return summary_frame

    mean_column_name = aggregation_spec.mean_column_name(metric_column_name)
    std_column_name = aggregation_spec.std_column_name(metric_column_name)

    if smoothing_spec.mode == "rolling_mean":
        return (
            summary_frame
            .sort(x_column_name)
            .with_columns(
                pl.col(mean_column_name)
                .rolling_mean(
                    window_size=smoothing_spec.window_size,
                    min_periods=smoothing_spec.min_periods,
                )
                .alias(mean_column_name),
                pl.col(std_column_name)
                .rolling_mean(
                    window_size=smoothing_spec.window_size,
                    min_periods=smoothing_spec.min_periods,
                )
                .alias(std_column_name),
            )
        )

    raise ValueError(f"Unsupported smoothing mode: {smoothing_spec.mode}")


def prepare_series_data(
    results_frame: pl.DataFrame,
    plot_request: PlotRequest,
) -> SummaryFrameBundle:
    raw_frames_by_series_name: dict[str, pl.DataFrame] = {}
    summary_frames_by_series_name: dict[str, pl.DataFrame] = {}

    x_column_name = plot_request.axis_spec.x_column_name
    run_column_name = plot_request.axis_spec.run_column_name
    aggregation_spec = plot_request.aggregation_spec
    smoothing_spec = plot_request.smoothing_spec

    for series_spec in plot_request.series_specs:
        filtered_frame = apply_filter(results_frame, series_spec.filter_spec)
        series_frame = select_series_columns(
            filtered_results_frame=filtered_frame,
            x_column_name=x_column_name,
            run_column_name=run_column_name,
            metric_column_name=series_spec.metric_spec.column_name,
        )

        summary_frame = build_series_summary(
            series_frame=series_frame,
            metric_column_name=series_spec.metric_spec.column_name,
            x_column_name=x_column_name,
            aggregation_spec=aggregation_spec,
        )

        summary_frame = apply_smoothing_to_summary(
            summary_frame=summary_frame,
            metric_column_name=series_spec.metric_spec.column_name,
            x_column_name=x_column_name,
            aggregation_spec=aggregation_spec,
            smoothing_spec=smoothing_spec,
        )

        raw_frames_by_series_name[series_spec.name] = series_frame
        summary_frames_by_series_name[series_spec.name] = summary_frame

    return SummaryFrameBundle(
        raw_frames_by_series_name=raw_frames_by_series_name,
        summary_frames_by_series_name=summary_frames_by_series_name,
    )


def plot_individual_runs_for_series(
    raw_series_frame: pl.DataFrame,
    series_spec: SeriesSpec,
    axis_spec: AxisSpec,
    axes,
) -> None:
    run_ids = (
        raw_series_frame
        .select(axis_spec.run_column_name)
        .unique()
        .to_series()
        .to_list()
    )

    for run_id in run_ids:
        run_frame = (
            raw_series_frame
            .filter(pl.col(axis_spec.run_column_name) == run_id)
            .sort(axis_spec.x_column_name)
        )

        x_values = run_frame[axis_spec.x_column_name].to_numpy()
        y_values = run_frame[series_spec.metric_spec.column_name].to_numpy()

        axes.plot(
            x_values,
            y_values,
            linestyle=series_spec.style.line_style,
            linewidth=1.0,
            alpha=series_spec.style.individual_run_alpha,
        )


def plot_series_summary(
    summary_frame: pl.DataFrame,
    raw_series_frame: pl.DataFrame,
    series_spec: SeriesSpec,
    axis_spec: AxisSpec,
    aggregation_spec: AggregationSpec,
    axes,
) -> None:
    metric_column_name = series_spec.metric_spec.column_name
    mean_column_name = aggregation_spec.mean_column_name(metric_column_name)
    std_column_name = aggregation_spec.std_column_name(metric_column_name)

    x_values = summary_frame[axis_spec.x_column_name].to_numpy()
    mean_values = summary_frame[mean_column_name].to_numpy()
    std_values = (
        summary_frame[std_column_name]
        .fill_null(aggregation_spec.fill_null_standard_deviation_with)
        .to_numpy()
    )

    if series_spec.style.show_individual_runs:
        plot_individual_runs_for_series(
            raw_series_frame=raw_series_frame,
            series_spec=series_spec,
            axis_spec=axis_spec,
            axes=axes,
        )

    label = series_spec.metric_spec.label
    if series_spec.is_baseline:
        label = f"{label} (baseline)"

    axes.plot(
        x_values,
        mean_values,
        linestyle=series_spec.style.line_style,
        linewidth=series_spec.style.line_width,
        label=label,
        clip_on=False,
        zorder=4,
    )

    if series_spec.style.show_standard_deviation_band:
        axes.fill_between(
            x_values,
            mean_values - std_values,
            mean_values + std_values,
            alpha=series_spec.style.band_alpha,
            zorder=1,
        )


def render_plot(
    plot_request: PlotRequest,
    summary_bundle: SummaryFrameBundle,
    *,
    output_path: str | Path | None = None,
    show: bool = True,
) -> None:
    apply_light_mode_plot_style()
    figure, axes = plt.subplots(figsize=plot_request.plot_spec.figure_size, facecolor="white")

    for series_spec in plot_request.series_specs:
        plot_series_summary(
            summary_frame=summary_bundle.summary_frames_by_series_name[series_spec.name],
            raw_series_frame=summary_bundle.raw_frames_by_series_name[series_spec.name],
            series_spec=series_spec,
            axis_spec=plot_request.axis_spec,
            aggregation_spec=plot_request.aggregation_spec,
            axes=axes,
        )

    axes.set_xlabel(plot_request.axis_spec.x_axis_label)
    axes.set_ylabel(plot_request.axis_spec.y_axis_label)

    if plot_request.plot_spec.title is not None:
        axes.set_title(plot_request.plot_spec.title, color="black")

    if plot_request.axis_spec.x_limits is not None:
        axes.set_xlim(*plot_request.axis_spec.x_limits)

    if plot_request.axis_spec.y_limits is not None:
        axes.set_ylim(*plot_request.axis_spec.y_limits)

    if plot_request.plot_spec.use_grid:
        axes.grid(True, alpha=0.3)

    if plot_request.plot_spec.show_legend:
        axes.legend(loc=plot_request.plot_spec.legend_location)

    _apply_light_mode_to_axes(figure, axes)

    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, bbox_inches="tight", facecolor="white", edgecolor="white", transparent=False)

    if show:
        plt.show()
    else:
        plt.close(figure)


def summarise_and_plot(
    results_frame: pl.DataFrame,
    plot_request: PlotRequest,
    *,
    output_path: str | Path | None = None,
    show: bool = True,
) -> SummaryFrameBundle:
    summary_bundle = prepare_series_data(
        results_frame=results_frame,
        plot_request=plot_request,
    )

    render_plot(
        plot_request=plot_request,
        summary_bundle=summary_bundle,
        output_path=output_path,
        show=show,
    )

    return summary_bundle


def _merged_filter_values(
    shared_filters: dict[str, Any],
    series_filters: dict[str, Any],
) -> dict[str, Any]:
    merged = dict(shared_filters)
    merged.update(series_filters)
    return merged


def notebook_series_to_series_spec(
    series: NotebookSeries,
    *,
    shared_filters: dict[str, Any],
) -> SeriesSpec:
    return SeriesSpec(
        name=series.name,
        filter_spec=FilterSpec(
            parameter_values=_merged_filter_values(shared_filters, series.filter_values),
        ),
        metric_spec=MetricSpec(
            column_name=series.metric_column_name,
            label=series.label,
        ),
        style=SeriesStyle(
            line_style=series.line_style,
            line_width=series.line_width,
            band_alpha=series.band_alpha,
            show_standard_deviation_band=series.show_standard_deviation_band,
            show_individual_runs=series.show_individual_runs,
            individual_run_alpha=series.individual_run_alpha,
        ),
        is_baseline=series.is_baseline,
    )


def notebook_plot_to_plot_request(plot: NotebookPlot) -> PlotRequest:
    return PlotRequest(
        series_specs=[
            notebook_series_to_series_spec(series, shared_filters=plot.shared_filters)
            for series in plot.series
        ],
        axis_spec=AxisSpec(
            x_column_name=plot.x_column_name,
            run_column_name=plot.run_column_name,
            x_axis_label=plot.x_axis_label,
            y_axis_label=plot.y_axis_label,
            y_limits=plot.y_limits,
            x_limits=plot.x_limits,
        ),
        aggregation_spec=AggregationSpec(
            mode=plot.aggregation_mode,
            mean_alias_suffix=plot.mean_alias_suffix,
            std_alias_suffix=plot.std_alias_suffix,
            fill_null_standard_deviation_with=plot.fill_null_standard_deviation_with,
        ),
        smoothing_spec=SmoothingSpec(
            mode=plot.smoothing_mode,
            window_size=plot.smoothing_window_size,
            min_periods=plot.smoothing_min_periods,
        ),
        plot_spec=PlotSpec(
            title=generate_notebook_plot_title(plot),
            figure_size=plot.figure_size,
            show_legend=plot.show_legend,
            legend_location=plot.legend_location,
            use_grid=plot.use_grid,
        ),
    )


def render_notebook_plot(
    results_frame: pl.DataFrame,
    plot: NotebookPlot,
    *,
    output_path: str | Path | None = None,
    show: bool = True,
) -> SummaryFrameBundle:
    resolved_output_path = output_path if output_path is not None else plot.output_path
    return summarise_and_plot(
        results_frame=results_frame,
        plot_request=notebook_plot_to_plot_request(plot),
        output_path=resolved_output_path,
        show=show,
    )


def render_alignment_contingency_heatmap(
    *,
    decoded_states: np.ndarray,
    true_states: np.ndarray,
    n_learned: int,
    n_true: int,
    title: str = "Alignment Contingency Heatmap",
    output_path: str | Path | None = None,
    show: bool = True,
) -> AlignmentHeatmapResult:
    from mercury_runs.infrastructure.paper_precision import compute_alignment, compute_contingency_matrix

    apply_light_mode_plot_style()

    contingency = compute_contingency_matrix(
        decoded_states=decoded_states,
        true_states=true_states,
        n_learned=n_learned,
        n_true=n_true,
    )
    alignment = compute_alignment(contingency)

    figure, axes = plt.subplots(figsize=(8, 6), facecolor="white")
    image = axes.imshow(contingency, cmap="Blues", aspect="auto")
    colorbar = figure.colorbar(image, ax=axes)
    colorbar.ax.yaxis.set_tick_params(color="black")
    plt.setp(colorbar.ax.get_yticklabels(), color="black")

    axes.set_title(title, color="black")
    axes.set_xlabel("True state", color="black")
    axes.set_ylabel("Learned state", color="black")
    axes.set_xticks(np.arange(int(n_true)))
    axes.set_yticks(np.arange(int(n_learned)))

    for learned_state in range(contingency.shape[0]):
        for true_state in range(contingency.shape[1]):
            count = int(contingency[learned_state, true_state])
            axes.text(
                true_state,
                learned_state,
                str(count),
                ha="center",
                va="center",
                color="black" if count < float(contingency.max(initial=0)) * 0.6 else "white",
            )

    axes.scatter(
        alignment,
        np.arange(contingency.shape[0]),
        s=120,
        facecolors="none",
        edgecolors="red",
        linewidths=2.0,
        label="Chosen alignment",
    )
    axes.legend(loc="upper right")
    _apply_light_mode_to_axes(figure, axes)

    resolved_output_path = Path(output_path) if output_path is not None else alignment_heatmap_output_path()
    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(resolved_output_path, bbox_inches="tight", facecolor="white", edgecolor="white", transparent=False)

    if show:
        plt.show()
    else:
        plt.close(figure)

    return AlignmentHeatmapResult(
        contingency=contingency,
        alignment=alignment,
        output_path=resolved_output_path,
    )


def _render_alignment_heatmap_from_matrix(
    *,
    contingency: np.ndarray,
    title: str,
    output_path: str | Path | None = None,
    show: bool = True,
) -> AlignmentHeatmapResult:
    from mercury_runs.infrastructure.paper_precision import compute_alignment

    apply_light_mode_plot_style()

    contingency_array = np.asarray(contingency, dtype=np.int64)
    if contingency_array.ndim != 2:
        raise ValueError(f"Expected contingency matrix with shape (n_learned, n_true), got {contingency_array.shape!r}")
    alignment = compute_alignment(contingency_array)

    n_learned, n_true = contingency_array.shape
    figure, axes = plt.subplots(figsize=(8, 6), facecolor="white")
    image = axes.imshow(contingency_array, cmap="Blues", aspect="auto")
    colorbar = figure.colorbar(image, ax=axes)
    colorbar.ax.yaxis.set_tick_params(color="black")
    plt.setp(colorbar.ax.get_yticklabels(), color="black")

    axes.set_title(title, color="black")
    axes.set_xlabel("Visited ground-truth state index (compressed)", color="black")
    axes.set_ylabel("Learned state", color="black")
    axes.set_xticks(np.arange(int(n_true)))
    axes.set_yticks(np.arange(int(n_learned)))

    max_count = float(contingency_array.max(initial=0))
    for learned_state in range(contingency_array.shape[0]):
        for true_state in range(contingency_array.shape[1]):
            count = int(contingency_array[learned_state, true_state])
            axes.text(
                true_state,
                learned_state,
                str(count),
                ha="center",
                va="center",
                color="black" if count < max_count * 0.6 else "white",
            )

    axes.scatter(
        alignment,
        np.arange(contingency_array.shape[0]),
        s=120,
        facecolors="none",
        edgecolors="red",
        linewidths=2.0,
        label="Chosen alignment",
    )
    axes.legend(loc="upper right")
    _apply_light_mode_to_axes(figure, axes)

    resolved_output_path = Path(output_path) if output_path is not None else alignment_heatmap_output_path(title)
    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(resolved_output_path, bbox_inches="tight", facecolor="white", edgecolor="white", transparent=False)

    if show:
        plt.show()
    else:
        plt.close(figure)

    return AlignmentHeatmapResult(
        contingency=contingency_array,
        alignment=alignment,
        output_path=resolved_output_path,
    )


def load_cooccurrence_matrix_from_paper_precision_json(
    paper_precision_json_path: str | Path,
    *,
    matrix_key: str = "latent_exact_state_cooccurrence_matrix",
) -> np.ndarray:
    path = Path(paper_precision_json_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    metrics = payload.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError(f"Expected 'metrics' object in {path}")
    matrix = metrics.get(matrix_key)
    if matrix is None:
        raise ValueError(f"Matrix key {matrix_key!r} not found in {path}")
    matrix_array = np.asarray(matrix, dtype=np.int64)
    if matrix_array.ndim != 2:
        raise ValueError(f"Expected matrix key {matrix_key!r} in {path} to have shape (n_learned, n_true)")
    return matrix_array


def load_cooccurrence_history_tensor_from_paper_precision_json(
    paper_precision_json_path: str | Path,
    *,
    matrix_key: str = "latent_exact_state_cooccurrence_matrix",
) -> tuple[np.ndarray, np.ndarray]:
    path = Path(paper_precision_json_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    history = payload.get("history")
    if not isinstance(history, list) or not history:
        raise ValueError(f"Expected non-empty 'history' array in {path}")

    steps: list[int] = []
    matrices: list[np.ndarray] = []
    max_learned = 0
    max_true = 0
    for entry in history:
        if not isinstance(entry, dict):
            raise ValueError(f"Expected history entries in {path} to be objects")
        if "step" not in entry:
            raise ValueError(f"Expected each history entry in {path} to contain 'step'")
        matrix = entry.get(matrix_key)
        if matrix is None:
            raise ValueError(f"Matrix key {matrix_key!r} not found in history entry for {path}")
        matrix_array = np.asarray(matrix, dtype=np.int64)
        if matrix_array.ndim != 2:
            raise ValueError(f"Expected matrix key {matrix_key!r} in {path} history to have shape (n_learned, n_true)")
        steps.append(int(entry["step"]))
        matrices.append(matrix_array)
        max_learned = max(max_learned, int(matrix_array.shape[0]))
        max_true = max(max_true, int(matrix_array.shape[1]))

    tensor = np.zeros((len(matrices), max_learned, max_true), dtype=np.int64)
    for index, matrix_array in enumerate(matrices):
        learned_count, true_count = matrix_array.shape
        tensor[index, :learned_count, :true_count] = matrix_array

    return np.asarray(steps, dtype=np.int64), tensor


def dominant_observation_ids_by_learned_state(
    states_parquet_path: str | Path,
    *,
    learned_state_column_name: str = "latent_bmu",
    observation_column_name: str = "sensory_bmu",
) -> dict[int, int]:
    frame = pl.read_parquet(states_parquet_path).select(learned_state_column_name, observation_column_name)
    if frame.is_empty():
        return {}
    grouped = (
        frame.group_by([learned_state_column_name, observation_column_name])
        .len()
        .sort([learned_state_column_name, "len", observation_column_name], descending=[False, True, False])
        .group_by(learned_state_column_name)
        .agg(pl.col(observation_column_name).first())
        .sort(learned_state_column_name)
    )
    learned = grouped.get_column(learned_state_column_name).to_list()
    observations = grouped.get_column(observation_column_name).to_list()
    return {
        int(learned_state): int(observation_id)
        for learned_state, observation_id in zip(learned, observations, strict=False)
    }


def observation_history_ids_by_checkpoint(
    states_parquet_path: str | Path,
    checkpoint_steps: np.ndarray,
    *,
    learned_state_column_name: str = "latent_bmu",
    observation_column_name: str = "sensory_bmu",
    max_learned_state_count: int | None = None,
) -> np.ndarray:
    frame = pl.read_parquet(states_parquet_path)
    if learned_state_column_name not in frame.columns or observation_column_name not in frame.columns:
        raise ValueError(
            f"Expected columns {learned_state_column_name!r} and {observation_column_name!r} in {states_parquet_path}",
        )
    if "step" in frame.columns:
        state_frame = frame.select("step", learned_state_column_name, observation_column_name)
    else:
        state_frame = (
            frame.select(learned_state_column_name, observation_column_name)
            .with_row_index(name="step", offset=1)
        )

    if state_frame.is_empty():
        width = 0 if max_learned_state_count is None else int(max_learned_state_count)
        return np.full((len(checkpoint_steps), width), -1, dtype=np.int64)

    learned_width = int(
        max_learned_state_count
        if max_learned_state_count is not None
        else (int(state_frame.get_column(learned_state_column_name).max()) + 1)
    )
    history = np.full((len(checkpoint_steps), learned_width), -1, dtype=np.int64)

    for checkpoint_index, checkpoint_step in enumerate(np.asarray(checkpoint_steps, dtype=np.int64).tolist()):
        prefix = state_frame.filter(pl.col("step") <= int(checkpoint_step))
        if prefix.is_empty():
            continue
        grouped = (
            prefix.group_by([learned_state_column_name, observation_column_name])
            .len()
            .sort([learned_state_column_name, "len", observation_column_name], descending=[False, True, False])
            .group_by(learned_state_column_name)
            .agg(pl.col(observation_column_name).first())
        )
        learned = grouped.get_column(learned_state_column_name).to_list()
        observations = grouped.get_column(observation_column_name).to_list()
        for learned_state, observation_id in zip(learned, observations, strict=False):
            learned_index = int(learned_state)
            if 0 <= learned_index < learned_width:
                history[checkpoint_index, learned_index] = int(observation_id)

    return history


def render_alignment_contingency_heatmap_from_paper_precision_json(
    *,
    paper_precision_json_path: str | Path,
    matrix_key: str = "latent_exact_state_cooccurrence_matrix",
    title: str | None = None,
    output_path: str | Path | None = None,
    show: bool = True,
) -> AlignmentHeatmapResult:
    contingency = load_cooccurrence_matrix_from_paper_precision_json(
        paper_precision_json_path,
        matrix_key=matrix_key,
    )
    resolved_title = title or f"{matrix_key.replace('_', ' ').title()} Alignment"
    return _render_alignment_heatmap_from_matrix(
        contingency=contingency,
        title=resolved_title,
        output_path=output_path,
        show=show,
    )


def render_cooccurrence_history_surface_from_paper_precision_json(
    *,
    paper_precision_json_path: str | Path,
    matrix_key: str = "latent_exact_state_cooccurrence_matrix",
    learned_state_index: int | None = None,
    title: str | None = None,
    output_path: str | Path | None = None,
    show: bool = True,
) -> CooccurrenceHistorySurfaceResult:
    apply_light_mode_plot_style()

    steps, tensor = load_cooccurrence_history_tensor_from_paper_precision_json(
        paper_precision_json_path,
        matrix_key=matrix_key,
    )
    if learned_state_index is None:
        learned_totals = tensor.sum(axis=(0, 2))
        learned_state_index = int(np.argmax(learned_totals))
    if learned_state_index < 0 or learned_state_index >= tensor.shape[1]:
        raise ValueError(f"learned_state_index {learned_state_index} is out of bounds for tensor with {tensor.shape[1]} learned states")

    true_states = np.arange(tensor.shape[2], dtype=np.int64)
    step_grid, true_grid = np.meshgrid(steps, true_states, indexing="ij")
    surface_values = tensor[:, learned_state_index, :]

    figure = plt.figure(figsize=(10, 6), facecolor="white")
    axes = figure.add_subplot(111, projection="3d")
    surface = axes.plot_surface(
        true_grid,
        step_grid,
        surface_values,
        cmap="Blues",
        edgecolor="none",
        antialiased=True,
    )
    colorbar = figure.colorbar(surface, ax=axes, shrink=0.7, pad=0.12)
    colorbar.ax.yaxis.set_tick_params(color="black")
    plt.setp(colorbar.ax.get_yticklabels(), color="black")

    resolved_title = title or f"{matrix_key.replace('_', ' ').title()} Surface | learned state {learned_state_index}"
    axes.set_title(resolved_title, color="black")
    axes.set_xlabel("Ground-truth state", color="black")
    axes.set_ylabel("Step", color="black")
    axes.set_zlabel("Cooccurrence count", color="black")
    axes.tick_params(colors="black")
    axes.xaxis.label.set_color("black")
    axes.yaxis.label.set_color("black")
    axes.zaxis.label.set_color("black")
    axes.xaxis.pane.set_facecolor((1.0, 1.0, 1.0, 1.0))
    axes.yaxis.pane.set_facecolor((1.0, 1.0, 1.0, 1.0))
    axes.zaxis.pane.set_facecolor((1.0, 1.0, 1.0, 1.0))
    axes.xaxis.pane.set_edgecolor("black")
    axes.yaxis.pane.set_edgecolor("black")
    axes.zaxis.pane.set_edgecolor("black")
    axes.view_init(elev=28, azim=-130)

    resolved_output_path = (
        Path(output_path)
        if output_path is not None
        else alignment_heatmap_output_path(f"{resolved_title} surface")
    )
    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(resolved_output_path, bbox_inches="tight", facecolor="white", edgecolor="white", transparent=False)

    if show:
        plt.show()
    else:
        plt.close(figure)

    return CooccurrenceHistorySurfaceResult(
        steps=steps,
        tensor=tensor,
        learned_state_index=learned_state_index,
        output_path=resolved_output_path,
    )


def render_alignment_history_from_paper_precision_json(
    *,
    paper_precision_json_path: str | Path,
    matrix_key: str = "latent_exact_state_cooccurrence_matrix",
    states_parquet_path: str | Path | None = None,
    learned_state_column_name: str = "latent_bmu",
    observation_column_name: str = "sensory_bmu",
    title: str | None = None,
    output_path: str | Path | None = None,
    show: bool = True,
) -> AlignmentHistoryResult:
    apply_light_mode_plot_style()

    steps, tensor = load_cooccurrence_history_tensor_from_paper_precision_json(
        paper_precision_json_path,
        matrix_key=matrix_key,
    )
    row_sums = tensor.sum(axis=2)
    assignments = np.argmax(tensor, axis=2)
    max_counts = np.max(tensor, axis=2)
    confidence = np.divide(
        max_counts,
        row_sums,
        out=np.zeros_like(max_counts, dtype=np.float64),
        where=row_sums > 0,
    )
    active_mask = row_sums > 0
    assignments = np.where(active_mask, assignments, -1)
    final_observation_ids = (
        dominant_observation_ids_by_learned_state(
            states_parquet_path,
            learned_state_column_name=learned_state_column_name,
            observation_column_name=observation_column_name,
        )
        if states_parquet_path is not None
        else {}
    )

    figure = plt.figure(figsize=(11, 8), facecolor="white")
    grid = figure.add_gridspec(
        2,
        2,
        width_ratios=[4.8, 1.2],
        height_ratios=[2.2, 1.4],
        wspace=0.28,
        hspace=0.18,
    )
    assignment_axes = figure.add_subplot(grid[0, 0])
    confidence_axes = figure.add_subplot(grid[1, 0], sharex=assignment_axes)
    observation_axes = figure.add_subplot(grid[:, 1])

    assignment_display = np.ma.masked_where(~active_mask, assignments)
    n_true_states = int(tensor.shape[2])
    assignment_colors = plt.get_cmap("tab10")(np.linspace(0, 1, max(n_true_states, 1)))
    assignment_cmap = ListedColormap(assignment_colors)
    assignment_cmap.set_bad(color="#efefef")
    assignment_norm = BoundaryNorm(np.arange(-0.5, n_true_states + 0.5, 1.0), assignment_cmap.N)

    assignment_image = assignment_axes.imshow(
        assignment_display.T,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        cmap=assignment_cmap,
        norm=assignment_norm,
    )
    assignment_colorbar = figure.colorbar(assignment_image, ax=assignment_axes, pad=0.01)
    assignment_colorbar.set_label("Assigned visited GT state index (compressed)", color="black")
    assignment_colorbar.ax.yaxis.set_tick_params(color="black")
    plt.setp(assignment_colorbar.ax.get_yticklabels(), color="black")
    if n_true_states > 0:
        assignment_colorbar.set_ticks(np.arange(n_true_states))

    confidence_display = np.ma.masked_where(~active_mask, confidence)
    confidence_cmap = plt.get_cmap("Blues").copy()
    confidence_cmap.set_bad(color="#efefef")
    confidence_image = confidence_axes.imshow(
        confidence_display.T,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        cmap=confidence_cmap,
        vmin=0.0,
        vmax=1.0,
    )
    confidence_colorbar = figure.colorbar(confidence_image, ax=confidence_axes, pad=0.01)
    confidence_colorbar.set_label("Assignment confidence", color="black")
    confidence_colorbar.ax.yaxis.set_tick_params(color="black")
    plt.setp(confidence_colorbar.ax.get_yticklabels(), color="black")

    final_observation_column = np.full((active_mask.shape[1],), -1, dtype=np.int64)
    for learned_state_index, observation_id in final_observation_ids.items():
        if 0 <= int(learned_state_index) < final_observation_column.shape[0]:
            final_observation_column[int(learned_state_index)] = int(observation_id)
    valid_observation_values = sorted({int(value) for value in final_observation_column[final_observation_column >= 0].tolist()})
    if valid_observation_values:
        observed_ids = valid_observation_values
        observed_count = max(len(observed_ids), 1)
        observation_colors = plt.get_cmap("Set3")(np.linspace(0, 1, observed_count))
        observation_cmap = ListedColormap(observation_colors)
        observation_cmap.set_bad(color="#efefef")
        id_to_dense = {obs_id: index for index, obs_id in enumerate(observed_ids)}
        dense_observation = np.full(final_observation_column.shape, -1, dtype=np.int64)
        for observed_id, dense_id in id_to_dense.items():
            dense_observation[final_observation_column == observed_id] = int(dense_id)
        observation_display = np.ma.masked_where(final_observation_column < 0, dense_observation)
        observation_norm = BoundaryNorm(np.arange(-0.5, len(observed_ids) + 0.5, 1.0), observation_cmap.N)
        observation_image = observation_axes.imshow(
            observation_display[:, np.newaxis],
            aspect="auto",
            origin="lower",
            interpolation="nearest",
            cmap=observation_cmap,
            norm=observation_norm,
        )
        observation_colorbar = figure.colorbar(observation_image, ax=observation_axes, pad=0.01)
        observation_colorbar.set_label("Dominant observation ID", color="black")
        observation_colorbar.ax.yaxis.set_tick_params(color="black")
        plt.setp(observation_colorbar.ax.get_yticklabels(), color="black")
        observation_colorbar.set_ticks(np.arange(len(observed_ids)))
        observation_colorbar.set_ticklabels([str(obs_id) for obs_id in observed_ids])
    else:
        observation_display = np.ma.masked_where(final_observation_column < 0, np.full(final_observation_column.shape, -1, dtype=np.int64))
        observation_cmap = ListedColormap(["#efefef"])
        observation_axes.imshow(
            observation_display[:, np.newaxis],
            aspect="auto",
            origin="lower",
            interpolation="nearest",
            cmap=observation_cmap,
        )

    resolved_title = title or f"{matrix_key.replace('_', ' ').title()} Alignment Over Time"
    assignment_axes.set_title(resolved_title, color="black")
    assignment_axes.set_ylabel("Learned state", color="black")
    confidence_axes.set_ylabel("Learned state", color="black")
    observation_axes.set_title("Final observation ID", color="black")

    learned_state_count = assignments.shape[1]
    ytick_positions = np.arange(learned_state_count)
    ytick_labels = [
        str(index)
        for index in range(learned_state_count)
    ]
    for axes_item in [assignment_axes, confidence_axes]:
        axes_item.set_yticks(ytick_positions)
        axes_item.set_yticklabels(ytick_labels)
    observation_axes.set_yticks(ytick_positions)
    observation_axes.set_yticklabels(ytick_labels)
    observation_axes.set_xticks([0])
    observation_axes.set_xticklabels(["final"], rotation=45, ha="right")

    tick_positions = np.arange(len(steps))
    tick_labels = [str(int(step)) for step in steps]
    confidence_axes.set_xticks(tick_positions)
    confidence_axes.set_xticklabels(tick_labels, rotation=45, ha="right")
    confidence_axes.set_xlabel("Evaluation checkpoint", color="black")

    for axes_item in [assignment_axes, confidence_axes, observation_axes]:
        axes_item.set_facecolor("white")
        axes_item.tick_params(colors="black")
        axes_item.grid(False)
        for spine in axes_item.spines.values():
            spine.set_color("black")

    assignment_axes.text(
        1.01,
        0.5,
        "light gray = nonexistent",
        transform=assignment_axes.transAxes,
        rotation=90,
        va="center",
        ha="left",
        color="black",
    )
    confidence_axes.text(
        1.01,
        0.5,
        "light gray = nonexistent",
        transform=confidence_axes.transAxes,
        rotation=90,
        va="center",
        ha="left",
        color="black",
    )
    observation_axes.text(
        1.01,
        0.5,
        "light gray = unavailable",
        transform=observation_axes.transAxes,
        rotation=90,
        va="center",
        ha="left",
        color="black",
    )
    resolved_output_path = (
        Path(output_path)
        if output_path is not None
        else alignment_heatmap_output_path(f"{resolved_title} history")
    )
    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(resolved_output_path, bbox_inches="tight", facecolor="white", edgecolor="white", transparent=False)

    if show:
        plt.show()
    else:
        plt.close(figure)

    return AlignmentHistoryResult(
        steps=steps,
        assignments=assignments,
        confidence=confidence,
        active_mask=active_mask,
        output_path=resolved_output_path,
    )


def render_learned_state_identity_graph(
    *,
    latent_graph_npz_path: str | Path,
    states_parquet_path: str | Path,
    paper_precision_json_path: str | Path,
    matrix_key: str = "latent_exact_state_cooccurrence_matrix",
    title: str | None = None,
    output_path: str | Path | None = None,
    show: bool = True,
) -> LearnedStateIdentityGraphResult:
    from mercury.graph.core import Graph

    apply_light_mode_plot_style()

    graph = Graph.from_npz(str(latent_graph_npz_path))
    adjacency = np.asarray(graph.adj, dtype=np.float64)
    node_count = int(adjacency.shape[0])

    dominant_observation_ids = dominant_observation_ids_by_learned_state(states_parquet_path)
    states_frame = pl.read_parquet(states_parquet_path).select("latent_bmu")
    occupancy_series = (
        states_frame.group_by("latent_bmu").len().sort("latent_bmu")
        if not states_frame.is_empty()
        else pl.DataFrame({"latent_bmu": [], "len": []})
    )
    occupancy = {
        int(node): int(count)
        for node, count in zip(
            occupancy_series.get_column("latent_bmu").to_list(),
            occupancy_series.get_column("len").to_list(),
            strict=False,
        )
    }

    contingency = load_cooccurrence_matrix_from_paper_precision_json(
        paper_precision_json_path,
        matrix_key=matrix_key,
    )
    assignments = np.argmax(contingency, axis=1) if contingency.size else np.asarray([], dtype=np.int64)
    compressed_ground_truth_assignments = {
        int(node_index): int(assignments[node_index])
        for node_index in range(min(node_count, int(assignments.shape[0])))
    }

    nx_graph = nx.DiGraph()
    nx_graph.add_nodes_from(range(node_count))
    for source_idx, target_idx in np.argwhere(adjacency > 0.0).tolist():
        nx_graph.add_edge(int(source_idx), int(target_idx), weight=float(adjacency[source_idx, target_idx]))
    positions = (
        nx.spring_layout(nx_graph, seed=7, weight="weight")
        if nx_graph.number_of_edges() > 0
        else {node: np.asarray([float(node), 0.0], dtype=np.float32) for node in range(node_count)}
    )

    observation_ids = sorted(set(dominant_observation_ids.values()))
    observation_index = {obs_id: index for index, obs_id in enumerate(observation_ids)}
    fill_values = np.asarray(
        [observation_index[dominant_observation_ids[node]] if node in dominant_observation_ids else -1 for node in range(node_count)],
        dtype=np.int64,
    )
    gt_ids = sorted(set(compressed_ground_truth_assignments.values()))
    gt_palette = plt.get_cmap("tab10")(np.linspace(0, 1, max(len(gt_ids), 1)))
    gt_color_lookup = {gt_id: gt_palette[index] for index, gt_id in enumerate(gt_ids)}
    node_edge_colors = [
        gt_color_lookup.get(compressed_ground_truth_assignments.get(node, -1), "#444444")
        for node in range(node_count)
    ]
    fill_palette = plt.get_cmap("Set3")(np.linspace(0, 1, max(len(observation_ids), 1)))
    node_face_colors = [
        fill_palette[int(fill_values[node])] if fill_values[node] >= 0 else "#efefef"
        for node in range(node_count)
    ]
    node_sizes = np.asarray(
        [220.0 + 18.0 * np.sqrt(float(occupancy.get(node, 0))) for node in range(node_count)],
        dtype=np.float64,
    )

    figure, axes = plt.subplots(figsize=(10, 8), facecolor="white")
    axes.set_title(
        title or "Learned State Identity Graph",
        color="black",
    )
    nx.draw_networkx_edges(
        nx_graph,
        pos=positions,
        ax=axes,
        edge_color="#9aa0a6",
        width=1.2,
        arrows=True,
        arrowsize=14,
        alpha=0.65,
        connectionstyle="arc3,rad=0.08",
    )
    nx.draw_networkx_nodes(
        nx_graph,
        pos=positions,
        ax=axes,
        node_color=node_face_colors,
        edgecolors=node_edge_colors,
        linewidths=2.4,
        node_size=node_sizes.tolist(),
    )
    node_labels = {
        node: f"{node}\nobs {dominant_observation_ids.get(node, '?')}"
        for node in range(node_count)
    }
    nx.draw_networkx_labels(nx_graph, pos=positions, labels=node_labels, ax=axes, font_size=8, font_color="black")
    axes.set_axis_off()
    _apply_light_mode_to_axes(figure, axes)

    resolved_output_path = (
        Path(output_path)
        if output_path is not None
        else alignment_heatmap_output_path("learned_state_identity_graph")
    )
    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(resolved_output_path, bbox_inches="tight", facecolor="white", edgecolor="white", transparent=False)

    if show:
        plt.show()
    else:
        plt.close(figure)

    return LearnedStateIdentityGraphResult(
        dominant_observation_ids=dominant_observation_ids,
        compressed_ground_truth_assignments=compressed_ground_truth_assignments,
        occupancy=occupancy,
        output_path=resolved_output_path,
    )


def render_observation_history_from_paper_precision_json(
    *,
    paper_precision_json_path: str | Path,
    states_parquet_path: str | Path,
    matrix_key: str = "latent_exact_state_cooccurrence_matrix",
    learned_state_column_name: str = "latent_bmu",
    observation_column_name: str = "sensory_bmu",
    title: str | None = None,
    output_path: str | Path | None = None,
    show: bool = True,
) -> ObservationHistoryResult:
    apply_light_mode_plot_style()

    steps, tensor = load_cooccurrence_history_tensor_from_paper_precision_json(
        paper_precision_json_path,
        matrix_key=matrix_key,
    )
    active_mask = tensor.sum(axis=2) > 0
    dominant_observation_ids = dominant_observation_ids_by_learned_state(
        states_parquet_path,
        learned_state_column_name=learned_state_column_name,
        observation_column_name=observation_column_name,
    )
    observation_matrix = np.full(active_mask.shape, -1, dtype=np.int64)
    for learned_state_index in range(active_mask.shape[1]):
        if learned_state_index in dominant_observation_ids:
            observation_matrix[:, learned_state_index] = dominant_observation_ids[learned_state_index]
    observation_display = np.ma.masked_where(~active_mask, observation_matrix)

    observed_ids = sorted(set(dominant_observation_ids.values()))
    observed_count = max(len(observed_ids), 1)
    observation_colors = plt.get_cmap("Set3")(np.linspace(0, 1, observed_count))
    observation_cmap = ListedColormap(observation_colors)
    observation_cmap.set_bad(color="#efefef")
    if observed_ids:
        id_to_dense = {obs_id: index for index, obs_id in enumerate(observed_ids)}
        dense_display = np.ma.masked_where(~active_mask, np.vectorize(lambda x: id_to_dense.get(int(x), -1))(observation_matrix))
        observation_norm = BoundaryNorm(np.arange(-0.5, len(observed_ids) + 0.5, 1.0), observation_cmap.N)
    else:
        dense_display = observation_display
        observation_norm = BoundaryNorm(np.arange(-0.5, 1.5, 1.0), observation_cmap.N)

    figure, axes = plt.subplots(figsize=(11, 4.8), facecolor="white")
    image = axes.imshow(
        dense_display.T,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        cmap=observation_cmap,
        norm=observation_norm,
    )
    colorbar = figure.colorbar(image, ax=axes, pad=0.01)
    colorbar.set_label("Dominant observation ID", color="black")
    colorbar.ax.yaxis.set_tick_params(color="black")
    plt.setp(colorbar.ax.get_yticklabels(), color="black")
    if observed_ids:
        colorbar.set_ticks(np.arange(len(observed_ids)))
        colorbar.set_ticklabels([str(obs_id) for obs_id in observed_ids])

    resolved_title = title or "Observation Labels Over Time"
    axes.set_title(resolved_title, color="black")
    axes.set_ylabel("Learned state", color="black")
    axes.set_xlabel("Evaluation checkpoint", color="black")
    tick_positions = np.arange(len(steps))
    tick_labels = [str(int(step)) for step in steps]
    axes.set_xticks(tick_positions)
    axes.set_xticklabels(tick_labels, rotation=45, ha="right")

    ytick_positions = np.arange(active_mask.shape[1])
    ytick_labels = [
        f"{index} | obs {dominant_observation_ids[index]}" if index in dominant_observation_ids else str(index)
        for index in range(active_mask.shape[1])
    ]
    axes.set_yticks(ytick_positions)
    axes.set_yticklabels(ytick_labels)
    axes.set_facecolor("white")
    axes.tick_params(colors="black")
    for spine in axes.spines.values():
        spine.set_color("black")
    axes.text(
        1.01,
        0.5,
        "light gray = nonexistent",
        transform=axes.transAxes,
        rotation=90,
        va="center",
        ha="left",
        color="black",
    )

    figure.tight_layout()
    resolved_output_path = (
        Path(output_path)
        if output_path is not None
        else alignment_heatmap_output_path("observation_history")
    )
    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(resolved_output_path, bbox_inches="tight", facecolor="white", edgecolor="white", transparent=False)

    if show:
        plt.show()
    else:
        plt.close(figure)

    return ObservationHistoryResult(
        steps=steps,
        observation_ids=observation_matrix,
        active_mask=active_mask,
        output_path=resolved_output_path,
    )


def display_alignment_contingency_heatmap(
    *,
    decoded_states: np.ndarray,
    true_states: np.ndarray,
    n_learned: int,
    n_true: int,
    title: str = "Alignment Contingency Heatmap",
    output_path: str | Path | None = None,
) -> AlignmentHeatmapResult:
    try:
        from IPython.display import Image, display
    except ImportError as exc:  # pragma: no cover - notebook-only fallback
        raise RuntimeError("display_alignment_contingency_heatmap requires an IPython notebook environment") from exc

    result = render_alignment_contingency_heatmap(
        decoded_states=decoded_states,
        true_states=true_states,
        n_learned=n_learned,
        n_true=n_true,
        title=title,
        output_path=output_path,
        show=False,
    )
    print(f"Saved plot: {result.output_path}")
    display(Image(filename=str(result.output_path), embed=True))
    return result


def display_alignment_contingency_heatmap_from_paper_precision_json(
    *,
    paper_precision_json_path: str | Path,
    matrix_key: str = "latent_exact_state_cooccurrence_matrix",
    title: str | None = None,
    output_path: str | Path | None = None,
) -> AlignmentHeatmapResult:
    try:
        from IPython.display import Image, display
    except ImportError as exc:  # pragma: no cover - notebook-only fallback
        raise RuntimeError(
            "display_alignment_contingency_heatmap_from_paper_precision_json requires an IPython notebook environment",
        ) from exc

    result = render_alignment_contingency_heatmap_from_paper_precision_json(
        paper_precision_json_path=paper_precision_json_path,
        matrix_key=matrix_key,
        title=title,
        output_path=output_path,
        show=False,
    )
    print(f"Saved plot: {result.output_path}")
    display(Image(filename=str(result.output_path), embed=True))
    return result


def display_cooccurrence_history_surface_from_paper_precision_json(
    *,
    paper_precision_json_path: str | Path,
    matrix_key: str = "latent_exact_state_cooccurrence_matrix",
    learned_state_index: int | None = None,
    title: str | None = None,
    output_path: str | Path | None = None,
) -> CooccurrenceHistorySurfaceResult:
    try:
        from IPython.display import Image, display
    except ImportError as exc:  # pragma: no cover - notebook-only fallback
        raise RuntimeError(
            "display_cooccurrence_history_surface_from_paper_precision_json requires an IPython notebook environment",
        ) from exc

    result = render_cooccurrence_history_surface_from_paper_precision_json(
        paper_precision_json_path=paper_precision_json_path,
        matrix_key=matrix_key,
        learned_state_index=learned_state_index,
        title=title,
        output_path=output_path,
        show=False,
    )
    print(f"Saved plot: {result.output_path}")
    display(Image(filename=str(result.output_path), embed=True))
    return result


def display_alignment_history_from_paper_precision_json(
    *,
    paper_precision_json_path: str | Path,
    matrix_key: str = "latent_exact_state_cooccurrence_matrix",
    states_parquet_path: str | Path | None = None,
    learned_state_column_name: str = "latent_bmu",
    observation_column_name: str = "sensory_bmu",
    title: str | None = None,
    output_path: str | Path | None = None,
) -> AlignmentHistoryResult:
    try:
        from IPython.display import Image, display
    except ImportError as exc:  # pragma: no cover - notebook-only fallback
        raise RuntimeError(
            "display_alignment_history_from_paper_precision_json requires an IPython notebook environment",
        ) from exc

    result = render_alignment_history_from_paper_precision_json(
        paper_precision_json_path=paper_precision_json_path,
        matrix_key=matrix_key,
        states_parquet_path=states_parquet_path,
        learned_state_column_name=learned_state_column_name,
        observation_column_name=observation_column_name,
        title=title,
        output_path=output_path,
        show=False,
    )
    print(f"Saved plot: {result.output_path}")
    display(Image(filename=str(result.output_path), embed=True))
    return result


def display_learned_state_identity_graph(
    *,
    latent_graph_npz_path: str | Path,
    states_parquet_path: str | Path,
    paper_precision_json_path: str | Path,
    matrix_key: str = "latent_exact_state_cooccurrence_matrix",
    title: str | None = None,
    output_path: str | Path | None = None,
) -> LearnedStateIdentityGraphResult:
    try:
        from IPython.display import Image, display
    except ImportError as exc:  # pragma: no cover - notebook-only fallback
        raise RuntimeError("display_learned_state_identity_graph requires an IPython notebook environment") from exc

    result = render_learned_state_identity_graph(
        latent_graph_npz_path=latent_graph_npz_path,
        states_parquet_path=states_parquet_path,
        paper_precision_json_path=paper_precision_json_path,
        matrix_key=matrix_key,
        title=title,
        output_path=output_path,
        show=False,
    )
    print(f"Saved plot: {result.output_path}")
    display(Image(filename=str(result.output_path), embed=True))
    return result


def display_observation_history_from_paper_precision_json(
    *,
    paper_precision_json_path: str | Path,
    states_parquet_path: str | Path,
    matrix_key: str = "latent_exact_state_cooccurrence_matrix",
    learned_state_column_name: str = "latent_bmu",
    observation_column_name: str = "sensory_bmu",
    title: str | None = None,
    output_path: str | Path | None = None,
) -> ObservationHistoryResult:
    try:
        from IPython.display import Image, display
    except ImportError as exc:  # pragma: no cover - notebook-only fallback
        raise RuntimeError("display_observation_history_from_paper_precision_json requires an IPython notebook environment") from exc

    result = render_observation_history_from_paper_precision_json(
        paper_precision_json_path=paper_precision_json_path,
        states_parquet_path=states_parquet_path,
        matrix_key=matrix_key,
        learned_state_column_name=learned_state_column_name,
        observation_column_name=observation_column_name,
        title=title,
        output_path=output_path,
        show=False,
    )
    print(f"Saved plot: {result.output_path}")
    display(Image(filename=str(result.output_path), embed=True))
    return result


def display_notebook_plot(
    results_frame: pl.DataFrame,
    plot: NotebookPlot,
    *,
    output_path: str | Path | None = None,
) -> SummaryFrameBundle:
    try:
        from IPython.display import Image, display
    except ImportError as exc:  # pragma: no cover - notebook-only fallback
        raise RuntimeError("display_notebook_plot requires an IPython notebook environment") from exc

    resolved_output_path = (
        Path(output_path)
        if output_path is not None
        else notebook_plot_output_path(plot)
    )

    summary_bundle = render_notebook_plot(
        results_frame=results_frame,
        plot=plot,
        output_path=resolved_output_path,
        show=False,
    )
    print(f"Saved plot: {resolved_output_path}")
    display(Image(filename=str(resolved_output_path), embed=True))
    return summary_bundle


def make_precision_series(
    name: str,
    label: str,
    metric_column_name: str,
    level: int,
    rand_prob: float,
    weight_memory: float,
    mercury_split_sensory_raw_latent_valid: bool,
    line_style: str = "-",
    is_baseline: bool = False,
    show_standard_deviation_band: bool = True,
) -> SeriesSpec:
    return SeriesSpec(
        name=name,
        filter_spec=FilterSpec(
            parameter_values={
                "level": level,
                "rand_prob": rand_prob,
                "weight_memory": weight_memory,
                "mercury_split_sensory_raw_latent_valid": mercury_split_sensory_raw_latent_valid,
            }
        ),
        metric_spec=MetricSpec(
            column_name=metric_column_name,
            label=label,
        ),
        style=SeriesStyle(
            line_style=line_style,
            show_standard_deviation_band=show_standard_deviation_band,
        ),
        is_baseline=is_baseline,
    )
