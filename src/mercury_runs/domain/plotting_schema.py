from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import polars as pl
import matplotlib.pyplot as plt


AggregationMode = Literal["mean_std"]
SmoothingMode = Literal["none", "rolling_mean"]


@dataclass(frozen=True)
class FilterSpec:
    parameter_values: dict[str, Any] = field(default_factory=dict)

    def to_polars_expressions(self) -> list[pl.Expr]:
        return [
            pl.col(parameter_name) == parameter_value
            for parameter_name, parameter_value in self.parameter_values.items()
        ]


@dataclass(frozen=True)
class AggregationSpec:
    mode: AggregationMode = "mean_std"
    mean_alias_suffix: str = "mean"
    std_alias_suffix: str = "std"
    fill_null_standard_deviation_with: float = 0.0

    def mean_column_name(self, metric_column_name: str) -> str:
        return f"{metric_column_name}_{self.mean_alias_suffix}"

    def std_column_name(self, metric_column_name: str) -> str:
        return f"{metric_column_name}_{self.std_alias_suffix}"


@dataclass(frozen=True)
class SmoothingSpec:
    mode: SmoothingMode = "none"
    window_size: int = 1
    min_periods: int = 1

    def enabled(self) -> bool:
        return self.mode != "none" and self.window_size > 1


@dataclass(frozen=True)
class MetricSpec:
    column_name: str
    label: str


@dataclass(frozen=True)
class SeriesStyle:
    line_style: str = "-"
    line_width: float = 2.0
    band_alpha: float = 0.2
    show_standard_deviation_band: bool = True
    show_individual_runs: bool = False
    individual_run_alpha: float = 0.1


@dataclass(frozen=True)
class SeriesSpec:
    """
    One plotted series.

    A series is:
    - one filtered subset
    - one metric column
    - one label/style
    """
    name: str
    filter_spec: FilterSpec
    metric_spec: MetricSpec
    style: SeriesStyle = field(default_factory=SeriesStyle)
    is_baseline: bool = False


@dataclass(frozen=True)
class AxisSpec:
    x_column_name: str = "step"
    run_column_name: str = "run_id"
    x_axis_label: str = "Step"
    y_axis_label: str = "Metric value"
    y_limits: tuple[float, float] | None = None
    x_limits: tuple[float, float] | None = None


@dataclass(frozen=True)
class PlotSpec:
    title: str | None = None
    figure_size: tuple[float, float] = (9, 5)
    show_legend: bool = True
    legend_location: str = "best"
    use_grid: bool = False


@dataclass(frozen=True)
class PlotRequest:
    series_specs: list[SeriesSpec]
    axis_spec: AxisSpec = field(default_factory=AxisSpec)
    aggregation_spec: AggregationSpec = field(default_factory=AggregationSpec)
    smoothing_spec: SmoothingSpec = field(default_factory=SmoothingSpec)
    plot_spec: PlotSpec = field(default_factory=PlotSpec)


@dataclass
class SummaryFrameBundle:
    """
    Stores the filtered raw frame and the aggregated frame for each series.
    """
    raw_frames_by_series_name: dict[str, pl.DataFrame]
    summary_frames_by_series_name: dict[str, pl.DataFrame]