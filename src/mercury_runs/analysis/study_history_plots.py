from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib
import polars as pl

matplotlib.use("Agg", force=True)
from matplotlib import pyplot as plt


PLOTTED_METRICS: tuple[tuple[str, str], ...] = (
    ("precision", "State Precision"),
    ("edge_precision", "Edge Precision"),
    ("edge_recall", "Edge Recall"),
    ("edge_f1", "Edge F1"),
    ("mean_total_variation", "Mean Total Variation"),
    ("action_conditioned_edge_precision", "Action-Conditioned Edge Precision"),
    ("action_conditioned_edge_recall", "Action-Conditioned Edge Recall"),
    ("action_conditioned_edge_f1", "Action-Conditioned Edge F1"),
    ("action_conditioned_mean_total_variation", "Action-Conditioned Mean Total Variation"),
)
RELATION_PLOTS: tuple[tuple[str, str, str, str], ...] = (
    ("edge_f1_vs_state_count_ratio_pareto", "edge_f1", "state_count_ratio", "Pareto Front: Edge F1 vs Node Count / Ground Truth State Count"),
)

CHANNELS: tuple[str, str] = ("sensory", "latent")
LINESTYLES: dict[str, str] = {"sensory": "-", "latent": "--"}
IDENTITY_COLUMNS: frozenset[str] = frozenset({"run_id", "method", "level", "seed", "step"})
SUMMARY_EXCLUDED_COLUMNS: frozenset[str] = frozenset(
    {
        "timestamp_utc",
        "run_status",
        "target_n1_accuracy",
        "target_log_likelihood_mean",
        "sensory_n1_accuracy",
        "latent_n1_accuracy",
        "sensory_log_likelihood_mean",
        "latent_log_likelihood_mean",
    }
)
SUMMARY_EXCLUDED_PREFIXES: tuple[str, ...] = ("paper_", "comp_", "pocml_", "cscg_")
RUN_CONFIG_EXCLUDED_COLUMNS: frozenset[str] = frozenset(
    {
        "study",
        "_worker_run",
        "method",
        "max_workers",
        "no_parallel",
        "resume",
        "continue_on_error",
        "retry_failed",
        "retry_incomplete",
        "wandb",
        "wandb_project",
        "wandb_entity",
        "wandb_group",
        "wandb_tags",
        "wandb_mode",
        "wandb_job_type",
        "wandb_log_artifacts",
        "wandb_run_name",
        "study_root",
        "study_name",
        "study_config",
        "run_id",
        "requested_run_id",
        "datasets_root",
        "reuse_existing_run",
        "no_reuse_existing_run",
        "no_reuse_existing_dataset",
    }
)


def _history_metric_columns() -> frozenset[str]:
    columns = set()
    for metric_key, _ in PLOTTED_METRICS:
        for channel in CHANNELS:
            columns.add(f"{channel}_{metric_key}")
            columns.add(f"{channel}_{metric_key}_mean")
            columns.add(f"{channel}_{metric_key}_std")
    for channel in CHANNELS:
        columns.add(f"{channel}_node_count")
        columns.add(f"{channel}_node_count_mean")
        columns.add(f"{channel}_state_count_ratio")
        columns.add(f"{channel}_state_count_ratio_mean")
    columns.add("seed_count")
    return frozenset(columns)


def _slug(value: Any) -> str:
    return str(value).replace(".", "p").replace("-", "m").replace("/", "_").replace(" ", "_")


def _is_numeric(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _normalize_param_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if _is_numeric(value):
        return float(value)
    if isinstance(value, str):
        return value
    return None


def _format_param_value(value: Any) -> str:
    if _is_numeric(value):
        return f"{float(value):g}"
    if isinstance(value, bool):
        return str(value).lower()
    return str(value)


def _summary_hyperparameter_columns(summary: pl.DataFrame) -> tuple[str, ...]:
    columns: list[str] = []
    for column in summary.columns:
        if column in IDENTITY_COLUMNS or column in SUMMARY_EXCLUDED_COLUMNS:
            continue
        if any(column.startswith(prefix) for prefix in SUMMARY_EXCLUDED_PREFIXES):
            continue
        columns.append(column)
    return tuple(columns)


def _run_config_plot_params(run_config: dict[str, Any], summary_columns: set[str]) -> dict[str, Any]:
    extras: dict[str, Any] = {}
    for key, raw_value in run_config.items():
        if key in summary_columns or key in IDENTITY_COLUMNS or key in RUN_CONFIG_EXCLUDED_COLUMNS:
            continue
        value = _normalize_param_value(raw_value)
        if value is not None:
            extras[key] = value
    return extras


def _resolve_paper_precision_path(study_root: Path, run_id: str) -> Path | None:
    run_root = study_root / run_id
    status_path = run_root / "run_status.json"
    if not status_path.exists():
        return None
    payload = json.loads(status_path.read_text(encoding="utf-8"))
    artifacts = payload.get("artifacts") or {}
    raw_path = artifacts.get("mercury_paper_precision_json") or artifacts.get("paper_precision_json")
    if not raw_path:
        return None
    path = Path(raw_path)
    return path if path.exists() else None


def _resolve_run_config(study_root: Path, run_id: str) -> dict[str, Any]:
    config_path = study_root / run_id / "run_config.json"
    if not config_path.exists():
        return {}
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    cli_args = payload.get("cli_args")
    return dict(cli_args) if isinstance(cli_args, dict) else {}


def _hyperparameter_columns(frame: pl.DataFrame) -> tuple[str, ...]:
    metric_columns = _history_metric_columns()
    return tuple(column for column in frame.columns if column not in IDENTITY_COLUMNS and column not in metric_columns)


def _varying_hyperparameter_columns(frame: pl.DataFrame) -> tuple[str, ...]:
    varying: list[str] = []
    for column in _hyperparameter_columns(frame):
        series = frame.get_column(column)
        unique_count = series.drop_nulls().n_unique()
        if unique_count > 1:
            varying.append(column)
    return tuple(varying)


def _load_history_rows(study_root: Path, summary: pl.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    summary_param_columns = _summary_hyperparameter_columns(summary)
    summary_param_column_set = set(summary_param_columns)
    for record in summary.iter_rows(named=True):
        run_config = _resolve_run_config(study_root, str(record["run_id"]))
        history_path = _resolve_paper_precision_path(study_root, str(record["run_id"]))
        if history_path is None:
            continue
        payload = json.loads(history_path.read_text(encoding="utf-8"))
        history = payload.get("history") or []
        for item in history:
            flattened: dict[str, Any] = {
                "run_id": str(record["run_id"]),
                "level": int(record["level"]),
                "seed": int(record["seed"]),
                "step": int(item["step"]),
            }
            for column in summary_param_columns:
                value = _normalize_param_value(record.get(column))
                if value is not None:
                    flattened[column] = value
            for column, value in _run_config_plot_params(run_config, summary_param_column_set).items():
                flattened.setdefault(column, value)
            for metric_key, _ in PLOTTED_METRICS:
                for channel in CHANNELS:
                    name = f"{channel}_{metric_key}"
                    if item.get(name) is not None:
                        flattened[name] = float(item[name])
            for channel in CHANNELS:
                inferred = item.get(f"{channel}_inferred_state_count")
                gt = item.get(f"{channel}_ground_truth_state_count")
                if inferred is not None:
                    flattened[f"{channel}_node_count"] = float(inferred)
                if inferred is not None and gt not in (None, 0):
                    flattened[f"{channel}_state_count_ratio"] = float(inferred) / float(gt)
            rows.append(flattened)
    return rows


def load_study_history_frame(study_root: Path) -> pl.DataFrame:
    summary_path = study_root / "study_summary.parquet"
    if not summary_path.exists():
        raise FileNotFoundError(f"Study summary not found: {summary_path}")
    summary = pl.read_parquet(summary_path).filter(pl.col("method") == "mercury")
    if summary.is_empty():
        raise RuntimeError(f"No mercury rows found in {summary_path}")
    rows = _load_history_rows(study_root, summary)
    if not rows:
        raise RuntimeError(f"No Mercury paper-precision histories found under {study_root}")
    return pl.DataFrame(rows)


def aggregate_study_history(frame: pl.DataFrame) -> pl.DataFrame:
    group_columns = ["level", "step", *_hyperparameter_columns(frame)]
    exprs: list[pl.Expr] = [pl.col("seed").n_unique().alias("seed_count")]
    for channel in CHANNELS:
        name = f"{channel}_node_count"
        if name in frame.columns:
            exprs.append(pl.col(name).mean().alias(f"{name}_mean"))
        ratio_name = f"{channel}_state_count_ratio"
        if ratio_name in frame.columns:
            exprs.append(pl.col(ratio_name).mean().alias(f"{ratio_name}_mean"))
    for metric_key, _ in PLOTTED_METRICS:
        for channel in CHANNELS:
            name = f"{channel}_{metric_key}"
            if name not in frame.columns:
                continue
            exprs.extend(
                [
                    pl.col(name).mean().alias(f"{name}_mean"),
                    pl.col(name).std(ddof=1).fill_null(0.0).alias(f"{name}_std"),
                ]
            )
    return frame.group_by(group_columns).agg(exprs).sort(group_columns)


def _combo_label(record: dict[str, Any], varying: tuple[str, ...]) -> str:
    return ", ".join(f"{name}={_format_param_value(record[name])}" for name in varying)


def _plot_path(output_dir: Path, *, level: int, metric_key: str, fixed_param: str | None = None, fixed_value: Any | None = None) -> Path:
    metric_dir = output_dir / f"level_{level}" / metric_key
    if fixed_param is None:
        return metric_dir / "all_vary" / "plot.png"
    else:
        return metric_dir / f"fixed_{fixed_param}" / f"value_{_slug(fixed_value)}.png"


def _filtered_level_frame(
    aggregated: pl.DataFrame,
    *,
    level: int,
    fixed_param: str | None = None,
    fixed_value: Any | None = None,
) -> tuple[pl.DataFrame, tuple[str, ...], str]:
    level_frame = aggregated.filter(pl.col("level") == int(level))
    title_suffix = "all inferred parameters varying"
    if fixed_param is None:
        varying = _varying_hyperparameter_columns(level_frame)
        return level_frame.sort(["step", *varying]), varying, title_suffix
    if fixed_value is None:
        raise ValueError("fixed_value is required when fixed_param is provided")
    title_suffix = f"fixed {fixed_param}={_format_param_value(fixed_value)}"
    filtered = level_frame.filter(pl.col(fixed_param) == fixed_value)
    varying = _varying_hyperparameter_columns(filtered)
    filtered = filtered.sort(["step", *varying])
    return filtered, varying, title_suffix


def plot_metric_history(
    aggregated: pl.DataFrame,
    *,
    level: int,
    metric_key: str,
    metric_label: str,
    output_dir: Path,
    fixed_param: str | None = None,
    fixed_value: Any | None = None,
) -> Path | None:
    level_frame, varying, title_suffix = _filtered_level_frame(
        aggregated,
        level=level,
        fixed_param=fixed_param,
        fixed_value=fixed_value,
    )
    if level_frame.is_empty():
        return None

    combo_columns = list(varying)
    combos = (
        [{}]
        if not combo_columns
        else list(level_frame.select(*combo_columns).unique().sort(combo_columns).iter_rows(named=True))
    )
    cmap = plt.get_cmap("tab10")
    figure, axis = plt.subplots(figsize=(11, 6.5))
    plotted = False

    for index, combo in enumerate(combos):
        combo_data = level_frame
        for key, value in combo.items():
            combo_data = combo_data.filter(pl.col(key) == value)
        combo_data = combo_data.sort("step")
        if combo_data.is_empty():
            continue
        color = cmap(index % cmap.N)
        label_prefix = _combo_label({**combo, **({fixed_param: fixed_value} if fixed_param is not None else {})}, varying) if varying else "fixed combo"
        for channel in CHANNELS:
            mean_col = f"{channel}_{metric_key}_mean"
            std_col = f"{channel}_{metric_key}_std"
            if mean_col not in combo_data.columns or std_col not in combo_data.columns:
                continue
            steps = combo_data.get_column("step").to_list()
            means = combo_data.get_column(mean_col).to_list()
            stds = combo_data.get_column(std_col).to_list()
            lower = [float(mean) - float(std) for mean, std in zip(means, stds, strict=True)]
            upper = [float(mean) + float(std) for mean, std in zip(means, stds, strict=True)]
            axis.plot(
                steps,
                means,
                color=color,
                linestyle=LINESTYLES[channel],
                linewidth=2.0,
                label=f"{label_prefix} | {channel}",
            )
            axis.fill_between(steps, lower, upper, color=color, alpha=0.14)
            plotted = True

    if not plotted:
        plt.close(figure)
        return None

    axis.set_title(f"{metric_label} | level={level} | {title_suffix}")
    axis.set_xlabel("Training Steps")
    axis.set_ylabel("Value")
    if metric_key == "state_count_ratio":
        max_value = 1.0
        for line in axis.lines:
            y_data = line.get_ydata()
            if len(y_data):
                max_value = max(max_value, max(float(value) for value in y_data))
        axis.set_ylim(0.0, max_value * 1.05)
    else:
        axis.set_ylim(0.0, 1.0)
    axis.grid(True, alpha=0.25)
    axis.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0, fontsize=9)
    figure.tight_layout(rect=(0.0, 0.0, 0.78, 1.0))
    output_path = _plot_path(output_dir, level=level, metric_key=metric_key, fixed_param=fixed_param, fixed_value=fixed_value)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return output_path


def plot_metric_relation(
    aggregated: pl.DataFrame,
    *,
    level: int,
    plot_key: str,
    y_metric_key: str,
    x_metric_key: str,
    plot_label: str,
    output_dir: Path,
    fixed_param: str | None = None,
    fixed_value: Any | None = None,
) -> Path | None:
    output_dir.mkdir(parents=True, exist_ok=True)
    level_frame, varying, title_suffix = _filtered_level_frame(
        aggregated,
        level=level,
        fixed_param=fixed_param,
        fixed_value=fixed_value,
    )
    if level_frame.is_empty():
        return None

    combo_columns = list(varying)
    combos = (
        [{}]
        if not combo_columns
        else list(level_frame.select(*combo_columns).unique().sort(combo_columns).iter_rows(named=True))
    )
    cmap = plt.get_cmap("tab10")
    figure, axis = plt.subplots(figsize=(11, 6.5))
    plotted = False
    max_x = 1.0
    frontier_points: dict[str, list[tuple[float, float]]] = {channel: [] for channel in CHANNELS}

    for index, combo in enumerate(combos):
        combo_data = level_frame
        for key, value in combo.items():
            combo_data = combo_data.filter(pl.col(key) == value)
        combo_data = combo_data.sort("step")
        if combo_data.is_empty():
            continue
        color = cmap(index % cmap.N)
        label_prefix = _combo_label({**combo, **({fixed_param: fixed_value} if fixed_param is not None else {})}, varying) if varying else "fixed combo"
        for channel in CHANNELS:
            x_col = f"{channel}_{x_metric_key}_mean"
            y_col = f"{channel}_{y_metric_key}_mean"
            if x_col not in combo_data.columns or y_col not in combo_data.columns:
                continue
            x_values = [float(value) for value in combo_data.get_column(x_col).to_list()]
            y_values = [float(value) for value in combo_data.get_column(y_col).to_list()]
            if not x_values or not y_values:
                continue
            max_x = max(max_x, max(x_values))
            axis.scatter(
                x_values,
                y_values,
                color=color,
                marker="o",
                s=24,
                alpha=0.65,
                label=f"{label_prefix} | {channel}",
            )
            frontier_points[channel].extend(zip(x_values, y_values, strict=True))
            plotted = True

    if not plotted:
        plt.close(figure)
        return None

    for channel in CHANNELS:
        frontier = _pareto_frontier(frontier_points[channel])
        if not frontier:
            continue
        xs = [point[0] for point in frontier]
        ys = [point[1] for point in frontier]
        axis.plot(
            xs,
            ys,
            color="#111111",
            linestyle=LINESTYLES[channel],
            linewidth=3.0,
            alpha=0.95,
            label=f"Pareto frontier | {channel}",
            zorder=5,
        )
        axis.scatter(xs, ys, color="#111111", s=26, zorder=6)

    axis.set_title(f"{plot_label} | level={level} | {title_suffix}")
    axis.set_xlabel("Node Count / Ground Truth State Count")
    axis.set_ylabel("Edge F1")
    axis.set_xlim(0.0, max_x * 1.05)
    axis.set_ylim(0.0, 1.0)
    axis.grid(True, alpha=0.25)
    axis.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0, fontsize=9)
    figure.tight_layout(rect=(0.0, 0.0, 0.78, 1.0))
    output_path = _plot_path(output_dir, level=level, metric_key=plot_key, fixed_param=fixed_param, fixed_value=fixed_value)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return output_path


def _pareto_frontier(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if not points:
        return []
    best_by_x: dict[float, float] = {}
    for x, y in points:
        best_by_x[x] = max(y, best_by_x.get(x, float("-inf")))
    ordered = sorted(best_by_x.items(), key=lambda item: item[0])
    frontier: list[tuple[float, float]] = []
    best_y = float("-inf")
    for x, y in ordered:
        if y > best_y:
            frontier.append((x, y))
            best_y = y
    return frontier


def generate_study_history_plots(*, study_root: Path, output_dir: Path | None = None) -> dict[str, Path]:
    study_root = Path(study_root)
    destination = output_dir if output_dir is not None else study_root / "plots" / "study_history"
    history = load_study_history_frame(study_root)
    aggregated = aggregate_study_history(history)
    outputs: dict[str, Path] = {}
    if destination.exists():
        shutil.rmtree(destination)

    for level in sorted(aggregated.get_column("level").unique().to_list()):
        level_parameters = _varying_hyperparameter_columns(aggregated.filter(pl.col("level") == int(level)))
        for metric_key, metric_label in PLOTTED_METRICS:
            full_key = f"level={int(level)}|metric={metric_key}|view=all"
            full_path = plot_metric_history(
                aggregated,
                level=int(level),
                metric_key=metric_key,
                metric_label=metric_label,
                output_dir=Path(destination),
            )
            if full_path is not None:
                outputs[full_key] = full_path

            for fixed_param in level_parameters:
                fixed_values = (
                    aggregated.filter(pl.col("level") == int(level))
                    .select(fixed_param)
                    .unique()
                    .sort(fixed_param)
                    .get_column(fixed_param)
                    .to_list()
                )
                for fixed_value in fixed_values:
                    slice_path = plot_metric_history(
                        aggregated,
                        level=int(level),
                        metric_key=metric_key,
                        metric_label=metric_label,
                        output_dir=Path(destination),
                        fixed_param=fixed_param,
                        fixed_value=fixed_value,
                    )
                    if slice_path is not None:
                        outputs[
                            f"level={int(level)}|metric={metric_key}|fixed={fixed_param}|value={_format_param_value(fixed_value)}"
                        ] = slice_path
        for plot_key, y_metric_key, x_metric_key, plot_label in RELATION_PLOTS:
            full_key = f"level={int(level)}|metric={plot_key}|view=all"
            full_path = plot_metric_relation(
                aggregated,
                level=int(level),
                plot_key=plot_key,
                y_metric_key=y_metric_key,
                x_metric_key=x_metric_key,
                plot_label=plot_label,
                output_dir=Path(destination),
            )
            if full_path is not None:
                outputs[full_key] = full_path

            for fixed_param in level_parameters:
                fixed_values = (
                    aggregated.filter(pl.col("level") == int(level))
                    .select(fixed_param)
                    .unique()
                    .sort(fixed_param)
                    .get_column(fixed_param)
                    .to_list()
                )
                for fixed_value in fixed_values:
                    slice_path = plot_metric_relation(
                        aggregated,
                        level=int(level),
                        plot_key=plot_key,
                        y_metric_key=y_metric_key,
                        x_metric_key=x_metric_key,
                        plot_label=plot_label,
                        output_dir=Path(destination),
                        fixed_param=fixed_param,
                        fixed_value=fixed_value,
                    )
                    if slice_path is not None:
                        outputs[
                            f"level={int(level)}|metric={plot_key}|fixed={fixed_param}|value={_format_param_value(fixed_value)}"
                        ] = slice_path
    return outputs
