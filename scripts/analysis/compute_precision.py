# scripts/compute_precision.py
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Optional

import numpy as np


def _is_plot_metric(name: str, value: Any) -> bool:
    if not isinstance(value, (int, float)):
        return False
    return (
        name.endswith("_precision")
        or name.endswith("_edge_f1")
        or name.endswith("_mean_total_variation")
    )


def _metric_sort_key(name: str) -> tuple[int, int, str]:
    if "_" not in name:
        return (99, 99, name)
    if name.startswith("sensory_"):
        channel = "sensory"
        metric = name[len("sensory_"):]
    elif name.startswith("latent_"):
        channel = "latent"
        metric = name[len("latent_"):]
    elif name.startswith("capacity_"):
        channel = "capacity"
        metric = name[len("capacity_"):]
    else:
        channel, metric = name.split("_", 1)
    metric_order = {
        "precision": 0,
        "edge_f1": 1,
        "action_conditioned_edge_f1": 2,
        "mean_total_variation": 3,
        "action_conditioned_mean_total_variation": 4,
    }
    channel_order = {
        "sensory": 0,
        "latent": 1,
        "capacity": 2,
    }
    return (
        metric_order.get(metric, 99),
        channel_order.get(channel, 99),
        name,
    )


def _display_metric_name(name: str) -> str:
    if "_" not in name:
        return name
    if name.startswith("sensory_"):
        channel = "sensory"
        metric = name[len("sensory_"):]
    elif name.startswith("latent_"):
        channel = "latent"
        metric = name[len("latent_"):]
    elif name.startswith("capacity_"):
        channel = "capacity"
        metric = name[len("capacity_"):]
    else:
        channel, metric = name.split("_", 1)
    metric_label = {
        "precision": "Precision",
        "edge_f1": "Edge F1",
        "mean_total_variation": "Mean Total Variation",
        "action_conditioned_edge_f1": "Action-Conditioned Edge F1",
        "action_conditioned_mean_total_variation": "Action-Conditioned Mean Total Variation",
    }.get(metric, metric.replace("_", " ").title())
    channel_label = {
        "sensory": "Sensory",
        "latent": "Latent",
        "capacity": "Capacity",
    }.get(channel, channel.replace("_", " ").title())
    return f"{metric_label} | {channel_label}"


def _preferred_plot_metric_names(metric_names: set[str]) -> list[str]:
    selected = set(metric_names)
    for name in list(metric_names):
        if name.endswith("_edge_precision"):
            f1_name = name[: -len("_edge_precision")] + "_edge_f1"
            if f1_name in metric_names:
                selected.discard(name)
                selected.add(f1_name)
    return sorted(selected, key=_metric_sort_key)


def _results_dir(results_root: Path, level: int, sensor: str, sensor_range: Optional[int]) -> Path:
    base = results_root / f"level={level}" / f"sensor={sensor}"
    if sensor == "cardinal distance":
        return base / f"range={sensor_range}"
    return base


def _latest_states_path(results_root: Path, level: int, sensor: str, sensor_range: Optional[int]) -> Path:
    directory = _results_dir(results_root, level, sensor, sensor_range)
    if not directory.exists():
        raise FileNotFoundError(f"Results directory not found: {directory}")

    candidates = sorted(directory.glob("*_states.parquet"))
    if not candidates:
        raise FileNotFoundError(f"No *_states.parquet in: {directory}")
    return candidates[-1]


def _paper_precision_json_path(output_root: Path, run_id: str) -> Path:
    return output_root / f"{run_id}_paper_precision.json"


def _plot_paper_precision(
    *,
    paper_precision_path: Path,
    plot_path: Path,
    method_metadata: dict[str, Any],
) -> Optional[str]:
    try:
        os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
        os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        payload = json.loads(paper_precision_path.read_text(encoding="utf-8"))
        metrics = dict(payload.get("metrics") or {})
        metrics_by_capacity = dict(payload.get("metrics_by_capacity") or {})
        protocol = dict(payload.get("protocol") or {})
        history = list(payload.get("history") or [])
        history_by_capacity = dict(payload.get("history_by_capacity") or {})
        precision_keys = _preferred_plot_metric_names(
            {
                key
                for key, value in metrics.items()
                if _is_plot_metric(key, value)
            }
        )
        figure, axis = plt.subplots(figsize=(8, 4.5))

        def _history_x_values(items: list[dict[str, Any]]) -> tuple[list[int], str]:
            if items and all("observed_samples" in item for item in items):
                return [int(item["observed_samples"]) for item in items], "Observed Samples"
            if items and all("step" in item for item in items):
                return [int(item["step"]) for item in items], "Step"
            if items and all("iteration" in item for item in items):
                return [int(item["iteration"]) for item in items], "Iteration"
            return list(range(1, len(items) + 1)), "Evaluation"

        if history_by_capacity:
            plotted = False
            method_name = str(payload.get("method") or method_metadata.get("method") or "").strip().lower()
            for capacity_key, capacity_history in sorted(history_by_capacity.items(), key=lambda item: int(item[0])):
                if not capacity_history:
                    continue
                x_values, x_label = _history_x_values(capacity_history)
                axis.set_xlabel(x_label)
                precision_series_keys = _preferred_plot_metric_names(
                    {
                        k
                        for item in capacity_history
                        for k, value in item.items()
                        if _is_plot_metric(k, value)
                    }
                )
                if method_name == "pocml":
                    preferred_key = "capacity_precision" if "capacity_precision" in precision_series_keys else (precision_series_keys[0] if precision_series_keys else None)
                    if preferred_key is not None:
                        axis.plot(x_values, [float(item[preferred_key]) for item in capacity_history], marker="o", label=f"K={capacity_key}")
                        plotted = True
                    continue
                for key in precision_series_keys:
                    axis.plot(
                        x_values,
                        [float(item[key]) for item in capacity_history],
                        label=f"{_display_metric_name(key)} | K={capacity_key}",
                    )
                    plotted = True
            if not plotted:
                plt.close(figure)
                return "no_paper_precision_data"
            axis.legend()
            axis.set_ylabel("Paper Precision")
            axis.set_title(f"Paper Precision | method={method_metadata['method']} | level={method_metadata['level']}")
        elif history:
            x_values, x_label = _history_x_values(history)
            axis.set_xlabel(x_label)
            plotted = False
            for key in _preferred_plot_metric_names(
                {
                    k
                    for item in history
                    for k, value in item.items()
                    if _is_plot_metric(k, value)
                }
            ):
                axis.plot(x_values, [float(item[key]) for item in history], marker="o", label=_display_metric_name(key))
                plotted = True
            if plotted:
                axis.legend()
            axis.set_ylabel("Paper Metrics")
            axis.set_title(f"Paper Precision | method={method_metadata['method']} | level={method_metadata['level']}")
            if not plotted:
                plt.close(figure)
                return "no_paper_precision_data"
        elif metrics_by_capacity:
            ordered_capacities = sorted(metrics_by_capacity.keys(), key=lambda value: int(value))
            precision_metric_names = _preferred_plot_metric_names(
                {
                    key
                    for capacity_metrics in metrics_by_capacity.values()
                    for key, value in dict(capacity_metrics or {}).items()
                    if _is_plot_metric(key, value)
                }
            )
            if not precision_metric_names:
                plt.close(figure)
                return "no_paper_precision_data"
            x_positions = np.arange(len(ordered_capacities), dtype=np.float32)
            width = 0.8 / max(1, len(precision_metric_names))
            for index, metric_name in enumerate(precision_metric_names):
                values = [float(dict(metrics_by_capacity.get(capacity) or {}).get(metric_name, 0.0)) for capacity in ordered_capacities]
                axis.bar(x_positions + (index * width), values, width=width, label=_display_metric_name(metric_name))
            axis.set_xticks(x_positions + width * max(0, len(precision_metric_names) - 1) / 2.0, [f"K={capacity}" for capacity in ordered_capacities])
            axis.set_ylim(0.0, 1.0)
            axis.set_ylabel("Paper Metrics")
            axis.set_title(f"Paper Precision | method={method_metadata['method']} | level={method_metadata['level']}")
            axis.legend()
        else:
            if not precision_keys:
                plt.close(figure)
                return "no_paper_precision_data"
            values = [float(metrics[key]) for key in precision_keys]
            axis.bar([_display_metric_name(key) for key in precision_keys], values, color="#3B82F6")
            axis.set_ylim(0.0, 1.0)
            axis.set_ylabel("Paper Metrics")
            axis.set_title(f"Paper Precision | method={method_metadata['method']} | level={method_metadata['level']}")
            summary_parts = []
            for key in ("mode", "num_walks", "walk_length"):
                if key in protocol:
                    summary_parts.append(f"{key}={protocol[key]}")
            if summary_parts:
                axis.text(
                    0.02,
                    0.98,
                    "\n".join(summary_parts),
                    transform=axis.transAxes,
                    ha="left",
                    va="top",
                    fontsize=9,
                    bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
                )

        figure.tight_layout()
        figure.savefig(plot_path, dpi=200)
        plt.close(figure)
        return None
    except Exception as exc:
        return repr(exc)


def write_paper_precision_plot(
    *,
    out_dir: str | Path,
    run_id: str,
    method_metadata: dict[str, Any],
    states_path: str | Path | None = None,
) -> None:
    output_root = Path(out_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    plot_path = output_root / f"{run_id}_precision.png"
    paper_precision_path = _paper_precision_json_path(output_root, run_id)
    plot_error: Optional[str] = None
    if paper_precision_path.exists():
        plot_error = _plot_paper_precision(
            paper_precision_path=paper_precision_path,
            plot_path=plot_path,
            method_metadata=method_metadata,
        )
        if plot_error is not None:
            if plot_path.exists():
                plot_path.unlink()
            plot_path = None
    else:
        plot_path = None

    metadata = {
        "run_id": run_id,
        "inputs": {
            "states_parquet": (str(states_path) if states_path is not None else None),
            "paper_precision_json": (str(paper_precision_path) if paper_precision_path.exists() else None),
        },
        "outputs": {"plot_png": (str(plot_path) if plot_path is not None else None)},
        "plot_error": plot_error,
        **method_metadata,
    }
    metadata_path = output_root / f"{run_id}_precision.metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def compute_metrics_for_states(
    *,
    states_path: str | Path,
    out_dir: str | Path,
    run_id: str,
    window_length: int,
    precision_columns: dict[str, str],
    method_metadata: dict[str, Any],
    latent_node_count_column: str = "latent_node_count",
) -> None:
    del window_length, precision_columns, latent_node_count_column
    write_paper_precision_plot(
        out_dir=out_dir,
        run_id=run_id,
        method_metadata=method_metadata,
        states_path=states_path,
    )


def compute_metrics(
    *,
    level: int,
    window_length: int = 100,
    sensor: str = "cartesian",
    sensor_range: Optional[int] = None,
    results_dir: str | Path = "results/current/mercury",
    out_dir: str | Path = "results/current/metrics",
    use_run_id: Optional[str] = None,
) -> None:
    del window_length
    results_root = Path(results_dir)
    if use_run_id is None:
        states_path = _latest_states_path(results_root, level, sensor, sensor_range)
        run_id = states_path.name.replace("_states.parquet", "")
    else:
        directory = _results_dir(results_root, level, sensor, sensor_range)
        states_path = directory / f"{use_run_id}_states.parquet"
        if not states_path.exists():
            raise FileNotFoundError(f"Missing: {states_path}")
        run_id = use_run_id

    write_paper_precision_plot(
        out_dir=out_dir,
        run_id=run_id,
        method_metadata={
            "method": "mercury",
            "level": level,
            "latent_sensor": sensor,
            "latent_sensor_range": sensor_range,
        },
        states_path=states_path,
    )
