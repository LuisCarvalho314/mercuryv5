# scripts/compute_precision.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from utils.metrics import compute_precision


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


def _load_column(parquet_path: Path, column_name: str) -> np.ndarray:
    frame = pl.read_parquet(parquet_path).select(column_name)
    return frame.get_column(column_name).to_numpy()


def _rolling_precision(series: np.ndarray, ground_truth: np.ndarray, window_length: int) -> np.ndarray:
    series = np.asarray(series).astype(np.int64, copy=False)
    ground_truth = np.asarray(ground_truth).astype(np.int64, copy=False)

    if series.shape[0] != ground_truth.shape[0]:
        raise ValueError(f"Length mismatch: series={series.shape[0]} vs ground_truth={ground_truth.shape[0]}")

    out = np.empty(series.shape[0] - 1, dtype=np.float32)
    for index in range(1, series.shape[0]):
        start = max(0, index - window_length)
        out[index - 1] = compute_precision(series[start:index], ground_truth[start:index])
    return out


def compute_metrics(
    *,
    level: int,
    window_length: int = 100,
    sensor: str = "cartesian",
    sensor_range: Optional[int] = None,
    results_dir: str | Path = "results/mercury",
    out_dir: str | Path = "results/metrics",
    use_run_id: Optional[str] = None,
) -> None:
    results_root = Path(results_dir)
    output_root = Path(out_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    if use_run_id is None:
        states_path = _latest_states_path(results_root, level, sensor, sensor_range)
        run_id = states_path.name.replace("_states.parquet", "")
    else:
        directory = _results_dir(results_root, level, sensor, sensor_range)
        states_path = directory / f"{use_run_id}_states.parquet"
        if not states_path.exists():
            raise FileNotFoundError(f"Missing: {states_path}")
        run_id = use_run_id

    ground_truth = _load_column(states_path, "ground_truth_bmu")
    sensory_bmu = _load_column(states_path, "sensory_bmu")
    latent_bmu = _load_column(states_path, "latent_bmu")
    latent_node_count = _load_column(states_path, "latent_node_count")

    sensory_precision = _rolling_precision(sensory_bmu, ground_truth, window_length)
    latent_precision = _rolling_precision(latent_bmu, ground_truth, window_length)

    # Plot
    figure, axis = plt.subplots(1, 2, figsize=(12, 4))
    x = np.arange(1, ground_truth.shape[0], dtype=np.int32)

    axis[0].plot(x, sensory_precision, label="sensory")
    axis[0].plot(x, latent_precision, label="latent")
    axis[0].set_xlabel("Iterations")
    axis[0].set_ylabel("Precision")
    axis[0].set_title(f"Precision | level={level} | latent_sensor={sensor}")
    axis[0].legend()

    axis[1].plot(np.arange(latent_node_count.shape[0], dtype=np.int32), latent_node_count, label="latent_node_count")
    axis[1].set_xlabel("Iterations")
    axis[1].set_ylabel("Count")
    axis[1].set_title(f"Latent node count | level={level} | latent_sensor={sensor}")

    plot_path = output_root / f"{run_id}_precision.png"
    figure.tight_layout()
    figure.savefig(plot_path, dpi=200)
    plt.close(figure)

    # Metrics parquet
    metrics_frame = pl.DataFrame(
        {
            "iteration": x,
            "sensory_precision": sensory_precision,
            "latent_precision": latent_precision,
            "latent_node_count": latent_node_count[1:],
        }
    )
    metrics_parquet = output_root / f"{run_id}_precision.parquet"
    metrics_frame.write_parquet(metrics_parquet, compression="zstd", statistics=True)

    metadata = {
        "run_id": run_id,
        "level": level,
        "latent_sensor": sensor,
        "latent_sensor_range": sensor_range,
        "window_length": window_length,
        "inputs": {"states_parquet": str(states_path)},
        "outputs": {"metrics_parquet": str(metrics_parquet), "plot_png": str(plot_path)},
    }
    metadata_path = output_root / f"{run_id}_precision.metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
