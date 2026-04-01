from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib
import numpy as np
import polars as pl
from matplotlib import pyplot as plt

from ..domain.models import ArtifactLayout


matplotlib.use("Agg")


_WEIGHTED_COMPONENT_COLUMNS = [
    "selected_memory_drive_weighted",
    "selected_undirected_drive_weighted",
    "selected_baseline_drive_weighted",
    "selected_action_drive_weighted",
]

_SHARE_COMPONENT_COLUMNS = [
    "selected_memory_drive_share",
    "selected_undirected_drive_share",
    "selected_baseline_drive_share",
    "selected_action_drive_share",
]

_COMPONENT_LABELS = {
    "selected_memory_drive_weighted": "memory",
    "selected_undirected_drive_weighted": "undirected graph",
    "selected_baseline_drive_weighted": "baseline graph",
    "selected_action_drive_weighted": "action",
    "selected_memory_drive_share": "memory",
    "selected_undirected_drive_share": "undirected graph",
    "selected_baseline_drive_share": "baseline graph",
    "selected_action_drive_share": "action",
}


def _save_figure(figure: plt.Figure, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _load_paper_precision_history(paper_precision_path: Path) -> list[dict[str, object]]:
    if not paper_precision_path.exists():
        return []
    payload = json.loads(paper_precision_path.read_text(encoding="utf-8"))
    history = payload.get("history")
    if not isinstance(history, list):
        return []
    return [entry for entry in history if isinstance(entry, dict) and entry.get("step") is not None]


def _checkpoint_summary_frame(
    attribution_frame: pl.DataFrame,
    paper_history: list[dict[str, object]],
) -> pl.DataFrame:
    if attribution_frame.is_empty() or not paper_history:
        return pl.DataFrame()
    checkpoint_rows: list[dict[str, float | int]] = []
    previous_step = 0
    for entry in sorted(paper_history, key=lambda item: int(item["step"])):
        checkpoint_step = int(entry["step"])
        checkpoint_slice = attribution_frame.filter(
            (pl.col("step") > previous_step) & (pl.col("step") <= checkpoint_step)
        )
        previous_step = checkpoint_step
        if checkpoint_slice.is_empty():
            continue
        row: dict[str, float | int] = {
            "step": checkpoint_step,
            "latent_edge_f1": float(entry.get("latent_edge_f1")) if entry.get("latent_edge_f1") is not None else np.nan,
            "latent_node_ratio": (
                float(entry.get("latent_inferred_state_count")) / float(entry.get("latent_ground_truth_state_count"))
                if entry.get("latent_inferred_state_count") is not None
                and entry.get("latent_ground_truth_state_count") not in {None, 0}
                else np.nan
            ),
        }
        for column_name in _WEIGHTED_COMPONENT_COLUMNS:
            row[column_name] = float(checkpoint_slice.get_column(column_name).mean())
        checkpoint_rows.append(row)
    return pl.DataFrame(checkpoint_rows) if checkpoint_rows else pl.DataFrame()


def _plot_vs_training_step(attribution_frame: pl.DataFrame, output_path: Path) -> None:
    figure, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    for column_name in _WEIGHTED_COMPONENT_COLUMNS:
        axes[0].plot(
            attribution_frame.get_column("step").to_numpy(),
            attribution_frame.get_column(column_name).to_numpy(),
            label=_COMPONENT_LABELS[column_name],
            linewidth=1.5,
        )
    axes[0].set_ylabel("weighted contribution")
    axes[0].set_title("BMU score attribution vs training step")
    axes[0].grid(alpha=0.25)

    for column_name in _SHARE_COMPONENT_COLUMNS:
        axes[1].plot(
            attribution_frame.get_column("step").to_numpy(),
            attribution_frame.get_column(column_name).to_numpy(),
            label=_COMPONENT_LABELS[column_name],
            linewidth=1.5,
        )
    axes[1].set_xlabel("training step")
    axes[1].set_ylabel("normalized contribution")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].grid(alpha=0.25)
    axes[1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    _save_figure(figure, output_path)


def _plot_component_relation(
    checkpoint_frame: pl.DataFrame,
    *,
    x_metric: str,
    x_label: str,
    output_path: Path,
) -> None:
    if checkpoint_frame.is_empty():
        return
    figure, axis = plt.subplots(figsize=(9, 6))
    x_values = checkpoint_frame.get_column(x_metric).to_numpy()
    for column_name in _WEIGHTED_COMPONENT_COLUMNS:
        y_values = checkpoint_frame.get_column(column_name).to_numpy()
        valid_mask = np.isfinite(x_values) & np.isfinite(y_values)
        if not np.any(valid_mask):
            continue
        axis.scatter(
            x_values[valid_mask],
            y_values[valid_mask],
            label=_COMPONENT_LABELS[column_name],
            alpha=0.8,
        )
        axis.plot(
            x_values[valid_mask],
            y_values[valid_mask],
            alpha=0.4,
            linewidth=1.0,
        )
    axis.set_xlabel(x_label)
    axis.set_ylabel("average weighted contribution")
    axis.grid(alpha=0.25)
    axis.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    _save_figure(figure, output_path)


def generate_bmu_attribution_plots(*, layout: ArtifactLayout) -> list[Path]:
    if not layout.mercury_attribution_parquet.exists():
        return []

    attribution_frame = pl.read_parquet(layout.mercury_attribution_parquet).sort("step")
    if attribution_frame.is_empty():
        return []

    generated_paths: list[Path] = []
    _plot_vs_training_step(attribution_frame, layout.mercury_bmu_attribution_step_png)
    generated_paths.append(layout.mercury_bmu_attribution_step_png)

    paper_history = _load_paper_precision_history(layout.mercury_paper_precision_json)
    checkpoint_frame = _checkpoint_summary_frame(attribution_frame, paper_history)
    if checkpoint_frame.is_empty():
        return generated_paths

    _plot_component_relation(
        checkpoint_frame,
        x_metric="latent_edge_f1",
        x_label="latent Edge F1",
        output_path=layout.mercury_bmu_attribution_edge_f1_png,
    )
    generated_paths.append(layout.mercury_bmu_attribution_edge_f1_png)

    _plot_component_relation(
        checkpoint_frame,
        x_metric="latent_node_ratio",
        x_label="latent node count / ground-truth state count",
        output_path=layout.mercury_bmu_attribution_node_ratio_png,
    )
    generated_paths.append(layout.mercury_bmu_attribution_node_ratio_png)
    return generated_paths
