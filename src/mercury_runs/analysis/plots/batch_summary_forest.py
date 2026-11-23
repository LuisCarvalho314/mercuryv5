"""Forest / matrix / profile comparison plots for batch-40 studies.

Canonical: consumes evaluation records produced by `evaluate_study`.
Must not load model snapshots, training datasets, or legacy CSV metrics.

Usage:
    python -m mercury_runs.analysis.plots.batch_summary_forest \\
        --study_root results/current/studies/<study> \\
        --protocol_id canonical-v4-<hash>
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import textwrap
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import numpy as np
import polars as pl
import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib import patches
from matplotlib.lines import Line2D
from scipy.stats import t

from mercury_runs.analysis.evaluation_records import (
    load_evaluation_step,
    load_study_manifest,
)
from mercury_runs.analysis.plots import publication_style as pub

DEFAULT_STUDY_ROOT = Path(
    "results/current/studies/mercury_cscg_hybrid_vs_cscg_batch_size_30_seed"
)
DEFAULT_PLOT_DIR = Path(
    "results/current/notebook_plots/mercury_cscg_hybrid_vs_cscg_batch_size_30_seed"
)

CANONICAL_LEVEL_ORDER = (13, 21, 20, 22)
LEVEL_LABELS = {13: "3x3", 21: "1x4", 20: "1x5", 22: "3x4"}

# PLOT_LEVELS selects which environments to plot; display order always follows
# CANONICAL_LEVEL_ORDER so figures stay comparable across studies.
_requested_levels = [int(item.strip()) for item in os.environ.get("PLOT_LEVELS", "").split(",") if item.strip()]
LEVEL_ORDER = [level for level in CANONICAL_LEVEL_ORDER if not _requested_levels or level in _requested_levels]
LEVEL_ORDER += [level for level in _requested_levels if level not in CANONICAL_LEVEL_ORDER]
HYBRID_SERIES_ORDER = [
    "hybrid",
    "tmgwr_hybrid",
    *pub.CSCG_VARIANT_ORDER,
    "ground_truth_state_1",
    "ground_truth_state_2",
    "uniform_obs_4",
    "uniform_obs_8",
]
MERCURY_SERIES_ORDER = [
    "mercury",
    "tmgwr_hybrid",
    *pub.CSCG_VARIANT_ORDER,
    "ground_truth_state_1",
    "ground_truth_state_2",
    "uniform_obs_4",
    "uniform_obs_8",
]
SERIES_LABELS = {
    name: pub.method_label(name)
    for name in MERCURY_SERIES_ORDER + HYBRID_SERIES_ORDER
}


def _series_labels_for_comparison(comparison: str) -> dict[str, str]:
    return {name: pub.method_label(name, comparison=comparison) for name in SERIES_LABELS}


def _series_colors_for_comparison(comparison: str) -> dict[str, str]:
    return {name: pub.method_color(name, comparison=comparison) for name in SERIES_COLORS}


def _series_label(name: str, series_labels: dict[str, str] | None = None) -> str:
    return (series_labels or SERIES_LABELS)[name]


def _series_color(name: str, series_colors: dict[str, str] | None = None) -> str:
    return (series_colors or SERIES_COLORS)[name]


SERIES_COLORS = {
    name: pub.method_color(name)
    for name in MERCURY_SERIES_ORDER + HYBRID_SERIES_ORDER
}
METRICS = [
    ("latent_precision", pub.metric_label("latent_precision")),
    ("latent_recall", pub.metric_label("latent_recall")),
    ("latent_mean_total_variation_similarity", "Policy-collapsed TV similarity"),
    ("latent_non_self_loop_total_variation_similarity", "Policy-collapsed no-self-loop TV similarity"),
    ("latent_action_conditioned_mean_total_variation_similarity", "Action-conditioned TV similarity"),
    ("latent_action_conditioned_non_self_loop_total_variation_similarity", "Action-conditioned no-self-loop TV similarity"),
]
PROFILE_FOCUSED_METRICS = [
    "latent_precision",
    "latent_recall",
    "latent_action_conditioned_mean_total_variation_similarity",
    "latent_action_conditioned_non_self_loop_total_variation_similarity",
]
CSCG_NLL_METRIC = "cscg_eval_nll_bps"
MATRIX_METRIC_LABELS = {
    "latent_precision": pub.metric_label("latent_precision").replace(" ", "\n"),
    "latent_recall": pub.metric_label("latent_recall").replace(" ", "\n"),
    "latent_mean_total_variation_similarity": "Policy-collapsed\nTV similarity",
    "latent_non_self_loop_total_variation_similarity": "Policy-collapsed\nno-self-loop TV",
    "latent_action_conditioned_mean_total_variation_similarity": "Action-conditioned\nTV similarity",
    "latent_action_conditioned_non_self_loop_total_variation_similarity": "Action-conditioned\nno-self-loop TV",
}
PROFILE_METRIC_COLORS = plt.get_cmap("tab10").colors
PROFILE_METRIC_STYLES = {
    "latent_precision": (pub.metric_label("latent_precision"), PROFILE_METRIC_COLORS[0], "o", 0.45),
    "latent_recall": (pub.metric_label("latent_recall"), PROFILE_METRIC_COLORS[1], "o", 0.27),
    "latent_mean_total_variation_similarity": ("Policy-collapsed TV similarity", PROFILE_METRIC_COLORS[2], "o", 0.09),
    "latent_non_self_loop_total_variation_similarity": ("Policy-collapsed no-self-loop TV similarity", PROFILE_METRIC_COLORS[3], "o", -0.09),
    "latent_action_conditioned_mean_total_variation_similarity": ("Action-conditioned TV similarity", PROFILE_METRIC_COLORS[4], "o", -0.27),
    "latent_action_conditioned_non_self_loop_total_variation_similarity": ("Action-conditioned no-self-loop TV similarity", PROFILE_METRIC_COLORS[5], "o", -0.45),
}
PROFILE_ROW_GAP = 1.35

COMPACT_TRANSPOSED_METRIC_LABELS = {
    "latent_precision": r"$P_Z$",
    "latent_recall": r"$C_Z$",
    "latent_action_conditioned_mean_total_variation_similarity": r"$S_{\mathrm{TV}}^A$",
    "latent_action_conditioned_non_self_loop_total_variation_similarity": r"$S_{\mathrm{TV}}^{A,\neg\mathrm{self}}$",
    CSCG_NLL_METRIC: "CSCG eval NLL",
}

COMPACT_TRANSPOSED_METHOD_LABELS = {
    "mercury": "MERCURY",
    "hybrid": "MERCURY",
    "tmgwr_hybrid": "TMGWR",
    "vanilla_cscg": "CSCG",
    "oracle_capacity": "Oracle-C",
    "oracle_capacity_first_order": "Oracle-C+1st",
    "oracle_capacity_structural": "Oracle-C+Struct",
}


def _wrap_row_label(label: str, width: int = 18) -> str:
    """Wrap a metric name so it fits inside one forest row's height."""
    return "\n".join(textwrap.wrap(str(label), width=width)) or str(label)


def _profile_metric_items(metric_keys: list[str] | None) -> list[tuple[str, str, object, str, float]]:
    metrics = [(metric, label) for metric, label in METRICS if metric_keys is None or metric in metric_keys]
    max_offset = 0.45 if len(metrics) >= 6 else 0.34
    offsets = np.linspace(max_offset, -max_offset, len(metrics))
    items = []
    for (metric, _), offset in zip(metrics, offsets):
        label, color, marker, _ = PROFILE_METRIC_STYLES[metric]
        items.append((metric, label, color, marker, float(offset)))
    return items


def _profile_legend_columns(metric_count: int) -> int:
    return 3 if metric_count > 4 else 2


def _transposed_method_offsets(method_count: int) -> np.ndarray:
    if method_count <= 1:
        return np.asarray([0.0])
    return np.linspace(0.30, -0.30, method_count)


def _transposed_method_legend_columns(method_count: int) -> int:
    # Keep the method key to two compact rows at single-column width.
    return max(1, (method_count + 1) // 2)


def _compact_transposed_method_label(
    name: str,
    series_labels: dict[str, str] | None = None,
) -> str:
    return COMPACT_TRANSPOSED_METHOD_LABELS.get(
        name,
        _series_label(name, series_labels),
    )


def _transposed_row_specs(
    metric_keys: list[str] | None,
    *,
    include_nll_axis: bool,
) -> list[tuple[str, str]]:
    rows = [(metric, label) for metric, label, _, _, _ in _profile_metric_items(metric_keys)]
    if include_nll_axis:
        rows.append((CSCG_NLL_METRIC, "CSCG eval NLL"))
    return rows

# Preset identifier → plot_series name
_PRESET_TO_SERIES = {
    "vanilla_cscg": "vanilla_cscg",
    "oracle_capacity": "oracle_capacity",
    "oracle_capacity_structural": "oracle_capacity_structural",
    "oracle_capacity_first_order": "oracle_capacity_first_order",
    "ground_truth_state_1": "ground_truth_state_1",
    "ground_truth_state_2": "ground_truth_state_2",
    "uniform_obs_4": "uniform_obs_4",
    "uniform_obs_8": "uniform_obs_8",
}
_SELECTED_CSCG_PRESETS = [
    item.strip()
    for item in os.environ.get("PLOT_CSCG_PRESETS", "").split(",")
    if item.strip()
]
if _SELECTED_CSCG_PRESETS:
    _selected_series = {_PRESET_TO_SERIES[preset] for preset in _SELECTED_CSCG_PRESETS if preset in _PRESET_TO_SERIES}
    HYBRID_SERIES_ORDER = [
        series
        for series in HYBRID_SERIES_ORDER
        if series in {"hybrid", "tmgwr_hybrid"} or series in _selected_series
    ]
    MERCURY_SERIES_ORDER = [
        series
        for series in MERCURY_SERIES_ORDER
        if series in {"mercury", "tmgwr_hybrid"} or series in _selected_series
    ]
# Comparison → nominal cumulative training exposure (None = last snapshot).
_COMPARISON_TO_STEP: dict[str, int | None] = {
    "final": None,
    "checkpoint_4000": 4000,
    "checkpoint_2000": 2000,
}


def _rand_slug(rand_prob: float) -> str:
    return format(float(rand_prob), "g").replace("-", "m").replace(".", "p")


def _transposed_output_name(output_name: str) -> str:
    path = Path(output_name)
    return f"{path.stem}_transposed{path.suffix}"


def _compact_transposed_output_name(output_name: str) -> str:
    path = Path(output_name)
    return f"{path.stem}_compact{path.suffix}"


def _record_lookup_for_comparison(method: str, comparison: str, nominal_step: int | None) -> tuple[int | None, str | None]:
    """Resolve plot comparison labels to the step IDs stored in evaluation records."""
    if method == "mercury_cscg_hybrid" and comparison == "checkpoint_2000":
        return 2000, "mercury_cscg_hybrid_mercury"
    if method == "mercury_cscg_hybrid" and comparison == "checkpoint_4000":
        # Hybrid CSCG evaluation records use the CSCG phase step. The figure
        # label is cumulative exposure: 2000 Mercury + 2000 CSCG = 4000.
        return 2000, "mercury_cscg_hybrid"
    if method == "tmgwr_cscg_hybrid" and comparison == "checkpoint_2000":
        return 2000, "tmgwr_cscg_hybrid_tmgwr"
    if method == "tmgwr_cscg_hybrid" and comparison == "checkpoint_4000":
        # Same phase-step convention as the Mercury hybrid: the CSCG-phase
        # records restart at 0, so cumulative 4000 = 2000 TMGWR + 2000 CSCG.
        return 2000, "tmgwr_cscg_hybrid"
    if method == "tmgwr_cscg_hybrid":
        return nominal_step, "tmgwr_cscg_hybrid"
    return nominal_step, str(method)


def _metrics_from_record(record: dict) -> dict[str, float]:
    """Extract canonical scalar metrics from an evaluation record."""
    policy = record["policy_collapsed"]
    action = record["action_conditioned"]
    collapsed_tv = float(policy["tv_occupancy"])
    collapsed_nonsl_tv = float(policy.get("tv_non_self_loop_occupancy", collapsed_tv))
    action_tv = float(action["tv_occupancy"])
    action_nonsl_tv = float(action.get("tv_non_self_loop_occupancy", action_tv))
    precision = float(record["state_representation"]["latent_precision"])
    recall = float(
        record["state_representation"].get(
            "latent_recall",
            record["state_representation"]["node_recall"],
        )
    )
    metrics = {
        "latent_precision": precision,
        "latent_recall": recall,
        "latent_mean_total_variation": collapsed_tv,
        "latent_non_self_loop_total_variation": collapsed_nonsl_tv,
        "latent_action_conditioned_mean_total_variation": action_tv,
        "latent_action_conditioned_non_self_loop_total_variation": action_nonsl_tv,
        "latent_mean_total_variation_similarity": 1.0 - collapsed_tv,
        "latent_non_self_loop_total_variation_similarity": 1.0 - collapsed_nonsl_tv,
        "latent_action_conditioned_mean_total_variation_similarity": 1.0 - action_tv,
        "latent_action_conditioned_non_self_loop_total_variation_similarity": 1.0 - action_nonsl_tv,
    }
    model_fit = record.get("model_fit")
    if isinstance(model_fit, dict):
        nll = model_fit.get("negative_log2_likelihood_mean", model_fit.get("bits_per_symbol"))
        if nll is not None:
            metrics[CSCG_NLL_METRIC] = float(nll)
    return metrics


def _history_path(run_root: Path, method: str) -> Path:
    return Path(run_root) / "metrics" / str(method) / f"{Path(run_root).name}_paper_precision.json"


def _load_hybrid_mercury_handoff_metrics(run_root: Path) -> dict[str, float] | None:
    """Load scalar Mercury-bootstrap metrics at the 2000-step handoff.

    Hybrid runs are trained as Mercury for 2000 steps, then CSCG for 2000
    steps. The Mercury handoff series is stored in the hybrid paper-precision
    history, not as a separate canonical evaluation snapshot.
    """
    path = _history_path(run_root, "mercury_cscg_hybrid")
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    candidates = [
        row for row in payload.get("history", [])
        if isinstance(row, dict)
        and str(row.get("method")) == "mercury_cscg_hybrid_mercury"
        and int(row.get("cumulative_step", row.get("step", -1))) == 2000
    ]
    if not candidates:
        return None
    row = candidates[-1]
    try:
        collapsed_tv = float(row["latent_mean_total_variation"])
        action_tv = float(row["latent_action_conditioned_mean_total_variation"])
        precision = float(row["latent_precision"])
        recall = float(row["latent_recall"])
    except Exception:
        return None
    return {
        "latent_precision": precision,
        "latent_recall": recall,
        "latent_mean_total_variation": collapsed_tv,
        "latent_non_self_loop_total_variation": None,
        "latent_action_conditioned_mean_total_variation": action_tv,
        "latent_action_conditioned_non_self_loop_total_variation": None,
        "latent_mean_total_variation_similarity": 1.0 - collapsed_tv,
        "latent_non_self_loop_total_variation_similarity": None,
        "latent_action_conditioned_mean_total_variation_similarity": 1.0 - action_tv,
        "latent_action_conditioned_non_self_loop_total_variation_similarity": None,
    }


def _parse_protocol_ids_by_rand(raw: str | None) -> dict[float, str]:
    if not raw:
        return {}
    mapping: dict[float, str] = {}
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        rand_text, protocol_id = item.split("=", 1)
        mapping[float(rand_text)] = protocol_id.strip()
    return mapping


def _build_frame(
    study_root: Path,
    protocol_id: str,
    *,
    protocol_ids_by_rand: dict[float, str] | None = None,
    batch_size: int = 40,
    levels: list[int] | None = None,
) -> pl.DataFrame:
    """Load evaluation records for all runs and return a tidy DataFrame."""
    levels = levels or LEVEL_ORDER
    manifest = load_study_manifest(study_root)
    rows = []
    for (method, level, seed, rand_prob, bs, preset), run_root in manifest.items():
        if int(bs) != batch_size:
            continue
        if int(level) not in levels:
            continue
        row_protocol_id = (protocol_ids_by_rand or {}).get(float(rand_prob), protocol_id)
        for comparison, nominal_step in _COMPARISON_TO_STEP.items():
            if method == "mercury_cscg_hybrid" and preset is None:
                if comparison == "checkpoint_2000":
                    step, eval_method = _record_lookup_for_comparison(str(method), str(comparison), nominal_step)
                    record = load_evaluation_step(run_root, row_protocol_id, step, method=eval_method)
                    metrics = _metrics_from_record(record) if record is not None else None
                    if metrics is None:
                        metrics = _load_hybrid_mercury_handoff_metrics(run_root)
                        if metrics is None:
                            continue
                    plot_series = "mercury"
                else:
                    step, eval_method = _record_lookup_for_comparison(str(method), str(comparison), nominal_step)
                    record = load_evaluation_step(run_root, row_protocol_id, step, method=eval_method)
                    if record is None:
                        continue
                    metrics = _metrics_from_record(record)
                    plot_series = "hybrid"
            elif method == "tmgwr_cscg_hybrid" and preset is None:
                step, eval_method = _record_lookup_for_comparison(str(method), str(comparison), nominal_step)
                record = load_evaluation_step(run_root, row_protocol_id, step, method=eval_method)
                if record is None:
                    continue
                metrics = _metrics_from_record(record)
                plot_series = "tmgwr_hybrid"
            elif method == "cscg" and preset in _PRESET_TO_SERIES:
                step, eval_method = _record_lookup_for_comparison(str(method), str(comparison), nominal_step)
                record = load_evaluation_step(run_root, row_protocol_id, step, method=eval_method)
                if record is None:
                    continue
                metrics = _metrics_from_record(record)
                plot_series = _PRESET_TO_SERIES[preset]
            else:
                continue
            rows.append(
                {
                    "comparison": comparison,
                    "level": int(level),
                    "seed": int(seed),
                    "rand_prob": float(rand_prob),
                    "cscg_batch_size": int(bs),
                    "plot_series": plot_series,
                    **metrics,
                }
            )
    return pl.DataFrame(rows, infer_schema_length=None) if rows else pl.DataFrame()


def _ci95_half_width(n: int, sd: float | None) -> float:
    if n <= 1 or sd is None or not np.isfinite(sd):
        return 0.0
    return float(t.ppf(0.975, n - 1) * sd / np.sqrt(n))


def _load_source(frame: pl.DataFrame, comparison: str, series_order: list[str]) -> pl.DataFrame:
    return (
        frame.filter(
            (pl.col("comparison") == comparison)
            & pl.col("level").is_in(LEVEL_ORDER)
            & pl.col("plot_series").is_in(series_order)
        )
    )


def _available_series_order(source: pl.DataFrame, series_order: list[str]) -> list[str]:
    available = set(source.get_column("plot_series").to_list())
    return [series for series in series_order if series in available]


def _summarise(frame: pl.DataFrame, metric: str) -> pl.DataFrame:
    return (
        frame.drop_nulls(["level", "plot_series", "seed", metric])
        .group_by(["level", "plot_series", "seed"])
        .agg(pl.col(metric).mean().alias("value"))
        .group_by(["level", "plot_series"])
        .agg(
            pl.len().alias("n"),
            pl.col("value").mean().alias("mean"),
            pl.col("value").std().fill_null(0.0).alias("sd"),
        )
    )


def _format_mean_ci(mean: float, half_width: float) -> str:
    lower = max(0.0, mean - half_width)
    upper = min(1.0, mean + half_width)

    def compact(value: float) -> str:
        return f"{value:.2f}".replace("0.", ".")

    return f"{compact(mean)} [{compact(lower)}-{compact(upper)}]"


def _summary_lookup(frame: pl.DataFrame) -> dict[tuple[int, str, str], tuple[float, float, float]]:
    lookup: dict[tuple[int, str, str], tuple[float, float, float]] = {}
    for metric, _ in METRICS:
        summary = _summarise(frame, metric)
        for row in summary.to_dicts():
            n = int(row["n"])
            mean = float(row["mean"])
            half_width = _ci95_half_width(n, float(row["sd"]))
            lookup[(int(row["level"]), str(row["plot_series"]), metric)] = (mean, half_width, float(row["sd"]))
    return lookup


def _summary_lookup_by_rand(frame: pl.DataFrame) -> dict[tuple[int, str, float, str], tuple[float, float, float]]:
    lookup: dict[tuple[int, str, float, str], tuple[float, float, float]] = {}
    for metric, _ in METRICS:
        summary = (
            frame.drop_nulls(["level", "plot_series", "rand_prob", "seed", metric])
            .group_by(["level", "plot_series", "rand_prob", "seed"])
            .agg(pl.col(metric).mean().alias("value"))
            .group_by(["level", "plot_series", "rand_prob"])
            .agg(
                pl.len().alias("n"),
                pl.col("value").mean().alias("mean"),
                pl.col("value").std().fill_null(0.0).alias("sd"),
            )
        )
        for row in summary.to_dicts():
            n = int(row["n"])
            mean = float(row["mean"])
            half_width = _ci95_half_width(n, float(row["sd"]))
            lookup[(int(row["level"]), str(row["plot_series"]), float(row["rand_prob"]), metric)] = (
                mean, half_width, float(row["sd"]),
            )
    return lookup


def _nll_axis_limits(values: list[float]) -> tuple[float, float]:
    finite = [float(value) for value in values if np.isfinite(float(value))]
    if not finite:
        return 0.0, 1.0
    low = min(finite)
    high = max(finite)
    if np.isclose(low, high):
        pad = max(0.05, abs(low) * 0.05)
    else:
        pad = 0.08 * (high - low)
    return max(0.0, low - pad), high + pad


def _nll_lookup(frame: pl.DataFrame) -> dict[tuple[int, str], tuple[float, float, float]]:
    if CSCG_NLL_METRIC not in frame.columns:
        return {}
    lookup: dict[tuple[int, str], tuple[float, float, float]] = {}
    summary = (
        frame.drop_nulls(["level", "plot_series", "seed", CSCG_NLL_METRIC])
        .group_by(["level", "plot_series", "seed"])
        .agg(pl.col(CSCG_NLL_METRIC).mean().alias("value"))
        .group_by(["level", "plot_series"])
        .agg(
            pl.len().alias("n"),
            pl.col("value").mean().alias("mean"),
            pl.col("value").std().fill_null(0.0).alias("sd"),
        )
    )
    for row in summary.to_dicts():
        n = int(row["n"])
        mean = float(row["mean"])
        half_width = _ci95_half_width(n, float(row["sd"]))
        lookup[(int(row["level"]), str(row["plot_series"]))] = (mean, half_width, float(row["sd"]))
    return lookup


def _nll_lookup_by_rand(frame: pl.DataFrame) -> dict[tuple[int, str, float], tuple[float, float, float]]:
    if CSCG_NLL_METRIC not in frame.columns:
        return {}
    lookup: dict[tuple[int, str, float], tuple[float, float, float]] = {}
    summary = (
        frame.drop_nulls(["level", "plot_series", "rand_prob", "seed", CSCG_NLL_METRIC])
        .group_by(["level", "plot_series", "rand_prob", "seed"])
        .agg(pl.col(CSCG_NLL_METRIC).mean().alias("value"))
        .group_by(["level", "plot_series", "rand_prob"])
        .agg(
            pl.len().alias("n"),
            pl.col("value").mean().alias("mean"),
            pl.col("value").std().fill_null(0.0).alias("sd"),
        )
    )
    for row in summary.to_dicts():
        n = int(row["n"])
        mean = float(row["mean"])
        half_width = _ci95_half_width(n, float(row["sd"]))
        lookup[(int(row["level"]), str(row["plot_series"]), float(row["rand_prob"]))] = (
            mean,
            half_width,
            float(row["sd"]),
        )
    return lookup


def plot_metric_matrix(
    frame: pl.DataFrame,
    *,
    series_order: list[str],
    title: str,
    output_name: str,
    gain_label: str | None,
    plot_dir: Path,
    series_labels: dict[str, str] | None = None,
    series_colors: dict[str, str] | None = None,
    emphasize_first: bool = True,
    show_gain: bool = True,
) -> Path:
    plt.style.use("default")
    plt.rcParams.update({"font.family": "DejaVu Sans", "figure.facecolor": "white", "savefig.facecolor": "white"})
    lookup = _summary_lookup(frame)
    column_specs = [(level, metric) for level in LEVEL_ORDER for metric, _ in METRICS]
    values = np.full((len(series_order), len(column_specs)), np.nan, dtype=float)
    text = [["" for _ in column_specs] for _ in series_order]
    for row_index, series_name in enumerate(series_order):
        for col_index, (level, metric) in enumerate(column_specs):
            result = lookup.get((level, series_name, metric))
            if result is None:
                continue
            mean, half_width, _ = result
            values[row_index, col_index] = mean
            text[row_index][col_index] = _format_mean_ci(mean, half_width)

    gains = []
    for level, metric in column_specs:
        first_mean = lookup.get((level, series_order[0], metric), (np.nan, 0.0, 0.0))[0]
        preset_means = [lookup.get((level, s, metric), (np.nan, 0.0, 0.0))[0] for s in series_order[1:]]
        preset_max = float(np.nanmax(preset_means)) if np.any(np.isfinite(preset_means)) else np.nan
        gains.append(first_mean - preset_max if np.isfinite(first_mean) and np.isfinite(preset_max) else np.nan)

    figure, axis = plt.subplots(figsize=(13.2, 4.9), dpi=300)
    masked = np.ma.masked_invalid(values)
    image = axis.imshow(masked, cmap="YlGnBu", vmin=0.0, vmax=1.0, aspect="auto")
    image.cmap.set_bad("#F5F7FA")

    axis.set_xticks(np.arange(len(column_specs)))
    axis.set_xticklabels([MATRIX_METRIC_LABELS[metric] for _, metric in column_specs], fontsize=8.4)
    display_rows = len(series_order) + (1 if show_gain else 0)
    axis.set_yticks(np.arange(display_rows))
    y_labels = [_series_label(name, series_labels) for name in series_order]
    if show_gain:
        y_labels.append(str(gain_label or "Gain"))
    axis.set_yticklabels(y_labels, fontsize=9.0)
    axis.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False, length=0)
    axis.set_ylim(display_rows - 0.5, -0.5)

    for level_index, level in enumerate(LEVEL_ORDER):
        center = level_index * len(METRICS) + (len(METRICS) - 1) / 2
        axis.text(center, -1.08, LEVEL_LABELS[level], ha="center", va="center", fontsize=10.5, fontweight="bold", clip_on=False)
        if level_index > 0:
            axis.axvline(level_index * len(METRICS) - 0.5, color="white", linewidth=3.0)
            axis.axvline(level_index * len(METRICS) - 0.5, color="#2F3740", linewidth=0.8)

    for row_index in range(len(series_order)):
        for col_index in range(len(column_specs)):
            value = values[row_index, col_index]
            if not np.isfinite(value):
                continue
            is_best = bool(np.isclose(value, np.nanmax(values[:, col_index]), atol=1e-12))
            color = "white" if value >= 0.66 else "#1F2933"
            axis.text(col_index, row_index, text[row_index][col_index], ha="center", va="center", fontsize=7.25, color=color, fontweight="bold" if is_best else "normal")
            if is_best:
                axis.add_patch(patches.Rectangle((col_index - 0.5, row_index - 0.5), 1.0, 1.0, fill=False, edgecolor="#111827", linewidth=1.7))

    if emphasize_first:
        first_row_color = _series_color(series_order[0], series_colors)
        axis.add_patch(patches.Rectangle((-0.5, -0.5), len(column_specs), 1.0, fill=False, edgecolor=first_row_color, linewidth=2.6))

    if show_gain:
        gain_row = len(series_order)
        axis.axhline(gain_row - 0.5, color="#2F3740", linewidth=1.0)
        for col_index, gain in enumerate(gains):
            axis.add_patch(patches.Rectangle((col_index - 0.5, gain_row - 0.5), 1.0, 1.0, facecolor="#F7FAFC", edgecolor="white", linewidth=1.0))
            if np.isfinite(gain):
                color = "#006D2C" if gain >= 0 else "#9F1239"
                axis.text(col_index, gain_row, f"{gain:+.2f}".replace("+0.", "+.").replace("-0.", "-."), ha="center", va="center", fontsize=8.2, color=color, fontweight="bold")

    axis.set_xticks(np.arange(-0.5, len(column_specs), 1), minor=True)
    axis.set_yticks(np.arange(-0.5, display_rows, 1), minor=True)
    axis.grid(which="minor", color="white", linewidth=1.0)
    axis.tick_params(which="minor", bottom=False, left=False)
    for spine in axis.spines.values():
        spine.set_visible(False)

    colorbar = figure.colorbar(image, ax=axis, fraction=0.022, pad=0.012)
    colorbar.set_label("Mean value", fontsize=8.5)
    colorbar.ax.tick_params(labelsize=7.8)
    figure.suptitle(title, fontsize=12.5, y=0.985)
    figure.subplots_adjust(left=0.115, right=0.955, top=0.76, bottom=0.085)
    png_path = plot_dir / output_name
    figure.savefig(png_path, bbox_inches="tight")
    figure.savefig(png_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)
    return png_path


def plot_metric_profile_by_rand(
    frame: pl.DataFrame,
    *,
    series_order: list[str],
    title: str,
    output_name: str,
    plot_dir: Path,
    metric_keys: list[str] | None = None,
    include_nll_axis: bool = False,
    series_labels: dict[str, str] | None = None,
    series_colors: dict[str, str] | None = None,
) -> Path:
    plt.style.use("default")
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.edgecolor": "#2F3740", "axes.labelcolor": "#1F2933", "xtick.color": "#1F2933", "ytick.color": "#1F2933", "figure.facecolor": "white", "savefig.facecolor": "white"})
    rand_order = [0.0, 0.3]
    row_specs = [(series_name, rand_prob) for series_name in series_order for rand_prob in rand_order]
    lookup = _summary_lookup_by_rand(frame)
    nll_lookup = _nll_lookup_by_rand(frame) if include_nll_axis else {}
    profile_items = _profile_metric_items(metric_keys)
    y_positions = np.arange(len(row_specs))[::-1] * PROFILE_ROW_GAP
    figure, axes = plt.subplots(nrows=1, ncols=len(LEVEL_ORDER), figsize=(12.2, 8.8), sharex=True, sharey=True, dpi=300, gridspec_kw={"wspace": 0.08})
    axes = np.ravel(axes)

    for col_index, (axis, level) in enumerate(zip(axes, LEVEL_ORDER)):
        nll_axis = None
        level_nll_values = [
            value[0]
            for key, value in nll_lookup.items()
            if key[0] == level
        ]
        if level_nll_values:
            nll_axis = axis.twiny()
            nll_axis.set_xlim(*_nll_axis_limits(level_nll_values))
            nll_axis.tick_params(axis="x", labelsize=7.2, colors="#111827", length=2.5, pad=1.5)
            nll_axis.spines["top"].set_color("#111827")
            nll_axis.spines["bottom"].set_visible(False)
            nll_axis.spines["left"].set_visible(False)
            nll_axis.spines["right"].set_visible(False)
            if col_index == len(LEVEL_ORDER) - 1:
                nll_axis.set_xlabel("CSCG eval NLL (bits/symbol)", fontsize=8.2, color="#111827", labelpad=4)
            else:
                nll_axis.set_xlabel("")
        axis.axhspan(y_positions[1] - 0.58, y_positions[0] + 0.58, color=_series_color(series_order[0], series_colors), alpha=0.075, zorder=0)
        for row_index, (series_name, rand_prob) in enumerate(row_specs):
            base_y = y_positions[row_index]
            if row_index > 0 and row_index % len(rand_order) == 0:
                axis.axhline(base_y + PROFILE_ROW_GAP / 2, color="#CBD5E1", linewidth=0.8, zorder=0)
            for metric, label, color, marker, offset in profile_items:
                seed_values = (
                    frame.filter((pl.col("level") == level) & (pl.col("plot_series") == series_name) & (pl.col("rand_prob") == rand_prob))
                    .drop_nulls(["seed", metric])
                    .group_by("seed")
                    .agg(pl.col(metric).mean().alias("value"))
                    .sort("seed")
                )
                for seed_index, value in enumerate(seed_values["value"]):
                    jitter = ((seed_index % 5) - 2) * 0.018
                    axis.scatter(float(value), base_y + offset + jitter, s=10, color=color, alpha=0.18, linewidths=0, zorder=1)
                result = lookup.get((level, series_name, rand_prob, metric))
                if result is None:
                    continue
                mean, half_width, _ = result
                axis.errorbar(mean, base_y + offset, xerr=half_width, fmt=marker, color=color, ecolor=color, elinewidth=1.9 if row_index < len(rand_order) else 1.4, capsize=3.2, capthick=1.9 if row_index < len(rand_order) else 1.4, markersize=4.4 if row_index < len(rand_order) else 3.8, markeredgecolor="#1F2933", markeredgewidth=0.25, label=label if col_index == 0 and row_index == 0 else None, zorder=3)
            if nll_axis is not None:
                nll_result = nll_lookup.get((level, series_name, rand_prob))
                if nll_result is not None:
                    seed_values = (
                        frame.filter((pl.col("level") == level) & (pl.col("plot_series") == series_name) & (pl.col("rand_prob") == rand_prob))
                        .drop_nulls(["seed", CSCG_NLL_METRIC])
                        .group_by("seed")
                        .agg(pl.col(CSCG_NLL_METRIC).mean().alias("value"))
                        .sort("seed")
                    )
                    for seed_index, value in enumerate(seed_values["value"]):
                        jitter = ((seed_index % 5) - 2) * 0.018
                        nll_axis.scatter(float(value), base_y - 0.52 + jitter, s=13, color="#111827", alpha=0.14, marker="D", linewidths=0, zorder=2)
                    mean, half_width, _ = nll_result
                    nll_axis.errorbar(mean, base_y - 0.52, xerr=half_width, fmt="D", color="#111827", ecolor="#111827", elinewidth=1.65, capsize=3.0, capthick=1.65, markersize=4.2, markeredgecolor="white", markeredgewidth=0.25, label="CSCG eval NLL" if col_index == 0 and row_index == 0 else None, zorder=4)
        axis.set_title(LEVEL_LABELS[level], fontsize=11.0, fontweight="bold", pad=8)
        axis.set_xlim(0.0, 1.0)
        axis.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
        axis.set_xticklabels(["0", "0.25", "0.50", "0.75", "1.00"])
        axis.set_ylim(-0.8, y_positions[0] + 0.8)
        axis.set_axisbelow(True)
        axis.grid(axis="x", which="major", color="#D8DEE6", linewidth=0.8, alpha=0.7)
        axis.tick_params(axis="both", labelsize=8.6)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.spines["left"].set_visible(col_index == 0)
        if col_index == 0:
            axis.set_yticks(y_positions, [f"{_series_label(series_name, series_labels)}  persistence {rand_prob:.1f}" for series_name, rand_prob in row_specs])
            for index, tick_label in enumerate(axis.get_yticklabels()):
                if index < len(rand_order):
                    tick_label.set_fontweight("bold")
            axis.tick_params(axis="y", labelsize=8.1, pad=5)
        else:
            axis.tick_params(axis="y", labelleft=False, left=False)

    handles, labels = axes[0].get_legend_handles_labels()
    if include_nll_axis:
        for axis in figure.axes:
            h, l = axis.get_legend_handles_labels()
            for handle, label in zip(h, l):
                if label and label not in labels:
                    handles.append(handle)
                    labels.append(label)
    figure.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.925), ncol=_profile_legend_columns(len(profile_items)), frameon=False, fontsize=8.6, handletextpad=0.4, columnspacing=1.2, labelspacing=0.55, borderaxespad=0.0)
    figure.suptitle(title, fontsize=12.5, y=0.975)
    figure.supxlabel("Metric value", fontsize=10.5, y=0.035)
    figure.subplots_adjust(left=0.18, right=0.995, top=0.80, bottom=0.09)
    png_path = plot_dir / output_name
    figure.savefig(png_path, bbox_inches="tight")
    figure.savefig(png_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)
    return png_path


def plot_metric_profile(
    frame: pl.DataFrame,
    *,
    series_order: list[str],
    title: str,
    output_name: str,
    plot_dir: Path,
    metric_keys: list[str] | None = None,
    include_nll_axis: bool = False,
    series_labels: dict[str, str] | None = None,
    series_colors: dict[str, str] | None = None,
    emphasize_first: bool = True,
) -> Path:
    plt.style.use("default")
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.edgecolor": "#2F3740", "axes.labelcolor": "#1F2933", "xtick.color": "#1F2933", "ytick.color": "#1F2933", "figure.facecolor": "white", "savefig.facecolor": "white"})
    lookup = _summary_lookup(frame)
    nll_lookup = _nll_lookup(frame) if include_nll_axis else {}
    profile_items = _profile_metric_items(metric_keys)
    y_positions = np.arange(len(series_order))[::-1] * PROFILE_ROW_GAP
    figure, axes = plt.subplots(nrows=1, ncols=len(LEVEL_ORDER), figsize=(12.2, 6.4), sharex=True, sharey=True, dpi=300, gridspec_kw={"wspace": 0.08})
    axes = np.ravel(axes)

    for col_index, (axis, level) in enumerate(zip(axes, LEVEL_ORDER)):
        nll_axis = None
        level_nll_values = [
            value[0]
            for key, value in nll_lookup.items()
            if key[0] == level
        ]
        if level_nll_values:
            nll_axis = axis.twiny()
            nll_axis.set_xlim(*_nll_axis_limits(level_nll_values))
            nll_axis.tick_params(axis="x", labelsize=7.2, colors="#111827", length=2.5, pad=1.5)
            nll_axis.spines["top"].set_color("#111827")
            nll_axis.spines["bottom"].set_visible(False)
            nll_axis.spines["left"].set_visible(False)
            nll_axis.spines["right"].set_visible(False)
            if col_index == len(LEVEL_ORDER) - 1:
                nll_axis.set_xlabel("CSCG eval NLL (bits/symbol)", fontsize=8.2, color="#111827", labelpad=4)
            else:
                nll_axis.set_xlabel("")
        if emphasize_first:
            axis.axhspan(y_positions[0] - 0.58, y_positions[0] + 0.58, color=_series_color(series_order[0], series_colors), alpha=0.075, zorder=0)
        for row_index, series_name in enumerate(series_order):
            base_y = y_positions[row_index]
            for metric, label, color, marker, offset in profile_items:
                seed_values = (
                    frame.filter((pl.col("level") == level) & (pl.col("plot_series") == series_name))
                    .drop_nulls(["seed", metric])
                    .group_by("seed")
                    .agg(pl.col(metric).mean().alias("value"))
                    .sort("seed")
                )
                for seed_index, value in enumerate(seed_values["value"]):
                    jitter = ((seed_index % 5) - 2) * 0.018
                    axis.scatter(float(value), base_y + offset + jitter, s=11, color=color, alpha=0.18, linewidths=0, zorder=1)
                result = lookup.get((level, series_name, metric))
                if result is None:
                    continue
                mean, half_width, _ = result
                is_emphasized = emphasize_first and row_index == 0
                axis.errorbar(mean, base_y + offset, xerr=half_width, fmt=marker, color=color, ecolor=color, elinewidth=1.95 if is_emphasized else 1.45, capsize=3.2, capthick=1.95 if is_emphasized else 1.45, markersize=4.6 if is_emphasized else 4.0, markeredgecolor="#1F2933", markeredgewidth=0.25, label=label if col_index == 0 and row_index == 0 else None, zorder=3)
            if nll_axis is not None:
                nll_result = nll_lookup.get((level, series_name))
                if nll_result is not None:
                    seed_values = (
                        frame.filter((pl.col("level") == level) & (pl.col("plot_series") == series_name))
                        .drop_nulls(["seed", CSCG_NLL_METRIC])
                        .group_by("seed")
                        .agg(pl.col(CSCG_NLL_METRIC).mean().alias("value"))
                        .sort("seed")
                    )
                    for seed_index, value in enumerate(seed_values["value"]):
                        jitter = ((seed_index % 5) - 2) * 0.018
                        nll_axis.scatter(float(value), base_y - 0.52 + jitter, s=13, color="#111827", alpha=0.14, marker="D", linewidths=0, zorder=2)
                    mean, half_width, _ = nll_result
                    is_emphasized = emphasize_first and row_index == 0
                    nll_axis.errorbar(mean, base_y - 0.52, xerr=half_width, fmt="D", color="#111827", ecolor="#111827", elinewidth=1.65 if is_emphasized else 1.25, capsize=3.0, capthick=1.65 if is_emphasized else 1.25, markersize=4.2, markeredgecolor="white", markeredgewidth=0.25, label="CSCG eval NLL" if col_index == 0 and row_index == 0 else None, zorder=4)
        axis.set_title(LEVEL_LABELS[level], fontsize=11.0, fontweight="bold", pad=8)
        axis.set_xlim(0.0, 1.0)
        axis.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
        axis.set_xticklabels(["0", "0.25", "0.50", "0.75", "1.00"])
        axis.set_ylim(-0.8, y_positions[0] + 0.8)
        axis.set_axisbelow(True)
        axis.grid(axis="x", which="major", color="#D8DEE6", linewidth=0.8, alpha=0.7)
        axis.tick_params(axis="both", labelsize=8.8)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.spines["left"].set_visible(col_index == 0)
        if col_index == 0:
            axis.set_yticks(y_positions, [_series_label(name, series_labels) for name in series_order])
            for index, tick_label in enumerate(axis.get_yticklabels()):
                if index == 0:
                    tick_label.set_fontweight("bold")
            axis.tick_params(axis="y", labelsize=9.0, pad=5)
        else:
            axis.tick_params(axis="y", labelleft=False, left=False)

    handles, labels = axes[0].get_legend_handles_labels()
    if include_nll_axis:
        for axis in figure.axes:
            h, l = axis.get_legend_handles_labels()
            for handle, label in zip(h, l):
                if label and label not in labels:
                    handles.append(handle)
                    labels.append(label)
    figure.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.915), ncol=_profile_legend_columns(len(profile_items)), frameon=False, fontsize=8.6, handletextpad=0.4, columnspacing=1.2, labelspacing=0.55, borderaxespad=0.0)
    figure.suptitle(title, fontsize=12.5, y=0.975)
    figure.supxlabel("Metric value", fontsize=10.5, y=0.035)
    figure.subplots_adjust(left=0.16, right=0.995, top=0.78, bottom=0.105)
    png_path = plot_dir / output_name
    figure.savefig(png_path, bbox_inches="tight")
    figure.savefig(png_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)
    return png_path


def plot_metric_profile_transposed(
    frame: pl.DataFrame,
    *,
    series_order: list[str],
    title: str,
    output_name: str,
    plot_dir: Path,
    metric_keys: list[str] | None = None,
    include_nll_axis: bool = False,
    series_labels: dict[str, str] | None = None,
    series_colors: dict[str, str] | None = None,
    emphasize_first: bool = True,
) -> Path:
    """Plot the original wide transposed metric profile."""
    plt.style.use("default")
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.edgecolor": "#2F3740", "axes.labelcolor": "#1F2933", "xtick.color": "#1F2933", "ytick.color": "#1F2933", "figure.facecolor": "white", "savefig.facecolor": "white"})
    lookup = _summary_lookup(frame)
    nll_lookup = _nll_lookup(frame) if include_nll_axis else {}
    row_specs = _transposed_row_specs(metric_keys, include_nll_axis=include_nll_axis)
    y_positions = np.arange(len(row_specs))[::-1] * PROFILE_ROW_GAP
    row_y = {metric: float(y_positions[index]) for index, (metric, _) in enumerate(row_specs)}
    method_offsets = np.linspace(0.42, -0.42, len(series_order)) if len(series_order) > 1 else np.asarray([0.0])
    figure_height = 6.6 if include_nll_axis else 6.0
    figure, axes = plt.subplots(nrows=1, ncols=len(LEVEL_ORDER), figsize=(12.2, figure_height), sharex=True, sharey=True, dpi=300, gridspec_kw={"wspace": 0.08})
    axes = np.ravel(axes)

    for col_index, (axis, level) in enumerate(zip(axes, LEVEL_ORDER)):
        nll_axis = None
        level_nll_values = [value[0] for key, value in nll_lookup.items() if key[0] == level]
        if level_nll_values:
            nll_axis = axis.twiny()
            nll_axis.set_xlim(*_nll_axis_limits(level_nll_values))
            nll_axis.tick_params(axis="x", labelsize=7.2, colors="#111827", length=2.5, pad=1.5)
            nll_axis.spines["top"].set_color("#111827")
            nll_axis.spines["bottom"].set_visible(False)
            nll_axis.spines["left"].set_visible(False)
            nll_axis.spines["right"].set_visible(False)
            nll_axis.set_xlabel("CSCG eval NLL (bits/symbol)" if col_index == len(LEVEL_ORDER) - 1 else "", fontsize=8.2, color="#111827", labelpad=4)

        for row_index, (metric, _) in enumerate(row_specs):
            base_y = row_y[metric]
            if row_index > 0:
                axis.axhline(base_y + PROFILE_ROW_GAP / 2, color="#E2E8F0", linewidth=0.8, zorder=0)
            target_axis = nll_axis if metric == CSCG_NLL_METRIC else axis
            if target_axis is None:
                continue
            for method_index, series_name in enumerate(series_order):
                offset_y = base_y + float(method_offsets[method_index])
                if metric == CSCG_NLL_METRIC:
                    result = nll_lookup.get((level, series_name))
                else:
                    result = lookup.get((level, series_name, metric))
                if result is None:
                    continue
                seed_values = (
                    frame.filter((pl.col("level") == level) & (pl.col("plot_series") == series_name))
                    .drop_nulls(["seed", metric])
                    .group_by("seed")
                    .agg(pl.col(metric).mean().alias("value"))
                    .sort("seed")
                )
                color = _series_color(series_name, series_colors)
                for seed_index, value in enumerate(seed_values["value"]):
                    jitter = ((seed_index % 5) - 2) * 0.010
                    target_axis.scatter(float(value), offset_y + jitter, s=11, color=color, alpha=0.18, linewidths=0, zorder=1)
                mean, half_width, _ = result
                is_emphasized = emphasize_first and method_index == 0
                target_axis.errorbar(mean, offset_y, xerr=half_width, fmt="o", color=color, ecolor=color, elinewidth=1.95 if is_emphasized else 1.35, capsize=3.0, capthick=1.95 if is_emphasized else 1.35, markersize=4.6 if is_emphasized else 3.9, markeredgecolor="#1F2933", markeredgewidth=0.25, zorder=3)

        axis.set_title(LEVEL_LABELS[level], fontsize=11.0, fontweight="bold", pad=8)
        axis.set_xlim(0.0, 1.0)
        axis.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
        axis.set_xticklabels(["0", "0.25", "0.50", "0.75", "1.00"])
        axis.set_ylim(y_positions[-1] - 0.72, y_positions[0] + 0.72)
        axis.set_axisbelow(True)
        axis.grid(axis="x", which="major", color="#D8DEE6", linewidth=0.8, alpha=0.7)
        axis.tick_params(axis="both", labelsize=8.8)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.spines["left"].set_visible(col_index == 0)
        if col_index == 0:
            axis.set_yticks(y_positions, [_wrap_row_label(label, width=24) for _, label in row_specs])
            axis.tick_params(axis="y", labelsize=8.4, pad=5)
        else:
            axis.tick_params(axis="y", labelleft=False, left=False)

    method_handles = [
        Line2D([0], [0], color=_series_color(name, series_colors), marker="o", linestyle="none", markersize=5, label=_series_label(name, series_labels))
        for name in series_order
    ]
    original_legend_columns = len(method_handles) if len(method_handles) <= 6 else (len(method_handles) + 1) // 2
    figure.legend(method_handles, [handle.get_label() for handle in method_handles], loc="upper center", bbox_to_anchor=(0.5, 0.91), ncol=original_legend_columns, frameon=False, fontsize=8.4, handletextpad=0.4, columnspacing=1.1, labelspacing=0.5, borderaxespad=0.0)
    figure.suptitle(title, fontsize=12.5, y=0.985)
    figure.supxlabel("Metric value", fontsize=10.5, y=0.035)
    figure.subplots_adjust(left=0.18, right=0.995, top=0.75, bottom=0.105)
    png_path = plot_dir / output_name
    figure.savefig(png_path, bbox_inches="tight")
    figure.savefig(png_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)
    return png_path


def plot_metric_profile_transposed_compact(
    frame: pl.DataFrame,
    *,
    series_order: list[str],
    title: str,
    output_name: str,
    plot_dir: Path,
    metric_keys: list[str] | None = None,
    include_nll_axis: bool = False,
    series_labels: dict[str, str] | None = None,
    series_colors: dict[str, str] | None = None,
    emphasize_first: bool = True,
) -> Path:
    """Plot a single-column-ready 2x2 metric profile with mean and 95% CI."""
    plt.style.use("default")
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.edgecolor": "#64748B", "axes.labelcolor": "#1F2933", "xtick.color": "#1F2933", "ytick.color": "#1F2933", "figure.facecolor": "white", "savefig.facecolor": "white", "pdf.fonttype": 42, "ps.fonttype": 42})
    lookup = _summary_lookup(frame)
    nll_lookup = _nll_lookup(frame) if include_nll_axis else {}
    row_specs = _transposed_row_specs(metric_keys, include_nll_axis=include_nll_axis)
    row_gap = 1.0
    y_positions = np.arange(len(row_specs))[::-1] * row_gap
    row_y = {metric: float(y_positions[index]) for index, (metric, _) in enumerate(row_specs)}
    method_offsets = _transposed_method_offsets(len(series_order))
    figure_height = 4.75 if include_nll_axis else 4.35
    figure, axes = plt.subplots(
        nrows=2,
        ncols=2,
        # The tight crop trims roughly 0.2 in of unused canvas, yielding a
        # 3.4--3.5 in exported page rather than requiring downstream scaling.
        figsize=(3.68, figure_height),
        sharex=True,
        sharey=True,
        dpi=300,
        gridspec_kw={"hspace": 0.28, "wspace": 0.16},
    )
    axes = np.ravel(axes)

    for col_index, (axis, level) in enumerate(zip(axes, LEVEL_ORDER)):
        nll_axis = None
        level_nll_values = [value[0] for key, value in nll_lookup.items() if key[0] == level]
        if level_nll_values:
            nll_axis = axis.twiny()
            nll_axis.set_xlim(*_nll_axis_limits(level_nll_values))
            nll_axis.tick_params(axis="x", labelsize=5.5, colors="#111827", length=2.0, pad=1.0)
            nll_axis.spines["top"].set_color("#111827")
            nll_axis.spines["bottom"].set_visible(False)
            nll_axis.spines["left"].set_visible(False)
            nll_axis.spines["right"].set_visible(False)
            nll_axis.set_xlabel("NLL (bits/symbol)" if col_index == len(LEVEL_ORDER) - 1 else "", fontsize=5.8, color="#111827", labelpad=2)

        for row_index, (metric, _) in enumerate(row_specs):
            base_y = row_y[metric]
            if row_index > 0:
                axis.axhline(base_y + row_gap / 2, color="#D7DEE7", linewidth=0.55, zorder=0)
            target_axis = nll_axis if metric == CSCG_NLL_METRIC else axis
            if target_axis is None:
                continue
            for method_index, series_name in enumerate(series_order):
                offset_y = base_y + float(method_offsets[method_index])
                if metric == CSCG_NLL_METRIC:
                    result = nll_lookup.get((level, series_name))
                else:
                    result = lookup.get((level, series_name, metric))
                if result is None:
                    continue
                color = _series_color(series_name, series_colors)
                seed_values = (
                    frame.filter((pl.col("level") == level) & (pl.col("plot_series") == series_name))
                    .drop_nulls(["seed", metric])
                    .group_by("seed")
                    .agg(pl.col(metric).mean().alias("value"))
                    .sort("seed")
                )
                seed_offsets = np.linspace(-0.025, 0.025, len(seed_values))
                target_axis.scatter(
                    seed_values["value"].to_numpy(),
                    offset_y + seed_offsets,
                    s=3.2,
                    color=color,
                    alpha=0.22,
                    linewidths=0,
                    zorder=1,
                )
                mean, half_width, _ = result
                is_emphasized = emphasize_first and method_index == 0
                target_axis.errorbar(mean, offset_y, xerr=half_width, fmt="o", color=color, ecolor=color, elinewidth=1.15 if is_emphasized else 0.8, capsize=1.8, capthick=1.15 if is_emphasized else 0.8, markersize=3.6 if is_emphasized else 2.9, markeredgecolor="#111827", markeredgewidth=0.3, zorder=3)

        axis.set_title(LEVEL_LABELS[level], fontsize=7.3, fontweight="bold", pad=3.0)
        axis.set_xlim(0.0, 1.0)
        axis.set_xticks([0.0, 0.5, 1.0])
        axis.set_xticklabels(["0", "0.5", "1.0"])
        axis.set_ylim(y_positions[-1] - 0.48, y_positions[0] + 0.48)
        axis.set_axisbelow(True)
        axis.grid(axis="x", which="major", color="#D8DEE6", linewidth=0.5, alpha=0.75)
        axis.tick_params(axis="x", labelsize=6.2, length=2.2, width=0.55, pad=1.5)
        axis.tick_params(axis="y", length=0)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.spines["left"].set_visible(False)
        axis.spines["bottom"].set_linewidth(0.55)
        if col_index % 2 == 0:
            axis.set_yticks(
                y_positions,
                [COMPACT_TRANSPOSED_METRIC_LABELS.get(metric, label) for metric, label in row_specs],
            )
            axis.tick_params(axis="y", labelsize=6.2, pad=3.0)
        else:
            axis.tick_params(axis="y", labelleft=False, left=False)

    method_handles = [
        Line2D([0], [0], color=_series_color(name, series_colors), marker="o", linestyle="none", markersize=3.5, label=_compact_transposed_method_label(name, series_labels))
        for name in series_order
    ]
    figure.legend(method_handles, [handle.get_label() for handle in method_handles], loc="upper center", bbox_to_anchor=(0.5, 0.975), ncol=_transposed_method_legend_columns(len(method_handles)), frameon=False, fontsize=5.9, handletextpad=0.25, columnspacing=0.65, labelspacing=0.25, borderaxespad=0.0)
    figure.supxlabel("Metric value", fontsize=6.5, y=0.018)
    figure.subplots_adjust(left=0.27, right=0.985, top=0.86, bottom=0.075)
    png_path = plot_dir / output_name
    figure.savefig(png_path, bbox_inches="tight", pad_inches=0.025, dpi=600)
    figure.savefig(png_path.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.025)
    plt.close(figure)
    return png_path


def plot_metric_profile_by_rand_transposed(
    frame: pl.DataFrame,
    *,
    series_order: list[str],
    title: str,
    output_name: str,
    plot_dir: Path,
    metric_keys: list[str] | None = None,
    include_nll_axis: bool = False,
    series_labels: dict[str, str] | None = None,
    series_colors: dict[str, str] | None = None,
) -> Path:
    """Plot metrics as rows, with method colours and persistence markers."""
    plt.style.use("default")
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.edgecolor": "#2F3740", "axes.labelcolor": "#1F2933", "xtick.color": "#1F2933", "ytick.color": "#1F2933", "figure.facecolor": "white", "savefig.facecolor": "white"})
    rand_order = [0.0, 0.3]
    rand_markers = {0.0: "o", 0.3: "s"}
    lookup = _summary_lookup_by_rand(frame)
    nll_lookup = _nll_lookup_by_rand(frame) if include_nll_axis else {}
    row_specs = _transposed_row_specs(metric_keys, include_nll_axis=include_nll_axis)
    y_positions = np.arange(len(row_specs))[::-1] * PROFILE_ROW_GAP
    row_y = {metric: float(y_positions[index]) for index, (metric, _) in enumerate(row_specs)}
    method_offsets = _transposed_method_offsets(len(series_order))
    persistence_offsets = {0.0: 0.025, 0.3: -0.025}
    figure_height = 7.0 if include_nll_axis else 6.4
    figure, axes = plt.subplots(nrows=1, ncols=len(LEVEL_ORDER), figsize=(12.2, figure_height), sharex=True, sharey=True, dpi=300, gridspec_kw={"wspace": 0.08})
    axes = np.ravel(axes)

    for col_index, (axis, level) in enumerate(zip(axes, LEVEL_ORDER)):
        nll_axis = None
        level_nll_values = [value[0] for key, value in nll_lookup.items() if key[0] == level]
        if level_nll_values:
            nll_axis = axis.twiny()
            nll_axis.set_xlim(*_nll_axis_limits(level_nll_values))
            nll_axis.tick_params(axis="x", labelsize=7.2, colors="#111827", length=2.5, pad=1.5)
            nll_axis.spines["top"].set_color("#111827")
            nll_axis.spines["bottom"].set_visible(False)
            nll_axis.spines["left"].set_visible(False)
            nll_axis.spines["right"].set_visible(False)
            nll_axis.set_xlabel("CSCG eval NLL (bits/symbol)" if col_index == len(LEVEL_ORDER) - 1 else "", fontsize=8.2, color="#111827", labelpad=4)

        for row_index, (metric, _) in enumerate(row_specs):
            base_y = row_y[metric]
            if row_index > 0:
                axis.axhline(base_y + PROFILE_ROW_GAP / 2, color="#E2E8F0", linewidth=0.8, zorder=0)
            target_axis = nll_axis if metric == CSCG_NLL_METRIC else axis
            if target_axis is None:
                continue
            for method_index, series_name in enumerate(series_order):
                color = _series_color(series_name, series_colors)
                for rand_prob in rand_order:
                    offset_y = base_y + float(method_offsets[method_index]) + persistence_offsets[rand_prob]
                    if metric == CSCG_NLL_METRIC:
                        result = nll_lookup.get((level, series_name, rand_prob))
                    else:
                        result = lookup.get((level, series_name, rand_prob, metric))
                    if result is None:
                        continue
                    seed_values = (
                        frame.filter((pl.col("level") == level) & (pl.col("plot_series") == series_name) & (pl.col("rand_prob") == rand_prob))
                        .drop_nulls(["seed", metric])
                        .group_by("seed")
                        .agg(pl.col(metric).mean().alias("value"))
                        .sort("seed")
                    )
                    for seed_index, value in enumerate(seed_values["value"]):
                        jitter = ((seed_index % 5) - 2) * 0.008
                        target_axis.scatter(float(value), offset_y + jitter, s=10, color=color, alpha=0.16, marker=rand_markers[rand_prob], linewidths=0, zorder=1)
                    mean, half_width, _ = result
                    target_axis.errorbar(mean, offset_y, xerr=half_width, fmt=rand_markers[rand_prob], color=color, ecolor=color, elinewidth=1.35, capsize=2.8, capthick=1.35, markersize=3.9, markeredgecolor="#1F2933", markeredgewidth=0.25, zorder=3)

        axis.set_title(LEVEL_LABELS[level], fontsize=11.0, fontweight="bold", pad=8)
        axis.set_xlim(0.0, 1.0)
        axis.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
        axis.set_xticklabels(["0", "0.25", "0.50", "0.75", "1.00"])
        axis.set_ylim(y_positions[-1] - 0.72, y_positions[0] + 0.72)
        axis.set_axisbelow(True)
        axis.grid(axis="x", which="major", color="#D8DEE6", linewidth=0.8, alpha=0.7)
        axis.tick_params(axis="both", labelsize=8.8)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.spines["left"].set_visible(col_index == 0)
        if col_index == 0:
            axis.set_yticks(y_positions, [_wrap_row_label(label, width=24) for _, label in row_specs])
            axis.tick_params(axis="y", labelsize=8.4, pad=5)
        else:
            axis.tick_params(axis="y", labelleft=False, left=False)

    method_handles = [
        Line2D([0], [0], color=_series_color(name, series_colors), marker="o", linestyle="none", markersize=5, label=_series_label(name, series_labels))
        for name in series_order
    ]
    persistence_handles = [
        Line2D([0], [0], color="#1F2933", marker=rand_markers[rand_prob], linestyle="none", markersize=5, label=f"Persistence {rand_prob:.1f}")
        for rand_prob in rand_order
    ]
    figure.legend(method_handles, [handle.get_label() for handle in method_handles], loc="upper center", bbox_to_anchor=(0.5, 0.925), ncol=_transposed_method_legend_columns(len(method_handles)), frameon=False, fontsize=8.4, handletextpad=0.4, columnspacing=1.1, labelspacing=0.5, borderaxespad=0.0)
    figure.legend(persistence_handles, [handle.get_label() for handle in persistence_handles], loc="upper center", bbox_to_anchor=(0.5, 0.84), ncol=2, frameon=False, fontsize=8.4, handletextpad=0.4, columnspacing=1.4, borderaxespad=0.0)
    figure.suptitle(title, fontsize=12.5, y=0.985)
    figure.supxlabel("Metric value", fontsize=10.5, y=0.035)
    figure.subplots_adjust(left=0.18, right=0.995, top=0.72, bottom=0.105)
    png_path = plot_dir / output_name
    figure.savefig(png_path, bbox_inches="tight")
    plt.close(figure)
    return png_path


def plot_forest(
    frame: pl.DataFrame,
    *,
    series_order: list[str],
    title: str,
    output_name: str,
    plot_dir: Path,
    series_labels: dict[str, str] | None = None,
    series_colors: dict[str, str] | None = None,
    emphasize_first: bool = True,
) -> Path:
    plt.style.use("default")
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.edgecolor": "#2F3740", "axes.labelcolor": "#1F2933", "xtick.color": "#1F2933", "ytick.color": "#1F2933", "figure.facecolor": "white", "savefig.facecolor": "white"})
    summaries = {metric: _summarise(frame, metric) for metric, _ in METRICS}
    n_metrics = len(METRICS)
    n_levels = len(LEVEL_ORDER)
    height_ratios = [0.68] + [1.0] * (n_metrics - 1)
    figure, axes = plt.subplots(
        nrows=n_metrics,
        ncols=n_levels,
        figsize=(9.6, 1.35 * n_metrics + 1.5),
        sharex=True,
        sharey=True,
        dpi=300,
        gridspec_kw={"height_ratios": height_ratios, "hspace": 0.22, "wspace": 0.08},
    )
    axes = np.asarray(axes).reshape(n_metrics, n_levels)
    y_positions = np.arange(len(series_order))[::-1]
    seed_offsets = np.linspace(-0.11, 0.11, 30)
    offset_by_seed = {seed: seed_offsets[index] for index, seed in enumerate(range(30))}

    for row_index, (metric, metric_label) in enumerate(METRICS):
        summary = summaries[metric]
        for col_index, level in enumerate(LEVEL_ORDER):
            axis = axes[row_index, col_index]
            panel = frame.filter(pl.col("level") == level)
            metric_summary = summary.filter(pl.col("level") == level)
            for series_index, series_name in enumerate(series_order):
                y = y_positions[series_index]
                color = _series_color(series_name, series_colors)
                seed_values = (
                    panel.filter(pl.col("plot_series") == series_name)
                    .drop_nulls(["seed", metric])
                    .group_by("seed")
                    .agg(pl.col(metric).mean().alias("value"))
                    .sort("seed")
                )
                for seed, value in zip(seed_values["seed"], seed_values["value"]):
                    axis.scatter(float(value), y + offset_by_seed.get(int(seed), 0.0), s=7, color=color, alpha=0.16, linewidths=0, zorder=1)
                row = metric_summary.filter(pl.col("plot_series") == series_name)
                if row.is_empty():
                    continue
                mean = float(row["mean"][0])
                error = _ci95_half_width(int(row["n"][0]), float(row["sd"][0]))
                is_emphasized = emphasize_first and series_index == 0
                linewidth = 2.2 if is_emphasized else 1.45
                markersize = 46 if is_emphasized else 32
                axis.errorbar(mean, y, xerr=error, fmt="o", color=color, ecolor=color, elinewidth=linewidth, capsize=3.0, capthick=linewidth, markersize=np.sqrt(markersize), markeredgecolor="#1F2933", markeredgewidth=0.35, zorder=3)
            axis.set_xlim(0.0, 1.0)
            axis.set_ylim(-0.65, len(series_order) - 0.35)
            axis.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
            axis.grid(axis="x", alpha=0.2, linewidth=0.8)
            axis.tick_params(axis="both", labelsize=8.5, length=3)
            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)
            axis.spines["left"].set_visible(col_index == 0)
            if row_index == 0:
                axis.set_title(LEVEL_LABELS[level], fontsize=10.5, fontweight="bold", pad=7)
            if col_index == 0:
                axis.set_yticks(y_positions, [_series_label(name, series_labels) for name in series_order])
                axis.tick_params(axis="y", labelsize=8.8, pad=5)
                # Wrap and pin the row label: unwrapped it is several times
                # taller than its row and overruns the neighbouring rows.
                axis.set_ylabel(_wrap_row_label(metric_label), fontsize=9.2, fontweight="bold", ha="center", va="center")
                axis.yaxis.set_label_coords(-0.82, 0.5)
            else:
                axis.tick_params(axis="y", labelleft=False, left=False)
            if row_index < len(METRICS) - 1:
                axis.tick_params(axis="x", labelbottom=False)

    figure.suptitle(title, fontsize=12.5, y=0.985)
    figure.supxlabel("Metric value", fontsize=10.5, y=0.03)
    figure.subplots_adjust(left=0.155, right=0.995, top=0.90, bottom=0.095)
    png_path = plot_dir / output_name
    figure.savefig(png_path, bbox_inches="tight")
    figure.savefig(png_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)
    return png_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study_root", type=Path, default=DEFAULT_STUDY_ROOT)
    parser.add_argument("--protocol_id", type=str, required=True,
                        help="Canonical-v4 protocol ID (e.g. canonical-v4-<hash>)")
    parser.add_argument("--protocol_ids_by_rand", type=str, default=None,
                        help="Optional comma-separated rand_prob=protocol_id mapping.")
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_PLOT_DIR)
    parser.add_argument("--batch_size", type=int, default=40)
    parser.add_argument("--comparisons", type=str, default="final,checkpoint_4000,checkpoint_2000")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading evaluation records …")
    full_frame = _build_frame(
        args.study_root,
        args.protocol_id,
        protocol_ids_by_rand=_parse_protocol_ids_by_rand(args.protocol_ids_by_rand),
        batch_size=args.batch_size,
    )
    if full_frame.is_empty():
        print("No evaluation records found. Run evaluate_study first.", file=sys.stderr)
        sys.exit(1)
    print(f"Loaded {len(full_frame)} rows from evaluation records.")

    forest_suffix = f"forest_{len(METRICS)}x{len(LEVEL_ORDER)}"
    batch_prefix = f"batch_{int(args.batch_size)}"
    plot_specs = [
        (
            "final",
            HYBRID_SERIES_ORDER,
            f"Final batch-{int(args.batch_size)} initialised methods versus CSCG variants",
            f"{batch_prefix}_final_selected_{forest_suffix}.png",
            f"{batch_prefix}_final_selected_metric_matrix.png",
            f"{batch_prefix}_final_selected_metric_profile.png",
            f"{batch_prefix}_final_selected_metric_profile_rand_all.png",
            f"{batch_prefix}_final_selected_metric_profile_rand_rows.png",
            "Initialised gain",
        ),
        (
            "checkpoint_4000",
            HYBRID_SERIES_ORDER,
            f"Checkpoint 4000 batch-{int(args.batch_size)} initialised methods versus CSCG variants",
            f"{batch_prefix}_checkpoint_4000_selected_{forest_suffix}.png",
            f"{batch_prefix}_checkpoint_4000_selected_metric_matrix.png",
            f"{batch_prefix}_checkpoint_4000_selected_metric_profile.png",
            f"{batch_prefix}_checkpoint_4000_selected_metric_profile_rand_all.png",
            f"{batch_prefix}_checkpoint_4000_selected_metric_profile_rand_rows.png",
            "Initialised gain",
        ),
        (
            "checkpoint_2000",
            MERCURY_SERIES_ORDER,
            f"Checkpoint 2000 batch-{int(args.batch_size)} MERCURY and TMGWR versus CSCG variants",
            f"{batch_prefix}_checkpoint_2000_selected_{forest_suffix}.png",
            f"{batch_prefix}_checkpoint_2000_selected_metric_matrix.png",
            f"{batch_prefix}_checkpoint_2000_selected_metric_profile.png",
            f"{batch_prefix}_checkpoint_2000_selected_metric_profile_rand_all.png",
            f"{batch_prefix}_checkpoint_2000_selected_metric_profile_rand_rows.png",
            "Best gain",
        ),
    ]
    selected_comparisons = {item.strip() for item in str(args.comparisons).split(",") if item.strip()}
    for (
        comparison, series_order, title,
        forest_output_name, matrix_output_name, profile_output_name,
        profile_rand_all_output_name, profile_rand_rows_output_name, gain_label,
    ) in plot_specs:
        if comparison not in selected_comparisons:
            continue
        source = _load_source(full_frame, comparison, series_order)
        if source.is_empty():
            print(f"  No data for comparison={comparison!r}, skipping.", file=sys.stderr)
            continue
        series_order = _available_series_order(source, series_order)
        series_labels = _series_labels_for_comparison(comparison)
        series_colors = _series_colors_for_comparison(comparison)
        kwargs = dict(series_order=series_order, title=title, plot_dir=args.output_dir, series_labels=series_labels, series_colors=series_colors)
        forest_path = plot_forest(source, output_name=forest_output_name, **kwargs)
        matrix_path = plot_metric_matrix(source, output_name=matrix_output_name, gain_label=gain_label, **kwargs)
        profile_path = plot_metric_profile(source, output_name=profile_output_name, **kwargs)
        focused_profile_path = plot_metric_profile(
            source,
            output_name=profile_output_name.replace("_metric_profile.png", "_metric_profile_focused.png"),
            metric_keys=PROFILE_FOCUSED_METRICS,
            **kwargs,
        )
        focused_profile_transposed_path = plot_metric_profile_transposed(
            source,
            output_name=_transposed_output_name(focused_profile_path.name),
            metric_keys=PROFILE_FOCUSED_METRICS,
            **{**kwargs, "title": pub.PUBLICATION_COMPARISON_TITLES[comparison]},
        )
        focused_profile_compact_transposed_path = plot_metric_profile_transposed_compact(
            source,
            output_name=_compact_transposed_output_name(focused_profile_transposed_path.name),
            metric_keys=PROFILE_FOCUSED_METRICS,
            **{**kwargs, "title": pub.PUBLICATION_COMPARISON_TITLES[comparison]},
        )
        nll_enabled = comparison in {"final", "checkpoint_4000", "checkpoint_2000"} and CSCG_NLL_METRIC in source.columns
        focused_profile_with_nll_path = None
        focused_profile_with_nll_transposed_path = None
        if nll_enabled:
            focused_profile_with_nll_path = plot_metric_profile(
                source,
                output_name=profile_output_name.replace("_metric_profile.png", "_metric_profile_focused_with_nll.png"),
                metric_keys=PROFILE_FOCUSED_METRICS,
                include_nll_axis=True,
                **kwargs,
            )
            focused_profile_with_nll_transposed_path = plot_metric_profile_transposed(
                source,
                output_name=_transposed_output_name(focused_profile_with_nll_path.name),
                metric_keys=PROFILE_FOCUSED_METRICS,
                include_nll_axis=True,
                **kwargs,
            )
        profile_rand_all_path = plot_metric_profile(
            source, output_name=profile_rand_all_output_name,
            title=f"{title} (persistence 0.0 and 0.3 aggregated)", plot_dir=args.output_dir, series_order=series_order, series_labels=series_labels, series_colors=series_colors,
        )
        focused_profile_rand_all_path = plot_metric_profile(
            source,
            output_name=profile_rand_all_output_name.replace("_metric_profile_rand_all.png", "_metric_profile_focused_rand_all.png"),
            title=f"{title} (persistence 0.0 and 0.3 aggregated)",
            plot_dir=args.output_dir,
            series_order=series_order,
            series_labels=series_labels,
            series_colors=series_colors,
            metric_keys=PROFILE_FOCUSED_METRICS,
        )
        focused_profile_rand_all_transposed_path = plot_metric_profile_transposed(
            source,
            output_name=_transposed_output_name(focused_profile_rand_all_path.name),
            title=f"{title} (persistence 0.0 and 0.3 aggregated)",
            plot_dir=args.output_dir,
            series_order=series_order,
            series_labels=series_labels,
            series_colors=series_colors,
            metric_keys=PROFILE_FOCUSED_METRICS,
        )
        focused_profile_rand_all_with_nll_path = None
        focused_profile_rand_all_with_nll_transposed_path = None
        if nll_enabled:
            focused_profile_rand_all_with_nll_path = plot_metric_profile(
                source,
                output_name=profile_rand_all_output_name.replace("_metric_profile_rand_all.png", "_metric_profile_focused_with_nll_rand_all.png"),
                title=f"{title} (persistence 0.0 and 0.3 aggregated)",
                plot_dir=args.output_dir,
                series_order=series_order,
                series_labels=series_labels,
                series_colors=series_colors,
                metric_keys=PROFILE_FOCUSED_METRICS,
                include_nll_axis=True,
            )
            focused_profile_rand_all_with_nll_transposed_path = plot_metric_profile_transposed(
                source,
                output_name=_transposed_output_name(focused_profile_rand_all_with_nll_path.name),
                title=f"{title} (persistence 0.0 and 0.3 aggregated)",
                plot_dir=args.output_dir,
                series_order=series_order,
                series_labels=series_labels,
                series_colors=series_colors,
                metric_keys=PROFILE_FOCUSED_METRICS,
                include_nll_axis=True,
            )
        profile_rand_rows_path = plot_metric_profile_by_rand(
            source, output_name=profile_rand_rows_output_name,
            title=f"{title} by persistence", plot_dir=args.output_dir, series_order=series_order, series_labels=series_labels, series_colors=series_colors,
        )
        focused_profile_rand_rows_path = plot_metric_profile_by_rand(
            source,
            output_name=profile_rand_rows_output_name.replace("_metric_profile_rand_rows.png", "_metric_profile_focused_rand_rows.png"),
            title=f"{title} by persistence",
            plot_dir=args.output_dir,
            series_order=series_order,
            series_labels=series_labels,
            series_colors=series_colors,
            metric_keys=PROFILE_FOCUSED_METRICS,
        )
        focused_profile_rand_rows_transposed_path = plot_metric_profile_by_rand_transposed(
            source,
            output_name=_transposed_output_name(focused_profile_rand_rows_path.name),
            title=f"{title} by persistence",
            plot_dir=args.output_dir,
            series_order=series_order,
            series_labels=series_labels,
            series_colors=series_colors,
            metric_keys=PROFILE_FOCUSED_METRICS,
        )
        focused_profile_rand_rows_with_nll_path = None
        focused_profile_rand_rows_with_nll_transposed_path = None
        if nll_enabled:
            focused_profile_rand_rows_with_nll_path = plot_metric_profile_by_rand(
                source,
                output_name=profile_rand_rows_output_name.replace("_metric_profile_rand_rows.png", "_metric_profile_focused_with_nll_rand_rows.png"),
                title=f"{title} by persistence",
                plot_dir=args.output_dir,
                series_order=series_order,
                series_labels=series_labels,
                series_colors=series_colors,
                metric_keys=PROFILE_FOCUSED_METRICS,
                include_nll_axis=True,
            )
            focused_profile_rand_rows_with_nll_transposed_path = plot_metric_profile_by_rand_transposed(
                source,
                output_name=_transposed_output_name(focused_profile_rand_rows_with_nll_path.name),
                title=f"{title} by persistence",
                plot_dir=args.output_dir,
                series_order=series_order,
                series_labels=series_labels,
                series_colors=series_colors,
                metric_keys=PROFILE_FOCUSED_METRICS,
                include_nll_axis=True,
            )
        rand_profile_paths = []
        focused_rand_profile_paths = []
        focused_rand_profile_transposed_paths = []
        focused_rand_profile_with_nll_paths = []
        focused_rand_profile_with_nll_transposed_paths = []
        for rand_prob in sorted(source["rand_prob"].unique()):
            rand_source = source.filter(pl.col("rand_prob") == rand_prob)
            if rand_source.is_empty():
                continue
            rand_output_name = profile_rand_rows_output_name.replace(
                "_rand_rows.png",
                f"_rand_{_rand_slug(float(rand_prob))}.png",
            )
            rand_profile_paths.append(
                plot_metric_profile(
                    rand_source,
                    output_name=rand_output_name,
                    title=f"{title} (persistence {float(rand_prob):.1f})",
                    plot_dir=args.output_dir,
                    series_order=series_order,
                    series_labels=series_labels,
                    series_colors=series_colors,
                )
            )
            focused_rand_profile_paths.append(
                plot_metric_profile(
                    rand_source,
                    output_name=rand_output_name.replace("_metric_profile_rand_", "_metric_profile_focused_rand_"),
                    title=f"{title} (persistence {float(rand_prob):.1f})",
                    plot_dir=args.output_dir,
                    series_order=series_order,
                    series_labels=series_labels,
                    series_colors=series_colors,
                    metric_keys=PROFILE_FOCUSED_METRICS,
                )
            )
            focused_rand_profile_transposed_paths.append(
                plot_metric_profile_transposed(
                    rand_source,
                    output_name=_transposed_output_name(focused_rand_profile_paths[-1].name),
                    title=f"{title} (persistence {float(rand_prob):.1f})",
                    plot_dir=args.output_dir,
                    series_order=series_order,
                    series_labels=series_labels,
                    series_colors=series_colors,
                    metric_keys=PROFILE_FOCUSED_METRICS,
                )
            )
            if nll_enabled:
                focused_rand_profile_with_nll_paths.append(
                    plot_metric_profile(
                        rand_source,
                        output_name=rand_output_name.replace("_metric_profile_rand_", "_metric_profile_focused_with_nll_rand_"),
                        title=f"{title} (persistence {float(rand_prob):.1f})",
                        plot_dir=args.output_dir,
                        series_order=series_order,
                        series_labels=series_labels,
                        series_colors=series_colors,
                        metric_keys=PROFILE_FOCUSED_METRICS,
                        include_nll_axis=True,
                    )
                )
                focused_rand_profile_with_nll_transposed_paths.append(
                    plot_metric_profile_transposed(
                        rand_source,
                        output_name=_transposed_output_name(focused_rand_profile_with_nll_paths[-1].name),
                        title=f"{title} (persistence {float(rand_prob):.1f})",
                        plot_dir=args.output_dir,
                        series_order=series_order,
                        series_labels=series_labels,
                        series_colors=series_colors,
                        metric_keys=PROFILE_FOCUSED_METRICS,
                        include_nll_axis=True,
                    )
                )
        for output_path in (
            forest_path,
            matrix_path,
            profile_path,
            focused_profile_path,
            focused_profile_transposed_path,
            focused_profile_compact_transposed_path,
            focused_profile_with_nll_path,
            focused_profile_with_nll_transposed_path,
            profile_rand_all_path,
            focused_profile_rand_all_path,
            focused_profile_rand_all_transposed_path,
            focused_profile_rand_all_with_nll_path,
            focused_profile_rand_all_with_nll_transposed_path,
            profile_rand_rows_path,
            focused_profile_rand_rows_path,
            focused_profile_rand_rows_transposed_path,
            focused_profile_rand_rows_with_nll_path,
            focused_profile_rand_rows_with_nll_transposed_path,
            *rand_profile_paths,
            *focused_rand_profile_paths,
            *focused_rand_profile_transposed_paths,
            *focused_rand_profile_with_nll_paths,
            *focused_rand_profile_with_nll_transposed_paths,
        ):
            if output_path is None:
                continue
            print(output_path)


if __name__ == "__main__":
    main()
