"""Structural state--flux portrait and supplementary trajectories for the age-ambiguity ablation."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

import matplotlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from scipy import stats

matplotlib.use("Agg", force=True)

from mercury_runs.analysis.plots.age_ambiguity_ablation import (
    PARAMETER_ORDER,
    SERIES_COLORS,
    SERIES_LABELS,
    SERIES_ORDER,
    _series_key,
)
from scripts.analysis.plot_mercury_structural_trajectories import (
    DEFAULT_LEVEL_ORDER,
    LEVEL_LABELS,
    METRICS,
    load_template_rows,
    load_trajectory_rows,
)


REFERENCE_COLOR = "#94A3B8"
FINAL_STEP = 2000
FLUX_WINDOW = 50
FLUX_SAMPLE_INTERVAL = 50
SENSITIVITY_WINDOWS = (25, 100)
CHURN_DISPLAY_STEPS = 50
TIME_CHECKPOINTS = (0, 250, 500, 750, 1000, 1250, 1500, 1750, 2000)
THRESHOLD_MARKERS = {10: "o", 20: "s", 30: "^", 40: "D"}
THRESHOLD_MARKER_SCALE = {10: 1.45, 20: 1.30, 30: 1.15, 40: 1.00}
SELECTED_CONFIGURATION = (30, 0.99)
THRESHOLD_COLORS = {
    10: "#0072B2",
    20: "#009E73",
    30: "#D55E00",
    40: "#7B4DCC",
}
PHASE_GROUP_COLUMNS = [
    "series",
    "series_label",
    "ambiguity_threshold",
    "ambiguity_decay",
    "level",
    "level_label",
]
SEED_GROUP_COLUMNS = [*PHASE_GROUP_COLUMNS, "run_id", "seed", "n_true_states"]


def _parameter_frame(study_root: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    allowed = set(PARAMETER_ORDER)
    for line in (study_root / "study_manifest.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        config = dict(json.loads(line)["config"])
        if str(config.get("method")) != "mercury":
            continue
        threshold = int(config["ambiguity_threshold"])
        decay = float(config["ambiguity_decay"])
        if (threshold, decay) not in allowed:
            continue
        key = _series_key(threshold, decay)
        rows.append(
            {
                "run_id": str(config.get("requested_run_id") or config.get("run_id")),
                "ambiguity_threshold": threshold,
                "ambiguity_decay": decay,
                "series": key,
                "series_label": SERIES_LABELS[key],
            }
        )
    return pd.DataFrame(rows).drop_duplicates(subset=["run_id"])


def _with_parameters(rows: pd.DataFrame, parameters: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return rows
    return rows.merge(parameters, on="run_id", how="inner", validate="many_to_one")


def _ci95(sd: pd.Series, n: pd.Series) -> np.ndarray:
    sd_values = sd.to_numpy(dtype=float)
    n_values = n.to_numpy(dtype=float)
    result = np.zeros_like(sd_values)
    usable = n_values >= 2
    result[usable] = stats.t.ppf(0.975, n_values[usable] - 1) * sd_values[usable] / np.sqrt(n_values[usable])
    return result


def summarise_trajectories(rows: pd.DataFrame) -> pd.DataFrame:
    summary = (
        rows.groupby(
            ["series", "series_label", "ambiguity_threshold", "ambiguity_decay", "level", "level_label", "metric", "step"],
            dropna=False,
        )["value"]
        .agg(["count", "mean", "std"])
        .reset_index()
        .rename(columns={"count": "n", "std": "sd"})
    )
    summary["sd"] = summary["sd"].fillna(0.0)
    summary["ci95"] = _ci95(summary["sd"], summary["n"])
    return summary


def _metric_matrix(rows: pd.DataFrame, metric: str, steps: set[int]) -> pd.DataFrame:
    selected = rows[(rows["metric"] == metric) & (rows["step"].isin(steps))]
    if selected.duplicated([*SEED_GROUP_COLUMNS, "step"]).any():
        raise ValueError(f"Duplicate {metric} rows were found for a seed and step.")
    matrix = selected.pivot(index=SEED_GROUP_COLUMNS, columns="step", values="value").sort_index()
    missing_steps = sorted(steps.difference(int(step) for step in matrix.columns))
    if missing_steps or matrix.isna().any(axis=None):
        raise ValueError(f"Incomplete {metric} trajectories at steps: {missing_steps}")
    return matrix


def construct_seed_phase_trajectories(
    rows: pd.DataFrame,
    *,
    window: int = FLUX_WINDOW,
    sample_interval: int = FLUX_SAMPLE_INTERVAL,
    final_step: int = FINAL_STEP,
) -> pd.DataFrame:
    """Compute trailing-window structural fluxes for each seed before aggregation."""
    if window <= 0 or sample_interval <= 0 or final_step % sample_interval:
        raise ValueError("window and sample_interval must be positive, and final_step must align to the sample grid.")
    required = {*SEED_GROUP_COLUMNS, "metric", "step", "value"}
    missing = sorted(required.difference(rows.columns))
    if missing:
        raise ValueError(f"Trajectory rows are missing required columns: {missing}")

    sample_times = list(range(sample_interval, final_step + 1, sample_interval))
    previous_times = {max(0, time - window) for time in sample_times}
    cumulative_steps = {1, *sample_times, *(time for time in previous_times if time > 0)}
    node_steps = {1, *sample_times}
    clones = _metric_matrix(rows, "clone_created_cumulative", cumulative_steps)
    deletions = _metric_matrix(rows, "node_prune_cumulative", cumulative_steps)
    nodes = _metric_matrix(rows, "node_ratio", node_steps)
    if not clones.index.equals(deletions.index) or not clones.index.equals(nodes.index):
        raise ValueError("Node, clone, and deletion trajectories do not cover the same seeds.")

    index_rows = clones.index.to_frame(index=False)
    n_true = clones.index.get_level_values("n_true_states").to_numpy(dtype=float)
    clone_positive = clones[final_step].to_numpy(dtype=float) > 0
    structurally_active = clone_positive | (deletions[final_step].to_numpy(dtype=float) > 0)
    activation_times = (
        rows[(rows["metric"] == "clone_created_cumulative") & (rows["value"] > 0)]
        .groupby(SEED_GROUP_COLUMNS, dropna=False)["step"]
        .min()
        .reindex(clones.index)
        .to_numpy(dtype=float)
    )
    initial_inventory = (
        nodes[1].to_numpy(dtype=float)
        - (clones[1].to_numpy(dtype=float) - deletions[1].to_numpy(dtype=float)) / n_true
    )
    phase_rows: list[pd.DataFrame] = []
    initial = index_rows.copy()
    initial["time"] = 0
    initial["node_ratio"] = initial_inventory
    initial["birth_flux"] = 0.0
    initial["death_flux"] = 0.0
    initial["net_flux"] = 0.0
    initial["gross_flux"] = 0.0
    initial["churn_flux"] = 0.0
    initial["cumulative_churn_per_gt"] = 0.0
    initial["clone_positive"] = clone_positive
    initial["structurally_active"] = structurally_active
    initial["activation_time"] = activation_times
    initial["flux_window"] = int(window)
    phase_rows.append(initial)

    for time in sample_times:
        previous = max(0, time - window)
        span = float(time - previous)
        clone_previous = 0.0 if previous == 0 else clones[previous].to_numpy(dtype=float)
        delete_previous = 0.0 if previous == 0 else deletions[previous].to_numpy(dtype=float)
        birth = (clones[time].to_numpy(dtype=float) - clone_previous) / span / n_true
        death = (deletions[time].to_numpy(dtype=float) - delete_previous) / span / n_true
        if np.any(birth < -1e-12) or np.any(death < -1e-12):
            raise ValueError("Cumulative event counts decreased within a flux window.")
        frame = index_rows.copy()
        frame["time"] = int(time)
        frame["node_ratio"] = nodes[time].to_numpy(dtype=float)
        frame["birth_flux"] = birth
        frame["death_flux"] = death
        frame["net_flux"] = birth - death
        frame["gross_flux"] = birth + death
        frame["churn_flux"] = birth + death
        frame["cumulative_churn_per_gt"] = (
            clones[time].to_numpy(dtype=float) + deletions[time].to_numpy(dtype=float)
        ) / n_true
        frame["clone_positive"] = clone_positive
        frame["structurally_active"] = structurally_active
        frame["activation_time"] = activation_times
        frame["flux_window"] = int(window)
        phase_rows.append(frame)
    return pd.concat(phase_rows, ignore_index=True).sort_values(
        ["level", "ambiguity_threshold", "ambiguity_decay", "seed", "time"]
    ).reset_index(drop=True)


def _trajectory_medoid(
    mode_rows: pd.DataFrame,
    feature_columns: tuple[str, str] = ("node_ratio", "churn_flux"),
) -> tuple[str, int, float]:
    trajectories = []
    identities: list[tuple[str, int]] = []
    expected_times: np.ndarray | None = None
    for (run_id, seed), trajectory in mode_rows.groupby(["run_id", "seed"], sort=True):
        trajectory = trajectory.sort_values("time")
        times = trajectory["time"].to_numpy(dtype=int)
        if expected_times is None:
            expected_times = times
        elif not np.array_equal(times, expected_times):
            raise ValueError("Seed phase trajectories do not share a common time grid.")
        trajectories.append(
            np.column_stack(
                [
                    trajectory[feature_columns[0]].to_numpy(dtype=float),
                    trajectory[feature_columns[1]].to_numpy(dtype=float),
                ]
            )
        )
        identities.append((str(run_id), int(seed)))
    values = np.stack(trajectories)
    scales = values.reshape(-1, 2).std(axis=0)
    scales[scales < 1e-12] = 1.0
    standardized = values / scales
    differences = standardized[:, None, :, :] - standardized[None, :, :, :]
    distances = np.sqrt(np.square(differences).sum(axis=3)).sum(axis=2)
    scores = distances.sum(axis=1)
    best = min(range(len(identities)), key=lambda index: (float(scores[index]), identities[index][1], identities[index][0]))
    run_id, seed = identities[best]
    return run_id, seed, float(scores[best])


def select_phase_medoids(
    seed_rows: pd.DataFrame,
    feature_columns: tuple[str, str] = ("node_ratio", "churn_flux"),
) -> pd.DataFrame:
    """Select observed representative paths, splitting only mechanistic mixtures."""
    outputs: list[pd.DataFrame] = []
    for _, condition in seed_rows.groupby(PHASE_GROUP_COLUMNS, sort=True, dropna=False):
        seed_metadata = condition.drop_duplicates(["run_id", "seed"])[
            ["run_id", "seed", "clone_positive", "structurally_active", "activation_time"]
        ].copy()
        n_seeds = int(len(seed_metadata))
        n_clone_positive = int(seed_metadata["clone_positive"].sum())
        n_structurally_active = int(seed_metadata["structurally_active"].sum())
        if n_clone_positive == n_seeds:
            seed_metadata["mode"] = "all"
        else:
            seed_metadata["mode"] = np.select(
                [
                    seed_metadata["clone_positive"],
                    seed_metadata["structurally_active"],
                ],
                ["active", "deletion_only"],
                default="inactive",
            )
        condition = condition.merge(
            seed_metadata[["run_id", "seed", "mode"]],
            on=["run_id", "seed"],
            how="left",
            validate="many_to_one",
        )
        for mode, mode_rows in condition.groupby("mode", sort=True):
            run_id, seed, score = _trajectory_medoid(mode_rows, feature_columns)
            representative = mode_rows[
                (mode_rows["run_id"] == run_id) & (mode_rows["seed"] == seed)
            ].copy()
            mode_metadata = seed_metadata[seed_metadata["mode"] == mode]
            activation = mode_metadata["activation_time"].dropna().to_numpy(dtype=float)
            representative["mode"] = mode
            representative["mode_n"] = int(len(mode_metadata))
            representative["mode_fraction"] = float(len(mode_metadata) / n_seeds)
            representative["n_seeds"] = n_seeds
            representative["n_clone_positive"] = n_clone_positive
            representative["n_zero_clone"] = n_seeds - n_clone_positive
            representative["n_structurally_active"] = n_structurally_active
            representative["n_quiescent"] = n_seeds - n_structurally_active
            representative["medoid_run_id"] = run_id
            representative["medoid_seed"] = seed
            representative["medoid_distance_sum"] = score
            representative["medoid_feature_space"] = "+".join(feature_columns)
            representative["activation_time_median"] = (
                float(np.median(activation)) if activation.size else np.nan
            )
            representative["activation_time_q25"] = (
                float(np.quantile(activation, 0.25)) if activation.size else np.nan
            )
            representative["activation_time_q75"] = (
                float(np.quantile(activation, 0.75)) if activation.size else np.nan
            )
            outputs.append(representative)
    return pd.concat(outputs, ignore_index=True).sort_values(
        ["level", "ambiguity_threshold", "ambiguity_decay", "mode", "time"]
    ).reset_index(drop=True)


def summarise_structural_flux_phase(seed_rows: pd.DataFrame) -> pd.DataFrame:
    """Summarise seed-wise net and gross 50-step flux coordinates by median."""
    required = {
        *PHASE_GROUP_COLUMNS,
        "run_id",
        "seed",
        "time",
        "net_flux",
        "gross_flux",
        "structurally_active",
    }
    missing = sorted(required.difference(seed_rows.columns))
    if missing:
        raise ValueError(f"Structural phase rows are missing required columns: {missing}")

    def aggregate(rows: pd.DataFrame, scope: str) -> pd.DataFrame:
        group_columns = [*PHASE_GROUP_COLUMNS, "time"]
        summary = (
            rows.groupby(group_columns, sort=True, dropna=False)
            .agg(
                net_flux_median=("net_flux", "median"),
                gross_flux_median=("gross_flux", "median"),
                n_seeds=("seed", "count"),
            )
            .reset_index()
        )
        summary["net_flux_50"] = CHURN_DISPLAY_STEPS * summary["net_flux_median"]
        summary["gross_turnover_50"] = CHURN_DISPLAY_STEPS * summary["gross_flux_median"]
        summary["scope"] = scope
        return summary

    outputs = [aggregate(seed_rows, "all_seeds")]
    active_subset = seed_rows[
        (seed_rows["level"] == 22)
        & (seed_rows["ambiguity_threshold"] == 40)
        & np.isclose(seed_rows["ambiguity_decay"], 0.99)
        & seed_rows["structurally_active"]
    ]
    if not active_subset.empty:
        outputs.append(aggregate(active_subset, "active_conditional"))
    result = pd.concat(outputs, ignore_index=True)
    tolerance = 1e-12
    invalid = result["gross_turnover_50"] + tolerance < np.abs(result["net_flux_50"])
    if invalid.any():
        raise ValueError("Median structural flux coordinates fell outside y >= |x|.")
    return result.sort_values(
        ["level", "ambiguity_threshold", "ambiguity_decay", "scope", "time"]
    ).reset_index(drop=True)


def _stable_spearman(left: np.ndarray, right: np.ndarray) -> float:
    if np.allclose(left, left[0]) or np.allclose(right, right[0]):
        return 1.0 if np.allclose(left, right) else 0.0
    result = stats.spearmanr(left, right).statistic
    return float(result) if np.isfinite(result) else 0.0


def window_sensitivity(reference: pd.DataFrame, alternatives: list[pd.DataFrame]) -> pd.DataFrame:
    """Compare alternate-window medoid paths with the chosen-window paths."""
    rows: list[dict[str, object]] = []
    keys = [*PHASE_GROUP_COLUMNS, "mode", "time"]
    for alternative in alternatives:
        merged = reference.merge(
            alternative,
            on=keys,
            how="inner",
            suffixes=("_reference", "_alternate"),
            validate="one_to_one",
        )
        sensitivity_groups = [*PHASE_GROUP_COLUMNS, "mode"]
        for group_keys, group in merged.groupby(sensitivity_groups, sort=True, dropna=False):
            net_reference = group["net_flux_reference"].to_numpy(dtype=float)
            net_alternate = group["net_flux_alternate"].to_numpy(dtype=float)
            churn_reference = group["churn_flux_reference"].to_numpy(dtype=float)
            churn_alternate = group["churn_flux_alternate"].to_numpy(dtype=float)
            item = dict(zip(sensitivity_groups, group_keys))
            item.update(
                {
                    "reference_window": int(group["flux_window_reference"].iloc[0]),
                    "alternate_window": int(group["flux_window_alternate"].iloc[0]),
                    "reference_medoid_run_id": str(group["medoid_run_id_reference"].iloc[0]),
                    "alternate_medoid_run_id": str(group["medoid_run_id_alternate"].iloc[0]),
                    "same_medoid": bool(
                        group["medoid_run_id_reference"].iloc[0]
                        == group["medoid_run_id_alternate"].iloc[0]
                    ),
                    "net_flux_spearman": _stable_spearman(net_reference, net_alternate),
                    "churn_flux_spearman": _stable_spearman(churn_reference, churn_alternate),
                    "net_flux_sign_agreement": float(
                        np.mean(np.sign(net_reference) == np.sign(net_alternate))
                    ),
                    "churn_activity_agreement": float(
                        np.mean((churn_reference > 0) == (churn_alternate > 0))
                    ),
                    "median_abs_net_flux_difference": float(
                        np.median(np.abs(net_reference - net_alternate))
                    ),
                    "median_abs_churn_flux_difference": float(
                        np.median(np.abs(churn_reference - churn_alternate))
                    ),
                    "max_abs_inventory_difference": float(
                        np.max(
                            np.abs(
                                group["node_ratio_reference"].to_numpy(dtype=float)
                                - group["node_ratio_alternate"].to_numpy(dtype=float)
                            )
                        )
                    ),
                }
            )
            rows.append(item)
    return pd.DataFrame(rows)


def summarise_templates(rows: pd.DataFrame) -> pd.DataFrame:
    group = ["series", "series_label", "ambiguity_threshold", "ambiguity_decay", "level", "level_label"]
    seed_means = (
        rows.groupby([*group, "run_id", "seed"], dropna=False)["template_length"]
        .mean()
        .reset_index()
    )
    summary = (
        seed_means.groupby(group, dropna=False)["template_length"]
        .agg(["count", "mean", "std"])
        .reset_index()
        .rename(columns={"count": "n_runs", "std": "sd"})
    )
    summary["sd"] = summary["sd"].fillna(0.0)
    summary["ci95"] = _ci95(summary["sd"], summary["n_runs"])
    node_stats = (
        rows.groupby(group, dropna=False)["template_length"]
        .agg(["count", "median", "min", "max"])
        .reset_index()
        .rename(columns={"count": "n_nodes", "median": "node_median", "min": "node_min", "max": "node_max"})
    )
    return summary.merge(node_stats, on=group, how="left")


def _apply_style() -> None:
    plt.style.use("default")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.edgecolor": "#2F3740",
            "axes.labelcolor": "#1F2933",
            "xtick.color": "#1F2933",
            "ytick.color": "#1F2933",
        }
    )


def _levels(frame: pd.DataFrame) -> list[int]:
    present = set(int(value) for value in frame["level"].unique())
    ordered = [level for level in DEFAULT_LEVEL_ORDER if level in present]
    ordered.extend(sorted(present.difference(ordered)))
    return ordered


def _linestyle(series: str) -> str | tuple[int, tuple[float, ...]]:
    return "-" if series.endswith("decay_0p99") else (0, (5.5, 2.2))


def _phase_segments(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    points = np.column_stack([x, y])
    return np.stack([points[:-1], points[1:]], axis=1)


def _categorical_ordered_colours(
    base_color: str,
    values: np.ndarray,
    norm: matplotlib.colors.Normalize,
) -> np.ndarray:
    """Combine categorical hue with ordered lightness for a product encoding."""
    base = np.asarray(matplotlib.colors.to_rgb(base_color), dtype=float)
    light = 0.72 * np.ones(3) + 0.28 * base
    dark = 0.58 * base
    scaled = np.asarray(norm(values), dtype=float)[:, None]
    rgb = (1.0 - scaled) * light + scaled * dark
    return np.column_stack([rgb, np.ones(len(rgb))])


def _direction_segment(x: np.ndarray, y: np.ndarray, x_span: float, y_span: float) -> int | None:
    distances = np.hypot(np.diff(x) / max(x_span, 1e-12), np.diff(y) / max(y_span, 1e-12))
    total = float(distances.sum())
    if total <= 1e-12:
        return None
    return int(np.searchsorted(np.cumsum(distances), 0.55 * total))


def _direction_arrows(
    x: np.ndarray,
    y: np.ndarray,
    x_span: float,
    y_span: float,
    fractions: tuple[float, ...] = (0.22, 0.50, 0.78),
) -> list[tuple[int, int]]:
    """Return separated vertex pairs for clear arrows along a phase path."""
    distances = np.hypot(np.diff(x) / max(x_span, 1e-12), np.diff(y) / max(y_span, 1e-12))
    cumulative = np.concatenate([[0.0], np.cumsum(distances)])
    total = float(cumulative[-1])
    if total <= 1e-12:
        return []
    arrows: list[tuple[int, int]] = []
    for fraction in fractions:
        start = int(np.searchsorted(cumulative, fraction * total, side="left"))
        end = int(np.searchsorted(cumulative, min((fraction + 0.07) * total, total), side="left"))
        start = min(start, len(x) - 2)
        end = min(max(end, start + 1), len(x) - 1)
        pair = (start, end)
        if pair not in arrows:
            arrows.append(pair)
    return arrows


def _direction_arrows_3d(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    x_span: float,
    y_span: float,
    z_span: float,
    fractions: tuple[float, ...] = (0.22, 0.50, 0.78),
) -> list[tuple[int, int]]:
    distances = np.sqrt(
        (np.diff(x) / max(x_span, 1e-12)) ** 2
        + (np.diff(y) / max(y_span, 1e-12)) ** 2
        + (np.diff(z) / max(z_span, 1e-12)) ** 2
    )
    cumulative = np.concatenate([[0.0], np.cumsum(distances)])
    total = float(cumulative[-1])
    if total <= 1e-12:
        return []
    arrows: list[tuple[int, int]] = []
    for fraction in fractions:
        start = int(np.searchsorted(cumulative, fraction * total, side="left"))
        end = int(np.searchsorted(cumulative, min((fraction + 0.075) * total, total), side="left"))
        start = min(start, len(x) - 2)
        end = min(max(end, start + 1), len(x) - 1)
        if (start, end) not in arrows:
            arrows.append((start, end))
    return arrows


def plot_structural_phase_portrait(
    seed_rows: pd.DataFrame,
    summary: pd.DataFrame,
    plot_dir: Path,
) -> list[Path]:
    if seed_rows.empty or summary.empty:
        raise ValueError("No structural phase trajectories were available to plot.")
    _apply_style()
    levels = _levels(summary)
    figure, axes = plt.subplots(
        1,
        len(levels),
        figsize=(7.16, 2.65),
        sharex=True,
        sharey=True,
        dpi=300,
        gridspec_kw={"wspace": 0.13},
    )
    axes = np.ravel(np.atleast_1d(axes))

    x_values = summary["node_ratio"].to_numpy(dtype=float)
    x_padding = 0.06 * max(float(np.ptp(x_values)), 1e-6)
    x_limits = (max(0.0, float(x_values.min()) - x_padding), float(x_values.max()) + x_padding)
    y_values = CHURN_DISPLAY_STEPS * summary["churn_flux"].to_numpy(dtype=float)
    y_limits = (0.0, max(float(y_values.max()) * 1.1, 1e-6))
    x_span = x_limits[1] - x_limits[0]
    y_span = y_limits[1] - y_limits[0]
    checkpoint_sizes = dict(zip(TIME_CHECKPOINTS, np.linspace(7.0, 21.0, len(TIME_CHECKPOINTS))))

    for axis, level in zip(axes, levels):
        panel_summary = summary[summary["level"] == level]
        for threshold in sorted(THRESHOLD_COLORS):
            color = THRESHOLD_COLORS[threshold]
            for decay in (0.99, 1.0):
                condition = panel_summary[
                    (panel_summary["ambiguity_threshold"] == threshold)
                    & np.isclose(panel_summary["ambiguity_decay"], decay)
                ]
                if condition.empty:
                    continue
                for mode, trajectory in condition.groupby("mode", sort=True):
                    trajectory = trajectory.sort_values("time")
                    x = trajectory["node_ratio"].to_numpy(dtype=float)
                    y = CHURN_DISPLAY_STEPS * trajectory["churn_flux"].to_numpy(dtype=float)
                    churn = trajectory["churn_flux"].to_numpy(dtype=float)
                    selected = (threshold, decay) == SELECTED_CONFIGURATION
                    if mode == "inactive" and np.allclose(churn, 0.0):
                        axis.scatter(
                            x[0],
                            0.0,
                            s=25,
                            marker="X",
                            facecolor=color,
                            edgecolor="white",
                            linewidth=0.45,
                            zorder=4,
                        )
                        continue
                    segments = _phase_segments(x, y)
                    linestyle = "solid" if np.isclose(decay, 0.99) else (0, (3.2, 2.0))
                    if selected:
                        halo = LineCollection(
                            segments,
                            colors="#111827",
                            linewidths=1.35,
                            linestyles=linestyle,
                            alpha=0.68,
                            zorder=1,
                        )
                        axis.add_collection(halo)
                    collection = LineCollection(
                        segments,
                        colors=color,
                        linewidths=0.72,
                        linestyles=linestyle,
                        alpha=0.82 if mode in {"all", "active"} else 0.62,
                        zorder=2,
                    )
                    axis.add_collection(collection)
                    checkpoints = trajectory[trajectory["time"].isin(TIME_CHECKPOINTS)]
                    for item in checkpoints.itertuples():
                        axis.scatter(
                            item.node_ratio,
                            CHURN_DISPLAY_STEPS * item.churn_flux,
                            s=(
                                checkpoint_sizes[int(item.time)]
                                * THRESHOLD_MARKER_SCALE[threshold]
                                * (1.32 if np.isclose(decay, 1.0) else 1.0)
                            ),
                            marker=THRESHOLD_MARKERS[threshold],
                            facecolor=color if np.isclose(decay, 0.99) else "none",
                            edgecolor="#111827" if selected else color,
                            linewidth=0.72 if selected else 0.48,
                            alpha=0.96,
                            zorder=3,
                        )
                    direction = _direction_segment(x, y, x_span, y_span)
                    if direction is not None:
                        axis.annotate(
                            "",
                            xy=(x[direction + 1], y[direction + 1]),
                            xytext=(x[direction], y[direction]),
                            arrowprops={
                                "arrowstyle": "-|>",
                                "color": "#111827" if selected else color,
                                "linewidth": 0.55,
                                "mutation_scale": 5.5,
                                "alpha": 0.82,
                            },
                            zorder=4,
                        )

        if level == 22:
            active_mode = panel_summary[
                (panel_summary["ambiguity_threshold"] == 40)
                & np.isclose(panel_summary["ambiguity_decay"], 0.99)
                & (panel_summary["mode"] == "active")
            ].iloc[0]
            inactive_mode = panel_summary[
                (panel_summary["ambiguity_threshold"] == 40)
                & np.isclose(panel_summary["ambiguity_decay"], 0.99)
                & (panel_summary["mode"] == "inactive")
            ].iloc[0]
            axis.text(
                0.98,
                0.96,
                f"active {int(active_mode['mode_n'])}/30 "
                f"($T_{{act}}$ IQR {active_mode['activation_time_q25']:.0f}--{active_mode['activation_time_q75']:.0f})\n"
                f"inactive {int(inactive_mode['mode_n'])}/30",
                transform=axis.transAxes,
                ha="right",
                va="top",
                fontsize=5.6,
                color="#4B5563",
            )

        axis.axvline(
            1.0,
            color=REFERENCE_COLOR,
            linewidth=0.65,
            linestyle=(0, (2.0, 2.2)),
            alpha=0.75,
            zorder=0,
        )
        axis.axhline(0.0, color=REFERENCE_COLOR, linewidth=0.65, alpha=0.65, zorder=0)
        level_title = LEVEL_LABELS.get(level, f"L{level}").replace("x", r"\times")
        axis.set_title(rf"${level_title}$", fontsize=8.2, fontweight="bold", pad=3)
        axis.set_xlim(*x_limits)
        axis.set_ylim(*y_limits)
        axis.grid(axis="both", color="#DDE3EA", linewidth=0.4, alpha=0.42)
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(axis="both", labelsize=6.0, length=2.2, pad=1.5)

    threshold_handles = [
        Line2D(
            [0], [0], color=color, linewidth=1.4, marker=THRESHOLD_MARKERS[threshold],
            markersize=3.0, label=rf"$\theta={threshold}$"
        )
        for threshold, color in THRESHOLD_COLORS.items()
    ]
    decay_handles = [
        Line2D(
            [0], [0], color="#4B5563", linewidth=1.5, linestyle="solid", marker="o",
            markersize=3.2, markerfacecolor="#4B5563", label=r"$\gamma=0.99$"
        ),
        Line2D(
            [0], [0], color="#4B5563", linewidth=1.5, linestyle=(0, (3.2, 2.0)), marker="o",
            markersize=3.2, markerfacecolor="white", label=r"$\gamma=1.0$"
        ),
    ]
    handles = [*threshold_handles, *decay_handles]
    figure.legend(
        handles,
        [handle.get_label() for handle in handles],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=6,
        frameon=False,
        fontsize=6.0,
        handlelength=2.2,
        handletextpad=0.4,
        columnspacing=1.0,
        borderaxespad=0,
    )
    figure.supxlabel(r"Graph inventory $N(t)/N_{\rm GT}$", fontsize=7.5, y=0.112)
    figure.supylabel(
        rf"Structural churn per GT state per {CHURN_DISPLAY_STEPS} steps",
        fontsize=7.2,
        x=0.008,
    )
    figure.text(
        0.5,
        0.018,
        rf"Observed trajectory medoids; {FLUX_WINDOW}-step local rates; markers at $t={','.join(str(value) for value in TIME_CHECKPOINTS)}$ grow with time; arrows show direction.",
        ha="center",
        va="bottom",
        fontsize=5.5,
        color="#4B5563",
    )
    figure.subplots_adjust(left=0.082, right=0.995, top=0.79, bottom=0.25)
    outputs = [
        plot_dir / "mercury_structural_trajectories.pdf",
        plot_dir / "mercury_structural_trajectories.png",
    ]
    for output in outputs:
        figure.savefig(output, bbox_inches="tight")
    plt.close(figure)
    return outputs


def plot_structural_phase_portrait_3d(
    summary: pd.DataFrame,
    plot_dir: Path,
) -> list[Path]:
    """Render the canonical four-environment portrait with explicit time."""
    if summary.empty:
        raise ValueError("No structural phase trajectories were available to plot.")
    _apply_style()
    levels = _levels(summary)
    figure, axes = plt.subplots(
        1,
        len(levels),
        figsize=(7.16, 3.15),
        subplot_kw={"projection": "3d"},
        dpi=300,
        gridspec_kw={"wspace": 0.01},
    )
    axes = np.ravel(np.atleast_1d(axes))

    x_values = summary["node_ratio"].to_numpy(dtype=float)
    x_padding = 0.05 * max(float(np.ptp(x_values)), 1e-6)
    x_limits = (max(0.0, float(x_values.min()) - x_padding), float(x_values.max()) + x_padding)
    y_values = CHURN_DISPLAY_STEPS * summary["churn_flux"].to_numpy(dtype=float)
    y_limits = (0.0, max(float(y_values.max()) * 1.10, 1e-6))
    z_limits = (0.0, float(summary["time"].max()))
    x_span = x_limits[1] - x_limits[0]
    y_span = y_limits[1] - y_limits[0]
    z_span = z_limits[1] - z_limits[0]

    for panel_index, (axis, level) in enumerate(zip(axes, levels)):
        panel = summary[summary["level"] == level]
        for threshold in sorted(THRESHOLD_COLORS):
            color = THRESHOLD_COLORS[threshold]
            for decay in (0.99, 1.0):
                condition = panel[
                    (panel["ambiguity_threshold"] == threshold)
                    & np.isclose(panel["ambiguity_decay"], decay)
                ]
                selected = (threshold, decay) == SELECTED_CONFIGURATION
                linestyle = "solid" if np.isclose(decay, 0.99) else (0, (3.2, 2.0))
                for mode, trajectory in condition.groupby("mode", sort=True):
                    trajectory = trajectory.sort_values("time")
                    x = trajectory["node_ratio"].to_numpy(dtype=float)
                    y = CHURN_DISPLAY_STEPS * trajectory["churn_flux"].to_numpy(dtype=float)
                    z = trajectory["time"].to_numpy(dtype=float)
                    alpha = 0.92 if mode in {"all", "active"} else 0.52
                    if selected:
                        axis.plot(
                            x, y, z, color="#111827", linewidth=1.45,
                            linestyle=linestyle, alpha=0.60, zorder=2,
                        )
                    axis.plot(
                        x, y, z, color=color, linewidth=0.78,
                        linestyle=linestyle, alpha=alpha, zorder=3,
                    )
                    for _, arrow_end in _direction_arrows_3d(
                        x, y, z, x_span, y_span, z_span, fractions=(0.34, 0.70)
                    ):
                        axis.scatter(
                            x[arrow_end], y[arrow_end], z[arrow_end],
                            color="#111827" if selected else color,
                            marker="^", s=9.5, linewidth=0.22,
                            edgecolor="white", depthshade=False, zorder=4,
                        )

        if level == 22:
            active = panel[
                (panel["ambiguity_threshold"] == 40)
                & np.isclose(panel["ambiguity_decay"], 0.99)
                & (panel["mode"] == "active")
            ].iloc[0]
            inactive = panel[
                (panel["ambiguity_threshold"] == 40)
                & np.isclose(panel["ambiguity_decay"], 0.99)
                & (panel["mode"] == "inactive")
            ].iloc[0]
            axis.text2D(
                0.98,
                0.97,
                f"active {int(active['mode_n'])}/30\ninactive {int(inactive['mode_n'])}/30",
                transform=axis.transAxes,
                ha="right",
                va="top",
                fontsize=4.5,
                color="#4B5563",
            )

        parity_y = np.linspace(*y_limits, 2)
        parity_z = np.linspace(*z_limits, 2)
        parity_y_grid, parity_z_grid = np.meshgrid(parity_y, parity_z)
        axis.plot_surface(
            np.ones_like(parity_y_grid), parity_y_grid, parity_z_grid,
            color=REFERENCE_COLOR, alpha=0.045, linewidth=0, shade=False,
        )
        axis.set_xlim(*x_limits)
        axis.set_ylim(*y_limits)
        axis.set_zlim(*z_limits)
        axis.set_zticks([0, 1000, 2000])
        axis.view_init(elev=23, azim=-57)
        axis.set_box_aspect((1.08, 0.80, 1.0))
        axis.tick_params(axis="both", labelsize=4.2, pad=-1.0, length=1.5)
        level_title = LEVEL_LABELS.get(level, f"L{level}").replace("x", r"\times")
        axis.set_title(rf"${level_title}$", fontsize=7.2, fontweight="bold", pad=-1)
        axis.set_xlabel(r"$N/N_{\rm GT}$", fontsize=5.2, labelpad=-5)
        axis.set_ylabel(rf"churn/{CHURN_DISPLAY_STEPS}", fontsize=5.0, labelpad=-6)
        if panel_index == len(levels) - 1:
            axis.set_zlabel("step", fontsize=5.0, labelpad=-7)
        else:
            axis.set_zticklabels([])
        for pane in (axis.xaxis.pane, axis.yaxis.pane, axis.zaxis.pane):
            pane.set_facecolor((1.0, 1.0, 1.0, 0.0))
            pane.set_edgecolor("#DDE3EA")
        axis.grid(True)

    threshold_handles = [
        Line2D([0], [0], color=color, linewidth=1.5, label=rf"$\theta={threshold}$")
        for threshold, color in THRESHOLD_COLORS.items()
    ]
    decay_handles = [
        Line2D([0], [0], color="#4B5563", linewidth=1.4, linestyle="solid", label=r"$\gamma=0.99$"),
        Line2D([0], [0], color="#4B5563", linewidth=1.4, linestyle=(0, (3.2, 2.0)), label=r"$\gamma=1.0$"),
    ]
    handles = [*threshold_handles, *decay_handles]
    figure.legend(
        handles,
        [handle.get_label() for handle in handles],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=6,
        frameon=False,
        fontsize=5.7,
        handlelength=2.0,
        handletextpad=0.35,
        columnspacing=0.85,
        borderaxespad=0,
    )
    figure.text(
        0.5,
        0.012,
        f"Observed trajectory medoids; {FLUX_WINDOW}-step local churn sampled every "
        f"{FLUX_SAMPLE_INTERVAL} steps; triangles point forward in learning time.",
        ha="center",
        va="bottom",
        fontsize=5.1,
        color="#4B5563",
    )
    figure.subplots_adjust(left=0.005, right=0.995, top=0.90, bottom=0.09)
    outputs = [
        plot_dir / "mercury_structural_trajectories.pdf",
        plot_dir / "mercury_structural_trajectories.png",
    ]
    for output in outputs:
        figure.savefig(output, bbox_inches="tight")
    plt.close(figure)
    return outputs


def plot_structural_phase_diagnostic_grid(
    summary: pd.DataFrame,
    plot_dir: Path,
) -> list[Path]:
    """Plot every environment/configuration combination in its own phase panel."""
    if summary.empty:
        raise ValueError("No structural phase trajectories were available to plot.")
    _apply_style()
    levels = _levels(summary)
    configurations = [
        (threshold, decay)
        for threshold in sorted(THRESHOLD_COLORS)
        for decay in (0.99, 1.0)
    ]
    figure, axes = plt.subplots(
        len(levels),
        len(configurations),
        figsize=(12.8, 7.0),
        sharex=True,
        sharey=True,
        dpi=300,
        gridspec_kw={"hspace": 0.18, "wspace": 0.10},
    )
    axes = np.asarray(axes).reshape(len(levels), len(configurations))

    x_values = summary["node_ratio"].to_numpy(dtype=float)
    x_padding = 0.05 * max(float(np.ptp(x_values)), 1e-6)
    x_limits = (max(0.0, float(x_values.min()) - x_padding), float(x_values.max()) + x_padding)
    y_values = CHURN_DISPLAY_STEPS * summary["churn_flux"].to_numpy(dtype=float)
    y_max = max(float(y_values.max()) * 1.10, 1e-6)
    # A small empty margin below zero keeps quiescent trajectories off the axis spine.
    y_limits = (-0.035 * y_max, y_max)
    x_span = x_limits[1] - x_limits[0]
    y_span = y_limits[1] - y_limits[0]
    for row_index, level in enumerate(levels):
        for col_index, (threshold, decay) in enumerate(configurations):
            axis = axes[row_index, col_index]
            color = THRESHOLD_COLORS[threshold]
            condition = summary[
                (summary["level"] == level)
                & (summary["ambiguity_threshold"] == threshold)
                & np.isclose(summary["ambiguity_decay"], decay)
            ]
            selected = (threshold, decay) == SELECTED_CONFIGURATION
            mode_counts = condition.groupby("mode")["mode_n"].first().to_dict()
            for mode, trajectory in condition.groupby("mode", sort=True):
                trajectory = trajectory.sort_values("time")
                x = trajectory["node_ratio"].to_numpy(dtype=float)
                y = CHURN_DISPLAY_STEPS * trajectory["churn_flux"].to_numpy(dtype=float)
                if mode == "inactive" and np.allclose(y, 0.0):
                    axis.scatter(
                        x[0], 0.0, s=24, marker="X", facecolor=color,
                        edgecolor="#111827", linewidth=0.45, zorder=5,
                    )
                    continue
                linestyle = "solid" if np.isclose(decay, 0.99) else (0, (3.2, 2.0))
                segments = _phase_segments(x, y)
                if selected:
                    axis.add_collection(
                        LineCollection(
                            segments, colors="#111827", linewidths=1.45,
                            linestyles=linestyle, alpha=0.62, zorder=2,
                        )
                    )
                axis.add_collection(
                    LineCollection(
                        segments, colors=color, linewidths=0.82,
                        linestyles=linestyle,
                        alpha=0.95 if mode in {"all", "active"} else 0.65,
                        zorder=3,
                    )
                )
                for arrow_start, arrow_end in _direction_arrows(x, y, x_span, y_span):
                    axis.annotate(
                        "",
                        xy=(x[arrow_end], y[arrow_end]),
                        xytext=(x[arrow_start], y[arrow_start]),
                        arrowprops={
                            "arrowstyle": "-|>", "color": "#111827" if selected else color,
                            "linewidth": 0.85, "mutation_scale": 8.0, "alpha": 0.95,
                            "shrinkA": 0.0, "shrinkB": 0.0,
                        },
                        zorder=5,
                    )

            if set(mode_counts) != {"all"}:
                axis.text(
                    0.97,
                    0.92,
                    "\n".join(f"{mode} {int(count)}/30" for mode, count in sorted(mode_counts.items())),
                    transform=axis.transAxes,
                    ha="right",
                    va="top",
                    fontsize=5.2,
                    color="#4B5563",
                )

            axis.axvline(
                1.0, color=REFERENCE_COLOR, linewidth=0.55,
                linestyle=(0, (2.0, 2.2)), alpha=0.7, zorder=0,
            )
            axis.axhline(0.0, color=REFERENCE_COLOR, linewidth=0.5, alpha=0.55, zorder=0)
            axis.set_xlim(*x_limits)
            axis.set_ylim(*y_limits)
            axis.grid(axis="both", color="#DDE3EA", linewidth=0.35, alpha=0.40)
            axis.spines[["top", "right"]].set_visible(False)
            axis.tick_params(axis="both", labelsize=5.2, length=1.8, pad=1.2)
            if selected:
                axis.set_facecolor("#F8FAFC")
            if row_index == 0:
                axis.set_title(
                    rf"$\theta={threshold}$" + "\n" + rf"$\gamma={decay:g}$",
                    fontsize=6.3,
                    color=color,
                    fontweight="bold" if selected else "normal",
                    pad=2.5,
                )
            if col_index == 0:
                level_title = LEVEL_LABELS.get(level, f"L{level}").replace("x", r"\times")
                axis.set_ylabel(rf"${level_title}$", fontsize=7.0, fontweight="bold", labelpad=7)

    figure.supxlabel(r"Graph inventory $N(t)/N_{\rm GT}$", fontsize=8.2, y=0.025)
    figure.supylabel(
        rf"Structural churn per GT state per {CHURN_DISPLAY_STEPS} steps",
        fontsize=8.0,
        x=0.008,
    )
    figure.text(
        0.5,
        0.992,
        f"Diagnostic configuration grid: observed trajectory medoids sampled every {FLUX_SAMPLE_INTERVAL} steps; arrows show time direction",
        ha="center",
        va="top",
        fontsize=8.2,
        color="#374151",
    )
    figure.subplots_adjust(left=0.065, right=0.995, top=0.91, bottom=0.085)
    outputs = [
        plot_dir / "mercury_structural_trajectories_diagnostic_grid.pdf",
        plot_dir / "mercury_structural_trajectories_diagnostic_grid.png",
    ]
    for output in outputs:
        figure.savefig(output, bbox_inches="tight")
    plt.close(figure)
    return outputs


def plot_structural_phase_diagnostic_combined(
    summary: pd.DataFrame,
    plot_dir: Path,
) -> list[Path]:
    """Overlay all configurations in one inventory--churn panel per environment."""
    if summary.empty:
        raise ValueError("No structural phase trajectories were available to plot.")
    _apply_style()
    levels = _levels(summary)
    figure, axes = plt.subplots(
        1,
        len(levels),
        figsize=(7.16, 2.65),
        sharex=True,
        sharey=True,
        dpi=300,
        gridspec_kw={"wspace": 0.13},
    )
    axes = np.ravel(np.atleast_1d(axes))
    x_values = summary["node_ratio"].to_numpy(dtype=float)
    x_padding = 0.05 * max(float(np.ptp(x_values)), 1e-6)
    x_limits = (max(0.0, float(x_values.min()) - x_padding), float(x_values.max()) + x_padding)
    y_values = CHURN_DISPLAY_STEPS * summary["churn_flux"].to_numpy(dtype=float)
    y_max = max(float(y_values.max()) * 1.10, 1e-6)
    y_limits = (-0.025 * y_max, y_max)
    x_span = x_limits[1] - x_limits[0]
    y_span = y_limits[1] - y_limits[0]

    for axis, level in zip(axes, levels):
        panel = summary[summary["level"] == level]
        for threshold in sorted(THRESHOLD_COLORS):
            color = THRESHOLD_COLORS[threshold]
            for decay in (0.99, 1.0):
                selected = (threshold, decay) == SELECTED_CONFIGURATION
                linestyle = "solid" if np.isclose(decay, 0.99) else (0, (3.2, 2.0))
                condition = panel[
                    (panel["ambiguity_threshold"] == threshold)
                    & np.isclose(panel["ambiguity_decay"], decay)
                ]
                for mode, trajectory in condition.groupby("mode", sort=True):
                    trajectory = trajectory.sort_values("time")
                    x = trajectory["node_ratio"].to_numpy(dtype=float)
                    y = CHURN_DISPLAY_STEPS * trajectory["churn_flux"].to_numpy(dtype=float)
                    segments = _phase_segments(x, y)
                    alpha = 0.90 if mode in {"all", "active"} else 0.45
                    if selected:
                        axis.add_collection(
                            LineCollection(
                                segments, colors="#111827", linewidths=1.60,
                                linestyles=linestyle, alpha=0.64, zorder=1,
                            )
                        )
                    axis.add_collection(
                        LineCollection(
                            segments, colors=color, linewidths=0.78,
                            linestyles=linestyle, alpha=alpha, zorder=2,
                        )
                    )
                    for arrow_start, arrow_end in _direction_arrows(
                        x, y, x_span, y_span, fractions=(0.58,)
                    ):
                        axis.annotate(
                            "",
                            xy=(x[arrow_end], y[arrow_end]),
                            xytext=(x[arrow_start], y[arrow_start]),
                            arrowprops={
                                "arrowstyle": "-|>",
                                "color": "#111827" if selected else color,
                                "linewidth": 0.72,
                                "mutation_scale": 7.0,
                                "alpha": 0.90,
                                "shrinkA": 0.0,
                                "shrinkB": 0.0,
                            },
                            zorder=4,
                        )

        if level == 22:
            active = panel[
                (panel["ambiguity_threshold"] == 40)
                & np.isclose(panel["ambiguity_decay"], 0.99)
                & (panel["mode"] == "active")
            ].iloc[0]
            inactive = panel[
                (panel["ambiguity_threshold"] == 40)
                & np.isclose(panel["ambiguity_decay"], 0.99)
                & (panel["mode"] == "inactive")
            ].iloc[0]
            axis.text(
                0.98, 0.96,
                rf"$\theta=40,\gamma=.99$: {int(active['mode_n'])}/30 active"
                + "\n"
                + f"{int(inactive['mode_n'])}/30 inactive",
                transform=axis.transAxes, ha="right", va="top",
                fontsize=4.5, color="#4B5563",
            )
        axis.axvline(
            1.0, color=REFERENCE_COLOR, linewidth=0.58,
            linestyle=(0, (2.0, 2.2)), alpha=0.72, zorder=0,
        )
        axis.axhline(0.0, color=REFERENCE_COLOR, linewidth=0.52, alpha=0.62, zorder=0)
        axis.set_xlim(*x_limits)
        axis.set_ylim(*y_limits)
        axis.grid(axis="both", color="#DDE3EA", linewidth=0.35, alpha=0.42)
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(axis="both", labelsize=5.4, length=1.8, pad=1.2)
        level_title = LEVEL_LABELS.get(level, f"L{level}").replace("x", r"\times")
        axis.set_title(rf"${level_title}$", fontsize=7.2, fontweight="bold", pad=3)
        axis.text(
            1.0, 0.025, "state-count parity",
            transform=axis.get_xaxis_transform(), rotation=90,
            ha="right", va="bottom", fontsize=4.0, color="#64748B",
        )

    threshold_handles = [
        Line2D([0], [0], color=color, linewidth=1.7, label=rf"$\theta={threshold}$")
        for threshold, color in THRESHOLD_COLORS.items()
    ]
    decay_handles = [
        Line2D([0], [0], color="#4B5563", linewidth=1.5, linestyle="solid", label=r"$\gamma=0.99$"),
        Line2D([0], [0], color="#4B5563", linewidth=1.5, linestyle=(0, (3.2, 2.0)), label=r"$\gamma=1.0$"),
    ]
    handles = [*threshold_handles, *decay_handles]
    figure.legend(
        handles, [handle.get_label() for handle in handles],
        loc="upper center", bbox_to_anchor=(0.5, 0.995), ncol=6,
        frameon=False, fontsize=5.7, handlelength=2.0,
        handletextpad=0.35, columnspacing=0.85, borderaxespad=0,
    )
    figure.supxlabel(r"Graph inventory $N(t)/N_{\rm GT}$", fontsize=7.0, y=0.095)
    figure.supylabel(
        f"Gross structural churn per GT state per {CHURN_DISPLAY_STEPS} steps",
        fontsize=6.4, x=0.008,
    )
    figure.text(
        0.5, 0.02,
        "All configurations overlaid; observed trajectory medoids; arrows show increasing learning time.",
        ha="center", va="bottom", fontsize=4.9, color="#4B5563",
    )
    figure.subplots_adjust(left=0.085, right=0.995, top=0.82, bottom=0.22)
    outputs = [
        plot_dir / "mercury_structural_trajectories_diagnostic_combined.pdf",
        plot_dir / "mercury_structural_trajectories_diagnostic_combined.png",
    ]
    for output in outputs:
        figure.savefig(output, bbox_inches="tight")
    plt.close(figure)
    return outputs


def plot_structural_flux_phase_space(
    summary: pd.DataFrame,
    plot_dir: Path,
    seed_rows: pd.DataFrame | None = None,
    *,
    flux_limit_override: float | None = None,
    output_stem: str = "mercury_structural_flux_phase_space",
    scale_note: str = "",
) -> list[Path]:
    """Plot median net structural flux against median gross turnover."""
    if summary.empty:
        raise ValueError("No structural flux summaries were available to plot.")
    _apply_style()
    levels = _levels(summary)
    figure, axes = plt.subplots(
        1,
        len(levels),
        figsize=(7.16, 2.72),
        sharex=True,
        sharey=True,
        dpi=300,
        gridspec_kw={"wspace": 0.12},
    )
    axes = np.ravel(np.atleast_1d(axes))
    coordinate_candidates = [
        float(summary["gross_turnover_50"].max()),
        float(np.abs(summary["net_flux_50"]).max()),
    ]
    coordinate_max = max(*coordinate_candidates, 1e-6)
    flux_limit = (
        float(flux_limit_override)
        if flux_limit_override is not None
        else 1.08 * coordinate_max
    )
    if flux_limit <= 0:
        raise ValueError("Structural flux axis limit must be positive.")
    x_limits = (-flux_limit, flux_limit)
    y_limits = (-0.025 * flux_limit, flux_limit)

    for panel_index, (axis, level) in enumerate(zip(axes, levels)):
        panel = summary[(summary["level"] == level) & (summary["scope"] == "all_seeds")]
        for threshold in sorted(THRESHOLD_COLORS):
            color = THRESHOLD_COLORS[threshold]
            for decay in (0.99, 1.0):
                trajectory = panel[
                    (panel["ambiguity_threshold"] == threshold)
                    & np.isclose(panel["ambiguity_decay"], decay)
                ].sort_values("time")
                if trajectory.empty:
                    continue
                x = trajectory["net_flux_50"].to_numpy(dtype=float)
                y = trajectory["gross_turnover_50"].to_numpy(dtype=float)
                segments = _phase_segments(x, y)
                linestyle = "solid" if np.isclose(decay, 0.99) else (0, (3.2, 2.0))
                selected = (threshold, decay) == SELECTED_CONFIGURATION
                if selected:
                    axis.add_collection(
                        LineCollection(
                            segments, colors="#111827", linewidths=1.55,
                            linestyles=linestyle, alpha=0.58, zorder=2,
                        )
                    )
                base_rgba = np.asarray(matplotlib.colors.to_rgba(color), dtype=float)
                segment_colours = np.tile(base_rgba, (len(segments), 1))
                segment_colours[:, 3] = np.linspace(0.24, 0.94, len(segments))
                axis.add_collection(
                    LineCollection(
                        segments, colors=segment_colours, linewidths=0.82,
                        linestyles=linestyle, zorder=3,
                    )
                )
                for arrow_start, arrow_end in _direction_arrows(
                    x, y, 2 * flux_limit, flux_limit, fractions=(0.62,)
                ):
                    axis.annotate(
                        "",
                        xy=(x[arrow_end], y[arrow_end]),
                        xytext=(x[arrow_start], y[arrow_start]),
                        arrowprops={
                            "arrowstyle": "-|>",
                            "color": "#111827" if selected else color,
                            "linewidth": 0.68,
                            "mutation_scale": 6.8,
                            "alpha": 0.90,
                            "shrinkA": 0.0,
                            "shrinkB": 0.0,
                        },
                        zorder=5,
                    )
                axis.scatter(
                    x[-1], y[-1], s=8.0, marker="o", facecolor=color,
                    edgecolor="#111827" if selected else "white",
                    linewidth=0.42, alpha=0.95, zorder=5,
                )

        if level == 22:
            active = summary[
                (summary["level"] == 22)
                & (summary["ambiguity_threshold"] == 40)
                & np.isclose(summary["ambiguity_decay"], 0.99)
                & (summary["scope"] == "active_conditional")
            ].sort_values("time")
            if not active.empty:
                x = active["net_flux_50"].to_numpy(dtype=float)
                y = active["gross_turnover_50"].to_numpy(dtype=float)
                segments = _phase_segments(x, y)
                rgba = np.tile(
                    np.asarray(matplotlib.colors.to_rgba(THRESHOLD_COLORS[40]), dtype=float),
                    (len(segments), 1),
                )
                rgba[:, 3] = np.linspace(0.30, 0.96, len(segments))
                axis.add_collection(
                    LineCollection(
                        segments, colors=rgba, linewidths=1.18,
                        linestyles=(0, (1.2, 1.2)), zorder=4,
                    )
                )
                peak = active.iloc[int(np.argmax(active["gross_turnover_50"].to_numpy()))]
                active_n = int(active["n_seeds"].iloc[0])
                total_n = int(
                    panel[
                        (panel["ambiguity_threshold"] == 40)
                        & np.isclose(panel["ambiguity_decay"], 0.99)
                    ]["n_seeds"].iloc[0]
                )
                axis.annotate(
                    rf"all-seed median + active conditional ({active_n}/{total_n})",
                    xy=(float(peak["net_flux_50"]), float(peak["gross_turnover_50"])),
                    xycoords="data",
                    xytext=(0.98, 0.96),
                    textcoords="axes fraction",
                    ha="right", va="top", fontsize=4.2, color="#4B5563",
                    arrowprops={
                        "arrowstyle": "-", "color": THRESHOLD_COLORS[40],
                        "linewidth": 0.48, "alpha": 0.65,
                    },
                )

        boundary_x = np.linspace(-flux_limit, flux_limit, 300)
        axis.plot(
            boundary_x[boundary_x <= 0],
            -boundary_x[boundary_x <= 0],
            color=REFERENCE_COLOR, linewidth=0.58,
            linestyle=(0, (2.0, 2.2)), alpha=0.75, zorder=0,
        )
        axis.plot(
            boundary_x[boundary_x >= 0],
            boundary_x[boundary_x >= 0],
            color=REFERENCE_COLOR, linewidth=0.58,
            linestyle=(0, (2.0, 2.2)), alpha=0.75, zorder=0,
        )
        axis.scatter(0.0, 0.0, s=7.0, facecolor="white", edgecolor=REFERENCE_COLOR, linewidth=0.5, zorder=1)
        axis.set_xlim(*x_limits)
        axis.set_ylim(*y_limits)
        axis.set_aspect("equal", adjustable="box")
        axis.grid(axis="both", color="#DDE3EA", linewidth=0.35, alpha=0.40)
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(axis="both", labelsize=5.2, length=1.8, pad=1.2)
        level_title = LEVEL_LABELS.get(level, f"L{level}").replace("x", r"\times")
        axis.set_title(rf"${level_title}$", fontsize=7.2, fontweight="bold", pad=3)
        if panel_index == 0:
            axis.text(0.48 * flux_limit, 0.57 * flux_limit, "Expansion", fontsize=4.2, color="#64748B")
            axis.text(-0.48 * flux_limit, 0.57 * flux_limit, "Pruning", fontsize=4.2, color="#64748B", ha="right")
            axis.text(0.0, 0.86 * flux_limit, "Turnover / replacement", fontsize=4.1, color="#64748B", ha="center")
            axis.text(0.0, 0.035 * flux_limit, "Quiescence", fontsize=4.0, color="#64748B", ha="center", va="bottom")

    threshold_handles = [
        Line2D([0], [0], color=color, linewidth=1.7, label=rf"$\theta={threshold}$")
        for threshold, color in THRESHOLD_COLORS.items()
    ]
    decay_handles = [
        Line2D([0], [0], color="#4B5563", linewidth=1.5, linestyle="solid", label=r"$\gamma=0.99$"),
        Line2D([0], [0], color="#4B5563", linewidth=1.5, linestyle=(0, (3.2, 2.0)), label=r"$\gamma=1.0$"),
    ]
    handles = [*threshold_handles, *decay_handles]
    figure.legend(
        handles, [handle.get_label() for handle in handles],
        loc="upper center", bbox_to_anchor=(0.5, 0.995), ncol=6,
        frameon=False, fontsize=5.7, handlelength=2.0,
        handletextpad=0.35, columnspacing=0.85, borderaxespad=0,
    )
    figure.supxlabel(
        r"Net structural flux $(B_{50}-D_{50})/N_{\rm GT}$",
        fontsize=6.8, y=0.088,
    )
    figure.supylabel(
        r"Gross structural turnover $(B_{50}+D_{50})/N_{\rm GT}$",
        fontsize=6.4, x=0.008,
    )
    figure.text(
        0.5, 0.018,
        "Pointwise seed-median trajectories; quantities are per 50 steps; "
        "opacity increases with time; circles mark final windows."
        + (f" {scale_note}" if scale_note else ""),
        ha="center", va="bottom", fontsize=4.8, color="#4B5563",
    )
    figure.subplots_adjust(left=0.085, right=0.995, top=0.82, bottom=0.22)
    outputs = [
        plot_dir / f"{output_stem}.pdf",
        plot_dir / f"{output_stem}.png",
    ]
    for output in outputs:
        figure.savefig(output, bbox_inches="tight")
    plt.close(figure)
    return outputs


def plot_structural_flux_phase_space_zoomed(
    summary: pd.DataFrame,
    plot_dir: Path,
    seed_rows: pd.DataFrame,
) -> list[Path]:
    """Central-range view that retains the full-scale phase-space plot separately."""
    seed_gross = CHURN_DISPLAY_STEPS * seed_rows["gross_flux"].to_numpy(dtype=float)
    seed_net = np.abs(CHURN_DISPLAY_STEPS * seed_rows["net_flux"].to_numpy(dtype=float))
    central_limit = max(
        float(np.quantile(seed_gross, 0.99)),
        float(np.quantile(seed_net, 0.99)),
        float(summary["gross_turnover_50"].max()),
        float(np.abs(summary["net_flux_50"]).max()),
        1e-6,
    )
    return plot_structural_flux_phase_space(
        summary,
        plot_dir,
        seed_rows=seed_rows,
        flux_limit_override=1.06 * central_limit,
        output_stem="mercury_structural_flux_phase_space_zoomed",
        scale_note="Axes use the central 99% seed-window range; only median trajectories are shown.",
    )


def plot_structural_phase_diagnostic_grid_3d(
    summary: pd.DataFrame,
    plot_dir: Path,
) -> list[Path]:
    """Plot each condition separately with learning time as the third dimension."""
    if summary.empty:
        raise ValueError("No structural phase trajectories were available to plot.")
    _apply_style()
    levels = _levels(summary)
    configurations = [
        (threshold, decay)
        for threshold in sorted(THRESHOLD_COLORS)
        for decay in (0.99, 1.0)
    ]
    figure, axes = plt.subplots(
        len(levels),
        len(configurations),
        figsize=(15.8, 8.9),
        subplot_kw={"projection": "3d"},
        dpi=300,
        gridspec_kw={"hspace": 0.04, "wspace": 0.01},
    )
    axes = np.asarray(axes).reshape(len(levels), len(configurations))

    x_values = summary["node_ratio"].to_numpy(dtype=float)
    x_padding = 0.05 * max(float(np.ptp(x_values)), 1e-6)
    x_limits = (max(0.0, float(x_values.min()) - x_padding), float(x_values.max()) + x_padding)
    y_values = CHURN_DISPLAY_STEPS * summary["churn_flux"].to_numpy(dtype=float)
    y_max = max(float(y_values.max()) * 1.10, 1e-6)
    y_limits = (0.0, y_max)
    z_limits = (0.0, float(summary["time"].max()))
    x_span = x_limits[1] - x_limits[0]
    y_span = y_limits[1] - y_limits[0]
    z_span = z_limits[1] - z_limits[0]

    for row_index, level in enumerate(levels):
        for col_index, (threshold, decay) in enumerate(configurations):
            axis = axes[row_index, col_index]
            color = THRESHOLD_COLORS[threshold]
            selected = (threshold, decay) == SELECTED_CONFIGURATION
            condition = summary[
                (summary["level"] == level)
                & (summary["ambiguity_threshold"] == threshold)
                & np.isclose(summary["ambiguity_decay"], decay)
            ]
            mode_counts = condition.groupby("mode")["mode_n"].first().to_dict()
            for mode, trajectory in condition.groupby("mode", sort=True):
                trajectory = trajectory.sort_values("time")
                x = trajectory["node_ratio"].to_numpy(dtype=float)
                y = CHURN_DISPLAY_STEPS * trajectory["churn_flux"].to_numpy(dtype=float)
                z = trajectory["time"].to_numpy(dtype=float)
                linestyle = "solid" if np.isclose(decay, 0.99) else (0, (3.2, 2.0))
                alpha = 0.95 if mode in {"all", "active"} else 0.58
                if selected:
                    axis.plot(x, y, z, color="#111827", linewidth=1.65, linestyle=linestyle, alpha=0.62)
                axis.plot(x, y, z, color=color, linewidth=0.95, linestyle=linestyle, alpha=alpha)
                for arrow_start, arrow_end in _direction_arrows_3d(
                    x, y, z, x_span, y_span, z_span
                ):
                    axis.scatter(
                        x[arrow_end],
                        y[arrow_end],
                        z[arrow_end],
                        color="#111827" if selected else color,
                        marker="^",
                        s=13,
                        linewidth=0.25,
                        edgecolor="white",
                        depthshade=False,
                    )

            if set(mode_counts) != {"all"}:
                axis.text2D(
                    0.97,
                    0.94,
                    "\n".join(f"{mode} {int(count)}/30" for mode, count in sorted(mode_counts.items())),
                    transform=axis.transAxes,
                    ha="right",
                    va="top",
                    fontsize=4.4,
                    color="#4B5563",
                )

            axis.plot(
                [1.0, 1.0], [0.0, 0.0], [z_limits[0], z_limits[1]],
                color=REFERENCE_COLOR, linewidth=0.5, linestyle=(0, (2.0, 2.2)), alpha=0.65,
            )
            axis.set_xlim(*x_limits)
            axis.set_ylim(*y_limits)
            axis.set_zlim(*z_limits)
            axis.view_init(elev=23, azim=-57)
            axis.set_box_aspect((1.15, 0.82, 1.0))
            axis.tick_params(axis="both", labelsize=3.4, pad=-1.5, length=1.5)
            axis.set_zticks([0, 1000, 2000])
            axis.grid(True)
            for pane in (axis.xaxis.pane, axis.yaxis.pane, axis.zaxis.pane):
                pane.set_facecolor((1.0, 1.0, 1.0, 0.0))
                pane.set_edgecolor("#DDE3EA")
            if selected:
                axis.set_facecolor("#F8FAFC")
            if row_index == 0:
                axis.set_title(
                    rf"$\theta={threshold}$" + "\n" + rf"$\gamma={decay:g}$",
                    fontsize=6.0,
                    color=color,
                    fontweight="bold" if selected else "normal",
                    pad=-1,
                )
            if row_index == len(levels) - 1:
                axis.set_xlabel(r"$N/N_{\rm GT}$", fontsize=4.8, labelpad=-6)
            if col_index == 0:
                axis.set_ylabel(
                    rf"churn/{CHURN_DISPLAY_STEPS}", fontsize=4.8, labelpad=-6
                )
            if col_index == len(configurations) - 1:
                axis.set_zlabel("step", fontsize=4.8, labelpad=-7)

    row_positions = np.linspace(0.82, 0.18, len(levels))
    for position, level in zip(row_positions, levels):
        level_title = LEVEL_LABELS.get(level, f"L{level}").replace("x", r"\times")
        figure.text(
            0.018, position, rf"${level_title}$", rotation=90,
            ha="center", va="center", fontsize=7.2, fontweight="bold",
        )
    figure.text(
        0.5,
        0.992,
        f"3D diagnostic grid: {FLUX_WINDOW}-step local churn, sampled every "
        f"{FLUX_SAMPLE_INTERVAL} steps; arrows show increasing time",
        ha="center",
        va="top",
        fontsize=8.5,
        color="#374151",
    )
    figure.subplots_adjust(left=0.035, right=0.995, top=0.94, bottom=0.025)
    outputs = [
        plot_dir / "mercury_structural_trajectories_diagnostic_grid_3d.pdf",
        plot_dir / "mercury_structural_trajectories_diagnostic_grid_3d.png",
    ]
    for output in outputs:
        figure.savefig(output, bbox_inches="tight")
    plt.close(figure)
    return outputs


def plot_structural_churn_hue_2d(
    summary: pd.DataFrame,
    plot_dir: Path,
) -> list[Path]:
    """Diagnostic inventory--time paths with local churn encoded by hue."""
    if summary.empty:
        raise ValueError("No structural phase trajectories were available to plot.")
    _apply_style()
    levels = _levels(summary)
    configurations = [
        (threshold, decay)
        for threshold in sorted(THRESHOLD_COLORS)
        for decay in (0.99, 1.0)
    ]
    figure, axes = plt.subplots(
        len(levels),
        len(configurations),
        figsize=(12.8, 7.0),
        sharex=True,
        sharey=True,
        dpi=300,
        gridspec_kw={"hspace": 0.16, "wspace": 0.10},
    )
    axes = np.asarray(axes).reshape(len(levels), len(configurations))

    x_values = summary["node_ratio"].to_numpy(dtype=float)
    x_padding = 0.05 * max(float(np.ptp(x_values)), 1e-6)
    x_limits = (max(0.0, float(x_values.min()) - x_padding), float(x_values.max()) + x_padding)
    churn_values = CHURN_DISPLAY_STEPS * summary["churn_flux"].to_numpy(dtype=float)
    churn_norm = matplotlib.colors.PowerNorm(
        gamma=0.45,
        vmin=0.0,
        vmax=max(float(churn_values.max()), 1e-12),
    )
    churn_cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "structural_churn",
        ["#E5E7EB", "#14B8A6", "#F59E0B", "#B91C1C"],
    )

    for row_index, level in enumerate(levels):
        for col_index, (threshold, decay) in enumerate(configurations):
            axis = axes[row_index, col_index]
            selected = (threshold, decay) == SELECTED_CONFIGURATION
            condition = summary[
                (summary["level"] == level)
                & (summary["ambiguity_threshold"] == threshold)
                & np.isclose(summary["ambiguity_decay"], decay)
            ]
            mode_counts = condition.groupby("mode")["mode_n"].first().to_dict()
            linestyle = "solid" if np.isclose(decay, 0.99) else (0, (3.2, 2.0))
            for mode, trajectory in condition.groupby("mode", sort=True):
                trajectory = trajectory.sort_values("time")
                x = trajectory["node_ratio"].to_numpy(dtype=float)
                time = trajectory["time"].to_numpy(dtype=float)
                churn = CHURN_DISPLAY_STEPS * trajectory["churn_flux"].to_numpy(dtype=float)
                segments = _phase_segments(x, time)
                segment_churn = 0.5 * (churn[:-1] + churn[1:])
                if selected:
                    axis.add_collection(
                        LineCollection(
                            segments, colors="#111827", linewidths=2.0,
                            linestyles=linestyle, alpha=0.56, zorder=2,
                        )
                    )
                collection = LineCollection(
                    segments,
                    cmap=churn_cmap,
                    norm=churn_norm,
                    linewidths=1.25 if mode in {"all", "active"} else 0.9,
                    linestyles=linestyle,
                    alpha=0.98 if mode in {"all", "active"} else 0.62,
                    zorder=3,
                )
                collection.set_array(segment_churn)
                axis.add_collection(collection)
                for fraction in (0.36, 0.72):
                    arrow_index = min(int(round(fraction * (len(time) - 1))), len(time) - 1)
                    axis.scatter(
                        x[arrow_index],
                        time[arrow_index],
                        s=11,
                        marker="^",
                        facecolor=churn_cmap(churn_norm(churn[arrow_index])),
                        edgecolor="#111827" if selected else "white",
                        linewidth=0.35,
                        zorder=4,
                    )

            if set(mode_counts) != {"all"}:
                axis.text(
                    0.97,
                    0.94,
                    "\n".join(f"{mode} {int(count)}/30" for mode, count in sorted(mode_counts.items())),
                    transform=axis.transAxes,
                    ha="right",
                    va="top",
                    fontsize=5.0,
                    color="#4B5563",
                )
            axis.axvline(
                1.0, color=REFERENCE_COLOR, linewidth=0.55,
                linestyle=(0, (2.0, 2.2)), alpha=0.70, zorder=0,
            )
            axis.set_xlim(*x_limits)
            axis.set_ylim(0, FINAL_STEP)
            axis.set_yticks([0, 500, 1000, 1500, 2000])
            axis.grid(axis="both", color="#DDE3EA", linewidth=0.35, alpha=0.42)
            axis.spines[["top", "right"]].set_visible(False)
            axis.tick_params(axis="both", labelsize=5.2, length=1.8, pad=1.2)
            if selected:
                axis.set_facecolor("#F8FAFC")
            if row_index == 0:
                axis.set_title(
                    rf"$\theta={threshold}$" + "\n" + rf"$\gamma={decay:g}$",
                    fontsize=6.3,
                    color=THRESHOLD_COLORS[threshold],
                    fontweight="bold" if selected else "normal",
                    pad=2.5,
                )
            if col_index == 0:
                level_title = LEVEL_LABELS.get(level, f"L{level}").replace("x", r"\times")
                axis.set_ylabel(
                    rf"${level_title}$" + "\nlearning step",
                    fontsize=6.5,
                    fontweight="bold",
                    labelpad=5,
                )

    figure.subplots_adjust(left=0.070, right=0.905, top=0.91, bottom=0.085)
    colour_mappable = matplotlib.cm.ScalarMappable(norm=churn_norm, cmap=churn_cmap)
    colour_mappable.set_array([])
    colour_axis = figure.add_axes([0.925, 0.16, 0.012, 0.68])
    colour_bar = figure.colorbar(
        colour_mappable,
        cax=colour_axis,
        orientation="vertical",
    )
    colour_bar.set_label(
        f"Structural churn per GT state per {CHURN_DISPLAY_STEPS} steps",
        fontsize=7.0,
    )
    colour_bar.ax.tick_params(labelsize=5.5, length=2.0)
    figure.supxlabel(r"Graph inventory $N(t)/N_{\rm GT}$", fontsize=8.0, y=0.025)
    figure.text(
        0.5,
        0.992,
        f"Inventory--time diagnostic with {FLUX_WINDOW}-step local churn encoded by line hue; "
        "triangles point forward in time",
        ha="center",
        va="top",
        fontsize=8.2,
        color="#374151",
    )
    outputs = [
        plot_dir / "mercury_structural_trajectories_churn_hue_2d.pdf",
        plot_dir / "mercury_structural_trajectories_churn_hue_2d.png",
    ]
    for output in outputs:
        figure.savefig(output, bbox_inches="tight")
    plt.close(figure)
    return outputs


def plot_structural_churn_hue_combined_2d(
    summary: pd.DataFrame,
    plot_dir: Path,
    *,
    cumulative: bool = False,
) -> list[Path]:
    """Overlay all configurations per environment with churn encoded by hue."""
    if summary.empty:
        raise ValueError("No structural phase trajectories were available to plot.")
    _apply_style()
    levels = _levels(summary)
    figure, axes = plt.subplots(
        1,
        len(levels),
        figsize=(7.16, 3.05),
        sharex=True,
        sharey=True,
        dpi=300,
        gridspec_kw={"wspace": 0.12},
    )
    axes = np.ravel(np.atleast_1d(axes))
    x_values = summary["node_ratio"].to_numpy(dtype=float)
    x_padding = 0.05 * max(float(np.ptp(x_values)), 1e-6)
    x_limits = (max(0.0, float(x_values.min()) - x_padding), float(x_values.max()) + x_padding)
    if cumulative:
        if "cumulative_churn_per_gt" not in summary.columns:
            raise ValueError("Cumulative normalized churn is missing from phase trajectories.")
        churn_values = summary["cumulative_churn_per_gt"].to_numpy(dtype=float)
        colourbar_label = r"Cumulative structural churn $[C(t)+D(t)]/N_{\rm GT}$"
        output_stem = "mercury_structural_trajectories_cumulative_churn_hue_2d"
        footer_churn = "cumulative churn"
        colourmap_name = "cumulative_structural_churn_combined"
        norm_gamma = 0.62
    else:
        churn_values = CHURN_DISPLAY_STEPS * summary["churn_flux"].to_numpy(dtype=float)
        colourbar_label = f"Structural churn per GT state per {CHURN_DISPLAY_STEPS} steps"
        output_stem = "mercury_structural_trajectories_churn_hue_2d"
        footer_churn = "local churn"
        colourmap_name = "structural_churn_combined"
        norm_gamma = 0.45
    churn_norm = matplotlib.colors.PowerNorm(
        gamma=norm_gamma,
        vmin=0.0,
        vmax=max(float(churn_values.max()), 1e-12),
    )
    churn_cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        colourmap_name,
        ["#E5E7EB", "#14B8A6", "#F59E0B", "#B91C1C"],
    )

    for axis, level in zip(axes, levels):
        panel = summary[summary["level"] == level]
        for threshold in sorted(THRESHOLD_COLORS):
            threshold_color = THRESHOLD_COLORS[threshold]
            for decay in (0.99, 1.0):
                selected = (threshold, decay) == SELECTED_CONFIGURATION
                linestyle = "solid" if np.isclose(decay, 0.99) else (0, (3.2, 2.0))
                condition = panel[
                    (panel["ambiguity_threshold"] == threshold)
                    & np.isclose(panel["ambiguity_decay"], decay)
                ]
                for mode, trajectory in condition.groupby("mode", sort=True):
                    trajectory = trajectory.sort_values("time")
                    x = trajectory["node_ratio"].to_numpy(dtype=float)
                    time = trajectory["time"].to_numpy(dtype=float)
                    churn = (
                        trajectory["cumulative_churn_per_gt"].to_numpy(dtype=float)
                        if cumulative
                        else CHURN_DISPLAY_STEPS * trajectory["churn_flux"].to_numpy(dtype=float)
                    )
                    segments = _phase_segments(x, time)
                    segment_churn = 0.5 * (churn[:-1] + churn[1:])
                    alpha = 0.95 if mode in {"all", "active"} else 0.55
                    if selected:
                        axis.add_collection(
                            LineCollection(
                                segments, colors="#111827",
                                linewidths=1.85 if cumulative else 2.45,
                                linestyles=linestyle, alpha=0.72, zorder=1,
                            )
                        )
                    if not cumulative:
                        axis.add_collection(
                            LineCollection(
                                segments, colors=threshold_color, linewidths=1.92,
                                linestyles=linestyle, alpha=0.78 * alpha, zorder=2,
                            )
                        )
                    if cumulative:
                        hue_line = LineCollection(
                            segments,
                            colors=_categorical_ordered_colours(
                                threshold_color, segment_churn, churn_norm
                            ),
                            linewidths=1.05,
                            linestyles=linestyle,
                            alpha=alpha,
                            zorder=3,
                        )
                    else:
                        hue_line = LineCollection(
                            segments,
                            cmap=churn_cmap,
                            norm=churn_norm,
                            linewidths=0.52,
                            linestyles=linestyle,
                            alpha=alpha,
                            zorder=3,
                        )
                        hue_line.set_array(segment_churn)
                    axis.add_collection(hue_line)
                    if not cumulative:
                        for checkpoint_time in (750, 1500):
                            matches = np.flatnonzero(time == checkpoint_time)
                            if not len(matches):
                                continue
                            arrow_index = int(matches[0])
                            axis.scatter(
                                x[arrow_index], time[arrow_index],
                                s=9.5, marker="^",
                                facecolor=churn_cmap(churn_norm(churn[arrow_index])),
                                edgecolor="#111827" if selected else threshold_color,
                                linewidth=0.45, zorder=4,
                            )

        if level == 22:
            active_path = panel[
                (panel["ambiguity_threshold"] == 40)
                & np.isclose(panel["ambiguity_decay"], 0.99)
                & (panel["mode"] == "active")
            ].sort_values("time")
            active = active_path.iloc[0]
            inactive = panel[
                (panel["ambiguity_threshold"] == 40)
                & np.isclose(panel["ambiguity_decay"], 0.99)
                & (panel["mode"] == "inactive")
            ].iloc[0]
            anchor = active_path.iloc[int(np.argmin(np.abs(active_path["time"].to_numpy() - 1800)))]
            axis.annotate(
                rf"$\theta=40,\gamma=.99$: {int(active['mode_n'])}/30 active"
                + "\n"
                + f"{int(inactive['mode_n'])}/30 inactive",
                xy=(float(anchor["node_ratio"]), float(anchor["time"])),
                xycoords="data",
                xytext=(0.98, 0.965),
                textcoords="axes fraction",
                ha="right",
                va="top",
                fontsize=4.6,
                color="#4B5563",
                arrowprops={
                    "arrowstyle": "-",
                    "color": THRESHOLD_COLORS[40],
                    "linewidth": 0.5,
                    "alpha": 0.65,
                },
            )
        axis.axvline(
            1.0, color=REFERENCE_COLOR, linewidth=0.55,
            linestyle=(0, (2.0, 2.2)), alpha=0.70, zorder=0,
        )
        axis.set_xlim(*x_limits)
        axis.set_ylim(0, FINAL_STEP)
        axis.set_yticks([0, 500, 1000, 1500, 2000])
        axis.grid(axis="both", color="#DDE3EA", linewidth=0.35, alpha=0.40)
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(axis="both", labelsize=5.4, length=1.8, pad=1.2)
        level_title = LEVEL_LABELS.get(level, f"L{level}").replace("x", r"\times")
        axis.set_title(rf"${level_title}$", fontsize=7.2, fontweight="bold", pad=3)
        axis.text(
            1.0,
            0.025,
            "state-count parity",
            transform=axis.get_xaxis_transform(),
            rotation=90,
            ha="right",
            va="bottom",
            fontsize=4.1,
            color="#64748B",
        )

    axes[0].set_ylabel("Learning step", fontsize=6.5)
    threshold_handles = [
        Line2D([0], [0], color=color, linewidth=1.7, label=rf"$\theta={threshold}$")
        for threshold, color in THRESHOLD_COLORS.items()
    ]
    decay_handles = [
        Line2D([0], [0], color="#4B5563", linewidth=1.5, linestyle="solid", label=r"$\gamma=0.99$"),
        Line2D([0], [0], color="#4B5563", linewidth=1.5, linestyle=(0, (3.2, 2.0)), label=r"$\gamma=1.0$"),
    ]
    handles = [*threshold_handles, *decay_handles]
    figure.legend(
        handles, [handle.get_label() for handle in handles],
        loc="upper center", bbox_to_anchor=(0.47, 0.995), ncol=6,
        frameon=False, fontsize=5.6, handlelength=2.0,
        handletextpad=0.35, columnspacing=0.8, borderaxespad=0,
    )
    figure.subplots_adjust(left=0.070, right=0.875, top=0.84, bottom=0.17)
    legend_cmap = (
        matplotlib.colors.LinearSegmentedColormap.from_list(
            "cumulative_churn_lightness", ["#E5E7EB", "#374151"]
        )
        if cumulative
        else churn_cmap
    )
    colour_mappable = matplotlib.cm.ScalarMappable(norm=churn_norm, cmap=legend_cmap)
    colour_mappable.set_array([])
    colour_axis = figure.add_axes([0.900, 0.20, 0.013, 0.58])
    colour_bar = figure.colorbar(colour_mappable, cax=colour_axis, orientation="vertical")
    colour_bar.set_label(colourbar_label, fontsize=6.2)
    colour_bar.ax.tick_params(labelsize=5.1, length=1.8)
    figure.supxlabel(r"Graph inventory $N(t)/N_{\rm GT}$", fontsize=7.0, y=0.075)
    figure.text(
        0.47, 0.025,
        (
            "All configurations overlaid; hue identifies threshold and light-to-dark progression identifies cumulative churn."
            if cumulative
            else f"All configurations overlaid; outer colour identifies threshold, inner hue identifies {footer_churn}; triangles mark 750-step intervals."
        ),
        ha="center", va="bottom", fontsize=4.9, color="#4B5563",
    )
    outputs = [
        plot_dir / f"{output_stem}.pdf",
        plot_dir / f"{output_stem}.png",
    ]
    for output in outputs:
        figure.savefig(output, bbox_inches="tight")
    plt.close(figure)
    return outputs


def plot_structural_orthogonal_inventory_churn_views(
    summary: pd.DataFrame,
    plot_dir: Path,
    *,
    environments_as_rows: bool = False,
) -> list[Path]:
    """Show orthogonal inventory--time and cumulative-churn--time projections."""
    if summary.empty:
        raise ValueError("No structural phase trajectories were available to plot.")
    if "cumulative_churn_per_gt" not in summary.columns:
        raise ValueError("Cumulative normalized churn is missing from phase trajectories.")
    expected_space = "node_ratio+cumulative_churn_per_gt"
    if "medoid_feature_space" not in summary.columns or set(summary["medoid_feature_space"]) != {expected_space}:
        raise ValueError(
            "Orthogonal projections require medoids selected jointly in inventory and cumulative-operations space."
        )
    _apply_style()
    levels = _levels(summary)
    if environments_as_rows:
        figure = plt.figure(figsize=(7.16, 9.3), dpi=300)
        outer = figure.add_gridspec(len(levels), 1, hspace=0.68)
    else:
        figure = plt.figure(figsize=(7.16, 4.65), dpi=300)
        outer = figure.add_gridspec(2, 2, hspace=0.48, wspace=0.26)
    paired_axes: list[tuple[plt.Axes, plt.Axes, int]] = []
    tick_fontsize = 10.2 if environments_as_rows else 5.1
    axis_label_fontsize = 11.2 if environments_as_rows else 5.5
    operations_label_fontsize = 10.2 if environments_as_rows else 5.0
    panel_title_fontsize = 11.8 if environments_as_rows else 5.8
    level_title_fontsize = 14.0 if environments_as_rows else 7.3
    annotation_fontsize = 8.6 if environments_as_rows else 4.4
    reference_fontsize = 7.8 if environments_as_rows else 3.9
    legend_fontsize = 10.2 if environments_as_rows else 5.8
    footer_fontsize = 8.8 if environments_as_rows else 5.0

    def projection_limits(panel: pd.DataFrame) -> tuple[tuple[float, float], tuple[float, float]]:
        inventory_values = panel["node_ratio"].to_numpy(dtype=float)
        inventory_padding = 0.055 * max(float(np.ptp(inventory_values)), 1e-6)
        inventory_limits = (
            max(0.0, min(1.0, float(inventory_values.min())) - inventory_padding),
            max(1.0, float(inventory_values.max())) + inventory_padding,
        )
        cumulative_values = panel["cumulative_churn_per_gt"].to_numpy(dtype=float)
        cumulative_limits = (0.0, 1.05 * max(float(cumulative_values.max()), 1e-6))
        return inventory_limits, cumulative_limits

    shared_inventory_limits, shared_cumulative_limits = projection_limits(summary)

    for index, level in enumerate(levels):
        if environments_as_rows:
            inner = outer[index, 0].subgridspec(1, 2, wspace=0.07)
        else:
            row, column = divmod(index, 2)
            inner = outer[row, column].subgridspec(1, 2, wspace=0.07)
        inventory_axis = figure.add_subplot(inner[0, 0])
        cumulative_axis = figure.add_subplot(inner[0, 1], sharey=inventory_axis)
        paired_axes.append((inventory_axis, cumulative_axis, level))
        panel = summary[summary["level"] == level]
        if environments_as_rows:
            inventory_limits, cumulative_limits = projection_limits(panel)
        else:
            inventory_limits = shared_inventory_limits
            cumulative_limits = shared_cumulative_limits

        for threshold in sorted(THRESHOLD_COLORS):
            color = THRESHOLD_COLORS[threshold]
            for decay in (0.99, 1.0):
                selected = (threshold, decay) == SELECTED_CONFIGURATION
                linestyle = "solid" if np.isclose(decay, 0.99) else (0, (3.2, 2.0))
                condition = panel[
                    (panel["ambiguity_threshold"] == threshold)
                    & np.isclose(panel["ambiguity_decay"], decay)
                ]
                for mode, trajectory in condition.groupby("mode", sort=True):
                    trajectory = trajectory.sort_values("time")
                    time = trajectory["time"].to_numpy(dtype=float)
                    alpha = 0.94 if mode in {"all", "active"} else 0.43
                    for axis, coordinate in (
                        (inventory_axis, trajectory["node_ratio"].to_numpy(dtype=float)),
                        (
                            cumulative_axis,
                            trajectory["cumulative_churn_per_gt"].to_numpy(dtype=float),
                        ),
                    ):
                        if selected:
                            axis.plot(
                                coordinate, time, color="#111827", linewidth=2.15,
                                linestyle=linestyle, alpha=0.78, zorder=2,
                            )
                        axis.plot(
                            coordinate, time, color=color, linewidth=0.92,
                            linestyle=linestyle, alpha=alpha, zorder=3,
                        )

        if level == 22:
            active_path = panel[
                (panel["ambiguity_threshold"] == 40)
                & np.isclose(panel["ambiguity_decay"], 0.99)
                & (panel["mode"] == "active")
            ].sort_values("time")
            active = active_path.iloc[0]
            inactive = panel[
                (panel["ambiguity_threshold"] == 40)
                & np.isclose(panel["ambiguity_decay"], 0.99)
                & (panel["mode"] == "inactive")
            ].iloc[0]
            anchor = active_path.iloc[int(np.argmin(np.abs(active_path["time"].to_numpy() - 1800)))]
            cumulative_axis.annotate(
                rf"$\theta=40,\gamma=.99$: {int(active['mode_n'])}/30 active"
                + "\n"
                + f"{int(inactive['mode_n'])}/30 inactive",
                xy=(float(anchor["cumulative_churn_per_gt"]), float(anchor["time"])),
                xycoords="data",
                xytext=(0.98, 0.965),
                textcoords="axes fraction",
                ha="right", va="top", fontsize=annotation_fontsize, color="#4B5563",
                arrowprops={
                    "arrowstyle": "-", "color": THRESHOLD_COLORS[40],
                    "linewidth": 0.48, "alpha": 0.65,
                },
            )

        inventory_axis.axvline(
            1.0, color=REFERENCE_COLOR, linewidth=0.58,
            linestyle=(0, (2.0, 2.2)), alpha=0.72, zorder=0,
        )
        inventory_axis.text(
            1.0, 0.025, r"$N=N_{\rm GT}$",
            transform=inventory_axis.get_xaxis_transform(), rotation=90,
            ha="right", va="bottom", fontsize=reference_fontsize, color="#64748B",
        )
        inventory_axis.set_xlim(*inventory_limits)
        cumulative_axis.set_xlim(*cumulative_limits)
        inventory_axis.set_ylim(0, FINAL_STEP)
        inventory_axis.set_yticks([0, 500, 1000, 1500, 2000])
        for axis in (inventory_axis, cumulative_axis):
            axis.grid(axis="both", color="#DDE3EA", linewidth=0.35, alpha=0.42)
            axis.spines[["top", "right"]].set_visible(False)
            axis.tick_params(axis="both", labelsize=tick_fontsize, length=1.8, pad=1.2)
        cumulative_axis.tick_params(axis="y", labelleft=False, left=False)
        cumulative_axis.spines["left"].set_visible(False)
        if not environments_as_rows or index == len(levels) - 1:
            inventory_axis.set_xlabel(
                r"Inventory $N/N_{\rm GT}$", fontsize=axis_label_fontsize, labelpad=2
            )
            cumulative_axis.set_xlabel(
                r"Cumulative structural operations $(C+D)/N_{\rm GT}$",
                fontsize=operations_label_fontsize,
                labelpad=2,
            )
        if not environments_as_rows:
            inventory_axis.set_ylabel("Learning step", fontsize=axis_label_fontsize)
        if not environments_as_rows or index == 0:
            inventory_axis.set_title("Inventory projection", fontsize=panel_title_fontsize, pad=3)
            cumulative_axis.set_title(
                "Structural-operations projection", fontsize=panel_title_fontsize, pad=3
            )
        level_title = LEVEL_LABELS.get(level, f"L{level}").replace("x", r"\times")
        inventory_axis.text(
            1.04, 1.17, rf"${level_title}$",
            transform=inventory_axis.transAxes, ha="center", va="bottom",
            fontsize=level_title_fontsize, fontweight="bold", clip_on=False,
        )

    threshold_handles = [
        Line2D([0], [0], color=color, linewidth=1.7, label=rf"$\theta={threshold}$")
        for threshold, color in THRESHOLD_COLORS.items()
    ]
    decay_handles = [
        Line2D([0], [0], color="#4B5563", linewidth=1.5, linestyle="solid", label=r"$\gamma=0.99$"),
        Line2D([0], [0], color="#4B5563", linewidth=1.5, linestyle=(0, (3.2, 2.0)), label=r"$\gamma=1.0$"),
    ]
    handles = [*threshold_handles, *decay_handles]
    figure.legend(
        handles, [handle.get_label() for handle in handles],
        loc="upper center", bbox_to_anchor=(0.5, 0.995), ncol=6,
        frameon=False, fontsize=legend_fontsize, handlelength=2.0,
        handletextpad=0.35, columnspacing=0.85, borderaxespad=0,
    )
    if not environments_as_rows:
        figure.text(
            0.5,
            0.018,
            r"Orthogonal projections of $(N/N_{\rm GT},[C+D]/N_{\rm GT},t)$ from one joint observed medoid; paired panels share learning step.",
            ha="center",
            va="bottom",
            fontsize=footer_fontsize,
            color="#4B5563",
        )
    if environments_as_rows:
        figure.supylabel("Learning step", fontsize=axis_label_fontsize, x=0.018)
        figure.subplots_adjust(left=0.085, right=0.995, top=0.89, bottom=0.055)
        output_stem = "mercury_structural_orthogonal_inventory_cumulative_churn_environment_rows"
    else:
        figure.subplots_adjust(left=0.075, right=0.995, top=0.87, bottom=0.10)
        output_stem = "mercury_structural_orthogonal_inventory_cumulative_churn"
    outputs = [
        plot_dir / f"{output_stem}.pdf",
        plot_dir / f"{output_stem}.png",
    ]
    for output in outputs:
        figure.savefig(output, bbox_inches="tight")
    plt.close(figure)
    return outputs


def plot_structural_churn_time_inventory_markers(
    summary: pd.DataFrame,
    plot_dir: Path,
) -> list[Path]:
    """Plot churn against time with inventory encoded at fixed checkpoints."""
    if summary.empty:
        raise ValueError("No structural phase trajectories were available to plot.")
    _apply_style()
    levels = _levels(summary)
    figure, axes = plt.subplots(
        1,
        len(levels),
        figsize=(7.16, 2.85),
        sharex=True,
        sharey=True,
        dpi=300,
        gridspec_kw={"wspace": 0.12},
    )
    axes = np.ravel(np.atleast_1d(axes))
    churn_values = CHURN_DISPLAY_STEPS * summary["churn_flux"].to_numpy(dtype=float)
    y_max = max(float(churn_values.max()) * 1.10, 1e-6)
    y_limits = (-0.025 * y_max, y_max)
    inventory_values = summary["node_ratio"].to_numpy(dtype=float)
    inventory_norm = matplotlib.colors.Normalize(
        vmin=float(inventory_values.min()),
        vmax=float(inventory_values.max()),
    )
    inventory_cmap = matplotlib.colormaps["cividis"]
    inventory_checkpoints = tuple(range(250, FINAL_STEP + 1, 250))

    for axis, level in zip(axes, levels):
        panel = summary[summary["level"] == level]
        for threshold in sorted(THRESHOLD_COLORS):
            color = THRESHOLD_COLORS[threshold]
            for decay in (0.99, 1.0):
                selected = (threshold, decay) == SELECTED_CONFIGURATION
                linestyle = "solid" if np.isclose(decay, 0.99) else (0, (3.2, 2.0))
                condition = panel[
                    (panel["ambiguity_threshold"] == threshold)
                    & np.isclose(panel["ambiguity_decay"], decay)
                ]
                for mode, trajectory in condition.groupby("mode", sort=True):
                    trajectory = trajectory.sort_values("time")
                    time = trajectory["time"].to_numpy(dtype=float)
                    churn = CHURN_DISPLAY_STEPS * trajectory["churn_flux"].to_numpy(dtype=float)
                    alpha = 0.94 if mode in {"all", "active"} else 0.42
                    if selected:
                        axis.plot(
                            time, churn, color="#111827", linewidth=1.75,
                            linestyle=linestyle, alpha=0.60, zorder=2,
                        )
                    axis.plot(
                        time, churn, color=color, linewidth=0.90,
                        linestyle=linestyle, alpha=alpha, zorder=3,
                    )
                    checkpoints = trajectory[trajectory["time"].isin(inventory_checkpoints)]
                    axis.scatter(
                        checkpoints["time"],
                        CHURN_DISPLAY_STEPS * checkpoints["churn_flux"],
                        c=checkpoints["node_ratio"],
                        cmap=inventory_cmap,
                        norm=inventory_norm,
                        s=12.0,
                        marker="o",
                        edgecolor="#111827" if selected else color,
                        linewidth=0.62 if selected else 0.40,
                        alpha=0.98 if mode in {"all", "active"} else 0.50,
                        zorder=4,
                    )

        if level == 22:
            active_path = panel[
                (panel["ambiguity_threshold"] == 40)
                & np.isclose(panel["ambiguity_decay"], 0.99)
                & (panel["mode"] == "active")
            ].sort_values("time")
            active = active_path.iloc[0]
            inactive = panel[
                (panel["ambiguity_threshold"] == 40)
                & np.isclose(panel["ambiguity_decay"], 0.99)
                & (panel["mode"] == "inactive")
            ].iloc[0]
            positive = active_path[active_path["churn_flux"] > 0]
            anchor = positive.iloc[0] if not positive.empty else active_path.iloc[-1]
            axis.annotate(
                rf"$\theta=40,\gamma=.99$: {int(active['mode_n'])}/30 active"
                + "\n"
                + f"{int(inactive['mode_n'])}/30 inactive",
                xy=(float(anchor["time"]), CHURN_DISPLAY_STEPS * float(anchor["churn_flux"])),
                xycoords="data",
                xytext=(0.98, 0.96),
                textcoords="axes fraction",
                ha="right",
                va="top",
                fontsize=4.5,
                color="#4B5563",
                arrowprops={
                    "arrowstyle": "-",
                    "color": THRESHOLD_COLORS[40],
                    "linewidth": 0.5,
                    "alpha": 0.65,
                },
            )

        axis.axhline(0.0, color=REFERENCE_COLOR, linewidth=0.55, alpha=0.65, zorder=0)
        axis.set_xlim(0, FINAL_STEP)
        axis.set_ylim(*y_limits)
        axis.set_xticks([0, 500, 1000, 1500, 2000])
        axis.grid(axis="both", color="#DDE3EA", linewidth=0.35, alpha=0.42)
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(axis="both", labelsize=5.3, length=1.8, pad=1.2)
        level_title = LEVEL_LABELS.get(level, f"L{level}").replace("x", r"\times")
        axis.set_title(rf"${level_title}$", fontsize=7.2, fontweight="bold", pad=3)

    axes[0].set_ylabel(
        f"Structural churn per GT state\nper {CHURN_DISPLAY_STEPS} steps",
        fontsize=6.3,
    )
    threshold_handles = [
        Line2D([0], [0], color=color, linewidth=1.7, label=rf"$\theta={threshold}$")
        for threshold, color in THRESHOLD_COLORS.items()
    ]
    decay_handles = [
        Line2D([0], [0], color="#4B5563", linewidth=1.5, linestyle="solid", label=r"$\gamma=0.99$"),
        Line2D([0], [0], color="#4B5563", linewidth=1.5, linestyle=(0, (3.2, 2.0)), label=r"$\gamma=1.0$"),
    ]
    handles = [*threshold_handles, *decay_handles]
    figure.legend(
        handles, [handle.get_label() for handle in handles],
        loc="upper center", bbox_to_anchor=(0.47, 0.995), ncol=6,
        frameon=False, fontsize=5.6, handlelength=2.0,
        handletextpad=0.35, columnspacing=0.8, borderaxespad=0,
    )
    figure.subplots_adjust(left=0.085, right=0.875, top=0.84, bottom=0.20)
    inventory_mappable = matplotlib.cm.ScalarMappable(norm=inventory_norm, cmap=inventory_cmap)
    inventory_mappable.set_array([])
    colour_axis = figure.add_axes([0.900, 0.22, 0.013, 0.55])
    colour_bar = figure.colorbar(inventory_mappable, cax=colour_axis, orientation="vertical")
    colour_bar.set_label(r"Checkpoint inventory $N/N_{\rm GT}$", fontsize=6.2)
    colour_bar.ax.tick_params(labelsize=5.1, length=1.8)
    figure.supxlabel("Learning step", fontsize=7.0, y=0.09)
    figure.text(
        0.47, 0.025,
        "Marker fill reports graph inventory at 250-step checkpoints; churn includes every clone and deletion operation.",
        ha="center", va="bottom", fontsize=4.8, color="#4B5563",
    )
    outputs = [
        plot_dir / "mercury_structural_churn_time_inventory_markers.pdf",
        plot_dir / "mercury_structural_churn_time_inventory_markers.png",
    ]
    for output in outputs:
        figure.savefig(output, bbox_inches="tight")
    plt.close(figure)
    return outputs


def plot_structural_trajectories(
    summary: pd.DataFrame,
    plot_dir: Path,
    *,
    output_stem: str = "mercury_structural_trajectories_supplementary",
) -> list[Path]:
    _apply_style()
    levels = _levels(summary)
    figure, axes = plt.subplots(
        len(METRICS),
        len(levels),
        figsize=(3.0 * len(levels) + 2.2, 1.65 * len(METRICS) + 2.1),
        sharex=True,
        dpi=300,
        gridspec_kw={"hspace": 0.2, "wspace": 0.12},
    )
    axes = np.asarray(axes).reshape(len(METRICS), len(levels))

    for row_index, (metric, metric_label, _) in enumerate(METRICS):
        metric_frame = summary[summary["metric"] == metric]
        upper = float((metric_frame["mean"] + metric_frame["ci95"]).max())
        lower = float((metric_frame["mean"] - metric_frame["ci95"]).min())
        if metric == "node_ratio":
            upper, lower = max(upper, 1.0), min(lower, 1.0)
        padding = 0.06 * max(upper - lower, 1e-6)
        limits = (max(0.0, lower - padding), upper + padding)
        for col_index, level in enumerate(levels):
            axis = axes[row_index, col_index]
            panel = metric_frame[metric_frame["level"] == level]
            for series in SERIES_ORDER:
                line = panel[panel["series"] == series].sort_values("step")
                if line.empty:
                    continue
                x = line["step"].to_numpy(dtype=float)
                y = line["mean"].to_numpy(dtype=float)
                ci = line["ci95"].to_numpy(dtype=float)
                color = SERIES_COLORS[series]
                axis.fill_between(x, y - ci, y + ci, color=color, alpha=0.07, linewidth=0)
                axis.plot(
                    x,
                    y,
                    color=color,
                    linestyle=_linestyle(series),
                    linewidth=1.25,
                    alpha=0.94,
                    label=SERIES_LABELS[series] if row_index == 0 and col_index == 0 else None,
                )
            if metric == "node_ratio":
                axis.axhline(1.0, color=REFERENCE_COLOR, linewidth=0.7, linestyle=(0, (2.0, 2.2)), alpha=0.8)
            axis.set_ylim(*limits)
            axis.grid(axis="both", color="#DDE3EA", linewidth=0.5, alpha=0.5)
            axis.spines[["top", "right"]].set_visible(False)
            axis.tick_params(axis="both", labelsize=7.0)
            if col_index == 0:
                axis.set_ylabel(metric_label, fontsize=7.8, fontweight="bold")
            else:
                axis.tick_params(axis="y", labelleft=False)
                axis.spines["left"].set_visible(False)
            if row_index == 0:
                axis.set_title(LEVEL_LABELS.get(level, f"L{level}"), fontsize=9.3, fontweight="bold")
            if row_index < len(METRICS) - 1:
                axis.tick_params(axis="x", labelbottom=False)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.935), ncol=4, frameon=False, fontsize=7.2)
    seed_counts = sorted(set(int(value) for value in summary["n"]))
    seed_label = f"n={seed_counts[0]} seeds/config" if len(seed_counts) == 1 else f"n={seed_counts[0]}-{seed_counts[-1]} seeds/config"
    figure.suptitle(f"Age ambiguity: Mercury graph growth ({seed_label}; mean ± 95% CI)", fontsize=11.5, y=0.992)
    figure.supxlabel("Mercury learning step", fontsize=8.8, y=0.025)
    figure.subplots_adjust(left=0.085, right=0.995, top=0.82, bottom=0.09)
    outputs = [plot_dir / f"{output_stem}.pdf", plot_dir / f"{output_stem}.png"]
    for output in outputs:
        figure.savefig(output, bbox_inches="tight")
    plt.close(figure)
    return outputs


def plot_template_lengths(rows: pd.DataFrame, summary: pd.DataFrame, plot_dir: Path) -> list[Path]:
    """Plot the template-length factorial ablation as compact 2x2 small multiples."""
    _apply_style()
    levels = _levels(rows)
    if len(levels) != 4:
        raise ValueError(f"The compact template-length layout requires four environments; found {len(levels)}.")
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(6.8, 4.6),
        sharex=True,
        sharey=True,
        dpi=300,
    )
    axes = np.ravel(axes)
    generator = np.random.default_rng(0)
    thresholds = (10, 20, 30, 40)
    decay_styles = {
        0.99: {
            "offset": -0.65,
            "color": "#34495E",
            "linestyle": "-",
            "markerfacecolor": "#34495E",
            "label": r"$\gamma=0.99$",
        },
        1.0: {
            "offset": 0.65,
            "color": "#7B8794",
            "linestyle": (0, (4.0, 2.2)),
            "markerfacecolor": "white",
            "label": r"$\gamma=1.0$",
        },
    }
    seed_means = (
        rows.groupby(
            [
                "ambiguity_threshold",
                "ambiguity_decay",
                "level",
                "level_label",
                "run_id",
                "seed",
            ],
            dropna=False,
        )["template_length"]
        .mean()
        .reset_index()
    )
    y_values = seed_means["template_length"].to_numpy(dtype=float)
    y_span = max(float(np.ptp(y_values)), 0.25)
    y_limits = (max(0.0, float(np.min(y_values)) - 0.06 * y_span), float(np.max(y_values)) + 0.08 * y_span)

    for panel_index, (axis, level) in enumerate(zip(axes, levels)):
        level_rows = rows[rows["level"] == level]
        level_summary = summary[summary["level"] == level]
        axis.axvline(30, color="#D9DEE5", linewidth=1.1, alpha=0.75, zorder=0)
        for decay, style in decay_styles.items():
            trace = level_summary[np.isclose(level_summary["ambiguity_decay"], decay)].set_index(
                "ambiguity_threshold"
            )
            if not set(thresholds).issubset(trace.index):
                raise ValueError(f"Incomplete template-length summary for level {level}, decay {decay}.")
            x_means = np.asarray(thresholds, dtype=float) + float(style["offset"])
            means = trace.loc[list(thresholds), "mean"].to_numpy(dtype=float)
            axis.plot(
                x_means,
                means,
                color=str(style["color"]),
                linestyle=style["linestyle"],
                linewidth=1.35,
                alpha=0.96,
                zorder=3,
            )
            for x_position, threshold in zip(x_means, thresholds):
                seed_values = (
                    level_rows[
                        (level_rows["ambiguity_threshold"] == threshold)
                        & np.isclose(level_rows["ambiguity_decay"], decay)
                    ]
                    .groupby(["run_id", "seed"])["template_length"]
                    .mean()
                    .to_numpy(dtype=float)
                )
                jitter = generator.uniform(-0.26, 0.26, size=seed_values.size)
                if decay == 0.99:
                    axis.scatter(
                        x_position + jitter,
                        seed_values,
                        s=8,
                        color=str(style["color"]),
                        alpha=0.16,
                        linewidths=0,
                        zorder=1,
                    )
                else:
                    axis.scatter(
                        x_position + jitter,
                        seed_values,
                        s=8,
                        facecolors="none",
                        edgecolors=str(style["color"]),
                        alpha=0.20,
                        linewidths=0.45,
                        zorder=1,
                    )
            ci95 = trace.loc[list(thresholds), "ci95"].to_numpy(dtype=float)
            axis.errorbar(
                x_means,
                means,
                yerr=ci95,
                fmt="o",
                color=str(style["color"]),
                ecolor=str(style["color"]),
                elinewidth=1.05,
                capsize=2.3,
                capthick=0.9,
                markersize=4.6,
                markerfacecolor=str(style["markerfacecolor"]),
                markeredgecolor=str(style["color"]),
                markeredgewidth=0.9,
                zorder=4,
            )
            if decay == SELECTED_CONFIGURATION[1]:
                selected = trace.loc[SELECTED_CONFIGURATION[0]]
                axis.scatter(
                    SELECTED_CONFIGURATION[0] + float(style["offset"]),
                    float(selected["mean"]),
                    s=38,
                    facecolors="none",
                    edgecolors="#111827",
                    linewidths=1.05,
                    zorder=5,
                )
        axis.set_title(LEVEL_LABELS.get(level, f"L{level}"), fontsize=9.5, fontweight="bold")
        axis.set_xticks(thresholds)
        axis.set_xlim(7.0, 43.0)
        axis.set_ylim(*y_limits)
        axis.grid(axis="y", color="#DDE3EA", linewidth=0.55, alpha=0.6)
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(axis="both", labelsize=7.4)
        if panel_index % 2:
            axis.tick_params(axis="y", labelleft=False, left=False)
            axis.spines["left"].set_visible(False)

    legend_handles = [
        Line2D(
            [0],
            [0],
            color=str(style["color"]),
            linestyle=style["linestyle"],
            linewidth=1.35,
            marker="o",
            markersize=4.8,
            markerfacecolor=str(style["markerfacecolor"]),
            markeredgecolor=str(style["color"]),
            markeredgewidth=0.9,
            label=str(style["label"]),
        )
        for style in decay_styles.values()
    ]
    figure.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.54, 0.995),
        ncol=2,
        frameon=False,
        fontsize=8.0,
        handlelength=2.8,
        columnspacing=1.5,
    )
    figure.supxlabel(r"$\theta$", fontsize=9.0, y=0.025)
    figure.supylabel("Mean sensory template length", fontsize=8.8, x=0.025)
    figure.subplots_adjust(left=0.10, right=0.995, top=0.91, bottom=0.12, hspace=0.28, wspace=0.08)
    outputs = [
        plot_dir / "mercury_sensory_template_lengths.pdf",
        plot_dir / "mercury_sensory_template_lengths.png",
    ]
    for output in outputs:
        figure.savefig(output, bbox_inches="tight")
    plt.close(figure)
    caption = plot_dir / "mercury_sensory_template_lengths_caption.tex"
    caption.write_text(
        r"\caption{Sensory-template-length sensitivity to MERCURY's ambiguity threshold. "
        r"Points show seed-level mean sensory-template lengths; larger markers and bars show "
        r"the across-seed mean and 95\% confidence interval. Filled solid and open dashed traces "
        r"denote $\gamma=0.99$ and $\gamma=1.0$, respectively. The faint vertical guide marks "
        r"the selected threshold, $\theta=30$; its $\gamma=0.99$ estimate is additionally outlined.}"
        "\n",
        encoding="utf-8",
    )
    return [*outputs, caption]


def write_structural_reporting(
    medoids: pd.DataFrame,
    sensitivity: pd.DataFrame,
    plot_dir: Path,
) -> list[Path]:
    sensitivity_summary = sensitivity.groupby("alternate_window").agg(
        same_medoid_fraction=("same_medoid", "mean"),
        mean_churn_spearman=("churn_flux_spearman", "mean"),
        mean_churn_activity_agreement=("churn_activity_agreement", "mean"),
        mean_net_direction_agreement=("net_flux_sign_agreement", "mean"),
    )
    sensitivity_lines = []
    for window, row in sensitivity_summary.iterrows():
        sensitivity_lines.append(
            f"- {int(window)} steps: same medoid in {100 * row.same_medoid_fraction:.1f}% of modes; "
            f"mean churn-profile Spearman rho={row.mean_churn_spearman:.3f}; "
            f"churn-active/quiescent agreement={100 * row.mean_churn_activity_agreement:.1f}%; "
            f"net-direction agreement={100 * row.mean_net_direction_agreement:.1f}%."
        )

    methods = f"""# Dynamic graph-inventory--churn portrait

## Plotted quantities and units

For each seed, local event rates use a trailing {FLUX_WINDOW}-step window (the available interval is used before step {FLUX_WINDOW}):

- `J_B(t) = [C(t) - C(t-tau)] / Delta t`, in clone events per learning step.
- `J_D(t) = [D(t) - D(t-tau)] / Delta t`, in deletion events per learning step.
- `J_gross = J_B + J_D`.
- `J_net = J_B - J_D`.
- `J_churn = J_gross = J_B + J_D`.

The portrait uses `x = N(t) / N_GT` and `y = {CHURN_DISPLAY_STEPS} J_churn(t) / N_GT`. Thus, y is gross local structural turnover per ground-truth state per {CHURN_DISPLAY_STEPS} learning steps: all clone and deletion operations contribute, including monotonic growth or pruning. Its normalized cumulative value through T is `[C(T) + D(T)] / N_GT`. The vertical `N/N_GT = 1` line denotes ground-truth state-count (capacity) parity only.

All differences and normalizations are computed within seed before any representative is selected. Paths are sampled every {FLUX_SAMPLE_INTERVAL} steps. Markers identify t={','.join(str(value) for value in TIME_CHECKPOINTS)}, and arrows show time direction.

## Seed aggregation and modes

Each displayed path is an observed trajectory medoid. Pairwise trajectory distance is the time-summed Euclidean distance in standardized `(N/N_GT, J_churn/N_GT)` coordinates. The medoid therefore remains a dynamically coherent realized run.

Conditions with universal cloning use one medoid. Mechanistic mixtures are split into clone-active, deletion-only, and quiescent modes only when those modes occur. For `3x4, theta=40, gamma=0.99`, the plot shows a clone-active medoid (14/30 seeds) and the inactive mode (16/30 seeds); inactive runs are observations, not missing values.

## Temporal-window sensitivity

The final {FLUX_WINDOW}-step window was compared with {', '.join(str(value) for value in SENSITIVITY_WINDOWS)} steps:

{chr(10).join(sensitivity_lines)}

The {FLUX_WINDOW}-step window was retained as the middle scale: {min(SENSITIVITY_WINDOWS)} steps is more impulse-sensitive, whereas {max(SENSITIVITY_WINDOWS)} steps blends longer portions of the birth--death process. Churn-active versus quiescent classification and net movement direction are largely preserved, although the exact medoid and fine trajectory shape can change with temporal scale.
"""

    caption = rf"""\caption{{Dynamic graph-inventory--churn portrait for MERCURY's ambiguity-control sensitivity analysis. Horizontal position is live graph inventory, $N(t)/N_{{\rm GT}}$; vertical position is ${CHURN_DISPLAY_STEPS}[J_B^{{({FLUX_WINDOW})}}(t)+J_D^{{({FLUX_WINDOW})}}(t)]/N_{{\rm GT}}$, expressed as gross structural turnover per ground-truth state per {CHURN_DISPLAY_STEPS} learning steps. Local birth and death rates were calculated within each seed using {FLUX_WINDOW}-step trailing differences before selecting an observed trajectory medoid in standardized inventory--churn space. Colour denotes ambiguity threshold $\theta$, line style denotes decay $\gamma$, markers denote common learning checkpoints, and arrows indicate time direction. The vertical dashed line marks ground-truth state-count parity, not representational optimality. The $3\times4$, $\theta=40$, $\gamma=0.99$ condition is shown as separate clone-active (14/30) and inactive (16/30) modes. Cumulative normalized churn through $T$ is $[C(T)+D(T)]/N_{{\rm GT}}$.}}
"""

    results = r"""Final latent-state and transition metrics were broadly stable across the intermediate ambiguity thresholds ($\theta=20$ and $\theta=30$). The structural trajectories show that lower thresholds initiate refinement earlier and generally induce greater structural activity, whereas $\theta=40$ can delay or suppress refinement within the finite acquisition horizon. Variation in ambiguity decay produced comparatively small qualitative changes. We therefore use $\theta=30,\gamma=0.99$ as a conservative interior operating point within the stable-performance regime.
"""

    hue_caption = rf"""\caption{{Inventory--time diagnostic for MERCURY's ambiguity-control sensitivity analysis. Horizontal position is live graph inventory, $N(t)/N_{{\rm GT}}$, and vertical position is learning step. The inner line hue encodes {CHURN_DISPLAY_STEPS}-step gross structural turnover, ${CHURN_DISPLAY_STEPS}[J_B^{{({FLUX_WINDOW})}}(t)+J_D^{{({FLUX_WINDOW})}}(t)]/N_{{\rm GT}}$, while the outer casing identifies ambiguity threshold $\theta$ and solid/dashed styling identifies decay $\gamma$. Triangles mark 750-step intervals. The dotted line marks ground-truth state-count parity, not representational optimality. Paths are observed trajectory medoids selected after within-seed rate calculation. For $3\times4$, the $\theta=40,\gamma=0.99$ condition is shown as separate active (14/30) and inactive (16/30) modes.}}
"""

    churn_time_caption = rf"""\caption{{Structural churn through learning for MERCURY's ambiguity-control sensitivity analysis. Learning step is shown horizontally and {CHURN_DISPLAY_STEPS}-step gross structural turnover, ${CHURN_DISPLAY_STEPS}[J_B^{{({FLUX_WINDOW})}}(t)+J_D^{{({FLUX_WINDOW})}}(t)]/N_{{\rm GT}}$, vertically. Line hue identifies ambiguity threshold $\theta$, solid/dashed styling identifies decay $\gamma$, and marker fill reports live graph inventory $N(t)/N_{{\rm GT}}$ at 250-step checkpoints. Zero churn denotes no clone or deletion operations within the local window; positive churn includes monotonic growth, monotonic pruning, and bidirectional turnover. Paths are observed trajectory medoids selected after within-seed rate calculation. For $3\times4$, the $\theta=40,\gamma=0.99$ condition is shown as separate active (14/30) and inactive (16/30) modes.}}
"""

    flux_phase_caption = r"""\caption{Structural flux phase space for MERCURY's ambiguity-control sensitivity analysis. Each 50-step window is represented by net structural flux, $(B_{50}-D_{50})/N_{\rm GT}$, horizontally and gross structural turnover, $(B_{50}+D_{50})/N_{\rm GT}$, vertically. Coordinates were computed within seed before taking pointwise medians across seeds; individual seed points and trajectories are omitted. The dashed boundaries $y=|x|$ delimit the physically admissible region: the positive and negative rays correspond to pure expansion and pure pruning, respectively; points near the vertical axis indicate approximately balanced replacement, and the origin indicates structural quiescence. Colour identifies ambiguity threshold $\theta$, line style identifies decay $\gamma$, increasing opacity indicates learning progression, and circles mark final windows. For $3\times4$, $\theta=40,\gamma=0.99$, the all-seed median retains all 30 observations and a labelled conditional median shows the 14/30 structurally active seeds; the remaining 16/30 inactive seeds are not discarded.}
"""

    cumulative_hue_caption = r"""\caption{Inventory--time diagnostic with cumulative structural churn for MERCURY's ambiguity-control sensitivity analysis. Horizontal position is live graph inventory, $N(t)/N_{\rm GT}$, vertical position is learning step, and line colour uses a factorized hue--lightness encoding: hue identifies ambiguity threshold $\theta$, while increasing darkness within each threshold's colour family encodes cumulative normalized structural activity, $[C(t)+D(t)]/N_{\rm GT}$. Solid/dashed styling identifies decay $\gamma$. Because cumulative churn counts every clone and deletion operation through time, it is monotone non-decreasing and records total structural work rather than only activity in the current 50-step window. The dotted line marks ground-truth state-count parity, not representational optimality. Paths are observed trajectory medoids selected after within-seed calculation. For $3\times4$, the $\theta=40,\gamma=0.99$ condition is shown as separate active (14/30) and inactive (16/30) modes.}
"""

    orthogonal_caption = r"""\caption{Orthogonal projections of MERCURY's structural learning trajectories. For each environment, the paired panels show graph inventory, $N(t)/N_{\rm GT}$, and cumulative normalized structural operations, $[C(t)+D(t)]/N_{\rm GT}$, against a shared learning-step axis. Together the panels are orthogonal views of the same trajectory in $(N/N_{\rm GT},[C+D]/N_{\rm GT},t)$ space, separating the graph state attained from the total structural work used to attain it. Each observed trajectory medoid was selected jointly in standardized inventory--cumulative-operations trajectory space and then projected into both panels; inventory and operations representatives were not selected independently. Colour identifies ambiguity threshold $\theta$, solid/dashed styling identifies decay $\gamma$, and a black outline denotes the selected $(\theta=30,\gamma=0.99)$ configuration. The dotted inventory reference marks $N=N_{\rm GT}$, not representational optimality. For $3\times4$, the $\theta=40,\gamma=0.99$ condition is shown as separate active (14/30) and inactive (16/30) modes.}
"""

    outputs = [
        plot_dir / "mercury_structural_inventory_churn_methods.md",
        plot_dir / "mercury_structural_inventory_churn_caption.tex",
        plot_dir / "mercury_structural_inventory_churn_results.tex",
        plot_dir / "mercury_structural_trajectories_churn_hue_2d_caption.tex",
        plot_dir / "mercury_structural_churn_time_inventory_markers_caption.tex",
        plot_dir / "mercury_structural_flux_phase_space_caption.tex",
        plot_dir / "mercury_structural_trajectories_cumulative_churn_hue_2d_caption.tex",
        plot_dir / "mercury_structural_orthogonal_inventory_cumulative_churn_caption.tex",
    ]
    for path, text in zip(
        outputs,
        (
            methods,
            caption,
            results,
            hue_caption,
            churn_time_caption,
            flux_phase_caption,
            cumulative_hue_caption,
            orthogonal_caption,
        ),
    ):
        path.write_text(text, encoding="utf-8")
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", type=Path, required=True)
    parser.add_argument("--plot-dir", type=Path, required=True)
    parser.add_argument("--protocol-id", required=True)
    parser.add_argument("--batch-size", type=int, default=40)
    parser.add_argument("--levels", default="13,21,20,22")
    parser.add_argument("--final-step", type=int, default=FINAL_STEP)
    args = parser.parse_args()

    levels = {int(value.strip()) for value in args.levels.split(",") if value.strip()}
    parameters = _parameter_frame(args.study_root)
    trajectory_rows = _with_parameters(
        load_trajectory_rows(args.study_root, args.protocol_id, args.batch_size, levels),
        parameters,
    )
    template_rows = _with_parameters(
        load_template_rows(args.study_root, args.batch_size, levels),
        parameters,
    )
    if trajectory_rows.empty or template_rows.empty:
        raise SystemExit("No age-ambiguity Mercury bundles were available for structural plots.")

    trajectory_summary = summarise_trajectories(trajectory_rows)
    phase_seed_rows = construct_seed_phase_trajectories(
        trajectory_rows,
        window=FLUX_WINDOW,
        sample_interval=FLUX_SAMPLE_INTERVAL,
        final_step=args.final_step,
    )
    phase_summary = select_phase_medoids(phase_seed_rows)
    orthogonal_projection_summary = select_phase_medoids(
        phase_seed_rows,
        feature_columns=("node_ratio", "cumulative_churn_per_gt"),
    )
    flux_phase_summary = summarise_structural_flux_phase(phase_seed_rows)
    alternate_phase_summaries = [
        select_phase_medoids(
            construct_seed_phase_trajectories(
                trajectory_rows,
                window=window,
                sample_interval=FLUX_SAMPLE_INTERVAL,
                final_step=args.final_step,
            )
        )
        for window in SENSITIVITY_WINDOWS
    ]
    sensitivity = window_sensitivity(phase_summary, alternate_phase_summaries)
    template_summary = summarise_templates(template_rows)
    args.plot_dir.mkdir(parents=True, exist_ok=True)
    outputs = [
        (trajectory_summary, args.plot_dir / "mercury_structural_trajectories_summary.csv"),
        (phase_seed_rows, args.plot_dir / "mercury_structural_phase_seed.csv"),
        (phase_summary, args.plot_dir / "mercury_structural_phase_medoids.csv"),
        (
            orthogonal_projection_summary,
            args.plot_dir / "mercury_structural_orthogonal_inventory_operations_medoids.csv",
        ),
        (flux_phase_summary, args.plot_dir / "mercury_structural_flux_phase_space_summary.csv"),
        (sensitivity, args.plot_dir / "mercury_structural_phase_window_sensitivity.csv"),
        (template_rows, args.plot_dir / "mercury_sensory_template_lengths_rows.csv"),
        (template_summary, args.plot_dir / "mercury_sensory_template_lengths_summary.csv"),
    ]
    for frame, path in outputs:
        frame.to_csv(path, index=False)
        print(f"Saved {path}")
    for path in plot_structural_phase_portrait_3d(phase_summary, args.plot_dir):
        print(f"Saved {path}")
    for path in plot_structural_phase_diagnostic_grid(phase_summary, args.plot_dir):
        print(f"Saved {path}")
    for path in plot_structural_phase_diagnostic_combined(phase_summary, args.plot_dir):
        print(f"Saved {path}")
    for path in plot_structural_flux_phase_space(
        flux_phase_summary, args.plot_dir, seed_rows=phase_seed_rows
    ):
        print(f"Saved {path}")
    for path in plot_structural_flux_phase_space_zoomed(
        flux_phase_summary, args.plot_dir, seed_rows=phase_seed_rows
    ):
        print(f"Saved {path}")
    for path in plot_structural_phase_diagnostic_grid_3d(phase_summary, args.plot_dir):
        print(f"Saved {path}")
    for path in plot_structural_churn_hue_combined_2d(phase_summary, args.plot_dir):
        print(f"Saved {path}")
    for path in plot_structural_churn_hue_combined_2d(
        phase_summary, args.plot_dir, cumulative=True
    ):
        print(f"Saved {path}")
    for path in plot_structural_orthogonal_inventory_churn_views(
        orthogonal_projection_summary, args.plot_dir
    ):
        print(f"Saved {path}")
    for path in plot_structural_orthogonal_inventory_churn_views(
        orthogonal_projection_summary, args.plot_dir, environments_as_rows=True
    ):
        print(f"Saved {path}")
    for path in plot_structural_churn_time_inventory_markers(phase_summary, args.plot_dir):
        print(f"Saved {path}")
    for path in plot_structural_trajectories(trajectory_summary, args.plot_dir):
        print(f"Saved {path}")
    for path in plot_template_lengths(template_rows, template_summary, args.plot_dir):
        print(f"Saved {path}")
    for path in write_structural_reporting(phase_summary, sensitivity, args.plot_dir):
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
