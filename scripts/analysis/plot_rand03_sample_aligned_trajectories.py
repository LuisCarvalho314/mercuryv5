from __future__ import annotations

import json
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy.stats import t


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from mercury_runs.analysis.plots import publication_style as pub

try:
    import mercury_runs.analysis.plotting as plotting

    plotting.apply_light_mode_plot_style()
except Exception:
    pass


STUDY_ROOT = (
    REPO_ROOT
    / "results"
    / "current"
    / "studies"
    / "mercury_cscg_hybrid_vs_cscg_batch_size_30_seed_common_eval_walks2"
)
PLOT_OUTPUT_DIRECTORY = (
    REPO_ROOT
    / "results"
    / "current"
    / "notebook_plots"
    / "mercury_cscg_hybrid_vs_cscg_batch_size_30_seed_common_eval_walks2"
    / "sample_aligned_trajectories"
)
RAND_PROB = 0.3
LEVELS = [13, 20, 21]
BATCH_ORDER = [1, 10, 40, 100]
SERIES_ORDER = [
    "mercury_cscg_hybrid",
    "ground_truth_state_1",
    "ground_truth_state_2",
    "uniform_obs_4",
    "uniform_obs_8",
]
SERIES_LABELS = {
    name: pub.method_label(name)
    for name in SERIES_ORDER
}
SERIES_COLORS = {
    name: pub.method_color(name)
    for name in SERIES_ORDER
}
SERIES_LINESTYLES = {
    "mercury_cscg_hybrid": "-",
    "ground_truth_state_1": "-",
    "ground_truth_state_2": "-",
    "uniform_obs_4": "--",
    "uniform_obs_8": "--",
}

METRICS = [
    ("latent_precision", pub.metric_label("latent_precision")),
    ("latent_recall", pub.metric_label("latent_recall")),
    ("latent_edge_precision", "Edge precision"),
    ("latent_edge_recall", "Edge recall"),
    ("latent_edge_f1", "Edge F1"),
    ("latent_action_conditioned_edge_precision", "Action edge precision"),
    ("latent_action_conditioned_edge_recall", "Action edge recall"),
    ("latent_action_conditioned_edge_f1", "Action edge F1"),
    ("latent_mean_total_variation_similarity", "TV similarity"),
    ("latent_action_conditioned_mean_total_variation_similarity", "Action TV similarity"),
]

SUMMARY_METRICS = [
    ("latent_precision", pub.metric_label("latent_precision")),
    ("latent_mean_total_variation_similarity", "TV similarity"),
    ("latent_action_conditioned_mean_total_variation_similarity", "Action-TV similarity"),
]

BATCH_COLORS = {
    1: "#BFD7EA",
    10: "#73A9CF",
    40: "#2C7FB8",
    100: "#084B83",
}
HYBRID_BATCH_COLORS = {
    1: "#E7D4F5",
    10: "#C59AE3",
    40: "#8F5BC5",
    100: "#5A2A8A",
}


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_manifest_configs(study_root: Path) -> dict[str, dict]:
    manifest_path = study_root / "study_manifest.jsonl"
    configs: dict[str, dict] = {}
    if not manifest_path.exists():
        return configs
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        config = record.get("config", {})
        run_id = config.get("requested_run_id") or record.get("run_id")
        if run_id:
            configs[str(run_id)] = dict(config)
    return configs


def _fallback_run_config(run_root: Path) -> dict:
    path = run_root / "run_config.json"
    if not path.exists():
        return {}
    payload = _read_json(path)
    return payload.get("cli_args") or payload.get("config") or payload


def _load_history(study_root: Path) -> pl.DataFrame:
    manifest_configs = _load_manifest_configs(study_root)
    rows: list[dict] = []
    for run_root in sorted(path for path in study_root.iterdir() if path.is_dir()):
        run_id = run_root.name
        config = manifest_configs.get(run_id) or _fallback_run_config(run_root)
        status_path = run_root / "run_status.json"
        status_payload = _read_json(status_path) if status_path.exists() else {"status": "missing"}
        method = config.get("method") or status_payload.get("config", {}).get("method")
        base = {
            "run_id": run_id,
            "run_status": status_payload.get("status", "missing"),
            "run_method": method,
            "method": method,
            "level": config.get("level"),
            "seed": config.get("seed"),
            "rand_prob": config.get("rand_prob"),
            "cscg_batch_size": config.get("cscg_batch_size"),
            "cscg_clone_preset": config.get("cscg_clone_preset"),
            "cscg_clone_allocation": config.get("cscg_clone_allocation"),
            "cscg_clones_per_state": config.get("cscg_clones_per_state"),
            "cscg_clones_per_obs": config.get("cscg_clones_per_obs"),

        }
        for metric_path in sorted((run_root / "metrics").glob("*/*_paper_precision.json")):
            payload = _read_json(metric_path)
            metric_base = dict(base)
            metric_base["metric_method"] = payload.get("method", method)
            metric_base["history_method"] = None
            metric_base["metric_source"] = str(
                metric_path.relative_to(run_root))
            metric_base["metric_directory"] = metric_path.parent.name

            for point in payload.get("history", []) or []:
                row = dict(metric_base)
                for key, value in point.items():
                    if isinstance(value, (dict, list, tuple)):
                        continue
                    if key == "method":
                        row["history_method"] = value
                        if value is None:
                            continue
                    row[key] = value
                if "observed_samples" not in row and "step" in row:
                    row["observed_samples"] = row["step"]
                rows.append(row)
    if not rows:
        return pl.DataFrame()
    return pl.DataFrame(rows, infer_schema_length=None)


def _ci95_half_width(n: int | float, sd: int | float | None) -> float:
    n = int(n or 0)
    sd = float(sd or 0.0)
    if n <= 1 or not np.isfinite(sd):
        return 0.0
    return float(t.ppf(0.975, n - 1) * sd / np.sqrt(n))


def _with_similarity_columns(frame: pl.DataFrame) -> pl.DataFrame:
    expressions = []
    if "latent_mean_total_variation" in frame.columns:
        expressions.append((1.0 - pl.col("latent_mean_total_variation")).alias("latent_mean_total_variation_similarity"))
    if "latent_action_conditioned_mean_total_variation" in frame.columns:
        expressions.append(
            (1.0 - pl.col("latent_action_conditioned_mean_total_variation")).alias(
                "latent_action_conditioned_mean_total_variation_similarity"
            )
        )
    return frame.with_columns(expressions) if expressions else frame

def _effective_method_expression() -> pl.Expr:
    """Resolve the per-history-row method.

    metric_method describes the metric payload and may label an entire
    Hybrid history as MERCURY. It is therefore unsuitable for identifying
    the phase of individual history rows.
    """
    return (
        pl.coalesce(
            [
                pl.col("history_method").cast(pl.Utf8, strict=False),
                pl.col("run_method").cast(pl.Utf8, strict=False),
            ]
        )
        .str.strip_chars()
        .str.to_lowercase()
        .alias("effective_method")
    )


def _series_expression() -> pl.Expr:
    allocation = pl.col("cscg_clone_allocation").cast(pl.Utf8, strict=False)
    clones_per_state = pl.col("cscg_clones_per_state").cast(
        pl.Int64, strict=False
    )
    clones_per_obs = pl.col("cscg_clones_per_obs").cast(
        pl.Int64, strict=False
    )

    run_method = (
        pl.col("run_method")
        .cast(pl.Utf8, strict=False)
        .str.strip_chars()
        .str.to_lowercase()
    )
    history_method = (
        pl.col("history_method")
        .cast(pl.Utf8, strict=False)
        .str.strip_chars()
        .str.to_lowercase()
    )

    is_hybrid_run = run_method == "mercury_cscg_hybrid"

    is_hybrid_mercury_phase = (
        is_hybrid_run
        & history_method.is_not_null()
        & (
            (history_method == "mercury")
            | history_method.str.ends_with("_mercury")
        )
    )

    is_hybrid_cscg_phase = (
        is_hybrid_run
        & (
            history_method.is_null()
            | (history_method == "cscg")
            | history_method.str.ends_with("_cscg")
        )
    )

    return (
        pl.when(is_hybrid_cscg_phase)
        .then(pl.lit("mercury_cscg_hybrid"))
        .when(is_hybrid_mercury_phase)
        .then(None)
        .when(
            (run_method == "cscg")
            & (allocation == "ground_truth")
            & (clones_per_state == 1)
        )
        .then(pl.lit("ground_truth_state_1"))
        .when(
            (run_method == "cscg")
            & (allocation == "ground_truth")
            & (clones_per_state == 2)
        )
        .then(pl.lit("ground_truth_state_2"))
        .when(
            (run_method == "cscg")
            & (allocation == "uniform")
            & (clones_per_obs == 4)
        )
        .then(pl.lit("uniform_obs_4"))
        .when(
            (run_method == "cscg")
            & (allocation == "uniform")
            & (clones_per_obs == 8)
        )
        .then(pl.lit("uniform_obs_8"))
        .otherwise(None)
        .alias("plot_series")
    )

def plot_sample_aligned_trajectory(
    history_metrics: pl.DataFrame,
    metric: str,
    label: str,
    level: int,
) -> Path | None:
    if history_metrics.is_empty() or metric not in history_metrics.columns:
        print(f"Skipping {metric}: missing data")
        return None
    frame, x_col = _prepare_metric_frame(
        history_metrics,
        metric,
        level,
    )
    if frame.is_empty():
        print(
            f"Skipping {metric}: no level={level}, "
            f"rand_prob={RAND_PROB} rows"
        )
        return None

    per_seed_level = (
        frame.group_by(
            [
                "plot_series",
                "cscg_batch_size",
                x_col,
                "seed",
                "level",
            ]
        )
        .agg(pl.col(metric).mean().alias("level_value"))
    )

    per_seed = (
        per_seed_level.group_by(
            [
                "plot_series",
                "cscg_batch_size",
                x_col,
                "seed",
            ]
        )
        .agg(pl.col("level_value").mean().alias("value"))
        .sort(["plot_series", "cscg_batch_size", x_col, "seed"])
    )
    summary = (
        per_seed.group_by(["plot_series", "cscg_batch_size", x_col])
        .agg(
            [
                pl.col("value").mean().alias("mean"),
                pl.col("value").std().fill_null(0.0).alias("sd"),
                pl.len().alias("n"),
            ]
        )
        .sort(["plot_series", "cscg_batch_size", x_col])
    )

    fig, axes = plt.subplots(1, len(BATCH_ORDER), figsize=(11.0, 3.2), sharey=True, dpi=220)
    axes = np.ravel(axes)
    for ax, batch in zip(axes, BATCH_ORDER):
        panel = summary.filter(pl.col("cscg_batch_size") == batch)
        for series_name in SERIES_ORDER:
            rows = panel.filter(pl.col("plot_series") == series_name).sort(x_col)
            if rows.is_empty():
                continue
            steps = rows.get_column(x_col).to_numpy()
            means = rows.get_column("mean").to_numpy()
            ci95 = np.asarray(
                [
                    _ci95_half_width(n, sd)
                    for n, sd in zip(rows.get_column("n").to_list(), rows.get_column("sd").fill_null(0.0).to_list())
                ],
                dtype=float,
            )
            color = SERIES_COLORS[series_name]
            ax.plot(
                steps,
                means,
                color=color,
                linestyle=SERIES_LINESTYLES[series_name],
                linewidth=1.75 if series_name == "mercury_cscg_hybrid" else 1.25,
                alpha=0.98 if series_name == "mercury_cscg_hybrid" else 0.88,
                label=SERIES_LABELS[series_name],
            )
            ax.fill_between(steps, means - ci95, means + ci95, color=color, alpha=0.14, linewidth=0.0)
        ax.set_title(f"batch {batch}")
        ax.set_xlabel("observed samples")
        ax.set_ylim(0.0, 1.08)
        ax.grid(axis="both", alpha=0.25)
    axes[0].set_ylabel(label)
    handles = []
    for series_name in SERIES_ORDER:
        handles.append(
            plt.Line2D(
                [0],
                [0],
                color=SERIES_COLORS[series_name],
                linestyle=SERIES_LINESTYLES[series_name],
                linewidth=1.75 if series_name == "mercury_cscg_hybrid" else 1.25,
                label=SERIES_LABELS[series_name],
            )
        )
    fig.legend(handles=handles, frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.01), ncol=5, fontsize=8.2)
    fig.suptitle(
        f"Sample-aligned {label} trajectories "
        f"(level = {level}, rand_prob = {RAND_PROB})",
        y=1.03,
    )
    fig.subplots_adjust(left=0.065, right=0.995, top=0.82, bottom=0.29, wspace=0.18)
    output_path = (
            PLOT_OUTPUT_DIRECTORY
            / (
                f"trajectory_sample_aligned_level_{level}_"
                f"rand_0p3_by_config_{metric}.png"
            )
    )
    fig.savefig(output_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(output_path)
    return output_path


def _summary_source(
    history_metrics: pl.DataFrame,
    metric: str,
    level: int,
) -> pl.DataFrame:
    frame, _ = _prepare_metric_frame(
        history_metrics,
        metric,
        level,
    )
    return frame

def _hybrid_batch_summary(
    frame: pl.DataFrame,
    metric: str,
    x_col: str,
) -> pl.DataFrame:
    hybrid = frame.filter(pl.col("plot_series") == "mercury_cscg_hybrid")
    if hybrid.is_empty():
        return pl.DataFrame()

    per_seed_level = (
        hybrid.group_by(
            [
                "cscg_batch_size",
                x_col,
                "seed",
                "level",
            ]
        )
        .agg(pl.col(metric).mean().alias("level_value"))
    )

    per_seed = (
        per_seed_level.group_by(
            [
                "cscg_batch_size",
                x_col,
                "seed",
            ]
        )
        .agg(pl.col("level_value").mean().alias("value"))
    )

    return (
        per_seed.group_by(["cscg_batch_size", x_col])
        .agg(
            [
                pl.col("value").mean().alias("mean"),
                pl.col("value").std().fill_null(0.0).alias("sd"),
                pl.len().alias("n"),
            ]
        )
        .sort(["cscg_batch_size", x_col])
    )


def _cscg_batch_envelope(
    frame: pl.DataFrame,
    metric: str,
    x_col: str,
) -> pl.DataFrame:
    cscg = frame.filter(pl.col("plot_series") != "mercury_cscg_hybrid")
    if cscg.is_empty():
        return pl.DataFrame()

    per_seed_level = (
        cscg.group_by(
            [
                "cscg_batch_size",
                "plot_series",
                x_col,
                "seed",
                "level",
            ]
        )
        .agg(pl.col(metric).mean().alias("level_value"))
    )

    per_seed = (
        per_seed_level.group_by(
            [
                "cscg_batch_size",
                "plot_series",
                x_col,
                "seed",
            ]
        )
        .agg(pl.col("level_value").mean().alias("value"))
    )

    preset_means = (
        per_seed.group_by(
            [
                "cscg_batch_size",
                "plot_series",
                x_col,
            ]
        )
        .agg(pl.col("value").mean().alias("mean"))
    )

    return (
        preset_means.group_by(["cscg_batch_size", x_col])
        .agg(
            [
                pl.col("mean").max().alias("mean"),
                pl.col("plot_series")
                .sort_by("mean")
                .last()
                .alias("best_preset"),
            ]
        )
        .sort(["cscg_batch_size", x_col])
    )


def plot_batch_envelope_summary(
    history_metrics: pl.DataFrame,
    metrics: list[tuple[str, str]],
    output_name: str,
    *,
    level: int,
    figsize: tuple[float, float],
) -> Path | None:
    x_col = "observed_samples" if "observed_samples" in history_metrics.columns else "step"
    fig, axes = plt.subplots(1, len(metrics), figsize=figsize, sharey=True, dpi=260)
    axes = np.ravel(axes)
    wrote_any = False
    for axis, (metric, label) in zip(axes, metrics):
        frame = _summary_source(
            history_metrics,
            metric,
            level,
        )
        if frame.is_empty():
            axis.set_visible(False)
            continue
        wrote_any = True
        hybrid = _hybrid_batch_summary(frame, metric, x_col)
        if not hybrid.is_empty():
            for batch in BATCH_ORDER:
                rows = hybrid.filter(pl.col("cscg_batch_size") == batch).sort(x_col)
                if rows.is_empty():
                    continue
                steps = rows.get_column(x_col).to_numpy()
                means = rows.get_column("mean").to_numpy()
                ci95 = np.asarray(
                    [
                        _ci95_half_width(n, sd)
                        for n, sd in zip(rows.get_column("n").to_list(), rows.get_column("sd").fill_null(0.0).to_list())
                    ],
                    dtype=float,
                )
                color = HYBRID_BATCH_COLORS.get(batch, pub.method_color("hybrid"))
                axis.plot(steps, means, color=color, linewidth=2.25, linestyle="-", label=f"{pub.method_label('hybrid')} batch {batch}")
                axis.fill_between(steps, means - ci95, means + ci95, color=color, alpha=0.13, linewidth=0.0)

        envelope = _cscg_batch_envelope(frame, metric, x_col)
        for batch in BATCH_ORDER:
            rows = envelope.filter(pl.col("cscg_batch_size") == batch).sort(x_col)
            if rows.is_empty():
                continue
            axis.plot(
                rows.get_column(x_col).to_numpy(),
                rows.get_column("mean").to_numpy(),
                color=BATCH_COLORS.get(batch, "#718096"),
                linewidth=1.4,
                linestyle="--",
                label=f"CSCG batch {batch}",
            )
        axis.set_title(label)
        axis.set_xscale("symlog", linthresh=2000, linscale=1.0)
        axis.set_xlim(left=0.0)
        axis.set_ylim(0.0, 1.02)
        axis.set_xlabel("observed samples")
        axis.grid(axis="both", alpha=0.24)
    if not wrote_any:
        plt.close(fig)
        return None
    axes[0].set_ylabel("metric value")
    handles = [
        *[
            plt.Line2D([0], [0], color=HYBRID_BATCH_COLORS[batch], linewidth=2.25, linestyle="-", label=f"{pub.method_label('hybrid')} {batch}")
            for batch in BATCH_ORDER
        ],
        *[
            plt.Line2D([0], [0], color=BATCH_COLORS[batch], linewidth=1.4, linestyle="--", label=f"CSCG {batch}")
            for batch in BATCH_ORDER
        ],
    ]
    fig.legend(handles=handles, frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.05), ncol=4, fontsize=8.2)
    fig.suptitle("Sample efficiency across CSCG update batch sizes", y=1.02)
    fig.text(
        0.5,
        0.925,
        (
            "Hybrid and pointwise-best CSCG clone preset shown separately "
            f"for each batch size, level = {level}, "
            f"rand_prob = {RAND_PROB}"
        ),
        ha="center",
        va="center",
        fontsize=9.4,
    )
    fig.subplots_adjust(left=0.065, right=0.995, top=0.80, bottom=0.34, wspace=0.16)
    output_path = PLOT_OUTPUT_DIRECTORY / output_name
    fig.savefig(output_path, dpi=260, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(output_path)
    return output_path


def write_envelope_summary(
    history_metrics: pl.DataFrame,
    level: int,
) -> Path:
    x_col = (
        "observed_samples"
        if "observed_samples" in history_metrics.columns
        else "step"
    )

    records: list[dict] = []

    for metric, label in SUMMARY_METRICS:
        frame = _summary_source(
            history_metrics,
            metric,
            level,
        )

        if frame.is_empty():
            continue

        envelope = _cscg_batch_envelope(
            frame,
            metric,
            x_col,
        )

        for row in envelope.iter_rows(named=True):
            records.append(
                {
                    "metric": metric,
                    "label": label,
                    "level": level,
                    "rand_prob": RAND_PROB,
                    "cscg_batch_size": row["cscg_batch_size"],
                    "observed_samples": row[x_col],
                    "envelope_mean": row["mean"],
                    "best_preset": row["best_preset"],
                }
            )

    output_path = (
        PLOT_OUTPUT_DIRECTORY
        / (
            f"trajectory_sample_aligned_level_{level}_"
            "rand_0p3_batch_envelope_summary.csv"
        )
    )

    pl.DataFrame(records).write_csv(output_path)
    print(output_path)
    return output_path

def _prepare_metric_frame(
    history_metrics: pl.DataFrame,
    metric: str,
    level: int,
) -> tuple[pl.DataFrame, str]:
    if history_metrics.is_empty() or metric not in history_metrics.columns:
        return pl.DataFrame(), "observed_samples"

    x_col = (
        "observed_samples"
        if "observed_samples" in history_metrics.columns
        else "step"
    )

    frame = (
        history_metrics
        .filter(pl.col("run_status") == "completed")
        .filter(pl.col("rand_prob") == RAND_PROB)
        .filter(
            pl.col("level").cast(pl.Int64, strict=False) == level
        )
        .with_columns(_effective_method_expression())
        .with_columns(_series_expression())
        .filter(pl.col("plot_series").is_in(SERIES_ORDER))
        .drop_nulls(
            [
                "plot_series",
                "cscg_batch_size",
                x_col,
                metric,
            ]
        )
    )

    _assert_unique_history_rows(frame, metric, x_col)
    return frame, x_col

def _assert_unique_history_rows(
    frame: pl.DataFrame,
    metric: str,
    x_col: str,
) -> None:
    if frame.is_empty():
        return

    duplicates = (
        frame
        .group_by(
            [
                "run_id",
                "plot_series",
                "cscg_batch_size",
                "seed",
                "level",
                x_col,
            ]
        )
        .agg(
            [
                pl.len().alias("n_rows"),
                pl.col("metric_source").n_unique().alias("n_sources"),
                pl.col("metric_source").unique().alias("sources"),
            ]
        )
        .filter(pl.col("n_rows") > 1)
        .sort("n_rows", descending=True)
    )

    if not duplicates.is_empty():
        raise RuntimeError(
            f"Multiple histories remain for metric {metric!r} at the same "
            "run/checkpoint. These must be filtered rather than averaged.\n"
            f"{duplicates.head(30)}"
        )

def main() -> None:
    PLOT_OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    history_metrics = _with_similarity_columns(
        _load_history(STUDY_ROOT)
    )

    hybrid_plot_audit = (
        history_metrics
        .filter(
            pl.col("run_method")
            == "mercury_cscg_hybrid"
        )
        .with_columns(_effective_method_expression())
        .with_columns(_series_expression())
        .with_columns(
            pl.when(
                pl.col("history_method")
                .cast(pl.Utf8, strict=False)
                .str.to_lowercase()
                .str.ends_with("_cscg")
            )
            .then(pl.lit("CSCG phase"))
            .when(
                pl.col("history_method")
                .cast(pl.Utf8, strict=False)
                .str.to_lowercase()
                .str.ends_with("_mercury")
            )
            .then(pl.lit("MERCURY phase"))
            .otherwise(pl.lit("unknown"))
            .alias("hybrid_phase")
        )
        .group_by(
            [
                "metric_method",
                "history_method",
                "hybrid_phase",
                "effective_method",
                "plot_series",
            ]
        )
        .len()
        .sort(["history_method"])
    )

    print("\nHybrid plot-series audit:")
    print(hybrid_plot_audit)

    for level in LEVELS:
        print(f"\nWriting plots for level {level}")

        paths = [
            plot_sample_aligned_trajectory(
                history_metrics,
                metric,
                label,
                level,
            )
            for metric, label in METRICS
        ]

        paths.append(
            plot_batch_envelope_summary(
                history_metrics,
                SUMMARY_METRICS,
                (
                    f"trajectory_sample_aligned_level_{level}_"
                    "rand_0p3_batch_envelope_summary_1x3.png"
                ),
                level=level,
                figsize=(10.8, 3.5),
            )
        )

        paths.append(
            plot_batch_envelope_summary(
                history_metrics,
                SUMMARY_METRICS[1:],
                (
                    f"trajectory_sample_aligned_level_{level}_"
                    "rand_0p3_batch_envelope_"
                    "transition_similarity_1x2.png"
                ),
                level=level,
                figsize=(8.0, 3.5),
            )
        )

        envelope_summary_path = write_envelope_summary(
            history_metrics,
            level,
        )

        written = [
            str(path)
            for path in paths
            if path is not None
        ]

        summary_path = (
            PLOT_OUTPUT_DIRECTORY
            / (
                f"trajectory_sample_aligned_level_{level}_"
                "rand_0p3_summary.json"
            )
        )

        summary_path.write_text(
            json.dumps(
                {
                    "study_root": str(STUDY_ROOT),
                    "level": level,
                    "rand_prob": RAND_PROB,
                    "metrics": [
                        metric
                        for metric, _ in METRICS
                    ],
                    "outputs": written,
                    "envelope_summary_csv": str(
                        envelope_summary_path
                    ),
                    "tv_similarity_definition": (
                        "1 - latent_mean_total_variation"
                    ),
                    "action_tv_similarity_definition": (
                        "1 - "
                        "latent_action_conditioned_"
                        "mean_total_variation"
                    ),
                    "cscg_envelope_definition": (
                        "For each batch size and "
                        "observed-sample checkpoint, "
                        "the CSCG line is the maximum "
                        "mean similarity across clone "
                        "presets. The selected preset "
                        "can change across checkpoints."
                    ),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        print(summary_path)


if __name__ == "__main__":
    main()
