"""Paired pass-2 to pass-1000 analysis of NLL and representation quality.

This replaces the former multi-panel delta plot with compact tables.  Every
comparison is made within one initialisation and environment across matching
training seeds; initialisations are never pooled.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
from scipy import stats

try:
    from scripts.analysis.plot_config_metric_trajectories import (
        DEFAULT_LEVEL_ORDER,
        LEVEL_LABELS,
        METRICS,
        SERIES_COLORS,
        SERIES_LABELS,
        SERIES_ORDER,
        _metrics_from_record,
        load_rows,
        rows_at_comparison_steps,
    )
    from scripts.analysis.study_selection import MAIN_STUDY_ROOT, PROTOCOL_ID
except ModuleNotFoundError:  # Support direct execution via ``python scripts/analysis/...``.
    from plot_config_metric_trajectories import (
        DEFAULT_LEVEL_ORDER,
        LEVEL_LABELS,
        METRICS,
        SERIES_COLORS,
        SERIES_LABELS,
        SERIES_ORDER,
        _metrics_from_record,
        load_rows,
        rows_at_comparison_steps,
    )
    from study_selection import MAIN_STUDY_ROOT, PROTOCOL_ID

DEFAULT_STUDY_ROOT = MAIN_STUDY_ROOT
DEFAULT_OUTPUT_DIR = Path(
    "results/current/notebook_plots/main_study/pass_2_to_1000_statistical_analysis"
)
DEFAULT_PROTOCOL_ID = PROTOCOL_ID
DEFAULT_OUTPUT_STEM = "pass_2_to_1000"
BASELINE_PASS_STEP = 4_000
THOUSANDTH_PASS_STEP = 2_000_000
INITIALISATION_STEP = 0
EXPECTED_SEEDS = tuple(range(30))
LATEX_LEVEL_LABELS = {
    13: r"$3\times3$",
    21: r"$1\times4$",
    20: r"$1\times5$",
    22: r"$3\times4$",
}
NLL_METRIC = "eval_nll_bps"
PRIMARY_METRIC = "action_tv_similarity"
SECONDARY_METRIC = "action_no_self_loop_tv_similarity"
PREDICTOR_METRICS = (PRIMARY_METRIC, SECONDARY_METRIC)

METRIC_LABELS = {
    "latent_precision": "Latent purity",
    "latent_recall": "Latent concentration",
    "action_edge_f1_auc": "Action-conditioned edge F1 AUC",
    "action_no_self_loop_edge_f1_auc": "Action-conditioned no-self-loop edge F1 AUC",
    PRIMARY_METRIC: "Action-conditioned TV similarity",
    SECONDARY_METRIC: "Action-conditioned no-self-loop TV similarity",
    NLL_METRIC: "CSCG evaluation NLL (bits/symbol)",
}
COMPACT_SERIES_LABELS = {
    "mercury_cscg_hybrid": "MERCURY",
    "tmgwr_cscg_hybrid": "TMGWR",
    "vanilla_cscg": "Standard CSCG",
    "oracle_capacity": "Oracle-C",
    "oracle_capacity_first_order": "Oracle-C+1st",
    "oracle_capacity_structural": "Oracle-C+Struct",
}
# Match the high-separation method palette used by the main forest-matrix
# profile so method identity is consistent across manuscript figures.
SERIES_COLORS = {
    "mercury_cscg_hybrid": "#D55E00",
    "tmgwr_cscg_hybrid": "#E6A700",
    "vanilla_cscg": "#27313D",
    "oracle_capacity": "#0072B2",
    "oracle_capacity_first_order": "#CC79A7",
    "oracle_capacity_structural": "#009E73",
}


def paired_improvement_deltas(
    rows: pd.DataFrame,
    *,
    baseline_step: int = BASELINE_PASS_STEP,
    final_step: int = THOUSANDTH_PASS_STEP,
) -> pd.DataFrame:
    """Return paired endpoints and changes, oriented so positive means improvement."""
    endpoint_rows = rows.loc[rows["step"].isin([baseline_step, final_step])].copy()
    keys = ["run_id", "series", "level", "seed", "metric"]
    if endpoint_rows.duplicated(keys + ["step"]).any():
        raise ValueError("Duplicate metric records found at an endpoint snapshot.")

    endpoint_counts = endpoint_rows.groupby(keys, dropna=False)["step"].nunique()
    incomplete = endpoint_counts[endpoint_counts != 2]
    if not incomplete.empty:
        raise ValueError(
            f"Missing pass-{baseline_step // 2_000:g} or pass-{final_step // 2_000:g} endpoint "
            f"for {len(incomplete)} run/metric pairs."
        )

    paired = endpoint_rows.pivot(index=keys, columns="step", values="value")
    if baseline_step not in paired or final_step not in paired:
        raise ValueError("No complete pass-2 to pass-1000 endpoint pairs were found.")
    if paired[[baseline_step, final_step]].isna().any().any():
        raise ValueError("Non-finite or missing endpoint values found in paired records.")
    paired = paired.reset_index().rename(
        columns={
            baseline_step: "value_pass_2",
            final_step: "value_pass_1000",
        }
    )
    paired["raw_change"] = paired["value_pass_1000"] - paired["value_pass_2"]
    paired["change"] = paired["raw_change"]
    nll_rows = paired["metric"] == NLL_METRIC
    paired.loc[nll_rows, "change"] *= -1.0
    # Retain the old column name for callers that consumed the previous helper.
    paired["delta"] = paired["change"]
    return paired


def _mean_change_interval(values: np.ndarray) -> dict[str, float | int]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    n = int(finite.size)
    mean = float(np.mean(finite)) if n else math.nan
    sd = float(np.std(finite, ddof=1)) if n > 1 else math.nan
    se = sd / math.sqrt(n) if n > 1 else math.nan
    if n < 2:
        low = high = math.nan
    elif sd == 0.0:
        low = high = mean
    else:
        margin = float(stats.t.ppf(0.975, n - 1) * se)
        low, high = mean - margin, mean + margin
    return {
        "n": n,
        "mean_change": mean,
        "sd_change": sd,
        "se_change": se,
        "ci95_low": low,
        "ci95_high": high,
    }


def _change_assessment(metric: str, low: float, high: float) -> str:
    if not np.isfinite(low) or not np.isfinite(high):
        return "unavailable"
    if metric == NLL_METRIC:
        if low > 0.0:
            return "reliably_decreased"
        if high < 0.0:
            return "reliably_increased"
        return "no_reliable_change"
    if low > 0.0:
        return "improved"
    if high < 0.0:
        return "deteriorated"
    return "unchanged_or_uncertain"


def summarise_paired_changes(paired: pd.DataFrame) -> pd.DataFrame:
    """Summarise absolute endpoints and paired changes with Student-t intervals."""
    output: list[dict[str, Any]] = []
    for (series, level, metric), group in paired.groupby(
        ["series", "level", "metric"], sort=False
    ):
        interval = _mean_change_interval(group["change"].to_numpy(dtype=float))
        output.append(
            {
                "series": series,
                "series_label": COMPACT_SERIES_LABELS.get(
                    str(series), SERIES_LABELS.get(str(series), str(series))
                ),
                "level": int(level),
                "level_label": LEVEL_LABELS.get(int(level), f"L{level}"),
                "metric": metric,
                "metric_label": METRIC_LABELS.get(str(metric), str(metric)),
                "mean_pass_2": float(group["value_pass_2"].mean()),
                "mean_pass_1000": float(group["value_pass_1000"].mean()),
                **interval,
                "assessment": _change_assessment(
                    str(metric), float(interval["ci95_low"]), float(interval["ci95_high"])
                ),
            }
        )
    return pd.DataFrame(output)


def predictor_correlations(
    paired: pd.DataFrame,
    *,
    metrics: Iterable[str] = PREDICTOR_METRICS,
) -> pd.DataFrame:
    """Relate NLL improvement to metric improvement without pooling conditions."""
    output: list[dict[str, Any]] = []
    for metric in metrics:
        merged = paired_predictor_values(paired, metric=metric)
        for (series, level), group in merged.groupby(["series", "level"], sort=False):
            values = group[["nll_improvement", "metric_change"]].dropna()
            n = len(values)
            reasons: list[str] = []
            if n < 2:
                rho = math.nan
                reasons.append("fewer than two paired seeds")
            elif values["nll_improvement"].nunique() < 2:
                rho = math.nan
                reasons.append("constant NLL improvement")
            elif values["metric_change"].nunique() < 2:
                rho = math.nan
                reasons.append("constant metric change")
            else:
                rho = float(
                    stats.spearmanr(
                        values["nll_improvement"], values["metric_change"]
                    ).statistic
                )
            output.append(
                {
                    "series": series,
                    "series_label": COMPACT_SERIES_LABELS.get(
                        str(series), SERIES_LABELS.get(str(series), str(series))
                    ),
                    "level": int(level),
                    "level_label": LEVEL_LABELS.get(int(level), f"L{level}"),
                    "metric": metric,
                    "metric_label": METRIC_LABELS.get(metric, metric),
                    "n_pairs": n,
                    "rho_s": rho,
                    "available": np.isfinite(rho),
                    "unavailable_reason": "; ".join(reasons),
                }
            )
    return pd.DataFrame(output)


def paired_predictor_values(
    paired: pd.DataFrame,
    *,
    metric: str = PRIMARY_METRIC,
) -> pd.DataFrame:
    """Align per-seed NLL and structural improvements within each condition."""
    keys = ["run_id", "series", "level", "seed"]
    nll = paired.loc[paired["metric"] == NLL_METRIC, keys + ["change"]].rename(
        columns={"change": "nll_improvement"}
    )
    metric_rows = paired.loc[paired["metric"] == metric, keys + ["change"]].rename(
        columns={"change": "metric_change"}
    )
    return nll.merge(metric_rows, on=keys, how="inner", validate="one_to_one")


def paired_absolute_values(
    paired: pd.DataFrame,
    *,
    metric: str = PRIMARY_METRIC,
) -> pd.DataFrame:
    """Align absolute NLL and structural endpoints for every matched seed."""
    keys = ["run_id", "series", "level", "seed"]
    endpoint_columns = keys + ["value_pass_2", "value_pass_1000"]
    nll = paired.loc[paired["metric"] == NLL_METRIC, endpoint_columns].rename(
        columns={
            "value_pass_2": "nll_pass_2",
            "value_pass_1000": "nll_pass_1000",
        }
    )
    metric_rows = paired.loc[paired["metric"] == metric, endpoint_columns].rename(
        columns={
            "value_pass_2": "metric_pass_2",
            "value_pass_1000": "metric_pass_1000",
        }
    )
    return nll.merge(metric_rows, on=keys, how="inner", validate="one_to_one")


def seed_response_frame(
    paired: pd.DataFrame,
    *,
    metric: str,
) -> pd.DataFrame:
    """Align seed-level metric endpoints, metric change, and NLL improvement."""
    keys = ["run_id", "series", "level", "seed"]
    nll = paired.loc[
        paired["metric"] == NLL_METRIC,
        keys + ["value_pass_2", "value_pass_1000", "change"],
    ].rename(
        columns={
            "value_pass_2": "nll_pass_2",
            "value_pass_1000": "nll_pass_1000",
            "change": "nll_improvement",
        }
    )
    metric_rows = paired.loc[
        paired["metric"] == metric,
        keys + ["value_pass_2", "value_pass_1000", "change"],
    ].rename(
        columns={
            "value_pass_2": "quality_pass_2",
            "value_pass_1000": "quality_pass_1000",
            "change": "quality_change",
        }
    )
    result = nll.merge(metric_rows, on=keys, how="inner", validate="one_to_one")
    result.insert(4, "metric", metric)
    return result


def three_checkpoint_seed_frame(
    rows: pd.DataFrame,
    *,
    metric: str,
    initialisation_step: int = INITIALISATION_STEP,
    pass_1_step: int | None = None,
    pass_2_step: int = BASELINE_PASS_STEP,
    pass_1000_step: int = THOUSANDTH_PASS_STEP,
    allow_missing_pass_1_series: set[str] | None = None,
) -> pd.DataFrame:
    """Align NLL and quality; hybrid initialisation is post-transfer CSCG step 0."""
    keys = ["run_id", "series", "level", "seed"]
    checkpoints = [initialisation_step]
    if pass_1_step is not None:
        checkpoints.append(pass_1_step)
    checkpoints.extend([pass_2_step, pass_1000_step])
    selected = rows.loc[
        rows["metric"].isin([NLL_METRIC, metric]) & rows["step"].isin(checkpoints),
        keys + ["metric", "step", "value"],
    ].copy()
    if selected.duplicated(keys + ["metric", "step"]).any():
        raise ValueError("Duplicate metric records found in three-checkpoint trajectories.")

    wide = selected.pivot(index=keys, columns=["metric", "step"], values="value")
    required = [(name, step) for name in (NLL_METRIC, metric) for step in checkpoints]
    if any(column not in wide.columns for column in required):
        if not (pass_1_step is not None and allow_missing_pass_1_series):
            raise ValueError("No complete three-checkpoint NLL/quality trajectories were found.")
        wide = wide.reindex(columns=required)
    for name in (NLL_METRIC, metric):
        for step in checkpoints:
            if step == pass_1_step and allow_missing_pass_1_series:
                required_rows = ~wide.index.get_level_values("series").isin(allow_missing_pass_1_series)
            else:
                required_rows = np.ones(len(wide), dtype=bool)
            if wide.loc[required_rows, (name, step)].isna().any():
                raise ValueError("Missing initialisation, pass-2, or pass-1000 value in seed/metric trajectories.")
    result = wide.loc[:, required].reset_index()
    checkpoint_labels = ["initialisation"]
    if pass_1_step is not None:
        checkpoint_labels.append("pass_1")
    checkpoint_labels.extend(["pass_2", "pass_1000"])
    result.columns = [
        *keys,
        *(f"nll_{label}" for label in checkpoint_labels),
        *(f"quality_{label}" for label in checkpoint_labels),
    ]
    result.insert(4, "metric", metric)
    return result


def plot_paired_absolute_values(
    paired: pd.DataFrame,
    output_dir: Path,
    *,
    output_stem: str,
) -> list[Path]:
    """Plot absolute pass-2 and pass-1000 NLL/structure endpoints for every seed."""
    values = paired_absolute_values(paired, metric=PRIMARY_METRIC)
    if values.empty:
        raise ValueError("No paired NLL and transition-similarity endpoints were found.")

    present_series = set(values["series"].astype(str))
    present_levels = set(values["level"].astype(int))
    series_order = [series for series in SERIES_ORDER if series in present_series]
    level_order = [level for level in DEFAULT_LEVEL_ORDER if level in present_levels]
    if not series_order or not level_order:
        raise ValueError("Paired predictor values do not contain recognised conditions.")

    figure, axes = plt.subplots(
        len(series_order),
        len(level_order),
        figsize=(10.2, 11.8),
        squeeze=False,
        constrained_layout=True,
    )
    pass_2_color = "#6B7280"
    pass_1000_color = "#0072B2"

    for row_index, series in enumerate(series_order):
        for column_index, level in enumerate(level_order):
            axis = axes[row_index, column_index]
            condition = values.loc[
                (values["series"] == series) & (values["level"] == level)
            ].sort_values("seed")
            if condition.empty:
                axis.set_visible(False)
                continue

            for seed_row in condition.itertuples(index=False):
                axis.plot(
                    [seed_row.nll_pass_2, seed_row.nll_pass_1000],
                    [seed_row.metric_pass_2, seed_row.metric_pass_1000],
                    color="#CBD5E1",
                    linewidth=0.7,
                    alpha=0.7,
                    zorder=1,
                )
            axis.scatter(
                condition["nll_pass_2"],
                condition["metric_pass_2"],
                c=pass_2_color,
                marker="o",
                s=20,
                alpha=0.72,
                edgecolors="white",
                linewidths=0.35,
                zorder=2,
            )
            axis.scatter(
                condition["nll_pass_1000"],
                condition["metric_pass_1000"],
                c=pass_1000_color,
                marker="^",
                s=25,
                alpha=0.82,
                edgecolors="white",
                linewidths=0.35,
                zorder=3,
            )
            axis.margins(x=0.12, y=0.15)
            axis.grid(True, color="#D1D5DB", linewidth=0.45, alpha=0.65, zorder=0)
            axis.xaxis.set_major_locator(MaxNLocator(nbins=3))
            axis.yaxis.set_major_locator(MaxNLocator(nbins=3))
            axis.tick_params(labelsize=7)

            axis.text(
                0.97, 0.95, f"$n={len(condition)}$", transform=axis.transAxes,
                ha="right", va="top", fontsize=7.5,
            )
            if row_index == 0:
                axis.set_title(LEVEL_LABELS.get(level, f"L{level}"), fontsize=10)
            if column_index == 0:
                method = COMPACT_SERIES_LABELS.get(series, SERIES_LABELS.get(series, series))
                axis.set_ylabel(rf"{method}" + "\n" + r"$S^A_{\mathrm{TV}}$", fontsize=9)
            if row_index == len(series_order) - 1:
                axis.set_xlabel("Evaluation NLL", fontsize=9)

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=pass_2_color,
            markeredgecolor="white",
            markersize=6,
            label="Pass 2",
        ),
        Line2D(
            [0],
            [0],
            marker="^",
            linestyle="none",
            markerfacecolor=pass_1000_color,
            markeredgecolor="white",
            markersize=6,
            label="Pass 1000",
        ),
    ]
    figure.suptitle(
        "Absolute NLL and transition similarity before and after refinement",
        fontsize=13,
        fontweight="semibold",
        y=1.035,
    )
    figure.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.015),
        ncol=2,
        frameon=False,
        fontsize=8.5,
    )
    figure.text(
        0.5,
        -0.012,
        (
            "Each line joins the pass-2 and pass-1000 endpoints for one matched training seed; "
            "axes are fitted independently. Lower NLL and higher transition similarity are desirable."
        ),
        ha="center",
        va="top",
        fontsize=8,
    )

    paths = [
        output_dir / f"{output_stem}_paired_absolute_values.png",
        output_dir / f"{output_stem}_paired_absolute_values.pdf",
    ]
    for path in paths:
        figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    return paths


def validate_analysis(
    paired: pd.DataFrame,
    summary: pd.DataFrame,
    correlations: pd.DataFrame,
    *,
    expected_seeds: Iterable[int] = EXPECTED_SEEDS,
) -> str:
    expected_seed_set = set(int(seed) for seed in expected_seeds)
    findings: list[str] = []
    for (series, level, metric), group in paired.groupby(["series", "level", "metric"]):
        seeds = set(int(seed) for seed in group["seed"])
        if seeds != expected_seed_set:
            findings.append(
                f"[ERROR] series={series}, level={level}, metric={metric}: "
                f"missing={sorted(expected_seed_set - seeds)}, extra={sorted(seeds - expected_seed_set)}"
            )
    if not summary["n"].eq(len(expected_seed_set)).all():
        findings.append("[ERROR] one or more summary rows has an unexpected pair count")
    if not correlations["n_pairs"].eq(len(expected_seed_set)).all():
        findings.append("[ERROR] one or more correlation rows has an unexpected pair count")
    expected_rows = len(SERIES_ORDER) * len(DEFAULT_LEVEL_ORDER) * len(METRICS)
    if len(summary) != expected_rows:
        findings.append(
            f"[ERROR] expected {expected_rows} method/environment/metric summaries, found {len(summary)}"
        )
    lines = [
        "Pass-2 to pass-1000 paired analysis validation report",
        "=" * 52,
        "",
        f"Paired seed/metric rows: {len(paired)}",
        f"Summary rows: {len(summary)}",
        f"Predictor-correlation rows: {len(correlations)}",
        f"Expected seeds per condition: {len(expected_seed_set)}",
        f"Initialisations: {sorted(paired['series'].unique().tolist())}",
        f"Environments: {sorted(int(value) for value in paired['level'].unique())}",
        f"Metrics: {sorted(paired['metric'].unique().tolist())}",
        "",
        "Findings",
        "-" * 8,
        *(findings or ["No validation warnings or errors."]),
    ]
    return "\n".join(lines) + "\n"


def _format_rho(value: float) -> str:
    return "NA" if not np.isfinite(value) else f"{value:+.3f}"


def _direction(before: float, after: float) -> str:
    if after > before:
        return "up"
    if after < before:
        return "down"
    return "unchanged"


def _change_supported(low: float, high: float) -> bool:
    return bool(np.isfinite(low) and np.isfinite(high) and (low > 0.0 or high < 0.0))


def _latex_endpoint_only(before: float, after: float) -> str:
    return rf"${before:.3f}\!\rightarrow\!{after:.3f}$"


def _latex_change_cell(mean: float, low: float, high: float) -> str:
    return rf"${mean:.3f}\,[{low:.3f},{high:.3f}]$"


def _markdown_change_cell(mean: float, low: float, high: float) -> str:
    return f"{mean:+.3f} [{low:+.3f}, {high:+.3f}]"


def _table_rows(
    summary: pd.DataFrame,
    correlations: pd.DataFrame,
    *,
    metric: str,
) -> list[dict[str, Any]]:
    metric_summary = summary.loc[summary["metric"] == metric].set_index(["series", "level"])
    nll_summary = summary.loc[summary["metric"] == NLL_METRIC].set_index(["series", "level"])
    correlation_rows = correlations.loc[correlations["metric"] == metric].set_index(
        ["series", "level"]
    )
    rows: list[dict[str, Any]] = []
    for series in SERIES_ORDER:
        for level in DEFAULT_LEVEL_ORDER:
            metric_row = metric_summary.loc[(series, level)]
            nll_row = nll_summary.loc[(series, level)]
            correlation = correlation_rows.loc[(series, level)]
            rows.append(
                {
                    "series": series,
                    "method": COMPACT_SERIES_LABELS.get(series, SERIES_LABELS.get(series, series)),
                    "level": level,
                    "environment": LEVEL_LABELS.get(level, f"L{level}"),
                    "metric": metric,
                    "metric_pass_2": metric_row["mean_pass_2"],
                    "metric_pass_1000": metric_row["mean_pass_1000"],
                    "metric_mean_change": metric_row["mean_change"],
                    "metric_ci95_low": metric_row["ci95_low"],
                    "metric_ci95_high": metric_row["ci95_high"],
                    "metric_assessment": metric_row["assessment"],
                    "nll_pass_2": nll_row["mean_pass_2"],
                    "nll_pass_1000": nll_row["mean_pass_1000"],
                    "nll_mean_improvement": nll_row["mean_change"],
                    "nll_ci95_low": nll_row["ci95_low"],
                    "nll_ci95_high": nll_row["ci95_high"],
                    "nll_assessment": nll_row["assessment"],
                    "rho_s": correlation["rho_s"],
                    "n_pairs": metric_row["n"],
                }
            )
    return rows


def publication_table_frame(
    summary: pd.DataFrame, correlations: pd.DataFrame, *, metric: str
) -> pd.DataFrame:
    return pd.DataFrame(_table_rows(summary, correlations, metric=metric))


def main_before_after_frame(
    summary: pd.DataFrame,
    correlations: pd.DataFrame,
) -> pd.DataFrame:
    rows = publication_table_frame(summary, correlations, metric=PRIMARY_METRIC).copy()
    rows["metric_direction"] = rows.apply(
        lambda row: _direction(float(row["metric_pass_2"]), float(row["metric_pass_1000"])),
        axis=1,
    )
    rows["metric_change_supported"] = rows.apply(
        lambda row: _change_supported(
            float(row["metric_ci95_low"]), float(row["metric_ci95_high"])
        ),
        axis=1,
    )
    rows["nll_direction"] = rows.apply(
        lambda row: _direction(float(row["nll_pass_2"]), float(row["nll_pass_1000"])),
        axis=1,
    )
    rows["nll_change_supported"] = rows.apply(
        lambda row: _change_supported(
            float(row["nll_ci95_low"]), float(row["nll_ci95_high"])
        ),
        axis=1,
    )
    return rows[
        [
            "series",
            "method",
            "level",
            "environment",
            "metric_pass_2",
            "metric_pass_1000",
            "metric_direction",
            "metric_change_supported",
            "nll_pass_2",
            "nll_pass_1000",
            "nll_direction",
            "nll_change_supported",
            "n_pairs",
        ]
    ]


def condition_metric_response_frame(
    summary: pd.DataFrame,
    *,
    metric: str,
) -> pd.DataFrame:
    """Return 24 condition means for an endpoint metric and NLL improvement."""
    metric_rows = summary.loc[summary["metric"] == metric].set_index(["series", "level"])
    nll = summary.loc[summary["metric"] == NLL_METRIC].set_index(["series", "level"])
    rows: list[dict[str, Any]] = []
    for series in SERIES_ORDER:
        for level in DEFAULT_LEVEL_ORDER:
            metric_row = metric_rows.loc[(series, level)]
            nll_row = nll.loc[(series, level)]
            rows.append(
                {
                    "series": series,
                    "initialisation": COMPACT_SERIES_LABELS.get(series, series),
                    "level": int(level),
                    "environment": LEVEL_LABELS.get(int(level), f"L{level}"),
                    "metric": metric,
                    "metric_label": METRIC_LABELS.get(metric, metric),
                    "n_pairs": int(metric_row["n"]),
                    "quality_pass_2": float(metric_row["mean_pass_2"]),
                    "quality_pass_1000": float(metric_row["mean_pass_1000"]),
                    "quality_change": float(metric_row["mean_change"]),
                    "quality_change_ci95_low": float(metric_row["ci95_low"]),
                    "quality_change_ci95_high": float(metric_row["ci95_high"]),
                    "nll_pass_2": float(nll_row["mean_pass_2"]),
                    "nll_pass_1000": float(nll_row["mean_pass_1000"]),
                    "nll_improvement": float(nll_row["mean_change"]),
                    "nll_improvement_ci95_low": float(nll_row["ci95_low"]),
                    "nll_improvement_ci95_high": float(nll_row["ci95_high"]),
                }
            )
    return pd.DataFrame(rows)


def condition_response_frame(summary: pd.DataFrame) -> pd.DataFrame:
    """Return the primary transition-metric frame with manuscript-facing names."""
    frame = condition_metric_response_frame(summary, metric=PRIMARY_METRIC)
    return frame.rename(
        columns={
            "quality_pass_2": "transition_similarity_pass_2",
            "quality_pass_1000": "transition_similarity_pass_1000",
            "quality_change": "structural_change",
            "quality_change_ci95_low": "structural_change_ci95_low",
            "quality_change_ci95_high": "structural_change_ci95_high",
        }
    )


def plot_condition_level_response(
    condition_summary: pd.DataFrame,
    output_dir: Path,
    *,
    output_stem: str,
    initial_column: str = "transition_similarity_pass_2",
    endpoint_column: str = "transition_similarity_pass_1000",
    change_column: str = "structural_change",
    change_ci_low_column: str = "structural_change_ci95_low",
    change_ci_high_column: str = "structural_change_ci95_high",
    initial_axis_label: str = r"Pass-2 transition similarity, $S^A_{\mathrm{TV},2}$",
    endpoint_axis_label: str = (
        r"Pass-1000 transition similarity, $S^A_{\mathrm{TV},1000}$"
    ),
    change_axis_label: str = (
        r"Structural change, $I_S=S^A_{\mathrm{TV},1000}-S^A_{\mathrm{TV},2}$"
    ),
    output_suffix: str = "condition_level_structural_response",
    seed_values: pd.DataFrame | None = None,
) -> list[Path]:
    """Plot condition-level endpoints and NLL improvement against metric change."""
    environment_markers = {13: "o", 21: "s", 20: "^", 22: "D"}
    is_seed_cloud = seed_values is not None
    figure_size = (7.85, 3.79) if is_seed_cloud else (7.15, 3.45)
    gridspec_kw = {"width_ratios": [0.82, 1.18]} if is_seed_cloud else None
    figure, axes = plt.subplots(1, 2, figsize=figure_size, gridspec_kw=gridspec_kw)
    bottom_margin = 0.27 if "\n" in initial_axis_label else 0.23
    if is_seed_cloud:
        bottom_margin -= 0.06
    figure.subplots_adjust(
        left=0.09 if is_seed_cloud else 0.12,
        right=0.99,
        bottom=bottom_margin,
        top=0.77 if is_seed_cloud else 0.82,
        wspace=0.24 if is_seed_cloud else 0.27,
    )

    for axis_index, axis in enumerate(axes):
        if seed_values is not None:
            for (series, level), cloud in seed_values.groupby(
                ["series", "level"], sort=False
            ):
                cloud_x = (
                    cloud["quality_pass_2"]
                    if axis_index == 0
                    else cloud["nll_improvement"]
                )
                cloud_y = (
                    cloud["quality_pass_1000"]
                    if axis_index == 0
                    else cloud["quality_change"]
                )
                axis.scatter(
                    cloud_x,
                    cloud_y,
                    marker=environment_markers[int(level)],
                    s=8,
                    color=SERIES_COLORS[str(series)],
                    alpha=0.14,
                    edgecolors="none",
                    zorder=2,
                )
        for row in condition_summary.itertuples(index=False):
            change = float(getattr(row, change_column))
            common_style = {
                "fmt": environment_markers[int(row.level)],
                "markersize": 5.3,
                "markerfacecolor": SERIES_COLORS[str(row.series)],
                "markeredgecolor": "white",
                "markeredgewidth": 0.55,
                "ecolor": SERIES_COLORS[str(row.series)],
                "elinewidth": 0.8,
                "capsize": 1.8,
                "alpha": 0.88,
                "zorder": 3,
            }
            if axis_index == 0:
                axis.errorbar(
                    float(getattr(row, initial_column)),
                    float(getattr(row, endpoint_column)),
                    **common_style,
                )
            else:
                y_error = np.asarray(
                    [
                        [change - float(getattr(row, change_ci_low_column))],
                        [float(getattr(row, change_ci_high_column)) - change],
                    ]
                )
                axis.errorbar(
                    float(row.nll_improvement),
                    change,
                    yerr=y_error,
                    **common_style,
                )
        if axis_index == 0:
            if seed_values is None:
                endpoint_values = condition_summary[
                    [initial_column, endpoint_column]
                ].to_numpy(dtype=float)
            else:
                endpoint_values = seed_values[
                    ["quality_pass_2", "quality_pass_1000"]
                ].to_numpy(dtype=float)
            lower = float(np.nanmin(endpoint_values))
            upper = float(np.nanmax(endpoint_values))
            padding = max((upper - lower) * 0.07, 0.01)
            limits = (lower - padding, upper + padding)
            axis.plot(limits, limits, color="#374151", linewidth=1.0, zorder=1)
            axis.set_xlim(limits)
            axis.set_ylim(limits)
            axis.set_aspect("equal", adjustable="box")
        else:
            axis.axhline(0.0, color="#374151", linewidth=1.0, zorder=1)
            axis.axvline(0.0, color="#6B7280", linewidth=0.8, linestyle="--", zorder=1)
        axis.grid(True, color="#D1D5DB", linewidth=0.45, alpha=0.55, zorder=0)
        axis.tick_params(labelsize=8)
        axis.yaxis.set_major_locator(MaxNLocator(nbins=5))
        axis.xaxis.set_major_locator(MaxNLocator(nbins=5))

    axes[0].set_xlabel(initial_axis_label, fontsize=8.5)
    axes[0].set_ylabel(endpoint_axis_label, fontsize=8.5)
    axes[1].set_xlabel(r"NLL improvement, $I_{\mathrm{NLL}}$", fontsize=8.5)
    axes[1].set_ylabel(change_axis_label, fontsize=8.5)
    axes[0].set_title(
        "a  Pass 2 versus pass 1000", loc="left", fontsize=9.5, fontweight="semibold"
    )
    axes[1].set_title("b  Optimisation outcomes", loc="left", fontsize=9.5, fontweight="semibold")

    initialisation_handles = [
        Line2D(
            [0], [0], marker="o", linestyle="none", markersize=5.5,
            markerfacecolor=SERIES_COLORS[series], markeredgecolor="white",
            label=COMPACT_SERIES_LABELS.get(series, series),
        )
        for series in SERIES_ORDER
    ]
    environment_handles = [
        Line2D(
            [0], [0], marker=environment_markers[level], linestyle="none", markersize=5,
            markerfacecolor="#6B7280", markeredgecolor="white",
            label=LEVEL_LABELS.get(level, f"L{level}").replace("x", r"$\times$"),
        )
        for level in DEFAULT_LEVEL_ORDER
    ]
    if is_seed_cloud:
        initialisation_handles = [
            initialisation_handles[index] for index in (0, 3, 1, 4, 2, 5)
        ]
        environment_handles = [environment_handles[index] for index in (0, 2, 1, 3)]
        initialisation_legend = figure.legend(
            handles=initialisation_handles,
            loc="upper center",
            bbox_to_anchor=(0.37, 0.975),
            ncol=3,
            frameon=False,
            fontsize=7.8,
            handletextpad=0.30,
            columnspacing=0.80,
            labelspacing=0.45,
            borderaxespad=0.0,
        )
        figure.add_artist(initialisation_legend)
        figure.legend(
            handles=environment_handles,
            loc="upper center",
            bbox_to_anchor=(0.88, 0.975),
            ncol=2,
            frameon=False,
            fontsize=7.8,
            handletextpad=0.30,
            columnspacing=0.85,
            labelspacing=0.45,
            borderaxespad=0.0,
        )
        figure.text(
            0.018, 0.932, "Methods", ha="left", va="center",
            fontsize=7.8, fontweight="semibold", color="#475569",
        )
        figure.text(
            0.655, 0.932, "Environments", ha="left", va="center",
            fontsize=7.8, fontweight="semibold", color="#475569",
        )
        figure.add_artist(
            Line2D(
                [0.64, 0.64], [0.89, 0.98], transform=figure.transFigure,
                color="#D7DEE7", linewidth=0.65,
            )
        )
    else:
        initialisation_legend = figure.legend(
            handles=initialisation_handles,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.015),
            ncol=6,
            frameon=False,
            fontsize=7.2,
            handletextpad=0.35,
            columnspacing=0.75,
        )
        figure.add_artist(initialisation_legend)
        figure.legend(
            handles=environment_handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.995),
            ncol=4,
            frameon=False,
            fontsize=7.2,
            handletextpad=0.35,
            columnspacing=1.0,
        )

    paths = [
        output_dir / f"{output_stem}_{output_suffix}.png",
        output_dir / f"{output_stem}_{output_suffix}.pdf",
    ]
    for path in paths:
        figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    return paths


def plot_combined_seed_cloud_responses(
    response_specs: list[dict[str, Any]],
    output_dir: Path,
    *,
    output_stem: str,
) -> list[Path]:
    """Combine purity, transition, and no-self response plots with one legend."""
    environment_markers = {13: "o", 21: "s", 20: "^", 22: "D"}
    figure, axes = plt.subplots(
        3,
        2,
        figsize=(7.85, 7.35),
        gridspec_kw={"width_ratios": [0.82, 1.18], "hspace": 0.30, "wspace": 0.24},
    )
    figure.subplots_adjust(left=0.09, right=0.99, bottom=0.065, top=0.86)

    for row_index, spec in enumerate(response_specs):
        condition_summary = spec["condition_summary"]
        seed_values = spec["seed_values"]
        left_axis, right_axis = axes[row_index]

        for axis_index, axis in enumerate((left_axis, right_axis)):
            for (series, level), cloud in seed_values.groupby(
                ["series", "level"], sort=False
            ):
                cloud_x = (
                    cloud["quality_pass_2"]
                    if axis_index == 0
                    else cloud["nll_improvement"]
                )
                cloud_y = (
                    cloud["quality_pass_1000"]
                    if axis_index == 0
                    else cloud["quality_change"]
                )
                axis.scatter(
                    cloud_x,
                    cloud_y,
                    marker=environment_markers[int(level)],
                    s=7,
                    color=SERIES_COLORS[str(series)],
                    alpha=0.13,
                    edgecolors="none",
                    zorder=2,
                )

            for row in condition_summary.itertuples(index=False):
                change = float(getattr(row, spec["change_column"]))
                common_style = {
                    "fmt": environment_markers[int(row.level)],
                    "markersize": 5.0,
                    "markerfacecolor": SERIES_COLORS[str(row.series)],
                    "markeredgecolor": "white",
                    "markeredgewidth": 0.5,
                    "ecolor": SERIES_COLORS[str(row.series)],
                    "elinewidth": 0.75,
                    "capsize": 1.6,
                    "alpha": 0.88,
                    "zorder": 3,
                }
                if axis_index == 0:
                    axis.errorbar(
                        float(getattr(row, spec["initial_column"])),
                        float(getattr(row, spec["endpoint_column"])),
                        **common_style,
                    )
                else:
                    y_error = np.asarray(
                        [
                            [change - float(getattr(row, spec["change_ci_low_column"]))],
                            [float(getattr(row, spec["change_ci_high_column"])) - change],
                        ]
                    )
                    axis.errorbar(
                        float(row.nll_improvement),
                        change,
                        yerr=y_error,
                        **common_style,
                    )

            if axis_index == 0:
                endpoint_values = seed_values[
                    ["quality_pass_2", "quality_pass_1000"]
                ].to_numpy(dtype=float)
                lower = float(np.nanmin(endpoint_values))
                upper = float(np.nanmax(endpoint_values))
                padding = max((upper - lower) * 0.07, 0.01)
                limits = (lower - padding, upper + padding)
                axis.plot(limits, limits, color="#374151", linewidth=0.95, zorder=1)
                axis.set_xlim(limits)
                axis.set_ylim(limits)
                axis.set_aspect("equal", adjustable="box")
            else:
                axis.axhline(0.0, color="#374151", linewidth=0.95, zorder=1)
                axis.axvline(
                    0.0, color="#6B7280", linewidth=0.75, linestyle="--", zorder=1
                )
            axis.grid(True, color="#D1D5DB", linewidth=0.42, alpha=0.52, zorder=0)
            axis.tick_params(labelsize=7.5)
            axis.yaxis.set_major_locator(MaxNLocator(nbins=4))
            axis.xaxis.set_major_locator(MaxNLocator(nbins=4))

        left_axis.set_ylabel(spec["endpoint_axis_label"], fontsize=8.0)
        right_axis.set_ylabel(spec["change_axis_label"], fontsize=8.0)
        if row_index == len(response_specs) - 1:
            left_axis.set_xlabel("Pass-2 metric value", fontsize=8.0)
            right_axis.set_xlabel(r"NLL improvement, $I_{\mathrm{NLL}}$", fontsize=8.0)
        panel_letter = chr(ord("a") + 2 * row_index)
        left_axis.set_title(
            f"{panel_letter}  {spec['panel_label']} endpoints",
            loc="left",
            fontsize=8.8,
            fontweight="semibold",
        )
        right_axis.set_title(
            f"{chr(ord(panel_letter) + 1)}  {spec['panel_label']} optimisation",
            loc="left",
            fontsize=8.8,
            fontweight="semibold",
        )

    initialisation_handles = [
        Line2D(
            [0], [0], marker="o", linestyle="none", markersize=5.2,
            markerfacecolor=SERIES_COLORS[series], markeredgecolor="white",
            label=COMPACT_SERIES_LABELS.get(series, series),
        )
        for series in SERIES_ORDER
    ]
    initialisation_handles = [
        initialisation_handles[index] for index in (0, 3, 1, 4, 2, 5)
    ]
    environment_handles = [
        Line2D(
            [0], [0], marker=environment_markers[level], linestyle="none", markersize=4.8,
            markerfacecolor="#6B7280", markeredgecolor="white",
            label=LEVEL_LABELS.get(level, f"L{level}").replace("x", r"$\times$"),
        )
        for level in DEFAULT_LEVEL_ORDER
    ]
    environment_handles = [environment_handles[index] for index in (0, 2, 1, 3)]
    method_legend = figure.legend(
        handles=initialisation_handles,
        loc="upper center",
        bbox_to_anchor=(0.37, 0.982),
        ncol=3,
        frameon=False,
        fontsize=7.8,
        handletextpad=0.30,
        columnspacing=0.80,
        labelspacing=0.45,
        borderaxespad=0.0,
    )
    figure.add_artist(method_legend)
    figure.legend(
        handles=environment_handles,
        loc="upper center",
        bbox_to_anchor=(0.88, 0.982),
        ncol=2,
        frameon=False,
        fontsize=7.8,
        handletextpad=0.30,
        columnspacing=0.85,
        labelspacing=0.45,
        borderaxespad=0.0,
    )
    figure.text(
        0.018, 0.947, "Methods", ha="left", va="center",
        fontsize=7.8, fontweight="semibold", color="#475569",
    )
    figure.text(
        0.655, 0.947, "Environments", ha="left", va="center",
        fontsize=7.8, fontweight="semibold", color="#475569",
    )
    figure.add_artist(
        Line2D(
            [0.64, 0.64], [0.91, 0.987], transform=figure.transFigure,
            color="#D7DEE7", linewidth=0.65,
        )
    )

    paths = [
        output_dir / f"{output_stem}_combined_condition_level_responses_with_seed_clouds.png",
        output_dir / f"{output_stem}_combined_condition_level_responses_with_seed_clouds.pdf",
    ]
    for path in paths:
        figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    caption_path = (
        output_dir
        / f"{output_stem}_combined_condition_level_responses_with_seed_clouds_caption.tex"
    )
    caption_path.write_text(
        (
            "\\caption{Metric response to continued CSCG refinement. Rows show latent purity, "
            "action-conditioned transition similarity, and the corresponding no-self-transition "
            "similarity. Left panels compare pass-2 and pass-1000 values; the identity line denotes "
            "no change. Right panels compare evaluation-NLL improvement with metric change; the "
            "horizontal and vertical reference lines denote zero change. Faint symbols show the 30 "
            "matched training seeds in each initialisation--environment condition, while larger "
            "symbols show condition means and vertical bars show 95\\% confidence intervals for "
            "paired mean metric change. Colour identifies initialisation and marker shape identifies "
            "environment.}\n"
        ),
        encoding="utf-8",
    )
    return [*paths, caption_path]


def plot_combined_absolute_phase_space_responses(
    response_specs: list[dict[str, Any]],
    output_dir: Path,
    *,
    output_stem: str,
    include_initialisation: bool = False,
    show_left_seed_points: bool = True,
    show_right_seed_points: bool = True,
    show_seed_trajectories: bool = True,
    delta_axes: bool = False,
) -> list[Path]:
    """Combine endpoint plots with absolute NLL--quality phase-space trajectories."""
    if delta_axes and not include_initialisation:
        raise ValueError("Delta-axis trajectories require the initialisation checkpoint.")
    environment_markers = {13: "o", 21: "s", 20: "^", 22: "D"}
    figure, axes = plt.subplots(
        3,
        2,
        figsize=(7.85, 7.35),
        gridspec_kw={"width_ratios": [0.82, 1.18], "hspace": 0.30, "wspace": 0.24},
    )
    figure.subplots_adjust(left=0.09, right=0.99, bottom=0.065, top=0.86)

    for row_index, spec in enumerate(response_specs):
        condition_summary = spec["condition_summary"]
        seed_values = spec["seed_values"]
        condition_trajectories = seed_values.groupby(
            ["series", "level"], sort=False
        ).mean(numeric_only=True)
        left_axis, right_axis = axes[row_index]

        for (series, level), cloud in seed_values.groupby(
            ["series", "level"], sort=False
        ):
            colour = SERIES_COLORS[str(series)]
            marker = environment_markers[int(level)]
            if show_left_seed_points:
                left_seed_x = cloud["quality_pass_2"]
                left_seed_y = cloud["quality_pass_1000"]
                if delta_axes:
                    left_seed_x = left_seed_x - cloud["quality_initialisation"]
                    left_seed_y = left_seed_y - cloud["quality_pass_2"]
                left_axis.scatter(
                    left_seed_x,
                    left_seed_y,
                    marker=marker,
                    s=7,
                    color=colour,
                    alpha=0.13,
                    edgecolors="none",
                    zorder=2,
                )
            # The seed paths are deliberately lines rather than arrows: 720 arrowheads
            # per panel obscure the condition means that carry the directional encoding.
            if show_seed_trajectories:
                for seed_row in cloud.itertuples(index=False):
                    trajectory_x = [
                        float(seed_row.nll_pass_2),
                        float(seed_row.nll_pass_1000),
                    ]
                    trajectory_y = [
                        float(seed_row.quality_pass_2),
                        float(seed_row.quality_pass_1000),
                    ]
                    if include_initialisation:
                        trajectory_x.insert(0, float(seed_row.nll_initialisation))
                        trajectory_y.insert(0, float(seed_row.quality_initialisation))
                    if delta_axes:
                        trajectory_x = [
                            0.0,
                            trajectory_x[0] - trajectory_x[1],
                            trajectory_x[1] - trajectory_x[2],
                        ]
                        trajectory_y = [
                            0.0,
                            trajectory_y[1] - trajectory_y[0],
                            trajectory_y[2] - trajectory_y[1],
                        ]
                    right_axis.plot(
                        trajectory_x,
                        trajectory_y,
                        color=colour,
                        linewidth=0.38,
                        alpha=0.065,
                        solid_capstyle="round",
                        zorder=1.5,
                    )
            if show_right_seed_points:
                right_axis.scatter(
                    cloud["nll_pass_2"],
                    cloud["quality_pass_2"],
                    marker=marker,
                    s=5,
                    facecolors="none",
                    edgecolors=colour,
                    linewidths=0.28,
                    alpha=0.09,
                    zorder=1.6,
                )
                if include_initialisation:
                    right_axis.scatter(
                        cloud["nll_initialisation"],
                        cloud["quality_initialisation"],
                        marker=marker,
                        s=5,
                        facecolors="none",
                        edgecolors=colour,
                        linewidths=0.28,
                        alpha=0.09,
                        zorder=1.6,
                    )
                right_axis.scatter(
                    cloud["nll_pass_1000"],
                    cloud["quality_pass_1000"],
                    marker=marker,
                    s=5,
                    color=colour,
                    alpha=0.09,
                    edgecolors="none",
                    zorder=1.6,
                )

        for row in condition_summary.itertuples(index=False):
            colour = SERIES_COLORS[str(row.series)]
            marker = environment_markers[int(row.level)]
            quality_pass_2 = float(getattr(row, spec["initial_column"]))
            quality_pass_1000 = float(getattr(row, spec["endpoint_column"]))
            trajectory_mean = condition_trajectories.loc[(row.series, row.level)]
            quality_initialisation = float(
                trajectory_mean["quality_initialisation"]
            ) if include_initialisation else math.nan
            left_mean_x = quality_pass_2
            left_mean_y = quality_pass_1000
            if delta_axes:
                left_mean_x -= quality_initialisation
                left_mean_y -= quality_pass_2

            left_axis.scatter(
                left_mean_x,
                left_mean_y,
                marker=marker,
                s=27,
                color=colour,
                edgecolors="white",
                linewidths=0.5,
                alpha=0.9,
                zorder=3,
            )

            nll_pass_2 = float(row.nll_pass_2)
            nll_pass_1000 = float(row.nll_pass_1000)
            if include_initialisation:
                nll_initialisation = float(trajectory_mean["nll_initialisation"])
                initialisation_x = nll_initialisation
                initialisation_y = quality_initialisation
                pass_2_x = nll_pass_2
                pass_2_y = quality_pass_2
                pass_1000_x = nll_pass_1000
                pass_1000_y = quality_pass_1000
                if delta_axes:
                    initialisation_x = initialisation_y = 0.0
                    pass_2_x = nll_initialisation - nll_pass_2
                    pass_2_y = quality_pass_2 - quality_initialisation
                    pass_1000_x = nll_pass_2 - nll_pass_1000
                    pass_1000_y = quality_pass_1000 - quality_pass_2
                right_axis.annotate(
                    "",
                    xy=(pass_2_x, pass_2_y),
                    xytext=(initialisation_x, initialisation_y),
                    arrowprops={
                        "arrowstyle": "-|>",
                        "color": colour,
                        "linewidth": 1.05,
                        "alpha": 0.82,
                        "mutation_scale": 7.0,
                        "shrinkA": 2.5,
                        "shrinkB": 2.5,
                    },
                    zorder=3,
                )
            right_axis.annotate(
                "",
                xy=(pass_1000_x if include_initialisation else nll_pass_1000,
                    pass_1000_y if include_initialisation else quality_pass_1000),
                xytext=(pass_2_x if include_initialisation else nll_pass_2,
                        pass_2_y if include_initialisation else quality_pass_2),
                arrowprops={
                    "arrowstyle": "-|>",
                    "color": colour,
                    "linewidth": 1.05,
                    "alpha": 0.82,
                    "mutation_scale": 7.0,
                    "shrinkA": 2.5,
                    "shrinkB": 2.5,
                },
                zorder=3,
            )
            right_axis.scatter(
                pass_2_x if include_initialisation else nll_pass_2,
                pass_2_y if include_initialisation else quality_pass_2,
                marker=marker,
                s=25,
                facecolors=colour if include_initialisation else "white",
                edgecolors="white" if include_initialisation else colour,
                linewidths=0.55 if include_initialisation else 0.9,
                alpha=0.48 if include_initialisation else 0.95,
                zorder=4,
            )
            if include_initialisation:
                right_axis.scatter(
                    initialisation_x,
                    initialisation_y,
                    marker=marker,
                    s=25,
                    facecolors="white",
                    edgecolors=colour,
                    linewidths=0.9,
                    alpha=0.95,
                    zorder=4,
                )
            right_axis.scatter(
                pass_1000_x if include_initialisation else nll_pass_1000,
                pass_1000_y if include_initialisation else quality_pass_1000,
                marker=marker,
                s=27,
                color=colour,
                edgecolors="white",
                linewidths=0.45,
                alpha=0.95,
                zorder=4,
            )

        if delta_axes:
            endpoint_values = np.column_stack(
                [
                    seed_values["quality_pass_2"] - seed_values["quality_initialisation"],
                    seed_values["quality_pass_1000"]
                    - seed_values["quality_pass_2"],
                ]
            )
        else:
            endpoint_values = seed_values[
                ["quality_pass_2", "quality_pass_1000"]
            ].to_numpy(dtype=float)
        lower = float(np.nanmin(endpoint_values))
        upper = float(np.nanmax(endpoint_values))
        if delta_axes:
            lower = min(lower, 0.0)
            upper = max(upper, 0.0)
        padding = max((upper - lower) * 0.07, 0.01)
        limits = (lower - padding, upper + padding)
        left_axis.plot(limits, limits, color="#374151", linewidth=0.95, zorder=1)
        left_axis.set_xlim(limits)
        left_axis.set_ylim(limits)
        left_axis.set_aspect("equal", adjustable="box")
        if delta_axes:
            left_axis.axhline(0.0, color="#94A3B8", linewidth=0.65, zorder=1)
            left_axis.axvline(0.0, color="#94A3B8", linewidth=0.65, zorder=1)
            right_axis.axhline(0.0, color="#94A3B8", linewidth=0.65, zorder=1)
            right_axis.axvline(0.0, color="#94A3B8", linewidth=0.65, zorder=1)

        for axis in (left_axis, right_axis):
            axis.grid(True, color="#D1D5DB", linewidth=0.42, alpha=0.52, zorder=0)
            axis.tick_params(labelsize=7.5)
            axis.yaxis.set_major_locator(MaxNLocator(nbins=4))
            axis.xaxis.set_major_locator(MaxNLocator(nbins=4))

        left_axis.set_ylabel(
            spec["left_delta_axis_label"]
            if delta_axes
            else spec["endpoint_axis_label"],
            fontsize=8.0,
        )
        right_axis.set_ylabel(
            spec["right_delta_axis_label"]
            if delta_axes
            else spec["absolute_axis_label"],
            fontsize=8.0,
        )
        if row_index == len(response_specs) - 1:
            left_axis.set_xlabel(
                r"Early change, initialisation$\to$pass 2"
                if delta_axes
                else "Pass-2 metric value",
                fontsize=8.0,
            )
            right_axis.set_xlabel(
                r"Checkpoint NLL improvement, $\Delta\mathrm{NLL}$"
                if delta_axes
                else "Evaluation NLL (bits/symbol; lower is better)",
                fontsize=8.0,
            )
        panel_letter = chr(ord("a") + 2 * row_index)
        left_axis.set_title(
            f"{panel_letter}  {spec['panel_label']} "
            f"{'changes' if delta_axes else 'endpoints'}",
            loc="left",
            fontsize=8.8,
            fontweight="semibold",
        )
        right_axis.set_title(
            f"{chr(ord(panel_letter) + 1)}  {spec['panel_label']} "
            f"{'change space' if delta_axes else 'phase space'}",
            loc="left",
            fontsize=8.8,
            fontweight="semibold",
        )
        if row_index == 0:
            phase_handles = []
            if include_initialisation:
                phase_handles.append(
                    Line2D(
                        [0], [0], marker="o", linestyle="none", markersize=4.8,
                        markerfacecolor="white", markeredgecolor="#64748B",
                        label="Initialisation",
                    )
                )
            phase_handles.extend(
                [
                    Line2D(
                        [0], [0], marker="o", linestyle="none", markersize=4.8,
                        markerfacecolor="#94A3B8" if include_initialisation else "white",
                        markeredgecolor="white" if include_initialisation else "#64748B",
                        alpha=0.65 if include_initialisation else 1.0,
                        label="Pass 2" + ("" if include_initialisation else " (start)"),
                    ),
                    Line2D(
                        [0], [0], marker="o", linestyle="none", markersize=4.8,
                        markerfacecolor="#64748B", markeredgecolor="white",
                        label="Pass 1000" + ("" if include_initialisation else " (end)"),
                    ),
                ]
            )
            right_axis.legend(
                handles=phase_handles,
                loc="lower right",
                frameon=False,
                fontsize=6.8,
                handletextpad=0.35,
                labelspacing=0.25,
            )

    initialisation_handles = [
        Line2D(
            [0], [0], marker="o", linestyle="none", markersize=5.2,
            markerfacecolor=SERIES_COLORS[series], markeredgecolor="white",
            label=COMPACT_SERIES_LABELS.get(series, series),
        )
        for series in SERIES_ORDER
    ]
    initialisation_handles = [
        initialisation_handles[index] for index in (0, 3, 1, 4, 2, 5)
    ]
    environment_handles = [
        Line2D(
            [0], [0], marker=environment_markers[level], linestyle="none", markersize=4.8,
            markerfacecolor="#6B7280", markeredgecolor="white",
            label=LEVEL_LABELS.get(level, f"L{level}").replace("x", r"$\times$"),
        )
        for level in DEFAULT_LEVEL_ORDER
    ]
    environment_handles = [environment_handles[index] for index in (0, 2, 1, 3)]
    method_legend = figure.legend(
        handles=initialisation_handles,
        loc="upper center",
        bbox_to_anchor=(0.37, 0.982),
        ncol=3,
        frameon=False,
        fontsize=7.8,
        handletextpad=0.30,
        columnspacing=0.80,
        labelspacing=0.45,
        borderaxespad=0.0,
    )
    figure.add_artist(method_legend)
    figure.legend(
        handles=environment_handles,
        loc="upper center",
        bbox_to_anchor=(0.88, 0.982),
        ncol=2,
        frameon=False,
        fontsize=7.8,
        handletextpad=0.30,
        columnspacing=0.85,
        labelspacing=0.45,
        borderaxespad=0.0,
    )
    figure.text(
        0.018, 0.947, "Methods", ha="left", va="center",
        fontsize=7.8, fontweight="semibold", color="#475569",
    )
    figure.text(
        0.655, 0.947, "Environments", ha="left", va="center",
        fontsize=7.8, fontweight="semibold", color="#475569",
    )
    figure.add_artist(
        Line2D(
            [0.64, 0.64], [0.91, 0.987], transform=figure.transFigure,
            color="#D7DEE7", linewidth=0.65,
        )
    )

    if delta_axes:
        output_suffix = (
            "combined_condition_level_responses_initialisation_pass_2_pass_1000_"
            "previous_checkpoint_delta_space"
        )
    elif include_initialisation:
        output_suffix = (
            "combined_condition_level_responses_initialisation_pass_2_pass_1000_phase_space"
        )
    else:
        output_suffix = "combined_condition_level_responses_absolute_phase_space"
    paths = [
        output_dir / f"{output_stem}_{output_suffix}.png",
        output_dir / f"{output_stem}_{output_suffix}.pdf",
    ]
    for path in paths:
        figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    caption_path = output_dir / f"{output_stem}_{output_suffix}_caption.tex"
    if delta_axes:
        trajectory_description = (
            "the initialisation is placed at $(0,0)$, the lightly filled pass-2 symbol is "
            "$(\\mathrm{NLL}_{\\mathrm{init}}-\\mathrm{NLL}_2, M_2-M_{\\mathrm{init}})$, "
            "and the filled pass-1000 symbol is "
            "$(\\mathrm{NLL}_2-\\mathrm{NLL}_{1000}, M_{1000}-M_2)$; arrows preserve the "
            "checkpoint order"
        )
    elif include_initialisation:
        trajectory_description = (
            "an open symbol marks the condition mean immediately after initialisation, a lightly "
            "filled symbol marks pass 2, and a filled symbol marks pass 1000; arrows join "
            "$(\\mathrm{NLL}_{\\mathrm{init}},M_{\\mathrm{init}})$ to "
            "$(\\mathrm{NLL}_2,M_2)$ and then to "
            "$(\\mathrm{NLL}_{1000},M_{1000})$"
        )
    else:
        trajectory_description = (
            "an open symbol marks the condition mean at pass 2 and a filled symbol marks pass "
            "1000; the arrow joins $(\\mathrm{NLL}_2,M_2)$ to "
            "$(\\mathrm{NLL}_{1000},M_{1000})$"
        )
    seed_descriptions = []
    if show_left_seed_points:
        seed_descriptions.append("faint symbols in the left panels show seed-level endpoints")
    if show_seed_trajectories:
        seed_descriptions.append("faint lines in the right panels show matched seed trajectories")
    if show_right_seed_points:
        seed_descriptions.append("faint symbols in the right panels show seed checkpoints")
    seed_description = (
        "; ".join(seed_descriptions).capitalize() + ". " if seed_descriptions else ""
    )
    if delta_axes:
        caption_opening = (
            "\\caption{Checkpoint-relative changes during CSCG refinement. Left panels compare "
            "metric change from initialisation to pass 2 (horizontal) with metric change from pass "
            "2 to pass 1000 (vertical). In each right panel, "
        )
        displacement_description = (
            "Positive horizontal values denote improved evaluation NLL relative to the previous "
            "checkpoint, and positive vertical values denote structural-metric improvement. "
        )
    else:
        caption_opening = (
            "\\caption{Absolute phase-space trajectories during continued CSCG refinement. "
            "Left panels retain the pass-2 versus pass-1000 endpoint comparison. In each right "
            "panel, "
        )
        displacement_description = (
            "Thus, horizontal displacement encodes change in evaluation NLL and vertical "
            "displacement encodes change in the corresponding structural metric. "
        )
    caption_path.write_text(
        (
            f"{caption_opening}{trajectory_description}. {displacement_description}"
            f"{seed_description}Colour identifies initialisation method and "
            "marker shape identifies environment. Lower NLL and higher "
            "structural-metric values are preferable.}\n"
        ),
        encoding="utf-8",
    )
    return [*paths, caption_path]


def plot_metric_likelihood_spaces(
    response_specs: list[dict[str, Any]],
    output_dir: Path,
    *,
    output_stem: str,
    separate_environments: bool = False,
) -> list[Path]:
    """Plot only the absolute metric/likelihood trajectories for condition means."""
    environment_markers = {13: "o", 21: "s", 20: "^", 22: "D"}
    figure_height = (2.2 if separate_environments else 1.8) * len(response_specs) + 1.65
    figure, axes = plt.subplots(
        len(response_specs),
        len(DEFAULT_LEVEL_ORDER) if separate_environments else 1,
        figsize=(12.8 if separate_environments else 6.45, figure_height),
        sharex="col",
        sharey="row" if separate_environments else False,
        squeeze=False,
    )
    if separate_environments:
        figure.subplots_adjust(
            left=0.08, right=0.99, bottom=0.08, top=0.85,
            wspace=0.12, hspace=0.28,
        )
    else:
        figure.subplots_adjust(
            left=0.135,
            right=0.99,
            bottom=0.53 / figure_height,
            top=1.0 - 1.06 / figure_height,
            hspace=0.30,
        )

    for row_index, spec in enumerate(response_specs):
        trajectories = spec["seed_values"].groupby(
            ["series", "level"], sort=False
        ).mean(numeric_only=True)
        for (series, level), trajectory in trajectories.iterrows():
            column_index = (
                DEFAULT_LEVEL_ORDER.index(int(level)) if separate_environments else 0
            )
            axis = axes[row_index, column_index]
            colour = SERIES_COLORS[str(series)]
            marker = "o" if separate_environments else environment_markers[int(level)]
            nll_initialisation = float(trajectory["nll_initialisation"])
            nll_pass_1 = float(trajectory.get("nll_pass_1", np.nan))
            nll_pass_2 = float(trajectory["nll_pass_2"])
            nll_pass_1000 = float(trajectory["nll_pass_1000"])
            quality_initialisation = float(trajectory["quality_initialisation"])
            quality_pass_1 = float(trajectory.get("quality_pass_1", np.nan))
            quality_pass_2 = float(trajectory["quality_pass_2"])
            quality_pass_1000 = float(trajectory["quality_pass_1000"])
            pass_1_series = spec.get("pass_1_series")
            show_pass_1 = (
                (pass_1_series is None or str(series) in pass_1_series)
                and np.isfinite(nll_pass_1)
                and np.isfinite(quality_pass_1)
            )

            checkpoints = [(nll_initialisation, quality_initialisation)]
            if show_pass_1:
                checkpoints.append((nll_pass_1, quality_pass_1))
            checkpoints.extend(
                [(nll_pass_2, quality_pass_2), (nll_pass_1000, quality_pass_1000)]
            )
            for start, end in zip(checkpoints, checkpoints[1:]):
                axis.annotate(
                    "",
                    xy=end,
                    xytext=start,
                    arrowprops={
                        "arrowstyle": "-|>",
                        "color": colour,
                        "linewidth": 1.05,
                        "alpha": 0.82,
                        "mutation_scale": 7.0,
                        "shrinkA": 2.5,
                        "shrinkB": 2.5,
                    },
                    zorder=3,
                )
            axis.scatter(
                nll_initialisation,
                quality_initialisation,
                marker=marker,
                s=25,
                facecolors="white",
                edgecolors=colour,
                linewidths=0.9,
                alpha=0.95,
                zorder=4,
            )
            if show_pass_1:
                axis.scatter(
                    nll_pass_1,
                    quality_pass_1,
                    marker=marker,
                    s=25,
                    facecolors=colour,
                    edgecolors="white",
                    linewidths=0.55,
                    alpha=0.28,
                    zorder=4,
                )
            axis.scatter(
                nll_pass_2,
                quality_pass_2,
                marker=marker,
                s=25,
                facecolors=colour,
                edgecolors="white",
                linewidths=0.55,
                alpha=0.48,
                zorder=4,
            )
            axis.scatter(
                nll_pass_1000,
                quality_pass_1000,
                marker=marker,
                s=27,
                color=colour,
                edgecolors="white",
                linewidths=0.45,
                alpha=0.95,
                zorder=4,
            )

        for column_index, axis in enumerate(axes[row_index]):
            axis.grid(True, color="#D1D5DB", linewidth=0.42, alpha=0.52, zorder=0)
            axis.tick_params(labelsize=7.5)
            axis.yaxis.set_major_locator(MaxNLocator(nbins=4))
            axis.xaxis.set_major_locator(MaxNLocator(nbins=5))
            if column_index == 0:
                axis.set_ylabel(spec["absolute_axis_label"], fontsize=8.0)
            if separate_environments:
                if row_index == 0:
                    axis.set_title(
                        LEVEL_LABELS[DEFAULT_LEVEL_ORDER[column_index]].replace("x", r"$\times$"),
                        fontsize=9.2, fontweight="semibold",
                    )
            else:
                axis.set_title(
                    f"{chr(ord('a') + row_index)}  {spec['panel_label']}/likelihood space",
                    loc="left", fontsize=8.8, fontweight="semibold",
                )
        if row_index == 0 and not separate_environments:
            axes[0, 0].legend(
                handles=[
                    Line2D(
                        [0], [0], marker="o", linestyle="none", markersize=4.8,
                        markerfacecolor="white", markeredgecolor="#64748B",
                        label="Initialisation",
                    ),
                    Line2D(
                        [0], [0], marker="o", linestyle="none", markersize=4.8,
                        markerfacecolor="#94A3B8", markeredgecolor="white", alpha=0.38,
                        label="Pass 1",
                    ),
                    Line2D(
                        [0], [0], marker="o", linestyle="none", markersize=4.8,
                        markerfacecolor="#94A3B8", markeredgecolor="white", alpha=0.65,
                        label="Pass 2",
                    ),
                    Line2D(
                        [0], [0], marker="o", linestyle="none", markersize=4.8,
                        markerfacecolor="#64748B", markeredgecolor="white",
                        label="Pass 1000",
                    ),
                ],
                loc="lower right",
                frameon=False,
                fontsize=6.8,
                handletextpad=0.35,
                labelspacing=0.25,
            )

    if separate_environments:
        figure.supxlabel("Evaluation NLL (bits/symbol; lower is better)", fontsize=9.0, y=0.02)
    else:
        axes[-1, 0].set_xlabel(
            "Evaluation NLL (bits/symbol; lower is better)", fontsize=8.0
        )
    method_handles = [
        Line2D(
            [0], [0], marker="o", linestyle="none", markersize=5.2,
            markerfacecolor=SERIES_COLORS[series], markeredgecolor="white",
            label=COMPACT_SERIES_LABELS.get(series, series),
        )
        for series in SERIES_ORDER
    ]
    if not separate_environments:
        method_handles = [method_handles[index] for index in (0, 3, 1, 4, 2, 5)]
    method_legend = figure.legend(
        handles=method_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985) if separate_environments else (0.37, 0.978),
        ncol=6 if separate_environments else 3,
        frameon=False,
        fontsize=7.8,
        handletextpad=0.30,
        columnspacing=0.80,
        labelspacing=0.45,
        borderaxespad=0.0,
    )
    figure.add_artist(method_legend)
    if separate_environments:
        figure.legend(
            handles=[
                Line2D([0], [0], marker="o", linestyle="none", markersize=5,
                       markerfacecolor=fill, markeredgecolor=edge, alpha=alpha,
                       label=label)
                for label, fill, edge, alpha in (
                    ("Initialisation", "white", "#64748B", 1.0),
                    ("Pass 1", "#94A3B8", "white", 0.38),
                    ("Pass 2", "#94A3B8", "white", 0.65),
                    ("Pass 1000", "#64748B", "white", 1.0),
                )
            ],
            loc="upper center", bbox_to_anchor=(0.5, 0.94), ncol=4,
            frameon=False, fontsize=7.8,
        )
    else:
        environment_handles = [
            Line2D(
                [0], [0], marker=environment_markers[level], linestyle="none", markersize=4.8,
                markerfacecolor="#6B7280", markeredgecolor="white",
                label=LEVEL_LABELS.get(level, f"L{level}").replace("x", r"$\times$"),
            )
            for level in DEFAULT_LEVEL_ORDER
        ]
        environment_handles = [environment_handles[index] for index in (0, 2, 1, 3)]
        figure.legend(
            handles=environment_handles,
            loc="upper center",
            bbox_to_anchor=(0.88, 0.978),
            ncol=2,
            frameon=False,
            fontsize=7.8,
            handletextpad=0.30,
            columnspacing=0.85,
            labelspacing=0.45,
            borderaxespad=0.0,
        )
        figure.text(
            0.018, 0.942, "Methods", ha="left", va="center",
            fontsize=7.8, fontweight="semibold", color="#475569",
        )
        figure.text(
            0.655, 0.942, "Environments", ha="left", va="center",
            fontsize=7.8, fontweight="semibold", color="#475569",
        )
        figure.add_artist(
            Line2D(
                [0.64, 0.64], [0.905, 0.983], transform=figure.transFigure,
                color="#D7DEE7", linewidth=0.65,
            )
        )

    output_suffix = (
        "condition_level_initialisation_pass_2_pass_1000_metric_likelihood_spaces"
    )
    if separate_environments:
        output_suffix += "_by_environment_prototype"
    paths = [
        output_dir / f"{output_stem}_{output_suffix}.png",
        output_dir / f"{output_stem}_{output_suffix}.pdf",
    ]
    for path in paths:
        figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    caption_path = output_dir / f"{output_stem}_{output_suffix}_caption.tex"
    environment_description = (
        "Column titles identify environments. "
        if separate_environments else
        "Marker shape identifies environment. "
    )
    caption_path.write_text(
        (
            "\\caption{Metric/likelihood spaces during continued CSCG refinement. Rows show "
            "latent purity, latent concentration, action-conditioned transition similarity, and "
            "no-self-transition similarity against evaluation NLL. For every "
            "initialisation--environment condition, "
            "an open symbol marks the mean immediately after initialisation, progressively filled "
            "symbols mark passes 1 and 2, and a filled symbol marks pass 1000; arrows indicate "
            "checkpoint order. Pass 1 is shown for Standard CSCG and the Oracle-C variants; it is "
            "omitted for MERCURY and TMGWR because their source one-pass representations have no "
            "comparable NLL. Colour identifies initialisation method. "
            + environment_description
            + "No seed-level points or trajectories are shown. Lower NLL and higher "
            "structural-metric values are preferable.}\n"
        ),
        encoding="utf-8",
    )
    return [*paths, caption_path]


def plot_mercury_initialised_cscg_one_pass_vs_initialisation(
    rows: pd.DataFrame,
    output_dir: Path,
    *,
    output_stem: str,
    one_pass_step: int = 2_000,
) -> list[Path]:
    """Compare MERCURY-initialised CSCG means before and after one CSCG pass."""
    series = "mercury_cscg_hybrid"
    metric_specs = (
        ("latent_precision", "Purity", r"Latent purity, $P$"),
        (PRIMARY_METRIC, "Transition similarity", r"Transition similarity, $S^A_{\mathrm{TV}}$"),
        (
            SECONDARY_METRIC,
            "No-self similarity",
            r"No-self similarity, $S^{A,\neg\mathrm{self}}_{\mathrm{TV}}$",
        ),
    )
    keys = ["run_id", "series", "level", "seed"]
    metrics = [NLL_METRIC, *(metric for metric, _, _ in metric_specs)]
    selected = rows.loc[
        (rows["series"] == series)
        & rows["step"].isin([INITIALISATION_STEP, one_pass_step])
        & rows["metric"].isin(metrics),
        keys + ["metric", "step", "value"],
    ].copy()
    if selected.duplicated(keys + ["metric", "step"]).any():
        raise ValueError("Duplicate MERCURY one-pass checkpoint records were found.")
    aligned = selected.pivot(index=keys, columns=["metric", "step"], values="value")
    required = [
        (metric, step)
        for metric in metrics
        for step in (INITIALISATION_STEP, one_pass_step)
    ]
    missing_columns = [column for column in required if column not in aligned.columns]
    if missing_columns:
        raise ValueError(f"Missing MERCURY one-pass checkpoint columns: {missing_columns}")
    incomplete = aligned.loc[:, required].isna().any(axis=1)
    if incomplete.any():
        raise ValueError(
            f"Missing checkpoint values for {int(incomplete.sum())} MERCURY seed trajectories."
        )
    aligned = aligned.loc[:, required]
    expected_conditions = len(DEFAULT_LEVEL_ORDER) * len(EXPECTED_SEEDS)
    if len(aligned) != expected_conditions:
        raise ValueError(
            f"Expected {expected_conditions} MERCURY environment/seed trajectories, "
            f"found {len(aligned)}."
        )

    environment_markers = {13: "o", 21: "s", 20: "^", 22: "D"}
    colour = SERIES_COLORS[series]
    figure, axes = plt.subplots(
        len(metric_specs),
        1,
        figsize=(6.15, 6.65),
        sharex=True,
        gridspec_kw={"hspace": 0.30},
    )
    axes = np.atleast_1d(axes)
    figure.subplots_adjust(left=0.145, right=0.99, bottom=0.08, top=0.855)

    for row_index, (axis, (metric, label, axis_label)) in enumerate(
        zip(axes, metric_specs)
    ):
        for level in DEFAULT_LEVEL_ORDER:
            condition = aligned.xs((series, level), level=("series", "level"))
            nll_initialisation = float(condition[(NLL_METRIC, INITIALISATION_STEP)].mean())
            nll_one_pass = float(condition[(NLL_METRIC, one_pass_step)].mean())
            quality_initialisation = float(condition[(metric, INITIALISATION_STEP)].mean())
            quality_one_pass = float(condition[(metric, one_pass_step)].mean())
            marker = environment_markers[level]
            axis.annotate(
                "",
                xy=(nll_one_pass, quality_one_pass),
                xytext=(nll_initialisation, quality_initialisation),
                arrowprops={
                    "arrowstyle": "-|>",
                    "color": colour,
                    "linewidth": 1.35,
                    "alpha": 0.9,
                    "mutation_scale": 8.0,
                    "shrinkA": 3.0,
                    "shrinkB": 3.0,
                },
                zorder=3,
            )
            axis.scatter(
                nll_initialisation,
                quality_initialisation,
                marker=marker,
                s=39,
                facecolors="white",
                edgecolors=colour,
                linewidths=1.1,
                zorder=4,
            )
            axis.scatter(
                nll_one_pass,
                quality_one_pass,
                marker=marker,
                s=41,
                color=colour,
                edgecolors="white",
                linewidths=0.55,
                zorder=4,
            )
        axis.grid(True, color="#D1D5DB", linewidth=0.45, alpha=0.55, zorder=0)
        axis.tick_params(labelsize=8.0)
        axis.yaxis.set_major_locator(MaxNLocator(nbins=4))
        axis.xaxis.set_major_locator(MaxNLocator(nbins=5))
        axis.set_ylabel(axis_label, fontsize=8.4)
        axis.set_title(
            f"{chr(ord('a') + row_index)}  {label}/likelihood space",
            loc="left",
            fontsize=9.2,
            fontweight="semibold",
        )
        if row_index == 0:
            axis.legend(
                handles=[
                    Line2D(
                        [0], [0], marker="o", linestyle="none", markersize=5.2,
                        markerfacecolor="white", markeredgecolor=colour,
                        label="Initialisation",
                    ),
                    Line2D(
                        [0], [0], marker="o", linestyle="none", markersize=5.2,
                        markerfacecolor=colour, markeredgecolor="white",
                        label="One pass",
                    ),
                ],
                loc="lower right",
                frameon=False,
                fontsize=7.2,
                handletextpad=0.35,
                labelspacing=0.25,
            )

    axes[-1].set_xlabel(
        "Evaluation NLL (bits/symbol; lower is better)", fontsize=8.4
    )
    figure.suptitle(
        "MERCURY-initialised CSCG: one CSCG pass versus initialisation",
        x=0.5,
        y=0.986,
        fontsize=10.5,
        fontweight="semibold",
    )
    environment_handles = [
        Line2D(
            [0], [0], marker=environment_markers[level], linestyle="none", markersize=5.2,
            markerfacecolor=colour, markeredgecolor="white",
            label=LEVEL_LABELS.get(level, f"L{level}").replace("x", r"$\times$"),
        )
        for level in DEFAULT_LEVEL_ORDER
    ]
    figure.legend(
        handles=environment_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.947),
        ncol=4,
        frameon=False,
        fontsize=7.8,
        handletextpad=0.35,
        columnspacing=1.0,
        borderaxespad=0.0,
    )

    output_suffix = (
        "mercury_initialised_cscg_one_pass_vs_initialisation_metric_likelihood_spaces"
    )
    paths = [
        output_dir / f"{output_stem}_{output_suffix}.png",
        output_dir / f"{output_stem}_{output_suffix}.pdf",
    ]
    for path in paths:
        figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    caption_path = output_dir / f"{output_stem}_{output_suffix}_caption.tex"
    caption_path.write_text(
        (
            "\\caption{MERCURY-initialised CSCG after one CSCG pass compared with its frozen "
            "initialisation in metric/likelihood space. Rows show latent purity, transition "
            "similarity, and no-self-transition similarity against evaluation NLL. Open symbols "
            "show step-0 condition means and filled symbols show one-CSCG-pass (2,000-step) means; "
            "arrows indicate optimisation direction. Marker shape identifies environment. Each "
            "mean contains 30 matched training seeds; seed-level points are not shown. Lower NLL "
            "and higher structural-metric values are preferable.}\n"
        ),
        encoding="utf-8",
    )
    return [*paths, caption_path]


def plot_mercury_source_vs_cscg_initialisation(
    rows: pd.DataFrame,
    study_root: Path,
    protocol_id: str,
    output_dir: Path,
    *,
    output_stem: str,
    source_step: int = 2_000,
) -> list[Path]:
    """Compare one-pass MERCURY source structures with their frozen CSCG initialisations."""
    series = "mercury_cscg_hybrid"
    metric_specs = (
        ("latent_precision", "Purity", r"Latent purity, $P$"),
        (PRIMARY_METRIC, "Transition similarity", r"Transition similarity, $S^A_{\mathrm{TV}}$"),
        (
            SECONDARY_METRIC,
            "No-self similarity",
            r"No-self similarity, $S^{A,\neg\mathrm{self}}_{\mathrm{TV}}$",
        ),
    )
    metric_names = [metric for metric, _, _ in metric_specs]
    initial = rows.loc[
        (rows["series"] == series)
        & (rows["step"] == INITIALISATION_STEP)
        & rows["metric"].isin(metric_names),
        ["run_id", "level", "seed", "metric", "value"],
    ].rename(columns={"value": "cscg_initialisation"})

    source_records: list[dict[str, Any]] = []
    for run_id, level, seed in initial[["run_id", "level", "seed"]].drop_duplicates().itertuples(
        index=False, name=None
    ):
        source_path = (
            study_root
            / str(run_id)
            / "evaluations"
            / protocol_id
            / "snapshots"
            / "mercury_cscg_hybrid_mercury"
            / f"step={source_step:08d}.json"
        )
        if not source_path.exists():
            raise ValueError(f"Missing one-pass MERCURY source snapshot: {source_path}")
        payload = json.loads(source_path.read_text(encoding="utf-8"))
        source_metrics = _metrics_from_record(payload)
        for metric in metric_names:
            if metric not in source_metrics:
                raise ValueError(f"Missing {metric} in one-pass MERCURY source snapshot: {source_path}")
            source_records.append(
                {
                    "run_id": str(run_id),
                    "level": int(level),
                    "seed": int(seed),
                    "metric": metric,
                    "mercury_one_pass": float(source_metrics[metric]),
                }
            )
    source = pd.DataFrame(source_records)
    paired = initial.merge(
        source,
        on=["run_id", "level", "seed", "metric"],
        how="inner",
        validate="one_to_one",
    )
    expected_rows = len(DEFAULT_LEVEL_ORDER) * len(EXPECTED_SEEDS) * len(metric_specs)
    if len(paired) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} paired MERCURY/source-initialisation metric rows, "
            f"found {len(paired)}."
        )

    environment_markers = {13: "o", 21: "s", 20: "^", 22: "D"}
    colour = SERIES_COLORS[series]
    figure, axes = plt.subplots(1, len(metric_specs), figsize=(7.85, 3.15))
    axes = np.atleast_1d(axes)
    figure.subplots_adjust(left=0.075, right=0.99, bottom=0.22, top=0.73, wspace=0.30)
    for panel_index, (axis, (metric, label, axis_label)) in enumerate(
        zip(axes, metric_specs)
    ):
        panel = paired.loc[paired["metric"] == metric]
        axis.scatter(
            panel["mercury_one_pass"],
            panel["cscg_initialisation"],
            s=8,
            color=colour,
            alpha=0.13,
            edgecolors="none",
            zorder=2,
        )
        for level in DEFAULT_LEVEL_ORDER:
            condition = panel.loc[panel["level"] == level]
            axis.scatter(
                float(condition["mercury_one_pass"].mean()),
                float(condition["cscg_initialisation"].mean()),
                marker=environment_markers[level],
                s=38,
                color=colour,
                edgecolors="white",
                linewidths=0.6,
                zorder=3,
            )
        values = panel[["mercury_one_pass", "cscg_initialisation"]].to_numpy(float)
        lower = float(np.nanmin(values))
        upper = float(np.nanmax(values))
        padding = max(0.07 * (upper - lower), 0.008)
        limits = (lower - padding, upper + padding)
        axis.plot(limits, limits, color="#374151", linewidth=0.95, zorder=1)
        axis.set_xlim(limits)
        axis.set_ylim(limits)
        axis.set_aspect("equal", adjustable="box")
        axis.grid(True, color="#D1D5DB", linewidth=0.42, alpha=0.52, zorder=0)
        axis.tick_params(labelsize=7.5)
        axis.xaxis.set_major_locator(MaxNLocator(nbins=4))
        axis.yaxis.set_major_locator(MaxNLocator(nbins=4))
        axis.set_xlabel("One-pass MERCURY metric value", fontsize=7.8)
        axis.set_ylabel("Frozen CSCG initialisation metric value", fontsize=7.8)
        axis.set_title(
            f"{chr(ord('a') + panel_index)}  {label}",
            loc="left",
            fontsize=9.0,
            fontweight="semibold",
        )

    figure.suptitle(
        "One-pass MERCURY source versus its CSCG initialisation",
        y=0.972,
        fontsize=10.2,
        fontweight="semibold",
    )
    environment_handles = [
        Line2D(
            [0], [0], marker=environment_markers[level], linestyle="none", markersize=5.2,
            markerfacecolor=colour, markeredgecolor="white",
            label=LEVEL_LABELS.get(level, f"L{level}").replace("x", r"$\times$"),
        )
        for level in DEFAULT_LEVEL_ORDER
    ]
    figure.legend(
        handles=environment_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.88),
        ncol=4,
        frameon=False,
        fontsize=7.8,
        handletextpad=0.35,
        columnspacing=1.0,
    )

    output_suffix = "mercury_one_pass_source_vs_cscg_initialisation_structural_endpoints"
    paths = [
        output_dir / f"{output_stem}_{output_suffix}.png",
        output_dir / f"{output_stem}_{output_suffix}.pdf",
    ]
    for path in paths:
        figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    caption_path = output_dir / f"{output_stem}_{output_suffix}_caption.tex"
    caption_path.write_text(
        (
            "\\caption{Structural metrics for the one-pass MERCURY source representation "
            "compared with the corresponding frozen CSCG initialisation. The identity line "
            "denotes exact preservation during initialisation. Faint points show 30 matched seeds "
            "per environment; larger symbols show environment means. Marker shape identifies "
            "environment. Evaluation NLL is omitted because it is not defined for the one-pass "
            "MERCURY source representation.}\n"
        ),
        encoding="utf-8",
    )
    return [*paths, caption_path]


def _latex_main_before_after_table(
    summary: pd.DataFrame, correlations: pd.DataFrame
) -> list[str]:
    rows = publication_table_frame(summary, correlations, metric=PRIMARY_METRIC)
    output = [
        r"\begin{table*}[!t]",
        r"\centering",
        (
            r"\caption{Effect of continued CSCG optimisation on transition similarity and "
            r"evaluation NLL.}"
        ),
        r"\label{tab:pass_2_to_1000_main}",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{7pt}",
        r"\renewcommand{\arraystretch}{1.05}",
        r"\begin{tabular}{@{}llcc@{}}",
        r"\toprule",
        (
            r"Initialisation & Env. & $S^A_{\mathrm{TV}}\ (\uparrow)$ & "
            r"Evaluation NLL $(\downarrow)$ \\"
        ),
        r"\midrule",
    ]
    for series_index, series in enumerate(SERIES_ORDER):
        method_rows = rows.loc[rows["series"] == series]
        for level_index, level in enumerate(DEFAULT_LEVEL_ORDER):
            row = method_rows.loc[method_rows["level"] == level].iloc[0]
            cells = [
                str(row["method"]) if level_index == 0 else "",
                LATEX_LEVEL_LABELS[int(level)],
                _latex_endpoint_only(
                    float(row["metric_pass_2"]),
                    float(row["metric_pass_1000"]),
                ),
                _latex_endpoint_only(
                    float(row["nll_pass_2"]),
                    float(row["nll_pass_1000"]),
                ),
            ]
            output.append(" & ".join(cells) + r" \\")
        if series_index < len(SERIES_ORDER) - 1:
            output.append(r"\addlinespace[2pt]")
    output.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\vspace{2pt}",
            (
                r"\parbox{\textwidth}{\footnotesize\textit{Note:} Entries are means across 30 "
                r"matched training seeds and report pass 2 $\rightarrow$ pass 1000. Higher "
                r"$S^A_{\mathrm{TV}}$ and lower evaluation NLL are desirable. Statistical analysis "
                r"of the paired changes is reported in Table~\ref{tab:pass_2_to_1000_changes}.}"
            ),
            r"\end{table*}",
            "",
        ]
    )
    return output


def _latex_paired_change_and_association_table(
    summary: pd.DataFrame,
    correlations: pd.DataFrame,
) -> list[str]:
    rows = publication_table_frame(summary, correlations, metric=PRIMARY_METRIC)
    output = [
        r"\begin{table*}[!t]",
        r"\centering",
        (
            r"\caption{Paired improvements in transition similarity and evaluation NLL, with "
            r"within-condition associations across seeds.}"
        ),
        r"\label{tab:pass_2_to_1000_changes}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{5pt}",
        r"\renewcommand{\arraystretch}{1.00}",
        r"\textit{Panel A: Paired mean improvements}\par\vspace{2pt}",
        r"\begin{tabular}{@{}llcc@{}}",
        r"\toprule",
        (
            r"Initialisation & Env. & $I_S$ [95\% CI] & "
            r"$I_{\mathrm{NLL}}$ [95\% CI] \\"
        ),
        r"\midrule",
    ]
    for series_index, series in enumerate(SERIES_ORDER):
        method_rows = rows.loc[rows["series"] == series]
        for level_index, level in enumerate(DEFAULT_LEVEL_ORDER):
            row = method_rows.loc[method_rows["level"] == level].iloc[0]
            output.append(
                " & ".join(
                    [
                        str(row["method"]) if level_index == 0 else "",
                        LATEX_LEVEL_LABELS[int(level)],
                        _latex_change_cell(
                            float(row["metric_mean_change"]),
                            float(row["metric_ci95_low"]),
                            float(row["metric_ci95_high"]),
                        ),
                        _latex_change_cell(
                            float(row["nll_mean_improvement"]),
                            float(row["nll_ci95_low"]),
                            float(row["nll_ci95_high"]),
                        ),
                    ]
                )
                + r" \\"
            )
        if series_index < len(SERIES_ORDER) - 1:
            output.append(r"\addlinespace[2pt]")
    output.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\par\vspace{6pt}",
            *_latex_predictor_tabular(correlations),
            r"\vspace{2pt}",
            (
                r"\parbox{\textwidth}{\scriptsize\textit{Note:} "
                r"$I_S=S^A_{\mathrm{TV},1000}-S^A_{\mathrm{TV},2}$ and "
                r"$I_{\mathrm{NLL}}=\mathrm{NLL}_2-\mathrm{NLL}_{1000}$; positive values "
                r"indicate improvement. Confidence intervals use 30 paired seeds. Panel B reports "
                r"Spearman $\rho$ between per-seed $I_{\mathrm{NLL}}$ and $I_S$ within each fixed "
                r"initialisation--environment condition; initialisations are not pooled. The "
                r"correlations are supporting descriptive evidence.}"
            ),
            r"\end{table*}",
            "",
        ]
    )
    return output


def _latex_predictor_tabular(correlations: pd.DataFrame) -> list[str]:
    rows = correlations.loc[correlations["metric"] == PRIMARY_METRIC].set_index(
        ["series", "level"]
    )
    output = [
        r"\textit{Panel B: Spearman $\rho(I_{\mathrm{NLL}},I_S)$ within each condition}\par\vspace{2pt}",
        r"\begin{tabular}{@{}lrrrr@{}}",
        r"\toprule",
        r"Initialisation & $3\times3$ & $1\times4$ & $1\times5$ & $3\times4$ \\",
        r"\midrule",
    ]
    for series in SERIES_ORDER:
        values = [
            _format_rho(float(rows.loc[(series, level), "rho_s"]))
            for level in DEFAULT_LEVEL_ORDER
        ]
        output.append(
            " & ".join([COMPACT_SERIES_LABELS.get(series, series), *values]) + r" \\"
        )
    output.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
        ]
    )
    return output


def _latex_no_self_loop_matrix(
    summary: pd.DataFrame, correlations: pd.DataFrame
) -> list[str]:
    rows = publication_table_frame(summary, correlations, metric=SECONDARY_METRIC).set_index(
        ["series", "level"]
    )
    output = [
        r"\begin{table*}[!t]",
        r"\centering",
        (
            r"\caption{Secondary diagnostic excluding self-transitions: action-conditioned "
            r"transition similarity before and after CSCG refinement.}"
        ),
        r"\label{tab:pass_2_to_1000_no_self}",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{@{}lcccc@{}}",
        r"\toprule",
        r"Initialisation & $3\times3$ & $1\times4$ & $1\times5$ & $3\times4$ \\",
        r"\midrule",
    ]
    for series in SERIES_ORDER:
        cells = []
        for level in DEFAULT_LEVEL_ORDER:
            row = rows.loc[(series, level)]
            cells.append(
                _latex_endpoint_only(
                    float(row["metric_pass_2"]),
                    float(row["metric_pass_1000"]),
                )
            )
        output.append(
            " & ".join([COMPACT_SERIES_LABELS.get(series, series), *cells]) + r" \\"
        )
    output.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\vspace{2pt}",
            (
                r"\parbox{\textwidth}{\footnotesize\textit{Note:} "
                r"$S^{A,\neg\mathrm{self}}_{\mathrm{TV}}$ is action-conditioned transition "
                r"similarity after conditioning on non-self transitions. Entries show pass-2 "
                r"$\rightarrow$ pass-1000 means across 30 matched training seeds.}"
            ),
            r"\end{table*}",
            "",
        ]
    )
    return output


def write_latex_tables(
    summary: pd.DataFrame,
    correlations: pd.DataFrame,
    output_dir: Path,
    *,
    output_stem: str,
) -> list[Path]:
    header = [
        "% Generated by plot_pass_2_to_1000_metric_delta.py",
        "% Requires: \\usepackage{booktabs}",
        "",
    ]
    outputs = {
        f"{output_stem}_main_table.tex": [
            *header,
            *_latex_main_before_after_table(summary, correlations),
        ],
        f"{output_stem}_paired_change_and_association.tex": [
            *header,
            *_latex_paired_change_and_association_table(summary, correlations),
        ],
        f"{output_stem}_no_self_transition_diagnostic.tex": [
            *header,
            *_latex_no_self_loop_matrix(summary, correlations),
        ],
    }
    paths = []
    for name, lines in outputs.items():
        path = output_dir / name
        path.write_text("\n".join(lines), encoding="utf-8")
        paths.append(path)
    return paths


def write_markdown_tables(
    summary: pd.DataFrame,
    correlations: pd.DataFrame,
    output_dir: Path,
    *,
    output_stem: str,
) -> list[Path]:
    primary = publication_table_frame(summary, correlations, metric=PRIMARY_METRIC)
    main_lines = [
        "| Initialisation | Env. | SᴬTV (↑) | Evaluation NLL (↓) |",
        "| --- | --- | ---: | ---: |",
    ]
    for row in primary.to_dict("records"):
        main_lines.append(
            "| "
            + " | ".join(
                [
                    str(row["method"]),
                    str(row["environment"]),
                    f"{row['metric_pass_2']:.3f} → {row['metric_pass_1000']:.3f}",
                    f"{row['nll_pass_2']:.3f} → {row['nll_pass_1000']:.3f}",
                ]
            )
            + " |"
        )
    main_lines.extend(["", "Entries are means across 30 matched training seeds."])

    correlation_rows = correlations.loc[correlations["metric"] == PRIMARY_METRIC].set_index(
        ["series", "level"]
    )
    no_self = publication_table_frame(summary, correlations, metric=SECONDARY_METRIC).set_index(
        ["series", "level"]
    )
    paired_lines = [
        "## Panel A: paired mean improvements",
        "",
        "| Initialisation | Env. | I_S [95% CI] | I_NLL [95% CI] |",
        "| --- | --- | ---: | ---: |",
    ]
    for row in primary.to_dict("records"):
        paired_lines.append(
            "| "
            + " | ".join(
                [
                    str(row["method"]),
                    str(row["environment"]),
                    _markdown_change_cell(
                        row["metric_mean_change"],
                        row["metric_ci95_low"],
                        row["metric_ci95_high"],
                    ),
                    _markdown_change_cell(
                        row["nll_mean_improvement"],
                        row["nll_ci95_low"],
                        row["nll_ci95_high"],
                    ),
                ]
            )
            + " |"
        )
    paired_lines.extend(
        [
            "",
            "## Panel B: Spearman ρ(I_NLL, I_S)",
            "",
            "| Initialisation | 3×3 | 1×4 | 1×5 | 3×4 |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for series in SERIES_ORDER:
        paired_lines.append(
            "| "
            + " | ".join(
                [
                    COMPACT_SERIES_LABELS.get(series, series),
                    *[
                        _format_rho(float(correlation_rows.loc[(series, level), "rho_s"]))
                        for level in DEFAULT_LEVEL_ORDER
                    ],
                ]
            )
            + " |"
        )

    no_self_lines = [
        "| Initialisation | 3×3 | 1×4 | 1×5 | 3×4 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for series in SERIES_ORDER:
        cells = []
        for level in DEFAULT_LEVEL_ORDER:
            row = no_self.loc[(series, level)]
            cells.append(f"{row['metric_pass_2']:.3f} → {row['metric_pass_1000']:.3f}")
        no_self_lines.append(
            "| "
            + " | ".join([COMPACT_SERIES_LABELS.get(series, series), *cells])
            + " |"
        )

    outputs = {
        f"{output_stem}_main_table.md": main_lines,
        f"{output_stem}_paired_change_and_association.md": paired_lines,
        f"{output_stem}_no_self_transition_diagnostic.md": no_self_lines,
    }
    paths = []
    for name, lines in outputs.items():
        path = output_dir / name
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        paths.append(path)
    return paths


def write_summary(
    summary: pd.DataFrame,
    correlations: pd.DataFrame,
    output_dir: Path,
    *,
    output_stem: str,
) -> tuple[Path, Path]:
    text_path = output_dir / f"{output_stem}_statistical_summary.txt"
    latex_path = output_dir / f"{output_stem}_results_summary.tex"
    nll = summary.loc[summary["metric"] == NLL_METRIC]
    primary = summary.loc[summary["metric"] == PRIMARY_METRIC]
    secondary = summary.loc[summary["metric"] == SECONDARY_METRIC]
    primary_correlations = correlations.loc[correlations["metric"] == PRIMARY_METRIC]
    lines = [
        "Pass-2 to pass-1000 NLL/representation analysis",
        "=" * 51,
        "",
        "All summaries are within initialisation and environment; no conditions are pooled.",
        f"NLL reliably decreased: {int((nll['assessment'] == 'reliably_decreased').sum())}/{len(nll)} conditions.",
        f"Primary structural metric improved: {int((primary['assessment'] == 'improved').sum())}/{len(primary)} conditions.",
        f"Primary structural metric deteriorated: {int((primary['assessment'] == 'deteriorated').sum())}/{len(primary)} conditions.",
        f"Primary structural metric unchanged/uncertain: {int((primary['assessment'] == 'unchanged_or_uncertain').sum())}/{len(primary)} conditions.",
        f"No-self-loop diagnostic improved: {int((secondary['assessment'] == 'improved').sum())}/{len(secondary)} conditions.",
        f"No-self-loop diagnostic deteriorated: {int((secondary['assessment'] == 'deteriorated').sum())}/{len(secondary)} conditions.",
        f"No-self-loop diagnostic unchanged/uncertain: {int((secondary['assessment'] == 'unchanged_or_uncertain').sum())}/{len(secondary)} conditions.",
        "",
        "Condition-level summary",
        "-" * 23,
    ]
    for series in SERIES_ORDER:
        method_primary = primary.loc[primary["series"] == series]
        method_nll = nll.loc[nll["series"] == series]
        method_rho = primary_correlations.loc[primary_correlations["series"] == series, "rho_s"]
        lines.append(
            f"{COMPACT_SERIES_LABELS.get(series, series)}: pass-2 S_TV^A range "
            f"[{method_primary['mean_pass_2'].min():.3f}, {method_primary['mean_pass_2'].max():.3f}], "
            f"mean-change range [{method_primary['mean_change'].min():+.3f}, "
            f"{method_primary['mean_change'].max():+.3f}], NLL reliably decreased in "
            f"{int((method_nll['assessment'] == 'reliably_decreased').sum())}/4 environments, "
            f"rho_s range [{method_rho.min():+.3f}, {method_rho.max():+.3f}]."
        )
    lines.extend(
        [
            "",
            "Interpretation",
            "-" * 14,
            (
                "Primary result: continued optimisation reliably reduced evaluation NLL in every "
                "condition, but transition similarity deteriorated for MERCURY, TMGWR, "
                "Oracle-C+1st, and Oracle-C+Struct while improving for Standard CSCG and "
                "Oracle-C. Lower NLL therefore does not imply improved transition reconstruction."
            ),
            (
                "Secondary result: within-condition Spearman coefficients were mixed. The amount "
                "of NLL reduction is not a consistently reliable indicator of the amount of "
                "structural improvement across initialisations and environments; this does not "
                "imply universal absence of association in every condition."
            ),
        ]
    )
    text_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    nll_reliable = int((nll["assessment"] == "reliably_decreased").sum())
    improved = int((primary["assessment"] == "improved").sum())
    deteriorated = int((primary["assessment"] == "deteriorated").sum())
    latex_path.write_text(
        (
            f"Evaluation NLL decreased reliably from pass 2 to pass 1000 in {nll_reliable} of "
            f"{len(nll)} method--environment conditions. In contrast, action-conditioned "
            f"transition similarity improved in {improved} conditions and deteriorated in "
            f"{deteriorated}: it deteriorated across every environment for MERCURY, TMGWR, "
            "Oracle-C+1st, and Oracle-C+Struct, but improved throughout for Standard CSCG and "
            "Oracle-C. Oracle-C+Struct began near structurally perfect transition similarity, so "
            "absolute endpoints are retained and initialisations are not pooled or normalised. "
            "Thus, lower NLL should not be interpreted as evidence of improved structural "
            "reconstruction. As a secondary result, the within-condition Spearman coefficients "
            "were heterogeneous: the magnitude of NLL reduction was not a consistently reliable "
            "indicator of structural improvement across conditions, without implying that every "
            "individual condition lacked an association.\n"
        ),
        encoding="utf-8",
    )
    return text_path, latex_path


def write_methods_snippet(output_dir: Path, *, output_stem: str) -> Path:
    path = output_dir / f"{output_stem}_methods_snippet.tex"
    path.write_text(
        "Within each initialisation and environment, pass-2 and pass-1000 results were paired by "
        "training seed ($n=30$). We report the absolute endpoint means and the paired mean change "
        "with a two-sided 95\\% Student-$t$ confidence interval. Metric change was defined as "
        "$M_{1000}-M_2$, while NLL improvement was defined as "
        "$I_{\\mathrm{NLL}}=\\mathrm{NLL}_2-\\mathrm{NLL}_{1000}$, so positive values always "
        "denote improvement. As a secondary descriptive analysis, Spearman's $\\rho_s$ related "
        "per-seed NLL improvement to per-seed transition-similarity change within each condition. "
        "Initialisations were not pooled, and no additional modelling or multiplicity correction "
        "was applied.\n",
        encoding="utf-8",
    )
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", type=Path, default=DEFAULT_STUDY_ROOT)
    parser.add_argument(
        "--output-dir", "--plot-dir", dest="output_dir", type=Path, default=DEFAULT_OUTPUT_DIR
    )
    parser.add_argument("--protocol-id", default=DEFAULT_PROTOCOL_ID)
    parser.add_argument("--batch-size", type=int, default=40)
    parser.add_argument("--output-stem", default=DEFAULT_OUTPUT_STEM)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rows = load_rows(
        args.study_root,
        args.protocol_id,
        args.batch_size,
        set(DEFAULT_LEVEL_ORDER),
    )
    comparison_rows = rows_at_comparison_steps(rows, args.protocol_id)
    paired = paired_improvement_deltas(comparison_rows)
    summary = summarise_paired_changes(paired)
    correlations = predictor_correlations(paired)
    validation = validate_analysis(paired, summary, correlations)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    paired_path = args.output_dir / f"{args.output_stem}_paired_seed_values.csv"
    summary_path = args.output_dir / f"{args.output_stem}_paired_summary.csv"
    correlation_path = args.output_dir / f"{args.output_stem}_predictor_correlations.csv"
    primary_path = args.output_dir / f"{args.output_stem}_main_before_after_table.csv"
    condition_response_path = (
        args.output_dir / f"{args.output_stem}_condition_level_structural_response.csv"
    )
    seed_cloud_path = (
        args.output_dir / f"{args.output_stem}_condition_level_seed_cloud_values.csv"
    )
    no_self_response_path = (
        args.output_dir / f"{args.output_stem}_condition_level_no_self_transition_response.csv"
    )
    purity_response_path = (
        args.output_dir / f"{args.output_stem}_condition_level_purity_response.csv"
    )
    validation_path = args.output_dir / f"{args.output_stem}_validation_report.txt"
    paired.to_csv(paired_path, index=False, float_format="%.17g")
    summary.to_csv(summary_path, index=False, float_format="%.17g")
    correlations.to_csv(correlation_path, index=False, float_format="%.17g")
    main_before_after_frame(summary, correlations).to_csv(
        primary_path, index=False, float_format="%.17g"
    )
    condition_response = condition_response_frame(summary)
    no_self_response = condition_metric_response_frame(summary, metric=SECONDARY_METRIC)
    purity_response = condition_metric_response_frame(summary, metric="latent_precision")
    primary_seed_values = seed_response_frame(paired, metric=PRIMARY_METRIC)
    no_self_seed_values = seed_response_frame(paired, metric=SECONDARY_METRIC)
    purity_seed_values = seed_response_frame(paired, metric="latent_precision")
    seed_cloud_values = pd.concat(
        [primary_seed_values, no_self_seed_values, purity_seed_values], ignore_index=True
    )
    condition_response.to_csv(condition_response_path, index=False, float_format="%.17g")
    no_self_response.to_csv(no_self_response_path, index=False, float_format="%.17g")
    purity_response.to_csv(purity_response_path, index=False, float_format="%.17g")
    seed_cloud_values.to_csv(seed_cloud_path, index=False, float_format="%.17g")
    validation_path.write_text(validation, encoding="utf-8")
    phase_space_specs = [
        {
            "panel_label": "Purity",
            "condition_summary": purity_response,
            "seed_values": purity_seed_values,
            "initial_column": "quality_pass_2",
            "endpoint_column": "quality_pass_1000",
            "endpoint_axis_label": r"Pass-1000 latent purity, $P_{1000}$",
            "absolute_axis_label": r"Latent purity, $P$",
            "left_delta_axis_label": r"Later change, $\Delta P_{2\to1000}$",
            "right_delta_axis_label": r"Checkpoint change, $\Delta P$",
        },
        {
            "panel_label": "Transition similarity",
            "condition_summary": condition_response,
            "seed_values": primary_seed_values,
            "initial_column": "transition_similarity_pass_2",
            "endpoint_column": "transition_similarity_pass_1000",
            "endpoint_axis_label": (
                r"Pass-1000 transition similarity, $S^A_{\mathrm{TV},1000}$"
            ),
            "absolute_axis_label": r"Transition similarity, $S^A_{\mathrm{TV}}$",
            "left_delta_axis_label": (
                r"Later change, $\Delta S^A_{\mathrm{TV},2\to1000}$"
            ),
            "right_delta_axis_label": (
                r"Checkpoint change, $\Delta S^A_{\mathrm{TV}}$"
            ),
        },
        {
            "panel_label": "No-self similarity",
            "condition_summary": no_self_response,
            "seed_values": no_self_seed_values,
            "initial_column": "quality_pass_2",
            "endpoint_column": "quality_pass_1000",
            "endpoint_axis_label": (
                "Pass-1000 no-self similarity, "
                r"$S^{A,\neg\mathrm{self}}_{\mathrm{TV},1000}$"
            ),
            "absolute_axis_label": (
                r"No-self similarity, $S^{A,\neg\mathrm{self}}_{\mathrm{TV}}$"
            ),
            "left_delta_axis_label": (
                r"Later change, "
                r"$\Delta S^{A,\neg\mathrm{self}}_{\mathrm{TV},2\to1000}$"
            ),
            "right_delta_axis_label": (
                r"Checkpoint change, "
                r"$\Delta S^{A,\neg\mathrm{self}}_{\mathrm{TV}}$"
            ),
        },
    ]
    initialisation_phase_space_specs = [
        {
            **spec,
            "seed_values": three_checkpoint_seed_frame(comparison_rows, metric=metric),
        }
        for spec, metric in zip(
            phase_space_specs,
            ("latent_precision", PRIMARY_METRIC, SECONDARY_METRIC),
        )
    ]
    pass_1_likelihood_series = {
        "vanilla_cscg",
        "oracle_capacity",
        "oracle_capacity_first_order",
        "oracle_capacity_structural",
    }
    metric_likelihood_space_specs = [
        {
            **spec,
            "seed_values": three_checkpoint_seed_frame(
                comparison_rows,
                metric=metric,
                pass_1_step=2_000,
                allow_missing_pass_1_series=(
                    {"mercury_cscg_hybrid", "tmgwr_cscg_hybrid"}
                    if comparison_rows is not rows else None
                ),
            ),
            "pass_1_series": pass_1_likelihood_series,
        }
        for spec, metric in zip(
            [
                initialisation_phase_space_specs[0],
                {
                    "panel_label": "Concentration",
                    "absolute_axis_label": r"Latent concentration, $C$",
                },
                *initialisation_phase_space_specs[1:],
            ],
            ("latent_precision", "latent_recall", PRIMARY_METRIC, SECONDARY_METRIC),
        )
    ]
    outputs = [
        paired_path,
        summary_path,
        correlation_path,
        primary_path,
        condition_response_path,
        no_self_response_path,
        purity_response_path,
        seed_cloud_path,
        *plot_condition_level_response(
            condition_response, args.output_dir, output_stem=args.output_stem
        ),
        *plot_combined_seed_cloud_responses(
            [
                {
                    "row_label": "Purity",
                    "panel_label": "Purity",
                    "condition_summary": purity_response,
                    "seed_values": purity_seed_values,
                    "initial_column": "quality_pass_2",
                    "endpoint_column": "quality_pass_1000",
                    "change_column": "quality_change",
                    "change_ci_low_column": "quality_change_ci95_low",
                    "change_ci_high_column": "quality_change_ci95_high",
                    "initial_axis_label": r"Pass-2 latent purity, $P_2$",
                    "endpoint_axis_label": r"Pass-1000 latent purity, $P_{1000}$",
                    "change_axis_label": r"Purity change, $I_P$",
                },
                {
                    "row_label": r"Transition similarity, $S^A_{\mathrm{TV}}$",
                    "panel_label": "Transition similarity",
                    "condition_summary": condition_response,
                    "seed_values": primary_seed_values,
                    "initial_column": "transition_similarity_pass_2",
                    "endpoint_column": "transition_similarity_pass_1000",
                    "change_column": "structural_change",
                    "change_ci_low_column": "structural_change_ci95_low",
                    "change_ci_high_column": "structural_change_ci95_high",
                    "initial_axis_label": r"Pass-2 transition similarity, $S^A_{\mathrm{TV},2}$",
                    "endpoint_axis_label": (
                        r"Pass-1000 transition similarity, $S^A_{\mathrm{TV},1000}$"
                    ),
                    "change_axis_label": r"Structural change, $I_S$",
                },
                {
                    "row_label": "No-self transition similarity",
                    "panel_label": "No-self similarity",
                    "condition_summary": no_self_response,
                    "seed_values": no_self_seed_values,
                    "initial_column": "quality_pass_2",
                    "endpoint_column": "quality_pass_1000",
                    "change_column": "quality_change",
                    "change_ci_low_column": "quality_change_ci95_low",
                    "change_ci_high_column": "quality_change_ci95_high",
                    "initial_axis_label": (
                        "Pass-2 no-self similarity, "
                        r"$S^{A,\neg\mathrm{self}}_{\mathrm{TV},2}$"
                    ),
                    "endpoint_axis_label": (
                        "Pass-1000 no-self similarity, "
                        r"$S^{A,\neg\mathrm{self}}_{\mathrm{TV},1000}$"
                    ),
                    "change_axis_label": (
                        r"No-self structural change, $I_{S,\neg\mathrm{self}}$"
                    ),
                },
            ],
            args.output_dir,
            output_stem=args.output_stem,
        ),
        *plot_combined_absolute_phase_space_responses(
            phase_space_specs,
            args.output_dir,
            output_stem=args.output_stem,
        ),
        *plot_combined_absolute_phase_space_responses(
            initialisation_phase_space_specs,
            args.output_dir,
            output_stem=args.output_stem,
            include_initialisation=True,
            show_left_seed_points=True,
            show_right_seed_points=False,
            show_seed_trajectories=False,
        ),
        *plot_combined_absolute_phase_space_responses(
            initialisation_phase_space_specs,
            args.output_dir,
            output_stem=args.output_stem,
            include_initialisation=True,
            show_left_seed_points=True,
            show_right_seed_points=False,
            show_seed_trajectories=False,
            delta_axes=True,
        ),
        *plot_metric_likelihood_spaces(
            metric_likelihood_space_specs,
            args.output_dir,
            output_stem=args.output_stem,
        ),
        *plot_metric_likelihood_spaces(
            metric_likelihood_space_specs,
            args.output_dir,
            output_stem=args.output_stem,
            separate_environments=True,
        ),
        *plot_mercury_source_vs_cscg_initialisation(
            rows,
            args.study_root,
            args.protocol_id,
            args.output_dir,
            output_stem=args.output_stem,
        ),
        *plot_condition_level_response(
            condition_response,
            args.output_dir,
            output_stem=args.output_stem,
            output_suffix="condition_level_structural_response_with_seed_clouds",
            seed_values=primary_seed_values,
        ),
        *plot_condition_level_response(
            no_self_response,
            args.output_dir,
            output_stem=args.output_stem,
            initial_column="quality_pass_2",
            endpoint_column="quality_pass_1000",
            change_column="quality_change",
            change_ci_low_column="quality_change_ci95_low",
            change_ci_high_column="quality_change_ci95_high",
            initial_axis_label=(
                "Pass-2 no-self transition\nsimilarity, "
                r"$S^{A,\neg\mathrm{self}}_{\mathrm{TV},2}$"
            ),
            endpoint_axis_label=(
                "Pass-1000 no-self transition\nsimilarity, "
                r"$S^{A,\neg\mathrm{self}}_{\mathrm{TV},1000}$"
            ),
            change_axis_label=(
                "No-self structural change,\n"
                r"$I_{S,\neg\mathrm{self}}$"
            ),
            output_suffix="condition_level_no_self_transition_response",
        ),
        *plot_condition_level_response(
            no_self_response,
            args.output_dir,
            output_stem=args.output_stem,
            initial_column="quality_pass_2",
            endpoint_column="quality_pass_1000",
            change_column="quality_change",
            change_ci_low_column="quality_change_ci95_low",
            change_ci_high_column="quality_change_ci95_high",
            initial_axis_label=(
                "Pass-2 no-self transition\nsimilarity, "
                r"$S^{A,\neg\mathrm{self}}_{\mathrm{TV},2}$"
            ),
            endpoint_axis_label=(
                "Pass-1000 no-self transition\nsimilarity, "
                r"$S^{A,\neg\mathrm{self}}_{\mathrm{TV},1000}$"
            ),
            change_axis_label=(
                "No-self structural change,\n"
                r"$I_{S,\neg\mathrm{self}}$"
            ),
            output_suffix="condition_level_no_self_transition_response_with_seed_clouds",
            seed_values=no_self_seed_values,
        ),
        *plot_condition_level_response(
            purity_response,
            args.output_dir,
            output_stem=args.output_stem,
            initial_column="quality_pass_2",
            endpoint_column="quality_pass_1000",
            change_column="quality_change",
            change_ci_low_column="quality_change_ci95_low",
            change_ci_high_column="quality_change_ci95_high",
            initial_axis_label=r"Pass-2 latent purity, $P_2$",
            endpoint_axis_label=r"Pass-1000 latent purity, $P_{1000}$",
            change_axis_label=r"Purity change, $I_P=P_{1000}-P_2$",
            output_suffix="condition_level_purity_response",
        ),
        *plot_condition_level_response(
            purity_response,
            args.output_dir,
            output_stem=args.output_stem,
            initial_column="quality_pass_2",
            endpoint_column="quality_pass_1000",
            change_column="quality_change",
            change_ci_low_column="quality_change_ci95_low",
            change_ci_high_column="quality_change_ci95_high",
            initial_axis_label=r"Pass-2 latent purity, $P_2$",
            endpoint_axis_label=r"Pass-1000 latent purity, $P_{1000}$",
            change_axis_label=r"Purity change, $I_P=P_{1000}-P_2$",
            output_suffix="condition_level_purity_response_with_seed_clouds",
            seed_values=purity_seed_values,
        ),
        *plot_paired_absolute_values(
            paired, args.output_dir, output_stem=args.output_stem
        ),
        *write_latex_tables(
            summary, correlations, args.output_dir, output_stem=args.output_stem
        ),
        *write_markdown_tables(
            summary, correlations, args.output_dir, output_stem=args.output_stem
        ),
        *write_summary(
            summary, correlations, args.output_dir, output_stem=args.output_stem
        ),
        write_methods_snippet(args.output_dir, output_stem=args.output_stem),
        validation_path,
    ]
    print(validation)
    print("Pass-2 to pass-1000 statistical analysis complete")
    for output in outputs:
        print(f"  {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
