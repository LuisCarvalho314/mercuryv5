"""Focused TV-similarity figures for batch-40 comparison studies.

Produces one figure per rand_prob, comparison, metric family, and self-loop
policy. Each figure shows seed-level points plus mean/95% CI across levels.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib
import numpy as np
import polars as pl
from matplotlib import pyplot as plt
from scipy.stats import t

from mercury_runs.analysis.plots.batch_summary_forest import (
    HYBRID_SERIES_ORDER,
    LEVEL_LABELS,
    LEVEL_ORDER,
    MERCURY_SERIES_ORDER,
    _build_frame,
    _parse_protocol_ids_by_rand,
)
from mercury_runs.analysis.plots import publication_style as pub


DEFAULT_STUDY_ROOT = Path(
    "results/current/studies/mercury_cscg_hybrid_vs_cscg_batch_size_30_seed_common_eval_walks2"
)
DEFAULT_OUTPUT_DIR = Path(
    "results/current/notebook_plots/mercury_cscg_hybrid_vs_cscg_batch_size_30_seed_common_eval_walks2"
)

COMPARISONS: dict[str, dict[str, Any]] = {
    "final": {
        "label": pub.PUBLICATION_COMPARISON_TITLES["final"],
        "series_order": HYBRID_SERIES_ORDER,
        "primary_series": "hybrid",
    },
    "checkpoint_4000": {
        "label": pub.PUBLICATION_COMPARISON_TITLES["checkpoint_4000"],
        "series_order": HYBRID_SERIES_ORDER,
        "primary_series": "hybrid",
    },
    "checkpoint_2000": {
        "label": pub.PUBLICATION_COMPARISON_TITLES["checkpoint_2000"],
        "series_order": MERCURY_SERIES_ORDER,
        "primary_series": "mercury",
    },
}

METRIC_SPECS: tuple[dict[str, str], ...] = (
    {
        "family": "tv_similarity",
        "self_loop_policy": "with_self_loops",
        "column": "latent_mean_total_variation_similarity",
        "label": "TV similarity",
    },
    {
        "family": "tv_similarity",
        "self_loop_policy": "without_self_loops",
        "column": "latent_non_self_loop_total_variation_similarity",
        "label": "TV similarity without self-loops",
    },
    {
        "family": "action_tv_similarity",
        "self_loop_policy": "with_self_loops",
        "column": "latent_action_conditioned_mean_total_variation_similarity",
        "label": "Action TV similarity",
    },
    {
        "family": "action_tv_similarity",
        "self_loop_policy": "without_self_loops",
        "column": "latent_action_conditioned_non_self_loop_total_variation_similarity",
        "label": "Action TV similarity without self-loops",
    },
)


def _parse_float_list(raw: str) -> list[float]:
    return [float(part.strip()) for part in raw.split(",") if part.strip()]


def _rand_slug(rand_prob: float) -> str:
    return "rand_" + f"{float(rand_prob):g}".replace("-", "m").replace(".", "p")


def _ci95_half_width(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    if values.size <= 1:
        return 0.0
    return float(t.ppf(0.975, values.size - 1) * values.std(ddof=1) / np.sqrt(values.size))


def _seed_values(frame: pl.DataFrame, *, level: int, series: str, metric: str) -> np.ndarray:
    values = (
        frame.filter((pl.col("level") == int(level)) & (pl.col("plot_series") == str(series)))
        .drop_nulls(["seed", metric])
        .group_by("seed")
        .agg(pl.col(metric).mean().alias("value"))
        .sort("seed")
    )
    if values.is_empty():
        return np.asarray([], dtype=np.float64)
    return np.asarray(values["value"].to_list(), dtype=np.float64)


def _summaries_for_csv(
    frame: pl.DataFrame,
    *,
    rand_prob: float,
    comparison: str,
    metric: str,
    metric_label: str,
    self_loop_policy: str,
    series_order: list[str],
    series_labels: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for level in LEVEL_ORDER:
        for series in series_order:
            values = _seed_values(frame, level=level, series=series, metric=metric)
            if values.size == 0:
                continue
            rows.append(
                {
                    "rand_prob": float(rand_prob),
                    "comparison": comparison,
                    "metric": metric_label,
                    "self_loop_policy": self_loop_policy,
                    "level": int(level),
                    "level_label": LEVEL_LABELS.get(int(level), str(level)),
                    "series": series,
                    "series_label": (series_labels or {}).get(series, pub.method_label(series, comparison=comparison)),
                    "n": int(values.size),
                    "mean": float(values.mean()),
                    "sd": float(values.std(ddof=1)) if values.size > 1 else 0.0,
                    "ci95": _ci95_half_width(values),
                }
            )
    return rows


def _warn_missing_series(frame: pl.DataFrame, *, comparison: str, rand_prob: float, series_order: list[str]) -> None:
    missing: list[str] = []
    for series in series_order:
        for level in LEVEL_ORDER:
            present = frame.filter((pl.col("level") == int(level)) & (pl.col("plot_series") == str(series)))
            if present.is_empty():
                missing.append(f"{pub.method_label(series, comparison=comparison)} level={level}")
    if missing:
        print(
            f"Warning: comparison={comparison} rand_prob={rand_prob:g} missing: {', '.join(missing)}",
            file=sys.stderr,
        )


def _missing_series(frame: pl.DataFrame, *, comparison: str, series_order: list[str]) -> list[dict[str, Any]]:
    missing: list[dict[str, Any]] = []
    for series in series_order:
        for level in LEVEL_ORDER:
            present = frame.filter((pl.col("level") == int(level)) & (pl.col("plot_series") == str(series)))
            if present.is_empty():
                missing.append(
                    {
                        "series": series,
                        "series_label": pub.method_label(series, comparison=comparison),
                        "level": int(level),
                        "level_label": LEVEL_LABELS.get(int(level), str(level)),
                    }
                )
    return missing


def _missing_metric_series(frame: pl.DataFrame, *, comparison: str, metric: str, series_order: list[str]) -> list[dict[str, Any]]:
    missing: list[dict[str, Any]] = []
    for series in series_order:
        for level in LEVEL_ORDER:
            values = _seed_values(frame, level=int(level), series=str(series), metric=str(metric))
            if values.size == 0:
                missing.append(
                    {
                        "series": series,
                        "series_label": pub.method_label(series, comparison=comparison),
                        "level": int(level),
                        "level_label": LEVEL_LABELS.get(int(level), str(level)),
                    }
                )
    return missing


def _available_series_order(source: pl.DataFrame, series_order: list[str]) -> list[str]:
    available = set(source.get_column("plot_series").to_list())
    return [series for series in series_order if series in available]


def _plot_one(
    frame: pl.DataFrame,
    *,
    output_path: Path,
    rand_prob: float,
    comparison: str | None = None,
    comparison_label: str,
    metric: str,
    metric_label: str,
    self_loop_policy: str,
    series_order: list[str],
    primary_series: str | None,
    series_labels: dict[str, str] | None = None,
    series_colors: dict[str, str] | None = None,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.style.use("default")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.edgecolor": "#2F3740",
            "axes.labelcolor": "#111827",
            "xtick.color": "#111827",
            "ytick.color": "#111827",
        }
    )

    y_positions = np.arange(len(series_order))[::-1]
    fig, axes = plt.subplots(1, len(LEVEL_ORDER), figsize=(10.4, 3.9), sharex=True, sharey=True, dpi=300)
    axes = np.ravel(axes)
    for axis_index, (axis, level) in enumerate(zip(axes, LEVEL_ORDER)):
        for series_index, series in enumerate(series_order):
            y = y_positions[series_index]
            color = (series_colors or {}).get(series, pub.method_color(series, comparison=comparison))
            values = _seed_values(frame, level=level, series=series, metric=metric)
            if values.size == 0:
                continue
            jitter = np.linspace(-0.11, 0.11, values.size) if values.size > 1 else np.asarray([0.0])
            axis.scatter(values, y + jitter, s=12, color=color, alpha=0.22, linewidths=0, zorder=1)
            mean = float(values.mean())
            ci95 = _ci95_half_width(values)
            emphasized = series == primary_series
            axis.errorbar(
                mean,
                y,
                xerr=ci95,
                fmt="o",
                color=color,
                ecolor=color,
                elinewidth=2.1 if emphasized else 1.35,
                capsize=3.0,
                capthick=2.1 if emphasized else 1.35,
                markersize=6.2 if emphasized else 5.0,
                markeredgecolor="#111827",
                markeredgewidth=0.35,
                zorder=3,
            )
            if emphasized:
                axis.axhspan(y - 0.38, y + 0.38, color=color, alpha=0.07, zorder=0)

        axis.set_title(LEVEL_LABELS.get(level, str(level)), fontsize=11, fontweight="bold", pad=8)
        axis.set_xlim(0.0, 1.0)
        axis.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
        axis.set_ylim(-0.65, len(series_order) - 0.35)
        axis.grid(axis="x", alpha=0.22, linewidth=0.8)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.spines["left"].set_visible(axis_index == 0)
        axis.tick_params(axis="both", labelsize=8.7)
        if axis_index == 0:
            axis.set_yticks(y_positions, [(series_labels or {}).get(series, pub.method_label(series, comparison=comparison)) for series in series_order])
            for label, series in zip(axis.get_yticklabels(), series_order):
                if series == primary_series:
                    label.set_fontweight("bold")
            axis.tick_params(axis="y", labelsize=9.0, pad=5)
        else:
            axis.tick_params(axis="y", labelleft=False, left=False)

    self_loop_label = "self-loops included" if self_loop_policy == "with_self_loops" else "self-loops excluded"
    fig.suptitle(
        f"{comparison_label}: {metric_label} ({self_loop_label}, rand_prob={rand_prob:g})",
        fontsize=12.2,
        y=0.985,
    )
    fig.supxlabel("Similarity (1 - total variation)", fontsize=10.3, y=0.045)
    fig.subplots_adjust(left=0.175, right=0.995, top=0.82, bottom=0.16, wspace=0.08)
    fig.savefig(output_path, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study_root", type=Path, default=DEFAULT_STUDY_ROOT)
    parser.add_argument("--protocol_id", type=str, required=True)
    parser.add_argument("--protocol_ids_by_rand", type=str, default=None)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--batch_size", type=int, default=40)
    parser.add_argument("--rand_probs", type=str, default="0.0,0.3")
    parser.add_argument("--comparisons", type=str, default="final,checkpoint_4000,checkpoint_2000")
    args = parser.parse_args()

    matplotlib.use("Agg", force=True)
    rand_probs = _parse_float_list(args.rand_probs)
    comparison_keys = [item.strip() for item in str(args.comparisons).split(",") if item.strip()]
    protocol_ids_by_rand = _parse_protocol_ids_by_rand(args.protocol_ids_by_rand)
    frame = _build_frame(
        args.study_root,
        args.protocol_id,
        protocol_ids_by_rand=protocol_ids_by_rand,
        batch_size=int(args.batch_size),
    )
    if frame.is_empty():
        print("No evaluation records found. Run evaluate_study first.", file=sys.stderr)
        sys.exit(1)

    csv_rows: list[dict[str, Any]] = []
    manifest: dict[str, Any] = {
        "study_root": str(args.study_root),
        "output_dir": str(args.output_dir),
        "batch_size": int(args.batch_size),
        "protocol_id": args.protocol_id,
        "protocol_ids_by_rand": {str(key): value for key, value in protocol_ids_by_rand.items()},
        "figures": [],
    }

    for rand_prob in rand_probs:
        rand_frame = frame.filter(pl.col("rand_prob") == float(rand_prob))
        if rand_frame.is_empty():
            print(f"Warning: no records for rand_prob={rand_prob:g}", file=sys.stderr)
            continue
        rand_dir = args.output_dir / _rand_slug(rand_prob)
        for comparison in comparison_keys:
            if comparison not in COMPARISONS:
                raise ValueError(f"Unknown comparison: {comparison}")
            comparison_spec = COMPARISONS[comparison]
            series_order = list(comparison_spec["series_order"])
            source = rand_frame.filter(pl.col("comparison") == comparison)
            if source.is_empty():
                print(f"Warning: no records for comparison={comparison} rand_prob={rand_prob:g}", file=sys.stderr)
                continue
            series_order = _available_series_order(source, series_order)
            _warn_missing_series(source, comparison=comparison, rand_prob=rand_prob, series_order=series_order)
            missing = _missing_series(source, comparison=comparison, series_order=series_order)
            for metric_spec in METRIC_SPECS:
                metric_missing = _missing_metric_series(
                    source,
                    comparison=comparison,
                    metric=str(metric_spec["column"]),
                    series_order=series_order,
                )
                if metric_missing:
                    missing_text = ", ".join(
                        f"{item['series_label']} level={item['level']}" for item in metric_missing
                    )
                    print(
                        f"Warning: comparison={comparison} rand_prob={rand_prob:g} "
                        f"metric={metric_spec['label']} missing: {missing_text}",
                        file=sys.stderr,
                    )
                out = (
                    rand_dir
                    / comparison
                    / metric_spec["family"]
                    / metric_spec["self_loop_policy"]
                    / f"{comparison}_{metric_spec['family']}_{metric_spec['self_loop_policy']}_{_rand_slug(rand_prob)}.png"
                )
                path = _plot_one(
                    source,
                    output_path=out,
                    rand_prob=rand_prob,
                    comparison=comparison,
                    comparison_label=str(comparison_spec["label"]),
                    metric=str(metric_spec["column"]),
                    metric_label=str(metric_spec["label"]),
                    self_loop_policy=str(metric_spec["self_loop_policy"]),
                    series_order=series_order,
                    primary_series=str(comparison_spec["primary_series"]),
                )
                print(path)
                manifest["figures"].append(
                    {
                        "path": str(path),
                        "rand_prob": float(rand_prob),
                        "comparison": comparison,
                        "metric_family": str(metric_spec["family"]),
                        "self_loop_policy": str(metric_spec["self_loop_policy"]),
                        "missing_series": missing,
                        "missing_metric_series": metric_missing,
                    }
                )
                csv_rows.extend(
                    _summaries_for_csv(
                        source,
                        rand_prob=rand_prob,
                        comparison=comparison,
                        metric=str(metric_spec["column"]),
                        metric_label=str(metric_spec["label"]),
                        self_loop_policy=str(metric_spec["self_loop_policy"]),
                        series_order=series_order,
                    )
                )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = args.output_dir / "tv_similarity_summary.csv"
    if csv_rows:
        with summary_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0].keys()))
            writer.writeheader()
            writer.writerows(csv_rows)
        print(summary_csv)
    (args.output_dir / "tv_similarity_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
