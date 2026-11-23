"""Prespecified paired statistical analysis for the MERCURY main study.

The independent experimental unit is the training seed. Canonical evaluation
records already pool the shared held-out walks within each seed; this script
never treats evaluation walks as independent observations.
"""
from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any, Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

from mercury_runs.analysis.evaluation_records import load_evaluation_step, load_study_manifest
from mercury_runs.analysis.plots import publication_style as pub
from mercury_runs.analysis.plots.batch_summary_forest import (
    _build_frame,
    _record_lookup_for_comparison,
)
from scripts.analysis.study_selection import MAIN_STUDY_ROOT, PROTOCOL_ID


# ---------------------------------------------------------------------------
# Editable analysis configuration
# ---------------------------------------------------------------------------

DEFAULT_STUDY_ROOT = MAIN_STUDY_ROOT
DEFAULT_OUTPUT_DIR = Path("results/current/notebook_plots/main_study/statistical_analysis")
DEFAULT_PROTOCOL_ID = PROTOCOL_ID
EXPECTED_SEEDS = tuple(range(30))
BATCH_SIZE = 40

ENVIRONMENTS = {
    13: "3x3",
    21: "1x4",
    20: "1x5",
    22: "3x4",
}

CHECKPOINTS = {
    "checkpoint_2000": {
        "label": "Single-pass acquisition",
        "short_label": "Single pass",
        "budget_step": 2_000,
    },
    "checkpoint_4000": {
        "label": "Matched-budget / two-pass refinement",
        "short_label": "Two pass",
        "budget_step": 4_000,
    },
    "final": {
        "label": "Extensive CSCG refinement",
        "short_label": "Extensive",
        "budget_step": 2_000_000,
    },
}

METRICS = {
    "latent_precision": {
        "label": "Latent purity",
        "latex": r"Latent purity",
    },
    "latent_recall": {
        "label": "Latent concentration",
        "latex": r"Latent concentration",
    },
    "latent_mean_total_variation_similarity": {
        "label": "Policy-collapsed TV similarity",
        "latex": r"Policy-collapsed TV similarity",
    },
    "latent_non_self_loop_total_variation_similarity": {
        "label": "Policy-collapsed no-self-loop TV similarity",
        "latex": r"Policy-collapsed no-self-loop TV similarity",
    },
    "latent_action_conditioned_mean_total_variation_similarity": {
        "label": "Action-conditioned TV similarity",
        "latex": r"Action-conditioned TV similarity",
    },
    "latent_action_conditioned_non_self_loop_total_variation_similarity": {
        "label": "Action-conditioned no-self-loop TV similarity",
        "latex": r"Action-conditioned no-self-loop TV similarity",
    },
}

METHOD_LABELS = {
    "mercury": "MERCURY",
    "mercury_initialised": "MERCURY-initialised CSCG",
    "tmgwr": "TMGWR",
    "tmgwr_initialised": "TMGWR-initialised CSCG",
    "standard_cscg": "Standard CSCG",
    "oracle_capacity": "Oracle capacity",
    "oracle_capacity_first_order": "Oracle capacity + first-order structure",
    "oracle_capacity_structural": "Oracle capacity + oracle structure",
}

SERIES_BY_METHOD = {
    "standard_cscg": "vanilla_cscg",
    "oracle_capacity": "oracle_capacity",
    "oracle_capacity_first_order": "oracle_capacity_first_order",
    "oracle_capacity_structural": "oracle_capacity_structural",
    "mercury": "mercury",
    "mercury_initialised": "hybrid",
    "tmgwr": "tmgwr_hybrid",
    "tmgwr_initialised": "tmgwr_hybrid",
}

PLANNED_CONTRASTS = (
    ("mercury_vs_standard", "mercury_derived", "standard_cscg"),
    ("tmgwr_vs_standard", "tmgwr_derived", "standard_cscg"),
    ("mercury_vs_tmgwr", "mercury_derived", "tmgwr_derived"),
    ("oracle_capacity_vs_standard", "oracle_capacity", "standard_cscg"),
    (
        "oracle_first_order_vs_capacity",
        "oracle_capacity_first_order",
        "oracle_capacity",
    ),
    (
        "oracle_structure_vs_first_order",
        "oracle_capacity_structural",
        "oracle_capacity_first_order",
    ),
    ("mercury_vs_oracle_capacity", "mercury_derived", "oracle_capacity"),
    (
        "mercury_vs_oracle_first_order",
        "mercury_derived",
        "oracle_capacity_first_order",
    ),
    (
        "mercury_vs_oracle_structure",
        "mercury_derived",
        "oracle_capacity_structural",
    ),
)

SUMMARY_CONTRAST_IDS = {
    "mercury_vs_standard",
    "mercury_vs_tmgwr",
    "mercury_vs_oracle_capacity",
    "mercury_vs_oracle_first_order",
    "mercury_vs_oracle_structure",
}

MAIN_BODY_CONTRAST_IDS = (
    "mercury_vs_standard",
    "mercury_vs_tmgwr",
    "mercury_vs_oracle_capacity",
    "mercury_vs_oracle_first_order",
    "mercury_vs_oracle_structure",
)

MAIN_BODY_COMPARATOR_LABELS = {
    "mercury_vs_standard": "Standard CSCG",
    "mercury_vs_tmgwr": "TMGWR",
    "mercury_vs_oracle_capacity": "Oracle capacity",
    "mercury_vs_oracle_first_order": "Oracle first-order",
    "mercury_vs_oracle_structure": "Oracle structure",
}

FULL_COLUMNS = [
    "metric",
    "metric_label",
    "environment",
    "environment_label",
    "checkpoint",
    "checkpoint_label",
    "checkpoint_step",
    "contrast_id",
    "contrast_label",
    "method_a",
    "method_b",
    "n_pairs",
    "mean_a",
    "mean_b",
    "mean_difference",
    "ci95_low",
    "ci95_high",
    "sd_difference",
    "se_difference",
    "t_statistic",
    "df",
    "p_raw",
    "p_holm",
    "significant_holm_0_05",
    "cohens_dz",
    "holm_family",
    "holm_family_n_tests",
    "expected_n_pairs",
    "n_pairs_is_expected",
    "available",
    "unavailable_reason",
    "missing_seeds_a",
    "missing_seeds_b",
    "unpaired_seeds_a",
    "unpaired_seeds_b",
    "nonfinite_seeds_a",
    "nonfinite_seeds_b",
]


@dataclass(frozen=True)
class ValidationFinding:
    severity: str
    category: str
    message: str


def resolve_checkpoint_method(method: str, checkpoint: str) -> str:
    """Resolve a scientific method role to the condition stored at a checkpoint."""
    if method == "mercury_derived":
        return "mercury" if checkpoint == "checkpoint_2000" else "mercury_initialised"
    if method == "tmgwr_derived":
        return "tmgwr" if checkpoint == "checkpoint_2000" else "tmgwr_initialised"
    return method


def method_label(method: str) -> str:
    return METHOD_LABELS.get(method, pub.method_label(method))


def contrast_label(method_a: str, method_b: str) -> str:
    return f"{method_label(method_a)} vs {method_label(method_b)}"


def parse_csv_list(raw: str | None, cast: Any = str) -> list[Any] | None:
    if raw is None:
        return None
    return [cast(item.strip()) for item in raw.split(",") if item.strip()]


def load_results(
    study_root: Path,
    protocol_id: str,
    *,
    environments: Iterable[int],
    batch_size: int = BATCH_SIZE,
) -> pd.DataFrame:
    """Load the exact canonical seed-level frame used by the paper plots."""
    frame = _build_frame(
        study_root,
        protocol_id,
        protocol_ids_by_rand={0.1: protocol_id},
        batch_size=batch_size,
        levels=list(environments),
    )
    if frame.is_empty():
        return pd.DataFrame()
    result = frame.to_pandas()
    result = result.rename(columns={"level": "environment", "plot_series": "series"})
    result["method"] = result.apply(
        lambda row: method_for_series(str(row["series"]), str(row["comparison"])), axis=1
    )
    result = result.rename(columns={"comparison": "checkpoint"})
    return result


def method_for_series(series: str, checkpoint: str) -> str:
    if series == "mercury":
        return "mercury"
    if series == "hybrid":
        return "mercury_initialised"
    if series == "tmgwr_hybrid":
        return "tmgwr" if checkpoint == "checkpoint_2000" else "tmgwr_initialised"
    return {
        "vanilla_cscg": "standard_cscg",
        "oracle_capacity": "oracle_capacity",
        "oracle_capacity_first_order": "oracle_capacity_first_order",
        "oracle_capacity_structural": "oracle_capacity_structural",
    }.get(series, series)


def aggregate_to_seed_level(
    results: pd.DataFrame,
    metrics: Iterable[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pool any repeated records exactly as the plotting pipeline does: by mean."""
    keys = ["environment", "checkpoint", "method", "seed"]
    duplicate_counts = results.groupby(keys, dropna=False).size().reset_index(name="row_count")
    duplicates = duplicate_counts.loc[duplicate_counts["row_count"] > 1].copy()
    columns = keys + list(metrics)
    seed_level = results[columns].groupby(keys, as_index=False, dropna=False).mean(numeric_only=True)
    return seed_level, duplicates


def expected_methods(checkpoint: str) -> list[str]:
    methods: list[str] = []
    for _, role_a, role_b in PLANNED_CONTRASTS:
        methods.extend(
            [resolve_checkpoint_method(role_a, checkpoint), resolve_checkpoint_method(role_b, checkpoint)]
        )
    return list(dict.fromkeys(methods))


def audit_evaluation_provenance(
    study_root: Path,
    protocol_id: str,
    *,
    environments: Iterable[int],
    checkpoints: Iterable[str],
    seeds: Iterable[int],
) -> list[ValidationFinding]:
    """Check that paired conditions use a common canonical held-out walk cache."""
    findings: list[ValidationFinding] = []
    manifest = load_study_manifest(study_root)
    series_sources = {
        "mercury": ("mercury_cscg_hybrid", None),
        "mercury_initialised": ("mercury_cscg_hybrid", None),
        "tmgwr": ("tmgwr_cscg_hybrid", None),
        "tmgwr_initialised": ("tmgwr_cscg_hybrid", None),
        "standard_cscg": ("cscg", "vanilla_cscg"),
        "oracle_capacity": ("cscg", "oracle_capacity"),
        "oracle_capacity_first_order": ("cscg", "oracle_capacity_first_order"),
        "oracle_capacity_structural": ("cscg", "oracle_capacity_structural"),
    }
    for environment in environments:
        for seed in seeds:
            for checkpoint in checkpoints:
                cache_ids: dict[str, str] = {}
                protocol_specs: dict[str, tuple[Any, Any]] = {}
                for method in expected_methods(checkpoint):
                    source_method, preset = series_sources[method]
                    run_root = manifest.get((source_method, int(environment), int(seed), 0.1, BATCH_SIZE, preset))
                    if run_root is None:
                        continue
                    nominal_step = CHECKPOINTS[checkpoint]["budget_step"] if checkpoint != "final" else None
                    lookup_method = source_method
                    if method == "mercury":
                        lookup_method = "mercury_cscg_hybrid_mercury"
                        nominal_step = 2_000
                    elif method == "tmgwr":
                        lookup_method = "tmgwr_cscg_hybrid_tmgwr"
                        nominal_step = 2_000
                    elif source_method in {"mercury_cscg_hybrid", "tmgwr_cscg_hybrid"}:
                        nominal_step, lookup_method = _record_lookup_for_comparison(
                            source_method, checkpoint, nominal_step
                        )
                    record = load_evaluation_step(
                        run_root, protocol_id, nominal_step, method=lookup_method
                    )
                    if record is None:
                        continue
                    cache_ids[method] = str(record.get("walk_cache_id"))
                    protocol = record.get("protocol") or {}
                    protocol_specs[method] = (
                        protocol.get("number_of_walks"), protocol.get("walk_length")
                    )
                if len(set(cache_ids.values())) > 1:
                    findings.append(
                        ValidationFinding(
                            "ERROR",
                            "walk_cache_mismatch",
                            f"environment={environment}, seed={seed}, checkpoint={checkpoint}: {cache_ids}",
                        )
                    )
                if len(set(protocol_specs.values())) > 1:
                    findings.append(
                        ValidationFinding(
                            "ERROR",
                            "evaluation_protocol_mismatch",
                            f"environment={environment}, seed={seed}, checkpoint={checkpoint}: {protocol_specs}",
                        )
                    )
    return findings


def validate_results(
    raw: pd.DataFrame,
    seed_level: pd.DataFrame,
    duplicates: pd.DataFrame,
    *,
    environments: list[int],
    checkpoints: list[str],
    metrics: list[str],
    expected_seeds: list[int],
    provenance_findings: list[ValidationFinding],
) -> tuple[str, list[ValidationFinding]]:
    findings = list(provenance_findings)
    lines = [
        "MERCURY planned paired analysis validation report",
        "=" * 50,
        "",
        f"Environments detected: {sorted(raw['environment'].unique().tolist()) if not raw.empty else []}",
        f"Methods detected: {sorted(raw['method'].unique().tolist()) if not raw.empty else []}",
        f"Checkpoints detected: {sorted(raw['checkpoint'].unique().tolist()) if not raw.empty else []}",
        f"Metrics detected: {[metric for metric in METRICS if metric in raw.columns]}",
        f"Metrics requested: {metrics}",
        f"Expected seeds: {expected_seeds}",
        f"Raw rows: {len(raw)}",
        f"Seed-level rows: {len(seed_level)}",
        f"Duplicate condition groups: {len(duplicates)}",
        "",
        "Seeds available per condition",
        "-" * 29,
    ]
    for environment in environments:
        for checkpoint in checkpoints:
            for method in expected_methods(checkpoint):
                subset = seed_level.loc[
                    (seed_level["environment"] == environment)
                    & (seed_level["checkpoint"] == checkpoint)
                    & (seed_level["method"] == method)
                ]
                seeds_found = sorted(int(seed) for seed in subset["seed"].tolist())
                lines.append(
                    f"environment={environment} ({ENVIRONMENTS[environment]}), checkpoint={checkpoint}, "
                    f"method={method}: n={len(seeds_found)}, seeds={seeds_found}"
                )
                missing = sorted(set(expected_seeds) - set(seeds_found))
                if missing:
                    findings.append(
                        ValidationFinding(
                            "ERROR",
                            "missing_condition_seeds",
                            f"environment={environment}, checkpoint={checkpoint}, method={method}: missing {missing}",
                        )
                    )
    lines.extend(["", "Paired seeds per requested comparison", "-" * 37])
    for metric in metrics:
        for contrast_id, role_a, role_b in PLANNED_CONTRASTS:
            for environment in environments:
                for checkpoint in checkpoints:
                    method_a = resolve_checkpoint_method(role_a, checkpoint)
                    method_b = resolve_checkpoint_method(role_b, checkpoint)
                    pairs, audit = build_paired_samples(
                        seed_level,
                        metric=metric,
                        environment=environment,
                        checkpoint=checkpoint,
                        method_a=method_a,
                        method_b=method_b,
                        expected_seeds=expected_seeds,
                    )
                    audit_detail = ", ".join(f"{key}={value}" for key, value in audit.items() if value)
                    lines.append(
                        f"metric={metric}, contrast={contrast_id}, environment={environment}, "
                        f"checkpoint={checkpoint}: n_pairs={len(pairs)}"
                        + (f", {audit_detail}" if audit_detail else "")
                    )
                    if len(pairs) != len(expected_seeds):
                        findings.append(
                            ValidationFinding(
                                "WARNING",
                                "unexpected_pair_count",
                                f"metric={metric}, contrast={contrast_id}, environment={environment}, "
                                f"checkpoint={checkpoint}: n_pairs={len(pairs)}, expected={len(expected_seeds)}"
                                + (f", {audit_detail}" if audit_detail else ""),
                            )
                        )
    if not duplicates.empty:
        for row in duplicates.to_dict("records"):
            findings.append(ValidationFinding("ERROR", "duplicate_rows", str(row)))
    for metric in metrics:
        if metric not in seed_level:
            findings.append(ValidationFinding("ERROR", "missing_metric", metric))
            continue
        bad = seed_level.loc[~np.isfinite(seed_level[metric].to_numpy(dtype=float))]
        for row in bad[["environment", "checkpoint", "method", "seed"]].to_dict("records"):
            findings.append(
                ValidationFinding("ERROR", "nonfinite_value", f"metric={metric}: {row}")
            )
    nonfinite_count = sum(item.category == "nonfinite_value" for item in findings)
    missing_count = sum(item.category == "missing_condition_seeds" for item in findings)
    lines.extend(
        [
            "",
            "Validation totals",
            "-" * 17,
            f"Missing condition/seed findings: {missing_count}",
            f"Non-finite metric values: {nonfinite_count}",
            f"Duplicate condition groups: {len(duplicates)}",
            f"Walk-cache mismatches: {sum(item.category == 'walk_cache_mismatch' for item in findings)}",
            f"Evaluation-protocol mismatches: {sum(item.category == 'evaluation_protocol_mismatch' for item in findings)}",
        ]
    )
    lines.extend(["", "Findings", "-" * 8])
    if findings:
        lines.extend(f"[{item.severity}] {item.category}: {item.message}" for item in findings)
    else:
        lines.append("No validation warnings or errors.")
    return "\n".join(lines) + "\n", findings


def _seed_text(values: Iterable[int]) -> str:
    return ";".join(str(int(value)) for value in sorted(values))


def build_paired_samples(
    seed_level: pd.DataFrame,
    *,
    metric: str,
    environment: int,
    checkpoint: str,
    method_a: str,
    method_b: str,
    expected_seeds: list[int],
) -> tuple[pd.DataFrame, dict[str, str]]:
    columns = ["seed", metric]
    a = seed_level.loc[
        (seed_level["environment"] == environment)
        & (seed_level["checkpoint"] == checkpoint)
        & (seed_level["method"] == method_a),
        columns,
    ].rename(columns={metric: "value_a"})
    b = seed_level.loc[
        (seed_level["environment"] == environment)
        & (seed_level["checkpoint"] == checkpoint)
        & (seed_level["method"] == method_b),
        columns,
    ].rename(columns={metric: "value_b"})
    seeds_a = set(int(value) for value in a["seed"])
    seeds_b = set(int(value) for value in b["seed"])
    nonfinite_a = set(int(value) for value in a.loc[~np.isfinite(a["value_a"]), "seed"])
    nonfinite_b = set(int(value) for value in b.loc[~np.isfinite(b["value_b"]), "seed"])
    merged = a.merge(b, on="seed", how="inner", validate="one_to_one")
    merged = merged.loc[
        np.isfinite(merged["value_a"].to_numpy(dtype=float))
        & np.isfinite(merged["value_b"].to_numpy(dtype=float))
    ].copy()
    merged["difference"] = merged["value_a"] - merged["value_b"]
    merged = merged.sort_values("seed").reset_index(drop=True)
    audit = {
        "missing_seeds_a": _seed_text(set(expected_seeds) - seeds_a),
        "missing_seeds_b": _seed_text(set(expected_seeds) - seeds_b),
        "unpaired_seeds_a": _seed_text(seeds_a - seeds_b),
        "unpaired_seeds_b": _seed_text(seeds_b - seeds_a),
        "nonfinite_seeds_a": _seed_text(nonfinite_a),
        "nonfinite_seeds_b": _seed_text(nonfinite_b),
    }
    return merged, audit


def compute_paired_contrast(values_a: np.ndarray, values_b: np.ndarray) -> dict[str, Any]:
    differences = np.asarray(values_a, dtype=float) - np.asarray(values_b, dtype=float)
    n = int(differences.size)
    result: dict[str, Any] = {
        "n_pairs": n,
        "mean_a": float(np.mean(values_a)) if n else math.nan,
        "mean_b": float(np.mean(values_b)) if n else math.nan,
        "mean_difference": float(np.mean(differences)) if n else math.nan,
        "sd_difference": float(np.std(differences, ddof=1)) if n > 1 else math.nan,
    }
    result["se_difference"] = (
        result["sd_difference"] / math.sqrt(n) if n > 1 else math.nan
    )
    if n < 2:
        result.update(
            ci95_low=math.nan,
            ci95_high=math.nan,
            t_statistic=math.nan,
            df=float(n - 1) if n else math.nan,
            p_raw=math.nan,
            cohens_dz=math.nan,
        )
        return result
    if result["sd_difference"] == 0.0:
        mean = result["mean_difference"]
        result.update(
            ci95_low=mean,
            ci95_high=mean,
            t_statistic=0.0 if mean == 0.0 else math.copysign(math.inf, mean),
            df=float(n - 1),
            p_raw=1.0 if mean == 0.0 else 0.0,
            cohens_dz=0.0 if mean == 0.0 else math.copysign(math.inf, mean),
        )
        return result
    test = stats.ttest_rel(values_a, values_b, alternative="two-sided")
    margin = float(stats.t.ppf(0.975, n - 1) * result["se_difference"])
    result.update(
        ci95_low=result["mean_difference"] - margin,
        ci95_high=result["mean_difference"] + margin,
        t_statistic=float(test.statistic),
        df=float(n - 1),
        p_raw=float(test.pvalue),
        cohens_dz=result["mean_difference"] / result["sd_difference"],
    )
    return result


def compute_all_contrasts(
    seed_level: pd.DataFrame,
    *,
    metrics: list[str],
    environments: list[int],
    checkpoints: list[str],
    expected_seeds: list[int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summaries: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    for metric in metrics:
        for contrast_id, role_a, role_b in PLANNED_CONTRASTS:
            for environment in environments:
                for checkpoint in checkpoints:
                    method_a = resolve_checkpoint_method(role_a, checkpoint)
                    method_b = resolve_checkpoint_method(role_b, checkpoint)
                    pairs, audit = build_paired_samples(
                        seed_level,
                        metric=metric,
                        environment=environment,
                        checkpoint=checkpoint,
                        method_a=method_a,
                        method_b=method_b,
                        expected_seeds=expected_seeds,
                    )
                    stats_row = compute_paired_contrast(
                        pairs["value_a"].to_numpy(dtype=float),
                        pairs["value_b"].to_numpy(dtype=float),
                    )
                    reasons = []
                    if stats_row["n_pairs"] < 2:
                        reasons.append("fewer than two finite paired seeds")
                    for key, value in audit.items():
                        if value:
                            reasons.append(f"{key}={value}")
                    label = contrast_label(method_a, method_b)
                    row = {
                        "metric": metric,
                        "metric_label": METRICS[metric]["label"],
                        "environment": environment,
                        "environment_label": ENVIRONMENTS[environment],
                        "checkpoint": checkpoint,
                        "checkpoint_label": CHECKPOINTS[checkpoint]["label"],
                        "checkpoint_step": CHECKPOINTS[checkpoint]["budget_step"],
                        "contrast_id": contrast_id,
                        "contrast_label": label,
                        "method_a": method_a,
                        "method_b": method_b,
                        **stats_row,
                        "p_holm": math.nan,
                        "significant_holm_0_05": False,
                        "holm_family": f"{metric}::{contrast_id}",
                        "holm_family_n_tests": 0,
                        "expected_n_pairs": len(expected_seeds),
                        "n_pairs_is_expected": stats_row["n_pairs"] == len(expected_seeds),
                        "available": stats_row["n_pairs"] >= 2,
                        "unavailable_reason": "; ".join(reasons),
                        **audit,
                    }
                    summaries.append(row)
                    for pair in pairs.to_dict("records"):
                        pair_rows.append(
                            {
                                "metric": metric,
                                "environment": environment,
                                "checkpoint": checkpoint,
                                "contrast_id": contrast_id,
                                "contrast_label": label,
                                "method_a": method_a,
                                "method_b": method_b,
                                **pair,
                            }
                        )
    return pd.DataFrame(summaries), pd.DataFrame(pair_rows)


def apply_holm_correction(results: pd.DataFrame) -> pd.DataFrame:
    adjusted = results.copy()
    for family, indices in adjusted.groupby("holm_family", sort=False).groups.items():
        valid_indices = [index for index in indices if np.isfinite(adjusted.at[index, "p_raw"])]
        adjusted.loc[list(indices), "holm_family_n_tests"] = len(valid_indices)
        if not valid_indices:
            continue
        reject, p_holm, _, _ = multipletests(
            adjusted.loc[valid_indices, "p_raw"].to_numpy(dtype=float), method="holm"
        )
        adjusted.loc[valid_indices, "p_holm"] = p_holm
        adjusted.loc[valid_indices, "significant_holm_0_05"] = reject
    return adjusted


def format_p_value(value: Any) -> str:
    if value is None or not np.isfinite(float(value)):
        return "NA"
    return "<0.001" if float(value) < 0.001 else f"{float(value):.3f}"


def publication_frame(results: pd.DataFrame) -> pd.DataFrame:
    output = results.copy()
    numeric_3dp = [
        "mean_a",
        "mean_b",
        "mean_difference",
        "ci95_low",
        "ci95_high",
        "sd_difference",
        "se_difference",
        "t_statistic",
        "cohens_dz",
    ]
    for column in numeric_3dp:
        output[column] = output[column].map(
            lambda value: "NA" if not np.isfinite(float(value)) else f"{float(value):.3f}"
        )
    output["df"] = output["df"].map(
        lambda value: "NA" if not np.isfinite(float(value)) else str(int(value))
    )
    output["p_raw"] = output["p_raw"].map(format_p_value)
    output["p_holm"] = output["p_holm"].map(format_p_value)
    return output


def latex_escape(value: Any) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in text)


def _latex_p_value(value: Any) -> str:
    if value is None or not np.isfinite(float(value)):
        return r"$\mathrm{NA}$"
    if float(value) < 0.001:
        return r"$<0.001$"
    return rf"${float(value):.3f}$"


def _latex_effect_cell(row: pd.Series) -> str:
    if not bool(row["available"]):
        return "Unavailable"
    delta = f"{row['mean_difference']:.3f}"
    interval = rf"[{row['ci95_low']:.3f},\,{row['ci95_high']:.3f}]"
    significant = np.isfinite(float(row["p_holm"])) and float(row["p_holm"]) < 0.05
    if significant:
        return rf"\textbf{{{delta}}} \textbf{{{interval}}}"
    return f"{delta} {interval}"


def _latex_p_cell(row: pd.Series) -> str:
    if not bool(row["available"]):
        return _latex_p_value(math.nan)
    value = _latex_p_value(row["p_holm"])
    significant = np.isfinite(float(row["p_holm"])) and float(row["p_holm"]) < 0.05
    return rf"\textbf{{{value}}}" if significant else value


def _latex_delta_cell(row: pd.Series) -> str:
    if not bool(row["available"]):
        return "Unavailable"
    difference = float(row["mean_difference"])
    value = f"{difference:+.3f}" if difference != 0.0 else "0.000"
    significant = np.isfinite(float(row["p_holm"])) and float(row["p_holm"]) < 0.05
    formatted = rf"\textbf{{{value}}}" if significant else value
    shade = 12 if significant else 4
    if difference > 0.0:
        return rf"\colorbox{{green!{shade}}}{{\strut\makebox[3.4em][r]{{{formatted}}}}}"
    if difference < 0.0:
        return rf"\colorbox{{red!{shade}}}{{\strut\makebox[3.4em][r]{{{formatted}}}}}"
    return formatted


LATEX_PRIMARY_METRIC = "latent_action_conditioned_mean_total_variation_similarity"
LATEX_NO_SELF_LOOP_METRIC = (
    "latent_action_conditioned_non_self_loop_total_variation_similarity"
)
LATEX_LATENT_METRIC = "latent_precision"
LATEX_CONTRAST_LABELS = {
    "mercury_vs_standard": "MERCURY vs Standard",
    "tmgwr_vs_standard": "TMGWR vs Standard",
    "mercury_vs_tmgwr": "MERCURY vs TMGWR",
    "oracle_capacity_vs_standard": "Oracle capacity vs Standard",
    "oracle_first_order_vs_capacity": "Oracle first-order vs capacity",
    "oracle_structure_vs_first_order": "Oracle structure vs first-order",
    "mercury_vs_oracle_capacity": "MERCURY vs Oracle capacity",
    "mercury_vs_oracle_first_order": "MERCURY vs Oracle first-order",
    "mercury_vs_oracle_structure": "MERCURY vs Oracle structure",
}

LATEX_ENVIRONMENT_LABELS = {
    13: r"$3\times3$",
    21: r"$1\times4$",
    20: r"$1\times5$",
    22: r"$3\times4$",
}


def _latex_primary_table(
    results: pd.DataFrame,
    *,
    metric: str,
    caption: str,
    label: str,
    contrast_ids: Iterable[str] | None = None,
) -> list[str]:
    requested_contrasts = set(contrast_ids or LATEX_CONTRAST_LABELS)
    rows = results.loc[
        (results["metric"] == metric)
        & results["contrast_id"].isin(requested_contrasts)
    ]
    output = [
        r"\begin{table*}[!t]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\renewcommand{\arraystretch}{1.10}",
        r"\begin{tabular}{@{}llcc cc cc@{}}",
        r"\toprule",
        (
            r"& & \multicolumn{2}{c}{Single pass} & "
            r"\multicolumn{2}{c}{Two-pass refinement} & "
            r"\multicolumn{2}{c}{Extensive refinement} \\"
        ),
        r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}",
        (
            r"Comparison & Env. & $\Delta$ [95\% CI] & $p_{\mathrm{Holm}}$ & "
            r"$\Delta$ [95\% CI] & $p_{\mathrm{Holm}}$ & "
            r"$\Delta$ [95\% CI] & $p_{\mathrm{Holm}}$ \\"
        ),
        r"\midrule",
    ]
    contrast_order = [
        contrast_id
        for contrast_id, _, _ in PLANNED_CONTRASTS
        if contrast_id in requested_contrasts
    ]
    for contrast_index, contrast_id in enumerate(contrast_order):
        for environment_index, environment in enumerate(ENVIRONMENTS):
            source = rows.loc[
                (rows["contrast_id"] == contrast_id)
                & (rows["environment"] == environment)
            ]
            cells: list[str] = []
            for checkpoint in CHECKPOINTS:
                cell = source.loc[source["checkpoint"] == checkpoint]
                if cell.empty:
                    cells.extend(["Unavailable", _latex_p_value(math.nan)])
                else:
                    cells.extend(
                        [
                            _latex_effect_cell(cell.iloc[0]),
                            _latex_p_cell(cell.iloc[0]),
                        ]
                    )
            output.append(
                " & ".join(
                    [
                        latex_escape(LATEX_CONTRAST_LABELS[contrast_id])
                        if environment_index == 0
                        else "",
                        LATEX_ENVIRONMENT_LABELS[environment],
                        *cells,
                    ]
                )
                + r" \\"
            )
        if contrast_index < len(contrast_order) - 1:
            output.append(r"\addlinespace[2pt]")
    output.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\vspace{2pt}",
            (
                r"\parbox{\textwidth}{\footnotesize\textit{Note:} Entries report the paired mean "
                r"difference $\Delta=A-B$ and its 95\% confidence interval across 30 training seeds. "
                r"Results with $p_{\mathrm{Holm}}<0.05$ are shown in bold. Positive differences favour "
                r"the first-named condition. At the single-pass checkpoint, MERCURY and TMGWR "
                r"denote the standalone learners; at subsequent checkpoints they denote the "
                r"corresponding CSCG initialisation conditions.}"
            ),
            r"\end{table*}",
            "",
        ]
    )
    return output


def _latex_main_body_table(results: pd.DataFrame) -> list[str]:
    """Render the condensed MERCURY-centred action-similarity results table."""
    rows = results.loc[
        (results["metric"] == LATEX_PRIMARY_METRIC)
        & results["contrast_id"].isin(MAIN_BODY_CONTRAST_IDS)
    ]
    output = [
        r"\begin{table}[!t]",
        r"\centering",
        (
            r"\caption{MERCURY-centred paired comparisons for action-conditioned transition "
            r"similarity across training checkpoints.}"
        ),
        r"\label{tab:planned_contrasts_action_tv_main_body}",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{7pt}",
        r"\setlength{\fboxsep}{1.5pt}",
        r"\renewcommand{\arraystretch}{1.05}",
        r"\begin{tabular}{@{}llccc@{}}",
        r"\toprule",
        (
            r"Comparator & Env. & Single pass $\Delta$ & "
            r"Two-pass $\Delta$ & Extensive $\Delta$ \\"
        ),
        r"\midrule",
    ]
    for contrast_index, contrast_id in enumerate(MAIN_BODY_CONTRAST_IDS):
        for environment_index, environment in enumerate(ENVIRONMENTS):
            source = rows.loc[
                (rows["contrast_id"] == contrast_id)
                & (rows["environment"] == environment)
            ]
            cells: list[str] = []
            for checkpoint in CHECKPOINTS:
                cell = source.loc[source["checkpoint"] == checkpoint]
                cells.append(
                    "Unavailable" if cell.empty else _latex_delta_cell(cell.iloc[0])
                )
            output.append(
                " & ".join(
                    [
                        latex_escape(MAIN_BODY_COMPARATOR_LABELS[contrast_id])
                        if environment_index == 0
                        else "",
                        LATEX_ENVIRONMENT_LABELS[environment],
                        *cells,
                    ]
                )
                + r" \\"
            )
        if contrast_index < len(MAIN_BODY_CONTRAST_IDS) - 1:
            output.append(r"\addlinespace[2pt]")
    output.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\vspace{2pt}",
            (
                r"\parbox{\columnwidth}{\footnotesize\textit{Note:} "
                r"$\Delta=\mathrm{MERCURY}-\mathrm{comparator}$; positive values favour MERCURY. "
                r"Green and red shading denote positive and negative paired differences, respectively. "
                r"Lighter shading denotes comparisons that are not significant after Holm correction; "
                r"significant comparisons are additionally shown in bold. Confidence intervals, exact "
                r"adjusted $p$-values, and all prespecified contrasts are reported in Appendix "
                r"Table~\ref{tab:planned_contrasts_action_tv}.}"
            ),
            r"\end{table}",
            "",
        ]
    )
    return output


def _latex_main_body_transposed_table(
    results: pd.DataFrame,
    *,
    metric: str,
    caption: str,
    label: str,
    appendix_label: str,
) -> list[str]:
    """Render MERCURY-centred effects with checkpoints as row groups."""
    rows = results.loc[
        (results["metric"] == metric)
        & results["contrast_id"].isin(MAIN_BODY_CONTRAST_IDS)
    ]
    comparator_headers = {
        "mercury_vs_standard": r"\shortstack{Standard\\CSCG}",
        "mercury_vs_tmgwr": "TMGWR",
        "mercury_vs_oracle_capacity": "Oracle-C",
        "mercury_vs_oracle_first_order": r"\shortstack{Oracle-C\\+1st}",
        "mercury_vs_oracle_structure": r"\shortstack{Oracle-C\\+Struct}",
    }
    output = [
        r"\begin{table}[!t]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{2pt}",
        r"\setlength{\fboxsep}{1.5pt}",
        r"\renewcommand{\arraystretch}{1.05}",
        r"\begin{tabular}{@{}llccccc@{}}",
        r"\toprule",
        " & ".join(
            [
                "Checkpoint",
                "Env.",
                *(comparator_headers[contrast_id] for contrast_id in MAIN_BODY_CONTRAST_IDS),
            ]
        )
        + r" \\",
        r"\midrule",
    ]
    checkpoint_labels = {
        "checkpoint_2000": r"Single pass $\Delta$",
        "checkpoint_4000": r"Two-pass $\Delta$",
        "final": r"Extensive $\Delta$",
    }
    for checkpoint_index, checkpoint in enumerate(CHECKPOINTS):
        for environment_index, environment in enumerate(ENVIRONMENTS):
            cells: list[str] = []
            for contrast_id in MAIN_BODY_CONTRAST_IDS:
                cell = rows.loc[
                    (rows["contrast_id"] == contrast_id)
                    & (rows["environment"] == environment)
                    & (rows["checkpoint"] == checkpoint)
                ]
                cells.append("Unavailable" if cell.empty else _latex_delta_cell(cell.iloc[0]))
            output.append(
                " & ".join(
                    [
                        checkpoint_labels[checkpoint] if environment_index == 0 else "",
                        LATEX_ENVIRONMENT_LABELS[environment],
                        *cells,
                    ]
                )
                + r" \\"
            )
        if checkpoint_index < len(CHECKPOINTS) - 1:
            output.append(r"\addlinespace[3pt]")
    output.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\vspace{2pt}",
            (
                r"\parbox{\columnwidth}{\footnotesize\textit{Note:} "
                r"$\Delta=\mathrm{MERCURY}-\mathrm{comparator}$; positive values favour MERCURY. "
                r"Green and red shading denote positive and negative paired differences, respectively. "
                r"Lighter shading denotes comparisons that are not significant after Holm correction; "
                r"significant comparisons are additionally shown in bold. Confidence intervals, exact "
                r"adjusted $p$-values, and all prespecified contrasts are reported in Appendix "
                rf"Table~\ref{{{appendix_label}}}.}}"
            ),
            r"\end{table}",
            "",
        ]
    )
    return output


def write_latex_tables(
    results: pd.DataFrame, output_dir: Path
) -> tuple[Path, Path, Path, Path]:
    appendix_path = output_dir / "planned_contrasts_appendix.tex"
    summary_path = output_dir / "planned_contrasts_summary.tex"
    main_body_path = output_dir / "planned_contrasts_main_body.tex"
    main_body_transposed_path = output_dir / "planned_contrasts_main_body_transposed.tex"
    action_caption = (
        r"All prespecified paired comparisons for action-conditioned transition similarity with "
        r"self-loops included. Effects are "
        r"reported as paired mean differences with 95\% confidence intervals across 30 training seeds."
    )
    no_self_loop_caption = (
        r"All prespecified paired comparisons for conditional non-self-loop action-conditioned "
        r"transition similarity. Effects are reported as paired mean differences with 95\% confidence "
        r"intervals across 30 training seeds."
    )
    latent_caption = (
        r"All prespecified paired comparisons for latent purity, reported using the same convention as "
        r"Table~\ref{tab:planned_contrasts_action_tv}. These results provide the corresponding "
        r"latent-state comparison for the transition-structure analysis."
    )
    header = [
        "% Generated by planned_paired_contrasts.py",
        "% Requires:",
        "% \\usepackage{booktabs}",
        "% \\usepackage[table]{xcolor}",
        "",
    ]
    action_table = _latex_primary_table(
        results,
        metric=LATEX_PRIMARY_METRIC,
        caption=action_caption,
        label="tab:planned_contrasts_action_tv",
    )
    no_self_loop_table = _latex_primary_table(
        results,
        metric=LATEX_NO_SELF_LOOP_METRIC,
        caption=no_self_loop_caption,
        label="tab:planned_contrasts_action_tv_no_self_loops",
    )
    appendix = [
        *header,
        *action_table,
        *no_self_loop_table,
        *_latex_primary_table(
            results,
            metric=LATEX_LATENT_METRIC,
            caption=latent_caption,
            label="tab:planned_contrasts_latent_purity",
        ),
    ]
    appendix_path.write_text("\n".join(appendix), encoding="utf-8")
    summary = [*header, *action_table]
    summary_path.write_text("\n".join(summary), encoding="utf-8")
    main_body = [*header, *_latex_main_body_table(results)]
    main_body_path.write_text("\n".join(main_body), encoding="utf-8")
    main_body_transposed = [
        *header,
        *_latex_main_body_transposed_table(
            results,
            metric=LATEX_PRIMARY_METRIC,
            caption=(
                r"MERCURY-centred paired comparisons for action-conditioned transition "
                r"similarity with self-loops included across training checkpoints."
            ),
            label="tab:planned_contrasts_action_tv_main_body_transposed",
            appendix_label="tab:planned_contrasts_action_tv",
        ),
        *_latex_main_body_transposed_table(
            results,
            metric=LATEX_NO_SELF_LOOP_METRIC,
            caption=(
                r"MERCURY-centred paired comparisons for conditional non-self-loop "
                r"action-conditioned transition similarity across training checkpoints."
            ),
            label="tab:planned_contrasts_action_tv_no_self_loops_main_body_transposed",
            appendix_label="tab:planned_contrasts_action_tv_no_self_loops",
        ),
    ]
    main_body_transposed_path.write_text(
        "\n".join(main_body_transposed), encoding="utf-8"
    )
    return appendix_path, summary_path, main_body_path, main_body_transposed_path


def metric_slug(metric: str) -> str:
    return metric.removeprefix("latent_").replace("_similarity", "_sim")


def plot_contrast_heatmaps(results: pd.DataFrame, output_dir: Path) -> list[Path]:
    finite_effects = results.loc[results["available"], "mean_difference"].to_numpy(dtype=float)
    global_limit = float(np.max(np.abs(finite_effects))) if finite_effects.size else 1.0
    global_limit = max(global_limit, 1e-12)
    paths: list[Path] = []
    contrast_order = [item[0] for item in PLANNED_CONTRASTS]
    columns = [(environment, checkpoint) for environment in ENVIRONMENTS for checkpoint in CHECKPOINTS]
    for metric in results["metric"].drop_duplicates():
        source = results.loc[results["metric"] == metric].set_index(
            ["contrast_id", "environment", "checkpoint"]
        )
        matrix = np.full((len(contrast_order), len(columns)), np.nan)
        significant = np.zeros_like(matrix, dtype=bool)
        row_labels: list[str] = []
        for row_index, contrast_id in enumerate(contrast_order):
            contrast_rows = results.loc[
                (results["metric"] == metric) & (results["contrast_id"] == contrast_id)
            ]
            row_labels.append(str(contrast_rows.iloc[0]["contrast_label"]))
            for column_index, (environment, checkpoint) in enumerate(columns):
                key = (contrast_id, environment, checkpoint)
                if key not in source.index:
                    continue
                row = source.loc[key]
                if bool(row["available"]):
                    matrix[row_index, column_index] = float(row["mean_difference"])
                    significant[row_index, column_index] = bool(row["significant_holm_0_05"])
        figure, axis = plt.subplots(figsize=(15.5, 7.2), constrained_layout=True)
        masked = np.ma.masked_invalid(matrix)
        image = axis.imshow(
            masked,
            cmap="RdBu_r",
            norm=TwoSlopeNorm(vmin=-global_limit, vcenter=0.0, vmax=global_limit),
            aspect="auto",
        )
        axis.set_xticks(range(len(columns)))
        axis.set_xticklabels(
            [f"{ENVIRONMENTS[environment]}\n{CHECKPOINTS[checkpoint]['short_label']}" for environment, checkpoint in columns],
            rotation=45,
            ha="right",
        )
        axis.set_yticks(range(len(row_labels)))
        axis.set_yticklabels(row_labels)
        for row_index in range(matrix.shape[0]):
            for column_index in range(matrix.shape[1]):
                if np.isfinite(matrix[row_index, column_index]):
                    marker = "\n●" if significant[row_index, column_index] else ""
                    axis.text(
                        column_index,
                        row_index,
                        f"{matrix[row_index, column_index]:.3f}{marker}",
                        ha="center",
                        va="center",
                        fontsize=7,
                        color="black",
                    )
                else:
                    axis.text(column_index, row_index, "NA", ha="center", va="center", fontsize=7)
        axis.set_title(
            f"{METRICS[metric]['label']}: paired mean differences\n"
            "Positive values favour the first-named condition; ● denotes Holm-adjusted p < 0.05"
        )
        colourbar = figure.colorbar(image, ax=axis, shrink=0.82)
        colourbar.set_label("Mean paired difference (A − B)")
        stem = output_dir / f"contrast_heatmap_{metric_slug(metric)}"
        for suffix, kwargs in ((".pdf", {}), (".png", {"dpi": 300})):
            path = stem.with_suffix(suffix)
            figure.savefig(path, bbox_inches="tight", **kwargs)
            paths.append(path)
        plt.close(figure)
    return paths


def write_statistical_summary(results: pd.DataFrame, output_dir: Path) -> Path:
    path = output_dir / "statistical_summary.txt"
    lines = [
        "Descriptive summary of planned paired contrasts",
        "=" * 48,
        "Positive effects favour the first-named condition. Counts are descriptive and are not claims of superiority.",
        "",
    ]
    for metric in results["metric"].drop_duplicates():
        lines.extend([METRICS[metric]["label"], "-" * len(METRICS[metric]["label"])])
        metric_rows = results.loc[results["metric"] == metric]
        for contrast_id, _, _ in PLANNED_CONTRASTS:
            rows = metric_rows.loc[
                (metric_rows["contrast_id"] == contrast_id) & metric_rows["available"]
            ]
            if rows.empty:
                lines.append(f"{contrast_id}: no available comparisons")
                continue
            effects = rows["mean_difference"].to_numpy(dtype=float)
            largest = rows.iloc[int(np.argmax(np.abs(effects)))]
            lines.append(
                f"{largest['contrast_label']}: available={len(rows)}, positive={int(np.sum(effects > 0))}, "
                f"negative={int(np.sum(effects < 0))}, zero={int(np.sum(effects == 0))}, "
                f"Holm p<0.05={int(rows['significant_holm_0_05'].sum())}, "
                f"effect range=[{effects.min():.6g}, {effects.max():.6g}], "
                f"largest absolute effect={largest['mean_difference']:.6g} at "
                f"{largest['environment_label']} / {largest['checkpoint_label']}"
            )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_methods_snippet(output_dir: Path) -> Path:
    path = output_dir / "statistical_methods_snippet.tex"
    text = (
        "The independent training seed was treated as the unit of analysis. "
        "Prespecified method contrasts were evaluated using two-sided paired $t$-tests across "
        "the 30 training seeds, with methods paired by seed and evaluated on common held-out "
        "trajectories. For each comparison, we report the mean paired difference and its two-sided "
        "95\\% confidence interval. To account for evaluation of the same prespecified contrast "
        "across the four environments and three training checkpoints, $p$-values within each "
        "metric-specific contrast family (12 tests) were adjusted using the Holm procedure. "
        "Held-out evaluation trajectories were pooled within training seed and were not treated as "
        "independent experimental replicates.\n"
    )
    path.write_text(text, encoding="utf-8")
    return path


def _contrast_result_rows(results: pd.DataFrame, contrast_id: str) -> pd.DataFrame:
    return results.loc[
        (results["metric"] == LATEX_PRIMARY_METRIC)
        & (results["contrast_id"] == contrast_id)
        & results["available"]
    ]


def _delta_range_text(rows: pd.DataFrame) -> str:
    return (
        rf"$\Delta={rows['mean_difference'].min():.3f}$--"
        rf"${rows['mean_difference'].max():.3f}$"
    )


def _holm_count_text(significant: int, total: int) -> str:
    if significant == total:
        return f"all {total} comparisons remained significant after Holm correction"
    if significant == 0:
        return f"none of the {total} comparisons was significant after Holm correction"
    return f"{significant} of {total} comparisons were significant after Holm correction"


def write_results_summary_snippet(results: pd.DataFrame, output_dir: Path) -> Path:
    path = output_dir / "statistical_results_summary.tex"
    standard = _contrast_result_rows(results, "mercury_vs_standard")
    tmgwr = _contrast_result_rows(results, "mercury_vs_tmgwr")
    oracle_capacity = _contrast_result_rows(results, "mercury_vs_oracle_capacity")
    oracle_first_order = _contrast_result_rows(results, "mercury_vs_oracle_first_order")
    oracle_structure = _contrast_result_rows(results, "mercury_vs_oracle_structure")

    def counts(rows: pd.DataFrame) -> tuple[int, int, int, int]:
        return (
            len(rows),
            int((rows["mean_difference"] > 0).sum()),
            int((rows["mean_difference"] < 0).sum()),
            int(rows["significant_holm_0_05"].sum()),
        )

    standard_n, standard_positive, _, standard_significant = counts(standard)
    tmgwr_n, tmgwr_positive, _, tmgwr_significant = counts(tmgwr)
    capacity_n, capacity_positive, _, capacity_significant = counts(oracle_capacity)
    first_order_n, first_order_positive, _, first_order_significant = counts(
        oracle_first_order
    )
    structure_n, _, structure_negative, structure_significant = counts(oracle_structure)

    standard_direction = (
        f"across all {standard_n} environment--checkpoint combinations"
        if standard_positive == standard_n
        else f"in {standard_positive} of {standard_n} environment--checkpoint combinations"
    )
    tmgwr_direction = (
        f"across all {tmgwr_n} environment--checkpoint combinations"
        if tmgwr_positive == tmgwr_n
        else f"in {tmgwr_positive} of {tmgwr_n} environment--checkpoint combinations"
    )
    structure_direction = (
        f"negative across all {structure_n} environment--checkpoint combinations"
        if structure_negative == structure_n
        else f"negative in {structure_negative} of {structure_n} environment--checkpoint combinations"
    )
    sentences = [
        (
            "MERCURY-derived conditions showed higher action-conditioned transition similarity "
            f"than Standard CSCG {standard_direction}, with paired differences ranging from "
            f"{_delta_range_text(standard)}; "
            f"{_holm_count_text(standard_significant, standard_n)}."
        ),
        (
            "MERCURY-derived conditions likewise had positive paired differences relative to "
            f"TMGWR-derived conditions {tmgwr_direction}, ranging from "
            f"{_delta_range_text(tmgwr)}; {_holm_count_text(tmgwr_significant, tmgwr_n)}."
        ),
        (
            f"Relative to Oracle capacity, paired differences were positive in {capacity_positive} "
            f"of {capacity_n} environment--checkpoint combinations and ranged from "
            f"{_delta_range_text(oracle_capacity)}; "
            f"{_holm_count_text(capacity_significant, capacity_n)}."
        ),
        (
            "Relative to Oracle capacity with first-order structure, paired differences were "
            f"positive in {first_order_positive} of {first_order_n} environment--checkpoint "
            f"combinations and ranged from {_delta_range_text(oracle_first_order)}; "
            f"{_holm_count_text(first_order_significant, first_order_n)}."
        ),
        (
            "The contrast with Oracle structure defines the remaining gap to privileged "
            "ground-truth action-conditioned structure: paired differences were "
            f"{structure_direction}, ranged from {_delta_range_text(oracle_structure)}, and "
            f"{_holm_count_text(structure_significant, structure_n)}."
        ),
    ]
    path.write_text("\n\n".join(sentences) + "\n", encoding="utf-8")
    return path


def write_csv_outputs(
    results: pd.DataFrame, pairs: pd.DataFrame, output_dir: Path
) -> tuple[Path, Path, Path]:
    full_path = output_dir / "planned_contrasts_full.csv"
    publication_path = output_dir / "planned_contrasts_publication.csv"
    pairs_path = output_dir / "planned_contrasts_seed_pairs.csv"
    results[FULL_COLUMNS].to_csv(full_path, index=False, float_format="%.17g")
    publication_frame(results[FULL_COLUMNS]).to_csv(publication_path, index=False)
    pairs.to_csv(pairs_path, index=False, float_format="%.17g")
    return full_path, publication_path, pairs_path


def print_console_summary(
    results: pd.DataFrame,
    *,
    metrics: list[str],
    environments: list[int],
    checkpoints: list[str],
    output_paths: list[Path],
    findings: list[ValidationFinding],
) -> None:
    print("\nStatistical analysis complete\n")
    print(f"Metrics: {len(metrics)}")
    print(f"Environments: {len(environments)}")
    print(f"Checkpoints: {len(checkpoints)}")
    print(f"Requested contrasts: {len(PLANNED_CONTRASTS)}")
    print(f"Valid paired comparisons: {int(results['available'].sum())}")
    print(f"Expected seed pairs: {len(EXPECTED_SEEDS)}")
    print("\nOutputs:")
    for path in output_paths:
        print(f"  {path}")
    warnings = results.loc[(results["n_pairs"] != len(EXPECTED_SEEDS)) | ~results["available"]]
    print("\nValidation warnings:")
    if warnings.empty and not findings:
        print("  None")
    for _, row in warnings.iterrows():
        print(
            f"  {row['metric']} | {row['contrast_id']} | environment={row['environment']} | "
            f"checkpoint={row['checkpoint']} | n={row['n_pairs']} | {row['unavailable_reason']}"
        )
    for finding in findings:
        print(f"  [{finding.severity}] {finding.category}: {finding.message}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", type=Path, default=DEFAULT_STUDY_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--protocol-id", default=DEFAULT_PROTOCOL_ID)
    parser.add_argument("--environments", default=",".join(str(value) for value in ENVIRONMENTS))
    parser.add_argument("--checkpoints", default=",".join(CHECKPOINTS))
    parser.add_argument("--metrics", default=",".join(METRICS))
    parser.add_argument("--expected-seeds", default=",".join(str(value) for value in EXPECTED_SEEDS))
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    environments = parse_csv_list(args.environments, int) or []
    checkpoints = parse_csv_list(args.checkpoints) or []
    metrics = parse_csv_list(args.metrics) or []
    expected_seeds = parse_csv_list(args.expected_seeds, int) or []
    unknown_environments = sorted(set(environments) - set(ENVIRONMENTS))
    unknown_checkpoints = sorted(set(checkpoints) - set(CHECKPOINTS))
    unknown_metrics = sorted(set(metrics) - set(METRICS))
    if unknown_environments or unknown_checkpoints or unknown_metrics:
        raise ValueError(
            f"Unknown configuration values: environments={unknown_environments}, "
            f"checkpoints={unknown_checkpoints}, metrics={unknown_metrics}"
        )
    if int(args.batch_size) != BATCH_SIZE:
        raise ValueError(f"This prespecified analysis fixes CSCG batch size at B={BATCH_SIZE}.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw = load_results(
        args.study_root,
        args.protocol_id,
        environments=environments,
        batch_size=int(args.batch_size),
    )
    if raw.empty:
        raise RuntimeError(f"No canonical evaluation records found under {args.study_root}")
    seed_level, duplicates = aggregate_to_seed_level(raw, metrics)
    provenance_findings = audit_evaluation_provenance(
        args.study_root,
        args.protocol_id,
        environments=environments,
        checkpoints=checkpoints,
        seeds=expected_seeds,
    )
    validation_text, findings = validate_results(
        raw,
        seed_level,
        duplicates,
        environments=environments,
        checkpoints=checkpoints,
        metrics=metrics,
        expected_seeds=expected_seeds,
        provenance_findings=provenance_findings,
    )
    validation_path = args.output_dir / "validation_report.txt"
    validation_path.write_text(validation_text, encoding="utf-8")
    print(validation_text)
    results, pairs = compute_all_contrasts(
        seed_level,
        metrics=metrics,
        environments=environments,
        checkpoints=checkpoints,
        expected_seeds=expected_seeds,
    )
    results = apply_holm_correction(results)
    csv_paths = list(write_csv_outputs(results, pairs, args.output_dir))
    latex_paths = list(write_latex_tables(results, args.output_dir))
    methods_path = write_methods_snippet(args.output_dir)
    results_summary_path = write_results_summary_snippet(results, args.output_dir)
    summary_path = write_statistical_summary(results, args.output_dir)
    heatmap_paths = plot_contrast_heatmaps(results, args.output_dir)
    output_paths = [
        *csv_paths,
        *latex_paths,
        methods_path,
        results_summary_path,
        summary_path,
        validation_path,
        *heatmap_paths,
    ]
    print_console_summary(
        results,
        metrics=metrics,
        environments=environments,
        checkpoints=checkpoints,
        output_paths=output_paths,
        findings=findings,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
