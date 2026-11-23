#!/usr/bin/env python3
"""Paired comparison of one-pass MERCURY with its frozen CSCG initialisation."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for import_root in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from scripts.analysis.planned_paired_contrasts import (
    apply_holm_correction,
    compute_paired_contrast,
    format_p_value,
)
from scripts.analysis.plot_config_metric_trajectories import (
    DEFAULT_LEVEL_ORDER,
    LEVEL_LABELS,
    _metrics_from_record,
    load_rows,
)


from scripts.analysis.study_selection import MAIN_STUDY_ROOT, PROTOCOL_ID

DEFAULT_STUDY_ROOT = MAIN_STUDY_ROOT
DEFAULT_OUTPUT_DIR = Path(
    "results/current/notebook_plots/main_study/pass_2_to_1000_statistical_analysis/"
    "mercury_source_vs_initialisation_statistical_analysis"
)
DEFAULT_PROTOCOL_ID = PROTOCOL_ID
EXPECTED_SEEDS = tuple(range(30))
SERIES = "mercury_cscg_hybrid"
SOURCE_METHOD = "mercury_cscg_hybrid_mercury"
SOURCE_STEP = 2_000
CONTRAST_ID = "cscg_initialisation_vs_mercury_source"
METRICS = {
    "latent_precision": "Latent purity",
    "action_tv_similarity": "Action-conditioned transition similarity",
    "action_no_self_loop_tv_similarity": "Action-conditioned no-self-loop similarity",
}


def load_paired_values(
    study_root: Path,
    protocol_id: str,
    *,
    batch_size: int,
) -> tuple[pd.DataFrame, list[str]]:
    """Load seed-paired source and initialisation metrics and audit provenance."""
    rows = load_rows(
        study_root,
        protocol_id,
        batch_size,
        set(DEFAULT_LEVEL_ORDER),
    )
    initial = rows.loc[
        (rows["series"] == SERIES)
        & (rows["step"] == 0)
        & rows["metric"].isin(METRICS),
        ["run_id", "level", "seed", "metric", "value"],
    ].rename(columns={"level": "environment", "value": "value_initialisation"})
    if initial.duplicated(["run_id", "environment", "seed", "metric"]).any():
        raise ValueError("Duplicate frozen-CSCG initialisation metric rows were found.")

    source_rows: list[dict[str, Any]] = []
    findings: list[str] = []
    conditions = initial[["run_id", "environment", "seed"]].drop_duplicates()
    for run_id, environment, seed in conditions.itertuples(index=False, name=None):
        snapshot_root = (
            study_root / str(run_id) / "evaluations" / protocol_id / "snapshots"
        )
        source_path = snapshot_root / SOURCE_METHOD / f"step={SOURCE_STEP:08d}.json"
        initial_path = snapshot_root / SERIES / "step=00000000.json"
        if not source_path.exists() or not initial_path.exists():
            findings.append(
                f"[ERROR] missing_snapshot: environment={environment}, seed={seed}, "
                f"source_exists={source_path.exists()}, initialisation_exists={initial_path.exists()}"
            )
            continue
        source_payload = json.loads(source_path.read_text(encoding="utf-8"))
        initial_payload = json.loads(initial_path.read_text(encoding="utf-8"))
        if source_payload.get("walk_cache_id") != initial_payload.get("walk_cache_id"):
            findings.append(
                f"[ERROR] walk_cache_mismatch: environment={environment}, seed={seed}"
            )
        source_protocol = source_payload.get("protocol") or {}
        initial_protocol = initial_payload.get("protocol") or {}
        for field in ("number_of_walks", "walk_length"):
            if source_protocol.get(field) != initial_protocol.get(field):
                findings.append(
                    f"[ERROR] protocol_mismatch: environment={environment}, seed={seed}, "
                    f"field={field}"
                )
        source_metrics = _metrics_from_record(source_payload)
        for metric in METRICS:
            value = source_metrics.get(metric)
            if value is None or not np.isfinite(float(value)):
                findings.append(
                    f"[ERROR] missing_source_metric: environment={environment}, seed={seed}, "
                    f"metric={metric}"
                )
                continue
            source_rows.append(
                {
                    "run_id": str(run_id),
                    "environment": int(environment),
                    "seed": int(seed),
                    "metric": metric,
                    "value_source": float(value),
                }
            )

    source = pd.DataFrame(source_rows)
    paired = initial.merge(
        source,
        on=["run_id", "environment", "seed", "metric"],
        how="inner",
        validate="one_to_one",
    )
    paired["difference"] = paired["value_initialisation"] - paired["value_source"]
    paired.insert(4, "metric_label", paired["metric"].map(METRICS))
    paired.insert(
        2,
        "environment_label",
        paired["environment"].map(lambda value: LEVEL_LABELS.get(int(value), f"L{value}")),
    )
    return paired.sort_values(["metric", "environment", "seed"]).reset_index(drop=True), findings


def validate_pairs(paired: pd.DataFrame, provenance_findings: list[str]) -> str:
    findings = list(provenance_findings)
    expected = set(EXPECTED_SEEDS)
    for metric in METRICS:
        for environment in DEFAULT_LEVEL_ORDER:
            condition = paired.loc[
                (paired["metric"] == metric) & (paired["environment"] == environment)
            ]
            seeds = set(int(seed) for seed in condition["seed"])
            if seeds != expected:
                findings.append(
                    f"[ERROR] seed_mismatch: metric={metric}, environment={environment}, "
                    f"missing={sorted(expected - seeds)}, extra={sorted(seeds - expected)}"
                )
            if not np.isfinite(
                condition[["value_initialisation", "value_source", "difference"]].to_numpy(float)
            ).all():
                findings.append(
                    f"[ERROR] nonfinite_values: metric={metric}, environment={environment}"
                )
    expected_rows = len(METRICS) * len(DEFAULT_LEVEL_ORDER) * len(EXPECTED_SEEDS)
    if len(paired) != expected_rows:
        findings.append(f"[ERROR] expected_rows={expected_rows}, observed_rows={len(paired)}")
    return "\n".join(
        [
            "One-pass MERCURY source vs frozen CSCG initialisation validation",
            "=" * 67,
            "",
            f"Paired metric rows: {len(paired)}",
            f"Expected paired metric rows: {expected_rows}",
            f"Expected seeds per environment: {len(EXPECTED_SEEDS)}",
            f"Environments: {list(DEFAULT_LEVEL_ORDER)}",
            f"Metrics: {list(METRICS)}",
            "",
            "Findings",
            "-" * 8,
            *(findings or ["No validation warnings or errors."]),
            "",
        ]
    )


def compute_results(paired: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for metric, metric_label in METRICS.items():
        for environment in DEFAULT_LEVEL_ORDER:
            condition = paired.loc[
                (paired["metric"] == metric) & (paired["environment"] == environment)
            ].sort_values("seed")
            statistics = compute_paired_contrast(
                condition["value_initialisation"].to_numpy(float),
                condition["value_source"].to_numpy(float),
            )
            records.append(
                {
                    "metric": metric,
                    "metric_label": metric_label,
                    "environment": int(environment),
                    "environment_label": LEVEL_LABELS.get(int(environment), f"L{environment}"),
                    "contrast_id": CONTRAST_ID,
                    "contrast_label": "Frozen CSCG initialisation vs one-pass MERCURY source",
                    "method_a": "frozen_cscg_initialisation",
                    "method_b": "one_pass_mercury_source",
                    **statistics,
                    "p_holm": math.nan,
                    "significant_holm_0_05": False,
                    "holm_family": f"{metric}::{CONTRAST_ID}",
                    "holm_family_n_tests": 0,
                    "expected_n_pairs": len(EXPECTED_SEEDS),
                    "n_pairs_is_expected": statistics["n_pairs"] == len(EXPECTED_SEEDS),
                    "available": statistics["n_pairs"] >= 2,
                    "unavailable_reason": "",
                }
            )
    return apply_holm_correction(pd.DataFrame(records))


def _markdown_table(results: pd.DataFrame) -> str:
    lines = [
        "| Metric | Environment | Initialisation mean | MERCURY mean | Δ | 95% CI | p (Holm) | dz |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in results.itertuples(index=False):
        lines.append(
            f"| {row.metric_label} | {row.environment_label} | {row.mean_a:.3f} | "
            f"{row.mean_b:.3f} | {row.mean_difference:+.3f} | "
            f"[{row.ci95_low:+.3f}, {row.ci95_high:+.3f}] | "
            f"{format_p_value(row.p_holm)} | {row.cohens_dz:+.3f} |"
        )
    return "\n".join(lines) + "\n"


def _latex_table(results: pd.DataFrame) -> str:
    lines = [
        r"\begin{table*}[!t]",
        r"\centering",
        (
            r"\caption{Paired comparison of the frozen CSCG initialisation with its one-pass "
            r"MERCURY source representation.}"
        ),
        r"\label{tab:mercury_source_vs_cscg_initialisation}",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{5pt}",
        r"\begin{tabular}{@{}llccccc@{}}",
        r"\toprule",
        (
            r"Metric & Env. & CSCG init. & MERCURY & $\Delta$ [95\% CI] & "
            r"$p_{\mathrm{Holm}}$ & $d_z$ \\"
        ),
        r"\midrule",
    ]
    for metric_index, metric in enumerate(METRICS):
        metric_rows = results.loc[results["metric"] == metric]
        for environment_index, environment in enumerate(DEFAULT_LEVEL_ORDER):
            row = metric_rows.loc[metric_rows["environment"] == environment].iloc[0]
            env = str(row["environment_label"]).replace("x", r"$\times$")
            p_holm = format_p_value(row["p_holm"])
            lines.append(
                " & ".join(
                    [
                        METRICS[metric] if environment_index == 0 else "",
                        env,
                        f"{row['mean_a']:.3f}",
                        f"{row['mean_b']:.3f}",
                        (
                            f"{row['mean_difference']:+.3f} "
                            f"[{row['ci95_low']:+.3f}, {row['ci95_high']:+.3f}]"
                        ),
                        p_holm,
                        f"{row['cohens_dz']:+.3f}",
                    ]
                )
                + r" \\"
            )
        if metric_index < len(METRICS) - 1:
            lines.append(r"\addlinespace[2pt]")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\vspace{2pt}",
            (
                r"\parbox{\textwidth}{\footnotesize\textit{Note:} "
                r"$\Delta=\mathrm{CSCG\ initialisation}-\mathrm{one\mbox{-}pass\ MERCURY}$. "
                r"Two-sided paired $t$-tests use 30 seed pairs per environment. Holm correction "
                r"is applied across the four environments separately for each metric.}"
            ),
            r"\end{table*}",
            "",
        ]
    )
    return "\n".join(lines)


def write_outputs(
    paired: pd.DataFrame,
    results: pd.DataFrame,
    validation: str,
    output_dir: Path,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "pairs": output_dir / "mercury_source_vs_initialisation_paired_seed_values.csv",
        "full": output_dir / "mercury_source_vs_initialisation_paired_contrasts_full.csv",
        "publication": output_dir / "mercury_source_vs_initialisation_paired_contrasts_publication.csv",
        "markdown": output_dir / "mercury_source_vs_initialisation_paired_contrasts.md",
        "latex": output_dir / "mercury_source_vs_initialisation_paired_contrasts.tex",
        "summary": output_dir / "mercury_source_vs_initialisation_statistical_summary.txt",
        "methods": output_dir / "mercury_source_vs_initialisation_methods_snippet.tex",
        "validation": output_dir / "mercury_source_vs_initialisation_validation_report.txt",
    }
    paired.to_csv(paths["pairs"], index=False, float_format="%.17g")
    results.to_csv(paths["full"], index=False, float_format="%.17g")
    publication = results.copy()
    for column in (
        "mean_a", "mean_b", "mean_difference", "ci95_low", "ci95_high",
        "sd_difference", "se_difference", "t_statistic", "cohens_dz",
    ):
        publication[column] = publication[column].map(
            lambda value: "NA" if not np.isfinite(float(value)) else f"{float(value):.3f}"
        )
    publication["p_raw"] = publication["p_raw"].map(format_p_value)
    publication["p_holm"] = publication["p_holm"].map(format_p_value)
    publication.to_csv(paths["publication"], index=False)
    paths["markdown"].write_text(_markdown_table(results), encoding="utf-8")
    paths["latex"].write_text(_latex_table(results), encoding="utf-8")
    summary_lines = [
        "One-pass MERCURY source vs frozen CSCG initialisation",
        "=" * 56,
        "",
        "Difference convention: frozen CSCG initialisation minus one-pass MERCURY source.",
        "Positive differences indicate that CSCG initialisation increased the metric.",
        "",
    ]
    for metric, label in METRICS.items():
        subset = results.loc[results["metric"] == metric]
        summary_lines.append(label)
        summary_lines.append("-" * len(label))
        for row in subset.itertuples(index=False):
            summary_lines.append(
                f"{row.environment_label}: Δ={row.mean_difference:+.6f}, "
                f"95% CI [{row.ci95_low:+.6f}, {row.ci95_high:+.6f}], "
                f"p_Holm={format_p_value(row.p_holm)}, dz={row.cohens_dz:+.3f}."
            )
        summary_lines.append("")
    paths["summary"].write_text("\n".join(summary_lines), encoding="utf-8")
    paths["methods"].write_text(
        (
            "The independent training seed was treated as the unit of analysis. The one-pass "
            "MERCURY source representation and its derived frozen CSCG initialisation were paired "
            "by seed within each environment ($n=30$). We report the paired mean difference "
            r"$\Delta=\mathrm{CSCG\ initialisation}-\mathrm{one\mbox{-}pass\ MERCURY}$, its "
            "two-sided 95\\% Student-$t$ confidence interval, a two-sided paired $t$-test, and "
            "Cohen's $d_z$. Within each metric, the four environment-specific $p$-values were "
            "adjusted using the Holm procedure. Both conditions were evaluated using the same "
            "canonical held-out walk cache; evaluation walks were not treated as independent "
            "replicates. Evaluation NLL was omitted because it is undefined for the standalone "
            "one-pass MERCURY representation.\n"
        ),
        encoding="utf-8",
    )
    paths["validation"].write_text(validation, encoding="utf-8")
    return list(paths.values())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", type=Path, default=DEFAULT_STUDY_ROOT)
    parser.add_argument("--protocol-id", default=DEFAULT_PROTOCOL_ID)
    parser.add_argument("--batch-size", type=int, default=40)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paired, provenance_findings = load_paired_values(
        args.study_root,
        args.protocol_id,
        batch_size=args.batch_size,
    )
    validation = validate_pairs(paired, provenance_findings)
    if "[ERROR]" in validation:
        raise RuntimeError(validation)
    results = compute_results(paired)
    outputs = write_outputs(paired, results, validation, args.output_dir)
    print(validation)
    print("MERCURY source/initialisation paired analysis complete")
    for path in outputs:
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
