"""Paired statistical analysis for the MERCURY ambiguity-response ablation.

The independent experimental unit is the training seed.  Every ablation is
paired with Full MERCURY by environment and seed and evaluated on the common
canonical held-out walks stored in the evaluation records.
"""
from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import sys
from typing import Any, Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import pandas as pd

import planned_paired_contrasts as main_stats
from mercury_runs.analysis.plots.ambiguity_response_ablation import _frame, _load_rows


DEFAULT_STUDY_ROOT = Path(
    "results/current/studies/ambiguity_response_ablation_rand_prob_0_1"
)
DEFAULT_OUTPUT_DIR = Path(
    "results/current/notebook_plots/ambiguity_response_ablation_rand_prob_0_1/statistical_analysis"
)
DEFAULT_PROTOCOL_ID = "canonical-v4-ea50f4caf128f37a"
EXPECTED_SEEDS = tuple(range(30))
ENVIRONMENTS = main_stats.ENVIRONMENTS
METRICS = main_stats.METRICS
CHECKPOINT = "final"
CHECKPOINT_LABEL = "Final MERCURY checkpoint"
CHECKPOINT_STEP = 2_000

REFERENCE_MODE = "full_mercury"
COMPARATORS = (
    "refinement_without_reconstruction",
    "local_repair_only",
    "no_ambiguity_response",
)
MODE_LABELS = {
    "full_mercury": "Full MERCURY",
    "refinement_without_reconstruction": "Refinement without reconstruction",
    "local_repair_only": "Local repair only",
    "no_ambiguity_response": "No ambiguity response",
}
COMPACT_MODE_LABELS = {
    "refinement_without_reconstruction": "No reconstruction",
    "local_repair_only": "Local repair",
    "no_ambiguity_response": "No response",
}
CONTRASTS = tuple(
    (f"full_mercury_vs_{mode}", REFERENCE_MODE, mode) for mode in COMPARATORS
)


def parse_csv_list(raw: str | None, cast: Any = str) -> list[Any] | None:
    if raw is None:
        return None
    return [cast(item.strip()) for item in raw.split(",") if item.strip()]


def contrast_label(method_a: str, method_b: str) -> str:
    return f"{MODE_LABELS[method_a]} vs {MODE_LABELS[method_b]}"


def load_results(study_root: Path, protocol_id: str) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    rows = _load_rows(study_root, protocol_id)
    result = _frame(rows).to_pandas().rename(
        columns={"level": "environment", "plot_series": "method"}
    )
    result["checkpoint"] = CHECKPOINT
    return result, rows


def aggregate_to_seed_level(
    results: pd.DataFrame, metrics: Iterable[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    keys = ["environment", "checkpoint", "method", "seed"]
    counts = results.groupby(keys, dropna=False).size().reset_index(name="row_count")
    duplicates = counts.loc[counts["row_count"] > 1].copy()
    seed_level = (
        results[keys + list(metrics)]
        .groupby(keys, as_index=False, dropna=False)
        .mean(numeric_only=True)
    )
    return seed_level, duplicates


def _seed_text(values: Iterable[int]) -> str:
    return ";".join(str(int(value)) for value in sorted(values))


def build_paired_samples(
    seed_level: pd.DataFrame,
    *,
    metric: str,
    environment: int,
    method_b: str,
    expected_seeds: list[int],
) -> tuple[pd.DataFrame, dict[str, str]]:
    columns = ["seed", metric]
    reference = seed_level.loc[
        (seed_level["environment"] == environment)
        & (seed_level["method"] == REFERENCE_MODE),
        columns,
    ].rename(columns={metric: "value_a"})
    comparator = seed_level.loc[
        (seed_level["environment"] == environment)
        & (seed_level["method"] == method_b),
        columns,
    ].rename(columns={metric: "value_b"})
    seeds_a = set(int(value) for value in reference["seed"])
    seeds_b = set(int(value) for value in comparator["seed"])
    nonfinite_a = set(
        int(value)
        for value in reference.loc[~np.isfinite(reference["value_a"]), "seed"]
    )
    nonfinite_b = set(
        int(value)
        for value in comparator.loc[~np.isfinite(comparator["value_b"]), "seed"]
    )
    pairs = reference.merge(comparator, on="seed", how="inner", validate="one_to_one")
    pairs = pairs.loc[
        np.isfinite(pairs["value_a"].to_numpy(dtype=float))
        & np.isfinite(pairs["value_b"].to_numpy(dtype=float))
    ].copy()
    pairs["difference"] = pairs["value_a"] - pairs["value_b"]
    pairs = pairs.sort_values("seed").reset_index(drop=True)
    audit = {
        "missing_seeds_a": _seed_text(set(expected_seeds) - seeds_a),
        "missing_seeds_b": _seed_text(set(expected_seeds) - seeds_b),
        "unpaired_seeds_a": _seed_text(seeds_a - seeds_b),
        "unpaired_seeds_b": _seed_text(seeds_b - seeds_a),
        "nonfinite_seeds_a": _seed_text(nonfinite_a),
        "nonfinite_seeds_b": _seed_text(nonfinite_b),
    }
    return pairs, audit


def compute_all_contrasts(
    seed_level: pd.DataFrame,
    *,
    metrics: list[str],
    environments: list[int],
    expected_seeds: list[int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summaries: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    for metric in metrics:
        for contrast_id, method_a, method_b in CONTRASTS:
            for environment in environments:
                pairs, audit = build_paired_samples(
                    seed_level,
                    metric=metric,
                    environment=environment,
                    method_b=method_b,
                    expected_seeds=expected_seeds,
                )
                statistics = main_stats.compute_paired_contrast(
                    pairs["value_a"].to_numpy(dtype=float),
                    pairs["value_b"].to_numpy(dtype=float),
                )
                reasons = []
                if statistics["n_pairs"] < 2:
                    reasons.append("fewer than two finite paired seeds")
                reasons.extend(f"{key}={value}" for key, value in audit.items() if value)
                label = contrast_label(method_a, method_b)
                summaries.append(
                    {
                        "metric": metric,
                        "metric_label": METRICS[metric]["label"],
                        "environment": environment,
                        "environment_label": ENVIRONMENTS[environment],
                        "checkpoint": CHECKPOINT,
                        "checkpoint_label": CHECKPOINT_LABEL,
                        "checkpoint_step": CHECKPOINT_STEP,
                        "contrast_id": contrast_id,
                        "contrast_label": label,
                        "method_a": method_a,
                        "method_b": method_b,
                        **statistics,
                        "p_holm": math.nan,
                        "significant_holm_0_05": False,
                        "holm_family": f"{metric}::{contrast_id}",
                        "holm_family_n_tests": 0,
                        "expected_n_pairs": len(expected_seeds),
                        "n_pairs_is_expected": statistics["n_pairs"] == len(expected_seeds),
                        "available": statistics["n_pairs"] >= 2,
                        "unavailable_reason": "; ".join(reasons),
                        **audit,
                    }
                )
                for pair in pairs.to_dict("records"):
                    pair_rows.append(
                        {
                            "metric": metric,
                            "environment": environment,
                            "checkpoint": CHECKPOINT,
                            "contrast_id": contrast_id,
                            "contrast_label": label,
                            "method_a": method_a,
                            "method_b": method_b,
                            **pair,
                        }
                    )
    return pd.DataFrame(summaries), pd.DataFrame(pair_rows)


def validate_results(
    raw: pd.DataFrame,
    seed_level: pd.DataFrame,
    duplicates: pd.DataFrame,
    source_rows: list[dict[str, Any]],
    *,
    metrics: list[str],
    environments: list[int],
    expected_seeds: list[int],
) -> tuple[str, list[str]]:
    findings: list[str] = []
    lines = [
        "MERCURY ambiguity-response paired analysis validation report",
        "=" * 61,
        "",
        f"Protocol ID: {source_rows[0]['record'].get('protocol_id') if source_rows else 'NA'}",
        f"Environments requested: {environments}",
        f"Modes detected: {sorted(raw['method'].unique().tolist())}",
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
        for method in (REFERENCE_MODE, *COMPARATORS):
            subset = seed_level.loc[
                (seed_level["environment"] == environment)
                & (seed_level["method"] == method)
            ]
            seeds = sorted(int(seed) for seed in subset["seed"])
            lines.append(
                f"environment={environment} ({ENVIRONMENTS[environment]}), method={method}: "
                f"n={len(seeds)}, seeds={seeds}"
            )
            missing = sorted(set(expected_seeds) - set(seeds))
            if missing:
                findings.append(
                    f"[ERROR] missing_condition_seeds: environment={environment}, "
                    f"method={method}, missing={missing}"
                )

    if not duplicates.empty:
        findings.extend(
            f"[ERROR] duplicate_rows: {row}"
            for row in duplicates.to_dict("records")
        )
    for metric in metrics:
        bad = seed_level.loc[~np.isfinite(seed_level[metric].to_numpy(dtype=float))]
        findings.extend(
            f"[ERROR] nonfinite_value: metric={metric}, condition={row}"
            for row in bad[["environment", "method", "seed"]].to_dict("records")
        )

    provenance = pd.DataFrame(
        {
            "environment": row["level"],
            "seed": row["seed"],
            "method": row["mode"],
            "walk_cache_id": row["record"].get("walk_cache_id"),
            "number_of_walks": (row["record"].get("protocol") or {}).get("number_of_walks"),
            "walk_length": (row["record"].get("protocol") or {}).get("walk_length"),
        }
        for row in source_rows
    )
    cache_mismatches = 0
    protocol_mismatches = 0
    for (environment, seed), group in provenance.groupby(["environment", "seed"]):
        cache_ids = group.set_index("method")["walk_cache_id"].to_dict()
        protocols = set(zip(group["number_of_walks"], group["walk_length"]))
        if len(set(cache_ids.values())) > 1:
            cache_mismatches += 1
            findings.append(
                f"[ERROR] walk_cache_mismatch: environment={environment}, seed={seed}: {cache_ids}"
            )
        if len(protocols) > 1:
            protocol_mismatches += 1
            findings.append(
                f"[ERROR] evaluation_protocol_mismatch: environment={environment}, "
                f"seed={seed}: {sorted(protocols)}"
            )
    lines.extend(
        [
            "",
            "Validation totals",
            "-" * 17,
            f"Non-finite metric values: {sum('nonfinite_value' in item for item in findings)}",
            f"Duplicate condition groups: {len(duplicates)}",
            f"Walk-cache mismatches: {cache_mismatches}",
            f"Evaluation-protocol mismatches: {protocol_mismatches}",
            "",
            "Findings",
            "-" * 8,
            *(findings or ["No validation warnings or errors."]),
        ]
    )
    return "\n".join(lines) + "\n", findings


def _latex_effect_cell(row: pd.Series) -> str:
    if not bool(row["available"]):
        return "Unavailable"
    effect = rf"{float(row['mean_difference']):+.3f} [{float(row['ci95_low']):.3f},\,{float(row['ci95_high']):.3f}]"
    return rf"\textbf{{{effect}}}" if float(row["p_holm"]) < 0.05 else effect


def _latex_delta_cell(row: pd.Series) -> str:
    return main_stats._latex_delta_cell(row)


def _primary_table(
    results: pd.DataFrame, *, metric: str, caption: str, label: str
) -> list[str]:
    rows = results.loc[results["metric"] == metric]
    output = [
        r"\begin{table}[!t]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\renewcommand{\arraystretch}{1.08}",
        r"\begin{tabular}{@{}llcc@{}}",
        r"\toprule",
        r"Comparator & Env. & $\Delta$ [95\% CI] & $p_{\mathrm{Holm}}$ \\",
        r"\midrule",
    ]
    for comparator_index, comparator in enumerate(COMPARATORS):
        contrast_id = f"full_mercury_vs_{comparator}"
        for environment_index, environment in enumerate(ENVIRONMENTS):
            row = rows.loc[
                (rows["contrast_id"] == contrast_id)
                & (rows["environment"] == environment)
            ].iloc[0]
            output.append(
                " & ".join(
                    [
                        main_stats.latex_escape(MODE_LABELS[comparator])
                        if environment_index == 0
                        else "",
                        main_stats.LATEX_ENVIRONMENT_LABELS[environment],
                        _latex_effect_cell(row),
                        main_stats._latex_p_cell(row),
                    ]
                )
                + r" \\"
            )
        if comparator_index < len(COMPARATORS) - 1:
            output.append(r"\addlinespace[2pt]")
    output.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\vspace{2pt}",
            (
                r"\parbox{\columnwidth}{\footnotesize\textit{Note:} "
                r"$\Delta=\mathrm{Full\ MERCURY}-\mathrm{comparator}$; positive values favour "
                r"Full MERCURY. Tests are paired across 30 training seeds. Bold entries have "
                r"$p_{\mathrm{Holm}}<0.05$. Holm correction is applied across the four "
                r"environments within each metric-specific comparator family.}"
            ),
            r"\end{table}",
            "",
        ]
    )
    return output


def _main_body_table(
    results: pd.DataFrame,
    *,
    metric: str,
    caption: str,
    label: str,
    appendix_label: str,
    transposed: bool,
) -> list[str]:
    rows = results.loc[results["metric"] == metric]
    output = [
        r"\begin{table}[!t]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label + ('_transposed' if transposed else '')}}}",
        r"\footnotesize",
        r"\setlength{\fboxsep}{1.5pt}",
        r"\renewcommand{\arraystretch}{1.05}",
    ]
    if not transposed:
        output.extend(
            [
                r"\begin{tabular}{@{}llc@{}}",
                r"\toprule",
                r"Comparator & Env. & $\Delta$ \\",
                r"\midrule",
            ]
        )
        for comparator_index, comparator in enumerate(COMPARATORS):
            contrast_id = f"full_mercury_vs_{comparator}"
            for environment_index, environment in enumerate(ENVIRONMENTS):
                row = rows.loc[
                    (rows["contrast_id"] == contrast_id)
                    & (rows["environment"] == environment)
                ].iloc[0]
                output.append(
                    " & ".join(
                        [
                            main_stats.latex_escape(MODE_LABELS[comparator])
                            if environment_index == 0
                            else "",
                            main_stats.LATEX_ENVIRONMENT_LABELS[environment],
                            _latex_delta_cell(row),
                        ]
                    )
                    + r" \\"
                )
            if comparator_index < len(COMPARATORS) - 1:
                output.append(r"\addlinespace[2pt]")
    else:
        output.extend(
            [
                r"\setlength{\tabcolsep}{4pt}",
                r"\begin{tabular}{@{}lccc@{}}",
                r"\toprule",
                "Env. & "
                + " & ".join(
                    main_stats.latex_escape(COMPACT_MODE_LABELS[item]) for item in COMPARATORS
                )
                + r" \\",
                r"\midrule",
            ]
        )
        for environment in ENVIRONMENTS:
            cells = []
            for comparator in COMPARATORS:
                row = rows.loc[
                    (rows["contrast_id"] == f"full_mercury_vs_{comparator}")
                    & (rows["environment"] == environment)
                ].iloc[0]
                cells.append(_latex_delta_cell(row))
            output.append(
                " & ".join([main_stats.LATEX_ENVIRONMENT_LABELS[environment], *cells])
                + r" \\"
            )
    output.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\vspace{2pt}",
            (
                r"\parbox{\columnwidth}{\footnotesize\textit{Note:} "
                r"$\Delta=\mathrm{Full\ MERCURY}-\mathrm{comparator}$. Green and red shading "
                r"denote positive and negative paired differences. Lighter shading is not "
                r"significant after Holm correction; significant results are bold. Full "
                r"intervals and adjusted $p$-values are reported in Appendix "
                rf"Table~\ref{{{appendix_label}}}.}}"
            ),
            r"\end{table}",
            "",
        ]
    )
    return output


def write_latex_tables(results: pd.DataFrame, output_dir: Path) -> list[Path]:
    header = [
        "% Generated by ambiguity_response_paired_contrasts.py",
        "% Requires:",
        "% \\usepackage{booktabs}",
        "% \\usepackage[table]{xcolor}",
        "",
    ]
    action = _primary_table(
        results,
        metric=main_stats.LATEX_PRIMARY_METRIC,
        caption=(
            "Paired ambiguity-response comparisons for action-conditioned transition "
            "similarity with self-loops included."
        ),
        label="tab:ambiguity_response_ablation_action_tv",
    )
    no_self_loops = _primary_table(
        results,
        metric=main_stats.LATEX_NO_SELF_LOOP_METRIC,
        caption=(
            "Paired ambiguity-response comparisons for conditional non-self-loop "
            "action-conditioned transition similarity."
        ),
        label="tab:ambiguity_response_ablation_action_tv_no_self_loops",
    )
    latent = _primary_table(
        results,
        metric=main_stats.LATEX_LATENT_METRIC,
        caption="Paired ambiguity-response comparisons for latent purity.",
        label="tab:ambiguity_response_ablation_latent_purity",
    )
    contents = {
        "planned_contrasts_appendix.tex": [*header, *action, *no_self_loops, *latent],
        "planned_contrasts_summary.tex": [*header, *action],
        "planned_contrasts_main_body.tex": [
            *header,
            *_main_body_table(
                results,
                metric=main_stats.LATEX_PRIMARY_METRIC,
                caption=(
                    "Full MERCURY comparisons for action-conditioned transition similarity "
                    "in the ambiguity-response ablation."
                ),
                label="tab:ambiguity_response_ablation_main",
                appendix_label="tab:ambiguity_response_ablation_action_tv",
                transposed=False,
            ),
            *_main_body_table(
                results,
                metric=main_stats.LATEX_LATENT_METRIC,
                caption=(
                    "Full MERCURY comparisons for latent purity in the "
                    "ambiguity-response ablation."
                ),
                label="tab:ambiguity_response_ablation_purity_main",
                appendix_label="tab:ambiguity_response_ablation_latent_purity",
                transposed=False,
            ),
        ],
        "planned_contrasts_main_body_transposed.tex": [
            *header,
            *_main_body_table(
                results,
                metric=main_stats.LATEX_PRIMARY_METRIC,
                caption=(
                    "Full MERCURY comparisons for action-conditioned transition similarity "
                    "in the ambiguity-response ablation."
                ),
                label="tab:ambiguity_response_ablation_main",
                appendix_label="tab:ambiguity_response_ablation_action_tv",
                transposed=True,
            ),
            *_main_body_table(
                results,
                metric=main_stats.LATEX_LATENT_METRIC,
                caption=(
                    "Full MERCURY comparisons for latent purity in the "
                    "ambiguity-response ablation."
                ),
                label="tab:ambiguity_response_ablation_purity_main",
                appendix_label="tab:ambiguity_response_ablation_latent_purity",
                transposed=True,
            ),
        ],
    }
    paths = []
    for name, lines in contents.items():
        path = output_dir / name
        path.write_text("\n".join(lines), encoding="utf-8")
        paths.append(path)
    return paths


def write_csv_outputs(
    results: pd.DataFrame, pairs: pd.DataFrame, output_dir: Path
) -> list[Path]:
    full = output_dir / "planned_contrasts_full.csv"
    publication = output_dir / "planned_contrasts_publication.csv"
    seed_pairs = output_dir / "planned_contrasts_seed_pairs.csv"
    results[main_stats.FULL_COLUMNS].to_csv(full, index=False, float_format="%.17g")
    main_stats.publication_frame(results[main_stats.FULL_COLUMNS]).to_csv(
        publication, index=False
    )
    pairs.to_csv(seed_pairs, index=False, float_format="%.17g")
    return [full, publication, seed_pairs]


def write_text_outputs(results: pd.DataFrame, output_dir: Path) -> list[Path]:
    methods = output_dir / "statistical_methods_snippet.tex"
    methods.write_text(
        "The independent training seed was treated as the unit of analysis. Each ambiguity-response "
        "ablation was compared with Full MERCURY using a two-sided paired $t$-test across 30 seeds "
        "evaluated on common held-out trajectories. We report the paired mean difference, its "
        "two-sided 95\\% confidence interval, and Cohen's $d_z$. Within each metric-specific "
        "comparator family, the four environment-level $p$-values were adjusted using the Holm "
        "procedure. Held-out walks were pooled within seed and were not treated as independent "
        "replicates.\n",
        encoding="utf-8",
    )
    summary = output_dir / "statistical_summary.txt"
    summary_lines = [
        "Descriptive summary of ambiguity-response paired contrasts",
        "=" * 58,
        "Positive effects favour Full MERCURY.",
        "",
    ]
    for metric in metrics_in_order(results):
        summary_lines.extend([METRICS[metric]["label"], "-" * len(METRICS[metric]["label"])])
        for contrast_id, _, comparator in CONTRASTS:
            rows = results.loc[
                (results["metric"] == metric) & (results["contrast_id"] == contrast_id)
            ]
            effects = rows["mean_difference"].to_numpy(dtype=float)
            summary_lines.append(
                f"Full MERCURY vs {MODE_LABELS[comparator]}: positive={int(np.sum(effects > 0))}, "
                f"negative={int(np.sum(effects < 0))}, Holm p<0.05="
                f"{int(rows['significant_holm_0_05'].sum())}/{len(rows)}, "
                f"effect range=[{effects.min():.6g}, {effects.max():.6g}]"
            )
        summary_lines.append("")
    summary.write_text("\n".join(summary_lines), encoding="utf-8")

    results_snippet = output_dir / "statistical_results_summary.tex"
    sentences = []
    for metric, result_label in (
        (main_stats.LATEX_PRIMARY_METRIC, "action-conditioned transition-similarity"),
        (main_stats.LATEX_LATENT_METRIC, "latent-purity"),
    ):
        metric_rows = results.loc[results["metric"] == metric]
        for contrast_id, _, comparator in CONTRASTS:
            rows = metric_rows.loc[metric_rows["contrast_id"] == contrast_id]
            positive = int((rows["mean_difference"] > 0).sum())
            significant = int(rows["significant_holm_0_05"].sum())
            sentences.append(
                f"Full MERCURY had positive paired {result_label} differences relative to "
                f"{MODE_LABELS[comparator]} in {positive} of {len(rows)} environments "
                f"($\\Delta={rows['mean_difference'].min():.3f}$--"
                f"${rows['mean_difference'].max():.3f}$); {significant} of {len(rows)} "
                "comparisons remained significant after Holm correction."
            )
    results_snippet.write_text("\n\n".join(sentences) + "\n", encoding="utf-8")
    return [methods, summary, results_snippet]


def metrics_in_order(results: pd.DataFrame) -> list[str]:
    return results["metric"].drop_duplicates().tolist()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", type=Path, default=DEFAULT_STUDY_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--protocol-id", default=DEFAULT_PROTOCOL_ID)
    parser.add_argument("--environments", default=",".join(str(value) for value in ENVIRONMENTS))
    parser.add_argument("--metrics", default=",".join(METRICS))
    parser.add_argument("--expected-seeds", default=",".join(str(value) for value in EXPECTED_SEEDS))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    environments = parse_csv_list(args.environments, int) or []
    metrics = parse_csv_list(args.metrics) or []
    expected_seeds = parse_csv_list(args.expected_seeds, int) or []
    unknown_environments = sorted(set(environments) - set(ENVIRONMENTS))
    unknown_metrics = sorted(set(metrics) - set(METRICS))
    if unknown_environments or unknown_metrics:
        raise ValueError(
            f"Unknown configuration values: environments={unknown_environments}, metrics={unknown_metrics}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw, source_rows = load_results(args.study_root, args.protocol_id)
    raw = raw.loc[raw["environment"].isin(environments)].copy()
    source_rows = [row for row in source_rows if int(row["level"]) in environments]
    seed_level, duplicates = aggregate_to_seed_level(raw, metrics)
    validation_text, findings = validate_results(
        raw,
        seed_level,
        duplicates,
        source_rows,
        metrics=metrics,
        environments=environments,
        expected_seeds=expected_seeds,
    )
    validation_path = args.output_dir / "validation_report.txt"
    validation_path.write_text(validation_text, encoding="utf-8")
    results, pairs = compute_all_contrasts(
        seed_level,
        metrics=metrics,
        environments=environments,
        expected_seeds=expected_seeds,
    )
    results = main_stats.apply_holm_correction(results)
    outputs = [
        *write_csv_outputs(results, pairs, args.output_dir),
        *write_latex_tables(results, args.output_dir),
        *write_text_outputs(results, args.output_dir),
        validation_path,
    ]
    print(validation_text)
    print("Statistical analysis complete")
    print(f"Valid paired comparisons: {int(results['available'].sum())}")
    print(f"Expected seed pairs per comparison: {len(expected_seeds)}")
    print("Outputs:")
    for path in outputs:
        print(f"  {path}")
    if findings:
        print(f"Validation findings: {len(findings)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
