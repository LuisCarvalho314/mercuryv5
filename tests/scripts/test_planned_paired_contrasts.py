from __future__ import annotations

import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = PROJECT_ROOT / "scripts" / "analysis"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import planned_paired_contrasts as analysis


def test_planned_contrasts_include_mercury_vs_oracle_first_order() -> None:
    assert (
        "mercury_vs_oracle_first_order",
        "mercury_derived",
        "oracle_capacity_first_order",
    ) in analysis.PLANNED_CONTRASTS
    assert (
        analysis.LATEX_CONTRAST_LABELS["mercury_vs_oracle_first_order"]
        == "MERCURY vs Oracle first-order"
    )


def test_main_body_table_contains_only_mercury_action_similarity_comparisons() -> None:
    rows = []
    for contrast_id in analysis.MAIN_BODY_CONTRAST_IDS:
        for environment in analysis.ENVIRONMENTS:
            for checkpoint in analysis.CHECKPOINTS:
                rows.append(
                    {
                        "metric": analysis.LATEX_PRIMARY_METRIC,
                        "contrast_id": contrast_id,
                        "environment": environment,
                        "checkpoint": checkpoint,
                        "available": True,
                        "mean_difference": 0.1,
                        "ci95_low": 0.05,
                        "ci95_high": 0.15,
                        "p_holm": 0.01,
                    }
                )
    table = "\n".join(analysis._latex_main_body_table(pd.DataFrame(rows)))
    assert "\\begin{table}[!t]" in table
    assert "\\begin{table*}" not in table
    assert "\\parbox{\\columnwidth}" in table
    assert "Comparator & Env." in table
    assert "Standard CSCG & $3\\times3$" in table
    assert "Oracle first-order & $3\\times3$" in table
    assert "MERCURY vs" not in table
    assert "Oracle first-order vs capacity" not in table
    assert "TMGWR vs Standard" not in table
    assert "& $p_{\\mathrm{Holm}}$" not in table
    assert "95\\% CI" not in table
    assert "[0.050" not in table
    assert "\\colorbox{green!12}{\\strut\\makebox[3.4em][r]{\\textbf{+0.100}}}" in table
    assert "Confidence intervals" in table
    assert "exact adjusted $p$-values" in table


def test_transposed_main_body_can_render_no_self_loop_metric() -> None:
    rows = []
    for contrast_id in analysis.MAIN_BODY_CONTRAST_IDS:
        for environment in analysis.ENVIRONMENTS:
            for checkpoint in analysis.CHECKPOINTS:
                rows.append(
                    {
                        "metric": analysis.LATEX_NO_SELF_LOOP_METRIC,
                        "contrast_id": contrast_id,
                        "environment": environment,
                        "checkpoint": checkpoint,
                        "available": True,
                        "mean_difference": 0.1,
                        "p_holm": 0.01,
                    }
                )
    table = "\n".join(
        analysis._latex_main_body_transposed_table(
            pd.DataFrame(rows),
            metric=analysis.LATEX_NO_SELF_LOOP_METRIC,
            caption="No-self-loop results.",
            label="tab:no_self_loops",
            appendix_label="tab:no_self_loops_appendix",
        )
    )
    assert "No-self-loop results." in table
    assert "\\label{tab:no_self_loops}" in table
    assert "Checkpoint & Env." in table
    assert "Single pass $\\Delta$ & $3\\times3$" in table
    assert "Table~\\ref{tab:no_self_loops_appendix}" in table
    assert table.count("\\colorbox{green!12}") == (
        len(analysis.MAIN_BODY_CONTRAST_IDS)
        * len(analysis.ENVIRONMENTS)
        * len(analysis.CHECKPOINTS)
    )


def test_main_body_delta_cells_encode_direction_and_significance() -> None:
    positive = pd.Series({"available": True, "mean_difference": 0.623, "p_holm": 0.001})
    negative = pd.Series({"available": True, "mean_difference": -0.084, "p_holm": 0.001})
    nonsignificant = pd.Series(
        {"available": True, "mean_difference": 0.021, "p_holm": 0.2}
    )
    negative_nonsignificant = pd.Series(
        {"available": True, "mean_difference": -0.006, "p_holm": 0.2}
    )
    assert analysis._latex_delta_cell(positive) == (
        r"\colorbox{green!12}{\strut\makebox[3.4em][r]{\textbf{+0.623}}}"
    )
    assert analysis._latex_delta_cell(negative) == (
        r"\colorbox{red!12}{\strut\makebox[3.4em][r]{\textbf{-0.084}}}"
    )
    assert analysis._latex_delta_cell(nonsignificant) == (
        r"\colorbox{green!4}{\strut\makebox[3.4em][r]{+0.021}}"
    )
    assert analysis._latex_delta_cell(negative_nonsignificant) == (
        r"\colorbox{red!4}{\strut\makebox[3.4em][r]{-0.006}}"
    )


def test_checkpoint_method_resolution_uses_standalone_then_initialised() -> None:
    assert analysis.resolve_checkpoint_method("mercury_derived", "checkpoint_2000") == "mercury"
    assert analysis.resolve_checkpoint_method("mercury_derived", "checkpoint_4000") == "mercury_initialised"
    assert analysis.resolve_checkpoint_method("mercury_derived", "final") == "mercury_initialised"
    assert analysis.resolve_checkpoint_method("tmgwr_derived", "checkpoint_2000") == "tmgwr"
    assert analysis.resolve_checkpoint_method("tmgwr_derived", "final") == "tmgwr_initialised"


def test_paired_statistics_match_scipy_and_student_t_ci() -> None:
    values_a = np.asarray([0.2, 0.5, 0.8, 0.9, 0.7])
    values_b = np.asarray([0.1, 0.4, 0.6, 0.8, 0.65])
    result = analysis.compute_paired_contrast(values_a, values_b)
    differences = values_a - values_b
    expected_test = stats.ttest_rel(values_a, values_b)
    expected_se = differences.std(ddof=1) / math.sqrt(len(differences))
    expected_margin = stats.t.ppf(0.975, len(differences) - 1) * expected_se

    assert result["n_pairs"] == 5
    assert np.isclose(result["mean_difference"], differences.mean())
    assert np.isclose(result["sd_difference"], differences.std(ddof=1))
    assert np.isclose(result["se_difference"], expected_se)
    assert np.isclose(result["ci95_low"], differences.mean() - expected_margin)
    assert np.isclose(result["ci95_high"], differences.mean() + expected_margin)
    assert np.isclose(result["t_statistic"], expected_test.statistic)
    assert np.isclose(result["p_raw"], expected_test.pvalue)
    assert np.isclose(result["cohens_dz"], differences.mean() / differences.std(ddof=1))


def test_constant_differences_have_exact_interval_and_defined_test() -> None:
    values_b = np.arange(5, dtype=float)
    result = analysis.compute_paired_contrast(values_b + 0.25, values_b)
    assert result["mean_difference"] == 0.25
    assert result["ci95_low"] == 0.25
    assert result["ci95_high"] == 0.25
    assert math.isinf(result["t_statistic"])
    assert result["p_raw"] == 0.0


def test_seed_pairing_reports_missing_unpaired_and_nonfinite_seeds() -> None:
    rows = [
        {"environment": 13, "checkpoint": "final", "method": "mercury_initialised", "seed": 0, "metric": 0.8},
        {"environment": 13, "checkpoint": "final", "method": "mercury_initialised", "seed": 1, "metric": np.nan},
        {"environment": 13, "checkpoint": "final", "method": "standard_cscg", "seed": 0, "metric": 0.6},
        {"environment": 13, "checkpoint": "final", "method": "standard_cscg", "seed": 2, "metric": 0.7},
    ]
    pairs, audit = analysis.build_paired_samples(
        pd.DataFrame(rows),
        metric="metric",
        environment=13,
        checkpoint="final",
        method_a="mercury_initialised",
        method_b="standard_cscg",
        expected_seeds=[0, 1, 2],
    )
    assert pairs["seed"].tolist() == [0]
    assert np.isclose(pairs.iloc[0]["difference"], 0.2)
    assert audit == {
        "missing_seeds_a": "2",
        "missing_seeds_b": "1",
        "unpaired_seeds_a": "1",
        "unpaired_seeds_b": "2",
        "nonfinite_seeds_a": "1",
        "nonfinite_seeds_b": "",
    }


def test_holm_correction_is_scoped_to_metric_specific_contrast_family() -> None:
    rows = []
    for family, p_values in (
        ("metric_a::contrast_1", [0.001, 0.02, 0.2]),
        ("metric_a::contrast_2", [0.01, 0.04]),
        ("metric_b::contrast_1", [0.03]),
    ):
        for p_value in p_values:
            rows.append(
                {
                    "holm_family": family,
                    "p_raw": p_value,
                    "p_holm": np.nan,
                    "significant_holm_0_05": False,
                    "holm_family_n_tests": 0,
                }
            )
    adjusted = analysis.apply_holm_correction(pd.DataFrame(rows))
    assert adjusted.groupby("holm_family")["holm_family_n_tests"].first().to_dict() == {
        "metric_a::contrast_1": 3,
        "metric_a::contrast_2": 2,
        "metric_b::contrast_1": 1,
    }
    expected = []
    for p_values in ([0.001, 0.02, 0.2], [0.01, 0.04], [0.03]):
        expected.extend(analysis.multipletests(p_values, method="holm")[1])
    assert np.allclose(adjusted["p_holm"], expected)


def test_duplicate_seed_rows_are_reported_before_plot_compatible_pooling() -> None:
    raw = pd.DataFrame(
        [
            {"environment": 13, "checkpoint": "final", "method": "standard_cscg", "seed": 0, "metric": 0.2},
            {"environment": 13, "checkpoint": "final", "method": "standard_cscg", "seed": 0, "metric": 0.4},
        ]
    )
    seed_level, duplicates = analysis.aggregate_to_seed_level(raw, ["metric"])
    assert len(duplicates) == 1
    assert duplicates.iloc[0]["row_count"] == 2
    assert np.isclose(seed_level.iloc[0]["metric"], 0.3)


def test_publication_p_value_formatting() -> None:
    assert analysis.format_p_value(0.0009) == "<0.001"
    assert analysis.format_p_value(0.0126) == "0.013"
    assert analysis.format_p_value(np.nan) == "NA"
