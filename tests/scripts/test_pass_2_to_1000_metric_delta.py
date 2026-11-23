from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from scripts.analysis.plot_pass_2_to_1000_metric_delta import (
    NLL_METRIC,
    PRIMARY_METRIC,
    _latex_main_before_after_table,
    SECONDARY_METRIC,
    _latex_no_self_loop_matrix,
    _latex_paired_change_and_association_table,
    condition_metric_response_frame,
    condition_response_frame,
    main_before_after_frame,
    paired_absolute_values,
    paired_improvement_deltas,
    paired_predictor_values,
    plot_paired_absolute_values,
    plot_condition_level_response,
    predictor_correlations,
    publication_table_frame,
    seed_response_frame,
    summarise_paired_changes,
    three_checkpoint_seed_frame,
)
from scripts.analysis.plot_config_metric_trajectories import rows_at_comparison_steps


def _row(*, step: int, metric: str, value: float) -> dict[str, object]:
    return {
        "run_id": "run-1",
        "series": "vanilla_cscg",
        "level": 13,
        "seed": 0,
        "metric": metric,
        "step": step,
        "value": value,
    }


def test_paired_improvement_deltas_orients_all_improvements_positive() -> None:
    rows = pd.DataFrame(
        [
            _row(step=4_000, metric="latent_precision", value=0.4),
            _row(step=2_000_000, metric="latent_precision", value=0.7),
            _row(step=4_000, metric="eval_nll_bps", value=2.5),
            _row(step=2_000_000, metric="eval_nll_bps", value=1.5),
        ]
    )

    deltas = paired_improvement_deltas(rows).set_index("metric")

    assert deltas.loc["latent_precision", "delta"] == pytest.approx(0.3)
    assert deltas.loc["eval_nll_bps", "delta"] == pytest.approx(1.0)
    assert deltas.loc["latent_precision", "value_pass_2"] == pytest.approx(0.4)
    assert deltas.loc["latent_precision", "value_pass_1000"] == pytest.approx(0.7)
    assert deltas.loc["eval_nll_bps", "raw_change"] == pytest.approx(-1.0)


def test_paired_improvement_deltas_rejects_missing_endpoint() -> None:
    rows = pd.DataFrame([_row(step=4_000, metric="latent_precision", value=0.4)])

    with pytest.raises(ValueError, match="Missing pass-2 or pass-1000 endpoint"):
        paired_improvement_deltas(rows)


def test_three_checkpoint_seed_frame_aligns_initialisation_pass_2_and_pass_1000() -> None:
    rows = pd.DataFrame(
        [
            _row(step=step, metric=metric, value=value)
            for metric, values in (
                (NLL_METRIC, (3.0, 2.0, 1.0)),
                (PRIMARY_METRIC, (0.2, 0.5, 0.8)),
            )
            for step, value in zip((0, 4_000, 2_000_000), values)
        ]
    )

    trajectory = three_checkpoint_seed_frame(rows, metric=PRIMARY_METRIC).iloc[0]

    assert trajectory["nll_initialisation"] == pytest.approx(3.0)
    assert trajectory["nll_pass_2"] == pytest.approx(2.0)
    assert trajectory["nll_pass_1000"] == pytest.approx(1.0)
    assert trajectory["quality_initialisation"] == pytest.approx(0.2)
    assert trajectory["quality_pass_2"] == pytest.approx(0.5)
    assert trajectory["quality_pass_1000"] == pytest.approx(0.8)


def test_matched_budget_hybrid_uses_total_samples_at_pass_2_and_final() -> None:
    protocol_id = "evaluation-v5-nonself-matched-budget-v1-test"
    records = []
    for series, checkpoints in (
        ("vanilla_cscg", ((0, 0), (2_000, 2_000), (4_000, 4_000), (2_000_000, 2_000_000))),
        ("mercury_cscg_hybrid", ((0, 0), (2_000, 4_000), (2_000_000, 2_000_000))),
    ):
        for phase_step, total_step in checkpoints:
            for metric in (NLL_METRIC, PRIMARY_METRIC):
                value = float(total_step if metric == NLL_METRIC else total_step / 2_000_000)
                if series == "mercury_cscg_hybrid" and phase_step == 0:
                    value = 2.5 if metric == NLL_METRIC else 0.35
                records.append(
                    {
                        "run_id": series,
                        "series": series,
                        "level": 13,
                        "seed": 0,
                        "metric": metric,
                        "step": phase_step,
                        "comparison_step": total_step,
                        "value": value,
                    }
                )
    raw = pd.DataFrame(records)
    aligned = rows_at_comparison_steps(raw, protocol_id)
    assert rows_at_comparison_steps(raw, "evaluation-v5-nonself-v2-test") is raw
    assert set(aligned.loc[aligned["series"] == "mercury_cscg_hybrid", "step"]) == {
        0, 4_000, 2_000_000
    }
    paired = paired_improvement_deltas(aligned)
    hybrid = paired[(paired["series"] == "mercury_cscg_hybrid") & (paired["metric"] == PRIMARY_METRIC)].iloc[0]
    assert hybrid["value_pass_2"] == pytest.approx(4_000 / 2_000_000)
    trajectory = three_checkpoint_seed_frame(
        aligned,
        metric=PRIMARY_METRIC,
        pass_1_step=2_000,
        allow_missing_pass_1_series={"mercury_cscg_hybrid"},
    ).set_index("series")
    assert np.isnan(trajectory.loc["mercury_cscg_hybrid", "quality_pass_1"])
    assert trajectory.loc["mercury_cscg_hybrid", "nll_initialisation"] == pytest.approx(2.5)
    assert trajectory.loc["mercury_cscg_hybrid", "quality_initialisation"] == pytest.approx(0.35)
    assert trajectory.loc["mercury_cscg_hybrid", "quality_pass_2"] == pytest.approx(4_000 / 2_000_000)
    assert trajectory.loc["vanilla_cscg", "quality_pass_1"] == pytest.approx(2_000 / 2_000_000)


def test_summary_retains_absolute_means_and_paired_student_t_interval() -> None:
    changes = np.asarray([0.1, 0.2, 0.0, 0.3, 0.15])
    paired = pd.DataFrame(
        {
            "run_id": [f"run-{seed}" for seed in range(5)],
            "series": "vanilla_cscg",
            "level": 13,
            "seed": range(5),
            "metric": PRIMARY_METRIC,
            "value_pass_2": [0.4] * 5,
            "value_pass_1000": 0.4 + changes,
            "raw_change": changes,
            "change": changes,
            "delta": changes,
        }
    )
    row = summarise_paired_changes(paired).iloc[0]
    se = changes.std(ddof=1) / np.sqrt(len(changes))
    margin = stats.t.ppf(0.975, len(changes) - 1) * se

    assert row["mean_pass_2"] == pytest.approx(0.4)
    assert row["mean_pass_1000"] == pytest.approx((0.4 + changes).mean())
    assert row["mean_change"] == pytest.approx(changes.mean())
    assert row["ci95_low"] == pytest.approx(changes.mean() - margin)
    assert row["ci95_high"] == pytest.approx(changes.mean() + margin)
    assert row["assessment"] == "improved"


def _predictor_pairs() -> pd.DataFrame:
    rows = []
    for series, metric_changes in (
        ("vanilla_cscg", [0.1, 0.2, 0.3, 0.4]),
        ("oracle_capacity", [0.4, 0.3, 0.2, 0.1]),
    ):
        for seed, metric_change in enumerate(metric_changes):
            common = {
                "run_id": f"{series}-{seed}",
                "series": series,
                "level": 13,
                "seed": seed,
            }
            rows.append(
                {
                    **common,
                    "metric": NLL_METRIC,
                    "change": float(seed + 1),
                }
            )
            rows.append(
                {
                    **common,
                    "metric": PRIMARY_METRIC,
                    "change": metric_change,
                }
            )
    return pd.DataFrame(rows)


def test_spearman_correlations_are_computed_within_each_initialisation() -> None:
    paired = _predictor_pairs()
    aligned = paired_predictor_values(paired, metric=PRIMARY_METRIC)
    correlations = predictor_correlations(paired, metrics=[PRIMARY_METRIC])
    values = correlations.set_index("series")["rho_s"]

    assert len(aligned) == 8
    assert aligned.groupby("series").size().tolist() == [4, 4]
    assert aligned.loc[aligned["series"] == "vanilla_cscg", "seed"].tolist() == [0, 1, 2, 3]
    assert values["vanilla_cscg"] == pytest.approx(1.0)
    assert values["oracle_capacity"] == pytest.approx(-1.0)
    assert correlations["n_pairs"].tolist() == [4, 4]


def test_spearman_reports_constant_metric_change_as_unavailable() -> None:
    paired = _predictor_pairs()
    paired.loc[
        (paired["series"] == "vanilla_cscg") & (paired["metric"] == PRIMARY_METRIC),
        "change",
    ] = 0.0
    row = predictor_correlations(paired, metrics=[PRIMARY_METRIC]).loc[
        lambda frame: frame["series"] == "vanilla_cscg"
    ].iloc[0]

    assert not row["available"]
    assert np.isnan(row["rho_s"])
    assert row["unavailable_reason"] == "constant metric change"


def test_paired_absolute_values_plot_writes_png_and_pdf(tmp_path: Path) -> None:
    changes = _predictor_pairs()
    paired = changes.assign(
        value_pass_2=np.where(changes["metric"] == NLL_METRIC, 2.0, 0.4),
        value_pass_1000=lambda frame: np.where(
            frame["metric"] == NLL_METRIC,
            frame["value_pass_2"] - frame["change"],
            frame["value_pass_2"] + frame["change"],
        ),
    )
    aligned = paired_absolute_values(paired)
    seed_values = seed_response_frame(paired, metric=PRIMARY_METRIC)

    paths = plot_paired_absolute_values(
        paired,
        tmp_path,
        output_stem="test",
    )

    assert len(aligned) == 8
    assert len(seed_values) == 8
    assert {
        "quality_pass_2",
        "quality_pass_1000",
        "quality_change",
        "nll_pass_2",
        "nll_pass_1000",
        "nll_improvement",
    } <= set(seed_values)
    assert {"nll_pass_2", "nll_pass_1000", "metric_pass_2", "metric_pass_1000"} <= set(aligned)
    assert [path.suffix for path in paths] == [".png", ".pdf"]
    assert all(path.exists() and path.stat().st_size > 0 for path in paths)


def test_primary_presentation_separates_before_after_changes_and_correlations(
    tmp_path: Path,
) -> None:
    summary_rows = []
    correlation_rows = []
    from scripts.analysis.plot_pass_2_to_1000_metric_delta import (
        DEFAULT_LEVEL_ORDER,
        SERIES_ORDER,
    )

    for series in SERIES_ORDER:
        for level in DEFAULT_LEVEL_ORDER:
            for metric, before, after, change in (
                (PRIMARY_METRIC, 0.4, 0.5, 0.1),
                (SECONDARY_METRIC, 0.6, 0.7, 0.1),
                (NLL_METRIC, 2.0, 1.5, 0.5),
            ):
                summary_rows.append(
                    {
                        "series": series,
                        "level": level,
                        "metric": metric,
                        "mean_pass_2": before,
                        "mean_pass_1000": after,
                        "mean_change": change,
                        "ci95_low": change - 0.02,
                        "ci95_high": change + 0.02,
                        "assessment": "improved",
                        "n": 30,
                    }
                )
            correlation_rows.append(
                {
                    "series": series,
                    "level": level,
                    "metric": PRIMARY_METRIC,
                    "rho_s": 0.25,
                }
            )
            correlation_rows.append(
                {
                    "series": series,
                    "level": level,
                    "metric": SECONDARY_METRIC,
                    "rho_s": 0.15,
                }
            )
    summary = pd.DataFrame(summary_rows)
    correlations = pd.DataFrame(correlation_rows)
    detailed = publication_table_frame(summary, correlations, metric=PRIMARY_METRIC)
    main = main_before_after_frame(summary, correlations)
    main_latex = "\n".join(_latex_main_before_after_table(summary, correlations))
    analysis_latex = "\n".join(
        _latex_paired_change_and_association_table(summary, correlations)
    )
    no_self_latex = "\n".join(_latex_no_self_loop_matrix(summary, correlations))
    condition_response = condition_response_frame(summary)
    no_self_response = condition_metric_response_frame(summary, metric=SECONDARY_METRIC)
    figure_paths = plot_condition_level_response(
        condition_response, tmp_path, output_stem="test"
    )

    assert len(detailed) == len(SERIES_ORDER) * len(DEFAULT_LEVEL_ORDER)
    assert main["metric_pass_2"].eq(0.4).all()
    assert main["metric_pass_1000"].eq(0.5).all()
    assert main["nll_pass_2"].eq(2.0).all()
    assert main["nll_pass_1000"].eq(1.5).all()
    assert "rho_s" not in main
    assert "ci95_low" not in main
    assert "$S^A_{\\mathrm{TV}}\\ (\\uparrow)$" in main_latex
    assert "Evaluation NLL $(\\downarrow)$" in main_latex
    assert "$0.400\\!\\rightarrow\\!0.500$" in main_latex
    assert ";\\," not in main_latex
    assert "^{*}" not in main_latex
    assert "rho_s" not in main_latex
    assert "$I_S$ [95\\% CI]" in analysis_latex
    assert "$I_{\\mathrm{NLL}}$ [95\\% CI]" in analysis_latex
    assert "Spearman $\\rho(I_{\\mathrm{NLL}},I_S)$" in analysis_latex
    assert "+0.250" in analysis_latex
    assert "initialisations are not pooled" in analysis_latex
    assert "$0.500\\,[0.480,0.520]$" in analysis_latex
    assert "[+0.480" not in analysis_latex
    assert "$0.600\\!\\rightarrow\\!0.700$" in no_self_latex
    assert ";\\," not in no_self_latex
    assert "^{*}" not in no_self_latex
    assert len(condition_response) == len(SERIES_ORDER) * len(DEFAULT_LEVEL_ORDER)
    assert condition_response["transition_similarity_pass_2"].eq(0.4).all()
    assert condition_response["structural_change"].eq(0.1).all()
    assert condition_response["nll_improvement"].eq(0.5).all()
    assert len(no_self_response) == len(condition_response)
    assert no_self_response["quality_pass_2"].eq(0.6).all()
    assert all(path.exists() and path.stat().st_size > 0 for path in figure_paths)
