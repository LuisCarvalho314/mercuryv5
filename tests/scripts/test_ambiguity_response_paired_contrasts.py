from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = PROJECT_ROOT / "scripts" / "analysis"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import ambiguity_response_paired_contrasts as analysis


METRIC = analysis.main_stats.LATEX_PRIMARY_METRIC


def _seed_level(environments: tuple[int, ...] = (13, 21, 20, 22)) -> pd.DataFrame:
    rows = []
    parameters = {
        "full_mercury": (0.30, 0.010),
        "refinement_without_reconstruction": (0.20, 0.005),
        "local_repair_only": (0.10, 0.004),
        "no_ambiguity_response": (0.00, 0.003),
    }
    for environment in environments:
        for method, (offset, slope) in parameters.items():
            for seed in range(5):
                rows.append(
                    {
                        "environment": environment,
                        "checkpoint": "final",
                        "method": method,
                        "seed": seed,
                        METRIC: offset + seed * slope,
                    }
                )
    return pd.DataFrame(rows)


def test_all_contrasts_use_full_mercury_as_the_reference() -> None:
    results, pairs = analysis.compute_all_contrasts(
        _seed_level((13,)),
        metrics=[METRIC],
        environments=[13],
        expected_seeds=list(range(5)),
    )

    assert set(results["method_a"]) == {"full_mercury"}
    assert set(results["method_b"]) == set(analysis.COMPARATORS)
    assert results["n_pairs"].tolist() == [5, 5, 5]
    assert np.all(results["mean_difference"].to_numpy() > 0)
    assert set(pairs["method_a"]) == {"full_mercury"}
    assert np.all(pairs["difference"].to_numpy() > 0)


def test_holm_families_contain_the_four_environment_tests() -> None:
    results, _ = analysis.compute_all_contrasts(
        _seed_level(),
        metrics=[METRIC],
        environments=list(analysis.ENVIRONMENTS),
        expected_seeds=list(range(5)),
    )
    adjusted = analysis.main_stats.apply_holm_correction(results)

    assert adjusted.groupby("holm_family")["holm_family_n_tests"].first().eq(4).all()
    assert adjusted["holm_family"].nunique() == 3


def test_main_body_tables_contain_only_the_three_ablation_comparators() -> None:
    results, _ = analysis.compute_all_contrasts(
        _seed_level(),
        metrics=[METRIC],
        environments=list(analysis.ENVIRONMENTS),
        expected_seeds=list(range(5)),
    )
    results = analysis.main_stats.apply_holm_correction(results)
    table_options = {
        "metric": METRIC,
        "caption": "Action similarity.",
        "label": "tab:test",
        "appendix_label": "tab:test_appendix",
    }
    standard = "\n".join(
        analysis._main_body_table(results, transposed=False, **table_options)
    )
    transposed = "\n".join(
        analysis._main_body_table(results, transposed=True, **table_options)
    )

    for label in analysis.MODE_LABELS.values():
        if label != "Full MERCURY":
            assert label in standard
    assert "No reconstruction" in transposed
    assert "Local repair" in transposed
    assert "No response" in transposed
    assert "Standard CSCG" not in standard + transposed
    assert standard.count("\\colorbox{green!12}") == 12


def test_main_body_outputs_report_latent_purity(tmp_path: Path) -> None:
    seed_level = _seed_level()
    latex_metrics = [
        METRIC,
        analysis.main_stats.LATEX_NO_SELF_LOOP_METRIC,
        analysis.main_stats.LATEX_LATENT_METRIC,
    ]
    for metric in latex_metrics[1:]:
        seed_level[metric] = seed_level[METRIC]
    results, _ = analysis.compute_all_contrasts(
        seed_level,
        metrics=latex_metrics,
        environments=list(analysis.ENVIRONMENTS),
        expected_seeds=list(range(5)),
    )
    results = analysis.main_stats.apply_holm_correction(results)

    paths = analysis.write_latex_tables(results, tmp_path)
    main_body_paths = [path for path in paths if "main_body" in path.name]
    assert len(main_body_paths) == 2
    for path in main_body_paths:
        table = path.read_text(encoding="utf-8")
        assert "latent purity" in table
        assert "tab:ambiguity_response_ablation_purity_main" in table
        assert "tab:ambiguity_response_ablation_latent_purity" in table

    analysis.write_text_outputs(results, tmp_path)
    results_summary = (tmp_path / "statistical_results_summary.tex").read_text(
        encoding="utf-8"
    )
    assert results_summary.count("latent-purity") == 3


def test_validation_checks_common_walk_provenance() -> None:
    seed_level = _seed_level((13,))
    raw = seed_level.copy()
    source_rows = []
    for row in seed_level[["environment", "seed", "method"]].to_dict("records"):
        source_rows.append(
            {
                "level": row["environment"],
                "seed": row["seed"],
                "mode": row["method"],
                "record": {
                    "protocol_id": "canonical-test",
                    "walk_cache_id": f"cache-{row['seed']}",
                    "protocol": {"number_of_walks": 100, "walk_length": 10_000},
                },
            }
        )
    report, findings = analysis.validate_results(
        raw,
        seed_level,
        pd.DataFrame(),
        source_rows,
        metrics=[METRIC],
        environments=[13],
        expected_seeds=list(range(5)),
    )
    assert findings == []
    assert "Walk-cache mismatches: 0" in report

    source_rows[0]["record"]["walk_cache_id"] = "different-cache"
    report, findings = analysis.validate_results(
        raw,
        seed_level,
        pd.DataFrame(),
        source_rows,
        metrics=[METRIC],
        environments=[13],
        expected_seeds=list(range(5)),
    )
    assert any("walk_cache_mismatch" in finding for finding in findings)
    assert "Walk-cache mismatches: 1" in report
