import json
import sys

import matplotlib.figure
import pandas as pd
import pytest

from scripts.analysis import cscg_distillation_allocation_report as report


@pytest.mark.parametrize("level, capacity", [(21, [1, 1, 2]), (23, [1, 1, 6, 1, 3, 1, 6, 3, 18])])
@pytest.mark.parametrize("observations", [2, 20000])
def test_single_cohort_report_needs_no_unmatched_fixed_capacity_runs(tmp_path, monkeypatch, level, capacity, observations):
    root = tmp_path / "ground_truth"
    root.mkdir()
    config = dict(clone_allocation="ground_truth", seeds=[0, 1], levels=[level],
                  observations=observations, checkpoints=[0, observations], methods=list(report.METHODS))
    (root / "experiment.json").write_text(json.dumps(config))
    (root / "implementation.json").write_text("{}")
    for seed in config["seeds"]:
        for method in report.METHODS:
            folder = root / f"level={level}/seed={seed}" / method
            (folder / "evaluations").mkdir(parents=True)
            (folder / "training.json").write_text(json.dumps(dict(provenance=dict(clone_allocation=capacity))))
            rows = []
            for step in config["checkpoints"]:
                rows.append(dict(level=level, seed=seed, method=method, step=step, train_seconds=step/10))
                score = .7 if method == "predictive" else .5
                record = dict(level=level, seed=seed, method=method, snapshot_step=step, score=score)
                (folder / "evaluations" / f"step={step:04d}.json").write_text(json.dumps(record))
            (folder / "rows.json").write_text(json.dumps(rows))
    monkeypatch.setattr(report, "canonical_metrics", lambda record: {metric: record["score"] for metric in report.METRICS})
    monkeypatch.setattr(report, "final_f1_curve", lambda record: [dict(threshold=.2, include_self_loops=keep,
                                                                     edge_f1=record["score"]) for keep in [True, False]])
    monkeypatch.setattr(report, "residual_plots", lambda *args, **kwargs: None)
    plot_labels = []
    monkeypatch.setattr(matplotlib.figure.Figure, "savefig",
                        lambda figure, *args, **kwargs: plot_labels.extend(ax.get_ylabel() for ax in figure.axes))
    monkeypatch.setattr(sys, "argv", ["report", "--ground-truth-only", "--ground-truth", str(root),
                                     "--fixed", str(tmp_path / "nonexistent_fixed_cohort")])
    report.main()
    output = root / "canonical_metrics"
    seeds = pd.read_csv(output / "per_seed_summary.csv")
    assert len(seeds) == 10
    assert f"Edge F1 at observation {observations:,}" in plot_labels
    assert set(seeds.allocation) == {"ground_truth"}
    provenance = json.loads((output / "provenance.json").read_text())
    assert provenance["fixed"] is None
    assert provenance["seeds"] == [0, 1]
    assert provenance["records"] == 20
    assert f"| ground_truth | {level} | {sum(capacity)} |" in (output / "results.md").read_text()
    differences = pd.read_csv(output / "per_seed_paired_differences.csv")
    assert "ground_truth_minus_fixed" not in set(differences.comparison)
    paired = pd.read_csv(output / "paired_summary.csv")
    scores = paired[(paired.method == "predictive") & (paired.comparison == "method_minus_long")]
    structural = scores[scores.metric == "final_action_tv_similarity"].iloc[0]
    assert structural["mean"] == pytest.approx(.2)
    assert structural["count"] == 2 and structural.wins == 2
    nll = scores[scores.metric == "final_eval_nll_bps"].iloc[0]
    assert nll.losses == 2  # Higher NLL is worse, despite the positive raw difference.
