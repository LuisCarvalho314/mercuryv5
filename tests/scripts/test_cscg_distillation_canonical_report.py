import copy

import numpy as np
import pandas as pd
import pytest

from scripts.analysis.cscg_distillation_canonical_report import (
    DEFAULT_EDGE_THRESHOLDS, METRICS, canonical_metrics, final_f1_curve, summarise,
)


def record():
    return dict(
        state_representation=dict(latent_precision=.9, node_recall=.8),
        alignment=dict(cooccurrence=[[5, 0], [0, 3], [1, 1]], learned_to_true=[0, 1, 0]),
        model_fit=dict(bits_per_symbol=.4),
        action_conditioned=dict(
            P_true_full=[[[.5, .5, 0], [.25, .75, 0]]],
            P_hat_full=[[[.1, .1, .8], [.5, .25, .25]]], valid_rows=[[True, True]],
            tv_occupancy=.65, tv_similarity_occupancy=.35,
            tv_non_self_loop_occupancy=11/18, tv_non_self_loop_similarity_occupancy=7/18,
            total_unmatched_target_mass=.525))


@pytest.mark.parametrize("include_self", [True, False])
def test_reference_f1_threshold_auc_includes_sink_and_does_not_renormalize(include_self):
    r = record()
    scores = canonical_metrics(r)
    truth = np.asarray(r["action_conditioned"]["P_true_full"])
    prediction = np.asarray(r["action_conditioned"]["P_hat_full"])
    mask = np.ones_like(truth, dtype=bool)
    if not include_self:
        mask[0, np.arange(2), np.arange(2)] = False
    values = []
    for threshold in DEFAULT_EDGE_THRESHOLDS:
        gt, pred = (truth > threshold) & mask, (prediction > threshold) & mask
        denominator = gt.sum() + pred.sum()
        values.append(2 * (gt & pred).sum() / denominator if denominator else 0.)
    expected = np.trapezoid(values, DEFAULT_EDGE_THRESHOLDS) / .5
    metric = "action_edge_f1_auc" if include_self else "action_no_self_loop_edge_f1_auc"
    assert scores[metric] == pytest.approx(expected)
    # Removing unmatched mass changes the canonical F1, rather than being ignored.
    removed = copy.deepcopy(r)
    removed["action_conditioned"]["P_hat_full"][0][0][-1] = 0
    assert canonical_metrics(removed)[metric] != pytest.approx(scores[metric])


def test_similarity_is_canonical_complement_and_recall_is_not_state_coverage():
    r = record()
    scores = canonical_metrics(r)
    assert scores["action_tv_similarity"] == pytest.approx(1 - .65)
    assert scores["action_no_self_loop_tv_similarity"] == pytest.approx(7/18)
    assert scores["latent_recall"] == .8
    assert scores["state_coverage"] == 1
    r["action_conditioned"]["tv_similarity_occupancy"] = .7
    with pytest.raises(ValueError, match="disagree"):
        canonical_metrics(r)


def test_temporal_mean_is_distinct_from_threshold_auc_and_has_correct_direction():
    rows = []
    for step in [0, 40, 100, 2000]:
        row = dict(iterations=5, level=21, seed=0, method="predictive", step=step, train_seconds=step/1000)
        row.update({metric: step/2000 for metric in METRICS})
        rows.append(row)
    summary = summarise(pd.DataFrame(rows)).iloc[0]
    assert summary["final_action_edge_f1_auc"] == 1
    assert summary["time_mean_action_edge_f1_auc"] == pytest.approx(.5)
    assert summary["time_mean_action_tv_similarity"] == pytest.approx(.5)


def test_final_step_f1_curve_contains_scores_at_every_threshold_not_auc():
    r = record()
    curve = final_f1_curve(r)
    assert len(curve) == 2 * len(DEFAULT_EDGE_THRESHOLDS)
    with_self = {row["threshold"]: row["edge_f1"] for row in curve if row["include_self_loops"]}
    without_self = {row["threshold"]: row["edge_f1"] for row in curve if not row["include_self_loops"]}
    # At 0.2, predicted edges are sink(0), target(1->0), self(1), sink(1).
    # Two of four true edges are recovered: F1 = 4 / (4 + 4).
    assert with_self[.2] == pytest.approx(.5)
    # Removing diagonal edges leaves two true and three predicted edges, one TP.
    assert without_self[.2] == pytest.approx(.4)
    assert with_self[.2] != pytest.approx(canonical_metrics(r)["action_edge_f1_auc"])
