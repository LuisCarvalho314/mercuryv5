from __future__ import annotations

import copy
import json

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pytest

from mercury_runs.analysis.spectral_transitions import (
    dominant_subspace, projected_operator, read_capacity_records, read_method_records,
    spectral_comparison, subspace_error,
)
from scripts.analysis.cscg_spectral_development import analyse_environment


def corridor():
    p = np.diag([.75, .5, .5, .75])
    p += np.diag([.25] * 3, 1) + np.diag([.25] * 3, -1)
    return p


def record(learned=None):
    gt = corridor()
    learned = gt if learned is None else learned
    full_gt = np.pad(gt, ((0, 0), (0, 1)))
    full = np.column_stack([learned, 1 - learned.sum(axis=1)])
    # Avoid tiny negative round-off in fixture probabilities.
    full[:, -1] = np.maximum(full[:, -1], 0)
    tv = np.abs(full - full_gt).sum(axis=1).mean() / 2
    return {
        "protocol_id": "test", "protocol": {"evaluation_seed": 0}, "method": "cscg", "level": 21,
        "snapshot_step": 0, "observed_samples": 0, "walk_cache_id": "shared", "ground_truth": {"state_order": list(range(4))},
        "state_representation": {"n_true_states": 4, "n_actions": 1, "n_learned_states": 4},
        "action_conditioned": {"P_true_full": [full_gt.tolist()], "P_hat_full": [full.tolist()],
                               "valid_rows": [[True] * 4], "source_action_counts": [[10] * 4],
                               "tv_similarity_occupancy": 1 - tv},
        "policy_collapsed": {"action_weights_evaluation_policy": [[1.] * 4],
                             "P_true": full_gt.tolist(), "P_hat": full.tolist()},
        "alignment": {"learned_to_true": list(range(4))}, "model_fit": {"bits_per_symbol": 1. + tv},
    }


def write_study(root):
    for preset in ("oracle_capacity", "vanilla_cscg"):
        for seed in range(2):
            run = root / f"{preset}_{seed}"
            run.mkdir(parents=True)
            (run / "run_config.json").write_text(json.dumps({"cli_args": {
                "method": "cscg", "level": 21, "seed": seed, "cscg_structural_preset": preset,
                "num_steps": 2000, "cscg_online_lambda": .99,
            }}))
            directory = run / "evaluations/test/snapshots/cscg"
            directory.mkdir(parents=True)
            for step, scale in ((0, .7), (2000, 1.)):
                r = record(scale * corridor())
                r["snapshot_step"] = step
                r["observed_samples"] = step
                (directory / f"step={step:08d}.json").write_text(json.dumps(r))


def test_corridor_reference_spectrum_and_leakage():
    p = corridor()
    stats, values, matched = spectral_comparison(p, p)
    expected = np.array([1., .5 + np.sqrt(2) / 4, .5, .5 - np.sqrt(2) / 4])
    np.testing.assert_allclose(values.real, expected, atol=1e-14)
    np.testing.assert_allclose(values, matched)
    assert stats["eigenvalue_rmse"] == 0
    assert stats["subspace_sin_max"] < 1e-14
    assert stats["modulus_gap"] == pytest.approx(1 - expected[1])
    r = record(.8 * p)
    original = copy.deepcopy(r)
    gt, learned, sink = projected_operator(r)
    np.testing.assert_allclose(learned, .8 * p)
    np.testing.assert_allclose(sink, .2)
    assert r == original
    leaking, _, _ = spectral_comparison(gt, learned)
    assert leaking["spectral_radius"] == pytest.approx(.8)
    assert leaking["eigenvalue_rmse"] > 0
    assert leaking["subspace_sin_max"] < 1e-14


def test_subspace_rotation_and_simultaneous_state_permutation():
    p = corridor()
    basis = dominant_subspace(p, 2)
    rotation = np.array([[1, 1j], [1j, 1]]) / np.sqrt(2)
    assert subspace_error(basis, basis @ rotation) < 1e-14
    learned = .8 * p + .2 * np.eye(4)
    a, _, _ = spectral_comparison(p, learned)
    perm = [2, 0, 3, 1]
    b, _, _ = spectral_comparison(p[np.ix_(perm, perm)], learned[np.ix_(perm, perm)])
    for key in a:
        assert a[key] == pytest.approx(b[key], abs=1e-12)


def test_complex_spectrum_and_degenerate_boundary_are_not_split():
    cycle = np.roll(np.eye(3), 1, axis=1)
    stats, values, matched = spectral_comparison(cycle, cycle)
    assert np.any(abs(values.imag) > .5)
    np.testing.assert_allclose(values, matched)
    assert not stats["subspace_defined"]
    assert np.isnan(stats["subspace_sin_max"])
    assert dominant_subspace(np.eye(4), 2) is None
    # Repeated eigenvalues inside a separated cluster remain comparable.
    p = np.diag([1., 1., .2, .1])
    assert subspace_error(dominant_subspace(p, 2), dominant_subspace(p, 2)) == 0


@pytest.mark.parametrize("damage", ["support", "policy", "mass"])
def test_projection_rejects_missing_support_or_hidden_mass(damage):
    r = record()
    if damage == "support":
        r["action_conditioned"]["source_action_counts"][0][0] = 0
    elif damage == "policy":
        r["policy_collapsed"]["action_weights_evaluation_policy"][0][0] = .5
    else:
        r["action_conditioned"]["P_hat_full"][0][0][0] = .1
    with pytest.raises(ValueError):
        projected_operator(r)


@pytest.mark.parametrize("field", ["walk_cache_id", "ground_truth", "protocol"])
def test_reader_rejects_incompatible_comparisons(tmp_path, field):
    write_study(tmp_path)
    path = next(tmp_path.glob("*/evaluations/test/snapshots/cscg/step=00002000.json"))
    r = json.loads(path.read_text())
    r[field] = "different" if field == "walk_cache_id" else {"different": True}
    path.write_text(json.dumps(r))
    with pytest.raises(ValueError, match="Incompatible"):
        read_capacity_records(tmp_path, "test", 21)


def test_record_to_figures_preserves_metric_values_and_pairs(tmp_path):
    study = tmp_path / "study"
    write_study(study)
    changed_path = study / "oracle_capacity_0/evaluations/test/snapshots/cscg/step=00002000.json"
    changed = json.loads(changed_path.read_text())
    changed["alignment"]["learned_to_true"] = [1, 0, 2, 3]
    changed_path.write_text(json.dumps(changed))
    output = tmp_path / "figures"
    frame = analyse_environment(study, "test", 21, output)
    assert len(frame) == 8
    assert (frame.loc[frame.snapshot_step == 2000, "action_tv_similarity"] == 1).all()
    assert (frame.loc[frame.snapshot_step == 2000, "nll_bits_per_symbol"] == 1).all()
    assert frame.loc[frame.snapshot_step == 0, "alignment_change_fraction"].isna().all()
    assert sorted(frame.loc[frame.snapshot_step == 2000, "alignment_change_fraction"]) == [0, 0, 0, .5]
    for name in ("spectral_trajectories", "eigenvalue_spectra", "spectral_relationships", "transition_pca"):
        assert (output / f"{name}.png").stat().st_size > 1000
        assert (output / f"{name}.pdf").stat().st_size > 1000
    with np.load(output / "operators_and_pca.npz") as arrays:
        recovered = (arrays["learned_vectors"] - arrays["feature_mean"]) @ arrays["components"].T
        np.testing.assert_allclose(frame[["pc1", "pc2"]], recovered, atol=1e-12)


def write_hybrids(root):
    write_study(root)
    for model, suffix in (("mercury", "mercury"), ("tmgwr", "tmgwr")):
        method = f"{model}_cscg_hybrid"
        for seed in range(2):
            run = root / f"{model}_{seed}"
            run.mkdir()
            (run / "run_config.json").write_text(json.dumps({"cli_args": {
                "method": method, "level": 21, "seed": seed, "num_steps": 2000,
                "cscg_structural_preset": "source_graph_clone_state",
            }}))
            for namespace, step, phase_samples in ((f"{method}_{suffix}", 2000, None),
                                                    (method, 0, 0), (method, 2000, 2000),
                                                    (method, 2000000, 1998000)):
                r = record(.8 * corridor())
                r.update(method=namespace, snapshot_step=step, observed_samples=step)
                if phase_samples is None:
                    r.pop("model_fit")
                    r["alignment"]["learned_to_true"] = [1, 0, 2, 3]
                else:
                    r["observed_samples"] = phase_samples
                    r["training_budget"] = {"source_samples": 2000, "cscg_phase_samples": phase_samples,
                                            "cumulative_samples": 2000 + phase_samples}
                directory = run / "evaluations/test/snapshots" / namespace
                directory.mkdir(parents=True, exist_ok=True)
                (directory / f"step={step:08d}.json").write_text(json.dumps(r))


def test_staged_models_preserve_clocks_missing_nll_and_transfer_boundaries(tmp_path):
    import pandas as pd

    study = tmp_path / "study"
    write_hybrids(study)
    output = tmp_path / "output"
    frame = analyse_environment(study, "test", 21, output, all_methods=True)
    assert len(frame) == 24
    assert set(frame.preset) == {"mercury", "tmgwr", "oracle_capacity", "vanilla_cscg"}
    assert frame.loc[frame.phase == "source", "nll_bits_per_symbol"].isna().all()
    assert (frame.loc[frame.phase == "handoff", "checkpoint_step"] == 2000).all()
    assert frame.loc[frame.phase == "handoff", "alignment_change_fraction"].isna().all()
    assert (frame.loc[(frame.phase == "hybrid") & (frame.snapshot_step == 2000000), "checkpoint_step"] == 2000000).all()
    changes = pd.read_csv(output / "interval_changes.csv")
    transfer = changes[changes.from_phase == "source"]
    assert len(transfer) == 4
    assert (transfer.to_phase == "handoff").all()
    assert (transfer.from_step == transfer.to_step).all()
    assert transfer.delta_nll_bits_per_symbol.isna().all()
    assert transfer.alignment_change_fraction.isna().all()
    assert transfer.subspace_step_sin_max.notna().all()
    correlations = pd.read_csv(output / "checkpoint_associations.csv")
    source_nll = correlations[(correlations.phase == "source") & (correlations.y == "nll_bits_per_symbol")]
    assert (source_nll.n_pairs == 0).all()
    assert source_nll.spearman_rho.isna().all()
    source_tv = correlations[(correlations.phase == "source") & (correlations.y == "action_tv_similarity")]
    assert (source_tv.n_pairs == 2).all()


@pytest.mark.parametrize("damage", ["budget", "missing_source"])
def test_staged_reader_rejects_wrong_clocks_and_missing_sources(tmp_path, damage):
    write_hybrids(tmp_path)
    run = tmp_path / "mercury_0/evaluations/test/snapshots"
    if damage == "budget":
        path = run / "mercury_cscg_hybrid/step=02000000.json"
        r = json.loads(path.read_text())
        r["training_budget"]["cumulative_samples"] += 2000
        path.write_text(json.dumps(r))
    else:
        (run / "mercury_cscg_hybrid_mercury/step=00002000.json").unlink()
    with pytest.raises(ValueError):
        read_method_records(tmp_path, "test", 21)
