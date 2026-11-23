from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mercury_runs.analysis.spatial_spectra import (
    spatial_laplacian, spatial_representation, align_spatial_modes, compare_spatial,
)
from scripts.analysis import plot_spatial_spectral_development as plotting


def ring():
    return .5 * np.eye(4) + .25 * np.roll(np.eye(4), 1, axis=1) + .25 * np.roll(np.eye(4), -1, axis=1)


def test_symmetric_operator_preserves_leakage_and_input():
    p = ring()
    original = p.copy()
    laplacian = spatial_laplacian(p)
    np.testing.assert_allclose(laplacian, np.eye(4) - p)
    leaking = spatial_laplacian(.8 * p)
    np.testing.assert_allclose(leaking, np.eye(4) - .8 * p)
    np.testing.assert_array_equal(p, original)
    np.testing.assert_allclose(leaking, leaking.T)
    assert np.linalg.eigvalsh(leaking).min() >= 0
    reference = spatial_representation(p)
    learned = spatial_representation(.8 * p)
    assert compare_spatial(reference, learned)["symmetric_eigenvalue_rmse"] > 0
    assert not np.allclose(reference["distances"], learned["distances"])


def test_degenerate_rotations_do_not_change_maps_distances_or_subspace(monkeypatch):
    p = ring()
    reference = spatial_representation(p, modes=2)
    original_eigh = np.linalg.eigh
    rotation = np.array([[.6, -.8], [.8, .6]])

    def rotated_eigh(matrix):
        values, vectors = original_eigh(matrix)
        assert values[1] == pytest.approx(values[2])
        vectors[:, 1:3] = vectors[:, 1:3] @ rotation
        vectors[:, 0] *= -1
        return values, vectors

    monkeypatch.setattr(np.linalg, "eigh", rotated_eigh)
    rotated = spatial_representation(p, modes=2)
    np.testing.assert_allclose(align_spatial_modes(rotated["basis"], reference["basis"]), reference["basis"], atol=1e-14)
    np.testing.assert_allclose(rotated["distances"], reference["distances"], atol=1e-14)
    assert compare_spatial(reference, rotated)["symmetric_subspace_error"] < 1e-14


def test_tied_boundaries_are_unresolved_and_distance_has_no_display_cutoff():
    p = ring()
    cut = spatial_representation(p, modes=1)
    whole = spatial_representation(p, modes=2)
    assert cut["basis"] is None
    assert whole["basis"] is not None
    np.testing.assert_allclose(cut["distances"], whole["distances"])
    disconnected = spatial_representation(np.eye(4))
    assert disconnected["basis"] is None
    assert disconnected["distances"] is None
    assert align_spatial_modes(None, whole["basis"]) is None


def test_spatial_distances_follow_simultaneous_state_permutations():
    p = ring()
    perm = [2, 0, 3, 1]
    original = spatial_representation(p)
    changed = spatial_representation(p[np.ix_(perm, perm)])
    np.testing.assert_allclose(original["eigenvalues"], changed["eigenvalues"], atol=1e-14)
    np.testing.assert_allclose(original["distances"][np.ix_(perm, perm)], changed["distances"], atol=1e-14)


@pytest.mark.parametrize("level,n,shape", [(13, 8, (3, 3)), (21, 4, (1, 4)), (20, 5, (1, 5)), (22, 12, (3, 4))])
def test_all_environment_layouts_respect_gt_state_order(level, n, shape):
    states = np.arange(n)
    grid = plotting._layout(states, level, states)
    assert grid.shape == shape
    assert sorted(grid[np.isfinite(grid)]) == list(states)
    assert np.isnan(grid).sum() == (1 if level == 13 else 0)
    np.testing.assert_array_equal(plotting._layout(states[::-1], level, states[::-1]), grid)


@pytest.mark.parametrize("matrix", [np.ones((3, 4)), np.full((3, 3), np.nan), -np.eye(3), 2 * np.eye(3)])
def test_invalid_spatial_operators_are_rejected(matrix):
    with pytest.raises(ValueError):
        spatial_laplacian(matrix)


def test_spatial_figures_keep_source_handoff_and_canonical_tv(tmp_path, monkeypatch):
    samples = []
    for model in plotting.MODELS:
        checkpoints = [(2000, "source"), (2000, "handoff"), (4000, "hybrid"), (2000000, "hybrid")] if model == "mercury" else [(0, "cscg"), (2000, "cscg"), (4000, "cscg"), (2000000, "cscg")]
        for seed in (0, 1):
            for step, phase in checkpoints:
                p = ring() if step else np.ones((4, 4)) / 4
                samples.append({"condition": model, "checkpoint_step": step, "phase": phase,
                                "truth": ring(), "operator": p, "leakage": np.zeros(4),
                                "run_id": f"{model}-{seed}", "config": {"seed": seed}, "record_path": "fixture",
                                "record": {"snapshot_step": step, "ground_truth": {"state_order": list(range(4))},
                                           "action_conditioned": {"tv_similarity_occupancy": .75}}})
    monkeypatch.setattr(plotting, "read_method_records", lambda *args: samples)
    frame = plotting.run(tmp_path, "test", 21, tmp_path / "output")
    assert len(frame) == 24
    assert (frame.action_tv_similarity == .75).all()
    assert set(frame.loc[frame.model == "mercury", "column"]) == {1, 2, 3, 4}
    assert not frame.loc[frame.checkpoint_step == 0, "modes_defined"].any()
    for figure in ("eigenvector_heatmaps", "spectral_convergence", "spectral_distance_matrices", "spectral_distance_residuals", "state_layout"):
        for suffix in ("png", "pdf"):
            assert (tmp_path / "output" / f"{figure}.{suffix}").stat().st_size > 1000
    saved = pd.read_csv(tmp_path / "output/spatial_metrics.csv")
    assert len(saved) == len(frame)
    with np.load(tmp_path / "output/spatial_representations.npz") as arrays:
        assert arrays["aligned_modes"].shape == (24, 4, 2)
        assert arrays["distances"].shape == (24, 4, 4)
