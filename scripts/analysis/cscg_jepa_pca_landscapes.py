"""Structural evaluation landscapes over the CSCG vs EM-JEPA projected-transition PCA plane.

Reuses cscg_jepa_pca_trajectories.load_case unchanged in substance: the same records, the same joint
per-level PCA (all arms, teachers, students, seeds, checkpoints) and the same GT placement. Nothing is
trained, decoded or re-projected; metrics come from the canonical paper_precision functions applied to
projected tensors P_hat_full (shape actions x GT states x (GT states + sink)).

Two landscapes per level and metric:
  * inverse-PCA: each point of a 100 x 100 grid is mapped back with pca.inverse_transform to a
    hypothetical projected tensor and scored against P_true_full. Points whose tensor is not a valid
    set of probability rows (negative or >1 entries, valid rows not summing to 1, invalid rows non-zero)
    are masked, never clipped. Frobenius distance needs no probability constraint and is never masked.
  * corrected inverse-PCA: every sample's tensor gets one Euclidean projection of each valid
    (action, state) row onto the probability simplex over GT states + sink (invalid rows set to zero, as
    stored), is reprojected once with the same PCA, scored, and linearly interpolated over a Delaunay
    triangulation of the corrected coordinates (NaN outside their hull). Samples are the static grid
    plus copies scaled about its centre (--sample-scales, default 1 8) because the static grid alone
    collapses inward and covers only about 55-65% of the plotting region.
  * empirical: each checkpoint's measured score (computed on its complete projected tensor, which is
    checked to equal the stored record) placed at its PC1/PC2 coordinates, linearly interpolated over a
    Delaunay triangulation of individual-seed checkpoints; NaN outside their convex hull.

These are structural *evaluation* landscapes (distance of a tensor from GT), not training-objective
landscapes: neither method descends these surfaces, so their minima or ridges say nothing about
optimisation local minima.

  PYTHONPATH=.:src .venv/bin/python scripts/analysis/cscg_jepa_pca_landscapes.py \
      --case 24:<em_baselines dir>:<jepa_mix1 dir> --case 25:... --output <pca_trajectories dir>/structural_landscapes
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LinearSegmentedColormap
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import Delaunay, cKDTree

from mercury_runs.infrastructure.paper_precision import (
    _action_tv_by_pair, compute_action_edge_metrics_canonical, compute_collapsed_distributions_canonical,
    compute_occupancy_weighted_action_tv, condition_non_selfloop_action_distributions,
)
from scripts.analysis.cscg_distillation_matched_eval import output_name
from scripts.analysis.cscg_jepa_pca_trajectories import ARMS, GRID, INK, MUTED, load_case
from scripts.analysis.plot_config_metric_trajectories import _action_edge_f1_auc

# (key, label, higher is better)
METRICS = [("frobenius", "Frobenius distance ‖P̂ − P_GT‖_F", False),
           ("action_tv_similarity", "Action-conditioned TV similarity (with self-loops)", True),
           ("action_no_self_loop_tv_similarity", "Action-conditioned TV similarity (no self-loops)", True),
           ("action_edge_precision", "Action edge precision (threshold 1e-6)", True),
           ("action_edge_recall", "Action edge recall (threshold 1e-6)", True),
           ("action_edge_f1", "Action edge F1 (threshold 1e-6)", True),
           ("action_edge_f1_auc", "Action edge F1, threshold AUC (with self-loops)", True),
           ("action_no_self_loop_edge_f1_auc", "Action edge F1, threshold AUC (no self-loops)", True),
           ("collapsed_tv_similarity", "Policy-collapsed TV similarity", True)]
# Recorded per checkpoint but not a function of the projected tensor (held-out decoding with the
# checkpoint's own clone model): empirical landscape only.
EMPIRICAL_ONLY = [("eval_nll_bps", "Held-out NLL (bits per symbol)", False)]
ALL_METRICS = METRICS + EMPIRICAL_ONLY
# Measured score columns written into the record by the evaluator, checked against our recomputation.
RECORDED = dict(action_tv_similarity="tv_similarity", action_no_self_loop_tv_similarity="tv_similarity_nonself",
                action_edge_precision="recorded_edge_precision", action_edge_recall="recorded_edge_recall",
                action_edge_f1="recorded_edge_f1", collapsed_tv_similarity="recorded_collapsed_tv_similarity",
                frobenius="dist_gt_full")
TOL = 1e-9   # floating-point slack for the probability constraints; never used to alter a tensor
# Single-hue sequential ramp (light -> dark = low -> high) kept neutral so the blue/orange arms stay legible.
plt.rcParams["hatch.color"] = "#c9c8c2"
SURFACE_CMAP = LinearSegmentedColormap.from_list("surface", ["#f7f7f5", "#d9d8d3", "#a7a59e", "#6f6d66", "#3d3c38"])


def valid_tensor(P, valid_rows):
    """True when every valid row is a probability vector and every invalid row is zero."""
    sums = P.sum(axis=2)
    return bool(P.min() >= -TOL and P.max() <= 1 + TOL and np.all(np.abs(sums[valid_rows] - 1) <= 1e-6)
                and np.all(np.abs(P[~valid_rows]) <= TOL))


def structural_metrics(P, context, probability_metrics=True):
    """Existing metric definitions on one projected tensor (same calls as evaluate_transition_structure_canonical)."""
    gt, valid, counts = context["gt"], context["valid_rows"], context["source_action_counts"]
    scores = dict(frobenius=float(np.linalg.norm(P - gt)))
    if not probability_metrics:
        return scores
    tv = compute_occupancy_weighted_action_tv(_action_tv_by_pair(P, gt, valid), valid, counts)[0]
    hat_ns, gt_ns, move = condition_non_selfloop_action_distributions(P, gt, valid)
    tv_ns = compute_occupancy_weighted_action_tv(_action_tv_by_pair(hat_ns, gt_ns, move), move, counts)[0]
    edge = compute_action_edge_metrics_canonical(P, gt, valid, (1e-6,))[1e-6]
    record = dict(action_conditioned=dict(P_true_full=gt, P_hat_full=P, valid_rows=valid))
    collapsed = compute_collapsed_distributions_canonical(P, gt, valid, counts)["evaluation_policy"]
    scores.update(action_tv_similarity=1 - tv, action_no_self_loop_tv_similarity=1 - tv_ns,
                  action_edge_precision=edge["action_probability_edge_precision"],
                  action_edge_recall=edge["action_probability_edge_recall"],
                  action_edge_f1=edge["action_probability_edge_f1"],
                  action_edge_f1_auc=_action_edge_f1_auc(record, include_self_loops=True),
                  action_no_self_loop_edge_f1_auc=_action_edge_f1_auc(record, include_self_loops=False),
                  collapsed_tv_similarity=1 - collapsed["tv_occ"])
    return scores


def _score_chunk(tensors, context):
    out = []
    for P in tensors:
        ok = valid_tensor(P, context["valid_rows"])
        row = {key: np.nan for key, _, _ in METRICS}
        row.update(structural_metrics(P, context, probability_metrics=ok), valid=ok, min_entry=float(P.min()))
        out.append(row)
    return out


def score_tensors(tensors, context, workers=1):
    """Metrics for each tensor; probability metrics are NaN (masked) where the tensor is invalid."""
    if workers <= 1 or len(tensors) < 200:
        return pd.DataFrame(_score_chunk(tensors, context))
    chunks = np.array_split(tensors, workers * 4)
    with ProcessPoolExecutor(workers) as pool:
        return pd.DataFrame([row for part in pool.map(_score_chunk, chunks, repeat(context)) for row in part])


def project_rows_to_simplex(tensors, valid_rows):
    """Euclidean projection of every valid (action, state) row onto the probability simplex over GT states
    plus the sink (sort-and-threshold algorithm, Duchi et al. 2008). Rows outside valid_rows are set to
    zero, as they are in every stored projected tensor. tensors: (n, actions, states, states + 1)."""
    rows = tensors[:, valid_rows]                       # (n, valid rows, states + 1)
    flat = rows.reshape(-1, rows.shape[-1])
    u = -np.sort(-flat, axis=1)
    css = np.cumsum(u, axis=1) - 1.0
    k = np.arange(1, flat.shape[1] + 1)
    rho = (u - css / k > 0).sum(axis=1)                 # condition holds for exactly the first rho entries
    theta = css[np.arange(len(flat)), rho - 1] / rho
    out = np.zeros_like(tensors)
    out[:, valid_rows] = np.maximum(flat - theta[:, None], 0.0).reshape(rows.shape)
    return out


def mean_paths(frame):
    """Per arm: step-wise means of seed coordinates (the existing thick lines) and of measured scores."""
    cols = ["c1_full", "c2_full"] + [f"full_{k}" for k, _, _ in ALL_METRICS]
    return {arm: part.groupby("step")[cols].mean().sort_index() for arm, part in frame.groupby("arm")}


def analyse(frame, grid_size, margin, workers=1, radius_fraction=.02, sample_scales=(1, 8)):
    pca, X, context = frame.attrs["model_full"], frame.attrs["features_full"], frame.attrs["context"]
    shape = context["gt"].shape
    coords = frame[["c1_full", "c2_full"]].to_numpy()
    gt_xy = frame.attrs["gt_full"]

    # Complete tensors: must reproduce the stored evaluation scores exactly (same definitions, same inputs).
    full = score_tensors(X.reshape(-1, *shape), context)
    if not full.valid.all():
        raise ValueError("A stored projected tensor violates the probability constraints")
    for key, column in RECORDED.items():
        if not np.allclose(full[key], frame[column], atol=1e-10, rtol=0):
            raise ValueError(f"Recomputed {key} differs from the evaluation record")
    two_pc = score_tensors(pca.inverse_transform(coords).reshape(-1, *shape), context)
    for key, _, _ in METRICS:
        frame[f"full_{key}"] = full[key].to_numpy()
        frame[f"pca2_{key}"] = two_pc[key].to_numpy()
    frame["pca2_valid"], frame["pca2_min_entry"] = two_pc.valid.to_numpy(), two_pc.min_entry.to_numpy()

    # Grid over seed trajectories (means lie inside their span) and GT, plus a margin.
    points = np.vstack([coords, gt_xy])
    lo, hi = points.min(0), points.max(0)
    pad = margin * (hi - lo)
    xs = np.linspace(lo[0] - pad[0], hi[0] + pad[0], grid_size)
    ys = np.linspace(lo[1] - pad[1], hi[1] + pad[1], grid_size)
    gx, gy = np.meshgrid(xs, ys)
    grid_xy = np.column_stack([gx.ravel(), gy.ravel()])
    grid_P = pca.inverse_transform(grid_xy)
    inverse = score_tensors(grid_P.reshape(-1, *shape), context, workers)

    # Corrected landscape: one simplex projection per row, one reprojection with the same fitted PCA.
    # Samples = the static plotting grid plus, for coverage, the same 100 x 100 grid scaled about its centre.
    valid_rows = context["valid_rows"]
    centre = np.array([(xs[0] + xs[-1]) / 2, (ys[0] + ys[-1]) / 2])
    sample_xy = np.vstack([centre + s * (grid_xy - centre) for s in sample_scales])
    sample_scale = np.repeat(np.asarray(sample_scales, dtype=float), len(grid_xy))
    raw_P = pca.inverse_transform(sample_xy)
    valid_before = np.array([valid_tensor(P, valid_rows) for P in raw_P.reshape(-1, *shape)])
    corrected_P = project_rows_to_simplex(raw_P.reshape(-1, *shape), valid_rows)
    corrected = score_tensors(corrected_P, context, workers)
    if not corrected.valid.all():
        raise ValueError("Simplex-projected tensor violates the probability constraints")
    corrected_xy = pca.transform(corrected_P.reshape(len(sample_xy), -1))
    displacement = np.linalg.norm(corrected_xy - sample_xy, axis=1)
    correction = np.linalg.norm(corrected_P.reshape(len(sample_xy), -1) - raw_P, axis=1)
    corrected_surface = {}
    for key, _, _ in METRICS:
        interpolate = LinearNDInterpolator(corrected_xy, corrected[key].to_numpy())
        corrected_surface[key] = interpolate(gx, gy)
        frame[f"corrected_{key}"] = interpolate(coords)   # corrected landscape read at each checkpoint
    two_pc_corrected = score_tensors(project_rows_to_simplex(
        pca.inverse_transform(coords).reshape(-1, *shape), valid_rows), context)
    for key, _, _ in METRICS:
        frame[f"pca2corr_{key}"] = two_pc_corrected[key].to_numpy()
    inside = ((corrected_xy[:, 0] >= xs[0]) & (corrected_xy[:, 0] <= xs[-1])
              & (corrected_xy[:, 1] >= ys[0]) & (corrected_xy[:, 1] <= ys[-1]))
    span = np.hypot(xs[-1] - xs[0], ys[-1] - ys[0])
    plot_points = np.column_stack([gx.ravel(), gy.ravel()])
    correction_summary = []
    for name, use in [(f"scale {s:g}", sample_scale == s) for s in sample_scales] + [("all samples", sample_scale > 0)]:
        bad = use & ~valid_before
        hull = Delaunay(corrected_xy[use])
        correction_summary.append(dict(
            samples=name, n=int(use.sum()), valid_before=float(valid_before[use].mean()),
            valid_after=float(corrected.valid.to_numpy()[use].mean()),
            displacement_mean=displacement[use].mean(), displacement_median=np.median(displacement[use]),
            displacement_max=displacement[use].max(),
            displacement_mean_invalid_only=displacement[bad].mean() if bad.any() else 0.0,
            displacement_mean_over_region_diagonal=displacement[use].mean() / span,
            tensor_correction_mean=correction[use].mean(), tensor_correction_max=correction[use].max(),
            tensor_correction_mean_invalid_only=correction[bad].mean() if bad.any() else 0.0,
            valid_samples_max_displacement=float(displacement[use & valid_before].max())
            if (use & valid_before).any() else np.nan,
            plot_region_covered=float((hull.find_simplex(plot_points) >= 0).mean()),
            corrected_samples_inside_region=float(inside[use].mean()),
            checkpoints_inside_corrected_hull=float((hull.find_simplex(coords) >= 0).mean()),
            gt_inside_corrected_hull=bool(hull.find_simplex(gt_xy[None])[0] >= 0),
            duplicate_pairs=int(len(cKDTree(corrected_xy[use]).query_pairs(1e-6 * span)))))
    corrected_consistency = local_consistency(corrected_xy, {k: corrected[k].to_numpy() for k, _, _ in METRICS},
                                              corrected_P.reshape(len(sample_xy), -1), radius_fraction * span)

    # Frobenius decomposition: ||P(x,y) - GT||^2 = (x - x_GT)^2 + (y - y_GT)^2 + c, c = GT residual off the plane.
    gt_vec = context["gt"].ravel()
    c = float(np.sum((gt_vec - pca.inverse_transform(gt_xy[None])[0]) ** 2))
    analytic = (grid_xy[:, 0] - gt_xy[0]) ** 2 + (grid_xy[:, 1] - gt_xy[1]) ** 2 + c
    frob_error = np.abs(inverse.frobenius.to_numpy() ** 2 - analytic)
    decomposition = dict(c=c, sqrt_c=np.sqrt(c), max_abs_error=float(frob_error.max()),
                         max_rel_error=float((frob_error / analytic).max()),
                         components_orthonormal_error=float(np.abs(pca.components_ @ pca.components_.T - np.eye(2)).max()))

    # Empirical surfaces from individual-seed checkpoints; LinearNDInterpolator is NaN outside the hull.
    empirical, loso = {}, {}
    frame["full_eval_nll_bps"] = frame["eval_nll_bps"]
    for key, _, _ in ALL_METRICS:
        values = frame[f"full_{key}"].to_numpy()
        empirical[key] = LinearNDInterpolator(coords, values)(gx, gy)
        held = np.full(len(frame), np.nan)
        for seed in frame.seed.unique():   # leave-one-seed-out: all arms of the other seeds
            out = (frame.seed == seed).to_numpy()
            held[out] = LinearNDInterpolator(coords[~out], values[~out])(coords[out])
        frame[f"loso_{key}"] = held
        loso[key] = held
    grids = dict(xs=xs, ys=ys, gt_xy=gt_xy, valid=inverse.valid.to_numpy().reshape(gx.shape),
                 min_entry=inverse.min_entry.to_numpy().reshape(gx.shape), frobenius_c=c,
                 explained_variance_ratio=pca.explained_variance_ratio_,
                 **{f"inverse_{k}": inverse[k].to_numpy().reshape(gx.shape) for k, _, _ in METRICS},
                 **{f"empirical_{k}": empirical[k] for k, _, _ in ALL_METRICS},
                 sample_xy=sample_xy, sample_scale=sample_scale, sample_valid_before=valid_before,
                 corrected_xy=corrected_xy, displacement=displacement, tensor_correction=correction,
                 **{f"corrected_samples_{k}": corrected[k].to_numpy() for k, _, _ in METRICS},
                 **{f"corrected_{k}": corrected_surface[k] for k, _, _ in METRICS})
    return grids, decomposition, correction_summary, corrected_consistency


def validation_summary(frame):
    rows = []
    for key, _, _ in ALL_METRICS:
        truth = pd.Series(frame[f"full_{key}"].to_numpy())   # plain Series: frame.attrs hold arrays
        spread = float(truth.max() - truth.min())
        for name, column in (("two_component_reconstruction", "pca2"),
                             ("two_component_reconstruction_simplex_corrected", "pca2corr"),
                             ("corrected_landscape_at_checkpoint", "corrected"),
                             ("empirical_leave_one_seed_out", "loso")):
            if f"{column}_{key}" not in frame:   # empirical-only metrics have no tensor-based estimate
                continue
            estimate = pd.Series(frame[f"{column}_{key}"].to_numpy())
            error = (estimate - truth).abs()
            rows.append(dict(metric=key, estimator=name, checkpoints=len(truth),
                             coverage=float(error.notna().mean()), mae=error.mean(), median_ae=error.median(),
                             p90_ae=error.quantile(.9), max_ae=error.max(),
                             mae_over_range=error.mean() / spread if spread else np.nan,
                             spearman=float(pd.concat([estimate, truth], axis=1).dropna().corr("spearman").iloc[0, 1])
                             if error.notna().sum() > 2 else np.nan))
    return pd.DataFrame(rows)


def local_consistency(coords, values, tensors, radius, arms=None, max_pairs=200_000):
    """Do samples that are close in the PCA plane (within radius) have similar scores?
    values: metric -> score per sample; tensors: (samples, features) for the tensor-space distance."""
    pairs = np.array(sorted(cKDTree(coords).query_pairs(radius)), dtype=np.int64).reshape(-1, 2)
    total = len(pairs)
    if total > max_pairs:   # many corrected samples can pile up; summarise a fixed random subset
        pairs = pairs[np.random.default_rng(0).choice(total, max_pairs, replace=False)]
    i, j = pairs.T
    plane = np.linalg.norm(coords[i] - coords[j], axis=1)
    tensor = np.linalg.norm(tensors[i] - tensors[j], axis=1)
    rows = []
    for key, v in values.items():
        spread = np.nanpercentile(v, 95) - np.nanpercentile(v, 5)
        d = np.abs(v[i] - v[j])
        d = d[np.isfinite(d)]
        rows.append(dict(metric=key, radius=radius, close_pairs=total, pairs_summarised=len(pairs),
                         median_abs_diff=np.median(d) if d.size else np.nan,
                         p90_abs_diff=np.percentile(d, 90) if d.size else np.nan,
                         max_abs_diff=d.max() if d.size else np.nan, score_spread_p5_p95=spread,
                         frac_close_pairs_diff_gt_10pct_spread=float((d > .1 * spread).mean()) if d.size else np.nan,
                         frac_close_pairs_diff_gt_25pct_spread=float((d > .25 * spread).mean()) if d.size else np.nan,
                         median_tensor_over_plane_distance=float(np.median(tensor / np.maximum(plane, 1e-12)))
                         if len(pairs) else np.nan,
                         cross_arm_close_pairs=int((arms[i] != arms[j]).sum()) if arms is not None else np.nan))
    return pd.DataFrame(rows)


def overlay_2d(ax, frame, means):
    for label, _, _, _, colour, ls in ARMS:
        part = frame[frame.arm == label]
        if part.empty:
            continue
        for _, seed in part.groupby("seed"):
            seed = seed.sort_values("step")
            ax.plot(seed.c1_full, seed.c2_full, color=colour, ls=ls, lw=.8, alpha=.5, zorder=3)
        mean = means[label]
        ax.plot(mean.c1_full, mean.c2_full, color=colour, ls=ls, lw=2.4, zorder=4)
        ax.scatter(mean.c1_full.iloc[0], mean.c2_full.iloc[0], s=55, facecolor="white", edgecolor=colour, lw=1.8, zorder=5)
        ax.scatter(mean.c1_full.iloc[-1], mean.c2_full.iloc[-1], s=60, marker="s", color=colour, edgecolor="white", lw=1, zorder=6)
    ax.scatter(*frame.attrs["gt_full"], marker="*", s=260, color=INK, edgecolor="white", lw=1.2, zorder=7)


def overlay_3d(ax, frame, means, heights, mean_heights, gt_height):
    """heights: per-checkpoint z (Series aligned with frame); mean_heights: arm -> z per mean step."""
    for label, _, _, _, colour, ls in ARMS:
        part = frame[frame.arm == label]
        if part.empty:
            continue
        for _, seed in part.groupby("seed"):
            seed = seed.sort_values("step")
            ax.plot(seed.c1_full, seed.c2_full, heights[seed.index], color=colour, ls=ls, lw=.8, alpha=.6, zorder=3)
        mean, z = means[label], mean_heights[label]
        ax.plot(mean.c1_full, mean.c2_full, z, color=colour, ls=ls, lw=2.4, zorder=4)
        ax.scatter(mean.c1_full.iloc[0], mean.c2_full.iloc[0], z[0], s=45, facecolor="white", edgecolor=colour, lw=1.6,
                   depthshade=False)
        ax.scatter(mean.c1_full.iloc[-1], mean.c2_full.iloc[-1], z[-1], s=50, marker="s", color=colour,
                   edgecolor="white", lw=.8, depthshade=False)
    if gt_height is None:   # metric undefined for the GT tensor
        return
    gx, gy = frame.attrs["gt_full"]
    ax.scatter(gx, gy, gt_height, marker="*", s=220, color=INK, edgecolor="white", lw=1, depthshade=False)


def style_2d(ax, variance, title):
    ax.set_xlabel(f"PC1 ({variance[0]:.0%})", fontsize=8, color=MUTED)
    ax.set_ylabel(f"PC2 ({variance[1]:.0%})", fontsize=8, color=MUTED)
    ax.set_title(title, fontsize=9, color=INK, loc="left")
    ax.tick_params(labelsize=7, colors=MUTED, length=0)
    for side in ax.spines.values():
        side.set_color(GRID)


def style_3d(ax, variance, title, label, zlim):
    ax.set_xlabel(f"PC1 ({variance[0]:.0%})", fontsize=7, color=MUTED, labelpad=-4)
    ax.set_ylabel(f"PC2 ({variance[1]:.0%})", fontsize=7, color=MUTED, labelpad=-4)
    ax.set_zlabel(label, fontsize=7, color=MUTED, labelpad=-2)
    ax.set_zlim(*zlim)
    ax.tick_params(labelsize=6, colors=MUTED, pad=-2)
    ax.set_title(title, fontsize=9, color=INK, loc="left")
    ax.view_init(elev=28, azim=-58)


def plot_metric(frame, grids, key, label, higher, path, dataset):
    variance = frame.attrs["variance"]["full"]
    xs, ys = grids["xs"], grids["ys"]
    gx, gy = np.meshgrid(xs, ys)
    empirical_only = f"inverse_{key}" not in grids
    emp = grids[f"empirical_{key}"]
    surfaces = [emp] if empirical_only else [grids[f"inverse_{key}"], grids[f"corrected_{key}"], emp]
    observed = frame[f"full_{key}"]
    extra = [] if empirical_only else [frame[f"pca2_{key}"].dropna()]
    values = np.concatenate([g[np.isfinite(g)] for g in surfaces] + [observed] + extra)
    norm = Normalize(values.min(), values.max() if values.max() > values.min() else values.min() + 1e-9)
    levels = np.linspace(norm.vmin, norm.vmax, 16)
    means = mean_paths(frame)
    direction = "higher is better" if higher else "lower is better"
    context, pca = frame.attrs["context"], frame.attrs["model_full"]
    gt_height = None if empirical_only else structural_metrics(context["gt"], context)[key]
    z_lo, z_hi = ((norm.vmin, norm.vmax) if gt_height is None   # z-range also shows GT's own score
                  else (min(norm.vmin, gt_height), max(norm.vmax, gt_height)))
    zlim = (z_lo - .03 * (z_hi - z_lo), z_hi + .03 * (z_hi - z_lo))
    masked_share = 0 if key == "frobenius" else 1 - grids["valid"].mean()

    # 3-D trajectory heights. Inverse-PCA: score of each 2-PC reconstruction (lies on that surface).
    # Corrected: the corrected surface read at the checkpoint coordinates (lies on that surface).
    # Empirical: the measured checkpoint scores themselves.
    mean_xy = {arm: m[["c1_full", "c2_full"]].to_numpy() for arm, m in means.items()}
    empirical_panel = (emp, f"Empirical: measured checkpoint scores, linear triangular interpolation\n"
                            f"blank = outside the checkpoints' convex hull ({1 - np.isfinite(emp).mean():.0%})",
                       observed, {a: m[f"full_{key}"].to_numpy() for a, m in means.items()},
                       "height = measured checkpoint score")
    if empirical_only:
        panels = [empirical_panel]
    else:
        inv, cor = surfaces[:2]
        corrected_at = LinearNDInterpolator(grids["corrected_xy"], grids[f"corrected_samples_{key}"])
        panels = [
        (inv, f"Inverse-PCA (masked): metric of the 2-PC reconstruction\n"
              f"hatched = not a probability tensor ({masked_share:.0%} of grid)",
         frame[f"pca2_{key}"], {a: score_tensors(pca.inverse_transform(xy).reshape(-1, *context["gt"].shape),
                                                 context)[key].to_numpy() for a, xy in mean_xy.items()},
         "height = score of 2-PC reconstruction"),
        (cor, f"Inverse-PCA, simplex-corrected and reprojected once\n"
              f"Delaunay over corrected coordinates; blank = outside their hull ({1 - np.isfinite(cor).mean():.0%})",
         pd.Series(frame[f"corrected_{key}"].to_numpy(), index=frame.index),
         {a: corrected_at(xy) for a, xy in mean_xy.items()}, "height = corrected surface at checkpoint"),
        empirical_panel]
    ncol = len(panels)
    fig = plt.figure(figsize=(6.5 * ncol + (1.5 if ncol == 1 else 0), 11.5))
    for col, (surface, title, heights, mean_heights, height_note) in enumerate(panels):
        ax = fig.add_subplot(2, ncol, col + 1)
        filled = ax.contourf(gx, gy, np.ma.masked_invalid(surface), levels=levels, cmap=SURFACE_CMAP, norm=norm, zorder=1)
        if col == 0 and key != "frobenius" and not empirical_only:
            ax.contourf(gx, gy, (~grids["valid"]).astype(float), levels=[.5, 1.5], colors="none", hatches=["////"],
                        zorder=2)
            ax.contour(gx, gy, grids["valid"].astype(float), levels=[.5], colors=[MUTED], linewidths=.7, zorder=2)
        overlay_2d(ax, frame, means)
        ax.set_xlim(xs[0], xs[-1]); ax.set_ylim(ys[0], ys[-1])
        style_2d(ax, variance, title)
        bar = fig.colorbar(filled, ax=ax, shrink=.85, pad=.02)
        bar.set_label(direction, fontsize=7, color=MUTED)
        bar.ax.tick_params(labelsize=6, colors=MUTED)
        ax = fig.add_subplot(2, ncol, col + 1 + ncol, projection="3d")
        ax.plot_surface(gx, gy, np.ma.masked_invalid(surface), cmap=SURFACE_CMAP, norm=norm, alpha=.75, lw=0,
                        rstride=2, cstride=2, antialiased=False)
        overlay_3d(ax, frame, means, heights, mean_heights, gt_height)
        ax.set_xlim(xs[0], xs[-1]); ax.set_ylim(ys[0], ys[-1])
        style_3d(ax, variance, f"3-D surface; trajectory {height_note}", label, zlim)

    present = set(frame.arm)
    handles = [Line2D([0], [0], color=c, ls=ls, lw=2.4, label=lab) for lab, _, _, _, c, ls in ARMS if lab in present]
    handles += [Line2D([0], [0], color=MUTED, lw=.8, label="thin = individual seeds, thick = mean over seeds"),
                Line2D([0], [0], marker="o", color="none", markerfacecolor="white", markeredgecolor=MUTED, label="start (step 0)"),
                Line2D([0], [0], marker="s", color="none", markerfacecolor=MUTED, label="end (step 20,000)"),
                Line2D([0], [0], marker="*", color="none", markerfacecolor=INK, markersize=12,
                       label="ground truth (2-D only: NLL undefined for the GT tensor)" if empirical_only
                       else "ground truth (3-D height = GT's own score)")]
    fig.legend(handles=handles, loc="lower center", ncol=2 if empirical_only else 7, frameon=False, fontsize=8)
    kind = ("Empirical held-out NLL landscape (each checkpoint's own decoding; no inverse-PCA version)\n"
            if empirical_only else "Structural evaluation landscape ")
    fig.suptitle(f"Level {int(frame.level.iloc[0])} — {label} ({direction}). {dataset}.\n{kind}"
                 f"over the joint projected-transition PCA plane; not a training-objective landscape.",
                 fontsize=10, color=INK, x=.01, ha="left")
    fig.tight_layout(rect=(0, .07 if empirical_only else .03, 1, .95 if empirical_only else .97))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_correction(frame, grids, summary, path, dataset):
    """Static grid, corrected-and-reprojected samples, and the static grid's displacement."""
    xs, ys = grids["xs"], grids["ys"]
    scale, sample_xy, corrected_xy = grids["sample_scale"], grids["sample_xy"], grids["corrected_xy"]
    static = scale == 1
    valid = grids["sample_valid_before"]
    first, union = summary[0], summary[-1]
    region = np.array([[xs[0], ys[0]], [xs[-1], ys[-1]]])
    shown = np.vstack([region, corrected_xy, sample_xy[static]])
    pad = .03 * (shown.max(0) - shown.min(0))
    lim = (shown.min(0) - pad, shown.max(0) + pad)
    means = mean_paths(frame)
    variance = frame.attrs["variance"]["full"]
    fig, axes = plt.subplots(1, 3, figsize=(19, 6.6))
    xy = sample_xy[static]
    axes[0].scatter(*xy[~valid[static]].T, s=2, color="#c9c8c2", lw=0, zorder=1)
    axes[0].scatter(*xy[valid[static]].T, s=2, color="#52514e", lw=0, zorder=2)
    style_2d(axes[0], variance, f"Static 100×100 grid: dark = valid tensor ({first['valid_before']:.0%}), light = invalid")
    axes[1].scatter(*corrected_xy[~static].T, s=2, color="#c9c8c2", lw=0, zorder=1)
    axes[1].scatter(*corrected_xy[static].T, s=2, color="#52514e", lw=0, zorder=2)
    style_2d(axes[1], variance, f"Corrected + reprojected (valid {union['valid_after']:.0%}): dark = static grid "
                                f"(covers {first['plot_region_covered']:.0%}),\nlight = expanded grid "
                                f"(union covers {union['plot_region_covered']:.0%} of the plotting region)")
    step = max(1, len(xs) // 25)   # every step-th static grid point in each direction
    sub = np.zeros((len(ys), len(xs)), dtype=bool); sub[::step, ::step] = True
    idx = np.flatnonzero(static)[sub.ravel()]
    d = corrected_xy[idx] - sample_xy[idx]
    axes[2].quiver(*sample_xy[idx].T, *d.T, angles="xy", scale_units="xy", scale=1, width=.0025,
                   color=np.where(valid[idx], "#52514e", "#a7a59e"), zorder=2)
    style_2d(axes[2], variance, f"Static-grid displacement (every {step}th point): mean {first['displacement_mean']:.2f}, "
                                f"median {first['displacement_median']:.2f}, max {first['displacement_max']:.2f}")
    for ax in axes:
        ax.add_patch(plt.Rectangle((xs[0], ys[0]), xs[-1] - xs[0], ys[-1] - ys[0], fill=False, ec=INK, lw=.8,
                                   ls=":", zorder=3))
        for label, _, _, _, colour, ls in ARMS:
            if label in means:
                ax.plot(means[label].c1_full, means[label].c2_full, color=colour, ls=ls, lw=2, zorder=4)
        ax.scatter(*frame.attrs["gt_full"], marker="*", s=200, color=INK, edgecolor="white", lw=1, zorder=5)
        ax.set_xlim(lim[0][0], lim[1][0]); ax.set_ylim(lim[0][1], lim[1][1])
    fig.suptitle(f"Level {int(frame.level.iloc[0])} — inverse-PCA grid correction (one simplex projection per row, one "
                 f"PCA reprojection). {dataset}. Dotted box = plotting region; lines = seed-mean trajectories; "
                 f"star = GT.", fontsize=10, color=INK, x=.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, .95))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case", action="append", required=True, help="LEVEL:EM_BASELINES_DIR:JEPA_DIR")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--dataset-label", default="")
    p.add_argument("--grid", type=int, default=100)
    p.add_argument("--margin", type=float, default=.05)
    p.add_argument("--radius", type=float, default=.02, help="close-pair radius, fraction of the plane's span")
    p.add_argument("--sample-scales", type=float, nargs="+", default=[1, 8],
                   help="corrected-landscape sampling grids: the static grid (1) plus copies scaled about its centre")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    p.add_argument("--walks", type=int, default=10)
    p.add_argument("--walk-length", type=int, default=500)
    p.add_argument("--rand-prob", type=float, default=.1)
    args = p.parse_args()
    evaluation = output_name(args.walks, args.walk_length, args.rand_prob)
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoints, summaries, consistency, decompositions, corrections = [], [], [], [], []
    for case in args.case:
        level, em, jepa = case.split(":")
        frame = load_case(int(level), {"em": Path(em), "jepa": Path(jepa)}, evaluation)
        grids, decomposition, correction, corrected_consistency = analyse(
            frame, args.grid, args.margin, args.workers, args.radius, tuple(args.sample_scales))
        np.savez_compressed(args.output / f"level{level}_landscape_grids.npz", **grids)
        for key, label, higher in ALL_METRICS:
            plot_metric(frame, grids, key, label, higher, args.output / f"level{level}_{key}.png", args.dataset_label)
        plot_correction(frame, grids, correction, args.output / f"level{level}_grid_correction.png", args.dataset_label)
        plain = frame.copy(); plain.attrs = {}
        checkpoints.append(plain)
        summaries.append(validation_summary(frame).assign(level=int(level)))
        radius = args.radius * np.hypot(grids["xs"][-1] - grids["xs"][0], grids["ys"][-1] - grids["ys"][0])
        consistency.append(local_consistency(
            frame[["c1_full", "c2_full"]].to_numpy(), {k: frame[f"full_{k}"].to_numpy() for k, _, _ in ALL_METRICS},
            frame.attrs["features_full"], radius, frame.arm.to_numpy()).assign(level=int(level), samples="checkpoints"))
        consistency.append(corrected_consistency.assign(level=int(level), samples="corrected_grid"))
        corrections += [dict(level=int(level), **row) for row in correction]
        decompositions.append(dict(level=int(level), grid_valid_fraction=float(grids["valid"].mean()),
                                   checkpoint_2pc_valid_fraction=float(frame.pca2_valid.mean()),
                                   explained_variance_pc1_pc2=float(frame.attrs["variance"]["full"].sum()),
                                   **decomposition))
        print(f"level {level}: static grid valid {correction[0]['valid_before']:.1%} -> "
              f"{correction[0]['valid_after']:.1%}; mean displacement {correction[0]['displacement_mean']:.3f}; "
              f"corrected hull covers {correction[0]['plot_region_covered']:.1%} (static) / "
              f"{correction[-1]['plot_region_covered']:.1%} (with expanded grid); Frobenius decomposition max |error| "
              f"{decomposition['max_abs_error']:.2e}")
    pd.concat(checkpoints, ignore_index=True).to_csv(args.output / "checkpoint_validation.csv", index=False)
    pd.concat(summaries, ignore_index=True).to_csv(args.output / "validation_summary.csv", index=False)
    pd.concat(consistency, ignore_index=True).to_csv(args.output / "local_consistency.csv", index=False)
    pd.DataFrame(decompositions).to_csv(args.output / "frobenius_decomposition.csv", index=False)
    pd.DataFrame(corrections).to_csv(args.output / "grid_correction_summary.csv", index=False)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
