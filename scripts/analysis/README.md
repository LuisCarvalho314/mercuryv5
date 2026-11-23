# Analysis Scripts

Purpose: post-run metrics computation and offline analysis.

## Key scripts
- `plot_spatial_spectral_development.py`: primary spectral visualisations for all four environments: GT-aligned leading-mode heatmaps on the environment layout, median/IQR convergence, pairwise spectral distances and distance residuals. Uses a separate symmetric normalised Laplacian that retains unmatched leakage; preserves the directed diagnostics. MERCURY and its CSCG continuation share one staged trajectory. Reports and PNG/PDF figures are under `spatial_spectral_development`, with a linked index and per-run numeric artifacts.
- `cscg_spectral_development.py`: compare directed spectra, separated invariant subspaces, TV similarity and NLL across oracle and standard CSCG capacities, using frozen evaluation records. Defaults to the matched-budget λ=0.99 dataset; runs 1×4 before 3×4. Produces PNG/PDF figures, checkpoint and paired-change tables, PCA bases, metadata and findings. Unmatched mass remains a row deficit; tied subspace boundaries are marked undefined.
  Add `--all-methods` to include MERCURY and TMGWR, each joined to its CSCG continuation as one staged model trajectory. The source endpoint and CSCG handoff are distinct records at the same total sample count. Subsequent hybrid checkpoints use the saved cumulative budget; source NLL and source training curves are unavailable. Outputs go to `model_spectral_development` alongside the original `cscg_spectral_development` results.
- `compute_precision.py`: emit paper-precision plots and metadata from `*_paper_precision.json` artifacts.
- `hyperparameter_optimisation.py`: feature/model analysis over study summaries.
- `run_health_report.py`: scan study outputs and summarize run completion/status health.
- `plot_method_graphs.py`: regenerate unified internal graph plots for enabled methods and a combined comparison figure for a run.
- `plot_study_histories.py`: aggregate Mercury paper-precision histories across seeds and write per-level training-step plots.

## Example usage
```bash
uv run python scripts/analysis/compute_precision.py --level 16 --sensor "cardinal distance" --sensor-range 1
uv run python scripts/analysis/hyperparameter_optimisation.py
uv run python scripts/analysis/run_health_report.py --study-root results/current/studies
uv run python scripts/analysis/cscg_spectral_development.py --dataset-root results/current/action_alignment_rerun_matched_budget_lambda099_20261004 --levels 21 22
uv run python scripts/analysis/cscg_spectral_development.py --dataset-root results/current/action_alignment_rerun_matched_budget_lambda099_20261004 --all-methods --levels 21 22
uv run python scripts/analysis/plot_spatial_spectral_development.py --levels 13 21 20 22
uv run python scripts/analysis/plot_method_graphs.py --run_root results/current/studies/pocml_test/<run_id>
uv run python scripts/analysis/plot_study_histories.py --study-root results/current/studies/levels_13_21_22_weight_study
```

Notes:
- These scripts are post-run tools only; execution and study orchestration live under `main.py` and `src/mercury_runs`.
- `*_precision.png` now visualizes paper precision from `*_paper_precision.json`; if paper precision is disabled, no precision PNG is written.
- `mean_total_variation` and `action_conditioned_mean_total_variation` are graph-structure metrics. They align learned nodes to ground-truth states, project learned adjacency or action-labelled transition tensors through that alignment, row-normalize the projected learned and ground-truth transition weights, and then compute mean TV. They are not computed directly from consecutive decoded-state transition counts alone.
- unified graph plots are now generated automatically during runs; `plot_method_graphs.py` is for regeneration or repair.
- `plot_method_graphs.py` reads `*_latent_graph.npz`, `*_embeddings.npz`, and `*_cscg_model.npz` when present, and writes per-method plots plus `plots/internal_graphs_comparison.png`.
- Canonical fixed-map paper-style precision is written separately to `metrics/<method>/*_paper_precision.json` and summarized through `comparison_summary.json`.
