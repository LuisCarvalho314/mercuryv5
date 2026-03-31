# Analysis Scripts

Purpose: post-run metrics computation and offline analysis.

## Key scripts
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
uv run python scripts/analysis/plot_method_graphs.py --run_root results/current/studies/pocml_test/<run_id>
uv run python scripts/analysis/plot_study_histories.py --study-root results/current/studies/levels_13_21_22_weight_study
```

Notes:
- These scripts are post-run tools only; execution and study orchestration live under `main.py` and `src/mercury_runs`.
- `*_precision.png` now visualizes paper precision from `*_paper_precision.json`; if paper precision is disabled, no precision PNG is written.
- unified graph plots are now generated automatically during runs; `plot_method_graphs.py` is for regeneration or repair.
- `plot_method_graphs.py` reads `*_latent_graph.npz`, `*_embeddings.npz`, and `*_cscg_model.npz` when present, and writes per-method plots plus `plots/internal_graphs_comparison.png`.
- Canonical fixed-map paper-style precision is written separately to `metrics/<method>/*_paper_precision.json` and summarized through `comparison_summary.json`.
