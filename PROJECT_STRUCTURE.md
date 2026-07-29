# Project Structure

## Source
- `src/`: application/package source code.
- `tests/`: unit/integration tests mirroring `src/` domains.

## Entry Points
- `main.py`: primary CLI for single runs and study orchestration.
- `scripts/analysis/`: thin shims (backward-compatible) pointing to canonical modules in `src/mercury_runs/analysis/`.
- `scripts/study/`: retired documentation stub pointing back to `main.py` and `make study*`.
- `scripts/live/`: retired documentation stub for removed live/debug scripts.

## Analysis Package (`src/mercury_runs/analysis/`)

Canonical analysis and plotting code — the only place where evaluation records are consumed:

- `evaluation_records.py`: library for loading and querying canonical-v4 evaluation records without model access. Key functions: `load_evaluation_step`, `list_evaluation_steps`, `load_study_manifest`, `collapsed_matrices`, `action_matrices`, `edge_metrics_at_threshold`, `diagnostics_from_record`.
- `transition_residual_maps.py`: per-run residual map generation (`generate_transition_residual_maps`).
- `residual_map_rendering.py`: figure assembly helpers (colorbars, TV-similarity bars, level labels). Public entry point: `save_selected_figure`.
- `plots/` — canonical study-level plot generators (run as `python -m mercury_runs.analysis.plots.<module>`):
  - `selected_residual_maps.py`: ground-truth-aligned multi-panel transition residual maps across seeds and methods.
  - `threshold_edge_metrics.py`: thresholded edge precision/recall/F1 from evaluation records.
  - `batch_summary_forest.py`: forest/matrix/profile comparison plots across batch-40 studies.
  - `threshold_robustness.py`: threshold robustness figures from edge-metrics CSVs.
  - `transition_residual_maps_cli.py`: CLI wrapper around `transition_residual_maps.generate_transition_residual_maps`.

### Canonical evaluation record layout

Evaluation records are stored at:
```
<run_root>/evaluations/<protocol_id>/snapshots/step=XXXXXXXX.json
```

Each record contains:
- `policy_collapsed`: P_true, P_hat (shape `(n_true, n_true+1)`), TV metrics
- `action_conditioned`: P_true_full, P_hat_full (shape `(n_actions, n_true, n_true+1)`), TV metrics
- `state_representation`: latent_precision, n_true_states, n_actions
- `alignment`: cooccurrence, learned_to_true vectors
- `ground_truth`: T_true_action, P_true_action, N_ai

The last column in P_true/P_hat tensors is the ⊥ sink state (absorbs unmatched-target mass).

## Configuration
- `configs/studies/`: JSON grid definitions for study mode (`--study_config`).
- `study_config.json`: legacy root config kept for compatibility.
- `external/POCML/`: vendored/cloned baseline method repository for comparison.
- `external/naturecomm_cscg/`: vendored/cloned CSCG baseline repository for comparison.

## Generated Artifacts (ignored)
- `datasets/level=*/`: parquet dataset exports.
- `results/current/`: active run bundles, metrics, study summaries.
- `results/legacy/`: timestamped snapshots of older result layouts.
  - includes migrated historical outputs from earlier layouts.
- `logs/`: runtime logs.
- `saved_states/`: local checkpoints.

### Study run directory shape
Typical per-run files under `results/current/studies/<study_name>/<run_id>/`:
- `run_config.json`: full run payload, CLI args, and reproduction command.
- `run_status.json`: lifecycle state, timestamps, artifact pointers, and error (if failed).
- `bundles/mercury/`: Mercury state parquet outputs.
- `bundles/pocml/`: native POCML outputs when enabled.
- `bundles/cscg/`: native CSCG outputs when enabled.
- `metrics/mercury/`: Mercury precision parquet outputs.
- `metrics/cscg/`: CSCG paper-precision JSON written by `--evaluate_study_snapshots`.
- `snapshots/cscg/paper_precision/step=<N>/snapshot.json`: CSCG model snapshots saved during training (one per checkpoint). Consumed by the post-training evaluation pass.
- `comparison_summary.json`: native-only baseline comparison summary when any baseline is enabled.

## Build/Packaging
- `pyproject.toml`: package/dependency metadata.
- `uv.lock`: locked dependency graph.
- `dist/`: package build outputs.
