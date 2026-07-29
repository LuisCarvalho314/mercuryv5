# Architecture

The movable codebase is now organized as a layered application under `src/mercury_runs`.

## Layers

- `domain/`
  - Typed Pydantic models and contracts for runs, studies, artifacts, and statuses.
- `algorithms/`
  - Canonical Python-first algorithm packages for Mercury, POCML, and CSCG, each split into step modules.
- `application/`
  - Single-run and study use cases. Includes `evaluation_service.py` for the canonical-v4 evaluation protocol.
- `infrastructure/`
  - Filesystem persistence, artifact layout, reporting, W&B logging, subprocess execution, and runtime helpers.
- `interfaces/`
  - CLI parsing and top-level dispatch.
- `analysis/`
  - Post-training evaluation record consumption and figure generation. Must not load model snapshots, generate walks, or compute alignments — all data comes from stored evaluation records.
  - `evaluation_records.py`: canonical library for reading evaluation records.
  - `transition_residual_maps.py`: per-run residual map figures.
  - `residual_map_rendering.py`: shared figure assembly helpers.
  - `plots/`: study-level canonical plot generators.

## Protected Dependencies

These directories are treated as external dependencies and are not modified by the rebuild:

- `external/`
- `src/mercury/`
- `src/maze_environment/`

All interaction with those directories should happen through adapters in the movable layers rather than direct business logic in the CLI.

## Canonical Surface

The canonical implementation now lives in:

- `algorithms/`
- `application/`
- `domain/`
- `infrastructure/`
- `interfaces/`

The old top-level wrapper modules were removed. New code should import canonical modules directly.


## Evaluation Pipeline

Paper-precision evaluation is a first-class, separately invocable three-phase procedure: training → evaluation records → plots. Each phase is independent and idempotent.

**Training phase**:
- Saves `snapshots/<method>/paper_precision/step=<N>/snapshot.json` at each configured checkpoint.
- No metrics are computed inline during training.

**Evaluation phase** (`evaluate_study` / `evaluate_study_snapshots`):
- Iterates over all run directories in a study using `study_manifest.jsonl`.
- Shares a `WalkCache` keyed by `(level, sensor, sensor_range, rand_prob, seed, n_walks, walk_length, valid_trajectories_only)`. Walks are generated once per unique protocol and reused across all runs and checkpoints.
- For each snapshot: loads the model, decodes the shared walks, builds evaluation structures, and calls `project_and_evaluate_transition_structure` (single alignment for both collapsed and action-conditioned TV).
- Writes immutable evaluation records at `<run_root>/evaluations/<protocol_id>/snapshots/step=XXXXXXXX.json`.
- Each record captures: `P_true_full`, `P_hat_full` (shape `(n_actions, n_true, n_true+1)`), collapsed matrices, TV metrics, alignment vectors, state-representation metrics. The `+1` column is the ⊥ sink state absorbing unmatched-target mass.
- The `protocol_id` (`canonical-v4-<hash>`) encodes all walk generation and evaluation hyperparameters, ensuring reproducibility.

**Plotting phase** (`src/mercury_runs/analysis/plots/`):
- Reads evaluation records via `evaluation_records.py` — no model loading, no walk generation, no alignment computation.
- `selected_residual_maps.py`: produces ground-truth-aligned transition residual map figures.
- `threshold_edge_metrics.py`: computes and plots thresholded edge precision/recall/F1.
- `batch_summary_forest.py`: forest/matrix/profile comparison across studies.
- `threshold_robustness.py`: threshold robustness figures from edge-metrics CSVs.
- All modules accept `--study_root` and `--protocol_id` as required CLI arguments.

**Single-run re-evaluation** (`reevaluate_paper_precision_snapshots` / `--reevaluate_paper_precision_snapshots`):
- Same evaluation pipeline for a single run, without the shared walk cache across runs.

**Current state**: CSCG fully implements this pattern. Mercury and POCML still compute metrics inline during training; migrating them to the snapshot pattern is the intended direction.
