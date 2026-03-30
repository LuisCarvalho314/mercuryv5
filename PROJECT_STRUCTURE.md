# Project Structure

## Source
- `src/`: application/package source code.
- `tests/`: unit/integration tests mirroring `src/` domains.

## Entry Points
- `main.py`: primary CLI for single runs and study orchestration.
- `scripts/analysis/`: supported offline analysis scripts.
- `scripts/study/`: retired documentation stub pointing back to `main.py` and `make study*`.
- `scripts/live/`: retired documentation stub for removed live/debug scripts.

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
- `comparison_summary.json`: native-only baseline comparison summary when any baseline is enabled.

## Build/Packaging
- `pyproject.toml`: package/dependency metadata.
- `uv.lock`: locked dependency graph.
- `dist/`: package build outputs.
