# Quick Start

## 1) Install

Using `uv`:

```bash
uv sync
```

Optional (for Weights & Biases logging):

```bash
uv add wandb
uv run wandb login
```

Optional (for POCML baseline comparisons):

```bash
git clone https://github.com/calvin4770/POCML external/POCML
uv add torch
```

Optional (for CSCG baseline comparisons):

```bash
git clone https://github.com/vicariousinc/naturecomm_cscg external/naturecomm_cscg
```


## 2) Verify CLI

```bash
python main.py --help
```

The canonical movable code lives in `src/mercury_runs`:
- `algorithms/` for Mercury, POCML, and CSCG
- `application/` for run and study orchestration
- `domain/` for typed Pydantic models
- `infrastructure/` for storage, reporting, runtime, and W&B
- `interfaces/` for the CLI


## 3) Run One Experiment

```bash
python main.py \
  --level 16 \
  --sensor "cardinal distance" \
  --sensor_range 1 \
  --memory_length 40
```

Outputs are written under:
- `results/current/studies/<study_name>/<run_id>/bundles/...`
- `results/current/studies/<study_name>/<run_id>/metrics/...`
- `results/current/studies/<study_name>/<run_id>/run_config.json`
- `results/current/studies/<study_name>/<run_id>/run_status.json`

Reuse is strict:
- only completed schema-valid runs with the exact expected artifacts are reused
- failed or polluted run folders are not reused
- if an auto-generated `run_id` collides with an invalid folder, a fresh suffixed run directory is created

Mercury is the only method that writes `metrics/*_precision.parquet`.

### Baselines

POCML:

```bash
python main.py \
  --level 16 \
  --sensor "cardinal distance" \
  --sensor_range 1 \
  --baseline_pocml True
```

CSCG:

```bash
python main.py \
  --level 16 \
  --sensor "cardinal distance" \
  --sensor_range 1 \
  --baseline_cscg True \
  --cscg_train_algorithm em
```

All methods:

```bash
python main.py \
  --level 16 \
  --sensor "cardinal distance" \
  --sensor_range 1 \
  --baseline_all True
```

Baseline outputs are native-only:
- POCML writes `bundles/pocml/.../*_states.parquet` and `*_embeddings.npz`
- CSCG writes `bundles/cscg/.../*_states.parquet` and `*_cscg_model.npz`
- no baseline precision parquet files are generated

### Direct Python API

For fine-grained algorithm control:

```python
from mercury_runs import MercuryConfig, POCMLConfig, CSCGConfig, run_mercury, run_pocml, run_cscg
```

Use the algorithm packages directly when you want to control preparation, training, evaluation, or artifact writing without going through the full CLI workflow.


## 4) Run a Study Grid

Default grid config:
- `configs/studies/default.json`

Run:

```bash
uv run make study STUDY_NAME=default MAX_WORKERS=4
```

Resume:

```bash
uv run make study-resume STUDY_NAME=default MAX_WORKERS=4
```


## 5) Evaluate CSCG Snapshots (post-training)

CSCG training saves model snapshots but does not compute paper-precision metrics inline. After a study completes, run the evaluation pass separately:

```bash
python main.py \
  --evaluate_study_snapshots \
  --study_root results/current/studies \
  --study_name default
```

This loads snapshots from every run directory in the study and evaluates them using a shared walk cache, writing `metrics/cscg/*_paper_precision.json` in each run folder. Walks are generated once per unique `(level, sensor, sensor_range, rand_prob, seed, n_walks, walk_length)` combination and reused across all runs and checkpoints.

To re-evaluate snapshots for a single run:

```bash
python main.py \
  --reevaluate_paper_precision_snapshots \
  --study_root results/current/studies \
  --study_name default \
  --run_id <run_id>
```


## 6) Recover Missing Outputs

Retry failed runs:

```bash
uv run make study-retry STUDY_NAME=default MAX_WORKERS=4
```

Retry incomplete runs from manifest:

```bash
uv run make study-retry-incomplete STUDY_NAME=default MAX_WORKERS=4
```

Run with W&B:

```bash
uv run make study-wandb STUDY_NAME=default WANDB_PROJECT=mercuryv5
uv run make study-wandb-offline STUDY_NAME=default WANDB_PROJECT=mercuryv5
uv run make study-compare-pocml STUDY_NAME=default MAX_WORKERS=4
uv run make study-compare-cscg STUDY_NAME=default MAX_WORKERS=4
uv run make study-compare-all STUDY_NAME=default MAX_WORKERS=4
```

Check:
- `results/current/studies/default/study_errors.jsonl`
- `results/current/studies/default/study_incomplete.jsonl`
- `results/current/studies/default/<run_id>/run_config.json`
- `results/current/studies/default/<run_id>/run_status.json`
- `results/current/studies/default/<run_id>/comparison_summary.json` (when `--baseline_pocml True`, `--baseline_cscg True`, or `--baseline_all True`)

W&B behavior:
- single runs create one W&B run
- study runs create one study-level W&B run plus one W&B run per worker/run
- only current-run artifacts and native baseline metrics are logged

If you do not install `wandb` into the current `uv` environment, W&B flags are safely ignored with a warning.
If you do not install `torch` into the current `uv` environment, `--baseline_pocml True` runs will fail with a dependency error.
If `numba` is unavailable, CSCG runs use a no-JIT fallback and may be slower.


## 7) Inspect Summary

Study summary parquet:

```bash
results/current/studies/<study_name>/study_summary.parquet
```

This includes core metrics plus tracked run parameters.

If baselines are enabled, individual run folders may also contain `comparison_summary.json` with native-only baseline sections.


## 8) Evaluate Study Snapshots (Canonical-v4 Protocol)

After running a study that uses the canonical-v4 evaluation service, generate evaluation records for all snapshots:

```bash
python main.py \
  --evaluate_study \
  --study_root results/current/studies/<study_name> \
  --protocol_id canonical-v4-<hash>
```

Records are written to `<run_root>/evaluations/<protocol_id>/snapshots/step=XXXXXXXX.json`.
The `protocol_id` encodes all walk-generation and evaluation hyperparameters — use the same ID for all downstream plotting.


## 9) Generate Canonical Plots

All plot generators live in `src/mercury_runs/analysis/plots/` and accept `--study_root` + `--protocol_id`.
Run them as Python modules (or via the thin shims in `scripts/analysis/`):

**Summary forest / matrix / profile plots:**
```bash
python -m mercury_runs.analysis.plots.batch_summary_forest \
  --study_root results/current/studies/<study_name> \
  --protocol_id canonical-v4-<hash> \
  --output_dir results/current/notebook_plots/<study_name>
```

**Transition residual maps (multi-seed, multi-method):**
```bash
python -m mercury_runs.analysis.plots.selected_residual_maps \
  --study_root results/current/studies/<study_name> \
  --protocol_id canonical-v4-<hash> \
  --output_dir results/current/notebook_plots/<study_name>_residual_maps
```

**Thresholded edge metrics:**
```bash
python -m mercury_runs.analysis.plots.threshold_edge_metrics \
  --study_root results/current/studies/<study_name> \
  --protocol_id canonical-v4-<hash> \
  --output_dir results/current/notebook_plots/<study_name>_threshold_metrics
```

**Threshold robustness figures** (reads CSVs from the step above):
```bash
python -m mercury_runs.analysis.plots.threshold_robustness \
  --collapsed_summary results/..../threshold_edge_metrics_summary.csv \
  --action_summary results/..../threshold_edge_metrics_summary.csv \
  --output_dir results/current/notebook_plots/<study_name>_robustness
```

**Per-run residual maps** (single run, reads snapshots directly):
```bash
python -m mercury_runs.analysis.plots.transition_residual_maps_cli \
  --run_root results/current/studies/<study_name>/<run_id>
```


## 10) Useful References

- Full documentation: [`README.md`](../README.md)
- Structure guide: [`PROJECT_STRUCTURE.md`](../PROJECT_STRUCTURE.md)
- Architecture guide: [`docs/ARCHITECTURE.md`](../docs/ARCHITECTURE.md)
- Evaluation architecture: [`docs/First-Class Snapshot Evaluation Architecture.md`](../docs/First-Class%20Snapshot%20Evaluation%20Architecture.md)
- Analysis tools: [`scripts/analysis/README.md`](../scripts/analysis/README.md)
