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


## 5) Recover Missing Outputs

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


## 6) Inspect Summary

Study summary parquet:

```bash
results/current/studies/<study_name>/study_summary.parquet
```

This includes core metrics plus tracked run parameters.

If baselines are enabled, individual run folders may also contain `comparison_summary.json` with native-only baseline sections.


## 7) Useful References

- Full documentation: [`README.md`](../README.md)
- Structure guide: [`PROJECT_STRUCTURE.md`](../PROJECT_STRUCTURE.md)
- Architecture guide: [`docs/ARCHITECTURE.md`](../docs/ARCHITECTURE.md)
- Analysis tools: [`scripts/analysis/README.md`](../scripts/analysis/README.md)
