# MercuryV5

MercuryV5 is an experimental research codebase for:
- generating maze datasets,
- running sensory + latent learning pipelines,
- running native baseline comparisons,
- packaging run outputs with metadata,
- computing study metrics and summaries.

The project supports both:
- single-run execution (`main.py`), and
- study/grid execution with resume/retry behavior.

The canonical movable code now lives under `src/mercury_runs` and is organized as:
- `algorithms/` for Mercury, POCML, and CSCG
- `application/` for run and study orchestration
- `domain/` for Pydantic models and contracts
- `infrastructure/` for storage, reporting, runtime, and W&B
- `interfaces/` for the CLI


## Table of Contents
- [1. Project Goals](#1-project-goals)
- [2. Repository Layout](#2-repository-layout)
- [3. Environment Setup](#3-environment-setup)
- [4. Core Workflow](#4-core-workflow)
- [5. Running Single Experiments](#5-running-single-experiments)
- [6. Running Studies (Grid Search)](#6-running-studies-grid-search)
- [7. Study Config Files](#7-study-config-files)
- [8. Output Artifacts](#8-output-artifacts)
- [9. Metadata and Reproducibility](#9-metadata-and-reproducibility)
- [10. Script Organization](#10-script-organization)
- [11. W&B Integration](#11-wb-integration)
- [12. POCML Baseline Comparison](#12-pocml-baseline-comparison)
- [13. CSCG Baseline Comparison](#13-cscg-baseline-comparison)
- [14. Troubleshooting](#14-troubleshooting)
- [15. Testing](#15-testing)
- [16. Common Commands](#16-common-commands)


## 1. Project Goals

This repository is designed to:
- simulate trajectories in maze environments,
- train/update sensory and latent state representations,
- evaluate precision metrics against ground-truth behavior,
- run repeatable study sweeps with parameter tracking,
- preserve run provenance in parquet metadata.


## 2. Repository Layout

High-level structure:

- `src/`
  - Core package code (`mercury`, `mercury_runs`, `maze_environment`, helpers).
- `src/mercury_runs/algorithms/`
  - Canonical Python APIs for Mercury, POCML, and CSCG, each split into `config`, `prepare`, `train`, `evaluate`, and artifact modules where applicable.
- `src/mercury_runs/application/`
  - Single-run and study orchestration.
- `src/mercury_runs/domain/`
  - Pydantic models for run identity, status, config, and study contracts.
- `src/mercury_runs/infrastructure/`
  - Artifact layout, persistence, reporting, W&B logging, and runtime helpers.
- `src/mercury_runs/interfaces/`
  - CLI parsing and dispatch.
- `main.py`
  - Main CLI entrypoint for single-run and study modes.
- `configs/studies/`
  - JSON grid configs for studies.
- `scripts/`
  - Supported offline analysis tools in `scripts/analysis/`.
- `tests/`
  - Unit tests.
- `datasets/`
  - Input/generated datasets.
- `results/`
  - Bundles, metrics, study outputs.
- `logs/`
  - Runtime logs.
- `Makefile`
  - Convenience commands for study runs.
- `external/POCML/`
  - Cloned baseline repository for side-by-side comparison runs.
- `external/naturecomm_cscg/`
  - Cloned CSCG baseline repository for side-by-side comparison runs.

Additional structure notes are in [`PROJECT_STRUCTURE.md`](PROJECT_STRUCTURE.md).

Documentation quick links:
- Quick start: [`docs/QUICKSTART.md`](docs/QUICKSTART.md)
- Analysis scripts: [`scripts/analysis/README.md`](scripts/analysis/README.md)
- Architecture: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)


## 3. Environment Setup

### Requirements
- Python 3.13+ (recommended to match current lock/build setup)
- Local writable workspace for `datasets/`, `results/`, and `logs/`

### Install dependencies

Use `uv`:

```bash
uv sync
```

Optional project dependencies can also be added with `uv`:

```bash
uv add wandb
uv add torch
```


## 4. Core Workflow

Typical lifecycle:

1. Generate/reuse dataset artifacts (`generate_data` path in `main.py`).
2. Run Mercury through `src/mercury_runs.algorithms.mercury`.
3. Optionally run native POCML and CSCG baselines through `src/mercury_runs.algorithms.pocml` and `src/mercury_runs.algorithms.cscg`.
4. Write run status, summaries, and Mercury precision metrics.
5. Summarize across runs for studies (`study_summary.parquet`).

In study mode, each run is tracked through:
- `study_manifest.jsonl` (scheduled runs),
- `study_errors.jsonl` (execution failures),
- optional `study_incomplete.jsonl` (missing outputs).

## Python API

The primary control surface is Python-first and typed:

```python
from mercury_runs import MercuryConfig, POCMLConfig, CSCGConfig, run_mercury, run_pocml, run_cscg
```

Important exports:
- `run_mercury(config)`
- `run_pocml(config)`
- `run_cscg(config)`
- `run_single(args)`
- `run_study(args)`

Use the CLI when you want end-to-end orchestration. Use the algorithm packages when you want direct control over preparation, training, evaluation, or artifact writing.


## 5. Running Single Experiments

Run one configuration:

```bash
python main.py \
  --level 16 \
  --sensor "cardinal distance" \
  --sensor_range 1 \
  --memory_length 40 \
  --activation_threshold 0.95 \
  --am_lr 0.001
```

To force dataset generation to save only collision-free trajectories, add:

```bash
python main.py --level 16 --valid_trajectories_only
```

Important behavior:
- `run_id` defaults to a stable hash of run payload if not provided.
- Reuse is strict: only completed schema-valid runs with the exact expected artifacts are reusable.
- Polluted or failed run directories are never reused.
- If an auto-generated `run_id` collides with an invalid directory, the pipeline writes a fresh suffixed run directory instead of mixing artifacts.
- Dataset generation can be reused unless `--no_reuse_existing_dataset` is set.
- `--valid_trajectories_only` makes dataset generation keep sampling until it saves `num_steps` non-collision rows, and those datasets are cached separately from default datasets.


## 6. Running Studies (Grid Search)

### Recommended path (Makefile)

Default config path:
- `configs/studies/default.json`

Run study:

```bash
make study STUDY_NAME=default MAX_WORKERS=4
```

Resume completed study:

```bash
make study-resume STUDY_NAME=default MAX_WORKERS=4
```

Retry failed runs from `study_errors.jsonl`:

```bash
make study-retry STUDY_NAME=default MAX_WORKERS=4
```

Retry incomplete runs from manifest (runs missing states/metrics):

```bash
make study-retry-incomplete STUDY_NAME=default MAX_WORKERS=4
```

Run study with Weights & Biases:

```bash
make study-wandb STUDY_NAME=default MAX_WORKERS=4 WANDB_PROJECT=mercuryv5
```

Run study with W&B offline mode:

```bash
make study-wandb-offline STUDY_NAME=default MAX_WORKERS=4 WANDB_PROJECT=mercuryv5
```

### Direct CLI equivalent

```bash
python main.py --study \
  --study_config configs/studies/default.json \
  --study_root results/current/studies \
  --study_name default \
  --max_workers 4
```


## 7. Study Config Files

Study configs are JSON objects where each key maps to either:
- a list (grid dimension), or
- a scalar (auto-normalized to a one-item list).

Example file:
- `configs/studies/default.json`

Common keys:
- dataset/run: `level`, `sensor`, `sensor_range`, `seed`, `rand_prob`, `num_steps`
- pipeline: `memory_length`, `window_length`
- sensory hyperparams: `activation_threshold`, `sensory_weighting`, etc.
- latent hyperparams: `ambiguity_threshold`, `trace_decay`, `mixture_alpha`, `mixture_beta`, `memory_replay`, `memory_disambiguation`
- action-map hyperparams: `am_n_codebook`, `am_lr`, `am_sigma`, `am_key`

`study_root` and `study_name` are injected from CLI/Make options.


## 8. Output Artifacts

### Dataset artifacts

Generated under:

- `datasets/level*/<sensor>/...`

Per dataset run:
- `<dataset_run_id>.parquet`
- `<dataset_run_id>.metadata.json`

### Mercury bundle artifacts

Per run folder:

- `bundles/mercury/.../*_states.parquet`
  - contains `ground_truth_bmu`, `sensory_bmu`, `latent_bmu`, `latent_node_count`
- `bundles/mercury/.../*_latent_graph.npz`
  - persisted final Mercury latent graph for post-run internal graph plotting
- `plots/mercury_internal_graph.png`
- `plots/internal_graphs_comparison.png`

- `metrics/mercury/*_precision.parquet`
  - rolling precision diagnostics
- `metrics/mercury/*_paper_precision.json`
  - canonical paper-style fixed-map precision over fresh evaluation walks

### Baseline artifacts

POCML:
- `bundles/pocml/.../<run_id>_states.parquet`
- `bundles/pocml/.../<run_id>_embeddings.npz`
- `bundles/pocml/.../<run_id>_pocml_model.pt`
- `metrics/pocml/<run_id>_precision.parquet` (rolling diagnostic precision)
- `metrics/pocml/<run_id>_paper_precision.json` (canonical paper-style precision, only when `--paper_precision True`)

CSCG:
- `bundles/cscg/.../<run_id>_states.parquet`
- `bundles/cscg/.../<run_id>_cscg_model.npz`
- `metrics/cscg/<run_id>_precision.parquet` (rolling diagnostic precision)
- `metrics/cscg/<run_id>_paper_precision.json` (canonical paper-style precision, only when `--paper_precision True`)

Important rule:
- all three methods write rolling precision diagnostics under `metrics/<method>/*_precision.parquet`
- all three methods also write canonical paper-style precision JSON artifacts under `metrics/<method>/*_paper_precision.json`

### Study outputs

Under `results/current/studies/<study_name>/`:
- run directories named by `run_id`
- per-run `run_config.json` (full CLI + resolved payload + reproduction command)
- per-run `run_status.json` (lifecycle, timestamps, artifacts, error if failed)
- `study_manifest.jsonl`
- `study_errors.jsonl` (if failures occurred)
- `study_incomplete.jsonl` (if missing outputs remain)
- `study_summary.parquet`

Historical outputs are preserved under:
- `results/legacy/<timestamp>/...`


## 9. Metadata and Reproducibility

Bundle metadata is embedded into parquet key-value metadata and parsed via:
- `src/mercury_runs/save_results.py`
- schema in `src/mercury_runs/schemas_results.py`

Saved metadata includes:
- run identity and timestamp,
- source dataset reference,
- sensory/latent/action-map parameters,
- memory length,
- full `run_parameters` payload,
- source dataset metadata (latent + ground-truth dataset refs),
- notes.

This enables reconstruction of experiment settings directly from artifacts.

In addition, each run directory stores:
- `run_config.json` for full execution config/repro command
- `run_status.json` for run lifecycle and artifact status


## 10. Script Organization

The supported script surface is intentionally small:

- `main.py` is the canonical execution entrypoint for single runs and studies.
- `Makefile` wraps the supported study commands.
- `scripts/analysis/` contains the supported post-run analysis tools.

Legacy study and live wrapper scripts were removed during the rebuild.


## 11. W&B Integration

W&B logging is optional and disabled by default.

CLI options:
- `--wandb [True|False]`
- `--wandb_project`
- `--wandb_entity`
- `--wandb_group`
- `--wandb_tags` (comma-separated)
- `--wandb_mode {online,offline,disabled}`
- `--wandb_job_type`
- `--wandb_log_artifacts [True|False]`
- `--wandb_run_name`

When enabled, single runs log one W&B run with validated config, subsystem status, Mercury precision stats, native baseline metrics, and current-run artifacts only.
Paper-style fixed-map precision is opt-in. Use `--paper_precision_mode final` for one post-training evaluation or `--paper_precision_mode per_iteration` to record training-time history. Mercury interprets `per_iteration` as step-interval evaluation along its online pass; CSCG and POCML use their native iteration units. The evaluation workload is configurable with `--paper_precision_num_walks` and `--paper_precision_walk_length`. When paper precision is enabled, `metrics/<method>/*_precision.png` visualizes paper precision; the rolling precision diagnostic remains in `*_precision.parquet`.
In study mode, each worker run logs its own W&B run and the parent process logs one aggregate study run.

If W&B is not installed, runs continue and emit a warning instead of failing.


## 12. POCML Baseline Comparison

You can run the `POCML` method from [`external/POCML`](external/POCML) against the same generated trajectory data used by Mercury runs.

Requirements:
- clone baseline repo:
  - `git clone https://github.com/calvin4770/POCML external/POCML`
- install PyTorch:
  - `uv add torch`

Single run with baseline:

```bash
python main.py \
  --level 16 \
  --sensor "cardinal distance" \
  --sensor_range 1 \
  --baseline_pocml True
```

Optional (POCML trainer node IDs from labels instead of observation IDs):

```bash
python main.py \
  --level 16 \
  --sensor "cardinal distance" \
  --sensor_range 1 \
  --baseline_pocml True \
  --pocml_use_ground_truth_node_ids True
```

Study run with baseline:

```bash
make study-compare-pocml STUDY_NAME=default MAX_WORKERS=4
```

Baseline artifacts are saved alongside Mercury run folders:
- `bundles/mercury/.../<run_id>_latent_graph.npz`
- `plots/mercury_internal_graph.png`
- `bundles/pocml/.../<run_id>_states.parquet`
- `bundles/pocml/.../<run_id>_embeddings.npz`
- `bundles/pocml/.../<run_id>_pocml_model.pt`
- `metrics/pocml/<run_id>_precision.parquet`
- `metrics/pocml/<run_id>_paper_precision.json` when `--paper_precision True`
- `plots/pocml_internal_graph.png` (when POCML is enabled)
- `comparison_summary.json` (native metrics plus rolling and paper-style precision summaries)

POCML run metadata also includes native evaluation signals:
- GT-space n-step observation prediction accuracy
- direct observation-space n-step prediction accuracy
- mean/std confidence on true next-observation labels
- trajectory log-likelihood

Metric interpretation:
- `pocml_native_metrics.n_step_observation_prediction_accuracy`: canonical POCML metric scored against future `ground_truth_bmu` via `obs_to_gt`
- `pocml_native_metrics.direct_observation_prediction_accuracy`: paper-aligned POCML observation-space metric family
- `pocml_capacity_precision`: rolling diagnostic precision summary computed from `latent_proxy_state_id`
- `pocml_paper_precision`: canonical fixed-map precision over fresh evaluation walks, still capacity-dependent because it uses `latent_proxy_state_id`
- POCML precision should be interpreted as `precision_at_assumed_state_capacity(K)`, not a clean eFeX-equivalent latent recovery score
- use the canonical GT-space metric for cross-method comparison and the direct observation metric for POCML-specific validation

This keeps the original Mercury artifacts unchanged while adding a comparable baseline output per run.


## 13. CSCG Baseline Comparison

You can also run CSCG from [`external/naturecomm_cscg`](external/naturecomm_cscg) against the same generated trajectory data used by Mercury runs.

Requirements:
- clone baseline repo:
  - `git clone https://github.com/vicariousinc/naturecomm_cscg external/naturecomm_cscg`
- install baseline dependencies (if available in your environment):
  - `uv add <required-package>`
  - if you need the exact vendored dependency set, install those packages into the same `uv` environment rather than using a separate `pip` workflow

Single run with CSCG baseline:

```bash
python main.py \
  --level 16 \
  --sensor "cardinal distance" \
  --sensor_range 1 \
  --baseline_cscg True
```

Study run with CSCG baseline:

```bash
make study-compare-cscg STUDY_NAME=default MAX_WORKERS=4
```

Artifacts:
- `bundles/mercury/.../<run_id>_latent_graph.npz`
- `bundles/cscg/.../<run_id>_states.parquet`
- `bundles/cscg/.../<run_id>_cscg_model.npz`
- `metrics/cscg/<run_id>_precision.parquet`
- `metrics/cscg/<run_id>_paper_precision.json` when `--paper_precision True`
- `plots/cscg_internal_graph.png` (when CSCG is enabled)
- `plots/internal_graphs_comparison.png`
- `comparison_summary.json` includes `cscg_native_metrics` and `cscg_config` when CSCG is enabled.
- `comparison_summary.json` also includes `cscg_metrics` and `cscg_paper_precision` when CSCG precision artifacts are present.

Training semantics:
- `--cscg_train_algorithm em` means EM-only training
- `--cscg_train_algorithm viterbi` means EM followed by Viterbi refinement
- CSCG reporting uses native CSCG objective metrics, decode summaries, and `n_step_observation_prediction_accuracy`

Run all three methods in one run:

```bash
python main.py \
  --level 16 \
  --sensor "cardinal distance" \
  --sensor_range 1 \
  --baseline_all True
```

Or in study mode:

```bash
make study-compare-all STUDY_NAME=default MAX_WORKERS=4
```

## 14. Troubleshooting

### “Outputs are incomplete”

Symptoms:
- run directories exist but missing `*_states.parquet` and/or `*_precision.parquet`.

Use:

```bash
make study-retry-incomplete STUDY_NAME=<name> MAX_WORKERS=4
```

Then inspect:
- `results/current/studies/<name>/study_incomplete.jsonl`
- `results/current/studies/<name>/study_errors.jsonl`

### `main.py --help` was slow/noisy

Run-only imports were moved into execution paths; current CLI help should be lightweight.

### “I need more direct control over algorithms”

Use the algorithm packages directly instead of routing everything through `main.py`:
- `mercury_runs.algorithms.mercury`
- `mercury_runs.algorithms.pocml`
- `mercury_runs.algorithms.cscg`

Each package is split into smaller step modules so you can control preparation, training, evaluation, and artifact writing independently.

### Matplotlib/font cache warnings

Set writable cache dirs, for example:

```bash
export MPLCONFIGDIR=/tmp/matplotlib
```


## 15. Testing

Run all tests:

```bash
pytest -q
```

Run focused suites:

```bash
pytest -q tests/src/data_helper/test_csv_loader.py
pytest -q tests/src/mercury_runs/test_validation_models.py
```


## 16. Common Commands

Show CLI options:

```bash
python main.py --help
```

Single run:

```bash
python main.py --level 17 --sensor "cardinal distance" --sensor_range 1
```

Study run:

```bash
make study STUDY_NAME=default MAX_WORKERS=4
```

Study recovery:

```bash
make study-retry STUDY_NAME=default MAX_WORKERS=4
make study-retry-incomplete STUDY_NAME=default MAX_WORKERS=4
```

Study with W&B:

```bash
make study-wandb STUDY_NAME=default WANDB_PROJECT=mercuryv5
make study-wandb-offline STUDY_NAME=default WANDB_PROJECT=mercuryv5
```

Study with POCML baseline comparison:

```bash
make study-compare-pocml STUDY_NAME=default MAX_WORKERS=4
```

Canonical help entrypoint:

```bash
python main.py --help
```
