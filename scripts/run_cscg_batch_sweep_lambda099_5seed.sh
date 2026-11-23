#!/usr/bin/env bash
# Train the five-seed CSCG batch-size sweep at lambda=0.99, evaluate its
# 0/2000/2000000-step snapshots with evaluation-v5, and render the reference products.
#
# Usage:
#   bash scripts/run_cscg_batch_sweep_lambda099_5seed.sh
#   RUN_TRAIN=0 bash scripts/run_cscg_batch_sweep_lambda099_5seed.sh
#
# PILOT_ROOT and MAX_WORKERS may be overridden. Training requires an empty
# source-study directory; RUN_TRAIN=0 reuses a completed study for evaluation.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PYTHONPATH="$PWD:$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/private/tmp/matplotlib-cscg-lambda099-pilot}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/private/tmp}"

PILOT_ROOT="${PILOT_ROOT:-results/current/action_alignment_cscg_lambda099_5seed}"
MAX_WORKERS="${MAX_WORKERS:-4}"
RUN_TRAIN="${RUN_TRAIN:-1}"
STUDY_NAME="cscg_batch_sweep"
STUDY_CONFIG="configs/studies/cscg_batch_sweep_lambda099_5seed.json"
SOURCE_STUDIES_ROOT="$PILOT_ROOT/studies"
EVALUATION_STUDIES_ROOT="$PILOT_ROOT/evaluation_v5/studies"
NOTEBOOK_PLOTS_ROOT="$PILOT_ROOT/notebook_plots"
SOURCE_STUDY="$SOURCE_STUDIES_ROOT/$STUDY_NAME"

if [[ "$RUN_TRAIN" == "1" ]]; then
    if [[ -d "$SOURCE_STUDY" ]] && [[ -n "$(find "$SOURCE_STUDY" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
        printf 'Training target already contains files: %s\n' "$SOURCE_STUDY" >&2
        printf 'Use a new PILOT_ROOT or RUN_TRAIN=0 for evaluation and plots.\n' >&2
        exit 2
    fi

    uv run python -u main.py \
        --study \
        --study_config "$STUDY_CONFIG" \
        --study_root "$SOURCE_STUDIES_ROOT" \
        --study_name "$STUDY_NAME" \
        --max_workers "$MAX_WORKERS"
elif [[ "$RUN_TRAIN" != "0" ]]; then
    printf 'RUN_TRAIN must be 0 or 1; got %s\n' "$RUN_TRAIN" >&2
    exit 2
fi

if [[ ! -f "$SOURCE_STUDY/study_manifest.jsonl" ]]; then
    printf 'Study manifest not found: %s\n' "$SOURCE_STUDY/study_manifest.jsonl" >&2
    exit 2
fi
manifest_rows="$(wc -l < "$SOURCE_STUDY/study_manifest.jsonl")"
if [[ "$manifest_rows" -ne 60 ]]; then
    printf 'Expected 60 study runs, found %s in %s\n' "$manifest_rows" "$SOURCE_STUDY/study_manifest.jsonl" >&2
    exit 2
fi

uv run python -u scripts/analysis/evaluate_v5.py \
    --source-studies-root "$SOURCE_STUDIES_ROOT" \
    --output-studies-root "$EVALUATION_STUDIES_ROOT" \
    --studies "$STUDY_NAME" \
    --levels 13 20 21 22 \
    --seeds 0-4 \
    --methods cscg \
    --checkpoints 0 2000 2000000 \
    --reference-step 2000 \
    --rand-prob 0.1 \
    --max-workers "$MAX_WORKERS"

uv run python -u scripts/analysis/postprocess_evaluation_v5.py \
    --evaluation-studies-root "$EVALUATION_STUDIES_ROOT" \
    --notebook-plots-root "$NOTEBOOK_PLOTS_ROOT" \
    --studies "$STUDY_NAME"

printf 'Plots and manifest: %s/%s_eval_v5\n' "$NOTEBOOK_PLOTS_ROOT" "$STUDY_NAME"
