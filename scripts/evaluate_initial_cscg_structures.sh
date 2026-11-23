#!/usr/bin/env bash
# Evaluate the six main-study CSCG initialisations before CSCG training.
#
# This workflow is isolated: main_study is read-only and all generated records,
# logs, reports, caches, and plots go under OUTPUT_ROOT.
#
# Examples:
#   bash scripts/evaluate_initial_cscg_structures.sh
#   LEVELS="13" SEEDS="0" EVAL_NUM_WALKS=2 EVAL_WALK_LENGTH=100 \
#     bash scripts/evaluate_initial_cscg_structures.sh
#   RUN_EVAL=0 bash scripts/evaluate_initial_cscg_structures.sh
set -euo pipefail

STUDY_ROOT="${STUDY_ROOT:-results/current/studies}"
STUDY_NAME="${STUDY_NAME:-main_study}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/current/initial_structure_evaluation/${STUDY_NAME}}"
PLOT_DIR="${PLOT_DIR:-${OUTPUT_ROOT}/plots}"
REFERENCE_STEP="${REFERENCE_STEP:-4000}"
BATCH_SIZE="${BATCH_SIZE:-40}"
RAND_PROB="${RAND_PROB:-0.1}"
EVAL_MAX_WORKERS="${EVAL_MAX_WORKERS:-4}"
EVAL_NUM_WALKS="${EVAL_NUM_WALKS:-100}"
EVAL_WALK_LENGTH="${EVAL_WALK_LENGTH:-10000}"
EVAL_SEED="${EVAL_SEED:-0}"
EVAL_FORCE="${EVAL_FORCE:-0}"
RUN_EVAL="${RUN_EVAL:-1}"
RUN_PLOTS="${RUN_PLOTS:-1}"

read -r -a LEVELS_ARRAY <<< "${LEVELS:-13 20 21 22}"
read -r -a METHODS_ARRAY <<< "${METHODS:-mercury_cscg_hybrid tmgwr_cscg_hybrid cscg}"
SEEDS="${SEEDS:-0-29}"

export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}.:src"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/private/tmp/matplotlib-${USER:-mercury}}"
mkdir -p "$MPLCONFIGDIR"

command=(
    python -u scripts/analysis/evaluate_initial_cscg_structures.py
    --study-root "$STUDY_ROOT"
    --study-name "$STUDY_NAME"
    --output-root "$OUTPUT_ROOT"
    --plot-dir "$PLOT_DIR"
    --levels "${LEVELS_ARRAY[@]}"
    --seeds "$SEEDS"
    --methods "${METHODS_ARRAY[@]}"
    --rand-prob "$RAND_PROB"
    --batch-size "$BATCH_SIZE"
    --reference-step "$REFERENCE_STEP"
    --max-workers "$EVAL_MAX_WORKERS"
    --eval-num-walks "$EVAL_NUM_WALKS"
    --eval-walk-length "$EVAL_WALK_LENGTH"
    --eval-seed "$EVAL_SEED"
)

if [[ "$EVAL_FORCE" == "1" ]]; then
    command+=(--force)
fi
if [[ "$RUN_EVAL" != "1" ]]; then
    command+=(--skip-eval)
fi
if [[ "$RUN_PLOTS" != "1" ]]; then
    command+=(--skip-plots)
fi

printf 'Source : %s/%s\n' "$STUDY_ROOT" "$STUDY_NAME"
printf 'Output : %s\n' "$OUTPUT_ROOT"
printf 'Levels : %s\n' "${LEVELS_ARRAY[*]}"
printf 'Seeds  : %s\n' "$SEEDS"
printf 'Methods: %s\n' "${METHODS_ARRAY[*]}"
printf 'Walks  : %s x %s steps\n\n' "$EVAL_NUM_WALKS" "$EVAL_WALK_LENGTH"

"${command[@]}"
