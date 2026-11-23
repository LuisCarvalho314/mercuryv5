#!/usr/bin/env bash
# Full corrected-study rerun, or isolated frozen evaluation and post-processing.
#
# Evaluation-only (existing behaviour):
#   bash scripts/evaluate_v5.sh
#
# Clean main-study and CSCG batch-sweep retrain + forced Evaluation-v5 + post-processing:
#   RUN_TRAIN=1 bash scripts/evaluate_v5.sh
#
# RUN_TRAIN defaults to a new isolated tree under ACTION_ALIGNMENT_RERUN_ROOT.
# Set TRAIN_RESUME=1 to resume that tree after an interrupted full rerun.
set -euo pipefail

UV_BIN="${UV_BIN:-uv}"
if ! command -v "$UV_BIN" >/dev/null 2>&1; then
    echo "uv executable not found: $UV_BIN" >&2
    echo "Install uv or set UV_BIN to its path." >&2
    exit 2
fi

RUN_TRAIN="${RUN_TRAIN:-0}"
ACTION_ALIGNMENT_RERUN_ROOT="${ACTION_ALIGNMENT_RERUN_ROOT:-results/current/action_alignment_rerun}"
if [[ "$RUN_TRAIN" == "1" ]]; then
    SOURCE_STUDIES_ROOT="${SOURCE_STUDIES_ROOT:-$ACTION_ALIGNMENT_RERUN_ROOT/studies}"
    OUTPUT_STUDIES_ROOT="${OUTPUT_STUDIES_ROOT:-$ACTION_ALIGNMENT_RERUN_ROOT/evaluation_v5/studies}"
    NOTEBOOK_PLOTS_ROOT="${NOTEBOOK_PLOTS_ROOT:-$ACTION_ALIGNMENT_RERUN_ROOT/notebook_plots}"
else
    SOURCE_STUDIES_ROOT="${SOURCE_STUDIES_ROOT:-results/current/studies}"
    OUTPUT_STUDIES_ROOT="${OUTPUT_STUDIES_ROOT:-results/current/evaluation_v5/studies}"
    NOTEBOOK_PLOTS_ROOT="${NOTEBOOK_PLOTS_ROOT:-results/current/notebook_plots}"
fi
MAX_WORKERS="${MAX_WORKERS:-4}"
TRAIN_MAX_WORKERS="${TRAIN_MAX_WORKERS:-$MAX_WORKERS}"
TRAIN_MAIN_STUDY_NAME="${TRAIN_MAIN_STUDY_NAME:-main_study}"
TRAIN_MAIN_STUDY_CONFIG="${TRAIN_MAIN_STUDY_CONFIG:-configs/studies/default_matrix_smoke_30_seed.json}"
TRAIN_BATCH_STUDY_NAME="${TRAIN_BATCH_STUDY_NAME:-cscg_batch_sweep}"
TRAIN_BATCH_STUDY_CONFIG="${TRAIN_BATCH_STUDY_CONFIG:-configs/studies/cscg_batch_sweep.json}"
TRAIN_RESUME="${TRAIN_RESUME:-0}"
SEEDS="${SEEDS:-0-29}"
RAND_PROB="${RAND_PROB:-0.1}"
EVAL_NUM_WALKS="${EVAL_NUM_WALKS:-100}"
EVAL_WALK_LENGTH="${EVAL_WALK_LENGTH:-10000}"
EVAL_SEED="${EVAL_SEED:-0}"
RUN_EVAL="${RUN_EVAL:-1}"
RUN_POSTPROCESS="${RUN_POSTPROCESS:-1}"
EVAL_FORCE="${EVAL_FORCE:-$RUN_TRAIN}"

DEFAULT_STUDIES="$TRAIN_MAIN_STUDY_NAME $TRAIN_BATCH_STUDY_NAME"
read -r -a STUDIES_ARRAY <<< "${STUDIES:-$DEFAULT_STUDIES}"
read -r -a LEVELS_ARRAY <<< "${LEVELS:-13 20 21 22}"
read -r -a CHECKPOINTS_ARRAY <<< "${CHECKPOINTS:-0 2000 4000 2000000}"

export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}.:src"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/private/tmp/matplotlib-evaluation-v5}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/private/tmp}"
mkdir -p "$MPLCONFIGDIR"

if [[ "$RUN_TRAIN" == "1" ]]; then
    TRAIN_STUDY_NAMES=("$TRAIN_MAIN_STUDY_NAME" "$TRAIN_BATCH_STUDY_NAME")
    TRAIN_STUDY_CONFIGS=("$TRAIN_MAIN_STUDY_CONFIG" "$TRAIN_BATCH_STUDY_CONFIG")

    if [[ "${#STUDIES_ARRAY[@]}" -ne 2 || \
          "${STUDIES_ARRAY[0]}" != "$TRAIN_MAIN_STUDY_NAME" || \
          "${STUDIES_ARRAY[1]}" != "$TRAIN_BATCH_STUDY_NAME" ]]; then
        echo "RUN_TRAIN=1 requires STUDIES=$DEFAULT_STUDIES; got: ${STUDIES_ARRAY[*]}" >&2
        exit 2
    fi

    # Validate both targets before starting either long-running study.
    for study_index in "${!TRAIN_STUDY_NAMES[@]}"; do
        train_study_name="${TRAIN_STUDY_NAMES[$study_index]}"
        train_study_config="${TRAIN_STUDY_CONFIGS[$study_index]}"
        if [[ ! -f "$train_study_config" ]]; then
            echo "Training study config not found: $train_study_config" >&2
            exit 2
        fi
        train_study_dir="$SOURCE_STUDIES_ROOT/$train_study_name"
        if [[ "$TRAIN_RESUME" != "1" && -d "$train_study_dir" ]] && \
           [[ -n "$(find "$train_study_dir" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
            echo "Refusing to mix a clean retrain with existing files: $train_study_dir" >&2
            echo "Choose a new ACTION_ALIGNMENT_RERUN_ROOT, or set TRAIN_RESUME=1 after an interrupted rerun." >&2
            exit 2
        fi
    done

    for study_index in "${!TRAIN_STUDY_NAMES[@]}"; do
        train_command=(
            "$UV_BIN" run python -u main.py
            --study
            --study_config "${TRAIN_STUDY_CONFIGS[$study_index]}"
            --study_root "$SOURCE_STUDIES_ROOT"
            --study_name "${TRAIN_STUDY_NAMES[$study_index]}"
            --max_workers "$TRAIN_MAX_WORKERS"
        )
        if [[ "$TRAIN_RESUME" == "1" ]]; then
            train_command+=(--resume)
        fi
        "${train_command[@]}"
    done
fi

if [[ "$RUN_EVAL" == "1" ]]; then
    command=(
        "$UV_BIN" run python -u scripts/analysis/evaluate_v5.py
        --source-studies-root "$SOURCE_STUDIES_ROOT"
        --output-studies-root "$OUTPUT_STUDIES_ROOT"
        --studies "${STUDIES_ARRAY[@]}"
        --levels "${LEVELS_ARRAY[@]}"
        --seeds "$SEEDS"
        --checkpoints "${CHECKPOINTS_ARRAY[@]}"
        --rand-prob "$RAND_PROB"
        --max-workers "$MAX_WORKERS"
        --eval-num-walks "$EVAL_NUM_WALKS"
        --eval-walk-length "$EVAL_WALK_LENGTH"
        --eval-seed "$EVAL_SEED"
    )
    if [[ "$EVAL_FORCE" == "1" ]]; then
        command+=(--force)
    fi
    "${command[@]}"
fi

if [[ "$RUN_POSTPROCESS" == "1" ]]; then
    "$UV_BIN" run python -u scripts/analysis/postprocess_evaluation_v5.py \
        --evaluation-studies-root "$OUTPUT_STUDIES_ROOT" \
        --notebook-plots-root "$NOTEBOOK_PLOTS_ROOT" \
        --studies "${STUDIES_ARRAY[@]}"
fi
