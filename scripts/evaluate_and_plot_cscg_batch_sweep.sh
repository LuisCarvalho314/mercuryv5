#!/usr/bin/env bash
# Evaluate every CSCG batch-size arm and plot its snapshot metrics.
#
# Usage:
#   bash scripts/evaluate_and_plot_cscg_batch_sweep.sh
#
# Useful overrides:
#   RUN_EVAL=0 bash scripts/evaluate_and_plot_cscg_batch_sweep.sh
#   RUN_PLOTS=0 EVAL_MAX_WORKERS=8 bash scripts/evaluate_and_plot_cscg_batch_sweep.sh
#   BATCH_SIZES="10 40" LEVELS="13 20" bash scripts/evaluate_and_plot_cscg_batch_sweep.sh
set -euo pipefail

STUDY_ROOT="${STUDY_ROOT:-results/current/studies}"
STUDY_NAME="${STUDY_NAME:-cscg_batch_sweep}"
OUT="${OUT:-results/current/notebook_plots/${STUDY_NAME}}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/private/tmp/matplotlib-${USER:-mercury}}"
mkdir -p "$MPLCONFIGDIR"

read -r -a LEVELS <<< "${LEVELS:-13 20 21 22}"
read -r -a BATCH_SIZES <<< "${BATCH_SIZES:-10 40 100}"
RAND_PROB="${RAND_PROB:-0.1}"
EVAL_MAX_WORKERS="${EVAL_MAX_WORKERS:-4}"
EVAL_FORCE="${EVAL_FORCE:-0}"
RUN_EVAL="${RUN_EVAL:-1}"
RUN_PLOTS="${RUN_PLOTS:-1}"
PLOT_LEVELS="${PLOT_LEVELS:-$(IFS=,; printf '%s' "${LEVELS[*]}")}"
BATCH_SIZE_LIST="$(IFS=,; printf '%s' "${BATCH_SIZES[*]}")"

protocol_id() {
    python -c "from mercury_runs.application.evaluation_protocol import EvaluationProtocol; print(EvaluationProtocol(sensor='cardinal distance', sensor_range=1, rand_prob=float('$RAND_PROB'), evaluation_seed=0, number_of_walks=100, walk_length=10000, edge_thresholds=[1e-6]).protocol_id())"
}

PROTOCOL_ID="$(protocol_id)"
STUDY_PATH="$STUDY_ROOT/$STUDY_NAME"

if [[ ! -f "$STUDY_PATH/study_manifest.jsonl" ]]; then
    printf 'Study manifest not found: %s\n' "$STUDY_PATH/study_manifest.jsonl" >&2
    exit 1
fi

EVAL_STEPS=0
PLOT_STEPS=0
if [[ "$RUN_EVAL" == "1" ]]; then
    EVAL_STEPS=$(( ${#LEVELS[@]} * ${#BATCH_SIZES[@]} ))
fi
if [[ "$RUN_PLOTS" == "1" ]]; then
    PLOT_STEPS=3
fi
TOTAL=$(( EVAL_STEPS + PLOT_STEPS ))
STEP=0
START=$(date +%s)

fmt_time() {
    printf '%dm%02ds' $(( $1 / 60 )) $(( $1 % 60 ))
}

step_start() {
    STEP=$(( STEP + 1 ))
    _STEP_START=$(date +%s)
    printf '[%d/%d] %s ...\n' "$STEP" "$TOTAL" "$1"
}

step_done() {
    local now elapsed step_elapsed
    now=$(date +%s)
    elapsed=$(( now - START ))
    step_elapsed=$(( now - _STEP_START ))
    printf '  -> done in %s (elapsed %s)\n' \
        "$(fmt_time "$step_elapsed")" "$(fmt_time "$elapsed")"
}

run_eval() {
    local level=$1 batch_size=$2
    local force_args=()
    if [[ "$EVAL_FORCE" == "1" ]]; then
        force_args+=(--eval_force)
    fi

    step_start "eval level=$level method=cscg batch_size=$batch_size rand_prob=$RAND_PROB"
    MERCURY_FORCE_TTY_PROGRESS=1 python -u main.py --evaluate_study \
        --study_root "$STUDY_ROOT" \
        --study_name "$STUDY_NAME" \
        --method cscg \
        --level "$level" \
        --sensor "cardinal distance" \
        --sensor_range 1 \
        --rand_prob "$RAND_PROB" \
        --eval_cscg_batch_size "$batch_size" \
        --max_workers "$EVAL_MAX_WORKERS" \
        ${force_args+"${force_args[@]}"}
    step_done
}

run_plot() {
    local label=$1
    shift
    step_start "plot $label"
    set +e
    python "$@" 2>&1 | sed 's/^/  /'
    local status=${PIPESTATUS[0]}
    set -e
    if (( status != 0 )); then
        return "$status"
    fi
    step_done
}

printf 'Study    : %s\n' "$STUDY_PATH"
printf 'Protocol : %s\n' "$PROTOCOL_ID"
printf 'Batches  : %s\n' "${BATCH_SIZES[*]}"
printf 'Levels   : %s\n\n' "${LEVELS[*]}"

if [[ "$RUN_EVAL" == "1" ]]; then
    for BATCH_SIZE in "${BATCH_SIZES[@]}"; do
        for LEVEL in "${LEVELS[@]}"; do
            run_eval "$LEVEL" "$BATCH_SIZE"
        done
    done
fi

if [[ "$RUN_PLOTS" == "1" ]]; then
    run_plot "final-checkpoint batch-size sensitivity" \
        scripts/analysis/plot_cscg_batch_size_sensitivity.py \
        --study-root "$STUDY_PATH" \
        --protocol-id "$PROTOCOL_ID" \
        --output-dir "$OUT/batch_size_sensitivity" \
        --batch-sizes "$BATCH_SIZE_LIST" \
        --levels "$PLOT_LEVELS"

    run_plot "snapshot metrics batch_sizes=$BATCH_SIZE_LIST" \
        scripts/analysis/plot_config_metric_trajectories.py \
        --study-root "$STUDY_PATH" \
        --plot-dir "$OUT/supplementary/metric_trajectories" \
        --protocol-id "$PROTOCOL_ID" \
        --batch-sizes "$BATCH_SIZE_LIST" \
        --levels "$PLOT_LEVELS" \
        --output-stem "cscg_batch_sweep_metric_trajectories" \
        --title "CSCG batch-size sweep metric trajectories"

    run_plot "cross-batch EM convergence" \
        scripts/analysis/plot_em_likelihood_trajectories.py \
        --study-root "$STUDY_PATH" \
        --out-dir "$OUT/em_likelihood_trajectories"
fi

total_elapsed=$(( $(date +%s) - START ))
printf '\nDone in %s. Plots -> %s\n' "$(fmt_time "$total_elapsed")" "$OUT"
