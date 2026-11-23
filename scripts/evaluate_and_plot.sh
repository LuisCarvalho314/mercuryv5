#!/usr/bin/env bash
set -euo pipefail

STUDY_ROOT="${STUDY_ROOT:-results/current/studies}"
STUDY_NAME="${STUDY_NAME:-main_study}"
OUT="${OUT:-results/current/notebook_plots/${STUDY_NAME}}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/private/tmp/matplotlib-${USER:-mercury}}"
mkdir -p "$MPLCONFIGDIR"

read -r -a LEVELS <<< "${LEVELS:-13 20 21 22}"
read -r -a RAND_PROBS <<< "${RAND_PROBS:-0.1}"
read -r -a METHODS <<< "${METHODS:-cscg mercury_cscg_hybrid}"
BATCH_SIZE="${BATCH_SIZE:-40}"
export PLOT_CSCG_BATCH_SIZE="$BATCH_SIZE"
EVAL_MAX_WORKERS="${EVAL_MAX_WORKERS:-4}"
EVAL_FORCE="${EVAL_FORCE:-0}"
RUN_EVAL="${RUN_EVAL:-1}"
RUN_PLOTS="${RUN_PLOTS:-1}"
RUN_SAMPLE_ALIGNED="${RUN_SAMPLE_ALIGNED:-1}"
COMPARISONS="${COMPARISONS:-final,checkpoint_4000,checkpoint_2000}"
SCENARIOS="${SCENARIOS:-all_methods_final,all_methods_checkpoint_4000,all_methods_checkpoint_2000,final_hybrid_vs_cscg,final_tmgwr_vs_cscg,hybrid_vs_cscg_checkpoint_4000,tmgwr_hybrid_vs_cscg_checkpoint_4000,mercury_vs_cscg_checkpoint_2000}"
PLOT_LEVELS="${PLOT_LEVELS:-$(IFS=,; printf '%s' "${LEVELS[*]}")}"
PLOT_SEEDS="${PLOT_SEEDS:-0-29}"
EVAL_STEPS=0
if [[ "$RUN_EVAL" == "1" ]]; then
    EVAL_STEPS=$(( ${#LEVELS[@]} * ${#METHODS[@]} * ${#RAND_PROBS[@]} ))
fi
PLOTS=0
if [[ "$RUN_PLOTS" == "1" ]]; then
    PLOTS=$(( 2 + ${#RAND_PROBS[@]} * 10 ))
    if [[ "$RUN_SAMPLE_ALIGNED" == "1" ]]; then
        PLOTS=$(( PLOTS + 1 ))
    fi
fi
TOTAL=$(( EVAL_STEPS + PLOTS ))
STEP=0
START=$(date +%s)

protocol_id_for_rand() {
    local rand_prob=$1
    python -c "from mercury_runs.application.evaluation_protocol import EvaluationProtocol; print(EvaluationProtocol(sensor='cardinal distance', sensor_range=1, rand_prob=float('$rand_prob'), evaluation_seed=0, number_of_walks=100, walk_length=10000, edge_thresholds=[1e-6]).protocol_id())"
}

rand_slug() {
    local rand_prob=$1
    python -c "print(('rand_' + format(float('$rand_prob'), 'g').replace('-', 'm').replace('.', 'p')))"
}

PROTOCOL_IDS_BY_RAND=""
for RAND_PROB in "${RAND_PROBS[@]}"; do
    PROTOCOL_ID="$(protocol_id_for_rand "$RAND_PROB")"
    if [[ -n "$PROTOCOL_IDS_BY_RAND" ]]; then
        PROTOCOL_IDS_BY_RAND+=","
    fi
    PROTOCOL_IDS_BY_RAND+="${RAND_PROB}=${PROTOCOL_ID}"
done

_bar() {
    local step=$1 total=$2 width=28
    local filled=$(( step * width / total ))
    local bar=""
    for ((i=0; i<filled; i++));  do bar+="█"; done
    for ((i=filled; i<width; i++)); do bar+="░"; done
    printf "%s" "$bar"
}

_fmt_time() {
    printf "%dm%02ds" $(( $1 / 60 )) $(( $1 % 60 ))
}

step_start() {
    STEP=$(( STEP + 1 ))
    local now elapsed pct
    now=$(date +%s); elapsed=$(( now - START ))
    pct=$(( (STEP - 1) * 100 / TOTAL ))
    printf "[%s] %3d%%  (%d/%d)  %s ...\n" \
        "$(_bar $(( STEP - 1 )) "$TOTAL")" "$pct" "$STEP" "$TOTAL" "$1"
}

step_done() {
    local now elapsed step_elapsed rem
    now=$(date +%s); elapsed=$(( now - START ))
    step_elapsed=$(( now - _STEP_START ))
    rem=$(( elapsed * (TOTAL - STEP) / STEP ))
    printf "  → done in %s  |  elapsed %s  |  eta %s\n" \
        "$(_fmt_time "$step_elapsed")" "$(_fmt_time "$elapsed")" "$(_fmt_time "$rem")"
}

run_eval() {
    local level=$1 method=$2 rand_prob=$3
    local protocol_id
    protocol_id="$(protocol_id_for_rand "$rand_prob")"
    step_start "eval  level=$level  method=$method  rand_prob=$rand_prob"
    _STEP_START=$(date +%s)
    printf "  using %s eval worker(s); per-run logs are under the study evaluation_logs directory\n" "$EVAL_MAX_WORKERS"
    set +e
    local force_args=()
    if [[ "$EVAL_FORCE" == "1" ]]; then
        force_args+=(--eval_force)
    fi
    MERCURY_FORCE_TTY_PROGRESS=1 python -u main.py --evaluate_study \
        --study_root "$STUDY_ROOT" --study_name "$STUDY_NAME" \
        --method "$method" --level "$level" \
        --sensor "cardinal distance" --sensor_range 1 --rand_prob "$rand_prob" \
        --eval_cscg_batch_size "$BATCH_SIZE" \
        --max_workers "$EVAL_MAX_WORKERS" \
        ${force_args+"${force_args[@]}"}
    local status=$?
    set -e
    if (( status != 0 )); then
        printf "  eval failed; check per-run logs under %s/%s/evaluation_logs/%s/level=%s_method=%s\n" \
            "$STUDY_ROOT" "$STUDY_NAME" "$protocol_id" "$level" "$method"
        return "$status"
    fi
    step_done
}

run_plot() {
    local label="$1"; shift
    step_start "plot  $label"
    _STEP_START=$(date +%s)
    set +e
    python "$@" \
        | grep -E "\.(png|pdf|csv)$" | sed 's/^/  /'
    local status=${PIPESTATUS[0]}
    set -e
    if (( status != 0 )); then
        return "$status"
    fi
    step_done
}

echo "Study : $STUDY_NAME"
echo "Steps : $TOTAL  (${#LEVELS[@]} levels × ${#METHODS[@]} methods × ${#RAND_PROBS[@]} rand probs + $PLOTS plots)"
echo ""

if [[ "$RUN_EVAL" == "1" ]]; then
    for RAND_PROB in "${RAND_PROBS[@]}"; do
        for LEVEL in "${LEVELS[@]}"; do
            for METHOD in "${METHODS[@]}"; do
                run_eval "$LEVEL" "$METHOD" "$RAND_PROB"
            done
        done
    done
fi

echo ""

if [[ "$RUN_PLOTS" != "1" ]]; then
    total_elapsed=$(( $(date +%s) - START ))
    printf "Done in $(_fmt_time "$total_elapsed").  Plots skipped.\n"
    exit 0
fi

run_plot "batch_summary_forest" \
    -m mercury_runs.analysis.plots.batch_summary_forest \
    --study_root "$STUDY_ROOT/$STUDY_NAME" \
    --protocol_id "$(protocol_id_for_rand "${RAND_PROBS[0]}")" \
    --protocol_ids_by_rand "$PROTOCOL_IDS_BY_RAND" \
    --batch_size "$BATCH_SIZE" \
    --comparisons "$COMPARISONS" \
    --output_dir "${OUT}/forest_matrix_profile"

run_plot "batch_tv_similarity" \
    -m mercury_runs.analysis.plots.batch_tv_similarity \
    --study_root "$STUDY_ROOT/$STUDY_NAME" \
    --protocol_id "$(protocol_id_for_rand "${RAND_PROBS[0]}")" \
    --protocol_ids_by_rand "$PROTOCOL_IDS_BY_RAND" \
    --batch_size "$BATCH_SIZE" \
    --rand_probs "$(IFS=,; printf '%s' "${RAND_PROBS[*]}")" \
    --comparisons "$COMPARISONS" \
    --output_dir "${OUT}/tv_similarity"

if [[ "$RUN_SAMPLE_ALIGNED" == "1" ]]; then
    run_plot "rand03_sample_aligned_trajectories" \
        scripts/analysis/plot_rand03_sample_aligned_trajectories.py
fi

for RAND_PROB in "${RAND_PROBS[@]}"; do
    PROTOCOL="$(protocol_id_for_rand "$RAND_PROB")"
    RAND_DIR="$(rand_slug "$RAND_PROB")"

    run_plot "selected_residual_maps rand_prob=$RAND_PROB" \
        -m mercury_runs.analysis.plots.selected_residual_maps \
        --study_root "$STUDY_ROOT/$STUDY_NAME" \
        --protocol_id "$PROTOCOL" \
        --rand_prob "$RAND_PROB" \
        --scenarios "$SCENARIOS" \
        --levels "$PLOT_LEVELS" \
        --seeds "$PLOT_SEEDS" \
        --output_dir "${OUT}/residual_maps/${RAND_DIR}/collapsed/without_self_loops"

    run_plot "selected_residual_maps (action) rand_prob=$RAND_PROB" \
        -m mercury_runs.analysis.plots.selected_residual_maps \
        --study_root "$STUDY_ROOT/$STUDY_NAME" \
        --protocol_id "$PROTOCOL" \
        --rand_prob "$RAND_PROB" \
        --scenarios "$SCENARIOS" \
        --levels "$PLOT_LEVELS" \
        --seeds "$PLOT_SEEDS" \
        --output_dir "${OUT}/residual_maps/${RAND_DIR}/action_conditioned/without_self_loops" \
        --action_conditioned

    run_plot "selected_residual_maps (self loops) rand_prob=$RAND_PROB" \
        -m mercury_runs.analysis.plots.selected_residual_maps \
        --study_root "$STUDY_ROOT/$STUDY_NAME" \
        --protocol_id "$PROTOCOL" \
        --rand_prob "$RAND_PROB" \
        --scenarios "$SCENARIOS" \
        --levels "$PLOT_LEVELS" \
        --seeds "$PLOT_SEEDS" \
        --output_dir "${OUT}/residual_maps/${RAND_DIR}/collapsed/with_self_loops" \
        --include_self_loops

    run_plot "selected_residual_maps (action, self loops) rand_prob=$RAND_PROB" \
        -m mercury_runs.analysis.plots.selected_residual_maps \
        --study_root "$STUDY_ROOT/$STUDY_NAME" \
        --protocol_id "$PROTOCOL" \
        --rand_prob "$RAND_PROB" \
        --scenarios "$SCENARIOS" \
        --levels "$PLOT_LEVELS" \
        --seeds "$PLOT_SEEDS" \
        --output_dir "${OUT}/residual_maps/${RAND_DIR}/action_conditioned/with_self_loops" \
        --action_conditioned \
        --include_self_loops

    run_plot "threshold_robustness data rand_prob=$RAND_PROB" \
        -m mercury_runs.analysis.plots.threshold_edge_metrics \
        --study_root "$STUDY_ROOT/$STUDY_NAME" \
        --protocol_id "$PROTOCOL" \
        --rand_prob "$RAND_PROB" \
        --scenarios "$SCENARIOS" \
        --levels "$PLOT_LEVELS" \
        --seeds "$PLOT_SEEDS" \
        --output_dir "${OUT}/threshold_edge_robustness/${RAND_DIR}/collapsed/without_self_loops" \
        --no_plots

    run_plot "threshold_robustness data (action) rand_prob=$RAND_PROB" \
        -m mercury_runs.analysis.plots.threshold_edge_metrics \
        --study_root "$STUDY_ROOT/$STUDY_NAME" \
        --protocol_id "$PROTOCOL" \
        --rand_prob "$RAND_PROB" \
        --scenarios "$SCENARIOS" \
        --levels "$PLOT_LEVELS" \
        --seeds "$PLOT_SEEDS" \
        --output_dir "${OUT}/threshold_edge_robustness/${RAND_DIR}/action_conditioned/without_self_loops" \
        --action_conditioned \
        --no_plots

    run_plot "threshold_robustness data (self loops) rand_prob=$RAND_PROB" \
        -m mercury_runs.analysis.plots.threshold_edge_metrics \
        --study_root "$STUDY_ROOT/$STUDY_NAME" \
        --protocol_id "$PROTOCOL" \
        --rand_prob "$RAND_PROB" \
        --scenarios "$SCENARIOS" \
        --levels "$PLOT_LEVELS" \
        --seeds "$PLOT_SEEDS" \
        --output_dir "${OUT}/threshold_edge_robustness/${RAND_DIR}/collapsed/with_self_loops" \
        --include_self_loops \
        --no_plots

    run_plot "threshold_robustness data (action, self loops) rand_prob=$RAND_PROB" \
        -m mercury_runs.analysis.plots.threshold_edge_metrics \
        --study_root "$STUDY_ROOT/$STUDY_NAME" \
        --protocol_id "$PROTOCOL" \
        --rand_prob "$RAND_PROB" \
        --scenarios "$SCENARIOS" \
        --levels "$PLOT_LEVELS" \
        --seeds "$PLOT_SEEDS" \
        --output_dir "${OUT}/threshold_edge_robustness/${RAND_DIR}/action_conditioned/with_self_loops" \
        --action_conditioned \
        --include_self_loops \
        --no_plots

    run_plot "threshold_robustness rand_prob=$RAND_PROB" \
        -m mercury_runs.analysis.plots.threshold_robustness \
        --collapsed_summary "${OUT}/threshold_edge_robustness/${RAND_DIR}/collapsed/without_self_loops/threshold_edge_metrics_summary.csv" \
        --action_summary "${OUT}/threshold_edge_robustness/${RAND_DIR}/action_conditioned/without_self_loops/threshold_edge_metrics_summary.csv" \
        --scenarios "$SCENARIOS" \
        --output_dir "${OUT}/threshold_edge_robustness/${RAND_DIR}/robustness/without_self_loops"

    run_plot "threshold_robustness (self loops) rand_prob=$RAND_PROB" \
        -m mercury_runs.analysis.plots.threshold_robustness \
        --collapsed_summary "${OUT}/threshold_edge_robustness/${RAND_DIR}/collapsed/with_self_loops/threshold_edge_metrics_summary.csv" \
        --action_summary "${OUT}/threshold_edge_robustness/${RAND_DIR}/action_conditioned/with_self_loops/threshold_edge_metrics_summary.csv" \
        --scenarios "$SCENARIOS" \
        --output_dir "${OUT}/threshold_edge_robustness/${RAND_DIR}/robustness/with_self_loops"
done

total_elapsed=$(( $(date +%s) - START ))
echo ""
printf "Done in $(_fmt_time "$total_elapsed").  Plots → %s\n" "$OUT"
