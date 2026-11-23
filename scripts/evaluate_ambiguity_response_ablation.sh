#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${CONFIG_PATH:-configs/studies/ambiguity_response_ablation_rand_prob_0_1.json}"
STUDY_ROOT="${STUDY_ROOT:-results/current/studies}"
STUDY_NAME="${STUDY_NAME:-$(basename "$CONFIG_PATH" .json)}"
OUTPUT_DIR="${OUTPUT_DIR:-results/current/notebook_plots/${STUDY_NAME}}"

if [[ ! -f "$CONFIG_PATH" ]]; then
    printf 'Study config not found: %s\n' "$CONFIG_PATH" >&2
    exit 1
fi

CONFIG_VALUES="$(uv run python - "$CONFIG_PATH" <<'PY'
import json
import shlex
import sys
from pathlib import Path

config = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
environment = config["environment"]
evaluation = config["evaluation"]


def one(section, key):
    values = section[key]
    if len(values) != 1:
        raise ValueError(f"{key} must contain exactly one value, received {values!r}")
    return values[0]


thresholds = one(evaluation, "structure_metrics_epsilons")
values = {
    "CONFIG_LEVELS": " ".join(str(value) for value in environment["level"]),
    "CONFIG_SENSOR": one(environment, "sensor"),
    "CONFIG_SENSOR_RANGE": one(environment, "sensor_range"),
    "CONFIG_RAND_PROB": one(environment, "rand_prob"),
    "CONFIG_FINAL_STEP": one(environment, "num_steps"),
    "CONFIG_EVAL_NUM_WALKS": one(evaluation, "paper_precision_num_walks"),
    "CONFIG_EVAL_WALK_LENGTH": one(evaluation, "paper_precision_walk_length"),
    "CONFIG_EVAL_EDGE_THRESHOLDS": " ".join(str(value) for value in thresholds),
}
for name, value in values.items():
    print(f"{name}={shlex.quote(str(value))}")
PY
)"
eval "$CONFIG_VALUES"

read -r -a LEVELS <<< "${LEVELS:-$CONFIG_LEVELS}"
SENSOR="${SENSOR:-$CONFIG_SENSOR}"
SENSOR_RANGE="${SENSOR_RANGE:-$CONFIG_SENSOR_RANGE}"
RAND_PROB="${RAND_PROB:-$CONFIG_RAND_PROB}"
EVAL_NUM_WALKS="${EVAL_NUM_WALKS:-$CONFIG_EVAL_NUM_WALKS}"
EVAL_WALK_LENGTH="${EVAL_WALK_LENGTH:-$CONFIG_EVAL_WALK_LENGTH}"
read -r -a EVAL_EDGE_THRESHOLDS <<< "${EVAL_EDGE_THRESHOLDS:-$CONFIG_EVAL_EDGE_THRESHOLDS}"
EVAL_SNAPSHOT_STEPS="${EVAL_SNAPSHOT_STEPS:-$CONFIG_FINAL_STEP}"
MAX_WORKERS="${MAX_WORKERS:-4}"
EVAL_FORCE="${EVAL_FORCE:-0}"
RUN_EVAL="${RUN_EVAL:-1}"
RUN_PLOTS="${RUN_PLOTS:-1}"
RUN_STATS="${RUN_STATS:-1}"

export MPLCONFIGDIR="${MPLCONFIGDIR:-/private/tmp/matplotlib-${USER:-mercury}}"
mkdir -p "$MPLCONFIGDIR"

PROTOCOL_ID="$(uv run python -c "from mercury_runs.application.evaluation_protocol import EvaluationProtocol; print(EvaluationProtocol(sensor='$SENSOR', sensor_range=int('$SENSOR_RANGE'), rand_prob=float('$RAND_PROB'), evaluation_seed=0, number_of_walks=int('$EVAL_NUM_WALKS'), walk_length=int('$EVAL_WALK_LENGTH'), edge_thresholds=[$(IFS=,; printf '%s' "${EVAL_EDGE_THRESHOLDS[*]}")]).protocol_id())")"

if [[ "$RUN_EVAL" == "1" ]]; then
    for level in "${LEVELS[@]}"; do
        printf 'Evaluating %s: level=%s, workers=%s\n' "$STUDY_NAME" "$level" "$MAX_WORKERS"
        command=(uv run python -u main.py \
            --evaluate_study \
            --study_root "$STUDY_ROOT" \
            --study_name "$STUDY_NAME" \
            --method mercury \
            --level "$level" \
            --sensor "$SENSOR" \
            --sensor_range "$SENSOR_RANGE" \
            --rand_prob "$RAND_PROB" \
            --eval_num_walks "$EVAL_NUM_WALKS" \
            --eval_walk_length "$EVAL_WALK_LENGTH" \
            --eval_edge_thresholds "${EVAL_EDGE_THRESHOLDS[@]}" \
            --max_workers "$MAX_WORKERS")
        if [[ "$EVAL_FORCE" == "1" ]]; then
            command+=(--eval_force)
        fi
        if [[ "$EVAL_SNAPSHOT_STEPS" != "all" ]]; then
            read -r -a snapshot_steps <<< "$EVAL_SNAPSHOT_STEPS"
            command+=(--eval_snapshot_steps "${snapshot_steps[@]}")
        fi
        MERCURY_FORCE_TTY_PROGRESS=0 "${command[@]}" | cat
    done
fi

if [[ "$RUN_PLOTS" == "1" ]]; then
    HISTORY_PATH="$(find "$STUDY_ROOT/$STUDY_NAME" -path '*metrics/mercury/*paper_precision*.json' -print -quit)"
    if [[ -n "$HISTORY_PATH" ]]; then
        printf 'Generating study-history plots in %s\n' "$OUTPUT_DIR/study_history"
        uv run python scripts/analysis/plot_study_histories.py \
            --study-root "$STUDY_ROOT/$STUDY_NAME" \
            --output-dir "$OUTPUT_DIR/study_history"
    else
        printf 'Skipping study-history plots: no inline paper-precision histories were produced.\n'
    fi

    printf 'Generating canonical plot suite in %s\n' "$OUTPUT_DIR/canonical_suite"
    uv run python -m mercury_runs.analysis.plots.ambiguity_response_ablation \
        --study_root "$STUDY_ROOT/$STUDY_NAME" \
        --output_dir "$OUTPUT_DIR/canonical_suite" \
        --protocol_id "$PROTOCOL_ID"
fi

if [[ "$RUN_STATS" == "1" ]]; then
    printf 'Generating paired statistical tables in %s\n' "$OUTPUT_DIR/statistical_analysis"
    uv run python scripts/analysis/ambiguity_response_paired_contrasts.py \
        --study-root "$STUDY_ROOT/$STUDY_NAME" \
        --output-dir "$OUTPUT_DIR/statistical_analysis" \
        --protocol-id "$PROTOCOL_ID" \
        --environments "$(IFS=,; printf '%s' "${LEVELS[*]}")"
fi

printf 'Evaluation records: %s/%s/*/evaluations/\n' "$STUDY_ROOT" "$STUDY_NAME"
