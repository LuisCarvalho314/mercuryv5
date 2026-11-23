#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${CONFIG_PATH:-configs/studies/age_ambiguity_ablation_rand_prob0_1.json}"
STUDY_ROOT="${STUDY_ROOT:-results/current/studies}"
STUDY_NAME="${STUDY_NAME:-age_ambiguity_ablation_revised_rand_prob_0_1}"
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


values = {
    "CONFIG_LEVELS": " ".join(str(value) for value in environment["level"]),
    "CONFIG_SENSOR": one(environment, "sensor"),
    "CONFIG_SENSOR_RANGE": one(environment, "sensor_range"),
    "CONFIG_RAND_PROB": one(environment, "rand_prob"),
    "CONFIG_FINAL_STEP": one(environment, "num_steps"),
    "CONFIG_EVAL_NUM_WALKS": one(evaluation, "paper_precision_num_walks"),
    "CONFIG_EVAL_WALK_LENGTH": one(evaluation, "paper_precision_walk_length"),
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
EVAL_SEED="${EVAL_SEED:-0}"
read -r -a EVAL_EDGE_THRESHOLDS <<< "${EVAL_EDGE_THRESHOLDS:-1e-6}"
EVAL_SNAPSHOT_STEPS="${EVAL_SNAPSHOT_STEPS:-$CONFIG_FINAL_STEP}"
PLOT_THRESHOLDS="${PLOT_THRESHOLDS:-0,1e-6,1e-5,1e-4,1e-3,0.002,0.005,0.01,0.02,0.05,0.1,0.15,0.2,0.25,0.3,0.4,0.5}"
MAX_WORKERS="${MAX_WORKERS:-4}"
EVAL_FORCE="${EVAL_FORCE:-0}"
RUN_EVAL="${RUN_EVAL:-1}"
RUN_PLOTS="${RUN_PLOTS:-1}"

export MPLCONFIGDIR="${MPLCONFIGDIR:-/private/tmp/matplotlib-${USER:-mercury}}"
mkdir -p "$MPLCONFIGDIR"

if [[ "$RUN_EVAL" == "1" ]]; then
    for level in "${LEVELS[@]}"; do
        printf 'Evaluating %s: level=%s, workers=%s\n' "$STUDY_NAME" "$level" "$MAX_WORKERS"
        command=(uv run python -u main.py
            --evaluate_study
            --study_root "$STUDY_ROOT"
            --study_name "$STUDY_NAME"
            --method mercury
            --level "$level"
            --sensor "$SENSOR"
            --sensor_range "$SENSOR_RANGE"
            --rand_prob "$RAND_PROB"
            --eval_num_walks "$EVAL_NUM_WALKS"
            --eval_walk_length "$EVAL_WALK_LENGTH"
            --eval_seed "$EVAL_SEED"
            --eval_edge_thresholds "${EVAL_EDGE_THRESHOLDS[@]}"
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
    EDGE_THRESHOLD_LIST="$(IFS=,; printf '%s' "${EVAL_EDGE_THRESHOLDS[*]}")"
    PROTOCOL_ID="$(uv run python -c "from mercury_runs.application.evaluation_protocol import EvaluationProtocol; print(EvaluationProtocol(sensor='$SENSOR', sensor_range=int('$SENSOR_RANGE'), rand_prob=float('$RAND_PROB'), evaluation_seed=int('$EVAL_SEED'), number_of_walks=int('$EVAL_NUM_WALKS'), walk_length=int('$EVAL_WALK_LENGTH'), edge_thresholds=[$EDGE_THRESHOLD_LIST]).protocol_id())")"
    PLOT_LEVELS="$(IFS=,; printf '%s' "${LEVELS[*]}")"

    printf 'Generating structural metric trajectories in %s\n' "$OUTPUT_DIR/metric_trajectories"
    uv run python scripts/analysis/plot_age_ambiguity_structural_trajectories.py \
        --study-root "$STUDY_ROOT/$STUDY_NAME" \
        --plot-dir "$OUTPUT_DIR/metric_trajectories" \
        --protocol-id "$PROTOCOL_ID" \
        --levels "$PLOT_LEVELS"

    printf 'Generating canonical plot suite in %s\n' "$OUTPUT_DIR"
    uv run python -m mercury_runs.analysis.plots.age_ambiguity_ablation \
        --study_root "$STUDY_ROOT/$STUDY_NAME" \
        --output_dir "$OUTPUT_DIR" \
        --protocol_id "$PROTOCOL_ID" \
        --thresholds "$PLOT_THRESHOLDS"
fi

printf 'Evaluation records: %s/%s/*/evaluations/\n' "$STUDY_ROOT" "$STUDY_NAME"
