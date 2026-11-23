I #!/usr/bin/env bash
# Generate the MERCURY-phase (mercury_cscg_hybrid_mercury) evaluation records
# for a hybrid study. evaluate_and_plot.sh does not cover this method, so the
# MERCURY panels stay empty until this runs.
#
# Usage: bash scripts/eval_mercury_phase.sh
set -euo pipefail

STUDY_ROOT="${STUDY_ROOT:-results/current/studies}"
STUDY_NAME="${STUDY_NAME:-default_matrix_smoke_30_seed_onehot_clean_more_mazes}"
RAND_PROB="${RAND_PROB:-0.3}"
BATCH_SIZE="${BATCH_SIZE:-40}"
MAX_WORKERS="${MAX_WORKERS:-4}"
read -r -a LEVELS <<< "${LEVELS:-13 20 21 22}"

export PYTHONPATH="${PYTHONPATH:-src}"

for LEVEL in "${LEVELS[@]}"; do
    echo "=== level=$LEVEL method=mercury_cscg_hybrid_mercury"
    python -u main.py --evaluate_study \
        --study_root "$STUDY_ROOT" \
        --study_name "$STUDY_NAME" \
        --method mercury_cscg_hybrid_mercury \
        --level "$LEVEL" \
        --sensor "cardinal distance" \
        --sensor_range 1 \
        --rand_prob "$RAND_PROB" \
        --eval_cscg_batch_size "$BATCH_SIZE" \
        --max_workers "$MAX_WORKERS"
done

echo ""
echo -n "MERCURY records written: "
find "$STUDY_ROOT/$STUDY_NAME" -path '*mercury_cscg_hybrid_mercury*' -name 'step=*.json' | wc -l
