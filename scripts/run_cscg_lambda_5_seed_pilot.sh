#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

study_root="${STUDY_ROOT:-results/current/studies}"
study_name="${STUDY_NAME:-cscg_lambda_5_seed_pilot}"
max_workers="${MAX_WORKERS:-4}"

echo "Running study: $study_name"
uv run python -u main.py \
    --study \
    --study_config configs/studies/default_matrix_smoke_5_seed_lambda_pilot.json \
    --study_root "$study_root" \
    --study_name "$study_name" \
    --max_workers "$max_workers" \
    --resume

for level in 13 20 21 22; do
    for method in cscg mercury_cscg_hybrid tmgwr_cscg_hybrid; do
        echo "Evaluating level=$level method=$method"
        uv run python -u main.py \
            --evaluate_study \
            --study_root "$study_root" \
            --study_name "$study_name" \
            --level "$level" \
            --method "$method" \
            --sensor "cardinal distance" \
            --sensor_range 1 \
            --rand_prob 0.1 \
            --max_workers "$max_workers"
    done
done
