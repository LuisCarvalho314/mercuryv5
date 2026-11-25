#!/usr/bin/env bash
# Controlled EM-CSCG vs gradient-based CSCG (docs/gradient_cscg_formulation.md).
#
#   OUT=results/diagnostics/cscg_em_vs_gradient_l23s0_v1 bash scripts/run_cscg_em_vs_gradient.sh
#
# Every arm is method "long" with identical stream, ground-truth clone allocation, initial
# tensor (RandomState(seed)), window context, carried boundary beliefs, forgetting, pseudocount
# and local iteration budget; only the optimiser of the per-arrival objective differs:
#   em                T <- rowNorm(B) at every local iteration
#   gradcscg_lr<eta>  reference gradCSCG -- logits, row-softmax, log-space forward, autodiff,
#                     standard Adam (Keras defaults)
# Set GRADCSCG_RATES="" (or RUN_EM=0) to skip arms already trained in OUT.
# Context = transitions in the window: W = CONTEXT + 1 observations.
# Then training-matched evaluation (carried filter + lag W - 1, as in training) and the report.
set -uo pipefail
cd "$(dirname "$0")/.."

LEVEL=${LEVEL:-23}
SEED=${SEED:-0}
STEPS=${STEPS:-20000}
CONTEXTS=${CONTEXTS:-"5 100"}
RUN_EM=${RUN_EM:-1}
GRADCSCG_RATES=${GRADCSCG_RATES-"0.0003 0.001 0.01 0.1"}
LOCAL_UPDATES=${LOCAL_UPDATES:-5}
FORGETTING=${FORGETTING:-1.0}
PSEUDOCOUNT=${PSEUDOCOUNT:-0.002}
RAND_PROB=${RAND_PROB:-0.3}
EVAL_RAND_PROB=${EVAL_RAND_PROB:-0.1}
EVAL_WALKS=${EVAL_WALKS:-10}
EVAL_WALK_LENGTH=${EVAL_WALK_LENGTH:-500}
OUT=${OUT:-results/diagnostics/cscg_em_vs_gradient_l23s0_v1}
PY=(env PYTHONPATH=.:src .venv/bin/python)

mkdir -p "$OUT"
run() {
  local name=$1 context=$2; shift 2
  "${PY[@]}" scripts/run_cscg_distillation_experiment.py --levels "$LEVEL" --seeds "$SEED" --steps "$STEPS" \
    --clone-allocation ground_truth --methods long --window $((context + 1)) --local-updates "$LOCAL_UPDATES" \
    --forgetting "$FORGETTING" --pseudocount "$PSEUDOCOUNT" --rand-prob "$RAND_PROB" \
    --walks "$EVAL_WALKS" --walk-length "$EVAL_WALK_LENGTH" "$@" --output "$OUT/context$context/$name" \
    > "$OUT/context$context/$name.log" 2>&1
  echo "context$context/$name train exit=$?" | tee -a "$OUT/status.txt"
}

for context in $CONTEXTS; do
  mkdir -p "$OUT/context$context"
  [[ "$RUN_EM" == 1 ]] && run em "$context" &
  for eta in $GRADCSCG_RATES; do
    run "gradcscg_lr$eta" "$context" --optimizer gradcscg --gradient-learning-rate "$eta" &
  done
done
wait

for dir in "$OUT"/context*/*/; do
  ls "$dir"level=*/seed=*/long/rows.json > /dev/null 2>&1 || { echo "skip evaluation of $dir (no completed case)"; continue; }
  "${PY[@]}" scripts/analysis/cscg_distillation_matched_eval.py evaluate --run "$dir" --steps all --workers 1 \
    --walks "$EVAL_WALKS" --walk-length "$EVAL_WALK_LENGTH" --rand-prob "$EVAL_RAND_PROB" \
    > "${dir%/}.eval.log" 2>&1
  echo "$dir eval exit=$?" | tee -a "$OUT/status.txt"
done
"${PY[@]}" scripts/analysis/cscg_em_vs_gradient_report.py --sweep "$OUT" \
  --walks "$EVAL_WALKS" --walk-length "$EVAL_WALK_LENGTH" --rand-prob "$EVAL_RAND_PROB"
