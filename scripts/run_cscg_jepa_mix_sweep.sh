#!/usr/bin/env bash
# Accumulated predictive JEPA-CSCG against matched EM baselines.
#
#   LEVELS=21 SEEDS=0 EVAL_WALKS=10 EVAL_WALK_LENGTH=500 \
#       OUT=results/diagnostics/cscg_jepa_accumulated_l21s0_v1 bash scripts/run_cscg_jepa_mix_sweep.sh
#
# Arms (run concurrently), identical streams, seeds, ground-truth clone allocations,
# initial tensors (RandomState(seed)), window/teacher context, teacher half-life,
# forgetting, pseudocount and local iteration budget; only the learning objective differs:
#   em_baselines   short + long EM
#   jepa_mix<rho>  accumulated JEPA: new counts (1 - rho) xi + rho D enter the running
#                  statistics (docs/online_jepa_cscg_formulation.md, Section 6)
# The transient-JEPA and legacy coupled arms of earlier sweeps are reproducible from
# commit 233eb86 and are deliberately not run here.
# Then: training-matched canonical evaluation of every case at every checkpoint
# (EVAL_WALKS x EVAL_WALK_LENGTH; canonical 100 x 10,000), and the sweep report.
set -uo pipefail
cd "$(dirname "$0")/.."

LEVELS=${LEVELS:-"21 22"}
SEEDS=${SEEDS:-"0 1 2"}
STEPS=${STEPS:-20000}
WORKERS=${WORKERS:-2}
MIXES=${MIXES:-"0.5 0.99 1"}
LOCAL_UPDATES=${LOCAL_UPDATES:-5}
STUDENT_CONTEXT=${STUDENT_CONTEXT:-1}  # student forward-backward history (lag one)
TEACHER_WINDOW=${TEACHER_WINDOW:-40}   # teacher context; the replay window (and long EM context) is locked to it
RAND_PROB=${RAND_PROB:-0.3}            # training stream
EVAL_RAND_PROB=${EVAL_RAND_PROB:-0.1}  # held-out walks (canonical 0.1)
EVAL_WALKS=${EVAL_WALKS:-100}
EVAL_WALK_LENGTH=${EVAL_WALK_LENGTH:-10000}
OUT=${OUT:-results/diagnostics/cscg_jepa_accumulated_v1}
COMMON=(--levels $LEVELS --seeds $SEEDS --steps "$STEPS" --clone-allocation ground_truth
        --student-inference smooth --student-context "$STUDENT_CONTEXT" --local-updates "$LOCAL_UPDATES" --forgetting 1.0 --pseudocount 0.002
        --window "$TEACHER_WINDOW" --teacher-window "$TEACHER_WINDOW" --half-life 40 --rand-prob "$RAND_PROB" --workers "$WORKERS")
PY=(env PYTHONPATH=.:src .venv/bin/python)

mkdir -p "$OUT"
run() {
  local name=$1; shift
  "${PY[@]}" scripts/run_cscg_distillation_experiment.py "${COMMON[@]}" "$@" --output "$OUT/$name" \
    > "$OUT/$name.log" 2>&1
  echo "$name train exit=$?" | tee -a "$OUT/status.txt"
}

run em_baselines --methods short long &
for rho in $MIXES; do
  run "jepa_mix$rho" --methods predictive --jepa-mix "$rho" &
done
wait

# Training-matched canonical evaluation (lag 1 for predictive/short, W-1 for long).
for dir in "$OUT"/*/; do
  [[ "$(basename "$dir")" == report* ]] && continue
  ls "$dir"level=*/seed=*/*/rows.json > /dev/null 2>&1 || { echo "skip evaluation of $dir (no completed case)"; continue; }
  "${PY[@]}" scripts/analysis/cscg_distillation_matched_eval.py evaluate --run "$dir" --steps all --workers "$WORKERS" \
    --walks "$EVAL_WALKS" --walk-length "$EVAL_WALK_LENGTH" --rand-prob "$EVAL_RAND_PROB" > "${dir%/}.eval.log" 2>&1
  echo "$(basename "$dir") eval exit=$?" | tee -a "$OUT/status.txt"
done
"${PY[@]}" scripts/analysis/cscg_jepa_mix_sweep_report.py --sweep "$OUT" \
  --walks "$EVAL_WALKS" --walk-length "$EVAL_WALK_LENGTH" --rand-prob "$EVAL_RAND_PROB"
