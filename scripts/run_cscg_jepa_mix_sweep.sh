#!/usr/bin/env bash
# Explicit JEPA mixing sweep with matched baselines.
#
#   bash scripts/run_cscg_jepa_mix_sweep.sh                                   # rho in {0, .25, .5, .75, 1}
#   MIXES="0.5 0.9 0.99 0.999 0.9999" OUT=results/diagnostics/cscg_jepa_dominated_20k_v1 \
#       bash scripts/run_cscg_jepa_mix_sweep.sh                               # JEPA-dominated sweep
#   LEVELS=21 SEEDS=0 STEPS=1000 ... bash scripts/run_cscg_jepa_mix_sweep.sh  # smoke
#
# Every arm uses identical streams, seeds, clone allocations, initial tensors
# (RandomState(seed)), window, teacher half-life, local iteration budget and
# evaluation protocol; only the objective differs. Arms (run concurrently):
#   em_baselines   short + long EM
#   legacy_eta1    legacy coupled predictive penalty (eta = 1), the reported configuration
#   jepa_mix<rho>  predictive with the explicit objective at every local iteration
# A ZeroPredictiveSupportError is recorded per case in failure.json (rho = 1 is
# expected to fail; docs/online_jepa_cscg_formulation.md, Section 5.11).
# Then: training-matched canonical evaluation of every completed case at every
# checkpoint, and the sweep report (scripts/analysis/cscg_jepa_mix_sweep_report.py).
set -uo pipefail
cd "$(dirname "$0")/.."

LEVELS=${LEVELS:-"21 22"}
SEEDS=${SEEDS:-"0 1 2"}
STEPS=${STEPS:-20000}
WORKERS=${WORKERS:-2}
MIXES=${MIXES:-"0 0.25 0.5 0.75 1"}
OUT=${OUT:-results/diagnostics/cscg_jepa_mix_sweep_v1}
COMMON=(--levels $LEVELS --seeds $SEEDS --steps "$STEPS" --clone-allocation ground_truth
        --student-inference smooth --local-updates 5 --forgetting 1.0 --pseudocount 0.002
        --window 40 --half-life 40 --workers "$WORKERS")
PY=(env PYTHONPATH=.:src .venv/bin/python)

mkdir -p "$OUT"
run() {
  local name=$1; shift
  "${PY[@]}" scripts/run_cscg_distillation_experiment.py "${COMMON[@]}" "$@" --output "$OUT/$name" \
    > "$OUT/$name.log" 2>&1
  echo "$name train exit=$?" | tee -a "$OUT/status.txt"
}

run em_baselines --methods short long &
run legacy_eta1 --methods predictive --weight 1.0 --coupled-auxiliary &
for rho in $MIXES; do
  run "jepa_mix$rho" --methods predictive --jepa-mix "$rho" &
done
wait

# Training-matched canonical evaluation (lag 1 for predictive/short, W-1 for long).
for dir in "$OUT"/*/; do
  ls "$dir"level=*/seed=*/*/rows.json > /dev/null 2>&1 || { echo "skip evaluation of $dir (no completed case)"; continue; }
  "${PY[@]}" scripts/analysis/cscg_distillation_matched_eval.py evaluate --run "$dir" --steps all --workers "$WORKERS" \
    > "${dir%/}.eval.log" 2>&1
  echo "$(basename "$dir") eval exit=$?" | tee -a "$OUT/status.txt"
done
"${PY[@]}" scripts/analysis/cscg_jepa_mix_sweep_report.py --sweep "$OUT"
