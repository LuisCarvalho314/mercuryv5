#!/usr/bin/env bash
# Explicit JEPA mixing sweep, rho in {0, .25, .5, .75, 1}: PREPARED, launch manually.
#
#   bash scripts/run_cscg_jepa_mix_sweep.sh                 # all arms, then matched evaluation
#   LEVELS="21" SEEDS="0" STEPS=2000 bash scripts/run_cscg_jepa_mix_sweep.sh   # reduced
#
# Every arm uses identical streams, seeds, clone allocations, initial tensors
# (RandomState(seed)), window, teacher half-life, local iteration budget and
# evaluation protocol; only the objective differs. Arms:
#   em_baselines   short + long EM
#   legacy_eta1    legacy coupled predictive penalty (eta = 1), the reported configuration
#   jepa_mix<rho>  predictive with the explicit objective at every local iteration
# rho = 0 must reproduce short EM exactly. The unmodified rho = 1 objective is expected to
# stop with ZeroPredictiveSupportError (docs/online_jepa_cscg_formulation.md, Section 5.11);
# its log records where, and the remaining arms still run.
set -uo pipefail
cd "$(dirname "$0")/.."

LEVELS=${LEVELS:-"21 22"}
SEEDS=${SEEDS:-"0 1 2"}
STEPS=${STEPS:-20000}
WORKERS=${WORKERS:-4}
MIXES=${MIXES:-"0 0.25 0.5 0.75 1"}
OUT=${OUT:-results/diagnostics/cscg_jepa_mix_sweep_v1}
COMMON=(--levels $LEVELS --seeds $SEEDS --steps "$STEPS" --clone-allocation ground_truth
        --student-inference smooth --local-updates 5 --forgetting 1.0 --pseudocount 0.002
        --window 40 --half-life 40 --workers "$WORKERS")
PY=(env PYTHONPATH=.:src .venv/bin/python)

mkdir -p "$OUT"
run() {
  local name=$1; shift
  echo "== $name"
  "${PY[@]}" scripts/run_cscg_distillation_experiment.py "${COMMON[@]}" "$@" --output "$OUT/$name" \
    > "$OUT/$name.log" 2>&1
  echo "$name exit=$?" | tee -a "$OUT/status.txt"
}

run em_baselines --methods short long
run legacy_eta1 --methods predictive --weight 1.0 --coupled-auxiliary
for rho in $MIXES; do
  run "jepa_mix$rho" --methods predictive --jepa-mix "$rho"
done

# Training-matched canonical evaluation of every completed arm (lag-1 decoder for
# predictive/short, lag W-1 for long). Each arm writes its own canonical metrics,
# keyed by level/seed/method; the matched_eval `report` subcommand expects all five
# legacy methods per condition, so cross-arm comparison joins those CSVs on level/seed.
for dir in "$OUT"/*/; do
  name=$(basename "$dir")
  [ -f "$dir/results.md" ] || { echo "skip evaluation of $name (incomplete)"; continue; }
  "${PY[@]}" scripts/analysis/cscg_distillation_matched_eval.py evaluate --run "$dir" --steps all --workers "$WORKERS"
done
