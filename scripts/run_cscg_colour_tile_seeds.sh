#!/usr/bin/env bash
# CSCG vs EM-JEPA vs gradCSCG on the colour-tile mazes (floor-colour sensor).
#
#   bash scripts/run_cscg_colour_tile_seeds.sh
#
# Levels 24 / 25 are colour_tile_levels level0 / level1: a 4x4 open interior whose cells
# emit only their floor colour (heavy aliasing). Same four arms and settings as
# results/diagnostics/cscg_vs_jepa_l23_3seed_20k:
#   em_baselines                       short + long EM (5 local updates)
#   jepa_mix1                          accumulated EM-JEPA, rho = 1 (5 local updates)
#   gradcscg_lr0.01_adam10             gradCSCG on the long-EM objective (Adam, lr 0.01, 10 updates)
#   gradcscg_jepa_mix1_lr0.01_adam10   gradCSCG on the accumulated JEPA objective
# Training streams: datasets/level={24,25}/sensor=floor (generate_data with {"sensor": "floor"}).
set -uo pipefail
cd "$(dirname "$0")/.."

LEVELS=${LEVELS:-"24 25"}
SEEDS=${SEEDS:-"0 1 2"}
STEPS=${STEPS:-20000}
WORKERS=${WORKERS:-2}
RAND_PROB=${RAND_PROB:-0.3}
EVAL_RAND_PROB=${EVAL_RAND_PROB:-0.1}
EVAL_WALKS=${EVAL_WALKS:-10}
EVAL_WALK_LENGTH=${EVAL_WALK_LENGTH:-500}
OUT=${OUT:-results/diagnostics/cscg_vs_jepa_colour_tiles_3seed_20k}
PY=(env PYTHONPATH=.:src .venv/bin/python)
COMMON=(--levels $LEVELS --seeds $SEEDS --steps "$STEPS" --clone-allocation ground_truth
        --student-inference smooth --student-context 1 --forgetting 1.0 --pseudocount 0.002
        --window 40 --teacher-window 40 --half-life 40 --rand-prob "$RAND_PROB"
        --walks "$EVAL_WALKS" --walk-length "$EVAL_WALK_LENGTH" --workers "$WORKERS")

for level in $LEVELS; do for seed in $SEEDS; do
  "${PY[@]}" -c "from src.maze_environment.generate_data import generate_data
generate_data($level, seed=$seed, rand_prob=$RAND_PROB, num_steps=max(2000, $STEPS), random_start=True,
              output_root='datasets', agent_sensors_list=[{'sensor': 'floor'}])"
done; done

mkdir -p "$OUT"
run() {
  local name=$1; shift
  "${PY[@]}" scripts/run_cscg_distillation_experiment.py "${COMMON[@]}" "$@" --output "$OUT/$name" \
    > "$OUT/$name.log" 2>&1
  echo "$name train exit=$?" | tee -a "$OUT/status.txt"
}

run em_baselines --methods short long --local-updates 5 &
run jepa_mix1 --methods predictive --jepa-mix 1 --local-updates 5 &
run gradcscg_lr0.01_adam10 --methods long --optimizer gradcscg --gradient-learning-rate 0.01 --local-updates 10 &
run gradcscg_jepa_mix1_lr0.01_adam10 --methods predictive --jepa-mix 1 --optimizer gradcscg \
  --gradient-learning-rate 0.01 --local-updates 10 &
wait

for dir in "$OUT"/*/; do
  ls "$dir"level=*/seed=*/*/rows.json > /dev/null 2>&1 || { echo "skip evaluation of $dir (no completed case)"; continue; }
  "${PY[@]}" scripts/analysis/cscg_distillation_matched_eval.py evaluate --run "$dir" --steps all --workers "$WORKERS" \
    --walks "$EVAL_WALKS" --walk-length "$EVAL_WALK_LENGTH" --rand-prob "$EVAL_RAND_PROB" \
    > "${dir%/}.eval.log" 2>&1
  echo "$(basename "$dir") eval exit=$?" | tee -a "$OUT/status.txt"
done
