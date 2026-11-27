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
CLONE_ALLOCATION=${CLONE_ALLOCATION:-ground_truth}   # or uniform: CLONES clones per observed colour
CLONES=${CLONES:-10}
HALF_LIFE=${HALF_LIFE:-40}   # EMA teacher half-life (observations); != 40 suffixes teacher arms with _hl<N>
ARMS=${ARMS:-"em_baselines jepa_mix1 gradcscg gradcscg_jepa"}
TEACHER_WINDOW=${TEACHER_WINDOW:-40}   # EMA teacher context; the replay window (and long-EM context) is tied to it
TW_SUFFIX=$([[ "$TEACHER_WINDOW" == 40 ]] && echo "" || echo "_tw$TEACHER_WINDOW")   # every arm: the window changes long EM too
JEPA_MIX=${JEPA_MIX:-1}   # accumulated-JEPA mix rho: new counts (1 - rho) xi + rho D
HL_SUFFIX=$([[ "$HALF_LIFE" == 40 ]] && echo "" || echo "_hl$HALF_LIFE")$TW_SUFFIX
if [[ "$CLONE_ALLOCATION" == uniform ]]; then
  OUT=${OUT:-results/diagnostics/cscg_vs_jepa_colour_tiles_uniform${CLONES}_3seed_20k}
else
  OUT=${OUT:-results/diagnostics/cscg_vs_jepa_colour_tiles_3seed_20k}
fi
PY=(env PYTHONPATH=.:src .venv/bin/python)
COMMON=(--levels $LEVELS --seeds $SEEDS --steps "$STEPS" --clone-allocation "$CLONE_ALLOCATION" --clones "$CLONES"
        --student-inference smooth --student-context 1 --forgetting 1.0 --pseudocount 0.002
        --window "$TEACHER_WINDOW" --teacher-window "$TEACHER_WINDOW" --half-life "$HALF_LIFE" --rand-prob "$RAND_PROB"
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

RAN=()
for arm in $ARMS; do
  case $arm in
    em_baselines) name=em_baselines$TW_SUFFIX; run $name --methods short long --local-updates 5 & ;;
    jepa_mix1) name=jepa_mix$JEPA_MIX$HL_SUFFIX; run $name --methods predictive --jepa-mix "$JEPA_MIX" --local-updates 5 & ;;
    gradcscg) name=gradcscg_lr0.01_adam10$TW_SUFFIX
      run $name --methods long --optimizer gradcscg --gradient-learning-rate 0.01 --local-updates 10 & ;;
    gradcscg_jepa) name=gradcscg_jepa_mix${JEPA_MIX}_lr0.01_adam10$HL_SUFFIX
      run $name --methods predictive --jepa-mix "$JEPA_MIX" --optimizer gradcscg --gradient-learning-rate 0.01 --local-updates 10 & ;;
    *) echo "unknown arm $arm" >&2; exit 2 ;;
  esac
  RAN+=("$name")
done
wait

for name in "${RAN[@]}"; do
  dir="$OUT/$name/"
  ls "$dir"level=*/seed=*/*/rows.json > /dev/null 2>&1 || { echo "skip evaluation of $dir (no completed case)"; continue; }
  "${PY[@]}" scripts/analysis/cscg_distillation_matched_eval.py evaluate --run "$dir" --steps all --workers "$WORKERS" \
    --walks "$EVAL_WALKS" --walk-length "$EVAL_WALK_LENGTH" --rand-prob "$EVAL_RAND_PROB" \
    > "${dir%/}.eval.log" 2>&1
  echo "$(basename "$dir") eval exit=$?" | tee -a "$OUT/status.txt"
done
