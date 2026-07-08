#!/usr/bin/env bash
# E2 (the heart): construction ablation for aim-vs-disinhibit. Four steering
# directions run at MATCHED effective strength (unit-normalized, same alpha) on
# the SAME eval set, at block L14. Tests which CONSTRUCTION aims vs disinhibits.
#
# PREREQ: run_e1.sh first — E2 reuses e1_clean as the baseline.
# GPU RULE: ONLY use CUDA_VISIBLE_DEVICES 5, 6, or 7.
set -euo pipefail
PY=/home/lukas/miniconda3/envs/sarim_awm/bin/python
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
ITEMS=experiments/data/black_referent_ambig_eval.jsonl
OUT=experiments/results; mkdir -p "$OUT"

# 0) Build the FairPCA direction (GPU, ~5 min) if missing.
[ -f directional_steering/race_black_fairpca.pt ] || \
  CUDA_VISIBLE_DEVICES=5 $PY experiments/build_fairpca.py

# 1) Matched-strength ablation. --normalize-direction makes the open-loop push
#    equal to alpha for EVERY construction (unit vector), so differences are due
#    to the direction's CONSTRUCTION, not its raw norm. Sweep a few matched alphas.
declare -A PT=(
  [groupmeandiff]=race_steering/race_black.pt
  [letter]=directional_steering/race_black_anchored.pt
  [answertext]=directional_steering/race_black_anchored_text.pt
  [fairpca]=directional_steering/race_black_fairpca.pt
)
ALPHAS="8 20 40"        # matched effective (unit) strengths to sweep
GPUS=(5 6 7); gi=0
for A in $ALPHAS; do
  for name in groupmeandiff letter answertext fairpca; do
    gpu=${GPUS[$gi]}
    CUDA_VISIBLE_DEVICES=$gpu $PY eval/bbq_eval.py \
      --items "$ITEMS" --layer 14 --direction-path "${PT[$name]}" \
      --normalize-direction --alpha "$A" \
      --out "$OUT/e2_${name}_a${A}.json" &
    gi=$(( (gi + 1) % 3 ))
    [ "$gi" -eq 0 ] && wait      # throttle: 3 concurrent (one per GPU)
  done
done
wait

echo "== E2 runs complete. Analyzing per matched alpha =="
for A in $ALPHAS; do $PY experiments/e2_analyze.py --alpha "$A"; done
