#!/usr/bin/env bash
# E1 (the gate): rescore the closed-loop conditions on the FULL Black-referent
# ambiguous set (n=1600, disjoint from the 400 anchored-build items) with the
# answer-text direction at block L14, so the +0.135 / +0.216 pilot gaps get
# bootstrap + McNemar CIs at large n.
#
# GPU RULE: ONLY use CUDA_VISIBLE_DEVICES 5, 6, or 7.
# Cost: ~45-60 min per 1600-item run; two runs fit on one 80 GB card.
set -euo pipefail
PY=/home/lukas/miniconda3/envs/sarim_awm/bin/python
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
ITEMS=experiments/data/black_referent_ambig_eval.jsonl
DIR=directional_steering/race_black_anchored_text.pt   # the answer-text winner
OUT=experiments/results; mkdir -p "$OUT"

# Build the eval set (CPU, seconds) if not present.
[ -f "$ITEMS" ] || $PY experiments/build_eval_set.py

run () {  # $1=gpu  $2=out-stem  $3...=extra eval args
  local gpu=$1 stem=$2; shift 2
  CUDA_VISIBLE_DEVICES=$gpu $PY eval/bbq_eval.py \
    --items "$ITEMS" --layer 14 --direction-path "$DIR" \
    --out "$OUT/$stem.json" "$@"
}

# 5 conditions across GPUs 5/6/7 (clean has no hook; open-loop uses the RAW
# vector to match the published anchored runs; clamp/cmom are closed-loop).
run 5 e1_clean     --alpha 0 &
run 6 e1_open_a4   --alpha 4 &
run 7 e1_open_a8   --alpha 8 &
wait
run 5 e1_clamp_c60 --steer-mode clamp --cstar 60 &
run 6 e1_cmom_c60  --steer-mode cmom  --cstar 60 --beta 0.8 &
wait

echo "== E1 runs complete. Analyzing (bootstrap + McNemar CIs) =="
$PY experiments/e1_analyze.py
