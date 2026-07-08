#!/usr/bin/env bash
# Build + run all three STEERING baselines on the E1 BBQ item set, each at its OWN
# layer/coeff (see each baselines/<name>/config.json). Ours (full-layer closed-loop)
# is NOT here -- these are the comparison points.
#
# GPU RULE: ONLY use CUDA_VISIBLE_DEVICES 5, 6 or 7.
# Env      : /home/lukas/miniconda3/envs/sarim_awm/bin/python
set -euo pipefail
PY=/home/lukas/miniconda3/envs/sarim_awm/bin/python
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"

ITEMS=experiments/data/black_referent_ambig_eval.jsonl   # E1 set (same items as ours)
OUT=baselines/results; mkdir -p "$OUT"

# Build the E1 eval set (CPU, seconds) if absent.
[ -f "$ITEMS" ] || "$PY" experiments/build_eval_set.py

# ----------------------------------------------------------------------------- #
# 1) BUILD each baseline's direction (GPU). Each on its own card.
# ----------------------------------------------------------------------------- #
echo "== building baseline directions =="
CUDA_VISIBLE_DEVICES=5 "$PY" baselines/caa/build_caa.py     --layer 13 --device cuda &
CUDA_VISIBLE_DEVICES=6 "$PY" baselines/actadd/build_actadd.py --layer 6  --device cuda &
CUDA_VISIBLE_DEVICES=7 bash  baselines/group_meandiff/build.sh 7 &
wait

# ----------------------------------------------------------------------------- #
# 2) EVAL each baseline on the SAME E1 items, each at its OWN layer/coeff.
# ----------------------------------------------------------------------------- #
echo "== evaluating baselines on the E1 item set =="
CUDA_VISIBLE_DEVICES=5 "$PY" eval/bbq_eval.py --items "$ITEMS" \
  --direction-path baselines/caa/caa_direction.pt         --layer 13 --alpha 4 \
  --out "$OUT/bbq_caa.json" &
CUDA_VISIBLE_DEVICES=6 "$PY" eval/bbq_eval.py --items "$ITEMS" \
  --direction-path baselines/actadd/actadd_direction.pt   --layer 6  --alpha 5 \
  --out "$OUT/bbq_actadd.json" &
CUDA_VISIBLE_DEVICES=7 "$PY" eval/bbq_eval.py --items "$ITEMS" \
  --direction-path bias_steering/directions/L14/race_color.pt --layer 14 --alpha 4 \
  --out "$OUT/bbq_group_meandiff.json" &
wait

echo "== baselines done. results in $OUT/ =="
