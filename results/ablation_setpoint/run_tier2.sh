#!/usr/bin/env bash
# Step 3 (optional): Tier 2, full 400-item evaluation per rotation (1,200
# position-balanced prompts pooled), for s* in {0.5, 0.9, 1.0}. Only run this
# after Tier 1 finished cleanly.
#
#   bash results/ablation_setpoint/run_tier2.sh
#
# About 2.5 hours on one GPU. Safe to rerun; finished runs are skipped.

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

OUT="$ABL/tier2"
SETPOINTS=(0.5 0.9 1.0)

for r in 0 1 2; do
  ITEMS="$ITEMS_DIR/_sweep400_rot$r.jsonl"
  for s in "${SETPOINTS[@]}"; do
    run_one tier2 "$r" "$(setpoint_tag "$s")" "$OUT/rot$r" -- \
      "${PI_ARGS[@]}" --setpoint "$s" --items "$ITEMS"
  done
done

"$PY" "$ABL/check_setpoints.py" "$OUT"
"$PY" "$ABL/score.py" "$OUT"
