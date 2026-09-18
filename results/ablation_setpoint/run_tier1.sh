#!/usr/bin/env bash
# Step 2: Tier 1 grid on the first 100 items of each rotation.
# Conditions: base (no steering) and PI at s* in {0.5, 0.7, 1.0}. The s* = 0.9
# runs come from run_repro.sh (same output directory), completing the grid.
#
#   bash results/ablation_setpoint/run_tier1.sh
#
# About 40 minutes on one GPU (the original 100-item runs took ~200 s each).
# Safe to rerun; finished runs are skipped. Scores at the end.

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

OUT="$ABL/tier1"
SETPOINTS=(0.5 0.7 1.0)

for r in 0 1 2; do
  ITEMS="$ITEMS_DIR/_sweep400_rot$r.jsonl"
  run_one tier1 "$r" dpid_base "$OUT/rot$r" -- \
    --cond base --limit 100 --items "$ITEMS"
  for s in "${SETPOINTS[@]}"; do
    run_one tier1 "$r" "$(setpoint_tag "$s")" "$OUT/rot$r" -- \
      "${PI_ARGS[@]}" --setpoint "$s" --limit 100 --items "$ITEMS"
  done
done

# The grid is complete only if the s*=0.9 runs from Step 1 are present.
for r in 0 1 2; do
  f="$OUT/rot$r/cond_$(setpoint_tag 0.9)_samples.jsonl"
  [[ -s "$f" ]] || { echo "Missing $f. Run run_repro.sh first." >&2; exit 1; }
done

"$PY" "$ABL/check_setpoints.py" "$OUT"
"$PY" "$ABL/score.py" "$OUT"
