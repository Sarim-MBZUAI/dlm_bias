#!/usr/bin/env bash
# Step 1: reproduction check. Rerun the primary PI setting (s* = 0.9) on the
# first 100 items of each rotation, then compare outputs row by row against the
# published run in results/balanced/results_balanced/rot*/cond_dpid_PI_samples.jsonl.
#
#   bash results/ablation_setpoint/run_repro.sh
#
# About 10 minutes on one GPU. Safe to rerun; finished runs are skipped.

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

OUT="$ABL/tier1"
TAG="$(setpoint_tag 0.9)"   # dpid_PI_s0p9

for r in 0 1 2; do
  run_one tier1 "$r" "$TAG" "$OUT/rot$r" -- \
    "${PI_ARGS[@]}" --setpoint 0.9 --limit 100 \
    --items "$ITEMS_DIR/_sweep400_rot$r.jsonl"
done

"$PY" "$ABL/compare_repro.py" --tier-dir "$OUT" --ref-dir "$REF_DIR" --limit 100 \
  --out "$ABL/repro.json"
