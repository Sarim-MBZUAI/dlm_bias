#!/usr/bin/env bash
# Dose-response for the targets that collapsed/strained at LLaDA defaults:
# GPU-3 queue = ASIAN decode-PI amax sweep {1,2,3,4} (amax 6 = 0.93 unparseable).
set -u
cd "${DLM_BIAS_ROOT:-$(dirname "$0")/..}"
PY="${PY:-python}"
export CUDA_VISIBLE_DEVICES=3
L=logs/dose_gpu3; mkdir -p "$L"
for A in 1 2 3 4; do
  $PY multirace/denoise_pid.py --target asian --cond PI --amax $A --tag dpid_asian_PI_amax$A \
    > "$L/asian_amax$A.log" 2>&1 && echo "[dose3] asian amax$A DONE"
done
echo "[dose3] ALL DONE $(date)"
