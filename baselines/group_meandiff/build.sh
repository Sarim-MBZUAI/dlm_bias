#!/usr/bin/env bash
# Group mean-difference baseline: a THIN wrapper over the EXISTING
# bias_steering/build_direction.py (CrowS-Pairs group mean-difference). We do NOT
# reimplement it -- this just invokes it at the baseline's own layer (L14) for the
# CrowS 'race-color' bias_type, writing bias_steering/directions/L14/race_color.pt.
#
# GPU RULE: ONLY use CUDA_VISIBLE_DEVICES 5, 6 or 7 (the USER runs this).
set -euo pipefail
PY=/home/lukas/miniconda3/envs/sarim_awm/bin/python
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"; cd "$ROOT"
LAYER=14
GPU=${1:-5}

CUDA_VISIBLE_DEVICES="$GPU" "$PY" bias_steering/build_direction.py \
  --source crows --layer "$LAYER" --categories race-color

echo "built -> bias_steering/directions/L${LAYER}/race_color.pt"
