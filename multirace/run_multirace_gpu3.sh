#!/usr/bin/env bash
# Multirace Phase-3 GPU-3 queue: targets WHITE + LATINO on LLaDA-8B-Instruct.
# Per target: build arrows, then base / decode-PI (ours) / normal a4 / CAA a2.
# Env pinned to sarim_awm (transformers 4.46.2).
set -u
cd "${DLM_BIAS_ROOT:-$(dirname "$0")/..}"
PY="${PY:-python}"
export CUDA_VISIBLE_DEVICES=3
L=logs/mr_gpu3; mkdir -p "$L"

for T in white latino; do
  [ -f "multirace/arrows_${T}.pt" ] || { $PY multirace/build_arrows.py --target $T > "$L/arrows_$T.log" 2>&1 && echo "[mr3] arrows $T DONE"; }
  $PY multirace/denoise_pid.py --target $T --cond base > "$L/${T}_base.log"  2>&1 && echo "[mr3] $T base   DONE"
  $PY multirace/denoise_pid.py --target $T --cond PI   > "$L/${T}_dpid.log"  2>&1 && echo "[mr3] $T dpidPI DONE"
  $PY multirace/normal.py      --target $T --alpha 4   > "$L/${T}_norm.log"  2>&1 && echo "[mr3] $T normal DONE"
  $PY multirace/caa.py         --target $T --alpha 2   > "$L/${T}_caa.log"   2>&1 && echo "[mr3] $T caa    DONE"
done
echo "[mr3] ALL DONE $(date)"
