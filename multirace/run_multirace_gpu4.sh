#!/usr/bin/env bash
# Multirace Phase-3 GPU-4 queue: targets ASIAN + ARAB on LLaDA-8B-Instruct.
# Per target: build arrows (asian pre-built), then base / decode-PI / normal a4 / CAA a2.
# Env pinned to sarim_awm (transformers 4.46.2).
set -u
cd /home/lukas/users/shashmi/dlm_bias
PY=/home/lukas/miniconda3/envs/sarim_awm/bin/python
export CUDA_VISIBLE_DEVICES=4
L=logs/mr_gpu4; mkdir -p "$L"

for T in asian arab; do
  [ -f "multirace/arrows_${T}.pt" ] || { $PY multirace/build_arrows.py --target $T > "$L/arrows_$T.log" 2>&1 && echo "[mr4] arrows $T DONE"; }
  $PY multirace/denoise_pid.py --target $T --cond base > "$L/${T}_base.log"  2>&1 && echo "[mr4] $T base   DONE"
  $PY multirace/denoise_pid.py --target $T --cond PI   > "$L/${T}_dpid.log"  2>&1 && echo "[mr4] $T dpidPI DONE"
  $PY multirace/normal.py      --target $T --alpha 4   > "$L/${T}_norm.log"  2>&1 && echo "[mr4] $T normal DONE"
  $PY multirace/caa.py         --target $T --alpha 2   > "$L/${T}_caa.log"   2>&1 && echo "[mr4] $T caa    DONE"
done
echo "[mr4] ALL DONE $(date)"
