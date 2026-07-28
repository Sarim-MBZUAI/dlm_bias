#!/usr/bin/env bash
# Phase-3 GPU-3 queue: OUR decode-space PID + layer-space PID/normal (prior work)
# on Dream-v0-Instruct-7B. Waits for dream/arrows.pt (built by dream/build_arrows.py),
# then queues 8 conditions x 400 items. Env pinned to transformers 4.46.2 (sarim_awm).
# STATUS: the 3 layer-space PID conditions (P/PI/PID at alpha 2, LLaDA magnitude) were
# killed mid-run — they produce ~97% unparseable output on Dream (see dream/RESULTS.md);
# they need a Dream-tuned (lower) alpha and are PENDING. Commands left unchanged.
set -u
cd /home/lukas/users/shashmi/dlm_bias
PY=/home/lukas/miniconda3/envs/sarim_awm/bin/python
export CUDA_VISIBLE_DEVICES=3
L=logs/gpu3; mkdir -p "$L"

echo "[gpu3] waiting for dream/arrows.pt ..."
while [ ! -f dream/arrows.pt ]; do sleep 15; done
echo "[gpu3] arrows ready; starting runs $(date)"

# --- our contribution: decode-space PID (feedback over the denoising step) ---
$PY dream/denoise_pid.py --cond base > "$L/decode_base.log" 2>&1 && echo "[gpu3] decode base  DONE"
$PY dream/denoise_pid.py --cond P    > "$L/decode_P.log"    2>&1 && echo "[gpu3] decode P     DONE"
$PY dream/denoise_pid.py --cond PI   > "$L/decode_PI.log"   2>&1 && echo "[gpu3] decode PI    DONE"
$PY dream/denoise_pid.py --cond PID  > "$L/decode_PID.log"  2>&1 && echo "[gpu3] decode PID   DONE"

# --- prior work: open-loop normal vector + layer-space PID ---
$PY dream/pid_steer.py --mode normal --source-layer 14 --alpha 4 > "$L/normal_a4.log" 2>&1 && echo "[gpu3] normal a4    DONE"
$PY dream/pid_steer.py --mode pid --cond P   --alpha 2 > "$L/layer_P.log"   2>&1 && echo "[gpu3] layer P      DONE"
$PY dream/pid_steer.py --mode pid --cond PI  --alpha 2 > "$L/layer_PI.log"  2>&1 && echo "[gpu3] layer PI     DONE"
$PY dream/pid_steer.py --mode pid --cond PID --alpha 2 > "$L/layer_PID.log" 2>&1 && echo "[gpu3] layer PID    DONE"

echo "[gpu3] ALL DONE $(date)"
