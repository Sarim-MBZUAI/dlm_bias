#!/usr/bin/env bash
# GPU-4 queue: LATINO decode-PI amax sweep {2,3,4} (amax 6 was 0.30 unparseable),
# plus fair lower-alpha normal-vector points where alpha 4 collapsed
# (asian 0.998 / white 0.988 unparseable).
set -u
cd /home/lukas/users/shashmi/dlm_bias
PY=/home/lukas/miniconda3/envs/sarim_awm/bin/python
export CUDA_VISIBLE_DEVICES=4
L=logs/dose_gpu4; mkdir -p "$L"
for A in 2 3 4; do
  $PY multirace/denoise_pid.py --target latino --cond PI --amax $A --tag dpid_latino_PI_amax$A \
    > "$L/latino_amax$A.log" 2>&1 && echo "[dose4] latino amax$A DONE"
done
for T in asian white; do
  for AL in 1 2; do
    $PY multirace/normal.py --target $T --alpha $AL --tag normal_${T}_a$AL \
      > "$L/${T}_norm_a$AL.log" 2>&1 && echo "[dose4] $T normal a$AL DONE"
  done
done
echo "[dose4] ALL DONE $(date)"
