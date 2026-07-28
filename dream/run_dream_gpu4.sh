#!/usr/bin/env bash
# Phase-3 GPU-4 queue: the 6 faithful baselines on Dream-v0-Instruct-7B.
# run_all --fit --run builds every fit artifact (calib caches, actadd dir, itic
# probes, aura auroc) then evaluates caa/actadd/meanact/linearact/aura/itic.
# Then aura --mode vanilla is the passing negative control (run_all does inject).
# Waits for dream/arrows.pt (caa/meanact read it). Env pinned to sarim_awm.
set -u
cd /home/lukas/users/shashmi/dlm_bias
PY=/home/lukas/miniconda3/envs/sarim_awm/bin/python
export CUDA_VISIBLE_DEVICES=4
L=logs/gpu4; mkdir -p "$L"

echo "[gpu4] waiting for dream/arrows.pt ..."
while [ ! -f dream/arrows.pt ]; do sleep 15; done
echo "[gpu4] arrows ready; fit+run 6 baselines $(date)"

$PY dream/baselines/run_all.py --fit --run --method all > "$L/run_all.log" 2>&1 && echo "[gpu4] run_all      DONE"
$PY dream/baselines/aura.py --run --mode vanilla --out_dir results/dream/aura --tag vanilla > "$L/aura_vanilla.log" 2>&1 && echo "[gpu4] aura vanilla DONE"

echo "[gpu4] ALL DONE $(date)"
