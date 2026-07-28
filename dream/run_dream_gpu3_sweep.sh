#!/usr/bin/env bash
# Dream dose-response for OUR decode-space PID. LLaDA's amax=6 drives Dream to
# ~97% unparseable (all-28-block actuation; Dream's residual stream is far more
# perturbation-sensitive than LLaDA's). Sweep amax low to find the coherence-
# preserving aim. cond=PI (the LLaDA headline). Kp=3 default => alpha pins near
# amax, so amax is the effective per-block injection magnitude.
set -u
cd /home/lukas/users/shashmi/dlm_bias
PY=/home/lukas/miniconda3/envs/sarim_awm/bin/python
export CUDA_VISIBLE_DEVICES=3
L=logs/gpu3sweep; mkdir -p "$L"
OUT=results/dream/decode_pid

for A in 0.25 0.5 1.0 1.5; do
  tag="dpid_PI_amax${A/./p}"
  $PY dream/denoise_pid.py --cond PI --amax "$A" --tag "$tag" --out-dir "$OUT" > "$L/$tag.log" 2>&1 \
    && echo "[gpu3sweep] $tag DONE"
done
echo "[gpu3sweep] ALL DONE $(date)"
