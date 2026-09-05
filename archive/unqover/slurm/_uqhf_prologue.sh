#!/bin/bash
# Shared prologue for the unqover_hf (Phase-2) jobs.
#
# WHY THE PREFLIGHT: titan's shard allocator accounts shards, not bytes, so a
# job can be placed on a card another process has already filled (we lost two
# jobs to exactly that: 84 GiB in use, 22 MiB free, OOM on model load).  LLaDA-8B
# in bf16 needs ~17 GiB plus activations, so we refuse to start on a card with
# less than MIN_FREE_MIB free and ask SLURM to requeue us onto another one.
# The restart counter stops that from becoming an infinite loop.
set -e
WT=/shared/home/sarim.hashmi/dlm_bias/.claude/worktrees/agent-a9f2f89e11792ec37
export DLM_BIAS_ROOT=$WT
export HF_MODULES_CACHE=/shared/home/sarim.hashmi/.cache/hf_modules
export HF_HOME=/shared/home/sarim.hashmi/.cache/huggingface
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$WT"
PY=/shared/home/sarim.hashmi/anaconda3/envs/dlm/bin/python

MIN_FREE_MIB=${MIN_FREE_MIB:-24000}
MAX_REQUEUE=${MAX_REQUEUE:-12}
FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
echo "[preflight] host=$(hostname) CVD=${CUDA_VISIBLE_DEVICES} free=${FREE}MiB "\
"need=${MIN_FREE_MIB}MiB restart=${SLURM_RESTART_COUNT:-0}"
if [ "${FREE:-0}" -lt "$MIN_FREE_MIB" ]; then
  if [ "${SLURM_RESTART_COUNT:-0}" -lt "$MAX_REQUEUE" ]; then
    echo "[preflight] card too full -- requeueing onto another GPU"
    scontrol requeue "$SLURM_JOB_ID"
    sleep 60
    exit 0
  fi
  echo "[preflight] out of requeues; proceeding anyway (may OOM)"
fi
