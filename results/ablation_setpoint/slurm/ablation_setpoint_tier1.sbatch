#!/bin/bash
#SBATCH --job-name=ablsp1
#SBATCH --account=nils
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --requeue
#SBATCH --output=/shared/home/sarim.hashmi/dlm_bias/logs/slurm/ablsp1-%j.out
#SBATCH --error=/shared/home/sarim.hashmi/dlm_bias/logs/slurm/ablsp1-%j.err
# Setpoint ablation, Tier 1: preflight + reproduction check + 100-item grid.
# Steps 0-2 of results/ablation_setpoint/README.md. ~1 h on one GPU.
# Requeue-safe: finished runs are skipped on resume.
set -euo pipefail
export HF_HOME=/shared/models/huggingface
export HF_HUB_CACHE=/shared/models/huggingface/hub
export HF_MODULES_CACHE=/shared/home/sarim.hashmi/.cache/hf_modules
export DLM_BIAS_ROOT=/shared/home/sarim.hashmi/dlm_bias
cd "$DLM_BIAS_ROOT"
export PY=/shared/home/sarim.hashmi/anaconda3/envs/dlm/bin/python
# Do NOT set CUDA_VISIBLE_DEVICES -- SLURM sets it.

echo "host=$(hostname) job=$SLURM_JOB_ID CVD=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv

bash results/ablation_setpoint/preflight.sh
bash results/ablation_setpoint/run_repro.sh
bash results/ablation_setpoint/run_tier1.sh
$PY results/ablation_setpoint/make_report.py

echo "ABLATION SETPOINT TIER 1 DONE"
