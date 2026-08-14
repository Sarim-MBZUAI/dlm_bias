#!/bin/bash
#SBATCH --job-name=r3smokeJ
#SBATCH --account=nils
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=/shared/home/sarim.hashmi/dlm_bias/logs/slurm/r3smokeJ-%j.out
#SBATCH --error=/shared/home/sarim.hashmi/dlm_bias/logs/slurm/r3smokeJ-%j.err
# E5 pre-run smoke: verify the suppression controller goes negative on-plant
# (reviewer-recommended gate before the 12h round3_jobJ run).
set -euo pipefail
export HF_HOME=/shared/models/huggingface
export HF_HUB_CACHE=/shared/models/huggingface/hub
export HF_MODULES_CACHE=/shared/home/sarim.hashmi/.cache/hf_modules
export DLM_BIAS_ROOT=/shared/home/sarim.hashmi/dlm_bias
cd "$DLM_BIAS_ROOT"
PY=/shared/home/sarim.hashmi/anaconda3/envs/dlm/bin/python

# Closed-loop suppression smoke: alpha(t) must be negative and non-constant.
$PY steering/denoise_pid.py --smoke --smoke-items 2 --setpoint 0.0 --amin -6 --amax 0 --kd 0

# Open-loop negative-alpha smoke (2 items).
$PY steering/pid_steer.py --mode normal --alpha -4 --limit 2 --out-dir /tmp/r3smokeJ_$SLURM_JOB_ID

echo "R3 SMOKE J PASSED"
