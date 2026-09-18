#!/bin/bash
#SBATCH --job-name=ablsp2
#SBATCH --account=nils
#SBATCH --partition=gpu
#SBATCH --gres=shard:3
# Titan rejects --gres=gpu:1; shards are ~9.6 GB each. LLaDA-8B in bf16 needs
# ~16 GB weights + small batch-1 activations, so 3 shards (~28.8 GB) leave margin
# under the enforced budget (N*9.6 GB + 5% + 0.5 GB). Run `titan-shards` for the guide.
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=05:00:00
#SBATCH --requeue
#SBATCH --output=/shared/home/sarim.hashmi/dlm_bias/logs/slurm/ablsp2-%j.out
#SBATCH --error=/shared/home/sarim.hashmi/dlm_bias/logs/slurm/ablsp2-%j.err
# Setpoint ablation, Tier 2: full 1,200-prompt evaluation at s* in {0.5, 0.9, 1.0}.
# Step 3 of results/ablation_setpoint/README.md. Only submit after Tier 1
# finished cleanly. ~2.5 h on one GPU. Requeue-safe.
#
# Run on the SAME GPU model as Tier 1 (check env.json / timings.tsv). Pin with
#   sbatch --nodelist=<node> results/ablation_setpoint/slurm/ablation_setpoint_tier2.sbatch
# if the partition mixes GPU types.
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

[[ -f results/ablation_setpoint/tier1/summary.json ]] || { echo "Tier 1 not scored; run tier1 first." >&2; exit 1; }
[[ -f results/ablation_setpoint/env.json ]] || { echo "env.json missing; run preflight first." >&2; exit 1; }
EXPECTED_GPU="$($PY -c "import json;print(json.load(open('results/ablation_setpoint/env.json'))['gpu'])")"
THIS_GPU="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1 | sed 's/^ *//;s/ *$//')"
if [[ "$EXPECTED_GPU" != "$THIS_GPU" ]]; then
  echo "GPU mismatch: Tier 1 ran on '$EXPECTED_GPU', this node has '$THIS_GPU'. Resubmit with --nodelist." >&2
  exit 1
fi

bash results/ablation_setpoint/run_tier2.sh
$PY results/ablation_setpoint/make_report.py

echo "ABLATION SETPOINT TIER 2 DONE"
