#!/bin/bash
#SBATCH --job-name=ssqaEff
#SBATCH --account=nils
#SBATCH --partition=gpu
#SBATCH --nodelist=mbz-titan-3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --output=/shared/home/sarim.hashmi/dlm_bias/logs/slurm/%x-%j.out
#SBATCH --error=/shared/home/sarim.hashmi/dlm_bias/logs/slurm/%x-%j.err

set -euo pipefail
export HF_HOME=/shared/models/huggingface
export HF_HUB_CACHE=/shared/models/huggingface/hub
export HF_MODULES_CACHE=/shared/home/sarim.hashmi/.cache/hf_modules
export DLM_BIAS_ROOT=/shared/home/sarim.hashmi/dlm_bias
cd "$DLM_BIAS_ROOT"
PY=/shared/home/sarim.hashmi/anaconda3/envs/dlm/bin/python

echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader

ALPHA=$($PY -m socialstigma.compute_effort_match --print-alpha)
echo "Preregistered label-free effort-matched alpha: $ALPHA"
for polarity in yes no; do
    for rotation in 0 1 2; do
        $PY -m socialstigma.run_one --condition normal_eff --effort-alpha "$ALPHA" \
            --polarity "$polarity" --rotation "$rotation"
    done
done

echo "ALL DONE (SocialStigmaQA effort-matched open loop alpha=$ALPHA)"
