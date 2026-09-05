#!/bin/bash
#SBATCH --job-name=ssqaD
#SBATCH --account=nils
#SBATCH --partition=gpu
#SBATCH --nodelist=mbz-titan-3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
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

for cond in aura_inject_g4 aura_vanilla itic_k48_a8; do
    for polarity in yes no; do
        for rotation in 0 1 2; do
            $PY -m socialstigma.run_one --condition "$cond" --polarity "$polarity" \
                --rotation "$rotation"
        done
    done
done

echo "ALL DONE (SocialStigmaQA D: AurA inject/vanilla/ITI-C)"
