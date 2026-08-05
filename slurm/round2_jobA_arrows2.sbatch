#!/bin/bash
#SBATCH --job-name=arrows2
#SBATCH --account=nils
#SBATCH --partition=gpu
#SBATCH --qos=normal-plus
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=/shared/home/sarim.hashmi/dlm_bias/logs/slurm/%x-%j.out
#SBATCH --error=/shared/home/sarim.hashmi/dlm_bias/logs/slurm/%x-%j.err

# ROUND 2 JOB A "arrows2": build the remaining steering directions.
#   * multirace/arrows_<T>.pt for T in {arab,asian,latino,white}
#     (multirace/build_arrows.py --target T; manifest heldout, 400 items each)
#   * dream/arrows.pt (dream/build_arrows.py; Black direction for Dream)
# Idempotent: per-artifact skip-guards (safe under preempt+requeue).
set -euo pipefail

export HF_HOME=/shared/models/huggingface
export HF_HUB_CACHE=/shared/models/huggingface/hub
export HF_MODULES_CACHE=/shared/home/sarim.hashmi/.cache/hf_modules
export DLM_BIAS_ROOT=/shared/home/sarim.hashmi/dlm_bias
cd "$DLM_BIAS_ROOT"
PY=/shared/home/sarim.hashmi/anaconda3/envs/dlm/bin/python
# Do NOT set CUDA_VISIBLE_DEVICES -- SLURM sets it.

run_if_missing() {  # $1 = expected artifact, rest = command
    local marker=$1; shift
    if [[ -f "$marker" ]]; then
        echo "SKIP (exists): $marker"
    else
        echo "RUN: $*"
        "$@"
    fi
}

for T in arab asian latino white; do
    run_if_missing "multirace/arrows_${T}.pt" \
        $PY multirace/build_arrows.py --target "$T"
done

run_if_missing "dream/arrows.pt" \
    $PY dream/build_arrows.py

echo "ALL DONE (arrows2: multirace arab/asian/latino/white + dream)"
