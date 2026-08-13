#!/bin/bash
#SBATCH --job-name=garrows
#SBATCH --account=nils
#SBATCH --partition=gpu
#SBATCH --qos=normal-plus
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=/shared/home/sarim.hashmi/dlm_bias/logs/slurm/%x-%j.out
#SBATCH --error=/shared/home/sarim.hashmi/dlm_bias/logs/slurm/%x-%j.err

# ROUND 3 JOB G "garrows" (E3 gender; no dependency -- needs only the committed
# multirace/items_manifest_gender.json + data/bbq_cache/Gender_identity.jsonl):
# build the two gender steering directions
#   multirace/arrows_{woman,man}.pt (+ direction_examples_{woman,man}.jsonl)
# via multirace/build_arrows.py --target {woman,man} (manifest heldout, 400
# items each; the negative option is the other-gendered person).
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

for T in woman man; do
    run_if_missing "multirace/arrows_${T}.pt" \
        $PY multirace/build_arrows.py --target "$T"
done

echo "ALL DONE (garrows: multirace woman/man)"
