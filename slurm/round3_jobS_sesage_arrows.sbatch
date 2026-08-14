#!/bin/bash
#SBATCH --job-name=saarrows
#SBATCH --account=nils
#SBATCH --partition=gpu
#SBATCH --qos=normal-plus
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --output=/shared/home/sarim.hashmi/dlm_bias/logs/slurm/%x-%j.out
#SBATCH --error=/shared/home/sarim.hashmi/dlm_bias/logs/slurm/%x-%j.err

# ROUND 3 JOB S "saarrows" (E9 SES + Age; no dependency -- needs only the
# committed multirace/items_manifest_{ses,age}.json + data/bbq_cache/
# {SES,Age}.jsonl): build the four E9 steering directions
#   multirace/arrows_{lowses,highses,old,young}.pt
#   (+ direction_examples_{lowses,highses,old,young}.jsonl)
# via multirace/build_arrows.py --target <T> (manifest heldout, 400 items
# each; the negative option is the OTHER pole of the axis -- highses for
# lowses and vice versa, nonOld/'young' for old and vice versa).
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

for T in lowses highses old young; do
    run_if_missing "multirace/arrows_${T}.pt" \
        $PY multirace/build_arrows.py --target "$T"
done

echo "ALL DONE (saarrows: multirace lowses/highses/old/young)"
