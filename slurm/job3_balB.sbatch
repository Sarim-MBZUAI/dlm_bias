#!/bin/bash
#SBATCH --job-name=balB
#SBATCH --account=nils
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=20:00:00
#SBATCH --output=/shared/home/sarim.hashmi/dlm_bias/logs/slurm/%x-%j.out
#SBATCH --error=/shared/home/sarim.hashmi/dlm_bias/logs/slurm/%x-%j.err

# JOB 3 "balB": position-balanced LLaDA Black-target baselines, second half.
# Methods aura(inject) / aura(vanilla) / itic x rotations 0,1,2 = 9 runs.
# Invocations, operating points, and output layout replicate
# balanced_all/run_balanced_gpu2.sh EXACTLY (so balanced_all/aggregate.py works
# unchanged) -- but without its hardcoded CUDA_VISIBLE_DEVICES.
# Idempotent per run via run_if_missing (safe under preempt+requeue).
set -euo pipefail

export HF_HOME=/shared/models/huggingface
export HF_HUB_CACHE=/shared/models/huggingface/hub
export HF_MODULES_CACHE=/shared/home/sarim.hashmi/.cache/hf_modules
export DLM_BIAS_ROOT=/shared/home/sarim.hashmi/dlm_bias
cd "$DLM_BIAS_ROOT"
PY=/shared/home/sarim.hashmi/anaconda3/envs/dlm/bin/python
# Do NOT set CUDA_VISIBLE_DEVICES -- SLURM sets it.

OUT=results/balanced_all/black

run_if_missing() {  # $1 = expected result json, rest = command
    local marker=$1; shift
    if [[ -f "$marker" ]]; then
        echo "SKIP (exists): $marker"
    else
        echo "RUN: $*"
        "$@"
    fi
}

for r in 0 1 2; do
    ITEMS=results/balanced/_sweep400_rot${r}.jsonl

    run_if_missing "$OUT/aura_inject/rot${r}/cond_inject_g4.json" \
        $PY baselines/aura.py --run --mode inject --gamma 4 \
            --items "$ITEMS" --out-dir "$OUT/aura_inject/rot${r}"

    run_if_missing "$OUT/aura_vanilla/rot${r}/cond_vanilla.json" \
        $PY baselines/aura.py --run --mode vanilla \
            --items "$ITEMS" --out-dir "$OUT/aura_vanilla/rot${r}"

    run_if_missing "$OUT/itic/rot${r}/cond_itic_K48_a8.json" \
        $PY baselines/itic.py --run --topk 48 --alpha 8 \
            --items "$ITEMS" --out_dir "$OUT/itic/rot${r}"
done

echo "ALL DONE (balB: aura_inject/aura_vanilla/itic x rot0-2)"
