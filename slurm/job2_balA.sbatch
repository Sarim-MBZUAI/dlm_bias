#!/bin/bash
#SBATCH --job-name=balA
#SBATCH --account=nils
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=20:00:00
#SBATCH --output=/shared/home/sarim.hashmi/dlm_bias/logs/slurm/%x-%j.out
#SBATCH --error=/shared/home/sarim.hashmi/dlm_bias/logs/slurm/%x-%j.err

# JOB 2 "balA": position-balanced LLaDA Black-target baselines, first half.
# Methods caa / actadd / meanact / linearact x rotations 0,1,2 = 12 runs.
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

    run_if_missing "$OUT/caa/rot${r}/cond_caa_L14_a16.json" \
        $PY baselines/caa.py --run --alpha 16 --layer 14 \
            --items "$ITEMS" --out_dir "$OUT/caa/rot${r}"

    run_if_missing "$OUT/actadd/rot${r}/cond_actadd_a16.json" \
        $PY baselines/actadd.py --run --alpha 16 --layer 14 \
            --items "$ITEMS" --out-dir "$OUT/actadd/rot${r}"

    run_if_missing "$OUT/meanact/rot${r}/cond_meanact_unit_s2.json" \
        $PY baselines/meanact.py --run --direction unit --strength 2 \
            --items "$ITEMS" --out-dir "$OUT/meanact/rot${r}"

    run_if_missing "$OUT/linearact/rot${r}/cond_gaussian_s1.json" \
        $PY baselines/linearact.py --run --variant gaussian --strength 1 \
            --items "$ITEMS" --out_dir "$OUT/linearact/rot${r}"
done

echo "ALL DONE (balA: caa/actadd/meanact/linearact x rot0-2)"
