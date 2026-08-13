#!/bin/bash
#SBATCH --job-name=gdose
#SBATCH --account=nils
#SBATCH --partition=gpu
#SBATCH --qos=normal-plus
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=/shared/home/sarim.hashmi/dlm_bias/logs/slurm/%x-%j.out
#SBATCH --error=/shared/home/sarim.hashmi/dlm_bias/logs/slurm/%x-%j.err

# ROUND 3 JOB H "gdose" (afterok garrows): E3 gender dose calibration, target
# WOMAN, UNROTATED items data/bbq_items/_sweep400_woman.jsonl (the runner
# defaults). 8 runs -> results/multirace/woman/{decode_pid,normal}:
#   base                          cond_dpid_woman_base.json
#   decode-PI amax {1,2,4,6}      cond_dpid_woman_PI_amax{A}.json
#   normal    alpha {1,2,4}       cond_normal_woman_a{A}.json
# Race precedent says high doses can collapse to unparseable (white a4 was
# 98.8% unparseable) -- this sweep picks WOMAN_AMAX / WOMAN_ALPHA for jobI.
# Idempotent via run_if_missing.
set -euo pipefail

export HF_HOME=/shared/models/huggingface
export HF_HUB_CACHE=/shared/models/huggingface/hub
export HF_MODULES_CACHE=/shared/home/sarim.hashmi/.cache/hf_modules
export DLM_BIAS_ROOT=/shared/home/sarim.hashmi/dlm_bias
cd "$DLM_BIAS_ROOT"
PY=/shared/home/sarim.hashmi/anaconda3/envs/dlm/bin/python
# Do NOT set CUDA_VISIBLE_DEVICES -- SLURM sets it.

OUT=results/multirace/woman

run_if_missing() {  # $1 = expected result json, rest = command
    local marker=$1; shift
    if [[ -f "$marker" ]]; then
        echo "SKIP (exists): $marker"
    else
        echo "RUN: $*"
        "$@"
    fi
}

run_if_missing "$OUT/decode_pid/cond_dpid_woman_base.json" \
    $PY multirace/denoise_pid.py --target woman --cond base

for A in 1 2 4 6; do
    run_if_missing "$OUT/decode_pid/cond_dpid_woman_PI_amax${A}.json" \
        $PY multirace/denoise_pid.py --target woman --cond PI --amax "$A" \
            --tag "dpid_woman_PI_amax${A}"
done

for AL in 1 2 4; do
    run_if_missing "$OUT/normal/cond_normal_woman_a${AL}.json" \
        $PY multirace/normal.py --target woman --alpha "$AL" \
            --tag "normal_woman_a${AL}"
done

echo "ALL DONE (gdose: woman base + decode-PI amax{1,2,4,6} + normal a{1,2,4})"
