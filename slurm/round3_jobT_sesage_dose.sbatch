#!/bin/bash
#SBATCH --job-name=sadose
#SBATCH --account=nils
#SBATCH --partition=gpu
#SBATCH --qos=normal-plus
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=/shared/home/sarim.hashmi/dlm_bias/logs/slurm/%x-%j.out
#SBATCH --error=/shared/home/sarim.hashmi/dlm_bias/logs/slurm/%x-%j.err

# ROUND 3 JOB T "sadose" (afterok saarrows -- submit with
#   sbatch --dependency=afterok:<jobS_id> slurm/round3_jobT_sesage_dose.sbatch
# ): E9 dose calibration on ONE pole per axis, targets LOWSES and OLD,
# UNROTATED items data/bbq_items/_sweep400_{lowses,old}.jsonl (the runner
# defaults). 12 runs -> results/multirace/{lowses,old}/{decode_pid,normal}:
#   per target T in {lowses, old}:
#     base                       cond_dpid_<T>_base.json
#     decode-PI amax {2,4,6}     cond_dpid_<T>_PI_amax{A}.json
#     normal    alpha {2,4}      cond_normal_<T>_a{A}.json
#
# POLE-MIRROR DOSE INHERITANCE (accepted design): the sweep runs on the
# lowses/old poles ONLY; the mirror poles (highses, young) INHERIT the dose
# picked here. Both poles of an axis live on the same kind of rows (identical
# 4-way disjoint split of one shared pool), so per-pole re-calibration would
# double the GPU cost for little information -- but it does ASSUME dose
# transfer across poles; jobU/jobV telemetry should be checked against this.
# Race precedent says high doses can collapse to unparseable (white a4 was
# 98.8% unparseable) -- this sweep picks SES_AMAX/SES_ALPHA (jobU) and
# AGE_AMAX/AGE_ALPHA (jobV).
# Idempotent via run_if_missing.
set -euo pipefail

export HF_HOME=/shared/models/huggingface
export HF_HUB_CACHE=/shared/models/huggingface/hub
export HF_MODULES_CACHE=/shared/home/sarim.hashmi/.cache/hf_modules
export DLM_BIAS_ROOT=/shared/home/sarim.hashmi/dlm_bias
cd "$DLM_BIAS_ROOT"
PY=/shared/home/sarim.hashmi/anaconda3/envs/dlm/bin/python
# Do NOT set CUDA_VISIBLE_DEVICES -- SLURM sets it.

run_if_missing() {  # $1 = expected result json, rest = command
    local marker=$1; shift
    if [[ -f "$marker" ]]; then
        echo "SKIP (exists): $marker"
    else
        echo "RUN: $*"
        "$@"
    fi
}

for T in lowses old; do
    OUT=results/multirace/$T

    run_if_missing "$OUT/decode_pid/cond_dpid_${T}_base.json" \
        $PY multirace/denoise_pid.py --target "$T" --cond base

    for A in 2 4 6; do
        run_if_missing "$OUT/decode_pid/cond_dpid_${T}_PI_amax${A}.json" \
            $PY multirace/denoise_pid.py --target "$T" --cond PI --amax "$A" \
                --tag "dpid_${T}_PI_amax${A}"
    done

    for AL in 2 4; do
        run_if_missing "$OUT/normal/cond_normal_${T}_a${AL}.json" \
            $PY multirace/normal.py --target "$T" --alpha "$AL" \
                --tag "normal_${T}_a${AL}"
    done
done

echo "ALL DONE (sadose: lowses+old base + decode-PI amax{2,4,6} + normal a{2,4})"
