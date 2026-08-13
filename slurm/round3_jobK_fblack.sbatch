#!/bin/bash
#SBATCH --job-name=fbdose
#SBATCH --account=nils
#SBATCH --partition=gpu
#SBATCH --qos=normal-plus
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=/shared/home/sarim.hashmi/dlm_bias/logs/slurm/%x-%j.out
#SBATCH --error=/shared/home/sarim.hashmi/dlm_bias/logs/slurm/%x-%j.err

# ROUND 3 JOB K "fbdose" (E6 fblack; no dependency -- needs only the
# committed multirace/items_manifest_fblack.json + the Race_ethnicity cache):
#   1. build the fblack steering direction multirace/arrows_fblack.pt
#      (+ direction_examples_fblack.jsonl) from the 312 manifest heldout rows
#      (documented 400/312 deviation; the negative option may be m-black by
#      policy -- empirically it is ALWAYS an other-race woman (0 m-black), so
#      this is a GENDER-CONDITIONED RACE direction (f-black vs other-race
#      women), not a full intersectional contrast -- see
#      multirace/build_arrows.py module doc).
#   2. dose calibration, UNROTATED items data/bbq_items/_sweep400_fblack.jsonl
#      (runner defaults). 6 runs -> results/multirace/fblack/{decode_pid,normal}:
#        base                       cond_dpid_fblack_base.json
#        decode-PI amax {2,4,6}     cond_dpid_fblack_PI_amax{A}.json
#        normal    alpha {2,4}      cond_normal_fblack_a{A}.json
# Race precedent says high doses can collapse to unparseable (white a4 was
# 98.8% unparseable) -- this sweep picks FBLACK_AMAX / FBLACK_ALPHA for jobL.
# Idempotent via run_if_missing (safe under preempt+requeue).
set -euo pipefail

export HF_HOME=/shared/models/huggingface
export HF_HUB_CACHE=/shared/models/huggingface/hub
export HF_MODULES_CACHE=/shared/home/sarim.hashmi/.cache/hf_modules
export DLM_BIAS_ROOT=/shared/home/sarim.hashmi/dlm_bias
cd "$DLM_BIAS_ROOT"
PY=/shared/home/sarim.hashmi/anaconda3/envs/dlm/bin/python
# Do NOT set CUDA_VISIBLE_DEVICES -- SLURM sets it.

OUT=results/multirace/fblack

run_if_missing() {  # $1 = expected result json, rest = command
    local marker=$1; shift
    if [[ -f "$marker" ]]; then
        echo "SKIP (exists): $marker"
    else
        echo "RUN: $*"
        "$@"
    fi
}

# ---- arrows (312 heldout items, ~40 min) ------------------------------------ #
run_if_missing "multirace/arrows_fblack.pt" \
    $PY multirace/build_arrows.py --target fblack

# ---- unrotated dose probes --------------------------------------------------- #
run_if_missing "$OUT/decode_pid/cond_dpid_fblack_base.json" \
    $PY multirace/denoise_pid.py --target fblack --cond base

for A in 2 4 6; do
    run_if_missing "$OUT/decode_pid/cond_dpid_fblack_PI_amax${A}.json" \
        $PY multirace/denoise_pid.py --target fblack --cond PI --amax "$A" \
            --tag "dpid_fblack_PI_amax${A}"
done

for AL in 2 4; do
    run_if_missing "$OUT/normal/cond_normal_fblack_a${AL}.json" \
        $PY multirace/normal.py --target fblack --alpha "$AL" \
            --tag "normal_fblack_a${AL}"
done

echo "ALL DONE (fbdose: arrows + fblack base + decode-PI amax{2,4,6} + normal a{2,4})"
