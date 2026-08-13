#!/bin/bash
#SBATCH --job-name=suppress
#SBATCH --account=nils
#SBATCH --partition=gpu
#SBATCH --qos=normal-plus
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=/shared/home/sarim.hashmi/dlm_bias/logs/slurm/%x-%j.out
#SBATCH --error=/shared/home/sarim.hashmi/dlm_bias/logs/slurm/%x-%j.err

# ROUND 3 JOB J "suppress" (E5 closed-loop bias SUPPRESSION; no dependency --
# needs only steering/arrows.pt + the committed Black rotation files, both
# exist).  For rot r in {0,1,2} on results/balanced/_sweep400_rot{r}.jsonl:
#
#   suppress        steering/denoise_pid.py --cond PI --setpoint 0.0
#                   --amin -6 --amax 0  (same PI loop as the attack, aimed at
#                   s*=0 with negative-only actuation: pushes AWAY from Black)
#                   -> results/balanced_all/black/suppress/rot{r}/cond_dpid_PI_suppress.json
#
#   openloop_debias steering/pid_steer.py --mode normal --alpha -4  (constant
#                   negative dose of the SAME vhat at all 32 layers -- the
#                   equally-strong open-loop control condition; the tag
#                   auto-renders '-' as 'm': cond_normalL14_am4.json)
#                   -> results/balanced_all/black/normal_neg/rot{r}/
#
# base + steered-toward (decode-PI attack) already exist from round 1 under
# results/balanced/results_balanced/rot{r}/ -- NOT rerun; the analysis
# (balanced_all/strict_round3.py) compares against them.
# 6 runs, idempotent via run_if_missing.
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

    run_if_missing "$OUT/suppress/rot${r}/cond_dpid_PI_suppress.json" \
        $PY steering/denoise_pid.py --cond PI \
            --setpoint 0.0 --amin -6 --amax 0 --tag dpid_PI_suppress \
            --items "$ITEMS" --out-dir "$OUT/suppress/rot${r}"

    run_if_missing "$OUT/normal_neg/rot${r}/cond_normalL14_am4.json" \
        $PY steering/pid_steer.py --mode normal --alpha -4 \
            --items "$ITEMS" --out-dir "$OUT/normal_neg/rot${r}"
done

echo "ALL DONE (suppress: closed-loop s*=0 amin=-6 + open-loop a=-4 x rot0-2)"
