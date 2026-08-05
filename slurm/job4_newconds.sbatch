#!/bin/bash
#SBATCH --job-name=newconds
#SBATCH --account=nils
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=15:00:00
#SBATCH --output=/shared/home/sarim.hashmi/dlm_bias/logs/slurm/%x-%j.out
#SBATCH --error=/shared/home/sarim.hashmi/dlm_bias/logs/slurm/%x-%j.err

# JOB 4 "newconds": 9 runs on the 3 Black rotation item files (needs only
# steering/arrows.pt from JOB 0; no baselines/cache fits):
#   (a) decode-time PID (full 64 steps)  -> black/decode_pid/rot{r}/cond_dpid_PID.json
#       (mirrors balanced_all/run_balanced_gpu4.sh's --items/--out-dir/--tag
#        conventions, with --cond PID)
#   (b) energy-matched open loop: pid_steer --mode normal --alpha 3.28
#       (controller's live-step mean actuation)
#                                        -> black/normal/rot{r}/cond_normalL14_a3p28.json
#       (tag built by pid_steer.py as f"normalL{L}_a{alpha:g}".replace('.','p'))
#   (c) decode-time PI at 32 steps       -> black/decode_pid_s32/rot{r}/cond_dpid_PI_s32.json
# Output layout keeps the results/balanced_all/black/<method>/rot{r}/cond_<tag>.json
# convention so balanced_all/aggregate.py works unchanged.
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

    # (a) decode-time PID, balanced (gpu4-script convention, --cond PID)
    run_if_missing "$OUT/decode_pid/rot${r}/cond_dpid_PID.json" \
        $PY steering/denoise_pid.py --cond PID \
            --items "$ITEMS" --out-dir "$OUT/decode_pid/rot${r}" --tag dpid_PID

    # (b) energy-matched open loop (single L14 vector, all layers, fixed alpha)
    run_if_missing "$OUT/normal/rot${r}/cond_normalL14_a3p28.json" \
        $PY steering/pid_steer.py --mode normal --alpha 3.28 \
            --items "$ITEMS" --out-dir "$OUT/normal/rot${r}"

    # (c) decode-time PI at 32 denoising steps
    run_if_missing "$OUT/decode_pid_s32/rot${r}/cond_dpid_PI_s32.json" \
        $PY steering/denoise_pid.py --cond PI --steps 32 \
            --items "$ITEMS" --out-dir "$OUT/decode_pid_s32/rot${r}" --tag dpid_PI_s32
done

echo "ALL DONE (newconds: decode_pid PID / normal a3.28 / PI s32 x rot0-2)"
