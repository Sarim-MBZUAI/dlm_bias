#!/bin/bash
#SBATCH --job-name=moebal2
#SBATCH --account=nils
#SBATCH --partition=gpu
#SBATCH --qos=normal-plus
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=/shared/home/sarim.hashmi/dlm_bias/logs/slurm/%x-%j.out
#SBATCH --error=/shared/home/sarim.hashmi/dlm_bias/logs/slurm/%x-%j.err

# ROUND 4 JOB AD2 "moebal2": the DISCLOSED SECONDARY LLaDA-MoE balanced
# condition promoted by the pre-registered PI calibration grid (jobAB3 job
# 20975; results/lladamoe/SWEEP_NOTES.md "Post-grid decision (2026-08-21)").
# The grid's headline bar was NOT met (no cell beat CAA a2's +0.237 at
# inv <= 0.01), so the HEADLINE decode-PI stays L8 amax2 (already run by
# jobAD in results/lladamoe_balanced/decode_pid).  The best clean cell --
# PI --layers 9-12 --amax 1: gap +0.225, inv 0.000, mean_alpha 0.94 -- gets
# this secondary balanced run so pooled 3x400 CIs can decide the CAA-parity
# claim.  Doses are FIXED (pre-registered by the grid), not overridable.
#
# 2 conditions x rot0-2 = 6 runs, SAME LLaDA Black rotation item files
# results/balanced/_sweep400_rot{r}.jsonl as jobAD:
#   decode-PI sec.  denoise_pid.py --cond PI --layers 9-12 --amax 1
#                   --tag dpid_PI_L9to12
#                       -> results/lladamoe_balanced/decode_pid_L9to12/rot{r}/
#                          cond_dpid_PI_L9to12.json (+ _samples.jsonl)
#   normal sec.     pid_steer.py --mode normal --source-layer 8
#                   --layers 9-12 --alpha 1 --tag normal_L9to12_a1
#                       -> results/lladamoe_balanced/normal_L9to12/rot{r}/
#                          cond_normal_L9to12_a1.json (+ _samples.jsonl)
#
# GEOMETRY/EFFORT MATCH: decode-PI's default unit mode injects the ONE source
# vector unit(r[8]) (denoise_pid.build_injection_matrix, layer_scale="unit")
# at every actuated block -- for --layers 9-12 that is alpha(t)*unit(r[8]) at
# blocks 9,10,11,12.  pid_steer --mode normal --source-layer 8 --layers 9-12
# injects exactly the same rows (alpha*unit(r[8]) at blocks 9-12), so the open
# loop is geometry-matched with NO code change (verified by pid_steer
# --selftest "normal L8 rows 9-12 == decode-PI unit rows 9-12").  Effort:
# constant alpha 1 = the grid cell's realized mean_alpha 0.94 rounded UP
# (conservative over-match, the round-3 Age / jobAD normal-L8 convention).
#
# STEMS: --tag dpid_PI_L9to12 / normal_L9to12_a1 -> cond_dpid_PI_L9to12 /
# cond_normal_L9to12_a1: no collision with jobAD's cond_dpid_PI /
# cond_normalL8_a2 (different stems AND different sub-dirs), so
# balanced_all/strict_round2.py family 9 discovers both operating points
# side by side (LLADAMOE_SUBS gained decode_pid_L9to12 + normal_L9to12,
# silently skipped while these dirs are absent).
# Directions come from llada_moe/arrows.pt (round4_jobAB fits) -- never from
# the LLaDA or Dream artifacts.  Idempotent per run via run_if_missing.
set -euo pipefail

export HF_HOME=/shared/models/huggingface
export HF_HUB_CACHE=/shared/models/huggingface/hub
export HF_MODULES_CACHE=/shared/home/sarim.hashmi/.cache/hf_modules
export DLM_BIAS_ROOT=/shared/home/sarim.hashmi/dlm_bias
cd "$DLM_BIAS_ROOT"
PY=/shared/home/sarim.hashmi/anaconda3/envs/dlm/bin/python
# Do NOT set CUDA_VISIBLE_DEVICES -- SLURM sets it.

OUT=results/lladamoe_balanced

run_if_missing() {  # $1 = expected result json, rest = command
    local marker=$1; shift
    if [[ -f "$marker" ]]; then
        echo "SKIP (exists): $marker"
    else
        echo "RUN: $*"
        "$@"
    fi
}

# Fit prerequisite (built by round4_jobAB): fail loud if absent.
test -f llada_moe/arrows.pt || { echo "FATAL: llada_moe/arrows.pt missing (run round4_jobAB first)"; exit 1; }

echo "secondary condition (fixed, pre-registered): PI --layers 9-12 --amax 1; normal --source-layer 8 --layers 9-12 --alpha 1"

for r in 0 1 2; do
    ITEMS=results/balanced/_sweep400_rot${r}.jsonl
    test -f "$ITEMS" || { echo "FATAL: $ITEMS missing"; exit 1; }

    run_if_missing "$OUT/decode_pid_L9to12/rot${r}/cond_dpid_PI_L9to12.json" \
        $PY llada_moe/denoise_pid.py --cond PI --layers 9-12 --amax 1 \
            --tag dpid_PI_L9to12 \
            --items "$ITEMS" --out-dir "$OUT/decode_pid_L9to12/rot${r}"

    run_if_missing "$OUT/normal_L9to12/rot${r}/cond_normal_L9to12_a1.json" \
        $PY llada_moe/pid_steer.py --mode normal --source-layer 8 \
            --layers 9-12 --alpha 1 --tag normal_L9to12_a1 \
            --items "$ITEMS" --out-dir "$OUT/normal_L9to12/rot${r}"
done

echo "ALL DONE (moebal2: 2 secondary conditions x rot0-2 -> results/lladamoe_balanced/{decode_pid_L9to12,normal_L9to12})"
