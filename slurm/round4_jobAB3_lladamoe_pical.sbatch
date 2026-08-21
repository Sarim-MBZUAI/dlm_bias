#!/bin/bash
#SBATCH --job-name=moeab3
#SBATCH --account=nils
#SBATCH --partition=gpu
#SBATCH --qos=normal-plus
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=/shared/home/sarim.hashmi/dlm_bias/logs/slurm/%x-%j.out
#SBATCH --error=/shared/home/sarim.hashmi/dlm_bias/logs/slurm/%x-%j.err

# ROUND 4 JOB AB3 "moeab3" -- decode-PI calibration sweep after the L8-amax2
# trajectory forensics (results/lladamoe/SWEEP_NOTES.md, "PI calibration round
# (2026-08-21)").  Diagnosis from cond_dpid_PI_L8_amax2_samples.jsonl:
#   * The answer slot COMMITS at step ~32/64 (median lock step 32) -- the
#     controllable window is only the first half of generation.
#   * (b) PLATEAU, not sluggishness: on the 210 failed items alpha is pinned at
#     amax=2 for essentially the whole pre-lock window (pre-lock mean 1.88) yet
#     p_black plateaus at pre-lock max 0.46 mean; at lock time only 36/162
#     pinned failures were still rising.  More time at alpha=2 cannot help.
#   * NOT bimodal: pre-lock p_max is a broad unimodal continuum (60% of items
#     reach 0.5, 8% reach the 0.9 setpoint).  Graded difficulty, shared ceiling.
#   * The controller itself is healthy: corr(p(t), alpha(t+1)) = -0.58;
#     items starting at p>=0.85 get mean alpha 0.78 (vs 1.96 for p<=0.30).
# => Calibrate the CEILING (amax at L8) and the SITE (L10/L12 out-signal
#    better than L8 per ITI-c probes: mean val_acc .751/.822 vs .675) and test
#    DISTRIBUTED actuation (CAA broke between const 2 and 4 at L8 -- the
#    single-site dose wall -- so spread the budget across sub-collapse sites).
# 7 runs x ~1.3-1.6h (measured) = ~9.5-11.2h; ordered by priority so a
# walltime kill costs the least informative cell.  Idempotent per run via
# run_if_missing (jobAB convention); all tags are NEW stems (no collision with
# cond_dpid_PI_amax* [all-16 broadcast] or cond_dpid_PI_L8_amax2).
set -euo pipefail

export HF_HOME=/shared/models/huggingface
export HF_HUB_CACHE=/shared/models/huggingface/hub
export HF_MODULES_CACHE=/shared/home/sarim.hashmi/.cache/hf_modules
export DLM_BIAS_ROOT=/shared/home/sarim.hashmi/dlm_bias
cd "$DLM_BIAS_ROOT"
PY=/shared/home/sarim.hashmi/anaconda3/envs/dlm/bin/python
# Do NOT set CUDA_VISIBLE_DEVICES -- SLURM sets it.

OUT=results/lladamoe

run_if_missing() {  # $1 = expected result json, rest = command
    local marker=$1; shift
    if [[ -f "$marker" ]]; then
        echo "SKIP (exists): $marker"
    else
        echo "RUN: $*"
        "$@"
    fi
}

# Prerequisites from round4_jobAB (fail loud if absent).
test -f llada_moe/arrows.pt || { echo "FATAL: llada_moe/arrows.pt missing (run round4_jobAB first)"; exit 1; }

# (1) L8 amax 3 -- lift the ceiling at the incumbent site.  Feedback thesis:
#     alpha=3 is spent only pre-lock on hard items and backs off on easy ones,
#     unlike CAA's constant dose.
run_if_missing "$OUT/decode_pid/cond_dpid_PI_L8_amax3.json" \
    $PY llada_moe/denoise_pid.py --cond PI --amax 3 --layers 8 \
        --tag dpid_PI_L8_amax3

# (2) L10 amax 4 -- relative-dose parity (4/||h||=4.61 = 87%, ~= L8-amax2's
#     2/2.4 = 83%) at a better-informed site (ITI-c probe mean .751 vs .675).
run_if_missing "$OUT/decode_pid/cond_dpid_PI_L10_amax4.json" \
    $PY llada_moe/denoise_pid.py --cond PI --amax 4 --layers 10 \
        --tag dpid_PI_L10_amax4

# (3) L8 amax 4 -- dose-response upper end.  CAA broke at CONSTANT 4 (12%
#     invalid); transient feedback-4 tests whether the wall is dose or duration.
run_if_missing "$OUT/decode_pid/cond_dpid_PI_L8_amax4.json" \
    $PY llada_moe/denoise_pid.py --cond PI --amax 4 --layers 8 \
        --tag dpid_PI_L8_amax4

# (4) L7,8,9 amax 1 -- distribute the budget (total L2 = 3) across three
#     mid-band sites, each at a sub-collapse per-site dose (CAA a1 was clean).
run_if_missing "$OUT/decode_pid/cond_dpid_PI_L789_amax1.json" \
    $PY llada_moe/denoise_pid.py --cond PI --amax 1 --layers 7,8,9 \
        --tag dpid_PI_L789_amax1

# (5) L10 amax 2 -- same nominal budget as the incumbent best at the
#     better-informed site (43% of local ||h||); separates site from dose.
run_if_missing "$OUT/decode_pid/cond_dpid_PI_L10_amax2.json" \
    $PY llada_moe/denoise_pid.py --cond PI --amax 2 --layers 10 \
        --tag dpid_PI_L10_amax2

# (6) L9-12 amax 1 -- distributed actuation over the highest-signal band
#     (probe means .730-.822), total budget 4, per-site rel dose <= 31%.
run_if_missing "$OUT/decode_pid/cond_dpid_PI_L9to12_amax1.json" \
    $PY llada_moe/denoise_pid.py --cond PI --amax 1 --layers 9-12 \
        --tag dpid_PI_L9to12_amax1

# (7) L12 amax 4 -- most separable actuation-plausible site (probe mean .822)
#     at 34% rel dose (amax 2 there would be a predictable 17% underdose --
#     that cell is skipped).
run_if_missing "$OUT/decode_pid/cond_dpid_PI_L12_amax4.json" \
    $PY llada_moe/denoise_pid.py --cond PI --amax 4 --layers 12 \
        --tag dpid_PI_L12_amax4

echo "ALL DONE (moeab3: L8 a{3,4} + L10 a{2,4} + L12 a4 + L789 a1 + L9-12 a1)"
