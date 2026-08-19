#!/bin/bash
#SBATCH --job-name=moefit
#SBATCH --account=nils
#SBATCH --partition=gpu
#SBATCH --qos=normal-plus
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --output=/shared/home/sarim.hashmi/dlm_bias/logs/slurm/%x-%j.out
#SBATCH --error=/shared/home/sarim.hashmi/dlm_bias/logs/slurm/%x-%j.err

# ROUND 4 JOB AC "moefit" (afterok moearw): calibration collection + per-method
# fit artifacts for the LLaDA-MoE-7B-A1B-Instruct baseline suite -- the exact
# round3_jobY (dfits) pattern on the MoE port (llada_moe/baselines/*).
# All fits are built from LLaDA-MoE activations (llada_moe/baselines/calib.py
# collects on the same contamination-safe held-out BBQ-Race contrast set as
# llada_moe/build_arrows.py); nothing is reused from the LLaDA or Dream caches.
#
# Artifacts (verified against the llada_moe fit functions):
#   calib.py --fit --where block -> llada_moe/baselines/cache/calib_block.pt
#   meanact.py --fit             -> llada_moe/baselines/cache/meanact_meandiff_block.pt
#                                   (reads calib_block.pt; only the 'fitted'
#                                    direction consumes it -- built for parity;
#                                    the jobAD run uses 'unit', which reads
#                                    llada_moe/arrows.pt)
#   actadd.py --fit              -> llada_moe/baselines/cache/actadd_dir.pt
#                                   (already built by round4_jobAB; guard skips)
#   linearact.py --fit           -> llada_moe/baselines/cache/linearact_stats.pt
#                                   (+ calib_mlp_hidden.pt as a side effect;
#                                    NOTE 'mlp_hidden' on this MoE model = the
#                                    2048-d MoE block OUTPUT, not a dense gated
#                                    hidden -- see llada_moe/README.md)
#   aura.py --fit                -> llada_moe/baselines/cache/aura_auroc.pt
#   itic.py --fit                -> llada_moe/baselines/cache/itic_probes.pt
# (caa needs no fit: it reads llada_moe/arrows.pt from round4_jobAB, asserted
#  below.  The aura/itic fits re-collect their own mlp_hidden / attn_head
#  activations internally -- no standalone calib step needed for them.)
# Env: LLaDA-MoE remote code loads under transformers==4.46.2 (the dlm env pin;
# verified by CPU config+tokenizer+meta-weights load).
# Each step is guarded on its llada_moe/baselines/cache/ artifact -> idempotent,
# safe under preempt+requeue.
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

CACHE=llada_moe/baselines/cache

# caa/meanact(unit) prerequisite (built by round4_jobAB moearw): fail loud.
test -f "llada_moe/arrows.pt" || {
    echo "FATAL: llada_moe/arrows.pt missing (run llada_moe/build_arrows.py / round4_jobAB first)"; exit 1; }

# Calibration activation collection (800 clean forwards: 400 items x 2).
run_if_missing "$CACHE/calib_block.pt" \
    $PY llada_moe/baselines/calib.py --fit --where block

# Per-method fit artifacts (each collects its own granularity where needed).
run_if_missing "$CACHE/actadd_dir.pt" \
    $PY llada_moe/baselines/actadd.py --fit
run_if_missing "$CACHE/meanact_meandiff_block.pt" \
    $PY llada_moe/baselines/meanact.py --fit
run_if_missing "$CACHE/linearact_stats.pt" \
    $PY llada_moe/baselines/linearact.py --fit
run_if_missing "$CACHE/aura_auroc.pt" \
    $PY llada_moe/baselines/aura.py --fit
run_if_missing "$CACHE/itic_probes.pt" \
    $PY llada_moe/baselines/itic.py --fit

echo "DONE (moefit: LLaDA-MoE calib block + actadd/meanact/linearact/aura/itic fits)"
