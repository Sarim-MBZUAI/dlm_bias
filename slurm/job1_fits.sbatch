#!/bin/bash
#SBATCH --job-name=fits
#SBATCH --account=nils
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --output=/shared/home/sarim.hashmi/dlm_bias/logs/slurm/%x-%j.out
#SBATCH --error=/shared/home/sarim.hashmi/dlm_bias/logs/slurm/%x-%j.err

# JOB 1 "fits": calibration collections + per-method fit artifacts (LLaDA).
# Each step is guarded on its baselines/cache/ artifact -> idempotent, safe
# under preempt+requeue (normal-plus).
#
# Artifacts (verified against the fit functions):
#   calib.py --fit --where block      -> baselines/cache/calib_block.pt
#   calib.py --fit --where mlp_hidden -> baselines/cache/calib_mlp_hidden.pt
#   calib.py --fit --where attn_head  -> baselines/cache/calib_attn_head.pt
#   actadd.py --fit                   -> baselines/cache/actadd_dir.pt
#   meanact.py --fit                  -> baselines/cache/meanact_meandiff_block.pt
#   linearact.py --fit                -> baselines/cache/linearact_stats.pt
#   aura.py --fit                     -> baselines/cache/aura_auroc.pt
#   itic.py --fit                     -> baselines/cache/itic_probes.pt
# (caa has no per-method fit artifact: it reuses steering/arrows.pt from JOB 0.)
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

CACHE=baselines/cache

# Calibration activation collections (800 forwards each: 400 items x 2).
run_if_missing "$CACHE/calib_block.pt" \
    $PY baselines/calib.py --fit --where block
run_if_missing "$CACHE/calib_mlp_hidden.pt" \
    $PY baselines/calib.py --fit --where mlp_hidden
run_if_missing "$CACHE/calib_attn_head.pt" \
    $PY baselines/calib.py --fit --where attn_head

# Per-method fit artifacts.
run_if_missing "$CACHE/actadd_dir.pt" \
    $PY baselines/actadd.py --fit
run_if_missing "$CACHE/meanact_meandiff_block.pt" \
    $PY baselines/meanact.py --fit
run_if_missing "$CACHE/linearact_stats.pt" \
    $PY baselines/linearact.py --fit
run_if_missing "$CACHE/aura_auroc.pt" \
    $PY baselines/aura.py --fit
run_if_missing "$CACHE/itic_probes.pt" \
    $PY baselines/itic.py --fit

echo "DONE (fits)"
