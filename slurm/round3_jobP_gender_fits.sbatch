#!/bin/bash
#SBATCH --job-name=gfits
#SBATCH --account=nils
#SBATCH --partition=gpu
#SBATCH --qos=normal-plus
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --output=/shared/home/sarim.hashmi/dlm_bias/logs/slurm/%x-%j.out
#SBATCH --error=/shared/home/sarim.hashmi/dlm_bias/logs/slurm/%x-%j.err

# ROUND 3 JOB P "gfits" (E8): baseline calibration collections + per-method fit
# artifacts for the GENDER targets woman + man -- the exact JOB 1 (job1_fits)
# pattern with --target, against the E3 gender-manifest heldout sets (400
# contrast items per target, multirace/items_manifest_gender.json).
# The 3 calib collections per target (block / mlp_hidden / attn_head; 800
# forwards each) are the expensive part; per-method fits mirror round 1
# (linearact/aura/itic re-collect their own granularity, as job1 did).
# Each step is guarded on its baselines/cache/ artifact -> idempotent, safe
# under preempt+requeue.
#
# Artifacts per target T in {woman, man}:
#   calib.py --fit --where block --target T      -> cache/calib_block_T.pt
#   calib.py --fit --where mlp_hidden --target T -> cache/calib_mlp_hidden_T.pt
#   calib.py --fit --where attn_head --target T  -> cache/calib_attn_head_T.pt
#   actadd.py --fit --target T                   -> cache/actadd_dir_T.pt
#   meanact.py --fit --target T                  -> cache/meanact_meandiff_block_T.pt
#   linearact.py --fit --target T                -> cache/linearact_stats_T.pt
#   aura.py --fit --target T                     -> cache/aura_auroc_T.pt
#   itic.py --fit --target T                     -> cache/itic_probes_T.pt
# (caa has no per-method fit: it reuses multirace/arrows_T.pt from JOB G,
#  asserted present below.)
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

for T in woman man; do
    # caa prerequisite (built by round3_jobG_gender_arrows): fail loud if absent.
    test -f "multirace/arrows_${T}.pt" || {
        echo "FATAL: multirace/arrows_${T}.pt missing (run jobG first)"; exit 1; }

    # Calibration activation collections (800 forwards each: 400 items x 2).
    run_if_missing "$CACHE/calib_block_${T}.pt" \
        $PY baselines/calib.py --fit --where block --target "$T"
    run_if_missing "$CACHE/calib_mlp_hidden_${T}.pt" \
        $PY baselines/calib.py --fit --where mlp_hidden --target "$T"
    run_if_missing "$CACHE/calib_attn_head_${T}.pt" \
        $PY baselines/calib.py --fit --where attn_head --target "$T"

    # Per-method fit artifacts.
    run_if_missing "$CACHE/actadd_dir_${T}.pt" \
        $PY baselines/actadd.py --fit --target "$T"
    run_if_missing "$CACHE/meanact_meandiff_block_${T}.pt" \
        $PY baselines/meanact.py --fit --target "$T"
    run_if_missing "$CACHE/linearact_stats_${T}.pt" \
        $PY baselines/linearact.py --fit --target "$T"
    run_if_missing "$CACHE/aura_auroc_${T}.pt" \
        $PY baselines/aura.py --fit --target "$T"
    run_if_missing "$CACHE/itic_probes_${T}.pt" \
        $PY baselines/itic.py --fit --target "$T"
done

echo "DONE (gfits: woman+man calib collections + method fits)"
