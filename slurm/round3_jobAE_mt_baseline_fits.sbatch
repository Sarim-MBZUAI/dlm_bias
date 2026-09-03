#!/bin/bash
#SBATCH --job-name=mtbfits
#SBATCH --account=nils
#SBATCH --partition=gpu
#SBATCH --qos=normal-plus
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=/shared/home/sarim.hashmi/dlm_bias/logs/slurm/%x-%j.out
#SBATCH --error=/shared/home/sarim.hashmi/dlm_bias/logs/slurm/%x-%j.err

# ROUND 3 JOB AE "mtbfits" (E9): baseline calibration collections + per-method
# fit artifacts for the E9 axis targets arab (race), lowses (SES) and old
# (Age) -- the EXACT JOB P (round3_jobP_gender_fits) pattern with --target,
# against each target's manifest heldout set (400 contrast items, resolved by
# multirace/build_arrows.heldout_from_manifest via calib.heldout_items; the
# SAME triples the arrows were fitted from).  Extends the gender fit suite to
# the three multi-target tables so they get the full Table-1 (Black) baseline
# method suite (slurm/round3_jobAF runs the balanced evals afterok this job).
#
# Prereq (built earlier): multirace/arrows_<T>.pt for T in {arab,lowses,old}
# (asserted present below -- caa has NO per-method fit, it reuses the arrows).
# baselines/common.py SUPPORTED_TARGETS was extended to include these targets;
# baselines/calib.py SOURCE_BY_TARGET carries their (seed-42) source strings.
#
# The 3 calib collections per target (block / mlp_hidden / attn_head; 800
# forwards each) are the expensive part; per-method fits mirror jobP exactly.
# Each step is guarded on its baselines/cache/ artifact -> idempotent, safe
# under preempt+requeue.
#
# Artifacts per target T in {arab, lowses, old}:
#   calib.py --fit --where block --target T      -> cache/calib_block_T.pt
#   calib.py --fit --where mlp_hidden --target T -> cache/calib_mlp_hidden_T.pt
#   calib.py --fit --where attn_head --target T  -> cache/calib_attn_head_T.pt
#   actadd.py --fit --target T                   -> cache/actadd_dir_T.pt
#   meanact.py --fit --target T                  -> cache/meanact_meandiff_block_T.pt
#   linearact.py --fit --target T                -> cache/linearact_stats_T.pt
#   aura.py --fit --target T                     -> cache/aura_auroc_T.pt
#   itic.py --fit --target T                     -> cache/itic_probes_T.pt
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

for T in arab old; do
    # caa prerequisite (built by multirace/build_arrows.py): fail loud if absent.
    test -f "multirace/arrows_${T}.pt" || {
        echo "FATAL: multirace/arrows_${T}.pt missing (build_arrows --target $T first)"; exit 1; }

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

echo "DONE (mtbfits: arab+lowses+old calib collections + method fits)"
