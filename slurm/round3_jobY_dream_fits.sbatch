#!/bin/bash
#SBATCH --job-name=dfits
#SBATCH --account=nils
#SBATCH --partition=gpu
#SBATCH --qos=normal-plus
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --output=/shared/home/sarim.hashmi/dlm_bias/logs/slurm/%x-%j.out
#SBATCH --error=/shared/home/sarim.hashmi/dlm_bias/logs/slurm/%x-%j.err

# ROUND 3 JOB Y "dfits" (Dream baseline parity): calibration collection +
# per-method fit artifacts for the Dream-v0-Instruct-7B baseline suite -- the
# exact JOB 1 (job1_fits) pattern on the DREAM port (dream/baselines/*).
# All fits are built from DREAM activations (dream/baselines/calib.py collects
# on the same contamination-safe held-out BBQ-Race contrast set as
# dream/build_arrows.py); nothing is reused from the LLaDA caches.
#
# Artifacts (verified against the dream fit functions):
#   calib.py --fit --where block -> dream/baselines/cache/calib_block.pt
#   meanact.py --fit             -> dream/baselines/cache/meanact_meandiff_block.pt
#                                   (reads calib_block.pt; only the 'fitted'
#                                    direction consumes it -- built for parity
#                                    with job1 so the fitted variant is
#                                    reproducible; the jobZ run uses 'unit',
#                                    which reads dream/arrows.pt)
#   actadd.py --fit              -> dream/baselines/cache/actadd_dir.pt
#                                   (already built by round2_jobF; guard skips)
#   linearact.py --fit           -> dream/baselines/cache/linearact_stats.pt
#                                   (+ calib_mlp_hidden.pt as a side effect)
#   aura.py --fit                -> dream/baselines/cache/aura_auroc.pt
#   itic.py --fit                -> dream/baselines/cache/itic_probes.pt
# (caa needs no fit: it reads dream/arrows.pt from round2_jobA, asserted below.
#  NOTE the dream aura/itic fits re-collect their own mlp_hidden / attn_head
#  activations internally, matching dream/run_dream_gpu4.sh -- no standalone
#  calib --where mlp_hidden/attn_head step is needed.)
# Env: Dream remote code needs transformers==4.46.2 (the dlm env pin).
# Each step is guarded on its dream/baselines/cache/ artifact -> idempotent,
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

CACHE=dream/baselines/cache

# caa/meanact(unit) prerequisite (built by round2_jobA arrows2): fail loud.
test -f "dream/arrows.pt" || {
    echo "FATAL: dream/arrows.pt missing (run dream/build_arrows.py / round2_jobA first)"; exit 1; }

# Calibration activation collection (800 clean forwards: 400 items x 2).
run_if_missing "$CACHE/calib_block.pt" \
    $PY dream/baselines/calib.py --fit --where block

# Per-method fit artifacts (each collects its own granularity where needed).
run_if_missing "$CACHE/actadd_dir.pt" \
    $PY dream/baselines/actadd.py --fit
run_if_missing "$CACHE/meanact_meandiff_block.pt" \
    $PY dream/baselines/meanact.py --fit
run_if_missing "$CACHE/linearact_stats.pt" \
    $PY dream/baselines/linearact.py --fit
run_if_missing "$CACHE/aura_auroc.pt" \
    $PY dream/baselines/aura.py --fit
run_if_missing "$CACHE/itic_probes.pt" \
    $PY dream/baselines/itic.py --fit

echo "DONE (dfits: Dream calib block + actadd/meanact/linearact/aura/itic fits)"
