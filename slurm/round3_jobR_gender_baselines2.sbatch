#!/bin/bash
#SBATCH --job-name=gbalB
#SBATCH --account=nils
#SBATCH --partition=gpu
#SBATCH --qos=normal-plus
#SBATCH --gres=gpu:1
#SBATCH --time=20:00:00
#SBATCH --output=/shared/home/sarim.hashmi/dlm_bias/logs/slurm/%x-%j.out
#SBATCH --error=/shared/home/sarim.hashmi/dlm_bias/logs/slurm/%x-%j.err

# ROUND 3 JOB R "gbalB" (E8, afterok gfits): position-balanced GENDER-target
# prior-work baselines, second half -- the exact JOB 3 (job3_balB) pattern with
# --target, at the round-1 published operating points (results/BASELINES.md).
# Methods aura(inject g4) / aura(vanilla) / itic(K48 a8) x targets woman,man
# x rot0-2 = 18 runs (~12 min each on 1 GPU).
# Output layout + stems replicate results/balanced_all/black EXACTLY (per
# target), so balanced_all/strict_round2.py's gender family (BASELINE_CONDS)
# discovers them unchanged. Idempotent per run via run_if_missing.
set -euo pipefail

export HF_HOME=/shared/models/huggingface
export HF_HUB_CACHE=/shared/models/huggingface/hub
export HF_MODULES_CACHE=/shared/home/sarim.hashmi/.cache/hf_modules
export DLM_BIAS_ROOT=/shared/home/sarim.hashmi/dlm_bias
cd "$DLM_BIAS_ROOT"
PY=/shared/home/sarim.hashmi/anaconda3/envs/dlm/bin/python
# Do NOT set CUDA_VISIBLE_DEVICES -- SLURM sets it.

ROTDIR=results/balanced_all/rotations

run_if_missing() {  # $1 = expected result json, rest = command
    local marker=$1; shift
    if [[ -f "$marker" ]]; then
        echo "SKIP (exists): $marker"
    else
        echo "RUN: $*"
        "$@"
    fi
}

for T in woman man; do
    OUT=results/balanced_all/$T
    for r in 0 1 2; do
        ITEMS=$ROTDIR/_sweep400_${T}_rot${r}.jsonl
        test -f "$ITEMS" || {
            echo "FATAL: $ITEMS missing (jobI builds the gender rotations)"; exit 1; }

        run_if_missing "$OUT/aura_inject/rot${r}/cond_inject_g4.json" \
            $PY baselines/aura.py --run --mode inject --gamma 4 --target "$T" \
                --items "$ITEMS" --out-dir "$OUT/aura_inject/rot${r}"

        run_if_missing "$OUT/aura_vanilla/rot${r}/cond_vanilla.json" \
            $PY baselines/aura.py --run --mode vanilla --target "$T" \
                --items "$ITEMS" --out-dir "$OUT/aura_vanilla/rot${r}"

        run_if_missing "$OUT/itic/rot${r}/cond_itic_K48_a8.json" \
            $PY baselines/itic.py --run --topk 48 --alpha 8 --target "$T" \
                --items "$ITEMS" --out_dir "$OUT/itic/rot${r}"
    done
done

echo "ALL DONE (gbalB: aura_inject/aura_vanilla/itic x woman,man x rot0-2)"
