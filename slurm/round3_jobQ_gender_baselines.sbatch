#!/bin/bash
#SBATCH --job-name=gbalA
#SBATCH --account=nils
#SBATCH --partition=gpu
#SBATCH --qos=normal-plus
#SBATCH --gres=gpu:1
#SBATCH --time=20:00:00
#SBATCH --output=/shared/home/sarim.hashmi/dlm_bias/logs/slurm/%x-%j.out
#SBATCH --error=/shared/home/sarim.hashmi/dlm_bias/logs/slurm/%x-%j.err

# ROUND 3 JOB Q "gbalA" (E8, afterok gfits): position-balanced GENDER-target
# prior-work baselines, first half -- the exact JOB 2 (job2_balA) pattern with
# --target, at the round-1 published operating points (results/BASELINES.md).
# Methods caa / actadd / meanact / linearact x targets woman,man x rot0-2
# = 24 runs (~12 min each on 1 GPU).
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

        run_if_missing "$OUT/caa/rot${r}/cond_caa_L14_a16.json" \
            $PY baselines/caa.py --run --alpha 16 --layer 14 --target "$T" \
                --items "$ITEMS" --out_dir "$OUT/caa/rot${r}"

        run_if_missing "$OUT/actadd/rot${r}/cond_actadd_a16.json" \
            $PY baselines/actadd.py --run --alpha 16 --layer 14 --target "$T" \
                --items "$ITEMS" --out-dir "$OUT/actadd/rot${r}"

        run_if_missing "$OUT/meanact/rot${r}/cond_meanact_unit_s2.json" \
            $PY baselines/meanact.py --run --direction unit --strength 2 --target "$T" \
                --items "$ITEMS" --out-dir "$OUT/meanact/rot${r}"

        run_if_missing "$OUT/linearact/rot${r}/cond_gaussian_s1.json" \
            $PY baselines/linearact.py --run --variant gaussian --strength 1 --target "$T" \
                --items "$ITEMS" --out_dir "$OUT/linearact/rot${r}"
    done
done

echo "ALL DONE (gbalA: caa/actadd/meanact/linearact x woman,man x rot0-2)"
