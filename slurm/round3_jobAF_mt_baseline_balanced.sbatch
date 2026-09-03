#!/bin/bash
#SBATCH --job-name=mtbbal
#SBATCH --account=nils
#SBATCH --partition=gpu
#SBATCH --qos=normal-plus
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=/shared/home/sarim.hashmi/dlm_bias/logs/slurm/%x-%j.out
#SBATCH --error=/shared/home/sarim.hashmi/dlm_bias/logs/slurm/%x-%j.err

# ROUND 3 JOB AF "mtbbal" (E9, afterok mtbfits): position-balanced prior-work
# baseline suite for the E9 axis targets arab (race), lowses (SES), old (Age)
# -- the EXACT JOB Q + JOB R (round3_jobQ/jobR gender baselines) pattern with
# --target, at the round-1 PUBLISHED operating points (results/BASELINES.md),
# so every new row is protocol-matched to Table 1 (results/balanced_all/black).
#
# Methods x targets x rot0-2:
#   caa L14 a16 / actadd a16 / meanact unit s2 / linearact gaussian s1 /
#   aura inject g4 / aura vanilla / itic K48 a8   (= the 7 BASELINE_CONDS)
#   x {arab, lowses, old} x {0,1,2}  = 7 x 3 x 3 = 63 runs (~12 min each).
#
# CAA NOTE: this writes the Table-1 caa stem cond_caa_L14_a16.json (baselines/
# caa.py, arrows_<T>.pt at layer 14 alpha 16) for ALL THREE targets.  lowses
# and old ALREADY have a DIFFERENT caa condition on disk -- cond_caa_<T>_a2.json
# from the E9 main job (multirace/caa.py, alpha 2), the E9-family "caa a2" row.
# The two coexist exactly as they do for woman/man (both cond_caa_woman_a2 and
# cond_caa_L14_a16 are committed).  The L14 a16 stem is the one Table 1 and
# balanced_all/strict_round2.py's BASELINE_CONDS discover, so it is required
# here for the Table-1 suite to be complete on arab/lowses/old.
#
# Output layout + stems replicate results/balanced_all/black EXACTLY (per
# target), so strict_round2.py discovers them: arab via family 1
# (family_multitarget soft baseline block), lowses via family 7, old via
# family 8 (both family_soft_targets, soft_baselines=True).  Idempotent per run
# via run_if_missing; rotations are the SAME item files the existing arab/
# lowses/old base+decode-PI+normal runs used.
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

for T in arab lowses old; do
    OUT=results/balanced_all/$T
    for r in 0 1 2; do
        ITEMS=$ROTDIR/_sweep400_${T}_rot${r}.jsonl
        test -f "$ITEMS" || {
            echo "FATAL: $ITEMS missing (E9 main job builds these rotations)"; exit 1; }

        # ---- gbalA half: caa / actadd / meanact / linearact ---------------- #
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

        # ---- gbalB half: aura(inject g4) / aura(vanilla) / itic(K48 a8) ---- #
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

echo "ALL DONE (mtbbal: caa/actadd/meanact/linearact/aura_inject/aura_vanilla/itic x arab,lowses,old x rot0-2)"
