#!/bin/bash
#SBATCH --job-name=moebal
#SBATCH --account=nils
#SBATCH --partition=gpu
#SBATCH --qos=normal-plus
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=/shared/home/sarim.hashmi/dlm_bias/logs/slurm/%x-%j.out
#SBATCH --error=/shared/home/sarim.hashmi/dlm_bias/logs/slurm/%x-%j.err

# ROUND 4 JOB AD "moebal" (afterok moefit): the FULL 9-condition LLaDA-MoE table
# through the position-balanced eval, SAME LLaDA Black rotation item files
# results/balanced/_sweep400_rot{r}.jsonl the LLaDA/Dream families used --
# the round2_jobF + round3_jobZ pattern merged into one job:
#   base            llada_moe/denoise_pid.py --cond base -> cond_dpid_base
#   decode-PI ours  --cond PI --amax $MOE_AMAX           -> cond_dpid_PI
#   caa             --alpha $MOE_CAA_ALPHA --layer 8     -> cond_caa_L8_a<A>
#   actadd          --alpha $MOE_ACTADD_ALPHA --layer 8  -> cond_actadd_a<A>
#   meanact         --direction unit --strength 2        -> cond_meanact_unit_s2
#   linearact       --variant gaussian --strength 1      -> cond_gaussian_s1
#   aura inject     --gamma 4                            -> cond_inject_g4
#   aura vanilla                                         -> cond_vanilla
#   itic            --topk 48 --alpha 8                  -> cond_itic_K48_a8
# = 9 conditions x rot0-2 = 27 runs -> results/lladamoe_balanced/<cond>/rot{r}.
# DOSES: decode-PI amax + caa/actadd alphas default to the conservative
# Dream-parity picks (amax 1.0, caa a2, actadd a8) and are OVERRIDABLE at
# submit time from the round4_jobAB sweep:
#     sbatch --export=ALL,MOE_AMAX=2,MOE_CAA_ALPHA=4,MOE_ACTADD_ALPHA=8 ...
# The fixed-point suite (meanact s2 / gaussian s1 / inject g4 / vanilla /
# itic K48 a8) replicates the LLaDA balanced operating points + stems EXACTLY
# (strict_round2.BASELINE_CONDS), so balanced_all/strict_round2.py's LLaDA-MoE
# family discovers them unchanged (glob discovery covers the dosed stems).
# Directions/gates come from the MoE fits of round4_jobAC -- never from the
# LLaDA or Dream artifacts.  If a condition collapses to unparseable, the
# strict table reports it as invalid (that IS the parity result, not a
# job failure).  Idempotent per run via run_if_missing.
set -euo pipefail

export HF_HOME=/shared/models/huggingface
export HF_HUB_CACHE=/shared/models/huggingface/hub
export HF_MODULES_CACHE=/shared/home/sarim.hashmi/.cache/hf_modules
export DLM_BIAS_ROOT=/shared/home/sarim.hashmi/dlm_bias
cd "$DLM_BIAS_ROOT"
PY=/shared/home/sarim.hashmi/anaconda3/envs/dlm/bin/python
# Do NOT set CUDA_VISIBLE_DEVICES -- SLURM sets it.

# Calibrated doses (submit-time overridable; defaults = conservative parity picks).
MOE_AMAX=${MOE_AMAX:-1.0}
MOE_CAA_ALPHA=${MOE_CAA_ALPHA:-2}
MOE_ACTADD_ALPHA=${MOE_ACTADD_ALPHA:-8}

OUT=results/lladamoe_balanced
CACHE=llada_moe/baselines/cache

run_if_missing() {  # $1 = expected result json, rest = command
    local marker=$1; shift
    if [[ -f "$marker" ]]; then
        echo "SKIP (exists): $marker"
    else
        echo "RUN: $*"
        "$@"
    fi
}

# Fit prerequisites (built by round4_jobAB/jobAC): fail loud if absent.
for f in llada_moe/arrows.pt "$CACHE/actadd_dir.pt" "$CACHE/linearact_stats.pt" \
         "$CACHE/aura_auroc.pt" "$CACHE/itic_probes.pt"; do
    test -f "$f" || { echo "FATAL: $f missing (run round4_jobAB + jobAC first)"; exit 1; }
done

echo "doses: MOE_AMAX=$MOE_AMAX MOE_CAA_ALPHA=$MOE_CAA_ALPHA MOE_ACTADD_ALPHA=$MOE_ACTADD_ALPHA"

for r in 0 1 2; do
    ITEMS=results/balanced/_sweep400_rot${r}.jsonl
    test -f "$ITEMS" || { echo "FATAL: $ITEMS missing"; exit 1; }

    run_if_missing "$OUT/base/rot${r}/cond_dpid_base.json" \
        $PY llada_moe/denoise_pid.py --cond base \
            --items "$ITEMS" --out-dir "$OUT/base/rot${r}"

    run_if_missing "$OUT/decode_pid/rot${r}/cond_dpid_PI.json" \
        $PY llada_moe/denoise_pid.py --cond PI --amax "$MOE_AMAX" \
            --items "$ITEMS" --out-dir "$OUT/decode_pid/rot${r}"

    run_if_missing "$OUT/caa/rot${r}/cond_caa_L8_a${MOE_CAA_ALPHA}.json" \
        $PY llada_moe/baselines/caa.py --run --alpha "$MOE_CAA_ALPHA" --layer 8 \
            --items "$ITEMS" --out_dir "$OUT/caa/rot${r}"

    run_if_missing "$OUT/actadd/rot${r}/cond_actadd_a${MOE_ACTADD_ALPHA}.json" \
        $PY llada_moe/baselines/actadd.py --run --alpha "$MOE_ACTADD_ALPHA" --layer 8 \
            --items "$ITEMS" --out_dir "$OUT/actadd/rot${r}"

    run_if_missing "$OUT/meanact/rot${r}/cond_meanact_unit_s2.json" \
        $PY llada_moe/baselines/meanact.py --run --direction unit --strength 2 \
            --items "$ITEMS" --out_dir "$OUT/meanact/rot${r}"

    run_if_missing "$OUT/linearact/rot${r}/cond_gaussian_s1.json" \
        $PY llada_moe/baselines/linearact.py --run --variant gaussian --strength 1 \
            --items "$ITEMS" --out_dir "$OUT/linearact/rot${r}"

    run_if_missing "$OUT/aura_inject/rot${r}/cond_inject_g4.json" \
        $PY llada_moe/baselines/aura.py --run --mode inject --gamma 4 \
            --items "$ITEMS" --out_dir "$OUT/aura_inject/rot${r}"

    run_if_missing "$OUT/aura_vanilla/rot${r}/cond_vanilla.json" \
        $PY llada_moe/baselines/aura.py --run --mode vanilla \
            --items "$ITEMS" --out_dir "$OUT/aura_vanilla/rot${r}"

    run_if_missing "$OUT/itic/rot${r}/cond_itic_K48_a8.json" \
        $PY llada_moe/baselines/itic.py --run --topk 48 --alpha 8 \
            --items "$ITEMS" --out_dir "$OUT/itic/rot${r}"
done

echo "ALL DONE (moebal: 9 conditions x rot0-2 -> results/lladamoe_balanced)"
