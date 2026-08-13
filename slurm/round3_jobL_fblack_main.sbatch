#!/bin/bash
#SBATCH --job-name=fbmain
#SBATCH --account=nils
#SBATCH --partition=gpu
#SBATCH --qos=normal-plus
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=/shared/home/sarim.hashmi/dlm_bias/logs/slurm/%x-%j.out
#SBATCH --error=/shared/home/sarim.hashmi/dlm_bias/logs/slurm/%x-%j.err

# ROUND 3 JOB L "fbmain" (afterok fbdose; submit AFTER reviewing jobK's dose
# sweep): E6 full position-balanced protocol, target FBLACK (Black women --
# a GENDER-CONDITIONED RACE direction, f-black vs other-race women, NOT a
# full intersectional contrast; compare its strict gap to the coarse black
# direction's pooled strict gaps (base +0.018, decode-PI +0.167) via
# balanced_all/strict_round2.py family 6).
#   1. rotations (CPU, deterministic, guarded) ->
#      results/balanced_all/rotations/_sweep400_fblack_rot{0,1,2}.jsonl
#   2. balanced_all/oracle_test.py --target fblack  (HARD FAIL via set -e)
#   3. per rot{0,1,2}: base, decode-PI (amax $FBLACK_AMAX), normal (alpha
#      $FBLACK_ALPHA), caa (alpha 2, layer 14)
#      -> results/balanced_all/fblack/{base,decode_pid,normal,caa}/rot{r}/
#      tags: cond_dpid_fblack_base / cond_dpid_fblack_PI /
#            cond_normal_fblack_a$FBLACK_ALPHA / cond_caa_fblack_a2
#      (strict_round2.py family 6 expects exactly these stems).
#
# !!! DOSES: FBLACK_AMAX / FBLACK_ALPHA below are PLACEHOLDER defaults (6 / 4,
# !!! the race-experiment defaults). The jobK dose sweep MUST be inspected and
# !!! these MUST be exported at submit time, e.g.
# !!!     sbatch --export=ALL,FBLACK_AMAX=4,FBLACK_ALPHA=2 slurm/round3_jobL_fblack_main.sbatch
# !!! Note the decode-PI marker/tag does NOT encode amax: rerunning with a
# !!! different FBLACK_AMAX after a partial run will SKIP completed rotations --
# !!! delete results/balanced_all/fblack/decode_pid (and normal at a changed
# !!! alpha) first if the dose changes.
# !!! FBLACK_ALPHA must be an INTEGER: normal.py's default output stem renders
# !!! alpha via %g with '.'->'p' (2.5 -> cond_normal_fblack_a2p5.json) while the
# !!! run_if_missing marker below interpolates the env value verbatim
# !!! (cond_normal_fblack_a2.5.json) -- a fractional alpha breaks the skip-guard
# !!! and reruns all 3 normal rotations on every requeue.
# 14 runs (12 GPU evals + rotations + oracle), idempotent via run_if_missing.
set -euo pipefail

export HF_HOME=/shared/models/huggingface
export HF_HUB_CACHE=/shared/models/huggingface/hub
export HF_MODULES_CACHE=/shared/home/sarim.hashmi/.cache/hf_modules
export DLM_BIAS_ROOT=/shared/home/sarim.hashmi/dlm_bias
cd "$DLM_BIAS_ROOT"
PY=/shared/home/sarim.hashmi/anaconda3/envs/dlm/bin/python
# Do NOT set CUDA_VISIBLE_DEVICES -- SLURM sets it.

T=fblack
FBLACK_AMAX="${FBLACK_AMAX:-6}"    # PLACEHOLDER -- set from jobK dose sweep!
FBLACK_ALPHA="${FBLACK_ALPHA:-4}"  # PLACEHOLDER -- set from jobK dose sweep!
echo "DOSES: amax=${FBLACK_AMAX} alpha=${FBLACK_ALPHA} (verify against jobK!)"

OUT=results/balanced_all
ROTDIR=$OUT/rotations

run_if_missing() {  # $1 = expected result json, rest = command
    local marker=$1; shift
    if [[ -f "$marker" ]]; then
        echo "SKIP (exists): $marker"
    else
        echo "RUN: $*"
        "$@"
    fi
}

# ---- rotations (CPU, deterministic) + oracle proof (hard fail) ------------- #
run_if_missing "$ROTDIR/_sweep400_${T}_rot2.jsonl" \
    $PY balanced_all/make_rotations.py \
        --items "data/bbq_items/_sweep400_${T}.jsonl" --out-dir "$ROTDIR"
$PY balanced_all/oracle_test.py \
    --items-glob "$ROTDIR/_sweep400_${T}_rot*.jsonl" \
    --target "$T" --orig "data/bbq_items/_sweep400_${T}.jsonl"

# ---- base, decode-PI, normal, caa x rot{0,1,2} ------------------------------ #
for r in 0 1 2; do
    ITEMS=$ROTDIR/_sweep400_${T}_rot${r}.jsonl

    run_if_missing "$OUT/$T/base/rot${r}/cond_dpid_${T}_base.json" \
        $PY multirace/denoise_pid.py --target "$T" --cond base \
            --items "$ITEMS" --out-dir "$OUT/$T/base/rot${r}"

    run_if_missing "$OUT/$T/decode_pid/rot${r}/cond_dpid_${T}_PI.json" \
        $PY multirace/denoise_pid.py --target "$T" --cond PI --amax "$FBLACK_AMAX" \
            --items "$ITEMS" --out-dir "$OUT/$T/decode_pid/rot${r}"

    run_if_missing "$OUT/$T/normal/rot${r}/cond_normal_${T}_a${FBLACK_ALPHA}.json" \
        $PY multirace/normal.py --target "$T" --alpha "$FBLACK_ALPHA" \
            --items "$ITEMS" --out-dir "$OUT/$T/normal/rot${r}"

    run_if_missing "$OUT/$T/caa/rot${r}/cond_caa_${T}_a2.json" \
        $PY multirace/caa.py --target "$T" --alpha 2 \
            --items "$ITEMS" --out-dir "$OUT/$T/caa/rot${r}"
done

echo "ALL DONE (fbmain: fblack base/decode-PI(amax${FBLACK_AMAX})/normal(a${FBLACK_ALPHA})/caa(a2) x rot0-2)"
