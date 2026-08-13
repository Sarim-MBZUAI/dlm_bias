#!/bin/bash
#SBATCH --job-name=gmainw
#SBATCH --account=nils
#SBATCH --partition=gpu
#SBATCH --qos=normal-plus
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=/shared/home/sarim.hashmi/dlm_bias/logs/slurm/%x-%j.out
#SBATCH --error=/shared/home/sarim.hashmi/dlm_bias/logs/slurm/%x-%j.err

# ROUND 3 JOB I "gmainw" (afterok garrows; submit AFTER reviewing jobH's dose
# sweep): E3 full position-balanced protocol, target WOMAN.
#   1. rotations (CPU, deterministic, guarded) ->
#      results/balanced_all/rotations/_sweep400_woman_rot{0,1,2}.jsonl
#   2. balanced_all/oracle_test.py --target woman  (HARD FAIL via set -e)
#   3. per rot{0,1,2}: base, decode-PI (amax $WOMAN_AMAX), normal (alpha
#      $WOMAN_ALPHA), caa (alpha 2, layer 14)
#      -> results/balanced_all/woman/{base,decode_pid,normal,caa}/rot{r}/
#      tags: cond_dpid_woman_base / cond_dpid_woman_PI /
#            cond_normal_woman_a$WOMAN_ALPHA / cond_caa_woman_a2
#      (strict_round2.py gender family expects exactly these stems).
#
# !!! DOSES: WOMAN_AMAX / WOMAN_ALPHA below are PLACEHOLDER defaults (6 / 4,
# !!! the race-experiment defaults). The jobH dose sweep MUST be inspected and
# !!! these MUST be exported at submit time, e.g.
# !!!     sbatch --export=ALL,WOMAN_AMAX=4,WOMAN_ALPHA=2 slurm/round3_jobI_gender_main.sbatch
# !!! Note the decode-PI marker/tag does NOT encode amax: rerunning with a
# !!! different WOMAN_AMAX after a partial run will SKIP completed rotations --
# !!! delete results/balanced_all/woman/decode_pid (and normal at a changed
# !!! alpha) first if the dose changes.
# 14 runs (12 GPU evals + rotations + oracle), idempotent via run_if_missing.
set -euo pipefail

export HF_HOME=/shared/models/huggingface
export HF_HUB_CACHE=/shared/models/huggingface/hub
export HF_MODULES_CACHE=/shared/home/sarim.hashmi/.cache/hf_modules
export DLM_BIAS_ROOT=/shared/home/sarim.hashmi/dlm_bias
cd "$DLM_BIAS_ROOT"
PY=/shared/home/sarim.hashmi/anaconda3/envs/dlm/bin/python
# Do NOT set CUDA_VISIBLE_DEVICES -- SLURM sets it.

T=woman
WOMAN_AMAX="${WOMAN_AMAX:-6}"    # PLACEHOLDER -- set from jobH dose sweep!
WOMAN_ALPHA="${WOMAN_ALPHA:-4}"  # PLACEHOLDER -- set from jobH dose sweep!
echo "DOSES: amax=${WOMAN_AMAX} alpha=${WOMAN_ALPHA} (verify against jobH!)"

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
        $PY multirace/denoise_pid.py --target "$T" --cond PI --amax "$WOMAN_AMAX" \
            --items "$ITEMS" --out-dir "$OUT/$T/decode_pid/rot${r}"

    run_if_missing "$OUT/$T/normal/rot${r}/cond_normal_${T}_a${WOMAN_ALPHA}.json" \
        $PY multirace/normal.py --target "$T" --alpha "$WOMAN_ALPHA" \
            --items "$ITEMS" --out-dir "$OUT/$T/normal/rot${r}"

    run_if_missing "$OUT/$T/caa/rot${r}/cond_caa_${T}_a2.json" \
        $PY multirace/caa.py --target "$T" --alpha 2 \
            --items "$ITEMS" --out-dir "$OUT/$T/caa/rot${r}"
done

echo "ALL DONE (gmainw: woman base/decode-PI(amax${WOMAN_AMAX})/normal(a${WOMAN_ALPHA})/caa(a2) x rot0-2)"
