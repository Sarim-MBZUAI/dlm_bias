#!/bin/bash
#SBATCH --job-name=agemain
#SBATCH --account=nils
#SBATCH --partition=gpu
#SBATCH --qos=normal-plus
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=/shared/home/sarim.hashmi/dlm_bias/logs/slurm/%x-%j.out
#SBATCH --error=/shared/home/sarim.hashmi/dlm_bias/logs/slurm/%x-%j.err

# ROUND 3 JOB V "agemain" (afterok saarrows; submit AFTER reviewing jobT's
# dose sweep): E9 full position-balanced protocol, AGE axis -- BOTH poles
# old and young in one job. Per target T:
#   1. rotations (CPU, deterministic, guarded) ->
#      results/balanced_all/rotations/_sweep400_<T>_rot{0,1,2}.jsonl
#   2. balanced_all/oracle_test.py --target <T>  (HARD FAIL via set -e)
#   3. per rot{0,1,2}: base, decode-PI (amax $AGE_AMAX), normal (alpha
#      $AGE_ALPHA), caa (alpha 2, layer 14)
#      -> results/balanced_all/<T>/{base,decode_pid,normal,caa}/rot{r}/
#      tags: cond_dpid_<T>_base / cond_dpid_<T>_PI /
#            cond_normal_<T>_a$AGE_ALPHA / cond_caa_<T>_a2
#      (strict_round2.py family 8 discovers exactly these stems).
#
# POLE-MIRROR DOSE INHERITANCE (accepted design): jobT sweeps OLD only;
# young INHERITS the same $AGE_AMAX/$AGE_ALPHA. See jobT header for the
# rationale and the assumption it carries.
#
# !!! DOSES: AGE_AMAX / AGE_ALPHA below are PLACEHOLDER defaults (4 / 4).
# !!! The jobT dose sweep MUST be inspected and these MUST be exported at
# !!! submit time, e.g.
# !!!     sbatch --export=ALL,AGE_AMAX=4,AGE_ALPHA=2 slurm/round3_jobV_age_main.sbatch
# !!! Note the decode-PI marker/tag does NOT encode amax: rerunning with a
# !!! different AGE_AMAX after a partial run will SKIP completed rotations --
# !!! delete results/balanced_all/{old,young}/decode_pid (and normal at a
# !!! changed alpha) first if the dose changes.
# !!! AGE_AMAX and AGE_ALPHA must be INTEGERS: normal.py's default output stem
# !!! renders alpha via %g with '.'->'p' (2.5 -> cond_normal_<T>_a2p5.json)
# !!! while the run_if_missing marker below interpolates the env value
# !!! verbatim (cond_normal_<T>_a2.5.json) -- a fractional alpha breaks the
# !!! skip-guard and reruns all normal rotations on every requeue.
# 2 targets x (rotations + oracle + 12 GPU evals) = 24 GPU evals, idempotent
# via run_if_missing.
set -euo pipefail

export HF_HOME=/shared/models/huggingface
export HF_HUB_CACHE=/shared/models/huggingface/hub
export HF_MODULES_CACHE=/shared/home/sarim.hashmi/.cache/hf_modules
export DLM_BIAS_ROOT=/shared/home/sarim.hashmi/dlm_bias
cd "$DLM_BIAS_ROOT"
PY=/shared/home/sarim.hashmi/anaconda3/envs/dlm/bin/python
# Do NOT set CUDA_VISIBLE_DEVICES -- SLURM sets it.

AGE_AMAX="${AGE_AMAX:-4}"    # PLACEHOLDER -- set from jobT dose sweep!
AGE_ALPHA="${AGE_ALPHA:-4}"  # PLACEHOLDER -- set from jobT dose sweep!
echo "DOSES: amax=${AGE_AMAX} alpha=${AGE_ALPHA} (verify against jobT!)"
echo "       young inherits the old-swept dose (pole-mirror design)"
if ! [[ "$AGE_AMAX" =~ ^[0-9]+$ && "$AGE_ALPHA" =~ ^[0-9]+$ ]]; then
    echo "!!! AGE_AMAX/AGE_ALPHA must be integers (skip-guard stems) -- got" \
         "amax=${AGE_AMAX} alpha=${AGE_ALPHA}" >&2
    exit 1
fi

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

for T in old young; do
    # ---- rotations (CPU, deterministic) + oracle proof (hard fail) --------- #
    run_if_missing "$ROTDIR/_sweep400_${T}_rot2.jsonl" \
        $PY balanced_all/make_rotations.py \
            --items "data/bbq_items/_sweep400_${T}.jsonl" --out-dir "$ROTDIR"
    $PY balanced_all/oracle_test.py \
        --items-glob "$ROTDIR/_sweep400_${T}_rot*.jsonl" \
        --target "$T" --orig "data/bbq_items/_sweep400_${T}.jsonl"

    # ---- base, decode-PI, normal, caa x rot{0,1,2} -------------------------- #
    for r in 0 1 2; do
        ITEMS=$ROTDIR/_sweep400_${T}_rot${r}.jsonl

        run_if_missing "$OUT/$T/base/rot${r}/cond_dpid_${T}_base.json" \
            $PY multirace/denoise_pid.py --target "$T" --cond base \
                --items "$ITEMS" --out-dir "$OUT/$T/base/rot${r}"

        run_if_missing "$OUT/$T/decode_pid/rot${r}/cond_dpid_${T}_PI.json" \
            $PY multirace/denoise_pid.py --target "$T" --cond PI --amax "$AGE_AMAX" \
                --items "$ITEMS" --out-dir "$OUT/$T/decode_pid/rot${r}"

        run_if_missing "$OUT/$T/normal/rot${r}/cond_normal_${T}_a${AGE_ALPHA}.json" \
            $PY multirace/normal.py --target "$T" --alpha "$AGE_ALPHA" \
                --items "$ITEMS" --out-dir "$OUT/$T/normal/rot${r}"

        run_if_missing "$OUT/$T/caa/rot${r}/cond_caa_${T}_a2.json" \
            $PY multirace/caa.py --target "$T" --alpha 2 \
                --items "$ITEMS" --out-dir "$OUT/$T/caa/rot${r}"
    done
done

echo "ALL DONE (agemain: old+young base/decode-PI(amax${AGE_AMAX})/normal(a${AGE_ALPHA})/caa(a2) x rot0-2)"
