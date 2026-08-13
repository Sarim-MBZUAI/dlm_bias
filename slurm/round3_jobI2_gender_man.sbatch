#!/bin/bash
#SBATCH --job-name=gmainm
#SBATCH --account=nils
#SBATCH --partition=gpu
#SBATCH --qos=normal-plus
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=/shared/home/sarim.hashmi/dlm_bias/logs/slurm/%x-%j.out
#SBATCH --error=/shared/home/sarim.hashmi/dlm_bias/logs/slurm/%x-%j.err

# ROUND 3 JOB I2 "gmainm" (afterok garrows; submit AFTER reviewing jobH's dose
# sweep): E3 full position-balanced protocol, target MAN. Mirror of
# round3_jobI_gender_main.sbatch -- see that file for the full description.
#
# !!! DOSES: MAN_AMAX / MAN_ALPHA are PLACEHOLDER defaults (6 / 4). The jobH
# !!! dose sweep (run on woman; gender doses are assumed shared unless the
# !!! reviewer decides otherwise) MUST be inspected and these MUST be exported
# !!! at submit time, e.g.
# !!!     sbatch --export=ALL,MAN_AMAX=4,MAN_ALPHA=2 slurm/round3_jobI2_gender_man.sbatch
# !!! The decode-PI marker/tag does NOT encode amax: delete
# !!! results/balanced_all/man/decode_pid before rerunning at a changed dose.
# !!! MAN_ALPHA must be an INTEGER: normal.py's default output stem renders
# !!! alpha via %g with '.'->'p' (2.5 -> cond_normal_man_a2p5.json) while the
# !!! run_if_missing marker below interpolates the env value verbatim
# !!! (cond_normal_man_a2.5.json) -- a fractional alpha breaks the skip-guard
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

T=man
MAN_AMAX="${MAN_AMAX:-6}"    # PLACEHOLDER -- set from jobH dose sweep!
MAN_ALPHA="${MAN_ALPHA:-4}"  # PLACEHOLDER -- set from jobH dose sweep!
echo "DOSES: amax=${MAN_AMAX} alpha=${MAN_ALPHA} (verify against jobH!)"

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
        $PY multirace/denoise_pid.py --target "$T" --cond PI --amax "$MAN_AMAX" \
            --items "$ITEMS" --out-dir "$OUT/$T/decode_pid/rot${r}"

    run_if_missing "$OUT/$T/normal/rot${r}/cond_normal_${T}_a${MAN_ALPHA}.json" \
        $PY multirace/normal.py --target "$T" --alpha "$MAN_ALPHA" \
            --items "$ITEMS" --out-dir "$OUT/$T/normal/rot${r}"

    run_if_missing "$OUT/$T/caa/rot${r}/cond_caa_${T}_a2.json" \
        $PY multirace/caa.py --target "$T" --alpha 2 \
            --items "$ITEMS" --out-dir "$OUT/$T/caa/rot${r}"
done

echo "ALL DONE (gmainm: man base/decode-PI(amax${MAN_AMAX})/normal(a${MAN_ALPHA})/caa(a2) x rot0-2)"
