#!/bin/bash
#SBATCH --job-name=arabcs
#SBATCH --account=nils
#SBATCH --partition=gpu
#SBATCH --qos=normal-plus
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=/shared/home/sarim.hashmi/dlm_bias/logs/slurm/%x-%j.out
#SBATCH --error=/shared/home/sarim.hashmi/dlm_bias/logs/slurm/%x-%j.err

# ROUND 3 JOB M "arabcs" (E7 case-full sensor; no dependency -- needs only
# multirace/arrows_arab.pt + the arab rotation files, both exist locally).
#
# Rationale (analysis/trajectory/FINDINGS.md RQ2): the decode-space sensor is
# case-BLIND (sums P over "A"/" A" uppercase ids only) while the strict parser
# accepts lowercase, and the arab vector induces lowercase answers -- 82.8% of
# arab strict-target successes end sensor-invisible (p_final~0), so the
# controller saturates on wins and losses alike (mean_alpha 4.56, sat 0.38).
# E7 gives the sensor lowercase vision (--sensor-case both: + "a"/" a" ids,
# tokenizer-encoded) and reruns arab decode-PI. PREDICTION: the controller
# backs off once it can see its wins (like black's post-lock alpha drop
# 2.9 -> ~1.9), i.e. lower mean_alpha / sat_frac at a comparable strict gap.
#
# Per rot r in {0,1,2} on results/balanced_all/rotations/_sweep400_arab_rot{r}.jsonl:
#   decode-PI --sensor-case both --amax 6
#   -> results/balanced_all/arab/decode_pid_cs/rot{r}/cond_dpid_arab_PI_cs.json
#
# The arab BASE and the case-BLIND decode-PI already exist under
# results/balanced_all/arab/{base,decode_pid}/rot{r}/ (round 2) -- NOT rerun;
# the analysis (balanced_all/strict_round3.py, E7 family) compares against them.
#
# The rotation item files are gitignored but exist in the main tree; if
# missing they regenerate deterministically via balanced_all/make_rotations.py
# (guarded below like jobI), and oracle_test.py re-proves them (hard fail).
# 3 GPU runs (+ rotations + oracle), idempotent via run_if_missing.
set -euo pipefail

export HF_HOME=/shared/models/huggingface
export HF_HUB_CACHE=/shared/models/huggingface/hub
export HF_MODULES_CACHE=/shared/home/sarim.hashmi/.cache/hf_modules
export DLM_BIAS_ROOT=/shared/home/sarim.hashmi/dlm_bias
cd "$DLM_BIAS_ROOT"
PY=/shared/home/sarim.hashmi/anaconda3/envs/dlm/bin/python
# Do NOT set CUDA_VISIBLE_DEVICES -- SLURM sets it.

T=arab
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

# ---- rotations (CPU, deterministic, guarded) + oracle proof (hard fail) ---- #
run_if_missing "$ROTDIR/_sweep400_${T}_rot2.jsonl" \
    $PY balanced_all/make_rotations.py \
        --items "data/bbq_items/_sweep400_${T}.jsonl" --out-dir "$ROTDIR"
$PY balanced_all/oracle_test.py \
    --items-glob "$ROTDIR/_sweep400_${T}_rot*.jsonl" \
    --target "$T" --orig "data/bbq_items/_sweep400_${T}.jsonl"

# ---- decode-PI with the case-FULL sensor x rot{0,1,2} ----------------------- #
for r in 0 1 2; do
    ITEMS=$ROTDIR/_sweep400_${T}_rot${r}.jsonl

    run_if_missing "$OUT/$T/decode_pid_cs/rot${r}/cond_dpid_${T}_PI_cs.json" \
        $PY multirace/denoise_pid.py --target "$T" --cond PI --amax 6 \
            --sensor-case both --tag "dpid_${T}_PI_cs" \
            --items "$ITEMS" --out-dir "$OUT/$T/decode_pid_cs/rot${r}"
done

echo "ALL DONE (arabcs: arab decode-PI --sensor-case both amax6 x rot0-2)"
