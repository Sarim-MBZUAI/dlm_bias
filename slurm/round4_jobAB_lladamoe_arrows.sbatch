#!/bin/bash
#SBATCH --job-name=moearw
#SBATCH --account=nils
#SBATCH --partition=gpu
#SBATCH --qos=normal-plus
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=/shared/home/sarim.hashmi/dlm_bias/logs/slurm/%x-%j.out
#SBATCH --error=/shared/home/sarim.hashmi/dlm_bias/logs/slurm/%x-%j.err

# ROUND 4 JOB AB "moearw" (after moesmk passes its gate): LLaDA-MoE arrows +
# dose calibration -- the round2_jobA + round3_jobH pattern on the MoE port.
#   (1) llada_moe/build_arrows.py -> llada_moe/arrows.pt (16x2048 diff-in-means,
#       contamination-safe held-out BBQ-Race contrast set, same items as
#       LLaDA/Dream arrows).
#   (2) dose sweep on the UNROTATED data/bbq_items/_sweep400.jsonl ->
#       results/lladamoe/{decode_pid,normal,caa}:
#         base                          cond_dpid_base.json
#         decode-PI amax {0.5,1,2,4}    cond_dpid_PI_amax{A}.json
#         normal L8 alpha {1,2,4}       cond_normalL8_a{A}.json
#         caa L8 alpha {1,2,4,8}        cond_caa_L8_a{A}.json
#         actadd L8 alpha {4,8,16}      cond_actadd_a{A}.json (fit guard first)
#       Dream precedent: LLaDA's amax 6 was 97% garbage on Dream (jobF used
#       amax 1.0); this sweep picks MOE_AMAX / MOE_CAA_ALPHA / MOE_ACTADD_ALPHA
#       for round4_jobAD.  Watch unparseable_rate collapse at high doses.
# Idempotent per artifact/run via run_if_missing.
set -euo pipefail

export HF_HOME=/shared/models/huggingface
export HF_HUB_CACHE=/shared/models/huggingface/hub
export HF_MODULES_CACHE=/shared/home/sarim.hashmi/.cache/hf_modules
export DLM_BIAS_ROOT=/shared/home/sarim.hashmi/dlm_bias
cd "$DLM_BIAS_ROOT"
PY=/shared/home/sarim.hashmi/anaconda3/envs/dlm/bin/python
# Do NOT set CUDA_VISIBLE_DEVICES -- SLURM sets it.

OUT=results/lladamoe

run_if_missing() {  # $1 = expected artifact/result json, rest = command
    local marker=$1; shift
    if [[ -f "$marker" ]]; then
        echo "SKIP (exists): $marker"
    else
        echo "RUN: $*"
        "$@"
    fi
}

# (1) Arrows (gitignored artifact; the whole downstream DAG needs it).
run_if_missing "llada_moe/arrows.pt" \
    $PY llada_moe/build_arrows.py

# (2a) clean base (also the sanity reference for every dose).
run_if_missing "$OUT/decode_pid/cond_dpid_base.json" \
    $PY llada_moe/denoise_pid.py --cond base

# (2b) decode-PI amax sweep (conservative start; Dream optimum was 1.0).
for A in 0.5 1 2 4; do
    TAGA=$(echo "$A" | tr . p)
    run_if_missing "$OUT/decode_pid/cond_dpid_PI_amax${TAGA}.json" \
        $PY llada_moe/denoise_pid.py --cond PI --amax "$A" \
            --tag "dpid_PI_amax${TAGA}"
done

# (2c) open-loop normal-vector dose sweep (all-16-block broadcast of unit(r[8])).
for AL in 1 2 4; do
    run_if_missing "$OUT/normal/cond_normalL8_a${AL}.json" \
        $PY llada_moe/pid_steer.py --mode normal --source-layer 8 --alpha "$AL" \
            --tag "normalL8_a${AL}" --out-dir "$OUT/normal"
done

# (2d) CAA single-layer dose sweep (L8 = 16-layer midpoint).
for AL in 1 2 4 8; do
    run_if_missing "$OUT/caa/cond_caa_L8_a${AL}.json" \
        $PY llada_moe/baselines/caa.py --run --alpha "$AL" --layer 8 \
            --out_dir "$OUT/caa"
done

# (2e) ActAdd fit (single-pair dir; guard on the cache artifact) + dose sweep.
run_if_missing "llada_moe/baselines/cache/actadd_dir.pt" \
    $PY llada_moe/baselines/actadd.py --fit
for AL in 4 8 16; do
    run_if_missing "$OUT/actadd/cond_actadd_a${AL}.json" \
        $PY llada_moe/baselines/actadd.py --run --alpha "$AL" --layer 8 \
            --out_dir "$OUT/actadd"
done

echo "ALL DONE (moearw: arrows + base + decode-PI amax{0.5,1,2,4} + normal a{1,2,4}"
echo "          + caa L8 a{1,2,4,8} + actadd a{4,8,16} -- pick doses for jobAD)"
