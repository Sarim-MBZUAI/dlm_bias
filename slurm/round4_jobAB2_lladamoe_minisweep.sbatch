#!/bin/bash
#SBATCH --job-name=moeab2
#SBATCH --account=nils
#SBATCH --partition=gpu
#SBATCH --qos=normal-plus
#SBATCH --gres=gpu:1
#SBATCH --time=10:00:00
#SBATCH --output=/shared/home/sarim.hashmi/dlm_bias/logs/slurm/%x-%j.out
#SBATCH --error=/shared/home/sarim.hashmi/dlm_bias/logs/slurm/%x-%j.err

# ROUND 4 JOB AB2 "moeab2" -- additive mini-sweep after the jobAB dose-sweep
# forensics (results/lladamoe/SWEEP_NOTES.md).  Two findings drive it:
#   (1) EVERY decode-PI / normal all-16-broadcast dose collapsed.  The apparent
#       coherence at amax 0.5 / 4 was a bbq_eval.parse_letter rule-2 artifact
#       ('mathbf...'->A, 'blackblack...'->B).  Cause: unit(r[8]) injected at all
#       16 blocks is ~135% of the pooled hidden norm at block 0 (||h|| grows
#       0.37->150 across the stack) -- the flat broadcast inverts the natural
#       norm profile and obliterates early layers.  CAA aims (a2: gap +0.237,
#       0% invalid) because its ONE mid-layer site keeps total added L2 = 2.
#   (2) ActAdd never saw a dose below its collapse zone (a4/a8/16 all invalid);
#       its geometry is identical to CAA (alpha*unit at L8), so a1/a2 -- the
#       doses where CAA is coherent -- were the missing operating points.
# CONDITIONS (400 unrotated items each, ~1-1.7h/run):
#   actadd a1, a2                        -> results/lladamoe/actadd/cond_actadd_a{1,2}
#   decode-PI L8-only amax 2             -> cond_dpid_PI_L8_amax2
#       (closed-loop CAA parity: same site+budget as CAA a2, but feedback
#        alpha(t) that can BACK OFF once p_black hits the 0.9 setpoint)
#   decode-PI raw-norm all-16 amax 2     -> cond_dpid_PI_raw16_amax2
#       (alpha(t)*r[k] RAW per block: preserves the natural top-heavy norm
#        profile, 7-32% of local ||h|| everywhere; alpha=1 == the natural
#        Black-vs-other mean shift, amax 2 gives 2x headroom)
# Tags are NEW stems -- no collision with jobAB's cond_dpid_PI_amax*/normalL8_*.
# Idempotent per run via run_if_missing (jobAB convention).
set -euo pipefail

export HF_HOME=/shared/models/huggingface
export HF_HUB_CACHE=/shared/models/huggingface/hub
export HF_MODULES_CACHE=/shared/home/sarim.hashmi/.cache/hf_modules
export DLM_BIAS_ROOT=/shared/home/sarim.hashmi/dlm_bias
cd "$DLM_BIAS_ROOT"
PY=/shared/home/sarim.hashmi/anaconda3/envs/dlm/bin/python
# Do NOT set CUDA_VISIBLE_DEVICES -- SLURM sets it.

OUT=results/lladamoe

run_if_missing() {  # $1 = expected result json, rest = command
    local marker=$1; shift
    if [[ -f "$marker" ]]; then
        echo "SKIP (exists): $marker"
    else
        echo "RUN: $*"
        "$@"
    fi
}

# Prerequisites from round4_jobAB (fail loud if absent).
for f in llada_moe/arrows.pt llada_moe/baselines/cache/actadd_dir.pt; do
    test -f "$f" || { echo "FATAL: $f missing (run round4_jobAB first)"; exit 1; }
done

# (1) ActAdd low doses (the sub-collapse operating points jobAB never tested).
for AL in 1 2; do
    run_if_missing "$OUT/actadd/cond_actadd_a${AL}.json" \
        $PY llada_moe/baselines/actadd.py --run --alpha "$AL" --layer 8 \
            --out_dir "$OUT/actadd"
done

# (2) decode-PI variant A: single-mid-layer actuation, CAA-a2 dose budget.
run_if_missing "$OUT/decode_pid/cond_dpid_PI_L8_amax2.json" \
    $PY llada_moe/denoise_pid.py --cond PI --amax 2 --layers 8 \
        --tag dpid_PI_L8_amax2

# (3) decode-PI variant B: all-16 with the NATURAL raw-norm profile.
run_if_missing "$OUT/decode_pid/cond_dpid_PI_raw16_amax2.json" \
    $PY llada_moe/denoise_pid.py --cond PI --amax 2 --layer-scale raw \
        --tag dpid_PI_raw16_amax2

echo "ALL DONE (moeab2: actadd a{1,2} + dpid_PI_L8_amax2 + dpid_PI_raw16_amax2)"
