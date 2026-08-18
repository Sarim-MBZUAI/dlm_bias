#!/bin/bash
#SBATCH --job-name=dbal
#SBATCH --account=nils
#SBATCH --partition=gpu
#SBATCH --qos=normal-plus
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=/shared/home/sarim.hashmi/dlm_bias/logs/slurm/%x-%j.out
#SBATCH --error=/shared/home/sarim.hashmi/dlm_bias/logs/slurm/%x-%j.err

# ROUND 3 JOB Z "dbal" (Dream baseline parity, afterok dfits): the prior-work
# baseline conditions still missing from the position-balanced Dream family --
# the exact round2_jobF pattern (same rotation item files
# results/balanced/_sweep400_rot{r}.jsonl the existing dream_balanced
# base/decode-PI/caa/actadd runs used), completing LLaDA parity:
#   meanact      --direction unit --strength 2  -> cond_meanact_unit_s2
#   linearact    --variant gaussian --strength 1 -> cond_gaussian_s1
#   aura inject  --gamma 4                       -> cond_inject_g4
#   aura vanilla                                 -> cond_vanilla
#   itic         --topk 48 --alpha 8             -> cond_itic_K48_a8
# = 5 conditions x rot0-2 = 15 runs (~8 min each on 1 GPU).
# Operating points and stems replicate the LLaDA balanced suite
# (results/balanced_all/black; strict_round2.BASELINE_CONDS) EXACTLY, so
# balanced_all/strict_round2.py's Dream family (DREAM_BASELINE_CONDS)
# discovers them unchanged. Directions/gates come from the DREAM fits of
# round3_jobY (dream/baselines/cache/*; dream/arrows.pt for meanact-unit) --
# never from the LLaDA artifacts. NOTE: on Dream the all-block raw meanact s2
# and itic a16 collapsed to 100% unparseable (dream/RESULTS.md); the balanced
# picks here are the milder LLaDA-published points (unit s2, itic a8) -- if a
# condition still collapses, the strict table reports it as invalid/unparse
# (that IS the parity result, not a job failure).
# Idempotent per run via run_if_missing.
set -euo pipefail

export HF_HOME=/shared/models/huggingface
export HF_HUB_CACHE=/shared/models/huggingface/hub
export HF_MODULES_CACHE=/shared/home/sarim.hashmi/.cache/hf_modules
export DLM_BIAS_ROOT=/shared/home/sarim.hashmi/dlm_bias
cd "$DLM_BIAS_ROOT"
PY=/shared/home/sarim.hashmi/anaconda3/envs/dlm/bin/python
# Do NOT set CUDA_VISIBLE_DEVICES -- SLURM sets it.

OUT=results/dream_balanced
CACHE=dream/baselines/cache

run_if_missing() {  # $1 = expected result json, rest = command
    local marker=$1; shift
    if [[ -f "$marker" ]]; then
        echo "SKIP (exists): $marker"
    else
        echo "RUN: $*"
        "$@"
    fi
}

# Fit prerequisites (built by round3_jobY dfits): fail loud if absent.
for f in dream/arrows.pt "$CACHE/linearact_stats.pt" "$CACHE/aura_auroc.pt" \
         "$CACHE/itic_probes.pt"; do
    test -f "$f" || { echo "FATAL: $f missing (run round3_jobY dfits first)"; exit 1; }
done

for r in 0 1 2; do
    ITEMS=results/balanced/_sweep400_rot${r}.jsonl
    test -f "$ITEMS" || { echo "FATAL: $ITEMS missing"; exit 1; }

    run_if_missing "$OUT/meanact/rot${r}/cond_meanact_unit_s2.json" \
        $PY dream/baselines/meanact.py --run --direction unit --strength 2 \
            --items "$ITEMS" --out_dir "$OUT/meanact/rot${r}"

    run_if_missing "$OUT/linearact/rot${r}/cond_gaussian_s1.json" \
        $PY dream/baselines/linearact.py --run --variant gaussian --strength 1 \
            --items "$ITEMS" --out_dir "$OUT/linearact/rot${r}"

    run_if_missing "$OUT/aura_inject/rot${r}/cond_inject_g4.json" \
        $PY dream/baselines/aura.py --run --mode inject --gamma 4 \
            --items "$ITEMS" --out_dir "$OUT/aura_inject/rot${r}"

    run_if_missing "$OUT/aura_vanilla/rot${r}/cond_vanilla.json" \
        $PY dream/baselines/aura.py --run --mode vanilla \
            --items "$ITEMS" --out_dir "$OUT/aura_vanilla/rot${r}"

    run_if_missing "$OUT/itic/rot${r}/cond_itic_K48_a8.json" \
        $PY dream/baselines/itic.py --run --topk 48 --alpha 8 \
            --items "$ITEMS" --out_dir "$OUT/itic/rot${r}"
done

echo "ALL DONE (dbal: meanact/linearact/aura_inject/aura_vanilla/itic x rot0-2)"
