#!/bin/bash
#SBATCH --job-name=seeds
#SBATCH --account=nils
#SBATCH --partition=gpu
#SBATCH --qos=normal-plus
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=/shared/home/sarim.hashmi/dlm_bias/logs/slurm/%x-%j.out
#SBATCH --error=/shared/home/sarim.hashmi/dlm_bias/logs/slurm/%x-%j.err

# ROUND 2 JOB D "seeds" (no dependency; needs only steering/arrows.pt, which
# exists): seed replication of the headline pair.  For seed s in {1,2,3} x rot
# r in {0,1,2}:
#   steering/denoise_pid.py --cond base -> results/balanced_seeds/seed{s}/base/rot{r}
#   steering/denoise_pid.py --cond PI   -> results/balanced_seeds/seed{s}/decode_pid/rot{r}
# Items: that seed's rotation file results/balanced/seeds/seed{s}/_sweep400_rot{r}.jsonl.
# Default tags dpid_base / dpid_PI.  18 runs, idempotent via run_if_missing.
set -euo pipefail

export HF_HOME=/shared/models/huggingface
export HF_HUB_CACHE=/shared/models/huggingface/hub
export HF_MODULES_CACHE=/shared/home/sarim.hashmi/.cache/hf_modules
export DLM_BIAS_ROOT=/shared/home/sarim.hashmi/dlm_bias
cd "$DLM_BIAS_ROOT"
PY=/shared/home/sarim.hashmi/anaconda3/envs/dlm/bin/python
# Do NOT set CUDA_VISIBLE_DEVICES -- SLURM sets it.

OUT=results/balanced_seeds

run_if_missing() {  # $1 = expected result json, rest = command
    local marker=$1; shift
    if [[ -f "$marker" ]]; then
        echo "SKIP (exists): $marker"
    else
        echo "RUN: $*"
        "$@"
    fi
}

for s in 1 2 3; do
    for r in 0 1 2; do
        ITEMS=results/balanced/seeds/seed${s}/_sweep400_rot${r}.jsonl

        run_if_missing "$OUT/seed${s}/base/rot${r}/cond_dpid_base.json" \
            $PY steering/denoise_pid.py --cond base \
                --items "$ITEMS" --out-dir "$OUT/seed${s}/base/rot${r}"

        run_if_missing "$OUT/seed${s}/decode_pid/rot${r}/cond_dpid_PI.json" \
            $PY steering/denoise_pid.py --cond PI \
                --items "$ITEMS" --out-dir "$OUT/seed${s}/decode_pid/rot${r}"
    done
done

echo "ALL DONE (seeds: base + decode-PI x seed1-3 x rot0-2)"
