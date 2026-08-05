#!/bin/bash
#SBATCH --job-name=mtbal1
#SBATCH --account=nils
#SBATCH --partition=gpu
#SBATCH --qos=normal-plus
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=/shared/home/sarim.hashmi/dlm_bias/logs/slurm/%x-%j.out
#SBATCH --error=/shared/home/sarim.hashmi/dlm_bias/logs/slurm/%x-%j.err

# ROUND 2 JOB B "mtbal1" (afterok arrows2): position-balanced multirace eval,
# targets ARAB + ASIAN.  Per target x rot{0,1,2}:
#   multirace/denoise_pid.py --cond base -> results/balanced_all/<T>/base/rot{r}
#   multirace/denoise_pid.py --cond PI   -> results/balanced_all/<T>/decode_pid/rot{r}
#   multirace/normal.py at the fair dose (arab a4, asian a2)
#                                        -> results/balanced_all/<T>/normal/rot{r}
# Mirrors balanced_all/run_balanced_gpu4.sh's layout/tags exactly so
# balanced_all/aggregate.py works unchanged (default tags: dpid_<T>_base,
# dpid_<T>_PI, normal_<T>_a<alpha>).  18 runs, idempotent via run_if_missing.
set -euo pipefail

export HF_HOME=/shared/models/huggingface
export HF_HUB_CACHE=/shared/models/huggingface/hub
export HF_MODULES_CACHE=/shared/home/sarim.hashmi/.cache/hf_modules
export DLM_BIAS_ROOT=/shared/home/sarim.hashmi/dlm_bias
cd "$DLM_BIAS_ROOT"
PY=/shared/home/sarim.hashmi/anaconda3/envs/dlm/bin/python
# Do NOT set CUDA_VISIBLE_DEVICES -- SLURM sets it.

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

for T in arab asian; do
    if [[ "$T" == arab ]]; then ALPHA=4; else ALPHA=2; fi
    for r in 0 1 2; do
        ITEMS=$ROTDIR/_sweep400_${T}_rot${r}.jsonl

        run_if_missing "$OUT/$T/base/rot${r}/cond_dpid_${T}_base.json" \
            $PY multirace/denoise_pid.py --target "$T" --cond base \
                --items "$ITEMS" --out-dir "$OUT/$T/base/rot${r}"

        run_if_missing "$OUT/$T/decode_pid/rot${r}/cond_dpid_${T}_PI.json" \
            $PY multirace/denoise_pid.py --target "$T" --cond PI \
                --items "$ITEMS" --out-dir "$OUT/$T/decode_pid/rot${r}"

        run_if_missing "$OUT/$T/normal/rot${r}/cond_normal_${T}_a${ALPHA}.json" \
            $PY multirace/normal.py --target "$T" --alpha "$ALPHA" \
                --items "$ITEMS" --out-dir "$OUT/$T/normal/rot${r}"
    done
done

echo "ALL DONE (mtbal1: arab a4 + asian a2, base/decode-PI/normal x rot0-2)"
