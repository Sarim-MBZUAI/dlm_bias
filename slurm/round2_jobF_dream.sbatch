#!/bin/bash
#SBATCH --job-name=dream
#SBATCH --account=nils
#SBATCH --partition=gpu
#SBATCH --qos=normal-plus
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=/shared/home/sarim.hashmi/dlm_bias/logs/slurm/%x-%j.out
#SBATCH --error=/shared/home/sarim.hashmi/dlm_bias/logs/slurm/%x-%j.err

# ROUND 2 JOB F "dream" (afterok arrows2, needs dream/arrows.pt): Dream parity
# through the position-balanced eval, SAME LLaDA Black rotation item files
# results/balanced/_sweep400_rot{r}.jsonl.
#   * dream/denoise_pid.py --cond base            -> results/dream_balanced/base/rot{r}
#   * dream/denoise_pid.py --cond PI --amax 1.0   -> results/dream_balanced/decode_pid/rot{r}
#     (Dream decode-PI optimum; LLaDA's amax 6 is 97% garbage on Dream)
#   * dream/baselines/actadd.py --fit (skip-guard on cache/actadd_dir.pt)
#   * caa    --alpha 2 (faithful dose)            -> results/dream_balanced/caa/rot{r}
#   * actadd --alpha 8 (faithful dose)            -> results/dream_balanced/actadd/rot{r}
# 12 runs + 1 fit, idempotent via run_if_missing.
set -euo pipefail

export HF_HOME=/shared/models/huggingface
export HF_HUB_CACHE=/shared/models/huggingface/hub
export HF_MODULES_CACHE=/shared/home/sarim.hashmi/.cache/hf_modules
export DLM_BIAS_ROOT=/shared/home/sarim.hashmi/dlm_bias
cd "$DLM_BIAS_ROOT"
PY=/shared/home/sarim.hashmi/anaconda3/envs/dlm/bin/python
# Do NOT set CUDA_VISIBLE_DEVICES -- SLURM sets it.

OUT=results/dream_balanced

run_if_missing() {  # $1 = expected result json/artifact, rest = command
    local marker=$1; shift
    if [[ -f "$marker" ]]; then
        echo "SKIP (exists): $marker"
    else
        echo "RUN: $*"
        "$@"
    fi
}

# ActAdd single-pair direction fit (GPU; cached artifact guard).
run_if_missing "dream/baselines/cache/actadd_dir.pt" \
    $PY dream/baselines/actadd.py --fit

for r in 0 1 2; do
    ITEMS=results/balanced/_sweep400_rot${r}.jsonl

    run_if_missing "$OUT/base/rot${r}/cond_dpid_base.json" \
        $PY dream/denoise_pid.py --cond base \
            --items "$ITEMS" --out-dir "$OUT/base/rot${r}"

    run_if_missing "$OUT/decode_pid/rot${r}/cond_dpid_PI.json" \
        $PY dream/denoise_pid.py --cond PI --amax 1.0 \
            --items "$ITEMS" --out-dir "$OUT/decode_pid/rot${r}"

    run_if_missing "$OUT/caa/rot${r}/cond_caa_L14_a2.json" \
        $PY dream/baselines/caa.py --run --alpha 2 --layer 14 \
            --items "$ITEMS" --out_dir "$OUT/caa/rot${r}"

    run_if_missing "$OUT/actadd/rot${r}/cond_actadd_a8.json" \
        $PY dream/baselines/actadd.py --run --alpha 8 --layer 14 \
            --items "$ITEMS" --out_dir "$OUT/actadd/rot${r}"
done

echo "ALL DONE (dream_balanced: base / decode-PI amax1 / caa a2 / actadd a8 x rot0-2)"
