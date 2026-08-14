#!/bin/bash
#SBATCH --job-name=saeffmatch
#SBATCH --account=nils
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --qos=normal-plus
#SBATCH --output=/shared/home/sarim.hashmi/dlm_bias/logs/slurm/saeffmatch-%j.out
#SBATCH --error=/shared/home/sarim.hashmi/dlm_bias/logs/slurm/saeffmatch-%j.err
# E9 effort-matched control: SES/Age open loop at alpha=3 (controller mean ~3.3),
# mirroring the gender jobO pattern. 3 rotations x 4 targets.
set -euo pipefail
export HF_HOME=/shared/models/huggingface
export HF_HUB_CACHE=/shared/models/huggingface/hub
export HF_MODULES_CACHE=/shared/home/sarim.hashmi/.cache/hf_modules
export DLM_BIAS_ROOT=/shared/home/sarim.hashmi/dlm_bias
cd "$DLM_BIAS_ROOT"
PY=/shared/home/sarim.hashmi/anaconda3/envs/dlm/bin/python
run_if_missing() { local marker="$1"; shift; if [ -f "$marker" ]; then echo "SKIP (exists): $marker"; else "$@"; fi; }
for T in lowses highses old young; do
  for R in 0 1 2; do
    ITEMS=results/balanced_all/rotations/_sweep400_${T}_rot$R.jsonl
    if [ ! -f "$ITEMS" ]; then echo "MISSING rotations for $T rot$R -- jobU/jobV generate them; aborting"; exit 1; fi
    OUT=results/balanced_all/$T/normal_eff/rot$R
    run_if_missing "$OUT/cond_normal_${T}_a3.json" \
      $PY multirace/normal.py --target $T --alpha 3 \
        --items "$ITEMS" --out-dir "$OUT" --tag "normal_${T}_a3"
  done
done
echo "ALL DONE (saeffmatch: lowses/highses/old/young open-loop a=3 x rot0-2)"
