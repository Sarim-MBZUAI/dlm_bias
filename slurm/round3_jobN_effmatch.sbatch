#!/bin/bash
#SBATCH --job-name=effmatch
#SBATCH --account=nils
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --qos=normal-plus
#SBATCH --output=/shared/home/sarim.hashmi/dlm_bias/logs/slurm/effmatch-%j.out
#SBATCH --error=/shared/home/sarim.hashmi/dlm_bias/logs/slurm/effmatch-%j.err
# E5 effort-matched control: open-loop alpha = -0.72 = suppression controller's
# own mean actuation (round-1 a3p28 precedent). 3 rotations.
set -euo pipefail
export HF_HOME=/shared/models/huggingface
export HF_HUB_CACHE=/shared/models/huggingface/hub
export HF_MODULES_CACHE=/shared/home/sarim.hashmi/.cache/hf_modules
export DLM_BIAS_ROOT=/shared/home/sarim.hashmi/dlm_bias
cd "$DLM_BIAS_ROOT"
PY=/shared/home/sarim.hashmi/anaconda3/envs/dlm/bin/python
run_if_missing() { local marker="$1"; shift; if [ -f "$marker" ]; then echo "SKIP (exists): $marker"; else "$@"; fi; }
for R in 0 1 2; do
  OUT=results/balanced_all/black/normal_neg/rot$R
  run_if_missing "$OUT/cond_normalL14_am0p72.json" \
    $PY steering/pid_steer.py --mode normal --alpha -0.72 \
      --items results/balanced/_sweep400_rot$R.jsonl --out-dir "$OUT"
done
echo "ALL DONE (effmatch: open-loop a=-0.72 x rot0-2)"
