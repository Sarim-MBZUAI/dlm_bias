#!/bin/bash
#SBATCH --job-name=uqnbase
#SBATCH --output=/shared/home/sarim.hashmi/dlm_bias/logs/slurm/%x-%j.out
#SBATCH --error=/shared/home/sarim.hashmi/dlm_bias/logs/slurm/%x-%j.err
#SBATCH --gres=gpu:1
#SBATCH --qos=normal-plus
#SBATCH --time=06:00:00
# Finish the 5 remaining NATIVE UnQover baselines on the eval split, reusing the
# native fits (unqover/cache_native) + native arrows already on disk. caa/actadd done.
set -e
export DLM_BIAS_ROOT=/shared/home/sarim.hashmi/dlm_bias
export HF_MODULES_CACHE=/shared/home/sarim.hashmi/.cache/hf_modules
cd "$DLM_BIAS_ROOT"
PY=/shared/home/sarim.hashmi/anaconda3/envs/dlm/bin/python
UQ=unqover; RES=results/unqover_native
ARROWS=$RES/arrows_unqover_black.pt
NCACHE=$UQ/cache_native
EVAL_ITEMS=$RES/eval_items.jsonl
for M in uq_meanact uq_linearact uq_aura_inject uq_aura_vanilla uq_itic; do
  if [ ! -f "$RES/$M.json" ]; then
    echo "RUN native baseline $M"
    $PY $UQ/baselines_unqover.py --method "$M" \
        --items "$EVAL_ITEMS" --out "$RES/$M.jsonl" \
        --native-cache "$NCACHE" --native-arrows "$ARROWS"
    $PY $UQ/unqover_metric.py --results "$RES/$M.jsonl" \
        --baseline "$RES/uq_clean.jsonl" --target-subject Black > "$RES/metric_$M.txt"
  else echo "SKIP $M (exists)"; fi
done
echo "ALL DONE (uqnbase: 5 native baselines on eval split)"
