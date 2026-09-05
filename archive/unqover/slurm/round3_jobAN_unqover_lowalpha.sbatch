#!/bin/bash
#SBATCH --job-name=uqloa
#SBATCH --output=/shared/home/sarim.hashmi/dlm_bias/logs/slurm/%x-%j.out
#SBATCH --error=/shared/home/sarim.hashmi/dlm_bias/logs/slurm/%x-%j.err
#SBATCH --gres=gpu:1
#SBATCH --qos=normal-plus
#SBATCH --time=03:00:00
# Low-alpha end of the curve: decode-PI at amax {1,2,3,4}, default gains, DEBIASED on eval.
set -e
export DLM_BIAS_ROOT=/shared/home/sarim.hashmi/dlm_bias
export HF_MODULES_CACHE=/shared/home/sarim.hashmi/.cache/hf_modules
cd "$DLM_BIAS_ROOT"
PY=/shared/home/sarim.hashmi/anaconda3/envs/dlm/bin/python
UQ=unqover; RES=results/unqover_native; AR=$RES/arrows_unqover_black.pt; EV=$RES/eval_items.jsonl
run(){ tag=$1; a=$2; [ -f "$RES/$tag.json" ] && { echo "SKIP $tag"; return; }; echo "RUN $tag amax=$a"; $PY $UQ/denoise_pid_unqover.py --cond PI --kp 3 --ki 0.1 --kd 0 --amax $a --arrows $AR --items $EV --target Black --out $RES/$tag.jsonl; $PY $UQ/unqover_metric.py --results $RES/$tag.jsonl --baseline $RES/uq_clean.jsonl --target-subject Black > $RES/metric_$tag.txt; }
run uq_PI_amax1 1
run uq_PI_amax2 2
run uq_PI_amax3 3
run uq_PI_amax4 4
echo "ALL DONE (uqloa)"
