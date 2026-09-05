#!/bin/bash
#SBATCH --job-name=uqhia
#SBATCH --output=/shared/home/sarim.hashmi/dlm_bias/logs/slurm/%x-%j.out
#SBATCH --error=/shared/home/sarim.hashmi/dlm_bias/logs/slurm/%x-%j.err
#SBATCH --gres=gpu:1
#SBATCH --qos=normal-plus
#SBATCH --time=04:00:00
# Test whether HIGHER alpha helps the DEBIASED gap on native UnQover (eval split).
# Raise amax and (for the pushed variants) Kp/Ki so the controller actually uses the headroom.
set -e
export DLM_BIAS_ROOT=/shared/home/sarim.hashmi/dlm_bias
export HF_MODULES_CACHE=/shared/home/sarim.hashmi/.cache/hf_modules
cd "$DLM_BIAS_ROOT"
PY=/shared/home/sarim.hashmi/anaconda3/envs/dlm/bin/python
UQ=unqover; RES=results/unqover_native; AR=$RES/arrows_unqover_black.pt; EV=$RES/eval_items.jsonl
run(){ tag=$1; shift; [ -f "$RES/$tag.json" ] && { echo "SKIP $tag"; return; }; echo "RUN $tag :: $*"; $PY $UQ/denoise_pid_unqover.py "$@" --arrows $AR --items $EV --target Black --out $RES/$tag.jsonl; $PY $UQ/unqover_metric.py --results $RES/$tag.jsonl --baseline $RES/uq_clean.jsonl --target-subject Black > $RES/metric_$tag.txt; }
# amax raises, default gains (isolate ceiling) then pushed gains (force higher realized alpha)
run uq_PI_amax8         --cond PI --kp 3 --ki 0.1 --kd 0 --amax 8
run uq_PI_amax8_push    --cond PI --kp 5 --ki 0.3 --kd 0 --amax 8
run uq_PI_amax12_push   --cond PI --kp 5 --ki 0.5 --kd 0 --amax 12
run uq_PID_amax10_push  --cond PID --kp 5 --ki 0.3 --kd 1 --amax 10
echo "ALL DONE (uqhia)"
