#!/bin/bash
#SBATCH --job-name=uqours
#SBATCH --output=/shared/home/sarim.hashmi/dlm_bias/logs/slurm/%x-%j.out
#SBATCH --error=/shared/home/sarim.hashmi/dlm_bias/logs/slurm/%x-%j.err
#SBATCH --gres=gpu:1
#SBATCH --qos=normal-plus
#SBATCH --time=03:00:00
# Fast repopulate of OUR native UnQover rows on the eval split: default decode-PI,
# decode-PID, and the EFFORT-MATCHED open loop (alpha=2.71 = PI's realized mean).
# Native direction, eval split -> fair. Runs independently of the slow tuning sweep.
set -e
export DLM_BIAS_ROOT=/shared/home/sarim.hashmi/dlm_bias
export HF_MODULES_CACHE=/shared/home/sarim.hashmi/.cache/hf_modules
cd "$DLM_BIAS_ROOT"
PY=/shared/home/sarim.hashmi/anaconda3/envs/dlm/bin/python
UQ=unqover; RES=results/unqover_native
ARROWS=$RES/arrows_unqover_black.pt; EV=$RES/eval_items.jsonl
run(){ [ -f "$RES/$1.json" ] && { echo "SKIP $1"; return; }; echo "RUN $1"; eval "$2"; $PY $UQ/unqover_metric.py --results "$RES/$1.jsonl" --baseline "$RES/uq_clean.jsonl" --target-subject Black > "$RES/metric_$1.txt"; }
run uq_decode_PI  "$PY $UQ/denoise_pid_unqover.py --cond PI  --arrows $ARROWS --items $EV --target Black --out $RES/uq_decode_PI.jsonl"
run uq_decode_PID "$PY $UQ/denoise_pid_unqover.py --cond PID --arrows $ARROWS --items $EV --target Black --out $RES/uq_decode_PID.jsonl"
run uq_normal_effort "$PY $UQ/pid_steer_unqover.py --mode normal --alpha 2.71 --arrows $ARROWS --items $EV --target Black --out $RES/uq_normal_effort.jsonl"
echo "ALL DONE (uqours)"
