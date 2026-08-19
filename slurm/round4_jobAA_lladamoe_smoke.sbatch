#!/bin/bash
#SBATCH --job-name=moesmk
#SBATCH --account=nils
#SBATCH --partition=debug
#SBATCH --qos=normal-plus
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=/shared/home/sarim.hashmi/dlm_bias/logs/slurm/%x-%j.out
#SBATCH --error=/shared/home/sarim.hashmi/dlm_bias/logs/slurm/%x-%j.err

# ROUND 4 JOB AA "moesmk" (debug partition): LLaDA-MoE-7B-A1B-Instruct gate
# BEFORE the real DAG (round4_jobAB -> jobAC -> jobAD).
#   (1) all 13 offline selftests (CPU-only math; cheap re-verification on node).
#   (2) 20-item CLEAN answer-format gate: common_lladamoe --smoke --smoke-items 20
#       prints counts incl. unparseable_rate.  GATE: if unparseable_rate is high
#       (>~0.2), STOP -- the model cannot do the letter-answer task at these
#       generation settings and the downstream jobs would only measure noise.
#   (3) 2-item decode-PID smoke with --dummy-arrows on the LLaDA balanced rot0
#       items file (verifies --items plumbing + the closed loop varies alpha).
# No result files are committed from here (cache/ + --smoke output only).
set -euo pipefail

export HF_HOME=/shared/models/huggingface
export HF_HUB_CACHE=/shared/models/huggingface/hub
export HF_MODULES_CACHE=/shared/home/sarim.hashmi/.cache/hf_modules
export DLM_BIAS_ROOT=/shared/home/sarim.hashmi/dlm_bias
cd "$DLM_BIAS_ROOT"
PY=/shared/home/sarim.hashmi/anaconda3/envs/dlm/bin/python
# Do NOT set CUDA_VISIBLE_DEVICES -- SLURM sets it.

echo "== (1) offline selftests (all 13 llada_moe modules) =="
for f in llada_moe/common_lladamoe.py llada_moe/build_arrows.py \
         llada_moe/denoise_pid.py llada_moe/pid_steer.py \
         llada_moe/baselines/directions.py llada_moe/baselines/calib.py \
         llada_moe/baselines/caa.py llada_moe/baselines/actadd.py \
         llada_moe/baselines/meanact.py llada_moe/baselines/linearact.py \
         llada_moe/baselines/aura.py llada_moe/baselines/itic.py \
         llada_moe/baselines/run_all.py; do
    echo "--- $f --selftest"
    $PY "$f" --selftest
done

echo "== (2) 20-item CLEAN answer-format gate (chat template, no steering) =="
$PY llada_moe/common_lladamoe.py --smoke --smoke-items 20

echo "== (3) 2-item decode-PID smoke (dummy arrows, balanced rot0 items) =="
$PY llada_moe/denoise_pid.py --smoke --dummy-arrows --limit 2 \
    --items results/balanced/_sweep400_rot0.jsonl

echo "SMOKE ALL DONE (moesmk: check unparseable_rate in step (2) before jobAB)"
