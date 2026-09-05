#!/bin/bash
#SBATCH --job-name=r2smoke
#SBATCH --account=nils
#SBATCH --partition=debug
#SBATCH --qos=normal-plus
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=/shared/home/sarim.hashmi/dlm_bias/logs/slurm/%x-%j.out
#SBATCH --error=/shared/home/sarim.hashmi/dlm_bias/logs/slurm/%x-%j.err

# ROUND 2 SMOKE (debug partition): 3 tiny GPU checks before the real DAG.
#   (1) multirace decode-PI, 2 items, arab rot0 file, --dummy-arrows (arrows not
#       built yet); --smoke writes NO result files.
#   (2) [REMOVED] UNQOVER clean eval -- track archived under archive/unqover/.
#   (3) dream/denoise_pid.py --help + 2-item Dream smoke with --dummy-arrows on
#       the LLaDA balanced rot0 items file (verifies --items plumbing).
set -euo pipefail

export HF_HOME=/shared/models/huggingface
export HF_HUB_CACHE=/shared/models/huggingface/hub
export HF_MODULES_CACHE=/shared/home/sarim.hashmi/.cache/hf_modules
export DLM_BIAS_ROOT=/shared/home/sarim.hashmi/dlm_bias
cd "$DLM_BIAS_ROOT"
PY=/shared/home/sarim.hashmi/anaconda3/envs/dlm/bin/python
# Do NOT set CUDA_VISIBLE_DEVICES -- SLURM sets it.

SMK=logs/smoke_round2

echo "== (1) multirace decode-PI smoke (arab rot0, dummy arrows, 2 items) =="
$PY multirace/denoise_pid.py --smoke --target arab --cond PI --dummy-arrows \
    --items results/balanced_all/rotations/_sweep400_arab_rot0.jsonl

# (2) UNQOVER clean smoke -- REMOVED.  The UNQOVER track is archived under
#     archive/unqover/ and is not part of the paper; see
#     archive/unqover/README.md.  The stage used to run
#     unqover/unqover_eval.py on data/unqover/ethnicity.items.smoke2.jsonl.
echo "== (2) UNQOVER clean smoke -- SKIPPED (track archived) =="

echo "== (3) dream --help + 2-item Dream smoke (dummy arrows, balanced rot0) =="
$PY dream/denoise_pid.py --help
$PY dream/denoise_pid.py --smoke --dummy-arrows --limit 2 \
    --items results/balanced/_sweep400_rot0.jsonl

echo "SMOKE ALL DONE"
