#!/bin/bash
#SBATCH --job-name=uqhf-effort
#SBATCH --output=/shared/home/sarim.hashmi/dlm_bias/.claude/worktrees/agent-a9f2f89e11792ec37/logs/slurm/%x-%j.out
#SBATCH --error=/shared/home/sarim.hashmi/dlm_bias/.claude/worktrees/agent-a9f2f89e11792ec37/logs/slurm/%x-%j.err
#SBATCH --gres=shard:3
#SBATCH --qos=guaranteed-nils
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --exclude=mbz-titan-2
# Phase 2, step 5: the EFFORT-MATCHED open-loop control, on the BUILD sweep
# subset.  alpha is the realized mean actuation of the selected controller over
# the FULL run (all 64 denoising steps) -- the pre-registered basis of
# unqover_hf/PREREG.md section 5.  Pass it as $1 (and the tag as $2).
# Usage:  sbatch slurm/unqover_hf_jobP3_effort.sbatch <alpha> <tag>
# BUILD SPLIT ONLY.
source /shared/home/sarim.hashmi/dlm_bias/.claude/worktrees/agent-a9f2f89e11792ec37/slurm/_uqhf_prologue.sh
ALPHA=${1:?need the effort-matched alpha}
TAG=${2:-normal_effortmatched}
OUT=$WT/results/unqover_hf/build_sweep
$PY -m unqover_hf.run_race --method normal --alpha "$ALPHA" --tag "$TAG" \
    --n-instances 200 --seed 42 \
    --split-path "$WT/data/unqover_hirundo/build_race.jsonl" --out-dir "$OUT" \
    --n-boot 1000 --n-noise 100
echo "ALL DONE (uqhf-effort alpha=$ALPHA)"
