#!/bin/bash
#SBATCH --job-name=uqhf-sweep
#SBATCH --output=/shared/home/sarim.hashmi/dlm_bias/.claude/worktrees/agent-a9f2f89e11792ec37/logs/slurm/%x-%A_%a.out
#SBATCH --error=/shared/home/sarim.hashmi/dlm_bias/.claude/worktrees/agent-a9f2f89e11792ec37/logs/slurm/%x-%A_%a.err
#SBATCH --gres=shard:3
#SBATCH --qos=guaranteed-nils
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=20:00:00
#SBATCH --requeue
#SBATCH --exclude=mbz-titan-2
#SBATCH --array=0-9%4
# Phase 2, step 3: the PRE-REGISTERED BUILD-split sweep (unqover_hf/PREREG.md
# section 3) -- 61 configs across 12 methods, EQUAL BUDGET, one array task per
# stride of 10 so each task loads the model once and covers a mix of methods.
# BUILD SPLIT ONLY.  The EVAL split is not touched anywhere in this job.
# Re-running the array resumes: a config whose *_summary.json exists is skipped.
source /shared/home/sarim.hashmi/dlm_bias/.claude/worktrees/agent-a9f2f89e11792ec37/slurm/_uqhf_prologue.sh
OUT=$WT/results/unqover_hf/build_sweep
$PY -m unqover_hf.run_race --sweep --stride 10 --offset "$SLURM_ARRAY_TASK_ID" \
    --split-path "$WT/data/unqover_hirundo/build_race.jsonl" --out-dir "$OUT" \
    --n-boot 1000 --n-noise 100
echo "ALL DONE (uqhf-sweep task $SLURM_ARRAY_TASK_ID)"
