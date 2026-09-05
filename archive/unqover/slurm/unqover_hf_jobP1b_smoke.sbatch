#!/bin/bash
#SBATCH --job-name=uqhf-smoke
#SBATCH --output=/shared/home/sarim.hashmi/dlm_bias/.claude/worktrees/agent-a9f2f89e11792ec37/logs/slurm/%x-%j.out
#SBATCH --error=/shared/home/sarim.hashmi/dlm_bias/.claude/worktrees/agent-a9f2f89e11792ec37/logs/slurm/%x-%j.err
#SBATCH --gres=shard:5
#SBATCH --qos=normal-plus
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --requeue
#SBATCH --exclude=mbz-titan-2
# Throughput probe on the BUILD split: seconds per item for the open-loop path
# (bbq_eval.generate) and the closed-loop path (controlled_generate_traced).
# Sets the pre-registered per-config sweep size.  No EVAL data is touched.
source /shared/home/sarim.hashmi/dlm_bias/.claude/worktrees/agent-a9f2f89e11792ec37/slurm/_uqhf_prologue.sh
OUT=$WT/results/unqover_hf/smoke
$PY -m unqover_hf.run_race --method base    --limit 40 --out-dir "$OUT" --tag smoke_base --force
$PY -m unqover_hf.run_race --method ours_PI --limit 40 --amax 6 --out-dir "$OUT" --tag smoke_ourspi --force
echo "ALL DONE (uqhf-smoke)"
