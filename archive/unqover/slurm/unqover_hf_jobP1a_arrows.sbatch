#!/bin/bash
#SBATCH --job-name=uqhf-arrows
#SBATCH --output=/shared/home/sarim.hashmi/dlm_bias/.claude/worktrees/agent-a9f2f89e11792ec37/logs/slurm/%x-%j.out
#SBATCH --error=/shared/home/sarim.hashmi/dlm_bias/.claude/worktrees/agent-a9f2f89e11792ec37/logs/slurm/%x-%j.err
#SBATCH --gres=shard:3
#SBATCH --qos=guaranteed-nils
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --exclude=mbz-titan-2
# Phase 2, step 1: UNQOVER-race native direction + every baseline fit, from the
# BUILD split ONLY.  Writes unqover_hf/arrows_race_black.pt and
# unqover_hf/cache_race/{actadd_dir_pair*,linearact_stats,aura_auroc,itic_probes}.pt
# Then a throughput probe (BUILD split) that sets the per-config sweep size.
source /shared/home/sarim.hashmi/dlm_bias/.claude/worktrees/agent-a9f2f89e11792ec37/slurm/_uqhf_prologue.sh
$PY -m unqover_hf.build_arrows_race --build
$PY -m unqover_hf.build_arrows_race --selftest
SM=$WT/results/unqover_hf/smoke
$PY -m unqover_hf.run_race --method base    --limit 40 --out-dir "$SM" --tag smoke_base   --force
$PY -m unqover_hf.run_race --method ours_PI --limit 40 --amax 6 --out-dir "$SM" --tag smoke_ourspi --force
$PY -m unqover_hf.run_race --method caa     --limit 40 --alpha 32 --out-dir "$SM" --tag smoke_caa --force
echo "ALL DONE (uqhf-arrows)"
