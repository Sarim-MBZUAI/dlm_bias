#!/bin/bash
#SBATCH --job-name=uqrelig
#SBATCH --account=nils
#SBATCH --partition=gpu
#SBATCH --qos=normal-plus
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --output=/shared/home/sarim.hashmi/dlm_bias/logs/slurm/%x-%j.out
#SBATCH --error=/shared/home/sarim.hashmi/dlm_bias/logs/slurm/%x-%j.err

# ROUND 3 JOB W "uqrelig" (no dependency; modeled on round2_jobE): UNQOVER
# RELIGION -- CLEAN BASELINE ONLY (E9 decision (a), see below).
#
# ITEM PIPELINE (CPU, guarded; the vendored official repo at
# data/unqover/unqover_repo generates the questions offline -- no network):
#   1. unqover/download_unqover.py --classes religion
#        -> data/unqover/generated/religion.source.json
#   2. unqover/unqover_loader.py --limit 2000 --seed 42
#        -> data/unqover/religion.items.jsonl  (2000 instances = 8000 items)
#   3. cap file = FIRST 2000 lines (= first 500 COMPLETE 4-item instances;
#      quadruples are contiguous, same convention as jobE's ethnicity cap)
#        -> data/unqover/religion.items.cap2000.jsonl
#
# CONDITION: clean only.
#   unqover/unqover_eval.py on the capped 2000 items -> uq_clean.jsonl, then
#   unqover/unqover_metric.py (official mu/eta/delta) + the signed pref_gap
#   toward 'Muslim' (exact subject string from the official word list
#   data/unqover/unqover_repo/word_lists/nouns/subjects/religion: Christian,
#   Hindu, Muslim, Buddhist, Jewish, Sikh, Mormon, Protestant, Orthodox,
#   Catholic, Atheist).
#
# WHY NO STEERED RELIGION RUNS (E9 DECISION (a), deliberate):
#   * BBQ Religion has only 600 ambiguous rows -- far below the 800-row
#     minimum of the item protocol (let alone the 1600 the 4-way disjoint
#     eval/heldout split needs), so a BBQ-built religion steering direction
#     is NOT possible.
#   * Reusing a direction built for ANOTHER attribute (e.g. the Black race
#     direction, as jobE's ethnicity runs legitimately do for their own
#     target) would not test religion steering -- it would test cross-class
#     transfer of a race direction and be reported as religion steering:
#     scientifically wrong, so we refuse it.
#   * An UnQover-native religion direction (built from UnQover items) is new
#     machinery and out of scope here -- documented as future work.
#   The clean run is still paper-worthy: it measures LLaDA's BASELINE
#   religion bias on UnQover (mu/eta/delta + pref_gap toward Muslim) and
#   frames the future steered study.
#
# Outputs under results/unqover_religion/.  Skip-guard marker = the .json
# CONFIG file (written only after a run completes, so partial jsonls re-run).
set -euo pipefail

export HF_HOME=/shared/models/huggingface
export HF_HUB_CACHE=/shared/models/huggingface/hub
export HF_MODULES_CACHE=/shared/home/sarim.hashmi/.cache/hf_modules
export DLM_BIAS_ROOT=/shared/home/sarim.hashmi/dlm_bias
cd "$DLM_BIAS_ROOT"
PY=/shared/home/sarim.hashmi/anaconda3/envs/dlm/bin/python
# Do NOT set CUDA_VISIBLE_DEVICES -- SLURM sets it.

RES=results/unqover_religion
mkdir -p "$RES"
SRC=data/unqover/generated/religion.source.json
FULL=data/unqover/religion.items.jsonl
ITEMS=data/unqover/religion.items.cap2000.jsonl

run_if_missing() {  # $1 = expected COMPLETION marker, rest = command
    local marker=$1; shift
    if [[ -f "$marker" ]]; then
        echo "SKIP (exists): $marker"
    else
        echo "RUN: $*"
        "$@"
    fi
}

# ---- items (CPU, guarded; normally already built at merge time) ----------- #
run_if_missing "$SRC" \
    $PY unqover/download_unqover.py --classes religion
run_if_missing "$FULL" \
    $PY unqover/unqover_loader.py --source "$SRC" --out "$FULL" \
        --limit 2000 --seed 42

# Deterministic cap file (first 2000 lines = 500 complete instances).
if [[ ! -f "$ITEMS" ]]; then
    head -n 2000 "$FULL" > "$ITEMS"
    echo "BUILT cap file: $ITEMS ($(wc -l < "$ITEMS") items)"
fi

# ---- clean baseline: all 2000 capped items, no steering ------------------- #
run_if_missing "$RES/uq_clean.json" \
    $PY unqover/unqover_eval.py --items "$ITEMS" --out "$RES/uq_clean.jsonl"

# ---- official metric on the clean run (CPU; always re-run, cheap) --------- #
echo "### metric: uq_clean (religion, baseline-only -- no steered conditions)"
$PY unqover/unqover_metric.py --results "$RES/uq_clean.jsonl" \
    --target-subject Muslim \
    | tee "$RES/metric_uq_clean.txt"

echo "ALL DONE (uqrelig: clean religion baseline + metric; steered runs"
echo "deliberately omitted -- see header, needs an UnQover-native direction)"
