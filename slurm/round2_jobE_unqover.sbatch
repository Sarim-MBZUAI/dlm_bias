#!/bin/bash
#SBATCH --job-name=unqover
#SBATCH --account=nils
#SBATCH --partition=gpu
#SBATCH --qos=normal-plus
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=/shared/home/sarim.hashmi/dlm_bias/logs/slurm/%x-%j.out
#SBATCH --error=/shared/home/sarim.hashmi/dlm_bias/logs/slurm/%x-%j.err

# ROUND 2 JOB E "unqover" (no dependency; uses the FIXED parser + existing
# steering/arrows.pt): four conditions on UNQOVER ethnicity items.
#
# ITEM CAP: unqover_eval.py has NO --limit, so all four conditions share ONE
# capped items file = the FIRST 2000 lines (= first 500 COMPLETE 4-item
# instances; quadruples verified contiguous) of data/unqover/ethnicity.items.jsonl:
#   data/unqover/ethnicity.items.cap2000.jsonl
# Clean runs all 2000 capped items.  The three steered adapters PRE-FILTER
# (load_target_items) to Black-containing pairs within the SAME capped file
# -> 256 items / 64 instances each.  unqover_metric.py regroups by instance and
# scores steered-vs-clean on the shared instances.
#
# Conditions (all --items = the capped file):
#   (1) clean     unqover_eval.py, no steering            -> uq_clean.jsonl
#   (2) decode-PI denoise_pid_unqover.py --cond PI
#                 (default target Black, steering/arrows.pt) -> uq_decode_PI.jsonl
#   (3) layer-PI  pid_steer_unqover.py --mode pid --cond PI --alpha 2
#                                                         -> uq_layer_PI_a2.jsonl
#   (4) open loop pid_steer_unqover.py --mode normal --alpha 4
#                                                         -> uq_normal_a4.jsonl
# Outputs under results/unqover_v2/.  Skip-guard marker = the .json CONFIG file
# (written only after a run completes, so partial jsonls are re-run).
set -euo pipefail

export HF_HOME=/shared/models/huggingface
export HF_HUB_CACHE=/shared/models/huggingface/hub
export HF_MODULES_CACHE=/shared/home/sarim.hashmi/.cache/hf_modules
export DLM_BIAS_ROOT=/shared/home/sarim.hashmi/dlm_bias
cd "$DLM_BIAS_ROOT"
PY=/shared/home/sarim.hashmi/anaconda3/envs/dlm/bin/python
# Do NOT set CUDA_VISIBLE_DEVICES -- SLURM sets it.

RES=results/unqover_v2
mkdir -p "$RES"
ITEMS=data/unqover/ethnicity.items.cap2000.jsonl

# Deterministic cap file (first 2000 lines = 500 complete instances).
if [[ ! -f "$ITEMS" ]]; then
    head -n 2000 data/unqover/ethnicity.items.jsonl > "$ITEMS"
    echo "BUILT cap file: $ITEMS ($(wc -l < "$ITEMS") items)"
fi

run_if_missing() {  # $1 = expected COMPLETION marker (config json), rest = command
    local marker=$1; shift
    if [[ -f "$marker" ]]; then
        echo "SKIP (exists): $marker"
    else
        echo "RUN: $*"
        "$@"
    fi
}

# (1) clean baseline: all 2000 capped items, no steering.
run_if_missing "$RES/uq_clean.json" \
    $PY unqover/unqover_eval.py --items "$ITEMS" --out "$RES/uq_clean.jsonl"

# (2) decode-space PI (closed loop; target Black, steering/arrows.pt defaults).
run_if_missing "$RES/uq_decode_PI.json" \
    $PY unqover/denoise_pid_unqover.py --cond PI \
        --items "$ITEMS" --out "$RES/uq_decode_PI.jsonl"

# (3) layer-space PI at alpha 2 (item-independent per-block injection).
run_if_missing "$RES/uq_layer_PI_a2.json" \
    $PY unqover/pid_steer_unqover.py --mode pid --cond PI --alpha 2 \
        --items "$ITEMS" --out "$RES/uq_layer_PI_a2.jsonl"

# (4) open loop: fixed vhat at all 32 blocks, alpha 4.
run_if_missing "$RES/uq_normal_a4.json" \
    $PY unqover/pid_steer_unqover.py --mode normal --alpha 4 \
        --items "$ITEMS" --out "$RES/uq_normal_a4.jsonl"

# Official metric + directional gap vs clean (CPU; always re-run, cheap).
for stem in uq_decode_PI uq_layer_PI_a2 uq_normal_a4; do
    [[ -f "$RES/$stem.jsonl" ]] || continue
    echo "### metric: $stem vs uq_clean"
    $PY unqover/unqover_metric.py --results "$RES/$stem.jsonl" \
        --baseline "$RES/uq_clean.jsonl" --target-subject Black \
        | tee "$RES/metric_${stem}.txt"
done

echo "ALL DONE (unqover_v2: clean / decode-PI / layer-PI a2 / normal a4)"
