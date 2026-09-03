#!/bin/bash
#SBATCH --job-name=uqfull
#SBATCH --account=nils
#SBATCH --partition=gpu
#SBATCH --qos=normal-plus
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=/shared/home/sarim.hashmi/dlm_bias/logs/slurm/%x-%j.out
#SBATCH --error=/shared/home/sarim.hashmi/dlm_bias/logs/slurm/%x-%j.err

# ROUND 3 JOB AH "uqfull" (no dependency): re-run the ENTIRE UNQOVER ethnicity
# comparison on the FULL uncapped items file so the method comparison is
# well-powered and fair, instead of the capped 64-instance subset.
#
# WHY: the existing results/unqover_v2/ steered runs used
# data/unqover/ethnicity.items.cap2000.jsonl (2000 rows / 500 instances, only
# 64 Black-containing -> 256 items), so every steered condition scored on just
# 64 Black instances (as few as 41 quadruple-complete after no_answer). The FULL
# file data/unqover/ethnicity.items.jsonl has 8000 rows / 2000 instances, of
# which 262 are Black-containing -> 1048 items (all 262 quadruple-complete before
# no_answer drops). That is ~4.1x the data -> the per-method winner is trustworthy.
#
# WHAT CHANGES vs jobE (round2, the 4 primary conds) + jobAG (round3, the 7
# baselines): ONLY the input items file (the FULL file, no cap; the steered
# adapters still Black-filter internally via load_target_items; NO --limit) and
# the OUTPUT dir (results/unqover_full/, NOT unqover_v2 -- the old 64-instance
# results are LEFT UNTOUCHED for verification). SAME operating points, SAME black
# fit artifacts, SAME gen config (gen_length 32 / steps 64 / block 32 / temp 0).
#
# CONDITIONS (11 that mirror the v2 table, + 2 optional cheap decode variants):
#   uq_clean         unqover_eval.py (no steering, ALL 8000 items)        base
#   uq_decode_PI     denoise_pid_unqover.py --cond PI    (Kp3 Ki0.1, amax6, arrows.pt L14)   ours (closed loop)
#   uq_layer_PI_a2   pid_steer_unqover.py --mode pid --cond PI --alpha 2  ours (layer-space Eq.18)
#   uq_normal_a4     pid_steer_unqover.py --mode normal --alpha 4         open-loop control
#   uq_caa           baselines_unqover.py --method uq_caa          caa L14 a16        steering/arrows.pt
#   uq_actadd        baselines_unqover.py --method uq_actadd       actadd L14 a16     baselines/cache/actadd_dir.pt
#   uq_meanact       baselines_unqover.py --method uq_meanact      meanact unit s2    steering/arrows.pt
#   uq_linearact     baselines_unqover.py --method uq_linearact    linearact gauss s1 baselines/cache/linearact_stats.pt
#   uq_aura_inject   baselines_unqover.py --method uq_aura_inject  aura inject g4     baselines/cache/aura_auroc.pt
#   uq_aura_vanilla  baselines_unqover.py --method uq_aura_vanilla aura vanilla       baselines/cache/aura_auroc.pt
#   uq_itic          baselines_unqover.py --method uq_itic         itic K48 a8        baselines/cache/itic_probes.pt
#   (optional, cheap) uq_decode_PID  denoise_pid_unqover.py --cond PID  (Kp3 Ki0.1 Kd1.0, amax6)
#   (optional, cheap) uq_decode_P    denoise_pid_unqover.py --cond P    (Kp3, amax6)
# = 11 core + 2 optional runs, single GPU, serial.
#
# EACH steered/baseline run Black-filters the FULL file to 1048 items / 262
# instances. The clean run scores ALL 8000 items (2000 instances); its Black
# subset (262 instances) is what the directional pref_gap vs clean uses.
#
# GPU ESTIMATE (from v2 elapsed: decode_PI 256 items 399s ~1.56 s/item; layer/
# normal 256 items ~381s ~1.49 s/item; base has no hooks -> ~1.4 s/item):
#   uq_clean        8000 items  ~1.4 s/item  ~3.1 h   (no hooks, but all instances)
#   3x decode       1048 items  ~1.6 s/item  ~27 min each  ~1.4 h
#   layer + normal  1048 items  ~1.5 s/item  ~26 min each  ~0.9 h
#   7x baselines    1048 items  ~1.6 s/item  ~28 min each  ~3.3 h
#   TOTAL           ~8.7 GPU-hours  (fits comfortably in the 12 h walltime)
#
# IDEMPOTENT: run_if_missing keys on the .json CONFIG (written ONLY after a run
# completes, so a partially-written .jsonl re-runs cleanly) -- same convention as
# jobE / jobAG. Re-submitting after a timeout resumes where it left off.
set -euo pipefail

export HF_HOME=/shared/models/huggingface
export HF_HUB_CACHE=/shared/models/huggingface/hub
export HF_MODULES_CACHE=/shared/home/sarim.hashmi/.cache/hf_modules
export DLM_BIAS_ROOT=/shared/home/sarim.hashmi/dlm_bias
cd "$DLM_BIAS_ROOT"
PY=/shared/home/sarim.hashmi/anaconda3/envs/dlm/bin/python
# Do NOT set CUDA_VISIBLE_DEVICES -- SLURM sets it.

UQ=unqover
RES=results/unqover_full                    # NEW dir; does NOT touch unqover_v2
ITEMS=data/unqover/ethnicity.items.jsonl    # FULL uncapped file (8000 rows)
CLEAN=$RES/uq_clean.jsonl                    # clean baseline for the metric
mkdir -p "$RES"

test -f "$ITEMS" || { echo "FATAL: $ITEMS missing (full unqover file)"; exit 1; }

run_if_missing() {  # $1 = expected result .json (completion marker), rest = command
    local marker=$1; shift
    if [[ -f "$marker" ]]; then
        echo "SKIP (exists): $marker"
    else
        echo "RUN: $*"
        "$@"
    fi
}

# --------------------------------------------------------------------------- #
# 1) GENERATION (single GPU, serial). Clean FIRST (it is the metric baseline).
# --------------------------------------------------------------------------- #

# (base) clean: ALL 8000 items, no steering.
run_if_missing "$RES/uq_clean.json" \
    $PY $UQ/unqover_eval.py --items "$ITEMS" --out "$RES/uq_clean.jsonl"

# (ours) decode-space closed-loop PI (target Black, steering/arrows.pt defaults).
run_if_missing "$RES/uq_decode_PI.json" \
    $PY $UQ/denoise_pid_unqover.py --cond PI \
        --items "$ITEMS" --out "$RES/uq_decode_PI.jsonl"

# (ours) layer-space PI at alpha 2 (item-independent per-block injection).
run_if_missing "$RES/uq_layer_PI_a2.json" \
    $PY $UQ/pid_steer_unqover.py --mode pid --cond PI --alpha 2 \
        --items "$ITEMS" --out "$RES/uq_layer_PI_a2.jsonl"

# (open loop) fixed vhat at all 32 blocks, alpha 4.
run_if_missing "$RES/uq_normal_a4.json" \
    $PY $UQ/pid_steer_unqover.py --mode normal --alpha 4 \
        --items "$ITEMS" --out "$RES/uq_normal_a4.jsonl"

# 7 prior-work baselines (Table-1 Black operating points, black fit artifacts).
# NOTE: --out is REQUIRED so these write into unqover_full, NOT the default
# results/unqover_v2 (which would overwrite the old runs).
BASELINES=(uq_caa uq_actadd uq_meanact uq_linearact uq_aura_inject uq_aura_vanilla uq_itic)
for M in "${BASELINES[@]}"; do
    run_if_missing "$RES/$M.json" \
        $PY $UQ/baselines_unqover.py --method "$M" \
            --items "$ITEMS" --out "$RES/$M.jsonl"
done

# (optional, cheap) two extra decode-space variants for a fuller ours-family row.
run_if_missing "$RES/uq_decode_PID.json" \
    $PY $UQ/denoise_pid_unqover.py --cond PID \
        --items "$ITEMS" --out "$RES/uq_decode_PID.jsonl"
run_if_missing "$RES/uq_decode_P.json" \
    $PY $UQ/denoise_pid_unqover.py --cond P \
        --items "$ITEMS" --out "$RES/uq_decode_P.jsonl"

# --------------------------------------------------------------------------- #
# 2) METRIC (CPU). Official mu/eta/delta + BBQ-comparable directional gap toward
#    'Black' vs the FULL clean baseline. metric_uq_<method>.txt in $RES.
# --------------------------------------------------------------------------- #
test -f "$CLEAN" || { echo "FATAL: $CLEAN missing (clean baseline needed for metric)"; exit 1; }

# Baseline-only report for the clean run itself.
if [[ ! -f "$RES/metric_uq_clean.txt" ]]; then
    echo "### metric: uq_clean (baseline-only)"
    $PY $UQ/unqover_metric.py --results "$CLEAN" --target-subject Black \
        | tee "$RES/metric_uq_clean.txt"
fi

STEERED=(uq_decode_PI uq_layer_PI_a2 uq_normal_a4 "${BASELINES[@]}" uq_decode_PID uq_decode_P)
for M in "${STEERED[@]}"; do
    [[ -f "$RES/$M.jsonl" ]] || { echo "SKIP metric (no $M.jsonl)"; continue; }
    if [[ -f "$RES/metric_$M.txt" ]]; then
        echo "SKIP metric (exists): $RES/metric_$M.txt"
        continue
    fi
    echo "### metric: $M vs uq_clean"
    $PY $UQ/unqover_metric.py --results "$RES/$M.jsonl" \
        --baseline "$CLEAN" --target-subject Black \
        | tee "$RES/metric_$M.txt"
done

echo "DONE jobAH: full-file UNQOVER -> $RES/ (uq_clean + decode_{PI,PID,P} + layer_PI_a2 + normal_a4 + 7 baselines, metric_uq_*.txt). Old results/unqover_v2 untouched."
