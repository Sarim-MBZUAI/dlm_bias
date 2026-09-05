#!/bin/bash
#SBATCH --job-name=uqnative
#SBATCH --account=nils
#SBATCH --partition=gpu
#SBATCH --qos=normal-plus
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=/shared/home/sarim.hashmi/dlm_bias/logs/slurm/%x-%j.out
#SBATCH --error=/shared/home/sarim.hashmi/dlm_bias/logs/slurm/%x-%j.err

# ROUND 3 JOB AI "uqnative" (no dependency): make UNQOVER a STANDALONE benchmark.
#
# WHY: today's UNQOVER runs (jobE/jobAG/jobAH) reuse the BBQ-built "prefer Black"
# direction (steering/arrows.pt row 14) and the BBQ black baseline fit artifacts
# (baselines/cache/*.pt). This job REPLACES that BBQ dependency with an
# UNQOVER-NATIVE direction + native baseline fits, built from UNQOVER's OWN
# held-out contrast data on a clean, committed, instance-disjoint train/eval
# split, then runs our PID controller AND all baselines on the EVAL split only.
#
# SPLIT (committed, reproducible): unqover/items_manifest_unqover_black.json --
# the 262 Black-containing UNQOVER ethnicity instances are split by instance_id
# (seed 42, 40% build) into BUILD (105 instances) and DISJOINT EVAL (157). The
# builder only ever sees BUILD; every condition below is evaluated on EVAL only,
# so there is ZERO train/eval contamination (build INTERSECT eval == empty).
#
# WHAT CHANGES vs jobAH: ONLY the direction/fit DATA SOURCE (UNQOVER build split
# instead of BBQ) and the eval set (the 157-instance EVAL split instead of all
# 262). SAME Table-1 operating points (caa L14 a16, actadd a16, meanact unit s2,
# linearact gaussian s1, aura inject g4 + vanilla, itic K48 a8), SAME PID gains/
# amax (decode Kp3 Ki0.1 [Kd1.0], amax6, s*0.9, vhat L14; layer PI alpha2; normal
# alpha4), SAME gen config (gen_length 32 / steps 64 / block 32 / temp 0).
# Output dir results/unqover_native/ -- does NOT touch unqover_v2 or unqover_full.
#
# STEPS:
#   1) BUILD (GPU): build_arrows_unqover.py --build -> UNQOVER-native
#      results/unqover_native/arrows_unqover_black.pt (CAA/Mean-AcT direction) +
#      unqover/cache_native/{actadd_dir,linearact_stats,aura_auroc,itic_probes,
#      meanact_meandiff_block,calib_*}.pt (the native baseline fits). Also emit
#      build_items.jsonl / eval_items.jsonl (the runners' --items files).
#   2) EVAL (GPU, EVAL split only): base + decode_{PI,PID,P} + layer_PI_a2 +
#      normal_a4 + 7 baselines, all with the native arrows/fits.
#   3) METRIC (CPU): unqover_metric.py per method vs the native clean baseline
#      -> metric_uq_<method>.txt.
#
# GPU ESTIMATE:
#   build   105 build inst x 2 fwd x 3 granularities = 630 short forwards +
#           CPU probe/AUROC fits                                    ~0.25 h
#   clean   628 eval items  ~1.4 s/item                            ~0.25 h
#   3x decode 628 items ~1.6 s/item ~17 min each                   ~0.85 h
#   layer + normal 628 items ~1.5 s/item ~16 min each              ~0.55 h
#   7x baselines 628 items ~1.6 s/item ~17 min each                ~2.0 h
#   TOTAL   ~3.9 GPU-hours  (comfortably within the 12 h walltime)
#
# IDEMPOTENT: run_if_missing keys on the .json CONFIG (written ONLY after a run
# completes); step 1 keys on the last-written fit (itic_probes.pt). Re-submitting
# after a timeout resumes where it left off. Same convention as jobAH.
set -euo pipefail

export HF_HOME=/shared/models/huggingface
export HF_HUB_CACHE=/shared/models/huggingface/hub
export HF_MODULES_CACHE=/shared/home/sarim.hashmi/.cache/hf_modules
export DLM_BIAS_ROOT=/shared/home/sarim.hashmi/dlm_bias
cd "$DLM_BIAS_ROOT"
PY=/shared/home/sarim.hashmi/anaconda3/envs/dlm/bin/python
# Do NOT set CUDA_VISIBLE_DEVICES -- SLURM sets it.

UQ=unqover
RES=results/unqover_native                       # NEW dir; NOT unqover_v2/unqover_full
NCACHE=unqover/cache_native                       # UNQOVER-native fit cache
ARROWS=$RES/arrows_unqover_black.pt               # UNQOVER-native direction (CAA/Mean-AcT)
MANIFEST=unqover/items_manifest_unqover_black.json # committed train/eval split
FULL=data/unqover/ethnicity.items.jsonl           # full 8000-item source
BUILD_ITEMS=$RES/build_items.jsonl                # direction/fit split (105 inst)
EVAL_ITEMS=$RES/eval_items.jsonl                  # held-out eval split (157 inst)
CLEAN=$RES/uq_clean.jsonl
mkdir -p "$RES" "$NCACHE"

test -f "$MANIFEST" || { echo "FATAL: $MANIFEST missing (committed split)"; exit 1; }
test -f "$FULL" || { echo "FATAL: $FULL missing (full unqover file)"; exit 1; }

run_if_missing() {  # $1 = expected marker (written only on completion), rest = command
    local marker=$1; shift
    if [[ -f "$marker" ]]; then echo "SKIP (exists): $marker"; else echo "RUN: $*"; "$@"; fi
}

# --------------------------------------------------------------------------- #
# 1) BUILD the UNQOVER-native direction + baseline fits from the BUILD split.  #
#    Also materialize the build/eval item files (deterministic from the        #
#    manifest) that every eval condition below consumes.                       #
# --------------------------------------------------------------------------- #
run_if_missing "$EVAL_ITEMS" \
    $PY $UQ/build_arrows_unqover.py --write-splits --out-dir "$RES"

# The direction + 5 fits (GPU). itic_probes.pt is written LAST -> its presence
# marks a complete build.
run_if_missing "$NCACHE/itic_probes.pt" \
    $PY $UQ/build_arrows_unqover.py --build \
        --manifest "$MANIFEST" --items "$FULL" \
        --cache-dir "$NCACHE" --arrows-out "$ARROWS"

test -f "$ARROWS" || { echo "FATAL: $ARROWS not built"; exit 1; }

# --------------------------------------------------------------------------- #
# 2) EVAL on the held-out EVAL split ONLY, with the UNQOVER-native arrows/fits. #
# --------------------------------------------------------------------------- #

# (base) native clean: EVAL split, no steering (the metric baseline).
run_if_missing "$RES/uq_clean.json" \
    $PY $UQ/unqover_eval.py --items "$EVAL_ITEMS" --out "$CLEAN"

# (ours) decode-space closed-loop PI/PID/P with the NATIVE direction (arrows L14).
run_if_missing "$RES/uq_decode_PI.json" \
    $PY $UQ/denoise_pid_unqover.py --cond PI \
        --arrows "$ARROWS" --items "$EVAL_ITEMS" --out "$RES/uq_decode_PI.jsonl"
run_if_missing "$RES/uq_decode_PID.json" \
    $PY $UQ/denoise_pid_unqover.py --cond PID \
        --arrows "$ARROWS" --items "$EVAL_ITEMS" --out "$RES/uq_decode_PID.jsonl"
run_if_missing "$RES/uq_decode_P.json" \
    $PY $UQ/denoise_pid_unqover.py --cond P \
        --arrows "$ARROWS" --items "$EVAL_ITEMS" --out "$RES/uq_decode_P.jsonl"

# (ours) layer-space PI at alpha 2 (item-independent per-block injection, native arrows).
run_if_missing "$RES/uq_layer_PI_a2.json" \
    $PY $UQ/pid_steer_unqover.py --mode pid --cond PI --alpha 2 \
        --arrows "$ARROWS" --items "$EVAL_ITEMS" --out "$RES/uq_layer_PI_a2.jsonl"

# (open loop) fixed native vhat at all 32 blocks, alpha 4.
run_if_missing "$RES/uq_normal_a4.json" \
    $PY $UQ/pid_steer_unqover.py --mode normal --alpha 4 \
        --arrows "$ARROWS" --items "$EVAL_ITEMS" --out "$RES/uq_normal_a4.jsonl"

# 7 prior-work baselines at the SAME Table-1 operating points, but with the
# UNQOVER-NATIVE fits (--native-cache) and native arrows (--native-arrows).
BASELINES=(uq_caa uq_actadd uq_meanact uq_linearact uq_aura_inject uq_aura_vanilla uq_itic)
for M in "${BASELINES[@]}"; do
    run_if_missing "$RES/$M.json" \
        $PY $UQ/baselines_unqover.py --method "$M" \
            --items "$EVAL_ITEMS" --out "$RES/$M.jsonl" \
            --native-cache "$NCACHE" --native-arrows "$ARROWS"
done

# --------------------------------------------------------------------------- #
# 3) METRIC (CPU): official mu/eta/delta + BBQ-comparable directional gap toward
#    'Black' vs the native clean baseline. metric_uq_<method>.txt in $RES.
# --------------------------------------------------------------------------- #
test -f "$CLEAN" || { echo "FATAL: $CLEAN missing (clean baseline needed for metric)"; exit 1; }

if [[ ! -f "$RES/metric_uq_clean.txt" ]]; then
    echo "### metric: uq_clean (baseline-only, EVAL split)"
    $PY $UQ/unqover_metric.py --results "$CLEAN" --target-subject Black \
        | tee "$RES/metric_uq_clean.txt"
fi

STEERED=(uq_decode_PI uq_decode_PID uq_decode_P uq_layer_PI_a2 uq_normal_a4 "${BASELINES[@]}")
for M in "${STEERED[@]}"; do
    [[ -f "$RES/$M.jsonl" ]] || { echo "SKIP metric (no $M.jsonl)"; continue; }
    if [[ -f "$RES/metric_$M.txt" ]]; then echo "SKIP metric (exists): $RES/metric_$M.txt"; continue; fi
    echo "### metric: $M vs uq_clean (native, EVAL split)"
    $PY $UQ/unqover_metric.py --results "$RES/$M.jsonl" \
        --baseline "$CLEAN" --target-subject Black \
        | tee "$RES/metric_$M.txt"
done

echo "DONE jobAI: UNQOVER-NATIVE standalone benchmark -> $RES/ (native arrows + fits from the BUILD split; base + decode_{PI,PID,P} + layer_PI_a2 + normal_a4 + 7 baselines on the 157-instance EVAL split; metric_uq_*.txt). unqover_v2 / unqover_full untouched."
