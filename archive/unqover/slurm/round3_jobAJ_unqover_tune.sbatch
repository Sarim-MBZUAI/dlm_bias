#!/bin/bash
#SBATCH --job-name=uqtune
#SBATCH --account=nils
#SBATCH --partition=gpu
#SBATCH --qos=normal-plus
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --output=/shared/home/sarim.hashmi/dlm_bias/logs/slurm/%x-%j.out
#SBATCH --error=/shared/home/sarim.hashmi/dlm_bias/logs/slurm/%x-%j.err

# ROUND 3 JOB AJ "uqtune": TUNE the UNQOVER decode-PI controller on the BUILD
# split and run the tuned controller + its EFFORT-MATCHED open-loop control on
# the held-out EVAL split. Mirrors jobAI conventions (env, cd main tree,
# run_if_missing idempotency, qos normal-plus, 1 GPU).
#
# WHY: the native decode-PI run (uq_decode_PI, Kp3/Ki0.1/Kd0/amax6/s*0.9) realizes
# mean alpha=2.71, sat_frac=0.11 and is BIMODAL -- 51% of items reach p>=0.9 then
# back off, 48% never move (p_final<0.3, alpha stuck mid-range, NOT escalating to
# amax). It UNDER-ACTUATES on hard items (Ki too weak / Kd=0). And the open-loop
# control it was compared against ran at a FIXED alpha=4 > the realized 2.71 -- an
# unfair over-strength comparator. This job (1) sweeps a small diagnosis-driven
# gain grid on BUILD, picks the winner by a PRE-REGISTERED rule
# (unqover/TUNING_PREREG_jobAJ.md), (2) runs that tuned config on EVAL, and (3)
# runs the fair EFFORT-MATCHED open loop at the controller's EXACT realized mean
# alpha (fractional --alpha, no integer rounding) -- both for the DEFAULT PI
# (alpha=2.71) and for the TUNED controller (its own realized mean).
#
# CONTAMINATION: tuning reads the BUILD split ONLY (105 inst / 420 items,
# disjoint from the 157-inst EVAL split -- verified). arrows_unqover_black.pt was
# fit on BUILD by jobAI. The EVAL split is read only for the final chosen config +
# its effort-matched control. NOTHING existing is overwritten (new stems only).
#
# GPU ESTIMATE (~1.6 s/item):
#   sweep   8 configs x 420 build items ~11 min each        ~1.5 h
#   eval    tuned decode-PI (628) ~17 min                   ~0.3 h
#           2 effort-matched open loops (628 each) ~16 min   ~0.55 h
#   TOTAL   ~2.4 GPU-hours (within the 6 h walltime; sweep+eval resume via markers)
#
# IDEMPOTENT: every GPU step is run_if_missing on its .json sidecar (written only
# on completion). The BUILD-split winner is chosen programmatically by
# select_tuned_pi.py -> CHOSEN.json; the EVAL runs read CHOSEN.json, so the whole
# pipeline is deterministic and resumable after a timeout.
set -euo pipefail

export HF_HOME=/shared/models/huggingface
export HF_HUB_CACHE=/shared/models/huggingface/hub
export HF_MODULES_CACHE=/shared/home/sarim.hashmi/.cache/hf_modules
export DLM_BIAS_ROOT=/shared/home/sarim.hashmi/dlm_bias
cd "$DLM_BIAS_ROOT"
PY=/shared/home/sarim.hashmi/anaconda3/envs/dlm/bin/python
# Do NOT set CUDA_VISIBLE_DEVICES -- SLURM sets it.

UQ=unqover
RES=results/unqover_native                       # jobAI's native benchmark dir
SWEEP=$RES/sweep                                 # BUILD-split tuning artifacts
ARROWS=$RES/arrows_unqover_black.pt              # UNQOVER-native direction (BUILD-fit)
BUILD_ITEMS=$RES/build_items.jsonl               # 105-inst tuning split
EVAL_ITEMS=$RES/eval_items.jsonl                 # 157-inst held-out eval split
CLEAN=$RES/uq_clean.jsonl                        # jobAI's EVAL clean baseline (for metric)
DEFAULT_PI=$RES/uq_decode_PI.jsonl               # existing default-PI EVAL run (kept)
CHOSEN=$SWEEP/CHOSEN.json
mkdir -p "$SWEEP"

# jobAI must have produced the native direction + splits first.
for f in "$ARROWS" "$BUILD_ITEMS" "$EVAL_ITEMS" "$CLEAN" "$DEFAULT_PI"; do
    test -f "$f" || { echo "FATAL: $f missing -- run round3_jobAI_unqover_native.sbatch first"; exit 1; }
done

run_if_missing() {  # $1 = marker (written only on completion), rest = command
    local marker=$1; shift
    if [[ -f "$marker" ]]; then echo "SKIP (exists): $marker"; else echo "RUN: $*"; "$@"; fi
}

# Common decode-PI run flags (native arrows, BUILD or EVAL items, fixed s*0.9 / L14).
DPI_COMMON=(--arrows "$ARROWS" --target Black)

# --------------------------------------------------------------------------- #
# 1) BUILD-SPLIT SWEEP: 8 diagnosis-driven configs (C0 = default reference).    #
#    cond/Kp/Ki/Kd/amax per unqover/TUNING_PREREG_jobAJ.md. setpoint fixed 0.9. #
#    For PI configs --kd is masked off by COND_MASK; passed only for readability.#
# --------------------------------------------------------------------------- #
# tag  cond  Kp  Ki   Kd  amax
SWEEP_CONFIGS=(
    "C0 PI  3 0.1 0 6"
    "C1 PI  3 0.3 0 6"
    "C2 PI  3 0.5 0 6"
    "C3 PI  5 0.3 0 6"
    "C4 PI  5 0.5 0 6"
    "C5 PID 3 0.3 1 6"
    "C6 PI  3 0.5 0 8"
    "C7 PID 5 0.5 1 8"
)
for spec in "${SWEEP_CONFIGS[@]}"; do
    read -r TAG COND KP KI KD AMAX <<< "$spec"
    run_if_missing "$SWEEP/build_${TAG}.json" \
        $PY $UQ/denoise_pid_unqover.py --cond "$COND" \
            --kp "$KP" --ki "$KI" --kd "$KD" --amax "$AMAX" \
            --items "$BUILD_ITEMS" "${DPI_COMMON[@]}" \
            --out "$SWEEP/build_${TAG}.jsonl"
done

# --------------------------------------------------------------------------- #
# 2) SELECT the winner deterministically from the BUILD-split sweep outputs.    #
#    Pre-registered rule in select_tuned_pi.py -> CHOSEN.json (+ printed table). #
# --------------------------------------------------------------------------- #
run_if_missing "$CHOSEN" \
    $PY $UQ/select_tuned_pi.py --sweep-dir "$SWEEP" --ref-tag C0 \
        --target Black --out "$CHOSEN"
# Always refresh the human-readable table (cheap, deterministic, read-only run).
$PY $UQ/select_tuned_pi.py --sweep-dir "$SWEEP" --ref-tag C0 --target Black \
    > "$SWEEP/sweep_table.txt"
cat "$SWEEP/sweep_table.txt"
test -f "$CHOSEN" || { echo "FATAL: selector did not write $CHOSEN"; exit 1; }

# Extract the chosen gains into shell vars (deterministic; from CHOSEN.json).
eval "$($PY - "$CHOSEN" <<'PYEOF'
import json, sys
c = json.load(open(sys.argv[1]))
kd = c.get("Kd") or 0
print(f"WCOND={c['cond']} WKP={c['Kp']} WKI={c['Ki']} WKD={kd} WAMAX={c['amax']} WTAG={c['tag']}")
PYEOF
)"
echo "CHOSEN tuned config: tag=$WTAG cond=$WCOND Kp=$WKP Ki=$WKI Kd=$WKD amax=$WAMAX"

# --------------------------------------------------------------------------- #
# 3) FINAL EVAL: the tuned decode-PI config on the held-out EVAL split.         #
#    NEW stem uq_decode_PI_tuned -- does NOT touch the existing uq_decode_PI.    #
# --------------------------------------------------------------------------- #
TUNED=$RES/uq_decode_PI_tuned.jsonl
run_if_missing "$RES/uq_decode_PI_tuned.json" \
    $PY $UQ/denoise_pid_unqover.py --cond "$WCOND" \
        --kp "$WKP" --ki "$WKI" --kd "$WKD" --amax "$WAMAX" \
        --items "$EVAL_ITEMS" "${DPI_COMMON[@]}" --out "$TUNED"
test -f "$TUNED" || { echo "FATAL: $TUNED not produced"; exit 1; }

# --------------------------------------------------------------------------- #
# 4) EFFORT-MATCHED OPEN LOOPS at the controllers' EXACT realized mean alpha.    #
#    (a) default PI (alpha = mean realized alpha of the existing EVAL PI run)    #
#    (b) tuned      (alpha = mean realized alpha of the tuned EVAL run above)    #
#    pid_steer_unqover.py --mode normal --alpha accepts fractional alpha, so the #
#    match is exact -- no integer rounding.                                      #
# --------------------------------------------------------------------------- #
mean_alpha() {  # $1 = decode-PI jsonl -> mean over items of alpha_mean
    $PY - "$1" <<'PYEOF'
import json, sys
xs = [json.loads(l)["alpha_mean"] for l in open(sys.argv[1]) if l.strip()]
print("%.4f" % (sum(xs) / len(xs)))
PYEOF
}

ALPHA_DEF=$(mean_alpha "$DEFAULT_PI")
ALPHA_TUNED=$(mean_alpha "$TUNED")
echo "effort-matched alpha: default=$ALPHA_DEF  tuned=$ALPHA_TUNED"

run_if_missing "$RES/uq_normal_effort.json" \
    $PY $UQ/pid_steer_unqover.py --mode normal --alpha "$ALPHA_DEF" \
        --arrows "$ARROWS" --items "$EVAL_ITEMS" --target Black \
        --out "$RES/uq_normal_effort.jsonl"

run_if_missing "$RES/uq_normal_effort_tuned.json" \
    $PY $UQ/pid_steer_unqover.py --mode normal --alpha "$ALPHA_TUNED" \
        --arrows "$ARROWS" --items "$EVAL_ITEMS" --target Black \
        --out "$RES/uq_normal_effort_tuned.jsonl"

# --------------------------------------------------------------------------- #
# 5) METRICS (CPU): tuned controller + both effort-matched controls vs clean.   #
# --------------------------------------------------------------------------- #
FINALS=(uq_decode_PI_tuned uq_normal_effort uq_normal_effort_tuned)
for M in "${FINALS[@]}"; do
    [[ -f "$RES/$M.jsonl" ]] || { echo "SKIP metric (no $M.jsonl)"; continue; }
    if [[ -f "$RES/metric_$M.txt" ]]; then echo "SKIP metric (exists): $RES/metric_$M.txt"; continue; fi
    echo "### metric: $M vs uq_clean (native, EVAL split)"
    $PY $UQ/unqover_metric.py --results "$RES/$M.jsonl" \
        --baseline "$CLEAN" --target-subject Black \
        | tee "$RES/metric_$M.txt"
done

echo "DONE jobAJ: BUILD-split sweep (${#SWEEP_CONFIGS[@]} configs) -> CHOSEN=$WTAG; EVAL runs uq_decode_PI_tuned + uq_normal_effort(a=$ALPHA_DEF) + uq_normal_effort_tuned(a=$ALPHA_TUNED); metrics in $RES/. Existing uq_decode_PI / uq_normal_a4 untouched."
