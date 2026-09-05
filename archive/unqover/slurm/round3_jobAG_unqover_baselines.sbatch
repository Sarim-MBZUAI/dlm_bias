#!/bin/bash
#SBATCH --job-name=uqbase
#SBATCH --account=nils
#SBATCH --partition=gpu
#SBATCH --qos=normal-plus
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --output=/shared/home/sarim.hashmi/dlm_bias/logs/slurm/%x-%j.out
#SBATCH --error=/shared/home/sarim.hashmi/dlm_bias/logs/slurm/%x-%j.err

# ROUND 3 JOB AG "uqbase" (no dependency): give the UNQOVER benchmark the SAME
# prior-work baseline suite the BBQ Table-1 (Black) has, so the paper's UNQOVER
# table can show ALL methods instead of only the 4 already run (uq_clean,
# uq_decode_PI, uq_layer_PI_a2, uq_normal_a4).
#
# The 7 prior-work baselines are run through unqover/baselines_unqover.py, which
# is a THIN ADAPTER: it IMPORTS each baseline's attach_fn from baselines/<m>.py
# at the EXACT Table-1 (Black) operating point, REUSING the black fit artifacts
# already on disk (NO re-fit), and only swaps the BBQ generation glue for the
# UNQOVER 2-choice glue (denoise_pid_unqover.py). Records are byte-schema
# identical to uq_decode_PI.jsonl, so unqover_metric.py consumes them unchanged.
#
# METHODS (stem -> operating point -> black fit artifact):
#   uq_caa          caa L14 a16          steering/arrows.pt
#   uq_actadd       actadd L14 a16       baselines/cache/actadd_dir.pt
#   uq_meanact      meanact unit s2      steering/arrows.pt
#   uq_linearact    linearact gaussian s1 baselines/cache/linearact_stats.pt
#   uq_aura_inject  aura inject g4       baselines/cache/aura_auroc.pt
#   uq_aura_vanilla aura vanilla         baselines/cache/aura_auroc.pt
#   uq_itic         itic K48 a8          baselines/cache/itic_probes.pt
# = 7 runs, single GPU. Item set = the SAME cap2000 file + Black subject filter
# the existing steered uq runs used (n_items=256 / 64 instances), so the new
# rows sit beside uq_decode_PI / uq_normal_a4 in the UNQOVER table. Each run is
# item-independent (fixed injection, no closed loop) -> ~6-7 min/run like
# uq_normal_a4 (elapsed ~381 s). aura/linearact/itic touch all 32 layers so are
# a touch slower; 06:00:00 is ample headroom for all 7.
#
# IDEMPOTENT: run_if_missing keys on the .json CONFIG (written ONLY after a run
# completes, so a partially-written .jsonl re-runs cleanly), the same
# completion-marker convention as round3_jobW / round2_jobE.
set -euo pipefail

export HF_HOME=/shared/models/huggingface
export HF_HUB_CACHE=/shared/models/huggingface/hub
export HF_MODULES_CACHE=/shared/home/sarim.hashmi/.cache/hf_modules
export DLM_BIAS_ROOT=/shared/home/sarim.hashmi/dlm_bias
cd "$DLM_BIAS_ROOT"
PY=/shared/home/sarim.hashmi/anaconda3/envs/dlm/bin/python
# Do NOT set CUDA_VISIBLE_DEVICES -- SLURM sets it.

UQ=unqover
RES=results/unqover_v2
CLEAN=$RES/uq_clean.jsonl                 # position/negation-debiased clean baseline
ITEMS=data/unqover/ethnicity.items.cap2000.jsonl
mkdir -p "$RES"

test -f "$ITEMS"  || { echo "FATAL: $ITEMS missing (unqover_loader must have run)"; exit 1; }
test -f "$CLEAN"  || { echo "FATAL: $CLEAN missing (clean uq baseline needed for metric)"; exit 1; }

run_if_missing() {  # $1 = expected result .json (completion marker), rest = command
    local marker=$1; shift
    if [[ -f "$marker" ]]; then
        echo "SKIP (exists): $marker"
    else
        echo "RUN: $*"
        "$@"
    fi
}

METHODS=(uq_caa uq_actadd uq_meanact uq_linearact uq_aura_inject uq_aura_vanilla uq_itic)

# 1) Run the 7 baselines through the UNQOVER eval harness (single GPU, serial).
for M in "${METHODS[@]}"; do
    run_if_missing "$RES/$M.json" \
        $PY $UQ/baselines_unqover.py --method "$M" --items "$ITEMS"
done

# 2) Official metric + BBQ-comparable directional gap vs the clean uq baseline
#    (CPU). Writes metric_uq_<method>.txt beside the existing metric_uq_*.txt so
#    the paper table can read the same columns (mu/eta/delta/mean|C| +
#    pref_gap toward 'Black', raw & debiased). Idempotent: skip if present.
echo "== UNQOVER analysis (official mu/eta/delta + directional gap vs clean) =="
for M in "${METHODS[@]}"; do
    [ -f "$RES/$M.jsonl" ] || { echo "SKIP metric (no $M.jsonl)"; continue; }
    if [[ -f "$RES/metric_$M.txt" ]]; then
        echo "SKIP metric (exists): $RES/metric_$M.txt"
        continue
    fi
    echo "### $M vs clean"
    $PY $UQ/unqover_metric.py --results "$RES/$M.jsonl" \
        --baseline "$CLEAN" --target-subject Black | tee "$RES/metric_$M.txt"
done

echo "DONE jobAG: 7 UNQOVER baselines -> $RES/uq_{caa,actadd,meanact,linearact,aura_inject,aura_vanilla,itic}.jsonl (+ metric_*.txt)"
