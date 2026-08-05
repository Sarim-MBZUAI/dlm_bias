#!/usr/bin/env bash
# Phase-3 GPU-4 queue: the 6 faithful baselines on Dream-v0-Instruct-7B, each via
# its OWN CLI (fit -> run) at a representative faithful strength matching the LLaDA
# results/BASELINES.md picks. (run_all.py's fit/strength dispatch is unreliable —
# meanact/linearact need --strength, actadd/itic/aura need their own --fit — so we
# drive each method directly, which is how the LLaDA baselines were actually run.)
# Waits for dream/arrows.pt. Env pinned to sarim_awm (transformers 4.46.2).
set -u
cd "${DLM_BIAS_ROOT:-$(dirname "$0")/..}"
PY="${PY:-python}"
export CUDA_VISIBLE_DEVICES=4
L=logs/gpu4; mkdir -p "$L"
B=dream/baselines

echo "[gpu4] waiting for dream/arrows.pt ..."
while [ ! -f dream/arrows.pt ]; do sleep 15; done
echo "[gpu4] arrows ready; running 6 baselines per-method $(date)"

# shared block-granularity calib cache (only meanact --direction fitted needs it;
# built for completeness so the fitted variant is reproducible).
$PY $B/calib.py --fit --where block > "$L/calib_block.log" 2>&1 && echo "[gpu4] calib block   DONE"

# CAA -- single-layer diff-in-means, faithful mult 2 (arrows.pt, no fit).
$PY $B/caa.py --run --layer 14 --alpha 2 > "$L/caa_a2.log" 2>&1 && echo "[gpu4] caa a2        DONE"

# Mean-AcT -- raw diff-in-means, all 28 blocks (arrows.pt, no fit).
$PY $B/meanact.py --run --direction raw --strength 2 > "$L/meanact.log" 2>&1 && echo "[gpu4] meanact raw   DONE"

# ActAdd -- single contrast pair (n=1); needs its own fit.
$PY $B/actadd.py --fit > "$L/actadd_fit.log" 2>&1 \
  && $PY $B/actadd.py --run --alpha 8 > "$L/actadd.log" 2>&1 && echo "[gpu4] actadd a8     DONE"

# Linear-AcT -- per-neuron 1-D OT on MLP-hidden (gaussian); fit collects mlp_hidden.
$PY $B/linearact.py --fit > "$L/linearact_fit.log" 2>&1 \
  && $PY $B/linearact.py --run --variant gaussian --strength 2 > "$L/linearact_g.log" 2>&1 && echo "[gpu4] linearact g   DONE"

# AURA -- per-neuron AUROC gate on MLP-hidden; inject (headline) + vanilla (control).
$PY $B/aura.py --fit > "$L/aura_fit.log" 2>&1 \
  && $PY $B/aura.py --run --mode inject --gamma 4 > "$L/aura_inject.log" 2>&1 \
  && $PY $B/aura.py --run --mode vanilla --tag vanilla > "$L/aura_vanilla.log" 2>&1 && echo "[gpu4] aura inj+van  DONE"

# ITI-C -- per-head contrastive attn shift; top-48 heads, alpha 16.
$PY $B/itic.py --fit > "$L/itic_fit.log" 2>&1 \
  && $PY $B/itic.py --run --topk 48 --alpha 16 > "$L/itic.log" 2>&1 && echo "[gpu4] itic k48a16   DONE"

echo "[gpu4] ALL DONE $(date)"
