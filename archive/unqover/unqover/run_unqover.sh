#!/usr/bin/env bash
# UNQOVER benchmark: fetch data -> load -> clean + our full-layer method
# (+ baseline placeholders) -> analyze with the official metric.
#
# GPU RULE: ONLY use CUDA_VISIBLE_DEVICES 5, 6, or 7.
# Generation is the USER's to run (needs a GPU + LLaDA weights). The download +
# loader + metric steps are CPU-only.
set -euo pipefail
PY="${PY:-python}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
UQ=unqover
DATA=data/unqover
RES=results/unqover; mkdir -p "$RES"

CLASS=${CLASS:-ethnicity}                 # ethnicity | religion | country | gender
LIMIT=${LIMIT:-2000}                      # number of instances (x4 items) to eval
TARGET=${TARGET:-Black}                   # subject for the BBQ-comparable directional
                                          # gap (matches the PID adapters' default)
# Direction: steering/arrows.pt (the Black direction). It is an arrows-format
# blob; unqover_eval.py derives unit(r[L14]) from it -- the same vhat the PID
# adapters use. Override DIR with any {"direction": ...} .pt or arrows-format .pt.
DIR=${DIR:-steering/arrows.pt}
LAYER=${LAYER:-14}

SRC=$DATA/generated/$CLASS.source.json
ITEMS=$DATA/$CLASS.items.jsonl

# 0) Fetch official UNQOVER data (clones allenai/unqover @ pinned commit, runs
#    its generator). CPU. Skipped if already present.
[ -f "$SRC" ] || $PY $UQ/download_unqover.py --classes "$CLASS"

# 1) Convert to our --items jsonl (keeps position/negation quadruples). CPU.
[ -f "$ITEMS" ] || $PY $UQ/unqover_loader.py --source "$SRC" --out "$ITEMS" --limit "$LIMIT" --seed 42

run () {  # $1=gpu  $2=out-stem  $3...=extra eval args
  local gpu=$1 stem=$2; shift 2
  CUDA_VISIBLE_DEVICES=$gpu $PY $UQ/unqover_eval.py \
    --items "$ITEMS" --out "$RES/$stem.jsonl" "$@"
}

# 2) Clean baseline + OUR full-layer method (clamp / cmom, --layers all).
run 5 uq_clean                                                         &
run 6 uq_clamp_all --steer-mode clamp --layers all --cstar 60 --direction-path "$DIR" &
run 7 uq_cmom_all  --steer-mode cmom  --layers all --cstar 60 --beta 0.8 --direction-path "$DIR" &
wait
# Single-layer open-loop reference at block L14 (optional).
run 5 uq_open_L14  --alpha 8 --layer "$LAYER" --direction-path "$DIR"   &
wait

# 3) BASELINE PLACEHOLDERS (fill in when the baselines are wired for UNQOVER):
#    e.g. Ghostwriter input-space attack, or a prompt-prefix debias baseline.
#    run 6 uq_ghostwriter --attack ghostwriter    # <- TODO: add --attack to unqover_eval

# 4) Official metric + BBQ-comparable directional gap (steered vs clean). CPU.
echo "== UNQOVER analysis (official mu/eta/delta + directional gap) =="
for stem in uq_clamp_all uq_cmom_all uq_open_L14; do
  [ -f "$RES/$stem.jsonl" ] || continue
  echo "### $stem vs clean"
  $PY $UQ/unqover_metric.py --results "$RES/$stem.jsonl" \
      --baseline "$RES/uq_clean.jsonl" --target-subject "$TARGET"
done
