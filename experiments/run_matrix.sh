#!/usr/bin/env bash
# Comparison matrix: every steering method vs OURS, across BBQ + UNQOVER.
#
#   rows (methods)      : clean · CAA · ActAdd · group-mean-diff · OURS-clamp-all · OURS-cmom-all
#   cols (benchmarks)   : BBQ (eval/bbq_eval.py --items) · UNQOVER (datasets/unqover/unqover_eval.py)
#
# Each BASELINE runs at ITS OWN layer+alpha (read from baselines/*/config.json).
# OURS runs FULL-LAYER closed-loop (--layers all, clamp/cmom); the answer-text
# direction dir has the co-located nat_proj_by_layer.json needed for per-layer c*.
#
# GPU RULE: ONLY CUDA_VISIBLE_DEVICES 5, 6, or 7.  ~45-60 min per BBQ run (n=1600);
# UNQOVER runtime depends on --limit.  Two runs fit on one 80 GB card.
set -euo pipefail
PY=/home/lukas/miniconda3/envs/sarim_awm/bin/python
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
OUT=experiments/results/matrix; mkdir -p "$OUT"

BBQ_ITEMS=experiments/data/black_referent_ambig_eval.jsonl
UQ_ITEMS=datasets/unqover/data/ethnicity.items.jsonl
OURS_DIR=directional_steering/race_black_anchored_text.pt   # has co-located nat_proj_by_layer.json
TARGET=African                                              # UNQOVER target subject

# --- prerequisites -----------------------------------------------------------
[ -f "$BBQ_ITEMS" ] || $PY experiments/build_eval_set.py
if [ ! -f "$UQ_ITEMS" ]; then
  echo "UNQOVER items missing. Run first (CPU):"
  echo "  $PY datasets/unqover/download_unqover.py --classes ethnicity"
  echo "  $PY datasets/unqover/unqover_loader.py --source datasets/unqover/data/generated/ethnicity.source.json --out $UQ_ITEMS --limit 2000 --seed 42"
  exit 1
fi
# baseline directions must be built (GPU) via baselines/run_baselines.sh
CAA_L=$($PY -c "import json;print(json.load(open('baselines/caa/config.json'))['layer'])")
CAA_A=$($PY -c "import json;print(json.load(open('baselines/caa/config.json'))['alpha'])")
AA_L=$($PY -c "import json;print(json.load(open('baselines/actadd/config.json'))['layer'])")
AA_A=$($PY -c "import json;print(json.load(open('baselines/actadd/config.json'))['alpha'])")
GM_L=$($PY -c "import json;print(json.load(open('baselines/group_meandiff/config.json'))['layer'])")
GM_A=$($PY -c "import json;print(json.load(open('baselines/group_meandiff/config.json'))['alpha'])")
GM_DIR=bias_steering/directions/L14/race_color.pt
for pt in baselines/caa/caa_direction.pt baselines/actadd/actadd_direction.pt "$GM_DIR"; do
  [ -f "$pt" ] || { echo "Missing $pt — run baselines/run_baselines.sh (GPU) first."; exit 1; }
done

# --- BBQ column --------------------------------------------------------------
bbq () { local gpu=$1 stem=$2; shift 2
  CUDA_VISIBLE_DEVICES=$gpu $PY eval/bbq_eval.py --items "$BBQ_ITEMS" \
    --out "$OUT/$stem.json" "$@"; }
# --- UNQOVER column ----------------------------------------------------------
uq () { local gpu=$1 stem=$2; shift 2
  CUDA_VISIBLE_DEVICES=$gpu $PY datasets/unqover/unqover_eval.py --items "$UQ_ITEMS" \
    --out "$OUT/$stem.jsonl" "$@"; }

echo "== BBQ column =="
bbq 5 bbq_clean         --alpha 0 &
bbq 6 bbq_caa           --direction-path baselines/caa/caa_direction.pt      --layer "$CAA_L" --alpha "$CAA_A" &
bbq 7 bbq_actadd        --direction-path baselines/actadd/actadd_direction.pt --layer "$AA_L"  --alpha "$AA_A" &
wait
bbq 5 bbq_group         --direction-path "$GM_DIR" --layer "$GM_L" --alpha "$GM_A" &
bbq 6 bbq_ours_clamp_all --direction-path "$OURS_DIR" --layers all --steer-mode clamp --cstar 60 &
bbq 7 bbq_ours_cmom_all  --direction-path "$OURS_DIR" --layers all --steer-mode cmom  --cstar 60 --beta 0.8 &
wait

echo "== UNQOVER column =="
uq 5 uq_clean          &
uq 6 uq_caa            --direction-path baselines/caa/caa_direction.pt      --layer "$CAA_L" --alpha "$CAA_A" &
uq 7 uq_actadd         --direction-path baselines/actadd/actadd_direction.pt --layer "$AA_L"  --alpha "$AA_A" &
wait
uq 5 uq_group          --direction-path "$GM_DIR" --layer "$GM_L" --alpha "$GM_A" &
uq 6 uq_ours_clamp_all --direction-path "$OURS_DIR" --layers all --steer-mode clamp --cstar 60 &
uq 7 uq_ours_cmom_all  --direction-path "$OURS_DIR" --layers all --steer-mode cmom  --cstar 60 --beta 0.8 &
wait

echo "== building comparison table =="
$PY experiments/build_comparison_table.py --target-subject "$TARGET"
