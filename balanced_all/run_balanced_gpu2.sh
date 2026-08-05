#!/usr/bin/env bash
# balanced_all/run_balanced_gpu2.sh -- LLaDA Black-target baselines through the
# position-balanced eval: 7 conditions x 3 rotations = 21 runs on GPU 2.
#
# Each condition is run at the EXACT operating point of the published raw table
# (results/BASELINES.md; verified against the cond_*.json configs that produced
# the table's rates):
#   caa        --alpha 16 --layer 14        (~ paper multiplier 2, ||r[14]||~8.5)
#   actadd     --alpha 16 --layer 14        (~ paper multiplier 2)
#   meanact    --direction unit --strength 2  (published row = cond_s2.json,
#              direction_source 'unit'; the raw_s* runs do NOT match the table)
#   linearact  --variant gaussian --strength 1 (published row = gaussian_s1;
#              gaussian degenerates at strength >= 2)
#   aura       --mode inject --gamma 4
#   aura       --mode vanilla               (negative control)
#   itic       --topk 48 --alpha 8          (published row = topk48_a8)
#
# Fit caches in baselines/cache/ are REUSED -- no --fit is run here.
# Items: the 3 existing Black rotation files results/balanced/_sweep400_rot{r}.jsonl.
# Output: results/balanced_all/black/<method>/rot{r}/cond_<default-tag>.json
# Aggregate afterwards with balanced_all/aggregate.py (see README.md).
set -euo pipefail

ROOT="${DLM_BIAS_ROOT:-$(dirname "$0")/..}"
PY="${PY:-python}"
export CUDA_VISIBLE_DEVICES=2
cd "$ROOT"

OUT=results/balanced_all/black

run_if_missing() {  # $1 = expected result json, rest = command
    local marker=$1; shift
    if [[ -f "$marker" ]]; then
        echo "SKIP (exists): $marker"
    else
        echo "RUN: $*"
        "$@"
    fi
}

for r in 0 1 2; do
    ITEMS=results/balanced/_sweep400_rot${r}.jsonl

    run_if_missing "$OUT/caa/rot${r}/cond_caa_L14_a16.json" \
        $PY baselines/caa.py --run --alpha 16 --layer 14 \
            --items "$ITEMS" --out_dir "$OUT/caa/rot${r}"

    run_if_missing "$OUT/actadd/rot${r}/cond_actadd_a16.json" \
        $PY baselines/actadd.py --run --alpha 16 --layer 14 \
            --items "$ITEMS" --out-dir "$OUT/actadd/rot${r}"

    run_if_missing "$OUT/meanact/rot${r}/cond_meanact_unit_s2.json" \
        $PY baselines/meanact.py --run --direction unit --strength 2 \
            --items "$ITEMS" --out-dir "$OUT/meanact/rot${r}"

    run_if_missing "$OUT/linearact/rot${r}/cond_gaussian_s1.json" \
        $PY baselines/linearact.py --run --variant gaussian --strength 1 \
            --items "$ITEMS" --out_dir "$OUT/linearact/rot${r}"

    run_if_missing "$OUT/aura_inject/rot${r}/cond_inject_g4.json" \
        $PY baselines/aura.py --run --mode inject --gamma 4 \
            --items "$ITEMS" --out-dir "$OUT/aura_inject/rot${r}"

    run_if_missing "$OUT/aura_vanilla/rot${r}/cond_vanilla.json" \
        $PY baselines/aura.py --run --mode vanilla \
            --items "$ITEMS" --out-dir "$OUT/aura_vanilla/rot${r}"

    run_if_missing "$OUT/itic/rot${r}/cond_itic_K48_a8.json" \
        $PY baselines/itic.py --run --topk 48 --alpha 8 \
            --items "$ITEMS" --out_dir "$OUT/itic/rot${r}"
done

echo "ALL DONE (gpu2 queue)"
