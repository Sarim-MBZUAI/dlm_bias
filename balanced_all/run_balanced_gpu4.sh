#!/usr/bin/env bash
# balanced_all/run_balanced_gpu4.sh -- position-balanced eval on GPU 4:
#   (a) decode-space PID (Black) x 3 existing Black rotations
#       -> results/balanced_all/black/decode_pid/rot{r}
#   (b) arab & white: generate rotations (deterministic; oracle test MUST pass),
#       then per target x 3 rotations run
#         multirace/denoise_pid.py --cond base   -> <target>/base/rot{r}
#         multirace/denoise_pid.py --cond PI     -> <target>/decode_pid/rot{r}
#         multirace/normal.py (arab a4, white a1, the fair coherent doses per
#         multirace/RESULTS.md: white normal a4 is 98.8% unparseable)
#                                                -> <target>/normal/rot{r}
set -euo pipefail

ROOT=/home/lukas/users/shashmi/dlm_bias
PY=/home/lukas/miniconda3/envs/sarim_awm/bin/python
export CUDA_VISIBLE_DEVICES=4
cd "$ROOT"

OUT=results/balanced_all
ROTDIR=$OUT/rotations

run_if_missing() {  # $1 = expected result json, rest = command
    local marker=$1; shift
    if [[ -f "$marker" ]]; then
        echo "SKIP (exists): $marker"
    else
        echo "RUN: $*"
        "$@"
    fi
}

# ---- (a) decode-space PID, Black, existing rotations ----------------------- #
for r in 0 1 2; do
    run_if_missing "$OUT/black/decode_pid/rot${r}/cond_dpid_PID.json" \
        $PY steering/denoise_pid.py --cond PID \
            --items results/balanced/_sweep400_rot${r}.jsonl \
            --out-dir "$OUT/black/decode_pid/rot${r}" --tag dpid_PID
done

# ---- (b) arab / white: rotations (CPU, deterministic) + oracle proof ------- #
for T in arab white; do
    $PY balanced_all/make_rotations.py \
        --items data/bbq_items/_sweep400_${T}.jsonl --out-dir "$ROTDIR"
    $PY balanced_all/oracle_test.py \
        --items-glob "$ROTDIR/_sweep400_${T}_rot*.jsonl" \
        --target "$T" --orig data/bbq_items/_sweep400_${T}.jsonl
done

# ---- (b) arab / white: base, decode-PI, normal (fair dose) ----------------- #
for T in arab white; do
    if [[ "$T" == arab ]]; then ALPHA=4; else ALPHA=1; fi
    for r in 0 1 2; do
        ITEMS=$ROTDIR/_sweep400_${T}_rot${r}.jsonl

        run_if_missing "$OUT/$T/base/rot${r}/cond_dpid_${T}_base.json" \
            $PY multirace/denoise_pid.py --target "$T" --cond base \
                --items "$ITEMS" --out-dir "$OUT/$T/base/rot${r}"

        run_if_missing "$OUT/$T/decode_pid/rot${r}/cond_dpid_${T}_PI.json" \
            $PY multirace/denoise_pid.py --target "$T" --cond PI \
                --items "$ITEMS" --out-dir "$OUT/$T/decode_pid/rot${r}"

        run_if_missing "$OUT/$T/normal/rot${r}/cond_normal_${T}_a${ALPHA}.json" \
            $PY multirace/normal.py --target "$T" --alpha "$ALPHA" \
                --items "$ITEMS" --out-dir "$OUT/$T/normal/rot${r}"
    done
done

echo "ALL DONE (gpu4 queue)"
