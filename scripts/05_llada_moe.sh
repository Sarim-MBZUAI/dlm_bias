#!/usr/bin/env bash
# LLaDA-MoE-7B-A1B-Instruct, Black target, on the LLaDA Black rotation files.
# SWEEPS=1 also runs the unrotated dose sweeps and the preregistered seven-cell
# controller grid that fixed the operating points.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python}
SWEEPS=${SWEEPS:-0}
run_if_missing() { local m=$1; shift; if [[ -f "$m" ]]; then echo "SKIP (exists): $m"; else echo "RUN: $*"; "$@"; fi; }
U=results/lladamoe
OUT=results/lladamoe_balanced
C=llada_moe/baselines/cache

# Direction (16-block arrows) + baseline fits from LLaDA-MoE activations.
run_if_missing llada_moe/arrows.pt          $PY llada_moe/build_arrows.py
run_if_missing $C/calib_block.pt            $PY llada_moe/baselines/calib.py --fit --where block
run_if_missing $C/actadd_dir.pt             $PY llada_moe/baselines/actadd.py --fit
run_if_missing $C/meanact_meandiff_block.pt $PY llada_moe/baselines/meanact.py --fit
run_if_missing $C/linearact_stats.pt        $PY llada_moe/baselines/linearact.py --fit
run_if_missing $C/aura_auroc.pt             $PY llada_moe/baselines/aura.py --fit
run_if_missing $C/itic_probes.pt            $PY llada_moe/baselines/itic.py --fit

if [[ "$SWEEPS" == 1 ]]; then
    # Appendix (LLaDA-MoE operating-point provenance): unrotated dose sweep.
    run_if_missing "$U/decode_pid/cond_dpid_base.json" $PY llada_moe/denoise_pid.py --cond base
    for A in 0.5 1 2 4; do
        TAGA=$(echo "$A" | tr . p)
        run_if_missing "$U/decode_pid/cond_dpid_PI_amax${TAGA}.json" \
            $PY llada_moe/denoise_pid.py --cond PI --amax "$A" --tag "dpid_PI_amax${TAGA}"
    done
    for AL in 1 2 4; do
        run_if_missing "$U/normal/cond_normalL8_a${AL}.json" \
            $PY llada_moe/pid_steer.py --mode normal --source-layer 8 --alpha "$AL" --tag "normalL8_a${AL}" --out-dir "$U/normal"
    done
    for AL in 1 2 4 8; do
        run_if_missing "$U/caa/cond_caa_L8_a${AL}.json" \
            $PY llada_moe/baselines/caa.py --run --alpha "$AL" --layer 8 --out_dir "$U/caa"
    done
    for AL in 1 2 4 8 16; do
        run_if_missing "$U/actadd/cond_actadd_a${AL}.json" \
            $PY llada_moe/baselines/actadd.py --run --alpha "$AL" --layer 8 --out_dir "$U/actadd"
    done
    # Single-site and raw-norm decode-PI probes (fixed the block-8 amax-2 headline).
    run_if_missing "$U/decode_pid/cond_dpid_PI_L8_amax2.json" \
        $PY llada_moe/denoise_pid.py --cond PI --amax 2 --layers 8 --tag dpid_PI_L8_amax2
    run_if_missing "$U/decode_pid/cond_dpid_PI_raw16_amax2.json" \
        $PY llada_moe/denoise_pid.py --cond PI --amax 2 --layer-scale raw --tag dpid_PI_raw16_amax2
fi

# Appendix (LLaDA-MoE protocol): preregistered seven-cell decode-PI grid, unrotated items.
grid() {  # $1 = tag, rest = flags
    local tag=$1; shift
    run_if_missing "$U/decode_pid/cond_${tag}.json" $PY llada_moe/denoise_pid.py --cond PI "$@" --tag "$tag"
}
if [[ "$SWEEPS" == 1 ]]; then
    grid dpid_PI_L8_amax3     --amax 3 --layers 8
    grid dpid_PI_L10_amax4    --amax 4 --layers 10
    grid dpid_PI_L8_amax4     --amax 4 --layers 8
    grid dpid_PI_L789_amax1   --amax 1 --layers 7,8,9
    grid dpid_PI_L10_amax2    --amax 2 --layers 10
    grid dpid_PI_L9to12_amax1 --amax 1 --layers 9-12
    grid dpid_PI_L12_amax4    --amax 4 --layers 12
fi

for r in 0 1 2; do
    ITEMS=results/balanced/_sweep400_rot${r}.jsonl

    # Table 3 + cross-model appendix: base, decode PI (block 8, amax 2), matched open loop
    # (block 8, a2), CAA L8 a2, ActAdd a2, remaining baselines at the LLaDA operating points.
    run_if_missing "$OUT/base/rot${r}/cond_dpid_base.json" \
        $PY llada_moe/denoise_pid.py --cond base --items "$ITEMS" --out-dir "$OUT/base/rot${r}"
    run_if_missing "$OUT/decode_pid/rot${r}/cond_dpid_PI.json" \
        $PY llada_moe/denoise_pid.py --cond PI --layers 8 --amax 2 --items "$ITEMS" --out-dir "$OUT/decode_pid/rot${r}"
    run_if_missing "$OUT/normal/rot${r}/cond_normalL8_a2.json" \
        $PY llada_moe/pid_steer.py --mode normal --source-layer 8 --layers 8 --alpha 2 --items "$ITEMS" --out-dir "$OUT/normal/rot${r}"
    run_if_missing "$OUT/caa/rot${r}/cond_caa_L8_a2.json" \
        $PY llada_moe/baselines/caa.py --run --alpha 2 --layer 8 --items "$ITEMS" --out_dir "$OUT/caa/rot${r}"
    run_if_missing "$OUT/actadd/rot${r}/cond_actadd_a2.json" \
        $PY llada_moe/baselines/actadd.py --run --alpha 2 --layer 8 --items "$ITEMS" --out_dir "$OUT/actadd/rot${r}"
    run_if_missing "$OUT/meanact/rot${r}/cond_meanact_unit_s2.json" \
        $PY llada_moe/baselines/meanact.py --run --direction unit --strength 2 --items "$ITEMS" --out_dir "$OUT/meanact/rot${r}"
    run_if_missing "$OUT/linearact/rot${r}/cond_gaussian_s1.json" \
        $PY llada_moe/baselines/linearact.py --run --variant gaussian --strength 1 --items "$ITEMS" --out_dir "$OUT/linearact/rot${r}"
    run_if_missing "$OUT/aura_inject/rot${r}/cond_inject_g4.json" \
        $PY llada_moe/baselines/aura.py --run --mode inject --gamma 4 --items "$ITEMS" --out_dir "$OUT/aura_inject/rot${r}"
    run_if_missing "$OUT/aura_vanilla/rot${r}/cond_vanilla.json" \
        $PY llada_moe/baselines/aura.py --run --mode vanilla --items "$ITEMS" --out_dir "$OUT/aura_vanilla/rot${r}"
    run_if_missing "$OUT/itic/rot${r}/cond_itic_K48_a8.json" \
        $PY llada_moe/baselines/itic.py --run --topk 48 --alpha 8 --items "$ITEMS" --out_dir "$OUT/itic/rot${r}"

    # Cross-model appendix: disclosed secondary (blocks 9-12, amax 1) + its matched open loop a1.
    run_if_missing "$OUT/decode_pid_L9to12/rot${r}/cond_dpid_PI_L9to12.json" \
        $PY llada_moe/denoise_pid.py --cond PI --layers 9-12 --amax 1 --tag dpid_PI_L9to12 \
            --items "$ITEMS" --out-dir "$OUT/decode_pid_L9to12/rot${r}"
    run_if_missing "$OUT/normal_L9to12/rot${r}/cond_normal_L9to12_a1.json" \
        $PY llada_moe/pid_steer.py --mode normal --source-layer 8 --layers 9-12 --alpha 1 --tag normal_L9to12_a1 \
            --items "$ITEMS" --out-dir "$OUT/normal_L9to12/rot${r}"
done

echo "DONE (05_llada_moe)"
