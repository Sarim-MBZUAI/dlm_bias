#!/usr/bin/env bash
# Dream-v0-Instruct-7B, Black target, on the LLaDA Black rotation files.
# SWEEPS=1 also runs the unrotated decode-PI amax sweep that fixed amax=1.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python}
SWEEPS=${SWEEPS:-0}
run_if_missing() { local m=$1; shift; if [[ -f "$m" ]]; then echo "SKIP (exists): $m"; else echo "RUN: $*"; "$@"; fi; }
OUT=results/dream_balanced
C=dream/baselines/cache

# Direction (Dream diff-in-means arrows) + baseline fits from Dream activations.
run_if_missing dream/arrows.pt                $PY dream/build_arrows.py
run_if_missing $C/calib_block.pt              $PY dream/baselines/calib.py --fit --where block
run_if_missing $C/actadd_dir.pt               $PY dream/baselines/actadd.py --fit
run_if_missing $C/meanact_meandiff_block.pt   $PY dream/baselines/meanact.py --fit
run_if_missing $C/linearact_stats.pt          $PY dream/baselines/linearact.py --fit
run_if_missing $C/aura_auroc.pt               $PY dream/baselines/aura.py --fit
run_if_missing $C/itic_probes.pt              $PY dream/baselines/itic.py --fit

if [[ "$SWEEPS" == 1 ]]; then
    # Appendix (Dream protocol): unrotated base, PI at the LLaDA bound, and amax sweep.
    D=results/dream/decode_pid
    run_if_missing "$D/cond_dpid_base.json" $PY dream/denoise_pid.py --cond base
    run_if_missing "$D/cond_dpid_PI.json"   $PY dream/denoise_pid.py --cond PI
    for A in 0.25 0.5 1.0 1.5; do
        tag="dpid_PI_amax${A/./p}"
        run_if_missing "$D/cond_${tag}.json" $PY dream/denoise_pid.py --cond PI --amax "$A" --tag "$tag" --out-dir "$D"
    done
fi

for r in 0 1 2; do
    ITEMS=results/balanced/_sweep400_rot${r}.jsonl

    # Table 3 + cross-model appendix: base, decode PI (amax 1), CAA a2, ActAdd a8.
    run_if_missing "$OUT/base/rot${r}/cond_dpid_base.json" \
        $PY dream/denoise_pid.py --cond base --items "$ITEMS" --out-dir "$OUT/base/rot${r}"
    run_if_missing "$OUT/decode_pid/rot${r}/cond_dpid_PI.json" \
        $PY dream/denoise_pid.py --cond PI --amax 1.0 --items "$ITEMS" --out-dir "$OUT/decode_pid/rot${r}"
    run_if_missing "$OUT/caa/rot${r}/cond_caa_L14_a2.json" \
        $PY dream/baselines/caa.py --run --alpha 2 --layer 14 --items "$ITEMS" --out_dir "$OUT/caa/rot${r}"
    run_if_missing "$OUT/actadd/rot${r}/cond_actadd_a8.json" \
        $PY dream/baselines/actadd.py --run --alpha 8 --layer 14 --items "$ITEMS" --out_dir "$OUT/actadd/rot${r}"

    # Cross-model appendix: remaining baselines at the LLaDA balanced operating points.
    run_if_missing "$OUT/meanact/rot${r}/cond_meanact_unit_s2.json" \
        $PY dream/baselines/meanact.py --run --direction unit --strength 2 --items "$ITEMS" --out_dir "$OUT/meanact/rot${r}"
    run_if_missing "$OUT/linearact/rot${r}/cond_gaussian_s1.json" \
        $PY dream/baselines/linearact.py --run --variant gaussian --strength 1 --items "$ITEMS" --out_dir "$OUT/linearact/rot${r}"
    run_if_missing "$OUT/aura_inject/rot${r}/cond_inject_g4.json" \
        $PY dream/baselines/aura.py --run --mode inject --gamma 4 --items "$ITEMS" --out_dir "$OUT/aura_inject/rot${r}"
    run_if_missing "$OUT/aura_vanilla/rot${r}/cond_vanilla.json" \
        $PY dream/baselines/aura.py --run --mode vanilla --items "$ITEMS" --out_dir "$OUT/aura_vanilla/rot${r}"
    run_if_missing "$OUT/itic/rot${r}/cond_itic_K48_a8.json" \
        $PY dream/baselines/itic.py --run --topk 48 --alpha 8 --items "$ITEMS" --out_dir "$OUT/itic/rot${r}"
done

# Dream static-alpha control: p0 and hindsight arms, amax 1; reads the Dream decode-PI samples.
for r in 0 1 2; do
    for ARM in p0 hind; do
        run_if_missing "$OUT/static_alpha/rot${r}/cond_static_${ARM}.json" \
            $PY dream/dream_static_alpha.py --arm "$ARM" --rot "$r" \
                --model-path Dream-v0-Instruct-7B --out-dir "$OUT/static_alpha/rot${r}"
    done
done

echo "DONE (04_dream)"
