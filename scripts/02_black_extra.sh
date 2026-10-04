#!/usr/bin/env bash
# LLaDA-8B-Instruct, Black target: seed draws, suppression, static-alpha arms, steps=32 repeat.
# Needs 01_black_main.sh (steering/arrows.pt and the headline decode-PI samples).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python}
run_if_missing() { local m=$1; shift; if [[ -f "$m" ]]; then echo "SKIP (exists): $m"; else echo "RUN: $*"; "$@"; fi; }
O=results/balanced_all/black

# Three extra 400-item draws (needs steering/direction_examples.jsonl from 01).
if [[ ! -f results/balanced/seeds/seed3/_sweep400_rot2.jsonl ]]; then
    $PY eval/balanced/make_superset.py
    $PY eval/balanced/make_seed_rotations.py
fi

# Appendix (seeds): base + decode PI on three extra 400-item draws x 3 rotations.
for s in 1 2 3; do
    for r in 0 1 2; do
        ITEMS=results/balanced/seeds/seed${s}/_sweep400_rot${r}.jsonl
        S=results/balanced_seeds/seed${s}
        run_if_missing "$S/base/rot${r}/cond_dpid_base.json" \
            $PY steering/denoise_pid.py --cond base --items "$ITEMS" --out-dir "$S/base/rot${r}"
        run_if_missing "$S/decode_pid/rot${r}/cond_dpid_PI.json" \
            $PY steering/denoise_pid.py --cond PI --items "$ITEMS" --out-dir "$S/decode_pid/rot${r}"
    done
done

for r in 0 1 2; do
    ITEMS=results/balanced/_sweep400_rot${r}.jsonl

    # Suppression (Sec. suppression / appendix): closed loop s*=0, alpha in [-6,0];
    # open loops a=-4 (mirror dose) and a=-0.72 (suppression controller's mean command).
    run_if_missing "$O/suppress/rot${r}/cond_dpid_PI_suppress.json" \
        $PY steering/denoise_pid.py --cond PI --setpoint 0.0 --amin -6 --amax 0 --tag dpid_PI_suppress \
            --items "$ITEMS" --out-dir "$O/suppress/rot${r}"
    run_if_missing "$O/normal_neg/rot${r}/cond_normalL14_am4.json" \
        $PY steering/pid_steer.py --mode normal --alpha -4 --items "$ITEMS" --out-dir "$O/normal_neg/rot${r}"
    run_if_missing "$O/normal_neg/rot${r}/cond_normalL14_am0p72.json" \
        $PY steering/pid_steer.py --mode normal --alpha -0.72 --items "$ITEMS" --out-dir "$O/normal_neg/rot${r}"

    # Table 1: static alpha per item (p0 arm, hindsight arm); reads headline PI samples.
    for ARM in p0 hind; do
        run_if_missing "$O/static_alpha/rot${r}/cond_static_${ARM}.json" \
            $PY steering/static_alpha_control.py --arm "$ARM" --rot "$r" \
                --model-path LLaDA-8B-Instruct --out-dir "$O/static_alpha/rot${r}"
    done

    # Appendix (repeat runs): decode PI at 32 denoising steps.
    run_if_missing "$O/decode_pid_s32/rot${r}/cond_dpid_PI_s32.json" \
        $PY steering/denoise_pid.py --cond PI --steps 32 --items "$ITEMS" \
            --out-dir "$O/decode_pid_s32/rot${r}" --tag dpid_PI_s32
done

echo "DONE (02_black_extra)"
