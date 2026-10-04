#!/usr/bin/env bash
# LLaDA-8B-Instruct, Black target: direction, baseline fits, all Table-1 conditions x 3 rotations.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python}
run_if_missing() { local m=$1; shift; if [[ -f "$m" ]]; then echo "SKIP (exists): $m"; else echo "RUN: $*"; "$@"; fi; }

# Direction (32-layer diff-in-means arrows).
run_if_missing steering/arrows.pt $PY steering/build_arrows.py

# Baseline calibration collections + per-method fits (CAA reuses steering/arrows.pt).
C=baselines/cache
run_if_missing $C/calib_block.pt            $PY baselines/calib.py --fit --where block
run_if_missing $C/calib_mlp_hidden.pt       $PY baselines/calib.py --fit --where mlp_hidden
run_if_missing $C/calib_attn_head.pt        $PY baselines/calib.py --fit --where attn_head
run_if_missing $C/actadd_dir.pt             $PY baselines/actadd.py --fit
run_if_missing $C/meanact_meandiff_block.pt $PY baselines/meanact.py --fit
run_if_missing $C/linearact_stats.pt        $PY baselines/linearact.py --fit
run_if_missing $C/aura_auroc.pt             $PY baselines/aura.py --fit
run_if_missing $C/itic_probes.pt            $PY baselines/itic.py --fit

for r in 0 1 2; do
    ITEMS=results/balanced/_sweep400_rot${r}.jsonl

    # Table 1: base, layer-wise PI (a=2), open loop (a=4), decode PI (headline).
    H=results/balanced/results_balanced/rot${r}
    run_if_missing "$H/cond_base.json" \
        $PY steering/pid_steer.py --mode pid --cond base --items "$ITEMS" --out-dir "$H"
    run_if_missing "$H/cond_PI_a2.json" \
        $PY steering/pid_steer.py --mode pid --cond PI --alpha 2 --items "$ITEMS" --out-dir "$H"
    run_if_missing "$H/cond_normalL14_a4.json" \
        $PY steering/pid_steer.py --mode normal --source-layer 14 --alpha 4 --items "$ITEMS" --out-dir "$H"
    run_if_missing "$H/cond_dpid_PI.json" \
        $PY steering/denoise_pid.py --cond PI --items "$ITEMS" --out-dir "$H" --tag dpid_PI

    # Table 1: prior-work baselines at the published operating points.
    O=results/balanced_all/black
    run_if_missing "$O/caa/rot${r}/cond_caa_L14_a16.json" \
        $PY baselines/caa.py --run --alpha 16 --layer 14 --items "$ITEMS" --out_dir "$O/caa/rot${r}"
    run_if_missing "$O/actadd/rot${r}/cond_actadd_a16.json" \
        $PY baselines/actadd.py --run --alpha 16 --layer 14 --items "$ITEMS" --out-dir "$O/actadd/rot${r}"
    run_if_missing "$O/meanact/rot${r}/cond_meanact_unit_s2.json" \
        $PY baselines/meanact.py --run --direction unit --strength 2 --items "$ITEMS" --out-dir "$O/meanact/rot${r}"
    run_if_missing "$O/linearact/rot${r}/cond_gaussian_s1.json" \
        $PY baselines/linearact.py --run --variant gaussian --strength 1 --items "$ITEMS" --out_dir "$O/linearact/rot${r}"
    run_if_missing "$O/aura_inject/rot${r}/cond_inject_g4.json" \
        $PY baselines/aura.py --run --mode inject --gamma 4 --items "$ITEMS" --out-dir "$O/aura_inject/rot${r}"
    run_if_missing "$O/aura_vanilla/rot${r}/cond_vanilla.json" \
        $PY baselines/aura.py --run --mode vanilla --items "$ITEMS" --out-dir "$O/aura_vanilla/rot${r}"
    run_if_missing "$O/itic/rot${r}/cond_itic_K48_a8.json" \
        $PY baselines/itic.py --run --topk 48 --alpha 8 --items "$ITEMS" --out_dir "$O/itic/rot${r}"

    # Table 1: decode PID; effort-matched open loop (a=3.28 = PI mean command over steps 1-32).
    run_if_missing "$O/decode_pid/rot${r}/cond_dpid_PID.json" \
        $PY steering/denoise_pid.py --cond PID --items "$ITEMS" --out-dir "$O/decode_pid/rot${r}" --tag dpid_PID
    run_if_missing "$O/normal/rot${r}/cond_normalL14_a3p28.json" \
        $PY steering/pid_steer.py --mode normal --alpha 3.28 --items "$ITEMS" --out-dir "$O/normal/rot${r}"
done

echo "DONE (01_black_main)"
