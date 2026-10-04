#!/usr/bin/env bash
# LLaDA-8B-Instruct, Arab / Asian / Latino / White / Old targets.
# SWEEPS=1 also runs the unrotated dose sweeps that fixed the per-target operating points.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python}
SWEEPS=${SWEEPS:-0}
run_if_missing() { local m=$1; shift; if [[ -f "$m" ]]; then echo "SKIP (exists): $m"; else echo "RUN: $*"; "$@"; fi; }
OUT=results/balanced_all
ROTDIR=$OUT/rotations
MR=results/multirace

# Directions per target (manifest held-out contrast sets).
for T in arab asian latino white old; do
    run_if_missing "multirace/arrows_${T}.pt" $PY multirace/build_arrows.py --target "$T"
done

if [[ "$SWEEPS" == 1 ]]; then
    # Appendix (per-target doses; race CAA claim): unrotated base / PI / open loop a4 / CAA a2.
    for T in arab asian latino white; do
        run_if_missing "$MR/$T/decode_pid/cond_dpid_${T}_base.json" $PY multirace/denoise_pid.py --target "$T" --cond base
        run_if_missing "$MR/$T/decode_pid/cond_dpid_${T}_PI.json"   $PY multirace/denoise_pid.py --target "$T" --cond PI
        run_if_missing "$MR/$T/normal/cond_normal_${T}_a4.json"     $PY multirace/normal.py --target "$T" --alpha 4
        run_if_missing "$MR/$T/caa/cond_caa_${T}_a2.json"           $PY multirace/caa.py --target "$T" --alpha 2
    done
    # Appendix (per-target doses): unrotated decode-PI amax and open-loop alpha sweeps.
    for A in 1 2 3 4; do
        run_if_missing "$MR/asian/decode_pid/cond_dpid_asian_PI_amax$A.json" \
            $PY multirace/denoise_pid.py --target asian --cond PI --amax $A --tag dpid_asian_PI_amax$A
    done
    for A in 2 3 4; do
        run_if_missing "$MR/latino/decode_pid/cond_dpid_latino_PI_amax$A.json" \
            $PY multirace/denoise_pid.py --target latino --cond PI --amax $A --tag dpid_latino_PI_amax$A
    done
    for T in asian white; do
        for AL in 1 2; do
            run_if_missing "$MR/$T/normal/cond_normal_${T}_a$AL.json" \
                $PY multirace/normal.py --target $T --alpha $AL --tag normal_${T}_a$AL
        done
    done
    # Appendix (Age construction): Old dose sweep, base / PI amax {2,4,6} / open loop a {2,4}.
    run_if_missing "$MR/old/decode_pid/cond_dpid_old_base.json" $PY multirace/denoise_pid.py --target old --cond base
    for A in 2 4 6; do
        run_if_missing "$MR/old/decode_pid/cond_dpid_old_PI_amax${A}.json" \
            $PY multirace/denoise_pid.py --target old --cond PI --amax "$A" --tag "dpid_old_PI_amax${A}"
    done
    for AL in 2 4; do
        run_if_missing "$MR/old/normal/cond_normal_old_a${AL}.json" \
            $PY multirace/normal.py --target old --alpha "$AL" --tag "normal_old_a${AL}"
    done
fi

# Table 3 (Arab) + appendix race targets: balanced base, decode PI, open loop at the per-target dose
# (arab a4, asian a2, latino a4, white a1).
for T in arab asian latino white; do
    case $T in arab|latino) ALPHA=4 ;; asian) ALPHA=2 ;; white) ALPHA=1 ;; esac
    for r in 0 1 2; do
        ITEMS=$ROTDIR/_sweep400_${T}_rot${r}.jsonl
        run_if_missing "$OUT/$T/base/rot${r}/cond_dpid_${T}_base.json" \
            $PY multirace/denoise_pid.py --target "$T" --cond base --items "$ITEMS" --out-dir "$OUT/$T/base/rot${r}"
        run_if_missing "$OUT/$T/decode_pid/rot${r}/cond_dpid_${T}_PI.json" \
            $PY multirace/denoise_pid.py --target "$T" --cond PI --items "$ITEMS" --out-dir "$OUT/$T/decode_pid/rot${r}"
        run_if_missing "$OUT/$T/normal/rot${r}/cond_normal_${T}_a${ALPHA}.json" \
            $PY multirace/normal.py --target "$T" --alpha "$ALPHA" --items "$ITEMS" --out-dir "$OUT/$T/normal/rot${r}"
    done
done

# Table 3 (Old): balanced base, decode PI amax 4, open loop a4, CAA a2, effort-matched open loop a3.
T=old
for r in 0 1 2; do
    ITEMS=$ROTDIR/_sweep400_${T}_rot${r}.jsonl
    run_if_missing "$OUT/$T/base/rot${r}/cond_dpid_${T}_base.json" \
        $PY multirace/denoise_pid.py --target "$T" --cond base --items "$ITEMS" --out-dir "$OUT/$T/base/rot${r}"
    run_if_missing "$OUT/$T/decode_pid/rot${r}/cond_dpid_${T}_PI.json" \
        $PY multirace/denoise_pid.py --target "$T" --cond PI --amax 4 --items "$ITEMS" --out-dir "$OUT/$T/decode_pid/rot${r}"
    run_if_missing "$OUT/$T/normal/rot${r}/cond_normal_${T}_a4.json" \
        $PY multirace/normal.py --target "$T" --alpha 4 --items "$ITEMS" --out-dir "$OUT/$T/normal/rot${r}"
    run_if_missing "$OUT/$T/caa/rot${r}/cond_caa_${T}_a2.json" \
        $PY multirace/caa.py --target "$T" --alpha 2 --items "$ITEMS" --out-dir "$OUT/$T/caa/rot${r}"
    run_if_missing "$OUT/$T/normal_eff/rot${r}/cond_normal_${T}_a3.json" \
        $PY multirace/normal.py --target "$T" --alpha 3 --items "$ITEMS" --out-dir "$OUT/$T/normal_eff/rot${r}" --tag "normal_${T}_a3"
done

# Appendix (Arab, Old): prior-work baseline fits + balanced suite at the Black operating points.
C=baselines/cache
for T in arab old; do
    run_if_missing "$C/calib_block_${T}.pt"            $PY baselines/calib.py --fit --where block --target "$T"
    run_if_missing "$C/calib_mlp_hidden_${T}.pt"       $PY baselines/calib.py --fit --where mlp_hidden --target "$T"
    run_if_missing "$C/calib_attn_head_${T}.pt"        $PY baselines/calib.py --fit --where attn_head --target "$T"
    run_if_missing "$C/actadd_dir_${T}.pt"             $PY baselines/actadd.py --fit --target "$T"
    run_if_missing "$C/meanact_meandiff_block_${T}.pt" $PY baselines/meanact.py --fit --target "$T"
    run_if_missing "$C/linearact_stats_${T}.pt"        $PY baselines/linearact.py --fit --target "$T"
    run_if_missing "$C/aura_auroc_${T}.pt"             $PY baselines/aura.py --fit --target "$T"
    run_if_missing "$C/itic_probes_${T}.pt"            $PY baselines/itic.py --fit --target "$T"

    B=$OUT/$T
    for r in 0 1 2; do
        ITEMS=$ROTDIR/_sweep400_${T}_rot${r}.jsonl
        run_if_missing "$B/caa/rot${r}/cond_caa_L14_a16.json" \
            $PY baselines/caa.py --run --alpha 16 --layer 14 --target "$T" --items "$ITEMS" --out_dir "$B/caa/rot${r}"
        run_if_missing "$B/actadd/rot${r}/cond_actadd_a16.json" \
            $PY baselines/actadd.py --run --alpha 16 --layer 14 --target "$T" --items "$ITEMS" --out-dir "$B/actadd/rot${r}"
        run_if_missing "$B/meanact/rot${r}/cond_meanact_unit_s2.json" \
            $PY baselines/meanact.py --run --direction unit --strength 2 --target "$T" --items "$ITEMS" --out-dir "$B/meanact/rot${r}"
        run_if_missing "$B/linearact/rot${r}/cond_gaussian_s1.json" \
            $PY baselines/linearact.py --run --variant gaussian --strength 1 --target "$T" --items "$ITEMS" --out_dir "$B/linearact/rot${r}"
        run_if_missing "$B/aura_inject/rot${r}/cond_inject_g4.json" \
            $PY baselines/aura.py --run --mode inject --gamma 4 --target "$T" --items "$ITEMS" --out-dir "$B/aura_inject/rot${r}"
        run_if_missing "$B/aura_vanilla/rot${r}/cond_vanilla.json" \
            $PY baselines/aura.py --run --mode vanilla --target "$T" --items "$ITEMS" --out-dir "$B/aura_vanilla/rot${r}"
        run_if_missing "$B/itic/rot${r}/cond_itic_K48_a8.json" \
            $PY baselines/itic.py --run --topk 48 --alpha 8 --target "$T" --items "$ITEMS" --out_dir "$B/itic/rot${r}"
    done
done

# Appendix (sensor): Arab decode PI with the case-inclusive sensor.
for r in 0 1 2; do
    ITEMS=$ROTDIR/_sweep400_arab_rot${r}.jsonl
    run_if_missing "$OUT/arab/decode_pid_cs/rot${r}/cond_dpid_arab_PI_cs.json" \
        $PY multirace/denoise_pid.py --target arab --cond PI --amax 6 --sensor-case both --tag dpid_arab_PI_cs \
            --items "$ITEMS" --out-dir "$OUT/arab/decode_pid_cs/rot${r}"
done

echo "DONE (03_other_targets)"
