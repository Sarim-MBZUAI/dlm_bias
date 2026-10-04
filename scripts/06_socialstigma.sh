#!/usr/bin/env bash
# SocialStigmaQA-MC3 on LLaDA-8B-Instruct: polarity-specific fits, all conditions x polarity x 3 rotations.
# Needs 00_setup.sh (pinned source + derived items). run_one skips finished (summary + samples) runs.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python}

# Yes/no-target artifacts (directions, baseline fits) from non-race calibration prompts.
$PY -m socialstigma.prepare --check
$PY -m socialstigma.fit_artifacts

# SocialStigmaQA table: preregistered conditions (decode_pi32 = same-GPU repeat, excluded from tables).
for cond in clean decode_pi64 decode_pid64 decode_pi32 normal_a4 layer_pi_a2 \
            caa_a16 actadd_a16 meanact_s2 linearact_s1 aura_inject_g4 aura_vanilla itic_k48_a8; do
    for polarity in yes no; do
        for rotation in 0 1 2; do
            $PY -m socialstigma.run_one --condition "$cond" --polarity "$polarity" --rotation "$rotation"
        done
    done
done

# SocialStigmaQA table: effort-matched open loop (label-free; alpha from decode-PI's first 32 commands, 2.65).
ALPHA=$($PY -m socialstigma.compute_effort_match --print-alpha)
echo "effort-matched alpha: $ALPHA"
for polarity in yes no; do
    for rotation in 0 1 2; do
        $PY -m socialstigma.run_one --condition normal_eff --effort-alpha "$ALPHA" \
            --polarity "$polarity" --rotation "$rotation"
    done
done

# ITI-C margin tie-break row (CPU margin added to existing probes, then eval).
$PY -m socialstigma.fit_artifacts --itic-margin
for polarity in yes no; do
    for rotation in 0 1 2; do
        $PY -m socialstigma.run_one --condition itic_k48_a8_tb --polarity "$polarity" --rotation "$rotation"
    done
done

echo "DONE (06_socialstigma; aggregate with scripts/07_score.sh)"
