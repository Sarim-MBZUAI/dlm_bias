#!/usr/bin/env bash
# Strict reparse, rotation pooling, and bootstrap CIs for every reported table (CPU only).
# SWEEPS=1 to also score the unrotated dose-selection sweeps (if you ran them).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python}
SWEEPS=${SWEEPS:-0}
S=results/strict
mkdir -p "$S"

# Table 1 (Black, LLaDA-8B): 14 conditions + steps=32 replicate.
$PY balanced_all/strict_pool.py --json "$S/black.json"

# Table 3 + appendix: other targets, seed draws, Dream, LLaDA-MoE, controller telemetry.
EXTRA=(); [[ "$SWEEPS" == 1 ]] || EXTRA=(--skip-unrotated)
$PY balanced_all/strict_extended.py --json "$S/extended.json" ${EXTRA[@]+"${EXTRA[@]}"}

# Suppression and Arab sensor appendix.
$PY balanced_all/strict_suppress_sensor.py --json "$S/suppress_sensor.json"

# Conditions outside the scripts above (static-alpha arms, effort-matched open loops),
# scored with the identical strict rule and bootstrap from balanced_all/strict_pool.py.
$PY - "$S/extra.json" <<'EOF'
import json, os, sys
sys.path.insert(0, "balanced_all")
import strict_pool as sp
R = lambda t: ["%s/rot%d/cond_%s_samples.jsonl" % (t[0], r, t[1]) for r in range(3)]
conds = {
    "Black static alpha p0":       ("results/balanced_all/black/static_alpha", "static_p0"),
    "Black static alpha hindsight": ("results/balanced_all/black/static_alpha", "static_hind"),
    "Black open loop a=-0.72":     ("results/balanced_all/black/normal_neg", "normalL14_am0p72"),
    "Old open loop a=3 (matched)": ("results/balanced_all/old/normal_eff", "normal_old_a3"),
    "Dream static alpha p0":       ("results/dream_balanced/static_alpha", "static_p0"),
    "Dream static alpha hindsight": ("results/dream_balanced/static_alpha", "static_hind"),
}
out = {}
for label, t in conds.items():
    paths = R(t)
    if not all(os.path.exists(p) for p in paths):
        print("MISSING (skipped):", label); continue
    s = sp.score(sp.load_rows(paths), sp.strict_letter)
    lo, hi = sp.bootstrap_ci(s["signs"])
    out[label] = {k: float(s[k]) for k in ("target", "comparator", "abstain", "invalid", "gap")}
    out[label]["gap_ci95"] = [lo, hi]
    print("%-30s tgt %.3f cmp %.3f abs %.3f inv %.3f gap %+.3f [%+.3f, %+.3f]" % (
        label, s["target"], s["comparator"], s["abstain"], s["invalid"], s["gap"], lo, hi))
json.dump(out, open(sys.argv[1], "w"), indent=1)
EOF

# SocialStigmaQA table (strict + semantic parsers, BH family, identity checks).
$PY -m socialstigma.aggregate

# Trajectory appendix (lock-in, saturation, success vs failure trajectories).
$PY analysis/trajectory/traj_analysis.py

# Item-clustered CIs (contrasts figure), held-out 300-item rescoring, seed draws without
# calibration items, within-sequence command statistics; static-alpha paired contrasts.
$PY analysis/reanalysis.py
$PY analysis/score_static.py
$PY analysis/score_dream.py

echo "DONE (07_score; JSON under $S)"
