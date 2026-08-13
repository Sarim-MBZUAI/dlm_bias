#!/usr/bin/env python3
"""Round-3 strict-parse analysis: E5 closed-loop bias SUPPRESSION family.

Same STRICT rule, pooling over rot0/rot1/rot2 (3x400 = 1200 items) and
10,000-resample item-level bootstrap (numpy default_rng(seed=0), fresh per
condition) as balanced_all/strict_pool.py / strict_round2.py.

Conditions:
  base            round-1 clean       results/balanced/results_balanced/
                                        rot{r}/cond_base_samples.jsonl
  decode_pi       round-1 attack      .../cond_dpid_PI_samples.jsonl
                  (decode-space PI, s*=0.9, alpha in [0, 6])
  suppress        NEW closed loop     results/balanced_all/black/suppress/
                                        rot{r}/cond_dpid_PI_suppress_samples.jsonl
                  (same PI loop, s*=0.0, alpha in [-6, 0]: pushes AWAY)
  openloop_debias NEW open loop       results/balanced_all/black/normal_neg/
                                        rot{r}/cond_normalL14_am4_samples.jsonl
                  (constant alpha=-4 of the same vhat at all 32 layers)

Reported per condition: strict target/comparator/abstain/invalid rates, gap
with 95% bootstrap CI, Gap@A / Gap@BC, permissive gap + invalid.  For the two
NEW conditions additionally Delta-g vs the round-1 BASE and vs the decode-PI
ATTACK (difference-of-replicates CIs), and the coherence cost (strict +
permissive invalid rate) side by side with the equally-strong open loop --
the E5 claim: closed-loop suppression moves the gap toward/below zero with
LESS coherence damage than an equally-strong negative open loop.

Telemetry: mean_alpha / mean_sat_frac of the suppress runs averaged over the
3 rotations (mean_alpha is NEGATIVE in suppression mode).

MISSING result files are reported gracefully (pre-run, the two NEW conditions
do not exist yet; the script still prints the baselines and exits 0).

Usage (repo root):  python balanced_all/strict_round3.py [--json OUT.json]
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import strict_pool as sp    # noqa: E402  (STRICT rule, score(), bootstrap)
import strict_round2 as r2  # noqa: E402  (analyze(), dgap_ci, rots, fmt, ...)

CONDS = [
    # key, label, samples template (rot %d), summary json template or None
    ("base", "Clean (base, round 1)",
     "results/balanced/results_balanced/rot%d/cond_base_samples.jsonl", None),
    ("decode_pi", "Decode PI attack (round 1)",
     "results/balanced/results_balanced/rot%d/cond_dpid_PI_samples.jsonl", None),
    ("suppress", "Suppress (closed loop)",
     "results/balanced_all/black/suppress/rot%d/cond_dpid_PI_suppress_samples.jsonl",
     "results/balanced_all/black/suppress/rot%d/cond_dpid_PI_suppress.json"),
    ("openloop_debias", "Open loop a=-4",
     "results/balanced_all/black/normal_neg/rot%d/cond_normalL14_am4_samples.jsonl",
     "results/balanced_all/black/normal_neg/rot%d/cond_normalL14_am4.json"),
]


def analyze_soft(tmpl, want_reps=False):
    """r2.analyze, but tolerate missing rotations: report and return None."""
    paths = r2.rots(tmpl)
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        for p in missing:
            print("   MISSING: %s" % os.path.relpath(p, REPO))
        return None
    return r2.analyze(paths, want_reps=want_reps)


def family_suppression(results):
    print("== FAMILY 5: SUPPRESSION (E5, pooled 3x400, strict) ==")
    print("%-28s %6s %6s %6s %6s | %25s | %6s %6s" %
          ("condition", "target", "compar", "abstn", "inval",
           "gap [95% CI]", "gap@A", "gap@BC"))
    fam = {}
    for key, label, tmpl, json_tmpl in CONDS:
        if json_tmpl is not None:
            jpaths = [p for p in r2.rots(json_tmpl) if os.path.exists(p)]
            if len(jpaths) == 3:
                r2.verify_summaries(jpaths)
        a = analyze_soft(tmpl, want_reps=True)
        if a is None:
            print("%-28s (not run yet)" % label)
            fam[key] = None
            continue
        print("%-28s %s" % (label, r2.fmt(a)))
        fam[key] = a

    base, attack = fam.get("base"), fam.get("decode_pi")
    print("\n-- Delta-g of the NEW conditions (vs round-1 base AND vs decode-PI attack) --")
    for key in ["suppress", "openloop_debias"]:
        a = fam.get(key)
        if a is None:
            print("%-28s MISSING (run slurm/round3_jobJ_suppress.sbatch first)" % key)
            continue
        for ref_name, ref in [("base", base), ("attack", attack)]:
            if ref is None:
                print("%-28s dg vs %-6s: baseline MISSING" % (key, ref_name))
                continue
            dg = a["gap"] - ref["gap"]
            ci = r2.dgap_ci(a["_reps"], ref["_reps"])
            a["dgap_vs_%s" % ref_name], a["dgap_vs_%s_ci95" % ref_name] = dg, ci
            print("%-28s dg vs %-6s = %+6.3f [%+6.3f,%+6.3f]"
                  % (key, ref_name, dg, ci[0], ci[1]))

    # Coherence cost: closed-loop suppression vs the equally-strong open loop.
    sup, ol = fam.get("suppress"), fam.get("openloop_debias")
    print("\n-- Coherence cost (invalid rates, strict + permissive) --")
    for key, a in [("suppress", sup), ("openloop_debias", ol)]:
        if a is None:
            print("%-28s MISSING" % key)
            continue
        print("%-28s strict_invalid=%.3f  perm_invalid=%.3f  abstain=%.3f"
              % (key, a["invalid"], a["perm_invalid"], a["abstain"]))
    if sup is not None and ol is not None:
        both_down = sup["gap"] <= 0.0 or sup.get("dgap_vs_base", 0.0) < 0.0
        cheaper = sup["invalid"] < ol["invalid"]
        print("E5 claim check: suppress gap %+0.3f (toward/below zero: %s); "
              "strict invalid %0.3f vs open-loop %0.3f (less damage: %s)"
              % (sup["gap"], "YES" if both_down else "NO",
                 sup["invalid"], ol["invalid"], "YES" if cheaper else "NO"))

    # Telemetry of the closed-loop suppressor (mean_alpha is NEGATIVE here).
    print("\n-- Telemetry: suppress controller effort (avg of 3 rotations) --")
    jt = [p for _, _, _, t in CONDS if t and "suppress" in t for p in r2.rots(t)]
    have = [p for p in jt if os.path.exists(p)]
    if len(have) == 3:
        t = r2.summary_avg(have, ["mean_alpha", "mean_sat_frac", "setpoint",
                                  "alpha_min", "alpha_max"])
        fam["suppress_telemetry"] = t
        print("suppress (s*=%g, alpha in [%g,%g])  mean_alpha=%.2f  sat_frac=%.2f"
              % (t["setpoint"], t["alpha_min"], t["alpha_max"],
                 t["mean_alpha"], t["mean_sat_frac"]))
    else:
        print("suppress telemetry MISSING (%d/3 summary jsons)" % len(have))

    for a in fam.values():
        if isinstance(a, dict):
            a.pop("_reps", None)
    results["suppression"] = fam


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    results = {}
    family_suppression(results)

    if args.json:
        with open(args.json, "w") as f:
            json.dump(results, f, indent=1)
        print("\nwrote %s" % args.json)


if __name__ == "__main__":
    main()
