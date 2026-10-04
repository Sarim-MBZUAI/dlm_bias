#!/usr/bin/env python3
"""Strict-parse analysis: closed-loop bias suppression + case-full sensor families.

Same strict rule, pooling over rot0/rot1/rot2 (3x400 = 1200 items) and
10,000-resample item-level bootstrap (numpy default_rng(seed=0), fresh per
condition) as balanced_all/strict_pool.py / strict_extended.py.

Suppression family:
  base            clean               results/balanced/results_balanced/
                                        rot{r}/cond_base_samples.jsonl
  decode_pi       attack              .../cond_dpid_PI_samples.jsonl
                  (decode-space PI, s*=0.9, alpha in [0, 6])
  suppress        closed loop         results/balanced_all/black/suppress/
                                        rot{r}/cond_dpid_PI_suppress_samples.jsonl
                  (same PI loop, s*=0.0, alpha in [-6, 0]: pushes away)
  openloop_debias open loop           results/balanced_all/black/normal_neg/
                                        rot{r}/cond_normalL14_am4_samples.jsonl
                  (constant alpha=-4 of the same vhat at all 32 layers)

Reported per condition: strict target/comparator/abstain/invalid rates, gap
with 95% bootstrap CI, Gap@A / Gap@BC, permissive gap + invalid. For suppress
and openloop_debias additionally Delta-g vs base and vs the decode-PI attack
(difference-of-replicates CIs), and the coherence cost (strict + permissive
invalid rate) of closed-loop suppression vs the equally-strong open loop.
Telemetry: mean_alpha / mean_sat_frac of the suppress runs averaged over the
3 rotations (mean_alpha is negative in suppression mode).

Case-full sensor family: the default decode-space sensor sums P over
uppercase letter ids only, while the strict parser also accepts lowercase
answers (which the arab direction induces). This family compares:
  arab_base       clean               results/balanced_all/arab/base/
                                        rot{r}/cond_dpid_arab_base_samples.jsonl
  arab_blind      attack, case-blind  results/balanced_all/arab/decode_pid/
                                        rot{r}/cond_dpid_arab_PI_samples.jsonl
  arab_casefull   attack, case-full   results/balanced_all/arab/decode_pid_cs/
                                        rot{r}/cond_dpid_arab_PI_cs_samples.jsonl
                  (multirace/denoise_pid.py --sensor-case both --amax 6)
Reported: strict rates + gap CI per condition, Delta-g of case-full vs the
case-blind run and vs base, invalid rates, and controller telemetry (whether
the controller backs off once it can see lowercase wins).

Missing result files are reported, not fatal.

Usage (from the repo root):  python balanced_all/strict_suppress_sensor.py [--json OUT.json]
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
import strict_extended as r2  # noqa: E402  (analyze(), dgap_ci, rots, fmt, ...)

CONDS = [
    # key, label, samples template (rot %d), summary json template or None
    ("base", "Clean (base)",
     "results/balanced/results_balanced/rot%d/cond_base_samples.jsonl", None),
    ("decode_pi", "Decode PI attack",
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
    print("== FAMILY 5: SUPPRESSION (pooled 3x400, strict) ==")
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
    print("\n-- Delta-g of the conditions (vs base AND vs decode-PI attack) --")
    for key in ["suppress", "openloop_debias"]:
        a = fam.get(key)
        if a is None:
            print("%-28s MISSING (run scripts/02_black_extra.sh first)" % key)
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
        print("Check: suppress gap %+0.3f (toward/below zero: %s); "
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


# ---------------------------------------------------------------------------
# Family 7: case-full sensor (arab decode-PI, sensor sees lowercase).
# ---------------------------------------------------------------------------
SENSOR_CONDS = [
    # key, label, samples template (rot %d), summary json template or None
    ("arab_base", "Arab clean (base)",
     "results/balanced_all/arab/base/rot%d/cond_dpid_arab_base_samples.jsonl",
     "results/balanced_all/arab/base/rot%d/cond_dpid_arab_base.json"),
    ("arab_blind", "Arab decode-PI case-BLIND",
     "results/balanced_all/arab/decode_pid/rot%d/cond_dpid_arab_PI_samples.jsonl",
     "results/balanced_all/arab/decode_pid/rot%d/cond_dpid_arab_PI.json"),
    ("arab_casefull", "Arab decode-PI case-FULL",
     "results/balanced_all/arab/decode_pid_cs/rot%d/cond_dpid_arab_PI_cs_samples.jsonl",
     "results/balanced_all/arab/decode_pid_cs/rot%d/cond_dpid_arab_PI_cs.json"),
]
SENSOR_TELE_KEYS = ["mean_alpha", "mean_sat_frac", "mean_final_alpha"]


def family_sensor_case(results):
    print("\n== FAMILY 7: CASE-FULL SENSOR (arab, pooled 3x400, strict) ==")
    print("%-28s %6s %6s %6s %6s | %25s | %6s %6s" %
          ("condition", "target", "compar", "abstn", "inval",
           "gap [95% CI]", "gap@A", "gap@BC"))
    fam = {}
    for key, label, tmpl, json_tmpl in SENSOR_CONDS:
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

    base, blind, full = (fam.get("arab_base"), fam.get("arab_blind"),
                         fam.get("arab_casefull"))
    print("\n-- Delta-g of the case-FULL run (vs case-BLIND attack AND vs base) --")
    if full is None:
        print("arab_casefull MISSING (run scripts/03_other_targets.sh first)")
    else:
        for ref_name, ref in [("blind", blind), ("base", base)]:
            if ref is None:
                print("%-28s dg vs %-6s: baseline MISSING" % ("arab_casefull", ref_name))
                continue
            dg = full["gap"] - ref["gap"]
            ci = r2.dgap_ci(full["_reps"], ref["_reps"])
            full["dgap_vs_%s" % ref_name], full["dgap_vs_%s_ci95" % ref_name] = dg, ci
            print("%-28s dg vs %-6s = %+6.3f [%+6.3f,%+6.3f]"
                  % ("arab_casefull", ref_name, dg, ci[0], ci[1]))

    print("\n-- Invalid rates (strict + permissive) --")
    for key in ["arab_base", "arab_blind", "arab_casefull"]:
        a = fam.get(key)
        if a is None:
            print("%-28s MISSING" % key)
            continue
        print("%-28s strict_invalid=%.3f  perm_invalid=%.3f  abstain=%.3f"
              % (key, a["invalid"], a["perm_invalid"], a["abstain"]))

    # Telemetry: does the controller BACK OFF once it can see lowercase wins?
    print("\n-- Telemetry: controller effort, case-blind vs case-full "
          "(avg of 3 rotations) --")
    tele = {}
    for key, _, _, json_tmpl in SENSOR_CONDS[1:]:
        jp = [p for p in r2.rots(json_tmpl) if os.path.exists(p)]
        if len(jp) != 3:
            print("%-28s telemetry MISSING (%d/3 summary jsons)" % (key, len(jp)))
            continue
        t = r2.summary_avg(jp, SENSOR_TELE_KEYS)
        tele[key] = t
        print("%-28s mean_alpha=%.2f  sat_frac=%.2f  final_alpha=%s"
              % (key, t["mean_alpha"], t["mean_sat_frac"],
                 "%.2f" % t["mean_final_alpha"]
                 if t["mean_final_alpha"] is not None else "n/a"))
    if "arab_blind" in tele and "arab_casefull" in tele:
        tb, tf = tele["arab_blind"], tele["arab_casefull"]
        backed_off = (tf["mean_alpha"] < tb["mean_alpha"]
                      and tf["mean_sat_frac"] < tb["mean_sat_frac"])
        print("Check: case-full mean_alpha %.2f vs blind %.2f, "
              "sat %.2f vs %.2f -> controller backs off: %s"
              % (tf["mean_alpha"], tb["mean_alpha"], tf["mean_sat_frac"],
                 tb["mean_sat_frac"], "YES" if backed_off else "NO"))
    fam["telemetry"] = tele or None

    for a in fam.values():
        if isinstance(a, dict):
            a.pop("_reps", None)
    results["sensor_case"] = fam


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    results = {}
    family_suppression(results)
    family_sensor_case(results)

    if args.json:
        with open(args.json, "w") as f:
            json.dump(results, f, indent=1)
        print("\nwrote %s" % args.json)


if __name__ == "__main__":
    main()
