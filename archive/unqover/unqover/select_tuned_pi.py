#!/usr/bin/env python
"""Deterministic selector for the UNQOVER decode-PI tuning sweep (jobAJ).

Reads the BUILD-split sweep outputs (build_*.jsonl + their .json sidecars written
by denoise_pid_unqover.py), computes per-config metrics via the OFFICIAL
unqover_metric module (no text parsing), applies the PRE-REGISTERED selection
rule, and writes CHOSEN.json describing the winning config. CPU-only.

PRE-REGISTERED SELECTION RULE (see unqover/TUNING_PREREG_jobAJ.md):
  Per config c on the BUILD split, compute:
    cov_c   = metric coverage (fraction of instances with all 4 records parseable)
    f90_c   = fraction of items with p_target_final >= 0.9   (control objective)
    alpha_c = mean over items of alpha_mean                  (realized actuation)
    sat_c   = mean sat_frac
    gdeb_c  = pref_gap_debiased toward the target (paper's fully-debiased bias)
    graw_c  = pref_gap_raw (BBQ-like) toward the target
  Reference = the default-PI config (tag REF, Kp3/Ki0.1/Kd0/amax6): f0, d0=gdeb.

  ELIGIBLE = { c : cov_c >= COV_MIN  AND  gdeb_c >= d0 - GDEB_SLACK }
      (coherent output AND the debiased bias signal is NOT collapsed by
       over-actuation relative to the default controller.)
  If some ELIGIBLE config has f90_c > f0 (tuning genuinely reduces the
  under-actuation the diagnosis found):
      winner = the ELIGIBLE config with the HIGHEST f90  (primary objective:
               fix the stuck-item under-actuation -> reach the p=0.9 setpoint).
  Else (no config improves actuation without collapsing the signal):
      winner = among coverage-passing configs, the HIGHEST gdeb (best debiased
               bias signal) -- i.e. keep the default unless something strictly
               helps.
  Deterministic tie-breaks (values within EPS_F90=0.01 on the primary key):
      higher gdeb -> lower mean alpha (efficiency) -> lexical tag (= grid order).

This rule uses BUILD-split data ONLY. The EVAL split is never read here.
"""
import argparse
import glob
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from unqover_metric import compute, target_gap, load  # noqa: E402

COV_MIN_DEFAULT = 0.95
GDEB_SLACK_DEFAULT = 0.01
EPS_F90 = 0.01


def config_metrics(jsonl_path, target):
    """All build-split metrics for one config from its .jsonl + .json sidecar."""
    recs = load(jsonl_path)
    m = compute(recs)
    g = target_gap(recs, target)
    # controller diagnostics live per-record (alpha_mean, sat_frac, p_target_final)
    am = [r["alpha_mean"] for r in recs if r.get("alpha_mean") is not None]
    sf = [r["sat_frac"] for r in recs if r.get("sat_frac") is not None]
    pf = [r["p_target_final"] for r in recs if r.get("p_target_final") is not None]
    side = jsonl_path[:-6] + ".json" if jsonl_path.endswith(".jsonl") else jsonl_path + ".json"
    cfg = json.load(open(side)) if os.path.exists(side) else {}
    gains = cfg.get("gains", {})
    tag = os.path.basename(jsonl_path)[:-6]
    if tag.startswith("build_"):
        tag = tag[len("build_"):]          # build_C3.jsonl -> tag "C3"
    return {
        "tag": tag,
        "jsonl": jsonl_path,
        "cond": cfg.get("condition"),
        "Kp": gains.get("Kp"), "Ki": gains.get("Ki"), "Kd": gains.get("Kd"),
        "amax": cfg.get("alpha_max"), "setpoint": cfg.get("setpoint"),
        "cov": m["coverage"], "mu": m["mu"],
        "f90": (sum(1 for p in pf if p >= 0.9) / len(pf)) if pf else 0.0,
        "mean_alpha": (sum(am) / len(am)) if am else 0.0,
        "sat_frac": (sum(sf) / len(sf)) if sf else 0.0,
        "gdeb": g["pref_gap_debiased"], "graw": g["pref_gap_raw"],
    }


def select(rows, ref_tag, cov_min=COV_MIN_DEFAULT, gdeb_slack=GDEB_SLACK_DEFAULT):
    """Pure, deterministic decision over a list of config-metric dicts.
    Returns (winner_row, reason_str). Tie-breaks are total and reproducible."""
    ref = next((r for r in rows if r["tag"] == ref_tag), None)
    if ref is None:
        # fall back to the config with the smallest Ki as the 'default' reference
        ref = sorted(rows, key=lambda r: (r["Ki"] if r["Ki"] is not None else 9e9,
                                          r["tag"]))[0]
    f0, d0 = ref["f90"], ref["gdeb"]

    cov_ok = [r for r in rows if r["cov"] >= cov_min]
    eligible = [r for r in cov_ok if r["gdeb"] >= d0 - gdeb_slack]

    # deterministic sort key for the "maximize f90" objective (higher is better):
    #   f90 desc -> gdeb desc -> mean_alpha asc -> tag asc
    def f90_key(r):
        return (-r["f90"], -r["gdeb"], r["mean_alpha"], r["tag"])

    def gdeb_key(r):
        return (-r["gdeb"], -r["f90"], r["mean_alpha"], r["tag"])

    improvers = [r for r in eligible if r["f90"] > f0 + 1e-12]
    if improvers:
        # bucket by f90 within EPS_F90 of the max, then apply the tie-break chain
        max_f90 = max(r["f90"] for r in improvers)
        near = [r for r in improvers if abs(r["f90"] - max_f90) <= EPS_F90]
        winner = sorted(near, key=f90_key)[0]
        reason = (f"PRIMARY: max f90 among eligible improvers "
                  f"(cov>={cov_min}, gdeb>=d0-{gdeb_slack}); "
                  f"ref {ref['tag']} f0={f0:.3f} d0={d0:+.4f}")
    else:
        winner = sorted(cov_ok, key=gdeb_key)[0] if cov_ok else sorted(rows, key=gdeb_key)[0]
        reason = (f"FALLBACK: no eligible config improves actuation over ref "
                  f"{ref['tag']} (f0={f0:.3f}); picked max gdeb among coverage-passing")
    return winner, reason


def main():
    ap = argparse.ArgumentParser(description="Select tuned decode-PI config (jobAJ).")
    ap.add_argument("--sweep-dir", help="dir with build_*.jsonl sweep outputs")
    ap.add_argument("--glob", default="build_*.jsonl", help="jsonl glob within sweep-dir")
    ap.add_argument("--target", default="Black")
    ap.add_argument("--ref-tag", default=None,
                    help="tag of the default-PI reference (default: auto = min Ki)")
    ap.add_argument("--cov-min", type=float, default=COV_MIN_DEFAULT)
    ap.add_argument("--gdeb-slack", type=float, default=GDEB_SLACK_DEFAULT)
    ap.add_argument("--out", default=None, help="write CHOSEN.json here")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        sys.exit(0 if selftest() else 1)
    if not args.sweep_dir:
        ap.error("--sweep-dir required (or --selftest)")

    paths = sorted(glob.glob(os.path.join(args.sweep_dir, args.glob)))
    if not paths:
        raise SystemExit(f"FATAL: no sweep outputs match {args.sweep_dir}/{args.glob}")
    rows = [config_metrics(p, args.target) for p in paths]
    ref_tag = args.ref_tag or sorted(
        rows, key=lambda r: (r["Ki"] if r["Ki"] is not None else 9e9, r["tag"]))[0]["tag"]

    # print a stable table
    hdr = ("tag", "cond", "Kp", "Ki", "Kd", "amax", "cov", "f90",
           "mean_a", "sat", "gdeb", "graw", "mu")
    print("  ".join("%-10s" % h for h in hdr))
    for r in sorted(rows, key=lambda r: r["tag"]):
        print("  ".join([
            "%-10s" % r["tag"], "%-10s" % r["cond"],
            "%-10s" % r["Kp"], "%-10s" % r["Ki"], "%-10s" % r["Kd"],
            "%-10s" % r["amax"], "%-10.3f" % r["cov"], "%-10.3f" % r["f90"],
            "%-10.3f" % r["mean_alpha"], "%-10.3f" % r["sat_frac"],
            "%-+10.4f" % r["gdeb"], "%-+10.4f" % r["graw"], "%-10.4f" % r["mu"]]))

    winner, reason = select(rows, ref_tag, args.cov_min, args.gdeb_slack)
    print(f"\nREF = {ref_tag}")
    print(f"REASON: {reason}")
    print(f"WINNER: {winner['tag']}  cond={winner['cond']} "
          f"Kp={winner['Kp']} Ki={winner['Ki']} Kd={winner['Kd']} amax={winner['amax']} "
          f"| f90={winner['f90']:.3f} mean_alpha={winner['mean_alpha']:.3f} "
          f"gdeb={winner['gdeb']:+.4f} graw={winner['graw']:+.4f}")

    chosen = {
        "tag": winner["tag"], "cond": winner["cond"],
        "Kp": winner["Kp"], "Ki": winner["Ki"], "Kd": winner["Kd"],
        "amax": winner["amax"], "setpoint": winner["setpoint"],
        "build_f90": winner["f90"], "build_mean_alpha": winner["mean_alpha"],
        "build_gdeb": winner["gdeb"], "build_graw": winner["graw"],
        "build_cov": winner["cov"], "ref_tag": ref_tag, "reason": reason,
        "target": args.target,
    }
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump(chosen, fh, indent=2)
        print(f"\nwrote {args.out}")


def selftest():
    """Pure-decision tests on synthetic config-metric rows (no files, no model)."""
    ok = True

    def R(tag, ki, f90, gdeb, cov=1.0, alpha=3.0, kd=0.0):
        return {"tag": tag, "cond": "PI", "Kp": 3.0, "Ki": ki, "Kd": kd, "amax": 6.0,
                "setpoint": 0.9, "cov": cov, "mu": 0.3, "f90": f90, "mean_alpha": alpha,
                "sat_frac": 0.1, "gdeb": gdeb, "graw": 0.2}

    # Case 1: a stronger config raises f90 and keeps gdeb -> it wins (PRIMARY).
    rows = [R("C0", 0.1, 0.50, 0.030), R("C1", 0.3, 0.72, 0.028),
            R("C2", 0.5, 0.85, 0.031)]
    w, _ = select(rows, "C0")
    t1 = w["tag"] == "C2"
    ok &= t1; print(f"[selftest] primary picks highest-f90 eligible: {'PASS' if t1 else 'FAIL(%s)'%w['tag']}")

    # Case 2: the highest-f90 config COLLAPSES gdeb (saturation) -> excluded by guard.
    rows = [R("C0", 0.1, 0.50, 0.030), R("C1", 0.3, 0.70, 0.029),
            R("C2", 0.5, 0.95, 0.005)]  # C2 collapses gdeb below d0-0.01
    w, _ = select(rows, "C0")
    t2 = w["tag"] == "C1"
    ok &= t2; print(f"[selftest] gdeb guard excludes saturated config: {'PASS' if t2 else 'FAIL(%s)'%w['tag']}")

    # Case 3: nothing improves f90 -> FALLBACK to max gdeb among cov-passing.
    rows = [R("C0", 0.1, 0.60, 0.030), R("C1", 0.3, 0.58, 0.045),
            R("C2", 0.5, 0.55, 0.040)]
    w, r = select(rows, "C0")
    t3 = w["tag"] == "C1" and "FALLBACK" in r
    ok &= t3; print(f"[selftest] fallback picks max gdeb: {'PASS' if t3 else 'FAIL(%s)'%w['tag']}")

    # Case 4: coverage gate drops an incoherent config even if f90 is highest.
    rows = [R("C0", 0.1, 0.50, 0.030), R("C1", 0.3, 0.70, 0.031),
            R("C2", 0.5, 0.99, 0.031, cov=0.80)]  # incoherent
    w, _ = select(rows, "C0")
    t4 = w["tag"] == "C1"
    ok &= t4; print(f"[selftest] coverage gate drops incoherent config: {'PASS' if t4 else 'FAIL(%s)'%w['tag']}")

    # Case 5: determinism -- ties on f90 resolved by gdeb then alpha then tag.
    rows = [R("C0", 0.1, 0.50, 0.030),
            R("Cx", 0.3, 0.80, 0.033, alpha=4.0),
            R("Cy", 0.5, 0.805, 0.033, alpha=3.5)]  # within EPS on f90, same gdeb -> lower alpha wins
    w, _ = select(rows, "C0")
    t5 = w["tag"] == "Cy"
    ok &= t5; print(f"[selftest] deterministic tie-break (lower alpha): {'PASS' if t5 else 'FAIL(%s)'%w['tag']}")

    print(f"[selftest] OVERALL: {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    main()
