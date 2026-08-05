#!/usr/bin/env python3
"""Round-2 strict-parse + pooling + bootstrap analysis.

Extends balanced_all/strict_pool.py (same STRICT rule, same pooling over
rot0/rot1/rot2, same 10,000-resample item-level bootstrap with
numpy default_rng(seed=0) fresh per condition) to the round-2 families:

  1. MULTI-TARGET BALANCED  results/balanced_all/{arab,asian,latino,white}/
     {base,decode_pid,normal}/rot{0,1,2}  -- samples use ``target_idx``
     (handled by strict_pool.target_index). Delta-g is vs the SAME target's
     pooled balanced base. Also recomputes the UNROTATED strict gaps from
     results/multirace/<T>/... at the same doses for comparison.
  2. SEEDS  results/balanced_seeds/seed{1,2,3}/{base,decode_pid}/rot{0,1,2}
     -- per-seed pooled strict gap for base and decode-PI, plus mean +/- sd
     over seeds.
  4. DREAM BALANCED  results/dream_balanced/{base,decode_pid,caa,actadd}/
     rot{0,1,2}  (Dream-v0-Instruct-7B). Delta-g vs the pooled dream base.

  T. TELEMETRY  mean_alpha / mean_sat_frac of every new decode-PI run,
     averaged over its 3 rotations (read from the summary cond_*.json).

  3. UNQOVER V2  results/unqover_v2/uq_{clean,decode_PI,layer_PI_a2,
     normal_a4}.jsonl -- two-choice subject-pick schema (``pred_subject``),
     scored with unqover/unqover_metric.py (official mu/eta/delta on
     quadruple-complete instances + BBQ-comparable pref_gap toward Black).
     Parse coverage = rows with a parsed pick / rows.

CIs: 95% percentile bootstrap of the strict gap (10k resamples, seed 0).
Delta-g CI: difference of the condition's and its baseline's bootstrap
replicates (both seed 0, independent draws).

Usage (repo root, branch portable-paths):
  python balanced_all/strict_round2.py [--json OUT.json] [--skip-unrotated]
"""
import argparse
import importlib.util
import json
import os
import statistics
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import strict_pool as sp  # noqa: E402  (STRICT rule, score(), bootstrap)

# ---------------------------------------------------------------------------
N_BOOT, SEED = sp.N_BOOT, sp.SEED


def gap_replicates(signs, seed=SEED):
    rng = np.random.default_rng(seed)
    v = np.asarray(signs, dtype=np.int8)
    idx = rng.integers(0, len(v), size=(N_BOOT, len(v)))
    return v[idx].mean(axis=1)


def analyze(paths, want_reps=False):
    """Pooled strict + permissive scoring with bootstrap CI over `paths`."""
    for p in paths:
        if not os.path.exists(p):
            sys.exit("MISSING: %s" % p)
    rows = sp.load_rows(paths)
    s = sp.score(rows, sp.strict_letter)
    perm = sp.score(rows, sp.permissive_letter)
    reps = gap_replicates(s["signs"])
    lo, hi = (float(np.percentile(reps, 2.5)), float(np.percentile(reps, 97.5)))
    out = {
        "n": s["n"], "counts": s["counts"],
        "target": float(s["target"]), "comparator": float(s["comparator"]),
        "abstain": float(s["abstain"]), "invalid": float(s["invalid"]),
        "gap": float(s["gap"]), "gap_ci95": [lo, hi],
        "gap_A": float(s.get("gap_A", 0)), "gap_BC": float(s.get("gap_BC", 0)),
        "perm_gap": float(perm["gap"]), "perm_invalid": float(perm["invalid"]),
    }
    if want_reps:
        out["_reps"] = reps
    return out


def dgap_ci(reps_cond, reps_base):
    d = reps_cond - reps_base
    return [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]


def rots(tmpl):
    return [os.path.join(REPO, tmpl % r) for r in range(3)]


def fmt(a, dg=None, dg_ci=None):
    row = ("%6.3f %6.3f %6.3f %6.3f | %+6.3f [%+6.3f,%+6.3f] | %+6.3f %+6.3f"
           % (a["target"], a["comparator"], a["abstain"], a["invalid"],
              a["gap"], a["gap_ci95"][0], a["gap_ci95"][1],
              a["gap_A"], a["gap_BC"]))
    if dg is not None:
        row += " | dg=%+6.3f [%+6.3f,%+6.3f]" % (dg, dg_ci[0], dg_ci[1])
    return row


def summary_avg(json_paths, keys):
    """Average numeric summary-JSON fields over rotations."""
    vals = {k: [] for k in keys}
    for p in json_paths:
        with open(p) as f:
            d = json.load(f)
        for k in keys:
            if d.get(k) is not None:
                vals[k].append(float(d[k]))
    return {k: (sum(v) / len(v) if v else None) for k, v in vals.items()}


def verify_summaries(json_paths, n_expect=400):
    """Every summary's counts must sum to its n (= n_expect)."""
    for p in json_paths:
        with open(p) as f:
            d = json.load(f)
        c = d["counts"]
        assert d["n"] == n_expect and sum(c.values()) == n_expect, \
            "count mismatch in %s" % p


# ---------------------------------------------------------------------------
# Family 1: multi-target balanced (+ unrotated multirace comparison)
# ---------------------------------------------------------------------------
MT_DOSES = {"arab": "a4", "asian": "a2", "latino": "a4", "white": "a1"}
MT_COND = [("base", "cond_dpid_%s_base"), ("decode_pid", "cond_dpid_%s_PI"),
           ("normal", "cond_normal_%s_%s")]


def mt_paths(t, sub, stem, ext="_samples.jsonl"):
    return rots("results/balanced_all/%s/%s/rot%%d/%s%s" % (t, sub, stem, ext))


def family_multitarget(results, skip_unrotated):
    print("\n== FAMILY 1: MULTI-TARGET BALANCED (pooled 3x400, strict) ==")
    hdr = ("%-22s %6s %6s %6s %6s | %25s | %6s %6s" %
           ("target/condition", "target", "compar", "abstn", "inval",
            "gap [95% CI]", "gap@A", "gap@BC"))
    print(hdr)
    fam = {}
    for t in ["arab", "asian", "latino", "white"]:
        base_stem = "cond_dpid_%s_base" % t
        base = analyze(mt_paths(t, "base", base_stem), want_reps=True)
        conds = {"base": base}
        for sub, stem_t in MT_COND[1:]:
            stem = stem_t % ((t, MT_DOSES[t]) if sub == "normal" else t)
            verify_summaries(mt_paths(t, sub, stem, ext=".json"))
            conds[sub] = analyze(mt_paths(t, sub, stem), want_reps=True)
        verify_summaries(mt_paths(t, "base", base_stem, ext=".json"))
        for sub in ["base", "decode_pid", "normal"]:
            a = conds[sub]
            label = "%s %s" % (t, sub if sub != "normal"
                               else "normal %s" % MT_DOSES[t])
            if sub == "base":
                print("%-22s %s" % (label, fmt(a)))
            else:
                dg = a["gap"] - base["gap"]
                ci = dgap_ci(a["_reps"], base["_reps"])
                print("%-22s %s" % (label, fmt(a, dg, ci)))
                a["dgap_vs_base"], a["dgap_ci95"] = dg, ci
        for a in conds.values():
            a.pop("_reps", None)
        fam[t] = conds

        if not skip_unrotated:
            # unrotated (results/multirace) strict recompute at the same doses
            unrot = {
                "base": "results/multirace/%s/decode_pid/cond_dpid_%s_base_samples.jsonl" % (t, t),
                "decode_pid": "results/multirace/%s/decode_pid/cond_dpid_%s_PI_samples.jsonl" % (t, t),
                "normal": "results/multirace/%s/normal/cond_normal_%s_%s_samples.jsonl" % (t, t, MT_DOSES[t]),
            }
            for sub, rel in unrot.items():
                p = os.path.join(REPO, rel)
                if not os.path.exists(p):
                    print("   (unrotated %s/%s missing: %s)" % (t, sub, rel))
                    continue
                u = analyze([p])
                fam[t][sub]["unrotated"] = u
                print("   unrot %-14s %s" % (sub, fmt(u)))
    results["multitarget"] = fam


# ---------------------------------------------------------------------------
# Family 2: seeds
# ---------------------------------------------------------------------------
def family_seeds(results):
    print("\n== FAMILY 2: SEED REPLICATION (pooled 3x400 per seed, strict) ==")
    fam = {}
    gaps = {"base": [], "decode_pid": []}
    for s in [1, 2, 3]:
        fam["seed%d" % s] = {}
        for sub, stem in [("base", "cond_dpid_base"), ("decode_pid", "cond_dpid_PI")]:
            tmpl = "results/balanced_seeds/seed%d/%s/rot%%d/%s" % (s, sub, stem)
            verify_summaries(rots(tmpl + ".json"))
            a = analyze(rots(tmpl + "_samples.jsonl"))
            fam["seed%d" % s][sub] = a
            gaps[sub].append(a["gap"])
            print("seed%d %-11s %s" % (s, sub, fmt(a)))
    for sub in ["base", "decode_pid"]:
        m = statistics.mean(gaps[sub])
        sd = statistics.stdev(gaps[sub])
        fam["%s_gap_mean" % sub], fam["%s_gap_sd" % sub] = m, sd
        print("%-11s over seeds: mean %+0.4f  sd %0.4f  (per-seed: %s)"
              % (sub, m, sd, ", ".join("%+0.3f" % g for g in gaps[sub])))
    results["seeds"] = fam


# ---------------------------------------------------------------------------
# Family 4: Dream balanced
# ---------------------------------------------------------------------------
DREAM_COND = [("base", "cond_dpid_base"), ("decode_pid", "cond_dpid_PI"),
              ("caa", "cond_caa_L14_a2"), ("actadd", "cond_actadd_a8")]


def family_dream(results):
    print("\n== FAMILY 4: DREAM BALANCED (Dream-v0-Instruct-7B, pooled 3x400) ==")
    fam = {}
    base = None
    for sub, stem in DREAM_COND:
        tmpl = "results/dream_balanced/%s/rot%%d/%s" % (sub, stem)
        verify_summaries(rots(tmpl + ".json"))
        a = analyze(rots(tmpl + "_samples.jsonl"), want_reps=True)
        if sub == "base":
            base = a
            print("%-12s %s" % (sub, fmt(a)))
        else:
            dg = a["gap"] - base["gap"]
            ci = dgap_ci(a["_reps"], base["_reps"])
            a["dgap_vs_base"], a["dgap_ci95"] = dg, ci
            print("%-12s %s" % (sub, fmt(a, dg, ci)))
        fam[sub] = a
    for a in fam.values():
        a.pop("_reps", None)
    results["dream"] = fam


# ---------------------------------------------------------------------------
# Telemetry (decode-PI controller effort, averaged over rotations)
# ---------------------------------------------------------------------------
def telemetry(results):
    print("\n== TELEMETRY: decode-PI mean_alpha / sat_frac (avg of 3 rotations) ==")
    tele = {}
    runs = [("multitarget %s (amax6)" % t,
             mt_paths(t, "decode_pid", "cond_dpid_%s_PI" % t, ext=".json"))
            for t in ["arab", "asian", "latino", "white"]]
    runs += [("seeds seed%d (amax6)" % s,
              rots("results/balanced_seeds/seed%d/decode_pid/rot%%d/cond_dpid_PI.json" % s))
             for s in [1, 2, 3]]
    runs += [("dream (amax1)",
              rots("results/dream_balanced/decode_pid/rot%d/cond_dpid_PI.json"))]
    for label, paths in runs:
        t = summary_avg(paths, ["mean_alpha", "mean_sat_frac"])
        tele[label] = t
        print("%-26s mean_alpha=%.2f  sat_frac=%.2f"
              % (label, t["mean_alpha"], t["mean_sat_frac"]))
    results["telemetry"] = tele


# ---------------------------------------------------------------------------
# Family 3: UNQOVER v2
# ---------------------------------------------------------------------------
UQ_CONDS = ["uq_clean", "uq_decode_PI", "uq_layer_PI_a2", "uq_normal_a4"]


def load_uq_metric():
    spec = importlib.util.spec_from_file_location(
        "unqover_metric", os.path.join(REPO, "unqover", "unqover_metric.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def family_unqover(results):
    print("\n== FAMILY 3: UNQOVER V2 (fixed generation-time parser) ==")
    uqm = load_uq_metric()
    fam = {}
    for cond in UQ_CONDS:
        rows = uqm.load(os.path.join(REPO, "results/unqover_v2/%s.jsonl" % cond))
        n = len(rows)
        no_ans = sum(1 for r in rows if r.get("pred_subject") is None)
        with open(os.path.join(REPO, "results/unqover_v2/%s.json" % cond)) as f:
            cfg = json.load(f)
        assert cfg.get("no_answer", no_ans) == no_ans, cond
        m = uqm.compute(rows)
        tg = uqm.target_gap(rows, "Black")
        fam[cond] = {
            "n_rows": n, "no_answer": no_ans,
            "parse_coverage": (n - no_ans) / n,
            "n_instances": m["n_instances"], "n_complete": m["n_complete"],
            "instance_coverage": m["coverage"],
            "mu": m["mu"], "eta": m["eta"], "delta": m["delta"],
            "mean_absC": m["mean_absC"],
            "pref_gap_raw": tg["pref_gap_raw"],
            "pref_gap_debiased": tg["pref_gap_debiased"],
            "n_black_instances": tg["n"],
        }
        f = fam[cond]
        print("%-16s rows=%4d no_ans=%2d (parse cov %.3f) | inst=%3d complete=%3d "
              "| mu=%.3f eta=%.3f delta=%.3f |C|=%.3f | prefB raw=%+.3f deb=%+.3f (n=%d)"
              % (cond, n, no_ans, f["parse_coverage"], m["n_instances"],
                 m["n_complete"], m["mu"], m["eta"], m["delta"], m["mean_absC"],
                 f["pref_gap_raw"], f["pref_gap_debiased"], tg["n"]))
    clean = fam["uq_clean"]
    print("-- deltas vs uq_clean --")
    for cond in UQ_CONDS[1:]:
        f = fam[cond]
        f["d_mu"] = f["mu"] - clean["mu"]
        f["d_pref_gap_raw"] = f["pref_gap_raw"] - clean["pref_gap_raw"]
        f["d_pref_gap_debiased"] = f["pref_gap_debiased"] - clean["pref_gap_debiased"]
        print("%-16s d_mu=%+.3f  d_prefB_raw=%+.4f  d_prefB_debiased=%+.4f"
              % (cond, f["d_mu"], f["d_pref_gap_raw"], f["d_pref_gap_debiased"]))
    results["unqover_v2"] = fam


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None)
    ap.add_argument("--skip-unrotated", action="store_true",
                    help="skip the unrotated results/multirace comparison")
    args = ap.parse_args()

    results = {}
    family_multitarget(results, args.skip_unrotated)
    family_seeds(results)
    family_dream(results)
    telemetry(results)
    family_unqover(results)

    if args.json:
        with open(args.json, "w") as f:
            json.dump(results, f, indent=1)
        print("\nwrote %s" % args.json)


if __name__ == "__main__":
    main()
