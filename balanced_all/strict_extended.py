#!/usr/bin/env python3
"""Strict-parse + pooling + bootstrap analysis for the extended experiment families.

Uses the same strict rule, pooling over rot0/rot1/rot2, and 10,000-resample
item-level bootstrap (numpy default_rng(seed=0), fresh per condition) as
balanced_all/strict_pool.py. Families:

  1. MULTI-TARGET  results/balanced_all/{arab,asian,latino,white}/
     {base,decode_pid,normal}/rot{0,1,2}; samples use ``target_idx``.
     Delta-g vs the same target's pooled base. Prior-work baselines
     (BASELINE_CONDS) are added when present in all 3 rotations. Also
     recomputes the unrotated strict gaps from results/multirace/<T>/.
  2. SEEDS  results/balanced_seeds/seed{1,2,3}/{base,decode_pid}/rot{0,1,2};
     per-seed pooled strict gap plus mean +/- sd over seeds.
  4. DREAM  results/dream_balanced/{base,decode_pid,caa,actadd}/rot{0,1,2}
     (Dream-v0-Instruct-7B), plus the remaining prior-work baselines at the
     LLaDA operating points (DREAM_BASELINE_CONDS). Delta-g vs Dream base.
  8. AGE  results/balanced_all/old/... (Old target, with the baseline suite).
  9. LLADA-MOE  results/lladamoe_balanced/<cond>/rot{0,1,2}
     (LLaDA-MoE-7B-A1B-Instruct); condition stems are discovered from disk.
     Skipped entirely if the directory does not exist.
  T. TELEMETRY  mean_alpha / mean_sat_frac of the decode-PI runs, averaged
     over the 3 rotations (from the summary cond_*.json).

The age family uses family_soft_targets: missing conditions are reported as
MISSING instead of aborting; the open-loop ("normal") dose is discovered
from the files on disk.

CIs: 95% percentile bootstrap of the strict gap (10k resamples, seed 0).
Delta-g CI: difference of the condition's and its baseline's bootstrap
replicates, drawn with identical resample indices (seed 0), i.e. a paired bootstrap.

Usage (from the repo root):
  python balanced_all/strict_extended.py [--json OUT.json] [--skip-unrotated]
"""
import argparse
import glob
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
        # Prior-work baseline suite (BASELINE_CONDS), discovered from disk:
        # only sub-dirs present in all 3 rotations are analyzed. Delta-g vs
        # this target's own pooled base; result keys are the BASELINE_CONDS keys.
        for b_sub, b_stem, b_label, b_ckey in BASELINE_CONDS:
            b_paths = mt_paths(t, b_sub, b_stem)
            if not all(os.path.exists(p) for p in b_paths):
                continue
            ba = analyze_soft(b_paths, want_reps=True)
            if ba is None:
                continue
            b_dg = ba["gap"] - base["gap"]
            b_ci = dgap_ci(ba["_reps"], base["_reps"])
            ba["dgap_vs_base"], ba["dgap_ci95"] = b_dg, b_ci
            ba.pop("_reps", None)
            print("%-22s %s" % ("%s %s" % (t, b_label), fmt(ba, b_dg, b_ci)))
            conds[b_ckey] = ba
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

# Dream baseline parity: the rest of the prior-work baseline suite at the same
# balanced operating points + stems as results/balanced_all/black
# (BASELINE_CONDS below), fitted from Dream activations and run on the same
# rotations. Tuples are (sub_dir, stem, print label, result key); missing runs
# are reported, not fatal. Dream's caa (L14 a2) / actadd (a8) doses are kept
# separately and not duplicated at the LLaDA alpha16 (alpha is norm-relative).
DREAM_BASELINE_CONDS = [
    ("meanact", "cond_meanact_unit_s2", "meanact unit s2", "meanact_unit_s2"),
    ("linearact", "cond_gaussian_s1", "linearact gauss s1", "linearact_gaussian_s1"),
    ("aura_inject", "cond_inject_g4", "aura inject g4", "aura_inject_g4"),
    ("aura_vanilla", "cond_vanilla", "aura vanilla", "aura_vanilla"),
    ("itic", "cond_itic_K48_a8", "itic K48 a8", "itic_K48_a8"),
]


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
            print("%-18s %s" % (sub, fmt(a)))
        else:
            dg = a["gap"] - base["gap"]
            ci = dgap_ci(a["_reps"], base["_reps"])
            a["dgap_vs_base"], a["dgap_ci95"] = dg, ci
            print("%-18s %s" % (sub, fmt(a, dg, ci)))
        fam[sub] = a
    # Parity baseline suite; missing runs are reported, not fatal.
    for sub, stem, label, ckey in DREAM_BASELINE_CONDS:
        paths = rots("results/dream_balanced/%s/rot%%d/%s_samples.jsonl"
                     % (sub, stem))
        a = analyze_soft(paths, want_reps=True)
        if a is None:
            print("%-18s (MISSING -- not run yet)" % label)
            continue
        dg = a["gap"] - base["gap"]
        ci = dgap_ci(a["_reps"], base["_reps"])
        a["dgap_vs_base"], a["dgap_ci95"] = dg, ci
        print("%-18s %s" % (label, fmt(a, dg, ci)))
        fam[ckey] = a
    for a in fam.values():
        a.pop("_reps", None)
    results["dream"] = fam


# ---------------------------------------------------------------------------
# Family 8: per-target balanced family (age);
# missing conditions are reported as MISSING, never fatal.
# ---------------------------------------------------------------------------
AGE_TARGETS = ["old"]


def analyze_soft(paths, want_reps=False, n_expect=400):
    """analyze() that reports MISSING instead of exiting; verifies summaries."""
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        for p in missing:
            print("   MISSING: %s" % os.path.relpath(p, REPO))
        return None
    verify_summaries([p.replace("_samples.jsonl", ".json") for p in paths],
                     n_expect=n_expect)
    return analyze(paths, want_reps)


def normal_stems(t):
    """Dosed normal stems present in ALL 3 rotations (discovered from disk,
    not hardcoded)."""
    stems = None
    for r in range(3):
        pat = os.path.join(REPO, "results/balanced_all/%s/normal/rot%d/cond_normal_%s_a*_samples.jsonl"
                           % (t, r, t))
        got = {os.path.basename(p)[:-len("_samples.jsonl")]
               for p in glob.glob(pat)}
        stems = got if stems is None else (stems & got)
    return sorted(stems or [])


# Prior-work baseline suite at the main (Black) operating points.
# (sub_dir, stem, print label, result key); sub-dirs and stems follow the
# results/balanced_all/black layout. Result keys are explicit so that the
# baseline CAA (L14 a16) does not shadow the caa-a2 condition ("caa").
BASELINE_CONDS = [
    ("caa", "cond_caa_L14_a16", "caa L14 a16", "caa_L14_a16"),
    ("actadd", "cond_actadd_a16", "actadd a16", "actadd_a16"),
    ("meanact", "cond_meanact_unit_s2", "meanact unit s2", "meanact_unit_s2"),
    ("linearact", "cond_gaussian_s1", "linearact gauss s1", "linearact_gaussian_s1"),
    ("aura_inject", "cond_inject_g4", "aura inject g4", "aura_inject_g4"),
    ("aura_vanilla", "cond_vanilla", "aura vanilla", "aura_vanilla"),
    ("itic", "cond_itic_K48_a8", "itic K48 a8", "itic_K48_a8"),
]


def family_soft_targets(results, key, title, targets, include_baselines=False,
                        soft_baselines=False):
    """Shared soft-missing family: pooled strict gap + CIs + Delta-g vs the
    SAME target's pooled base + gap@A/gap@BC + invalid rates + decode-PI
    telemetry, per target in `targets`. include_baselines additionally reports
    the prior-work baseline suite (BASELINE_CONDS). With soft_baselines=False
    an absent baseline prints a MISSING line; with soft_baselines=True only the
    baselines present in ALL 3 rotations are appended."""
    print("\n== %s ==" % title)
    hdr = ("%-22s %6s %6s %6s %6s | %25s | %6s %6s" %
           ("target/condition", "target", "compar", "abstn", "inval",
            "gap [95% CI]", "gap@A", "gap@BC"))
    print(hdr)
    fam = {}
    for t in targets:
        conds = [("base", "cond_dpid_%s_base" % t, "base", "base"),
                 ("decode_pid", "cond_dpid_%s_PI" % t, "decode_pid", "decode_pid")]
        nstems = normal_stems(t)
        if not nstems:
            print("%-22s (MISSING -- no cond_normal_%s_a* in all 3 rotations)"
                  % ("%s normal" % t, t))
        conds += [("normal", stem, stem.replace("cond_", ""),
                   stem.replace("cond_", "")) for stem in nstems]
        conds += [("caa", "cond_caa_%s_a2" % t, "caa a2", "caa")]
        if include_baselines:
            if soft_baselines:
                conds += [c for c in BASELINE_CONDS
                          if all(os.path.exists(p) for p in mt_paths(t, c[0], c[1]))]
            else:
                conds += BASELINE_CONDS
        base = None
        fam[t] = {}
        for sub, stem, label, ckey in conds:
            a = analyze_soft(mt_paths(t, sub, stem), want_reps=True)
            if a is None:
                print("%-22s (MISSING -- not run yet)" % ("%s %s" % (t, label)))
                continue
            if sub == "base":
                base = a
                print("%-22s %s" % ("%s %s" % (t, label), fmt(a)))
            elif base is not None:
                dg = a["gap"] - base["gap"]
                ci = dgap_ci(a["_reps"], base["_reps"])
                a["dgap_vs_base"], a["dgap_ci95"] = dg, ci
                print("%-22s %s" % ("%s %s" % (t, label), fmt(a, dg, ci)))
            else:
                print("%-22s %s | dg=N/A (base missing)"
                      % ("%s %s" % (t, label), fmt(a)))
            fam[t][ckey] = a
        for a in fam[t].values():
            a.pop("_reps", None)
        # telemetry: decode-PI controller effort (only if the runs exist)
        tele_paths = mt_paths(t, "decode_pid", "cond_dpid_%s_PI" % t, ext=".json")
        if all(os.path.exists(p) for p in tele_paths):
            te = summary_avg(tele_paths, ["mean_alpha", "mean_sat_frac"])
            fam[t]["decode_pid_telemetry"] = te
            print("   telemetry %-11s mean_alpha=%.2f  sat_frac=%.2f"
                  % (t, te["mean_alpha"], te["mean_sat_frac"]))
    results[key] = fam
    return fam


def family_age(results):
    family_soft_targets(results, "age",
                        "FAMILY 8: AGE BALANCED (Old target, pooled 3x400, strict)",
                        AGE_TARGETS, include_baselines=True, soft_baselines=True)


# ---------------------------------------------------------------------------
# Family 9: LLaDA-MoE balanced (results/lladamoe_balanced; llada_moe port).
# If the directory does not exist the family prints nothing and adds no
# result key. Conditions (base, decode-PI, normal open-loop, caa, actadd,
# meanact, linearact, aura-inject, aura-vanilla, itic) are discovered from
# disk as stems present in all 3 rotations, since the dosed conditions are
# set by the run configuration; the fixed-point baseline stems match the
# LLaDA balanced operating points (cond_meanact_unit_s2 / cond_gaussian_s1 /
# cond_inject_g4 / cond_vanilla / cond_itic_K48_a8). Delta-g is vs the pooled
# lladamoe base. Secondary conditions: decode_pid_L9to12 (cond_dpid_PI_L9to12;
# PI --layers 9-12 --amax 1) and normal_L9to12 (cond_normal_L9to12_a1; its
# geometry/effort-matched open loop); these are skipped silently when absent
# (LLADAMOE_SOFT_SUBS). As in every family, each sample's model_output is
# re-parsed with strict_pool.strict_letter; summary counts are only
# consistency-checked, never scored, so the lenient eval-time parser cannot
# leak into the pooled strict table.
# ---------------------------------------------------------------------------
LLADAMOE_DIR = "results/lladamoe_balanced"
LLADAMOE_SUBS = ["decode_pid", "decode_pid_L9to12", "normal", "normal_L9to12",
                 "caa", "actadd", "meanact", "linearact", "aura_inject",
                 "aura_vanilla", "itic"]
# Secondary sub-dirs: skipped silently while absent.
LLADAMOE_SOFT_SUBS = {"decode_pid_L9to12", "normal_L9to12"}


def lladamoe_stems(sub):
    """Condition stems present in ALL 3 rotations of one lladamoe sub-dir
    (discovered from disk, not hardcoded)."""
    stems = None
    for r in range(3):
        pat = os.path.join(REPO, "%s/%s/rot%d/cond_*_samples.jsonl"
                           % (LLADAMOE_DIR, sub, r))
        got = {os.path.basename(p)[:-len("_samples.jsonl")]
               for p in glob.glob(pat)}
        stems = got if stems is None else (stems & got)
    return sorted(stems or [])


def family_lladamoe(results):
    if not os.path.isdir(os.path.join(REPO, LLADAMOE_DIR)):
        return   # no LLaDA-MoE results: print nothing
    print("\n== FAMILY 9: LLADA-MOE BALANCED (LLaDA-MoE-7B-A1B-Instruct, "
          "pooled 3x400) ==")
    fam = {}
    base = analyze_soft(
        rots(LLADAMOE_DIR + "/base/rot%d/cond_dpid_base_samples.jsonl"),
        want_reps=True)
    if base is None:
        print("%-18s (MISSING -- not run yet)" % "base")
    else:
        print("%-18s %s" % ("base", fmt(base)))
        fam["base"] = base
    for sub in LLADAMOE_SUBS:
        stems = lladamoe_stems(sub)
        if not stems:
            if sub not in LLADAMOE_SOFT_SUBS:
                print("%-18s (MISSING -- not run yet)" % sub)
            continue
        for stem in stems:
            a = analyze_soft(rots("%s/%s/rot%%d/%s_samples.jsonl"
                                  % (LLADAMOE_DIR, sub, stem)), want_reps=True)
            if a is None:
                continue
            label = stem.replace("cond_", "")
            if base is not None:
                dg = a["gap"] - base["gap"]
                ci = dgap_ci(a["_reps"], base["_reps"])
                a["dgap_vs_base"], a["dgap_ci95"] = dg, ci
                print("%-18s %s" % (label, fmt(a, dg, ci)))
            else:
                print("%-18s %s | dg=N/A (base missing)" % (label, fmt(a)))
            fam[label] = a
    for a in fam.values():
        a.pop("_reps", None)
    # telemetry: decode-PI controller effort (only if the runs exist), for the
    # headline decode_pid sub and the secondary decode_pid_L9to12.
    for sub in [s for s in LLADAMOE_SUBS if s.startswith("decode_pid")]:
        for stem in lladamoe_stems(sub):
            tele_paths = rots("%s/%s/rot%%d/%s.json" % (LLADAMOE_DIR, sub, stem))
            if all(os.path.exists(p) for p in tele_paths):
                te = summary_avg(tele_paths, ["mean_alpha", "mean_sat_frac"])
                fam["%s_telemetry" % stem.replace("cond_", "")] = te
                print("   telemetry %-11s mean_alpha=%.2f  sat_frac=%.2f"
                      % (stem.replace("cond_", ""), te["mean_alpha"],
                         te["mean_sat_frac"]))
    results["lladamoe"] = fam


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
    family_age(results)
    family_lladamoe(results)
    telemetry(results)

    if args.json:
        with open(args.json, "w") as f:
            json.dump(results, f, indent=1)
        print("\nwrote %s" % args.json)


if __name__ == "__main__":
    main()
