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
     PARITY: the family ALSO reports the rest of the prior-work baseline
     suite at the LLaDA balanced operating points (DREAM_BASELINE_CONDS:
     meanact unit s2, linearact gaussian s1, aura inject g4, aura vanilla,
     itic K48 a8 -- slurm/round3_jobY/jobZ; directions/gates fitted from
     DREAM activations), soft-missing until those runs land.

  5. GENDER BALANCED (E3)  results/balanced_all/{woman,man}/
     {base,decode_pid,normal,caa}/rot{0,1,2} -- pooled strict gap + 10k
     bootstrap CIs + Delta-g vs the SAME target's pooled base + gap@A/gap@BC
     + invalid rates, exactly like family 1. The normal dose is discovered
     from the files on disk (cond_normal_<T>_a*.json -- the alpha is fixed at
     submit time by the jobH sweep). Conditions not run yet are reported as
     MISSING, never crash.
     E8: the family ALSO reports the full prior-work baseline suite at the
     round-1 published operating points (results/BASELINES.md; jobQ/jobR):
     caa L14 a16, actadd a16, meanact unit s2, linearact gaussian s1,
     aura inject g4, aura vanilla, itic K48 a8 -- same sub-dirs + stems as
     results/balanced_all/black (BASELINE_CONDS), per target, soft-missing.

  6. FBLACK BALANCED (E6)  results/balanced_all/fblack/
     {base,decode_pid,normal,caa}/rot{0,1,2} -- same soft-missing machinery
     as family 5 (shared family_soft_targets). The fblack direction is a
     GENDER-CONDITIONED RACE direction (f-black vs other-race women; 0
     m-black negatives -- see multirace/build_arrows.py), NOT a full
     intersectional contrast. CRITICAL COMPARISON: once the fblack runs
     exist, the fblack pooled strict gaps are printed next to the round-1
     COARSE black direction's pooled gaps (recomputed from the committed
     results/balanced/results_balanced samples; black base gap +0.018,
     decode-PI gap +0.167). Different item sets, so the comparison is
     descriptive (no paired bootstrap).

  7. SES BALANCED (E9)  results/balanced_all/{lowses,highses}/
     {base,decode_pid,normal,caa}/rot{0,1,2} -- same soft-missing machinery
     as family 5 (shared family_soft_targets): pooled strict gap + CIs +
     Delta-g vs the SAME pole's pooled base + gap@A/gap@BC + decode-PI
     telemetry; the normal dose is discovered from disk; MISSING pre-run.

  8. AGE BALANCED (E9)  results/balanced_all/{old,young}/
     {base,decode_pid,normal,caa}/rot{0,1,2} -- identical machinery. The
     'young' target tag-matches BBQ's 'nonOld' tag (multirace/targets.py).

  9. LLADA-MOE BALANCED  results/lladamoe_balanced/{base,decode_pid,caa,
     actadd,meanact,linearact,aura_inject,aura_vanilla,itic}/rot{0,1,2}
     (LLaDA-MoE-7B-A1B-Instruct; llada_moe port, round4_jobAB..jobAD).
     Delta-g vs the pooled lladamoe base.  FULLY soft: the family prints
     NOTHING while results/lladamoe_balanced does not exist (output stays
     byte-identical pre-round-4); dosed stems (decode-PI amax, caa/actadd
     alpha) are discovered from disk like normal_stems.

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
import glob
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

# Dream baseline PARITY: the rest of the LLaDA prior-work baseline suite, at
# the SAME balanced operating points + stems as results/balanced_all/black
# (BASELINE_CONDS below), fitted from DREAM activations and run on the same
# rotations by slurm/round3_jobY (fits) + round3_jobZ (runs).  (sub_dir, stem,
# print label, result key); soft-missing until those jobs land.  Dream's caa
# (L14 a2) / actadd (a8) doses stay the round-2 Dream-faithful mult-2 picks --
# they are NOT duplicated at the LLaDA alpha16 (alpha is norm-relative).
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
    # Parity baseline suite (jobY/jobZ), soft-missing pre-run.
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
# Families 5/6: soft-missing per-target balanced families (E3 gender woman/man,
# E6 fblack) -- report MISSING pre-run, never crash.
# ---------------------------------------------------------------------------
GENDER_TARGETS = ["woman", "man"]
FBLACK_TARGETS = ["fblack"]
SES_TARGETS = ["lowses", "highses"]   # E9 (family 7)
AGE_TARGETS = ["old", "young"]        # E9 (family 8; young = BBQ tag nonOld)


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
    """Dosed normal stems present in ALL 3 rotations (alpha fixed at submit
    time by the jobH/jobK sweep; discovered from disk, not hardcoded)."""
    stems = None
    for r in range(3):
        pat = os.path.join(REPO, "results/balanced_all/%s/normal/rot%d/cond_normal_%s_a*_samples.jsonl"
                           % (t, r, t))
        got = {os.path.basename(p)[:-len("_samples.jsonl")]
               for p in glob.glob(pat)}
        stems = got if stems is None else (stems & got)
    return sorted(stems or [])


# E8: the full prior-work baseline suite at the round-1 published operating
# points (results/BASELINES.md). (sub_dir, stem, print label, result key) --
# sub-dirs and stems are EXACTLY the round-1 results/balanced_all/black layout,
# so slurm/round3_jobQ/jobR write where this discovery looks. Result keys are
# explicit because the baselines-CAA (L14 a16) must not shadow the multirace
# caa-a2 condition ("caa") in the family dict.
BASELINE_CONDS = [
    ("caa", "cond_caa_L14_a16", "caa L14 a16", "caa_L14_a16"),
    ("actadd", "cond_actadd_a16", "actadd a16", "actadd_a16"),
    ("meanact", "cond_meanact_unit_s2", "meanact unit s2", "meanact_unit_s2"),
    ("linearact", "cond_gaussian_s1", "linearact gauss s1", "linearact_gaussian_s1"),
    ("aura_inject", "cond_inject_g4", "aura inject g4", "aura_inject_g4"),
    ("aura_vanilla", "cond_vanilla", "aura vanilla", "aura_vanilla"),
    ("itic", "cond_itic_K48_a8", "itic K48 a8", "itic_K48_a8"),
]


def family_soft_targets(results, key, title, targets, include_baselines=False):
    """Shared soft-missing family: pooled strict gap + CIs + Delta-g vs the
    SAME target's pooled base + gap@A/gap@BC + invalid rates + decode-PI
    telemetry, per target in `targets`. include_baselines additionally reports
    the round-1 prior-work baseline suite (BASELINE_CONDS), soft-missing.
    Returns the family dict."""
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


def family_gender(results):
    family_soft_targets(results, "gender",
                        "FAMILY 5: GENDER BALANCED (E3, pooled 3x400, strict; "
                        "E8 incl. prior-work baseline suite)",
                        GENDER_TARGETS, include_baselines=True)


def family_fblack(results):
    fam = family_soft_targets(
        results, "fblack",
        "FAMILY 6: FBLACK BALANCED (E6 gender-conditioned race direction, "
        "pooled 3x400, strict)",
        FBLACK_TARGETS)
    # CRITICAL COMPARISON: fblack (gender-conditioned race direction: f-black
    # vs other-race women, see multirace/build_arrows.py) vs the round-1
    # COARSE black direction, recomputed from the committed round-1 balanced
    # samples (results/balanced/results_balanced; pooled strict gaps: base
    # +0.018, decode-PI +0.167). Different item sets and different steering
    # vectors -> descriptive comparison, no paired bootstrap. Printed only
    # once the fblack runs exist.
    fb = fam.get("fblack") or {}
    if not fb.get("base"):
        print("   (fblack-vs-black comparison: skipped -- fblack base not run yet)")
        return
    cmp_out = {}
    print("-- fblack (gender-conditioned race direction: f-black vs "
          "other-race women) vs black (coarse, round 1) -- "
          "descriptive: different item sets --")
    pairs = [("base", "cond_base"), ("decode_pid", "cond_dpid_PI")]
    for sub, black_stem in pairs:
        if not fb.get(sub):
            continue
        blk = analyze(rots("results/balanced/results_balanced/rot%%d/%s_samples.jsonl"
                           % black_stem))
        f, b = fb[sub], blk
        cmp_out[sub] = {"fblack_gap": f["gap"], "fblack_gap_ci95": f["gap_ci95"],
                        "black_gap": b["gap"], "black_gap_ci95": b["gap_ci95"],
                        "diff": f["gap"] - b["gap"]}
        print("%-22s fblack %+6.3f [%+6.3f,%+6.3f]  vs  black %+6.3f "
              "[%+6.3f,%+6.3f]  (diff %+6.3f)"
              % (sub, f["gap"], f["gap_ci95"][0], f["gap_ci95"][1],
                 b["gap"], b["gap_ci95"][0], b["gap_ci95"][1],
                 f["gap"] - b["gap"]))
    if fb.get("decode_pid") and cmp_out.get("base") and cmp_out.get("decode_pid"):
        dg_f = fb["decode_pid"].get("dgap_vs_base")
        dg_b = cmp_out["decode_pid"]["black_gap"] - cmp_out["base"]["black_gap"]
        cmp_out["dgap_decode_pid"] = {"fblack": dg_f, "black": dg_b}
        print("%-22s fblack dg=%+6.3f  vs  black dg=%+6.3f"
              % ("decode_pid dgap", dg_f, dg_b))
    results["fblack_vs_black"] = cmp_out


def family_ses(results):
    family_soft_targets(results, "ses",
                        "FAMILY 7: SES BALANCED (E9, pooled 3x400, strict)",
                        SES_TARGETS)


def family_age(results):
    family_soft_targets(results, "age",
                        "FAMILY 8: AGE BALANCED (E9, pooled 3x400, strict; "
                        "young = BBQ tag nonOld)",
                        AGE_TARGETS)


# ---------------------------------------------------------------------------
# Family 9: LLaDA-MoE balanced (results/lladamoe_balanced; llada_moe port,
# slurm/round4_jobAB..jobAD).  FULLY soft: when the directory does not exist
# yet (pre-round-4 trees) the family prints NOTHING and adds no result key, so
# the script's output stays byte-identical.  Once round4_jobAD lands, the 9
# conditions (base, decode-PI ours, caa, actadd, meanact, linearact,
# aura-inject, aura-vanilla, itic) are discovered from disk -- glob stems
# present in all 3 rotations, like normal_stems -- because the caa/actadd/
# decode-PI doses are fixed at submit time from the round4_jobAB sweep
# ($MOE_AMAX / $MOE_CAA_ALPHA / $MOE_ACTADD_ALPHA), while the fixed-point
# suite stems replicate the LLaDA balanced operating points exactly
# (cond_meanact_unit_s2 / cond_gaussian_s1 / cond_inject_g4 / cond_vanilla /
# cond_itic_K48_a8).  Delta-g is vs the pooled lladamoe base.
# ---------------------------------------------------------------------------
LLADAMOE_DIR = "results/lladamoe_balanced"
LLADAMOE_SUBS = ["decode_pid", "caa", "actadd", "meanact", "linearact",
                 "aura_inject", "aura_vanilla", "itic"]


def lladamoe_stems(sub):
    """Condition stems present in ALL 3 rotations of one lladamoe sub-dir
    (dose fixed at submit time; discovered from disk, not hardcoded)."""
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
        return   # pre-run trees: print nothing, keep output byte-identical
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
    # telemetry: decode-PI controller effort (only if the runs exist)
    for stem in lladamoe_stems("decode_pid"):
        tele_paths = rots("%s/decode_pid/rot%%d/%s.json" % (LLADAMOE_DIR, stem))
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
    family_gender(results)
    family_fblack(results)
    family_ses(results)
    family_age(results)
    family_lladamoe(results)
    telemetry(results)
    family_unqover(results)

    if args.json:
        with open(args.json, "w") as f:
            json.dump(results, f, indent=1)
        print("\nwrote %s" % args.json)


if __name__ == "__main__":
    main()
