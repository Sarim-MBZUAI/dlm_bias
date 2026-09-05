#!/usr/bin/env python
"""BUILD / EVAL split for UNQOVER-race with STRICT axis disjointness.

WHY.  The previous UNQOVER track split instances at random: BUILD and EVAL were
instance-disjoint but shared context strings, all templates and all questions,
so a direction fitted on BUILD transferred partly because EVAL was the same
sentence frames and the same questions.  Here the split is a PRODUCT split over
the *generative* axes, and instances that straddle the axis assignment are
DISCARDED rather than silently assigned:

    BUILD = { i : attribute(i) in A_b and template(i) in T_b [and pair(i) in P_b] }
    EVAL  = { i : attribute(i) in A_e and template(i) in T_e [and pair(i) in P_e] }
    A_b n A_e = T_b n T_e = P_b n P_e = {}         (partitions of each axis)

AXES.  These are the official generation's own factors (loader.py):
    attribute   the UNQOVER attribute obj0. 50 total, 38 after the allowlist.
                This is the right question axis: it is what gamma(x,a) is
                indexed by, and it is coarser than a question string, so the
                surface paraphrases ("Who speed..."/"Who would speed...") that
                a qid-level rule would leak across splits cannot leak.
    template    the context template tid. 14 values.  Disjoint templates imply
                disjoint context strings, since a context is a template with
                two subject names substituted.
    pair        the unordered subject pair. 105 total, 94 after the
                within-family exclusion.  OPTIONAL third axis (--axes
                attribute,template,pair): stricter, but it costs about half the
                retained data and it thins the per-family instance counts.
                NOT the individual subjects -- with 15 names a subject-disjoint
                split would put the steering target in one split only.

DEFAULT.  attribute + template (the audited recommendation).  --relaxations
prints every subset so the trade-off is argued from numbers.

CLUSTERS.  Rows are not independent: 14 templates, 38 allowed attributes, 94
pairs, and `context` is perfectly confounded with the subject pair.  The
manifest reports, per split, the number of attribute clusters and
(pair, attribute) clusters -- the units any interval must be resampled over.

COST.  A product split over k axes retains ~p^k + (1-p)^k.  A 30/70 split at
100% retention is therefore impossible; p is set so the RATIO of what survives
is ~30/70 (p^k : (1-p)^k = 3 : 7) and the absolute counts are whatever the data
gives.

CPU only.
  python -m unqover_hf.splits --selftest
  python -m unqover_hf.splits                 # writes build_race / eval_race
  python -m unqover_hf.splits --relaxations --seed-spread
"""
import argparse
import json
import os
import random
import sys
from collections import Counter
from itertools import combinations

from unqover_hf.loader import (DEFAULT_OUT as CANONICAL_OUT, SUBJECT_FAMILY,
                               FAMILY_MEMBERS, load_jsonl, write_jsonl)

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_DATA = os.path.join(_ROOT, "data", "unqover_hirundo")

DEFAULT_BUILD = os.path.join(_DATA, "build_race.jsonl")
DEFAULT_EVAL = os.path.join(_DATA, "eval_race.jsonl")
DEFAULT_MANIFEST = os.path.join(_DATA, "split_manifest_race.json")

SEED = 42
DEFAULT_AXES = ("attribute", "template")

AXES = {"attribute": "attribute_id", "template": "template_id",
        "pair": "subject_pair"}
# Everything that must come out disjoint. attribute_id/act_cluster and
# template_id/tid are functionally linked; context follows from template.
VERIFY_FIELDS = {
    "attribute": ["attribute_id", "attribute", "q0_question", "q1_question"],
    "template": ["template_id", "tid", "context"],
    "pair": ["subject_pair"],
}


def axis_frac_for_ratio(n_axes, build_share=0.3):
    """p with p^k : (1-p)^k = build_share : (1-build_share)."""
    r = (build_share / (1.0 - build_share)) ** (1.0 / n_axes)
    return r / (1.0 + r)


def _values(recs, name):
    if name == "template":                     # context is derived, not a field
        return {r[AXES[name]] for r in recs}
    return {r[AXES[name]] for r in recs}


def _record_values(r, field):
    """The values of `field` carried by one instance (contexts: one per order)."""
    if field == "context":
        return [o["context"] for o in r["orders"]]
    return [r[field]]


def partition_axis(values, frac, rng):
    vals = sorted(values)
    rng.shuffle(vals)
    k = int(round(frac * len(vals)))
    k = max(1, min(len(vals) - 1, k))          # never leave a side empty
    return set(vals[:k]), set(vals[k:])


def make_split(recs, seed=SEED, frac=None, axes=DEFAULT_AXES):
    """Product split over `axes`. Returns (build, eval, dropped, assignment)."""
    axes = tuple(axes)
    if frac is None:
        frac = axis_frac_for_ratio(len(axes))
    rng = random.Random(seed)
    assign = {}
    for name in sorted(axes):                  # sorted -> stable rng consumption
        b, e = partition_axis(_values(recs, name), frac, rng)
        assign[name] = {"field": AXES[name], "build": sorted(b), "eval": sorted(e)}
    bsets = {n: set(assign[n]["build"]) for n in axes}
    esets = {n: set(assign[n]["eval"]) for n in axes}
    build, ev, dropped = [], [], []
    for r in recs:
        if all(r[AXES[n]] in bsets[n] for n in axes):
            build.append(r)
        elif all(r[AXES[n]] in esets[n] for n in axes):
            ev.append(r)
        else:
            dropped.append(r)
    return build, ev, dropped, assign


# --------------------------------------------------------------------------- #
def verify(build, ev, axes=DEFAULT_AXES, strict=True):
    """Recompute the overlap of every key implied by `axes`, plus instance_id.
    Returns {field: n_shared}; raises AssertionError on any leak when strict."""
    fields = [f for n in axes for f in VERIFY_FIELDS[n]] + ["instance_id"]
    out = {}
    for f in fields:
        bs = {v for r in build for v in _record_values(r, f)}
        es = {v for r in ev for v in _record_values(r, f)}
        shared = bs & es
        out[f] = len(shared)
        if strict and shared:
            raise AssertionError("SPLIT LEAK: %d shared value(s) of %r; e.g. %r"
                                 % (len(shared), f, sorted(shared)[:3]))
    return out


def subject_counts(recs):
    c = Counter()
    for r in recs:
        c[r["subj_a"]] += 1
        c[r["subj_b"]] += 1
    return dict(sorted(c.items()))


def family_counts(recs):
    c = Counter()
    for r in recs:
        for fam in {r["family_a"], r["family_b"]}:
            c[fam] += 1
    return {f: c.get(f, 0) for f in sorted(FAMILY_MEMBERS)}


def cluster_counts(recs):
    return {"attribute": len({r["attribute_id"] for r in recs}),
            "template": len({r["template_id"] for r in recs}),
            "subject_pair": len({r["subject_pair"] for r in recs}),
            "pair_x_attribute": len({r["cluster_key"] for r in recs})}


# --------------------------------------------------------------------------- #
def relaxation_table(recs, seed=SEED):
    rows = []
    names = sorted(AXES)
    for k in range(len(names), 0, -1):
        for combo in combinations(names, k):
            b, e, d, _ = make_split(recs, seed=seed, axes=combo)
            rows.append({"axes": list(combo), "build": len(b), "eval": len(e),
                         "dropped": len(d),
                         "retained_frac": round((len(b) + len(e)) / len(recs), 4),
                         "build_frac_of_kept": round(len(b) / max(1, len(b) + len(e)), 4),
                         "black_build": family_counts(b)["black"],
                         "black_eval": family_counts(e)["black"],
                         "clusters_build": cluster_counts(b),
                         "clusters_eval": cluster_counts(e)})
    # reference: the previous track's rule (random instance split, no axes)
    rng = random.Random(seed)
    order = list(range(len(recs)))
    rng.shuffle(order)
    cut = int(round(0.3 * len(recs)))
    b = [recs[i] for i in sorted(order[:cut])]
    e = [recs[i] for i in sorted(order[cut:])]
    rows.append({"axes": ["<none: random instance split>"], "build": len(b),
                 "eval": len(e), "dropped": 0, "retained_frac": 1.0,
                 "build_frac_of_kept": round(len(b) / len(recs), 4),
                 "black_build": family_counts(b)["black"],
                 "black_eval": family_counts(e)["black"],
                 "clusters_build": cluster_counts(b), "clusters_eval": cluster_counts(e),
                 "leak": verify(b, e, axes=tuple(AXES), strict=False)})
    return rows


def seed_spread(recs, seeds=range(40, 50), axes=DEFAULT_AXES):
    out = []
    for s in seeds:
        b, e, _, _ = make_split(recs, seed=s, axes=axes)
        out.append({"seed": s, "build": len(b), "eval": len(e),
                    "retained_frac": round((len(b) + len(e)) / len(recs), 4),
                    "black_build": family_counts(b)["black"],
                    "black_eval": family_counts(e)["black"]})
    return out


# --------------------------------------------------------------------------- #
def build_manifest(canonical, usable, build, ev, dropped, assign, seed, frac, axes, leaks):
    return {
        "dataset": "unqover-race (hirundo selection, completed from "
                   "allenai/unqover ethnicity.source.json)",
        "canonical_instances": len(canonical),
        "usable_instances": len(usable),
        "seed": seed,
        "axes": list(axes),
        "axis_frac": frac,
        "constraint": ("BUILD and EVAL are disjoint on ALL of: %s. Disjoint "
                       "attributes imply disjoint q0/q1 question strings; disjoint "
                       "templates imply disjoint context strings. Instances "
                       "straddling the axis assignment are DISCARDED."
                       % ", ".join("%s (%s)" % (n, AXES[n]) for n in axes)),
        "input_filters": ("usable only: attribute in the 38-attribute allowlist "
                          "AND the two subjects come from different families"),
        "axis_partition": {n: {"field": a["field"], "n_build": len(a["build"]),
                               "n_eval": len(a["eval"]),
                               "build": a["build"], "eval": a["eval"]}
                           for n, a in assign.items()},
        "counts": {
            "build": len(build), "eval": len(ev), "dropped": len(dropped),
            "retained_frac": round((len(build) + len(ev)) / max(1, len(usable)), 4),
            "build_frac_of_usable": round(len(build) / max(1, len(usable)), 4),
            "eval_frac_of_usable": round(len(ev) / max(1, len(usable)), 4),
            "build_frac_of_kept": round(len(build) / max(1, len(build) + len(ev)), 4),
            "generations_build": 4 * len(build), "generations_eval": 4 * len(ev),
        },
        "clusters": {"build": cluster_counts(build), "eval": cluster_counts(ev)},
        "disjointness_verification": leaks,
        "shared_context_strings": leaks.get("context", None),
        "family_instances": {"build": family_counts(build), "eval": family_counts(ev)},
        "subject_instances": {"build": subject_counts(build), "eval": subject_counts(ev)},
        "family_members": FAMILY_MEMBERS,
        "templates": {"build": sorted({r["template_id"] for r in build}),
                      "eval": sorted({r["template_id"] for r in ev})},
        "attributes": {"build": sorted({r["attribute_id"] for r in build}),
                       "eval": sorted({r["attribute_id"] for r in ev})},
    }


def print_manifest(man):
    p = print
    c = man["counts"]
    p("=" * 76)
    p("UNQOVER-race BUILD/EVAL split  (seed=%s, axes=%s, axis_frac=%.4f)"
      % (man["seed"], "+".join(man["axes"]), man["axis_frac"]))
    p("=" * 76)
    p("  canonical quadruples : %d   usable (post-filter): %d"
      % (man["canonical_instances"], man["usable_instances"]))
    p("  BUILD                : %5d instances (%5d generations)  %.1f%% of usable"
      % (c["build"], c["generations_build"], 100 * c["build_frac_of_usable"]))
    p("  EVAL                 : %5d instances (%5d generations)  %.1f%% of usable"
      % (c["eval"], c["generations_eval"], 100 * c["eval_frac_of_usable"]))
    p("  DISCARDED (straddle) : %5d" % c["dropped"])
    p("  retained             : %.1f%%   BUILD share of retained = %.1f%%"
      % (100 * c["retained_frac"], 100 * c["build_frac_of_kept"]))
    p("")
    p("  AXIS PARTITION")
    for n in sorted(man["axis_partition"]):
        a = man["axis_partition"][n]
        p("    %-10s (%-12s) build=%-3d eval=%-3d" % (n, a["field"], a["n_build"],
                                                      a["n_eval"]))
    p("")
    p("  CLUSTER COUNTS PER SPLIT (resample these, never rows)")
    p("    %-6s %10s %9s %13s %17s" % ("split", "attribute", "template",
                                       "subject_pair", "pair_x_attribute"))
    for side in ("build", "eval"):
        cc = man["clusters"][side]
        p("    %-6s %10d %9d %13d %17d" % (side, cc["attribute"], cc["template"],
                                           cc["subject_pair"], cc["pair_x_attribute"]))
    p("")
    p("  DISJOINTNESS VERIFICATION (shared values BUILD n EVAL; all must be 0)")
    for k, v in man["disjointness_verification"].items():
        p("    %-16s %d   %s" % (k, v, "OK" if v == 0 else "*** LEAK ***"))
    p("")
    p("  INSTANCES CONTAINING EACH SUBJECT FAMILY (merged steering targets)")
    p("    %-10s %8s %8s   members" % ("family", "build", "eval"))
    for fam in sorted(man["family_instances"]["build"]):
        p("    %-10s %8d %8d   %s"
          % (fam, man["family_instances"]["build"][fam],
             man["family_instances"]["eval"][fam], ", ".join(man["family_members"][fam])))
    p("")
    p("  INSTANCES CONTAINING EACH LITERAL SUBJECT")
    sb, se = man["subject_instances"]["build"], man["subject_instances"]["eval"]
    for s in sorted(set(sb) | set(se)):
        p("    %-18s %6d %6d" % (s, sb.get(s, 0), se.get(s, 0)))
    p("=" * 76)


# --------------------------------------------------------------------------- #
def _synthetic(n_templates=4, n_attrs=6, n_pairs=8):
    subs = ["S%d" % i for i in range(6)]
    pairs = [(subs[i], subs[j]) for i in range(len(subs))
             for j in range(i + 1, len(subs))][:n_pairs]
    recs, k = [], 0
    for t in range(n_templates):
        for a in range(n_attrs):
            for x, y in pairs:
                k += 1
                recs.append({
                    "instance_id": "i%04d" % k,
                    "template_id": "T%02d" % t, "tid": str(t),
                    "attribute_id": "A%02d" % a, "attribute": "attr%d" % a,
                    "q0_question": "q0_%d" % a, "q1_question": "q1_%d" % a,
                    "subj_a": x, "subj_b": y, "subject_pair": "%s::%s" % (x, y),
                    "family_a": "f" + x, "family_b": "f" + y,
                    "cluster_key": "%s::%s::A%02d" % (x, y, a),
                    "usable": True,
                    "orders": [{"order": 0, "context": "T%d %s %s" % (t, x, y),
                                "subj0": x, "subj1": y},
                               {"order": 1, "context": "T%d %s %s" % (t, y, x),
                                "subj0": y, "subj1": x}],
                })
    return recs


def selftest():
    ok = True

    def check(name, cond, extra=""):
        nonlocal ok
        ok &= bool(cond)
        print("[selftest] %-56s %s%s" % (name, "PASS" if cond else "FAIL",
                                         (" " + extra) if extra else ""))

    check("axis_frac: 2 axes gives ~0.395",
          abs(axis_frac_for_ratio(2) - 0.3956) < 1e-3,
          "(%.5f)" % axis_frac_for_ratio(2))
    check("axis_frac: 3 axes gives ~0.430",
          abs(axis_frac_for_ratio(3) - 0.4299) < 1e-3,
          "(%.5f)" % axis_frac_for_ratio(3))

    recs = _synthetic()
    b, e, d, assign = make_split(recs)
    check("synthetic: split non-empty", b and e, "(build=%d eval=%d)" % (len(b), len(e)))
    check("synthetic: build+eval+dropped == n", len(b) + len(e) + len(d) == len(recs))
    leaks = verify(b, e, strict=False)
    check("synthetic: all disjointness keys clean",
          all(v == 0 for v in leaks.values()), str(leaks))
    check("synthetic: shared context strings == 0", leaks["context"] == 0)
    nb = len(assign["attribute"]["build"]) * len(assign["template"]["build"]) * 8
    check("synthetic: |BUILD| == product of build axis sizes", len(b) == nb,
          "(%d vs %d)" % (len(b), nb))
    b3, e3, _, a3 = make_split(recs, axes=("attribute", "template", "pair"))
    l3 = verify(b3, e3, axes=("attribute", "template", "pair"), strict=False)
    check("synthetic: 3-axis variant also clean", all(v == 0 for v in l3.values()), str(l3))
    check("synthetic: 3-axis retains less", len(b3) + len(e3) < len(b) + len(e),
          "(%d vs %d)" % (len(b3) + len(e3), len(b) + len(e)))

    b2, e2, _, a2 = make_split(recs)
    check("determinism: same seed -> identical split",
          [r["instance_id"] for r in b2] == [r["instance_id"] for r in b]
          and [r["instance_id"] for r in e2] == [r["instance_id"] for r in e]
          and a2 == assign)
    bx, _, _, _ = make_split(recs, seed=SEED + 1)
    check("determinism: different seed -> different split",
          [r["instance_id"] for r in bx] != [r["instance_id"] for r in b])

    planted = False
    try:
        verify(b, e + [dict(b[0])], strict=True)
    except AssertionError:
        planted = True
    check("verifier: raises on a planted instance leak", planted)
    planted2 = False
    leaky = dict(e[0], instance_id="planted", template_id=b[0]["template_id"],
                 orders=list(b[0]["orders"]))
    try:
        verify(b, e + [leaky], strict=True)
    except AssertionError:
        planted2 = True
    check("verifier: raises on a planted template/context leak", planted2)
    planted3 = False
    leaky3 = dict(e[0], instance_id="planted3", attribute_id=b[0]["attribute_id"],
                  attribute=b[0]["attribute"], q0_question=b[0]["q0_question"],
                  q1_question=b[0]["q1_question"])
    try:
        verify(b, e + [leaky3], strict=True)
    except AssertionError:
        planted3 = True
    check("verifier: raises on a planted attribute leak", planted3)

    if os.path.exists(CANONICAL_OUT):
        canon = load_jsonl(CANONICAL_OUT)
        usable = [r for r in canon if r["usable"]]
        rb, re_, rd, _ = make_split(usable)
        rl = verify(rb, re_, strict=False)
        check("REAL: split non-empty", rb and re_,
              "(build=%d eval=%d dropped=%d)" % (len(rb), len(re_), len(rd)))
        check("REAL: all disjointness keys clean", all(v == 0 for v in rl.values()), str(rl))
        fb, fe = family_counts(rb), family_counts(re_)
        check("REAL: every family present in both splits",
              all(fb[f] > 0 and fe[f] > 0 for f in fb), str({"build": fb, "eval": fe}))
        check("REAL: merged black target has >=200 instances per split",
              fb["black"] >= 200 and fe["black"] >= 200,
              "(build=%d eval=%d)" % (fb["black"], fe["black"]))
        check("REAL: no unusable instance leaked into a split",
              all(r["usable"] for r in rb + re_))
    else:
        print("[selftest] (canonical jsonl absent -- real-data checks skipped; run "
              "`python -m unqover_hf.loader` first)")

    print("[selftest] splits OVERALL: %s" % ("PASS" if ok else "FAIL"))
    return ok


def main():
    ap = argparse.ArgumentParser(description="Strictly disjoint BUILD/EVAL split.")
    ap.add_argument("--canonical", default=CANONICAL_OUT)
    ap.add_argument("--build-out", default=DEFAULT_BUILD)
    ap.add_argument("--eval-out", default=DEFAULT_EVAL)
    ap.add_argument("--manifest", default=DEFAULT_MANIFEST)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--axes", default=",".join(DEFAULT_AXES),
                    help="comma list from %s" % ",".join(sorted(AXES)))
    ap.add_argument("--axis-frac", type=float, default=None,
                    help="default: solve p^k:(1-p)^k = 30:70 for k axes")
    ap.add_argument("--relaxations", action="store_true")
    ap.add_argument("--seed-spread", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        sys.exit(0 if selftest() else 1)

    axes = tuple(a.strip() for a in args.axes.split(",") if a.strip())
    for a in axes:
        if a not in AXES:
            ap.error("unknown axis %r (choose from %s)" % (a, sorted(AXES)))
    frac = args.axis_frac if args.axis_frac is not None else axis_frac_for_ratio(len(axes))

    canon = load_jsonl(args.canonical)
    usable = [r for r in canon if r["usable"]]
    build, ev, dropped, assign = make_split(usable, seed=args.seed, frac=frac, axes=axes)
    leaks = verify(build, ev, axes=axes, strict=True)     # FAILS LOUDLY on a leak
    man = build_manifest(canon, usable, build, ev, dropped, assign,
                         args.seed, frac, axes, leaks)
    print_manifest(man)

    if args.relaxations:
        print("\n  CONSTRAINT COST (seed=%s; axis_frac solved per axis count)" % args.seed)
        print("    %-34s %6s %6s %8s %7s %11s %s"
              % ("enforced axes", "build", "eval", "dropped", "retain",
                 "black b/e", "leaks"))
        for row in relaxation_table(usable, args.seed):
            leak = row.get("leak")
            lt = ("-" if leak is None else
                  ",".join("%s=%d" % (k, v) for k, v in leak.items() if v))
            print("    %-34s %6d %6d %8d %6.1f%% %5d/%-5d %s"
                  % ("+".join(row["axes"]), row["build"], row["eval"], row["dropped"],
                     100 * row["retained_frac"], row["black_build"], row["black_eval"], lt))
    if args.seed_spread:
        print("\n  SEED SENSITIVITY (axes=%s)" % "+".join(axes))
        for row in seed_spread(usable, axes=axes):
            print("    seed=%-4d build=%-5d eval=%-5d retained=%5.1f%%  black b/e=%d/%d"
                  % (row["seed"], row["build"], row["eval"], 100 * row["retained_frac"],
                     row["black_build"], row["black_eval"]))

    if args.dry_run:
        print("\n[dry-run] nothing written.")
        return
    write_jsonl(args.build_out, build)
    write_jsonl(args.eval_out, ev)
    os.makedirs(os.path.dirname(os.path.abspath(args.manifest)), exist_ok=True)
    with open(args.manifest, "w") as fh:
        json.dump(man, fh, indent=1)
    print("\nwrote %d -> %s" % (len(build), args.build_out))
    print("wrote %d -> %s" % (len(ev), args.eval_out))
    print("wrote manifest -> %s" % args.manifest)


if __name__ == "__main__":
    main()
