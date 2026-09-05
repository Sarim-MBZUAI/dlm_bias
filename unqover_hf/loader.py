#!/usr/bin/env python
"""Canonical UNQOVER-race instances: the hirundo HF item selection, COMPLETED
into full UNQOVER quadruples from the official generation, then filtered.

--------------------------------------------------------------------------- #
WHY THE HF PARQUETS ALONE ARE NOT ENOUGH  (two audited defects)
--------------------------------------------------------------------------- #
DEFECT 1 -- the label columns are positional artefacts.
  `unqover-race-bias.answer` equals ans0, the subject mentioned FIRST in the
  context, in 10000/10000 rows; its marginal is flat over all 15 subjects.  It
  is a POSITION label, not a stereotype label.  `unqover-race-bias-free-text`
  is a refusal string drawn from 10 variants.  NEITHER may be used for scoring,
  for labels, or to build contrast pairs.  UNQOVER has no gold stereotype label
  by construction: bias is only recoverable from the 4-way contrast.  Both
  columns are carried through into `hirundo_aux` for provenance only, and the
  eval harness never reads them.

DEFECT 2 -- the 2-order structure is collapsed.
  UNQOVER needs 2 SUBJECT ORDERS x 2 QUESTION POLARITIES per instance.  The
  polarities are intact (q0/q1 both shipped), but the flipped-order twin is
  present for only 692/10000 rows: of 9654 distinct unordered instances only
  346 have both orders.  B, C, mu, eta and delta are undefined on the rest.

THE FIX -- complete the subsample instead of abandoning it.  The 10000 hirundo
rows are an exact, unmodified 6.8% subsample of the official generation

  data/unqover/generated/ethnicity.source.json   (allenai/unqover @ 3e47969)
  147000 items = 14 templates x 50 attributes x 210 ordered subject pairs

verified here row by row: all 10000 rows match a source item exactly on
(context, q0.question, q1.question, ans0, ans1), with no ambiguous match.  Each
hirundo row's missing flipped-order twin is recovered from that file, so every
canonical instance ships BOTH orderings and the real metric is computable.  The
user's item selection is preserved exactly -- nothing is added except twins.

Source key: "scluster0|scluster1|subj0|subj1|tid|act_cluster|obj0|obj1"
  tid          template index, 0..13          -> template_id
  obj0 (=obj1) the attribute phrase, 50 values -> attribute / attribute_id
  act_cluster  coarse attribute family, 5 values (crime/indecent/negative/poor/rude)
  the flipped twin is the same key with subj0/subj1 (and scluster0/1, obj0/1) swapped.

--------------------------------------------------------------------------- #
FILTERS (both recorded in the manifest; nothing is silently dropped)
--------------------------------------------------------------------------- #
ATTRIBUTE ALLOWLIST.  12 of the 50 attributes have a q1 that is NOT the logical
negation of q0, so the negation subtraction in Eq. 5 is invalid on them; they
are excluded (ATTRIBUTE_DENYLIST below).  Two are additionally disqualifying
because they put the race construct inside the question text itself
("...very dark and looked untrustworthy", "...refused to serve visitors because
of their skin color").  Identified by ATTRIBUTE, never by row text.

SUBJECT FAMILIES.  The 15 subject names are not 15 disjoint groups: several are
nested or near-synonymous.  Instances pairing two members of the SAME family are
excluded (the comparison is not a between-group comparison), and a steering
target is the MERGED family, not a literal name -- targeting the string "Black"
alone leaves far too few instances to be a benchmark.

    black     African, African-American, Black
    hispanic  Hispanic, Latino, South-American
    european  European, White, Caucasian
    arab      Arab, Middle-Eastern
    native    Alaskan, Native American
    asian     Asian                (singleton)
    jewish    Jewish               (singleton)

--------------------------------------------------------------------------- #
CLUSTERING (for anyone computing an interval on these numbers)
--------------------------------------------------------------------------- #
The rows are NOT independent: everything rests on 14 templates, 50 attributes
and 105 unordered pairs, and `context` is perfectly confounded with the subject
pair (no context ever occurs with a different pair).  Do NOT bootstrap over
rows.  Cluster at the attribute or (pair, attribute) level; every record carries
`attribute_id` and `cluster_key` = (subject_pair, attribute_id) for that, and
metric.py's bootstrap resamples clusters, not rows.

--------------------------------------------------------------------------- #
Output: one jsonl line per unordered INSTANCE (= one UNQOVER quadruple, which
the harness expands to 4 generations: 2 orders x 2 polarities).

CPU only.
  python -m unqover_hf.loader --selftest
  python -m unqover_hf.loader --report            # audit only, no write
  python -m unqover_hf.loader                     # writes canonical_race.jsonl
"""
import argparse
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)

_PARQUET_SEARCH = [
    os.path.join(_ROOT, "data", "unqover_hirundo"),
    "/shared/home/sarim.hashmi/dlm_bias/data/unqover_hirundo",
]
_SOURCE_SEARCH = [
    os.path.join(_ROOT, "data", "unqover", "generated", "ethnicity.source.json"),
    "/shared/home/sarim.hashmi/dlm_bias/data/unqover/generated/ethnicity.source.json",
]
_SHARDS = {
    "race": "unqover-race/data/train-00000-of-00001.parquet",
    "bias": "unqover-race-bias/data/train-00000-of-00001.parquet",
    "free_text": "unqover-race-bias-free-text/data/train-00000-of-00001.parquet",
}

DEFAULT_OUT = os.path.join(_ROOT, "data", "unqover_hirundo", "canonical_race.jsonl")

# --------------------------------------------------------------------------- #
# 12 attributes whose q1 is not the negation of q0 (Eq. 5 invalid on them).
# The first two additionally name the race construct inside the question.
ATTRIBUTE_DENYLIST = [
    "were very dark and looked untrustworthy",
    "were angry and refused to serve visitors because of their skin color",
    "tried to scam for money",
    "ate exotic dishes and smelled strange to the others",
    "smelled of marijuana",
    "were very scrawny, with swollen bellies",
    "drank and dosed drugs on the weekend",
    "were likely to hijack a plane",
    "drank a lot of alcohol and slept on street",
    "didn't even have cellphones",
    "were only good at doing nails and spa",
    "were in drug cartels",
]

SUBJECT_FAMILY = {
    "African": "black", "African-American": "black", "Black": "black",
    "Hispanic": "hispanic", "Latino": "hispanic", "South-American": "hispanic",
    "European": "european", "White": "european", "Caucasian": "european",
    "Arab": "arab", "Middle-Eastern": "arab",
    "Alaskan": "native", "Native American": "native",
    "Asian": "asian", "Jewish": "jewish",
}
FAMILY_MEMBERS = defaultdict(list)
for _s, _f in sorted(SUBJECT_FAMILY.items()):
    FAMILY_MEMBERS[_f].append(_s)
FAMILY_MEMBERS = dict(FAMILY_MEMBERS)


def resolve(cands, what):
    for c in cands:
        if c and os.path.exists(c):
            return c
    raise SystemExit("could not find %s in any of: %s" % (what, ", ".join(map(str, cands))))


def resolve_parquet_dir(explicit=None):
    cands = [explicit] if explicit else _PARQUET_SEARCH
    for c in cands:
        if c and all(os.path.exists(os.path.join(c, p)) for p in _SHARDS.values()):
            return c
    raise SystemExit("could not find the three UNQOVER-race parquets in any of: "
                     + ", ".join(map(str, cands)))


# --------------------------------------------------------------------------- #
# source-key helpers
# --------------------------------------------------------------------------- #
def parse_key(key):
    scl0, scl1, subj0, subj1, tid, act_cluster, obj0, obj1 = key.split("|")
    return {"s_cluster0": scl0, "s_cluster1": scl1, "subj0": subj0, "subj1": subj1,
            "tid": tid, "act_cluster": act_cluster, "obj0": obj0, "obj1": obj1}


def flip_key(key):
    """The same item with the two subjects swapped (UNQOVER's tau_{2,1})."""
    f = key.split("|")
    f[0], f[1] = f[1], f[0]
    f[2], f[3] = f[3], f[2]
    f[6], f[7] = f[7], f[6]
    return "|".join(f)


def signature(entry):
    """The tuple on which a hirundo row is matched against the official source."""
    return (entry["context"], entry["q0"]["question"], entry["q1"]["question"],
            entry["q0"]["ans0"]["text"], entry["q0"]["ans1"]["text"])


def _ids(values, prefix, width=2):
    return {v: "%s%0*d" % (prefix, width, i) for i, v in enumerate(sorted(values))}


# --------------------------------------------------------------------------- #
def build_records(parquet_dir, source_path):
    """Returns (instances, report). Anomalies are counted, never raised."""
    import pandas as pd

    with open(source_path) as fh:
        source = json.load(fh)
    race = pd.read_parquet(os.path.join(parquet_dir, _SHARDS["race"]))
    bias = pd.read_parquet(os.path.join(parquet_dir, _SHARDS["bias"]))
    free = pd.read_parquet(os.path.join(parquet_dir, _SHARDS["free_text"]))

    rep = {"parquet_dir": parquet_dir, "source_path": source_path,
           "n_source_items": len(source), "n_hirundo_rows": int(len(race)),
           "n_rows_bias": int(len(bias)), "n_rows_free_text": int(len(free))}

    # ---- 1. index the official source by signature ------------------------ #
    by_sig = defaultdict(list)
    for k, v in source.items():
        by_sig[signature(v)].append(k)
    rep["source_signature_collisions"] = sum(1 for v in by_sig.values() if len(v) > 1)

    # ---- 2. match every hirundo row -------------------------------------- #
    bias_map = dict(zip(bias["question"].astype(str), bias["answer"].astype(str)))
    free_map = dict(zip(free["question"].astype(str), free["answer"].astype(str)))
    rep["bias_key_dups"] = int(len(bias) - len(bias_map))
    rep["free_text_key_dups"] = int(len(free) - len(free_map))

    matched, unmatched, ambiguous = {}, [], 0
    ster_is_s0 = ster_is_s1 = ster_bad = 0
    for ctx, q0, q1 in zip(race["context"], race["q0"], race["q1"]):
        entry = {"context": str(ctx),
                 "q0": {"question": str(q0["question"]),
                        "ans0": {"text": str(q0["ans0"]["text"])},
                        "ans1": {"text": str(q0["ans1"]["text"])}},
                 "q1": {"question": str(q1["question"]),
                        "ans0": {"text": str(q1["ans0"]["text"])},
                        "ans1": {"text": str(q1["ans1"]["text"])}}}
        ks = by_sig.get(signature(entry))
        if not ks:
            unmatched.append(signature(entry))
            continue
        if len(ks) > 1:
            ambiguous += 1
        key = sorted(ks)[0]
        jk = entry["context"] + " " + entry["q0"]["question"]
        ster = bias_map.get(jk)
        if ster == entry["q0"]["ans0"]["text"]:
            ster_is_s0 += 1
        elif ster == entry["q0"]["ans1"]["text"]:
            ster_is_s1 += 1
        else:
            ster_bad += 1
        matched[key] = {"bias_answer": ster, "free_text_answer": free_map.get(jk),
                        "join_key": jk}
    rep["hirundo_matched"] = len(matched) + ambiguous * 0
    rep["hirundo_unmatched"] = len(unmatched)
    rep["hirundo_ambiguous_matches"] = ambiguous
    rep["hirundo_match_rate"] = round(len(matched) / max(1, len(race)), 6)
    rep["stereotyped_is_subj0"] = ster_is_s0
    rep["stereotyped_is_subj1"] = ster_is_s1
    rep["stereotyped_not_in_pair"] = ster_bad
    rep["refusal_variants"] = dict(Counter(
        v["free_text_answer"] for v in matched.values()).most_common())

    # ---- 3. group into unordered instances, recover the twins ------------- #
    groups = defaultdict(set)          # instance key -> {ordered source keys}
    for key in matched:
        f = parse_key(key)
        groups[(f["tid"], f["act_cluster"], f["obj0"],
                tuple(sorted((f["subj0"], f["subj1"]))))].add(key)

    rep["n_instances_raw"] = len(groups)
    rep["instances_with_both_orders_in_hirundo"] = sum(1 for v in groups.values() if len(v) == 2)
    rep["twins_needed"] = sum(1 for v in groups.values() if len(v) == 1)

    recovered = twin_missing = qtext_mismatch = 0
    tid_ids = _ids({k[0] for k in groups}, "T")
    attr_ids = _ids({k[2] for k in groups}, "A")

    instances = []
    for gkey in sorted(groups):
        tid, act_cluster, attribute, pair = gkey
        keys = set(groups[gkey])
        for k in list(keys):
            tw = flip_key(k)
            if tw in keys:
                continue
            if tw not in source:
                twin_missing += 1
                continue
            keys.add(tw)
            recovered += 1
        if len(keys) != 2:
            continue                                   # cannot be completed
        # order 0 = the context naming the alphabetically FIRST subject first
        subj_a, subj_b = pair
        by_first = {parse_key(k)["subj0"]: k for k in keys}
        if set(by_first) != set(pair):
            continue
        k0, k1 = by_first[subj_a], by_first[subj_b]
        e0, e1 = source[k0], source[k1]
        if (e0["q0"]["question"] != e1["q0"]["question"]
                or e0["q1"]["question"] != e1["q1"]["question"]):
            qtext_mismatch += 1
            continue

        fam_a, fam_b = SUBJECT_FAMILY[subj_a], SUBJECT_FAMILY[subj_b]
        within = fam_a == fam_b
        allowed = attribute not in ATTRIBUTE_DENYLIST
        aux = matched.get(k0) or matched.get(k1) or {}
        iid = "U" + hashlib.sha1(
            ("|".join((tid, attribute, subj_a, subj_b))).encode("utf-8")).hexdigest()[:14]
        instances.append({
            "instance_id": iid,
            "tid": tid, "template_id": tid_ids[tid],
            "attribute": attribute, "attribute_id": attr_ids[attribute],
            "act_cluster": act_cluster,
            "q0_question": e0["q0"]["question"], "q1_question": e0["q1"]["question"],
            "subj_a": subj_a, "subj_b": subj_b,
            "subject_pair": "::".join((subj_a, subj_b)),
            "family_a": fam_a, "family_b": fam_b,
            "family_pair": "::".join(sorted((fam_a, fam_b))),
            "within_family": within,
            "attribute_allowed": allowed,
            "usable": allowed and not within,
            "cluster_key": "%s::%s" % ("::".join((subj_a, subj_b)), attr_ids[attribute]),
            "orders": [
                {"order": 0, "context": e0["context"],
                 "subj0": subj_a, "subj1": subj_b,
                 "source_key": k0, "from_hirundo": k0 in matched},
                {"order": 1, "context": e1["context"],
                 "subj0": subj_b, "subj1": subj_a,
                 "source_key": k1, "from_hirundo": k1 in matched},
            ],
            # provenance ONLY -- see DEFECT 1. Never used for scoring.
            "hirundo_aux": {"bias_answer": aux.get("bias_answer"),
                            "free_text_answer": aux.get("free_text_answer"),
                            "note": "bias_answer == the first-named subject in "
                                    "10000/10000 rows: a POSITION artefact, not a "
                                    "stereotype label. Do not score against it."},
        })

    rep["twins_recovered"] = recovered
    rep["twins_missing_from_source"] = twin_missing
    rep["question_text_mismatch_between_orders"] = qtext_mismatch
    rep["n_complete_quadruples"] = len(instances)
    rep["n_records_after_completion"] = 2 * len(instances)
    rep["n_generations_full_set"] = 4 * len(instances)

    # ---- 4. filter accounting -------------------------------------------- #
    usable = [r for r in instances if r["usable"]]
    rep["n_denied_attribute"] = sum(1 for r in instances if not r["attribute_allowed"])
    rep["n_within_family"] = sum(1 for r in instances if r["within_family"])
    rep["n_usable"] = len(usable)
    rep["attribute_denylist"] = list(ATTRIBUTE_DENYLIST)
    rep["n_attributes_total"] = len(attr_ids)
    rep["n_attributes_allowed"] = len({r["attribute"] for r in usable})
    rep["n_templates"] = len(tid_ids)
    rep["attribute_ids"] = {v: k for k, v in attr_ids.items()}
    rep["hirundo_rows_denied_attribute"] = sum(
        1 for k in matched if parse_key(k)["obj0"] in ATTRIBUTE_DENYLIST)
    rep["hirundo_rows_within_family"] = sum(
        1 for k in matched
        if SUBJECT_FAMILY[parse_key(k)["subj0"]] == SUBJECT_FAMILY[parse_key(k)["subj1"]])

    # ---- 5. cluster + target accounting ---------------------------------- #
    rep["clusters_usable"] = {
        "attribute": len({r["attribute_id"] for r in usable}),
        "template": len({r["template_id"] for r in usable}),
        "subject_pair": len({r["subject_pair"] for r in usable}),
        "pair_x_attribute": len({r["cluster_key"] for r in usable}),
    }
    rep["usable_by_family"] = {}
    for fam in sorted(FAMILY_MEMBERS):
        rep["usable_by_family"][fam] = sum(
            1 for r in usable if fam in (r["family_a"], r["family_b"]))
    rep["usable_by_literal_subject"] = dict(Counter(
        s for r in usable for s in (r["subj_a"], r["subj_b"])).most_common())
    rep["usable_by_attribute"] = dict(Counter(r["attribute_id"] for r in usable).most_common())
    rep["usable_by_template"] = dict(Counter(r["template_id"] for r in usable).most_common())
    rep["family_members"] = FAMILY_MEMBERS
    # position balance of the FIRST-named slot, over the completed set (must be
    # exactly 0.5 by construction now that both orders exist)
    first = Counter()
    for r in instances:
        for o in r["orders"]:
            first[o["subj0"]] += 1
    tot = Counter()
    for r in instances:
        tot[r["subj_a"]] += 2
        tot[r["subj_b"]] += 2
    rep["first_slot_frac"] = {s: round(first[s] / tot[s], 4) for s in sorted(tot)}
    return instances, rep


# --------------------------------------------------------------------------- #
def print_report(rep):
    p = print
    p("=" * 76)
    p("UNQOVER-race -- hirundo selection COMPLETED from the official generation")
    p("=" * 76)
    p("  hirundo parquets      : %s" % rep["parquet_dir"])
    p("  official source       : %s  (%d items)"
      % (rep["source_path"], rep["n_source_items"]))
    p("")
    p("  SOURCE MATCH (context, q0, q1, ans0, ans1)")
    p("    hirundo rows                     : %d" % rep["n_hirundo_rows"])
    p("    matched exactly                  : %d  (rate=%.4f)"
      % (rep["hirundo_matched"], rep["hirundo_match_rate"]))
    p("    unmatched / ambiguous            : %d / %d"
      % (rep["hirundo_unmatched"], rep["hirundo_ambiguous_matches"]))
    p("    signature collisions in source   : %d" % rep["source_signature_collisions"])
    p("")
    p("  QUADRUPLE COMPLETION (DEFECT 2)")
    p("    distinct unordered instances     : %d" % rep["n_instances_raw"])
    p("    ... already 2-order in hirundo   : %d"
      % rep["instances_with_both_orders_in_hirundo"])
    p("    twins needed / RECOVERED / lost  : %d / %d / %d"
      % (rep["twins_needed"], rep["twins_recovered"], rep["twins_missing_from_source"]))
    p("    q-text mismatch between orders   : %d" % rep["question_text_mismatch_between_orders"])
    p("    COMPLETE QUADRUPLES              : %d   (= %d records, %d generations)"
      % (rep["n_complete_quadruples"], rep["n_records_after_completion"],
         rep["n_generations_full_set"]))
    p("")
    p("  LABEL COLUMNS (DEFECT 1)")
    p("    bias_answer == subj0 (FIRST-named): %d / %d"
      % (rep["stereotyped_is_subj0"], rep["n_hirundo_rows"]))
    p("    bias_answer == subj1 (SECOND)     : %d / %d"
      % (rep["stereotyped_is_subj1"], rep["n_hirundo_rows"]))
    p("    bias_answer outside the pair      : %d" % rep["stereotyped_not_in_pair"])
    p("    free-text answer variants         : %d (all refusal strings)"
      % len(rep["refusal_variants"]))
    if rep["stereotyped_is_subj1"] == 0:
        p("    *** unqover-race-bias.answer is a POSITION label, not a stereotype")
        p("        label. Not used for scoring, labels or contrast pairs. ***")
    p("")
    p("  FILTERS")
    p("    attributes total / allowed        : %d / %d  (%d denied)"
      % (rep["n_attributes_total"], rep["n_attributes_allowed"],
         len(rep["attribute_denylist"])))
    p("    instances denied by attribute     : %d" % rep["n_denied_attribute"])
    p("    instances with a within-family pair: %d" % rep["n_within_family"])
    p("    hirundo ROWS hit by those filters : %d attribute / %d within-family"
      % (rep["hirundo_rows_denied_attribute"], rep["hirundo_rows_within_family"]))
    p("    USABLE instances                  : %d  (%d generations)"
      % (rep["n_usable"], 4 * rep["n_usable"]))
    p("")
    p("  CLUSTERS IN THE USABLE SET (bootstrap over these, never over rows)")
    for k, v in rep["clusters_usable"].items():
        p("    %-18s %d" % (k, v))
    p("")
    p("  USABLE INSTANCES CONTAINING EACH SUBJECT FAMILY (merged targets)")
    for fam, n in sorted(rep["usable_by_family"].items(), key=lambda kv: -kv[1]):
        p("    %-10s %5d   members: %s" % (fam, n, ", ".join(rep["family_members"][fam])))
    p("  ... by LITERAL subject name (why a merged target is necessary)")
    for s, n in rep["usable_by_literal_subject"].items():
        p("    %-18s %5d" % (s, n))
    p("")
    p("  FIRST-NAMED SLOT SHARE PER SUBJECT (0.5 exactly once quadruples are complete)")
    bad = {s: f for s, f in rep["first_slot_frac"].items() if abs(f - 0.5) > 1e-9}
    p("    all 0.5000: %s%s" % (not bad, ("  offenders=%s" % bad) if bad else ""))
    p("=" * 76)


# --------------------------------------------------------------------------- #
def load_jsonl(path):
    with open(path) as fh:
        return [json.loads(l) for l in fh if l.strip()]


def write_jsonl(path, recs):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as fh:
        for r in recs:
            fh.write(json.dumps(r) + "\n")


# --------------------------------------------------------------------------- #
def selftest():
    ok = True

    def check(name, got, want):
        nonlocal ok
        good = got == want
        ok &= good
        print("[selftest] %-54s %s" % (name, "PASS" if good else
                                       "FAIL got=%r want=%r" % (got, want)))

    k = "None|None|African|Jewish|0|rude|had a reputation of being barbarians|" \
        "had a reputation of being barbarians"
    f = parse_key(k)
    check("parse_key: subjects", (f["subj0"], f["subj1"]), ("African", "Jewish"))
    check("parse_key: tid/act/obj", (f["tid"], f["act_cluster"], f["obj0"]),
          ("0", "rude", "had a reputation of being barbarians"))
    check("flip_key: swaps subjects", parse_key(flip_key(k))["subj0"], "Jewish")
    check("flip_key: is an involution", flip_key(flip_key(k)), k)
    k2 = "c0|c1|Black|White|7|crime|objA|objB"
    check("flip_key: swaps clusters and objs too", flip_key(k2),
          "c1|c0|White|Black|7|crime|objB|objA")
    check("flip_key: keeps tid and act_cluster",
          (parse_key(flip_key(k2))["tid"], parse_key(flip_key(k2))["act_cluster"]),
          ("7", "crime"))

    check("signature: field order", signature(
        {"context": "C", "q0": {"question": "Q0", "ans0": {"text": "X"},
                                "ans1": {"text": "Y"}},
         "q1": {"question": "Q1", "ans0": {"text": "X"}, "ans1": {"text": "Y"}}}),
        ("C", "Q0", "Q1", "X", "Y"))

    check("denylist: 12 attributes", len(ATTRIBUTE_DENYLIST), 12)
    check("denylist: no duplicates", len(set(ATTRIBUTE_DENYLIST)), 12)
    check("families: 15 subjects", len(SUBJECT_FAMILY), 15)
    check("families: 7 families", len(FAMILY_MEMBERS), 7)
    check("families: black family members", FAMILY_MEMBERS["black"],
          ["African", "African-American", "Black"])
    check("families: nested names share a family",
          SUBJECT_FAMILY["African"] == SUBJECT_FAMILY["African-American"], True)
    check("families: unrelated names do not",
          SUBJECT_FAMILY["Asian"] == SUBJECT_FAMILY["Jewish"], False)
    check("ids: deterministic + sorted", _ids({"b", "a", "c"}, "A"),
          {"a": "A00", "b": "A01", "c": "A02"})

    # canonical file, if built: structural invariants that must hold on real data
    if os.path.exists(DEFAULT_OUT):
        recs = load_jsonl(DEFAULT_OUT)
        check("REAL: every instance has exactly 2 orders",
              all(len(r["orders"]) == 2 for r in recs), True)
        check("REAL: order 0 names subj_a first",
              all(r["orders"][0]["subj0"] == r["subj_a"] for r in recs), True)
        check("REAL: order 1 names subj_b first",
              all(r["orders"][1]["subj0"] == r["subj_b"] for r in recs), True)
        check("REAL: the two orders use different contexts",
              all(r["orders"][0]["context"] != r["orders"][1]["context"] for r in recs), True)
        check("REAL: both subject names appear in both contexts",
              all(all(s in o["context"] for s in (r["subj_a"], r["subj_b"]))
                  for r in recs for o in r["orders"]), True)
        check("REAL: instance_id unique", len({r["instance_id"] for r in recs}), len(recs))
        check("REAL: usable == allowed and not within-family",
              all(r["usable"] == (r["attribute_allowed"] and not r["within_family"])
                  for r in recs), True)
        check("REAL: no usable instance carries a denied attribute",
              any(r["usable"] and r["attribute"] in ATTRIBUTE_DENYLIST for r in recs), False)
        check("REAL: no usable instance pairs one family with itself",
              any(r["usable"] and r["family_a"] == r["family_b"] for r in recs), False)
        check("REAL: at least one order of each instance came from hirundo",
              all(any(o["from_hirundo"] for o in r["orders"]) for r in recs), True)
    else:
        print("[selftest] (canonical jsonl absent -- real-data checks skipped)")

    print("[selftest] loader OVERALL: %s" % ("PASS" if ok else "FAIL"))
    return ok


def main():
    ap = argparse.ArgumentParser(
        description="hirundo UNQOVER-race -> completed, filtered canonical quadruples.")
    ap.add_argument("--parquet-dir", default=None)
    ap.add_argument("--source", default=None, help="ethnicity.source.json (official)")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--report-json", default=None)
    ap.add_argument("--report", action="store_true", help="audit only, do not write")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        sys.exit(0 if selftest() else 1)

    recs, rep = build_records(
        resolve_parquet_dir(args.parquet_dir),
        resolve([args.source] if args.source else _SOURCE_SEARCH, "ethnicity.source.json"))
    print_report(rep)
    if args.report_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.report_json)), exist_ok=True)
        with open(args.report_json, "w") as fh:
            json.dump(rep, fh, indent=1)
        print("wrote report -> %s" % args.report_json)
    if not args.report:
        write_jsonl(args.out, recs)
        print("wrote %d canonical quadruples (%d usable) -> %s"
              % (len(recs), rep["n_usable"], args.out))


if __name__ == "__main__":
    main()
