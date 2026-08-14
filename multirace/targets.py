#!/usr/bin/env python
"""multirace/targets.py -- shared registry for multi-target steering targets.

Generalizes the Black-referent setup (steering/pid_steer.py) to five race
targets plus two gender targets (E3). Tags come from BBQ answer_info group
tags, matched WHOLE and lowercase (compound tags like "f-black" or "trans_f"
are literal tags, NOT split).

GENDER (BBQ Gender_identity, E3): STRICT tag sets --
    woman = {f, woman, girl}    man = {m, man, boy}
The trans/nontrans compound tags (trans_f, trans_m, nontrans_f, nontrans_m)
and the bare trans/nontrans tags are DELIBERATELY EXCLUDED from both sets:
steering toward/away from trans-identified options is a dual-use hazard we do
not touch, so those rows simply never match a gender target.

SES (BBQ SES, E9): lowses = {lowses}, highses = {highses}. The raw
answer_info tags are 'lowSES'/'highSES'; matching is lowercase whole-tag, so
the canonical registry forms are lowercase. Every ambiguous SES row carries
BOTH poles plus unknown (same-rows situation as gender), so eval/heldout of
the two poles must come from ONE shuffle of the shared pool, cut into four
mutually disjoint sets (see make_items.py).

AGE (BBQ Age, E9): old = {old}, young = {nonold}. The BBQ tag for the
younger person is literally 'nonOld' -- the target is NAMED 'young' for
readability but TAG-MATCHES 'nonold' (lowercase whole-tag; note whole-tag
matching is what keeps 'nonold' from ever matching the 'old' target even
though 'old' is a substring). Same both-poles-per-row situation as SES.

INTERSECTIONAL TARGET (BBQ Race_ethnicity, E6): fblack = {f-black} -- Black
WOMEN specifically, via the single compound tag, matched whole. Note fblack
is a semantic SUBSET of black (BLACK_TAGS contains "f-black"), so an f-black
row matches BOTH the fblack and the black target -- expected and relied upon
for the fblack-vs-coarse-black comparison. CAVEAT: the target GROUP is
intersectional, but the built steering DIRECTION is not a full
intersectional contrast -- BBQ Race_ethnicity pairs same-gender people, so
all realized heldout negatives are other-race women (0 m-black): a
GENDER-CONDITIONED RACE direction, f-black vs other-race women (see
build_arrows.py). The mirror target m-black is NOT built: only 378 usable
rows survive the Black-experiment exclusions, too few for the 400-eval
protocol.

Exports
    TARGET_TAGS            dict target -> frozenset of lowercase group tags
    TARGET_CATEGORY        dict target -> BBQ category (cache file stem)
    NEW_TARGETS            race targets built in round 2 (black excluded)
    GENDER_TARGETS         ("woman", "man")
    INTERSECTIONAL_TARGETS ("fblack",)  E6
    SES_TARGETS            ("lowses", "highses")  E9
    AGE_TARGETS            ("old", "young")       E9 (young tag-matches 'nonold')
    target_idx_of(row, t)  index of the FIRST option whose tag is in
                           TARGET_TAGS[t], else None (mirrors
                           pid_steer.black_idx_of, via bbq_eval.get_answer_info)
    unk_idx_of(row)        re-exported from steering/pid_steer.py

Offline check:  python multirace/targets.py --selftest
"""
import os
import sys

ROOT = os.environ.get("DLM_BIAS_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "eval"))
sys.path.insert(0, os.path.join(ROOT, "steering"))
import bbq_eval  # noqa: E402
from pid_steer import BLACK_TAGS, unk_idx_of  # noqa: E402,F401

TARGET_TAGS = {
    "black":  frozenset(BLACK_TAGS),  # existing reference set, NOT rebuilt here
    "white":  frozenset({"white", "caucasian", "f-white", "m-white", "european"}),
    "asian":  frozenset({"asian", "f-asian", "m-asian"}),
    "latino": frozenset({"latino", "hispanic", "f-latino", "m-latino"}),
    "arab":   frozenset({"arab", "middle eastern", "f-arab", "m-arab"}),
    # E3 gender targets (STRICT: no trans_/nontrans_ compounds -- see module doc).
    "woman":  frozenset({"f", "woman", "girl"}),
    "man":    frozenset({"m", "man", "boy"}),
    # E6 intersectional target: Black women only (subset of black -- see doc).
    "fblack": frozenset({"f-black"}),
    # E9 SES poles (BBQ SES; raw tags lowSES/highSES, matched lowercase whole).
    "lowses":  frozenset({"lowses"}),
    "highses": frozenset({"highses"}),
    # E9 Age poles (BBQ Age). 'young' is a READABILITY NAME: the BBQ tag for
    # the younger person is 'nonOld', so the tag set is {nonold} (lowercase
    # whole-tag -- 'nonold' never matches the 'old' target despite the
    # substring, because tags are compared whole, never split).
    "old":   frozenset({"old"}),
    "young": frozenset({"nonold"}),
}
NEW_TARGETS = ("white", "asian", "latino", "arab")  # black excluded: already built
GENDER_TARGETS = ("woman", "man")                   # E3; NOT in NEW_TARGETS (race-only loops)
INTERSECTIONAL_TARGETS = ("fblack",)                # E6; NOT in NEW_TARGETS
SES_TARGETS = ("lowses", "highses")                 # E9; NOT in NEW_TARGETS
AGE_TARGETS = ("old", "young")                      # E9; NOT in NEW_TARGETS

# BBQ category (= data/bbq_cache/<category>.jsonl stem) each target lives in.
TARGET_CATEGORY = {
    **{t: "Race_ethnicity" for t in ("black", "white", "asian", "latino", "arab")},
    **{t: "Gender_identity" for t in GENDER_TARGETS},
    **{t: "Race_ethnicity" for t in INTERSECTIONAL_TARGETS},
    **{t: "SES" for t in SES_TARGETS},
    **{t: "Age" for t in AGE_TARGETS},
}


def target_idx_of(row, target):
    """First option index whose answer_info tag is in TARGET_TAGS[target], else
    None. Same logic as pid_steer.black_idx_of, tag set parameterized."""
    tags = TARGET_TAGS[target]
    for k in range(3):
        p = bbq_eval.get_answer_info(row, k)
        if p and str(p[-1]).strip().lower() in tags:
            return k
    return None


def target_option_count(row, target):
    """How many of the 3 options carry a TARGET_TAGS[target] tag."""
    tags = TARGET_TAGS[target]
    n = 0
    for k in range(3):
        p = bbq_eval.get_answer_info(row, k)
        if p and str(p[-1]).strip().lower() in tags:
            n += 1
    return n


# --------------------------------------------------------------------------- #
# Offline self-test (no GPU, no data files).
# --------------------------------------------------------------------------- #
def _mk_row(tags):
    return {"answer_info": {f"ans{k}": [f"w{k}", tags[k]] for k in range(3)}}


def _selftest():
    ok = True

    def check(name, cond):
        nonlocal ok
        ok &= bool(cond)
        print(f"[selftest-targets] {name:56s} : {'PASS' if cond else 'FAIL'}")

    check("black tags == pid_steer.BLACK_TAGS", TARGET_TAGS["black"] == frozenset(BLACK_TAGS))
    check("all tag sets are lowercase frozensets",
          all(isinstance(s, frozenset) and all(t == t.lower() for t in s)
              for s in TARGET_TAGS.values()))

    # Compound tags match WHOLE: "F-Asian" is an asian tag, "f-black" is NOT white.
    r = _mk_row(["F-Asian", "white", "unknown"])
    check("compound tag F-Asian -> asian idx 0", target_idx_of(r, "asian") == 0)
    check("compound tag not split (f-black not white)",
          target_idx_of(_mk_row(["f-black", "unknown", "european"]), "white") == 2)

    # Two-target row (Black-vs-White): right index per target.
    r2 = _mk_row(["Black", "Caucasian", "unknown"])
    check("two-target row: black -> 0", target_idx_of(r2, "black") == 0)
    check("two-target row: white -> 1", target_idx_of(r2, "white") == 1)
    check("two-target row: asian -> None", target_idx_of(r2, "asian") is None)

    # Casing/whitespace tolerated; unknown detection re-export works.
    r3 = _mk_row([" Middle eastern ", "Hispanic", "Unknown"])
    check("arab matches ' Middle eastern '", target_idx_of(r3, "arab") == 0)
    check("latino matches 'Hispanic'", target_idx_of(r3, "latino") == 1)
    check("unk_idx_of re-export -> 2", unk_idx_of(r3) == 2)

    # No match / counts.
    check("no target tag -> None", target_idx_of(_mk_row(["a", "b", "unknown"]), "arab") is None)
    check("count: two white options -> 2",
          target_option_count(_mk_row(["white", "european", "unknown"]), "white") == 2)

    # --- E3 gender targets (STRICT sets) --------------------------------- #
    check("registry has woman/man + categories",
          all(t in TARGET_TAGS and TARGET_CATEGORY[t] == "Gender_identity"
              for t in GENDER_TARGETS)
          and TARGET_CATEGORY["black"] == "Race_ethnicity")
    # bare f/m tags (dominant vocabulary in Gender_identity).
    g1 = _mk_row(["F", "M", "unknown"])
    check("bare tag F -> woman idx 0", target_idx_of(g1, "woman") == 0)
    check("bare tag M -> man idx 1", target_idx_of(g1, "man") == 1)
    # literal woman/man + girl/boy.
    check("literal 'woman' matches woman",
          target_idx_of(_mk_row(["man", "Woman", "unknown"]), "woman") == 1)
    check("literal 'boy' matches man",
          target_idx_of(_mk_row(["girl", "boy", "unknown"]), "man") == 1)
    # STRICT: trans/nontrans compounds match NEITHER gender target.
    gt = _mk_row(["trans_f", "nontrans_f", "unknown"])
    check("trans_f/nontrans_f row -> woman None (strict)",
          target_idx_of(gt, "woman") is None)
    check("trans_m/nontrans_m row -> man None (strict)",
          target_idx_of(_mk_row(["trans_m", "nontrans_m", "unknown"]), "man") is None)
    # woman-vs-man row: each target matches exactly once, at opposite options.
    g2 = _mk_row(["f", "m", "unknown"])
    check("woman-vs-man row: each target exactly one option",
          target_option_count(g2, "woman") == 1 and target_option_count(g2, "man") == 1
          and target_idx_of(g2, "woman") != target_idx_of(g2, "man"))
    # gender tags never leak into race targets and vice versa.
    check("gender tags don't match race targets",
          all(target_idx_of(g2, t) is None for t in NEW_TARGETS + ("black",)))

    # --- E6 intersectional target fblack ---------------------------------- #
    check("registry has fblack, category Race_ethnicity",
          "fblack" in TARGET_TAGS and TARGET_CATEGORY["fblack"] == "Race_ethnicity"
          and INTERSECTIONAL_TARGETS == ("fblack",))
    fb = _mk_row(["F-Black", "M-White", "unknown"])
    check("f-black row matches fblack (idx 0)", target_idx_of(fb, "fblack") == 0)
    # plain 'black' and m-black rows do NOT match fblack (compound tag, whole).
    check("plain 'black' tag does NOT match fblack",
          target_idx_of(_mk_row(["Black", "white", "unknown"]), "fblack") is None)
    check("m-black tag does NOT match fblack",
          target_idx_of(_mk_row(["M-Black", "F-White", "unknown"]), "fblack") is None)
    # fblack SUBSET of black: an f-black row ALSO matches the black target
    # (expected -- BLACK_TAGS contains "f-black"; the E6 comparison relies on it).
    check("fblack row ALSO matches black (fblack SUBSET of black)",
          target_idx_of(fb, "black") == 0)
    # f-black vs m-black row: fblack picks the woman, black picks the FIRST
    # Black option -- the intersectional target disambiguates within race.
    fb2 = _mk_row(["m-black", "f-black", "unknown"])
    check("f-black vs m-black row: fblack -> 1, black -> 0 (first)",
          target_idx_of(fb2, "fblack") == 1 and target_idx_of(fb2, "black") == 0
          and target_option_count(fb2, "fblack") == 1)

    # --- E9 SES targets ---------------------------------------------------- #
    check("registry has lowses/highses, category SES",
          all(t in TARGET_TAGS and TARGET_CATEGORY[t] == "SES" for t in SES_TARGETS)
          and SES_TARGETS == ("lowses", "highses"))
    # raw cache casing (lowSES/highSES) is tolerated via .lower().
    s1 = _mk_row(["lowSES", "highSES", "unknown"])
    check("raw tag lowSES -> lowses idx 0", target_idx_of(s1, "lowses") == 0)
    check("raw tag highSES -> highses idx 1", target_idx_of(s1, "highses") == 1)
    # both-pole row: each SES target matches exactly once, at opposite options.
    check("lowses-vs-highses row: each target exactly one option",
          target_option_count(s1, "lowses") == 1
          and target_option_count(s1, "highses") == 1
          and target_idx_of(s1, "lowses") != target_idx_of(s1, "highses"))
    # each pole matches ONLY its own tag (whole-tag, no cross-pole leakage).
    check("lowses does not match a highses-only row",
          target_idx_of(_mk_row(["highSES", "black", "unknown"]), "lowses") is None)
    check("highses does not match a lowses-only row",
          target_idx_of(_mk_row(["lowSES", "black", "unknown"]), "highses") is None)
    # SES tags never leak into race/gender targets and vice versa.
    check("SES tags don't match race/gender targets",
          all(target_idx_of(s1, t) is None
              for t in NEW_TARGETS + ("black",) + GENDER_TARGETS))

    # --- E9 Age targets (young tag-matches 'nonold') ------------------------ #
    check("registry has old/young, category Age",
          all(t in TARGET_TAGS and TARGET_CATEGORY[t] == "Age" for t in AGE_TARGETS)
          and AGE_TARGETS == ("old", "young"))
    check("young tag set is {nonold} (readability name, BBQ tag nonOld)",
          TARGET_TAGS["young"] == frozenset({"nonold"}))
    a1 = _mk_row(["nonOld", "old", "unknown"])
    check("raw tag nonOld -> young idx 0", target_idx_of(a1, "young") == 0)
    check("raw tag old -> old idx 1", target_idx_of(a1, "old") == 1)
    # WHOLE-TAG matching: 'nonold' must NOT match the old target ('old' is a
    # substring of 'nonold' -- whole-tag comparison is what protects this).
    check("whole-tag: 'nonold' never matches the old target",
          target_idx_of(_mk_row(["nonOld", "black", "unknown"]), "old") is None)
    check("whole-tag: 'old' never matches the young target",
          target_idx_of(_mk_row(["old", "black", "unknown"]), "young") is None)
    # both-pole row: each Age target matches exactly once, at opposite options.
    check("old-vs-young row: each target exactly one option",
          target_option_count(a1, "old") == 1 and target_option_count(a1, "young") == 1
          and target_idx_of(a1, "old") != target_idx_of(a1, "young"))
    # Age tags never leak into other targets.
    check("Age tags don't match race/gender/SES targets",
          all(target_idx_of(a1, t) is None
              for t in NEW_TARGETS + ("black",) + GENDER_TARGETS + SES_TARGETS))

    print(f"[selftest-targets] OVERALL: {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        sys.exit(0 if _selftest() else 1)
    ap.print_help()
