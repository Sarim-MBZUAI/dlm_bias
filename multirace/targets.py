#!/usr/bin/env python
"""multirace/targets.py -- shared registry for multi-race steering targets.

Generalizes the Black-referent setup (steering/pid_steer.py) to five targets.
Tags come from BBQ answer_info group tags, matched WHOLE and lowercase
(compound tags like "f-black" are literal tags, NOT split).

Exports
    TARGET_TAGS            dict target -> frozenset of lowercase group tags
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
}
NEW_TARGETS = ("white", "asian", "latino", "arab")  # black excluded: already built


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
