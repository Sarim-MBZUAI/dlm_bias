#!/usr/bin/env python
"""Reconstruct data/bbq_items/black_referent_ambig_eval.jsonl (the 1600-item
superset that make_seed_rotations.py samples from) deterministically.

Selection (reproduces the seed1/2/3 rot0 sets exactly):
  BBQ Race_ethnicity cache rows, IN CACHE FILE ORDER, with
    context_condition == "ambig"
    exactly ONE option carrying a Black group tag (pid_steer.BLACK_TAGS)
    an "unknown" option present
  MINUS the direction-build keys (steering/direction_examples.jsonl).
2000 rows match the predicate; removing the 400 direction-build keys leaves
exactly 1600. NOTE: the superset deliberately does NOT exclude _sweep400.jsonl
keys -- the seed draws are independent replications, not disjoint from the
primary eval set (they ARE disjoint from the direction-build set).

Usage: python eval/balanced/make_superset.py
"""
import json
import os
import sys

ROOT = os.environ.get("DLM_BIAS_ROOT") or os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "eval"))
import bbq_eval  # noqa: E402

CACHE = os.path.join(ROOT, "data", "bbq_cache", "Race_ethnicity.jsonl")
BUILD = os.path.join(ROOT, "steering", "direction_examples.jsonl")
OUT = os.path.join(ROOT, "data", "bbq_items", "black_referent_ambig_eval.jsonl")
BLACK_TAGS = {"black", "african american", "f-black", "m-black", "african"}  # == pid_steer.BLACK_TAGS


def key(r):
    return (r["example_id"], str(r.get("question_index", "")))


def is_black_referent_ambig(r):
    if r.get("context_condition") != "ambig":
        return False
    hits = [k for k in range(3)
            if (p := bbq_eval.get_answer_info(r, k))
            and str(p[-1]).strip().lower() in BLACK_TAGS]
    return len(hits) == 1 and bbq_eval.unknown_index(r) is not None


def main():
    rows = [json.loads(l) for l in open(CACHE) if l.strip()]
    build_keys = {key(json.loads(l)) for l in open(BUILD)}
    big = [r for r in rows if is_black_referent_ambig(r) and key(r) not in build_keys]
    assert len(big) == 1600, f"expected 1600 superset rows, got {len(big)}"
    with open(OUT, "w") as f:
        for r in big:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {OUT}  ({len(big)} rows; build-key overlap = "
          f"{len({key(r) for r in big} & build_keys)}, must be 0)")


if __name__ == "__main__":
    main()
