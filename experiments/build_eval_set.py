#!/usr/bin/env python
"""E1 evaluation set: the FULL Black-referent AMBIGUOUS Race_ethnicity items,
disjoint from the 400 anchored-build items.

Selection logic is IDENTICAL to directional_steering/build_anchored.py
(BLACK_TAGS, exactly-one-Black-option, context_condition == "ambig"). We EXCLUDE
the 400 anchored-build items (by (example_id, question_index)) so the eval set is
uncontaminated by the items the answer-text direction was built from.

Writes experiments/data/black_referent_ambig_eval.jsonl (full BBQ rows, ready to
feed to `bbq_eval.py --items`).

Asserts (fail-loud):
  * eval set is DISJOINT from the 400 build keys;
  * the seed-42 random-1000 n=37 Black-referent ambiguous items are a SUBSET of
    this eval set (so E1's large-n number is a strict superset of the pilot n=37).

CPU only, no model.  Run:
  /home/lukas/miniconda3/envs/sarim_awm/bin/python experiments/build_eval_set.py
"""
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_ROOT, "eval"))
import bbq_eval  # load_bbq / get_answer_info / unknown_index  (same loader as the eval)

# Same Black tag set as build_anchored.py / black_analysis.py.
BLACK_TAGS = {"black", "african american", "f-black", "m-black", "african"}
# Use the eval's canonical cache dir (absolute) so this works from any checkout.
RACE_CACHE = os.path.join(bbq_eval.BBQ_CACHE_DIR, "Race_ethnicity.jsonl")
BUILD_ITEMS = os.path.join(_ROOT, "directional_steering", "data", "anchored_items.jsonl")
OUT = os.path.join(_HERE, "data", "black_referent_ambig_eval.jsonl")
DATASET = "nyu-mll/BBQ (jsonl)"


def key_of(row):
    return (int(row.get("example_id", -1)), str(row.get("question_index", "")))


def black_options(row):
    out = []
    for k in range(3):
        pair = bbq_eval.get_answer_info(row, k)
        if pair and str(pair[-1]).strip().lower() in BLACK_TAGS:
            out.append(k)
    return out


def is_black_referent_ambig(row):
    """Ambiguous + exactly one Black-tagged option (same rule as build_anchored)."""
    return row.get("context_condition") == "ambig" and len(black_options(row)) == 1


def load_full_race():
    rows = []
    with open(RACE_CACHE) as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                r["category"] = "Race_ethnicity"
                rows.append(r)
    return rows


def build_item_keys():
    """(example_id, question_index) of the 400 anchored-build items."""
    keys = set()
    with open(BUILD_ITEMS) as f:
        for line in f:
            line = line.strip()
            if line:
                d = json.loads(line)
                keys.add((int(d["example_id"]), str(d["question_index"])))
    return keys


def seed42_n37_keys():
    """The Black-referent AMBIGUOUS keys inside the seed-42 random-1000 sample
    (the historical n=37 pilot denominator)."""
    sample = bbq_eval.load_bbq(DATASET, 42, 1000, None)
    keys = set()
    for r in sample:
        if r.get("category") == "Race_ethnicity" and is_black_referent_ambig(r):
            keys.add(key_of(r))
    return keys


def main():
    full = load_full_race()
    build_keys = build_item_keys()
    all_br_ambig = [r for r in full if is_black_referent_ambig(r)]
    total = len(all_br_ambig)

    eval_rows = [r for r in all_br_ambig if key_of(r) not in build_keys]
    eval_keys = {key_of(r) for r in eval_rows}

    print("=" * 72)
    print("E1 eval set: FULL Black-referent AMBIGUOUS Race_ethnicity, minus build items")
    print("=" * 72)
    print(f"  full Race_ethnicity rows                 : {len(full)}")
    print(f"  Black-referent ambiguous (all)           : {total}")
    print(f"  anchored-build items (excluded)          : {len(build_keys)}")
    print(f"  overlap build∩(BR-ambig) excluded        : {total - len(eval_rows)}")
    print(f"  --> EVAL N                               : {len(eval_rows)}")

    # ---- assertions ----
    assert eval_keys.isdisjoint(build_keys), "eval set overlaps the 400 build items!"
    n37 = seed42_n37_keys()
    print(f"  seed-42 pilot n=37 Black-referent ambig  : {len(n37)}")
    missing = n37 - eval_keys
    assert not missing, (
        f"{len(missing)} of the seed-42 pilot items are NOT in the eval set "
        f"(they may have leaked into the 400 build items): {sorted(missing)[:5]}"
    )
    print(f"  seed-42 pilot ⊆ eval set                 : PASS ({len(n37)} items covered)")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        for r in eval_rows:
            f.write(json.dumps(r) + "\n")
    print(f"\nWrote {len(eval_rows)} rows -> {OUT}")
    print("Feed to the eval with:  --items experiments/data/black_referent_ambig_eval.jsonl")


if __name__ == "__main__":
    main()
