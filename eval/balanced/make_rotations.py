#!/usr/bin/env python
"""Build 3 CYCLIC-ROTATION copies of _sweep400.jsonl for position-balanced eval.

BBQ items have 3 options rendered A/B/C (ans0->A, ans1->B, ans2->C in
build_prompt). The model has a letter-position preference; under strong steering
it can collapse onto a single letter, faking a high "black-pick" rate whenever the
Black option happens to sit at that letter. We neutralize this by evaluating every
item under all 3 cyclic rotations so the Black option sits at A, B and C once each.

ROTATION r (r = 0,1,2), content-only remap:
    new_ans[j]         = old_ans[(j - r) % 3]
    new answer_info[j] = old answer_info[(j - r) % 3]
    new label          = (old_label + r) % 3
=> the option originally at index i lands at (i + r) % 3, so across r=0,1,2 the
Black option (and every option) visits A, B, C exactly once.

rot0 (r=0) is byte-identical to the original for the rotated fields. ALL other
fields are preserved verbatim. Only ans0/ans1/ans2, answer_info and label change.

Writes _sweep400_rot0.jsonl / _rot1.jsonl / _rot2.jsonl into results/balanced/.
"""
import json
import os

ROOT = "/home/lukas/users/shashmi/dlm_bias"
SWEEP400 = os.path.join(ROOT, "data", "bbq_items", "_sweep400.jsonl")
OUTDIR = os.path.join(ROOT, "results", "balanced")


def rotate_row(row, r):
    """Return a new row with options cyclically rotated by r (content-only)."""
    new = dict(row)  # shallow copy preserves all untouched fields
    old_ans = [row["ans0"], row["ans1"], row["ans2"]]
    old_info = row["answer_info"]
    for j in range(3):
        src = (j - r) % 3
        new[f"ans{j}"] = old_ans[src]
    # answer_info is its own object -> rebuild so we never alias the original.
    new_info = dict(old_info)
    for j in range(3):
        src = (j - r) % 3
        new_info[f"ans{j}"] = old_info[f"ans{src}"]
    new["answer_info"] = new_info
    if isinstance(row.get("label"), int):
        new["label"] = (row["label"] + r) % 3
    return new


def main():
    rows = [json.loads(l) for l in open(SWEEP400) if l.strip()]
    os.makedirs(OUTDIR, exist_ok=True)
    for r in range(3):
        out = os.path.join(OUTDIR, f"_sweep400_rot{r}.jsonl")
        with open(out, "w") as f:
            for row in rows:
                f.write(json.dumps(rotate_row(row, r)) + "\n")
        print(f"wrote {out}  ({len(rows)} rows, rotation r={r})")


if __name__ == "__main__":
    main()
