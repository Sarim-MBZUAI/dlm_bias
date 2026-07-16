#!/usr/bin/env python
"""Draw 3 independent 400-item eval samples (seeds 1/2/3) from the 1600 superset,
each disjoint from the direction-build items, and write its 3 cyclic rotations.
Rotation logic identical to make_rotations.py (oracle-validated)."""
import json
import os
import random

ROOT = "/home/lukas/users/shashmi/dlm_bias"
BIG = os.path.join(ROOT, "experiments", "data", "black_referent_ambig_eval.jsonl")
BUILD = os.path.join(ROOT, "pid_steering", "direction_examples.jsonl")
OUTBASE = os.path.join(ROOT, "pid_steering", "balanced", "seeds")
SEEDS = [1, 2, 3]
N = 400


def key(r):
    return (r["example_id"], str(r.get("question_index", "")))


def rotate(row, r):
    """new_ans[j] = old_ans[(j-r)%3]; answer_info remapped; label=(label+r)%3."""
    a = [row["ans0"], row["ans1"], row["ans2"]]
    ai = row["answer_info"]
    old_ai = [ai["ans0"], ai["ans1"], ai["ans2"]]
    out = dict(row)
    for j in range(3):
        src = (j - r) % 3
        out[f"ans{j}"] = a[src]
    out["answer_info"] = {f"ans{j}": old_ai[(j - r) % 3] for j in range(3)}
    out["label"] = (row["label"] + r) % 3
    return out


big = [json.loads(l) for l in open(BIG)]
build_keys = {key(json.loads(l)) for l in open(BUILD)}
pool = [r for r in big if key(r) not in build_keys]     # clean pool (no contamination)
assert len(pool) >= N, f"pool too small: {len(pool)}"

for s in SEEDS:
    rng = random.Random(s)
    sample = rng.sample(pool, N)
    d = os.path.join(OUTBASE, f"seed{s}")
    os.makedirs(d, exist_ok=True)
    for r in [0, 1, 2]:
        with open(os.path.join(d, f"_sweep400_rot{r}.jsonl"), "w") as f:
            for row in sample:
                f.write(json.dumps(rotate(row, r), ensure_ascii=False) + "\n")
    # contamination check for this seed
    contam = len({key(x) for x in sample} & build_keys)
    print(f"seed {s}: 400 items, build-overlap={contam} (must be 0)")
print(f"clean pool size = {len(pool)} of {len(big)}")
