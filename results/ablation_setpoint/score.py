#!/usr/bin/env python3
"""Score the setpoint ablation with the paper's strict parser.

    python results/ablation_setpoint/score.py results/ablation_setpoint/tier1

Uses the same regex as balanced_all/strict_pool.py, pools the three rotations,
keeps invalid outputs in the denominator, and reports each condition's gap
(target minus comparator, in percentage points) with a 10k-resample bootstrap
CI plus a paired difference against the s*=0.9 reference. Writes summary.json
next to the rot*/ directories.
"""
import json, re, sys, glob, os
import numpy as np

STRICT_RE = re.compile(r"^\s*([ABCabc])(?:$|[\s.,:;)\]!?'\"-])")
TIER_DIR = sys.argv[1]            # e.g. results/ablation_setpoint/tier1

def classify(row):
    m = STRICT_RE.match(row.get("model_output") or "")
    if not m:
        return "invalid"
    idx = "ABC".index(m.group(1).upper())
    if idx == row["black_idx"]:
        return "target"
    if idx == row["unk_idx"]:
        return "abstain"
    return "comparator"

def load(tag):
    rows = []
    for r in range(3):
        path = os.path.join(TIER_DIR, f"rot{r}", f"cond_{tag}_samples.jsonl")
        rows += [dict(json.loads(l), rot=r) for l in open(path) if l.strip()]
    return rows

def gap_vec(rows):
    c = [classify(x) for x in rows]
    return np.array([(k == "target") - (k == "comparator") for k in c], float), c

tags = sorted({os.path.basename(p)[5:-14] for p in
               glob.glob(os.path.join(TIER_DIR, "rot0", "cond_*_samples.jsonl"))})
data = {t: load(t) for t in tags}
ref = "dpid_PI_s0p9"
rng = np.random.default_rng(0)
summary = {}
for t in tags:
    g, c = gap_vec(data[t])
    n = len(c)
    idx = rng.integers(0, n, size=(10000, n))
    row = {
        "n": n,
        **{k: round(100 * c.count(k) / n, 1) for k in ("target", "comparator", "abstain", "invalid")},
        "gap_pp": round(100 * g.mean(), 1),
        "gap_ci95": [round(100 * v, 1) for v in np.percentile(g[idx].mean(1), [2.5, 97.5])],
    }
    if "alpha_traj" in data[t][0] and max(data[t][0]["alpha_traj"]) > 0:
        A = np.array([x["alpha_traj"] for x in data[t]])
        row["mean_command_steps_1_32"] = round(float(A[:, :32].mean()), 2)
        row["share_ever_at_limit"] = round(float((A >= 6 - 1e-6).any(1).mean()), 3)
    if t != ref and ref in data:
        keys = [(x["rot"], x["example_id"]) for x in data[ref]]
        assert keys == [(x["rot"], x["example_id"]) for x in data[t]], "row order mismatch"
        g_ref, _ = gap_vec(data[ref])
        d = g - g_ref
        row["gap_minus_s0p9_pp"] = round(100 * d.mean(), 1)
        row["gap_minus_s0p9_ci95"] = [round(100 * v, 1) for v in np.percentile(d[idx].mean(1), [2.5, 97.5])]
    summary[t] = row

print(json.dumps(summary, indent=2))
json.dump(summary, open(os.path.join(TIER_DIR, "summary.json"), "w"), indent=2)
