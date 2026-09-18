#!/usr/bin/env python3
"""Step 1 comparison: match the rerun s*=0.9 outputs against the published PI
run row by row (on example_id) and report the fraction of identical outputs.

    python results/ablation_setpoint/compare_repro.py \
        --tier-dir results/ablation_setpoint/tier1 \
        --ref-dir results/balanced/results_balanced --limit 100 \
        --out results/ablation_setpoint/repro.json

Reports, per rotation and pooled: exact model_output match rate, strict-class
agreement rate (target / comparator / abstain / invalid), and the strict gap of
both runs on the compared rows. A high but imperfect match is expected on a
different GPU type; report it either way.
"""
import argparse
import json
import os
import re

STRICT_RE = re.compile(r"^\s*([ABCabc])(?:$|[\s.,:;)\]!?'\"-])")


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


def gap_pp(rows):
    c = [classify(r) for r in rows]
    return round(100.0 * (c.count("target") - c.count("comparator")) / len(c), 1)


def load(path, limit=None):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    return rows[:limit] if limit else rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier-dir", required=True)
    ap.add_argument("--ref-dir", required=True)
    ap.add_argument("--tag", default="dpid_PI_s0p9")
    ap.add_argument("--ref-tag", default="dpid_PI")
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    per_rot = {}
    pooled_same = pooled_agree = pooled_n = 0
    pooled_new, pooled_ref = [], []
    for r in range(3):
        new = load(os.path.join(args.tier_dir, f"rot{r}", f"cond_{args.tag}_samples.jsonl"))
        ref = load(os.path.join(args.ref_dir, f"rot{r}", f"cond_{args.ref_tag}_samples.jsonl"), args.limit)
        ref_by_id = {x["example_id"]: x for x in ref}
        missing = [x["example_id"] for x in new if x["example_id"] not in ref_by_id]
        if missing:
            raise SystemExit(f"rot{r}: {len(missing)} rerun example_ids not in the first "
                             f"{args.limit} reference rows (e.g. {missing[:5]}); "
                             f"was --limit {args.limit} used for the rerun?")
        if len(new) != len(ref):
            raise SystemExit(f"rot{r}: rerun has {len(new)} rows, reference slice has {len(ref)}")
        same = agree = 0
        diffs = []
        for x in new:
            y = ref_by_id[x["example_id"]]
            if x["model_output"] == y["model_output"]:
                same += 1
            else:
                diffs.append({"example_id": x["example_id"],
                              "rerun": x["model_output"], "published": y["model_output"]})
            if classify(x) == classify(y):
                agree += 1
        n = len(new)
        per_rot[f"rot{r}"] = {
            "n": n,
            "identical_output_frac": round(same / n, 3),
            "strict_class_agreement_frac": round(agree / n, 3),
            "gap_pp_rerun": gap_pp(new),
            "gap_pp_published": gap_pp([ref_by_id[x["example_id"]] for x in new]),
            "n_differing": len(diffs),
            "differing_examples": diffs[:10],
        }
        pooled_same += same
        pooled_agree += agree
        pooled_n += n
        pooled_new += new
        pooled_ref += [ref_by_id[x["example_id"]] for x in new]

    report = {
        "tag": args.tag,
        "ref_tag": args.ref_tag,
        "ref_dir": args.ref_dir,
        "limit": args.limit,
        "pooled": {
            "n": pooled_n,
            "identical_output_frac": round(pooled_same / pooled_n, 3),
            "strict_class_agreement_frac": round(pooled_agree / pooled_n, 3),
            "gap_pp_rerun": gap_pp(pooled_new),
            "gap_pp_published": gap_pp(pooled_ref),
        },
        **per_rot,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(report, open(args.out, "w"), indent=2)
    p = report["pooled"]
    print(f"identical outputs: {p['identical_output_frac']:.3f}  "
          f"strict-class agreement: {p['strict_class_agreement_frac']:.3f}  "
          f"gap rerun {p['gap_pp_rerun']} pp vs published {p['gap_pp_published']} pp  "
          f"(n={p['n']}) -> {args.out}")


if __name__ == "__main__":
    main()
