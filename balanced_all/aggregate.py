#!/usr/bin/env python
"""balanced_all/aggregate.py -- pool one condition's 3 rotation results.

Takes a glob matching the 3 per-rotation result JSONs of ONE condition and
prints the pooled position-balanced numbers plus a ready-to-paste markdown row.

Handles BOTH result schemas:
  * baselines/common.py + steering/denoise_pid.py: counts {black, nonblack,
    abstain, unparseable}                     (black -> target)
  * multirace/common_eval.py + multirace/denoise_pid.py: counts {target,
    nontarget, abstain, unparseable}

Outputs:
  * pooled rates over the pooled n (1200 for 400 items x 3 rotations)
  * balanced gap = target_rate - nontarget_rate  (letter-jam-immune)
  * per-position gap @A/@B/@C, recomputed from the sibling _samples.jsonl files
    by joining each item's predicted class against the LETTER the target option
    sits on in that rotation.

Usage:
    python balanced_all/aggregate.py \
        --cond-glob 'results/balanced_all/black/caa/rot*/cond_caa.json' \
        [--label 'CAA a16 L14']
Selftest (offline, synthetic JSONs):  python balanced_all/aggregate.py --selftest
"""
import argparse
import glob
import json
import os
import sys

LETTERS = ["A", "B", "C"]
CLS_MAP = {"black": "target", "nonblack": "nontarget",
           "target": "target", "nontarget": "nontarget",
           "abstain": "abstain", "unparseable": "unparseable"}
CLASSES = ("target", "nontarget", "abstain", "unparseable")


def norm_counts(counts):
    out = {c: 0 for c in CLASSES}
    for k, v in counts.items():
        out[CLS_MAP[k]] += v
    return out


def target_letter_of(sample):
    """Letter the target option sits on, whichever schema wrote the sample."""
    if sample.get("target_letter") in LETTERS:
        return sample["target_letter"]
    for key in ("target_idx", "black_idx"):
        idx = sample.get(key)
        if isinstance(idx, int) and 0 <= idx < 3:
            return LETTERS[idx]
    return None


def samples_path_of(result_path):
    return result_path[:-len(".json")] + "_samples.jsonl"


def aggregate(cond_glob):
    files = sorted(glob.glob(cond_glob))
    assert len(files) == 3, f"expected 3 rotation result JSONs, glob matched {files}"

    pooled = {c: 0 for c in CLASSES}
    n = 0
    for p in files:
        d = json.load(open(p))
        pooled_p = norm_counts(d["counts"])
        for c in CLASSES:
            pooled[c] += pooled_p[c]
        n += d["n"]
    assert sum(pooled.values()) == n, f"counts {pooled} do not sum to n={n}"

    rates = {c: pooled[c] / n for c in CLASSES}
    gap = rates["target"] - rates["nontarget"]

    # per-position gap from the sample files
    pos = {L: {c: 0 for c in CLASSES} for L in LETTERS}
    for p in files:
        sp = samples_path_of(p)
        for line in open(sp):
            if not line.strip():
                continue
            s = json.loads(line)
            L = target_letter_of(s)
            assert L is not None, f"cannot derive target letter in {sp}"
            pos[L][CLS_MAP[s["pred_class"]]] += 1
    pos_gap = {}
    for L in LETTERS:
        nL = sum(pos[L].values())
        pos_gap[L] = (pos[L]["target"] - pos[L]["nontarget"]) / nL if nL else float("nan")

    return {"files": files, "n": n, "counts": pooled, "rates": rates,
            "gap": gap, "pos_n": {L: sum(pos[L].values()) for L in LETTERS},
            "pos_gap": pos_gap}


def report(agg, label):
    r, pg = agg["rates"], agg["pos_gap"]
    print(f"condition : {label}")
    print(f"files     : {agg['files']}")
    print(f"pooled n  : {agg['n']}  (per-position n: "
          + " ".join(f"{L}={agg['pos_n'][L]}" for L in LETTERS) + ")")
    print(f"rates     : target={r['target']:.3f} nontarget={r['nontarget']:.3f} "
          f"abstain={r['abstain']:.3f} unparseable={r['unparseable']:.3f}")
    print(f"balanced gap (target-nontarget) = {agg['gap']:+.3f}")
    print(f"per-position gap: @A={pg['A']:+.2f} @B={pg['B']:+.2f} @C={pg['C']:+.2f}")
    print("\nmarkdown row:")
    print(f"| {label} | {r['target']:.3f} | {r['nontarget']:.3f} | "
          f"{r['abstain']:.3f} | {r['unparseable']:.3f} | **{agg['gap']:+.3f}** | "
          f"{pg['A']:+.2f} / {pg['B']:+.2f} / {pg['C']:+.2f} |")


# --------------------------------------------------------------------------- #
# Selftest: synthetic 3-rotation condition in BOTH schemas (no GPU, no repo data)
# --------------------------------------------------------------------------- #
def _write_cond(d, rot, schema, rows):
    """rows: list of (target_idx, pred_class-in-target-schema)."""
    cdir = os.path.join(d, f"rot{rot}")
    os.makedirs(cdir, exist_ok=True)
    to_schema = ({"target": "black", "nontarget": "nonblack"} if schema == "black"
                 else {})
    counts = {c: 0 for c in CLASSES}
    samples = []
    for tidx, cls in rows:
        counts[cls] += 1
        s = {"pred_class": to_schema.get(cls, cls)}
        if schema == "black":
            s["black_idx"] = tidx
        else:
            s["target_idx"] = tidx
            s["target_letter"] = LETTERS[tidx]
        samples.append(s)
    if schema == "black":
        counts = {"black": counts["target"], "nonblack": counts["nontarget"],
                  "abstain": counts["abstain"], "unparseable": counts["unparseable"]}
    with open(os.path.join(cdir, "cond_x.json"), "w") as f:
        json.dump({"counts": counts, "n": len(rows)}, f)
    with open(os.path.join(cdir, "cond_x_samples.jsonl"), "w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")


def _selftest():
    import tempfile
    ok = True

    def chk(name, cond):
        nonlocal ok
        ok &= bool(cond)
        print(f"[selftest-aggregate] {name:52s} : {'PASS' if cond else 'FAIL'}")

    for schema in ("black", "target"):
        with tempfile.TemporaryDirectory() as d:
            # rot0: target@A; rot1: target@B; rot2: target@C.  4 rows each.
            _write_cond(d, 0, schema, [(0, "target"), (0, "target"),
                                       (0, "nontarget"), (0, "abstain")])
            _write_cond(d, 1, schema, [(1, "target"), (1, "nontarget"),
                                       (1, "nontarget"), (1, "unparseable")])
            _write_cond(d, 2, schema, [(2, "abstain"), (2, "abstain"),
                                       (2, "target"), (2, "target")])
            agg = aggregate(os.path.join(d, "rot*", "cond_x.json"))
            chk(f"[{schema}] pooled n == 12", agg["n"] == 12)
            chk(f"[{schema}] pooled counts t/nt/ab/un == 5/3/3/1",
                agg["counts"] == {"target": 5, "nontarget": 3,
                                  "abstain": 3, "unparseable": 1})
            chk(f"[{schema}] gap == (5-3)/12", abs(agg["gap"] - 2 / 12) < 1e-12)
            chk(f"[{schema}] per-position n == 4/4/4",
                agg["pos_n"] == {"A": 4, "B": 4, "C": 4})
            want = {"A": (2 - 1) / 4, "B": (1 - 2) / 4, "C": (2 - 0) / 4}
            chk(f"[{schema}] gap@A/@B/@C == +.25/-.25/+.50",
                all(abs(agg["pos_gap"][L] - want[L]) < 1e-12 for L in LETTERS))

    print(f"[selftest-aggregate] OVERALL: {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cond-glob", default=None,
                    help="glob matching one condition's 3 rotation result JSONs")
    ap.add_argument("--label", default=None, help="row label (default: glob)")
    ap.add_argument("--selftest", action="store_true",
                    help="offline check on synthetic result files (no GPU)")
    args = ap.parse_args()

    if args.selftest:
        sys.exit(0 if _selftest() else 1)
    if not args.cond_glob:
        ap.error("--cond-glob required (or --selftest)")
    report(aggregate(args.cond_glob), args.label or args.cond_glob)


if __name__ == "__main__":
    main()
