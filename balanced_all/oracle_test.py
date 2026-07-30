#!/usr/bin/env python
"""balanced_all/oracle_test.py -- OFFLINE correctness proof for ANY rotation set.

Generalization of eval/balanced/oracle_test.py (Black-only) to any target in
multirace/targets.py (target_idx_of / unk_idx_of are reused by import; the tag
registry is the single source of truth). No model, no GPU.

Checks (all must PASS; exits non-zero otherwise):
  1. BALANCE       target and Unknown each sit at A/B/C exactly n/3 per position
                   across the pooled rotations.
  2. pick-TARGET   oracle that always predicts the target letter -> rate 1.000.
  3. pick-UNKNOWN  oracle that always predicts the Unknown letter -> abstain 1.000.
  4. always-"A"    pure letter-spammer -> target rate exactly 1/3 (letter-jam null).
  5. rot0==orig    the _rot0 file is identical to --orig on the rotated fields.

Usage:
    python balanced_all/oracle_test.py \
        --items-glob 'results/balanced_all/rotations/_sweep400_arab_rot*.jsonl' \
        --target arab --orig data/bbq_items/_sweep400_arab.jsonl
"""
import argparse
import glob
import importlib.util
import json
import os
import sys
from collections import Counter

ROOT = "/home/lukas/users/shashmi/dlm_bias"


def _load_targets():
    p = os.path.join(ROOT, "multirace", "targets.py")
    spec = importlib.util.spec_from_file_location("multirace_targets", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


T = _load_targets()  # target_idx_of(row, target), unk_idx_of(row), TARGET_TAGS

ROTATED_FIELDS = ["ans0", "ans1", "ans2", "answer_info", "label"]


def load(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def classify(pred_idx, row, target):
    if pred_idx is None:
        return "unparseable"
    if pred_idx == T.target_idx_of(row, target):
        return "target"
    if pred_idx == T.unk_idx_of(row):
        return "abstain"
    return "nontarget"


def score(oracle, pool, target):
    c = Counter(classify(oracle(r), r, target) for r in pool)
    n = len(pool)
    return {k: c.get(k, 0) / n for k in ("target", "abstain", "nontarget",
                                         "unparseable")}, c


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--items-glob", required=True,
                    help="glob matching the 3 rotation jsonl files")
    ap.add_argument("--target", required=True, choices=sorted(T.TARGET_TAGS),
                    help="steering target whose option must be position-balanced")
    ap.add_argument("--orig", required=True,
                    help="original (unrotated) items jsonl; rot0 must equal it "
                         "on the rotated fields")
    args = ap.parse_args()

    files = sorted(glob.glob(args.items_glob))
    assert len(files) == 3, f"expected 3 rotation files, glob matched {files}"

    failures = 0

    def check(name, ok, detail):
        nonlocal failures
        failures += (not ok)
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")

    pool = []
    for p in files:
        pool.extend(load(p))
    n = len(pool)
    nr = n // 3
    print(f"pooled rows = {n} over {len(files)} rotations (n/3 = {nr}); "
          f"target = {args.target}\n")
    check("0  pooled count divisible by 3", n == 3 * nr, f"n={n}")

    tgt_of = lambda r: T.target_idx_of(r, args.target)  # noqa: E731
    tpos = Counter(tgt_of(r) for r in pool)
    upos = Counter(T.unk_idx_of(r) for r in pool)

    t_counts = [tpos.get(k, 0) for k in range(3)]
    u_counts = [upos.get(k, 0) for k in range(3)]
    t_bal = (tpos.get(None, 0) == 0) and all(c == nr for c in t_counts)
    u_bal = (upos.get(None, 0) == 0) and all(c == nr for c in u_counts)
    check(f"1a BALANCE {args.target} A/B/C", t_bal,
          f"A={t_counts[0]} B={t_counts[1]} C={t_counts[2]} "
          f"(want {nr}/{nr}/{nr}), none={tpos.get(None, 0)}")
    check("1b BALANCE Unknown A/B/C", u_bal,
          f"A={u_counts[0]} B={u_counts[1]} C={u_counts[2]} "
          f"(want {nr}/{nr}/{nr}), none={upos.get(None, 0)}")

    rpt, cpt = score(tgt_of, pool, args.target)
    check("2  pick-TARGET oracle target rate", abs(rpt["target"] - 1.0) < 1e-12,
          f"{rpt['target']:.4f} (want 1.0000)  counts={dict(cpt)}")

    rpu, cpu = score(T.unk_idx_of, pool, args.target)
    check("3  pick-UNKNOWN oracle abstain rate", abs(rpu["abstain"] - 1.0) < 1e-12,
          f"{rpu['abstain']:.4f} (want 1.0000)  counts={dict(cpu)}")

    rpa, cpa = score(lambda r: 0, pool, args.target)
    a_t = cpa.get("target", 0)
    check("4  always-A oracle target rate", a_t == nr and abs(rpa["target"] - 1 / 3) < 1e-9,
          f"{rpa['target']:.4f} ({a_t}/{n}, want {nr}/{n} = 0.3333)  counts={dict(cpa)}")

    orig = load(args.orig)
    rot0 = load(files[0])
    same = len(orig) == len(rot0) and all(
        all(a.get(f) == b.get(f) for f in ROTATED_FIELDS)
        for a, b in zip(orig, rot0))
    check("5  rot0 == original (rotated fields)", same,
          f"{len(rot0)} rows vs {len(orig)} compared on {ROTATED_FIELDS}")

    print()
    if failures:
        print(f"OVERALL: FAIL ({failures} check(s) failed)")
        sys.exit(1)
    print("OVERALL: PASS (all checks)")


if __name__ == "__main__":
    main()
