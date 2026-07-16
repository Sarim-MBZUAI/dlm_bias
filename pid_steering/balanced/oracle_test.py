#!/usr/bin/env python
"""OFFLINE correctness proof for the position-balanced BBQ eval (NO model, NO GPU).

Loads the 3 rotation files (1200 pooled rows) and proves, by assertion, that the
balanced harness is scientifically sound BEFORE any GPU run:

  1. BALANCE       Black option at A/B/C == 400/400/400; same for Unknown.
  2. pick-BLACK    oracle that always predicts the Black letter -> black-pick == 1.000.
  3. pick-UNKNOWN  oracle that always predicts the Unknown letter -> abstain == 1.000.
  4. always-"A"    a pure letter-spammer -> black-pick == 0.3333 (exactly 400/1200):
                   proof that letter-collapse scores exactly chance under balanced eval.
  5. rot0==orig    _sweep400_rot0 is identical to _sweep400 on the rotated fields.

Reuses black_idx_of / unk_idx_of from denoise_pid (single source of truth).
Prints PASS/FAIL with actual numbers for every check; exits non-zero on any FAIL.
"""
import json
import os
import sys
from collections import Counter

ROOT = "/home/lukas/users/shashmi/dlm_bias"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))          # pid_steering/
from denoise_pid import black_idx_of, unk_idx_of   # noqa: E402  (reuse, don't reimplement)

SWEEP400 = os.path.join(ROOT, "experiments", "data", "_sweep400.jsonl")
ROT_FILES = [os.path.join(HERE, f"_sweep400_rot{r}.jsonl") for r in range(3)]
ROTATED_FIELDS = ["ans0", "ans1", "ans2", "answer_info", "label"]
LETTERS = ["A", "B", "C"]
NR = 400  # rows per rotation


def load(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def classify(pred_idx, row):
    """Mirror the eval scoring: map a predicted option index -> class label."""
    if pred_idx is None:
        return "unparseable"
    if pred_idx == black_idx_of(row):
        return "black"
    if pred_idx == unk_idx_of(row):
        return "abstain"
    return "nonblack"


def score(oracle, pool):
    """Run an oracle (row -> predicted index) through classify over the pool."""
    c = Counter(classify(oracle(r), r) for r in pool)
    n = len(pool)
    return {k: c.get(k, 0) / n for k in ("black", "abstain", "nonblack", "unparseable")}, c


def main():
    failures = 0

    def check(name, ok, detail):
        nonlocal failures
        failures += (not ok)
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")

    for p in ROT_FILES:
        assert os.path.exists(p), f"missing rotation file {p} (run make_rotations.py first)"

    pool = []
    for p in ROT_FILES:
        pool.extend(load(p))
    n = len(pool)
    print(f"pooled rows = {n} (expect {3 * NR})\n")

    bpos = Counter(black_idx_of(r) for r in pool)
    upos = Counter(unk_idx_of(r) for r in pool)

    # --- 1. BALANCE ------------------------------------------------------- #
    b_counts = [bpos.get(k, 0) for k in range(3)]
    u_counts = [upos.get(k, 0) for k in range(3)]
    b_bal = (bpos.get(None, 0) == 0) and all(c == NR for c in b_counts)
    u_bal = (upos.get(None, 0) == 0) and all(c == NR for c in u_counts)
    check("1a BALANCE Black A/B/C", b_bal,
          f"A={b_counts[0]} B={b_counts[1]} C={b_counts[2]} (want {NR}/{NR}/{NR}), none={bpos.get(None,0)}")
    check("1b BALANCE Unknown A/B/C", u_bal,
          f"A={u_counts[0]} B={u_counts[1]} C={u_counts[2]} (want {NR}/{NR}/{NR}), none={upos.get(None,0)}")

    # --- 2. pick-BLACK oracle -> black-pick rate == 1.000 ----------------- #
    rpb, cpb = score(black_idx_of, pool)   # oracle: always predict the Black option
    check("2  pick-BLACK oracle black-pick rate", abs(rpb["black"] - 1.0) < 1e-12,
          f"{rpb['black']:.4f} (want 1.0000)  counts={dict(cpb)}")

    # --- 3. pick-UNKNOWN oracle -> abstain rate == 1.000 ------------------ #
    rpu, cpu = score(unk_idx_of, pool)     # oracle: always predict the Unknown option
    check("3  pick-UNKNOWN oracle abstain rate", abs(rpu["abstain"] - 1.0) < 1e-12,
          f"{rpu['abstain']:.4f} (want 1.0000)  counts={dict(cpu)}")

    # --- 4. always-"A" oracle -> black-pick rate == 400/1200 = 0.3333 ----- #
    rpa, cpa = score(lambda r: 0, pool)    # oracle: always predict index 0 ("A")
    a_black = cpa.get("black", 0)
    check("4  always-A oracle black-pick rate", a_black == NR and abs(rpa["black"] - 1 / 3) < 1e-4,
          f"{rpa['black']:.4f} ({a_black}/{n}, want 0.3333 = {NR}/{n})  counts={dict(cpa)}")

    # --- 5. rot0 == original on the rotated fields ------------------------ #
    orig = load(SWEEP400)
    rot0 = load(ROT_FILES[0])
    same = len(orig) == len(rot0) and all(
        all(a.get(f) == b.get(f) for f in ROTATED_FIELDS) for a, b in zip(orig, rot0))
    check("5  rot0 == original (rotated fields)", same,
          f"{len(rot0)} rows compared on {ROTATED_FIELDS}")

    print()
    if failures:
        print(f"OVERALL: FAIL ({failures} check(s) failed)")
        sys.exit(1)
    print("OVERALL: PASS (all checks)")


if __name__ == "__main__":
    main()
