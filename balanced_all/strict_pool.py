#!/usr/bin/env python3
"""Strict reparse + pooling + bootstrap CIs for the position-balanced runs.

Strict parsing rule:

  A response is *valid* only when the answer letter (A, B, or C, either
  case) appears at the very start of ``model_output`` (leading whitespace
  allowed) and is followed by whitespace, punctuation, or end of string.
  Anything else is *invalid* and stays in the denominator.

Each valid row is classified with ``black_idx``/``unk_idx``:
  letter index == black_idx -> target
  letter index == unk_idx   -> abstain
  otherwise                 -> comparator

For every condition the three rotations (rot0/rot1/rot2, 400 items each)
are pooled to 1200 items.  Reported per condition:
  strict     target / comparator / abstain / invalid rates, gap, and the
             per-position gaps Gap@A and Gap@(B,C mean) using the per-row
             letter position of the Black option;
  permissive the same rates under the old generation-time parser (first
             standalone A/B/C token, else any A/B/C character);
  bootstrap  95% percentile CI of the strict gap from a 10,000-resample
             item-level bootstrap (resample the 1200 pooled items with
             replacement; numpy default_rng(seed=0), fresh per condition).

Usage:
  python balanced_all/strict_pool.py [--json OUT.json]
Run from the repo root (paths are resolved relative to this file's repo).
"""
import argparse
import json
import os
import re
import sys
from decimal import Decimal, ROUND_HALF_UP
from fractions import Fraction

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

LETTERS = "ABC"
# Letter at the very start, followed by whitespace/punctuation/end.
STRICT_RE = re.compile(r"^\s*([ABCabc])(?:$|[\s.,:;)\]!?'\"-])")
# The old permissive rule (first standalone A/B/C token, case-insensitive,
# falling back to any A/B/C character).
PERMISSIVE_WORD_RE = re.compile(r"\b([ABCabc])\b")
PERMISSIVE_CHAR_RE = re.compile(r"[ABCabc]")

N_BOOT = 10_000
SEED = 0


def r3(x):
    """Round to 3 decimals with decimal (not binary) half-up rounding."""
    if isinstance(x, Fraction):
        d = Decimal(x.numerator) / Decimal(x.denominator)
    else:
        d = Decimal(repr(float(x)))
    return str(d.quantize(Decimal("0.001"), rounding=ROUND_HALF_UP))


def strict_letter(text):
    m = STRICT_RE.match(text or "")
    return m.group(1).upper() if m else None


def permissive_letter(text):
    m = PERMISSIVE_WORD_RE.search(text or "")
    if m:
        return m.group(1).upper()
    m = PERMISSIVE_CHAR_RE.search(text or "")
    return m.group(0).upper() if m else None


def target_index(row):
    """Letter index of the steered-target option.

    Black-target samples store it as ``black_idx``; the
    multi-target samples store it as ``target_idx``. Same semantics.
    """
    return row["target_idx"] if "target_idx" in row else row["black_idx"]


def classify_letter(letter, row):
    if letter is None:
        return "invalid"
    idx = LETTERS.index(letter)
    if idx == target_index(row):
        return "target"
    if idx == row["unk_idx"]:
        return "abstain"
    return "comparator"


def load_rows(paths):
    rows = []
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows


def score(rows, parser):
    counts = {"target": 0, "comparator": 0, "abstain": 0, "invalid": 0}
    by_pos = {p: {"target": 0, "comparator": 0, "n": 0} for p in LETTERS}
    signs = []  # per-item bootstrap value: +1 target, -1 comparator, 0 else
    for row in rows:
        cls = classify_letter(parser(row.get("model_output", "")), row)
        counts[cls] += 1
        pos = LETTERS[target_index(row)]
        by_pos[pos]["n"] += 1
        if cls in ("target", "comparator"):
            by_pos[pos][cls] += 1
        signs.append(1 if cls == "target" else (-1 if cls == "comparator" else 0))
    n = len(rows)
    out = {
        "n": n,
        "counts": counts,
        "target": Fraction(counts["target"], n),
        "comparator": Fraction(counts["comparator"], n),
        "abstain": Fraction(counts["abstain"], n),
        "invalid": Fraction(counts["invalid"], n),
        "gap": Fraction(counts["target"] - counts["comparator"], n),
        "signs": signs,
    }
    if by_pos["A"]["n"]:
        out["gap_A"] = Fraction(by_pos["A"]["target"] - by_pos["A"]["comparator"],
                                by_pos["A"]["n"])
    nbc = by_pos["B"]["n"] + by_pos["C"]["n"]
    if nbc:
        out["gap_BC"] = Fraction((by_pos["B"]["target"] + by_pos["C"]["target"])
                                 - (by_pos["B"]["comparator"] + by_pos["C"]["comparator"]),
                                 nbc)
    out["by_pos"] = by_pos
    return out


def bootstrap_ci(signs, n_boot=N_BOOT, seed=SEED):
    """95% percentile CI of the gap: resample items with replacement."""
    rng = np.random.default_rng(seed)
    v = np.asarray(signs, dtype=np.int8)
    n = len(v)
    idx = rng.integers(0, n, size=(n_boot, n))
    gaps = v[idx].mean(axis=1)
    lo, hi = np.percentile(gaps, [2.5, 97.5])
    return float(lo), float(hi)


# label -> list of the 3 rotation samples paths (relative to repo root)
def bal_all(subdir, stem):
    return ["results/balanced_all/black/%s/rot%d/cond_%s_samples.jsonl" % (subdir, r, stem)
            for r in range(3)]


def bal_old(stem):
    return ["results/balanced/results_balanced/rot%d/cond_%s_samples.jsonl" % (r, stem)
            for r in range(3)]


CONDITIONS = [
    # --- headline conditions (results/balanced) ---
    ("Clean (base)",              bal_old("base")),
    ("Layer PI (a=2)",            bal_old("PI_a2")),
    ("Open loop (a=4)",           bal_old("normalL14_a4")),
    ("Decode PI",                 bal_old("dpid_PI")),
    # --- baseline and ablation conditions (results/balanced_all/black) ---
    ("CAA L14 (a=16)",            bal_all("caa", "caa_L14_a16")),
    ("ActAdd (a=16)",             bal_all("actadd", "actadd_a16")),
    ("Mean-AcT unit (s=2)",       bal_all("meanact", "meanact_unit_s2")),
    ("Linear-AcT gaussian (s=1)", bal_all("linearact", "gaussian_s1")),
    ("AurA-inject (g=4)",         bal_all("aura_inject", "inject_g4")),
    ("AurA vanilla",              bal_all("aura_vanilla", "vanilla")),
    ("ITI-C (K=48, a=8)",         bal_all("itic", "itic_K48_a8")),
    ("Decode PID",                bal_all("decode_pid", "dpid_PID")),
    ("Open loop (a=3.28)",        bal_all("normal", "normalL14_a3p28")),
    # Not a separate condition: with gen_length == block_length == 32, steps
    # 33-64 commit no tokens, so this is a numerical replicate of Decode PI
    # (independent run).
    ("Decode PI replicate (\"steps=32\")", bal_all("decode_pid_s32", "dpid_PI_s32")),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None, help="also write results as JSON")
    args = ap.parse_args()

    header = ("%-26s %6s %6s %6s %6s | %7s [%6s,%6s] | %6s %6s | perm: %6s %6s %6s %6s %7s"
              % ("condition", "target", "compar", "abstn", "inval",
                 "gap", "lo", "hi", "gap@A", "gap@BC",
                 "target", "compar", "abstn", "inval", "gap"))
    print(header)
    print("-" * len(header))

    results = {}
    for label, rels in CONDITIONS:
        paths = [os.path.join(REPO, p) for p in rels]
        for p in paths:
            if not os.path.exists(p):
                sys.exit("MISSING: %s" % p)
        rows = load_rows(paths)
        s = score(rows, strict_letter)
        p = score(rows, permissive_letter)
        lo, hi = bootstrap_ci(s["signs"])
        print("%-26s %6s %6s %6s %6s | %7s [%6s,%6s] | %6s %6s | perm: %6s %6s %6s %6s %7s"
              % (label, r3(s["target"]), r3(s["comparator"]), r3(s["abstain"]),
                 r3(s["invalid"]), r3(s["gap"]), r3(lo), r3(hi),
                 r3(s.get("gap_A", 0)), r3(s.get("gap_BC", 0)),
                 r3(p["target"]), r3(p["comparator"]), r3(p["abstain"]),
                 r3(p["invalid"]), r3(p["gap"])))
        results[label] = {
            "paths": rels,
            "n": s["n"],
            "strict": {
                "counts": s["counts"],
                "target": float(s["target"]), "comparator": float(s["comparator"]),
                "abstain": float(s["abstain"]), "invalid": float(s["invalid"]),
                "gap": float(s["gap"]),
                "gap_A": float(s.get("gap_A", 0)), "gap_BC": float(s.get("gap_BC", 0)),
                "by_pos": s["by_pos"],
                "gap_ci95": [lo, hi],
            },
            "permissive": {
                "counts": p["counts"],
                "target": float(p["target"]), "comparator": float(p["comparator"]),
                "abstain": float(p["abstain"]), "invalid": float(p["invalid"]),
                "gap": float(p["gap"]),
            },
        }

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=1)
        print("\nwrote %s" % args.json)


if __name__ == "__main__":
    main()
