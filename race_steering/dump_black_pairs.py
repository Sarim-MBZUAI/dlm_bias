#!/usr/bin/env python
"""Dump the exact CrowS-Pairs subset used to build the Black-bias steering vector.

Reuses build_direction's OWN load_crows + subset_by_terms so the pairs are
identical to what race_black.pt was built from. Writes race_steering/data/
black_pairs.csv (one row per pair: idx, stereotype=sent_more, anti=sent_less).

Run:  python race_steering/dump_black_pairs.py
"""
import csv
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_ROOT, "bias_steering"))
import build_direction as bd

TERMS = ["black", "african american", "african-american"]  # as used to build race_black.pt
OUT = os.path.join(_HERE, "data", "black_pairs.csv")


def main():
    groups = bd.load_crows(bd.DEFAULT_CROWS_URL, bd.CROWS_CACHE)
    # pick the race category key (hyphen/underscore agnostic)
    race_key = next(k for k in groups if k.replace("-", "_").lower() == "race_color")
    race_all = groups[race_key]
    sub = bd.subset_by_terms({race_key: race_all}, [t.lower() for t in TERMS])
    pairs = sub.get(race_key, [])

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["idx", "stereotype_sent_more", "antistereotype_sent_less"])
        for i, (more, less) in enumerate(pairs):
            w.writerow([i, more, less])

    print(f"race category key   : {race_key!r}")
    print(f"all race pairs      : {len(race_all)}")
    print(f"terms               : {TERMS}")
    print(f"Black-referent pairs: {len(pairs)}   -> {OUT}")
    print("\nfirst 8 pairs (stereotype  ||  anti-stereotype):")
    for i, (more, less) in enumerate(pairs[:8]):
        print(f"  {i:3d}. {more!r}\n       || {less!r}")


if __name__ == "__main__":
    main()
