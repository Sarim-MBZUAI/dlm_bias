#!/usr/bin/env python
"""Build the Black-bias STEERING dataset from REAL benchmark data only.

Combines two real minimal-pair sources (NO synthetic/templated pairs):
  * CrowS-Pairs  -- race_color category, Black-referent subset (reuses
    build_direction.load_crows + subset_by_terms so the CrowS side is IDENTICAL
    to what the original race_black.pt was built from).
  * StereoSet dev.json (race domain) -- Sub-Saharan-African / Black targets only.
    intrasentence: the labeled 'stereotype' vs 'anti-stereotype' completions of
      the same context (the sentence field is already the fully-filled sentence).
    intersentence: the 'stereotype' vs 'anti-stereotype' continuation of the same
      context.

Writes race_steering/data/:
  black_pairs.csv  (idx, source, stereotype, anti_stereotype)  -- human-readable
  black_pairs.json (list of {"stereotype":..., "anti_stereotype":...}) -- for
                   build_direction.py --source json

Run:  python race_steering/build_black_data.py
"""
import csv
import json
import os
import sys
import urllib.request

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_ROOT, "bias_steering"))
import build_direction as bd  # noqa: E402

# --- CrowS side (identical to the original race_black.pt build) -------------- #
CROWS_TERMS = ["black", "african american", "african-american"]

# --- StereoSet side ---------------------------------------------------------- #
STEREOSET_URL = (
    "https://raw.githubusercontent.com/moinnadeem/StereoSet/master/data/dev.json"
)
STEREOSET_CACHE = os.path.join(_HERE, ".stereoset_cache", "dev.json")
# Sub-Saharan-African / Black-referent race targets. StereoSet's race targets are
# ethnicity/nationality strings; we DELIBERATELY include only clearly Sub-Saharan
# African / "Black African" referents and EXCLUDE North-African/Arab (Morocco,
# Arab), Middle-Eastern, Asian, European, and Latin-American nationalities, which
# are not "Black/African" referents. "African" is the explicit pan-ethnic target.
STEREOSET_BLACK_TARGETS = [
    "African",      # explicit pan-African / Black-African referent
    "Ethiopian", "Ethiopia",
    "Somalia",
    "Ghanaian",
    "Cameroon",
    "Cape Verde",
    "Eritrean", "Eriteria",  # both spellings present in the data
    "Sierra Leon",
]

CSV_OUT = os.path.join(_HERE, "data", "black_pairs.csv")
JSON_OUT = os.path.join(_HERE, "data", "black_pairs.json")


def load_crows_black():
    """Reuse build_direction's loader to get the Black-referent race_color pairs."""
    groups = bd.load_crows(bd.DEFAULT_CROWS_URL, bd.CROWS_CACHE)
    race_key = next(k for k in groups if k.replace("-", "_").lower() == "race_color")
    sub = bd.subset_by_terms({race_key: groups[race_key]}, CROWS_TERMS)
    return sub.get(race_key, [])


def load_stereoset():
    """Download (cache) StereoSet dev.json; return (data_dict, all_race_targets)."""
    if not (os.path.exists(STEREOSET_CACHE) and os.path.getsize(STEREOSET_CACHE) > 0):
        os.makedirs(os.path.dirname(STEREOSET_CACHE), exist_ok=True)
        urllib.request.urlretrieve(STEREOSET_URL, STEREOSET_CACHE)
    d = json.load(open(STEREOSET_CACHE))["data"]
    race_targets = sorted({
        e["target"]
        for split in ("intrasentence", "intersentence")
        for e in d[split]
        if e["bias_type"] == "race"
    })
    return d, race_targets


def _labeled(sentences, label):
    """Return the sentence string for the given gold_label, or None."""
    for s in sentences:
        if s["gold_label"] == label:
            return s["sentence"].strip()
    return None


def stereoset_black_pairs(data):
    """Extract (stereotype, anti_stereotype) pairs for the selected Black targets.

    Both intrasentence and intersentence: pair the labeled stereotype sentence
    with the labeled anti-stereotype sentence of the same example.
    """
    wanted = set(STEREOSET_BLACK_TARGETS)
    pairs = []
    for split in ("intrasentence", "intersentence"):
        for e in data[split]:
            if e["bias_type"] != "race" or e["target"] not in wanted:
                continue
            ster = _labeled(e["sentences"], "stereotype")
            anti = _labeled(e["sentences"], "anti-stereotype")
            if ster and anti:
                pairs.append((ster, anti))
    return pairs


def main():
    print("=" * 70)
    print("Black-bias STEERING dataset builder (REAL data only: CrowS + StereoSet)")
    print("=" * 70)

    # ---- CrowS ---- #
    crows = load_crows_black()
    print(f"\nCrowS-Pairs (race_color, terms={CROWS_TERMS}): {len(crows)} pairs")

    # ---- StereoSet ---- #
    data, all_race_targets = load_stereoset()
    print(f"\nStereoSet cache: {STEREOSET_CACHE}")
    print(f"StereoSet race domain: {len(all_race_targets)} DISTINCT race targets:")
    print(f"  {all_race_targets}")
    print(f"\nSELECTED Black/African subset ({len(STEREOSET_BLACK_TARGETS)} targets):")
    print(f"  {STEREOSET_BLACK_TARGETS}")
    print("  rationale: only clearly Sub-Saharan-African / Black-African referents;"
          " EXCLUDES North-African/Arab, Middle-Eastern, Asian, European, and"
          " Latin-American nationalities (not Black/African referents).")
    stereo = stereoset_black_pairs(data)
    print(f"\nStereoSet Black pairs extracted (pre-dedup): {len(stereo)}")

    # ---- Combine + dedup (exact) ---- #
    seen = set()
    rows = []  # (source, stereotype, anti)
    for src, pairs in (("crows", crows), ("stereoset", stereo)):
        for ster, anti in pairs:
            key = (ster, anti)
            if key in seen:
                continue
            seen.add(key)
            rows.append((src, ster, anti))

    per_source = {}
    for src, _, _ in rows:
        per_source[src] = per_source.get(src, 0) + 1

    # ---- Write CSV + JSON ---- #
    os.makedirs(os.path.dirname(CSV_OUT), exist_ok=True)
    with open(CSV_OUT, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["idx", "source", "stereotype", "anti_stereotype"])
        for i, (src, ster, anti) in enumerate(rows):
            w.writerow([i, src, ster, anti])
    with open(JSON_OUT, "w") as f:
        json.dump(
            [{"stereotype": ster, "anti_stereotype": anti} for _, ster, anti in rows],
            f, indent=2,
        )

    print("\n" + "-" * 70)
    print(f"FINAL combined total : {len(rows)} pairs (exact-dedup applied)")
    print(f"  per-source         : {per_source}")
    print(f"  csv  -> {CSV_OUT}")
    print(f"  json -> {JSON_OUT}")
    print("-" * 70)


if __name__ == "__main__":
    main()
