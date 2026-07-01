#!/usr/bin/env python
"""Black-SPECIFIC (directional) metrics for the Black-targeted steering sweep on BBQ.

Question: does the race_black steering direction (built on the Black subset of
CrowS, injected at block L14) produce a DIRECTIONAL bias toward Black -- i.e.
does black_pick_rate on Black-referent AMBIGUOUS items rise MORE than
nonblack_pick_rate as alpha grows -- or is it (like the generic race_color
direction) just a non-directional abstention suppressor that pushes picks off
"Unknown" onto BOTH named options roughly equally?

Pipeline
  1. Re-load the SAME BBQ 1000-item sample (seed 42, same loader as bbq_eval.py)
     and build a map (example_id, question_index) -> {option_index: group_tag}.
  2. Flag "Black-referent" Race_ethnicity items: exactly ONE of the 3 options
     carries a Black group tag (the named, non-Unknown option). Record that idx.
  3. On Black-referent AMBIGUOUS items (gold == Unknown), per run, compute:
       black_pick / nonblack_pick / abstention / no_answer rates,
       the neg-vs-nonneg polarity split of black_pick,
       and flips vs the clean baseline (clean==Unknown -> steered Black/non-Black).
  4. Do the SAME on the generic race_color runs, and print race_black vs
     race_color side by side.
  5. Figure: black_pick & nonblack_pick & abstention vs alpha (steering only).

Reads only the saved JSON/JSONL result files -- no GPU, no model. The runs all
used the same seed-42 sample, so items align by (example_id, question_index).
"""
import json
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_ROOT, "eval"))
import bbq_eval  # same BBQ loader / get_answer_info used by the eval

# --------------------------------------------------------------------------- #
# Config: result files. black runs live in THIS worktree; clean + race_color
# comparators live in the sibling main-repo eval/results.
# --------------------------------------------------------------------------- #
BLACK_DIR = os.path.join(_HERE, "results")
MAIN_RESULTS = "/home/lukas/users/shashmi/dlm_bias/eval/results"

CLEAN = os.path.join(MAIN_RESULTS, "bbq_clean_samples.jsonl")
BLACK_RUNS = {a: os.path.join(BLACK_DIR, f"bbq_L14_race_black_a{a}_samples.jsonl")
              for a in (8, 16, 32)}
COLOR_RUNS = {a: os.path.join(MAIN_RESULTS, f"bbq_L14_race_color_a{a}_samples.jsonl")
              for a in (8, 16, 32)}
BLACK_METRICS = {a: os.path.join(BLACK_DIR, f"bbq_L14_race_black_a{a}.json")
                 for a in (8, 16, 32)}
COLOR_METRICS = {a: os.path.join(MAIN_RESULTS, f"bbq_L14_race_color_a{a}.json")
                 for a in (8, 16, 32)}
CLEAN_METRICS = os.path.join(MAIN_RESULTS, "bbq_clean.json")

# Case-insensitive Black group tags in BBQ Race_ethnicity answer_info.
# 'African' pairs with 'Black' ("The African one" / "The Black student"), so
# it denotes Black/African people here (NOT an African-nationality distractor).
BLACK_TAGS = {"black", "african american", "f-black", "m-black", "african"}

FIG_PATH = os.path.join(_HERE, "figs", "black_steering.png")


# --------------------------------------------------------------------------- #
# Step 1-2: build Black-referent map from the seed-42 BBQ sample.
# --------------------------------------------------------------------------- #
def build_black_map():
    rows = bbq_eval.load_bbq("nyu-mll/BBQ (jsonl)", 42, 1000, None)
    black_idx = {}          # (ex_id, q_idx) -> option index that is Black
    seen_black_tags = set()
    n_race = 0
    for r in rows:
        if r["category"] != "Race_ethnicity":
            continue
        n_race += 1
        black_opts = []
        for k in range(3):
            pair = bbq_eval.get_answer_info(r, k)
            if not pair:
                continue
            tag = str(pair[-1]).strip().lower()
            if tag in BLACK_TAGS:
                black_opts.append(k)
                seen_black_tags.add(tag)
        # Black-referent iff EXACTLY ONE option is Black-tagged.
        if len(black_opts) == 1:
            key = (int(r.get("example_id", -1)), str(r.get("question_index", "")))
            black_idx[key] = black_opts[0]
    return black_idx, seen_black_tags, n_race


def load_samples(path):
    out = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            out[(int(d["example_id"]), str(d["question_index"]))] = d
    return out


# --------------------------------------------------------------------------- #
# Step 2-3: Black-referent metrics on AMBIGUOUS items (gold == Unknown).
# --------------------------------------------------------------------------- #
def black_metrics(samples, black_idx, clean_samples=None):
    """samples/clean_samples: (key)->row. Returns metric dict over Black-referent
    ambiguous items. pred_index==black_idx -> black pick; ==unknown_idx ->
    abstention; None -> no_answer; else -> nonblack pick."""
    n = 0
    black = nonblack = abst = noans = 0
    black_neg = black_nonneg = 0
    neg_n = nonneg_n = 0
    flip_to_black = flip_to_nonblack = flip_stay_unk = clean_unk_n = 0
    for key, bidx in black_idx.items():
        row = samples.get(key)
        if row is None:
            continue
        if row["context_condition"] != "ambig":
            continue  # ambiguous items only (gold == Unknown)
        n += 1
        pred = row["pred_index"]
        unk = row["unknown_idx"]
        neg = row["polarity"] == "neg"
        if neg:
            neg_n += 1
        else:
            nonneg_n += 1
        if pred is None:
            noans += 1
        elif pred == bidx:
            black += 1
            if neg:
                black_neg += 1
            else:
                black_nonneg += 1
        elif pred == unk:
            abst += 1
        else:
            nonblack += 1
        # flips vs clean: only items the CLEAN run answered Unknown.
        if clean_samples is not None:
            crow = clean_samples.get(key)
            if crow is not None and crow["pred_index"] == crow["unknown_idx"]:
                clean_unk_n += 1
                if pred == bidx:
                    flip_to_black += 1
                elif pred == unk:
                    flip_stay_unk += 1
                elif pred is not None:
                    flip_to_nonblack += 1
    def rate(x):
        return x / n if n else 0.0
    return {
        "n": n,
        "black_pick_rate": rate(black),
        "nonblack_pick_rate": rate(nonblack),
        "abstention_rate": rate(abst),
        "no_answer_rate": rate(noans),
        "black_neg_rate": (black_neg / neg_n) if neg_n else 0.0,
        "black_nonneg_rate": (black_nonneg / nonneg_n) if nonneg_n else 0.0,
        "neg_n": neg_n, "nonneg_n": nonneg_n,
        "clean_unk_n": clean_unk_n,
        "flip_to_black": flip_to_black,
        "flip_to_nonblack": flip_to_nonblack,
        "flip_stay_unk": flip_stay_unk,
    }


def acc_disambig(metrics_json):
    with open(metrics_json) as f:
        return json.load(f)["overall"]["acc_disambig"]


# --------------------------------------------------------------------------- #
# Report.
# --------------------------------------------------------------------------- #
def main():
    black_idx, seen_tags, n_race = build_black_map()
    print("=" * 78)
    print("BLACK-SPECIFIC (DIRECTIONAL) BBQ ANALYSIS  --  race_black vs race_color")
    print("=" * 78)
    print(f"Race_ethnicity items in seed-42 sample : {n_race}")
    print(f"Black group tags used (case-insensitive): {sorted(BLACK_TAGS)}")
    print(f"  observed in sample                    : {sorted(seen_tags)}")
    print(f"Black-referent items (exactly 1 Black opt): {len(black_idx)}")

    clean = load_samples(CLEAN)
    # n_black_items on AMBIGUOUS subset (the metric denominator).
    amb_keys = [k for k in black_idx if clean.get(k) and clean[k]["context_condition"] == "ambig"]
    print(f"  ... of which AMBIGUOUS (gold=Unknown)  : {len(amb_keys)}  <- metric n")

    # a=0 clean baseline row.
    runs = {"race_black": {0: clean}, "race_color": {0: clean}}
    for a in (8, 16, 32):
        runs["race_black"][a] = load_samples(BLACK_RUNS[a])
        runs["race_color"][a] = load_samples(COLOR_RUNS[a])

    m = {fam: {a: black_metrics(runs[fam][a], black_idx, clean) for a in (0, 8, 16, 32)}
         for fam in ("race_black", "race_color")}

    accd = {"race_black": {0: acc_disambig(CLEAN_METRICS)},
            "race_color": {0: acc_disambig(CLEAN_METRICS)}}
    for a in (8, 16, 32):
        accd["race_black"][a] = acc_disambig(BLACK_METRICS[a])
        accd["race_color"][a] = acc_disambig(COLOR_METRICS[a])

    # ---- Main comparison table ----
    print("\n" + "-" * 78)
    print("Black-referent AMBIGUOUS items  (rates over n={})".format(m["race_black"][0]["n"]))
    print("-" * 78)
    hdr = f"{'dir':11s} {'a':>3s} {'black':>7s} {'nonblk':>7s} {'abst':>7s} {'noans':>6s} {'blk_neg':>8s} {'blk_nn':>7s} {'accDis':>7s}"
    print(hdr)
    for fam in ("race_black", "race_color"):
        for a in (0, 8, 16, 32):
            r = m[fam][a]
            print(f"{fam:11s} {a:3d} {r['black_pick_rate']:7.3f} {r['nonblack_pick_rate']:7.3f} "
                  f"{r['abstention_rate']:7.3f} {r['no_answer_rate']:6.3f} "
                  f"{r['black_neg_rate']:8.3f} {r['black_nonneg_rate']:7.3f} {accd[fam][a]:7.3f}")
        if fam == "race_black":
            print()

    # ---- Flips vs clean ----
    print("\n" + "-" * 78)
    print("FLIPS vs clean baseline (items clean answered Unknown; n_clean_unk={})".format(
        m["race_black"][8]["clean_unk_n"]))
    print("-" * 78)
    print(f"{'dir':11s} {'a':>3s} {'->Black':>8s} {'->nonBlk':>9s} {'stayUnk':>8s}")
    for fam in ("race_black", "race_color"):
        for a in (8, 16, 32):
            r = m[fam][a]
            print(f"{fam:11s} {a:3d} {r['flip_to_black']:8d} {r['flip_to_nonblack']:9d} {r['flip_stay_unk']:8d}")
        if fam == "race_black":
            print()

    # ---- Directionality summary ----
    print("\n" + "-" * 78)
    print("DIRECTIONALITY  (delta from a=0; black_pick minus nonblack_pick change)")
    print("-" * 78)
    print(f"{'dir':11s} {'a':>3s} {'d_black':>8s} {'d_nonblk':>9s} {'d_gap':>7s}")
    for fam in ("race_black", "race_color"):
        b0 = m[fam][0]["black_pick_rate"]
        nb0 = m[fam][0]["nonblack_pick_rate"]
        for a in (8, 16, 32):
            db = m[fam][a]["black_pick_rate"] - b0
            dnb = m[fam][a]["nonblack_pick_rate"] - nb0
            print(f"{fam:11s} {a:3d} {db:+8.3f} {dnb:+9.3f} {db - dnb:+7.3f}")
        if fam == "race_black":
            print()

    make_figure(m)
    print(f"\nSaved figure -> {FIG_PATH}")
    return m


# --------------------------------------------------------------------------- #
# Step 4: figure (steering only, race_black direction).
# --------------------------------------------------------------------------- #
def make_figure(m):
    os.makedirs(os.path.dirname(FIG_PATH), exist_ok=True)
    alphas = [0, 8, 16, 32]
    fam = "race_black"
    black = [m[fam][a]["black_pick_rate"] for a in alphas]
    nonblack = [m[fam][a]["nonblack_pick_rate"] for a in alphas]
    abst = [m[fam][a]["abstention_rate"] for a in alphas]

    fig, ax = plt.subplots(figsize=(7, 4.6))
    ax.plot(alphas, black, "o-", color="#c0392b", lw=2.2, ms=8, label="Black pick")
    ax.plot(alphas, nonblack, "s-", color="#2c7fb8", lw=2.2, ms=7, label="non-Black pick")
    ax.plot(alphas, abst, "^--", color="#7f8c8d", lw=1.8, ms=7, label="abstention (Unknown)")

    ax.set_xlabel("steering strength  alpha  (race_black dir, block L14)")
    ax.set_ylabel("rate on Black-referent ambiguous items")
    ax.set_title("Black-targeted steering: does Black-pick outrun non-Black-pick?")
    ax.set_xticks(alphas)
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.axvline(0, color="k", lw=0.8, alpha=0.4)
    ax.annotate("clean\nbaseline", xy=(0, 0.02), xytext=(1.5, 0.08),
                fontsize=8, color="k", alpha=0.7)
    ax.legend(loc="upper left", frameon=True)
    # annotate the last point rates for readability
    for xs, ys, c in ((alphas, black, "#c0392b"), (alphas, nonblack, "#2c7fb8")):
        ax.annotate(f"{ys[-1]:.2f}", xy=(xs[-1], ys[-1]),
                    xytext=(xs[-1] - 3.5, ys[-1] + 0.03), fontsize=8, color=c)
    fig.tight_layout()
    fig.savefig(FIG_PATH, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
