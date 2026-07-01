#!/usr/bin/env python
"""Black-referent (DIRECTIONAL) BBQ metrics for the ITEM-ANCHORED (answer-text)
steering vector -- the payoff analysis.

Question: does the coherent (split-half 0.98) "prefer the Black option"
answer-text-anchored direction, injected at block L14, produce a DIRECTIONAL
bias toward Black -- i.e. does black_pick_rate on Black-referent AMBIGUOUS items
rise MORE than nonblack_pick_rate as alpha grows (a real AIMing lever) -- or is
it (like the group mean-diff race_black direction, and every prior vector) just a
non-directional abstention suppressor that pushes picks off "Unknown" onto BOTH
named options roughly equally?

Reference: adapted from race_steering/black_analysis.py. Same Black-referent
logic, same seed-42 BBQ sample, same n=37 AMBIGUOUS Black-referent denominator.

Pipeline
  1. Re-load the SAME BBQ 1000-item sample (seed 42, same loader as bbq_eval.py)
     and build a map (example_id, question_index) -> the option index that is
     Black-tagged, for "Black-referent" Race_ethnicity items (exactly ONE of the
     3 options carries a Black group tag).
  2. On Black-referent AMBIGUOUS items (gold == Unknown), per run, compute:
       black_pick / nonblack_pick / abstention / no_answer rates,
       the neg-vs-nonneg polarity split of black_pick,
       and flips vs the clean baseline (clean==Unknown -> steered Black/non-Black).
  3. Do the SAME on the group mean-diff race_black comparison runs and print
     anchored vs race_black (group mean-diff) side by side + peak directional gap.
  4. Figure: black_pick & nonblack_pick & abstention vs alpha (anchored only),
     with an acc_disambig competence line and competence-cliff shading if any.

Reads only the saved JSON/JSONL result files -- no GPU, no model.
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)          # dlm_bias-anchored worktree
sys.path.insert(0, os.path.join(_ROOT, "eval"))
import bbq_eval  # same BBQ loader / get_answer_info used by the eval

# --------------------------------------------------------------------------- #
# Config. Anchored runs live in THIS worktree's directional_steering/results.
# Clean baseline + the group mean-diff race_black comparator live in the sibling
# MAIN repo (dlm_bias), not in this worktree.
# --------------------------------------------------------------------------- #
ANCH_DIR = os.path.join(_HERE, "results")
MAIN = "/home/lukas/users/shashmi/dlm_bias"
MAIN_EVAL = os.path.join(MAIN, "eval", "results")
RACE_DIR = os.path.join(MAIN, "race_steering", "results")

ANCH_ALPHAS = (1, 2, 3, 4, 6, 8)             # item-anchored (answer-text), this worktree
RACE_ALPHAS = (4, 8, 12, 16, 24, 32)         # group mean-diff race_black, main repo

CLEAN = os.path.join(MAIN_EVAL, "bbq_clean_samples.jsonl")
CLEAN_METRICS = os.path.join(MAIN_EVAL, "bbq_clean.json")

ANCH_RUNS = {a: os.path.join(ANCH_DIR, f"bbq_L14_anchored_a{a}_samples.jsonl") for a in ANCH_ALPHAS}
ANCH_METRICS = {a: os.path.join(ANCH_DIR, f"bbq_L14_anchored_a{a}.json") for a in ANCH_ALPHAS}
RACE_RUNS = {a: os.path.join(RACE_DIR, f"bbq_L14_race_black_a{a}_samples.jsonl") for a in RACE_ALPHAS}
RACE_METRICS = {a: os.path.join(RACE_DIR, f"bbq_L14_race_black_a{a}.json") for a in RACE_ALPHAS}

# Case-insensitive Black group tags -- identical to race_steering/black_analysis.py.
BLACK_TAGS = {"black", "african american", "f-black", "m-black", "african"}

FIG_PATH = os.path.join(_HERE, "figs", "anchored_steering.png")


# --------------------------------------------------------------------------- #
# Step 1: build Black-referent map from the seed-42 BBQ sample.
# --------------------------------------------------------------------------- #
def build_black_map():
    rows = bbq_eval.load_bbq("nyu-mll/BBQ (jsonl)", 42, 1000, None)
    black_idx = {}
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
        if len(black_opts) == 1:            # Black-referent iff EXACTLY ONE Black opt
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
# Step 2: Black-referent metrics on AMBIGUOUS items (gold == Unknown).
# --------------------------------------------------------------------------- #
def black_metrics(samples, black_idx, clean_samples=None):
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
            continue                        # ambiguous items only (gold == Unknown)
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
        if clean_samples is not None:       # flips: only items CLEAN answered Unknown
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
    print("=" * 82)
    print("ITEM-ANCHORED (answer-text) DIRECTIONAL BBQ ANALYSIS")
    print("   anchored (split-half 0.98)  vs  group mean-diff race_black")
    print("=" * 82)
    print(f"Race_ethnicity items in seed-42 sample     : {n_race}")
    print(f"Black group tags used (case-insensitive)   : {sorted(BLACK_TAGS)}")
    print(f"  observed in sample                       : {sorted(seen_tags)}")
    print(f"Black-referent items (exactly 1 Black opt) : {len(black_idx)}")

    clean = load_samples(CLEAN)
    amb_keys = [k for k in black_idx if clean.get(k) and clean[k]["context_condition"] == "ambig"]
    print(f"  ... of which AMBIGUOUS (gold=Unknown)    : {len(amb_keys)}  <- metric n (n=37 caveat)")

    fam_alphas = {"anchored": (0,) + tuple(ANCH_ALPHAS),
                  "race_black": (0,) + tuple(RACE_ALPHAS)}
    fam_runs = {"anchored": ANCH_RUNS, "race_black": RACE_RUNS}
    fam_metrics = {"anchored": ANCH_METRICS, "race_black": RACE_METRICS}

    runs = {fam: {0: clean} for fam in fam_alphas}
    for fam in fam_alphas:
        for a in fam_alphas[fam]:
            if a == 0:
                continue
            runs[fam][a] = load_samples(fam_runs[fam][a])

    m = {fam: {a: black_metrics(runs[fam][a], black_idx, clean) for a in fam_alphas[fam]}
         for fam in fam_alphas}

    accd = {fam: {0: acc_disambig(CLEAN_METRICS)} for fam in fam_alphas}
    for fam in fam_alphas:
        for a in fam_alphas[fam]:
            if a == 0:
                continue
            accd[fam][a] = acc_disambig(fam_metrics[fam][a])

    # ---- Anchored per-alpha Black-referent table ----
    print("\n" + "-" * 82)
    print("ANCHORED (answer-text) -- Black-referent AMBIGUOUS items (rates over n={})".format(m["anchored"][0]["n"]))
    print("-" * 82)
    hdr = (f"{'a':>3s} {'black':>7s} {'nonblk':>7s} {'abst':>7s} {'noans':>6s} "
           f"{'blk_neg':>8s} {'blk_nn':>7s} {'accDis':>7s}")
    print(hdr)
    for a in fam_alphas["anchored"]:
        r = m["anchored"][a]
        print(f"{a:3d} {r['black_pick_rate']:7.3f} {r['nonblack_pick_rate']:7.3f} "
              f"{r['abstention_rate']:7.3f} {r['no_answer_rate']:6.3f} "
              f"{r['black_neg_rate']:8.3f} {r['black_nonneg_rate']:7.3f} {accd['anchored'][a]:7.3f}")

    # ---- Anchored directionality (delta from a=0) ----
    print("\n" + "-" * 82)
    print("ANCHORED DIRECTIONALITY (delta from a=0; gap = d_black - d_nonblk)")
    print("-" * 82)
    print(f"{'a':>3s} {'d_black':>8s} {'d_nonblk':>9s} {'d_gap':>7s}")
    b0 = m["anchored"][0]["black_pick_rate"]
    nb0 = m["anchored"][0]["nonblack_pick_rate"]
    anch_gaps = {}
    for a in fam_alphas["anchored"]:
        if a == 0:
            continue
        db = m["anchored"][a]["black_pick_rate"] - b0
        dnb = m["anchored"][a]["nonblack_pick_rate"] - nb0
        anch_gaps[a] = db - dnb
        print(f"{a:3d} {db:+8.3f} {dnb:+9.3f} {db - dnb:+7.3f}")

    # ---- Flips at each anchored alpha ----
    print("\n" + "-" * 82)
    print("ANCHORED FLIPS vs clean baseline (items clean answered Unknown; n_clean_unk={})".format(
        m["anchored"][ANCH_ALPHAS[0]]["clean_unk_n"]))
    print("-" * 82)
    print(f"{'a':>3s} {'->Black':>8s} {'->nonBlk':>9s} {'stayUnk':>8s}")
    for a in fam_alphas["anchored"]:
        if a == 0:
            continue
        r = m["anchored"][a]
        print(f"{a:3d} {r['flip_to_black']:8d} {r['flip_to_nonblack']:9d} {r['flip_stay_unk']:8d}")

    # ---- race_black (group mean-diff) directionality, for comparison ----
    print("\n" + "-" * 82)
    print("GROUP MEAN-DIFF race_black DIRECTIONALITY (comparator; delta from a=0)")
    print("-" * 82)
    print(f"{'a':>3s} {'d_black':>8s} {'d_nonblk':>9s} {'d_gap':>7s} {'accDis':>7s}")
    rb0 = m["race_black"][0]["black_pick_rate"]
    rnb0 = m["race_black"][0]["nonblack_pick_rate"]
    race_gaps = {}
    for a in fam_alphas["race_black"]:
        if a == 0:
            continue
        db = m["race_black"][a]["black_pick_rate"] - rb0
        dnb = m["race_black"][a]["nonblack_pick_rate"] - rnb0
        race_gaps[a] = db - dnb
        print(f"{a:3d} {db:+8.3f} {dnb:+9.3f} {db - dnb:+7.3f} {accd['race_black'][a]:7.3f}")

    # ---- Peak directional gap side-by-side ----
    # Competent range only for the "honest" peak: acc_disambig must not have cratered.
    def peak(gaps, accd_fam, min_acc=0.30):
        cand = {a: g for a, g in gaps.items() if accd_fam[a] >= min_acc}
        pool = cand if cand else gaps
        a_best = max(pool, key=pool.get)
        return a_best, pool[a_best]

    a_anch, g_anch = max(anch_gaps, key=anch_gaps.get), max(anch_gaps.values())
    a_anch_c, g_anch_c = peak(anch_gaps, accd["anchored"])
    a_race, g_race = max(race_gaps, key=race_gaps.get), max(race_gaps.values())
    a_race_c, g_race_c = peak(race_gaps, accd["race_black"])

    print("\n" + "=" * 82)
    print("PEAK DIRECTIONAL GAP  (d_black - d_nonblk):  ANCHORED  vs  GROUP MEAN-DIFF")
    print("=" * 82)
    print(f"{'direction':22s} {'peak_gap':>9s} {'@alpha':>7s} {'accDis@peak':>12s}")
    print(f"{'anchored (any a)':22s} {g_anch:+9.3f} {a_anch:>7d} {accd['anchored'][a_anch]:12.3f}")
    print(f"{'anchored (accDis>=.30)':22s} {g_anch_c:+9.3f} {a_anch_c:>7d} {accd['anchored'][a_anch_c]:12.3f}")
    print(f"{'race_black (any a)':22s} {g_race:+9.3f} {a_race:>7d} {accd['race_black'][a_race]:12.3f}")
    print(f"{'race_black (accDis>=.30)':22s} {g_race_c:+9.3f} {a_race_c:>7d} {accd['race_black'][a_race_c]:12.3f}")

    make_figure(m, accd)
    print(f"\nSaved figure -> {FIG_PATH}")
    return m


# --------------------------------------------------------------------------- #
# Figure (anchored direction only).
# --------------------------------------------------------------------------- #
def make_figure(m, accd):
    os.makedirs(os.path.dirname(FIG_PATH), exist_ok=True)
    fam = "anchored"
    alphas = [0] + list(ANCH_ALPHAS)
    black = [m[fam][a]["black_pick_rate"] for a in alphas]
    nonblack = [m[fam][a]["nonblack_pick_rate"] for a in alphas]
    abst = [m[fam][a]["abstention_rate"] for a in alphas]
    accd_series = [accd[fam][a] for a in alphas]

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.plot(alphas, black, "o-", color="#c0392b", lw=2.4, ms=8, label="Black pick", zorder=3)
    ax.plot(alphas, nonblack, "s-", color="#2c7fb8", lw=2.4, ms=7, label="non-Black pick", zorder=3)
    ax.plot(alphas, abst, "^--", color="#7f8c8d", lw=1.8, ms=7, label="abstention (Unknown)", zorder=2)
    ax.plot(alphas, accd_series, "d:", color="#27ae60", lw=1.6, ms=6,
            label="acc_disambig (competence)", alpha=0.85, zorder=1)

    cliff = next((a for a in ANCH_ALPHAS if accd[fam][a] < 0.5), None)
    if cliff is not None:
        ax.axvspan(cliff - 0.5, alphas[-1] + 0.5, color="#e74c3c", alpha=0.06, zorder=0)
        ax.annotate("competence\ncliff (acc<0.5)", xy=(cliff, 0.45),
                    xytext=(cliff - 2.0, 0.62), fontsize=8, color="#c0392b",
                    arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1))

    ax.set_xlabel("steering strength  alpha  (item-anchored answer-text dir, block L14)")
    ax.set_ylabel("rate on Black-referent ambiguous items  (n=37)")
    ax.set_title("Item-anchored (answer-text, split-half 0.98) steering:\n"
                 "does Black-pick outrun non-Black-pick?")
    ax.set_xticks(alphas)
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.axvline(0, color="k", lw=0.8, alpha=0.4)
    ax.annotate("clean\nbaseline", xy=(0, 0.02), xytext=(0.1, 0.10), fontsize=8, color="k", alpha=0.7)
    ax.legend(loc="center right", frameon=True, fontsize=9)
    for a, yb, ynb in zip(alphas, black, nonblack):
        if a == 0:
            continue
        ax.annotate(f"{yb:.2f}", xy=(a, yb), xytext=(a - 0.12, yb + 0.03), fontsize=7.5, color="#c0392b")
        ax.annotate(f"{ynb:.2f}", xy=(a, ynb), xytext=(a - 0.12, ynb - 0.055), fontsize=7.5, color="#2c7fb8")
    fig.tight_layout()
    fig.savefig(FIG_PATH, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
