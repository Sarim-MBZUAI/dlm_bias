#!/usr/bin/env python
"""E1 analysis: rescore the closed-loop conditions on the FULL Black-referent
ambiguous set (n=1600) with uncertainty.

For each condition (clean, open a4, open a8, clamp C60, cmom C60) we compute on
the Black-referent AMBIGUOUS items: black-pick / non-black / abstention rates,
acc_disambig, and the directional gap  d_gap = Δblack − Δnonblack  vs clean, with:
  * a BOOTSTRAP 95% CI on d_gap (paired item resampling, seeded, n_boot=10000);
  * an exact McNEMAR test (clean vs steered) on the per-item black-pick change.

The per-item Black-referent classification reuses the SAME logic as
directional_steering/anchored_analysis.black_metrics (pred==black_idx → black;
pred==unknown → abstain; pred is None → no_answer; else non-black).

OFFLINE VALIDATION: before trusting the big-run analysis, we recompute d_gap on
the historical seed-42 n=37 items from the EXISTING result files
(bbq_L14_anchored_a8 vs bbq_clean) and check it reproduces +0.135
(black 0.568 / non-black 0.351).  PASS/FAIL is printed.

CPU only.  Run AFTER the E1 runs (see run_e1.sh):
  /home/lukas/miniconda3/envs/sarim_awm/bin/python experiments/e1_analyze.py
Validation-only (no big-run files needed):
  ... experiments/e1_analyze.py --validate-only
"""
import argparse
import json
import math
import os
import random
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_ROOT, "eval"))
sys.path.insert(0, os.path.join(_ROOT, "directional_steering"))
import bbq_eval
import anchored_analysis as aa  # build_black_map / black_metrics / load_samples

BLACK_TAGS = aa.BLACK_TAGS
EVAL_SET = os.path.join(_HERE, "data", "black_referent_ambig_eval.jsonl")
RESULTS = os.path.join(_HERE, "results")

# E1 conditions -> result-file stem (written by run_e1.sh into experiments/results/).
E1_CONDITIONS = [
    ("clean",      "e1_clean"),
    ("open a=4",   "e1_open_a4"),
    ("open a=8",   "e1_open_a8"),
    ("clamp C60",  "e1_clamp_c60"),
    ("cmom C60",   "e1_cmom_c60"),
]


# --------------------------------------------------------------------------- #
# Black-referent map from a jsonl of full BBQ rows (the E1 eval set).
# --------------------------------------------------------------------------- #
def black_map_from_jsonl(path):
    """key (example_id, question_index) -> Black option index, for the eval rows."""
    bmap = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            opts = [k for k in range(3)
                    if (bbq_eval.get_answer_info(r, k)
                        and str(bbq_eval.get_answer_info(r, k)[-1]).strip().lower() in BLACK_TAGS)]
            if len(opts) == 1:
                key = (int(r.get("example_id", -1)), str(r.get("question_index", "")))
                bmap[key] = opts[0]
    return bmap


def per_item_indicators(samples, black_idx):
    """key -> (is_black, is_nonblack, is_abst) for Black-referent AMBIGUOUS items
    present in `samples`. Same classification as anchored_analysis.black_metrics."""
    ind = {}
    for key, bidx in black_idx.items():
        row = samples.get(key)
        if row is None or row["context_condition"] != "ambig":
            continue
        pred, unk = row["pred_index"], row["unknown_idx"]
        is_black = int(pred == bidx)
        is_abst = int(pred == unk)
        is_none = pred is None
        is_nonblack = int((not is_black) and (not is_abst) and (not is_none))
        ind[key] = (is_black, is_nonblack, is_abst)
    return ind


def d_gap_from_rates(clean_ind, steer_ind, keys):
    """d_gap = (Δblack) − (Δnonblack) over the given keys (paired)."""
    n = len(keys)
    if n == 0:
        return 0.0
    cb = sum(clean_ind[k][0] for k in keys) / n
    cnb = sum(clean_ind[k][1] for k in keys) / n
    sb = sum(steer_ind[k][0] for k in keys) / n
    snb = sum(steer_ind[k][1] for k in keys) / n
    return (sb - cb) - (snb - cnb)


def bootstrap_ci(clean_ind, steer_ind, n_boot=10000, seed=0):
    keys = [k for k in steer_ind if k in clean_ind]
    point = d_gap_from_rates(clean_ind, steer_ind, keys)
    rng = random.Random(seed)
    n = len(keys)
    boots = []
    for _ in range(n_boot):
        sample = [keys[rng.randrange(n)] for _ in range(n)]
        boots.append(d_gap_from_rates(clean_ind, steer_ind, sample))
    boots.sort()
    lo = boots[int(0.025 * n_boot)]
    hi = boots[int(0.975 * n_boot)]
    return point, lo, hi, n


def mcnemar_exact(clean_ind, steer_ind):
    """Exact two-sided McNemar on the black-pick indicator (paired). Returns
    (b, c, p) where b = clean0→steer1 (gained black), c = clean1→steer0."""
    b = c = 0
    for k in steer_ind:
        if k not in clean_ind:
            continue
        c0, s0 = clean_ind[k][0], steer_ind[k][0]
        if c0 == 0 and s0 == 1:
            b += 1
        elif c0 == 1 and s0 == 0:
            c += 1
    n = b + c
    if n == 0:
        return b, c, 1.0
    lo = min(b, c)
    tail = sum(math.comb(n, i) for i in range(lo + 1)) * (0.5 ** n)
    p = min(1.0, 2.0 * tail)
    return b, c, p


# --------------------------------------------------------------------------- #
# Offline validation: reproduce the published n=37 +0.135 from existing files.
# --------------------------------------------------------------------------- #
def validate_n37():
    main_eval = os.path.join(_ROOT, "eval", "results")
    anch = os.path.join(_ROOT, "directional_steering", "results")
    clean_p = os.path.join(main_eval, "bbq_clean_samples.jsonl")
    a8_p = os.path.join(anch, "bbq_L14_anchored_a8_samples.jsonl")
    if not (os.path.exists(clean_p) and os.path.exists(a8_p)):
        print("VALIDATION SKIPPED (missing existing result files).")
        return None
    black_idx, _, _ = aa.build_black_map()             # seed-42 map (the n=37 pilot)
    clean = aa.load_samples(clean_p)
    a8 = aa.load_samples(a8_p)
    m0 = aa.black_metrics(clean, black_idx, clean)
    m8 = aa.black_metrics(a8, black_idx, clean)
    db = m8["black_pick_rate"] - m0["black_pick_rate"]
    dnb = m8["nonblack_pick_rate"] - m0["nonblack_pick_rate"]
    d_gap = db - dnb
    ok = (m8["n"] == 37 and abs(m8["black_pick_rate"] - 0.568) < 0.01
          and abs(m8["nonblack_pick_rate"] - 0.351) < 0.01 and abs(d_gap - 0.135) < 0.01)
    print("-" * 72)
    print("OFFLINE VALIDATION (seed-42 n=37, existing files: anchored a=8 vs clean)")
    print(f"  n={m8['n']}  black={m8['black_pick_rate']:.3f} (exp 0.568)  "
          f"nonblk={m8['nonblack_pick_rate']:.3f} (exp 0.351)  d_gap={d_gap:+.3f} (exp +0.135)")
    print(f"  --> {'PASS' if ok else 'FAIL'}")
    print("-" * 72)
    return ok


def main():
    ap = argparse.ArgumentParser(description="E1 full-set analysis with CIs.")
    ap.add_argument("--eval-set", default=EVAL_SET)
    ap.add_argument("--results", default=RESULTS)
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--validate-only", action="store_true")
    args = ap.parse_args()

    ok = validate_n37()
    if args.validate_only:
        return

    black_idx = black_map_from_jsonl(args.eval_set)
    print(f"\nFull Black-referent eval map: {len(black_idx)} items (from {os.path.basename(args.eval_set)})")

    # load per-condition samples
    ind = {}
    accd = {}
    missing = []
    for label, stem in E1_CONDITIONS:
        sp = os.path.join(args.results, f"{stem}_samples.jsonl")
        mp = os.path.join(args.results, f"{stem}.json")
        if not os.path.exists(sp):
            missing.append(sp)
            continue
        ind[label] = per_item_indicators(aa.load_samples(sp), black_idx)
        accd[label] = json.load(open(mp))["overall"]["acc_disambig"] if os.path.exists(mp) else float("nan")
    if missing:
        print("\nNOTE: missing E1 result files (run run_e1.sh first):")
        for m in missing:
            print(f"  {m}")
        if "clean" not in ind:
            print("Cannot analyze without the clean run. Exiting.")
            return

    clean_ind = ind["clean"]
    print("\n" + "=" * 96)
    print("E1 -- FULL Black-referent ambiguous set  (d_gap = Δblack − Δnonblack vs clean)")
    print("=" * 96)
    hdr = (f"{'condition':12s} {'n':>5s} {'black':>7s} {'nonblk':>7s} {'abst':>7s} "
           f"{'d_gap':>7s} {'95% CI':>18s} {'McNemar p':>10s} {'accDis':>7s}")
    print(hdr)
    print("-" * 96)
    for label, _ in E1_CONDITIONS:
        if label not in ind:
            continue
        si = ind[label]
        keys = list(si.keys())
        n = len(keys)
        black = sum(si[k][0] for k in keys) / n
        nonblk = sum(si[k][1] for k in keys) / n
        abst = sum(si[k][2] for k in keys) / n
        if label == "clean":
            print(f"{label:12s} {n:5d} {black:7.3f} {nonblk:7.3f} {abst:7.3f} "
                  f"{'—':>7s} {'—':>18s} {'—':>10s} {accd.get(label, float('nan')):7.3f}")
            continue
        point, lo, hi, npair = bootstrap_ci(clean_ind, si, args.n_boot, args.seed)
        b, c, p = mcnemar_exact(clean_ind, si)
        ci = f"[{lo:+.3f}, {hi:+.3f}]"
        print(f"{label:12s} {n:5d} {black:7.3f} {nonblk:7.3f} {abst:7.3f} "
              f"{point:+7.3f} {ci:>18s} {p:10.4g} {accd.get(label, float('nan')):7.3f}")
    print("-" * 96)
    print("d_gap>0 = aims at Black; CI excluding 0 = significant. McNemar tests the")
    print("per-item black-pick change (b=gained, c=lost) between clean and steered.")


if __name__ == "__main__":
    main()
