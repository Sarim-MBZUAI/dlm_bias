#!/usr/bin/env python
"""E2 construction ablation (THE aim-vs-disinhibit heart).

Four steering-direction CONSTRUCTIONS, all run at MATCHED effective strength
(unit-normalized direction, same alpha, same layer L14, same eval set), scored on
the FULL Black-referent ambiguous set:

  1. group mean-diff       (race_steering/race_black.pt)          -- expected: disinhibits
  2. letter-anchored       (directional_steering/race_black_anchored.pt)
  3. answer-text-anchored  (directional_steering/race_black_anchored_text.pt) -- expected: AIMS
  4. FairPCA subspace      (directional_steering/race_black_fairpca.pt)

Columns: construction | split-half coherence (from .pt) | d_gap (AIM) |
Δabstention (DISINHIBIT) | acc_disambig (COMPETENCE) | aiming ratio.

  aiming ratio = d_gap / (abstention_drop)   where abstention_drop = abst_clean − abst_steer
A pure disinhibitor collapses abstention with ~0 gap  -> ratio ~ 0.
An aimer produces gap without dumping everything off Unknown -> high ratio.

Reuses the metric helpers from e1_analyze (same per-item Black-referent logic).
CPU only. Run AFTER the E2 runs (see run_e2.sh):
  /home/lukas/miniconda3/envs/sarim_awm/bin/python experiments/e2_analyze.py
"""
import argparse
import json
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_ROOT, "directional_steering"))
import e1_analyze as e1
import anchored_analysis as aa

RESULTS = os.path.join(_HERE, "results")
EVAL_SET = e1.EVAL_SET

# construction label -> (direction .pt path, E2 result stem prefix)
CONSTRUCTIONS = [
    ("group mean-diff", os.path.join(_ROOT, "race_steering", "race_black.pt"),                      "e2_groupmeandiff"),
    ("letter-anchored", os.path.join(_ROOT, "directional_steering", "race_black_anchored.pt"),      "e2_letter"),
    ("answer-text",     os.path.join(_ROOT, "directional_steering", "race_black_anchored_text.pt"), "e2_answertext"),
    ("FairPCA",         os.path.join(_ROOT, "directional_steering", "race_black_fairpca.pt"),        "e2_fairpca"),
]
CLEAN_STEM = "e1_clean"   # E2 reuses the E1 clean run as the reference baseline


def coherence(pt_path):
    if not os.path.exists(pt_path):
        return float("nan")
    d = torch.load(pt_path, map_location="cpu")
    v = d.get("splithalf_cosine")
    return float(v) if v is not None else float("nan")


def rates(ind):
    keys = list(ind)
    n = len(keys) or 1
    return (sum(ind[k][0] for k in keys) / n,   # black
            sum(ind[k][1] for k in keys) / n,   # nonblack
            sum(ind[k][2] for k in keys) / n)   # abstention


def main():
    ap = argparse.ArgumentParser(description="E2 construction ablation at matched strength.")
    ap.add_argument("--alpha", type=int, default=20,
                    help="matched effective (unit) strength to tabulate; must match a "
                         "value swept in run_e2.sh (stems e2_<name>_a<alpha>).")
    args = ap.parse_args()
    A = args.alpha
    black_idx = e1.black_map_from_jsonl(EVAL_SET)
    clean_sp = os.path.join(RESULTS, f"{CLEAN_STEM}_samples.jsonl")
    if not os.path.exists(clean_sp):
        print(f"Missing clean run {clean_sp} (run run_e1.sh first — E2 reuses it). Exiting.")
        return
    clean_ind = e1.per_item_indicators(aa.load_samples(clean_sp), black_idx)
    cb, cnb, c_abst = rates(clean_ind)

    print("=" * 100)
    print(f"E2 -- construction ablation @ matched unit strength alpha={A} (full Black-referent set)")
    print(f"     clean: black={cb:.3f} nonblk={cnb:.3f} abst={c_abst:.3f}  (n={len(clean_ind)})")
    print("=" * 100)
    hdr = (f"{'construction':16s} {'coherence':>9s} {'d_gap':>7s} {'Δabst':>7s} "
           f"{'accDis':>7s} {'aimRatio':>9s}")
    print(hdr)
    print("-" * 100)
    for label, pt, prefix in CONSTRUCTIONS:
        stem = f"{prefix}_a{A}"
        sp = os.path.join(RESULTS, f"{stem}_samples.jsonl")
        mp = os.path.join(RESULTS, f"{stem}.json")
        coh = coherence(pt)
        if not os.path.exists(sp):
            print(f"{label:16s} {coh:9.3f} {'(no result — run run_e2.sh)':>40s}")
            continue
        ind = e1.per_item_indicators(aa.load_samples(sp), black_idx)
        keys = [k for k in ind if k in clean_ind]
        b, nb, abst = rates({k: ind[k] for k in keys})
        d_gap = (b - cb) - (nb - cnb)
        d_abst = abst - c_abst                       # negative = abstention collapsed
        drop = c_abst - abst
        aim = d_gap / drop if abs(drop) > 1e-6 else float("nan")
        accd = json.load(open(mp))["overall"]["acc_disambig"] if os.path.exists(mp) else float("nan")
        print(f"{label:16s} {coh:9.3f} {d_gap:+7.3f} {d_abst:+7.3f} {accd:7.3f} {aim:9.3f}")
    print("-" * 100)
    print("Success signature: answer-text is the ONLY construction high on d_gap / aimRatio;")
    print("group mean-diff & FairPCA collapse abstention (large −Δabst) with ~0 gap (disinhibit).")


if __name__ == "__main__":
    main()
