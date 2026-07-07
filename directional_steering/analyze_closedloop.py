#!/usr/bin/env python
"""Open-loop vs closed-loop injection of the Black bias on BBQ.

Reuses anchored_analysis' Black-referent logic. For each run computes, on the
Black-referent AMBIGUOUS items (n~37, seed-42 sample): black_pick, nonblack_pick,
directional gap = (Δblack - Δnonblack) vs clean, abstention, acc_disambig.

Runs compared:
  clean            (a=0)
  open-loop  a=4, a=8   (additive alpha*dir  -- the original method)
  closed clamp  (P)     bbq_L14_anchored_clampC60
  closed cmom   (PI)    bbq_L14_anchored_cmomC60
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import anchored_analysis as A

RES = A.ANCH_DIR
RUNS = [
    ("clean",         A.CLEAN,                                   A.CLEAN_METRICS),
    ("open a=4",      f"{RES}/bbq_L14_anchored_a4_samples.jsonl", f"{RES}/bbq_L14_anchored_a4.json"),
    ("open a=8",      f"{RES}/bbq_L14_anchored_a8_samples.jsonl", f"{RES}/bbq_L14_anchored_a8.json"),
    ("clamp (P) c60", f"{RES}/bbq_L14_anchored_clampC60_samples.jsonl", f"{RES}/bbq_L14_anchored_clampC60.json"),
    ("cmom (PI) c60", f"{RES}/bbq_L14_anchored_cmomC60_samples.jsonl",  f"{RES}/bbq_L14_anchored_cmomC60.json"),
    ("ALL-lyr add .25",  f"{RES}/bbq_alllayer_add025_samples.jsonl",   f"{RES}/bbq_alllayer_add025.json"),
    ("ALL-lyr clamp o2", f"{RES}/bbq_alllayer_clamp2_samples.jsonl",   f"{RES}/bbq_alllayer_clamp2.json"),
    ("ALL-lyr cmom o2",  f"{RES}/bbq_alllayer_cmom2_samples.jsonl",    f"{RES}/bbq_alllayer_cmom2.json"),
]

def main():
    black_idx, _, _ = A.build_black_map()
    clean = A.load_samples(A.CLEAN)
    cm = A.black_metrics(clean, black_idx)
    b0, nb0 = cm["black_pick_rate"], cm["nonblack_pick_rate"]
    print(f"clean baseline: black={b0:.3f} nonblack={nb0:.3f}\n")
    print(f"{'run':16}{'black':>8}{'nonblk':>8}{'abst':>7}{'d_black':>9}{'d_nonblk':>9}{'d_gap':>8}{'accDis':>8}")
    for name, sf, mf in RUNS:
        if not os.path.exists(sf):
            print(f"{name:16}  (missing: {os.path.basename(sf)})"); continue
        m = A.black_metrics(A.load_samples(sf), black_idx, clean)
        db = m["black_pick_rate"] - b0
        dnb = m["nonblack_pick_rate"] - nb0
        gap = db - dnb
        acc = A.acc_disambig(mf) if os.path.exists(mf) else float("nan")
        print(f"{name:16}{m['black_pick_rate']:>8.3f}{m['nonblack_pick_rate']:>8.3f}"
              f"{m['abstention_rate']:>7.3f}{db:>+9.3f}{dnb:>+9.3f}{gap:>+8.3f}{acc:>8.3f}")
    print("\nREAD: d_gap = (Δblack − Δnonblack) = directionality (higher = more AIMED at Black).")
    print("Closed-loop wins if it gives a LARGER d_gap and/or higher accDis than open-loop.")

if __name__ == "__main__":
    main()
