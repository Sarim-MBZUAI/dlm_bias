#!/usr/bin/env python
"""Capstone comparison table: every steering method vs OURS, across BBQ + UNQOVER.

Reads the matrix result files written by run_matrix.sh into experiments/results/matrix/
and emits ONE markdown table (experiments/results/comparison_table.md, also printed).

  BBQ block     (reuses experiments/e1_analyze):  d_gap [95% CI] · McNemar p ·
                abstain · black(target) · nonblk · acc_disambig
  UNQOVER block (reuses datasets/unqover/unqover_metric):  mu · mean|C| ·
                pref_gap(target) · Delta-mu · Delta-pref_gap  (vs clean)

Baselines run at THEIR OWN config (layer/alpha, annotated per row); OURS is
full-layer closed-loop. Missing result files render as "—" so a partial matrix
still produces a table.

CPU only.
  Full table:  python experiments/build_comparison_table.py [--target-subject African]
  Self-test :  python experiments/build_comparison_table.py --selftest
               (reproduces the +0.135 pilot from existing files; no matrix runs needed)
"""
import argparse
import json
import math
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_ROOT, "eval"))
sys.path.insert(0, os.path.join(_ROOT, "directional_steering"))
sys.path.insert(0, os.path.join(_ROOT, "datasets", "unqover"))

import e1_analyze as e1                 # black_map_from_jsonl / per_item_indicators / bootstrap_ci / mcnemar_exact
import anchored_analysis as aa          # load_samples / build_black_map
import unqover_metric as uq             # load / compute / target_gap

MATRIX = os.path.join(_HERE, "results", "matrix")
EVAL_SET = e1.EVAL_SET
OUT_MD = os.path.join(_HERE, "results", "comparison_table.md")

# (label, bbq stem, unqover stem, config annotation)
METHODS = [
    ("clean",                "bbq_clean",          "uq_clean",          "—"),
    ("CAA",                  "bbq_caa",            "uq_caa",            "own cfg: L{caa}"),
    ("ActAdd",               "bbq_actadd",         "uq_actadd",         "own cfg: L{actadd}"),
    ("group mean-diff",      "bbq_group",          "uq_group",          "own cfg: L{group}"),
    ("**OURS clamp (P)**",   "bbq_ours_clamp_all", "uq_ours_clamp_all", "full-layer, c*=60"),
    ("**OURS cmom (PI)**",   "bbq_ours_cmom_all",  "uq_ours_cmom_all",  "full-layer, c*=60, β=0.8"),
]


def _cfg(path, key, default="?"):
    try:
        return json.load(open(path))[key]
    except Exception:
        return default


def _annot(tmpl):
    return (tmpl.replace("{caa}", str(_cfg(os.path.join(_ROOT, "baselines/caa/config.json"), "layer")))
                .replace("{actadd}", str(_cfg(os.path.join(_ROOT, "baselines/actadd/config.json"), "layer")))
                .replace("{group}", str(_cfg(os.path.join(_ROOT, "baselines/group_meandiff/config.json"), "layer"))))


# --------------------------------------------------------------------------- #
# BBQ block
# --------------------------------------------------------------------------- #
def bbq_indicators(stem, black_idx, results_dir):
    sp = os.path.join(results_dir, f"{stem}_samples.jsonl")
    if not os.path.exists(sp):
        return None, float("nan")
    ind = e1.per_item_indicators(aa.load_samples(sp), black_idx)
    mp = os.path.join(results_dir, f"{stem}.json")
    accd = json.load(open(mp))["overall"]["acc_disambig"] if os.path.exists(mp) else float("nan")
    return ind, accd


def bbq_row(ind, accd, clean_ind, n_boot, seed):
    keys = list(ind)
    n = len(keys)
    black = sum(ind[k][0] for k in keys) / n if n else float("nan")
    nonblk = sum(ind[k][1] for k in keys) / n if n else float("nan")
    abst = sum(ind[k][2] for k in keys) / n if n else float("nan")
    if clean_ind is None or ind is clean_ind:
        return dict(n=n, black=black, nonblk=nonblk, abst=abst, dgap=None, ci=None, p=None, accd=accd)
    point, lo, hi, _ = e1.bootstrap_ci(clean_ind, ind, n_boot=n_boot, seed=seed)
    _, _, p = e1.mcnemar_exact(clean_ind, ind)
    return dict(n=n, black=black, nonblk=nonblk, abst=abst, dgap=point, ci=(lo, hi), p=p, accd=accd)


# --------------------------------------------------------------------------- #
# UNQOVER block
# --------------------------------------------------------------------------- #
def uq_row(stem, results_dir, target, clean_m, clean_tg):
    rp = os.path.join(results_dir, f"{stem}.jsonl")
    if not os.path.exists(rp):
        return None
    recs = uq.load(rp)
    m = uq.compute(recs)
    tg = uq.target_gap(recs, target) if target else {"pref_gap_debiased": float("nan")}
    dmu = (m["mu"] - clean_m["mu"]) if clean_m else None
    dgap = ((tg["pref_gap_debiased"] - clean_tg["pref_gap_debiased"])
            if (clean_tg and target) else None)
    return dict(n=m["n_complete"], mu=m["mu"], absC=m["mean_absC"],
                prefgap=tg["pref_gap_debiased"], dmu=dmu, dgap=dgap)


# --------------------------------------------------------------------------- #
# rendering
# --------------------------------------------------------------------------- #
def _f(x, s="{:+.3f}"):
    return "—" if x is None or (isinstance(x, float) and math.isnan(x)) else s.format(x)


def render(rows, target):
    L = []
    L.append(f"# Method × benchmark comparison  (BBQ + UNQOVER; target = {target})\n")
    L.append("Baselines run at their **own** config (layer/α); **OURS** is full-layer "
             "closed-loop. d_gap>0 = aims at the target group; CI excluding 0 = significant.\n")
    L.append("| method | config | BBQ n | BBQ d_gap [95% CI] | McNemar p | abstain | target-pick | nontarget | accDis "
             "| UQ n | μ | mean\\|C\\| | pref_gap | Δμ | Δpref_gap |")
    L.append("|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|")
    for label, annot, b, u in rows:
        if b is None:
            bbq = "| — | — | — | — | — | — | — |"
        else:
            if b["dgap"] is None:
                dcell = "—"
            else:
                ci = f" [{b['ci'][0]:+.3f}, {b['ci'][1]:+.3f}]" if b["ci"] else ""
                dcell = f"{b['dgap']:+.3f}{ci}"
            pcell = f"{b['p']:.3g}" if b["p"] is not None else "—"
            bbq = (f"| {b['n']} | {dcell} | {pcell} | {_f(b['abst'],'{:.3f}')} "
                   f"| {_f(b['black'],'{:.3f}')} | {_f(b['nonblk'],'{:.3f}')} | {_f(b['accd'],'{:.3f}')} |")
        if u is None:
            uqs = "— | — | — | — | — | — |"
        else:
            uqs = (f"{u['n']} | {_f(u['mu'],'{:.3f}')} | {_f(u['absC'],'{:.3f}')} "
                   f"| {_f(u['prefgap'])} | {_f(u['dmu'])} | {_f(u['dgap'])} |")
        L.append(f"| {label} | {annot} {bbq} {uqs}")
    return "\n".join(L) + "\n"


def build(results_dir, eval_set, target, n_boot, seed):
    black_idx = e1.black_map_from_jsonl(eval_set)
    clean_ind, clean_accd = bbq_indicators("bbq_clean", black_idx, results_dir)
    # UNQOVER clean baseline for deltas
    clean_rp = os.path.join(results_dir, "uq_clean.jsonl")
    clean_m = clean_tg = None
    if os.path.exists(clean_rp):
        cr = uq.load(clean_rp)
        clean_m = uq.compute(cr)
        clean_tg = uq.target_gap(cr, target) if target else None
    rows = []
    for label, bstem, ustem, annot_tmpl in METHODS:
        annot = _annot(annot_tmpl)
        ind, accd = bbq_indicators(bstem, black_idx, results_dir)
        brow = bbq_row(ind, accd, clean_ind, n_boot, seed) if ind is not None else None
        urow = uq_row(ustem, results_dir, target, clean_m, clean_tg)
        rows.append((label, annot, brow, urow))
    return render(rows, target)


# --------------------------------------------------------------------------- #
# offline self-test: reproduce the +0.135 pilot from EXISTING files
# --------------------------------------------------------------------------- #
def selftest():
    clean_p = os.path.join(_ROOT, "eval", "results", "bbq_clean_samples.jsonl")
    a8_p = os.path.join(_ROOT, "directional_steering", "results", "bbq_L14_anchored_a8_samples.jsonl")
    if not (os.path.exists(clean_p) and os.path.exists(a8_p)):
        print("SELFTEST SKIPPED (missing existing result files).")
        return None
    black_idx, _, _ = aa.build_black_map()            # seed-42 n=37 pilot map
    clean_ind = e1.per_item_indicators(aa.load_samples(clean_p), black_idx)
    a8_ind = e1.per_item_indicators(aa.load_samples(a8_p), black_idx)
    row = bbq_row(a8_ind, float("nan"), clean_ind, n_boot=2000, seed=0)
    ok = (row["n"] == 37 and abs(row["black"] - 0.568) < 0.01
          and abs(row["nonblk"] - 0.351) < 0.01 and abs(row["dgap"] - 0.135) < 0.01)
    # render a 2-row table (clean + anchored-a8 as a stand-in "OURS") to prove the BBQ column renders
    rows = [("clean", "—", bbq_row(clean_ind, float("nan"), clean_ind, 0, 0), None),
            ("anchored a=8 (pilot)", "L14 α=8", row, None)]
    print(render(rows, target="African"))
    print("-" * 72)
    print(f"SELFTEST: n={row['n']} black={row['black']:.3f} (exp .568) "
          f"nonblk={row['nonblk']:.3f} (exp .351) d_gap={row['dgap']:+.3f} (exp +0.135)")
    print(f"  BBQ column renders + reproduces pilot --> {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    ap = argparse.ArgumentParser(description="Comparison matrix table (BBQ + UNQOVER).")
    ap.add_argument("--results", default=MATRIX)
    ap.add_argument("--eval-set", default=EVAL_SET)
    ap.add_argument("--target-subject", default="African")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return

    if not os.path.exists(args.eval_set):
        print(f"Missing BBQ eval set {args.eval_set} — run experiments/build_eval_set.py first.")
        return
    md = build(args.results, args.eval_set, args.target_subject, args.n_boot, args.seed)
    os.makedirs(os.path.dirname(OUT_MD), exist_ok=True)
    with open(OUT_MD, "w") as f:
        f.write(md)
    print(md)
    print(f"[written to {OUT_MD}]")


if __name__ == "__main__":
    main()
