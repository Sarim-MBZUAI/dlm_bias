#!/usr/bin/env python
"""PID-Steering final assembly. Reads every cond_*.json in results/, computes the
directional gap relative to the base (clean) condition, and writes a markdown table.

    d_gap = (black_pick - clean_black_pick) - (nonblack_pick - clean_nonblack_pick)

d_gap > 0 means the intervention aims the model toward the Black option MORE than it
raises the non-Black option, relative to clean. The base (no-hook) condition must be
present to define the clean reference.
"""
import glob
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
OUT_MD = os.path.join(HERE, "results_table.md")


def load_conds(results_dir):
    res = {}
    for p in sorted(glob.glob(os.path.join(results_dir, "cond_*.json"))):
        if p.endswith("_samples.jsonl"):
            continue
        r = json.load(open(p))
        tag = os.path.basename(p)[len("cond_"):-len(".json")]
        res[tag] = r
    return res


def main():
    res = load_conds(RESULTS)
    if not res:
        raise SystemExit(f"no cond_*.json in {RESULTS}")

    base = res.get("base")
    if base is None:
        raise SystemExit("base (clean) condition missing; cannot compute d_gap")
    bpc = base["rates"]["black_pick_rate"]
    nbc = base["rates"]["nonblack_pick_rate"]

    # order: base first, then by condition family then alpha
    def sort_key(t):
        r = res[t]
        fam_order = {"base": 0, "P": 1, "PI": 2, "PID": 3}
        return (fam_order.get(r["condition"], 9), r.get("alpha", 0.0))
    tags = sorted(res.keys(), key=sort_key)

    lines = []
    lines.append("# PID-Steering on LLaDA-8B-Instruct -- BBQ (400 Black-referent ambiguous items)\n")
    meta = base
    lines.append(f"- eval set: `experiments/data/_sweep400.jsonl` (n={meta['n']})")
    lines.append(f"- gen/steps/block: {meta['gen_length']}/{meta['steps']}/{meta['block_length']}, "
                 f"temperature {meta.get('temperature', 0.0)}")
    am = base.get("arrows", {})
    lines.append(f"- arrows: {am.get('source')} (n_items={am.get('n_items')}), "
                 f"unit-normed per layer, injected at all 32 blocks every denoising step")
    lines.append(f"- gains: Kp=1.0, Ki=0.05, Kd=0.02 (P/PI/PID select active terms)\n")

    hdr = ("| condition | alpha | Kp | Ki | Kd | black | nonblack | abstain | "
           "unparse | d_gap | acc_dis |")
    sep = "|" + "|".join(["---"] * 11) + "|"
    lines.append(hdr)
    lines.append(sep)
    for t in tags:
        r = res[t]
        rr = r["rates"]
        g = r["gains"]
        d_gap = (rr["black_pick_rate"] - bpc) - (rr["nonblack_pick_rate"] - nbc)
        acc = r.get("acc_disambig")
        acc_s = f"{acc:.3f}" if acc is not None else "n/a"
        lines.append(
            f"| {r['condition']} | {r.get('alpha', 0):g} | {g['Kp']:g} | {g['Ki']:g} | "
            f"{g['Kd']:g} | {rr['black_pick_rate']:.3f} | {rr['nonblack_pick_rate']:.3f} | "
            f"{rr['abstain_rate']:.3f} | {rr['unparseable_rate']:.3f} | "
            f"{d_gap:+.3f} | {acc_s} |"
        )
    lines.append("\n(d_gap is relative to the base/clean condition; positive = aimed toward Black.)")

    txt = "\n".join(lines) + "\n"
    with open(OUT_MD, "w") as f:
        f.write(txt)
    print(txt)
    print(f"wrote {OUT_MD}")


if __name__ == "__main__":
    main()
