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
PRESWEEP_NORMAL = os.path.join(HERE, "presweep_normal")
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

    # Row order for the combined table: base, normalL*, P, PI, PID (then by alpha).
    def fam(cond):
        if cond == "base":
            return 0
        if cond.startswith("normal"):
            return 1
        return {"P": 2, "PI": 3, "PID": 4}.get(cond, 9)
    tags = sorted(res.keys(), key=lambda t: (fam(res[t]["condition"]), res[t].get("alpha", 0.0)))

    lines = []
    lines.append("# PID-Steering vs normal steering vector on LLaDA-8B-Instruct")
    lines.append("## BBQ, 400 Black-referent ambiguous items\n")
    meta = base
    lines.append(f"- eval set: `experiments/data/_sweep400.jsonl` (n={meta['n']})")
    lines.append(f"- gen/steps/block: {meta['gen_length']}/{meta['steps']}/{meta['block_length']}, "
                 f"temperature {meta.get('temperature', 0.0)}")
    am = base.get("arrows", {})
    lines.append(f"- arrows: {am.get('source')} (n_items={am.get('n_items')}), "
                 f"injected at all 32 blocks every denoising step")
    lines.append("- **normalL14**: single fixed vector v̂ = unit(r14) added identically at all "
                 "32 blocks (Kp-only, no per-layer/I/D)")
    lines.append("- **P / PI / PID**: per-layer PID combine of rhat(k) "
                 "(Kp=1.0, Ki=0.05, Kd=0.02; each condition selects active terms)\n")

    lines.append("### Combined comparison at alpha = 2\n")
    hdr = "| condition | alpha | black_pick | nonblack | abstain | unparse | d_gap (vs base) |"
    sep = "|" + "|".join(["---"] * 7) + "|"
    lines.append(hdr)
    lines.append(sep)
    for t in tags:
        r = res[t]
        rr = r["rates"]
        d_gap = (rr["black_pick_rate"] - bpc) - (rr["nonblack_pick_rate"] - nbc)
        lines.append(
            f"| {r['condition']} | {r.get('alpha', 0):g} | "
            f"{rr['black_pick_rate']:.3f} | {rr['nonblack_pick_rate']:.3f} | "
            f"{rr['abstain_rate']:.3f} | {rr['unparseable_rate']:.3f} | {d_gap:+.3f} |"
        )
    lines.append("\n(d_gap = (black - base_black) - (nonblack - base_nonblack); "
                 "positive = aimed toward Black relative to the clean model.)")

    # Second table: normal steering-vector alpha-sweep (100 items).
    sweep = []
    for p in sorted(glob.glob(os.path.join(PRESWEEP_NORMAL, "cond_normal_a*.json"))):
        if p.endswith("_samples.jsonl"):
            continue
        sweep.append(json.load(open(p)))
    if sweep:
        sweep.sort(key=lambda r: r.get("alpha", 0.0))
        sn = sweep[0]["n"]
        lines.append(f"\n### Normal steering-vector alpha-sweep (v̂ = unit(r14), first {sn} items)\n")
        lines.append("| alpha | black | nonblack | abstain | unparse | raw gap (black-nonblack) |")
        lines.append("|" + "|".join(["---"] * 6) + "|")
        for r in sweep:
            rr = r["rates"]
            raw = rr["black_pick_rate"] - rr["nonblack_pick_rate"]
            lines.append(
                f"| {r.get('alpha', 0):g} | {rr['black_pick_rate']:.2f} | "
                f"{rr['nonblack_pick_rate']:.2f} | {rr['abstain_rate']:.2f} | "
                f"{rr['unparseable_rate']:.2f} | {raw:+.2f} |"
            )

    txt = "\n".join(lines) + "\n"
    with open(OUT_MD, "w") as f:
        f.write(txt)
    print(txt)
    print(f"wrote {OUT_MD}")


if __name__ == "__main__":
    main()
