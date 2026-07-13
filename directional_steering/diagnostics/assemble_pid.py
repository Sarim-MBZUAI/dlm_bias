#!/usr/bin/env python
"""STEP 4/final. Assemble per-condition results into pid_layerwise.json + .png.

Reads cond_<name>.json for each of the 5 conditions, computes d_gap relative to
clean, a competence proxy (unparseable rate; abstain shown separately), and writes
the final persistent json + bar chart to the diagnostics dir.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

DIAG = "/home/lukas/users/shashmi/dlm_bias/directional_steering/diagnostics"
CONDS = ["clean", "L14_open_a8", "layerwise_P", "layerwise_PI", "layerwise_PID"]


def main():
    res = {}
    for c in CONDS:
        p = os.path.join(DIAG, f"cond_{c}.json")
        if os.path.exists(p):
            res[c] = json.load(open(p))
    if "clean" not in res:
        raise SystemExit("clean condition missing; cannot compute d_gap")

    bpc = res["clean"]["rates"]["black_pick_rate"]
    nbc = res["clean"]["rates"]["nonblack_pick_rate"]

    metrics = {}
    for c, r in res.items():
        rr = r["rates"]
        d_gap = (rr["black_pick_rate"] - bpc) - (rr["nonblack_pick_rate"] - nbc)
        metrics[c] = {
            **rr,
            "d_gap": d_gap,
            "competence_unparseable_rate": rr["unparseable_rate"],
            "counts": r["counts"], "n": r["n"], "elapsed_s": r.get("elapsed_s"),
        }

    arrows = torch.load(os.path.join(DIAG, "perlayer_arrows.pt"), map_location="cpu")
    gains = res.get("layerwise_PID", res.get("layerwise_P", {})).get("gains", {})
    devices = {c: res[c].get("device", "cuda") for c in res}

    final = {
        "meta": {
            "eval_set": "experiments/data/_sweep400.jsonl",
            "n": res["clean"]["n"],
            "gpus": "5,6,7 (per-job CUDA_VISIBLE_DEVICES; see cond logs)",
            "gains": gains,
            "gen_steps_block": [res["clean"]["gen_length"], res["clean"]["steps"],
                                res["clean"]["block_length"]],
            "arrows_n_items": arrows["n_items"],
            "arrows_source": arrows["source"],
            "per_layer_arrow_norm": [round(x, 3) for x in arrows["per_layer_raw_norm"]],
        },
        "conditions": metrics,
    }
    outj = os.path.join(DIAG, "pid_layerwise.json")
    with open(outj, "w") as f:
        json.dump(final, f, indent=2)
    print("wrote", outj)

    # ---- plot ----
    present = [c for c in CONDS if c in metrics]
    x = range(len(present))
    w = 0.2
    black = [metrics[c]["black_pick_rate"] for c in present]
    nonb = [metrics[c]["nonblack_pick_rate"] for c in present]
    abst = [metrics[c]["abstain_rate"] for c in present]
    unp = [metrics[c]["unparseable_rate"] for c in present]
    dgap = [metrics[c]["d_gap"] for c in present]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6),
                                   gridspec_kw={"width_ratios": [3, 1]})
    ax1.bar([i - 1.5 * w for i in x], black, w, label="black_pick", color="#1f77b4")
    ax1.bar([i - 0.5 * w for i in x], nonb, w, label="nonblack_pick", color="#ff7f0e")
    ax1.bar([i + 0.5 * w for i in x], abst, w, label="abstain", color="#2ca02c")
    ax1.bar([i + 1.5 * w for i in x], unp, w, label="unparseable", color="#d62728")
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(present, rotation=20, ha="right")
    ax1.set_ylabel("rate")
    ax1.set_title("Pick-rate breakdown per condition (400 Black-referent ambig items)")
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3)

    colors = ["#888" if c == "clean" else ("#9467bd" if c == "L14_open_a8" else "#17becf")
              for c in present]
    ax2.bar(list(x), dgap, color=colors)
    ax2.axhline(0, color="k", lw=0.8)
    ax2.set_xticks(list(x))
    ax2.set_xticklabels(present, rotation=20, ha="right")
    ax2.set_ylabel("d_gap  (aim toward Black option)")
    ax2.set_title("d_gap vs clean")
    ax2.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    outp = os.path.join(DIAG, "pid_layerwise.png")
    fig.savefig(outp, dpi=130)
    print("wrote", outp)

    # console table
    print(f"\n{'cond':16s}{'black':>8s}{'nonblk':>8s}{'abst':>8s}{'unparse':>8s}{'d_gap':>8s}")
    for c in present:
        m = metrics[c]
        print(f"{c:16s}{m['black_pick_rate']:8.3f}{m['nonblack_pick_rate']:8.3f}"
              f"{m['abstain_rate']:8.3f}{m['unparseable_rate']:8.3f}{m['d_gap']:8.3f}")


if __name__ == "__main__":
    main()
