#!/usr/bin/env python
"""Ghostwriter (input-space) vs activation-steering (block L14) on BBQ.

Reads the single Ghostwriter metrics JSON from baseline/results/ and the shared
clean+steering JSONs from eval/results/, prints the comparison table, and
(re)generates baseline/figs/ghostwriter_vs_steering.png:
  (L) target-rate vs abstention: the steering sweep + the single Ghostwriter point
  (R) competence panel:          abstention_rate vs acc_disambig
Ghostwriter is a single fabricated-evidence injection (no strength dial), so it is
one point, not a dose curve.
Run:  python baseline/compare_baseline.py     (deps: matplotlib)
"""
import os, json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))   # .../baseline
ROOT = os.path.dirname(HERE)                         # repo root
GW_RES = os.path.join(HERE, "results")               # baseline/results (ghostwriter)
STEER_RES = os.path.join(ROOT, "eval", "results")    # shared clean+steering
FIG = os.path.join(HERE, "figs")                     # baseline/figs
os.makedirs(FIG, exist_ok=True)

# (label, file, series)  series: "gw" | "steer"
ROWS = [
    ("clean",       "bbq_clean.json",              None),
    ("ghostwriter", "bbq_ghostwriter.json",        "gw"),
    ("steer a=8",   "bbq_L14_race_color_a8.json",  "steer"),
    ("steer a=16",  "bbq_L14_race_color_a16.json", "steer"),
    ("steer a=32",  "bbq_L14_race_color_a32.json", "steer"),
]


def load():
    out = []
    for label, fn, series in ROWS:
        # Ghostwriter runs live in baseline/results; clean+steering in eval/results
        base = GW_RES if series == "gw" else STEER_RES
        p = os.path.join(base, fn)
        if not os.path.exists(p):
            continue
        j = json.load(open(p))
        am, o = j["attack_metrics"], j["overall"]
        out.append(dict(label=label, series=series,
                        abstain=am["abstention_rate"], target=am["target_rate"],
                        nontarget=am["nontarget_rate"], acc_dis=o["acc_disambig"],
                        s_AMB=o["s_AMB"]))
    return out


def print_table(rows):
    hdr = ["run", "abstain", "target", "nontarget", "acc_disambig", "s_AMB"]
    print(f"{hdr[0]:<12}{hdr[1]:>9}{hdr[2]:>9}{hdr[3]:>11}{hdr[4]:>14}{hdr[5]:>9}")
    for r in rows:
        print(f"{r['label']:<12}{r['abstain']:>9.3f}{r['target']:>9.3f}"
              f"{r['nontarget']:>11.3f}{r['acc_dis']:>14.3f}{r['s_AMB']:>+9.3f}")


def make_fig(rows):
    gw = [r for r in rows if r["series"] == "gw"]
    st = [r for r in rows if r["series"] == "steer"]
    base = next(r for r in rows if r["label"] == "clean")
    GW_C, ST_C, B_C = "#c0392b", "#1f6fb2", "#555555"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.6, 5.2))

    # Ghostwriter is now a SINGLE injection (no strength dial), so it plots as one
    # reference point alongside the continuous steering alpha sweep -- the "two
    # doors to the same abstention-collapse room" made visual.
    def series(ax, x, y, only=None):
        ax.plot([r[x] for r in st], [r[y] for r in st], "s-", color=ST_C,
                lw=1.6, ms=8, label="Activation steering (L14)")
        ax.scatter([base[x]], [base[y]], marker="*", s=220, color=B_C,
                   zorder=5, label="clean baseline (a=0)")
        if gw:
            ax.scatter([gw[0][x]], [gw[0][y]], marker="D", s=90, color=GW_C,
                       zorder=5, label="Ghostwriter (input-space)")
            ax.annotate("ghostwriter", (gw[0][x], gw[0][y]),
                        textcoords="offset points", xytext=(6, 6), fontsize=8,
                        color=GW_C)
        LOFF = {"clean": (8, -13), "a=8": (8, -13),
                "a=16": (-40, -14), "a=32": (-40, 4)}
        for r in st + [base]:
            lb = r["label"].replace("steer ", "")
            if only is not None and lb not in only:
                continue
            dx, dy = LOFF.get(lb, (6, 5))
            ax.annotate(lb, (r[x], r[y]), textcoords="offset points",
                        xytext=(dx, dy), fontsize=8)

    # (L) abstention vs target-rate along the steering alpha sweep
    series(ax1, "abstain", "target")
    ax1.set_xlabel("abstention rate (ambiguous items)")
    ax1.set_ylabel("target (stereotyped-group) pick rate")
    ax1.set_title("Steering sweep: target-rate vs abstention")
    ax1.invert_xaxis()  # attack strength increases left->right
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(alpha=0.25)

    # (R) competence: abstention vs disambig accuracy (all points cluster at
    # ~0.97 except a=32, so label only clean + the a=32 cliff to avoid overlap)
    series(ax2, "abstain", "acc_dis", only={"clean", "a=8", "a=16", "a=32"})
    ax2.axhline(0.95, ls="--", c="grey", lw=1.0)
    ax2.set_xlabel("abstention rate (ambiguous items)")
    ax2.set_ylabel("disambiguated accuracy (task competence)")
    ax2.set_title("Competence cliff: a=32 wrecks\n"
                  "disambiguated accuracy (0.97->0.64)")
    ax2.invert_xaxis()
    ax2.set_ylim(0.60, 1.0)
    ax2.legend(loc="lower left", fontsize=9)
    ax2.grid(alpha=0.25)

    fig.suptitle("Activation steering (L14) dose-response vs Ghostwriter (single injection) on BBQ",
                 fontsize=12.5, y=1.02)
    fig.tight_layout()
    out = os.path.join(FIG, "ghostwriter_vs_steering.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


if __name__ == "__main__":
    rows = load()
    print_table(rows)
    out = make_fig(rows)
    print("\nWROTE", out)
