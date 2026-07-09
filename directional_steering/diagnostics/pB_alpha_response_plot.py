#!/usr/bin/env python3
"""Render the alpha->p_target response curve from pB_alpha_response.json.

Standalone re-plot (the sweep script's inline matplotlib call crashed on a None
in yerr once the p_target readout became undefined at high alpha). Robust to
None: pick-rates are plotted over the full alpha range; p_target is plotted only
where the readout was usable (n_usable>0), and the breakdown region is shaded.
"""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
d = json.load(open(os.path.join(HERE, "pB_alpha_response.json")))
c = d["conditions"]
alphas = sorted((int(a) for a in c), key=int)

def series(key, need_usable=False):
    xs, ys = [], []
    for a in alphas:
        r = c[str(a)]
        v = r.get(key)
        if v is None:
            continue
        if need_usable and not r.get("n_usable"):
            continue
        xs.append(a); ys.append(v)
    return xs, ys

# reliable across full range (from parse_letter)
_, tgt = series("target_pick_rate")
_, non = series("non_target_pick_rate")
_, ab  = series("abstain_rate")
# p_target readout: only where usable
xf, finalpt = series("final_p_target_mean", need_usable=True)
xp, precpt  = series("precommit_p_target_mean", need_usable=True)

# first alpha whose readout is unusable => breakdown boundary
breakdown = next((a for a in alphas if not c[str(a)].get("n_usable")), None)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 9), sharex=True)

# Panel 1: pick-rates (full range)
ax1.plot(alphas, tgt, "o-", color="#1f77b4", label="target pick-rate")
ax1.plot(alphas, non, "s-", color="#d62728", label="non-target pick-rate")
ax1.plot(alphas, ab,  "^-", color="#7f7f7f", label="abstain rate")
peak = max(range(len(tgt)), key=lambda i: tgt[i])
ax1.axvline(alphas[peak], ls="--", color="#1f77b4", alpha=0.5)
ax1.annotate(f"peak aiming @ a={alphas[peak]}", (alphas[peak], tgt[peak]),
             textcoords="offset points", xytext=(8, 6), color="#1f77b4")
ax1.set_ylabel("pick-rate (n=64, from parse_letter)")
ax1.set_title("Formulation-B actuation-response: steering gain a  vs  aiming\n"
              "monotonic controllable band up to a~8, then reversal (nontarget overtakes)")
ax1.legend(loc="center right"); ax1.grid(alpha=0.3); ax1.set_ylim(-0.02, 1.0)

# Panel 2: p_target (usable region only)
ax2.plot(xf, finalpt, "o-", color="#2ca02c", label="final p_target (committed)")
ax2.plot(xp, precpt,  "d--", color="#ff7f0e", label="pre-commit p_target (steps 0-31)")
ax2.axhline(1/3, ls=":", color="k", alpha=0.5, label="uniform 3-way (1/3)")
if breakdown is not None:
    ax2.axvspan(breakdown - 1, alphas[-1] + 1, color="red", alpha=0.08)
    ax2.annotate("readout undefined\n(competence breakdown:\nanswer not locatable)",
                 ((breakdown + alphas[-1]) / 2, 0.5), ha="center", va="center",
                 color="#b22222", fontsize=9)
ax2.set_xlabel("steering gain  alpha  (raw direction, L14, add mode)")
ax2.set_ylabel("p_target at answer position")
ax2.legend(loc="upper right"); ax2.grid(alpha=0.3); ax2.set_ylim(-0.02, 1.0)

fig.tight_layout()
out = os.path.join(HERE, "pB_alpha_response.png")
fig.savefig(out, dpi=130, bbox_inches="tight")
print("wrote", out)
