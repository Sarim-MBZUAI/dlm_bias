#!/usr/bin/env python3
"""E1: trajectory-level analysis of the decode-space PI/PID controller.

Reads the per-item denoising trajectories logged by every decode-PI/PID run
(`pblack_traj`/`ptarget_traj` = P(target letter at the answer slot) at each
denoising step, `alpha_traj` = the injection strength applied at each step)
and answers five questions:

  RQ1  When does the steered answer lock in (early/prior vs late/commit)?
  RQ2  How do trajectories differ between steerable (black, arab) and
       unsteerable (asian, white) targets?
  RQ3  Does lock-in happen before or after the controller first saturates?
  RQ4  How do success trajectories differ from failure trajectories
       (STRICT-parsed outcome classes)?
  RQ5  Is there a trajectory-level signature separating PI / PID / steps=32?

Definitions (all robustness variants are computed and written to CSV):

  STRICT outcome   Replicates balanced_all/strict_pool.py exactly: a response
                   is valid only if an A/B/C letter (either case) starts
                   `model_output` (leading whitespace allowed) followed by
                   whitespace/punctuation/end.  Valid letters classify via
                   target_idx (or black_idx) / unk_idx into
                   target / abstain / comparator; else invalid.

  lock-in step     t_lock(th) = the first step t such that p(s) >= th for ALL
                   s >= t (i.e. the last upward crossing of th that is never
                   undone).  Primary th = 0.5; robustness th in
                   {0.3, 0.5, 0.7, 0.9}.  Undefined when p_final < th.

  freeze step      t_freeze(eps) = 1 + the last step with |p(t+1)-p(t)| >= eps
                   (0 if the trajectory never moves by eps).  Proxy for the
                   irreversible commitment of the answer-slot token: once
                   LLaDA transfers that token, the measured distribution
                   pins.  Primary eps = 0.02; robustness {0.01, 0.05}.
                   Unlike t_lock it is defined for every item, including
                   failures (where p collapses and freezes near 0).

  first pin step   t_pin = the first step with alpha(t) >= amax (the upper
                   actuator clamp; amax = 6.0 for LLaDA runs, 1.0 for Dream).
                   Computed from alpha_traj directly ("pinned at the ceiling"),
                   NOT from the logged sat flag (which also counts lower-clamp
                   raw < 0 events).

Trajectory conventions:
  LLaDA runs (steering/denoise_pid.py, multirace/denoise_pid.py):
      alpha_traj[t] is APPLIED at step t and pblack_traj[t] is measured from
      the same forward pass, so the two are aligned.
  Dream runs (dream/denoise_pid.py):
      alpha_traj[t] is the command issued AFTER observing pblack_traj[t]; it
      drives forward t+1.  We re-align by shifting: applied[t] = cmd[t-1],
      applied[0] = 0 (the step-0 forward is an unsteered probe).  The last
      command (cmd[63]) is never applied and is dropped.

Base (unsteered) runs from balanced_seeds/*/base, balanced_all/*/base and
dream_balanced/base log the same trajectories with alpha == 0 and are used as
the no-steering reference.  The round-1 black base
(results/balanced/results_balanced/rot*/cond_base_samples.jsonl) predates
trajectory logging and has none; the seed bases stand in for it.

Outputs (all under analysis/trajectory/):
  per_item.csv          one row per item per run: outcome + all derived steps
  summary_runs.csv      per-run outcome rates + trajectory telemetry
  summary_lockin.csv    lock-in stats per run x threshold (+ per-rot medians)
  summary_ordering.csv  freeze-vs-first-pin ordering stats per run x class
  figs/fig_*.pdf/.png   the paper figures (Okabe-Ito, colorblind-safe)
  stdout                an ASCII summary of every headline number

Usage:
  python analysis/trajectory/traj_analysis.py [--no-figs]
Rerunnable from anywhere; paths resolve from DLM_BIAS_ROOT or the repo root
two levels above this file.  CPU-only, needs numpy (+ matplotlib for figures).
"""
import argparse
import csv
import json
import os
import re
import sys
from collections import Counter

import numpy as np

ROOT = os.environ.get("DLM_BIAS_ROOT") or os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_DIR = os.path.join(ROOT, "analysis", "trajectory")
FIG_DIR = os.path.join(OUT_DIR, "figs")

# ---------------------------------------------------------------------------
# STRICT parser (verbatim from balanced_all/strict_pool.py)
# ---------------------------------------------------------------------------
LETTERS = "ABC"
STRICT_RE = re.compile(r"^\s*([ABCabc])(?:$|[\s.,:;)\]!?'\"-])")


def strict_letter(text):
    m = STRICT_RE.match(text or "")
    return m.group(1) if m else None          # keep case (upper/lower matters)


def classify(row):
    """(strict_class, letter_or_None). Classes: target/abstain/comparator/invalid."""
    raw = strict_letter(row.get("model_output", ""))
    if raw is None:
        return "invalid", None
    idx = LETTERS.index(raw.upper())
    tgt = row["target_idx"] if "target_idx" in row else row["black_idx"]
    if idx == tgt:
        return "target", raw
    if idx == row["unk_idx"]:
        return "abstain", raw
    return "comparator", raw


# ---------------------------------------------------------------------------
# Run registry.  Every entry: paths per rotation, actuator ceiling, model,
# and whether alpha_traj needs the Dream one-step re-alignment.
# ---------------------------------------------------------------------------
def _r3(fmt):
    return [(r, fmt % r) for r in range(3)]


RUNS = [
    # key                 kind     model   amax  steps dream? paths[(rot,relpath)]
    ("black_PI",          "steer", "llada", 6.0, 64, False,
     _r3("results/balanced/results_balanced/rot%d/cond_dpid_PI_samples.jsonl")),
    ("black_PID",         "steer", "llada", 6.0, 64, False,
     _r3("results/balanced_all/black/decode_pid/rot%d/cond_dpid_PID_samples.jsonl")),
    ("black_PI_s32",      "steer", "llada", 6.0, 32, False,
     _r3("results/balanced_all/black/decode_pid_s32/rot%d/cond_dpid_PI_s32_samples.jsonl")),
    ("black_PI_seed1",    "steer", "llada", 6.0, 64, False,
     _r3("results/balanced_seeds/seed1/decode_pid/rot%d/cond_dpid_PI_samples.jsonl")),
    ("black_PI_seed2",    "steer", "llada", 6.0, 64, False,
     _r3("results/balanced_seeds/seed2/decode_pid/rot%d/cond_dpid_PI_samples.jsonl")),
    ("black_PI_seed3",    "steer", "llada", 6.0, 64, False,
     _r3("results/balanced_seeds/seed3/decode_pid/rot%d/cond_dpid_PI_samples.jsonl")),
    ("black_base_seed1",  "base",  "llada", 6.0, 64, False,
     _r3("results/balanced_seeds/seed1/base/rot%d/cond_dpid_base_samples.jsonl")),
    ("black_base_seed2",  "base",  "llada", 6.0, 64, False,
     _r3("results/balanced_seeds/seed2/base/rot%d/cond_dpid_base_samples.jsonl")),
    ("black_base_seed3",  "base",  "llada", 6.0, 64, False,
     _r3("results/balanced_seeds/seed3/base/rot%d/cond_dpid_base_samples.jsonl")),
    ("arab_PI",           "steer", "llada", 6.0, 64, False,
     _r3("results/balanced_all/arab/decode_pid/rot%d/cond_dpid_arab_PI_samples.jsonl")),
    ("asian_PI",          "steer", "llada", 6.0, 64, False,
     _r3("results/balanced_all/asian/decode_pid/rot%d/cond_dpid_asian_PI_samples.jsonl")),
    ("latino_PI",         "steer", "llada", 6.0, 64, False,
     _r3("results/balanced_all/latino/decode_pid/rot%d/cond_dpid_latino_PI_samples.jsonl")),
    ("white_PI",          "steer", "llada", 6.0, 64, False,
     _r3("results/balanced_all/white/decode_pid/rot%d/cond_dpid_white_PI_samples.jsonl")),
    ("arab_base",         "base",  "llada", 6.0, 64, False,
     _r3("results/balanced_all/arab/base/rot%d/cond_dpid_arab_base_samples.jsonl")),
    ("asian_base",        "base",  "llada", 6.0, 64, False,
     _r3("results/balanced_all/asian/base/rot%d/cond_dpid_asian_base_samples.jsonl")),
    ("latino_base",       "base",  "llada", 6.0, 64, False,
     _r3("results/balanced_all/latino/base/rot%d/cond_dpid_latino_base_samples.jsonl")),
    ("white_base",        "base",  "llada", 6.0, 64, False,
     _r3("results/balanced_all/white/base/rot%d/cond_dpid_white_base_samples.jsonl")),
    ("dream_PI",          "steer", "dream", 1.0, 64, True,
     _r3("results/dream_balanced/decode_pid/rot%d/cond_dpid_PI_samples.jsonl")),
    ("dream_base",        "base",  "dream", 1.0, 64, True,
     _r3("results/dream_balanced/base/rot%d/cond_dpid_base_samples.jsonl")),
]

THETAS = (0.3, 0.5, 0.7, 0.9)   # lock-in thresholds; 0.5 is primary
EPSES = (0.01, 0.02, 0.05)      # freeze epsilons; 0.02 is primary
PIN_TOL = 1e-6


# ---------------------------------------------------------------------------
# Per-item metric extraction
# ---------------------------------------------------------------------------
def lock_step(p, theta):
    """First t such that p[s] >= theta for every s >= t; None if p[-1] < theta."""
    above = p >= theta
    if not above[-1]:
        return None
    below = np.flatnonzero(~above)
    return 0 if below.size == 0 else int(below[-1] + 1)


def freeze_step(p, eps):
    """1 + last step with |dp| >= eps (0 if the trajectory never moves)."""
    moved = np.flatnonzero(np.abs(np.diff(p)) >= eps)
    return 0 if moved.size == 0 else int(moved[-1] + 1)


def first_pin(a, amax):
    w = np.flatnonzero(a >= amax - PIN_TOL)
    return int(w[0]) if w.size else None


def load_run(key, kind, model, amax, steps, dream, paths):
    items = []
    for rot, rel in paths:
        path = os.path.join(ROOT, rel)
        if not os.path.exists(path):
            print(f"[warn] MISSING {rel} -- run {key} rot{rot} skipped", file=sys.stderr)
            continue
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                p = np.asarray(row.get("pblack_traj") or row["ptarget_traj"], dtype=float)
                a = np.asarray(row["alpha_traj"], dtype=float)
                if dream:
                    # re-align: command at t drives forward t+1 (see module doc)
                    a = np.concatenate(([0.0], a[:-1]))
                assert len(p) == steps and len(a) == steps, (key, len(p), len(a))
                cls, letter = classify(row)
                it = {
                    "run": key, "kind": kind, "model": model, "amax": amax,
                    "steps": steps, "rot": rot,
                    "example_id": row["example_id"],
                    "target_letter": row["target_letter"],
                    "context_condition": row.get("context_condition", ""),
                    "question_polarity": row.get("question_polarity", ""),
                    "strict_class": cls,
                    "strict_letter": letter or "",
                    "letter_lower": bool(letter and letter.islower()),
                    "p0": float(p[0]), "p_final": float(p[-1]),
                    "p_peak": float(p.max()), "t_peak": int(p.argmax()),
                    "alpha_mean": float(a.mean()), "alpha_final": float(a[-1]),
                    "pin_frac": float((a >= amax - PIN_TOL).mean()),
                    "p": p, "a": a,
                }
                fp = first_pin(a, amax)
                it["first_pin"] = fp
                it["ever_pin"] = fp is not None
                for th in THETAS:
                    it[f"lock{th}"] = lock_step(p, th)
                for eps in EPSES:
                    it[f"freeze{eps}"] = freeze_step(p, eps)
                items.append(it)
    return items


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------
CLASSES = ("target", "comparator", "abstain", "invalid")


def q(x, v):
    return float(np.percentile(x, v)) if len(x) else float("nan")


def fmt(x, nd=3):
    return "nan" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def outcome_summary(items):
    n = len(items)
    cnt = Counter(it["strict_class"] for it in items)
    row = {"n": n}
    for c in CLASSES:
        row[c] = cnt.get(c, 0) / n if n else float("nan")
    tgt = [it for it in items if it["strict_class"] == "target"]
    row["target_lower_frac"] = (np.mean([it["letter_lower"] for it in tgt])
                                if tgt else float("nan"))
    # sensor-visible successes: strict target AND the sensor saw it commit
    row["target_pfinal_ge05"] = (np.mean([it["p_final"] >= 0.5 for it in tgt])
                                 if tgt else float("nan"))
    for c in CLASSES:
        sub = [it["p_final"] for it in items if it["strict_class"] == c]
        row[f"pfinal_med_{c}"] = float(np.median(sub)) if sub else float("nan")
    row["alpha_mean"] = float(np.mean([it["alpha_mean"] for it in items]))
    row["pin_frac_mean"] = float(np.mean([it["pin_frac"] for it in items]))
    row["ever_pin_frac"] = float(np.mean([it["ever_pin"] for it in items]))
    return row


def lockin_summary(items, theta, steps):
    ls = [it[f"lock{theta}"] for it in items if it[f"lock{theta}"] is not None]
    n = len(items)
    out = {"theta": theta, "n_items": n, "n_locked": len(ls)}
    if not ls:
        out.update({k: float("nan") for k in
                    ("lock_med", "lock_q10", "lock_q90",
                     "frac_early", "frac_wave", "frac_late")})
        return out
    ls = np.asarray(ls, float)
    e, w = steps / 8.0, 3 * steps / 8.0        # early <= 1/8; wave in [3/8, 5/8]
    out.update({
        "lock_med": float(np.median(ls)), "lock_q10": q(ls, 10), "lock_q90": q(ls, 90),
        "frac_early": float((ls <= e).mean()),
        "frac_wave": float(((ls >= w) & (ls <= 5 * steps / 8.0)).mean()),
        "frac_late": float((ls > 5 * steps / 8.0).mean()),
    })
    return out


def ordering_summary(items, cls_group, amax_key="first_pin"):
    """freeze(0.02) vs first-pin ordering inside one outcome group."""
    sub = [it for it in items if it["strict_class"] in cls_group]
    ever = [it for it in sub if it["ever_pin"]]
    out = {"n": len(sub), "ever_pin_frac": len(ever) / len(sub) if sub else float("nan")}
    if ever:
        d = np.array([it["freeze0.02"] - it[amax_key] for it in ever], float)
        out.update({"n_pin": len(ever),
                    "frac_freeze_before_pin": float((d < 0).mean()),
                    "med_pin_minus_freeze": float(np.median(-d)),
                    "freeze_med": float(np.median([it["freeze0.02"] for it in ever])),
                    "pin_med": float(np.median([it[amax_key] for it in ever]))})
    else:
        out.update({"n_pin": 0, "frac_freeze_before_pin": float("nan"),
                    "med_pin_minus_freeze": float("nan"),
                    "freeze_med": float("nan"), "pin_med": float("nan")})
    return out


def mean_band(trajs):
    """(mean, 95% normal CI half-width) per step."""
    M = np.stack(trajs)
    mu = M.mean(0)
    hw = 1.96 * M.std(0, ddof=1) / np.sqrt(M.shape[0])
    return mu, hw


# ---------------------------------------------------------------------------
# Figures (Okabe-Ito; palette orderings validated for CVD separation)
# ---------------------------------------------------------------------------
C_BLUE, C_ORANGE, C_GREEN, C_SKY, C_PURPLE = (
    "#0072B2", "#E69F00", "#009E73", "#56B4E9", "#CC79A7")
C_VERM, C_GRAY = "#D55E00", "#7F7F7F"

TARGET_COLOR = {"black": C_BLUE, "arab": C_ORANGE, "latino": C_GREEN,
                "white": C_SKY, "asian": C_PURPLE}
CLASS_COLOR = {"target": C_BLUE, "comparator": C_VERM,
               "abstain": C_GREEN, "invalid": C_PURPLE}


def style_ax(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.6)
    ax.set_axisbelow(True)


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG_DIR, f"{name}.{ext}"),
                    bbox_inches="tight", dpi=200)
    import matplotlib.pyplot as plt
    plt.close(fig)
    print(f"[fig] {name}.pdf/.png")


def xfrac(steps):
    return np.arange(steps) / (steps - 1)


def make_figures(data):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9,
        "legend.fontsize": 8, "legend.frameon": False,
        "axes.linewidth": 0.8, "lines.linewidth": 1.8,
        "xtick.labelsize": 8, "ytick.labelsize": 8,
    })
    os.makedirs(FIG_DIR, exist_ok=True)

    def band(ax, run, color, label, ls="-", key="p"):
        items = data[run]
        mu, hw = mean_band([it[key] for it in items])
        x = xfrac(items[0]["steps"])
        ax.plot(x, mu, color=color, label=label, linestyle=ls)
        ax.fill_between(x, mu - hw, mu + hw, color=color, alpha=0.18, linewidth=0)

    # ---- Figure 1: mean p_target(t) per condition -------------------------
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.7))
    ax = axes[0]
    seeds_base = data["black_base_seed1"] + data["black_base_seed2"] + data["black_base_seed3"]
    mu, hw = mean_band([it["p"] for it in seeds_base])
    ax.plot(xfrac(64), mu, color=C_GRAY, linestyle="--", label="base (unsteered, seeds)")
    ax.fill_between(xfrac(64), mu - hw, mu + hw, color=C_GRAY, alpha=0.18, linewidth=0)
    band(ax, "black_PI", C_BLUE, "decode-PI (64 steps)")
    band(ax, "black_PID", C_ORANGE, "decode-PID")
    band(ax, "black_PI_s32", C_GREEN, "decode-PI (32 steps)")
    ax.set_title("Black target (LLaDA)")
    ax.set_xlabel("decode progress  t / (T−1)")
    ax.set_ylabel("mean  $p_{\\mathrm{target}}(t)$")
    ax.set_ylim(0, 0.55)
    ax.legend(loc="upper left")
    style_ax(ax)

    ax = axes[1]
    band(ax, "black_PI", C_BLUE, "black")
    band(ax, "arab_PI", C_ORANGE, "arab")
    band(ax, "latino_PI", C_GREEN, "latino")
    band(ax, "white_PI", C_SKY, "white")
    band(ax, "asian_PI", C_PURPLE, "asian")
    ax.set_title("decode-PI by target (LLaDA, amax=6)")
    ax.set_xlabel("decode progress  t / (T−1)")
    ax.set_ylim(0, 0.55)
    ax.legend(loc="upper left", ncol=2)
    style_ax(ax)
    fig.tight_layout()
    save(fig, "fig_p_mean")

    # ---- Figure 2: lock-in step distributions -----------------------------
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.7))
    ax = axes[0]
    pooled = {"decode-PI (round 1)": (data["black_PI"], C_BLUE),
              "decode-PI (seeds 1-3)": (data["black_PI_seed1"] + data["black_PI_seed2"]
                                        + data["black_PI_seed3"], C_SKY),
              "base (seeds 1-3)": (seeds_base, C_GRAY)}
    bins = np.arange(0, 68, 4)
    for label, (items, color) in pooled.items():
        ls = [it["lock0.5"] for it in items if it["lock0.5"] is not None]
        ax.hist(ls, bins=bins, density=True, histtype="step",
                color=color, label=f"{label}  (n={len(ls)})", linewidth=1.8)
    ax.set_title("lock-in step, $\\theta=0.5$ (black target)")
    ax.set_xlabel("lock-in step $t_{\\mathrm{lock}}$ (of 64)")
    ax.set_ylabel("density")
    ax.legend(loc="upper left")
    style_ax(ax)

    ax = axes[1]
    shades = ["#B7D8EC", "#6FAED2", "#2E86B5", "#00507E"]  # light->dark one-hue ramp
    for th, color in zip(THETAS, shades):
        ls = sorted(it[f"lock{th}"] for it in data["black_PI"]
                    if it[f"lock{th}"] is not None)
        if not ls:
            continue
        y = np.arange(1, len(ls) + 1) / len(ls)
        ax.step(ls, y, where="post", color=color,
                label=f"$\\theta$={th} (n={len(ls)})", linewidth=1.6)
    ax.set_title("robustness to threshold (decode-PI)")
    ax.set_xlabel("lock-in step $t_{\\mathrm{lock}}$ (of 64)")
    ax.set_ylabel("ECDF among locked items")
    ax.legend(loc="upper left")
    style_ax(ax)
    fig.tight_layout()
    save(fig, "fig_lockin")

    # ---- Figure 3: mean alpha(t) by target (steerable vs unsteerable) -----
    fig, ax = plt.subplots(figsize=(3.7, 2.7))
    for run, label, ls in [("black_PI", "black", "-"), ("arab_PI", "arab", "-"),
                           ("latino_PI", "latino", "-."), ("white_PI", "white", "--"),
                           ("asian_PI", "asian", "--")]:
        band(ax, run, TARGET_COLOR[label], label, ls=ls, key="a")
    ax.axhline(6.0, color="#BBBBBB", linewidth=0.8, linestyle=":")
    ax.text(0.01, 6.05, "$\\alpha_{\\max}$", color="#888888", fontsize=8)
    ax.set_title("controller effort by target (decode-PI)")
    ax.set_xlabel("decode progress  t / (T−1)")
    ax.set_ylabel("mean applied $\\alpha(t)$")
    ax.set_ylim(0, 6.6)
    ax.legend(loc="lower right", ncol=2)
    style_ax(ax)
    fig.tight_layout()
    save(fig, "fig_alpha_targets")

    # ---- Figure 4: success vs failure (black decode-PI) -------------------
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.7))
    for ax, key, ylab, title in [
            (axes[0], "p", "mean  $p_{\\mathrm{target}}(t)$", "sensor signal by outcome"),
            (axes[1], "a", "mean applied $\\alpha(t)$", "controller effort by outcome")]:
        for c in CLASSES:
            items = [it for it in data["black_PI"] if it["strict_class"] == c]
            if not items:
                continue
            mu, hw = mean_band([it[key] for it in items])
            x = xfrac(64)
            ax.plot(x, mu, color=CLASS_COLOR[c], label=f"{c} (n={len(items)})")
            ax.fill_between(x, mu - hw, mu + hw, color=CLASS_COLOR[c],
                            alpha=0.18, linewidth=0)
        ax.set_xlabel("decode progress  t / (T−1)")
        ax.set_ylabel(ylab)
        ax.set_title(title + "  (black decode-PI)")
        style_ax(ax)
    axes[0].legend(loc="upper left")   # shared class legend for both panels
    axes[1].axhline(6.0, color="#BBBBBB", linewidth=0.8, linestyle=":")
    axes[1].set_ylim(0, 6.6)
    fig.tight_layout()
    save(fig, "fig_success_failure")

    # ---- Figure 5: commitment vs first saturation scatter -----------------
    fig, ax = plt.subplots(figsize=(3.7, 3.2))
    rng = np.random.default_rng(0)
    for run, color, label, marker in [("black_PI", C_BLUE, "black failures", "o"),
                                      ("arab_PI", C_ORANGE, "arab failures", "^")]:
        sub = [it for it in data[run]
               if it["strict_class"] != "target" and it["ever_pin"]]
        x = np.array([it["freeze0.02"] for it in sub], float) + rng.uniform(-.35, .35, len(sub))
        y = np.array([it["first_pin"] for it in sub], float) + rng.uniform(-.35, .35, len(sub))
        ax.scatter(x, y, s=7, alpha=0.35, color=color, label=label,
                   marker=marker, linewidths=0)
    lim = 64
    ax.plot([0, lim], [0, lim], color="#888888", linewidth=0.8, linestyle="--")
    ax.text(38, 20, "saturates before\ncommitment", fontsize=8, color="#555555")
    ax.text(2, 52, "saturates after\ncommitment", fontsize=8, color="#555555")
    ax.set_xlabel("freeze step $t_{\\mathrm{freeze}}$ (commitment proxy)")
    ax.set_ylabel("first saturation step $t_{\\mathrm{pin}}$")
    ax.set_title("commitment vs first saturation")
    ax.set_xlim(-1, lim)
    ax.set_ylim(-1, lim)
    ax.legend(loc="lower right")
    style_ax(ax)
    fig.tight_layout()
    save(fig, "fig_commit_vs_sat")

    # ---- Figure 6: PI vs PID signature ------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.7))
    ax = axes[0]
    band(ax, "black_PI", C_BLUE, "PI", key="a")
    band(ax, "black_PID", C_ORANGE, "PID (Kd=1)", key="a")
    ax.set_title("mean applied $\\alpha(t)$, PI vs PID (black)")
    ax.set_xlabel("decode progress  t / (T−1)")
    ax.set_ylabel("mean applied $\\alpha(t)$")
    ax.legend(loc="lower right")
    style_ax(ax)

    ax = axes[1]
    for run, color, label in [("black_PI", C_BLUE, "PI"),
                              ("black_PID", C_ORANGE, "PID (Kd=1)")]:
        rough = [float(np.abs(np.diff(it["a"])).mean()) for it in data[run]]
        xs = sorted(rough)
        ax.step(xs, np.arange(1, len(xs) + 1) / len(xs), where="post",
                color=color, label=label, linewidth=1.8)
    ax.set_title("actuation roughness  mean$_t\\,|\\Delta\\alpha(t)|$")
    ax.set_xlabel("per-item mean $|\\Delta\\alpha|$")
    ax.set_ylabel("ECDF")
    ax.legend(loc="lower right")
    style_ax(ax)
    fig.tight_layout()
    save(fig, "fig_pi_vs_pid")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--no-figs", action="store_true", help="skip matplotlib figures")
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    data = {}
    for key, kind, model, amax, steps, dream, paths in RUNS:
        items = load_run(key, kind, model, amax, steps, dream, paths)
        if items:
            data[key] = items
            print(f"[load] {key:18s} n={len(items)}")
        else:
            print(f"[warn] run {key} has no data; dropped", file=sys.stderr)

    # ---- per_item.csv ------------------------------------------------------
    cols = ["run", "kind", "model", "steps", "amax", "rot", "example_id",
            "target_letter", "context_condition", "question_polarity",
            "strict_class", "strict_letter", "letter_lower",
            "p0", "p_final", "p_peak", "t_peak",
            "alpha_mean", "alpha_final", "pin_frac", "ever_pin", "first_pin"] \
        + [f"lock{th}" for th in THETAS] + [f"freeze{e}" for e in EPSES]
    with open(os.path.join(OUT_DIR, "per_item.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for key in data:
            for it in data[key]:
                w.writerow([it[c] if it[c] is not None else "" for c in cols])
    print(f"[csv] per_item.csv ({sum(len(v) for v in data.values())} rows)")

    # ---- summary_runs.csv --------------------------------------------------
    run_rows = []
    for key in data:
        row = {"run": key}
        row.update(outcome_summary(data[key]))
        run_rows.append(row)
    with open(os.path.join(OUT_DIR, "summary_runs.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(run_rows[0].keys()))
        w.writeheader()
        w.writerows(run_rows)
    print("[csv] summary_runs.csv")

    # ---- summary_lockin.csv ------------------------------------------------
    lock_rows = []
    for key in data:
        steps = data[key][0]["steps"]
        for th in THETAS:
            row = {"run": key}
            row.update(lockin_summary(data[key], th, steps))
            for rot in range(3):
                sub = [it for it in data[key] if it["rot"] == rot]
                ls = [it[f"lock{th}"] for it in sub if it[f"lock{th}"] is not None]
                row[f"lock_med_rot{rot}"] = float(np.median(ls)) if ls else float("nan")
            lock_rows.append(row)
    with open(os.path.join(OUT_DIR, "summary_lockin.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(lock_rows[0].keys()))
        w.writeheader()
        w.writerows(lock_rows)
    print("[csv] summary_lockin.csv")

    # ---- summary_ordering.csv ----------------------------------------------
    ord_rows = []
    for key in data:
        if data[key][0]["kind"] != "steer":
            continue
        for label, grp in [("success", ("target",)),
                           ("failure", ("comparator", "abstain", "invalid"))]:
            row = {"run": key, "group": label}
            row.update(ordering_summary(data[key], grp))
            ord_rows.append(row)
    with open(os.path.join(OUT_DIR, "summary_ordering.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(ord_rows[0].keys()))
        w.writeheader()
        w.writerows(ord_rows)
    print("[csv] summary_ordering.csv")

    # ---- ASCII report ------------------------------------------------------
    print("\n" + "=" * 78)
    print("OUTCOMES (STRICT) + telemetry")
    print("=" * 78)
    hdr = ("run", "n", "tgt", "comp", "abst", "inval", "tgt_lc",
           "a_mean", "pin_fr", "ever_pin")
    print("%-18s %5s %6s %6s %6s %6s %6s %6s %6s %8s" % hdr)
    for r in run_rows:
        print("%-18s %5d %6s %6s %6s %6s %6s %6s %6s %8s" % (
            r["run"], r["n"], fmt(r["target"]), fmt(r["comparator"]),
            fmt(r["abstain"]), fmt(r["invalid"]), fmt(r["target_lower_frac"]),
            fmt(r["alpha_mean"], 2), fmt(r["pin_frac_mean"], 2),
            fmt(r["ever_pin_frac"], 2)))

    print("\n" + "=" * 78)
    print("LOCK-IN (theta=0.5): median [q10,q90], early(<=T/8) / wave(3T/8..5T/8) / late")
    print("=" * 78)
    for row in lock_rows:
        if row["theta"] != 0.5:
            continue
        print("%-18s locked %4d/%4d  med %4s [%4s,%4s]  early %5s wave %5s late %5s"
              "  rot-med %s/%s/%s" % (
                  row["run"], row["n_locked"], row["n_items"],
                  fmt(row["lock_med"], 0), fmt(row["lock_q10"], 0),
                  fmt(row["lock_q90"], 0), fmt(row["frac_early"], 2),
                  fmt(row["frac_wave"], 2), fmt(row["frac_late"], 2),
                  fmt(row["lock_med_rot0"], 0), fmt(row["lock_med_rot1"], 0),
                  fmt(row["lock_med_rot2"], 0)))

    print("\n" + "=" * 78)
    print("ORDERING: freeze(0.02) vs first alpha pin (per outcome group)")
    print("=" * 78)
    for row in ord_rows:
        print("%-18s %-8s n=%4d ever_pin %5s | n_pin %4d  freeze<pin %5s  "
              "med(pin-freeze) %5s  (freeze med %4s, pin med %4s)" % (
                  row["run"], row["group"], row["n"], fmt(row["ever_pin_frac"], 2),
                  row["n_pin"], fmt(row["frac_freeze_before_pin"], 2),
                  fmt(row["med_pin_minus_freeze"], 1), fmt(row["freeze_med"], 0),
                  fmt(row["pin_med"], 0)))

    # ---- extra stats used in FINDINGS.md ------------------------------------
    print("\n" + "=" * 78)
    print("EXTRAS")
    print("=" * 78)
    # early-lock composition: target letter position of early vs wave locks
    for key in ("black_PI", "black_PI_seed1", "black_PI_seed2", "black_PI_seed3"):
        if key not in data:
            continue
        locked = [(it["lock0.5"], it["target_letter"]) for it in data[key]
                  if it["lock0.5"] is not None]
        early = [L for t, L in locked if t <= 8]
        late = [L for t, L in locked if t > 8]
        fe = np.mean([L == "A" for L in early]) if early else float("nan")
        fl = np.mean([L == "A" for L in late]) if late else float("nan")
        print(f"{key}: early locks n={len(early)} frac(target@A)={fmt(fe,2)} | "
              f"later locks n={len(late)} frac(target@A)={fmt(fl,2)}")
    # jump size at lock (is lock a discrete commit?)
    for key in ("black_PI", "black_PID"):
        jumps = [it["p"][it["lock0.5"]] - it["p"][it["lock0.5"] - 1]
                 for it in data.get(key, []) if it.get("lock0.5")]
        if jumps:
            print(f"{key}: p-jump at lock med {fmt(np.median(jumps),2)} "
                  f"(q25 {fmt(q(jumps,25),2)}, q75 {fmt(q(jumps,75),2)})")
    # PI vs PID paired-by-item agreement
    if "black_PI" in data and "black_PID" in data:
        A = {(it["rot"], it["example_id"]): it for it in data["black_PI"]}
        B = {(it["rot"], it["example_id"]): it for it in data["black_PID"]}
        keys = sorted(set(A) & set(B))
        agree = np.mean([A[k]["strict_class"] == B[k]["strict_class"] for k in keys])
        both_t = [k for k in keys if A[k]["strict_class"] == "target"
                  and B[k]["strict_class"] == "target"]
        dl = [B[k]["lock0.5"] - A[k]["lock0.5"] for k in both_t
              if A[k]["lock0.5"] is not None and B[k]["lock0.5"] is not None]
        ra = float(np.mean([np.abs(np.diff(A[k]["a"])).mean() for k in keys]))
        rb = float(np.mean([np.abs(np.diff(B[k]["a"])).mean() for k in keys]))
        print(f"PI vs PID paired (n={len(keys)}): class agreement {fmt(agree,3)}; "
              f"both-target n={len(both_t)}, lock-step diff med {fmt(np.median(dl),1)} "
              f"(q25 {fmt(q(dl,25),1)}, q75 {fmt(q(dl,75),1)}); "
              f"roughness mean|dA| PI {fmt(ra,3)} vs PID {fmt(rb,3)}")
    # p at commit-wave onset: pre-commit lift under steering (base vs steered)
    for base_key, steer_key in [("black_base_seed1", "black_PI_seed1"),
                                ("black_base_seed2", "black_PI_seed2"),
                                ("black_base_seed3", "black_PI_seed3")]:
        if base_key in data and steer_key in data:
            mb = np.mean([it["p"][:24].mean() for it in data[base_key]])
            ms = np.mean([it["p"][:24].mean() for it in data[steer_key]])
            print(f"pre-commit mean p (t<24): {base_key} {fmt(mb,3)} vs "
                  f"{steer_key} {fmt(ms,3)}")

    if not args.no_figs:
        try:
            make_figures(data)
        except ImportError as e:
            print(f"[warn] matplotlib unavailable ({e}); CSVs written, figures skipped",
                  file=sys.stderr)


if __name__ == "__main__":
    main()
