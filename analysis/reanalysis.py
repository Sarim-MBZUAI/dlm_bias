"""Post-hoc analyses of the BBQ outputs (no new generations).

- item-clustered bootstrap: resample the 400 source items, keeping all
  three rotations of an item together (prompt-level CIs shown alongside).
- held-out rescoring of the Black target: drop the 100 calibration items
  (the first 100 rows of the evaluation pool) and rescore on the other 300.
- within-sequence controller adaptation: how much of the PI command over
  the 32 token-committing steps is explained by the item's first command
  alpha_1, which depends only on the unsteered probe p_0.

No new generations. Uses the repository's strict parser.
Usage:  python analysis/reanalysis.py
"""
import glob
import importlib.util
import json
import os
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("sp", ROOT / "balanced_all/strict_pool.py")
sp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sp)

N_BOOT = 10_000


def key(r):
    return (str(r["category"]), str(r["example_id"]), str(r["question_index"]))


def read(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def load_condition(paths):
    """paths: 3 rotation files -> dict key -> [row_rot0, row_rot1, row_rot2]."""
    by_key = {}
    for rot, p in enumerate(paths):
        for r in read(p):
            by_key.setdefault(key(r), [None, None, None])[rot] = r
    assert all(all(x is not None for x in v) for v in by_key.values()), paths[0]
    return by_key


def sign(r):
    c = sp.classify_letter(sp.strict_letter(r.get("model_output", "")), r)
    return {"target": 1, "comparator": -1}.get(c, 0), c == "invalid"


ROOTS = {"dream": "results/dream_balanced", "lladamoe": "results/lladamoe_balanced"}


def target_dir(target):
    return ROOT / ROOTS[target] if target in ROOTS else ROOT / "results/balanced_all" / target


def discover(target):
    """label -> 3 rotation paths for every complete condition of a target."""
    conds = {}
    base_dir = target_dir(target)
    for f in sorted(glob.glob(str(base_dir / "*/rot0/cond_*_samples.jsonl"))):
        method = Path(f).parts[-3]
        stem = Path(f).name
        paths = [f.replace("/rot0/", f"/rot{r}/") for r in range(3)]
        if all(os.path.exists(p) for p in paths):
            label = "base" if method == "base" else f"{method}:{stem[5:-14]}"
            conds[label] = paths
    if target == "black":
        old = lambda s: [str(ROOT / f"results/balanced/results_balanced/rot{r}/cond_{s}_samples.jsonl") for r in range(3)]
        conds["base"] = old("base")
        conds["layer_pi:PI_a2"] = old("PI_a2")
        conds["open_loop:normalL14_a4"] = old("normalL14_a4")
        conds["decode_pid:dpid_PI"] = old("dpid_PI")
    return conds


def matrices(conds, order=None):
    data = {k: load_condition(v) for k, v in conds.items()}
    if order is None:
        order = sorted(data["base"])
    out = {}
    for k, d in data.items():
        assert set(d) == set(order), (k, len(d), len(order))
        S = np.array([[sign(r)[0] for r in d[q]] for q in order], dtype=float)
        I = np.array([[sign(r)[1] for r in d[q]] for q in order], dtype=float)
        out[k] = (S, I, d)
    return out, order


def ci(x):
    return [float(v) for v in np.quantile(x, [0.025, 0.975])]


def summarize(mats, pi_label, idx_mask=None):
    """Gap, prompt-level and item-level CIs, paired contrasts."""
    keys = list(mats)
    n_all = len(next(iter(mats.values()))[0])
    sel = np.arange(n_all) if idx_mask is None else np.flatnonzero(idx_mask)
    n = len(sel)
    rng_i = np.random.default_rng(0).integers(0, n, size=(N_BOOT, n))
    rng_p = np.random.default_rng(0).integers(0, 3 * n, size=(N_BOOT, 3 * n))
    item = {k: mats[k][0][sel].mean(axis=1) for k in keys}
    prompt = {k: mats[k][0][sel].reshape(-1) for k in keys}
    res = {}
    for k in keys:
        res[k] = {
            "n_items": int(n),
            "gap_pp": 100 * float(item[k].mean()),
            "invalid_pct": 100 * float(mats[k][1][sel].mean()),
            "ci_prompt_pp": [100 * v for v in ci(prompt[k][rng_p].mean(axis=1))],
            "ci_item_pp": [100 * v for v in ci(item[k][rng_i].mean(axis=1))],
        }
        d = item[k] - item["base"]
        res[k]["dg_pp"] = 100 * float(d.mean())
        res[k]["dg_ci_item_pp"] = [100 * v for v in ci(d[rng_i].mean(axis=1))]
        if pi_label in item and k != pi_label:
            d = item[pi_label] - item[k]
            res[k]["pi_minus_this_pp"] = 100 * float(d.mean())
            res[k]["pi_minus_this_ci_item_pp"] = [100 * v for v in ci(d[rng_i].mean(axis=1))]
    return res


def fmt(res, pi_label):
    lines = []
    head = f"{'condition':34s} {'gap':>6s} {'prompt CI':>15s} {'item CI':>15s} {'inv%':>5s} {'dg':>6s} {'dg item CI':>15s} {'PI-this':>7s} {'item CI':>15s}"
    lines.append(head)
    for k, r in sorted(res.items(), key=lambda kv: -kv[1]["gap_pp"]):
        c = lambda v: f"[{v[0]:6.1f},{v[1]:6.1f}]"
        pm = f"{r['pi_minus_this_pp']:7.1f} {c(r['pi_minus_this_ci_item_pp'])}" if "pi_minus_this_pp" in r else ""
        lines.append(f"{k:34s} {r['gap_pp']:6.1f} {c(r['ci_prompt_pp'])} {c(r['ci_item_pp'])} {r['invalid_pct']:5.1f} {r['dg_pp']:6.1f} {c(r['dg_ci_item_pp'])} {pm}")
    return "\n".join(lines)


def pi_label_for(conds, target):
    cands = [k for k in conds if k.startswith("decode_pid:") and k.endswith("_PI")]
    cands = [k for k in cands if "_cs" not in k] or cands
    return cands[0] if cands else None


report = {"scope": __doc__.strip().splitlines()[0]}
text = []

# ---- item-clustered CIs for every target ----
report["item_clustered"] = {}
for target in ["black", "arab", "old", "asian", "latino", "white", "woman", "man", "fblack", "lowses", "highses", "young", "dream", "lladamoe"]:
    if not target_dir(target).exists():
        continue
    conds = discover(target)
    if "base" not in conds:
        continue
    try:
        mats, order = matrices(conds)
    except AssertionError as e:
        text.append(f"\n## {target}: skipped ({e})")
        continue
    pl = pi_label_for(conds, target)
    res = summarize(mats, pl)
    report["item_clustered"][target] = {"pi_label": pl, "conditions": res}
    text.append(f"\n## item-clustered {target}  (PI = {pl}; {len(order)} items x 3 rotations)\n" + fmt(res, pl))
    if target == "black":
        black_mats, black_order, black_pl = mats, order, pl

# ---- Black held-out 300 items ----
cal = [key(r) for r in read(ROOT / "data/bbq_items/_sweep400.jsonl")][:100]  # gain-calibration items = first 100 rows of the pool
pool = [key(r) for r in read(ROOT / "results/balanced/_sweep400_rot0.jsonl")]
assert cal == pool[:100]
cal_set = set(cal)
mask = np.array([q not in cal_set for q in black_order])
res = summarize(black_mats, black_pl, mask)
report["black_heldout300"] = {"n_calibration_items": len(cal_set), "conditions": res}
text.append(f"\n## black, held-out {int(mask.sum())} items (calibration items removed)\n" + fmt(res, black_pl))
res_cal = summarize(black_mats, black_pl, ~mask)
report["black_calibration100"] = {"conditions": {k: {kk: v[kk] for kk in ("gap_pp", "ci_item_pp")} for k, v in res_cal.items()}}
text.append("\n## black, the 100 calibration items only: " + ", ".join(
    f"{k} {v['gap_pp']:.1f}" for k, v in res_cal.items() if k in ("base", black_pl, "open_loop:normalL14_a4", "normal:normalL14_a3p28")))

# additional question draws, with calibration items removed
seeds = {}
for s in (1, 2, 3):
    d = ROOT / f"results/balanced_seeds/seed{s}"
    conds = {"base": sorted(glob.glob(str(d / "base/rot*/cond_*_samples.jsonl"))),
             "pi": sorted(glob.glob(str(d / "decode_pid/rot*/cond_*_samples.jsonl")))}
    if any(len(v) != 3 for v in conds.values()):
        continue
    mats, order = matrices(conds)
    m = np.array([q not in cal_set for q in order])
    full = summarize(mats, "pi")
    held = summarize(mats, "pi", m)
    seeds[f"seed{s}"] = {"n_items": len(order), "n_calibration_overlap": int((~m).sum()),
                         "pi_gap_all_pp": full["pi"]["gap_pp"], "pi_dg_all_pp": full["pi"]["dg_pp"],
                         "pi_gap_heldout_pp": held["pi"]["gap_pp"], "pi_dg_heldout_pp": held["pi"]["dg_pp"],
                         "pi_dg_heldout_ci_item_pp": held["pi"]["dg_ci_item_pp"]}
report["seed_draws_heldout"] = seeds
text.append("\n## additional draws (calibration overlap removed)\n" + "\n".join(
    f"{s}: overlap {v['n_calibration_overlap']}, PI gap all {v['pi_gap_all_pp']:.1f} -> held-out {v['pi_gap_heldout_pp']:.1f}; "
    f"dg held-out {v['pi_dg_heldout_pp']:.1f} {['%.1f' % x for x in v['pi_dg_heldout_ci_item_pp']]}" for s, v in seeds.items()))

# ---- within-sequence adaptation of the PI command ----
def within_sequence_stats(label, mats):
    S, _, d = mats
    rows = [r for q in sorted(d) for r in d[q]]
    A = np.array([r["alpha_traj"][:32] for r in rows], dtype=float)  # steps 1..32
    outcome = np.array([sign(r)[0] for r in rows])
    seq_mean = A.mean(axis=1)
    between = float(seq_mean.var())
    within = float(A.var(axis=1).mean())
    # explain alpha_t from alpha_1 (a static per-item rule knows only p_0)
    ss_res = ss_tot = 0.0
    for t in range(1, 32):
        X = np.c_[np.ones(len(A)), A[:, 0]]
        beta, *_ = np.linalg.lstsq(X, A[:, t], rcond=None)
        ss_res += float(((A[:, t] - X @ beta) ** 2).sum())
        ss_tot += float(((A[:, t] - A[:, t].mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot
    # non-linear check: best static rule = per-step mean of alpha_t within 20 quantile bins of alpha_1
    bins = np.quantile(A[:, 0], np.linspace(0, 1, 21)[1:-1])
    b = np.digitize(A[:, 0], bins)
    ss_res_b = 0.0
    for t in range(1, 32):
        fit = np.zeros(len(A))
        for g in np.unique(b):
            fit[b == g] = A[b == g, t].mean()
        ss_res_b += float(((A[:, t] - fit) ** 2).sum())
    r2_binned = 1 - ss_res_b / ss_tot
    drops = np.diff(A, axis=1)
    frac_decrease = float((drops.min(axis=1) < -0.05).mean())
    rng = A.max(axis=1) - A.min(axis=1)
    # does the trajectory carry outcome information beyond alpha_1?
    r_a1 = float(np.corrcoef(A[:, 0], outcome == 1)[0, 1])
    r_late = float(np.corrcoef(A[:, 16:32].mean(axis=1), outcome == 1)[0, 1])
    return {"n_seq": len(A), "var_between_seq_means": between, "mean_var_within_seq": within,
            "share_within": within / (within + between),
            "r2_alpha_t_from_alpha1_steps2to32": r2, "r2_binned_alpha1": r2_binned,
            "frac_seq_with_command_decrease": frac_decrease,
            "median_within_seq_range": float(np.median(rng)),
            "corr_alpha1_vs_target": r_a1, "corr_mean_alpha_steps17to32_vs_target": r_late}

report["within_sequence"] = {}
for target, info in report["item_clustered"].items():
    pl = info["pi_label"]
    if pl is None:
        continue
    conds = discover(target)
    try:
        mats, _ = matrices({"base": conds["base"], pl: conds[pl]})
        report["within_sequence"][target] = within_sequence_stats(pl, mats[pl])
    except (KeyError, AssertionError):
        pass
text.append("\n## within-sequence PI command (steps 1-32)\n" + "\n".join(
    f"{t:8s} share_within={v['share_within']:.2f}  R2(alpha_t|alpha_1)={v['r2_alpha_t_from_alpha1_steps2to32']:.2f} binned={v['r2_binned_alpha1']:.2f}  "
    f"decrease={v['frac_seq_with_command_decrease']:.2f}  range_med={v['median_within_seq_range']:.2f}  "
    f"corr(a1,tgt)={v['corr_alpha1_vs_target']:.2f} corr(a17-32,tgt)={v['corr_mean_alpha_steps17to32_vs_target']:.2f}"
    for t, v in report["within_sequence"].items()))

out = Path(__file__).with_name("reanalysis.json")
out.write_text(json.dumps(report, indent=1) + "\n")
Path(__file__).with_name("reanalysis.txt").write_text("\n".join(text) + "\n")
print("\n".join(text))
