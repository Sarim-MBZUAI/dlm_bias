"""Score the per-item static-alpha control against decode-time PI (Black, Dream-7B).

Strict parser from balanced_all/strict_pool.py. Paired item-clustered bootstrap
(400 items, 3 rotations kept together, 10,000 resamples, default_rng(0)).
Usage:  python analysis/score_dream.py
"""
import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("sp", ROOT / "balanced_all/strict_pool.py")
sp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sp)

old = lambda s: [ROOT / f"results/balanced/results_balanced/rot{r}/cond_{s}_samples.jsonl" for r in range(3)]
new = lambda d, s: [ROOT / f"results/balanced_all/black/{d}/rot{r}/cond_{s}_samples.jsonl" for r in range(3)]
dr = lambda d, s: [ROOT / f"results/dream_balanced/{d}/rot{r}/cond_{s}_samples.jsonl" for r in range(3)]
CONDS = {
    "Decode PI": dr("decode_pid", "dpid_PI"),
    "Static alpha from p0": dr("static_alpha", "static_p0"),
    "Static alpha, hindsight": dr("static_alpha", "static_hind"),
    "Unsteered base": dr("base", "dpid_base"),
}
key = lambda r: (str(r["category"]), str(r["example_id"]), str(r["question_index"]))


def load(paths):
    by = {}
    for rot, p in enumerate(paths):
        for l in open(p):
            if l.strip():
                r = json.loads(l)
                by.setdefault(key(r), [None] * 3)[rot] = r
    return by


data = {k: load(v) for k, v in CONDS.items()}
order = sorted(data["Decode PI"])
S, INV, ALPHA = {}, {}, {}
for k, d in data.items():
    assert set(d) == set(order), k
    cls = [[sp.classify_letter(sp.strict_letter(r.get("model_output", "")), r) for r in d[q]] for q in order]
    S[k] = np.array([[{"target": 1, "comparator": -1}.get(c, 0) for c in row] for row in cls], float)
    INV[k] = np.array([[c == "invalid" for c in row] for row in cls], float)
    ALPHA[k] = np.array([[float(np.mean(r["alpha_traj"])) if "alpha_traj" in r else np.nan
                          for r in d[q]] for q in order])

idx = np.random.default_rng(0).integers(0, len(order), size=(10_000, len(order)))
ci = lambda v: [100 * x for x in np.quantile(v[idx].mean(axis=1), [0.025, 0.975])]
item = {k: v.mean(axis=1) for k, v in S.items()}
out = {}
print(f"{'condition':28s} {'gap pp':>7s} {'item CI':>16s} {'inv %':>6s} {'mean alpha':>12s} {'PI - this':>9s} {'paired CI':>16s}")
for k in CONDS:
    g = 100 * item[k].mean()
    row = {"gap_pp": g, "ci_item_pp": ci(item[k]), "invalid_pct": 100 * INV[k].mean(),
           "mean_alpha_all_steps": float(np.nanmean(ALPHA[k]))}
    line = f"{k:28s} {g:7.1f} [{row['ci_item_pp'][0]:6.1f},{row['ci_item_pp'][1]:6.1f}] {row['invalid_pct']:6.1f} {row['mean_alpha_all_steps']:12.2f}"
    if k != "Decode PI":
        d = item["Decode PI"] - item[k]
        row["pi_minus_this_pp"], row["pi_minus_this_ci_pp"] = 100 * d.mean(), ci(d)
        line += f" {row['pi_minus_this_pp']:9.1f} [{row['pi_minus_this_ci_pp'][0]:6.1f},{row['pi_minus_this_ci_pp'][1]:6.1f}]"
    out[k] = row
    print(line)
Path(__file__).with_name("dream_static_alpha_results.json").write_text(json.dumps(out, indent=1) + "\n")
