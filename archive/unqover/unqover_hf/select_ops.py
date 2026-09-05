#!/usr/bin/env python
"""Apply the PRE-REGISTERED selection rule to the BUILD-split sweep.  CPU only.

Implements unqover_hf/PREREG.md section 4 mechanically and identically for ours
and for every baseline, and writes:

    <sweep-dir>/BUILD_TABLE.md    every method x every config
    <sweep-dir>/SELECTED.json     the chosen operating points + the clause that
                                  chose each, the gate value, and the
                                  effort-matched open-loop alpha (section 5)

THE RULE (PREREG section 4, verbatim in the docstring of `select`):

    G:  strict invalid_rate(c) <= max(0.10, base_inv + 0.10)
    CLAUSE 1: argmax pref_gap_raw over the gate-passing configs of the method
    CLAUSE 2 (fallback, reported): every config fails G -> lowest invalid_rate
    TIE-BREAKS within EPS_GAP=0.02 of the clause-1 max:
        higher coverage -> lower invalid -> lower realized actuation -> tag

Reads only BUILD-split summaries.  It never opens an EVAL file.

  python -m unqover_hf.select_ops --selftest
  python -m unqover_hf.select_ops --sweep-dir results/unqover_hf/build_sweep
"""
import argparse
import glob
import json
import os
import sys

INVALID_FLOOR = 0.10          # the max(0.10, ...) floor in the gate
INVALID_MARGIN = 0.10         # base_inv + this
EPS_GAP = 0.02                # tie-break band on the primary endpoint

BASE_TAG = "base"
# Methods whose dose is an additive vector, so realized actuation is a norm.
ADDITIVE = {"ours_PI", "ours_PID", "normal", "caa", "actadd",
            "meanact_raw", "meanact_unit", "itic"}


# --------------------------------------------------------------------------- #
def load_rows(sweep_dir):
    rows = []
    for p in sorted(glob.glob(os.path.join(sweep_dir, "*_summary.json"))):
        with open(p) as fh:
            s = json.load(fh)
        m = s["metric"]
        ts = m["target_stats"]
        po = s.get("pick_rate_by_order", {})
        opt = s.get("target_outcome_by_option_position", {})
        act = s.get("actuation") or {}
        rows.append({
            "tag": s["tag"],
            "method": s.get("sweep_method"),
            "knobs": s.get("knobs", {}),
            "gap_raw": ts["pref_gap_raw"],
            "gap_deb": ts["pref_gap_debiased"],
            "invalid": s["invalid_rate"],
            "coverage": s["coverage"],
            "mu": m["mu"], "eta": m["eta"], "raw_skew_mu": m["raw_skew_mu"],
            "epsilon": m["epsilon"], "delta": m["delta"],
            "n_complete": m["n_complete"], "n_instances": m["n_instances"],
            "n_clusters": ts.get("n_clusters"),
            "tgt_first": ts["target_pick_rate_first_named"],
            "tgt_second": ts["target_pick_rate_second_named"],
            "order0_A": po.get("0", {}).get("A"), "order0_B": po.get("0", {}).get("B"),
            "order1_A": po.get("1", {}).get("A"), "order1_B": po.get("1", {}).get("B"),
            "tgt_at_A": opt.get("A", {}).get("target"),
            "tgt_at_B": opt.get("B", {}).get("target"),
            "delta_norm_mean": s.get("delta_norm_mean_over_layers"),
            "mean_alpha_full": act.get("mean_alpha_full_run"),
            "mean_alpha_commit": act.get("mean_alpha_commit_window"),
            "mean_commit_step": act.get("mean_commit_step"),
            "sat_frac": act.get("mean_sat_frac"),
            "p_target_final": act.get("mean_p_target_final"),
            "hook_fire_count": s.get("hook_fire_count"),
            "hook_fire_per_item": s.get("hook_fire_per_item"),
            "elapsed_s": s.get("elapsed_s"),
            "degenerate": bool(m["epsilon"] > 0.5 and m["mu"] < 0.1),
            "ci_raw": ts.get("ci_raw"), "ci_debiased": ts.get("ci_debiased"),
            "noise_floor": m.get("noise_floor"),
            "article_ambiguous_rate": s.get("article_ambiguous_rate"),
            "summary_path": p,
        })
    return rows


def mu_above_floor(r):
    """Is this run's mu above the chance floor the SAME run's own validity
    pattern produces?  metric.py: mu is only interpretable above that line."""
    nf = r.get("noise_floor")
    if not nf or "mu" not in nf:
        return None
    return r["mu"] > nf["mu"]["p95"]


def actuation_of(r):
    """Realized actuation for tie-break (t3): the added-vector norm for the
    additive methods (for ours, the realized mean alpha over the full run),
    otherwise the magnitude of the dose knob."""
    if r["method"] in ("ours_PI", "ours_PID") and r["mean_alpha_full"] is not None:
        return float(r["mean_alpha_full"])
    if r["method"] in ADDITIVE and r["delta_norm_mean"] is not None:
        return float(r["delta_norm_mean"])
    k = r["knobs"] or {}
    for key in ("gamma", "strength", "alpha"):
        if key in k:
            return abs(float(k[key]))
    return 0.0


def select(rows, invalid_floor=INVALID_FLOOR, invalid_margin=INVALID_MARGIN,
           eps_gap=EPS_GAP):
    """PREREG section 4, applied mechanically.  -> (gate, {method: (row, clause)})."""
    base = next((r for r in rows if r["tag"] == BASE_TAG), None)
    assert base is not None, ("the unsteered `base` run is missing; the gate is "
                              "anchored to it and cannot be computed without it")
    gate = max(invalid_floor, base["invalid"] + invalid_margin)

    by_method = {}
    for r in rows:
        by_method.setdefault(r["method"], []).append(r)

    def tb(r):                                  # tie-break chain t1..t4
        return (-r["coverage"], r["invalid"], actuation_of(r), r["tag"])

    out = {}
    for m, rs in sorted(by_method.items()):
        if m == BASE_TAG:
            continue
        elig = [r for r in rs if r["invalid"] <= gate + 1e-12]
        if elig:
            top = max(r["gap_raw"] for r in elig)
            near = [r for r in elig if r["gap_raw"] >= top - eps_gap]
            win = sorted(near, key=tb)[0]
            clause = ("CLAUSE 1: max pref_gap_raw among the %d/%d configs passing "
                      "the gate (invalid <= %.4f)" % (len(elig), len(rs), gate))
            if len(near) > 1:
                clause += ("; %d configs within EPS_GAP=%.2f of the max, broken by "
                           "coverage -> invalid -> actuation -> tag" % (len(near), eps_gap))
        else:
            win = sorted(rs, key=lambda r: (r["invalid"], r["tag"]))[0]
            clause = ("CLAUSE 2 (FALLBACK): all %d configs FAIL the coherence gate "
                      "(invalid <= %.4f); picked the lowest invalid_rate" % (len(rs), gate))
        out[m] = (win, clause)
    return gate, out, base


def effort_matched_alpha(selected, rows):
    """PREREG section 5: the open-loop alpha that matches the selected
    controller's realized mean actuation over the FULL run (the pre-registered
    basis), with the committing-window figure reported beside it."""
    out = {}
    for m in ("ours_PI", "ours_PID"):
        if m not in selected:
            continue
        r = selected[m][0]
        if r["mean_alpha_full"] is None:
            continue
        near = None
        cands = [q for q in rows if q["method"] == "normal"]
        if cands:
            near = min(cands, key=lambda q: abs(float(q["knobs"]["alpha"])
                                                - r["mean_alpha_full"]))
        out[m] = {
            "selected_tag": r["tag"],
            "basis": "mean of alpha(t) over ALL 64 denoising steps, averaged over items",
            "effort_matched_alpha_full_run": round(r["mean_alpha_full"], 4),
            "alt_basis_commit_window": round(r["mean_alpha_commit"], 4)
            if r["mean_alpha_commit"] is not None else None,
            "mean_commit_step": r["mean_commit_step"],
            "amax": (r["knobs"] or {}).get("amax"),
            "sat_frac": r["sat_frac"],
            "nearest_swept_normal_alpha": (float(near["knobs"]["alpha"])
                                           if near else None),
            "note": ("the effort-matched control is a NEW `normal` run at "
                     "alpha = effort_matched_alpha_full_run; the nearest swept "
                     "alpha is reported only for orientation"),
        }
    return out


# --------------------------------------------------------------------------- #
COLS = [("tag", "%-24s", "%-24s"), ("gap_raw", "%9s", "%+9.4f"),
        ("gap_deb", "%9s", "%+9.4f"), ("invalid", "%8s", "%8.4f"),
        ("cov", "%6s", "%6.3f"), ("o0_A", "%6s", "%6.3f"), ("o1_A", "%6s", "%6.3f"),
        ("T@A", "%6s", "%6.3f"), ("T@B", "%6s", "%6.3f"),
        ("mu", "%6s", "%6.3f"), ("rawskew", "%8s", "%8.3f"),
        ("eps", "%6s", "%6.3f"), ("act", "%8s", "%8.3f"),
        ("fires", "%9s", "%9d"), ("s", "%6s", "%6.0f")]


def fmt_table(rows, gate, selected):
    key = {"tag": lambda r: r["tag"], "gap_raw": lambda r: r["gap_raw"],
           "gap_deb": lambda r: r["gap_deb"], "invalid": lambda r: r["invalid"],
           "cov": lambda r: r["coverage"],
           "o0_A": lambda r: r["order0_A"] or 0.0, "o1_A": lambda r: r["order1_A"] or 0.0,
           "T@A": lambda r: r["tgt_at_A"] or 0.0, "T@B": lambda r: r["tgt_at_B"] or 0.0,
           "mu": lambda r: r["mu"], "rawskew": lambda r: r["raw_skew_mu"],
           "eps": lambda r: r["epsilon"], "act": actuation_of,
           "fires": lambda r: r["hook_fire_count"] or 0,
           "s": lambda r: r["elapsed_s"] or 0.0}
    lines = []
    hdr = " ".join(h % n for n, h, _ in COLS) + "  flags"
    lines.append(hdr)
    lines.append("-" * len(hdr))
    order = ["base"] + [m for m in sorted(selected)]
    seen = set()
    for m in order:
        rs = sorted([r for r in rows if r["method"] == m], key=lambda r: r["tag"])
        if not rs:
            continue
        lines.append("## %s" % m)
        for r in rs:
            seen.add(r["tag"])
            flags = []
            if r["invalid"] > gate + 1e-12:
                flags.append("GATE-FAIL")
            if m in selected and selected[m][0]["tag"] == r["tag"]:
                flags.append("<< SELECTED")
            if r["degenerate"]:
                flags.append("DEGENERATE")
            lines.append(" ".join(f % key[n](r) for n, _, f in COLS)
                         + "  " + " ".join(flags))
    missing = [r for r in rows if r["tag"] not in seen]
    if missing:
        lines.append("## (ungrouped)")
        for r in missing:
            lines.append(" ".join(f % key[n](r) for n, _, f in COLS))
    return "\n".join(lines)


def write_report(sweep_dir, rows, gate, selected, base, effort):
    md = [
        "# UNQOVER-race BUILD-split sweep (Phase 2)",
        "",
        "Selection rule: `unqover_hf/PREREG.md` section 4, applied mechanically.",
        "",
        "* configs run: **%d**" % len(rows),
        "* base strict-invalid rate: **%.4f**" % base["invalid"],
        "* **coherence gate: strict invalid_rate <= max(0.10, %.4f + 0.10) = %.4f**"
        % (base["invalid"], gate),
        "* primary endpoint: `pref_gap_raw` toward the merged black family",
        "* sweep subset: %d instances / %d generations per config"
        % (base["n_instances"], base["n_instances"] * 4),
        "",
        "Columns: `o0_A`/`o1_A` = fraction answering **A** in subject order 0 / 1;",
        "`T@A`/`T@B` = target picked when the target sits at option A / B;",
        "`act` = realized actuation (see PREREG section 4 t3); `fires` = hook fire count.",
        "",
        "```",
        fmt_table(rows, gate, selected),
        "```",
        "",
        "## Selected operating points",
        "",
        "| method | selected tag | knobs | gap_raw | gap_deb | invalid | cov | clause |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for m in sorted(selected):
        r, clause = selected[m]
        md.append("| %s | `%s` | %s | %+.4f | %+.4f | %.4f | %.3f | %s |"
                  % (m, r["tag"], r["knobs"], r["gap_raw"], r["gap_deb"],
                     r["invalid"], r["coverage"], clause))
    md += ["", "## mu vs its own chance floor (PREREG section 2: mu is SECONDARY)", "",
           "| method | tag | mu | chance mu (p95, this run's n and coverage) | above floor? | raw_skew_mu | epsilon |",
           "|---|---|---|---|---|---|---|"]
    for m in ["base"] + sorted(selected):
        r = base if m == "base" else selected[m][0]
        nf = r.get("noise_floor") or {}
        p95 = nf.get("mu", {}).get("p95")
        md.append("| %s | `%s` | %.4f | %s | %s | %.4f | %.4f |"
                  % (m, r["tag"], r["mu"],
                     ("%.4f" % p95) if p95 is not None else "n/a",
                     {True: "yes", False: "**no**", None: "n/a"}[mu_above_floor(r)],
                     r["raw_skew_mu"], r["epsilon"]))
    md += ["", "## Effort matching (PREREG section 5)", ""]
    for m, e in sorted(effort.items()):
        md.append("* **%s** (`%s`): effort-matched open-loop alpha = "
                  "**%.4f** (basis: %s); committing-window basis = %s "
                  "(mean commit step %s of 64); saturated fraction %.3f."
                  % (m, e["selected_tag"], e["effort_matched_alpha_full_run"],
                     e["basis"], e["alt_basis_commit_window"],
                     ("%.1f" % e["mean_commit_step"]) if e["mean_commit_step"] else "n/a",
                     e["sat_frac"] or 0.0))
    if not effort:
        md.append("* (no controller run present)")
    md.append("")
    path = os.path.join(sweep_dir, "BUILD_TABLE.md")
    with open(path, "w") as fh:
        fh.write("\n".join(md) + "\n")

    sel_json = {
        "prereg": "unqover_hf/PREREG.md section 4",
        "coherence_gate": {"formula": "max(0.10, base_invalid + 0.10)",
                           "base_invalid": base["invalid"], "value": gate},
        "primary_endpoint": "pref_gap_raw toward the merged black family",
        "eps_gap": EPS_GAP,
        "n_configs": len(rows),
        "selected": {m: {"tag": r["tag"], "knobs": r["knobs"], "clause": c,
                         "gap_raw": r["gap_raw"], "ci_raw": r["ci_raw"],
                         "gap_deb": r["gap_deb"], "ci_debiased": r["ci_debiased"],
                         "invalid": r["invalid"], "coverage": r["coverage"],
                         "order0_A": r["order0_A"], "order1_A": r["order1_A"],
                         "target_at_A": r["tgt_at_A"], "target_at_B": r["tgt_at_B"],
                         "mu": r["mu"], "eta": r["eta"],
                         "raw_skew_mu": r["raw_skew_mu"],
                         "delta": r["delta"], "epsilon": r["epsilon"],
                         "degenerate": r["degenerate"],
                         "mu_above_own_chance_floor": mu_above_floor(r),
                         "noise_floor": r["noise_floor"],
                         "realized_actuation": actuation_of(r),
                         "hook_fire_count": r["hook_fire_count"],
                         "hook_fire_per_item": r["hook_fire_per_item"],
                         "elapsed_s": r["elapsed_s"],
                         "summary_path": r["summary_path"]}
                     for m, (r, c) in selected.items()},
        "effort_matching": effort,
        "base": {k: base[k] for k in ("tag", "gap_raw", "ci_raw", "gap_deb",
                                      "invalid", "coverage", "order0_A",
                                      "order1_A", "tgt_at_A", "tgt_at_B", "mu",
                                      "eta", "raw_skew_mu", "epsilon", "delta",
                                      "n_instances", "n_clusters",
                                      "noise_floor")},
    }
    jpath = os.path.join(sweep_dir, "SELECTED.json")
    with open(jpath, "w") as fh:
        json.dump(sel_json, fh, indent=1)
    return path, jpath


# --------------------------------------------------------------------------- #
def _row(tag, method, gap, inv, cov=1.0, knobs=None, act=None, eps=0.2, mu=0.3):
    return {"tag": tag, "method": method, "knobs": knobs or {},
            "gap_raw": gap, "gap_deb": gap / 2, "invalid": inv, "coverage": cov,
            "mu": mu, "eta": 0.3, "raw_skew_mu": 0.4, "epsilon": eps, "delta": 0.2,
            "n_complete": 100, "n_instances": 100, "n_clusters": 90,
            "tgt_first": 0.5, "tgt_second": 0.5, "order0_A": 0.5, "order0_B": 0.5,
            "order1_A": 0.5, "order1_B": 0.5, "tgt_at_A": 0.5, "tgt_at_B": 0.5,
            "delta_norm_mean": act, "mean_alpha_full": None,
            "mean_alpha_commit": None, "mean_commit_step": None, "sat_frac": None,
            "p_target_final": None, "hook_fire_count": 10, "hook_fire_per_item": 1.0,
            "elapsed_s": 1.0, "degenerate": bool(eps > 0.5 and mu < 0.1),
            "summary_path": tag}


def selftest():
    ok = True

    def check(name, cond, extra=""):
        nonlocal ok
        ok &= bool(cond)
        print("[selftest] %-58s %s%s" % (name, "PASS" if cond else "FAIL",
                                         (" " + extra) if extra else ""))

    # gate arithmetic
    rows = [_row("base", "base", 0.0, 0.04)]
    rows += [_row("m_a", "M", 0.30, 0.05), _row("m_b", "M", 0.50, 0.20),
             _row("m_c", "M", 0.40, 0.10)]
    gate, sel, base = select(rows)
    check("gate = max(0.10, base_inv+0.10) with a low base",
          abs(gate - 0.14) < 1e-12, "(%.4f)" % gate)
    check("CLAUSE 1 picks the max gap among gate-passers, not the global max",
          sel["M"][0]["tag"] == "m_c" and "CLAUSE 1" in sel["M"][1],
          sel["M"][0]["tag"])

    rows2 = [_row("base", "base", 0.0, 0.30)]
    rows2 += [_row("n_a", "N", 0.30, 0.35), _row("n_b", "N", 0.60, 0.39)]
    gate2, sel2, _ = select(rows2)
    check("gate follows a high base invalid rate", abs(gate2 - 0.40) < 1e-12,
          "(%.4f)" % gate2)
    check("both configs pass the looser gate; max gap wins",
          sel2["N"][0]["tag"] == "n_b")

    rows3 = [_row("base", "base", 0.0, 0.02)]
    rows3 += [_row("p_a", "P", 0.90, 0.50), _row("p_b", "P", 0.80, 0.40)]
    gate3, sel3, _ = select(rows3)
    check("CLAUSE 2 fires when every config fails the gate",
          "CLAUSE 2" in sel3["P"][1] and sel3["P"][0]["tag"] == "p_b",
          sel3["P"][0]["tag"])

    # tie-breaks
    rows4 = [_row("base", "base", 0.0, 0.02)]
    rows4 += [_row("q_hi_cov", "Q", 0.50, 0.05, cov=0.99, act=5.0),
              _row("q_lo_cov", "Q", 0.515, 0.05, cov=0.80, act=1.0)]
    _, sel4, _ = select(rows4)
    check("t1: within EPS_GAP, higher coverage wins over a bigger gap",
          sel4["Q"][0]["tag"] == "q_hi_cov", sel4["Q"][0]["tag"])
    rows5 = [_row("base", "base", 0.0, 0.02)]
    rows5 += [_row("r_hi_act", "caa", 0.50, 0.05, cov=0.9, act=9.0),
              _row("r_lo_act", "caa", 0.505, 0.05, cov=0.9, act=1.0)]
    _, sel5, _ = select(rows5)
    check("t3: equal coverage+invalid -> lower actuation wins",
          sel5["caa"][0]["tag"] == "r_lo_act", sel5["caa"][0]["tag"])
    rows5b = [_row("base", "base", 0.0, 0.02)]
    rows5b += [_row("v_hi", "aura_inject", 0.50, 0.05, cov=0.9, knobs={"gamma": 8}),
               _row("v_lo", "aura_inject", 0.505, 0.05, cov=0.9, knobs={"gamma": 1})]
    _, sel5b, _ = select(rows5b)
    check("t3: non-additive methods tie-break on the |dose knob|",
          sel5b["aura_inject"][0]["tag"] == "v_lo", sel5b["aura_inject"][0]["tag"])
    rows6 = [_row("base", "base", 0.0, 0.02)]
    rows6 += [_row("s_bbb", "S", 0.50, 0.05, act=1.0),
              _row("s_aaa", "S", 0.50, 0.05, act=1.0)]
    _, sel6, _ = select(rows6)
    check("t4: full ties resolve on the lexically smaller tag",
          sel6["S"][0]["tag"] == "s_aaa", sel6["S"][0]["tag"])

    check("a gap outside the EPS_GAP band is NOT overturned by coverage",
          select([_row("base", "base", 0.0, 0.02),
                  _row("u_big", "U", 0.60, 0.05, cov=0.70),
                  _row("u_cov", "U", 0.50, 0.05, cov=1.00)])[1]["U"][0]["tag"]
          == "u_big")

    check("the rule is method-agnostic: renaming a method changes nothing",
          select([_row("base", "base", 0.0, 0.04), _row("x1", "ours_PI", 0.3, 0.05),
                  _row("x2", "ours_PI", 0.4, 0.10)])[1]["ours_PI"][0]["tag"]
          == select([_row("base", "base", 0.0, 0.04), _row("x1", "caa", 0.3, 0.05),
                     _row("x2", "caa", 0.4, 0.10)])[1]["caa"][0]["tag"])

    check("degeneracy is FLAGGED, not gated out",
          _row("d", "D", 1.0, 0.01, eps=0.9, mu=0.0)["degenerate"] is True
          and select([_row("base", "base", 0.0, 0.02),
                      _row("d", "D", 1.0, 0.01, eps=0.9, mu=0.0)])[1]["D"][0]["tag"] == "d")

    missing = False
    try:
        select([_row("m", "M", 0.1, 0.1)])
    except AssertionError:
        missing = True
    check("missing base run raises instead of inventing a gate", missing)

    print("[selftest] select_ops OVERALL: %s" % ("PASS" if ok else "FAIL"))
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--sweep-dir", default=None)
    args = ap.parse_args()
    if args.selftest:
        sys.exit(0 if selftest() else 1)
    if not args.sweep_dir:
        ap.error("--sweep-dir required (or --selftest)")
    rows = load_rows(args.sweep_dir)
    assert rows, "no *_summary.json under %s" % args.sweep_dir
    gate, selected, base = select(rows)
    effort = effort_matched_alpha(selected, rows)
    print(fmt_table(rows, gate, selected))
    print("\ncoherence gate = max(%.2f, %.4f + %.2f) = %.4f"
          % (INVALID_FLOOR, base["invalid"], INVALID_MARGIN, gate))
    for m in sorted(selected):
        r, c = selected[m]
        print("  %-14s -> %-24s gap_raw=%+.4f  %s" % (m, r["tag"], r["gap_raw"], c))
    for m, e in sorted(effort.items()):
        print("  effort-matched alpha for %s: %.4f (full run) / %s (commit window)"
              % (m, e["effort_matched_alpha_full_run"], e["alt_basis_commit_window"]))
    p, j = write_report(args.sweep_dir, rows, gate, selected, base, effort)
    print("wrote %s\nwrote %s" % (p, j))


if __name__ == "__main__":
    main()
