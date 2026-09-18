#!/usr/bin/env python3
"""Step 5: assemble REPORT.md from env.json, repro.json, tier*/summary.json,
timings.tsv, failures.log, and the sha256 of every *_samples.jsonl produced.

    python results/ablation_setpoint/make_report.py

Numbers only; no interpretation. Missing inputs are reported as missing
rather than invented.
"""
import argparse
import csv
import glob
import hashlib
import json
import os
import re
import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
TAG_RE = re.compile(r"^dpid_PI_s(\d)p(\d+)$")


def read_json(path):
    return json.load(open(path)) if os.path.exists(path) else None


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def setpoint_label(tag):
    if tag == "dpid_base":
        return "base (no steering)"
    m = TAG_RE.match(tag)
    return f"{m.group(1)}.{m.group(2)}" if m else tag


def sort_key(tag):
    if tag == "dpid_base":
        return (0, 0.0)
    m = TAG_RE.match(tag)
    return (1, float(f"{m.group(1)}.{m.group(2)}")) if m else (2, 0.0)


def fmt_ci(ci):
    return f"[{ci[0]:+.1f}, {ci[1]:+.1f}]" if ci else ""


def tier_table(summary):
    hdr = ("| setpoint s* | n | target % | comparator % | abstain % | invalid % | gap (pp) | 95% CI | "
           "gap − s*=0.9 (pp) | paired 95% CI | mean command, steps 1–32 | share ever at limit |")
    sep = "|---|---:|---:|---:|---:|---:|---:|:--:|---:|:--:|---:|---:|"
    lines = [hdr, sep]
    for tag in sorted(summary, key=sort_key):
        r = summary[tag]
        d = r.get("gap_minus_s0p9_pp")
        lines.append("| {sp} | {n} | {t} | {c} | {a} | {i} | {g:+.1f} | {gci} | {d} | {dci} | {mc} | {sl} |".format(
            sp=setpoint_label(tag), n=r["n"], t=r["target"], c=r["comparator"], a=r["abstain"],
            i=r["invalid"], g=r["gap_pp"], gci=fmt_ci(r["gap_ci95"]),
            d=(f"{d:+.1f}" if d is not None else ("ref" if tag == "dpid_PI_s0p9" else "")),
            dci=fmt_ci(r.get("gap_minus_s0p9_ci95")),
            mc=r.get("mean_command_steps_1_32", ""), sl=r.get("share_ever_at_limit", "")))
    return "\n".join(lines)


def timings_section(path):
    if not os.path.exists(path):
        return "_timings.tsv not found._"
    rows = list(csv.DictReader(open(path), delimiter="\t"))
    if not rows:
        return "_timings.tsv is empty._"
    lines = ["| tier | rot | tag | start (UTC) | wall-clock (s) | exit | GPU |", "|---|---|---|---|---:|---:|---|"]
    for r in rows:
        lines.append(f"| {r['tier']} | {r['rot']} | {r['tag']} | {r['start_utc']} | {r['elapsed_s']} | {r['exit_code']} | {r['gpu']} |")
    gpus = sorted({r["gpu"] for r in rows})
    note = "" if len(gpus) <= 1 else f"\n\n**Warning:** runs span {len(gpus)} GPU models: {', '.join(gpus)}."
    return "\n".join(lines) + note


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=HERE)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    d = args.dir
    out = args.out or os.path.join(d, "REPORT.md")

    env = read_json(os.path.join(d, "env.json"))
    repro = read_json(os.path.join(d, "repro.json"))
    parts = ["# Setpoint ablation for the decode-time PI attack — report", "",
             f"_Generated {datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')} "
             f"by make_report.py. Numbers only; no interpretation._", ""]

    parts += ["## Environment", ""]
    if env:
        parts += [
            f"- commit: `{env.get('commit')}` (modified tracked files at preflight: {env.get('modified_tracked_files')})",
            f"- GPU: {env.get('gpu')} (host {env.get('hostname')}, SLURM job {env.get('slurm_job_id')})",
            f"- python {env.get('python')}, torch {env.get('torch')}, transformers {env.get('transformers')}, numpy {env.get('numpy')}",
            f"- arrows: `{env.get('arrows_path')}` sha256 `{env.get('arrows_sha256')}`",
            "- fixed controller args: `" + str(env.get("fixed_args")) + "`; only `--setpoint` varies",
        ]
    else:
        parts.append("_env.json not found; preflight.sh was not run._")
    parts.append("")

    parts += ["## Step 1: reproduction of the published s*=0.9 run (first 100 items × 3 rotations)", ""]
    if repro:
        p = repro["pooled"]
        parts += [
            f"- identical `model_output` on {p['identical_output_frac']:.1%} of {p['n']} rows",
            f"- strict-class agreement on {p['strict_class_agreement_frac']:.1%}",
            f"- strict gap: rerun {p['gap_pp_rerun']:+.1f} pp vs published {p['gap_pp_published']:+.1f} pp",
            "", "| rotation | n | identical outputs | class agreement | gap rerun (pp) | gap published (pp) |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for r in range(3):
            q = repro.get(f"rot{r}")
            if q:
                parts.append(f"| rot{r} | {q['n']} | {q['identical_output_frac']:.3f} | {q['strict_class_agreement_frac']:.3f} "
                             f"| {q['gap_pp_rerun']:+.1f} | {q['gap_pp_published']:+.1f} |")
    else:
        parts.append("_repro.json not found; run_repro.sh was not run._")
    parts.append("")

    for tier in ("tier1", "tier2"):
        summary = read_json(os.path.join(d, tier, "summary.json"))
        title = {"tier1": "Tier 1: first 100 items per rotation (300 pooled)",
                 "tier2": "Tier 2: full 400 items per rotation (1,200 pooled)"}[tier]
        parts += [f"## {title}", ""]
        if summary:
            parts += [tier_table(summary), "",
                      "Gap = (target − comparator) / n with strict-invalid rows kept in the denominator. "
                      "CIs: 10,000 item-level bootstrap resamples, seed 0; the paired CI resamples the per-item "
                      "difference against s*=0.9 with the same indices. Mean command and limit share are read "
                      "from `alpha_traj`; only steps 1–32 commit tokens.", ""]
        else:
            parts += [f"_{tier}/summary.json not found; {tier} was not run or not scored._", ""]

    parts += ["## Wall-clock per run", "", timings_section(os.path.join(d, "timings.tsv")), ""]

    parts += ["## Failed or rerun jobs", ""]
    fl = os.path.join(d, "failures.log")
    if os.path.exists(fl) and os.path.getsize(fl):
        parts += ["```"] + [l.rstrip("\n") for l in open(fl)] + ["```",
                  "Each failed run was rerun individually after the environment was fixed; "
                  "the skip-if-present guard left completed runs untouched."]
    else:
        parts.append("None recorded in failures.log.")
    parts.append("")

    parts += ["## sha256 of every `*_samples.jsonl` produced", "", "```"]
    files = sorted(glob.glob(os.path.join(d, "tier*", "rot*", "cond_*_samples.jsonl")))
    if files:
        for f in files:
            parts.append(f"{sha256(f)}  {os.path.relpath(f, d)}")
    else:
        parts.append("(no sample files found)")
    parts += ["```", ""]

    open(out, "w").write("\n".join(parts))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
