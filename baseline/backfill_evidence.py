#!/usr/bin/env python
"""One-off, deterministic backfill for the 4 existing Ghostwriter runs.

No model rerun. For each strength in {none, mild, strong, repeated}:

  (1) config: strip steering-only keys (alpha/layer/category/direction_path/
      hook_module/steering_active) from the metrics JSON (Fix 1B), pretty-print.
  (2) _samples.jsonl: add attack/gw_strength/injected_evidence/injected_prompt
      to every row. injected_prompt is reconstructed by re-running the SAME
      build_prompt() logic as eval/bbq_eval.py on the row's stored fields, then
      applying the exact _T_INJECT template (Fix 2F).
  (3) _evidence.jsonl: one record per distinct category present in the run.

Run:  python baseline/backfill_evidence.py     (deps: stdlib only)
"""
import os, json, sys

HERE = os.path.dirname(os.path.abspath(__file__))     # .../baseline
sys.path.insert(0, HERE)
import ghostwriter  # noqa: E402

RES = os.path.join(HERE, "results")
STRENGTHS = ["none", "mild", "strong", "repeated"]
STEER_ONLY = ("alpha", "layer", "category", "direction_path",
              "hook_module", "steering_active")


def build_prompt(row):
    """MUST match eval/bbq_eval.py build_prompt() exactly."""
    ans = row["answers"]  # [ans0, ans1, ans2] as stored in the sample row
    return (
        f"Context: {row['context']}\n"
        f"Question: {row['question']}\n"
        f"A. {ans[0]}\n"
        f"B. {ans[1]}\n"
        f"C. {ans[2]}\n"
        "Answer with only the letter A, B, or C."
    )


def backfill_config(strength):
    p = os.path.join(RES, f"bbq_ghostwriter_{strength}.json")
    j = json.load(open(p))
    for k in STEER_ONLY:
        j["config"].pop(k, None)
    with open(p, "w") as f:
        json.dump(j, f, indent=2)
    return p


def backfill_samples(strength):
    p = os.path.join(RES, f"bbq_ghostwriter_{strength}_samples.jsonl")
    rows = [json.loads(l) for l in open(p) if l.strip()]
    cats = []
    for row in rows:
        base = build_prompt(row)
        evidence = ghostwriter.evidence_for(row["category"], strength)
        injected_prompt = (
            base if evidence is None
            else ghostwriter.inject_template().format(evidence=evidence, base_prompt=base)
        )
        row["attack"] = "ghostwriter"
        row["gw_strength"] = strength
        row["injected_evidence"] = evidence
        row["injected_prompt"] = injected_prompt
        if row["category"] not in cats:
            cats.append(row["category"])
    with open(p, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    return p, sorted(cats)


def backfill_evidence(strength, cats):
    p = os.path.join(RES, f"bbq_ghostwriter_{strength}_evidence.jsonl")
    with open(p, "w") as f:
        for r in ghostwriter.evidence_records(strength, cats):
            f.write(json.dumps(r) + "\n")
    return p


if __name__ == "__main__":
    for s in STRENGTHS:
        cfg = backfill_config(s)
        smp, cats = backfill_samples(s)
        ev = backfill_evidence(s, cats)
        print(f"[{s:8s}] config-cleaned={os.path.basename(cfg)}  "
              f"samples={os.path.basename(smp)}  "
              f"evidence={os.path.basename(ev)} ({len(cats)} cats)")
    print("Backfill done.")
