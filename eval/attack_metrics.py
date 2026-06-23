#!/usr/bin/env python
"""Compute the absolute pick-rate attack metrics from saved *_samples.jsonl WITHOUT
re-running the model, backfill them into the metrics JSONs, and print a dose-response table.

Usage:
  # backfill the 4 current runs and print the table (baseline = clean run):
  python eval/attack_metrics.py \
      --baseline eval/results/bbq_clean.json \
      eval/results/bbq_clean.json \
      eval/results/bbq_L14_race_color_a8.json \
      eval/results/bbq_L14_race_color_a16.json \
      eval/results/bbq_L14_race_color_a32.json \
      --write
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bias_metrics as bm


def load(metrics_path):
    d = json.load(open(metrics_path))
    sf = d.get("samples_file") or (os.path.basename(metrics_path)[:-5] + "_samples.jsonl")
    sp = os.path.join(os.path.dirname(metrics_path), sf)
    if not os.path.exists(sp):
        raise FileNotFoundError(f"samples file not found for {metrics_path}: {sp}")
    samples = [json.loads(l) for l in open(sp) if l.strip()]
    return d, samples, sp


def main():
    ap = argparse.ArgumentParser(description="Backfill/print absolute pick-rate attack metrics from saved samples.")
    ap.add_argument("results", nargs="+", help="metrics JSON files to analyze")
    ap.add_argument("--baseline", default=None, help="metrics JSON to use as the flip baseline")
    ap.add_argument("--write", action="store_true",
                    help="patch each metrics JSON in place, adding an 'attack_metrics' block")
    args = ap.parse_args()

    base_samples = None
    if args.baseline:
        _, base_samples, _ = load(args.baseline)

    rows = []
    for path in args.results:
        d, samples, sp = load(path)
        am = bm.attack_metrics(samples)
        alpha = d.get("config", {}).get("alpha", "?")
        fl = bm.flips(base_samples, samples) if base_samples is not None else None
        if args.write:
            d["attack_metrics"] = am
            if fl is not None:
                d["flips_vs_baseline"] = fl
            with open(path, "w") as f:
                json.dump(d, f, indent=2)
        rows.append((os.path.basename(path), alpha, am, fl))

    # ---- dose-response table ----
    print("=" * 92)
    print("ATTACK METRICS  (ambiguous items, gold=unknown; absolute pick rates)")
    print("-" * 92)
    print(f"{'run':34}{'alpha':>6}{'abstain':>9}{'TARGET':>9}{'non-tgt':>9}{'no_ans':>8}{'n':>6}")
    for name, alpha, am, _ in rows:
        print(f"{name[:34]:34}{str(alpha):>6}{am['abstention_rate']:>9.3f}"
              f"{am['target_rate']:>9.3f}{am['nontarget_rate']:>9.3f}"
              f"{am['no_answer_rate']:>8.3f}{am['n_ambig']:>6}")
    if base_samples is not None:
        print("-" * 92)
        print(f"FLIPS from baseline-'unknown' items (n_base_unknown shown):")
        print(f"{'run':34}{'->target':>10}{'->non-tgt':>10}{'->no_ans':>10}{'stay':>8}{'base_unk':>9}")
        for name, alpha, _, fl in rows:
            if fl is None:
                continue
            print(f"{name[:34]:34}{fl['flip_to_target']:>10}{fl['flip_to_nontarget']:>10}"
                  f"{fl['flip_to_no_answer']:>10}{fl['stayed_unknown']:>8}{fl['baseline_unknown_n']:>9}")
    print("=" * 92)
    if args.write:
        print("Patched 'attack_metrics'"
              + (" + 'flips_vs_baseline'" if base_samples is not None else "")
              + " into the metrics JSON files above.")


if __name__ == "__main__":
    main()
