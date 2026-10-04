#!/usr/bin/env python
"""Compute the preregistered label-free SocialStigmaQA effort-match alpha.

The alpha is the pooled mean PI controller output over the first 32 live
denoising steps of the decode_pi64 runs (no labels used).

Usage:
    python socialstigma/compute_effort_match.py [--print-alpha]
"""
from __future__ import annotations

import argparse
import json
import os
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path


ROOT = Path(os.environ.get("DLM_BIAS_ROOT") or Path(__file__).resolve().parents[1])
RESULTS = ROOT / "results" / "socialstigma"
OUT = RESULTS / "effort_match.json"


def compute(results_root: Path = RESULTS) -> dict:
    values = []
    files = []
    per_polarity = {}
    for polarity, expected_per_rotation in (("yes", 210), ("no", 345)):
        pv = []
        for rotation in (0, 1, 2):
            path = (results_root / "decode_pid" / polarity / f"rot{rotation}"
                    / "cond_dpid_PI_samples.jsonl")
            if not path.exists():
                raise FileNotFoundError(path)
            rows = [json.loads(line) for line in path.open() if line.strip()]
            if len(rows) != expected_per_rotation:
                raise AssertionError(f"{path}: {len(rows)} != {expected_per_rotation}")
            for row in rows:
                traj = row.get("alpha_traj")
                if not isinstance(traj, list) or len(traj) != 64:
                    raise AssertionError(f"bad alpha trajectory in {path}")
                pv.extend(float(x) for x in traj[:32])
            files.append(str(path.relative_to(ROOT)))
        per_polarity[polarity] = sum(pv) / len(pv)
        values.extend(pv)
    raw = sum(values) / len(values)
    rounded = float(Decimal(str(raw)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))
    result = {
        "benchmark": "SocialStigmaQA-MC3",
        "source_condition": "decode_pi64",
        "definition": "pooled mean alpha over first 32 live denoising steps",
        "uses_labels": False,
        "n_items": 1665,
        "n_alpha_values": len(values),
        "mean_live_alpha_raw": raw,
        "mean_live_alpha_rounded_2dp": rounded,
        "per_polarity_raw": per_polarity,
        "source_files": files,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--print-alpha", action="store_true")
    args = ap.parse_args()
    result = compute()
    if args.print_alpha:
        print(f"{result['mean_live_alpha_rounded_2dp']:g}")
    else:
        print(json.dumps(result, indent=2, sort_keys=True))
        print(f"[effort-match] PASS -> {OUT}")


if __name__ == "__main__":
    main()
