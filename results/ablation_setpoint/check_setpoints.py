#!/usr/bin/env python3
"""Confirm every cond_*.json in a tier directory records the intended
hyperparameters: the setpoint encoded in its tag, Kp=3, Ki=0.1, Kd=0 (or the
PI mask), amax=6, 64 steps, gen_length 32, and n = 100 (tier1) or 400 (tier2).

    python results/ablation_setpoint/check_setpoints.py results/ablation_setpoint/tier1

Optional second argument overrides the expected n (used by selftest.py).
"""
import glob
import json
import os
import re
import sys

TAG_RE = re.compile(r"^dpid_PI_s(\d)p(\d+)$")
EXPECTED_N = {"tier1": 100, "tier2": 400}


def expected_setpoint(tag):
    m = TAG_RE.match(tag)
    if not m:
        return None
    return float(f"{m.group(1)}.{m.group(2)}")


def check(tier_dir, n_expected=None):
    tier = os.path.basename(os.path.normpath(tier_dir))
    if n_expected is None:
        n_expected = EXPECTED_N.get(tier)
    problems = []
    seen = 0
    for path in sorted(glob.glob(os.path.join(tier_dir, "rot*", "cond_*.json"))):
        tag = os.path.basename(path)[len("cond_"):-len(".json")]
        d = json.load(open(path))
        seen += 1
        where = os.path.relpath(path, tier_dir)

        if tag == "dpid_base":
            if d.get("condition") != "base":
                problems.append(f"{where}: condition={d.get('condition')!r}, expected 'base'")
        else:
            s_exp = expected_setpoint(tag)
            if s_exp is None:
                problems.append(f"{where}: unrecognised tag {tag!r}")
                continue
            if d.get("condition") != "PI":
                problems.append(f"{where}: condition={d.get('condition')!r}, expected 'PI'")
            if abs(float(d.get("setpoint", -1)) - s_exp) > 1e-9:
                problems.append(f"{where}: setpoint={d.get('setpoint')} but tag says {s_exp}")
            gains = {k.lower(): v for k, v in (d.get("gains") or {}).items()}  # stored as Kp/Ki/Kd
            kp, ki, kd = (gains.get("kp"), gains.get("ki"), gains.get("kd"))
            if kp is None or ki is None or kd is None:
                problems.append(f"{where}: gains missing from JSON ({d.get('gains')!r})")
            if kp is not None and abs(kp - 3.0) > 1e-9:
                problems.append(f"{where}: kp={kp}, expected 3")
            if ki is not None and abs(ki - 0.1) > 1e-9:
                problems.append(f"{where}: ki={ki}, expected 0.1")
            if kd not in (None, 0, 0.0):
                problems.append(f"{where}: kd={kd}, expected 0 (PI condition)")
            if abs(float(d.get("alpha_max", -1)) - 6.0) > 1e-9:
                problems.append(f"{where}: alpha_max={d.get('alpha_max')}, expected 6")
            if d.get("alpha_min", 0.0) not in (0, 0.0):
                problems.append(f"{where}: alpha_min={d.get('alpha_min')}, expected 0")
            if d.get("sensor_case", "upper") != "upper":
                problems.append(f"{where}: sensor_case={d.get('sensor_case')!r}, expected 'upper'")
            if d.get("target_mapping", "oracle") != "oracle":
                problems.append(f"{where}: target_mapping={d.get('target_mapping')!r}, expected 'oracle'")

        if d.get("steps") != 64:
            problems.append(f"{where}: steps={d.get('steps')}, expected 64")
        if d.get("gen_length") != 32:
            problems.append(f"{where}: gen_length={d.get('gen_length')}, expected 32")
        if n_expected is not None and d.get("n") != n_expected:
            problems.append(f"{where}: n={d.get('n')}, expected {n_expected}")

        samples = path[:-len(".json")] + "_samples.jsonl"
        if not os.path.exists(samples):
            problems.append(f"{where}: missing {os.path.basename(samples)}")
        else:
            rows = sum(1 for l in open(samples) if l.strip())
            if rows != d.get("n"):
                problems.append(f"{where}: {rows} sample rows but n={d.get('n')}")

    print(f"checked {seen} result files under {tier_dir}")
    for p in problems:
        print("  PROBLEM:", p)
    return not problems and seen > 0


if __name__ == "__main__":
    if len(sys.argv) not in (2, 3):
        sys.exit(__doc__)
    ok = check(sys.argv[1], int(sys.argv[2]) if len(sys.argv) == 3 else None)
    print("OK" if ok else "FAILED")
    sys.exit(0 if ok else 1)
