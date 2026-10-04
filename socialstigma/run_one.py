#!/usr/bin/env python
"""Run one preregistered SocialStigmaQA-MC3 condition/polarity/rotation.

Usage:
    python socialstigma/run_one.py --condition decode_pi64 --polarity yes --rotation 0
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


ROOT = Path(os.environ.get("DLM_BIAS_ROOT") or Path(__file__).resolve().parents[1])
POLARITIES = ("yes", "no")
CONDITIONS = (
    "clean",
    "decode_pi64",
    "decode_pid64",
    # NOTE: decode_pi32 is NOT a separate condition. With gen_length ==
    # block_length == 32, steps 33-64 of the 64-step run commit no tokens, so
    # a 32-step run is the same procedure; it is kept as a same-GPU numerical
    # replicate and aggregate.py checks output identity instead of tabulating it.
    "decode_pi32",
    "normal_a4",
    "normal_eff",
    "layer_pi_a2",
    "caa_a16",
    "actadd_a16",
    "meanact_s2",
    "linearact_s1",
    "aura_inject_g4",
    "aura_vanilla",
    "itic_k48_a8",
    "itic_k48_a8_tb",
)


def count_lines(path: Path) -> int:
    with path.open(encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def configure_paths(polarity: str) -> tuple[Path, Path]:
    cache = ROOT / "socialstigma" / "cache" / polarity
    arrows = cache / "arrows.pt"
    if not arrows.exists():
        raise FileNotFoundError(f"missing {arrows}; run socialstigma.fit_artifacts first")
    os.environ["DLM_BIAS_ROOT"] = str(ROOT)
    os.environ["DLM_BASELINE_CACHE_DIR"] = str(cache)
    os.environ["DLM_ARROWS_PATH"] = str(arrows)
    return cache, arrows


def expected(condition: str, effort_alpha: float | None) -> tuple[str, str]:
    fixed = {
        "clean": ("clean", "dpid_base"),
        "decode_pi64": ("decode_pid", "dpid_PI"),
        "decode_pid64": ("decode_pid", "dpid_PID"),
        "decode_pi32": ("decode_pid_s32", "dpid_PI_s32"),
        "normal_a4": ("normal", "normalL14_a4"),
        "layer_pi_a2": ("layer_pi", "layerPI_a2"),
        "caa_a16": ("caa", "caa_L14_a16"),
        "actadd_a16": ("actadd", "actadd_a16"),
        "meanact_s2": ("meanact", "meanact_unit_s2"),
        "linearact_s1": ("linearact", "gaussian_s1"),
        "aura_inject_g4": ("aura_inject", "inject_g4"),
        "aura_vanilla": ("aura_vanilla", "vanilla"),
        "itic_k48_a8": ("itic", "itic_K48_a8"),
        # Same probes as itic_k48_a8, val_acc ties broken by head margin.
        "itic_k48_a8_tb": ("itic_tiebreak", "itic_K48_a8_tb"),
    }
    if condition in fixed:
        return fixed[condition]
    if condition == "normal_eff":
        if effort_alpha is None:
            raise ValueError("normal_eff requires --effort-alpha")
        stem = f"normalEff_a{effort_alpha:g}".replace(".", "p").replace("-", "m")
        return "normal_eff", stem
    raise ValueError(condition)


def run(args) -> Path:
    _, arrows = configure_paths(args.polarity)

    # Import only after environment isolation is configured: the baseline
    # modules resolve cache/arrows paths at import time.
    sys.path.insert(0, str(ROOT / "eval"))
    sys.path.insert(0, str(ROOT / "steering"))
    sys.path.insert(0, str(ROOT / "baselines"))
    import denoise_pid  # noqa: E402
    import pid_steer  # noqa: E402
    import caa  # noqa: E402
    import actadd  # noqa: E402
    import meanact  # noqa: E402
    import linearact  # noqa: E402
    import aura  # noqa: E402
    import itic  # noqa: E402

    items = ROOT / "data" / "socialstigma" / "items" / \
        f"eval_{args.polarity}_rot{args.rotation}.jsonl"
    if not items.exists():
        raise FileNotFoundError(items)
    full_n = count_lines(items)
    expected_n = min(args.limit, full_n) if args.limit else full_n
    method_dir, stem = expected(args.condition, args.effort_alpha)
    out_dir = Path(args.out_root) / method_dir / args.polarity / f"rot{args.rotation}"
    summary = out_dir / f"cond_{stem}.json"
    samples = out_dir / f"cond_{stem}_samples.jsonl"
    if summary.exists() and samples.exists():
        try:
            saved = json.loads(summary.read_text())
            if int(saved["n"]) == expected_n and count_lines(samples) == expected_n:
                print(f"[run_one] SKIP complete n={expected_n}: {summary}", flush=True)
                return summary
        except (KeyError, ValueError, json.JSONDecodeError):
            pass
    out_dir.mkdir(parents=True, exist_ok=True)

    common = dict(limit=args.limit, items_path=str(items), out_dir=str(out_dir))
    if args.condition in {"clean", "decode_pi64", "decode_pid64", "decode_pi32"}:
        cond = {"clean": "base", "decode_pi64": "PI", "decode_pid64": "PID",
                "decode_pi32": "PI"}[args.condition]
        steps = 32 if args.condition == "decode_pi32" else 64
        tag = {"clean": "dpid_base", "decode_pi64": "dpid_PI",
               "decode_pid64": "dpid_PID", "decode_pi32": "dpid_PI_s32"}[args.condition]
        denoise_pid.run(cond=cond, kp=3.0, ki=0.1, kd=1.0, amax=6.0,
                        tag=tag, steps=steps, setpoint=0.9, amin=0.0,
                        sensor_case="upper", **common)
    elif args.condition in {"normal_a4", "normal_eff", "layer_pi_a2"}:
        if args.condition == "normal_a4":
            mode, cond, alpha, prefix = "normal", None, 4.0, None
        elif args.condition == "normal_eff":
            mode, cond, alpha, prefix = "normal", None, float(args.effort_alpha), "normalEff"
        else:
            mode, cond, alpha, prefix = "pid", "PI", 2.0, "layerPI"
        pid_steer.run(mode=mode, cond=cond, alpha=alpha,
                      arrows_path=str(arrows), source_layer=14,
                      gen_len=32, steps=64, blk=32, dummy_arrows=False,
                      tag_prefix=prefix, **common)
    elif args.condition == "caa_a16":
        caa.run(alpha=16, layer=14, tag=stem, **common)
    elif args.condition == "actadd_a16":
        actadd.run(alpha=16, layer=14, tag=stem, **common)
    elif args.condition == "meanact_s2":
        meanact.run(strength=2, direction="unit", tag=stem, **common)
    elif args.condition == "linearact_s1":
        linearact.run(variant="gaussian", strength=1, tag=stem, **common)
    elif args.condition == "aura_inject_g4":
        aura.run(mode="inject", gamma=4, tag=stem, **common)
    elif args.condition == "aura_vanilla":
        aura.run(mode="vanilla", tag=stem, **common)
    elif args.condition == "itic_k48_a8":
        itic.run(K=48, alpha=8, tag=stem, **common)
    elif args.condition == "itic_k48_a8_tb":
        itic.run(K=48, alpha=8, tag=stem, tiebreak="margin", **common)
    else:
        raise AssertionError(args.condition)

    if not summary.exists() or not samples.exists():
        raise AssertionError(f"runner did not create expected outputs: {summary}, {samples}")
    saved = json.loads(summary.read_text())
    if int(saved["n"]) != expected_n or count_lines(samples) != expected_n:
        raise AssertionError(f"incomplete output for {summary}")
    print(f"[run_one] PASS n={expected_n}: {summary}", flush=True)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--condition", choices=CONDITIONS, required=True)
    ap.add_argument("--polarity", choices=POLARITIES, required=True)
    ap.add_argument("--rotation", type=int, choices=(0, 1, 2), required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--effort-alpha", type=float, default=None)
    ap.add_argument("--out-root", type=Path,
                    default=ROOT / "results" / "socialstigma")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
