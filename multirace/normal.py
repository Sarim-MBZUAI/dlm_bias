#!/usr/bin/env python
"""multirace/normal.py -- target-parameterized open-loop normal-vector baseline (LLaDA).

Port of steering/pid_steer.py's `--mode normal` path: vhat = unit(r[14]) from
multirace/arrows_<target>.pt, and the SAME fixed alpha*vhat is injected at ALL 32
blocks every denoising step (default alpha 4). Eval loop + result schema shared
with caa.py via common_eval.run_openloop (target / nontarget / abstain /
unparseable; gap_within; d_gap null unless --baseline-rate).

MODES
    --selftest                          offline injection-shape check (no GPU).
    --target T [--alpha 4] [...]        eval; out-dir results/multirace/<target>/normal.
"""
import argparse
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common_eval as CE  # noqa: E402


def build_inject(r, alpha, layer=CE.LAYER):
    """{block: alpha*unit(r[layer])} for ALL 32 blocks (same vector everywhere)."""
    v = alpha * CE.vhat_of(r, layer)
    return {k: v for k in range(CE.N_LAYERS)}


def selftest():
    torch.manual_seed(0)
    r = torch.randn(CE.N_LAYERS, CE.D_MODEL)
    alpha = 4.0
    inj = build_inject(r, alpha)
    vhat = r[CE.LAYER] / r[CE.LAYER].norm()
    ok = True

    def check(name, cond):
        nonlocal ok
        ok &= bool(cond)
        print(f"[selftest-normal] {name:48s} : {'PASS' if cond else 'FAIL'}")

    check("covers all 32 blocks", sorted(inj) == list(range(CE.N_LAYERS)))
    check("each vec shape == (4096,)",
          all(tuple(v.shape) == (CE.D_MODEL,) for v in inj.values()))
    check("each vec == alpha*vhat (norm==alpha)",
          all(torch.allclose(v, alpha * vhat, atol=1e-6) for v in inj.values())
          and abs(inj[0].norm().item() - alpha) < 1e-4)
    print(f"[selftest-normal] OVERALL: {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--target", choices=CE.TARGETS)
    ap.add_argument("--alpha", type=float, default=4.0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--items", default=None)
    ap.add_argument("--arrows", default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--dummy-arrows", action="store_true")
    ap.add_argument("--baseline-rate", type=float, default=None)
    args = ap.parse_args()

    if args.selftest:
        sys.exit(0 if selftest() else 1)
    if not args.target:
        ap.error("--target required (or use --selftest)")
    CE.warn_if_fallback()
    r, meta = CE.load_r(args.arrows or CE.default_arrows(args.target), args.dummy_arrows)
    CE.run_openloop(
        "normal", args.target, build_inject(r, args.alpha), args.alpha, meta,
        args.items or CE.default_items(args.target), args.limit,
        args.out_dir or os.path.join(CE.ROOT, "results", "multirace", args.target, "normal"),
        args.tag, args.baseline_rate,
        extra_config={"mode": "normal_all32", "dummy_arrows": bool(args.dummy_arrows)})


if __name__ == "__main__":
    main()
