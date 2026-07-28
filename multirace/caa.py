#!/usr/bin/env python
"""multirace/caa.py -- target-parameterized faithful single-layer CAA baseline (LLaDA).

Faithful CAA (verified single-layer per the paper): alpha*unit(r[layer]) injected
at ONE transformer block only (default layer 14, alpha 2), every denoising step.
Arrows from multirace/arrows_<target>.pt. Eval loop + result schema shared with
normal.py via common_eval.run_openloop.

MODES
    --selftest                                   offline injection-shape check (no GPU).
    --target T [--layer 14] [--alpha 2] [...]    eval; out-dir results/multirace/<target>/caa.
"""
import argparse
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common_eval as CE  # noqa: E402


def build_inject(r, alpha, layer):
    """{layer: alpha*unit(r[layer])} -- ONE block only."""
    return {layer: alpha * CE.vhat_of(r, layer)}


def selftest():
    torch.manual_seed(0)
    r = torch.randn(CE.N_LAYERS, CE.D_MODEL)
    alpha, layer = 2.0, 14
    inj = build_inject(r, alpha, layer)
    vhat = r[layer] / r[layer].norm()
    ok = True

    def check(name, cond):
        nonlocal ok
        ok &= bool(cond)
        print(f"[selftest-caa] {name:48s} : {'PASS' if cond else 'FAIL'}")

    check("exactly ONE block steered (layer 14)", sorted(inj) == [layer])
    check("vec shape == (4096,)", tuple(inj[layer].shape) == (CE.D_MODEL,))
    check("vec == alpha*unit(r[layer]) (norm==alpha)",
          torch.allclose(inj[layer], alpha * vhat, atol=1e-6)
          and abs(inj[layer].norm().item() - alpha) < 1e-4)
    inj9 = build_inject(r, alpha, 9)
    check("--layer respected (layer 9 distinct)",
          sorted(inj9) == [9] and not torch.allclose(inj9[9], inj[layer], atol=1e-3))
    print(f"[selftest-caa] OVERALL: {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--target", choices=CE.TARGETS)
    ap.add_argument("--layer", type=int, default=14)
    ap.add_argument("--alpha", type=float, default=2.0)
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
        "caa", args.target, build_inject(r, args.alpha, args.layer), args.alpha, meta,
        args.items or CE.default_items(args.target), args.limit,
        args.out_dir or os.path.join(CE.ROOT, "results", "multirace", args.target, "caa"),
        args.tag, args.baseline_rate,
        extra_config={"mode": f"caa_single_layer_{args.layer}", "caa_layer": args.layer,
                      "dummy_arrows": bool(args.dummy_arrows)})


if __name__ == "__main__":
    main()
