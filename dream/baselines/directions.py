#!/usr/bin/env python
"""dream/baselines/directions.py -- shared FIT PRIMITIVES for the Dream port.

The per-neuron OT / AURA math is MODEL-AGNOSTIC (pure numpy/torch), so we do NOT
reimplement it: gaussian_ot, empirical_ot_fit, auroc_per_neuron (+ _auroc_numpy)
are IMPORTED verbatim from the LLaDA baselines/directions.py (loaded by absolute
path to dodge the basename clash with THIS module).  The ONLY Dream-specific
override is load_arrows(): it points at dream/arrows.pt and expects the 28-layer,
3584-wide Dream arrow set instead of LLaDA's (32,4096).

INJECTION convention (unchanged): "Black" is the OT DESTINATION.  dream/arrows.pt
stores r = mean(h_black - h_other) (dream/build_arrows.py), i.e. it already points
toward Black; positive strength injects toward Black.

CLI:  python dream/baselines/directions.py --selftest   (offline, no GPU)
"""
import argparse
import importlib.util
import os
import sys

import torch

ROOT = "/home/lukas/users/shashmi/dlm_bias"
sys.path.insert(0, os.path.join(ROOT, "eval"))
sys.path.insert(0, os.path.join(ROOT, "steering"))
import pid_steer  # noqa: E402  (unit_rows)


# Import the pure primitives from the LLaDA baselines/directions.py VERBATIM.
def _load_llada_directions():
    path = os.path.join(ROOT, "baselines", "directions.py")
    spec = importlib.util.spec_from_file_location("llada_directions", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_LD = _load_llada_directions()
gaussian_ot = _LD.gaussian_ot            # per-neuron Gaussian 1-D OT map
empirical_ot_fit = _LD.empirical_ot_fit  # sorted-1D-OT + closed-form LS affine
auroc_per_neuron = _LD.auroc_per_neuron  # per-neuron AUROC + AURA gate
_auroc_numpy = _LD._auroc_numpy

# Dream arrow set: 28 layers x 3584 (vs LLaDA's steering/arrows.pt 32x4096).
# Point at THIS tree's dream/arrows.pt (dream/baselines/ -> dream/), so a worktree
# reads the arrows dream/build_arrows.py wrote in the same tree.
_HERE = os.path.dirname(os.path.abspath(__file__))
ARROWS_PATH = os.path.join(os.path.dirname(_HERE), "arrows.pt")
N_LAYERS = 28
D_MODEL = 3584


def load_arrows(path=ARROWS_PATH, raw=False):
    """Return the (28,3584) Black-minus-other arrow set from dream/arrows.pt.

    Default: per-layer unit-normalized (pid_steer.unit_rows).  raw=True: the raw
    (un-normed) diff-in-means (dream/build_arrows.py)."""
    blob = torch.load(path, map_location="cpu")
    r = blob["r"].to(torch.float32)          # (28,3584) raw
    return r if raw else pid_steer.unit_rows(r)


def _selftest():
    ok = True

    def check(name, cond):
        nonlocal ok
        ok &= bool(cond)
        print(f"[selftest-directions] {name:48s} : {'PASS' if cond else 'FAIL'}")

    # The primitives are the SAME objects as the LLaDA module (dim-free math).
    check("gaussian_ot imported from LLaDA baselines", gaussian_ot is _LD.gaussian_ot)
    check("empirical_ot_fit imported from LLaDA baselines",
          empirical_ot_fit is _LD.empirical_ot_fit)
    check("auroc_per_neuron imported from LLaDA baselines",
          auroc_per_neuron is _LD.auroc_per_neuron)

    # Tiny end-to-end sanity on the primitives (not re-deriving the LLaDA proofs).
    D = 8
    mu1 = torch.randn(D); mu2 = torch.randn(D); sig = torch.rand(D) + 0.5
    f = gaussian_ot(mu1, sig, mu2, sig)
    x = torch.randn(5, D)
    check("gaussian_ot pure shift == x + (mu2-mu1)",
          torch.allclose(f(x), x + (mu2 - mu1), atol=1e-5))
    a = torch.rand(6) * 2 + 0.5; b = torch.randn(6)
    src = torch.randn(2000, 6); dst = a * src + b
    beta, bias = empirical_ot_fit(src, dst)
    check("empirical_ot_fit recovers beta/bias",
          torch.allclose(beta, a, atol=5e-2) and torch.allclose(bias, b, atol=5e-2))

    # load_arrows shape/unit, only if the Dream arrows file exists (no GPU).
    if os.path.exists(ARROWS_PATH):
        u = load_arrows()
        norms = u.norm(dim=1)
        check("load_arrows unit rows: (28,3584), ||row||~1",
              tuple(u.shape) == (N_LAYERS, D_MODEL)
              and torch.allclose(norms, torch.ones(N_LAYERS), atol=1e-4))
    else:
        print(f"[selftest-directions] load_arrows: SKIP (no {ARROWS_PATH})")

    print(f"[selftest-directions] OVERALL: {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true",
                    help="offline check: primitives imported + load_arrows (no GPU)")
    args = ap.parse_args()
    if args.selftest:
        sys.exit(0 if _selftest() else 1)
    ap.error("nothing to do: pass --selftest")


if __name__ == "__main__":
    main()
