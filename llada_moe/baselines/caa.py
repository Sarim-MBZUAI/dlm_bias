#!/usr/bin/env python
"""llada_moe/baselines/caa.py -- CAA (Contrastive Activation Addition) on
LLaDA-MoE-7B-A1B-Instruct.  Port of dream/baselines/caa.py.

CAA builds ONE steering vector as the diff-in-means between residual activations
on contrastive (Black vs other) pairs, then ADDS it to the residual stream at a
SINGLE layer, every token position.  No fit.

  steer(h) = h + alpha * unit(mu_black - mu_other)      [applied at one layer]

MoE MAPPING
  * The contrastive vector IS llada_moe/arrows.pt: r[k] = mean(h_black - h_other)
    at block k (llada_moe/build_arrows.py).  directions.load_arrows()
    unit-normalizes each layer row, so load_arrows()[layer] == unit(r[layer]);
    positive alpha -> Black.
  * NATIVE granularity = residual at ONE block (2048-d).  We add at
    model.layers[layer] OUTPUT[0] via common_lladamoe.add_vec_hook (block TUPLE
    contract preserved, all positions, every diffusion step).

Single-layer, unit-vector, scalar-alpha CAA (Rimsky et al.); layer + alpha are CLI
knobs (default layer 8 = the 16-layer midpoint, matching LLaDA/Dream's mid-stack
L14 convention).  CAA needs no per-method fit: its only artifact is
llada_moe/arrows.pt (llada_moe/build_arrows.py; NEEDS GPU).  --fit here only
verifies that prerequisite.

CLI:
    python llada_moe/baselines/caa.py --selftest              # offline math, no GPU
    python llada_moe/baselines/caa.py --fit                   # verify arrows.pt exists
    python llada_moe/baselines/caa.py --run --alpha 2 --layer 8    # GPU (SLURM sets it)
"""
import argparse
import os
import sys

import torch

ROOT = os.environ.get("DLM_BIAS_ROOT") or os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "eval"))
sys.path.insert(0, os.path.join(ROOT, "steering"))
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))   # common_lladamoe
sys.path.insert(0, _HERE)                     # sibling modules

import common_lladamoe as C  # noqa: E402  (re-exports AddVec from the LLaDA steering/pid_steer.py)
import directions       # noqa: E402  (load_arrows -> the CAA contrastive vector)

DEFAULT_LAYER = 8
RESULTS_DIR = os.path.join(ROOT, "results", "lladamoe", "caa")


def build_injection(layer=DEFAULT_LAYER, alpha=1.0, arrows=None):
    """Return the CAA additive vector for one layer:  alpha * unit(r[layer]).

    directions.load_arrows() already unit-normalizes each row; we scale by alpha.
    arrows: optional pre-loaded (16,H) UNIT arrow set (the selftest passes one)."""
    u = directions.load_arrows() if arrows is None else arrows
    assert 0 <= int(layer) < u.shape[0], f"layer {layer} out of range [0,{u.shape[0]})"
    return (float(alpha) * u[int(layer)]).to(torch.float32)


def make_attach_fn(layer=DEFAULT_LAYER, alpha=1.0, arrows=None):
    """attach_fn(model)->handles adding the CAA vector at ONE block output."""
    vec = build_injection(layer=layer, alpha=alpha, arrows=arrows)

    def attach_fn(model):
        return C.attach(model, [C.block_path(int(layer))], C.add_vec_hook(vec))

    return attach_fn


def run(alpha=1.0, layer=DEFAULT_LAYER, out_dir=RESULTS_DIR, tag=None,
        items_path=None, limit=0, baseline_black_rate=None, model=None, tok=None):
    """Run one CAA condition end-to-end via common_lladamoe.run_items.  NEEDS A GPU."""
    if tag is None:
        tag = f"caa_L{int(layer)}_a{alpha:g}"
    return C.run_items(
        attach_fn=make_attach_fn(layer=layer, alpha=alpha),
        items_path=items_path or C.SWEEP400,
        out_dir=out_dir, tag=tag, limit=limit, model=model, tok=tok,
        baseline_black_rate=baseline_black_rate,
        config_extra={
            "method": "caa",
            "granularity": "block_residual_single_layer",
            "layer": int(layer), "alpha": float(alpha),
            "direction": "unit_diff_in_means_black_minus_other",
            "arrows_path": directions.ARROWS_PATH,
            "intervention_position": "all",
            "reference": "Rimsky et al. Contrastive Activation Addition",
        })


def fit():
    """CAA has no per-method fit.  Its vector is llada_moe/arrows.pt
    (llada_moe/build_arrows.py; NEEDS GPU).  Report whether it exists."""
    p = directions.ARROWS_PATH
    if os.path.exists(p):
        blob = torch.load(p, map_location="cpu")
        print(f"[caa:fit] arrows present: {p}  r={tuple(blob['r'].shape)} "
              f"n_items={blob.get('n_items')} -- CAA needs no further fit.")
    else:
        print(f"[caa:fit] MISSING {p}. Build it (NEEDS GPU):\n"
              f"    python llada_moe/build_arrows.py")


def _selftest():
    torch.manual_seed(0)
    ok = True

    def check(name, cond):
        nonlocal ok
        ok &= bool(cond)
        print(f"[selftest-caa] {name:52s} : {'PASS' if cond else 'FAIL'}")

    H, L = C.D_MODEL, C.N_LAYERS
    raw = torch.randn(L, H)
    u = torch.stack([raw[k] / raw[k].norm() for k in range(L)], 0)

    layer, alpha = DEFAULT_LAYER, 4.0
    vec = build_injection(layer=layer, alpha=alpha, arrows=u)
    check("build_injection shape (H,)", tuple(vec.shape) == (H,))
    check("||vec|| == |alpha| (unit direction scaled)",
          torch.allclose(vec.norm(), torch.tensor(float(abs(alpha))), atol=1e-4))
    check("vec direction == unit(r[layer])",
          torch.allclose(vec / vec.norm(), u[layer], atol=1e-5))
    check("positive alpha aligns with arrow (toward Black)",
          float(torch.dot(vec, u[layer])) > 0)
    neg = build_injection(layer=layer, alpha=-alpha, arrows=u)
    check("negative alpha flips direction",
          float(torch.dot(neg, u[layer])) < 0 and torch.allclose(neg, -vec, atol=1e-6))

    # Hook math on the MoE block tuple contract (hidden,)+..., all positions.
    B, S = 2, 5
    h = torch.randn(B, S, H)
    tup = (h.clone(), "attn_sentinel")
    C.reset_fire_count()
    r = C.add_vec_hook(vec)(None, None, tup)
    check("add_vec_hook preserves (hidden,)+... tuple",
          isinstance(r, tuple) and len(r) == 2 and r[1] == "attn_sentinel")
    check("edit adds vec at EVERY position", torch.allclose(r[0], h + vec))
    check("fire counter bumped", C.get_fire_count() == 1)

    # Cross-check against pid_steer.AddVec (repo's own additive edit).
    fired = {}
    ref = C.AddVec(layer, vec, fired)
    ref_out = ref._hook(None, None, (h.clone(), "c"))
    check("matches pid_steer.AddVec reference edit",
          torch.allclose(ref_out[0], r[0], atol=1e-6) and fired[layer] == 1)

    # make_attach_fn wiring on a fabricated 16-block MoE-shaped model.
    import torch.nn as nn

    class Blk(nn.Module):
        def forward(self, x):
            return (x,)   # LLaDAMoEDecoderLayer returns a TUPLE (hidden,)+...

    class Inner(nn.Module):
        def __init__(self):
            super().__init__(); self.layers = nn.ModuleList([Blk() for _ in range(L)])

    class Model(nn.Module):
        def __init__(self):
            super().__init__(); self.model = Inner()   # BLOCKS_PATH=model.layers

    model = Model()
    C.reset_fire_count()
    handles = make_attach_fn(layer=layer, alpha=alpha, arrows=u)(model)
    try:
        check("attach_fn returns exactly ONE handle", len(handles) == 1)
        x0 = torch.randn(1, 3, H)
        blocks = model.model.layers
        edited = blocks[layer](x0)[0]
        other = blocks[(layer + 1) % L](x0)[0]
        check("target block adds vec", torch.allclose(edited, x0 + vec, atol=1e-5))
        check("non-target block unchanged", torch.allclose(other, x0))
        check("only target block fired once", C.get_fire_count() == 1)
    finally:
        for hd in handles:
            hd.remove()

    print(f"[selftest-caa] OVERALL: {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--fit", action="store_true", help="verify arrows.pt prerequisite")
    ap.add_argument("--run", action="store_true", help="run BBQ eval (NEEDS GPU)")
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--layer", type=int, default=DEFAULT_LAYER)
    ap.add_argument("--items", default=None,
                    help="BBQ items jsonl (default: sweep400; e.g. a position-balance rotation file)")
    ap.add_argument("--out_dir", default=RESULTS_DIR)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--baseline_black_rate", type=float, default=None)
    args = ap.parse_args()

    if args.selftest:
        sys.exit(0 if _selftest() else 1)
    if args.fit:
        fit(); return
    if args.run:
        run(alpha=args.alpha, layer=args.layer, out_dir=args.out_dir, tag=args.tag,
            items_path=args.items, limit=args.limit,
            baseline_black_rate=args.baseline_black_rate)
        return
    ap.error("nothing to do: pass --selftest, --fit, or --run")


if __name__ == "__main__":
    main()
