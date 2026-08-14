#!/usr/bin/env python
"""baselines/caa.py -- CAA (Contrastive Activation Addition), NATIVE granularity.

Faithful port of Contrastive Activation Addition (Rimsky, Gurnee, et al.,
"Steering Llama 2 via Contrastive Activation Addition", 2023) onto our LLaDA-8B
masked-diffusion BBQ bias-INJECTION harness.

WHAT CAA IS
-----------
CAA builds ONE steering vector as the difference-in-means between the residual
activations on contrastive (positive vs negative) prompt pairs, then ADDS that
vector to the residual stream at a SINGLE layer during generation, at every
token position.  There is no fit/optimization -- the vector IS the mean
difference.

  steer(h) = h + alpha * unit(mu_pos - mu_neg)          [applied at one layer]

MAPPING ONTO THIS REPO
----------------------
  * The contrastive vector is EXACTLY our diff-in-means arrow set
    steering/arrows.pt: r[k] = mean_i(h_black(i) - h_other(i)) at layer k
    (steering/build_arrows.py:182,190).  "Black" is the positive class, so the
    arrow already points toward Black -> POSITIVE alpha injects toward Black.
    We reuse it contamination-safely via directions.load_arrows() (per-layer
    unit-normalized; directions.py:39-47), never refitting.
  * NATIVE granularity for CAA is the residual stream at a single block
    (block-level, H=4096).  We add at model.transformer.blocks[layer] OUTPUT[0]
    via the shared common.add_vec_hook factory (block tuple contract preserved,
    all positions, every denoising step, bidirectional -- common.py:133-135).

FAITHFULNESS TO THE AcT REFERENCE
---------------------------------
The mean-shift edit is the same primitive AcT's mean map applies, cited here for
the shared reference tree:
  * OnlyMeanHook / GaussianOTHook.forward mean branch adds the mean difference
    to the activation: z_ot = (z - c*mu1) + c*mu2 = z + c*(mu2-mu1)
    (act/hooks/transport.py:258-259; the PID mean variant z + c*diff at
    transport.py:483-484 with diff = mu2 - mu1, transport.py:412).
  * AcT applies that edit per-neuron across ALL fitted layers; native CAA is the
    single-layer, whole-vector special case with a unit-normalized direction and
    a scalar strength (Rimsky et al.).  We keep CAA's single-layer, unit-vector
    form and expose the layer + strength as CLI knobs.
  * intervention_position="all" in the AcT hook (transport.py:87) == our
    all-token, bidirectional application (LLaDA masked diffusion; the hook fires
    ~steps times per item over every position). There is NO "last"-token mode.

ARTIFACT
--------
CAA needs no per-method fit: its only artifact is steering/arrows.pt, produced by
steering/build_arrows.py (NEEDS GPU).  --fit here documents/verifies that
prerequisite; it does NOT rebuild the arrows.

CLI
---
    python -m baselines.caa --selftest                 # offline math, no GPU
    python -m baselines.caa --fit                      # verify arrows.pt exists
    python -m baselines.caa --run --alpha 4 --layer 14 # eval (NEEDS GPU)
"""
import argparse
import os
import sys

import torch

# Harness import (ROOT = MAIN tree; mirrors steering/pid_steer.py:49-51).
ROOT = os.environ.get("DLM_BIAS_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "eval"))
sys.path.insert(0, os.path.join(ROOT, "steering"))

import common      # noqa: E402  (shared driver: hooks, attach, run_baseline)
import directions  # noqa: E402  (load_arrows -> the CAA contrastive vector)

from common import (  # noqa: E402,F401
    N_LAYERS,        # 32
    H_MODEL,         # 4096 residual width
    add_vec_hook,    # block-tuple-safe additive residual edit (increments _FIRE)
    attach,          # register hooks on dotted module paths
    block_paths,     # 'model.transformer.blocks.<k>'
    run_baseline,    # the shared item loop + result-file writer
    AddVec,          # pid_steer reference additive hook (used in selftest to
                     # cross-check that common's hook matches the repo's own)
    SWEEP400,
)

DEFAULT_LAYER = 14          # native CAA steers at a single mid-network layer
RESULTS_DIR = os.path.join(ROOT, "results", "caa")


# --------------------------------------------------------------------------- #
# The CAA steering vector.                                                     #
# --------------------------------------------------------------------------- #
def build_injection(layer=DEFAULT_LAYER, alpha=1.0, arrows=None, target="black"):
    """Return the CAA additive vector for one layer:  alpha * unit(r[layer]).

    r is the diff-in-means target-minus-other arrow set (steering/arrows.pt for
    the default target="black"; multirace/arrows_<target>.pt for the E3 gender
    targets -- same format and sign convention, see directions.load_arrows).
    directions.load_arrows() already unit-normalizes each layer row
    (directions.py:39-47), so load_arrows()[layer] == unit(r[layer]); we scale by
    alpha.  POSITIVE alpha injects toward the target (the arrow points
    target-minus-other).

    arrows: optional pre-loaded (32,H) UNIT arrow set (skips the file read; the
    selftest passes a synthetic one).  Returns a (H,) float32 tensor.
    """
    u = directions.load_arrows(target=target) if arrows is None else arrows
    assert 0 <= int(layer) < u.shape[0], f"layer {layer} out of range [0,{u.shape[0]})"
    return (float(alpha) * u[int(layer)]).to(torch.float32)


def make_attach_fn(layer=DEFAULT_LAYER, alpha=1.0, arrows=None, target="black"):
    """Build attach_fn(model) -> handles that adds the CAA vector at ONE block.

    Uses common.add_vec_hook (block tuple contract + shared fire counter) on the
    single dotted path 'model.transformer.blocks.<layer>'.  All positions, every
    denoising step, bidirectional.
    """
    vec = build_injection(layer=layer, alpha=alpha, arrows=arrows, target=target)

    def attach_fn(model):
        return attach(model, block_paths([int(layer)]), add_vec_hook(vec), pre=False)

    return attach_fn


# --------------------------------------------------------------------------- #
# Eval driver (NEEDS GPU).                                                     #
# --------------------------------------------------------------------------- #
def run(alpha=1.0, layer=DEFAULT_LAYER, out_dir=RESULTS_DIR, tag=None,
        items_path=None, limit=0, baseline_black_rate=None,
        model=None, tok=None, target="black", **run_kwargs):
    """Run one CAA condition end-to-end via common.run_baseline.  NEEDS A GPU.

    target="black" (default) is byte-identical to round 1 (arrows.pt direction,
    black_idx classification, _sweep400.jsonl items when items_path is None).
    target="woman"/"man" uses multirace/arrows_<target>.pt and classifies
    against the WOMAN/MAN option (multirace/targets.py registry)."""
    if tag is None:
        tag = f"caa_L{int(layer)}_a{alpha:g}"
    attach_fn = make_attach_fn(layer=layer, alpha=alpha, target=target)
    return run_baseline(
        attach_fn,
        items_path=items_path,
        out_dir=out_dir,
        tag=tag,
        limit=limit,
        model=model,
        tok=tok,
        baseline_black_rate=baseline_black_rate,
        target=target,
        config_extra={
            "method": "caa",
            "granularity": "block_residual_single_layer",
            "layer": int(layer),
            "alpha": float(alpha),
            "target": target,
            "direction": f"unit_diff_in_means_{target}_minus_other",
            "arrows_path": directions.arrows_path_for(target),
            "intervention_position": "all",
            "reference": "Rimsky et al. Contrastive Activation Addition; "
                         "AcT mean map transport.py:258-259",
        },
        **run_kwargs,
    )


# --------------------------------------------------------------------------- #
# Fit prerequisite check (the ONLY artifact is arrows.pt; NEEDS GPU to build).  #
# --------------------------------------------------------------------------- #
def fit(target="black"):
    """CAA has no per-method fit.  Its contrastive vector is steering/arrows.pt
    (black) or multirace/arrows_<target>.pt (gender), built by the respective
    build_arrows.py (NEEDS GPU).  Report whether it exists."""
    p = directions.arrows_path_for(target)
    builder = ("steering/build_arrows.py" if target == "black"
               else f"multirace/build_arrows.py --target {target}")
    if os.path.exists(p):
        blob = torch.load(p, map_location="cpu")
        r = blob["r"]
        print(f"[caa:fit] arrows present: {p}  r={tuple(r.shape)} "
              f"n_items={blob.get('n_items')} -- CAA needs no further fit.")
    else:
        print(f"[caa:fit] MISSING {p}. Build it (NEEDS GPU):\n"
              f"    cd {ROOT} && python {builder}\n"
              f"CAA reuses that diff-in-means arrow set as its steering vector.")


# --------------------------------------------------------------------------- #
# Offline self-test: vector math + single-layer attach on synthetic tensors.   #
# --------------------------------------------------------------------------- #
def _selftest():
    torch.manual_seed(0)
    ok = True

    def check(name, cond):
        nonlocal ok
        ok &= bool(cond)
        print(f"[selftest-caa] {name:52s} : {'PASS' if cond else 'FAIL'}")

    H = H_MODEL
    # Synthetic UNIT arrow set (stand-in for directions.load_arrows()).
    raw = torch.randn(N_LAYERS, H)
    u = torch.stack([raw[k] / raw[k].norm() for k in range(N_LAYERS)], 0)

    layer, alpha = DEFAULT_LAYER, 4.0
    vec = build_injection(layer=layer, alpha=alpha, arrows=u)

    check("build_injection shape (H,)", tuple(vec.shape) == (H,))
    check("||vec|| == |alpha| (unit direction, scaled)",
          torch.allclose(vec.norm(), torch.tensor(float(abs(alpha))), atol=1e-4))
    check("vec direction == unit(r[layer])",
          torch.allclose(vec / vec.norm(), u[layer], atol=1e-5))
    # POSITIVE alpha injects toward Black: aligns with the Black-minus-other arrow.
    check("positive alpha aligns with arrow (toward Black)",
          float(torch.dot(vec, u[layer])) > 0)
    neg = build_injection(layer=layer, alpha=-alpha, arrows=u)
    check("negative alpha flips direction (away from Black)",
          float(torch.dot(neg, u[layer])) < 0
          and torch.allclose(neg, -vec, atol=1e-6))

    # Single-layer targeting: exactly one dotted block path, correct index.
    paths = block_paths([layer])
    check("block_paths selects ONE block", len(paths) == 1)
    check("path is transformer.blocks.<layer>",
          paths[0].endswith(f".blocks.{layer}"))

    # Hook math on the LLaDA block tuple contract (hidden, cache), all positions.
    B, S = 2, 5
    h = torch.randn(B, S, H)
    tup = (h.clone(), "cache_sentinel")
    common.reset_fire_count()
    hook = add_vec_hook(vec)
    r = hook(None, None, tup)
    check("add_vec_hook preserves (hidden, cache) tuple",
          isinstance(r, tuple) and len(r) == 2 and r[1] == "cache_sentinel")
    check("edit adds vec at EVERY position (broadcast over B,seq)",
          torch.allclose(r[0], h + vec))
    check("fire counter bumped (run_baseline asserts >0)",
          common.get_fire_count() == 1)

    # Cross-check: common.add_vec_hook matches the repo's own AddVec math.
    fired = {}
    ref = AddVec(layer, vec, fired)
    ref_out = ref._hook(None, None, (h.clone(), "c"))
    check("matches pid_steer.AddVec reference edit",
          torch.allclose(ref_out[0], r[0], atol=1e-6) and fired[layer] == 1)

    # make_attach_fn wiring: fabricate a 32-block model and confirm ONLY the
    # target block is edited and fires.
    import torch.nn as nn

    class Blk(nn.Module):
        def __init__(self, i):
            super().__init__(); self.i = i
        def forward(self, x):
            return (x, None)   # LLaDA block returns (hidden, cache)

    class TF(nn.Module):
        def __init__(self):
            super().__init__(); self.blocks = nn.ModuleList([Blk(i) for i in range(N_LAYERS)])

    class Inner(nn.Module):
        def __init__(self):
            super().__init__(); self.transformer = TF()

    class Model(nn.Module):  # BLOCKS_PATH = 'model.transformer.blocks'
        def __init__(self):
            super().__init__(); self.model = Inner()

    model = Model()
    common.reset_fire_count()
    attach_fn = make_attach_fn(layer=layer, alpha=alpha, arrows=u)
    handles = attach_fn(model)
    try:
        check("attach_fn returns exactly ONE handle", len(handles) == 1)
        x0 = torch.randn(1, 3, H)
        blocks = model.model.transformer.blocks
        edited = blocks[layer](x0)[0]
        other = blocks[(layer + 1) % N_LAYERS](x0)[0]
        check("target block adds vec", torch.allclose(edited, x0 + vec, atol=1e-5))
        check("non-target block unchanged", torch.allclose(other, x0))
        check("only target block fired once", common.get_fire_count() == 1)
    finally:
        for hd in handles:
            hd.remove()

    print(f"[selftest-caa] OVERALL: {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true",
                    help="offline vector-math + single-layer attach check (no GPU)")
    ap.add_argument("--fit", action="store_true",
                    help="verify the arrows.pt prerequisite (build NEEDS GPU)")
    ap.add_argument("--run", action="store_true",
                    help="run the BBQ eval for one CAA condition (NEEDS GPU)")
    ap.add_argument("--alpha", type=float, default=1.0,
                    help="steering strength (positive -> inject toward Black); sweepable")
    ap.add_argument("--layer", type=int, default=DEFAULT_LAYER,
                    help=f"single block to steer (default {DEFAULT_LAYER})")
    ap.add_argument("--target", choices=common.SUPPORTED_TARGETS, default="black",
                    help="steering target (black = round-1 default; woman/man = "
                         "E3 gender, multirace arrows + classification)")
    ap.add_argument("--items", default=None,
                    help="BBQ items jsonl (e.g. a position-balance rotation "
                         "file); default = the target's own _sweep400 file")
    ap.add_argument("--out_dir", default=RESULTS_DIR)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--baseline_black_rate", type=float, default=None)
    args = ap.parse_args()

    if args.selftest:
        sys.exit(0 if _selftest() else 1)
    if args.fit:
        fit(target=args.target)
        return
    if args.run:
        run(alpha=args.alpha, layer=args.layer, out_dir=args.out_dir, tag=args.tag,
            items_path=args.items, limit=args.limit,
            baseline_black_rate=args.baseline_black_rate, target=args.target)
        return
    ap.error("nothing to do: pass --selftest (offline), --fit (verify arrows), "
             "or --run (GPU eval)")


if __name__ == "__main__":
    main()
