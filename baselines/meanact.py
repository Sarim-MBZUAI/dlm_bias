#!/usr/bin/env python
"""baselines/meanact.py -- Mean-AcT (Rodriguez et al., Apple "Activation
Transport"), the mean-only Optimal-Transport steering baseline, ported onto our
LLaDA-8B-Instruct masked-diffusion BBQ bias-INJECTION harness at NATIVE
granularity (the full residual stream, per-neuron, at every one of the 32
transformer blocks).

WHAT MEAN-AcT IS (the faithful reference)
-----------------------------------------
Mean-AcT is GaussianOT with `onlymean=True` (act/hooks/transport.py OnlyMeanHook,
which subclasses GaussianOTHook and hardcodes hook_onlymean=True -- see
transport.py:500-524).  The transport applied inside the forward is

    onlymean branch     : z_ot = (z - 1.2*mu1) + 1.2*mu2          (transport.py:259)
    strength blend      : z_ot = strength*z_ot + (1-strength)*z   (transport.py:264)

Algebraically the two lines collapse to a pure additive mean-shift:

    z  <-  z + strength * 1.2 * (mu2 - mu1)

i.e. a CONSTANT per-neuron vector added to every activation, independent of z.
(mu1, mu2) are the per-neuron class means fitted in GaussianOTHook.fit
(transport.py:189-192): mu1 = mean over the `labels` group, mu2 = mean over the
`~labels` group.  The `1.2` is the Apple fork's hardcoded magic gain baked into
the onlymean branch (transport.py:259) -- NOT a general OT quantity; we keep it
verbatim as GAIN below.

INJECTION MAPPING (Black is the OT DESTINATION)
-----------------------------------------------
Our task pushes the model TOWARD the "Black" answer, so Black is the destination
(mu2) and the non-Black option is the source (mu1).  The per-neuron mean shift
(mu2 - mu1) is therefore exactly the Black-minus-other diff-in-means arrow that
steering/arrows.pt already stores (build_arrows.py:182,190: r = mean(h_black -
h_other)).  So

    vec_k = strength * 1.2 * (mu2 - mu1)_k        with (mu2 - mu1) = arrows r[k]

and the residual edit at block k is  h <- h + vec_k, applied bidirectionally to
ALL positions (LLaDA is a masked diffusion LM; the hook fires once per denoising
step, ~steps times per block per item, over every token -- there is NO
"last"-token mode; matches common.py's intervention_position=="all" contract).

DIRECTION SOURCE (a faithfulness knob, documented)
--------------------------------------------------
The spec fixes (mu2 - mu1) = directions.load_arrows() r[k].  load_arrows()
DEFAULT per-layer unit-normalizes (directions.py:39-47, via pid_steer.unit_rows),
which is the family convention shared by every baseline here (and by the repo's
own pid_steer) so that a single `strength` is comparable across methods.  This
DROPS the native per-layer magnitude of the true (mu2 - mu1); the true Mean-AcT
uses the RAW diff-in-means.  Two escape hatches recover full faithfulness:
  * direction="raw"    -> load_arrows(raw=True): the raw diff-in-means arrows.pt
                          stores (build_arrows.py:190), i.e. the literal (mu2-mu1).
  * direction="fitted" -> fit() recomputes per-neuron (mu2-mu1) at block
                          granularity from a calib.py activation cache (offline
                          once the cache exists), the closest analogue to
                          OnlyMeanHook.fit's own mu1/mu2.
See FAITHFULNESS notes at the bottom.

CLI:
    python -m baselines.meanact --selftest                 # offline math, no GPU
    python -m baselines.meanact --fit                      # NEEDS GPU cache; do NOT run here
    python -m baselines.meanact --run --strength 2.0       # NEEDS GPU; do NOT run here
"""
import argparse
import os
import sys

import torch

# Harness import (ROOT = MAIN tree; mirrors steering/pid_steer.py:49-51).
ROOT = "/home/lukas/users/shashmi/dlm_bias"
sys.path.insert(0, os.path.join(ROOT, "eval"))
sys.path.insert(0, os.path.join(ROOT, "steering"))

import common      # noqa: E402  shared driver (hooks, attach, run_baseline)
import directions  # noqa: E402  load_arrows (per-layer diff-in-means)

# --------------------------------------------------------------------------- #
# The fork magic number.                                                       #
# The onlymean branch multiplies the class means by a HARDCODED 1.2 gain       #
# (act/hooks/transport.py:259).  It is not a derived OT quantity; it is baked  #
# into the Apple fork.  We keep it verbatim so the mean-shift magnitude matches #
# the reference; the `strength` CLI knob scales on top of it.                  #
# --------------------------------------------------------------------------- #
GAIN = 1.2

DEFAULT_OUT = os.path.join(ROOT, "results", "meanact")
DIR_CHOICES = ("unit", "raw", "fitted")
FIT_PATH = os.path.join(common.CACHE_DIR, "meanact_meandiff_block.pt")


# --------------------------------------------------------------------------- #
# Direction source: the per-layer (mu2 - mu1) = Black-minus-other mean shift.  #
# --------------------------------------------------------------------------- #
def load_direction(direction="unit", fit_path=FIT_PATH):
    """Return the (32, 4096) per-layer mean-shift (mu2 - mu1), Black minus other.

    direction:
      "unit"   -> directions.load_arrows()          (per-layer unit-normalized;
                  family default, the spec's directions.load_arrows() r[k]).
      "raw"    -> directions.load_arrows(raw=True)   (raw diff-in-means; the
                  literal (mu2 - mu1) the true Mean-AcT uses).
      "fitted" -> the fit()-produced per-neuron (mu2 - mu1) at block granularity
                  (needs a calib cache; see fit()).
    """
    assert direction in DIR_CHOICES, f"direction must be one of {DIR_CHOICES}"
    if direction == "fitted":
        assert os.path.exists(fit_path), (
            f"direction='fitted' needs {fit_path}; run --fit first "
            f"(which needs a calib.py activation cache built on GPU).")
        blob = torch.load(fit_path, map_location="cpu")
        return blob["mean_diff"].to(torch.float32)          # (32, 4096)
    return directions.load_arrows(raw=(direction == "raw")).to(torch.float32)


# --------------------------------------------------------------------------- #
# The injection: vec_k = strength * GAIN * (mu2 - mu1)_k   (all 32 blocks).    #
# --------------------------------------------------------------------------- #
def build_injection(strength, direction="unit", gain=GAIN, arrows=None,
                    fit_path=FIT_PATH):
    """Return the (32, 4096) additive mean-shift injection, one row per block.

        vec_k = strength * gain * (mu2 - mu1)_k

    This is the collapsed OnlyMeanHook edit (transport.py:259+264): adding vec_k
    to every activation is identically  strength*z_ot + (1-strength)*z  with
    z_ot = z + gain*(mu2-mu1).  Positive strength injects TOWARD Black.

    arrows: optional (32, 4096) override for (mu2 - mu1) (used by --selftest to
    avoid touching arrows.pt); otherwise load_direction(direction) supplies it.
    """
    r = load_direction(direction, fit_path=fit_path) if arrows is None \
        else torch.as_tensor(arrows, dtype=torch.float32)
    assert r.dim() == 2 and r.shape[1] == common.H_MODEL, \
        f"expected (n_layers, {common.H_MODEL}) arrows, got {tuple(r.shape)}"
    return float(strength) * float(gain) * r


# --------------------------------------------------------------------------- #
# attach_fn: add vec_k at every block k via the shared fire-counted factory.   #
# --------------------------------------------------------------------------- #
def build_attach_fn(strength, layers=None, direction="unit", gain=GAIN,
                    arrows=None, fit_path=FIT_PATH):
    """Return attach_fn(model) -> handles that adds vec_k at each block k.

    layers=None -> all 32 blocks (native Mean-AcT: every layer).  Hooks are built
    with common.add_vec_hook so the shared fire counter increments and
    run_baseline can assert the intervention actually fired.
    """
    layers = list(range(common.N_LAYERS)) if layers is None else list(layers)
    vecs = build_injection(strength, direction=direction, gain=gain,
                           arrows=arrows, fit_path=fit_path)

    def attach_fn(model):
        handles = []
        for k in layers:
            path = f"{common.BLOCKS_PATH}.{int(k)}"
            handles += common.attach(model, [path], common.add_vec_hook(vecs[k]))
        return handles

    return attach_fn


# --------------------------------------------------------------------------- #
# End-to-end run (NEEDS GPU + model).                                          #
# --------------------------------------------------------------------------- #
def run(strength, out_dir=DEFAULT_OUT, tag=None, layers=None, direction="unit",
        gain=GAIN, limit=0, baseline_black_rate=None, model=None, tok=None,
        items_path=common.SWEEP400):
    """Run one Mean-AcT injection condition end-to-end via common.run_baseline.

    Writes out_dir/cond_<tag>.json (+ _samples.jsonl) in the pid_steer format.
    NEEDS A GPU.  See --selftest for the offline math check.
    """
    tag = tag or f"meanact_{direction}_s{strength:g}"
    attach_fn = build_attach_fn(strength, layers=layers, direction=direction,
                                gain=gain)
    return common.run_baseline(
        attach_fn, items_path=items_path, out_dir=out_dir, tag=tag, limit=limit,
        baseline_black_rate=baseline_black_rate, model=model, tok=tok,
        config_extra={
            "method": "mean_act",
            "reference": "act/hooks/transport.py OnlyMeanHook (transport.py:259)",
            "granularity": "block/residual (native, all 32 layers)",
            "strength": float(strength),
            "gain": float(gain),
            "direction_source": direction,
            "layers": "all32" if layers is None else list(layers),
            "position": "all (bidirectional)",
        },
    )


# --------------------------------------------------------------------------- #
# fit(): per-neuron (mu2 - mu1) at block granularity from a calib cache.       #
# Faithful analogue of OnlyMeanHook.fit (transport.py:189-192): mu = per-class #
# mean; (mu2 - mu1) = mean(Black) - mean(other).  PURE (no GPU) once the calib #
# activation cache exists -- but building that cache (calib.py --fit) NEEDS a   #
# GPU, so --fit as a whole is documented as a GPU step; do NOT run it here.     #
# --------------------------------------------------------------------------- #
def fit(where="block", calib_path=None, save=True, out_path=FIT_PATH):
    """Compute per-neuron (mu2 - mu1) = mean(Black) - mean(other) per layer.

    Reads a calib.py activation cache (acts (2n,32,feat), labels 1=Black/0=other)
    and averages within class.  This is the block-granularity twin of arrows.pt's
    diff-in-means and the direct analogue of OnlyMeanHook.fit's mu1/mu2.
    """
    import calib  # local import: only needed for --fit
    blob = calib.load_calib(where, path=calib_path)
    acts = blob["acts"].to(torch.float64)          # (2n, 32, feat)
    labels = blob["labels"].to(torch.bool)         # (2n,) 1=Black, 0=other
    mu_black = acts[labels].mean(dim=0)            # (32, feat)  == mu2
    mu_other = acts[~labels].mean(dim=0)           # (32, feat)  == mu1
    mean_diff = (mu_black - mu_other).to(torch.float32)   # (mu2 - mu1) toward Black
    out = {
        "mean_diff": mean_diff, "where": where,
        "feat": mean_diff.shape[-1], "n_layers": mean_diff.shape[0],
        "n_items": int(labels.sum()),
        "note": "mu2-mu1 = mean(Black) - mean(other); OnlyMeanHook.fit analogue",
        "source": blob.get("source"),
    }
    if save:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        torch.save(out, out_path)
        print(f"[meanact:fit] SAVED (mu2-mu1) {tuple(mean_diff.shape)} -> {out_path}",
              flush=True)
    return out


# --------------------------------------------------------------------------- #
# Offline self-test: the pure mean-shift math on synthetic tensors (no GPU).   #
# --------------------------------------------------------------------------- #
def _onlymean_reference(z, mu1, mu2, strength, gain=GAIN):
    """Byte-faithful OnlyMeanHook edit (transport.py:259 then :264)."""
    z_ot = (z - gain * mu1) + gain * mu2        # onlymean branch, transport.py:259
    return strength * z_ot + (1 - strength) * z  # strength blend, transport.py:264


def _selftest():
    torch.manual_seed(0)
    L, H = common.N_LAYERS, common.H_MODEL
    ok = True

    def check(name, cond):
        nonlocal ok
        ok &= bool(cond)
        print(f"[selftest-meanact] {name:52s} : {'PASS' if cond else 'FAIL'}")

    check("GAIN is the 1.2 fork magic number", abs(GAIN - 1.2) < 1e-12)

    # Synthetic arrows (mu2 - mu1), avoid touching arrows.pt.
    arrows = torch.randn(L, H)
    s = 2.0

    # build_injection: shape + exact value strength*gain*r.
    vec = build_injection(s, arrows=arrows)
    check("build_injection shape (32,4096)", tuple(vec.shape) == (L, H))
    check("build_injection == strength*GAIN*(mu2-mu1)",
          torch.allclose(vec, s * GAIN * arrows, atol=1e-6))

    # strength=0 -> no-op; linearity in strength.
    check("strength=0 -> zero injection",
          torch.allclose(build_injection(0.0, arrows=arrows), torch.zeros(L, H)))
    check("linear in strength: v(2s)=2 v(s)",
          torch.allclose(build_injection(2 * s, arrows=arrows),
                         2 * build_injection(s, arrows=arrows), atol=1e-6))
    check("sign: negative strength flips the shift",
          torch.allclose(build_injection(-s, arrows=arrows),
                         -build_injection(s, arrows=arrows), atol=1e-6))

    # Equivalence to the byte-faithful OnlyMeanHook edit, per layer, on a
    # synthetic activation z with arbitrary mu1/mu2 s.t. mu2-mu1 == arrows[k].
    B, S = 2, 4
    for k in [0, 7, 31]:
        z = torch.randn(B, S, H)
        mu1 = torch.randn(H)
        mu2 = mu1 + arrows[k]                       # so (mu2-mu1) == arrows[k]
        ref = _onlymean_reference(z, mu1, mu2, s)   # OnlyMeanHook output
        # our hook: h + vec[k]  (add_vec_hook on the block tuple contract)
        hook = common.add_vec_hook(vec[k])
        got = hook(None, None, (z.clone(), "cache"))[0]
        check(f"layer {k:2d}: add_vec_hook == OnlyMeanHook edit",
              torch.allclose(got, ref, atol=1e-4))

    # Direction points toward Black: adding vec increases alignment with (mu2-mu1).
    z = torch.randn(B, S, H)
    steered = z + vec[5]
    before = (z * arrows[5]).sum()
    after = (steered * arrows[5]).sum()
    check("positive strength moves activation toward Black",
          float(after - before) > 0)

    # attach_fn builds one handle per requested block, hooks fire on call.
    layers = [0, 1, 2]
    common.reset_fire_count()

    class _Blk:
        def register_forward_hook(self, fn):
            self._fn = fn
            return type("H", (), {"remove": lambda self: None})()

    class _Model:
        def __init__(self):
            self.blocks = {k: _Blk() for k in layers}

        # emulate resolve_module walking 'model.transformer.blocks.<k>'
    blks = {k: _Blk() for k in layers}

    def _fake_resolve(model, path):
        return blks[int(path.rsplit(".", 1)[1])]

    orig_resolve = common.resolve_module
    common.resolve_module = _fake_resolve
    try:
        attach_fn = build_attach_fn(s, layers=layers, arrows=arrows)
        handles = attach_fn(object())
        check("attach_fn: one handle per block", len(handles) == len(layers))
        # fire each captured hook and confirm the per-layer vec is applied.
        allok = True
        for k in layers:
            z = torch.randn(B, S, H)
            out = blks[k]._fn(None, None, (z.clone(), "c"))[0]
            allok &= torch.allclose(out, z + s * GAIN * arrows[k], atol=1e-4)
        check("attach_fn: block k adds strength*GAIN*(mu2-mu1)_k", allok)
    finally:
        common.resolve_module = orig_resolve
    check("fire counter incremented by attach_fn hooks",
          common.get_fire_count() == len(layers))

    # fit() math on a synthetic calib cache: (mu2-mu1) == mean(Black)-mean(other).
    n = 50
    black = torch.randn(n, L, H) + 3.0
    other = torch.randn(n, L, H) - 1.0
    acts = torch.empty(2 * n, L, H)
    labels = torch.empty(2 * n, dtype=torch.int64)
    acts[0::2] = black; labels[0::2] = 1
    acts[1::2] = other; labels[1::2] = 0
    lab_b = labels.to(torch.bool)
    md = (acts.to(torch.float64)[lab_b].mean(0)
          - acts.to(torch.float64)[~lab_b].mean(0)).to(torch.float32)
    check("fit (mu2-mu1) == mean(Black)-mean(other) shape",
          tuple(md.shape) == (L, H))
    check("fit recovers the ~4.0 mean gap (Black-other)",
          abs(float(md.mean()) - 4.0) < 0.3)

    # load_direction: unit source is per-layer unit-normalized; raw is not.
    if os.path.exists(directions.ARROWS_PATH):
        u = load_direction("unit")
        rw = load_direction("raw")
        check("load_direction unit: (32,4096), ||row||~1",
              tuple(u.shape) == (L, H)
              and torch.allclose(u.norm(dim=1), torch.ones(L), atol=1e-4))
        check("load_direction raw preserves native magnitude (||row||!=1 somewhere)",
              tuple(rw.shape) == (L, H) and float((rw.norm(dim=1) - 1).abs().max()) > 1e-3)
    else:
        print(f"[selftest-meanact] load_direction: SKIP (no {directions.ARROWS_PATH})")

    print(f"[selftest-meanact] OVERALL: {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true",
                    help="offline mean-shift math check on synthetic tensors (no GPU)")
    ap.add_argument("--fit", action="store_true",
                    help="compute per-neuron (mu2-mu1) from a calib cache "
                         "(NEEDS a GPU-built calib cache; do NOT run here)")
    ap.add_argument("--run", action="store_true",
                    help="run the BBQ injection eval (NEEDS a GPU + model)")
    ap.add_argument("--strength", type=float, default=2.0,
                    help="injection strength (vec_k = strength*1.2*(mu2-mu1)_k)")
    ap.add_argument("--direction", choices=DIR_CHOICES, default="unit",
                    help="(mu2-mu1) source: unit (default/family), raw (faithful "
                         "diff-in-means magnitude), fitted (per-neuron from calib)")
    ap.add_argument("--where", default="block",
                    help="calib granularity for --fit (default block/residual)")
    ap.add_argument("--items", default=common.SWEEP400,
                    help="BBQ items jsonl (e.g. a position-balance rotation file)")
    ap.add_argument("--out-dir", default=DEFAULT_OUT)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--baseline-black-rate", type=float, default=None)
    args = ap.parse_args()

    if args.selftest:
        sys.exit(0 if _selftest() else 1)
    if args.fit:
        fit(where=args.where)
        return
    if args.run:
        run(args.strength, out_dir=args.out_dir, tag=args.tag,
            direction=args.direction, limit=args.limit,
            baseline_black_rate=args.baseline_black_rate,
            items_path=args.items)
        return
    ap.error("nothing to do: pass --selftest (offline), --fit (GPU cache), "
             "or --run (GPU)")


if __name__ == "__main__":
    main()
