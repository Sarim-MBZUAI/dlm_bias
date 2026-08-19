#!/usr/bin/env python
"""llada_moe/baselines/meanact.py -- Mean-AcT (Apple "Activation Transport") on
LLaDA-MoE-7B-A1B-Instruct.  Port of dream/baselines/meanact.py at NATIVE
granularity (the full residual stream, per-neuron, at every one of the 16
transformer blocks).

Mean-AcT = GaussianOT with onlymean=True (OnlyMeanHook, transport.py:500-524):

    onlymean branch : z_ot = (z - GAIN*mu1) + GAIN*mu2         (transport.py:259)
    strength blend  : z_ot = strength*z_ot + (1-strength)*z    (transport.py:264)
  collapse to a pure additive mean-shift:
    z  <-  z + strength * GAIN * (mu2 - mu1)

Black is the OT DESTINATION (mu2), the non-Black option the source (mu1), so
(mu2 - mu1) is exactly the Black-minus-other diff-in-means llada_moe/arrows.pt
stores.  The residual edit at block k is h <- h + vec_k, applied to ALL positions
every diffusion step.  GAIN=1.2 is the Apple fork's hardcoded onlymean gain,
kept verbatim.

DIRECTION SOURCE (faithfulness knob):
  unit   -> directions.load_arrows()        (per-layer unit-normalized; family default)
  raw    -> directions.load_arrows(raw=True) (raw diff-in-means; the literal mu2-mu1)
  fitted -> fit(): per-neuron (mu2-mu1) from a calib block cache (OnlyMeanHook.fit twin)

CLI:
    python llada_moe/baselines/meanact.py --selftest                 # offline
    python llada_moe/baselines/meanact.py --fit                      # needs calib cache
    python llada_moe/baselines/meanact.py --run --strength 2.0       # NEEDS GPU
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
sys.path.insert(0, _HERE)

import common_lladamoe as C  # noqa: E402
import directions        # noqa: E402
import calib             # noqa: E402  (CACHE_DIR + fit reads a calib cache)

GAIN = 1.2               # Apple fork onlymean magic gain (transport.py:259)
DEFAULT_OUT = os.path.join(ROOT, "results", "lladamoe", "meanact")
DIR_CHOICES = ("unit", "raw", "fitted")
FIT_PATH = os.path.join(calib.CACHE_DIR, "meanact_meandiff_block.pt")


def load_direction(direction="unit", fit_path=FIT_PATH):
    """Return the (16,2048) per-layer mean-shift (mu2 - mu1), Black minus other."""
    assert direction in DIR_CHOICES, f"direction must be one of {DIR_CHOICES}"
    if direction == "fitted":
        assert os.path.exists(fit_path), (
            f"direction='fitted' needs {fit_path}; run --fit first "
            f"(which needs a calib block cache built on GPU).")
        return torch.load(fit_path, map_location="cpu")["mean_diff"].to(torch.float32)
    return directions.load_arrows(raw=(direction == "raw")).to(torch.float32)


def build_injection(strength, direction="unit", gain=GAIN, arrows=None,
                    fit_path=FIT_PATH):
    """Return the (16,2048) additive mean-shift  vec_k = strength*gain*(mu2-mu1)_k."""
    r = load_direction(direction, fit_path=fit_path) if arrows is None \
        else torch.as_tensor(arrows, dtype=torch.float32)
    assert r.dim() == 2 and r.shape[1] == C.D_MODEL, \
        f"expected (n_layers, {C.D_MODEL}) arrows, got {tuple(r.shape)}"
    return float(strength) * float(gain) * r


def build_attach_fn(strength, layers=None, direction="unit", gain=GAIN,
                    arrows=None, fit_path=FIT_PATH):
    """attach_fn(model)->handles adding vec_k at each block k (default all 16)."""
    layers = list(range(C.N_LAYERS)) if layers is None else list(layers)
    vecs = build_injection(strength, direction=direction, gain=gain,
                           arrows=arrows, fit_path=fit_path)

    def attach_fn(model):
        handles = []
        for k in layers:
            handles += C.attach(model, [C.block_path(int(k))], C.add_vec_hook(vecs[k]))
        return handles

    return attach_fn


def run(strength, out_dir=DEFAULT_OUT, tag=None, layers=None, direction="unit",
        gain=GAIN, items_path=None, limit=0, baseline_black_rate=None,
        model=None, tok=None):
    """Run one Mean-AcT injection condition end-to-end.  NEEDS A GPU."""
    tag = tag or f"meanact_{direction}_s{strength:g}"
    return C.run_items(
        attach_fn=build_attach_fn(strength, layers=layers, direction=direction, gain=gain),
        items_path=items_path or C.SWEEP400,
        out_dir=out_dir, tag=tag, limit=limit,
        baseline_black_rate=baseline_black_rate, model=model, tok=tok,
        config_extra={
            "method": "mean_act",
            "reference": "act/hooks/transport.py OnlyMeanHook (transport.py:259)",
            "granularity": "block/residual (native, all 16 layers)",
            "strength": float(strength), "gain": float(gain),
            "direction_source": direction,
            "layers": "all16" if layers is None else list(layers),
            "position": "all (bidirectional)"})


def fit(where="block", calib_path=None, save=True, out_path=FIT_PATH):
    """Per-neuron (mu2 - mu1) = mean(Black) - mean(other) per layer from a calib cache."""
    blob = calib.load_calib(where, path=calib_path)
    acts = blob["acts"].to(torch.float64)
    labels = blob["labels"].to(torch.bool)
    mean_diff = (acts[labels].mean(0) - acts[~labels].mean(0)).to(torch.float32)
    out = {"mean_diff": mean_diff, "where": where, "feat": mean_diff.shape[-1],
           "n_layers": mean_diff.shape[0], "n_items": int(labels.sum()),
           "note": "mu2-mu1 = mean(Black)-mean(other); OnlyMeanHook.fit analogue",
           "source": blob.get("source")}
    if save:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        torch.save(out, out_path)
        print(f"[meanact:fit] SAVED (mu2-mu1) {tuple(mean_diff.shape)} -> {out_path}",
              flush=True)
    return out


def _onlymean_reference(z, mu1, mu2, strength, gain=GAIN):
    z_ot = (z - gain * mu1) + gain * mu2
    return strength * z_ot + (1 - strength) * z


def _selftest():
    torch.manual_seed(0)
    L, H = C.N_LAYERS, C.D_MODEL
    ok = True

    def check(name, cond):
        nonlocal ok
        ok &= bool(cond)
        print(f"[selftest-meanact] {name:52s} : {'PASS' if cond else 'FAIL'}")

    check("GAIN is the 1.2 fork magic number", abs(GAIN - 1.2) < 1e-12)

    arrows = torch.randn(L, H); s = 2.0
    vec = build_injection(s, arrows=arrows)
    check("build_injection shape (16,2048)", tuple(vec.shape) == (L, H))
    check("build_injection == strength*GAIN*(mu2-mu1)",
          torch.allclose(vec, s * GAIN * arrows, atol=1e-6))
    check("strength=0 -> zero injection",
          torch.allclose(build_injection(0.0, arrows=arrows), torch.zeros(L, H)))
    check("linear in strength", torch.allclose(build_injection(2 * s, arrows=arrows),
                                                2 * build_injection(s, arrows=arrows), atol=1e-6))
    check("negative strength flips", torch.allclose(build_injection(-s, arrows=arrows),
                                                    -build_injection(s, arrows=arrows), atol=1e-6))

    # Equivalence to the byte-faithful OnlyMeanHook edit, per layer.
    B, S = 2, 4
    for k in [0, 7, 15]:
        z = torch.randn(B, S, H)
        mu1 = torch.randn(H); mu2 = mu1 + arrows[k]
        ref = _onlymean_reference(z, mu1, mu2, s)
        got = C.add_vec_hook(vec[k])(None, None, (z.clone(), "cache"))[0]
        check(f"layer {k:2d}: add_vec_hook == OnlyMeanHook edit",
              torch.allclose(got, ref, atol=1e-4))

    # attach_fn: one handle per block, per-layer vec applied, fire counter bumps.
    layers = [0, 1, 2]
    C.reset_fire_count()
    captured = {}

    class _Blk:
        def __init__(self, k):
            self.k = k
        def register_forward_hook(self, fn):
            captured[self.k] = fn
            return type("H", (), {"remove": lambda self: None})()

    blkmap = {k: _Blk(k) for k in layers}

    def _fake_resolve(model, path):
        return blkmap[int(path.rsplit(".", 1)[1])]

    orig = C.resolve_module
    C.resolve_module = _fake_resolve
    try:
        handles = build_attach_fn(s, layers=layers, arrows=arrows)(object())
        check("attach_fn: one handle per block", len(handles) == len(layers))
        allok = True
        for k in layers:
            z = torch.randn(B, S, H)
            out = captured[k](None, None, (z.clone(), "c"))[0]
            allok &= torch.allclose(out, z + s * GAIN * arrows[k], atol=1e-4)
        check("attach_fn: block k adds strength*GAIN*(mu2-mu1)_k", allok)
    finally:
        C.resolve_module = orig
    check("fire counter incremented by attach_fn hooks", C.get_fire_count() == len(layers))

    # fit() math on a synthetic calib cache.
    n = 50
    black = torch.randn(n, L, H) + 3.0; other = torch.randn(n, L, H) - 1.0
    acts = torch.empty(2 * n, L, H); labels = torch.empty(2 * n, dtype=torch.int64)
    acts[0::2] = black; labels[0::2] = 1
    acts[1::2] = other; labels[1::2] = 0
    lab_b = labels.to(torch.bool)
    md = (acts.to(torch.float64)[lab_b].mean(0)
          - acts.to(torch.float64)[~lab_b].mean(0)).to(torch.float32)
    check("fit recovers ~4.0 mean gap (Black-other)", abs(float(md.mean()) - 4.0) < 0.3)

    if os.path.exists(directions.ARROWS_PATH):
        u = load_direction("unit"); rw = load_direction("raw")
        check("load_direction unit: (16,2048), ||row||~1",
              tuple(u.shape) == (L, H)
              and torch.allclose(u.norm(dim=1), torch.ones(L), atol=1e-4))
        check("load_direction raw preserves native magnitude",
              tuple(rw.shape) == (L, H)
              and float((rw.norm(dim=1) - 1).abs().max()) > 1e-3)
    else:
        print(f"[selftest-meanact] load_direction: SKIP (no {directions.ARROWS_PATH})")

    print(f"[selftest-meanact] OVERALL: {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--fit", action="store_true", help="per-neuron (mu2-mu1) from calib cache")
    ap.add_argument("--run", action="store_true", help="run BBQ injection eval (NEEDS GPU)")
    ap.add_argument("--strength", type=float, default=2.0)
    ap.add_argument("--direction", choices=DIR_CHOICES, default="unit")
    ap.add_argument("--where", default="block")
    ap.add_argument("--items", default=None,
                    help="BBQ items jsonl (default: sweep400; e.g. a position-balance rotation file)")
    ap.add_argument("--out_dir", default=DEFAULT_OUT)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--baseline_black_rate", type=float, default=None)
    args = ap.parse_args()

    if args.selftest:
        sys.exit(0 if _selftest() else 1)
    if args.fit:
        fit(where=args.where); return
    if args.run:
        run(args.strength, out_dir=args.out_dir, tag=args.tag,
            direction=args.direction, items_path=args.items, limit=args.limit,
            baseline_black_rate=args.baseline_black_rate)
        return
    ap.error("nothing to do: pass --selftest, --fit (GPU cache), or --run (GPU)")


if __name__ == "__main__":
    main()
