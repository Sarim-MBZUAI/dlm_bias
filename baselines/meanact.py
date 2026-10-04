#!/usr/bin/env python
"""Mean-AcT baseline (Rodriguez et al., 2025, "Controlling Language and
Diffusion Models by Transporting Activations"): the mean-only
Optimal-Transport steering map, applied to the residual stream at all 32
LLaDA blocks.

Mean-AcT is GaussianOT with onlymean=True (act/hooks/transport.py
OnlyMeanHook).  The forward edit is
    onlymean branch : z_ot = (z - 1.2*mu1) + 1.2*mu2
    strength blend  : z_ot = strength*z_ot + (1-strength)*z
which collapses to a constant per-neuron additive shift
    z <- z + strength * 1.2 * (mu2 - mu1).
The 1.2 is a hardcoded gain in the reference onlymean branch (not a derived
OT quantity); it is kept as GAIN.

The target answer is the OT destination (mu2) and the non-target option the
source (mu1), so (mu2 - mu1) is the target-minus-other diff-in-means stored in
steering/arrows.pt.  The edit h <- h + vec_k is applied at every block, all
positions, every denoising step.

Direction source (--direction):
  unit    (default) directions.load_arrows(): per-layer unit-normalized, the
          convention shared by all baselines so one strength is comparable;
          drops the native per-layer magnitude of (mu2 - mu1).
  raw     load_arrows(raw=True): the literal diff-in-means (mu2 - mu1).
  fitted  fit() recomputes per-neuron (mu2 - mu1) at block granularity from a
          calib.py activation cache (analogue of OnlyMeanHook.fit).

Usage:
    python baselines/meanact.py --selftest               # offline, no GPU
    python baselines/meanact.py --fit                    # needs a calib cache
    python baselines/meanact.py --run --strength 2.0     # GPU
"""
import argparse
import os
import sys

import torch

# Harness import (same convention as steering/pid_steer.py).
ROOT = os.environ.get("DLM_BIAS_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "eval"))
sys.path.insert(0, os.path.join(ROOT, "steering"))

import common      # noqa: E402  shared driver (hooks, attach, run_baseline)
import directions  # noqa: E402  load_arrows (per-layer diff-in-means)

# --------------------------------------------------------------------------- #
# The onlymean branch multiplies the class means by a hardcoded 1.2 gain       #
# (act/hooks/transport.py).  It is not a derived OT quantity; it is kept so    #
# the mean-shift magnitude matches the reference; `strength` scales on top.    #
# --------------------------------------------------------------------------- #
GAIN = 1.2

DEFAULT_OUT = os.path.join(ROOT, "results", "meanact")
DIR_CHOICES = ("unit", "raw", "fitted")
FIT_PATH = os.path.join(common.CACHE_DIR, "meanact_meandiff_block.pt")


def fit_path_for(target):
    """cache/meanact_meandiff_block.pt for black;
    cache/meanact_meandiff_block_<target>.pt otherwise."""
    return os.path.join(common.CACHE_DIR,
                        f"meanact_meandiff_block{common.target_suffix(target)}.pt")


# --------------------------------------------------------------------------- #
# Direction source: the per-layer (mu2 - mu1) = target-minus-other mean shift. #
# --------------------------------------------------------------------------- #
def load_direction(direction="unit", fit_path=None, target="black"):
    """Return the (32, 4096) per-layer mean-shift (mu2 - mu1), target minus
    other (Black minus other for the default target).

    direction:
      "unit"   -> directions.load_arrows(target=...)  (per-layer unit-normalized;
                  family default, the spec's directions.load_arrows() r[k]).
      "raw"    -> directions.load_arrows(raw=True, target=...)  (raw
                  diff-in-means; the literal (mu2 - mu1) the true Mean-AcT uses).
      "fitted" -> the fit()-produced per-neuron (mu2 - mu1) at block granularity
                  (needs the target's calib cache; see fit()).
    """
    assert direction in DIR_CHOICES, f"direction must be one of {DIR_CHOICES}"
    fit_path = fit_path or fit_path_for(target)
    if direction == "fitted":
        assert os.path.exists(fit_path), (
            f"direction='fitted' needs {fit_path}; run --fit first "
            f"(which needs a calib.py activation cache built on GPU).")
        blob = torch.load(fit_path, map_location="cpu")
        return blob["mean_diff"].to(torch.float32)          # (32, 4096)
    return directions.load_arrows(raw=(direction == "raw"),
                                  target=target).to(torch.float32)


# --------------------------------------------------------------------------- #
# The injection: vec_k = strength * GAIN * (mu2 - mu1)_k   (all 32 blocks).    #
# --------------------------------------------------------------------------- #
def build_injection(strength, direction="unit", gain=GAIN, arrows=None,
                    fit_path=None, target="black"):
    """Return the (32, 4096) additive mean-shift injection, one row per block.

        vec_k = strength * gain * (mu2 - mu1)_k

    This is the collapsed OnlyMeanHook edit (transport.py): adding vec_k
    to every activation is identically  strength*z_ot + (1-strength)*z  with
    z_ot = z + gain*(mu2-mu1).  Positive strength injects TOWARD Black.

    arrows: optional (32, 4096) override for (mu2 - mu1) (used by --selftest to
    avoid touching arrows.pt); otherwise load_direction(direction) supplies it.
    """
    r = load_direction(direction, fit_path=fit_path, target=target) if arrows is None \
        else torch.as_tensor(arrows, dtype=torch.float32)
    assert r.dim() == 2 and r.shape[1] == common.H_MODEL, \
        f"expected (n_layers, {common.H_MODEL}) arrows, got {tuple(r.shape)}"
    return float(strength) * float(gain) * r


# --------------------------------------------------------------------------- #
# attach_fn: add vec_k at every block k via the shared fire-counted factory.   #
# --------------------------------------------------------------------------- #
def build_attach_fn(strength, layers=None, direction="unit", gain=GAIN,
                    arrows=None, fit_path=None, target="black"):
    """Return attach_fn(model) -> handles that adds vec_k at each block k.

    layers=None -> all 32 blocks (native Mean-AcT: every layer).  Hooks are built
    with common.add_vec_hook so the shared fire counter increments and
    run_baseline can assert the intervention actually fired.
    """
    layers = list(range(common.N_LAYERS)) if layers is None else list(layers)
    vecs = build_injection(strength, direction=direction, gain=gain,
                           arrows=arrows, fit_path=fit_path, target=target)

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
        items_path=None, target="black"):
    """Run one Mean-AcT injection condition end-to-end via common.run_baseline.

    Writes out_dir/cond_<tag>.json (+ _samples.jsonl) in the pid_steer format.
    target selects the (mu2 - mu1) source arrows/fit AND the eval
    classification.  Needs a GPU.  See
    --selftest for the offline math check.
    """
    tag = tag or f"meanact_{direction}_s{strength:g}"
    attach_fn = build_attach_fn(strength, layers=layers, direction=direction,
                                gain=gain, target=target)
    return common.run_baseline(
        attach_fn, items_path=items_path, out_dir=out_dir, tag=tag, limit=limit,
        baseline_black_rate=baseline_black_rate, model=model, tok=tok,
        target=target,
        config_extra={
            "method": "mean_act",
            "reference": "act/hooks/transport.py OnlyMeanHook (transport.py:259)",
            "granularity": "block/residual (native, all 32 layers)",
            "strength": float(strength),
            "gain": float(gain),
            "direction_source": direction,
            "target": target,
            "layers": "all32" if layers is None else list(layers),
            "position": "all (bidirectional)",
        },
    )


# --------------------------------------------------------------------------- #
# fit(): per-neuron (mu2 - mu1) at block granularity from a calib cache.       #
# Analogue of OnlyMeanHook.fit: mu = per-class mean; (mu2 - mu1) =            #
# mean(Black) - mean(other).  Pure (no GPU) once the calib activation cache    #
# exists; building that cache (calib.py --fit) needs a GPU.                    #
# --------------------------------------------------------------------------- #
def fit(where="block", calib_path=None, save=True, out_path=None, target="black"):
    """Compute per-neuron (mu2 - mu1) = mean(target) - mean(other) per layer.

    Reads the target's calib.py activation cache (acts (2n,32,feat), labels
    1=target/0=other) and averages within class.  This is the block-granularity
    twin of the target's arrows diff-in-means and the direct analogue of
    OnlyMeanHook.fit's mu1/mu2.  target="black" (default) reads
    cache/calib_block.pt.
    """
    import calib  # local import: only needed for --fit
    out_path = out_path or fit_path_for(target)
    blob = calib.load_calib(where, path=calib_path, target=target)
    acts = blob["acts"].to(torch.float64)          # (2n, 32, feat)
    labels = blob["labels"].to(torch.bool)         # (2n,) 1=target, 0=other
    mu_target = acts[labels].mean(dim=0)           # (32, feat)  == mu2
    mu_other = acts[~labels].mean(dim=0)           # (32, feat)  == mu1
    mean_diff = (mu_target - mu_other).to(torch.float32)  # (mu2 - mu1) toward target
    out = {
        "mean_diff": mean_diff, "where": where,
        "feat": mean_diff.shape[-1], "n_layers": mean_diff.shape[0],
        "n_items": int(labels.sum()),
        "note": "mu2-mu1 = mean(target) - mean(other); OnlyMeanHook.fit analogue",
        "source": blob.get("source"),
        "target": target,
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
    """Reference OnlyMeanHook edit (onlymean branch, then strength blend)."""
    z_ot = (z - gain * mu1) + gain * mu2        # onlymean branch
    return strength * z_ot + (1 - strength) * z  # strength blend


def _selftest():
    torch.manual_seed(0)
    L, H = common.N_LAYERS, common.H_MODEL
    ok = True

    def check(name, cond):
        nonlocal ok
        ok &= bool(cond)
        print(f"[selftest-meanact] {name:52s} : {'PASS' if cond else 'FAIL'}")

    check("GAIN is the reference-implementation onlymean gain 1.2", abs(GAIN - 1.2) < 1e-12)

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

    # Equivalence to the reference OnlyMeanHook edit, per layer, on a
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

    # target-parameterized fit-artifact paths.
    check("fit_path_for('black') == FIT_PATH (regression)",
          fit_path_for("black") == FIT_PATH)
    check("fit_path_for gender -> cache/meanact_meandiff_block_<t>.pt",
          fit_path_for("woman").endswith("cache/meanact_meandiff_block_woman.pt")
          and fit_path_for("man").endswith("cache/meanact_meandiff_block_man.pt"))

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
                         "(needs a GPU-built calib cache)")
    ap.add_argument("--run", action="store_true",
                    help="run the BBQ injection eval (NEEDS a GPU + model)")
    ap.add_argument("--strength", type=float, default=2.0,
                    help="injection strength (vec_k = strength*1.2*(mu2-mu1)_k)")
    ap.add_argument("--direction", choices=DIR_CHOICES, default="unit",
                    help="(mu2-mu1) source: unit (default/family), raw (faithful "
                         "diff-in-means magnitude), fitted (per-neuron from calib)")
    ap.add_argument("--where", default="block",
                    help="calib granularity for --fit (default block/residual)")
    ap.add_argument("--target", choices=common.SUPPORTED_TARGETS, default="black",
                    help="steering target (black = default; woman/man = "
                         "gender arrows/calib + classification)")
    ap.add_argument("--items", default=None,
                    help="BBQ items jsonl (e.g. a position-balance rotation "
                         "file); default = the target's own _sweep400 file")
    ap.add_argument("--out-dir", default=DEFAULT_OUT)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--baseline-black-rate", type=float, default=None)
    args = ap.parse_args()

    if args.selftest:
        sys.exit(0 if _selftest() else 1)
    if args.fit:
        fit(where=args.where, target=args.target)
        return
    if args.run:
        run(args.strength, out_dir=args.out_dir, tag=args.tag,
            direction=args.direction, limit=args.limit,
            baseline_black_rate=args.baseline_black_rate,
            items_path=args.items, target=args.target)
        return
    ap.error("nothing to do: pass --selftest (offline), --fit (GPU cache), "
             "or --run (GPU)")


if __name__ == "__main__":
    main()
