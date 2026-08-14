#!/usr/bin/env python
"""baselines/linearact.py -- Linear-AcT (Rodriguez et al., "Activation Transport")
per-neuron 1-D Optimal-Transport steering, ported to our LLaDA-8B-Instruct BBQ
bias-INJECTION harness at NATIVE (per-neuron) granularity, hooked at the
MLP-HIDDEN units (the 12288-d gated activation, INPUT to each block's ff_out).

WHAT THIS IS (faithfulness map to the Apple AcT reference tree under
.../pid-steering/Mean-AcT/act):

  Two variants of a per-neuron 1-D OT map fitted on labelled (Black vs other)
  MLP-hidden activations, with "Black" as the OT DESTINATION so the map pushes
  toward the Black answer (positive strength => stronger injection):

    (i)  gaussian  -- assumes each neuron is Gaussian; transports N(mu_src,sig_src)
         -> N(mu_dst,sig_dst) with the closed-form affine map
             z_ot = (sig_dst/sig_src) * (z - mu_src) + mu_dst
         This is GaussianOTHook.forward (transport.py:261) and is produced here
         via directions.gaussian_ot(mu1=src, sig1=src, mu2=dst, sig2=dst).

    (ii) empirical -- the LEARNABLE Linear-AcT map: per neuron, sort the source
         and destination samples (1-D OT pairing, ot_maps.py:solve_ot_1d:9-21)
         and fit a closed-form least-squares affine beta*z+bias
         (LinearProj.optimize, archs.py:30-54; driven by LearnableOTHook.fit,
         transport.py:693-721).  Produced here via directions.empirical_ot_fit.

  Both variants are affine per neuron:  z -> beta*z + bias.  The AcT strength
  knob interpolates/extrapolates against the identity (transport.py:264 and 756):
             z_final = strength * (beta*z + bias) + (1 - strength) * z
  which is ITSELF affine with
             beta_eff = strength*beta + (1 - strength),  bias_eff = strength*bias.
  strength=1 => full transport; strength=0 => identity; strength>1 EXTRAPOLATES
  beyond the destination (stronger injection).  We fold strength into
  (beta_eff, bias_eff) and apply once per denoising step at every position.

  Quantile pool-mask: AcT's default quantiles_src="q_all" == [-1e6, 1e6]
  (transport.py:220-221, 731-732) makes pool_mask all-ones (no filtering); we
  follow that default, so every unit is transported.

GRANULARITY / HOOK SITE (max faithfulness, user's choice = MLP-hidden):
  In LLaDA's fused MLP  x = act(ff_proj(x)) * up_proj(x); x = ff_out(x)
  (modeling_llada.py:924-930) there is NO module whose OUTPUT is the 12288-d
  gated activation -- it is exactly the INPUT to ff_out.  AcT hooks a module
  whose OUTPUT is the target activation; the faithful equivalent here is a
  forward_PRE hook on ff_out (edits input[0]).  So the affine map is applied via
  common.affine_pre_hook on blocks[k].ff_out (NOT affine_hook, which would edit
  ff_out's 4096-d output and mismatch the 12288-d per-neuron stats).  This is the
  exact activation calib.collect_activations('mlp_hidden') captures (calib.py:187),
  so fit and apply use the same units.  Default: ALL 32 layers.

INTERVENTION POSITION: ALL tokens, bidirectional (LLaDA masked-diffusion; the
hook fires once per denoising step over every position -- there is no "last").

CLI:
  python -m baselines.linearact --selftest                      # offline, no GPU
  python -m baselines.linearact --fit                           # NEEDS GPU (do NOT run here)
  python -m baselines.linearact --run --variant gaussian  --strength 2.0   # NEEDS GPU
  python -m baselines.linearact --run --variant empirical --strength 2.0   # NEEDS GPU
"""
import argparse
import os
import sys

import torch

# Harness import (ROOT = MAIN tree; mirrors steering/pid_steer.py:49-51).
ROOT = os.environ.get("DLM_BIAS_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "eval"))
sys.path.insert(0, os.path.join(ROOT, "steering"))

import common  # noqa: E402
import calib  # noqa: E402
import directions  # noqa: E402
from common import (  # noqa: E402
    CACHE_DIR,
    N_LAYERS,
    H_MLP,
    BLOCKS_PATH,
    submodule_paths,
    attach,
    affine_pre_hook,
    run_baseline,
)

WHERE = "mlp_hidden"          # user chose max-faithfulness MLP-hidden units
VARIANTS = ("gaussian", "empirical")
STATS_PATH = os.path.join(CACHE_DIR, "linearact_stats.pt")
RESULTS_DIR = os.path.join(ROOT, "results", "linearact")
EPS = 1e-4                    # matches directions.gaussian_ot / transport.std_eps


def stats_path_for(target):
    """cache/linearact_stats.pt for black (round-1 path, unchanged);
    cache/linearact_stats_<target>.pt otherwise."""
    return os.path.join(CACHE_DIR,
                        f"linearact_stats{common.target_suffix(target)}.pt")


# --------------------------------------------------------------------------- #
# PURE fit math (no GPU): labelled per-layer acts -> per-neuron OT parameters. #
# Factored out of fit() so --selftest can exercise it on synthetic tensors.    #
# --------------------------------------------------------------------------- #
def fit_stats_from_blob(blob):
    """Fit BOTH OT variants per layer/neuron from a calib blob.

    blob: {'acts': (2n, L, feat) float, 'labels': (2n,) with 1=Black, 0=other, ...}
    Black is the DESTINATION (label==1), other is the SOURCE (label==0), so both
    maps push toward Black (matches the injection convention; directions.py:14).

    Returns a stats dict (see fit()).  Gaussian stats are the per-neuron
    (mean,std) of each class (transport.GaussianOTHook.fit:189-192 form, but with
    src=other/dst=Black to inject).  Empirical stats are the per-neuron
    (beta,bias) from directions.empirical_ot_fit (sorted 1-D OT + closed-form LS).
    """
    acts = torch.as_tensor(blob["acts"], dtype=torch.float32)
    labels = torch.as_tensor(blob["labels"]).to(torch.bool)
    L, feat = acts.shape[1], acts.shape[2]
    src_sel = ~labels          # other  (label 0) = OT source
    dst_sel = labels           # Black  (label 1) = OT destination

    mu_src = torch.empty(L, feat); sig_src = torch.empty(L, feat)
    mu_dst = torch.empty(L, feat); sig_dst = torch.empty(L, feat)
    e_beta = torch.empty(L, feat); e_bias = torch.empty(L, feat)
    for k in range(L):
        a = acts[:, k, :]
        xs = a[src_sel]        # (n_src, feat)
        xd = a[dst_sel]        # (n_dst, feat)
        # (i) gaussian per-neuron stats  (unbiased std, like torch.std default).
        mu_src[k] = xs.mean(0); sig_src[k] = xs.std(0)
        mu_dst[k] = xd.mean(0); sig_dst[k] = xd.std(0)
        # (ii) empirical per-neuron affine (needs equal counts; calib gives 1:1).
        e_beta[k], e_bias[k] = directions.empirical_ot_fit(xs, xd)

    return {
        "where": blob.get("where", WHERE),
        "feat": feat,
        "n_layers": L,
        "n_items": blob.get("n_items"),
        "source": blob.get("source"),
        "gaussian": {"mu_src": mu_src, "sig_src": sig_src,
                     "mu_dst": mu_dst, "sig_dst": sig_dst},
        "empirical": {"beta": e_beta, "bias": e_bias},
    }


# --------------------------------------------------------------------------- #
# FIT (NEEDS GPU): collect mlp_hidden activations, fit both variants, cache.   #
# --------------------------------------------------------------------------- #
def fit(cap=calib.CAP, model=None, tok=None, out_path=None, target="black"):
    """Collect target-vs-other MLP-hidden activations and fit both OT variants.

    Calls calib.collect_activations('mlp_hidden', target=...) (target=label 1 =>
    OT DESTINATION; black default = the round-1 Black heldout, gender = the E3
    manifest heldout), fits gaussian + empirical per-neuron maps for all 32
    layers, and caches cache/linearact_stats[_<target>].pt.  NEEDS A GPU + the
    model -- do NOT run here."""
    out_path = out_path or stats_path_for(target)
    blob = calib.collect_activations(WHERE, model=model, tok=tok, cap=cap,
                                     save=True, target=target)
    stats = fit_stats_from_blob(blob)
    stats["target"] = target
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save(stats, out_path)
    print(f"[linearact.fit] gaussian+empirical stats "
          f"({stats['n_layers']}x{stats['feat']}) -> {out_path}", flush=True)
    return stats


def load_stats(path=None, target="black"):
    return torch.load(path or stats_path_for(target), map_location="cpu")


# --------------------------------------------------------------------------- #
# Build the per-layer effective affine (beta_eff, bias_eff) for a variant.     #
# gaussian: beta = sig_dst/sig_src, bias = mu_dst - beta*mu_src  (== the map    #
#           directions.gaussian_ot returns, transport.py:261).                 #
# empirical: (beta,bias) straight from the fitted LS affine.                    #
# strength folds in via  beta_eff = s*beta + (1-s),  bias_eff = s*bias          #
# (transport.py:264/756).                                                       #
# --------------------------------------------------------------------------- #
def build_injection(variant, strength=1.0, stats=None, layers=None):
    """Return (beta_eff, bias_eff, layers): per-selected-layer affine params.

    variant in {'gaussian','empirical'}.  strength=1 => full OT transport toward
    Black; strength>1 EXTRAPOLATES (stronger injection).  Shapes (n_sel, feat)."""
    assert variant in VARIANTS, f"variant must be one of {VARIANTS}"
    if stats is None:
        stats = load_stats()
    feat = stats["feat"]
    L = stats["n_layers"]
    layers = list(range(L)) if layers is None else list(layers)

    if variant == "gaussian":
        g = stats["gaussian"]
        # FAITHFUL to AcT (transport.py:203-210,270): low-variance neurons in
        # EITHER class are left at IDENTITY, not transported. Clamping sig_src
        # instead (old bug) let near-dead source neurons get beta=sig_dst/1e-4
        # ~1e4 and explosively over-inject. Mask them to beta=1, bias=0.
        ss, sd = g["sig_src"], g["sig_dst"]
        valid = (ss > EPS) & (sd > EPS)
        ratio = torch.where(valid, sd / ss.clamp(min=EPS), torch.ones_like(ss))
        beta = ratio
        bias = torch.where(valid, g["mu_dst"] - ratio * g["mu_src"], torch.zeros_like(ss))
    else:
        e = stats["empirical"]
        beta = e["beta"].clone()
        bias = e["bias"].clone()

    beta = beta[layers].to(torch.float32)
    bias = bias[layers].to(torch.float32)
    # AcT strength interp/extrap against identity (affine composition).
    beta_eff = strength * beta + (1.0 - strength)
    bias_eff = strength * bias
    assert beta_eff.shape == (len(layers), feat), beta_eff.shape
    return beta_eff, bias_eff, layers


# --------------------------------------------------------------------------- #
# Attach: one per-layer affine_pre_hook on each blocks[k].ff_out.              #
# (pre-hook => edits ff_out's INPUT == the 12288-d MLP-hidden activation.)     #
# --------------------------------------------------------------------------- #
def attach_fn(model, variant="empirical", strength=1.0, stats=None, layers=None):
    """Register the Linear-AcT affine on the MLP-hidden units of each layer.

    Returns the list of forward-pre-hook handles (run_baseline detaches them).
    Each layer gets its OWN (beta,bias) row, so we register per layer (the hooks
    are built with common.affine_pre_hook => they bump the shared fire counter)."""
    beta_eff, bias_eff, layers = build_injection(variant, strength, stats, layers)
    paths = submodule_paths("ff_out", layers)     # blocks[k].ff_out, in `layers` order
    handles = []
    for i, p in enumerate(paths):
        hk = affine_pre_hook(beta_eff[i], bias_eff[i])
        handles += attach(model, [p], hk, pre=True)
    return handles


# --------------------------------------------------------------------------- #
# RUN one BBQ condition (NEEDS GPU).                                           #
# --------------------------------------------------------------------------- #
def run(variant="empirical", strength=1.0, out_dir=RESULTS_DIR, tag=None,
        layers=None, limit=0, model=None, tok=None, baseline_black_rate=None,
        items_path=None, target="black"):
    """Run the full sweep400 BBQ injection for one variant/strength.

    Writes out_dir/cond_<tag>.json (+ _samples.jsonl); default tag encodes the
    strength (<variant>_s<strength>) so sweeps at different strengths don't
    overwrite each other.  target selects the fitted OT stats
    (cache/linearact_stats[_<target>].pt) AND the eval classification (black
    default = round-1 behavior).  NEEDS A GPU."""
    assert variant in VARIANTS, f"variant must be one of {VARIANTS}"
    tag = tag or f"{variant}_s{strength:g}"
    stats = load_stats(target=target)

    def _attach(m):
        return attach_fn(m, variant=variant, strength=strength,
                         stats=stats, layers=layers)

    return run_baseline(
        _attach, items_path=items_path, out_dir=out_dir, tag=tag, limit=limit,
        model=model, tok=tok, baseline_black_rate=baseline_black_rate,
        target=target,
        config_extra={"method": "linear_act", "variant": variant,
                      "strength": strength, "where": WHERE, "target": target,
                      "layers": "all" if layers is None else list(layers),
                      "intervention_position": "all_bidirectional",
                      "stats_path": stats_path_for(target)},
    )


# --------------------------------------------------------------------------- #
# Offline self-test: OT maps recover analytic shift/scale + affine hook math.  #
# No GPU / no model / no fits.                                                 #
# --------------------------------------------------------------------------- #
def _selftest():
    torch.manual_seed(0)
    ok = True

    def check(name, cond):
        nonlocal ok
        ok &= bool(cond)
        print(f"[selftest-linearact] {name:52s} : {'PASS' if cond else 'FAIL'}")

    L, feat, n = 3, 5, 4000
    # Ground-truth per-neuron classes: other=N(mu_o,s_o), Black=N(mu_b,s_b).
    mu_o = torch.randn(feat); s_o = torch.rand(feat) + 0.5
    mu_b = torch.randn(feat); s_b = torch.rand(feat) + 0.5
    acts, labels = [], []
    for _ in range(n):
        acts.append(mu_b + s_b * torch.randn(L, feat)); labels.append(1)   # Black
        acts.append(mu_o + s_o * torch.randn(L, feat)); labels.append(0)   # other
    blob = {"acts": torch.stack(acts), "labels": torch.tensor(labels),
            "where": WHERE, "feat": feat, "n_layers": L, "n_items": n}

    stats = fit_stats_from_blob(blob)
    g = stats["gaussian"]
    check("gaussian mu_dst ~ Black mean", torch.allclose(g["mu_dst"][0], mu_b, atol=0.1))
    check("gaussian mu_src ~ other mean", torch.allclose(g["mu_src"][0], mu_o, atol=0.1))
    check("gaussian sig_dst ~ Black std",  torch.allclose(g["sig_dst"][0], s_b, atol=0.1))
    check("gaussian sig_src ~ other std",  torch.allclose(g["sig_src"][0], s_o, atol=0.1))

    # (i) gaussian affine == directions.gaussian_ot map, and transports other->Black.
    bg, big, _ = build_injection("gaussian", strength=1.0, stats=stats)
    x = mu_o + s_o * torch.randn(500, feat)                      # source-distributed
    ref = directions.gaussian_ot(g["mu_src"][0], g["sig_src"][0],
                                 g["mu_dst"][0], g["sig_dst"][0])
    y_affine = bg[0] * x + big[0]
    check("gaussian build == directions.gaussian_ot map",
          torch.allclose(y_affine, ref(x), atol=1e-4))
    # Analytic OT identity: map sends source moments EXACTLY to dst moments.
    check("gaussian maps mu_src -> mu_dst exactly",
          torch.allclose(bg[0] * g["mu_src"][0] + big[0], g["mu_dst"][0], atol=1e-4))
    check("gaussian scales sig_src -> sig_dst exactly",
          torch.allclose(bg[0].abs() * g["sig_src"][0], g["sig_dst"][0], atol=1e-4))

    # (ii) empirical: fit dst = a*src + b per neuron, recover a,b (sorted-OT LS).
    a = torch.rand(feat) * 2 + 0.5; b = torch.randn(feat)
    src = torch.randn(2000, feat)
    dst = a * src + b
    blob2 = {"acts": torch.stack([torch.stack([dst[i]] * L) for i in range(2000)]
                                 + [torch.stack([src[i]] * L) for i in range(2000)]),
             "labels": torch.tensor([1] * 2000 + [0] * 2000),
             "where": WHERE, "feat": feat, "n_layers": L, "n_items": 2000}
    st2 = fit_stats_from_blob(blob2)
    be, bi, _ = build_injection("empirical", strength=1.0, stats=st2)
    check("empirical recovers beta (a)", torch.allclose(be[0], a, atol=5e-2))
    check("empirical recovers bias (b)", torch.allclose(bi[0], b, atol=5e-2))

    # strength folding: s=1 identity-of-map; s>1 extrapolates; s=0 => identity.
    be1, bi1, _ = build_injection("empirical", strength=1.0, stats=st2)
    be2, bi2, _ = build_injection("empirical", strength=2.0, stats=st2)
    be0, bi0, _ = build_injection("empirical", strength=0.0, stats=st2)
    check("strength=1 -> beta_eff==beta", torch.allclose(be1[0], a, atol=5e-2))
    check("strength=2 -> beta_eff==2beta-1",
          torch.allclose(be2[0], 2 * be1[0] - 1.0, atol=1e-5)
          and torch.allclose(bi2[0], 2 * bi1[0], atol=1e-5))
    check("strength=0 -> identity (beta=1,bias=0)",
          torch.allclose(be0[0], torch.ones(feat), atol=1e-5)
          and torch.allclose(bi0[0], torch.zeros(feat), atol=1e-5))

    # affine_pre_hook math: edits input[0] as beta_eff*x + bias_eff, tuple-safe.
    common.reset_fire_count()
    xin = torch.randn(2, 7, feat)
    hk = affine_pre_hook(be2[0], bi2[0])
    r = hk(None, (xin.clone(), "extra"))
    check("affine_pre_hook: beta_eff*x + bias_eff, extras kept",
          isinstance(r, tuple) and len(r) == 2 and r[1] == "extra"
          and torch.allclose(r[0], be2[0] * xin + bi2[0], atol=1e-5))
    check("affine_pre_hook bumps fire counter", common.get_fire_count() == 1)

    # E8 target-parameterized fit-artifact paths.
    check("stats_path_for('black') == STATS_PATH (round-1 regression)",
          stats_path_for("black") == STATS_PATH)
    check("stats_path_for gender -> cache/linearact_stats_<t>.pt",
          stats_path_for("woman").endswith("cache/linearact_stats_woman.pt")
          and stats_path_for("man").endswith("cache/linearact_stats_man.pt"))

    # feat / layer wiring sanity (MLP-hidden = 12288, all 32 layers).
    check("H_MLP == 12288 and N_LAYERS == 32", H_MLP == 12288 and N_LAYERS == 32)
    check("submodule_paths('ff_out') -> blocks[k].ff_out",
          submodule_paths("ff_out", [0, 31]) ==
          [f"{BLOCKS_PATH}.0.ff_out", f"{BLOCKS_PATH}.31.ff_out"])

    print(f"[selftest-linearact] OVERALL: {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true",
                    help="offline OT-map + affine-hook math check (no GPU)")
    ap.add_argument("--fit", action="store_true",
                    help="collect mlp_hidden acts + fit both variants (NEEDS GPU)")
    ap.add_argument("--run", action="store_true",
                    help="run one BBQ injection condition (NEEDS GPU)")
    ap.add_argument("--variant", choices=VARIANTS, default="empirical")
    ap.add_argument("--strength", type=float, default=2.0,
                    help="AcT transport strength; 1=full OT, >1 extrapolates")
    ap.add_argument("--cap", type=int, default=calib.CAP)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--target", choices=common.SUPPORTED_TARGETS, default="black",
                    help="steering target (black = round-1 default; woman/man = "
                         "E3 gender calib fit + classification)")
    ap.add_argument("--items", default=None,
                    help="BBQ items jsonl (e.g. a position-balance rotation "
                         "file); default = the target's own _sweep400 file")
    ap.add_argument("--out_dir", default=RESULTS_DIR)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    if args.selftest:
        sys.exit(0 if _selftest() else 1)
    if args.fit:
        fit(cap=args.cap, target=args.target)             # NEEDS GPU
        return
    if args.run:
        run(variant=args.variant, strength=args.strength,   # NEEDS GPU
            out_dir=args.out_dir, tag=args.tag, limit=args.limit,
            items_path=args.items, target=args.target)
        return
    ap.error("nothing to do: pass --selftest (offline), --fit or --run (GPU)")


if __name__ == "__main__":
    main()
