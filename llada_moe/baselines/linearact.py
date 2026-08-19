#!/usr/bin/env python
"""llada_moe/baselines/linearact.py -- Linear-AcT (Apple "Activation Transport")
per-neuron 1-D OT steering on LLaDA-MoE-7B-A1B-Instruct.  Port of
dream/baselines/linearact.py, hooked at the MoE-MLP OUTPUT units (the 2048-d
pre-residual MLP delta each block's LLaDAMoESparseMoeBlock emits).

Two per-neuron 1-D OT map variants fitted on labelled (Black vs other) MLP
activations, Black = OT DESTINATION so both push toward Black:

  (i)  gaussian  -- z_ot = (sig_dst/sig_src)*(z - mu_src) + mu_dst
       (GaussianOTHook.forward, transport.py:261; directions.gaussian_ot).
  (ii) empirical -- per neuron: sort src/dst (1-D OT pairing) + closed-form LS
       affine beta*z+bias (LinearProj.optimize / LearnableOTHook.fit;
       directions.empirical_ot_fit).

Both are affine per neuron: z -> beta*z + bias.  AcT strength folds against the
identity (transport.py:264/756):
    beta_eff = strength*beta + (1-strength),  bias_eff = strength*bias
strength=1 full transport, 0 identity, >1 extrapolates (stronger injection).

GRANULARITY / HOOK SITE (MoE DEVIATION): LLaDA-MoE routes tokens across 64
1024-d experts, so there is NO dense gated MLP-hidden bank (Dream's 18944-d /
LLaDA's 12288-d).  The per-neuron MLP bank every token passes through is the
MoE block OUTPUT (model.layers[k].mlp, bare tensor, 2048-d) -- and since it IS
a module OUTPUT, the affine map is applied with common_lladamoe.affine_hook
(a plain forward hook), which is exactly AcT's InterventionHook contract (edit
the module OUTPUT), with no pre-hook workaround needed.  It is the SAME
activation calib.collect_activations('mlp_hidden') captures.  Default: ALL 16
layers, all positions, every diffusion step.

CLI:
  python llada_moe/baselines/linearact.py --selftest                          # offline
  python llada_moe/baselines/linearact.py --fit                               # NEEDS GPU
  python llada_moe/baselines/linearact.py --run --variant gaussian  --strength 1.0
  python llada_moe/baselines/linearact.py --run --variant empirical --strength 1.0
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
import calib             # noqa: E402
import directions        # noqa: E402

WHERE = "mlp_hidden"          # MoE-port MLP bank = MoE block OUTPUT (2048-d)
VARIANTS = ("gaussian", "empirical")
STATS_PATH = os.path.join(calib.CACHE_DIR, "linearact_stats.pt")
RESULTS_DIR = os.path.join(ROOT, "results", "lladamoe", "linearact")
EPS = 1e-4                    # matches directions.gaussian_ot / transport.std_eps


def fit_stats_from_blob(blob):
    """Fit BOTH OT variants per layer/neuron from a calib blob (Black=1 dst, other=0 src)."""
    acts = torch.as_tensor(blob["acts"], dtype=torch.float32)
    labels = torch.as_tensor(blob["labels"]).to(torch.bool)
    L, feat = acts.shape[1], acts.shape[2]
    src_sel = ~labels          # other (label 0) = OT source
    dst_sel = labels           # Black (label 1) = OT destination

    mu_src = torch.empty(L, feat); sig_src = torch.empty(L, feat)
    mu_dst = torch.empty(L, feat); sig_dst = torch.empty(L, feat)
    e_beta = torch.empty(L, feat); e_bias = torch.empty(L, feat)
    for k in range(L):
        a = acts[:, k, :]
        xs = a[src_sel]; xd = a[dst_sel]
        mu_src[k] = xs.mean(0); sig_src[k] = xs.std(0)
        mu_dst[k] = xd.mean(0); sig_dst[k] = xd.std(0)
        e_beta[k], e_bias[k] = directions.empirical_ot_fit(xs, xd)

    return {
        "where": blob.get("where", WHERE), "feat": feat, "n_layers": L,
        "n_items": blob.get("n_items"), "source": blob.get("source"),
        "gaussian": {"mu_src": mu_src, "sig_src": sig_src,
                     "mu_dst": mu_dst, "sig_dst": sig_dst},
        "empirical": {"beta": e_beta, "bias": e_bias},
    }


def fit(cap=calib.CAP, model=None, tok=None, out_path=STATS_PATH):
    """Collect Black-vs-other MoE-MLP activations and fit both variants.  NEEDS GPU."""
    blob = calib.collect_activations(WHERE, model=model, tok=tok, cap=cap, save=True)
    stats = fit_stats_from_blob(blob)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save(stats, out_path)
    print(f"[linearact.fit] gaussian+empirical stats "
          f"({stats['n_layers']}x{stats['feat']}) -> {out_path}", flush=True)
    return stats


def load_stats(path=STATS_PATH):
    return torch.load(path, map_location="cpu")


def build_injection(variant, strength=1.0, stats=None, layers=None):
    """Return (beta_eff, bias_eff, layers): per-selected-layer affine params."""
    assert variant in VARIANTS, f"variant must be one of {VARIANTS}"
    if stats is None:
        stats = load_stats()
    feat = stats["feat"]; L = stats["n_layers"]
    layers = list(range(L)) if layers is None else list(layers)

    if variant == "gaussian":
        g = stats["gaussian"]
        # FAITHFUL to AcT (transport.py:203-210,270): low-variance neurons in
        # EITHER class stay at IDENTITY (mask to beta=1,bias=0), not transported --
        # else near-dead source neurons get beta=sig_dst/1e-4 and over-inject.
        ss, sd = g["sig_src"], g["sig_dst"]
        valid = (ss > EPS) & (sd > EPS)
        ratio = torch.where(valid, sd / ss.clamp(min=EPS), torch.ones_like(ss))
        beta = ratio
        bias = torch.where(valid, g["mu_dst"] - ratio * g["mu_src"], torch.zeros_like(ss))
    else:
        e = stats["empirical"]
        beta = e["beta"].clone(); bias = e["bias"].clone()

    beta = beta[layers].to(torch.float32); bias = bias[layers].to(torch.float32)
    beta_eff = strength * beta + (1.0 - strength)
    bias_eff = strength * bias
    assert beta_eff.shape == (len(layers), feat), beta_eff.shape
    return beta_eff, bias_eff, layers


def attach_fn(model, variant="empirical", strength=1.0, stats=None, layers=None):
    """Register the Linear-AcT affine on each layer's MoE-mlp OUTPUT (2048-d).

    Plain forward hook (post) -- the MoE block's OUTPUT is the activation itself
    (AcT's native InterventionHook contract), unlike the Dream/LLaDA pre-hook
    workaround for fused dense MLPs."""
    beta_eff, bias_eff, layers = build_injection(variant, strength, stats, layers)
    handles = []
    for i, k in enumerate(layers):
        hk = C.affine_hook(beta_eff[i], bias_eff[i])
        handles += C.attach(model, [C.moe_mlp_path(int(k))], hk, pre=False)
    return handles


def run(variant="empirical", strength=1.0, out_dir=RESULTS_DIR, tag=None,
        layers=None, items_path=None, limit=0, model=None, tok=None,
        baseline_black_rate=None):
    """Run the sweep400 BBQ injection for one variant/strength.  NEEDS A GPU.

    Default tag encodes the strength (<variant>_s<strength>) so sweeps at
    different strengths don't overwrite each other; an explicit tag still wins."""
    assert variant in VARIANTS, f"variant must be one of {VARIANTS}"
    tag = tag or f"{variant}_s{strength:g}"
    stats = load_stats()

    def _attach(m):
        return attach_fn(m, variant=variant, strength=strength, stats=stats, layers=layers)

    return C.run_items(
        attach_fn=_attach, items_path=items_path or C.SWEEP400,
        out_dir=out_dir, tag=tag, limit=limit,
        model=model, tok=tok, baseline_black_rate=baseline_black_rate,
        config_extra={"method": "linear_act", "variant": variant,
                      "strength": strength, "where": WHERE,
                      "hook": "affine_hook on layers[k].mlp OUTPUT (2048-d MoE delta)",
                      "layers": "all" if layers is None else list(layers),
                      "intervention_position": "all_bidirectional",
                      "stats_path": STATS_PATH})


def _selftest():
    torch.manual_seed(0)
    ok = True

    def check(name, cond):
        nonlocal ok
        ok &= bool(cond)
        print(f"[selftest-linearact] {name:52s} : {'PASS' if cond else 'FAIL'}")

    L, feat, n = 3, 5, 4000
    mu_o = torch.randn(feat); s_o = torch.rand(feat) + 0.5
    mu_b = torch.randn(feat); s_b = torch.rand(feat) + 0.5
    acts, labels = [], []
    for _ in range(n):
        acts.append(mu_b + s_b * torch.randn(L, feat)); labels.append(1)
        acts.append(mu_o + s_o * torch.randn(L, feat)); labels.append(0)
    blob = {"acts": torch.stack(acts), "labels": torch.tensor(labels),
            "where": WHERE, "feat": feat, "n_layers": L, "n_items": n}

    stats = fit_stats_from_blob(blob)
    g = stats["gaussian"]
    check("gaussian mu_dst ~ Black mean", torch.allclose(g["mu_dst"][0], mu_b, atol=0.1))
    check("gaussian mu_src ~ other mean", torch.allclose(g["mu_src"][0], mu_o, atol=0.1))
    check("gaussian sig_dst ~ Black std", torch.allclose(g["sig_dst"][0], s_b, atol=0.1))
    check("gaussian sig_src ~ other std", torch.allclose(g["sig_src"][0], s_o, atol=0.1))

    bg, big, _ = build_injection("gaussian", strength=1.0, stats=stats)
    x = mu_o + s_o * torch.randn(500, feat)
    ref = directions.gaussian_ot(g["mu_src"][0], g["sig_src"][0],
                                 g["mu_dst"][0], g["sig_dst"][0])
    check("gaussian build == directions.gaussian_ot map",
          torch.allclose(bg[0] * x + big[0], ref(x), atol=1e-4))
    check("gaussian maps mu_src -> mu_dst exactly",
          torch.allclose(bg[0] * g["mu_src"][0] + big[0], g["mu_dst"][0], atol=1e-4))
    check("gaussian scales sig_src -> sig_dst exactly",
          torch.allclose(bg[0].abs() * g["sig_src"][0], g["sig_dst"][0], atol=1e-4))

    # low-variance identity mask: a dead neuron in either class stays at identity.
    st = {"feat": feat, "n_layers": 1,
          "gaussian": {"mu_src": torch.zeros(1, feat), "sig_src": torch.full((1, feat), 1e-9),
                       "mu_dst": torch.ones(1, feat), "sig_dst": torch.ones(1, feat)}}
    bd, bib, _ = build_injection("gaussian", strength=1.0, stats=st)
    check("low-variance src neuron masked to beta=1,bias=0",
          torch.allclose(bd[0], torch.ones(feat)) and torch.allclose(bib[0], torch.zeros(feat)))

    a = torch.rand(feat) * 2 + 0.5; b = torch.randn(feat)
    src = torch.randn(2000, feat); dst = a * src + b
    blob2 = {"acts": torch.stack([torch.stack([dst[i]] * L) for i in range(2000)]
                                 + [torch.stack([src[i]] * L) for i in range(2000)]),
             "labels": torch.tensor([1] * 2000 + [0] * 2000),
             "where": WHERE, "feat": feat, "n_layers": L, "n_items": 2000}
    st2 = fit_stats_from_blob(blob2)
    be, bi, _ = build_injection("empirical", strength=1.0, stats=st2)
    check("empirical recovers beta (a)", torch.allclose(be[0], a, atol=5e-2))
    check("empirical recovers bias (b)", torch.allclose(bi[0], b, atol=5e-2))

    be2, bi2, _ = build_injection("empirical", strength=2.0, stats=st2)
    be0, bi0, _ = build_injection("empirical", strength=0.0, stats=st2)
    check("strength=2 -> beta_eff==2beta-1",
          torch.allclose(be2[0], 2 * be[0] - 1.0, atol=1e-5)
          and torch.allclose(bi2[0], 2 * bi[0], atol=1e-5))
    check("strength=0 -> identity (beta=1,bias=0)",
          torch.allclose(be0[0], torch.ones(feat), atol=1e-5)
          and torch.allclose(bi0[0], torch.zeros(feat), atol=1e-5))

    # POST forward hook on a bare tensor (the MoE mlp OUTPUT contract).
    C.reset_fire_count()
    xin = torch.randn(2, 7, feat)
    r = C.affine_hook(be2[0], bi2[0])(None, None, xin.clone())
    check("affine_hook (MoE mlp OUTPUT): beta_eff*x + bias_eff, bare tensor",
          torch.is_tensor(r) and torch.allclose(r, be2[0] * xin + bi2[0], atol=1e-5))
    check("affine_hook bumps fire counter", C.get_fire_count() == 1)

    check("D_MLP == 2048 and N_LAYERS == 16", C.D_MLP == 2048 and C.N_LAYERS == 16)
    check("moe_mlp_path -> layers[k].mlp",
          C.moe_mlp_path(0).endswith(".0.mlp")
          and C.moe_mlp_path(15).endswith(".15.mlp"))

    print(f"[selftest-linearact] OVERALL: {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--fit", action="store_true", help="collect mlp acts + fit (NEEDS GPU)")
    ap.add_argument("--run", action="store_true", help="run one condition (NEEDS GPU)")
    ap.add_argument("--variant", choices=VARIANTS, default="empirical")
    ap.add_argument("--strength", type=float, default=2.0)
    ap.add_argument("--cap", type=int, default=calib.CAP)
    ap.add_argument("--items", default=None,
                    help="BBQ items jsonl (default: sweep400; e.g. a position-balance rotation file)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out_dir", default=RESULTS_DIR)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    if args.selftest:
        sys.exit(0 if _selftest() else 1)
    if args.fit:
        fit(cap=args.cap); return
    if args.run:
        run(variant=args.variant, strength=args.strength,
            out_dir=args.out_dir, tag=args.tag, items_path=args.items,
            limit=args.limit)
        return
    ap.error("nothing to do: pass --selftest, --fit or --run (GPU)")


if __name__ == "__main__":
    main()
