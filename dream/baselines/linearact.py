#!/usr/bin/env python
"""dream/baselines/linearact.py -- Linear-AcT (Apple "Activation Transport")
per-neuron 1-D OT steering on Dream-v0-Instruct-7B.  Port of baselines/linearact.py
at NATIVE (per-neuron) granularity, hooked at the MLP-HIDDEN units (the 18944-d
gated activation, INPUT to each block's down_proj).

Two per-neuron 1-D OT map variants fitted on labelled (Black vs other) MLP-hidden
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

GRANULARITY / HOOK SITE: In Dream's MLP  down_proj(act(gate_proj(x))*up_proj(x))
there is NO module whose OUTPUT is the 18944-d gated activation -- it is exactly
the INPUT to down_proj.  So the affine map is applied via common_dream.affine_pre_hook
on model.model.layers[k].mlp.down_proj (edits input[0]), the SAME activation
calib.collect_activations('mlp_hidden') captures.  Default: ALL 28 layers, all
positions, every diffusion step.

CLI:
  python dream/baselines/linearact.py --selftest                              # offline
  CUDA_VISIBLE_DEVICES=4 python dream/baselines/linearact.py --fit             # NEEDS GPU
  CUDA_VISIBLE_DEVICES=4 python dream/baselines/linearact.py --run --variant gaussian  --strength 2.0
  CUDA_VISIBLE_DEVICES=4 python dream/baselines/linearact.py --run --variant empirical --strength 2.0
"""
import argparse
import os
import sys

import torch

ROOT = "/home/lukas/users/shashmi/dlm_bias"
sys.path.insert(0, os.path.join(ROOT, "eval"))
sys.path.insert(0, os.path.join(ROOT, "steering"))
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))   # common_dream
sys.path.insert(0, _HERE)

import common_dream as C  # noqa: E402
import calib             # noqa: E402
import directions        # noqa: E402

WHERE = "mlp_hidden"          # max-faithfulness MLP-hidden units (18944-d)
VARIANTS = ("gaussian", "empirical")
STATS_PATH = os.path.join(calib.CACHE_DIR, "linearact_stats.pt")
RESULTS_DIR = os.path.join(ROOT, "results", "dream", "linearact")
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
    """Collect Black-vs-other MLP-hidden activations and fit both variants.  NEEDS GPU."""
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
    """Register the Linear-AcT affine on each layer's down_proj INPUT (18944-d)."""
    beta_eff, bias_eff, layers = build_injection(variant, strength, stats, layers)
    handles = []
    for i, k in enumerate(layers):
        hk = C.affine_pre_hook(beta_eff[i], bias_eff[i])
        handles += C.attach(model, [C.down_proj_path(int(k))], hk, pre=True)
    return handles


def run(variant="empirical", strength=1.0, out_dir=RESULTS_DIR, tag=None,
        layers=None, limit=0, model=None, tok=None, baseline_black_rate=None):
    """Run the sweep400 BBQ injection for one variant/strength.  NEEDS A GPU."""
    assert variant in VARIANTS, f"variant must be one of {VARIANTS}"
    tag = tag or variant
    stats = load_stats()

    def _attach(m):
        return attach_fn(m, variant=variant, strength=strength, stats=stats, layers=layers)

    return C.run_items(
        attach_fn=_attach, out_dir=out_dir, tag=tag, limit=limit,
        model=model, tok=tok, baseline_black_rate=baseline_black_rate,
        config_extra={"method": "linear_act", "variant": variant,
                      "strength": strength, "where": WHERE,
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

    C.reset_fire_count()
    xin = torch.randn(2, 7, feat)
    r = C.affine_pre_hook(be2[0], bi2[0])(None, (xin.clone(), "extra"))
    check("affine_pre_hook: beta_eff*x + bias_eff, extras kept",
          isinstance(r, tuple) and len(r) == 2 and r[1] == "extra"
          and torch.allclose(r[0], be2[0] * xin + bi2[0], atol=1e-5))
    check("affine_pre_hook bumps fire counter", C.get_fire_count() == 1)

    check("D_MLP == 18944 and N_LAYERS == 28", C.D_MLP == 18944 and C.N_LAYERS == 28)
    check("down_proj_path -> layers[k].mlp.down_proj",
          C.down_proj_path(0).endswith(".0.mlp.down_proj")
          and C.down_proj_path(27).endswith(".27.mlp.down_proj"))

    print(f"[selftest-linearact] OVERALL: {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--fit", action="store_true", help="collect mlp_hidden + fit (NEEDS GPU)")
    ap.add_argument("--run", action="store_true", help="run one condition (NEEDS GPU)")
    ap.add_argument("--variant", choices=VARIANTS, default="empirical")
    ap.add_argument("--strength", type=float, default=2.0)
    ap.add_argument("--cap", type=int, default=calib.CAP)
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
            out_dir=args.out_dir, tag=args.tag, limit=args.limit)
        return
    ap.error("nothing to do: pass --selftest, --fit or --run (GPU)")


if __name__ == "__main__":
    main()
