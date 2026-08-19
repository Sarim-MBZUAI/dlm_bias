#!/usr/bin/env python
"""llada_moe/baselines/itic.py -- ITI-C (Inference-Time Intervention, constant
variant) on LLaDA-MoE-7B-A1B-Instruct.  Port of dream/baselines/itic.py.
Li et al. 2023 (arXiv:2306.03341) -- NOT an AcT method; built directly from the
paper.

ITI (paper Sec. 3):
  * attention is head-wise: x_{l+1} = x_l + sum_h P_h^l Att_h^l(...)  (Eq. 1), so
    each head contributes a d_head activation into the residual via its slice of
    the output projection.
  * per head, fit a linear probe (val-accuracy) to predict the label (Black vs
    non-Black); select the TOP-K heads by validation accuracy.
  * at inference shift each selected head along a concept direction theta_h by a
    CONSTANT: x_h <- x_h + alpha * sigma_h * theta_h  (Eq. 2), theta_h a UNIT
    mass-mean-shift (mean_Black - mean_other), sigma_h the std of activations
    projected onto theta_h, alpha a single global strength.  Positive alpha injects
    toward Black.

LLaDA-MoE specifics (verified in common_lladamoe.py):
  * n_heads=16, d_model=2048 => d_head=128 (16*128=2048).  num_key_value_heads
    == 16 (no GQA), so the o_proj INPUT is the plain 2048-d concat of the 16
    head outputs; the per-head reshape is (...,16,128).
  * the per-head activation is the INPUT to model.layers[k].self_attn.o_proj
    (o_proj is the output projection P^l), the same submodule
    calib.collect_activations('attn_head') pools.  We steer it via a forward_pre_hook.
  * hook fires once per diffusion step over ALL positions, bidirectional.

CLI:
    python llada_moe/baselines/itic.py --selftest              # offline math, NO GPU
    python llada_moe/baselines/itic.py --fit                   # NEEDS GPU
    python llada_moe/baselines/itic.py --run --topk 48 --alpha 8
"""
import argparse
import os
import sys

import numpy as np
import torch

ROOT = os.environ.get("DLM_BIAS_ROOT") or os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "eval"))
sys.path.insert(0, os.path.join(ROOT, "steering"))
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))   # common_lladamoe
sys.path.insert(0, _HERE)

import common_lladamoe as C  # noqa: E402  (re-exports unit_rows from the LLaDA steering/pid_steer.py)
import calib            # noqa: E402

# ITI's hard invariant: reassembled attention splits EXACTLY into n_heads d_head-slices.
assert C.N_HEADS * C.HEAD_DIM == C.D_MODEL, (
    f"ITI needs n_heads*d_head == d_model; got {C.N_HEADS}*{C.HEAD_DIM} != {C.D_MODEL}")

PROBES_PATH = os.path.join(calib.CACHE_DIR, "itic_probes.pt")
DEFAULT_TOPK = 48
DEFAULT_ALPHA = 8.0
RESULTS_DIR = os.path.join(ROOT, "results", "lladamoe", "itic")


def _fit_one_head(Xh, y, seed=0, val_frac=0.25):
    """Fit ITI's per-head quantities: (val_acc, theta_unit, sigma).

    theta_unit = unit(mean_Black - mean_other) (mass mean shift toward Black);
    sigma = std of activations projected onto theta_unit; val_acc = linear-probe
    validation accuracy (LogisticRegression, else diff-in-means threshold)."""
    Xh = torch.as_tensor(Xh, dtype=torch.float32)
    y = torch.as_tensor(y).long()
    N = Xh.shape[0]

    mu_pos = Xh[y == 1].mean(dim=0); mu_neg = Xh[y == 0].mean(dim=0)
    theta_unit = C.unit_rows((mu_pos - mu_neg).unsqueeze(0)).squeeze(0)
    sigma = float((Xh @ theta_unit).std(unbiased=False))

    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(N, generator=g)
    n_val = max(1, int(round(val_frac * N)))
    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    Xtr, ytr = Xh[tr_idx].numpy(), y[tr_idx].numpy()
    Xva, yva = Xh[val_idx].numpy(), y[val_idx].numpy()

    val_acc = None
    try:
        from sklearn.linear_model import LogisticRegression
        if len(np.unique(ytr)) == 2:
            clf = LogisticRegression(max_iter=1000, C=1.0)
            clf.fit(Xtr, ytr)
            val_acc = float((clf.predict(Xva) == yva).mean())
    except Exception:
        val_acc = None
    if val_acc is None:
        tr_pos = Xh[tr_idx][ytr == 1] if (ytr == 1).any() else mu_pos.unsqueeze(0)
        tr_neg = Xh[tr_idx][ytr == 0] if (ytr == 0).any() else mu_neg.unsqueeze(0)
        thr = 0.5 * (float(tr_pos.mean(0) @ theta_unit) + float(tr_neg.mean(0) @ theta_unit))
        pred = (Xh[val_idx] @ theta_unit > thr).long().numpy().astype(yva.dtype)
        val_acc = float((pred == yva).mean())

    return val_acc, theta_unit, sigma


def fit(model=None, tok=None, cap=calib.CAP, save=True, out_path=PROBES_PATH):
    """Collect attn_head activations, fit per-(layer,head) probes.  NEEDS A GPU.

    Saves theta (16,16,128) unit dirs, sigma (16,16), val_acc (16,16).  Top-K and
    the additive vectors happen in build_injection so K/alpha sweep without refit."""
    blob = calib.collect_activations("attn_head", model=model, tok=tok, cap=cap, save=False)
    acts = blob["acts"].to(torch.float32)   # (2n, 16, 2048)
    labels = blob["labels"].long()
    assert acts.shape[1] == C.N_LAYERS and acts.shape[2] == C.D_MODEL, acts.shape

    A = acts.view(acts.shape[0], C.N_LAYERS, C.N_HEADS, C.HEAD_DIM)
    theta = torch.zeros(C.N_LAYERS, C.N_HEADS, C.HEAD_DIM, dtype=torch.float32)
    sigma = torch.zeros(C.N_LAYERS, C.N_HEADS, dtype=torch.float32)
    val_acc = torch.zeros(C.N_LAYERS, C.N_HEADS, dtype=torch.float32)
    for k in range(C.N_LAYERS):
        for hh in range(C.N_HEADS):
            va, th, sg = _fit_one_head(A[:, k, hh, :], labels)
            val_acc[k, hh] = va; theta[k, hh] = th; sigma[k, hh] = sg
        print(f"[itic:fit] layer {k+1}/{C.N_LAYERS} "
              f"best_head_acc={float(val_acc[k].max()):.3f}", flush=True)

    out = {"theta": theta, "sigma": sigma, "val_acc": val_acc,
           "n_layers": C.N_LAYERS, "n_heads": C.N_HEADS, "d_head": C.HEAD_DIM,
           "n_items": blob["n_items"], "source": blob["source"],
           "method": "iti_c", "direction": "mass_mean_shift_black_minus_other"}
    if save:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        torch.save(out, out_path)
        print(f"[itic:fit] SAVED probes -> {out_path}", flush=True)
    return out


def load_probes(path=PROBES_PATH):
    return torch.load(path, map_location="cpu")


def select_topk(val_acc, K):
    """Bool mask (16,16) with the TOP-K heads by validation accuracy."""
    va = torch.as_tensor(val_acc, dtype=torch.float32)
    flat = va.reshape(-1)
    K = int(min(max(K, 0), flat.numel()))
    mask = torch.zeros_like(flat, dtype=torch.bool)
    if K > 0:
        mask[torch.topk(flat, K).indices] = True
    return mask.reshape(va.shape)


def build_injection(K=DEFAULT_TOPK, alpha=DEFAULT_ALPHA, probes=None):
    """Constant per-head shifts alpha*sigma_h*theta_unit_h on top-K heads (0 else)."""
    if probes is None:
        probes = load_probes()
    theta = probes["theta"].to(torch.float32)
    sigma = probes["sigma"].to(torch.float32)
    val_acc = probes["val_acc"].to(torch.float32)

    mask = select_topk(val_acc, K)
    delta_heads = (alpha * sigma.unsqueeze(-1) * theta) * mask.unsqueeze(-1).to(theta.dtype)
    layers = [k for k in range(C.N_LAYERS) if bool(mask[k].any())]
    return {"delta_heads": delta_heads, "mask": mask, "layers": layers,
            "K": int(mask.sum()), "alpha": float(alpha)}


def _head_add_edit(delta_hd):
    """edit(x) for one layer: reshape o_proj input to heads, add per-head shift, reshape back."""
    def edit(x):
        B, T, Cc = x.shape
        xh = x.reshape(B, T, C.N_HEADS, C.HEAD_DIM)
        xh = xh + delta_hd.to(xh.dtype).to(xh.device)
        return xh.reshape(B, T, Cc)
    return edit


def attach_fn(model, injection):
    """ITI-C pre-hooks on layers[k].self_attn.o_proj for every layer with a selected head."""
    delta_heads = injection["delta_heads"]
    handles = []
    for k in injection["layers"]:
        edit = _head_add_edit(delta_heads[k])
        handles += C.attach(model, [C.o_proj_path(int(k))], C.make_pre_edit_hook(edit), pre=True)
    return handles


def run(K=DEFAULT_TOPK, alpha=DEFAULT_ALPHA, out_dir=RESULTS_DIR, tag=None,
        items_path=None, limit=0, model=None, tok=None, baseline_black_rate=None,
        probes=None):
    """Build the top-K injection and evaluate on the BBQ sweep.  NEEDS A GPU."""
    inj = build_injection(K=K, alpha=alpha, probes=probes)
    if tag is None:
        tag = f"itic_K{inj['K']}_a{alpha:g}"
    return C.run_items(
        attach_fn=lambda m: attach_fn(m, inj),
        items_path=items_path or C.SWEEP400,
        out_dir=out_dir, tag=tag, limit=limit, model=model, tok=tok,
        baseline_black_rate=baseline_black_rate,
        config_extra={"method": "iti_c", "K": inj["K"], "alpha": float(alpha),
                      "n_layers_hooked": len(inj["layers"]), "probes_path": PROBES_PATH,
                      "hook_target": "layers[k].self_attn.o_proj (input, per-head)"})


def _selftest():
    torch.manual_seed(0)
    ok = True

    def check(name, cond):
        nonlocal ok
        ok &= bool(cond)
        print(f"[selftest-itic] {name:52s} : {'PASS' if cond else 'FAIL'}")

    L, Hh, d, Dm = C.N_LAYERS, C.N_HEADS, C.HEAD_DIM, C.D_MODEL
    check("n_heads*d_head == d_model (2048)", Hh * d == Dm)

    # (a) head reshape round-trips EXACTLY.
    B, T = 2, 5
    x = torch.randn(B, T, Dm)
    xh = x.reshape(B, T, Hh, d)
    check("reshape (B,T,2048)->(B,T,16,128)->back is identity",
          torch.equal(xh.reshape(B, T, Dm), x))
    j = 7
    check("head-j reshape slice == flat [j*128:(j+1)*128]",
          torch.equal(xh[..., j, :], x[..., j * d:(j + 1) * d]))

    # (b) top-K selection.
    va = torch.rand(L, Hh); K = 48
    mask = select_topk(va, K)
    check("select_topk picks exactly K heads", int(mask.sum()) == K)
    thr = torch.sort(va.reshape(-1), descending=True).values[K - 1]
    check("all selected heads >= K-th largest", bool((va[mask] >= thr - 1e-6).all()))
    check("no unselected head strictly exceeds a selected one",
          float(va[mask].min()) >= float(va[~mask].max()) - 1e-6)
    check("select_topk clamps K>total", int(select_topk(va, 10 ** 6).sum()) == L * Hh)
    check("select_topk K=0 -> none", int(select_topk(va, 0).sum()) == 0)

    # (c) build_injection + edit: shift lands ONLY on selected heads.
    theta = torch.randn(L, Hh, d); theta = theta / theta.norm(dim=-1, keepdim=True)
    sigma = torch.rand(L, Hh) + 0.5
    probes = {"theta": theta, "sigma": sigma, "val_acc": va}
    alpha = 3.0
    inj = build_injection(K=K, alpha=alpha, probes=probes)
    dmask = inj["mask"]; dh = inj["delta_heads"]
    check("delta zero on all non-selected heads",
          torch.equal(dh[~dmask], torch.zeros_like(dh[~dmask])))
    check("delta == alpha*sigma*theta on selected heads",
          torch.allclose(dh[dmask], (alpha * sigma.unsqueeze(-1) * theta)[dmask], atol=1e-6))
    check("build_injection.layers == layers with a selected head",
          inj["layers"] == [k for k in range(L) if bool(dmask[k].any())])

    k0 = inj["layers"][0]
    x = torch.randn(B, T, Dm)
    y = _head_add_edit(dh[k0])(x.clone())
    yh = y.reshape(B, T, Hh, d); xh = x.reshape(B, T, Hh, d)
    sel = dmask[k0]
    check("edit moves selected heads by exactly delta",
          torch.allclose(yh[..., sel, :] - xh[..., sel, :],
                         dh[k0][sel].expand(B, T, int(sel.sum()), d), atol=1e-5))
    check("edit leaves non-selected heads untouched",
          torch.equal(yh[..., ~sel, :], xh[..., ~sel, :]))

    # pre-hook wiring: make_pre_edit_hook applies the edit to input[0], bumps counter.
    C.reset_fire_count()
    r = C.make_pre_edit_hook(_head_add_edit(dh[k0]))(None, (x.clone(), "extra"))
    check("make_pre_edit_hook: edits input[0], extras kept",
          isinstance(r, tuple) and len(r) == 2 and r[1] == "extra"
          and torch.allclose(r[0], _head_add_edit(dh[k0])(x)))
    check("make_pre_edit_hook bumps fire counter", C.get_fire_count() == 1)

    # (d) _fit_one_head on separable synthetic data.
    N = 400
    base = torch.randn(N, d)
    y = torch.cat([torch.ones(N // 2), torch.zeros(N // 2)]).long()
    bump = torch.zeros(d); bump[0] = 6.0
    Xh = base + (y.float().unsqueeze(1) * bump)
    acc, th, sg = _fit_one_head(Xh, y)
    check("_fit_one_head separable val_acc > 0.9", acc > 0.9)
    check("_fit_one_head theta oriented toward Black (dim0>0)", float(th[0]) > 0)
    check("_fit_one_head theta is unit", abs(float(th.norm()) - 1.0) < 1e-5)
    check("_fit_one_head sigma > 0", sg > 0)

    check("o_proj_path -> layers[k].self_attn.o_proj",
          C.o_proj_path(0).endswith(".0.self_attn.o_proj"))

    print(f"[selftest-itic] OVERALL: {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--fit", action="store_true", help="fit per-head probes (NEEDS GPU)")
    ap.add_argument("--run", action="store_true", help="evaluate ITI-C (NEEDS GPU)")
    ap.add_argument("--topk", type=int, default=DEFAULT_TOPK)
    ap.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    ap.add_argument("--items", default=None,
                    help="BBQ items jsonl (default: sweep400; e.g. a position-balance rotation file)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out_dir", default=RESULTS_DIR)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    if args.selftest:
        sys.exit(0 if _selftest() else 1)
    if args.fit:
        fit(); return
    if args.run:
        run(K=args.topk, alpha=args.alpha, out_dir=args.out_dir, tag=args.tag,
            items_path=args.items, limit=args.limit)
        return
    ap.error("nothing to do: pass --selftest, --fit or --run (GPU)")


if __name__ == "__main__":
    main()
