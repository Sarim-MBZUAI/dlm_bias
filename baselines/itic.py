#!/usr/bin/env python
"""ITI-C baseline: Inference-Time Intervention, constant variant (Li et al.,
2023, "Inference-Time Intervention: Eliciting Truthful Answers from a Language
Model", NeurIPS 2023, arXiv:2306.03341).  Implemented from the paper.

  * Attention is written head-wise, x_{l+1} = x_l + sum_h P_h^l Att_h^l(...)
    (paper Eq. 1), so each head contributes a d_head-dim activation.
  * For every head fit a linear probe on its activation to predict the label
    (here target vs. non-target answer) and select the top-K heads by probe
    validation accuracy.
  * At inference shift each selected head by a constant amount (paper Eq. 2):
        x_h <- x_h + alpha * sigma_h * theta_h
    theta_h is the unit mass-mean-shift direction (mean_pos - mean_neg,
    oriented toward the target, so positive alpha injects toward it) and
    sigma_h the std of the head activations projected onto theta_h.

LLaDA specifics: n_heads=32, d_model=4096, d_head=128.  The per-head activation
is the INPUT to blocks[k].attn_out (att.transpose(1,2).contiguous().view(B,T,C)
in LLaDABlock.attention()), so the shift is applied via a forward_pre_hook on
attn_out, reshaped to (B,T,32,128).  All positions, every denoising step.

Defaults: K=48 heads, alpha=15.  --tiebreak margin breaks val_acc ties by the
standardized head margin (see head_margin).

Usage:
    python baselines/itic.py --selftest                   # offline, no GPU
    python baselines/itic.py --fit                        # GPU
    python baselines/itic.py --run --topk 48 --alpha 15   # GPU
"""
import argparse
import os
import sys

import numpy as np
import torch

# Harness import (same convention as steering/pid_steer.py).  ROOT holds
# model/arrows/data.
ROOT = os.environ.get("DLM_BIAS_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "eval"))
sys.path.insert(0, os.path.join(ROOT, "steering"))

import bbq_eval   # noqa: E402,F401  (harness; imported for parity w/ siblings)
import pid_steer  # noqa: E402,F401

from common import (  # noqa: E402
    unit_rows,          # per-layer / per-row unit-normalize (N,H)
    submodule_paths,    # 'model.transformer.blocks.<k>.attn_out'
    attach,             # register hooks, returns handles
    make_pre_edit_hook, # wrap edit(x)->x into a fire-counting forward_pre_hook
    run_baseline,       # the shared GPU eval loop
    load_model,
    CACHE_DIR,
    SWEEP400,           # default BBQ items file (black; see sweep400_path)
    N_LAYERS,           # 32
    N_HEADS,            # 32
    D_HEAD,             # 128
    H_MODEL,            # 4096
    SUPPORTED_TARGETS,
    target_suffix,      # '' for black, '_<t>' otherwise
)
import calib  # noqa: E402  (collect_activations('attn_head'))

# Hard invariant ITI relies on: the reassembled attention activation splits
# EXACTLY into n_heads d_head-slices with no remainder (paper Eq. 1).
assert N_HEADS * D_HEAD == H_MODEL, (
    f"ITI needs n_heads*d_head == d_model; got {N_HEADS}*{D_HEAD} != {H_MODEL}")

PROBES_PATH = os.path.join(CACHE_DIR, "itic_probes.pt")
DEFAULT_TOPK = 48
DEFAULT_ALPHA = 15.0
RESULTS_DIR = os.path.join(ROOT, "results", "itic")


def probes_path_for(target):
    """cache/itic_probes.pt for black; cache/itic_probes_<target>.pt otherwise."""
    return os.path.join(CACHE_DIR, f"itic_probes{target_suffix(target)}.pt")


# --------------------------------------------------------------------------- #
# Per-head probe fit (pure; operates on already-collected activations).        #
# --------------------------------------------------------------------------- #
def _fit_one_head(Xh, y, seed=0, val_frac=0.25):
    """Fit ITI's per-head quantities on one head's activations.

    Xh : (N, d_head) float  -- pooled per-head activation for N calib responses.
    y  : (N,)         int   -- 1 = Black (positive), 0 = non-Black.

    Returns (val_acc, theta_unit, sigma):
      * val_acc    : linear-probe validation accuracy used ONLY for top-K
                     selection (paper Sec. 3).  Logistic regression when sklearn
                     is available (faithful to the paper's "linear probe"); a
                     diff-in-means projection classifier otherwise (same decision
                     boundary family, no extra deps).
      * theta_unit : (d_head,) UNIT mass-mean-shift direction
                     unit(mean_Black - mean_other), oriented toward Black
                     (paper Sec. 3, "mass mean shift").
      * sigma      : std of the head activations PROJECTED onto theta_unit
                     (the sigma_h scalar in paper Eq. 2).
    """
    Xh = torch.as_tensor(Xh, dtype=torch.float32)
    y = torch.as_tensor(y).long()
    N = Xh.shape[0]

    # theta: mass-mean-shift toward Black, unit-normalized (paper Sec. 3).
    mu_pos = Xh[y == 1].mean(dim=0)
    mu_neg = Xh[y == 0].mean(dim=0)
    theta = mu_pos - mu_neg
    theta_unit = unit_rows(theta.unsqueeze(0)).squeeze(0)  # (d_head,), ||.||=1

    # sigma_h: std of activations along theta_unit (paper Eq. 2 magnitude).
    proj_all = Xh @ theta_unit
    sigma = float(proj_all.std(unbiased=False))

    # ---- validation-accuracy probe (selection only) ----
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
        # Diff-in-means projection classifier: threshold at the midpoint of the
        # two TRAIN class-mean projections; accuracy on the held-out split.
        tr_pos = Xh[tr_idx][ytr == 1] if (ytr == 1).any() else mu_pos.unsqueeze(0)
        tr_neg = Xh[tr_idx][ytr == 0] if (ytr == 0).any() else mu_neg.unsqueeze(0)
        thr = 0.5 * (float((tr_pos.mean(0) @ theta_unit)) +
                     float((tr_neg.mean(0) @ theta_unit)))
        pred = (Xh[val_idx] @ theta_unit > thr).long().numpy().astype(yva.dtype)
        val_acc = float((pred == yva).mean())

    return val_acc, theta_unit, sigma


def head_margin(Xh, y):
    """Standardized class separation of one head along its mass-mean-shift
    direction: ||mean_pos - mean_neg|| / sigma_h, where sigma_h is the std of
    all activations projected onto the unit direction (the same sigma that
    scales the ITI shift).  Used ONLY as a tie-breaker for top-K selection
    when many heads reach the same probe validation accuracy (e.g. on
    SocialStigmaQA, where >600 of 1024 heads reach val_acc = 1.0 and the
    paper's accuracy criterion no longer ranks anything)."""
    Xh = torch.as_tensor(Xh, dtype=torch.float32)
    y = torch.as_tensor(y).long()
    diff = Xh[y == 1].mean(dim=0) - Xh[y == 0].mean(dim=0)
    norm = float(diff.norm())
    if norm == 0.0:
        return 0.0
    sigma = float((Xh @ (diff / norm)).std(unbiased=False))
    return norm / sigma if sigma > 0 else float("inf")


def fit(model=None, tok=None, cap=calib.CAP, save=True, out_path=None,
        target="black"):
    """FIT stage (needs a GPU).

    1. calib.collect_activations('attn_head', target=...) -> pooled per-(layer)
       activation of width 4096 for 2*n_items labelled target/other calib
       responses.
    2. For every (layer, head), slice out the head's d_head=128 activation and
       run _fit_one_head -> (val_acc, theta_unit, sigma).
    3. Save cache/itic_probes[_<target>].pt with:
         theta   : (N_LAYERS, N_HEADS, D_HEAD)  unit directions (target-oriented)
         sigma   : (N_LAYERS, N_HEADS)          per-head projection std
         val_acc : (N_LAYERS, N_HEADS)          probe validation accuracy
       plus meta (n_items, source, dims).

    Selection of top-K and the concrete additive vectors happen later in
    build_injection so K/alpha can be swept without re-collecting activations.
    """
    out_path = out_path or probes_path_for(target)
    blob = calib.collect_activations("attn_head", model=model, tok=tok, cap=cap,
                                     save=False, target=target)
    acts = blob["acts"].to(torch.float32)   # (2n, N_LAYERS, 4096)
    labels = blob["labels"].long()          # (2n,)
    assert acts.shape[1] == N_LAYERS and acts.shape[2] == H_MODEL, acts.shape

    # reshape residual-width -> heads: (2n, N_LAYERS, N_HEADS, D_HEAD)
    A = acts.view(acts.shape[0], N_LAYERS, N_HEADS, D_HEAD)

    theta = torch.zeros(N_LAYERS, N_HEADS, D_HEAD, dtype=torch.float32)
    sigma = torch.zeros(N_LAYERS, N_HEADS, dtype=torch.float32)
    val_acc = torch.zeros(N_LAYERS, N_HEADS, dtype=torch.float32)
    margin = torch.zeros(N_LAYERS, N_HEADS, dtype=torch.float32)
    for k in range(N_LAYERS):
        for hh in range(N_HEADS):
            va, th, sg = _fit_one_head(A[:, k, hh, :], labels)
            val_acc[k, hh] = va
            theta[k, hh] = th
            sigma[k, hh] = sg
            margin[k, hh] = head_margin(A[:, k, hh, :], labels)
        print(f"[itic:fit] layer {k+1}/{N_LAYERS} "
              f"best_head_acc={float(val_acc[k].max()):.3f}", flush=True)

    out = {
        "theta": theta, "sigma": sigma, "val_acc": val_acc, "margin": margin,
        "n_layers": N_LAYERS, "n_heads": N_HEADS, "d_head": D_HEAD,
        "n_items": blob["n_items"], "source": blob["source"],
        "method": "iti_c", "target": target,
        "direction": f"mass_mean_shift_{target}_minus_other",
    }
    if save:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        torch.save(out, out_path)
        print(f"[itic:fit] SAVED probes -> {out_path}", flush=True)
    return out


def load_probes(path=None, target="black"):
    """Load cache/itic_probes[_<target>].pt (produced by fit(); needs a GPU)."""
    return torch.load(path or probes_path_for(target), map_location="cpu")


# --------------------------------------------------------------------------- #
# Selection + additive vectors (pure; offline-testable).                       #
# --------------------------------------------------------------------------- #
def select_topk(val_acc, K, margin=None):
    """Indices of the TOP-K heads by validation accuracy (paper Sec. 3).

    val_acc: (N_LAYERS, N_HEADS).  Returns a bool mask (N_LAYERS, N_HEADS) with
    exactly min(K, N_LAYERS*N_HEADS) True entries (the highest-accuracy heads).

    Ties in val_acc are broken
      * by flattened index (torch.topk order) when margin is None (default,
        used for BBQ); or
      * by descending `margin` (N_LAYERS, N_HEADS; see head_margin) when it is
        given, i.e. lexicographic (val_acc desc, margin desc, index asc).
    """
    va = torch.as_tensor(val_acc, dtype=torch.float32)
    flat = va.reshape(-1)
    K = int(min(max(K, 0), flat.numel()))
    mask = torch.zeros_like(flat, dtype=torch.bool)
    if K > 0:
        if margin is None:
            top = torch.topk(flat, K).indices
        else:
            mg = torch.as_tensor(margin, dtype=torch.float32).reshape(-1)
            assert mg.numel() == flat.numel(), (mg.shape, flat.shape)
            idx = np.arange(flat.numel())
            # np.lexsort sorts by the LAST key first: primary = -val_acc,
            # secondary = -margin, tertiary = index (all ascending).
            order = np.lexsort((idx, -mg.numpy().astype(np.float64),
                                -flat.numpy().astype(np.float64)))
            top = torch.as_tensor(order[:K].copy())
        mask[top] = True
    return mask.reshape(va.shape)


def build_injection(K=DEFAULT_TOPK, alpha=DEFAULT_ALPHA, probes=None,
                    target="black", tiebreak="index"):
    """Build the constant per-head shift vectors for ITI-C (paper Eq. 2).

    For each SELECTED head (top-K by val_acc) the shift is
        alpha * sigma_h * theta_unit_h     (d_head vector)
    placed in that head's slice; non-selected heads get zero.

    tiebreak: "index" (default; torch.topk order) or "margin" (requires
    probes["margin"]; see select_topk / head_margin).

    Returns a dict:
        delta_heads : (N_LAYERS, N_HEADS, D_HEAD)  the constant shifts
        mask        : (N_LAYERS, N_HEADS)          selected-head bool mask
        layers      : sorted list of layers with >=1 selected head
        K, alpha, tiebreak, n_tied_at_top
    """
    if probes is None:
        probes = load_probes(target=target)
    theta = probes["theta"].to(torch.float32)     # (L,H,d) unit
    sigma = probes["sigma"].to(torch.float32)     # (L,H)
    val_acc = probes["val_acc"].to(torch.float32) # (L,H)

    if tiebreak == "index":
        margin = None
    elif tiebreak == "margin":
        if "margin" not in probes:
            raise KeyError("tiebreak='margin' needs probes['margin'] "
                           "(refit, or add it with head_margin on the cached "
                           "attn_head activations)")
        margin = probes["margin"].to(torch.float32)
    else:
        raise ValueError(f"unknown tiebreak {tiebreak!r}")

    mask = select_topk(val_acc, K, margin=margin)  # (L,H) bool
    # alpha * sigma_h * theta_unit_h, zeroed on non-selected heads.
    delta_heads = (alpha * sigma.unsqueeze(-1) * theta)      # (L,H,d)
    delta_heads = delta_heads * mask.unsqueeze(-1).to(delta_heads.dtype)
    layers = [k for k in range(N_LAYERS) if bool(mask[k].any())]
    # How many heads share the top accuracy: if > K, selection among them is
    # decided entirely by the tie-break rule.
    n_tied = int((val_acc >= float(val_acc.max()) - 1e-6).sum())
    return {"delta_heads": delta_heads, "mask": mask, "layers": layers,
            "K": int(mask.sum()), "alpha": float(alpha),
            "tiebreak": tiebreak, "n_tied_at_top": n_tied}


# --------------------------------------------------------------------------- #
# Apply: head-level forward_pre_hook on attn_out INPUT.                         #
# --------------------------------------------------------------------------- #
def _head_add_edit(delta_hd):
    """edit(x) for one layer: reshape attn_out input to heads, add the constant
    per-head shift, reshape back.  delta_hd: (N_HEADS, D_HEAD)."""
    def edit(x):
        B, T, C = x.shape
        xh = x.reshape(B, T, N_HEADS, D_HEAD)
        xh = xh + delta_hd.to(xh.dtype).to(xh.device)
        return xh.reshape(B, T, C)
    return edit


def attach_fn(model, injection):
    """Attach ITI-C pre-hooks on blocks[k].attn_out for every layer with a
    selected head.  Built via common.make_pre_edit_hook so the shared fire
    counter increments and run_baseline's >0 assertion holds.

    The per-head activation only exists as attn_out's input, hence the
    sub-module pre-hook rather than a block-output hook."""
    delta_heads = injection["delta_heads"]
    handles = []
    for k in injection["layers"]:
        edit = _head_add_edit(delta_heads[k])
        path = submodule_paths("attn_out", layers=[k])  # one dotted path
        handles += attach(model, path, make_pre_edit_hook(edit), pre=True)
    return handles


def run(K=DEFAULT_TOPK, alpha=DEFAULT_ALPHA, out_dir=RESULTS_DIR, tag=None,
        limit=0, model=None, tok=None, baseline_black_rate=None, probes=None,
        items_path=None, target="black", tiebreak="index"):
    """RUN stage (NEEDS A GPU).  Build the top-K injection and evaluate on the
    400-item BBQ sweep via the shared run_baseline loop.  target selects the
    per-head probe fit (cache/itic_probes[_<target>].pt) and the eval
    classification.  tiebreak="margin"
    breaks val_acc ties by standardized head margin (see select_topk)."""
    inj = build_injection(K=K, alpha=alpha, probes=probes, target=target,
                          tiebreak=tiebreak)
    if tag is None:
        tag = f"itic_K{inj['K']}_a{alpha:g}" + ("_tb" if tiebreak == "margin" else "")
    return run_baseline(
        attach_fn=lambda m: attach_fn(m, inj),
        items_path=items_path,
        out_dir=out_dir, tag=tag, limit=limit, model=model, tok=tok,
        baseline_black_rate=baseline_black_rate,
        target=target,
        config_extra={"method": "iti_c", "K": inj["K"], "alpha": float(alpha),
                      "n_layers_hooked": len(inj["layers"]),
                      "layers_hooked": inj["layers"],
                      "tiebreak": inj["tiebreak"],
                      "n_heads_tied_at_top_val_acc": inj["n_tied_at_top"],
                      "probes_path": probes_path_for(target), "target": target,
                      "hook_target": "blocks[k].attn_out (input, per-head)"},
    )


# --------------------------------------------------------------------------- #
# Offline self-test: reshape round-trip, top-K, head-localized shift.          #
# --------------------------------------------------------------------------- #
def _selftest():
    torch.manual_seed(0)
    ok = True

    def check(name, cond):
        nonlocal ok
        ok &= bool(cond)
        print(f"[selftest-itic] {name:52s} : {'PASS' if cond else 'FAIL'}")

    check("n_heads*d_head == d_model (4096)", N_HEADS * D_HEAD == H_MODEL)

    # target-parameterized fit-artifact paths.
    check("probes_path_for('black') == PROBES_PATH (regression)",
          probes_path_for("black") == PROBES_PATH)
    check("probes_path_for gender -> cache/itic_probes_<t>.pt",
          probes_path_for("woman").endswith("cache/itic_probes_woman.pt")
          and probes_path_for("man").endswith("cache/itic_probes_man.pt"))

    # (a) head reshape round-trips EXACTLY.
    B, T = 2, 5
    x = torch.randn(B, T, H_MODEL)
    xh = x.reshape(B, T, N_HEADS, D_HEAD)
    check("reshape (B,T,4096)->(B,T,32,128)->back is identity",
          torch.equal(xh.reshape(B, T, H_MODEL), x))
    # head slice j equals flat slice [j*128:(j+1)*128]
    j = 7
    check("head-j reshape slice == flat [j*128:(j+1)*128]",
          torch.equal(xh[..., j, :], x[..., j * D_HEAD:(j + 1) * D_HEAD]))

    # (b) top-K selection picks the K highest val_acc.
    va = torch.rand(N_LAYERS, N_HEADS)
    K = 48
    mask = select_topk(va, K)
    check("select_topk picks exactly K heads", int(mask.sum()) == K)
    thr = torch.sort(va.reshape(-1), descending=True).values[K - 1]
    check("all selected heads >= K-th largest val_acc",
          bool((va[mask] >= thr - 1e-6).all()))
    check("no unselected head strictly exceeds a selected one",
          float(va[mask].min()) >= float(va[~mask].max()) - 1e-6)
    # clamp behaviour
    check("select_topk clamps K>total to total",
          int(select_topk(va, 10 ** 6).sum()) == N_LAYERS * N_HEADS)
    check("select_topk K=0 -> none", int(select_topk(va, 0).sum()) == 0)

    # (c) build_injection + hook: shift lands ONLY on selected heads and equals
    #     alpha*sigma_h*theta_unit_h there.
    theta = torch.randn(N_LAYERS, N_HEADS, D_HEAD)
    theta = theta / theta.norm(dim=-1, keepdim=True)     # unit rows
    sigma = torch.rand(N_LAYERS, N_HEADS) + 0.5
    probes = {"theta": theta, "sigma": sigma, "val_acc": va}
    alpha = 3.0
    inj = build_injection(K=K, alpha=alpha, probes=probes)
    dmask = inj["mask"]
    dh = inj["delta_heads"]
    # non-selected heads have exactly zero shift
    check("delta zero on all non-selected heads",
          torch.equal(dh[~dmask], torch.zeros_like(dh[~dmask])))
    # selected heads equal alpha*sigma*theta
    want = (alpha * sigma.unsqueeze(-1) * theta)[dmask]
    check("delta == alpha*sigma*theta on selected heads",
          torch.allclose(dh[dmask], want, atol=1e-6))
    check("build_injection.layers == layers with a selected head",
          inj["layers"] == [k for k in range(N_LAYERS) if bool(dmask[k].any())])

    # apply the per-layer edit closure on a synthetic attn_out input and confirm
    # only the selected heads of THAT layer move, by exactly the delta.
    k0 = inj["layers"][0]
    x = torch.randn(B, T, H_MODEL)
    edit = _head_add_edit(dh[k0])
    y = edit(x.clone())
    yh = y.reshape(B, T, N_HEADS, D_HEAD)
    xh = x.reshape(B, T, N_HEADS, D_HEAD)
    sel = dmask[k0]
    moved_ok = torch.allclose(yh[..., sel, :] - xh[..., sel, :],
                              dh[k0][sel].expand(B, T, int(sel.sum()), D_HEAD),
                              atol=1e-5)
    untouched_ok = torch.equal(yh[..., ~sel, :], xh[..., ~sel, :])
    check("edit moves selected heads by exactly delta", moved_ok)
    check("edit leaves non-selected heads untouched", untouched_ok)

    # (d) _fit_one_head on separable synthetic data: high val_acc, theta toward
    #     Black, sigma>0.
    N, d = 400, D_HEAD
    base = torch.randn(N, d)
    y = torch.cat([torch.ones(N // 2), torch.zeros(N // 2)]).long()
    bump = torch.zeros(d); bump[0] = 6.0
    Xh = base + (y.float().unsqueeze(1) * bump)          # Black shifted on dim 0
    acc, th, sg = _fit_one_head(Xh, y)
    check("_fit_one_head separable val_acc > 0.9", acc > 0.9)
    check("_fit_one_head theta oriented toward Black (dim0>0)", float(th[0]) > 0)
    check("_fit_one_head theta is unit", abs(float(th.norm()) - 1.0) < 1e-5)
    check("_fit_one_head sigma > 0", sg > 0)

    # (f) tie-breaking.  With all val_acc tied, index selection is the first
    # K flattened indices; margin selection is the K largest margins.
    va_tied = torch.ones(N_LAYERS, N_HEADS)
    mg = torch.rand(N_LAYERS, N_HEADS)
    m_legacy = select_topk(va_tied, 48)
    m_margin = select_topk(va_tied, 48, margin=mg)
    check("legacy tie-break is deterministic and picks K heads",
          int(m_legacy.sum()) == 48
          and torch.equal(m_legacy, select_topk(va_tied, 48)))
    check("margin tie-break selects the 48 largest margins",
          set(m_margin.reshape(-1).nonzero().squeeze(1).tolist())
          == set(torch.topk(mg.reshape(-1), 48).indices.tolist()))
    check("margin tie-break selects exactly K heads", int(m_margin.sum()) == 48)
    # val_acc still dominates margin: a head with lower accuracy but huge
    # margin must not be selected ahead of a full-accuracy head.
    va_mix = torch.ones(N_LAYERS, N_HEADS); va_mix[5, 5] = 0.5
    mg_mix = torch.zeros(N_LAYERS, N_HEADS); mg_mix[5, 5] = 1e6
    check("val_acc dominates margin (lexicographic)",
          not bool(select_topk(va_mix, 1023, margin=mg_mix)[5, 5]))
    va_r = torch.rand(N_LAYERS, N_HEADS)
    check("margin tie-break with distinct val_acc == legacy",
          torch.equal(select_topk(va_r, 48), select_topk(va_r, 48, margin=mg)))
    check("head_margin: separable data has margin > 1",
          head_margin(Xh, y) > 1.0)
    check("build_injection(tiebreak='margin') needs probes['margin']",
          _raises(KeyError, lambda: build_injection(
              K=4, alpha=1.0, tiebreak="margin",
              probes={"theta": torch.zeros(N_LAYERS, N_HEADS, D_HEAD),
                      "sigma": torch.ones(N_LAYERS, N_HEADS),
                      "val_acc": torch.ones(N_LAYERS, N_HEADS)})))

    print(f"[selftest-itic] OVERALL: {'PASS' if ok else 'FAIL'}")
    return ok


def _raises(exc, fn):
    try:
        fn()
    except exc:
        return True
    except Exception:
        return False
    return False


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true",
                    help="offline math check on synthetic tensors (NO GPU)")
    ap.add_argument("--fit", action="store_true",
                    help="collect activations + fit per-head probes (NEEDS GPU)")
    ap.add_argument("--run", action="store_true",
                    help="evaluate ITI-C on the BBQ sweep (NEEDS GPU)")
    ap.add_argument("--topk", type=int, default=DEFAULT_TOPK,
                    help=f"number of top heads to steer (default {DEFAULT_TOPK})")
    ap.add_argument("--alpha", type=float, default=DEFAULT_ALPHA,
                    help=f"constant injection strength (default {DEFAULT_ALPHA})")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--target", choices=SUPPORTED_TARGETS, default="black",
                    help="steering target (black = default; woman/man = "
                         "gender calib fit + classification)")
    ap.add_argument("--items", default=None,
                    help="BBQ items jsonl (e.g. a position-balance rotation "
                         "file); default = the target's own _sweep400 file")
    ap.add_argument("--out_dir", default=RESULTS_DIR)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--tiebreak", choices=("index", "margin"), default="index",
                    help="val_acc tie-break for top-K selection: 'index' "
                         "(legacy torch.topk order) or 'margin' (standardized "
                         "head margin; needs probes['margin'])")
    args = ap.parse_args()

    if args.selftest:
        sys.exit(0 if _selftest() else 1)
    if args.fit:
        fit(target=args.target)
        return
    if args.run:
        run(K=args.topk, alpha=args.alpha, out_dir=args.out_dir, tag=args.tag,
            limit=args.limit, items_path=args.items, target=args.target,
            tiebreak=args.tiebreak)
        return
    ap.error("nothing to do: pass --selftest (offline), --fit or --run (GPU)")


if __name__ == "__main__":
    main()
