#!/usr/bin/env python
"""baselines/directions.py -- shared FIT PRIMITIVES (method-agnostic math).

Pure, offline-testable building blocks the 7 AcT baseline method files fit their
per-neuron transports / gates from.  Each primitive cites the AcT reference file
it is faithful to.  NONE of these touch the model or a GPU.

Reference tree (read for faithfulness):
    act/hooks/transport.py          GaussianOTHook / LearnableOTHook
    act/optimal_transport/ot_maps.py  solve_ot_1d (sorted 1-D OT)
    act/optimal_transport/archs.py    LinearProj.optimize (closed-form LS)
    act/hooks/aura_hook.py + act/utils/auroc.py   AURA gate

INJECTION convention: "Black" is the OT DESTINATION.  The diff-in-means arrow in
steering/arrows.pt is r = mean(h_black - h_other) (build_arrows.py:182,190), i.e.
it already points toward Black; positive strength injects toward Black.

CLI:  python -m baselines.directions --selftest   (offline, no GPU)
"""
import argparse
import os
import sys

import numpy as np
import torch

# Harness import (ROOT = MAIN tree; mirrors steering/pid_steer.py:49-51).
ROOT = "/home/lukas/users/shashmi/dlm_bias"
sys.path.insert(0, os.path.join(ROOT, "eval"))
sys.path.insert(0, os.path.join(ROOT, "steering"))
import pid_steer  # noqa: E402  (unit_rows)

ARROWS_PATH = os.path.join(ROOT, "steering", "arrows.pt")


# --------------------------------------------------------------------------- #
# 1) Diff-in-means arrows, per-layer unit-normalized.                          #
# --------------------------------------------------------------------------- #
def load_arrows(path=ARROWS_PATH, raw=False):
    """Return the (32,4096) Black-minus-other arrow set from steering/arrows.pt.

    Default: per-layer unit-normalized via pid_steer.unit_rows (pid_steer.py:91).
    raw=True: the raw (un-normed) diff-in-means (build_arrows.py:190).
    """
    blob = torch.load(path, map_location="cpu")
    r = blob["r"].to(torch.float32)          # (32,4096) raw
    return r if raw else pid_steer.unit_rows(r)


# --------------------------------------------------------------------------- #
# 2) Gaussian (per-neuron 1-D) Optimal Transport map.                          #
#    Faithful to GaussianOTHook.forward (transport.py:261):                    #
#        z_ot = (std2/std1) * (z - mu1) + mu2                                  #
#    with source = (mu1,std1), destination = (mu2,std2).                       #
# --------------------------------------------------------------------------- #
def gaussian_ot(mu1, sig1, mu2, sig2, eps=1e-4):
    """Return a callable x -> (sig2/sig1)*(x - mu1) + mu2  (per-neuron 1-D
    Gaussian OT from N(mu1,sig1) to N(mu2,sig2)).

    All args are 1-D tensors of length D (per-neuron stats), or scalars.  The
    returned map broadcasts over any leading (..., D) shape.

    To INJECT toward Black: source = non-Black stats (mu1,sig1), destination =
    Black stats (mu2,sig2).  std_eps guards against divide-by-zero on dead
    neurons (transport.py:208-210 uses the same std_eps mask idea)."""
    mu1 = torch.as_tensor(mu1, dtype=torch.float32)
    sig1 = torch.as_tensor(sig1, dtype=torch.float32)
    mu2 = torch.as_tensor(mu2, dtype=torch.float32)
    sig2 = torch.as_tensor(sig2, dtype=torch.float32)
    ratio = sig2 / sig1.clamp(min=eps)

    def _map(x):
        r = ratio.to(x.dtype).to(x.device)
        m1 = mu1.to(x.dtype).to(x.device)
        m2 = mu2.to(x.dtype).to(x.device)
        return r * (x - m1) + m2

    return _map


# --------------------------------------------------------------------------- #
# 3) Empirical (per-neuron) 1-D OT fit -> affine (beta, bias).                 #
#    Faithful to LearnableOTHook.fit (transport.py:693-721):                   #
#      * per neuron, sort src and dst (solve_ot_1d, ot_maps.py:9-21) to get the #
#        monotone 1-D OT pairing;                                              #
#      * closed-form least squares on the sorted pairs (archs.py:30-54):        #
#            beta = sum(xc*yc)/sum(xc^2),  bias = mean(y) - beta*mean(x).       #
# --------------------------------------------------------------------------- #
def empirical_ot_fit(x_src, x_dst, eps=1e-8):
    """Fit a per-neuron affine map src -> dst by sorted-quantile matching + LS.

    x_src: (Ns, D) source samples (e.g. non-Black activations).
    x_dst: (Nd, D) destination samples (e.g. Black activations).
    Requires Ns == Nd (solve_ot_1d matches equal counts; ot_maps.py:15-18).

    Returns (beta, bias), each (D,), so that  beta*x + bias  transports a source
    neuron toward its destination.  Use with common.affine_hook / affine_pre_hook.
    """
    x_src = torch.as_tensor(x_src, dtype=torch.float64)
    x_dst = torch.as_tensor(x_dst, dtype=torch.float64)
    assert x_src.shape == x_dst.shape, (
        f"empirical_ot_fit needs equal-count, equal-dim tensors (1-D OT sorting); "
        f"got src {tuple(x_src.shape)} dst {tuple(x_dst.shape)}")
    # Per-neuron monotone OT pairing == sort each column independently.
    xs, _ = torch.sort(x_src, dim=0)
    ys, _ = torch.sort(x_dst, dim=0)
    # add tiny noise to xs to avoid 0/0 on constant neurons (archs.py:37).
    xs = xs + eps * torch.randn_like(xs)
    mx = xs.mean(dim=0, keepdim=True)
    my = ys.mean(dim=0, keepdim=True)
    xc = xs - mx
    yc = ys - my
    beta = (xc * yc).sum(dim=0) / (xc ** 2).sum(dim=0).clamp(min=eps)
    bias = (my.squeeze(0) - beta * mx.squeeze(0))
    return beta.to(torch.float32), bias.to(torch.float32)


# --------------------------------------------------------------------------- #
# 4) AUROC per neuron + AURA gate.                                             #
#    Faithful to compute_auroc (utils/auroc.py) and AURAHook._post_load        #
#    (aura_hook.py:62-67):  gate = 1 for auroc<=0.5, else 1 - 2*(auroc-0.5).   #
#    Equivalently  gate = 1 - 2*max(auroc-0.5, 0).                             #
# --------------------------------------------------------------------------- #
def _auroc_numpy(acts, labels):
    """Rank-based per-neuron AUROC (Mann-Whitney U), numpy fallback.
    acts: (N, D) float, labels: (N,) in {0,1}.  Returns (D,) AUROC."""
    labels = np.asarray(labels).astype(bool)
    pos = acts[labels]         # (Np, D)
    neg = acts[~labels]        # (Nn, D)
    Np, Nn = pos.shape[0], neg.shape[0]
    D = acts.shape[1]
    out = np.empty(D, dtype=np.float64)
    for j in range(D):
        order = np.argsort(acts[:, j], kind="mergesort")
        ranks = np.empty(acts.shape[0], dtype=np.float64)
        col = acts[order, j]
        i = 0
        n = col.shape[0]
        # average ranks for ties (1-indexed)
        while i < n:
            k = i
            while k + 1 < n and col[k + 1] == col[i]:
                k += 1
            avg = (i + k) / 2.0 + 1.0
            ranks[i:k + 1] = avg
            i = k + 1
        rank_full = np.empty(n, dtype=np.float64)
        rank_full[order] = ranks
        sum_pos = rank_full[labels].sum()
        out[j] = (sum_pos - Np * (Np + 1) / 2.0) / (Np * Nn)
    return out


def auroc_per_neuron(acts, labels, use_sklearn=True):
    """Per-neuron AUROC and the AURA dampening gate.

    acts: (N, D) responses; labels: (N,) in {0,1} (1 == the concept to detect,
    here "Black").  Returns (auroc, gate), each (D,) float32.

        gate = 1 - 2*max(auroc - 0.5, 0)   (aura_hook.py:65-67)

    Neurons that do NOT separate the classes (auroc<=0.5) keep gate=1 (untouched);
    strongly Black-selective neurons (auroc->1) get gate->0 (fully dampened).
    """
    acts = torch.as_tensor(acts, dtype=torch.float32).cpu().numpy()
    labels = torch.as_tensor(labels).cpu().numpy()
    auroc = None
    if use_sklearn:
        try:
            # Match act/utils/auroc.py: one roc_auc_score call per column set.
            from sklearn.metrics import roc_auc_score
            auroc = roc_auc_score(
                labels[:, None].repeat(acts.shape[1], 1), acts, average=None
            )
            auroc = np.asarray(auroc, dtype=np.float64)
        except Exception:
            auroc = None
    if auroc is None:
        auroc = _auroc_numpy(acts, labels)
    auroc_t = torch.tensor(auroc, dtype=torch.float32)
    gate = 1.0 - 2.0 * (auroc_t - 0.5).clamp(min=0.0)
    return auroc_t, gate


# --------------------------------------------------------------------------- #
# Offline self-test (analytic ground truth).                                   #
# --------------------------------------------------------------------------- #
def _selftest():
    torch.manual_seed(0)
    np.random.seed(0)
    ok = True

    def check(name, cond):
        nonlocal ok
        ok &= bool(cond)
        print(f"[selftest-directions] {name:48s} : {'PASS' if cond else 'FAIL'}")

    # (2) Gaussian OT: pure shift (sig1==sig2) recovers x + (mu2-mu1).
    D = 8
    mu1 = torch.randn(D); mu2 = torch.randn(D); sig = torch.rand(D) + 0.5
    f = gaussian_ot(mu1, sig, mu2, sig)
    x = torch.randn(5, D)
    check("gaussian_ot pure shift == x + (mu2-mu1)",
          torch.allclose(f(x), x + (mu2 - mu1), atol=1e-5))
    # scale+shift form
    sig2 = torch.rand(D) + 0.5
    f2 = gaussian_ot(mu1, sig, mu2, sig2)
    check("gaussian_ot scale+shift == (s2/s1)(x-m1)+m2",
          torch.allclose(f2(x), (sig2 / sig) * (x - mu1) + mu2, atol=1e-5))

    # (3) Empirical OT fit: dst = a*src + b  ->  recover beta~a, bias~b.
    N, D = 2000, 6
    a = torch.rand(D) * 2 + 0.5
    b = torch.randn(D)
    src = torch.randn(N, D)
    dst = a * src + b                     # same ordering, so sort pairs align
    beta, bias = empirical_ot_fit(src, dst)
    check("empirical_ot_fit recovers beta (a)",
          torch.allclose(beta, a, atol=5e-2))
    check("empirical_ot_fit recovers bias (b)",
          torch.allclose(bias, b, atol=5e-2))
    # pure shift special case
    beta2, bias2 = empirical_ot_fit(src, src + 3.0)
    check("empirical_ot_fit pure shift: beta~1, bias~3",
          torch.allclose(beta2, torch.ones(D), atol=5e-2)
          and torch.allclose(bias2, torch.full((D,), 3.0), atol=5e-2))

    # (4) AUROC / AURA gate.
    N = 1000
    labels = torch.cat([torch.ones(N // 2), torch.zeros(N // 2)])
    sep = torch.cat([torch.randn(N // 2) + 8.0, torch.randn(N // 2) - 8.0])  # separable
    rnd = torch.randn(N)                                                    # uninformative
    acts = torch.stack([sep, rnd], dim=1)                                   # (N,2)
    auroc, gate = auroc_per_neuron(acts, labels)
    check("auroc separable neuron ~1", auroc[0] > 0.99)
    check("auroc random neuron ~0.5", abs(float(auroc[1]) - 0.5) < 0.1)
    check("gate separable neuron ~0", gate[0] < 0.02)
    check("gate random neuron ~1", abs(float(gate[1]) - 1.0) < 0.05)
    # gate formula exactness
    man = 1.0 - 2.0 * (auroc - 0.5).clamp(min=0.0)
    check("gate == 1 - 2*max(auroc-0.5,0)", torch.allclose(gate, man))

    # (1) load_arrows shape/unit (only if the arrows file exists; no GPU needed).
    if os.path.exists(ARROWS_PATH):
        u = load_arrows()
        norms = u.norm(dim=1)
        check("load_arrows unit rows: (32,4096), ||row||~1",
              tuple(u.shape) == (32, 4096)
              and torch.allclose(norms, torch.ones(32), atol=1e-4))
    else:
        print(f"[selftest-directions] load_arrows: SKIP (no {ARROWS_PATH})")

    print(f"[selftest-directions] OVERALL: {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true",
                    help="offline math check on analytic synthetic data (no GPU)")
    args = ap.parse_args()
    if args.selftest:
        sys.exit(0 if _selftest() else 1)
    ap.error("nothing to do: pass --selftest")


if __name__ == "__main__":
    main()
