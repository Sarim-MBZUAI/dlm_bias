#!/usr/bin/env python
"""E2 construction #4: a FairPCA-subspace steering direction at L14.

Contrast with the group mean-difference vector (race_black.pt):
  * mean-diff   = MEAN of the per-pair difference vectors  d_i = emb(stereo)-emb(anti).
  * FairPCA     = leading PRINCIPAL COMPONENT (top singular vector) of the stacked
                  difference vectors D = [d_1; ...; d_n]  (i.e. top eigenvector of
                  the uncentered second-moment  D^T D  =  Σ d_i d_i^T).

The top PC is the rank-1 direction maximizing Σ (d_i · v)^2 — the axis the pairs
most consistently vary along. For perfectly coherent pairs it coincides with the
mean-diff; when pairs are noisy it down-weights inconsistent dimensions, giving a
"cleaner" group axis (the FairPCA / concept-subspace recipe, cf. DiffLens /
FairImagen in docs/literature.md §4). We ORIENT the sign toward the Black group
(dot with the mean-diff > 0) so +alpha pushes toward Black like the others.

Pairs come from race_steering/data/black_pairs.json (792 CrowS+StereoSet Black
pairs), embedded at block L14 via build_direction's masked-mean capture machinery.

GPU (the user runs this):
  CUDA_VISIBLE_DEVICES=5 /home/lukas/miniconda3/envs/sarim_awm/bin/python \
      experiments/build_fairpca.py
Offline PCA unit-test (no model, no GPU):
  /home/lukas/miniconda3/envs/sarim_awm/bin/python experiments/build_fairpca.py --self-test
"""
import argparse
import json
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_ROOT, "eval"))
sys.path.insert(0, os.path.join(_ROOT, "bias_steering"))

MODEL_PATH = "/home/lukas/users/shashmi/dlm_bias/LLaDA-8B-Instruct"
LAYER = 14
DEVICE = "cuda"
SPLITHALF_SEED = 1234
PAIRS = os.path.join(_ROOT, "race_steering", "data", "black_pairs.json")
OUT_PT = os.path.join(_ROOT, "directional_steering", "race_black_fairpca.pt")


def fairpca_direction(diffs):
    """Top principal component (unit) of stacked diff vectors, oriented toward
    the mean-diff. diffs: (n, H) float tensor. Returns (unit_pc, mean_diff)."""
    D = diffs.to(torch.float32)
    mean_diff = D.mean(dim=0)
    # top right singular vector of D == top eigenvector of D^T D (uncentered 2nd moment).
    U, S, Vh = torch.linalg.svd(D, full_matrices=False)
    pc = Vh[0]                                   # (H,), unit norm
    if torch.dot(pc, mean_diff) < 0:             # orient toward the Black (stereo) side
        pc = -pc
    return pc / pc.norm(), mean_diff


def cosine(a, b):
    return float(torch.dot(a, b) / (a.norm() * b.norm()).clamp(min=1e-12))


def self_test():
    """PCA recovers a known principal axis from synthetic diff vectors."""
    torch.manual_seed(0)
    H, n = 64, 500
    axis = torch.randn(H); axis = axis / axis.norm()
    # diffs = scaled axis + small isotropic noise (coherent group axis).
    coeffs = torch.randn(n, 1).abs() + 1.0
    diffs = coeffs * axis.unsqueeze(0) + 0.05 * torch.randn(n, H)
    pc, mean_diff = fairpca_direction(diffs)
    c_axis = abs(cosine(pc, axis))
    c_mean = cosine(pc, mean_diff)
    print("PCA self-test (synthetic, no model):")
    print(f"  cos(recovered PC, true axis)  = {c_axis:.4f}  (expect > 0.99)")
    print(f"  cos(recovered PC, mean-diff)  = {c_mean:.4f}  (expect > 0.99, coherent case)")
    print(f"  oriented toward mean-diff     : {c_mean > 0}")
    ok = c_axis > 0.99 and c_mean > 0.99
    # Anti-coherent case: half the pairs flipped -> PC still an axis, mean-diff shrinks.
    flip = torch.ones(n, 1); flip[: n // 2] = -1.0
    diffs2 = flip * coeffs * axis.unsqueeze(0) + 0.05 * torch.randn(n, H)
    pc2, mean2 = fairpca_direction(diffs2)
    print(f"  anti-coherent: |mean_diff| {diffs2.mean(0).norm():.3f} << PC still axis "
          f"cos={abs(cosine(pc2, axis)):.3f}")
    print(f"  --> {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true", help="offline PCA math check, no model")
    args = ap.parse_args()
    if args.self_test:
        self_test()
        return

    import build_direction as bd
    from transformers import AutoModel, AutoTokenizer

    pairs = json.load(open(PAIRS))
    print(f"Loaded {len(pairs)} Black pairs from {PAIRS}")
    print("Loading model ...")
    model = (AutoModel.from_pretrained(MODEL_PATH, trust_remote_code=True,
                                       torch_dtype=torch.bfloat16).to(DEVICE).eval())
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    embed_means, handles, state = bd.make_multilayer_embedder(
        model, tok, [LAYER], DEVICE, bd.DEFAULT_HOOK_MODULE)
    diffs, embed_norms = [], []
    try:
        for j, p in enumerate(pairs):
            es = embed_means(p["stereotype"])[LAYER]
            ea = embed_means(p["anti_stereotype"])[LAYER]
            diffs.append((es - ea).cpu())
            embed_norms.append(float(es.norm())); embed_norms.append(float(ea.norm()))
            if (j + 1) % 100 == 0 or (j + 1) == len(pairs):
                print(f"  [{j + 1}/{len(pairs)}]")
    finally:
        for h in handles:
            h.remove()

    D = torch.stack(diffs, dim=0)                    # (n, H)
    pc, mean_diff = fairpca_direction(D)
    # scale the unit PC to the mean-diff norm so raw_norm is comparable to race_black
    direction = pc * mean_diff.norm()

    # split-half coherence, same protocol as build_direction/build_anchored.
    n = D.shape[0]
    g = torch.Generator().manual_seed(SPLITHALF_SEED)
    perm = torch.randperm(n, generator=g)
    half = n // 2
    pa, _ = fairpca_direction(D[perm[:half]])
    pb, _ = fairpca_direction(D[perm[half:]])
    splithalf = cosine(pa, pb)

    torch.save({
        "direction": direction.cpu(),
        "hook_module": bd.DEFAULT_HOOK_MODULE,
        "hidden_size": int(direction.shape[0]),
        "layer": LAYER,
        "method": "fairpca_subspace",
        "contrast": "black_group_pairs_top_principal_component",
        "n_pairs": int(n),
        "raw_norm": float(direction.norm()),
        "avg_embed_norm": float(sum(embed_norms) / len(embed_norms)),
        "norm_ratio": float(direction.norm() / D.norm(dim=1).mean()),
        "splithalf_cosine": splithalf,
        "cos_to_mean_diff": cosine(pc, mean_diff),
        "source": "race_black_pairs_fairpca",
    }, OUT_PT)
    print(f"\nsplithalf_cosine={splithalf:.4f}  cos(PC, mean-diff)={cosine(pc, mean_diff):.4f}")
    print(f"saved -> {OUT_PT}")


if __name__ == "__main__":
    main()
