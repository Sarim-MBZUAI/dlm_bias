#!/usr/bin/env python
"""DIAGNOSTIC: does the steering DIRECTION change across the denoising trajectory?

Build the sentiment contrastive direction at several MASK RATIOS r (r=noise level:
0 = clean/late-step analogue, 0.9 = mostly-masked/early-step analogue), by randomly
masking r of each example's tokens before the forward pass and capturing L14.
Report split-half coherence per ratio + the cosine between directions at different r.

If cos(v_r0, v_r0.9) ~ 1  -> one fixed vector suffices; adaptive-direction is pointless.
If it drops well below 1  -> the direction is noise-level-dependent -> adaptive-direction
is a real, unexplored signal worth a method.
"""
import os, sys, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run as R
from transformers import AutoModel, AutoTokenizer

DEV = "cuda"; L = 14; SEED = 42
model = AutoModel.from_pretrained(R.MODEL, trust_remote_code=True,
                                  torch_dtype=torch.bfloat16).to(DEV).eval()
tok = AutoTokenizer.from_pretrained(R.MODEL, trust_remote_code=True)
blocks = R.resolve_blocks(model)
pad = tok.pad_token_id if tok.pad_token_id is not None else -1

@torch.no_grad()
def dir_at_ratio(r):
    pos, neg = R.pairs("sentiment")
    cap = {}
    h = blocks[L].register_forward_hook(
        lambda m, i, o: cap.__setitem__("h", (o[0] if isinstance(o, tuple) else o).detach()))
    g = torch.Generator(device=DEV).manual_seed(SEED)
    def rep(sents):
        rows = []
        for s in sents:
            ids = torch.tensor(tok(s)["input_ids"], device=DEV).unsqueeze(0)
            if r > 0:
                T = ids.shape[1]; k = max(1, int(r * T))
                idx = torch.randperm(T, generator=g, device=DEV)[:k]
                ids = ids.clone(); ids[0, idx] = R.MASK_ID
            model(ids)
            hs = cap["h"].float()
            rows.append(R.masked_mean(hs, ids, pad).squeeze(0))
        return torch.stack(rows)
    P = rep(pos); N = rep(neg); h.remove()
    d = P.mean(0) - N.mean(0)
    vc = d / d.norm()
    a = (P[::2].mean(0) - N[::2].mean(0)); b = (P[1::2].mean(0) - N[1::2].mean(0))
    shc = torch.nn.functional.cosine_similarity(a, b, dim=0).item()
    return vc, shc

ratios = [0.0, 0.25, 0.5, 0.75, 0.9]
vs, shcs = [], []
for r in ratios:
    v, s = dir_at_ratio(r); vs.append(v); shcs.append(s)
    print(f"r={r:.2f}  split_half_cos={s:.3f}", flush=True)

print("\ncosine between directions at different mask ratios:")
print("        " + "".join(f"r{r:<6.2f}" for r in ratios))
for i, r in enumerate(ratios):
    row = "".join(f"{torch.nn.functional.cosine_similarity(vs[i], vs[j], dim=0).item():<7.3f}" for j in range(len(ratios)))
    print(f"r{r:<6.2f}{row}")
print("\nREAD: cos(r0.00, r0.90) near 1 -> fixed vector fine (adaptive-dir pointless). "
      "Well below 1 -> direction is noise-level-dependent -> adaptive-direction is real.")
