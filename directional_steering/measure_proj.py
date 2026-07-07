#!/usr/bin/env python
"""Calibration check: what is the NATURAL projection <h, v_hat> of L14 activations
onto the anchored Black direction, in the BBQ generation context? This tells us
whether the clamp target c* is sensible (too low -> no-op; way above natural ->
huge push). Also verifies the clamp hook actually changes the projection + output.
"""
import os, sys, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "eval"))
import bbq_eval as B

DEV = "cuda"; L = 14
model = B.AutoModel.from_pretrained(B.DEFAULT_MODEL_PATH, trust_remote_code=True,
                                    torch_dtype=torch.bfloat16).to(DEV).eval()
tok = B.AutoTokenizer.from_pretrained(B.DEFAULT_MODEL_PATH, trust_remote_code=True)
saved = torch.load("directional_steering/race_black_anchored_text.pt", map_location="cpu")
d = saved["direction"].float().to(DEV)
vh = (d / d.norm()).to(DEV)
print(f"direction raw_norm={d.norm():.2f}  (open-loop a=8 adds {8*float(d.norm()):.1f} to projection)")

rows = B.load_bbq("nyu-mll/BBQ (jsonl)", 42, 1000, None)
rows = [r for r in rows if r["context_condition"] == "ambig"][:4]

blocks = B.resolve_module(model, "model.transformer.blocks")
cap = {}
h = blocks[L].register_forward_hook(lambda m,i,o: cap.__setitem__("h",(o[0] if isinstance(o,tuple) else o).detach()))

@torch.no_grad()
def proj_for(row):
    base = B.build_prompt(row)
    pt = tok.apply_chat_template([{"role":"user","content":base}], add_generation_prompt=True, tokenize=False)
    ids = torch.tensor(tok(pt)["input_ids"], device=DEV).unsqueeze(0)
    plen = ids.shape[1]
    x = torch.full((1, plen+32), B.MASK_ID, dtype=torch.long, device=DEV); x[:,:plen]=ids
    model(x)
    hs = cap["h"].float()[0]                      # (T,H)
    p = (hs * vh).sum(-1)                          # (T,) projection per position
    return p[:plen], p[plen:]                      # prompt, gen(mask)

allp=[]; allg=[]
for r in rows:
    pp, pg = proj_for(r); allp.append(pp); allg.append(pg)
h.remove()
import torch as T
pp = T.cat(allp); pg = T.cat(allg)
print(f"\nNATURAL projection <h,v_hat> at L14 (over {len(rows)} ambiguous items):")
print(f"  PROMPT positions: mean={pp.mean():.2f}  min={pp.min():.2f}  max={pp.max():.2f}")
print(f"  GEN(mask) positions: mean={pg.mean():.2f}  min={pg.min():.2f}  max={pg.max():.2f}")
print(f"\nclamp target c*=60 -> deficit vs natural gen ~ {60-float(pg.mean()):.1f} (this is what clamp injects/step)")
print("READ: if natural gen-proj is already ~60, c*=60 is a near no-op; if it's small, c*=60 pushes hard.")
