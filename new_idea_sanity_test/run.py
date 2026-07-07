#!/usr/bin/env python
"""Convergence-aware steering on LLaDA-8B-Instruct.

Question: does decaying the steering strength as the attribute CONVERGES let us
reach the same steered output in FEWER denoising steps than constant (uniform)
steering? Efficiency of STEERING, not answer correctness.

Conditions (identical everything else):
  none    : lambda_t = 0                    (reference, all steps)
  uniform : lambda_t = lambda0              (standard steering baseline, all steps)
  decay   : lambda_t = lambda0*(1 - c_t)    (ablation: no over-steer, all steps)
  conv    : lambda_t = lambda0*(1 - c_t) + EARLY-COMMIT at c_t>=rho   (the method)

c_t = fraction of GENERATED positions whose argmax has been unchanged for the
last k steps (a passive stability read-out, NOT the top1-top2 gap -> not gameable
by the steering push). Steering vector v_c is added to the block-L residual on
generated positions only. Everything runs at temperature 0 (deterministic).

Output: convergence_steering/results/{attr}_{cond}.jsonl  (one row/prompt:
prompt, text, nfe = #forward passes used, gen_length, condition, attr).
"""
import argparse, json, os, torch
from transformers import AutoModel, AutoTokenizer

MODEL = "/home/lukas/users/shashmi/dlm_bias/LLaDA-8B-Instruct"
MASK_ID = 126336
HERE = os.path.dirname(os.path.abspath(__file__))

# ---- contrastive pairs (templated, topic-matched) to BUILD the direction ----
_TOPICS = ["movie", "meal", "trip", "book", "concert", "hotel", "game", "service",
           "gift", "day", "class", "phone", "car", "show", "party", "job",
           "restaurant", "app", "flight", "neighborhood"]
_SENT = {
    "pos": ["I absolutely loved the {x}; it was wonderful and made me so happy.",
            "The {x} was fantastic, easily the best I have ever experienced.",
            "What a delightful {x} — I enjoyed every single moment of it."],
    "neg": ["I absolutely hated the {x}; it was awful and made me miserable.",
            "The {x} was terrible, easily the worst I have ever experienced.",
            "What a dreadful {x} — I disliked every single moment of it."],
}
_FORM = {
    "pos": ["I would like to formally request your assistance regarding the {x}.",
            "Please be advised that the {x} requires your immediate attention.",
            "I am writing to kindly inquire about the status of the {x}."],
    "neg": ["hey can u help me out with the {x}? kinda stuck lol",
            "yo the {x} is a total mess, no clue whats going on tbh",
            "omg the {x} thing is so annoying, ugh whatever"],
}
# for formality, "pos" = formal (the steered target), "neg" = casual

def pairs(attr):
    d = _SENT if attr == "sentiment" else _FORM
    pos = [t.format(x=x) for x in _TOPICS for t in d["pos"]]
    neg = [t.format(x=x) for x in _TOPICS for t in d["neg"]]
    return pos, neg

# ---- neutral generation prompts (tone is what steering changes) ----
_GEN_TOPICS = ["your weekend", "the new coffee shop downtown", "a movie you saw",
    "the weather this week", "your morning commute", "a book you finished",
    "the office lunch", "a walk in the park", "your favorite hobby", "the local gym",
    "a recent phone call", "the neighborhood", "a cooking attempt", "your desk setup",
    "a train ride", "the grocery store", "a work meeting", "a video game",
    "the city library", "a rainy afternoon", "your running routine", "a new restaurant",
    "the bus schedule", "a home repair"]
def gen_prompts(n):
    ps = [f"Write two or three sentences about {t}." for t in _GEN_TOPICS]
    return ps[:n]

# --------------------------------------------------------------------------- #
def resolve_blocks(model):
    for path in ("model.transformer.blocks", "transformer.blocks"):
        obj = model
        ok = True
        for p in path.split("."):
            if hasattr(obj, p): obj = getattr(obj, p)
            else: ok = False; break
        if ok: return obj
    raise RuntimeError("could not locate transformer.blocks")

def masked_mean(h, ids, pad_id):
    m = (ids != pad_id).unsqueeze(-1).to(h.dtype)  # (1,T,1)
    return (h * m).sum(1) / m.sum(1).clamp(min=1)

@torch.no_grad()
def build_direction(model, tok, blocks, L, attr, device):
    pos, neg = pairs(attr)
    cap = {}
    h = blocks[L].register_forward_hook(
        lambda m, i, o: cap.__setitem__("h", (o[0] if isinstance(o, tuple) else o).detach()))
    def rep(sents):
        out, norms = [], []
        for s in sents:
            ids = torch.tensor(tok(s)["input_ids"], device=device).unsqueeze(0)
            model(ids)
            hs = cap["h"].float()                      # (1,T,H)
            out.append(masked_mean(hs, ids, tok.pad_token_id or -1).squeeze(0))
            norms.append(hs.norm(dim=-1).mean().item())
        return torch.stack(out), sum(norms)/len(norms)
    P, npos = rep(pos); N, nneg = rep(neg)
    h.remove()
    d = (P.mean(0) - N.mean(0))
    vc = d / d.norm()
    # split-half cosine (deterministic halves)
    def half(M, idx): return M[idx].mean(0)
    ip = list(range(len(P))); ineg = list(range(len(N)))
    a = (half(P, ip[::2]) - half(N, ineg[::2]))
    b = (half(P, ip[1::2]) - half(N, ineg[1::2]))
    shc = torch.nn.functional.cosine_similarity(a, b, dim=0).item()
    act_norm = (npos + nneg) / 2
    return vc, shc, act_norm, P.mean(0), N.mean(0)

# --------------------------------------------------------------------------- #
@torch.no_grad()
def generate(model, blocks, L, prompt, gen_length, block_length, steps,
             cond, vc, lam0, k, rho, device, cstar=0.0, tau=0.5, beta=0.0,
             mu=None, slerp_t=0.2, mask=None, delta=4.0):
    if cond == "sadidir" and mask is not None:            # SADI dim-mask + directional additive
        vm = vc * mask; vc = vm / vm.norm().clamp(min=1e-6)
    mode = ("sadi" if cond == "sadi" else "sphere" if cond == "sphere"
            else "nadd" if cond == "nadd"
            else "clamp" if cond in ("clamp", "clampconf", "cmom") else "add")
    momentum = beta if cond in ("amom", "cmom") else 0.0
    st = {"lam": 0.0, "vc": vc.to(device), "g0": prompt.shape[1],
          "mode": mode, "cstar": cstar, "gate": cond == "clampconf",
          "tau": tau, "gap": torch.zeros(gen_length, device=device),
          "beta": momentum, "vel": None,
          "mu": (mu.to(device) if mu is not None else None), "t": slerp_t,
          "mask": (mask.view(1, 1, -1).to(device) if mask is not None else None), "delta": delta}
    def steer(m, i, o):
        h = o[0] if isinstance(o, tuple) else o
        vh = st["vc"].to(h.dtype)
        g = h[:, st["g0"]:, :]
        if st["mode"] == "sadi":                               # SADI eq5: A' = A + delta*(A ⊙ M)
            g = g + st["delta"] * (g * st["mask"].to(g.dtype))
            h[:, st["g0"]:, :] = g
            return (h,) + tuple(o[1:]) if isinstance(o, tuple) else h
        if st["mode"] == "nadd":                               # norm-PRESERVING additive
            g2 = g + st["lam"] * vh                            # additive move
            nrm = g.norm(dim=-1, keepdim=True)
            g = nrm * g2 / g2.norm(dim=-1, keepdim=True).clamp(min=1e-6)  # rescale to original norm
            h[:, st["g0"]:, :] = g
            return (h,) + tuple(o[1:]) if isinstance(o, tuple) else h
        if st["mode"] == "sphere":                             # norm-preserving Slerp rotation
            mu = st["mu"].to(h.dtype)                          # (H,) unit target-class direction
            nrm = g.norm(dim=-1, keepdim=True)                 # (1,Tg,1) preserve this
            hhat = g / nrm.clamp(min=1e-6)
            cos = (hhat * mu).sum(-1, keepdim=True).clamp(-0.9999, 0.9999)
            th = torch.arccos(cos)                             # angle to target (1,Tg,1)
            s = torch.sin(th).clamp(min=1e-6)
            t = st["t"]
            hrot = (torch.sin((1 - t) * th) / s) * hhat + (torch.sin(t * th) / s) * mu
            g = nrm * hrot                                     # rotate, keep original norm
            h[:, st["g0"]:, :] = g
            return (h,) + tuple(o[1:]) if isinstance(o, tuple) else h
        if st["mode"] == "clamp":
            a = (g * vh).sum(-1, keepdim=True)                 # current projection (1,Tg,1)
            base = st["cstar"] - a                             # deficit to target
            if st["gate"]:                                     # per-position confidence coupling
                gate = (st["gap"] < st["tau"]).to(g.dtype).unsqueeze(0).unsqueeze(-1)
                base = base * gate                             # correct only undecided positions
        else:
            if st["lam"] == 0.0 and st["beta"] == 0.0: return o
            base = torch.full((1, g.shape[1], 1), st["lam"], device=g.device, dtype=g.dtype)
        if st["beta"] > 0.0:                                   # momentum: EMA (bounded, steady-state = base)
            b = st["beta"]
            st["vel"] = base if st["vel"] is None else b * st["vel"] + (1 - b) * base
            corr = st["vel"]
        else:
            corr = base
        g = g + corr * vh
        h[:, st["g0"]:, :] = g
        return (h,) + tuple(o[1:]) if isinstance(o, tuple) else h
    handle = blocks[L].register_forward_hook(steer)

    x = torch.full((1, prompt.shape[1] + gen_length), MASK_ID, dtype=torch.long, device=device)
    x[:, :prompt.shape[1]] = prompt.clone()
    g0 = prompt.shape[1]
    num_blocks = gen_length // block_length
    steps_per_block = steps // num_blocks

    prev = None; streak = torch.zeros(gen_length, dtype=torch.long, device=device)
    nfe = 0; ct = 0.0; gprev = 0.0; committed = False

    def num_transfer(mask_blk, spb):
        n = mask_blk.sum(); base = n // spb; rem = n % spb
        arr = torch.zeros(spb, dtype=torch.long, device=device) + base
        arr[:rem] += 1
        return arr

    for nb in range(num_blocks):
        if committed: break
        b0 = g0 + nb * block_length; b1 = g0 + (nb + 1) * block_length
        nt = num_transfer(x[:, b0:b1] == MASK_ID, steps_per_block)
        for i in range(steps_per_block):
            # set lambda for THIS step from current convergence c_t
            if cond in ("clamp", "clampconf", "cmom", "sphere", "sadi"): pass  # handled inside the hook
            elif cond == "none": st["lam"] = 0.0
            elif cond in ("uniform", "amom", "nadd", "sadidir"): st["lam"] = lam0
            elif cond == "warmup": st["lam"] = lam0 * gprev   # steer MORE as confidence gap grows
            else: st["lam"] = lam0 * (1.0 - ct)          # decay / conv
            logits = model(x).logits; nfe += 1
            x0 = torch.argmax(logits, dim=-1)            # (1, T), temp 0
            # confidence gap G_t = mean(p_top1 - p_top2) over generated region (for warmup)
            p2 = torch.softmax(logits[0, g0:].float(), dim=-1)
            top2 = torch.topk(p2, 2, dim=-1).values
            gap_pos = top2[:, 0] - top2[:, 1]                 # per-position confidence gap
            gprev = gap_pos.mean().item()
            st["gap"] = gap_pos                               # for clampconf gating next step
            # ---- stability read-out over generated region ----
            cur = x0[0, g0:]
            if prev is not None:
                same = cur == prev
                streak = torch.where(same, streak + 1, torch.zeros_like(streak))
            prev = cur.clone()
            ct = (streak >= k).float().mean().item()
            # ---- early commit (conv only) ----
            if cond == "conv" and ct >= rho:
                mask_index = x == MASK_ID
                x[mask_index] = x0[mask_index]
                committed = True
                break
            # ---- standard low-confidence unmasking within block ----
            mask_index = x == MASK_ID
            p = torch.softmax(logits.float(), dim=-1)
            x0p = torch.gather(p, -1, x0.unsqueeze(-1)).squeeze(-1)
            x0p[:, b1:] = float("-inf")
            conf = torch.where(mask_index, x0p, torch.full_like(x0p, float("-inf")))
            x0 = torch.where(mask_index, x0, x)
            sel = torch.topk(conf[0], k=int(nt[i]))[1]
            x[0, sel] = x0[0, sel]
    handle.remove()
    return x, nfe

# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attr", required=True, choices=["sentiment", "formality"])
    ap.add_argument("--conditions", default="none,uniform,decay,conv")
    ap.add_argument("--layer", type=int, default=14)
    ap.add_argument("--n-prompts", type=int, default=24)
    ap.add_argument("--gen-length", type=int, default=64)
    ap.add_argument("--block-length", type=int, default=32)
    ap.add_argument("--steps", type=int, default=64)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--rho", type=float, default=0.9)
    ap.add_argument("--lam-frac", type=float, default=0.15,
                    help="lambda0 = lam_frac * mean L14 activation norm")
    ap.add_argument("--sign", type=float, default=1.0,
                    help="steer toward +target (1.0) or away (-1.0)")
    ap.add_argument("--clamp-target", type=float, default=0.0,
                    help="override clamp c* (0 = use natural build-example level)")
    ap.add_argument("--tau", type=float, default=0.5,
                    help="clampconf: only correct positions with confidence gap < tau (still undecided)")
    ap.add_argument("--momentum", type=float, default=0.8,
                    help="momentum beta for amom/cmom conditions (velocity accumulation)")
    ap.add_argument("--slerp-t", type=float, default=0.2,
                    help="sphere: geodesic rotation fraction toward target-class mean (0..1)")
    ap.add_argument("--sadi-topk", type=float, default=0.1,
                    help="SADI: fraction of critical dims kept in the binary mask")
    ap.add_argument("--sadi-delta", type=float, default=4.0,
                    help="SADI (eq5): A' = A + delta*(A ⊙ M) amplification strength")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    outdir = os.path.join(HERE, "results"); os.makedirs(outdir, exist_ok=True)
    print(f"[{args.attr}] loading LLaDA ...", flush=True)
    model = AutoModel.from_pretrained(MODEL, trust_remote_code=True,
                                      torch_dtype=torch.bfloat16).to(args.device).eval()
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    blocks = resolve_blocks(model)

    vc, shc, act_norm, Pmean, Nmean = build_direction(model, tok, blocks, args.layer, args.attr, args.device)
    vc = args.sign * vc
    lam0 = args.lam_frac * act_norm
    # clamp target c* = projection of the TARGET class (the one v_hat points toward)
    cstar = max((Pmean @ vc).item(), (Nmean @ vc).item())
    if args.clamp_target > 0: cstar = args.clamp_target
    # sphere: target-class mean DIRECTION (unit) to rotate toward (neg if sign<0 else pos)
    mu_raw = Nmean if args.sign < 0 else Pmean
    mu = mu_raw / mu_raw.norm()
    # SADI eq3: binary mask = top-K dims by |mean(pos)-mean(neg)| (behavior-critical dims)
    d_sel = (Pmean - Nmean).abs()
    kdim = max(1, int(args.sadi_topk * d_sel.numel()))
    mask = torch.zeros_like(d_sel); mask[torch.topk(d_sel, kdim).indices] = 1.0
    print(f"[{args.attr}] direction: split_half_cos={shc:.3f}  act_norm~{act_norm:.1f}  "
          f"lambda0={lam0:.2f}  cstar={cstar:.2f}  (layer L{args.layer})", flush=True)
    torch.save({"vc": vc.cpu(), "split_half_cos": shc, "act_norm": act_norm,
                "lam0": lam0, "layer": args.layer, "attr": args.attr},
               os.path.join(outdir, f"{args.attr}_direction.pt"))

    prompts = gen_prompts(2 if args.smoke else args.n_prompts)
    conds = args.conditions.split(",")
    for cond in conds:
        rows = []
        for j, pr in enumerate(prompts):
            ptext = tok.apply_chat_template([{"role": "user", "content": pr}],
                                            add_generation_prompt=True, tokenize=False)
            ids = torch.tensor(tok(ptext)["input_ids"], device=args.device).unsqueeze(0)
            out, nfe = generate(model, blocks, args.layer, ids, args.gen_length,
                                args.block_length, args.steps, cond, vc, lam0,
                                args.k, args.rho, args.device, cstar, args.tau, args.momentum,
                                mu, args.slerp_t, mask, args.sadi_delta)
            text = tok.batch_decode(out[:, ids.shape[1]:], skip_special_tokens=True)[0].strip()
            rows.append({"attr": args.attr, "cond": cond, "prompt": pr, "text": text,
                         "nfe": nfe, "steps": args.steps, "gen_length": args.gen_length})
            if args.smoke:
                print(f"\n[{cond}] nfe={nfe}/{args.steps}\n  {pr}\n  -> {text}", flush=True)
        if not args.smoke:
            fp = os.path.join(outdir, f"{args.attr}_{cond}.jsonl")
            with open(fp, "w") as f:
                for r in rows: f.write(json.dumps(r) + "\n")
            avg_nfe = sum(r["nfe"] for r in rows) / len(rows)
            print(f"[{args.attr}] {cond}: {len(rows)} rows, avg_nfe={avg_nfe:.1f} -> {fp}", flush=True)

if __name__ == "__main__":
    main()
