#!/usr/bin/env python
"""PID-Steering (arXiv:2510.04309) ported to LLaDA-8B-Instruct, control axis = LAYER DEPTH.

Paper Eq. 18. For each transformer block k = 0..31, with per-layer diff-in-means
arrow r(k) and unit direction rhat(k) = r(k)/||r(k)||:

    u(k) = Kp * rhat(k)
         + Ki * sum_{j=0}^{k-1} rhat(j)          # integral over shallower layers
         + Kd * ( rhat(k) - rhat(k-1) )          # derivative,  rhat(-1) := 0

The residual stream is steered at EVERY block k, at EVERY denoising step:

    hidden  <-  hidden + alpha * u(k)

alpha is the single intervention-strength knob. Conditions differ ONLY in gains:
    base = no hook           P = (Ki=0,  Kd=0)
    PI   = (Ki=0.05, Kd=0)   PID = (Ki=0.05, Kd=0.02)     [Kp=1.0 throughout]

This is ALL 32 layers -- no single-layer anchor, no noise term, no +0.1 offset
(those were the reference repo's steady-state-error demo hacks, NOT the method).

Porting decisions for the masked-diffusion LM: (a) inject at all 32 blocks on every
denoising step (the hook fires once per model forward, and generate() calls the
model steps*num_blocks times); (b) unit-normalize each layer's arrow independently;
(c) arrows are built by answer-text-span pooling (see build_arrows.py).

MODES
    --selftest            offline PID-math check (no GPU, no model); prints PASS/FAIL.
    --cond {base,P,PI,PID} --alpha A  [--limit N] [--dummy-arrows]  run the eval.
Reuses bbq_eval.generate / build_prompt / parse_letter verbatim.
"""
import argparse
import json
import os
import sys
import time

import torch

ROOT = "/home/lukas/users/shashmi/dlm_bias"
sys.path.insert(0, os.path.join(ROOT, "eval"))
import bbq_eval  # noqa: E402

MODEL_PATH = os.path.join(ROOT, "LLaDA-8B-Instruct")
SWEEP400 = os.path.join(ROOT, "experiments", "data", "_sweep400.jsonl")

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_ARROWS = os.path.join(HERE, "arrows.pt")
RESULTS = os.path.join(HERE, "results")

N_LAYERS = 32
BLACK_TAGS = {"black", "african american", "f-black", "m-black", "african"}

# Paper reference gains. Conditions select which terms are active.
GAINS = {
    "base": None,                       # no hook
    "P":   (1.0, 0.0,  0.0),
    "PI":  (1.0, 0.05, 0.0),
    "PID": (1.0, 0.05, 0.02),
}


# --------------------------------------------------------------------------- #
# Core PID combine (Eq. 18).  rhat: (N,H) ALREADY unit-normed per layer.
# --------------------------------------------------------------------------- #
def build_u(rhat, kp, ki, kd):
    """Return u: (N,H), u(k)=Kp*rhat(k)+Ki*sum_{j<k}rhat(j)+Kd*(rhat(k)-rhat(k-1))."""
    n = rhat.shape[0]
    cum = torch.zeros_like(rhat[0])        # sum_{j<k} rhat(j)
    U = []
    for k in range(n):
        p = kp * rhat[k]
        integ = ki * cum.clone()
        # Reference-code boundary (llama_many_layers.py: shifted_ref_dir[0]=ref_dir[0]):
        # rhat(-1) := rhat(0) so the derivative term is ZERO at k=0.
        deriv = kd * (rhat[k] - (rhat[k - 1] if k > 0 else rhat[k]))
        U.append(p + integ + deriv)
        cum = cum + rhat[k]
    return torch.stack(U, 0)


def unit_rows(r, eps=1e-12):
    """Unit-normalize each row (layer) of r: (N,H)."""
    return torch.stack([r[k] / r[k].norm().clamp(min=eps) for k in range(r.shape[0])], 0)


def build_injection(r, kp, ki, kd, alpha):
    """Full per-block injection vectors alpha*u(k) from RAW arrows r: (N,H)."""
    return alpha * build_u(unit_rows(r), kp, ki, kd)


# --------------------------------------------------------------------------- #
# Hook: add a fixed per-block vector to the residual stream (all positions,
# every denoising step). Fires once per model forward; counts invocations so the
# smoke test can verify all 32 blocks are steered.
# --------------------------------------------------------------------------- #
class AddVec:
    def __init__(self, li, vec, fired):
        self.li = li
        self.vec = vec
        self.fired = fired      # shared dict {li: count}
        self.h = None

    def _hook(self, mod, inp, out):
        hidden = bbq_eval.hidden_from_output(out)
        self.fired[self.li] = self.fired.get(self.li, 0) + 1
        steer = self.vec.to(hidden.dtype).to(hidden.device)
        return bbq_eval.output_with_hidden(out, hidden + steer)

    def attach(self, module):
        self.h = module.register_forward_hook(self._hook)

    def detach(self):
        if self.h is not None:
            self.h.remove()
            self.h = None


def black_idx_of(row):
    for k in range(3):
        p = row["answer_info"].get(f"ans{k}")
        if p and str(p[-1]).strip().lower() in BLACK_TAGS:
            return k
    return None


def unk_idx_of(row):
    for k in range(3):
        p = row["answer_info"].get(f"ans{k}")
        if p and str(p[-1]).strip().lower() == "unknown":
            return k
    return None


# --------------------------------------------------------------------------- #
# Offline PID-math self-test (no GPU, no model).
# --------------------------------------------------------------------------- #
def selftest():
    torch.manual_seed(0)
    N, H = 7, 5
    rhat = torch.randn(N, H)  # synthetic per-layer unit dirs (values arbitrary; math only)

    # Reference: explicit exclusive cumsum + first difference.
    cumexcl = torch.zeros(N, H)
    for k in range(1, N):
        cumexcl[k] = cumexcl[k - 1] + rhat[k - 1]
    diff = torch.zeros(N, H)  # diff[0]=0 (rhat(-1):=rhat(0), matches reference code)
    for k in range(1, N):
        diff[k] = rhat[k] - rhat[k - 1]

    ok_all = True
    for name, (kp, ki, kd) in {"P": GAINS["P"], "PI": GAINS["PI"], "PID": GAINS["PID"]}.items():
        u = build_u(rhat, kp, ki, kd)
        expect = kp * rhat + ki * cumexcl + kd * diff
        p_ok = torch.allclose(u, expect, atol=1e-6)
        ok_all &= p_ok
        print(f"[selftest] {name:3s} (Kp={kp},Ki={ki},Kd={kd})  full formula: "
              f"{'PASS' if p_ok else 'FAIL'}")

    # Term isolation: P<-base, I<-(PI-P), D<-(PID-PI).
    uP = build_u(rhat, *GAINS["P"])
    uPI = build_u(rhat, *GAINS["PI"])
    uPID = build_u(rhat, *GAINS["PID"])
    p_term = torch.allclose(uP, GAINS["P"][0] * rhat, atol=1e-6)
    i_term = torch.allclose(uPI - uP, GAINS["PI"][1] * cumexcl, atol=1e-6)
    d_term = torch.allclose(uPID - uPI, GAINS["PID"][2] * diff, atol=1e-6)
    print(f"[selftest] P term  == Kp*rhat            : {'PASS' if p_term else 'FAIL'}")
    print(f"[selftest] I term  == Ki*cumsum_excl     : {'PASS' if i_term else 'FAIL'}")
    print(f"[selftest] D term  == Kd*(rhat_k-rhat_k-1): {'PASS' if d_term else 'FAIL'}")

    # Reference boundary: deriv(0)=0 and integral empty -> u(0)==Kp*rhat(0).
    b_ok = torch.allclose(uPID[0], GAINS["PID"][0] * rhat[0], atol=1e-6)
    print(f"[selftest] k=0 boundary  u(0)==Kp*rhat(0)      : {'PASS' if b_ok else 'FAIL'}")

    ok_all &= p_term and i_term and d_term and b_ok
    print(f"[selftest] OVERALL: {'PASS' if ok_all else 'FAIL'}")
    return ok_all


# --------------------------------------------------------------------------- #
# Eval runner.
# --------------------------------------------------------------------------- #
def run(cond, alpha, arrows_path, limit, gen_len, steps, blk, dummy_arrows, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    kp = ki = kd = 0.0
    if cond != "base":
        kp, ki, kd = GAINS[cond]

    # arrows -> per-block injection vectors (skip for base / dummy handled below).
    if dummy_arrows:
        torch.manual_seed(1234)
        r = torch.randn(N_LAYERS, 4096, dtype=torch.float32)
        arrows_meta = {"n_items": -1, "source": "DUMMY_RANDOM_SMOKE", "per_layer_raw_norm": None}
    else:
        blob = torch.load(arrows_path, map_location="cpu")
        r = blob["r"].to(torch.float32)
        arrows_meta = {"n_items": blob.get("n_items"), "source": blob.get("source"),
                       "per_layer_raw_norm": blob.get("per_layer_raw_norm")}
    inject = None if cond == "base" else build_injection(r, kp, ki, kd, alpha)

    rows = [json.loads(l) for l in open(SWEEP400) if l.strip()]
    use_rows = rows[:limit] if limit else rows

    from transformers import AutoModel, AutoTokenizer
    model = (AutoModel.from_pretrained(MODEL_PATH, trust_remote_code=True,
                                       torch_dtype=torch.bfloat16).to("cuda").eval())
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    print(f"[{cond}] model on {model.device} | CUDA_VISIBLE_DEVICES="
          f"{os.environ.get('CUDA_VISIBLE_DEVICES')} | alpha={alpha} "
          f"Kp={kp} Ki={ki} Kd={kd} | n={len(use_rows)} dummy={dummy_arrows}", flush=True)

    fired = {}
    steerers = []
    if cond != "base":
        blocks = bbq_eval.resolve_module(model, bbq_eval.BLOCKS_PATH)
        assert len(blocks) == N_LAYERS, f"expected {N_LAYERS} blocks, got {len(blocks)}"
        for li in range(N_LAYERS):
            s = AddVec(li, inject[li], fired)
            s.attach(blocks[li])
            steerers.append(s)

    per_item = []
    counts = {"black": 0, "nonblack": 0, "abstain": 0, "unparseable": 0}
    dis_n = dis_correct = 0
    t0 = time.time()
    try:
        for idx, row in enumerate(use_rows):
            prompt = bbq_eval.build_prompt(row)
            ptxt = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                           add_generation_prompt=True, tokenize=False)
            ids = torch.tensor(tok(ptxt)["input_ids"], device=model.device).unsqueeze(0)
            out = bbq_eval.generate(model, ids, steps=steps, gen_length=gen_len,
                                    block_length=blk, temperature=0.0, cfg_scale=0.0,
                                    remasking="low_confidence")
            gen = tok.batch_decode(out[:, ids.shape[1]:], skip_special_tokens=True)[0].strip()
            letter = bbq_eval.parse_letter(gen, row)
            pred = bbq_eval.LETTERS.index(letter) if letter else None
            bidx = black_idx_of(row)
            uidx = unk_idx_of(row)
            if pred is None:
                cls = "unparseable"; counts["unparseable"] += 1
            elif pred == bidx:
                cls = "black"; counts["black"] += 1
            elif pred == uidx:
                cls = "abstain"; counts["abstain"] += 1
            else:
                cls = "nonblack"; counts["nonblack"] += 1
            if row.get("context_condition") == "disambig" and pred is not None:
                dis_n += 1
                dis_correct += int(pred == int(row["label"]))
            per_item.append({
                "example_id": int(row.get("example_id", -1)),
                "question_index": str(row.get("question_index", "")),
                "polarity": row.get("question_polarity"),
                "context_condition": row.get("context_condition"),
                "black_idx": bidx, "unk_idx": uidx, "pred_index": pred,
                "pred_letter": letter, "model_output": gen, "pred_class": cls,
            })
            if (idx + 1) % 50 == 0:
                print(f"[{cond}] {idx+1}/{len(use_rows)} b={counts['black']} "
                      f"nb={counts['nonblack']} ab={counts['abstain']} "
                      f"un={counts['unparseable']} ({time.time()-t0:.0f}s)", flush=True)
    finally:
        for s in steerers:
            s.detach()

    n = len(use_rows)
    rates = {
        "black_pick_rate": counts["black"] / n,
        "nonblack_pick_rate": counts["nonblack"] / n,
        "abstain_rate": counts["abstain"] / n,
        "unparseable_rate": counts["unparseable"] / n,
    }
    acc_disambig = (dis_correct / dis_n) if dis_n else None
    n_fired = len(fired)
    fire_counts = sorted(set(fired.values())) if fired else []
    result = {
        "condition": cond, "alpha": alpha, "n": n, "counts": counts, "rates": rates,
        "acc_disambig": acc_disambig, "n_disambig": dis_n,
        "gains": {"Kp": kp, "Ki": ki, "Kd": kd},
        "gen_length": gen_len, "steps": steps, "block_length": blk,
        "temperature": 0.0, "dummy_arrows": bool(dummy_arrows),
        "hooks_fired_layers": n_fired, "hook_fire_counts": fire_counts,
        "arrows": arrows_meta, "elapsed_s": time.time() - t0,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    tag = f"{cond}" if cond == "base" else f"{cond}_a{alpha:g}".replace(".", "p")
    outp = os.path.join(out_dir, f"cond_{tag}.json")
    with open(outp, "w") as f:
        json.dump(result, f, indent=2)
    with open(os.path.join(out_dir, f"cond_{tag}_samples.jsonl"), "w") as f:
        for it in per_item:
            f.write(json.dumps(it) + "\n")
    print(f"[{cond}] DONE rates={rates} acc_dis={acc_disambig} "
          f"hooks_fired_layers={n_fired}/{N_LAYERS} fire_counts={fire_counts} -> {outp}",
          flush=True)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true", help="offline PID-math check; no GPU")
    ap.add_argument("--cond", choices=list(GAINS.keys()))
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--arrows", default=DEFAULT_ARROWS)
    ap.add_argument("--limit", type=int, default=0, help="0 = all 400")
    ap.add_argument("--gen-length", type=int, default=32)
    ap.add_argument("--steps", type=int, default=64)
    ap.add_argument("--block-length", type=int, default=32)
    ap.add_argument("--dummy-arrows", action="store_true",
                    help="use a random (32,4096) arrow set (GPU smoke test only)")
    ap.add_argument("--out-dir", default=RESULTS)
    args = ap.parse_args()

    if args.selftest:
        ok = selftest()
        sys.exit(0 if ok else 1)

    if not args.cond:
        ap.error("--cond required (or use --selftest)")
    run(args.cond, args.alpha, args.arrows, args.limit, args.gen_length,
        args.steps, args.block_length, args.dummy_arrows, args.out_dir)


if __name__ == "__main__":
    main()
