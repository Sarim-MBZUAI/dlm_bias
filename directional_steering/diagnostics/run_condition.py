#!/usr/bin/env python
"""STEP 3 (PID layer-depth study). Run one or more steering conditions on the
400-item BBQ Black-referent set and write a PERSISTENT per-condition results json.

Conditions (--conds, comma list):
  clean          no hook (alpha=0 baseline)
  L14_open_a8    single hook at block 14, RAW race_black_anchored_text.pt, add, alpha=8
  layerwise_P    per-block add u[l]=Kp*rhat[l]
  layerwise_PI   per-block add u[l]=Kp*rhat[l] + Ki*sum_{j<l} rhat[j]
  layerwise_PID  per-block add u[l]=Kp*rhat[l] + Ki*sum_{j<l} rhat[j] + Kd*(rhat[l]-rhat[l-1])

Each finished condition writes cond_<name>.json (metrics + raw counts + config) and
cond_<name>_samples.jsonl to the diagnostics dir, so results survive a crash of any
sibling job. Reuses bbq_eval.generate / build_prompt / parse_letter verbatim.
"""
import argparse
import json
import os
import sys
import time

import torch
from transformers import AutoModel, AutoTokenizer

ROOT = "/home/lukas/users/shashmi/dlm_bias"
DIAG = os.path.join(ROOT, "directional_steering", "diagnostics")
sys.path.insert(0, os.path.join(ROOT, "eval"))
import bbq_eval

MODEL_PATH = os.path.join(ROOT, "LLaDA-8B-Instruct")
SWEEP400 = os.path.join(ROOT, "experiments", "data", "_sweep400.jsonl")
ARROWS = os.path.join(DIAG, "perlayer_arrows.pt")
WINNER = os.path.join(ROOT, "directional_steering", "race_black_anchored_text.pt")
BLACK_TAGS = {"black", "african american", "f-black", "m-black", "african"}
N_LAYERS = 32

# gains (Step 2 defaults; overridable)
KP, KI, KD = 2.0, 0.05, 0.02
L14_ALPHA = 8.0


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


class AddVec:
    """Add a constant vector to a block/wte output hidden state (all positions)."""
    def __init__(self, vec):
        self.vec = vec
        self.h = None

    def _hook(self, mod, inp, out):
        hidden = bbq_eval.hidden_from_output(out)
        steer = self.vec.to(hidden.dtype).to(hidden.device)
        return bbq_eval.output_with_hidden(out, hidden + steer)

    def attach(self, module):
        self.h = module.register_forward_hook(self._hook)

    def detach(self):
        if self.h is not None:
            self.h.remove()
            self.h = None


def build_layerwise_vectors(r, kp, ki, kd):
    """r: (32,H). Returns dict cond -> list[32] of injection vectors (float32)."""
    rhat = torch.stack([r[l] / r[l].norm().clamp(min=1e-12) for l in range(N_LAYERS)], 0)
    cum = torch.zeros_like(rhat[0])
    P, PI, PID = [], [], []
    for l in range(N_LAYERS):
        p = kp * rhat[l]
        integ = ki * cum.clone()                       # sum_{j<l} rhat[j]
        deriv = kd * (rhat[l] - (rhat[l - 1] if l > 0 else torch.zeros_like(rhat[l])))
        P.append(p)
        PI.append(p + integ)
        PID.append(p + integ + deriv)
        cum = cum + rhat[l]
    return {"layerwise_P": P, "layerwise_PI": PI, "layerwise_PID": PID}


def attach_condition(model, cond, layerwise, winner_dir):
    steerers = []
    if cond == "clean":
        return steerers
    if cond == "L14_open_a8":
        s = AddVec(L14_ALPHA * winner_dir)             # raw direction * alpha
        s.attach(bbq_eval.resolve_module(model, bbq_eval.BLOCKS_PATH)[14])
        steerers.append(s)
        return steerers
    # layerwise_*
    vecs = layerwise[cond]
    blocks = bbq_eval.resolve_module(model, bbq_eval.BLOCKS_PATH)
    for l in range(N_LAYERS):
        s = AddVec(vecs[l])
        s.attach(blocks[l])
        steerers.append(s)
    return steerers


def run_one(cond, rows, model, tok, layerwise, winner_dir, gen_len, steps, blk, limit):
    steerers = attach_condition(model, cond, layerwise, winner_dir)
    use_rows = rows[:limit] if limit else rows
    per_item = []
    counts = {"black": 0, "nonblack": 0, "abstain": 0, "unparseable": 0}
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
            per_item.append({
                "example_id": int(row.get("example_id", -1)),
                "question_index": str(row.get("question_index", "")),
                "polarity": row.get("question_polarity"),
                "black_idx": bidx, "unk_idx": uidx, "pred_index": pred,
                "pred_letter": letter, "model_output": gen, "pred_class": cls,
            })
            if (idx + 1) % 50 == 0:
                print(f"[{cond}] {idx+1}/{len(use_rows)} "
                      f"b={counts['black']} nb={counts['nonblack']} "
                      f"ab={counts['abstain']} un={counts['unparseable']} "
                      f"({time.time()-t0:.0f}s)", flush=True)
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
    result = {"condition": cond, "n": n, "counts": counts, "rates": rates,
              "gen_length": gen_len, "steps": steps, "block_length": blk,
              "gains": {"Kp": KP, "Ki": KI, "Kd": KD, "L14_alpha": L14_ALPHA},
              "elapsed_s": time.time() - t0}
    outp = os.path.join(DIAG, f"cond_{cond}.json")
    with open(outp, "w") as f:
        json.dump(result, f, indent=2)
    with open(os.path.join(DIAG, f"cond_{cond}_samples.jsonl"), "w") as f:
        for it in per_item:
            f.write(json.dumps(it) + "\n")
    print(f"[{cond}] DONE {rates} -> {outp}", flush=True)
    return result


def main():
    global KP, KI, KD
    ap = argparse.ArgumentParser()
    ap.add_argument("--conds", required=True, help="comma list of condition names")
    ap.add_argument("--limit", type=int, default=0, help="0 = all 400 (else first N, calibration)")
    ap.add_argument("--gen-length", type=int, default=32)
    ap.add_argument("--steps", type=int, default=64)
    ap.add_argument("--block-length", type=int, default=32)
    ap.add_argument("--kp", type=float, default=KP)
    ap.add_argument("--ki", type=float, default=KI)
    ap.add_argument("--kd", type=float, default=KD)
    args = ap.parse_args()

    KP, KI, KD = args.kp, args.ki, args.kd

    print(f"[worker] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')} "
          f"conds={args.conds} limit={args.limit} gains Kp={KP} Ki={KI} Kd={KD}", flush=True)

    rows = [json.loads(l) for l in open(SWEEP400) if l.strip()]
    arrows = torch.load(ARROWS, map_location="cpu")
    r = arrows["r"].to(torch.float32)
    layerwise = build_layerwise_vectors(r, KP, KI, KD)
    winner = torch.load(WINNER, map_location="cpu")["direction"].to(torch.float32)

    model = (AutoModel.from_pretrained(MODEL_PATH, trust_remote_code=True,
                                       torch_dtype=torch.bfloat16).to("cuda").eval())
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    print(f"[worker] model on {model.device}", flush=True)

    for cond in args.conds.split(","):
        cond = cond.strip()
        if not cond:
            continue
        run_one(cond, rows, model, tok, layerwise, winner,
                args.gen_length, args.steps, args.block_length, args.limit)
    print("[worker] ALL DONE", flush=True)


if __name__ == "__main__":
    main()
