#!/usr/bin/env python
"""STEP 1 (do this BEFORE any convergence stuff): does the steering vector
actually steer? Sweep lambda for uniform steering and measure the attribute.

- builds v_c at block L (reuses run.py)
- for each lambda, generates for N prompts with UNIFORM steering (all steps,
  single block), scores sentiment with distilbert-sst2, reports mean P(negative)
  and mean output length (crude fluency guard) + one sample.

Success = P(negative) rises monotonically with lambda while text stays coherent
-> we have a working steering vector. If it never moves, the vector/layer is the
problem and there is nothing to make convergence-aware.
"""
import argparse, os, sys, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run as R
from transformers import (AutoModel, AutoTokenizer,
                          AutoModelForSequenceClassification, AutoTokenizer as ATok)

SST2 = "distilbert-base-uncased-finetuned-sst-2-english"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attr", default="sentiment")
    ap.add_argument("--layer", type=int, default=14)
    ap.add_argument("--sign", type=float, default=-1.0)      # toward negative
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--fracs", default="0,0.3,0.6,0.9,1.2")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    print("loading LLaDA ...", flush=True)
    model = AutoModel.from_pretrained(R.MODEL, trust_remote_code=True,
                                      torch_dtype=torch.bfloat16).to(args.device).eval()
    tok = AutoTokenizer.from_pretrained(R.MODEL, trust_remote_code=True)
    blocks = R.resolve_blocks(model)
    vc, shc, act_norm, _, _ = R.build_direction(model, tok, blocks, args.layer, args.attr, args.device)
    vc = args.sign * vc
    print(f"direction: split_half_cos={shc:.3f}  act_norm~{act_norm:.1f}  (steer sign={args.sign})", flush=True)

    print("loading sentiment scorer ...", flush=True)
    sc = AutoModelForSequenceClassification.from_pretrained(SST2).to(args.device).eval()
    stok = ATok.from_pretrained(SST2)
    neg_idx = [i for i, l in sc.config.id2label.items() if l.upper().startswith("NEG")][0]

    @torch.no_grad()
    def p_negative(texts):
        enc = stok(texts, return_tensors="pt", padding=True, truncation=True, max_length=128).to(args.device)
        p = torch.softmax(sc(**enc).logits, -1)[:, neg_idx]
        return p.tolist()

    prompts = R.gen_prompts(args.n)
    print(f"\n{'lambda':>8}{'P(neg)':>9}{'len':>7}   sample[prompt0]")
    for fr in [float(x) for x in args.fracs.split(",")]:
        lam0 = fr * act_norm
        texts = []
        for pr in prompts:
            pt = tok.apply_chat_template([{"role": "user", "content": pr}],
                                         add_generation_prompt=True, tokenize=False)
            ids = torch.tensor(tok(pt)["input_ids"], device=args.device).unsqueeze(0)
            out, _ = R.generate(model, blocks, args.layer, ids, 64, 64, 64,
                                "uniform", vc, lam0, 3, 0.9, args.device)
            texts.append(tok.batch_decode(out[:, ids.shape[1]:], skip_special_tokens=True)[0].strip())
        pn = p_negative(texts)
        mlen = sum(len(t.split()) for t in texts) / len(texts)
        s0 = texts[0][:90].replace("\n", " ")
        print(f"{fr:>8.2f}{sum(pn)/len(pn):>9.3f}{mlen:>7.1f}   {s0}", flush=True)

if __name__ == "__main__":
    main()
