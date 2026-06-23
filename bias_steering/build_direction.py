#!/usr/bin/env python
"""Build a social/demographic bias STEERING DIRECTION for LLaDA-8B-Instruct.

Training-free port of "Implicit Bias Injection" (IBI) to LLaDA's masked-diffusion
architecture. We compute an INPUT-EMBEDDING-LAYER direction that points from the
anti-stereotype side toward the stereotype side of minimal pairs.

Method (frozen model, no gradients):
  1. Register a forward hook on the token-embedding module (model.transformer.wte
     reached via model.model.transformer.wte for the AutoModel object). The hook
     only CAPTURES the embedding output (B, T, H) -- it does not modify it.
  2. For each (stereotype, anti_stereotype) pair, run a forward pass on each
     sentence, take the masked-mean of the captured embeddings over real tokens
     -> a single (H,) vector per sentence.
  3. direction = mean over pairs of (emb_mean[stereotype] - emb_mean[anti]).

Because LLaDA applies NO embedding scaling, NO additive positional embedding
(RoPE is applied inside attention), and embedding dropout is a no-op at p=0,
the captured wte output is exactly the true input embedding fed into block 0.
The saved norms let you calibrate alpha in bias_llada.py.

NOTE: run this on a GPU; this script invokes the model. The user runs it.
"""
import argparse
import json

import torch
from transformers import AutoModel, AutoTokenizer

DEFAULT_MODEL_PATH = "/home/lukas/users/shashmi/dlm_bias/LLaDA-8B-Instruct"
DEFAULT_PAIRS = "/home/lukas/users/shashmi/dlm_bias/bias_steering/example_pairs.json"
DEFAULT_OUT = "/home/lukas/users/shashmi/dlm_bias/bias_steering/direction.pt"
DEFAULT_HOOK_MODULE = "model.transformer.wte"
PAD_TOKEN_ID = 126081


def resolve_module(model, dotted_path):
    """Resolve a dotted attribute path (e.g. 'model.transformer.wte') on model."""
    obj = model
    for part in dotted_path.split("."):
        obj = getattr(obj, part)
    return obj


def parse_args():
    p = argparse.ArgumentParser(
        description="Build an embedding-layer bias steering direction for LLaDA."
    )
    p.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    p.add_argument("--pairs", default=DEFAULT_PAIRS)
    p.add_argument("--hook-module", default=DEFAULT_HOOK_MODULE)
    p.add_argument("--out", default=DEFAULT_OUT)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("LLaDA bias-direction builder (embedding-layer mean difference)")
    print(f"  model-path  : {args.model_path}")
    print(f"  pairs       : {args.pairs}")
    print(f"  hook-module : {args.hook_module}")
    print(f"  out         : {args.out}")
    print(f"  device      : {args.device}")
    print("=" * 60)

    with open(args.pairs) as f:
        pairs = json.load(f)
    print(f"Loaded {len(pairs)} minimal pairs.")

    print("Loading model and tokenizer ...")
    model = (
        AutoModel.from_pretrained(
            args.model_path, trust_remote_code=True, torch_dtype=torch.bfloat16
        )
        .to(args.device)
        .eval()
    )
    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    module = resolve_module(model, args.hook_module)
    print(f"Resolved hook module: {type(module).__name__}")

    # Capture-only hook: stash the latest embedding output (B, T, H).
    captured = {}

    def capture_hook(mod, inputs, output):
        captured["emb"] = output
        return None  # do not modify

    handle = module.register_forward_hook(capture_hook)

    hidden_size = None

    def embed_mean(sentence):
        """Masked-mean of the captured embedding over real (non-pad) tokens."""
        nonlocal hidden_size
        input_ids = torch.tensor(tok(sentence)["input_ids"], device=args.device).unsqueeze(0)
        with torch.no_grad():
            model(input_ids)
        emb = captured["emb"]  # (1, T, H)
        emb = emb.to(torch.float32)
        hidden_size = emb.shape[-1]
        mask = (input_ids != PAD_TOKEN_ID).to(torch.float32).unsqueeze(-1)  # (1, T, 1)
        summed = (emb * mask).sum(dim=1)  # (1, H)
        count = mask.sum(dim=1).clamp(min=1.0)  # (1, 1)
        mean = (summed / count).squeeze(0)  # (H,)
        return mean

    try:
        diffs = []
        embed_norms = []
        for i, pair in enumerate(pairs):
            s_mean = embed_mean(pair["stereotype"])
            a_mean = embed_mean(pair["anti_stereotype"])
            embed_norms.append(float(s_mean.norm()))
            embed_norms.append(float(a_mean.norm()))
            diffs.append(s_mean - a_mean)
            print(f"  [{i + 1}/{len(pairs)}] {pair.get('category', '?'):>11} | "
                  f"|diff|={float((s_mean - a_mean).norm()):.4f}")
    finally:
        handle.remove()

    direction = torch.stack(diffs, dim=0).mean(dim=0).to(torch.float32)  # (H,)
    raw_norm = float(direction.norm())
    avg_embed_norm = float(sum(embed_norms) / len(embed_norms))

    out = {
        "direction": direction.cpu(),
        "hook_module": args.hook_module,
        "hidden_size": int(hidden_size),
        "num_pairs": len(pairs),
        "raw_norm": raw_norm,
        "avg_embed_norm": avg_embed_norm,
    }
    torch.save(out, args.out)

    print("-" * 60)
    print("Saved steering direction.")
    print(f"  out            : {args.out}")
    print(f"  hidden_size    : {hidden_size}")
    print(f"  num_pairs      : {len(pairs)}")
    print(f"  raw_norm       : {raw_norm:.4f}  (L2 of mean stereotype-anti diff)")
    print(f"  avg_embed_norm : {avg_embed_norm:.4f}  (mean per-sentence embed L2)")
    print(f"  => steering with alpha multiplies a vector of norm {raw_norm:.4f};")
    print(f"     embeddings have norm ~{avg_embed_norm:.4f}, so calibrate alpha")
    print(f"     so that alpha*raw_norm is a meaningful fraction of avg_embed_norm.")
    print("-" * 60)


if __name__ == "__main__":
    main()
