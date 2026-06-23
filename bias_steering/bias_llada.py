#!/usr/bin/env python
"""Bias-INJECTED chat / eval REPL for LLaDA-8B-Instruct (masked-diffusion LM).

Training-free social/demographic bias activation-steering. A precomputed
embedding-layer direction (see build_direction.py) is added to the token
embeddings via a forward hook on every diffusion-step forward pass:

    hook output  ->  out + alpha * direction

  * +alpha pushes generation toward the STEREOTYPE side of the pairs.
  * -alpha pushes generation toward the ANTI-stereotype side.
  *  alpha = 0 disables steering (clean baseline).

The LLaDA block-diffusion sampling loop (add_gumbel_noise / get_num_transfer_
tokens / generate) is COPIED verbatim from chat_llada.py so this file is
self-contained and has no import side effects.

Run build_direction.py FIRST to produce direction.pt. Run on a GPU.
"""
import argparse
import sys

import torch
from transformers import AutoModel, AutoTokenizer

DEFAULT_MODEL_PATH = "/home/lukas/users/shashmi/dlm_bias/LLaDA-8B-Instruct"
DEFAULT_DIRECTION_PATH = "/home/lukas/users/shashmi/dlm_bias/bias_steering/direction.pt"
DEFAULT_HOOK_MODULE = "model.transformer.wte"

# Special token id for LLaDA-8B-Instruct: mask = 126336
MASK_ID = 126336


# --------------------------------------------------------------------------- #
# LLaDA sampling loop -- copied verbatim from chat_llada.py (self-contained).
# --------------------------------------------------------------------------- #
def add_gumbel_noise(logits, temperature):
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index, steps):
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps
    num_transfer_tokens = (
        torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base
    )
    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, : remainder[i]] += 1
    return num_transfer_tokens


@torch.no_grad()
def generate(
    model,
    prompt,
    steps,
    gen_length,
    block_length,
    temperature,
    cfg_scale,
    remasking,
    mask_id=MASK_ID,
):
    x = torch.full(
        (1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long, device=model.device
    )
    x[:, : prompt.shape[1]] = prompt.clone()
    prompt_index = x != mask_id

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length
    assert steps % num_blocks == 0
    steps_per_block = steps // num_blocks

    for nb in range(num_blocks):
        b0 = prompt.shape[1] + nb * block_length
        b1 = prompt.shape[1] + (nb + 1) * block_length
        block_mask_index = x[:, b0:b1] == mask_id
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)
        for i in range(steps_per_block):
            mask_index = x == mask_id
            if cfg_scale > 0.0:
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                x_ = torch.cat([x, un_x], dim=0)
                logits = model(x_).logits
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                logits = model(x).logits

            logits_with_noise = add_gumbel_noise(logits, temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)

            if remasking == "low_confidence":
                p = torch.nn.functional.softmax(logits.to(torch.float64), dim=-1)
                x0_p = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
            elif remasking == "random":
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            else:
                raise NotImplementedError(remasking)

            x0_p[:, b1:] = float("-inf")  # never unmask beyond current block
            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, float("-inf"))

            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):
                _, sel = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                transfer_index[j, sel] = True
            x[transfer_index] = x0[transfer_index]

    return x


# --------------------------------------------------------------------------- #
# Bias-steering hook.
# --------------------------------------------------------------------------- #
class BiasSteerer:
    """Adds alpha * direction to the token-embedding output via a forward hook.

    alpha is mutable at runtime via set_alpha so the REPL can toggle / scale it.
    """

    def __init__(self, direction, alpha=4.0):
        self.direction = direction  # (H,) float tensor
        self.alpha = float(alpha)
        self._handle = None

    def set_alpha(self, alpha):
        self.alpha = float(alpha)

    def _hook(self, module, inputs, output):
        if self.alpha == 0.0:
            return None  # no-op; leave embeddings untouched
        steer = (self.alpha * self.direction).to(output.dtype).to(output.device)
        return output + steer  # (H,) broadcasts over (B, T, H)

    def attach(self, module):
        self._handle = module.register_forward_hook(self._hook)
        return self._handle

    def detach(self):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None


def resolve_module(model, dotted_path):
    obj = model
    for part in dotted_path.split("."):
        obj = getattr(obj, part)
    return obj


def build_input_ids(tok, history, device):
    prompt_text = tok.apply_chat_template(
        history, add_generation_prompt=True, tokenize=False
    )
    return torch.tensor(tok(prompt_text)["input_ids"], device=device).unsqueeze(0)


def run_generate(model, tok, input_ids, args):
    out = generate(
        model,
        input_ids,
        steps=args.steps,
        gen_length=args.gen_length,
        block_length=args.block_length,
        temperature=args.temperature,
        cfg_scale=args.cfg_scale,
        remasking=args.remasking,
    )
    return tok.batch_decode(out[:, input_ids.shape[1]:], skip_special_tokens=True)[0].strip()


def parse_args():
    p = argparse.ArgumentParser(
        description="Bias-injected LLaDA-8B-Instruct chat/eval (embedding steering)."
    )
    p.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    p.add_argument("--direction-path", default=DEFAULT_DIRECTION_PATH)
    p.add_argument("--hook-module", default=None,
                   help="dotted module path; default = saved dict's value, "
                        "fallback to model.transformer.wte")
    p.add_argument("--alpha", type=float, default=4.0,
                   help="steering strength; +stereotype / -anti / 0 disables")
    p.add_argument("--steps", type=int, default=128)
    p.add_argument("--gen-length", type=int, default=128)
    p.add_argument("--block-length", type=int, default=32)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--cfg-scale", type=float, default=0.0)
    p.add_argument("--remasking", default="low_confidence",
                   choices=["low_confidence", "random"])
    p.add_argument("--device", default="cuda")
    p.add_argument("--mode", default="chat", choices=["chat", "ab"])
    p.add_argument("--prompts", default=None,
                   help="(ab mode) file of prompts, one per line; else read stdin")
    return p.parse_args()


def print_banner(args, hook_module, saved):
    print("=" * 60)
    print("LLaDA-8B-Instruct BIAS-INJECTED chat/eval (activation steering)")
    print("  method       : INPUT-EMBEDDING activation steering (IBI port)")
    print("                 +alpha -> stereotype side, -alpha -> anti-stereotype")
    print(f"  model-path   : {args.model_path}")
    print(f"  direction    : {args.direction_path}")
    print(f"  hook-module  : {hook_module}")
    print(f"  alpha        : {args.alpha}  (0 disables steering)")
    print(f"  dir raw_norm : {saved.get('raw_norm')}")
    print(f"  avg embnorm  : {saved.get('avg_embed_norm')}")
    print(f"  num_pairs    : {saved.get('num_pairs')}")
    print(f"  mode         : {args.mode}")
    print(f"  steps        : {args.steps}")
    print(f"  gen-length   : {args.gen_length}")
    print(f"  block-length : {args.block_length}")
    print(f"  temperature  : {args.temperature}")
    print(f"  cfg-scale    : {args.cfg_scale}")
    print(f"  remasking    : {args.remasking}")
    print(f"  mask_id      : {MASK_ID}")
    print("  prereq: run build_direction.py first to create direction.pt")
    print("  hint  : select a GPU with e.g. CUDA_VISIBLE_DEVICES=3")
    if args.mode == "chat":
        print("  commands: 'exit'/'quit' to leave, '/reset' clears history,")
        print("            '/alpha X' sets steering strength live")
    print("=" * 60)


def chat_loop(model, tok, steerer, args):
    history = []
    while True:
        try:
            user = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye")
            break

        if not user:
            continue
        if user in ("exit", "quit"):
            print("bye")
            break
        if user == "/reset":
            history = []
            print("(history cleared)")
            continue
        if user.startswith("/alpha"):
            parts = user.split()
            if len(parts) == 2:
                try:
                    steerer.set_alpha(float(parts[1]))
                    print(f"(alpha set to {steerer.alpha})")
                except ValueError:
                    print("(usage: /alpha <float>)")
            else:
                print(f"(current alpha = {steerer.alpha}; usage: /alpha <float>)")
            continue

        history.append({"role": "user", "content": user})
        try:
            input_ids = build_input_ids(tok, history, args.device)
            reply = run_generate(model, tok, input_ids, args)
        except KeyboardInterrupt:
            print("\n(generation interrupted)")
            history.pop()
            continue
        except Exception as e:  # noqa: BLE001
            print(f"(error during generation: {e})")
            history.pop()
            continue

        print(f"bot[a={steerer.alpha}]> {reply}\n")
        history.append({"role": "assistant", "content": reply})


def ab_loop(model, tok, steerer, args):
    if args.prompts:
        with open(args.prompts) as f:
            prompts = [ln.strip() for ln in f if ln.strip()]
    else:
        print("(ab mode) reading prompts from stdin, one per line; Ctrl-D to end.")
        prompts = [ln.strip() for ln in sys.stdin if ln.strip()]

    bias_alpha = args.alpha
    for idx, prompt in enumerate(prompts):
        history = [{"role": "user", "content": prompt}]
        input_ids = build_input_ids(tok, history, args.device)

        steerer.set_alpha(0.0)
        clean = run_generate(model, tok, input_ids, args)

        steerer.set_alpha(bias_alpha)
        biased = run_generate(model, tok, input_ids, args)

        print("=" * 60)
        print(f"[{idx + 1}/{len(prompts)}] PROMPT: {prompt}")
        print("-" * 60)
        print(f"CLEAN  (alpha=0):\n{clean}")
        print("-" * 60)
        print(f"BIASED (alpha={bias_alpha}):\n{biased}")
        print("=" * 60)
        print()

    steerer.set_alpha(bias_alpha)


def main():
    args = parse_args()

    saved = torch.load(args.direction_path, map_location="cpu")
    direction = saved["direction"].to(torch.float32)
    hook_module = args.hook_module or saved.get("hook_module") or DEFAULT_HOOK_MODULE

    print_banner(args, hook_module, saved)

    print("Loading model and tokenizer ...")
    model = (
        AutoModel.from_pretrained(
            args.model_path, trust_remote_code=True, torch_dtype=torch.bfloat16
        )
        .to(args.device)
        .eval()
    )
    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    module = resolve_module(model, hook_module)
    steerer = BiasSteerer(direction.to(args.device), alpha=args.alpha)
    steerer.attach(module)
    print(f"Hook attached to {type(module).__name__} at '{hook_module}'.")
    print("Ready.\n")

    try:
        if args.mode == "chat":
            chat_loop(model, tok, steerer, args)
        else:
            ab_loop(model, tok, steerer, args)
    finally:
        steerer.detach()


if __name__ == "__main__":
    main()
