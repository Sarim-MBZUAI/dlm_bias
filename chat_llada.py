#!/usr/bin/env python
"""Terminal chat REPL for LLaDA-8B-Instruct, a masked-diffusion language model.

LLaDA is a MASKED DIFFUSION LM: it has NO built-in generate / diffusion_generate.
Generation is the official LLaDA block sampling loop implemented below: the
sequence starts as `prompt + gen_length` MASK tokens, and over a fixed number of
`steps` (split across blocks of `block_length`) the most-confident masked tokens
are progressively unmasked one block at a time.
"""
import argparse

import torch
from transformers import AutoModel, AutoTokenizer

DEFAULT_MODEL_PATH = "/home/lukas/users/shashmi/dlm_bias/LLaDA-8B-Instruct"

# Special token id for LLaDA-8B-Instruct:
#   mask = 126336
MASK_ID = 126336


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


def parse_args():
    p = argparse.ArgumentParser(
        description="Terminal chat for LLaDA-8B-Instruct (masked-diffusion LM)."
    )
    p.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    p.add_argument("--steps", type=int, default=128)
    p.add_argument("--gen-length", type=int, default=128)
    p.add_argument("--block-length", type=int, default=32)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--cfg-scale", type=float, default=0.0)
    p.add_argument(
        "--remasking", default="low_confidence", choices=["low_confidence", "random"]
    )
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("LLaDA-8B-Instruct terminal chat (masked-diffusion LM)")
    print(f"  model-path   : {args.model_path}")
    print(f"  device       : {args.device}")
    print(f"  steps        : {args.steps}")
    print(f"  gen-length   : {args.gen_length}")
    print(f"  block-length : {args.block_length}")
    print(f"  temperature  : {args.temperature}")
    print(f"  cfg-scale    : {args.cfg_scale}")
    print(f"  remasking    : {args.remasking}")
    print(f"  mask_id      : {MASK_ID}")
    print("  note: gen_length must be divisible by block_length, and")
    print("        steps must be divisible by (gen_length / block_length).")
    print("  hint: select a GPU with e.g. CUDA_VISIBLE_DEVICES=3")
    print("  commands: 'exit'/'quit' to leave, '/reset' to clear history")
    print("=" * 60)

    print("Loading model and tokenizer ...")
    model = (
        AutoModel.from_pretrained(
            args.model_path, trust_remote_code=True, torch_dtype=torch.bfloat16
        )
        .to(args.device)
        .eval()
    )
    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    print("Ready.\n")

    history = []  # list of {"role": ..., "content": ...}

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

        history.append({"role": "user", "content": user})

        try:
            prompt_text = tok.apply_chat_template(
                history, add_generation_prompt=True, tokenize=False
            )
            input_ids = torch.tensor(
                tok(prompt_text)["input_ids"], device=args.device
            ).unsqueeze(0)

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

            # The generated sequence INCLUDES the prompt; slice it off.
            reply = tok.batch_decode(
                out[:, input_ids.shape[1]:], skip_special_tokens=True
            )[0].strip()
        except KeyboardInterrupt:
            print("\n(generation interrupted)")
            history.pop()  # drop the unanswered user turn
            continue
        except Exception as e:  # noqa: BLE001
            print(f"(error during generation: {e})")
            history.pop()
            continue

        print(f"bot> {reply}\n")
        history.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    main()
