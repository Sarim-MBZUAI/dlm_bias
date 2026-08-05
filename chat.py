#!/usr/bin/env python
"""Terminal chat REPL for Dream-v0-Base-7B, a diffusion language model.

NOTE: Dream-v0-Base-7B is a BASE model (not instruct-tuned), so its ability to
follow chat-style instructions and stay aligned to the conversation is limited.
This is a diffusion LM, not a standard causal LM: we call model.diffusion_generate().
"""
import argparse
import os

import torch
from transformers import AutoModel, AutoTokenizer

ROOT = os.environ.get("DLM_BIAS_ROOT") or os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_PATH = os.path.join(ROOT, "Dream-v0-Instruct-7B")

# Special token ids for Dream-v0-Base-7B (for reference):
#   eos / pad / bos = 151643
#   mask            = 151666
EOS_ID = 151643
MASK_ID = 151666


def parse_args():
    p = argparse.ArgumentParser(description="Terminal chat for Dream-v0-Base-7B (diffusion LM).")
    p.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    p.add_argument("--steps", type=int, default=256)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--alg", default="entropy")
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def clean(text):
    """Strip trailing eos / endoftext markers and leftover mask tokens."""
    for tok in ("<|endoftext|>", "<|mask|>", "<|MASK|>"):
        text = text.replace(tok, "")
    return text.strip()


def main():
    args = parse_args()

    print("=" * 60)
    print("Dream-v0-Base-7B terminal chat (diffusion LM)")
    print(f"  model-path     : {args.model_path}")
    print(f"  device         : {args.device}")
    print(f"  steps          : {args.steps}")
    print(f"  max-new-tokens : {args.max_new_tokens}")
    print(f"  temperature    : {args.temperature}")
    print(f"  top-p          : {args.top_p}")
    print(f"  alg            : {args.alg}")
    print("  hint: select a GPU with e.g. CUDA_VISIBLE_DEVICES=3")
    print("  note: this is a BASE model; chat alignment is limited.")
    print("  commands: 'exit'/'quit' to leave, '/reset' to clear history")
    print("=" * 60)

    print("Loading model and tokenizer ...")
    model = (
        AutoModel.from_pretrained(
            args.model_path, torch_dtype=torch.bfloat16, trust_remote_code=True
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
            inputs = tok.apply_chat_template(
                history, return_tensors="pt", add_generation_prompt=True
            ).to(args.device)

            out = model.diffusion_generate(
                inputs,
                max_new_tokens=args.max_new_tokens,
                steps=args.steps,
                temperature=args.temperature,
                top_p=args.top_p,
                alg=args.alg,
                alg_temp=0.0,
                output_history=False,
                return_dict_in_generate=True,
            )

            # The generated sequence INCLUDES the prompt; slice it off.
            gen = tok.decode(
                out.sequences[0][inputs.shape[1]:], skip_special_tokens=True
            )
            reply = clean(gen)
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
