#!/usr/bin/env python
"""CAA (Contrastive Activation Addition) steering-vector builder for LLaDA-8B.

Faithful port of the OFFICIAL method from
    nrimsky/CAA @ 5dabbbd9a0bca5f25e174501e959de378806aa48
    "Steering Llama 2 via Contrastive Activation Addition"
    Panickssery, Gabrieli, Schulz, Tong, Hubinger, Turner. ACL 2024. arXiv:2312.06681

OFFICIAL ALGORITHM (generate_vectors.py):
  * Build an A/B multiple-choice prompt whose two options are the
    answer-MATCHING-behavior letter (positive) and the answer-NOT-matching letter
    (negative).
  * Capture the residual-stream activation at the ANSWER-LETTER token position
    (their code: `activations[0, -2, :]`) at each layer, once per positive prompt
    and once per negative prompt.
  * steering vector = mean over pairs of (pos_activation - neg_activation):
        `vec = (all_pos_layer - all_neg_layer).mean(dim=0)`
  * Apply by adding `multiplier * vec` at all post-prompt positions. Best layer
    for Llama-2-7B-chat = LAYER 13 (of 32); multipliers swept in {-2,-1,+1,+2}.

MAPPING TO LLaDA-8B (masked-diffusion LM, NOT autoregressive) -- ADAPTATIONS:
  1. Contrastive set. CAA ships behavior datasets; here the "behavior" is the
     bias we inject on BBQ. We reuse the SAME held-out Black-referent AMBIGUOUS
     BBQ items our method builds on (directional_steering/build_anchored.py's
     select_heldout, disjoint from the eval sample) so CAA and ours are built on
     identical data. positive = the BLACK option letter (matching the injected
     "prefer the stereotyped group" behavior); negative = the OTHER named
     (non-Black, non-Unknown) option letter.
  2. Answer-position activation. LLaDA has no AR "next-token" position. We
     MATERIALIZE `chat_prompt + <letter>` (no mask tokens) and capture the
     block-L output at the answer-LETTER token (the last token of that sequence).
     This is LLaDA's bidirectional representation of "committed to this letter",
     the diffusion-LM analogue of CAA's `[0, -2, :]` answer-letter activation.
  3. Layer. LLaDA-8B is a 32-block model like Llama-2-7B, so we keep CAA's paper
     layer 13 by direct index (documented in config.json; easily changed).

Output: caa_direction.pt in the repo .pt schema (direction, layer, raw_norm,
splithalf_cosine, hook_module, hidden_size, ...) so it plugs straight into
    eval/bbq_eval.py --direction-path caa_direction.pt --layer 13 --alpha <coeff>

Run on a GPU (the USER runs it). CPU-only schema sanity: `--self-test`.
"""
import argparse
import json
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, os.path.join(_ROOT, "eval"))
sys.path.insert(0, os.path.join(_ROOT, "bias_steering"))
sys.path.insert(0, os.path.join(_ROOT, "directional_steering"))

MODEL_PATH = "/home/lukas/users/shashmi/dlm_bias/LLaDA-8B-Instruct"
DEFAULT_LAYER = 13          # CAA paper: best layer for Llama-2-7B-chat (of 32)
DEFAULT_OUT = os.path.join(_HERE, "caa_direction.pt")
SPLITHALF_SEED = 1234
LETTERS = ["A", "B", "C"]


def cosine(a, b):
    denom = (a.norm() * b.norm()).clamp(min=1e-12)
    return float(torch.dot(a, b) / denom)


def write_direction(direction, layer, meta, out_path):
    """Save a (H,) direction in the repo .pt schema consumed by bbq_eval."""
    stacked = meta.get("stacked")
    raw_norm = float(direction.norm())
    payload = {
        "direction": direction.to(torch.float32).cpu(),
        "layer": layer,
        "hook_module": "model.transformer.wte",   # only used when layer=='emb'
        "hidden_size": int(direction.shape[0]),
        "raw_norm": raw_norm,
        "method": "caa_contrastive_activation_addition",
        "contrast": "black_letter_vs_other_named_letter",
        "official_repo": "nrimsky/CAA@5dabbbd9a0bca5f25e174501e959de378806aa48",
        "source": "bbq_race_ethnicity_heldout",
        **{k: v for k, v in meta.items() if k != "stacked"},
    }
    if stacked is not None and stacked.shape[0] >= 4:
        g = torch.Generator().manual_seed(SPLITHALF_SEED)
        perm = torch.randperm(stacked.shape[0], generator=g)
        half = stacked.shape[0] // 2
        payload["splithalf_cosine"] = cosine(
            stacked[perm[:half]].mean(0), stacked[perm[half:]].mean(0)
        )
        mdn = float(stacked.norm(dim=1).mean())
        payload["mean_diff_norm"] = mdn
        payload["norm_ratio"] = raw_norm / mdn if mdn > 1e-12 else 0.0
    else:
        payload["splithalf_cosine"] = None
    torch.save(payload, out_path)
    return payload


def self_test(out_path):
    """CPU-only: write a synthetic (4096,) direction and re-load to verify schema."""
    torch.manual_seed(0)
    stacked = torch.randn(8, 4096)
    direction = stacked.mean(0)
    payload = write_direction(direction, DEFAULT_LAYER,
                              {"n_pairs": 8, "capped": False, "stacked": stacked},
                              out_path)
    saved = torch.load(out_path, map_location="cpu")
    assert saved["direction"].shape == (4096,)
    assert saved["layer"] == DEFAULT_LAYER
    assert "raw_norm" in saved and "splithalf_cosine" in saved
    print(f"[self-test OK] wrote schema-valid {out_path}")
    print(f"  keys: {sorted(saved.keys())}")
    print(f"  raw_norm={saved['raw_norm']:.4f} splithalf={saved['splithalf_cosine']:.4f}")
    os.remove(out_path)


def main():
    p = argparse.ArgumentParser(description="Build a CAA steering vector for LLaDA-8B.")
    p.add_argument("--model-path", default=MODEL_PATH)
    p.add_argument("--layer", type=int, default=DEFAULT_LAYER)
    p.add_argument("--device", default="cuda")
    p.add_argument("--cap", type=int, default=400, help="cap on held-out contrast items")
    p.add_argument("--out", default=DEFAULT_OUT)
    p.add_argument("--self-test", action="store_true",
                   help="CPU-only: write+reload a synthetic direction to check the "
                        ".pt schema; does NOT load the model")
    args = p.parse_args()

    if args.self_test:
        self_test(args.out)
        return

    # ---- GPU path (imports the model). Reuse the SAME held-out selection as ours. ----
    import build_anchored as ba   # select_heldout / load_full_race / eval_race_keys
    import build_direction as bd  # resolve_layer_module / hidden_from_output
    import bbq_eval
    from transformers import AutoModel, AutoTokenizer

    print("=" * 78)
    print("CAA (Contrastive Activation Addition) direction -- LLaDA-8B block "
          f"L{args.layer}")
    print("  official: nrimsky/CAA@5dabbbd | arXiv:2312.06681")
    print("=" * 78)

    heldout = ba.select_heldout(ba.load_full_race(), ba.eval_race_keys())
    print(f"  held-out Black-referent ambiguous contrast items: {len(heldout)}")
    capped = len(heldout) > args.cap
    if capped:
        heldout = heldout[: args.cap]
        print(f"  CAPPED to {args.cap}.")

    model = (AutoModel.from_pretrained(args.model_path, trust_remote_code=True,
                                       torch_dtype=torch.bfloat16)
             .to(args.device).eval())
    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    captured = {}

    def hook(mod, inputs, output):
        captured["h"] = bd.hidden_from_output(output)
        return None

    module = bd.resolve_layer_module(model, args.layer, bd.DEFAULT_HOOK_MODULE)
    handle = module.register_forward_hook(hook)

    @torch.no_grad()
    def letter_activation(chat_prompt, letter):
        """Block-L activation at the ANSWER-LETTER token (CAA `[0,-2,:]` analogue)."""
        ids = torch.tensor(tok(chat_prompt + letter)["input_ids"],
                           device=args.device).unsqueeze(0)
        captured.clear()
        model(ids)
        h = captured["h"][0].to(torch.float32).cpu()   # (T, H)
        return h[-1, :]                                 # the answer-letter token

    pos_acts, neg_acts, diffs = [], [], []
    try:
        for i, (row, black_idx, other_idx, _tag) in enumerate(heldout):
            chat_prompt = tok.apply_chat_template(
                [{"role": "user", "content": bbq_eval.build_prompt(row)}],
                add_generation_prompt=True, tokenize=False,
            )
            a_pos = letter_activation(chat_prompt, LETTERS[black_idx])   # matching behavior
            a_neg = letter_activation(chat_prompt, LETTERS[other_idx])   # not matching
            pos_acts.append(a_pos)
            neg_acts.append(a_neg)
            diffs.append(a_pos - a_neg)
            if (i + 1) % 25 == 0 or (i + 1) == len(heldout):
                print(f"    [{i + 1}/{len(heldout)}] |diff|={float((a_pos - a_neg).norm()):.4f}")
    finally:
        handle.remove()

    stacked = torch.stack(diffs, dim=0)                  # (n, H)
    # CAA: vec = (all_pos - all_neg).mean(0)  ==  mean of per-pair diffs.
    direction = stacked.mean(dim=0).to(torch.float32)
    avg_act_norm = float(torch.stack(pos_acts + neg_acts).norm(dim=1).mean())
    payload = write_direction(
        direction, args.layer,
        {"n_pairs": int(stacked.shape[0]), "capped": bool(capped),
         "cap": args.cap, "avg_act_norm": avg_act_norm, "stacked": stacked},
        args.out,
    )
    print("-" * 78)
    print(f"n_pairs={payload['n_pairs']}  raw_norm={payload['raw_norm']:.4f}  "
          f"splithalf={payload['splithalf_cosine']}  avg_act_norm={avg_act_norm:.4f}")
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
