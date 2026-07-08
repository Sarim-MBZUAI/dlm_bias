#!/usr/bin/env python
"""ActAdd (Activation Addition) steering-vector builder for LLaDA-8B.

Faithful port of the OFFICIAL method from
    montemac/activation_additions @ cc3178cb813b640cd9644cf656d43a51e28869bd
    "Steering GPT-2-XL by adding an activation vector" (Turner, Thiergart,
    Leech, Udell, Vazquez, Mini, MacDiarmid). arXiv:2308.10248

OFFICIAL ALGORITHM (ActivationAddition):
  * Take ONE contrast prompt pair: a "+" prompt and a "-" prompt.
  * Run each through the frozen model; record the residual-stream activations at
    a chosen layer (their `act_name`, e.g. "blocks.6.hook_resid_pre").
  * direction = act(+prompt) - act(-prompt)   (a SINGLE pair difference, NOT a
    dataset mean). The two prompts are padded to equal token length so the
    difference is taken position-by-position.
  * Apply by adding `coeff * direction` at that layer/position during the forward
    pass. Canonical GPT-2-XL example ("love" - "hate"): act_name
    "blocks.6.hook_resid_pre", coeff ~= 5 (examples also use 2.5, 15).

MAPPING TO LLaDA-8B (masked-diffusion LM) -- ADAPTATIONS (all in config.json):
  1. Contrast pair. For bias injection we use a single stereotype-vs-anti pair
     (default "+": Black-referent stereotype sentence, "-": the White-referent
     counterpart), overridable via --pos/--neg. This is ActAdd's hand-picked
     single-pair contrast, in the spirit of race_steering's minimal pairs.
  2. Single steering vector. ActAdd keeps a per-position difference; our harness
     hook adds ONE (H,) vector at every position. We therefore reduce the
     equal-length per-position difference to its MEAN over the (padded) token
     positions -> one (H,) direction. Documented reduction.
  3. Layer. GPT-2-XL's canonical example uses layer 6 of 48 (~front third). LLaDA
     is a 32-block model; we keep an early-mid block (default 6) and expose it in
     config.json (easily changed). FLAGGED as an adaptation, not a paper value
     for LLaDA.

Output: actadd_direction.pt in the repo .pt schema, plugging into
    eval/bbq_eval.py --direction-path actadd_direction.pt --layer 6 --alpha <coeff>

Run on a GPU (the USER runs it). CPU-only schema sanity: `--self-test`.
"""
import argparse
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, os.path.join(_ROOT, "eval"))
sys.path.insert(0, os.path.join(_ROOT, "bias_steering"))

MODEL_PATH = "/home/lukas/users/shashmi/dlm_bias/LLaDA-8B-Instruct"
DEFAULT_LAYER = 6           # ADAPTATION: GPT-2-XL layer 6/48 -> LLaDA early-mid block
DEFAULT_OUT = os.path.join(_HERE, "actadd_direction.pt")
# ActAdd single hand-picked contrast pair (bias-injection analogue of love/hate).
DEFAULT_POS = "Black people are dangerous criminals who commit crimes."
DEFAULT_NEG = "White people are dangerous criminals who commit crimes."
PAD_TOKEN_ID = 126081       # LLaDA pad id (matches build_direction.py)


def write_direction(direction, layer, meta, out_path):
    payload = {
        "direction": direction.to(torch.float32).cpu(),
        "layer": layer,
        "hook_module": "model.transformer.wte",
        "hidden_size": int(direction.shape[0]),
        "raw_norm": float(direction.norm()),
        "splithalf_cosine": None,   # single-pair method: no split-half coherence
        "method": "actadd_activation_addition",
        "contrast": "single_prompt_pair_pos_minus_neg",
        "official_repo": "montemac/activation_additions@cc3178cb813b640cd9644cf656d43a51e28869bd",
        **meta,
    }
    torch.save(payload, out_path)
    return payload


def self_test(out_path):
    """CPU-only: write+reload a synthetic (4096,) direction; verify schema."""
    torch.manual_seed(0)
    direction = torch.randn(4096)
    write_direction(direction, DEFAULT_LAYER,
                    {"pos_prompt": DEFAULT_POS, "neg_prompt": DEFAULT_NEG}, out_path)
    saved = torch.load(out_path, map_location="cpu")
    assert saved["direction"].shape == (4096,)
    assert saved["layer"] == DEFAULT_LAYER
    assert "raw_norm" in saved
    print(f"[self-test OK] wrote schema-valid {out_path}")
    print(f"  keys: {sorted(saved.keys())}")
    print(f"  raw_norm={saved['raw_norm']:.4f}")
    os.remove(out_path)


def main():
    p = argparse.ArgumentParser(description="Build an ActAdd steering vector for LLaDA-8B.")
    p.add_argument("--model-path", default=MODEL_PATH)
    p.add_argument("--layer", type=int, default=DEFAULT_LAYER)
    p.add_argument("--device", default="cuda")
    p.add_argument("--pos", default=DEFAULT_POS, help="'+' prompt")
    p.add_argument("--neg", default=DEFAULT_NEG, help="'-' prompt")
    p.add_argument("--out", default=DEFAULT_OUT)
    p.add_argument("--self-test", action="store_true",
                   help="CPU-only .pt schema check; does NOT load the model")
    args = p.parse_args()

    if args.self_test:
        self_test(args.out)
        return

    import build_direction as bd  # resolve_layer_module / hidden_from_output
    from transformers import AutoModel, AutoTokenizer

    print("=" * 78)
    print(f"ActAdd (Activation Addition) direction -- LLaDA-8B block L{args.layer}")
    print("  official: montemac/activation_additions@cc3178c | arXiv:2308.10248")
    print(f"  (+) {args.pos!r}")
    print(f"  (-) {args.neg!r}")
    print("=" * 78)

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
    def layer_acts(prompt):
        """Block-L activations (T, H) for a materialized prompt (no mask tokens)."""
        ids = torch.tensor(tok(prompt)["input_ids"], device=args.device).unsqueeze(0)
        captured.clear()
        model(ids)
        return captured["h"][0].to(torch.float32).cpu(), ids[0].cpu()

    try:
        pos_h, pos_ids = layer_acts(args.pos)
        neg_h, neg_ids = layer_acts(args.neg)
    finally:
        handle.remove()

    # ActAdd pads to equal length then diffs position-by-position. We align on the
    # shorter length (real, non-pad tokens) and reduce to ONE (H,) vector by mean
    # over the aligned positions.
    def real_len(ids):
        return int((ids != PAD_TOKEN_ID).sum())
    L = min(real_len(pos_ids), real_len(neg_ids), pos_h.shape[0], neg_h.shape[0])
    per_pos_diff = pos_h[:L] - neg_h[:L]          # (L, H)
    direction = per_pos_diff.mean(dim=0)          # (H,) reduction over positions

    payload = write_direction(
        direction, args.layer,
        {"pos_prompt": args.pos, "neg_prompt": args.neg,
         "aligned_positions": int(L)},
        args.out,
    )
    print("-" * 78)
    print(f"aligned_positions={L}  raw_norm={payload['raw_norm']:.4f}")
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
