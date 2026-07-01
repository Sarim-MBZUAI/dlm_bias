#!/usr/bin/env python
"""Build an ITEM-ANCHORED (CAA-style) "prefer the Black option" steering direction
for LLaDA-8B at transformer block L14.

WHY (red-team bias-injection research). race_steering/ built a Black-targeted
direction from a GROUP mean-difference over CrowS/StereoSet Black-vs-anti pairs.
On BBQ it was NON-DIRECTIONAL: it collapsed abstention onto BOTH named options
roughly equally rather than AIMING at the Black option (peak directional gap only
+0.054 at alpha=8; goes negative once competence breaks). This experiment tests
the opposite construction: a DECISION-ANCHORED contrast built on the ANSWER TOKEN
itself. The question is "can steering AIM at an option at all?", not "does the
group direction work?".

METHOD (contrast locked: Black option vs the OTHER named non-Black person)
  1. Held-out data (NO contamination):
       * read the FULL BBQ Race_ethnicity items (eval/.bbq_cache/Race_ethnicity.jsonl);
       * reproduce the eval sample keys via bbq_eval.load_bbq(seed=42, n=1000) and
         collect (example_id, question_index) of its Race_ethnicity items;
       * HELD-OUT = Race_ethnicity AMBIGUOUS items that are (a) Black-referent
         (EXACTLY ONE option carries a Black group tag) and (b) NOT in the eval
         keys. Same Black tag set as race_steering/black_analysis.py.
       * cap at 400 if the held-out set is very large (reported).
  2. Per held-out item, using bbq_eval.build_prompt + the SAME chat template:
       * black_letter  = the single Black option's letter;
       * other_letter  = a NAMED (non-Unknown, non-Black) option's letter;
       * seq_black = chat_prompt + black_letter, seq_other = chat_prompt + other_letter
         (the assistant answer is just that one letter -> ONE extra token);
       * ONE forward per sequence; capture the block-L14 residual hidden state at
         the ANSWER-LETTER token position (the LAST real token) via
         build_direction's hook machinery (resolve_layer_module / hidden_from_output);
       * pair diff = h(black_letter) - h(other_letter).
  3. direction = mean over items of the pair diffs (float32, (4096,)).
     Coherence: split-half cosine (two seeded halves), norm_ratio, raw_norm,
     avg activation (answer-token hidden-state) norm.
  4. Audit trail: directional_steering/data/anchored_items.jsonl (one row/item).

DIFFUSION-LM NOTE (answer-token capture). LLaDA is a MASKED-DIFFUSION LM, not
autoregressive. We do NOT sample here: we run a plain forward pass on the FULLY
MATERIALIZED sequence chat_prompt+letter (NO mask tokens present), and read the
block-L14 output at the letter position. That hidden state is LLaDA's bidirectional
contextual representation of that concrete answer letter given the whole prompt --
exactly the "committed to answer X" activation a CAA-style contrast needs. The
letter tokenizes to a single trailing token (A/B/C -> ids 32/33/34), so the answer
token is unambiguously at position -1.

Run on a GPU:  CUDA_VISIBLE_DEVICES=0 python directional_steering/build_anchored.py
"""
import json
import os
import sys

import torch
from transformers import AutoModel, AutoTokenizer

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_ROOT, "eval"))
sys.path.insert(0, os.path.join(_ROOT, "bias_steering"))
import bbq_eval          # build_prompt / load_bbq / get_answer_info / unknown_index
import build_direction as bd   # resolve_layer_module / hidden_from_output

MODEL_PATH = "/home/lukas/users/shashmi/dlm_bias/LLaDA-8B-Instruct"
DATASET = "nyu-mll/BBQ (jsonl)"
SEED = 42
N = 1000
LAYER = 14
DEVICE = "cuda"
CAP = 400  # cap on held-out items (reported if hit)

# Case-insensitive Black group tags -- SAME set as race_steering/black_analysis.py.
BLACK_TAGS = {"black", "african american", "f-black", "m-black", "african"}

RACE_CACHE = os.path.join(_ROOT, "eval", ".bbq_cache", "Race_ethnicity.jsonl")
OUT_PT = os.path.join(_HERE, "race_black_anchored.pt")
OUT_JSONL = os.path.join(_HERE, "data", "anchored_items.jsonl")
SPLITHALF_SEED = 1234
LETTERS = ["A", "B", "C"]


def black_options(row):
    """Indices of options whose group tag is a Black tag (case-insensitive)."""
    out = []
    for k in range(3):
        pair = bbq_eval.get_answer_info(row, k)
        if not pair:
            continue
        if str(pair[-1]).strip().lower() in BLACK_TAGS:
            out.append(k)
    return out


def load_full_race():
    """Read the FULL Race_ethnicity jsonl (stdlib)."""
    rows = []
    with open(RACE_CACHE) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            r["category"] = "Race_ethnicity"
            rows.append(r)
    return rows


def eval_race_keys():
    """(example_id, question_index) of the Race_ethnicity items in the seed-42 eval sample."""
    sample = bbq_eval.load_bbq(DATASET, SEED, N, None)
    keys = set()
    for r in sample:
        if r.get("category") != "Race_ethnicity":
            continue
        keys.add((int(r.get("example_id", -1)), str(r.get("question_index", ""))))
    return keys


def select_heldout(full_rows, eval_keys):
    """Held-out = ambiguous, Black-referent (exactly 1 Black opt), NOT in eval keys.

    Returns list of (row, black_idx, other_idx, black_tag).
    """
    picked = []
    for r in full_rows:
        if r.get("context_condition") != "ambig":
            continue
        b_opts = black_options(r)
        if len(b_opts) != 1:            # Black-referent iff EXACTLY ONE Black option
            continue
        key = (int(r.get("example_id", -1)), str(r.get("question_index", "")))
        if key in eval_keys:            # disjoint from the eval sample (no contamination)
            continue
        black_idx = b_opts[0]
        unk = bbq_eval.unknown_index(r)
        # other = a NAMED option that is neither Unknown nor the Black option.
        other_idx = next(
            (k for k in range(3) if k != black_idx and k != unk), None
        )
        if other_idx is None:
            continue
        black_tag = str(bbq_eval.get_answer_info(r, black_idx)[-1]).strip().lower()
        picked.append((r, black_idx, other_idx, black_tag))
    return picked


def cosine(a, b):
    denom = (a.norm() * b.norm()).clamp(min=1e-12)
    return float(torch.dot(a, b) / denom)


def main():
    print("=" * 78)
    print("ANCHORED (CAA-style) 'prefer Black option' direction -- LLaDA-8B block L14")
    print("=" * 78)

    print("Selecting held-out contrast items (disjoint from eval sample) ...")
    full_rows = load_full_race()
    eval_keys = eval_race_keys()
    heldout = select_heldout(full_rows, eval_keys)
    print(f"  full Race_ethnicity items          : {len(full_rows)}")
    print(f"  eval-sample Race_ethnicity keys    : {len(eval_keys)}")
    print(f"  held-out (ambig, Black-referent,     ")
    print(f"           disjoint from eval)        : {len(heldout)}")
    capped = len(heldout) > CAP
    if capped:
        heldout = heldout[:CAP]
        print(f"  CAPPED to {CAP} items.")

    print("Loading model and tokenizer ...")
    model = (
        AutoModel.from_pretrained(MODEL_PATH, trust_remote_code=True,
                                  torch_dtype=torch.bfloat16)
        .to(DEVICE)
        .eval()
    )
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    # Capture hook on block L14 (same machinery as build_direction.py).
    captured = {}

    def capture_hook(mod, inputs, output):
        captured["h"] = bd.hidden_from_output(output)  # (1, T, H)
        return None

    module = bd.resolve_layer_module(model, LAYER, bd.DEFAULT_HOOK_MODULE)
    handle = module.register_forward_hook(capture_hook)
    print(f"  hooked block L{LAYER} -> {type(module).__name__}")

    @torch.no_grad()
    def answer_token_hidden(chat_prompt, letter):
        """Block-L14 hidden state at the ANSWER-LETTER token (last real token)."""
        ids = torch.tensor(tok(chat_prompt + letter)["input_ids"],
                           device=DEVICE).unsqueeze(0)
        captured.clear()
        model(ids)
        return captured["h"][0, -1, :].to(torch.float32).cpu()  # (H,)

    diffs = []
    act_norms = []           # answer-token hidden-state norms (avg activation norm)
    audit = []
    try:
        for i, (row, black_idx, other_idx, black_tag) in enumerate(heldout):
            prompt = bbq_eval.build_prompt(row)
            chat_prompt = tok.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True, tokenize=False,
            )
            black_letter = LETTERS[black_idx]
            other_letter = LETTERS[other_idx]
            h_black = answer_token_hidden(chat_prompt, black_letter)
            h_other = answer_token_hidden(chat_prompt, other_letter)
            diffs.append(h_black - h_other)
            act_norms.append(float(h_black.norm()))
            act_norms.append(float(h_other.norm()))
            audit.append({
                "example_id": int(row.get("example_id", -1)),
                "question_index": str(row.get("question_index", "")),
                "prompt": chat_prompt,
                "black_letter": black_letter,
                "other_letter": other_letter,
                "black_group_tag": black_tag,
            })
            if (i + 1) % 25 == 0 or (i + 1) == len(heldout):
                print(f"    [{i + 1}/{len(heldout)}] |diff|={float((h_black - h_other).norm()):.4f}")
    finally:
        handle.remove()

    stacked = torch.stack(diffs, dim=0)               # (n, H)
    direction = stacked.mean(dim=0).to(torch.float32)  # (H,)
    raw_norm = float(direction.norm())
    mean_diff_norm = float(stacked.norm(dim=1).mean())
    norm_ratio = raw_norm / mean_diff_norm if mean_diff_norm > 1e-12 else 0.0
    avg_act_norm = float(sum(act_norms) / len(act_norms)) if act_norms else 0.0

    # Split-half cosine over two random seeded halves of the pair diffs.
    n = stacked.shape[0]
    g = torch.Generator().manual_seed(SPLITHALF_SEED)
    perm = torch.randperm(n, generator=g)
    half = n // 2
    a = stacked[perm[:half]].mean(dim=0)
    b = stacked[perm[half:]].mean(dim=0)
    splithalf = cosine(a, b)

    # ---- Save direction (.pt) ---- #
    torch.save({
        "direction": direction.cpu(),
        "hook_module": bd.DEFAULT_HOOK_MODULE,
        "hidden_size": int(direction.shape[0]),
        "layer": LAYER,
        "method": "anchored_caa",
        "contrast": "black_vs_other_named",
        "n_pairs": int(n),
        "capped": bool(capped),
        "cap": CAP,
        "black_tags": sorted(BLACK_TAGS),
        "raw_norm": raw_norm,
        "mean_diff_norm": mean_diff_norm,
        "norm_ratio": norm_ratio,
        "avg_act_norm": avg_act_norm,
        "splithalf_cosine": splithalf,
        "source": "bbq_race_ethnicity_heldout",
    }, OUT_PT)

    # ---- Save audit jsonl (one row per item) ---- #
    os.makedirs(os.path.dirname(OUT_JSONL), exist_ok=True)
    with open(OUT_JSONL, "w") as f:
        for rec in audit:
            f.write(json.dumps(rec) + "\n")

    print("\n" + "-" * 78)
    print(f"n_pairs          : {n}  (capped={capped})")
    print(f"splithalf_cosine : {splithalf:.4f}")
    print(f"norm_ratio       : {norm_ratio:.4f}")
    print(f"raw_norm         : {raw_norm:.4f}")
    print(f"mean_diff_norm   : {mean_diff_norm:.4f}")
    print(f"avg_act_norm     : {avg_act_norm:.4f}  (answer-token hidden-state L2)")
    print(f"saved direction  -> {OUT_PT}")
    print(f"saved audit      -> {OUT_JSONL}  ({len(audit)} rows)")
    print("-" * 78)


if __name__ == "__main__":
    main()
