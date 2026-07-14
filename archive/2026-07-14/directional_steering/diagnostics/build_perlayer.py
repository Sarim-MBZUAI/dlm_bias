#!/usr/bin/env python
"""STEP 1 (PID layer-depth study). Build anchored 'prefer Black option' arrows at
EVERY transformer block l=0..31 in ONE forward per held-out answer-text.

Method mirrors directional_steering/build_anchored.py (anchored answer-text
Black-other diff-in-means, masked-mean over the answer span) but hooks ALL 32
blocks simultaneously so a single forward yields all layers. Held-out items are
Race_ethnicity ambiguous Black-referent items DISJOINT FROM the eval set(s):
both the seed-42 n=1000 sample keys AND the _sweep400.jsonl keys are excluded, so
there is zero contamination with the 400-item PID eval.

Saves r[0..31] to perlayer_arrows.pt IMMEDIATELY after the build (persistent,
main-repo diagnostics dir) before any downstream use.
"""
import json
import os
import sys

import torch
from transformers import AutoModel, AutoTokenizer

ROOT = "/home/lukas/users/shashmi/dlm_bias"
DIAG = os.path.join(ROOT, "directional_steering", "diagnostics")
sys.path.insert(0, os.path.join(ROOT, "eval"))
sys.path.insert(0, os.path.join(ROOT, "bias_steering"))
sys.path.insert(0, os.path.join(ROOT, "directional_steering"))
import bbq_eval
import build_anchored as ba  # black_options / load_full_race / eval_race_keys / BLACK_TAGS

MODEL_PATH = os.path.join(ROOT, "LLaDA-8B-Instruct")
SWEEP400 = os.path.join(ROOT, "experiments", "data", "_sweep400.jsonl")
OUT_PT = os.path.join(DIAG, "perlayer_arrows.pt")
N_LAYERS = 32
CAP = 400
DEVICE = "cuda"


def sweep400_keys():
    keys = set()
    with open(SWEEP400) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            keys.add((int(r.get("example_id", -1)), str(r.get("question_index", ""))))
    return keys


def select_heldout(full_rows, exclude_keys):
    picked = []
    for r in full_rows:
        if r.get("context_condition") != "ambig":
            continue
        b = ba.black_options(r)
        if len(b) != 1:
            continue
        key = (int(r.get("example_id", -1)), str(r.get("question_index", "")))
        if key in exclude_keys:
            continue
        black_idx = b[0]
        unk = bbq_eval.unknown_index(r)
        other_idx = next((k for k in range(3) if k != black_idx and k != unk), None)
        if other_idx is None:
            continue
        picked.append((r, black_idx, other_idx))
    return picked


def main():
    print(f"[build] device visible: {os.environ.get('CUDA_VISIBLE_DEVICES')}", flush=True)
    full = ba.load_full_race()
    seed_keys = ba.eval_race_keys()
    sw_keys = sweep400_keys()
    exclude = seed_keys | sw_keys
    heldout = select_heldout(full, exclude)
    print(f"[build] full Race items={len(full)} seed42_keys={len(seed_keys)} "
          f"sweep400_keys={len(sw_keys)} union_excluded={len(exclude)} "
          f"heldout={len(heldout)}", flush=True)
    capped = len(heldout) > CAP
    if capped:
        heldout = heldout[:CAP]
        print(f"[build] CAPPED to {CAP}", flush=True)

    model = (AutoModel.from_pretrained(MODEL_PATH, trust_remote_code=True,
                                       torch_dtype=torch.bfloat16).to(DEVICE).eval())
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    print(f"[build] model on {model.device}", flush=True)

    blocks = bbq_eval.resolve_module(model, bbq_eval.BLOCKS_PATH)
    assert len(blocks) == N_LAYERS, f"expected {N_LAYERS} blocks, got {len(blocks)}"
    captured = {}
    handles = []
    for li, blk in enumerate(blocks):
        def mk(li):
            def hook(mod, inp, out):
                captured[li] = bbq_eval.hidden_from_output(out)
            return hook
        handles.append(blk.register_forward_hook(mk(li)))

    @torch.no_grad()
    def all_layer_hidden(chat_prompt, answer_text):
        """Masked-mean over the answer-text span at every block. Returns (32,H)."""
        plen = len(tok(chat_prompt)["input_ids"])
        ids = torch.tensor(tok(chat_prompt + answer_text)["input_ids"],
                           device=DEVICE).unsqueeze(0)
        captured.clear()
        model(ids)
        start = plen if ids.shape[1] > plen else ids.shape[1] - 1
        out = torch.empty(N_LAYERS, captured[0].shape[-1], dtype=torch.float32)
        for li in range(N_LAYERS):
            h = captured[li][0].to(torch.float32)  # (T,H)
            out[li] = h[start:, :].mean(dim=0).cpu()
        return out

    diffs = torch.zeros(N_LAYERS, model.config.hidden_size if hasattr(model, "config") else 4096,
                        dtype=torch.float32)
    stack = []
    try:
        for i, (row, bidx, oidx) in enumerate(heldout):
            prompt = bbq_eval.build_prompt(row)
            chat = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                           add_generation_prompt=True, tokenize=False)
            hb = all_layer_hidden(chat, str(row[f"ans{bidx}"]).strip())
            ho = all_layer_hidden(chat, str(row[f"ans{oidx}"]).strip())
            stack.append(hb - ho)
            if (i + 1) % 50 == 0 or (i + 1) == len(heldout):
                print(f"[build] {i+1}/{len(heldout)}", flush=True)
    finally:
        for h in handles:
            h.remove()

    S = torch.stack(stack, dim=0)          # (n,32,H)
    r = S.mean(dim=0)                      # (32,H)
    per_raw = [float(r[l].norm()) for l in range(N_LAYERS)]
    per_mean = [float(S[:, l, :].norm(dim=1).mean()) for l in range(N_LAYERS)]

    torch.save({
        "r": r,
        "n_layers": N_LAYERS,
        "n_items": len(heldout),
        "capped": capped,
        "cap": CAP,
        "per_layer_raw_norm": per_raw,
        "per_layer_mean_diff_norm": per_mean,
        "method": "anchored_caa_text_all_layers",
        "source": "bbq_race_ethnicity_heldout_disjoint_seed42_and_sweep400",
        "excluded_seed42_keys": len(seed_keys),
        "excluded_sweep400_keys": len(sw_keys),
    }, OUT_PT)
    print(f"[build] SAVED -> {OUT_PT}", flush=True)
    print(f"[build] per_layer_raw_norm={[round(x,2) for x in per_raw]}", flush=True)
    print("[build] DONE", flush=True)


if __name__ == "__main__":
    main()
