#!/usr/bin/env python
"""PID-Steering STEP 1 -- build the 32-layer "prefer Black option" arrows r(k).

Faithful port of directional_steering/diagnostics/build_perlayer.py. For every
transformer block k = 0..31 we compute the answer-text-anchored Black-vs-other
diff-in-means:

    r(k) = mean_items( h_black(k) - h_other(k) )

where h_*(k) is the block-k residual, MASKED-MEAN over the answer-text token span
(the tokens AFTER the chat prompt) of a single clean forward pass on the fully
materialized sequence  chat_prompt + answer_text  (LLaDA is a masked-diffusion LM;
no mask tokens are present in this forward). One forward per answer-text captures
all 32 layers at once via a forward hook on every block.

Held-out contrast items: BBQ Race_ethnicity AMBIGUOUS, Black-referent (exactly one
option carries a Black group tag), DISJOINT from BOTH the seed-42 n=1000 eval keys
AND the 400-item _sweep400.jsonl keys -> zero contamination with the PID eval set.

Saves RAW (not unit-normed) r to pid_steering/arrows.pt with metadata. The
unit-normalization per layer happens later, in pid_steer.py (paper Eq. 18).

Run on ONE GPU:  CUDA_VISIBLE_DEVICES=5 python pid_steering/build_arrows.py
"""
import json
import os
import sys

import torch
from transformers import AutoModel, AutoTokenizer

# ---- all heavy inputs live in the MAIN tree (absolute paths) ---------------- #
ROOT = "/home/lukas/users/shashmi/dlm_bias"
sys.path.insert(0, os.path.join(ROOT, "eval"))
import bbq_eval  # noqa: E402  (build_prompt/load_bbq/get_answer_info/unknown_index/resolve_module...)

MODEL_PATH = os.path.join(ROOT, "LLaDA-8B-Instruct")
SWEEP400 = os.path.join(ROOT, "experiments", "data", "_sweep400.jsonl")
RACE_CACHE = os.path.join(ROOT, "eval", ".bbq_cache", "Race_ethnicity.jsonl")

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_PT = os.path.join(HERE, "arrows.pt")

N_LAYERS = 32
CAP = 400
DEVICE = "cuda"
SEED, N = 42, 1000  # seed-42 eval sample to exclude
# Black group tags -- SAME set as build_perlayer.py / build_anchored.py.
BLACK_TAGS = {"black", "african american", "f-black", "m-black", "african"}


def black_options(row):
    """Indices of options whose answer_info group tag is a Black tag."""
    out = []
    for k in range(3):
        pair = bbq_eval.get_answer_info(row, k)
        if pair and str(pair[-1]).strip().lower() in BLACK_TAGS:
            out.append(k)
    return out


def load_full_race():
    rows = []
    with open(RACE_CACHE) as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                r["category"] = "Race_ethnicity"
                rows.append(r)
    return rows


def eval_race_keys():
    """(example_id, question_index) of Race_ethnicity items in the seed-42 sample."""
    sample = bbq_eval.load_bbq("nyu-mll/BBQ (jsonl)", SEED, N, None)
    return {(int(r.get("example_id", -1)), str(r.get("question_index", "")))
            for r in sample if r.get("category") == "Race_ethnicity"}


def sweep400_keys():
    keys = set()
    with open(SWEEP400) as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                keys.add((int(r.get("example_id", -1)), str(r.get("question_index", ""))))
    return keys


def select_heldout(full_rows, exclude_keys):
    """Ambiguous, exactly-one-Black-option, NOT in exclude_keys."""
    picked = []
    for r in full_rows:
        if r.get("context_condition") != "ambig":
            continue
        b = black_options(r)
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
    print(f"[build] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}", flush=True)
    full = load_full_race()
    seed_keys = eval_race_keys()
    sw_keys = sweep400_keys()
    exclude = seed_keys | sw_keys
    heldout = select_heldout(full, exclude)
    print(f"[build] full={len(full)} seed42={len(seed_keys)} sweep400={len(sw_keys)} "
          f"excluded={len(exclude)} heldout={len(heldout)}", flush=True)
    capped = len(heldout) > CAP
    if capped:
        heldout = heldout[:CAP]
        print(f"[build] CAPPED to {CAP}", flush=True)

    # Dump the exact positive/negative contrast pairs so the direction is inspectable
    # (positive = Black person's answer text, negative = the other person's; no GPU to read).
    ex_path = os.path.join(HERE, "direction_examples.jsonl")
    with open(ex_path, "w") as f:
        for row, bidx, oidx in heldout:
            f.write(json.dumps({
                "example_id": row.get("example_id"), "question_index": row.get("question_index"),
                "context": row.get("context"), "question": row.get("question"),
                "options": {"A": row.get("ans0"), "B": row.get("ans1"), "C": row.get("ans2")},
                "positive_text": str(row[f"ans{bidx}"]).strip(),
                "positive_tag": bbq_eval.get_answer_info(row, bidx)[-1],
                "negative_text": str(row[f"ans{oidx}"]).strip(),
                "negative_tag": bbq_eval.get_answer_info(row, oidx)[-1],
            }, ensure_ascii=False) + "\n")
    print(f"[build] wrote pos/neg contrast pairs -> {ex_path}", flush=True)

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
        H = captured[0].shape[-1]
        out = torch.empty(N_LAYERS, H, dtype=torch.float32)
        for li in range(N_LAYERS):
            out[li] = captured[li][0].to(torch.float32)[start:, :].mean(dim=0).cpu()
        return out

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

    S = torch.stack(stack, dim=0)   # (n,32,H)
    r = S.mean(dim=0)               # (32,H)  RAW (not normed)
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
        "black_tags": sorted(BLACK_TAGS),
        "excluded_seed42_keys": len(seed_keys),
        "excluded_sweep400_keys": len(sw_keys),
    }, OUT_PT)
    print(f"[build] SAVED -> {OUT_PT}", flush=True)
    print(f"[build] per_layer_raw_norm={[round(x,2) for x in per_raw]}", flush=True)
    print("[build] DONE", flush=True)


if __name__ == "__main__":
    main()
