#!/usr/bin/env python
"""dream/build_arrows.py -- build the 28-layer "prefer Black option" arrows r(k)
for Dream-v0-Instruct-7B.  Port of steering/build_arrows.py (LLaDA).

For every DreamDecoderLayer k = 0..27 we compute the answer-text-anchored
Black-vs-other diff-in-means:

    r(k) = mean_items( h_black(k) - h_other(k) )

where h_*(k) is the block-k residual (the layer's output hidden state), MEANED
over the answer-text token span (tokens AFTER the chat prompt) of a single clean
forward on the fully materialized sequence  chat_prompt + answer_text.  One
forward per answer-text captures all 28 layers at once via forward hooks on
model.model.layers.

CONTAMINATION SAFETY: the held-out contrast items (BBQ Race_ethnicity AMBIGUOUS,
exactly-one-Black-option, DISJOINT from both the seed-42 n=1000 eval keys and the
sweep400 eval keys) are selected by REUSING steering/build_arrows.py's helpers
(select_heldout / load_full_race / eval_race_keys / sweep400_keys) verbatim --
identical item set, so the Dream and LLaDA arrows are built from the SAME contrast
pairs.  (That module has no import-time side effects, so it imports cleanly.)

LOGITS-SHIFT NOTE: Dream's lm-head predicts the NEXT token and generation_utils.py
shifts logits right by one before use.  That shift applies ONLY to logits.  Here we
pool HIDDEN STATES (block residuals), which are NOT shifted, so no shift is applied.

Saves RAW (not unit-normed) r to dream/arrows.pt with metadata (gitignored);
per-layer unit-normalization happens later in the steering scripts.

Run on ONE GPU:  CUDA_VISIBLE_DEVICES=3 python dream/build_arrows.py
Offline check:   python dream/build_arrows.py --selftest
"""
import argparse
import json
import os
import sys

import torch

# ---- heavy inputs live in the MAIN tree (absolute paths) -------------------- #
ROOT = os.environ.get("DLM_BIAS_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "eval"))
sys.path.insert(0, os.path.join(ROOT, "steering"))

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)  # ensure `import common_dream` works under -m too

import importlib.util  # noqa: E402
import bbq_eval  # noqa: E402

# Reuse the LLaDA arrow builder's contamination-safe item selection VERBATIM.
# It shares this file's basename (build_arrows.py), so load it by absolute path
# to avoid the name clash rather than via a plain `import`.
def _load_llada_builder():
    path = os.path.join(ROOT, "steering", "build_arrows.py")
    spec = importlib.util.spec_from_file_location("llada_build_arrows", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_LB = _load_llada_builder()
select_heldout = _LB.select_heldout
load_full_race = _LB.load_full_race
eval_race_keys = _LB.eval_race_keys
sweep400_keys = _LB.sweep400_keys
BLACK_TAGS = _LB.BLACK_TAGS

import common_dream as C  # noqa: E402

OUT_PT = os.path.join(HERE, "arrows.pt")

N_LAYERS = C.N_LAYERS   # 28
CAP = 400
DEVICE = "cuda"


def build(model, tok, heldout):
    """Forward each (prompt+answer) pair, mean-pool block residuals over the
    answer span at all 28 layers, return stacked (n,28,H) black-minus-other."""
    blocks = bbq_eval.resolve_module(model, C.BLOCKS_PATH)
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
        """Mean over the answer-text span at every block -> (28,H).  Reads HIDDEN
        states (not logits), so Dream's next-token logits shift does NOT apply."""
        pref = tok(chat_prompt)["input_ids"]
        plen = len(pref)
        full = tok(chat_prompt + answer_text)["input_ids"]
        assert full[:plen] == pref, (
            f"BPE merge across the prompt/answer boundary: tok(chat+answer)[:{plen}] "
            f"!= tok(chat) for answer {answer_text!r} -- the answer span would be "
            f"shifted; refusing to pool a misaligned span.")
        ids = torch.tensor(full, device=DEVICE).unsqueeze(0)
        captured.clear()
        model(ids, use_cache=False)   # Dream custom forward: no KV cache
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
    return torch.stack(stack, dim=0)   # (n,28,H)


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

    # Dump the exact positive/negative contrast pairs (inspectable, no GPU to read).
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

    model, tok = C.load_model_tok(device=DEVICE)
    print(f"[build] model on {model.device}", flush=True)

    S = build(model, tok, heldout)      # (n,28,H)
    r = S.mean(dim=0)                    # (28,H) RAW (not normed)
    per_raw = [float(r[l].norm()) for l in range(N_LAYERS)]
    per_mean = [float(S[:, l, :].norm(dim=1).mean()) for l in range(N_LAYERS)]

    torch.save({
        "r": r,
        "n_layers": N_LAYERS,
        "n_items": len(heldout),
        "capped": capped,
        "cap": CAP,
        "d_model": C.D_MODEL,
        "per_layer_raw_norm": per_raw,
        "per_layer_mean_diff_norm": per_mean,
        "method": "anchored_caa_text_all_layers",
        "model": "Dream-v0-Instruct-7B",
        "source": "bbq_race_ethnicity_heldout_disjoint_seed42_and_sweep400",
        "black_tags": sorted(BLACK_TAGS),
        "excluded_seed42_keys": len(seed_keys),
        "excluded_sweep400_keys": len(sw_keys),
    }, OUT_PT)
    print(f"[build] SAVED -> {OUT_PT}", flush=True)
    print(f"[build] per_layer_raw_norm={[round(x,2) for x in per_raw]}", flush=True)
    print("[build] DONE", flush=True)


# --------------------------------------------------------------------------- #
# Offline self-test: item-selection disjointness logic (no GPU, no data files). #
# --------------------------------------------------------------------------- #
def _mk_row(eid, qi, cc, tags):
    """Synthetic BBQ row: tags = [tag0,tag1,tag2] for ans0/1/2 answer_info."""
    return {
        "example_id": eid, "question_index": qi, "context_condition": cc,
        "context": "c", "question": "q",
        "ans0": "a0", "ans1": "a1", "ans2": "a2",
        "answer_info": {f"ans{k}": [f"w{k}", tags[k]] for k in range(3)},
    }


def _selftest():
    ok = True

    def check(name, cond):
        nonlocal ok
        ok &= bool(cond)
        print(f"[selftest-arrows] {name:52s} : {'PASS' if cond else 'FAIL'}")

    # Full pool: mix of ambig/disambig, black-count 0/1/2, and an excluded key.
    full = [
        _mk_row(1, "q1", "ambig",   ["black", "asian", "unknown"]),   # keep
        _mk_row(2, "q2", "ambig",   ["latino", "black", "unknown"]),  # keep
        _mk_row(3, "q3", "disambig",["black", "asian", "unknown"]),   # drop: disambig
        _mk_row(4, "q4", "ambig",   ["black", "african", "unknown"]), # drop: 2 black
        _mk_row(5, "q5", "ambig",   ["asian", "latino", "unknown"]),  # drop: 0 black
        _mk_row(6, "q6", "ambig",   ["black", "asian", "unknown"]),   # drop: excluded
    ]
    exclude = {(6, "q6")}
    picked = select_heldout(full, exclude)
    pk = {(int(r.get("example_id", -1)), str(r.get("question_index", ""))) for r, _, _ in picked}

    check("selects exactly the 2 valid held-out items", pk == {(1, "q1"), (2, "q2")})
    check("picked disjoint from exclude_keys", pk.isdisjoint(exclude))
    check("every picked item is ambiguous",
          all(r.get("context_condition") == "ambig" for r, _, _ in picked))
    # black idx != other idx, and each is a real 0..2 option.
    idx_ok = all(0 <= b < 3 and 0 <= o < 3 and b != o for _, b, o in picked)
    check("black_idx/other_idx valid and distinct", idx_ok)
    # black_idx must carry a Black tag; other must not, and must not be unknown.
    def tag(row, k):
        return str(row["answer_info"][f"ans{k}"][-1]).strip().lower()
    tag_ok = all(tag(r, b) in BLACK_TAGS and tag(r, o) not in BLACK_TAGS
                 and tag(r, o) != "unknown" for r, b, o in picked)
    check("positive=Black tag, negative=non-Black non-unknown", tag_ok)

    print(f"[selftest-arrows] OVERALL: {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true",
                    help="offline item-selection disjointness check (no GPU)")
    args = ap.parse_args()
    if args.selftest:
        sys.exit(0 if _selftest() else 1)
    main()
