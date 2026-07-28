#!/usr/bin/env python
"""multirace/build_arrows.py -- per-target "prefer <target> option" arrows r(k)
for LLaDA-8B-Instruct.  Port of steering/build_arrows.py parameterized by
--target in {white, asian, latino, arab} (black = existing steering/arrows.pt,
NOT rebuilt here).

For every transformer block k = 0..31, the answer-text-anchored diff-in-means:

    r(k) = mean_items( h_target(k) - h_other(k) )

where h_*(k) is the block-k residual, masked-mean over the answer-text token
span of a single clean forward on  chat_prompt + answer_text  (identical
pooling to steering/build_arrows.py).

ITEMS: the target's HELDOUT keys from multirace/items_manifest.json (written by
make_items.py, seed 42) -- the manifest is the single source of truth, nothing
is recomputed here, so the item set is deterministic and disjoint from both the
target's own _sweep400_<target>.jsonl eval items and the Black experiment's
seed-42 / sweep400 keys.

REUSE: steering/build_arrows.py's load_full_race is imported by absolute path
(importlib, same pattern as dream/build_arrows.py -- this file shares its
basename); pooling/hook code mirrors it line-for-line via eval/bbq_eval.py.

Saves RAW (32,4096) r -> multirace/arrows_<target>.pt (+ metadata; gitignored).

Run on ONE GPU:  CUDA_VISIBLE_DEVICES=3 python multirace/build_arrows.py --target asian
Offline check:   python multirace/build_arrows.py --selftest
"""
import argparse
import importlib.util
import json
import os
import sys

import torch

ROOT = "/home/lukas/users/shashmi/dlm_bias"
sys.path.insert(0, os.path.join(ROOT, "eval"))

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import bbq_eval  # noqa: E402
from targets import TARGET_TAGS, NEW_TARGETS, target_idx_of, unk_idx_of  # noqa: E402
from make_items import load_llada_builder, row_key, MANIFEST  # noqa: E402

MODEL_PATH = os.path.join(ROOT, "LLaDA-8B-Instruct")
N_LAYERS = 32
DEVICE = "cuda"


def heldout_from_manifest(target):
    """(row, target_idx, other_idx) triples for the manifest's heldout keys,
    in manifest order. other = the non-target, non-unknown option."""
    with open(MANIFEST) as f:
        mani = json.load(f)
    tinfo = mani["targets"][target]
    assert sorted(TARGET_TAGS[target]) == tinfo["tags"], \
        f"[{target}] manifest tags drifted from targets.py -- rerun make_items.py"
    keys = [tuple(k) for k in tinfo["heldout_keys"]]
    by_key = {}
    for r in load_llada_builder().load_full_race():
        by_key[row_key(r)] = r
    triples = []
    for key in keys:
        row = by_key[key]  # KeyError = manifest/cache drift, fail loud
        tidx = target_idx_of(row, target)
        unk = unk_idx_of(row)
        oidx = next((k for k in range(3) if k != tidx and k != unk), None)
        assert tidx is not None and unk is not None and oidx is not None, \
            f"[{target}] row {key} no longer qualifies"
        triples.append((row, tidx, oidx))
    return triples, mani


def main(target):
    print(f"[build:{target}] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}",
          flush=True)
    heldout, mani = heldout_from_manifest(target)
    print(f"[build:{target}] heldout={len(heldout)} (manifest seed={mani['seed']})", flush=True)
    out_pt = os.path.join(HERE, f"arrows_{target}.pt")

    # Dump the exact positive/negative contrast pairs (inspectable, no GPU to read).
    ex_path = os.path.join(HERE, f"direction_examples_{target}.jsonl")
    with open(ex_path, "w") as f:
        for row, tidx, oidx in heldout:
            f.write(json.dumps({
                "example_id": row.get("example_id"), "question_index": row.get("question_index"),
                "context": row.get("context"), "question": row.get("question"),
                "options": {"A": row.get("ans0"), "B": row.get("ans1"), "C": row.get("ans2")},
                "positive_text": str(row[f"ans{tidx}"]).strip(),
                "positive_tag": bbq_eval.get_answer_info(row, tidx)[-1],
                "negative_text": str(row[f"ans{oidx}"]).strip(),
                "negative_tag": bbq_eval.get_answer_info(row, oidx)[-1],
            }, ensure_ascii=False) + "\n")
    print(f"[build:{target}] wrote pos/neg contrast pairs -> {ex_path}", flush=True)

    from transformers import AutoModel, AutoTokenizer
    model = (AutoModel.from_pretrained(MODEL_PATH, trust_remote_code=True,
                                       torch_dtype=torch.bfloat16).to(DEVICE).eval())
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    print(f"[build:{target}] model on {model.device}", flush=True)

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
        for i, (row, tidx, oidx) in enumerate(heldout):
            prompt = bbq_eval.build_prompt(row)
            chat = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                           add_generation_prompt=True, tokenize=False)
            ht = all_layer_hidden(chat, str(row[f"ans{tidx}"]).strip())
            ho = all_layer_hidden(chat, str(row[f"ans{oidx}"]).strip())
            stack.append(ht - ho)
            if (i + 1) % 50 == 0 or (i + 1) == len(heldout):
                print(f"[build:{target}] {i+1}/{len(heldout)}", flush=True)
    finally:
        for h in handles:
            h.remove()

    S = torch.stack(stack, dim=0)   # (n,32,H)
    r = S.mean(dim=0)               # (32,H)  RAW (not normed)
    per_raw = [float(r[l].norm()) for l in range(N_LAYERS)]
    per_mean = [float(S[:, l, :].norm(dim=1).mean()) for l in range(N_LAYERS)]

    torch.save({
        "r": r,
        "target": target,
        "target_tags": sorted(TARGET_TAGS[target]),
        "n_layers": N_LAYERS,
        "n_items": len(heldout),
        "per_layer_raw_norm": per_raw,
        "per_layer_mean_diff_norm": per_mean,
        "method": "anchored_caa_text_all_layers",
        "source": f"multirace_manifest_heldout_seed{mani['seed']}_disjoint_black_seed42_and_sweep400",
        "manifest": MANIFEST,
    }, out_pt)
    print(f"[build:{target}] SAVED -> {out_pt}", flush=True)
    print(f"[build:{target}] per_layer_raw_norm={[round(x, 2) for x in per_raw]}", flush=True)
    print(f"[build:{target}] DONE", flush=True)


# --------------------------------------------------------------------------- #
# Offline self-test: manifest -> triples resolution (no GPU, real manifest).   #
# --------------------------------------------------------------------------- #
def _selftest():
    ok = True

    def check(name, cond):
        nonlocal ok
        ok &= bool(cond)
        print(f"[selftest-marrows] {name:56s} : {'PASS' if cond else 'FAIL'}")

    lb = load_llada_builder()
    exclude = lb.eval_race_keys() | lb.sweep400_keys()
    for target in NEW_TARGETS:
        triples, mani = heldout_from_manifest(target)
        tinfo = mani["targets"][target]
        check(f"{target}: heldout resolves ({len(triples)} triples)",
              len(triples) == tinfo["n_heldout"])
        keys = {row_key(r) for r, _, _ in triples}
        with open(tinfo["eval_file"]) as f:
            ev_keys = {row_key(json.loads(l)) for l in f if l.strip()}
        check(f"{target}: heldout disjoint from eval file", keys.isdisjoint(ev_keys))
        check(f"{target}: heldout disjoint from Black-exp keys", keys.isdisjoint(exclude))
        tag_ok = all(
            str(bbq_eval.get_answer_info(r, t)[-1]).strip().lower() in TARGET_TAGS[target]
            and str(bbq_eval.get_answer_info(r, o)[-1]).strip().lower()
            not in (TARGET_TAGS[target] | {"unknown"})
            for r, t, o in triples)
        check(f"{target}: positive=target tag, negative=non-target non-unknown", tag_ok)

    print(f"[selftest-marrows] OVERALL: {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target", choices=list(NEW_TARGETS))
    ap.add_argument("--selftest", action="store_true",
                    help="offline manifest-resolution check (no GPU)")
    args = ap.parse_args()
    if args.selftest:
        sys.exit(0 if _selftest() else 1)
    if not args.target:
        ap.error("--target required (or --selftest)")
    main(args.target)
