#!/usr/bin/env python
"""baselines/calib.py -- activation collection for the fit-based AcT baselines.

Builds the labelled (Black vs other) calibration responses the OT / AURA method
files fit their per-neuron transports and gates from.  Uses the SAME
contamination-safe held-out BBQ Race_ethnicity contrast set that
steering/build_arrows.py uses (disjoint from the seed-42 n=1000 eval keys AND
the 400-item _sweep400.jsonl keys), and the SAME answer-text-span masked-mean
pooling on a single clean forward of the fully materialized prompt+answer (LLaDA
is a masked-diffusion LM; no MASK tokens are present in this forward).

Granularities (`where`), with the exact LLaDA submodule verified against
LLaDA-8B-Instruct/modeling_llada.py (LLaDALlamaBlock, block_type="llama"):

  block       blocks[k] OUTPUT[0]  (residual, H=4096, config d_model).
              hidden_from_output(out) -- same tensor build_arrows pools.
  mlp_hidden  INPUT to blocks[k].ff_out  (H=12288, config mlp_hidden_size).
              In forward() this is  x = act(ff_proj(x)) * up_proj(x)  and then
              x = ff_out(x)  (modeling_llada.py:924-930).  There is NO module
              whose OUTPUT is this gated activation, so we capture it as the
              INPUT of ff_out (inp[0]).
  attn_head   INPUT to blocks[k].attn_out  (H=4096, reshapeable to
              n_heads=32 x d_head=128).  In attention() the per-head outputs are
              re-assembled  att = att.transpose(1,2).contiguous().view(B,T,C)
              and then projected by attn_out (modeling_llada.py:720-724); the
              per-head activation is exactly that INPUT (inp[0]), which a method
              file reshapes to (...,32,128).

Pooling matches build_arrows.all_layer_hidden (build_arrows.py:159-172):
    plen  = len(tok(chat_prompt).input_ids)
    ids   = tok(chat_prompt + answer_text).input_ids
    start = plen if seq_len > plen else seq_len - 1
    pooled_layer_k = captured[k][0][start:, :].float().mean(dim=0)

NOTE ON FAITHFULNESS: build_arrows.all_layer_hidden is a CLOSURE nested inside
build_arrows.main() and is NOT importable; its pooling logic is re-implemented
here verbatim (see _pool_captured) and cited.  The held-out SELECTION helpers
(select_heldout, load_full_race, eval_race_keys, sweep400_keys) ARE module-level
and are imported, not reimplemented.

collect_activations() NEEDS A GPU + the model.  --selftest only validates the
pure pooling/labeling logic on synthetic captured tensors (no forward, no GPU).

CLI:
    python -m baselines.calib --selftest                 # offline logic check
    python -m baselines.calib --fit --where block        # NEEDS GPU (do not run here)
"""
import argparse
import os
import sys

import torch

# Harness import (ROOT = MAIN tree; mirrors steering/pid_steer.py:49-51).
ROOT = os.environ.get("DLM_BIAS_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "eval"))
sys.path.insert(0, os.path.join(ROOT, "steering"))
import bbq_eval  # noqa: E402
import build_arrows  # noqa: E402  (select_heldout + contrast helpers, module-level)

from common import (  # noqa: E402
    resolve_module,
    hidden_from_output,
    BLOCKS_PATH,
    N_LAYERS,
    load_model,
    CACHE_DIR,
    H_MODEL,
    H_MLP,
    N_HEADS,
    D_HEAD,
)

WHERE_CHOICES = ("block", "mlp_hidden", "attn_head")
FEAT_DIM = {"block": H_MODEL, "mlp_hidden": H_MLP, "attn_head": H_MODEL}
CAP = 400


# --------------------------------------------------------------------------- #
# Held-out contrast set (identical selection to build_arrows.py).              #
# --------------------------------------------------------------------------- #
def heldout_items(tok=None, cap=CAP):
    """Return the contamination-safe held-out BBQ-Race contrast items.

    Selection is build_arrows' own: ambiguous, exactly-one-Black-option,
    Race_ethnicity, disjoint from BOTH the seed-42 eval keys and _sweep400.jsonl
    (build_arrows.select_heldout, .load_full_race, .eval_race_keys,
    .sweep400_keys), capped at `cap` (build_arrows CAP=400).

    Each returned dict has:
        row, black_idx, other_idx,
        black_answer_text, other_answer_text,
        chat_prompt  (None unless a tokenizer is passed -- apply_chat_template).

    Reads cached BBQ jsonl only; no GPU/network (files are cached under
    data/bbq_cache/).  A tokenizer is optional and only needed to materialize
    the chat_prompt string for collect_activations().
    """
    full = build_arrows.load_full_race()
    exclude = build_arrows.eval_race_keys() | build_arrows.sweep400_keys()
    heldout = build_arrows.select_heldout(full, exclude)
    if len(heldout) > cap:
        heldout = heldout[:cap]
    items = []
    for row, bidx, oidx in heldout:
        prompt = bbq_eval.build_prompt(row)
        chat = None
        if tok is not None:
            chat = tok.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True, tokenize=False,
            )
        items.append({
            "row": row,
            "black_idx": bidx,
            "other_idx": oidx,
            "black_answer_text": str(row[f"ans{bidx}"]).strip(),
            "other_answer_text": str(row[f"ans{oidx}"]).strip(),
            "chat_prompt": chat,
        })
    return items


# --------------------------------------------------------------------------- #
# Pure pooling helper (masked-mean over the answer-text span).                 #
# Re-implements build_arrows.all_layer_hidden's pooling (build_arrows.py:167-172)
# because that function is a non-importable closure.                           #
# --------------------------------------------------------------------------- #
def span_start(plen, seq_len):
    """start index of the answer-text span (build_arrows.py:167)."""
    return plen if seq_len > plen else seq_len - 1


def _pool_captured(captured, start, n_layers=N_LAYERS):
    """captured: {layer -> (B,T,feat)}.  Return (n_layers, feat) masked-mean over
    tokens [start:] of batch element 0 (build_arrows.py:169-172)."""
    feat = captured[0].shape[-1]
    out = torch.empty(n_layers, feat, dtype=torch.float32)
    for li in range(n_layers):
        out[li] = captured[li][0].to(torch.float32)[start:, :].mean(dim=0).cpu()
    return out


# --------------------------------------------------------------------------- #
# Collection (NEEDS GPU + model).                                              #
# --------------------------------------------------------------------------- #
def collect_activations(where, model=None, tok=None, items=None, cap=CAP,
                        save=True, out_path=None):
    """Collect labelled Black/other pooled activations at granularity `where`.

    For each held-out contrast item, run TWO clean forwards (materialized
    chat_prompt + black_answer_text, and + other_answer_text), pool each with the
    build_arrows masked-mean, and label Black=1 / other=0.

    Returns a dict:
        acts   : (2*n_items, N_LAYERS, feat)  float32
        labels : (2*n_items,)                 int64 (1=Black, 0=other)
        where, feat, n_items, source
    and (if save) writes it to cache/calib_<where>.pt.

    NEEDS A GPU.  See --selftest for the offline logic check.
    """
    assert where in WHERE_CHOICES, f"where must be one of {WHERE_CHOICES}"
    if model is None or tok is None:
        model, tok = load_model()
    if items is None:
        items = heldout_items(tok=tok, cap=cap)

    blocks = resolve_module(model, BLOCKS_PATH)
    assert len(blocks) == N_LAYERS, f"expected {N_LAYERS} blocks, got {len(blocks)}"

    captured = {}
    handles = []
    for li, blk in enumerate(blocks):
        if where == "block":
            target = blk
        elif where == "mlp_hidden":
            target = blk.ff_out
        else:  # attn_head
            target = blk.attn_out

        def mk(li):
            def hook(mod, inp, out):
                if where == "block":
                    captured[li] = hidden_from_output(out)
                else:
                    captured[li] = inp[0]   # INPUT to ff_out / attn_out
            return hook
        handles.append(target.register_forward_hook(mk(li)))

    @torch.no_grad()
    def pooled(chat, answer_text):
        plen = len(tok(chat)["input_ids"])
        ids = torch.tensor(tok(chat + answer_text)["input_ids"],
                           device=model.device).unsqueeze(0)
        captured.clear()
        model(ids)
        start = span_start(plen, ids.shape[1])
        return _pool_captured(captured, start)

    acts, labels = [], []
    try:
        for i, it in enumerate(items):
            chat = it["chat_prompt"]
            hb = pooled(chat, it["black_answer_text"])
            ho = pooled(chat, it["other_answer_text"])
            acts.append(hb); labels.append(1)
            acts.append(ho); labels.append(0)
            if (i + 1) % 50 == 0 or (i + 1) == len(items):
                print(f"[calib:{where}] {i+1}/{len(items)}", flush=True)
    finally:
        for h in handles:
            h.remove()

    A = torch.stack(acts, dim=0)                 # (2n, 32, feat)
    L = torch.tensor(labels, dtype=torch.int64)  # (2n,)
    blob = {
        "acts": A, "labels": L, "where": where,
        "feat": A.shape[-1], "n_layers": N_LAYERS, "n_items": len(items),
        "n_heads": N_HEADS, "d_head": D_HEAD,
        "source": "bbq_race_ethnicity_heldout_disjoint_seed42_and_sweep400",
    }
    if save:
        out_path = out_path or os.path.join(CACHE_DIR, f"calib_{where}.pt")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        torch.save(blob, out_path)
        print(f"[calib:{where}] SAVED {tuple(A.shape)} -> {out_path}", flush=True)
    return blob


def load_calib(where, path=None):
    """Load a previously fitted cache/calib_<where>.pt."""
    path = path or os.path.join(CACHE_DIR, f"calib_{where}.pt")
    return torch.load(path, map_location="cpu")


# --------------------------------------------------------------------------- #
# Offline self-test: pooling + labeling on synthetic captured tensors.         #
# --------------------------------------------------------------------------- #
def _selftest():
    torch.manual_seed(0)
    ok = True

    def check(name, cond):
        nonlocal ok
        ok &= bool(cond)
        print(f"[selftest-calib] {name:46s} : {'PASS' if cond else 'FAIL'}")

    # span_start logic (build_arrows.py:167).
    check("span_start normal (seq>plen) == plen", span_start(5, 9) == 5)
    check("span_start degenerate (seq<=plen) == seq-1", span_start(9, 9) == 8)

    # _pool_captured: masked-mean over [start:] on every layer.
    n_layers, T, feat = N_LAYERS, 7, 4
    start = 3
    captured = {li: torch.randn(1, T, feat) for li in range(n_layers)}
    pooled = _pool_captured(captured, start, n_layers)
    manual0 = captured[0][0, start:, :].mean(dim=0)
    check("pool shape (N_LAYERS, feat)", tuple(pooled.shape) == (n_layers, feat))
    check("pool == mean over [start:] tokens", torch.allclose(pooled[0], manual0, atol=1e-6))
    manualL = captured[n_layers - 1][0, start:, :].mean(dim=0)
    check("pool correct on last layer", torch.allclose(pooled[-1], manualL, atol=1e-6))

    # labeling order matches collect_activations (Black=1 pushed before other=0).
    labels = []
    for _ in range(3):
        labels.append(1); labels.append(0)
    L = torch.tensor(labels)
    check("labels alternate 1,0 per item (Black,other)",
          L.tolist() == [1, 0, 1, 0, 1, 0]
          and int((L == 1).sum()) == 3 and int((L == 0).sum()) == 3)

    # attn_head reshape sanity: 4096 flattens to (n_heads, d_head).
    check("attn_head feat 4096 reshapes to (32,128)",
          FEAT_DIM["attn_head"] == N_HEADS * D_HEAD)
    check("feat dims block/mlp/attn = 4096/12288/4096",
          (FEAT_DIM["block"], FEAT_DIM["mlp_hidden"], FEAT_DIM["attn_head"])
          == (4096, 12288, 4096))

    print(f"[selftest-calib] OVERALL: {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true",
                    help="offline pooling/labeling check on synthetic tensors (no GPU)")
    ap.add_argument("--fit", action="store_true",
                    help="collect activations (NEEDS GPU + model)")
    ap.add_argument("--where", choices=WHERE_CHOICES, default="block")
    ap.add_argument("--cap", type=int, default=CAP)
    args = ap.parse_args()

    if args.selftest:
        sys.exit(0 if _selftest() else 1)
    if args.fit:
        collect_activations(args.where, cap=args.cap)
        return
    ap.error("nothing to do: pass --selftest (offline) or --fit (GPU)")


if __name__ == "__main__":
    main()
