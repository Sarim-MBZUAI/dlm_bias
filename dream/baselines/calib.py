#!/usr/bin/env python
"""dream/baselines/calib.py -- activation collection for the fit-based Dream baselines.

Port of baselines/calib.py to Dream-v0-Instruct-7B.  Builds the labelled (Black vs
other) calibration responses the OT / AURA / ITI method files fit from, using the
SAME contamination-safe held-out BBQ Race_ethnicity contrast set as
dream/build_arrows.py (disjoint from the seed-42 n=1000 eval keys AND the 400-item
_sweep400.jsonl keys), and the SAME answer-text-span masked-mean pooling on a
single clean forward of the fully materialized chat_prompt+answer.

Granularities (`where`) -- Dream analogues verified in common_dream.py:
  block       block_path(k) OUTPUT[0]     (residual, 3584 = D_MODEL).
              hidden_from_output(out) -- the SAME tensor dream/build_arrows pools.
  mlp_hidden  INPUT to down_proj_path(k)  (18944 = D_MLP; gated MLP activation).
              In Dream's MLP  down_proj(act(gate_proj(x))*up_proj(x))  there is NO
              module whose OUTPUT is the gated activation, so we capture it as the
              INPUT of down_proj (inp[0]) -- LLaDA ff_out analogue.
  attn_head   INPUT to o_proj_path(k)     (3584 = N_HEADS*HEAD_DIM = 28*128).
              o_proj's INPUT is the concat of the 28 QUERY-head outputs (GQA has 4
              kv heads, but the o_proj input is the full 3584-d query-head concat),
              reshapeable to (...,28,128) -- LLaDA attn_out analogue.

Pooling matches dream/build_arrows.all_layer_hidden:
    plen  = len(tok(chat_prompt).input_ids)
    ids   = tok(chat_prompt + answer_text).input_ids
    start = plen if seq_len > plen else seq_len - 1
    pooled_layer_k = captured[k][0][start:, :].float().mean(dim=0)
Hidden states are read (not logits), so Dream's next-token logits shift is N/A.

collect_activations() NEEDS A GPU + the model.  --selftest only validates the pure
pooling/labeling logic on synthetic captured tensors (no forward, no GPU).

CLI:
    python dream/baselines/calib.py --selftest                # offline logic check
    CUDA_VISIBLE_DEVICES=4 python dream/baselines/calib.py --fit --where block  # GPU
"""
import argparse
import importlib.util
import os
import sys

import torch

ROOT = "/home/lukas/users/shashmi/dlm_bias"
sys.path.insert(0, os.path.join(ROOT, "eval"))
sys.path.insert(0, os.path.join(ROOT, "steering"))

_HERE = os.path.dirname(os.path.abspath(__file__))
_DREAM = os.path.dirname(_HERE)
sys.path.insert(0, _DREAM)   # common_dream

import bbq_eval        # noqa: E402
import common_dream as C  # noqa: E402


# Reuse dream/build_arrows.py's contamination-safe selection (same tree as this
# file), loaded by absolute path to avoid the build_arrows basename clash with
# steering/build_arrows.py.
def _load_dream_build_arrows():
    path = os.path.join(_DREAM, "build_arrows.py")
    spec = importlib.util.spec_from_file_location("dream_build_arrows", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_DBA = _load_dream_build_arrows()

CACHE_DIR = os.path.join(_HERE, "cache")
WHERE_CHOICES = ("block", "mlp_hidden", "attn_head")
FEAT_DIM = {"block": C.D_MODEL, "mlp_hidden": C.D_MLP, "attn_head": C.D_MODEL}
CAP = 400


# --------------------------------------------------------------------------- #
# Held-out contrast set (identical selection to dream/build_arrows.py).        #
# --------------------------------------------------------------------------- #
def heldout_items(tok=None, cap=CAP):
    """Return the contamination-safe held-out BBQ-Race contrast items.

    Ambiguous, exactly-one-Black-option, Race_ethnicity, disjoint from BOTH the
    seed-42 eval keys and _sweep400.jsonl (dream/build_arrows select_heldout /
    load_full_race / eval_race_keys / sweep400_keys), capped at `cap`.  A
    tokenizer is optional and only needed to materialize the chat_prompt string.
    """
    full = _DBA.load_full_race()
    exclude = _DBA.eval_race_keys() | _DBA.sweep400_keys()
    heldout = _DBA.select_heldout(full, exclude)
    if len(heldout) > cap:
        heldout = heldout[:cap]
    items = []
    for row, bidx, oidx in heldout:
        prompt = bbq_eval.build_prompt(row)
        chat = None
        if tok is not None:
            chat = tok.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True, tokenize=False)
        items.append({
            "row": row, "black_idx": bidx, "other_idx": oidx,
            "black_answer_text": str(row[f"ans{bidx}"]).strip(),
            "other_answer_text": str(row[f"ans{oidx}"]).strip(),
            "chat_prompt": chat,
        })
    return items


# --------------------------------------------------------------------------- #
# Pure pooling helper (masked-mean over the answer-text span).                 #
# Re-implements dream/build_arrows.all_layer_hidden's pooling (a closure).      #
# --------------------------------------------------------------------------- #
def span_start(plen, seq_len):
    """start index of the answer-text span (build_arrows: plen if seq>plen else seq-1)."""
    return plen if seq_len > plen else seq_len - 1


def _pool_captured(captured, start, n_layers=C.N_LAYERS):
    """captured: {layer -> (B,T,feat)}. Return (n_layers, feat) masked-mean over
    tokens [start:] of batch element 0."""
    feat = captured[0].shape[-1]
    out = torch.empty(n_layers, feat, dtype=torch.float32)
    for li in range(n_layers):
        out[li] = captured[li][0].to(torch.float32)[start:, :].mean(dim=0).cpu()
    return out


# --------------------------------------------------------------------------- #
# Collection (NEEDS GPU + model).                                             #
# --------------------------------------------------------------------------- #
def collect_activations(where, model=None, tok=None, items=None, cap=CAP,
                        save=True, out_path=None):
    """Collect labelled Black/other pooled activations at granularity `where`.

    For each held-out contrast item, run TWO clean forwards (chat_prompt +
    black_answer, and + other_answer), pool each with the build_arrows masked-mean,
    and label Black=1 / other=0.  Returns {acts (2n,28,feat), labels (2n,), ...}
    and (if save) writes cache/calib_<where>.pt.  NEEDS A GPU."""
    assert where in WHERE_CHOICES, f"where must be one of {WHERE_CHOICES}"
    if model is None or tok is None:
        model, tok = C.load_model_tok()
    if items is None:
        items = heldout_items(tok=tok, cap=cap)

    blocks = C.resolve_module(model, C.BLOCKS_PATH)
    assert len(blocks) == C.N_LAYERS, f"expected {C.N_LAYERS} blocks, got {len(blocks)}"

    captured = {}
    handles = []
    for li, blk in enumerate(blocks):
        if where == "block":
            target = blk
        elif where == "mlp_hidden":
            target = C.resolve_module(model, C.down_proj_path(li))
        else:  # attn_head
            target = C.resolve_module(model, C.o_proj_path(li))

        def mk(li):
            def hook(mod, inp, out):
                if where == "block":
                    captured[li] = bbq_eval.hidden_from_output(out)
                else:
                    captured[li] = inp[0]   # INPUT to down_proj / o_proj
            return hook
        handles.append(target.register_forward_hook(mk(li)))

    @torch.no_grad()
    def pooled(chat, answer_text):
        plen = len(tok(chat)["input_ids"])
        ids = torch.tensor(tok(chat + answer_text)["input_ids"],
                           device=model.device).unsqueeze(0)
        captured.clear()
        model(ids, use_cache=False)   # Dream custom forward: no KV cache
        start = span_start(plen, ids.shape[1])
        return _pool_captured(captured, start)

    acts, labels = [], []
    try:
        for i, it in enumerate(items):
            chat = it["chat_prompt"]
            acts.append(pooled(chat, it["black_answer_text"])); labels.append(1)
            acts.append(pooled(chat, it["other_answer_text"])); labels.append(0)
            if (i + 1) % 50 == 0 or (i + 1) == len(items):
                print(f"[calib:{where}] {i+1}/{len(items)}", flush=True)
    finally:
        for h in handles:
            h.remove()

    A = torch.stack(acts, dim=0)                 # (2n, 28, feat)
    L = torch.tensor(labels, dtype=torch.int64)  # (2n,)
    blob = {
        "acts": A, "labels": L, "where": where,
        "feat": A.shape[-1], "n_layers": C.N_LAYERS, "n_items": len(items),
        "n_heads": C.N_HEADS, "d_head": C.HEAD_DIM,
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

    check("span_start normal (seq>plen) == plen", span_start(5, 9) == 5)
    check("span_start degenerate (seq<=plen) == seq-1", span_start(9, 9) == 8)

    n_layers, T, feat = C.N_LAYERS, 7, 4
    start = 3
    captured = {li: torch.randn(1, T, feat) for li in range(n_layers)}
    pooled = _pool_captured(captured, start, n_layers)
    manual0 = captured[0][0, start:, :].mean(dim=0)
    check("pool shape (N_LAYERS, feat)", tuple(pooled.shape) == (n_layers, feat))
    check("pool == mean over [start:] tokens", torch.allclose(pooled[0], manual0, atol=1e-6))
    manualL = captured[n_layers - 1][0, start:, :].mean(dim=0)
    check("pool correct on last layer", torch.allclose(pooled[-1], manualL, atol=1e-6))

    labels = []
    for _ in range(3):
        labels.append(1); labels.append(0)
    L = torch.tensor(labels)
    check("labels alternate 1,0 per item (Black,other)",
          L.tolist() == [1, 0, 1, 0, 1, 0]
          and int((L == 1).sum()) == 3 and int((L == 0).sum()) == 3)

    check("attn_head feat 3584 reshapes to (28,128)",
          FEAT_DIM["attn_head"] == C.N_HEADS * C.HEAD_DIM)
    check("feat dims block/mlp/attn = 3584/18944/3584",
          (FEAT_DIM["block"], FEAT_DIM["mlp_hidden"], FEAT_DIM["attn_head"])
          == (3584, 18944, 3584))

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
