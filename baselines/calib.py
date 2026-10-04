#!/usr/bin/env python
"""Activation collection for the fit-based steering baselines.

Builds the labelled (target vs other) calibration responses the OT / AURA
method files fit their per-neuron transports and gates from.  Uses the same
contamination-safe held-out BBQ contrast set as steering/build_arrows.py
(disjoint from the seed-42 n=1000 eval keys and the 400-item _sweep400.jsonl
keys), and the same answer-text-span masked-mean pooling on a single clean
forward of the fully materialized prompt+answer (no MASK tokens present).

Granularities (`where`), per LLaDA-8B-Instruct/modeling_llada.py
(LLaDALlamaBlock, block_type="llama"):

  block       blocks[k] OUTPUT[0]  (residual, H=4096, config d_model);
              the same tensor build_arrows pools.
  mlp_hidden  INPUT to blocks[k].ff_out  (H=12288, config mlp_hidden_size).
              In forward() x = act(ff_proj(x)) * up_proj(x), then
              x = ff_out(x).  No module outputs this gated activation, so it
              is captured as the INPUT of ff_out (inp[0]).
  attn_head   INPUT to blocks[k].attn_out  (H=4096, reshapeable to
              n_heads=32 x d_head=128): the re-assembled per-head outputs
              att.transpose(1,2).contiguous().view(B,T,C) before projection.

Pooling matches build_arrows.all_layer_hidden:
    plen  = len(tok(chat_prompt).input_ids)
    ids   = tok(chat_prompt + answer_text).input_ids
    start = plen if seq_len > plen else seq_len - 1
    pooled_layer_k = captured[k][0][start:, :].float().mean(dim=0)
build_arrows.all_layer_hidden is a non-importable closure, so its pooling is
re-implemented here (_pool_captured); the held-out selection helpers are
imported from build_arrows.

Targets: target="black" (default) uses the Race_ethnicity held-out set and
cache/calib_<where>.pt.  Other targets use the multirace manifest heldout keys
(multirace/build_arrows.py heldout_from_manifest: 400 keys per target, disjoint
from the eval files and from each other), positive = the target-tagged option's
answer text, negative = the other person's option, cached as
cache/calib_<where>_<target>.pt.  Labels: 1 = target response, 0 = other.

collect_activations() needs a GPU + the model.  --selftest validates the pure
pooling/labeling logic on synthetic tensors and, when the data caches are
present, the offline heldout selection per target.

Usage:
    python baselines/calib.py --selftest
    python baselines/calib.py --fit --where block [--target woman]   # GPU
"""
import argparse
import json
import os
import sys

import torch

# Harness import (same convention as steering/pid_steer.py).
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
    CODE_ROOT,
    H_MODEL,
    H_MLP,
    N_HEADS,
    D_HEAD,
    SUPPORTED_TARGETS,
    target_suffix,
)

WHERE_CHOICES = ("block", "mlp_hidden", "attn_head")
FEAT_DIM = {"block": H_MODEL, "mlp_hidden": H_MLP, "attn_head": H_MODEL}
CAP = 400

# Calib-blob source strings; non-black targets mirror the string
# multirace/build_arrows.py stamps on arrows_<target>.pt (all seed 42): arab is
# a Race_ethnicity pole, lowses/old are the SES/Age cross_target_disjoint poles.
SOURCE_BY_TARGET = {
    "black": "bbq_race_ethnicity_heldout_disjoint_seed42_and_sweep400",
    "woman": "multirace_gender_manifest_heldout_seed42_cross_target_disjoint",
    "man": "multirace_gender_manifest_heldout_seed42_cross_target_disjoint",
    "arab": "multirace_manifest_heldout_seed42_disjoint_black_seed42_and_sweep400",
    "lowses": "multirace_ses_manifest_heldout_seed42_cross_target_disjoint",
    "old": "multirace_age_manifest_heldout_seed42_cross_target_disjoint",
}

_MB = None


def _multirace_builder():
    """Load multirace/build_arrows.py from this code tree by absolute path
    (it shares its basename with steering/build_arrows.py, which is already
    imported as `build_arrows`)."""
    global _MB
    if _MB is None:
        import importlib.util
        path = os.path.join(CODE_ROOT, "multirace", "build_arrows.py")
        spec = importlib.util.spec_from_file_location("multirace_build_arrows", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _MB = mod
    return _MB


def calib_path(where, target="black"):
    """cache/calib_<where>.pt for black; cache/calib_<where>_<target>.pt
    otherwise."""
    return os.path.join(CACHE_DIR, f"calib_{where}{target_suffix(target)}.pt")


# --------------------------------------------------------------------------- #
# Held-out contrast set (identical selection to build_arrows.py).              #
# --------------------------------------------------------------------------- #
def heldout_items(tok=None, cap=CAP, target="black"):
    """Return the contamination-safe held-out contrast items for `target`.

    target="black" (default): selection is build_arrows' own -- ambiguous,
    exactly-one-Black-option, Race_ethnicity, disjoint from BOTH the seed-42
    eval keys and _sweep400.jsonl (build_arrows.select_heldout,
    .load_full_race, .eval_race_keys, .sweep400_keys), capped at `cap`
    (build_arrows CAP=400).

    Other targets: the multirace manifest heldout keys (400 keys per target,
    disjoint from the _sweep400_<target> eval files and from each other),
    resolved by multirace/build_arrows.heldout_from_manifest -- the same
    triples the target arrows were fitted from.  Positive = the target-tagged option's
    answer text; negative = the other-gendered person's option (the mirror of
    the black/other pairing).

    Each returned dict has:
        row, target_idx, other_idx,
        target_answer_text, other_answer_text,
        chat_prompt  (None unless a tokenizer is passed -- apply_chat_template)
    plus, for target="black" only, the aliases black_idx / black_answer_text
    (same values).

    Reads cached BBQ jsonl + committed manifests only; no GPU/network.  A
    tokenizer is optional and only needed to materialize the chat_prompt
    string for collect_activations().
    """
    if target == "black":
        full = build_arrows.load_full_race()
        exclude = build_arrows.eval_race_keys() | build_arrows.sweep400_keys()
        heldout = build_arrows.select_heldout(full, exclude)
    else:
        target_suffix(target)  # validate
        heldout, _mani = _multirace_builder().heldout_from_manifest(target)
    if len(heldout) > cap:
        heldout = heldout[:cap]
    items = []
    for row, tidx, oidx in heldout:
        prompt = bbq_eval.build_prompt(row)
        chat = None
        if tok is not None:
            chat = tok.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True, tokenize=False,
            )
        it = {
            "row": row,
            "target_idx": tidx,
            "other_idx": oidx,
            "target_answer_text": str(row[f"ans{tidx}"]).strip(),
            "other_answer_text": str(row[f"ans{oidx}"]).strip(),
            "chat_prompt": chat,
        }
        if target == "black":  # aliases (black key names)
            it["black_idx"] = tidx
            it["black_answer_text"] = it["target_answer_text"]
        items.append(it)
    return items


# --------------------------------------------------------------------------- #
# Pure pooling helper (masked-mean over the answer-text span).                 #
# Re-implements build_arrows.all_layer_hidden's pooling because that function #
# is a non-importable closure.                                                 #
# --------------------------------------------------------------------------- #
def span_start(plen, seq_len):
    """start index of the answer-text span (as in build_arrows)."""
    return plen if seq_len > plen else seq_len - 1


def _pool_captured(captured, start, n_layers=N_LAYERS):
    """captured: {layer -> (B,T,feat)}.  Return (n_layers, feat) masked-mean over
    tokens [start:] of batch element 0 (as in build_arrows)."""
    feat = captured[0].shape[-1]
    out = torch.empty(n_layers, feat, dtype=torch.float32)
    for li in range(n_layers):
        out[li] = captured[li][0].to(torch.float32)[start:, :].mean(dim=0).cpu()
    return out


# --------------------------------------------------------------------------- #
# Collection (NEEDS GPU + model).                                              #
# --------------------------------------------------------------------------- #
def collect_activations(where, model=None, tok=None, items=None, cap=CAP,
                        save=True, out_path=None, target="black"):
    """Collect labelled target/other pooled activations at granularity `where`.

    For each held-out contrast item, run TWO clean forwards (materialized
    chat_prompt + target_answer_text, and + other_answer_text), pool each with
    the build_arrows masked-mean, and label target=1 / other=0.

    Returns a dict:
        acts   : (2*n_items, N_LAYERS, feat)  float32
        labels : (2*n_items,)                 int64 (1=target, 0=other)
        where, feat, n_items, source, target
    and (if save) writes it to cache/calib_<where>.pt for black or
    cache/calib_<where>_<target>.pt otherwise.

    NEEDS A GPU.  See --selftest for the offline logic check.
    """
    assert where in WHERE_CHOICES, f"where must be one of {WHERE_CHOICES}"
    assert target in SUPPORTED_TARGETS, f"target must be one of {SUPPORTED_TARGETS}"
    if model is None or tok is None:
        model, tok = load_model()
    if items is None:
        items = heldout_items(tok=tok, cap=cap, target=target)

    blocks = resolve_module(model, BLOCKS_PATH)
    assert len(blocks) == N_LAYERS, f"expected {N_LAYERS} blocks, got {len(blocks)}"

    captured = {}
    handles = []
    for li, blk in enumerate(blocks):
        # NOTE: must not be named `target` -- that would shadow the function's
        # target parameter, which is still needed at save time (SOURCE_BY_TARGET).
        if where == "block":
            hook_mod = blk
        elif where == "mlp_hidden":
            hook_mod = blk.ff_out
        else:  # attn_head
            hook_mod = blk.attn_out

        def mk(li):
            def hook(mod, inp, out):
                if where == "block":
                    captured[li] = hidden_from_output(out)
                else:
                    captured[li] = inp[0]   # INPUT to ff_out / attn_out
            return hook
        handles.append(hook_mod.register_forward_hook(mk(li)))

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
            hb = pooled(chat, it["target_answer_text"])
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
        "source": SOURCE_BY_TARGET[target],
        "target": target,
    }
    if save:
        out_path = out_path or calib_path(where, target)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        torch.save(blob, out_path)
        print(f"[calib:{where}] SAVED {tuple(A.shape)} -> {out_path}", flush=True)
    return blob


def load_calib(where, path=None, target="black"):
    """Load a previously fitted cache/calib_<where>[_<target>].pt."""
    path = path or calib_path(where, target)
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

    # span_start logic (as in build_arrows).
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

    # --- target parameterization: cache paths (pure). ------------------------ #
    check("calib_path black == cache/calib_<where>.pt",
          all(calib_path(w, "black") == os.path.join(CACHE_DIR, f"calib_{w}.pt")
              for w in WHERE_CHOICES))
    check("calib_path gender == cache/calib_<where>_<target>.pt",
          calib_path("block", "woman").endswith("cache/calib_block_woman.pt")
          and calib_path("attn_head", "man").endswith("cache/calib_attn_head_man.pt"))
    check("black source string unchanged",
          SOURCE_BY_TARGET["black"]
          == "bbq_race_ethnicity_heldout_disjoint_seed42_and_sweep400")

    # --- heldout selection (offline data checks; SKIP when caches absent). --- #
    race_cache = os.path.join(ROOT, "data", "bbq_cache", "Race_ethnicity.jsonl")
    gender_cache = os.path.join(ROOT, "data", "bbq_cache", "Gender_identity.jsonl")
    dir_ex = os.path.join(CODE_ROOT, "steering", "direction_examples.jsonl")
    if os.path.exists(race_cache) and os.path.exists(dir_ex):
        items = heldout_items(target="black")
        with open(dir_ex) as f:
            ref = [json.loads(l) for l in f if l.strip()]
        check("black: 400 heldout items (cap)",
              len(items) == len(ref) == 400)
        check("black REGRESSION: selection == committed direction_examples.jsonl "
              "(order, keys, pos/neg answer texts)",
              all(it["row"].get("example_id") == r["example_id"]
                  and str(it["row"].get("question_index")) == str(r["question_index"])
                  and it["target_answer_text"] == r["positive_text"]
                  and it["other_answer_text"] == r["negative_text"]
                  for it, r in zip(items, ref)))
        check("black: legacy aliases black_idx/black_answer_text present+equal",
              all(it["black_idx"] == it["target_idx"]
                  and it["black_answer_text"] == it["target_answer_text"]
                  for it in items))
    else:
        print("[selftest-calib] black heldout regression: SKIP (data caches or "
              "steering/direction_examples.jsonl not available)")
    if os.path.exists(gender_cache) and os.path.exists(
            os.path.join(ROOT, "multirace", "items_manifest_gender.json")):
        mb = _multirace_builder()
        gkeys = {}
        for t in ("woman", "man"):
            other = "man" if t == "woman" else "woman"
            gitems = heldout_items(target=t)
            check(f"{t}: heldout resolves to 400 rows", len(gitems) == 400)
            check(f"{t}: positive tag in TARGET_TAGS[{t}]",
                  all(str(bbq_eval.get_answer_info(it["row"], it["target_idx"])[-1])
                      .strip().lower() in mb.TARGET_TAGS[t] for it in gitems))
            check(f"{t}: negative = other-gendered person (mirror pairing)",
                  all(str(bbq_eval.get_answer_info(it["row"], it["other_idx"])[-1])
                      .strip().lower() in mb.TARGET_TAGS[other] for it in gitems))
            check(f"{t}: no legacy black_* keys on gender items",
                  all("black_idx" not in it and "black_answer_text" not in it
                      for it in gitems))
            gkeys[t] = {(int(it["row"].get("example_id", -1)),
                         str(it["row"].get("question_index", ""))) for it in gitems}
            ev_path = os.path.join(ROOT, "data", "bbq_items", f"_sweep400_{t}.jsonl")
            if os.path.exists(ev_path):
                with open(ev_path) as f:
                    ev = {(int(r.get("example_id", -1)), str(r.get("question_index", "")))
                          for r in map(json.loads, filter(str.strip, f))}
                check(f"{t}: heldout disjoint from its 400-item eval file",
                      gkeys[t].isdisjoint(ev) and len(ev) == 400)
        check("woman/man heldout sets mutually disjoint",
              gkeys["woman"].isdisjoint(gkeys["man"]))
    else:
        print("[selftest-calib] gender heldout checks: SKIP (no Gender_identity cache)")

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
    ap.add_argument("--target", choices=SUPPORTED_TARGETS, default="black",
                    help="steering target; black (default) = default selection "
                         "and paths, woman/man = gender manifest heldout")
    args = ap.parse_args()

    if args.selftest:
        sys.exit(0 if _selftest() else 1)
    if args.fit:
        collect_activations(args.where, cap=args.cap, target=args.target)
        return
    ap.error("nothing to do: pass --selftest (offline) or --fit (GPU)")


if __name__ == "__main__":
    main()
