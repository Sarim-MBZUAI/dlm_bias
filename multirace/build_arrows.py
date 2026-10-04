#!/usr/bin/env python
"""multirace/build_arrows.py -- per-target "prefer <target> option" arrows r(k)
for LLaDA-8B-Instruct.

Generalizes steering/build_arrows.py to --target in the race targets
{white, asian, latino, arab}, the intersectional target fblack, and the
pair-axis targets {woman, man, lowses, highses, old, young}. The black arrows
are steering/arrows.pt and are not rebuilt here.

For every transformer block k = 0..31, the answer-text-anchored diff-in-means:

    r(k) = mean_items( h_target(k) - h_other(k) )

where h_*(k) is the block-k residual, masked-mean over the answer-text token
span of a single clean forward on  chat_prompt + answer_text  (same pooling
as steering/build_arrows.py).

Items are the target's HELDOUT keys from the category manifest written by
make_items.py (seed 42); nothing is recomputed here.
  * race targets: multirace/items_manifest.json; heldout is disjoint from the
    target's own eval items and from the Black experiment's eval keys.
  * pair-axis targets (gender / SES / Age): items_manifest_{gender,ses,age}.json;
    the four sets eval/heldout x pole are mutually disjoint. The negative is
    the other pole. 'young' matches BBQ's 'nonOld' tag (see targets.py).
  * fblack: items_manifest_fblack.json over the Race_ethnicity cache
    (heldout = 312, all remaining rows; see make_items.py). The negative is
    any non-fblack, non-unknown option, which admits m-black. BBQ
    Race_ethnicity pairs people of the same gender, so the realized negatives
    are other-race women (0 m-black; recorded in the manifest and the saved
    .pt), i.e. the direction is "Black woman vs other-race woman".

steering/build_arrows.py's load_full_race is imported by file path (this file
shares its basename); pooling/hook code follows it via eval/bbq_eval.py.

Saves RAW (32,4096) r -> multirace/arrows_<target>.pt (+ metadata).

Run:           python multirace/build_arrows.py --target asian
Offline check: python multirace/build_arrows.py --selftest
"""
import argparse
import importlib.util
import json
import os
import sys

import torch

ROOT = os.environ.get("DLM_BIAS_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "eval"))

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import bbq_eval  # noqa: E402
from targets import (TARGET_TAGS, NEW_TARGETS, GENDER_TARGETS,  # noqa: E402
                     INTERSECTIONAL_TARGETS, SES_TARGETS, AGE_TARGETS,
                     TARGET_CATEGORY, target_idx_of, unk_idx_of)
from make_items import (load_llada_builder, load_category_cache, row_key,  # noqa: E402
                        negative_composition,
                        MANIFEST, MANIFEST_GENDER, MANIFEST_FBLACK,
                        MANIFEST_SES, MANIFEST_AGE)

MODEL_PATH = os.path.join(ROOT, "LLaDA-8B-Instruct")
N_LAYERS = 32
DEVICE = "cuda"

# category -> pair-axis manifest (gender, SES, Age; race handled below).
_CATEGORY_MANIFEST = {"Gender_identity": MANIFEST_GENDER,
                      "SES": MANIFEST_SES, "Age": MANIFEST_AGE}


def manifest_path_for(target):
    if target in INTERSECTIONAL_TARGETS:
        return MANIFEST_FBLACK
    return _CATEGORY_MANIFEST.get(TARGET_CATEGORY[target], MANIFEST)


def load_cache_rows(target):
    """Full BBQ cache rows for the target's category."""
    cat = TARGET_CATEGORY[target]
    if cat == "Race_ethnicity":
        return load_llada_builder().load_full_race()
    return load_category_cache(cat)


def heldout_from_manifest(target):
    """(row, target_idx, other_idx) triples for the manifest's heldout keys,
    in manifest order. other = the non-target, non-unknown option."""
    with open(manifest_path_for(target)) as f:
        mani = json.load(f)
    tinfo = mani["targets"][target]
    assert sorted(TARGET_TAGS[target]) == tinfo["tags"], \
        f"[{target}] manifest tags drifted from targets.py -- rerun make_items.py"
    keys = [tuple(k) for k in tinfo["heldout_keys"]]
    by_key = {}
    for r in load_cache_rows(target):
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

    if target in INTERSECTIONAL_TARGETS:
        source = (f"multirace_fblack_manifest_heldout_seed{mani['seed']}"
                  "_disjoint_black_seed42_and_sweep400")
    elif TARGET_CATEGORY[target] == "Gender_identity":
        source = f"multirace_gender_manifest_heldout_seed{mani['seed']}_cross_target_disjoint"
    elif TARGET_CATEGORY[target] in ("SES", "Age"):
        source = (f"multirace_{TARGET_CATEGORY[target].lower()}_manifest_heldout"
                  f"_seed{mani['seed']}_cross_target_disjoint")
    else:
        source = f"multirace_manifest_heldout_seed{mani['seed']}_disjoint_black_seed42_and_sweep400"
    # Composition of the realized negatives (for fblack: how many contrasts
    # are m-black vs other-race), recomputed from the triples actually used.
    neg_comp = negative_composition([row for row, _, _ in heldout], target)

    torch.save({
        "r": r,
        "target": target,
        "target_tags": sorted(TARGET_TAGS[target]),
        "n_layers": N_LAYERS,
        "n_items": len(heldout),
        "per_layer_raw_norm": per_raw,
        "per_layer_mean_diff_norm": per_mean,
        "method": "anchored_caa_text_all_layers",
        "source": source,
        "negative_composition": neg_comp,
        "manifest": os.path.relpath(manifest_path_for(target), ROOT).replace(os.sep, "/"),
    }, out_pt)
    print(f"[build:{target}] negative composition: {neg_comp}", flush=True)
    print(f"[build:{target}] SAVED -> {out_pt}", flush=True)
    print(f"[build:{target}] per_layer_raw_norm={[round(x, 2) for x in per_raw]}", flush=True)
    print(f"[build:{target}] DONE", flush=True)


# --------------------------------------------------------------------------- #
# Offline self-test: manifest -> triples resolution (no GPU, real manifests).  #
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
        # eval_file is ROOT-relative; join() is a no-op for absolute paths.
        with open(os.path.join(ROOT, tinfo["eval_file"])) as f:
            ev_keys = {row_key(json.loads(l)) for l in f if l.strip()}
        check(f"{target}: heldout disjoint from eval file", keys.isdisjoint(ev_keys))
        check(f"{target}: heldout disjoint from Black-exp keys", keys.isdisjoint(exclude))
        tag_ok = all(
            str(bbq_eval.get_answer_info(r, t)[-1]).strip().lower() in TARGET_TAGS[target]
            and str(bbq_eval.get_answer_info(r, o)[-1]).strip().lower()
            not in (TARGET_TAGS[target] | {"unknown"})
            for r, t, o in triples)
        check(f"{target}: positive=target tag, negative=non-target non-unknown", tag_ok)

    # --- gender manifest: same checks + 4-way cross-target disjointness ---- #
    gsets = {}
    for target in GENDER_TARGETS:
        triples, mani = heldout_from_manifest(target)
        tinfo = mani["targets"][target]
        check(f"{target}: gender manifest flags cross_target_disjoint",
              mani.get("cross_target_disjoint") is True)
        check(f"{target}: heldout resolves ({len(triples)} triples)",
              len(triples) == tinfo["n_heldout"] == 400)
        keys = {row_key(r) for r, _, _ in triples}
        with open(os.path.join(ROOT, tinfo["eval_file"])) as f:
            ev_keys = {row_key(json.loads(l)) for l in f if l.strip()}
        gsets[f"heldout({target})"] = keys
        gsets[f"eval({target})"] = ev_keys
        other = "man" if target == "woman" else "woman"
        tag_ok = all(
            str(bbq_eval.get_answer_info(r, t)[-1]).strip().lower() in TARGET_TAGS[target]
            and str(bbq_eval.get_answer_info(r, o)[-1]).strip().lower()
            not in (TARGET_TAGS[target] | {"unknown"})
            for r, t, o in triples)
        check(f"{target}: positive=target tag, negative=non-target non-unknown", tag_ok)
        # gender-specific: the negative is the OTHER-gendered person.
        check(f"{target}: negative carries the other gender's tag",
              all(str(bbq_eval.get_answer_info(r, o)[-1]).strip().lower()
                  in TARGET_TAGS[other] for r, _, o in triples))
    names = sorted(gsets)
    check("gender: 4 sets pairwise disjoint (eval/heldout x woman/man)",
          all(gsets[a].isdisjoint(gsets[b])
              for i, a in enumerate(names) for b in names[i + 1:]))

    # --- SES/Age manifests: same 4-way cross-target disjointness ----------- #
    for axis_name, axis_targets in (("ses", SES_TARGETS), ("age", AGE_TARGETS)):
        xsets = {}
        for target in axis_targets:
            triples, mani = heldout_from_manifest(target)
            tinfo = mani["targets"][target]
            check(f"{target}: {axis_name} manifest flags cross_target_disjoint",
                  mani.get("cross_target_disjoint") is True
                  and mani["category"] == TARGET_CATEGORY[target])
            check(f"{target}: heldout resolves ({len(triples)} triples)",
                  len(triples) == tinfo["n_heldout"] == 400)
            keys = {row_key(r) for r, _, _ in triples}
            with open(os.path.join(ROOT, tinfo["eval_file"])) as f:
                ev_keys = {row_key(json.loads(l)) for l in f if l.strip()}
            xsets[f"heldout({target})"] = keys
            xsets[f"eval({target})"] = ev_keys
            other = axis_targets[1] if target == axis_targets[0] else axis_targets[0]
            tag_ok = all(
                str(bbq_eval.get_answer_info(r, t)[-1]).strip().lower() in TARGET_TAGS[target]
                and str(bbq_eval.get_answer_info(r, o)[-1]).strip().lower()
                not in (TARGET_TAGS[target] | {"unknown"})
                for r, t, o in triples)
            check(f"{target}: positive=target tag, negative=non-target non-unknown", tag_ok)
            # pair-axis specific: the negative carries the OTHER pole's tag.
            check(f"{target}: negative carries the other pole's tag",
                  all(str(bbq_eval.get_answer_info(r, o)[-1]).strip().lower()
                      in TARGET_TAGS[other] for r, _, o in triples))
        names = sorted(xsets)
        check(f"{axis_name}: 4 sets pairwise disjoint (eval/heldout x poles)",
              all(xsets[a].isdisjoint(xsets[b])
                  for i, a in enumerate(names) for b in names[i + 1:]))

    # --- fblack manifest: race exclusions DO apply; negatives policy ------- #
    triples, mani = heldout_from_manifest("fblack")
    tinfo = mani["targets"]["fblack"]
    check("fblack: heldout resolves (312 triples, documented deviation)",
          len(triples) == tinfo["n_heldout"] == 312)
    check("fblack: manifest records the 400/312 deviation",
          mani.get("n_heldout") == 312 and "deviation" in mani)
    keys = {row_key(r) for r, _, _ in triples}
    with open(os.path.join(ROOT, tinfo["eval_file"])) as f:
        ev_keys = {row_key(json.loads(l)) for l in f if l.strip()}
    check("fblack: heldout disjoint from eval file (400 items)",
          keys.isdisjoint(ev_keys) and len(ev_keys) == 400)
    check("fblack: heldout disjoint from Black-exp keys (same category!)",
          keys.isdisjoint(exclude))
    check("fblack: eval disjoint from Black-exp keys", ev_keys.isdisjoint(exclude))
    # positives are exactly f-black; negatives NEVER carry the f-black tag
    # (they MAY be m-black by policy -- see module doc).
    check("fblack: every positive tagged f-black",
          all(str(bbq_eval.get_answer_info(r, t)[-1]).strip().lower() == "f-black"
              for r, t, _ in triples))
    check("fblack: negatives never f-black, never unknown",
          all(str(bbq_eval.get_answer_info(r, o)[-1]).strip().lower()
              not in {"f-black", "unknown"} for r, _, o in triples))
    # composition recomputed from triples == manifest record.
    comp = {}
    for r, _, o in triples:
        tag = str(bbq_eval.get_answer_info(r, o)[-1]).strip().lower()
        comp[tag] = comp.get(tag, 0) + 1
    mcomp = tinfo["heldout_negative_composition"]
    check("fblack: negative composition matches manifest record",
          comp == mcomp["by_tag"]
          and mcomp["m-black"] + mcomp["other"] == len(triples))
    print(f"[selftest-marrows] fblack negatives: m-black={mcomp['m-black']} "
          f"other={mcomp['other']} by_tag={mcomp['by_tag']}")

    print(f"[selftest-marrows] OVERALL: {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target", choices=(list(NEW_TARGETS) + list(GENDER_TARGETS)
                                         + list(INTERSECTIONAL_TARGETS)
                                         + list(SES_TARGETS) + list(AGE_TARGETS)))
    ap.add_argument("--selftest", action="store_true",
                    help="offline manifest-resolution check (no GPU)")
    args = ap.parse_args()
    if args.selftest:
        sys.exit(0 if _selftest() else 1)
    if not args.target:
        ap.error("--target required (or --selftest)")
    main(args.target)
