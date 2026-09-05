#!/usr/bin/env python
"""UNQOVER-race NATIVE direction + every baseline fit, from the BUILD split ONLY.

Phase-2 step 1.  Builds the "prefer the black family" steering direction with the
paper's own recipe -- per-layer diff-in-means over block residuals, exactly
steering/build_arrows.py -- but on UNQOVER-race BUILD instances, and with the
three tokenisation fixes this benchmark forces (README section 8):

  * pool over the COMPLETE subject-name span, never the first token
    ("African" is a strict token-PREFIX of "African-American"; both start with
    id 12066, so first-token pooling would merge the two names);
  * length-normalise -- spans run 1-3 tokens, so the span mean (not the sum) is
    the pooled vector;
  * extract the span from the IN-CONTEXT (leading-space) encoding, never a
    standalone encode ("Caucasian" is 1 token in context, 3 standalone).

The answer text appended to the chat prompt is therefore " <Name>" (leading
space), and the pooled span is tokens [len(tok(chat)):] -- which this file
ASSERTS equals tok(" <Name>", add_special_tokens=False), i.e. the in-context
encoding, for every single contrast forward.  Nothing is pooled from a
standalone encode and nothing is truncated to a first token.

CONTRAST (one per BUILD instance x subject order)
-------------------------------------------------
Each BUILD instance whose subject pair contains the merged black family
{Black, African-American, African} pairs that member against a non-black
subject.  For BOTH subject orders we take the SAME 2-option A/B prompt the eval
harness builds (unqover_hf.eval_harness.build_prompt, q0 polarity) and run two
clean forwards on the fully materialised sequence:

    positive = chat_prompt + " <black-family member>"      (label 1)
    negative = chat_prompt + " <other subject>"            (label 0)

Both orders are used so the direction is not confounded with mention position
(order 0 names subj_a first, order 1 names subj_b first).

    r(k) = mean over contrast items of ( h_pos(k) - h_neg(k) )        (32, 4096)

pooled with calib._pool_captured -- the SAME masked-mean the BBQ arrows use.

EQUAL BUDGET
------------
The SAME labelled pooled activations, collected at where in {block, mlp_hidden,
attn_head}, feed EVERY baseline fit.  No method gets a bigger or cleaner fit set
than any other, and no method's fit touches the EVAL split:

    caa / meanact       -> arrows_race_black.pt        (block, diff-in-means)
    actadd              -> cache_race/actadd_dir_pair{0,1,2}.pt (single pair)
    linearact           -> cache_race/linearact_stats.pt        (mlp_hidden OT)
    aura                -> cache_race/aura_auroc.pt             (mlp_hidden AUROC)
    itic                -> cache_race/itic_probes.pt            (attn_head probes)

DISJOINTNESS
------------
The fit set is asserted disjoint from the EVAL split on instance_id, attribute
(attribute_id + the literal attribute string + both question strings) and
template (template_id + tid + the literal context strings).  The assertion
OUTPUT is written into the arrows metadata, together with the git SHA of the
code that built it.

  CPU:  python -m unqover_hf.build_arrows_race --selftest
  GPU:  CUDA_VISIBLE_DEVICES=6 python -m unqover_hf.build_arrows_race --build
"""
import argparse
import json
import os
import subprocess
import sys
import time

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))                 # unqover_hf/
_ROOT = os.path.dirname(_HERE)

sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "baselines"))
sys.path.insert(0, os.path.join(_ROOT, "eval"))
sys.path.insert(0, os.path.join(_ROOT, "steering"))

from unqover_hf.loader import FAMILY_MEMBERS, load_jsonl  # noqa: E402
from unqover_hf.splits import DEFAULT_BUILD, DEFAULT_EVAL, VERIFY_FIELDS, _record_values  # noqa: E402
from unqover_hf import eval_harness as EH  # noqa: E402

TARGET_FAMILY = "black"
N_LAYERS = 32
WHERES = ("block", "mlp_hidden", "attn_head")
# ActAdd is an n=1 method, so WHICH single contrast pair it is fitted from is a
# real hyperparameter and PREREG section 3 sweeps it.  Contrast items come in
# (instance, order-0), (instance, order-1) pairs, so consecutive item indices
# would give the SAME subject pair twice; pair index p therefore maps to item
# p * (n_items // N_ACTADD_PAIRS) -- three well-separated instances, hence three
# genuinely different subject pairs / templates / attributes.
N_ACTADD_PAIRS = 3
ACTADD_PAIRS = tuple(range(N_ACTADD_PAIRS))


def actadd_item_index(pair_index, items):
    """Contrast-item index for ActAdd pair `pair_index`.

    Items are ordered (instance, order0), (instance, order1), ..., so adjacent
    indices are the SAME subject pair.  We therefore index into a deterministic
    SPREAD order -- items sorted by (attribute, template, instance, order) -- and
    take evenly spaced thirds, which lands the three pairs on three different
    attributes, templates and subject pairs.  `build` asserts exactly that."""
    n = len(items)
    spread = sorted(range(n), key=lambda j: (items[j]["attribute_id"],
                                             items[j]["template_id"],
                                             items[j]["instance_id"],
                                             items[j]["order"]))
    return spread[int(pair_index) * (n // N_ACTADD_PAIRS)]

ARROWS_OUT = os.path.join(_HERE, "arrows_race_black.pt")
CACHE_DIR = os.path.join(_HERE, "cache_race")
EXAMPLES_OUT = os.path.join(_HERE, "direction_examples_race.jsonl")

SOURCE = "unqover_hf_race_BUILD_split_black_family_span_pooled_both_orders"


def git_sha():
    try:
        return subprocess.check_output(
            ["git", "-C", _ROOT, "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:                                          # noqa: BLE001
        return None


# --------------------------------------------------------------------------- #
# Contrast-item construction (CPU).                                            #
# --------------------------------------------------------------------------- #
def target_instances(recs, family=TARGET_FAMILY):
    """BUILD instances whose subject pair contains the merged target family."""
    members = FAMILY_MEMBERS[family]
    out = []
    for r in recs:
        in_a, in_b = r["subj_a"] in members, r["subj_b"] in members
        if in_a == in_b:                       # neither, or a within-family pair
            continue                           # (within-family already excluded)
        out.append(r)
    return out


def build_contrast_items(recs, tok=None, family=TARGET_FAMILY, cap=0):
    """One contrast item per (instance, subject order), in calib's item shape.

    positive = " <black-family member>", negative = " <other subject>" -- both
    with the LEADING SPACE, so the appended span is the in-context encoding of
    the complete name (never a standalone encode, never a first token)."""
    members = FAMILY_MEMBERS[family]
    items = []
    for r in recs:
        tgt = r["subj_a"] if r["subj_a"] in members else r["subj_b"]
        oth = r["subj_b"] if r["subj_a"] in members else r["subj_a"]
        for o in r["orders"]:
            prompt = EH.build_prompt({"context": o["context"],
                                      "question": r["q0_question"],
                                      "subj0": o["subj0"], "subj1": o["subj1"]})
            chat = None
            if tok is not None:
                chat = tok.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    add_generation_prompt=True, tokenize=False)
            items.append({
                "instance_id": r["instance_id"],
                "template_id": r["template_id"], "attribute_id": r["attribute_id"],
                "order": o["order"],
                "chat_prompt": chat,
                "prompt": prompt,
                "target_answer_text": " " + tgt,      # LEADING SPACE (in-context)
                "other_answer_text": " " + oth,       # LEADING SPACE (in-context)
                "target_subject": tgt, "other_subject": oth,
            })
    return items[:cap] if cap else items


def disjointness(fit_recs, eval_recs):
    """Shared-value count for every key the split verifier checks. All must be 0."""
    fields = [f for n in ("attribute", "template") for f in VERIFY_FIELDS[n]] \
        + ["instance_id"]
    out = {}
    for f in fields:
        bs = {v for r in fit_recs for v in _record_values(r, f)}
        es = {v for r in eval_recs for v in _record_values(r, f)}
        out[f] = len(bs & es)
    return out


def write_actadd(acts, items, cache_dir, sha):
    """The ActAdd n=1 artifacts: r = h_target - h_other for ONE contrast item.

    Pure CPU tensor math on the pooled block activations, so it can be re-run
    from cache_race/calib_block.pt without a GPU."""
    os.makedirs(cache_dir, exist_ok=True)
    out = []
    for pi in ACTADD_PAIRS:
        j = actadd_item_index(pi, items)
        r_a = (acts[2 * j] - acts[2 * j + 1]).to(torch.float32)
        p = os.path.join(cache_dir, "actadd_dir_pair%d.pt" % pi)
        torch.save({"r": r_a, "n_layers": N_LAYERS, "n_items": 1,
                    "pair_index": pi, "contrast_item_index": j,
                    "method": "actadd_single_pair", "granularity": "block_residual",
                    "target_subject": items[j]["target_subject"],
                    "other_subject": items[j]["other_subject"],
                    "instance_id": items[j]["instance_id"],
                    "template_id": items[j]["template_id"],
                    "attribute_id": items[j]["attribute_id"],
                    "order": items[j]["order"],
                    "source": SOURCE, "target": "black_unqover_race",
                    "git_sha": sha,
                    "per_layer_raw_norm": [float(r_a[k].norm())
                                           for k in range(N_LAYERS)]}, p)
        print("[actadd] pair%d = contrast item %d: %s vs %s (%s, %s) -> %s"
              % (pi, j, items[j]["target_subject"], items[j]["other_subject"],
                 items[j]["template_id"], items[j]["attribute_id"], p), flush=True)
        out.append({"pair_index": pi, "item": j,
                    "target": items[j]["target_subject"],
                    "other": items[j]["other_subject"],
                    "instance_id": items[j]["instance_id"],
                    "template_id": items[j]["template_id"],
                    "attribute_id": items[j]["attribute_id"]})
    assert len({(d["target"], d["other"]) for d in out}) == len(out), (
        "ActAdd pair sweep is degenerate: %s -- the pair indices must give "
        "DIFFERENT subject pairs" % out)
    assert len({d["instance_id"] for d in out}) == len(out), \
        "ActAdd pair indices must come from different instances: %s" % out
    assert len({d["attribute_id"] for d in out}) == len(out), \
        "ActAdd pair indices must come from different attributes: %s" % out
    assert len({d["template_id"] for d in out}) == len(out), \
        "ActAdd pair indices must come from different templates: %s" % out
    return out


def refit_actadd(build_path=DEFAULT_BUILD, cache_dir=CACHE_DIR,
                 family=TARGET_FAMILY):
    """Rebuild the ActAdd artifacts from the saved calib_block.pt.  CPU ONLY."""
    blob = torch.load(os.path.join(cache_dir, "calib_block.pt"), map_location="cpu")
    fit_recs = target_instances(load_jsonl(build_path), family)
    items = build_contrast_items(fit_recs, tok=None, family=family)
    assert blob["acts"].shape[0] == 2 * len(items), (
        "calib_block.pt has %d rows but the contrast set has %d items"
        % (blob["acts"].shape[0], len(items)))
    return write_actadd(blob["acts"].to(torch.float32), items, cache_dir, git_sha())


# --------------------------------------------------------------------------- #
# GPU build.                                                                   #
# --------------------------------------------------------------------------- #
def build(build_path=DEFAULT_BUILD, eval_path=DEFAULT_EVAL, cap=0,
          arrows_out=ARROWS_OUT, cache_dir=CACHE_DIR, family=TARGET_FAMILY):
    import common as C          # noqa: E402
    import calib                # noqa: E402
    import directions           # noqa: E402
    import linearact            # noqa: E402
    import itic                 # noqa: E402

    t0 = time.time()
    build_recs = load_jsonl(build_path)
    eval_recs = load_jsonl(eval_path)
    fit_recs = target_instances(build_recs, family)

    dis = disjointness(fit_recs, eval_recs)
    assert all(v == 0 for v in dis.values()), \
        "FIT/EVAL LEAK: %s" % {k: v for k, v in dis.items() if v}
    print("[build] BUILD=%d instances, target-family=%d, EVAL=%d" %
          (len(build_recs), len(fit_recs), len(eval_recs)), flush=True)
    print("[build] fit-vs-EVAL shared values (all must be 0): %s" % dis, flush=True)

    model, tok = C.load_model()
    items = build_contrast_items(fit_recs, tok=tok, family=family, cap=cap)
    print("[build] %d contrast items (%d instances x 2 subject orders)%s"
          % (len(items), len(fit_recs), " CAPPED" if cap else ""), flush=True)

    # --- span audit: the appended answer span IS the in-context name span ---- #
    span_len = {}
    for it in items:
        for key in ("target_answer_text", "other_answer_text"):
            name = it[key]
            plen = len(tok(it["chat_prompt"])["input_ids"])
            full = tok(it["chat_prompt"] + name)["input_ids"]
            want = tok(name, add_special_tokens=False)["input_ids"]
            assert full[:plen] == tok(it["chat_prompt"])["input_ids"], \
                "chat prompt is not a token prefix of chat+answer (%r)" % name
            assert full[plen:] == want, (
                "pooled span %r != in-context encoding %r for %r"
                % (full[plen:], want, name))
            span_len.setdefault(name.strip(), len(want))
    print("[build] span audit OK. in-context span lengths: %s"
          % dict(sorted(span_len.items())), flush=True)
    assert min(span_len.values()) >= 1 and max(span_len.values()) <= 3

    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(arrows_out)), exist_ok=True)
    with open(EXAMPLES_OUT, "w") as fh:
        for it in items:
            fh.write(json.dumps({k: v for k, v in it.items()
                                 if k != "chat_prompt"}, ensure_ascii=False) + "\n")

    # --- pooled activations at all three granularities (SAME items) --------- #
    blobs = {}
    for where in WHERES:
        blob = calib.collect_activations(where, model=model, tok=tok, items=items,
                                         save=False, target="black")
        blob["source"] = SOURCE
        blob["target"] = "black_unqover_race"
        blob["benchmark"] = "unqover_hf_race"
        blobs[where] = blob
        print("[build] calib %s: acts=%s" % (where, tuple(blob["acts"].shape)),
              flush=True)

    bblob = blobs["block"]
    acts = bblob["acts"].to(torch.float32)          # (2n, 32, 4096)
    labels = bblob["labels"].to(torch.bool)
    r = acts[labels].mean(0) - acts[~labels].mean(0)

    torch.save({k: v for k, v in bblob.items() if k != "acts"} |
               {"acts": bblob["acts"]},
               os.path.join(cache_dir, "calib_block.pt"))

    meta = {
        "r": r,
        "n_layers": N_LAYERS,
        "n_items": len(items),
        "n_instances": len(fit_recs),
        "n_contrast_forwards": 2 * len(items),
        "method": "diff_in_means_block_residual (steering/build_arrows.py recipe)",
        "span_pooling": ("masked mean over the COMPLETE in-context (leading-space) "
                         "subject-name token span appended as the answer text; "
                         "length-normalised (mean, not sum); never the first token"),
        "span_lengths_tokens": dict(sorted(span_len.items())),
        "orders_used": [0, 1],
        "polarity_used": "q0",
        "source": SOURCE,
        "target_family": family,
        "target_members": sorted(FAMILY_MEMBERS[family]),
        "split": {"build_path": build_path, "eval_path": eval_path,
                  "build_instances": len(build_recs),
                  "fit_instances": len(fit_recs),
                  "eval_instances": len(eval_recs)},
        "disjointness_assertion": {
            "keys": dis,
            "all_zero": all(v == 0 for v in dis.values()),
            "note": ("shared values between the FIT set (BUILD, target family) "
                     "and the EVAL split; asserted == 0 before any forward"),
        },
        "per_layer_raw_norm": [float(r[k].norm()) for k in range(N_LAYERS)],
        "per_layer_mean_diff_norm": [
            float((acts[labels][:, k, :] - acts[~labels][:, k, :]).norm(dim=1).mean())
            for k in range(N_LAYERS)],
        "git_sha": git_sha(),
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    torch.save(meta, arrows_out)
    print("[build] arrows -> %s  raw_norms[12:16]=%s" %
          (arrows_out, [round(meta["per_layer_raw_norm"][k], 2) for k in range(12, 16)]),
          flush=True)

    # --- ActAdd: the single-pair block diff, for 3 DIFFERENT subject pairs --- #
    write_actadd(acts, items, cache_dir, meta["git_sha"])

    # --- Mean-AcT "fitted" variant (block per-neuron mean diff == r) --------- #
    torch.save({"mean_diff": r, "where": "block", "n_layers": N_LAYERS,
                "n_items": bblob["n_items"], "source": SOURCE,
                "target": "black_unqover_race", "git_sha": meta["git_sha"]},
               os.path.join(cache_dir, "meanact_meandiff_block.pt"))

    # --- Linear-AcT: per-neuron gaussian + empirical OT on mlp_hidden -------- #
    stats = linearact.fit_stats_from_blob(blobs["mlp_hidden"])
    stats.update({"source": SOURCE, "target": "black_unqover_race",
                  "git_sha": meta["git_sha"]})
    torch.save(stats, os.path.join(cache_dir, "linearact_stats.pt"))
    print("[build] linearact_stats (%sx%s) written" % (stats["n_layers"], stats["feat"]),
          flush=True)

    # --- AurA: per-neuron AUROC on mlp_hidden -------------------------------- #
    mblob = blobs["mlp_hidden"]
    feat = mblob["acts"].shape[-1]
    auroc = torch.empty(N_LAYERS, feat, dtype=torch.float32)
    for k in range(N_LAYERS):
        # .contiguous() is NOT cosmetic: acts is (2n, 32, feat), so acts[:, k, :]
        # is a strided view whose consecutive rows sit 32*feat*4 bytes apart.
        # sklearn's roc_auc_score reads it COLUMN by column, so on the strided
        # view every column touches 1800 cache lines spread over the whole
        # tensor -- measured 15x slower at 4 layers and far worse at 32 (148 s
        # vs 9.8 s per layer, i.e. hours vs minutes for the loop).
        a_k, _g = directions.auroc_per_neuron(
            mblob["acts"][:, k, :].contiguous(), mblob["labels"])
        auroc[k] = a_k
        if (k + 1) % 4 == 0 or k == 0:
            print("[build] aura auroc layer %d/%d" % (k + 1, N_LAYERS), flush=True)
    torch.save({"auroc": auroc, "where": "mlp_hidden", "feat": feat,
                "n_layers": N_LAYERS, "n_items": mblob["n_items"],
                "source": SOURCE, "target": "black_unqover_race",
                "git_sha": meta["git_sha"]},
               os.path.join(cache_dir, "aura_auroc.pt"))
    print("[build] aura_auroc (%dx%d) written  max=%.3f"
          % (N_LAYERS, feat, float(auroc.max())), flush=True)

    # --- ITI-C: per-head probes on attn_head --------------------------------- #
    ablob = blobs["attn_head"]
    A = ablob["acts"].to(torch.float32).view(ablob["acts"].shape[0], N_LAYERS,
                                             itic.N_HEADS, itic.D_HEAD)
    # same striding hazard as AURA above: make each layer's block contiguous once
    # instead of handing 32 strided per-head views to the probe fitter.
    lab = ablob["labels"].long()
    theta = torch.zeros(N_LAYERS, itic.N_HEADS, itic.D_HEAD)
    sigma = torch.zeros(N_LAYERS, itic.N_HEADS)
    val_acc = torch.zeros(N_LAYERS, itic.N_HEADS)
    for k in range(N_LAYERS):
        Ak = A[:, k, :, :].contiguous()
        for hh in range(itic.N_HEADS):
            va, th, sg = itic._fit_one_head(Ak[:, hh, :], lab)
            val_acc[k, hh] = va
            theta[k, hh] = th
            sigma[k, hh] = sg
        print("[build] itic layer %d/%d" % (k + 1, N_LAYERS), flush=True)
    torch.save({"theta": theta, "sigma": sigma, "val_acc": val_acc,
                "n_layers": N_LAYERS, "n_heads": itic.N_HEADS, "d_head": itic.D_HEAD,
                "n_items": ablob["n_items"], "source": SOURCE, "method": "iti_c",
                "target": "black_unqover_race", "git_sha": meta["git_sha"],
                "direction": "mass_mean_shift_blackfamily_minus_other"},
               os.path.join(cache_dir, "itic_probes.pt"))
    print("[build] itic_probes written  best_val_acc=%.3f" % float(val_acc.max()),
          flush=True)
    print("[build] DONE in %.0fs" % (time.time() - t0), flush=True)


# --------------------------------------------------------------------------- #
def selftest(build_path=DEFAULT_BUILD, eval_path=DEFAULT_EVAL):
    ok = True

    def check(name, cond, extra=""):
        nonlocal ok
        ok &= bool(cond)
        print("[selftest] %-58s %s%s" % (name, "PASS" if cond else "FAIL",
                                         (" " + extra) if extra else ""))

    have = os.path.exists(build_path) and os.path.exists(eval_path)
    check("BUILD and EVAL split files exist", have)
    if not have:
        print("[selftest] run `python -m unqover_hf.splits` first")
        return False

    b = load_jsonl(build_path)
    e = load_jsonl(eval_path)
    fit = target_instances(b)
    check("BUILD target-family instances == 450", len(fit) == 450, "(%d)" % len(fit))
    members = FAMILY_MEMBERS[TARGET_FAMILY]
    check("every fit instance has exactly one black-family subject",
          all((r["subj_a"] in members) != (r["subj_b"] in members) for r in fit))

    dis = disjointness(fit, e)
    check("fit set disjoint from EVAL on every key", all(v == 0 for v in dis.values()),
          str(dis))

    items = build_contrast_items(fit, tok=None)
    check("2 contrast items per instance (both subject orders)",
          len(items) == 2 * len(fit), "(%d)" % len(items))
    check("every positive is a black-family member, negative is not",
          all(it["target_subject"] in members and it["other_subject"] not in members
              for it in items))
    check("both answer texts carry the LEADING SPACE (in-context encoding)",
          all(it["target_answer_text"].startswith(" ")
              and it["other_answer_text"].startswith(" ") for it in items))
    check("both orders present for every instance",
          sorted({it["order"] for it in items}) == [0, 1])
    per_inst = {}
    for it in items:
        per_inst.setdefault(it["instance_id"], set()).add(it["order"])
    check("each instance contributes order 0 AND order 1",
          all(v == {0, 1} for v in per_inst.values()))
    p = items[0]["prompt"]
    check("contrast prompt is the harness's own 2-option A/B prompt",
          p.endswith("Answer with only the letter A or B.")
          and "C." not in p and "unknown" not in p.lower())

    idxs = [actadd_item_index(p, items) for p in ACTADD_PAIRS]
    trip = [(items[j]["target_subject"], items[j]["other_subject"],
             items[j]["template_id"], items[j]["attribute_id"]) for j in idxs]
    check("the 3 ActAdd pair indices give 3 DIFFERENT subject pairs",
          len({t[:2] for t in trip}) == 3, str(trip))
    check("the 3 ActAdd pair indices come from 3 different instances",
          len({items[j]["instance_id"] for j in idxs}) == 3)
    check("the 3 ActAdd pair indices span 3 templates and 3 attributes",
          len({t[2] for t in trip}) == 3 and len({t[3] for t in trip}) == 3,
          str(trip))

    if os.path.exists(ARROWS_OUT):
        blob = torch.load(ARROWS_OUT, map_location="cpu")
        check("arrows_race_black.pt: r is (32,4096)",
              tuple(blob["r"].shape) == (32, 4096))
        check("arrows metadata carries the disjointness assertion",
              blob["disjointness_assertion"]["all_zero"] is True)
        check("arrows metadata carries a git SHA", bool(blob.get("git_sha")))
        check("arrows metadata carries the span-pooling mode + span lengths",
              "span_pooling" in blob and blob.get("span_lengths_tokens"))
        check("arrows metadata carries per-layer norms",
              len(blob["per_layer_raw_norm"]) == 32)
    else:
        print("[selftest] arrows checks: SKIP (not built yet)")
    for name in ["actadd_dir_pair%d.pt" % i for i in ACTADD_PAIRS] + \
                ["linearact_stats.pt", "aura_auroc.pt", "itic_probes.pt",
                 "meanact_meandiff_block.pt"]:
        q = os.path.join(CACHE_DIR, name)
        if os.path.exists(q):
            blob = torch.load(q, map_location="cpu")
            check("%s: UNQOVER-race-native source" % name, blob.get("source") == SOURCE)
        else:
            print("[selftest] %s: SKIP (not built yet)" % name)

    print("[selftest] build_arrows_race OVERALL: %s" % ("PASS" if ok else "FAIL"))
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--build", action="store_true", help="NEEDS A GPU")
    ap.add_argument("--refit-actadd", action="store_true",
                    help="CPU: rebuild the ActAdd n=1 artifacts from calib_block.pt")
    ap.add_argument("--build-path", default=DEFAULT_BUILD)
    ap.add_argument("--eval-path", default=DEFAULT_EVAL)
    ap.add_argument("--cap", type=int, default=0, help="0 = all contrast items")
    ap.add_argument("--arrows-out", default=ARROWS_OUT)
    ap.add_argument("--cache-dir", default=CACHE_DIR)
    args = ap.parse_args()
    if args.selftest:
        sys.exit(0 if selftest(args.build_path, args.eval_path) else 1)
    if args.refit_actadd:
        refit_actadd(args.build_path, args.cache_dir)
        return
    if args.build:
        build(args.build_path, args.eval_path, args.cap, args.arrows_out,
              args.cache_dir)
        return
    ap.error("pass --selftest (CPU) or --build (GPU)")


if __name__ == "__main__":
    main()
