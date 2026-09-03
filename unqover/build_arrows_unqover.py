#!/usr/bin/env python
"""UNQOVER-NATIVE direction + baseline fits (make UNQOVER a STANDALONE benchmark).

Today every UNQOVER run reuses the BBQ-built "prefer Black" direction
(steering/arrows.pt row 14) and the BBQ black baseline fit artifacts
(baselines/cache/*.pt).  This script REPLACES that dependency: it builds the
"prefer Black" steering direction AND every fit-based baseline artifact from
UNQOVER's OWN held-out contrast data, on a clean, committed, instance-disjoint
train/eval split -- so the UNQOVER numbers no longer borrow anything from BBQ.

MIRRORS steering/build_arrows.py (diff-in-means) and the baselines/ fit machinery
(baselines/calib.py + directions.py + the 6 method files), but pointed at UNQOVER
contrast items instead of BBQ Race_ethnicity items.

CONTRAST (UNQOVER-native, per BUILD-split instance)
---------------------------------------------------
Each Black-containing UNQOVER instance pairs the subject "Black" with one paired
OTHER subject (e.g. "African", "White", ...).  For every BUILD instance we take
its 2-choice A/B prompt (unqover_eval.build_prompt -- the SAME prompt the eval
uses) and run TWO clean forwards on the fully-materialized sequence:

    positive  = chat_prompt + "Black"          (label 1, OT destination)
    negative  = chat_prompt + <other subject>  (label 0, OT source)

masked-mean pooled over the appended answer-text span (calib._pool_captured,
identical to build_arrows.all_layer_hidden).  The per-layer diff-in-means

    r(k) = mean_build( h_Black(k) - h_other(k) )

is the UNQOVER-native "prefer Black" arrow set (saved arrows_unqover_black.pt,
same {'r': (32,4096)} format as steering/arrows.pt).  The SAME labelled pooled
activations, collected at where in {block, mlp_hidden, attn_head}, feed the
baseline fits (reusing baselines/ code verbatim; only the data source changes):

    caa / mean-act     -> the UNQOVER arrows (via --native-arrows)
    actadd             -> single-pair block diff (actadd_dir.pt)
    linear-act         -> per-neuron gaussian+empirical OT (linearact_stats.pt)
    aura               -> per-neuron AUROC (aura_auroc.pt)
    iti-c              -> per-head probes (itic_probes.pt)

CLEAN SPLIT (committed, reproducible, contamination-safe)
---------------------------------------------------------
The 262 Black-containing instances are split by instance_id with a fixed seed
into a BUILD split (direction + fits) and a DISJOINT EVAL split (never seen by
the builder).  The split is recorded in unqover/items_manifest_unqover_black.json
(build/eval instance-id lists) so it is auditable and reproducible, and so the
runners can eval on the EVAL split only.

STEPS (this file)
-----------------
  --build-manifest : CPU. Read the full items, find the 262 Black instances,
                     deterministic split, write the manifest json.  (run here)
  --write-splits   : CPU. From the manifest, emit build_items.jsonl and
                     eval_items.jsonl (the runners' --items files).  (run by sbatch)
  --build          : GPU. Collect UNQOVER-native activations from the BUILD split,
                     save arrows_unqover_black.pt + the 5 native baseline fits.
  --selftest       : CPU. Manifest disjointness + counts + contrast build + a
                     tiny load check of any artifacts that already exist. NO model.

Native artifacts (all UNQOVER-native, separate from steering/arrows.pt and
baselines/cache/):
    results/unqover_native/arrows_unqover_black.pt        (CAA / Mean-AcT direction)
    unqover/cache_native/calib_{block,mlp_hidden,attn_head}.pt  (pooled acts)
    unqover/cache_native/actadd_dir.pt
    unqover/cache_native/linearact_stats.pt
    unqover/cache_native/aura_auroc.pt
    unqover/cache_native/itic_probes.pt
"""
import argparse
import json
import os
import random
import sys
from collections import OrderedDict, defaultdict

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))                       # unqover/
_ROOT = os.environ.get("DLM_BIAS_ROOT") or os.path.dirname(_HERE)        # MAIN tree

# baselines/ importable as top-level (import common/calib/directions + methods),
# unqover/ for the UNQOVER prompt, eval/+steering/ for the harness (via calib).
sys.path.insert(0, os.path.join(_ROOT, "baselines"))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_ROOT, "eval"))
sys.path.insert(0, os.path.join(_ROOT, "steering"))

import unqover_eval as U  # noqa: E402  (build_prompt -- the SAME 2-choice A/B prompt)

# ----- paths (data reads honor DLM_BIAS_ROOT; committed outputs live in-tree) -- #
FULL_ITEMS = os.path.join(_ROOT, "data", "unqover", "ethnicity.items.jsonl")
MANIFEST = os.path.join(_HERE, "items_manifest_unqover_black.json")       # COMMITTED
NATIVE_CACHE = os.path.join(_HERE, "cache_native")                        # local *.pt
ARROWS_OUT = os.path.join(_ROOT, "results", "unqover_native",
                          "arrows_unqover_black.pt")                       # local *.pt

TARGET = "Black"                 # the UNQOVER subject the direction prefers
SEED = 42                        # repo convention
BUILD_FRAC = 0.40                # ~40% build / ~60% eval (task's suggested ratio)
N_LAYERS = 32
WHERES = ("block", "mlp_hidden", "attn_head")

# UNQOVER-native provenance string stamped on every artifact (replaces the BBQ
# "bbq_race_ethnicity_heldout_disjoint_seed42_and_sweep400" source string).
SOURCE = "unqover_ethnicity_black_build_split_seed42_instance_disjoint"


# --------------------------------------------------------------------------- #
# Data: the 262 Black-containing instances (grouped, complete quadruples).      #
# --------------------------------------------------------------------------- #
def load_black_instances(items_path=FULL_ITEMS):
    """OrderedDict instance_id -> [4 records] for every Black-containing instance,
    in first-seen file order (deterministic)."""
    by_inst = OrderedDict()
    with open(items_path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            by_inst.setdefault(r["instance_id"], []).append(r)
    black = OrderedDict()
    for iid, recs in by_inst.items():
        subs = set()
        for r in recs:
            subs.add(r["subj0"]); subs.add(r["subj1"])
        if TARGET in subs:
            black[iid] = recs
    return black


def make_split(black_ids, seed=SEED, build_frac=BUILD_FRAC):
    """Deterministic instance-disjoint split of the Black instance ids."""
    ids = list(black_ids)
    random.Random(seed).shuffle(ids)
    n_build = int(round(build_frac * len(ids)))
    build = sorted(ids[:n_build])
    ev = sorted(ids[n_build:])
    return build, ev


# --------------------------------------------------------------------------- #
# Manifest (COMMITTED, the reproducible source of truth for the split).        #
# --------------------------------------------------------------------------- #
def build_manifest(items_path=FULL_ITEMS, out_path=MANIFEST,
                   seed=SEED, build_frac=BUILD_FRAC):
    black = load_black_instances(items_path)
    build, ev = make_split(list(black.keys()), seed, build_frac)
    assert set(build).isdisjoint(set(ev)), "build/eval overlap!"
    assert set(build) | set(ev) == set(black.keys()), "split lost instances!"
    manifest = {
        "benchmark": "unqover_ethnicity",
        "target_subject": TARGET,
        "source_items": os.path.relpath(items_path, _ROOT),
        "bias_class": "ethnicity",
        "seed": seed,
        "build_frac": build_frac,
        "split_by": "instance_id (order-independent unordered-pair key)",
        "n_total_instances": None,             # filled below (needs full file)
        "n_black_instances": len(black),
        "n_build": len(build),
        "n_eval": len(ev),
        "disjoint": True,
        "note": ("Instance-disjoint train/eval split of the 262 Black-containing "
                 "UNQOVER ethnicity instances. BUILD builds the UNQOVER-native "
                 "diff-in-means direction + all baseline fits; EVAL is the "
                 "held-out eval set (never seen by the builder). Every listed "
                 "instance has 4 complete records (2 positions x q0/q1)."),
        "build_instance_ids": build,
        "eval_instance_ids": ev,
    }
    # count total instances (all classes) for context
    total = set()
    with open(items_path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                total.add(json.loads(line)["instance_id"])
    manifest["n_total_instances"] = len(total)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"[manifest] {len(black)} Black instances -> build={len(build)} "
          f"eval={len(ev)} (seed={seed}, frac={build_frac}) -> {out_path}")
    return manifest


def load_manifest(path=MANIFEST):
    with open(path) as fh:
        return json.load(fh)


# --------------------------------------------------------------------------- #
# Split item files (the runners' --items) -- emitted from the manifest.        #
# --------------------------------------------------------------------------- #
def write_splits(manifest_path=MANIFEST, items_path=FULL_ITEMS,
                 out_dir=os.path.dirname(ARROWS_OUT)):
    m = load_manifest(manifest_path)
    black = load_black_instances(items_path)
    os.makedirs(out_dir, exist_ok=True)
    outs = {}
    for name, ids in (("build_items.jsonl", m["build_instance_ids"]),
                      ("eval_items.jsonl", m["eval_instance_ids"])):
        p = os.path.join(out_dir, name)
        n = 0
        with open(p, "w") as fh:
            for iid in ids:
                for r in black[iid]:
                    fh.write(json.dumps(r) + "\n")
                    n += 1
        outs[name] = (p, n, len(ids))
        print(f"[splits] {name}: {len(ids)} instances / {n} items -> {p}")
    return outs


# --------------------------------------------------------------------------- #
# Contrast items (Black vs paired-other, at the 2-choice prompt answer span).   #
# --------------------------------------------------------------------------- #
def build_contrast_items(black, build_ids, tok=None):
    """One contrast item per BUILD instance in calib.collect_activations shape:
    {chat_prompt, target_answer_text, other_answer_text, ...}.  A canonical
    record (Black first, q0) supplies the prompt; positive=Black subject text,
    negative=the paired other subject text.  chat_prompt is None unless a
    tokenizer is passed (only needed for the GPU forward)."""
    items = []
    for iid in build_ids:
        recs = black[iid]
        rec = next((r for r in recs if r["subj0"] == TARGET and r["qid"] == "q0"),
                   None)
        if rec is None:
            rec = next((r for r in recs if TARGET in (r["subj0"], r["subj1"])
                        and r["qid"] == "q0"), recs[0])
        other = rec["subj1"] if rec["subj0"] == TARGET else rec["subj0"]
        tidx = 0 if rec["subj0"] == TARGET else 1
        chat = None
        if tok is not None:
            prompt = U.build_prompt(rec)
            chat = tok.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True, tokenize=False)
        items.append({
            "instance_id": iid,
            "chat_prompt": chat,
            "target_answer_text": TARGET,
            "other_answer_text": other,
            "target_idx": tidx,
            "other_idx": 1 - tidx,
            "target_subject": TARGET,
            "other_subject": other,
        })
    return items


# --------------------------------------------------------------------------- #
# GPU build: native activations -> arrows + 5 baseline fits.                    #
# --------------------------------------------------------------------------- #
def build_all(manifest_path=MANIFEST, items_path=FULL_ITEMS,
              cache_dir=NATIVE_CACHE, arrows_out=ARROWS_OUT):
    """NEEDS A GPU.  Collect UNQOVER-native pooled activations from the BUILD
    split and write the arrow set + every native baseline fit artifact."""
    import common      # noqa: E402  (load_model)
    import calib       # noqa: E402  (collect_activations pooling, reused verbatim)
    import directions  # noqa: E402  (auroc_per_neuron)
    import meanact     # noqa: E402
    import linearact   # noqa: E402
    import itic        # noqa: E402

    m = load_manifest(manifest_path)
    black = load_black_instances(items_path)
    model, tok = common.load_model()
    items = build_contrast_items(black, m["build_instance_ids"], tok=tok)
    print(f"[build] {len(items)} BUILD contrast items (Black vs paired other), "
          f"target={TARGET!r}", flush=True)

    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(os.path.dirname(arrows_out), exist_ok=True)

    # ---- 1) pooled activations at all 3 granularities (SAME items) ---------- #
    blobs = {}
    for where in WHERES:
        blob = calib.collect_activations(where, model=model, tok=tok,
                                         items=items, save=False, target="black")
        blob["source"] = SOURCE                  # UNQOVER-native provenance
        blob["target"] = "black_unqover"
        blob["benchmark"] = "unqover_ethnicity"
        out = os.path.join(cache_dir, f"calib_{where}.pt")
        torch.save(blob, out)
        print(f"[build] calib {where}: acts={tuple(blob['acts'].shape)} -> {out}",
              flush=True)
        blobs[where] = blob

    # ---- 2) UNQOVER-native arrows (diff-in-means at the block residual) ------ #
    bblob = blobs["block"]
    acts = bblob["acts"].to(torch.float32)       # (2n,32,4096)
    labels = bblob["labels"].to(torch.bool)      # (2n,) 1=Black
    r = acts[labels].mean(0) - acts[~labels].mean(0)     # (32,4096) RAW diff
    torch.save({
        "r": r, "n_layers": N_LAYERS, "n_items": int(labels.sum()),
        "capped": False, "cap": None,
        "per_layer_raw_norm": [float(r[l].norm()) for l in range(N_LAYERS)],
        "method": "unqover_native_diff_in_means_block",
        "source": SOURCE, "target_subject": TARGET,
    }, arrows_out)
    print(f"[build] arrows_unqover_black -> {arrows_out} "
          f"raw_norms[:4]={[round(float(r[l].norm()),2) for l in range(4)]}",
          flush=True)

    # ---- 3) actadd single-pair (block diff of BUILD item 0) ----------------- #
    h_t = acts[0]                                 # item0 target (label 1)
    h_o = acts[1]                                 # item0 other  (label 0)
    r_actadd = (h_t - h_o).to(torch.float32)      # (32,4096)
    ap = os.path.join(cache_dir, "actadd_dir.pt")
    torch.save({
        "r": r_actadd, "n_layers": N_LAYERS, "n_items": 1, "pair_index": 0,
        "method": "actadd_single_pair", "granularity": "block_residual",
        "other_subject": items[0]["other_subject"],
        "source": SOURCE, "target": "black_unqover",
        "per_layer_raw_norm": [float(r_actadd[k].norm()) for k in range(N_LAYERS)],
    }, ap)
    print(f"[build] actadd_dir (single pair, other={items[0]['other_subject']!r}) "
          f"-> {ap}", flush=True)

    # ---- 4) mean-act per-neuron (mu2-mu1) at block granularity -------------- #
    mp = os.path.join(cache_dir, "meanact_meandiff_block.pt")
    meanact.fit(where="block",
                calib_path=os.path.join(cache_dir, "calib_block.pt"),
                save=True, out_path=mp, target="black")
    print(f"[build] meanact_meandiff_block -> {mp}", flush=True)

    # ---- 5) linear-act gaussian+empirical OT (mlp_hidden) ------------------- #
    stats = linearact.fit_stats_from_blob(blobs["mlp_hidden"])
    stats["source"] = SOURCE
    stats["target"] = "black_unqover"
    lp = os.path.join(cache_dir, "linearact_stats.pt")
    torch.save(stats, lp)
    print(f"[build] linearact_stats ({stats['n_layers']}x{stats['feat']}) -> {lp}",
          flush=True)

    # ---- 6) aura per-neuron AUROC (mlp_hidden) ------------------------------ #
    mblob = blobs["mlp_hidden"]
    m_acts = mblob["acts"]
    m_lab = mblob["labels"]
    feat = m_acts.shape[-1]
    auroc = torch.empty(N_LAYERS, feat, dtype=torch.float32)
    for k in range(N_LAYERS):
        a_k, _gate = directions.auroc_per_neuron(m_acts[:, k, :], m_lab)
        auroc[k] = a_k
    up = os.path.join(cache_dir, "aura_auroc.pt")
    torch.save({"auroc": auroc, "where": "mlp_hidden", "feat": feat,
                "n_layers": N_LAYERS, "n_items": mblob["n_items"],
                "source": SOURCE, "target": "black_unqover"}, up)
    print(f"[build] aura_auroc ({N_LAYERS}x{feat}) -> {up}", flush=True)

    # ---- 7) iti-c per-head probes (attn_head) ------------------------------- #
    ablob = blobs["attn_head"]
    a_acts = ablob["acts"].to(torch.float32)      # (2n,32,4096)
    a_lab = ablob["labels"].long()
    n_heads, d_head = itic.N_HEADS, itic.D_HEAD
    A = a_acts.view(a_acts.shape[0], N_LAYERS, n_heads, d_head)
    theta = torch.zeros(N_LAYERS, n_heads, d_head)
    sigma = torch.zeros(N_LAYERS, n_heads)
    val_acc = torch.zeros(N_LAYERS, n_heads)
    for k in range(N_LAYERS):
        for hh in range(n_heads):
            va, th, sg = itic._fit_one_head(A[:, k, hh, :], a_lab)
            val_acc[k, hh] = va; theta[k, hh] = th; sigma[k, hh] = sg
    ip = os.path.join(cache_dir, "itic_probes.pt")
    torch.save({"theta": theta, "sigma": sigma, "val_acc": val_acc,
                "n_layers": N_LAYERS, "n_heads": n_heads, "d_head": d_head,
                "n_items": ablob["n_items"], "source": SOURCE,
                "method": "iti_c", "target": "black_unqover",
                "direction": "mass_mean_shift_Black_minus_other"}, ip)
    print(f"[build] itic_probes -> {ip}  best_val_acc={float(val_acc.max()):.3f}",
          flush=True)
    print("[build] DONE: UNQOVER-native arrows + 5 baseline fits written.",
          flush=True)


# --------------------------------------------------------------------------- #
# CPU self-test: split disjointness + counts + contrast build + load checks.    #
# --------------------------------------------------------------------------- #
def selftest(manifest_path=MANIFEST, items_path=FULL_ITEMS,
             cache_dir=NATIVE_CACHE, arrows_out=ARROWS_OUT):
    ok = True

    def check(name, cond):
        nonlocal ok
        ok &= bool(cond)
        print(f"[selftest] {name:56s} : {'PASS' if cond else 'FAIL'}")

    have_data = os.path.exists(items_path)
    check(f"full items file resolves ({os.path.relpath(items_path, _ROOT)})",
          have_data)
    have_manifest = os.path.exists(manifest_path)
    check("manifest exists", have_manifest)

    if have_manifest:
        m = load_manifest(manifest_path)
        b, e = set(m["build_instance_ids"]), set(m["eval_instance_ids"])
        check("build INTERSECT eval == empty (instance-disjoint)", b.isdisjoint(e))
        check("no dup ids within build",
              len(m["build_instance_ids"]) == len(b))
        check("no dup ids within eval", len(m["eval_instance_ids"]) == len(e))
        check(f"n_build+n_eval == n_black ({m['n_build']}+{m['n_eval']}"
              f"=={m['n_black_instances']})",
              m["n_build"] + m["n_eval"] == m["n_black_instances"]
              and len(b) + len(e) == m["n_black_instances"])
        print(f"[selftest] split: build={m['n_build']} eval={m['n_eval']} "
              f"(of {m['n_black_instances']} Black instances; seed={m['seed']}, "
              f"frac={m['build_frac']})")

        if have_data:
            black = load_black_instances(items_path)
            check(f"data has {m['n_black_instances']} Black instances "
                  f"(matches manifest)", len(black) == m["n_black_instances"])
            check("every manifest id is a real Black instance",
                  (b | e).issubset(set(black.keys())))
            check("every Black instance has 4 complete records",
                  all(len(black[i]) == 4 for i in (b | e)))
            # contrast build (CPU: chat_prompt None) -- shape + Black membership
            items = build_contrast_items(black, m["build_instance_ids"], tok=None)
            check(f"contrast items == n_build ({len(items)})",
                  len(items) == m["n_build"])
            check("every contrast: positive=Black, negative!=Black, other in pair",
                  all(it["target_answer_text"] == TARGET
                      and it["other_answer_text"] != TARGET
                      and TARGET in (black[it["instance_id"]][0]["subj0"],
                                     black[it["instance_id"]][0]["subj1"])
                      for it in items))
            # a real 2-choice prompt for the first build instance
            rec0 = black[m["build_instance_ids"][0]][0]
            prompt = U.build_prompt(rec0)
            nchoice = sum(1 for ln in prompt.splitlines() if ln[:2] in ("A.", "B."))
            check("build_prompt yields a 2-choice A/B prompt (no C.)",
                  nchoice == 2 and "C." not in prompt)

    # optional artifact load checks (only if a GPU build already ran)
    if os.path.exists(arrows_out):
        blob = torch.load(arrows_out, map_location="cpu")
        check("arrows_unqover_black.pt: r is (32,4096)",
              tuple(blob["r"].shape) == (32, 4096))
    else:
        print(f"[selftest] arrows load check: SKIP (not built yet: {arrows_out})")
    for name, key in (("calib_block.pt", "acts"), ("actadd_dir.pt", "r"),
                      ("linearact_stats.pt", "feat"), ("aura_auroc.pt", "auroc"),
                      ("itic_probes.pt", "theta"),
                      ("meanact_meandiff_block.pt", "mean_diff")):
        p = os.path.join(cache_dir, name)
        if os.path.exists(p):
            blob = torch.load(p, map_location="cpu")
            check(f"{name}: has '{key}' + UNQOVER-native source",
                  key in blob and blob.get("source") == SOURCE)
        else:
            print(f"[selftest] {name}: SKIP (not built yet)")

    print(f"[selftest] OVERALL: {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build-manifest", action="store_true",
                    help="CPU: write the committed train/eval split manifest")
    ap.add_argument("--write-splits", action="store_true",
                    help="CPU: emit build_items.jsonl / eval_items.jsonl from the manifest")
    ap.add_argument("--build", action="store_true",
                    help="GPU: build native arrows + all baseline fits from the BUILD split")
    ap.add_argument("--selftest", action="store_true",
                    help="CPU: disjointness + counts + contrast build + load checks")
    ap.add_argument("--items", default=FULL_ITEMS)
    ap.add_argument("--manifest", default=MANIFEST)
    ap.add_argument("--cache-dir", default=NATIVE_CACHE)
    ap.add_argument("--arrows-out", default=ARROWS_OUT)
    ap.add_argument("--out-dir", default=os.path.dirname(ARROWS_OUT),
                    help="dir for --write-splits jsonl files")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--build-frac", type=float, default=BUILD_FRAC)
    args = ap.parse_args()

    if args.build_manifest:
        build_manifest(args.items, args.manifest, args.seed, args.build_frac)
        return
    if args.write_splits:
        write_splits(args.manifest, args.items, args.out_dir)
        return
    if args.build:
        build_all(args.manifest, args.items, args.cache_dir, args.arrows_out)
        return
    if args.selftest:
        sys.exit(0 if selftest(args.manifest, args.items, args.cache_dir,
                               args.arrows_out) else 1)
    ap.error("nothing to do: pass --build-manifest / --write-splits (CPU), "
             "--build (GPU), or --selftest (CPU)")


if __name__ == "__main__":
    main()
