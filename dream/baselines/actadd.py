#!/usr/bin/env python
"""dream/baselines/actadd.py -- ActAdd (Turner et al. 2023) on Dream-v0-Instruct-7B.

Port of baselines/actadd.py.  ActAdd = a steering vector from a SINGLE contrast
prompt-PAIR (one Black minus one other) at ONE layer, scaled by a coefficient and
ADDED to the residual stream.  The n=1, single-layer, purely-additive special case
of the diff-in-means / CAA family.

    r_actadd[k] = h_black[k] - h_other[k]      (ONE held-out item, not the n mean)
    steer(h)    = h + alpha * unit(r_actadd[layer])          at a single block

Defining property vs CAA: n=1 contrast pair, NOT the n=400 dataset mean
(dream/arrows.pt).  We deliberately do NOT auto-substitute arrows.pt: if the fit
artifact is missing, build_injection RAISES (that would be CAA, not ActAdd).
NATIVE granularity = residual at ONE block (3584-d), model.layers[layer]
OUTPUT[0] via common_dream.add_vec_hook, all positions, every diffusion step.

CLI:
    python dream/baselines/actadd.py --selftest                       # offline
    CUDA_VISIBLE_DEVICES=4 python dream/baselines/actadd.py --fit      # NEEDS GPU
    CUDA_VISIBLE_DEVICES=4 python dream/baselines/actadd.py --run --alpha 8 --layer 14
"""
import argparse
import os
import sys

import torch

ROOT = os.environ.get("DLM_BIAS_ROOT") or os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "eval"))
sys.path.insert(0, os.path.join(ROOT, "steering"))
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))   # common_dream
sys.path.insert(0, _HERE)

import common_dream as C  # noqa: E402  (re-exports unit_rows from the LLaDA steering/pid_steer.py)
import calib            # noqa: E402

DEFAULT_LAYER = 14
DEFAULT_ALPHA = 8.0
DIR_PATH = os.path.join(calib.CACHE_DIR, "actadd_dir.pt")
RESULTS_DIR = os.path.join(ROOT, "results", "dream", "actadd")


def fit(pair_index=0, model=None, tok=None, save=True, out_path=DIR_PATH):
    """Build the ActAdd single-pair direction r_actadd (28,3584) and cache it.

    Picks ONE Black-vs-other contrast pair (held-out item `pair_index`), reuses the
    calib block-granularity masked-mean pool on that single item, and computes
    r_actadd[k] = h_black[k] - h_other[k] for every block.  NEEDS A GPU."""
    if model is None or tok is None:
        model, tok = C.load_model_tok()
    items = calib.heldout_items(tok=tok)
    assert len(items) > 0, "no held-out contrast items available"
    assert 0 <= pair_index < len(items), \
        f"pair_index {pair_index} out of range [0,{len(items)})"
    item = items[pair_index]

    blob = calib.collect_activations("block", model=model, tok=tok,
                                     items=[item], save=False)
    acts = blob["acts"].to(torch.float32)      # (2, 28, 3584)
    labels = blob["labels"]                    # tensor([1, 0])
    h_black = acts[(labels == 1).nonzero(as_tuple=True)[0][0]]   # (28,3584)
    h_other = acts[(labels == 0).nonzero(as_tuple=True)[0][0]]
    r = h_black - h_other                      # (28,3584)

    out = {
        "r": r, "n_layers": C.N_LAYERS, "n_items": 1, "pair_index": pair_index,
        "method": "actadd_single_pair", "granularity": "block_residual",
        "black_idx": item["black_idx"], "other_idx": item["other_idx"],
        "black_answer_text": item["black_answer_text"],
        "other_answer_text": item["other_answer_text"],
        "source": "bbq_race_ethnicity_heldout_disjoint_seed42_and_sweep400",
        "per_layer_raw_norm": [float(r[k].norm()) for k in range(C.N_LAYERS)],
    }
    if save:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        torch.save(out, out_path)
        print(f"[actadd:fit] SAVED single-pair dir {tuple(r.shape)} "
              f"(pair_index={pair_index}) -> {out_path}", flush=True)
    return out


def load_direction(path=DIR_PATH):
    """Load the cached single-pair r_actadd (28,3584), raw."""
    return torch.load(path, map_location="cpu")["r"].to(torch.float32)


def build_injection(alpha=DEFAULT_ALPHA, layer=DEFAULT_LAYER, r=None, path=DIR_PATH):
    """Return the additive injection at `layer`:  alpha * unit(r_actadd[layer]).

    If r is None we load the fitted cache/actadd_dir.pt.  RAISES if missing -- we
    do NOT silently substitute arrows.pt (that is CAA n=400, not ActAdd n=1)."""
    if r is None:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"ActAdd direction not fitted: {path} missing. Run "
                f"`python dream/baselines/actadd.py --fit` (GPU) first. Do NOT "
                f"substitute arrows.pt -- that is CAA (n=400), not ActAdd (n=1).")
        r = load_direction(path)
    r = torch.as_tensor(r, dtype=torch.float32)
    assert r.shape == (C.N_LAYERS, C.D_MODEL), \
        f"expected r shape ({C.N_LAYERS},{C.D_MODEL}), got {tuple(r.shape)}"
    assert 0 <= layer < C.N_LAYERS, f"layer {layer} out of range"
    return alpha * C.unit_rows(r)[layer]     # (3584,)


def make_attach_fn(alpha=DEFAULT_ALPHA, layer=DEFAULT_LAYER, r=None, path=DIR_PATH):
    """attach_fn(model)->handles adding alpha*unit(r[layer]) at block `layer`."""
    vec = build_injection(alpha=alpha, layer=layer, r=r, path=path)

    def attach_fn(model):
        return C.attach(model, [C.block_path(int(layer))], C.add_vec_hook(vec))

    return attach_fn


def run(out_dir=RESULTS_DIR, alpha=DEFAULT_ALPHA, layer=DEFAULT_LAYER,
        items_path=None, limit=0, baseline_black_rate=None, model=None, tok=None,
        tag=None):
    """Run the ActAdd BBQ-injection condition end-to-end.  NEEDS A GPU.

    Default tag encodes alpha (actadd_a<alpha>) so sweeps at different alphas
    don't overwrite each other; an explicit tag still wins."""
    tag = tag or f"actadd_a{alpha:g}"
    return C.run_items(
        attach_fn=make_attach_fn(alpha=alpha, layer=layer),
        items_path=items_path or C.SWEEP400,
        out_dir=out_dir, tag=tag, limit=limit,
        baseline_black_rate=baseline_black_rate, model=model, tok=tok,
        config_extra={"method": "actadd_single_pair", "alpha": alpha,
                      "layer": layer, "n_items_fit": 1,
                      "granularity": "block_residual"})


def _selftest():
    torch.manual_seed(0)
    ok = True

    def check(name, cond):
        nonlocal ok
        ok &= bool(cond)
        print(f"[selftest-actadd] {name:52s} : {'PASS' if cond else 'FAIL'}")

    L, H = C.N_LAYERS, C.D_MODEL
    h_black = torch.randn(L, H); h_other = torch.randn(L, H)
    r = h_black - h_other
    check("single-pair r == h_black - h_other", torch.allclose(r, h_black - h_other))
    check("r shape (28,3584)", tuple(r.shape) == (L, H))

    alpha, layer = 8.0, DEFAULT_LAYER
    vec = build_injection(alpha=alpha, layer=layer, r=r)
    check("vec shape (3584,)", tuple(vec.shape) == (H,))
    check("||vec|| == alpha", torch.allclose(vec.norm(), torch.tensor(alpha), atol=1e-4))
    cos = torch.dot(vec, r[layer]) / (vec.norm() * r[layer].norm())
    check("vec parallel to r[layer] (cos==1)",
          torch.allclose(cos, torch.tensor(1.0), atol=1e-4))
    check("positive alpha points toward Black", float(torch.dot(vec, r[layer])) > 0)
    check("negative alpha flips",
          float(torch.dot(build_injection(alpha=-alpha, layer=layer, r=r), r[layer])) < 0)
    check("alpha=0 -> zero vector",
          torch.allclose(build_injection(alpha=0.0, layer=layer, r=r), torch.zeros(H)))

    hook = C.add_vec_hook(vec)
    B, S = 2, 5
    hh = torch.randn(B, S, H)
    out = hook(None, None, (hh.clone(), "cache"))
    check("additive hook: h <- h + vec (tuple preserved)",
          isinstance(out, tuple) and out[1] == "cache" and torch.allclose(out[0], hh + vec))

    # missing-artifact guard raises (no silent CAA fallback).
    missing = os.path.join(calib.CACHE_DIR, "definitely_absent_actadd_dir.pt")
    raised = False
    try:
        build_injection(alpha=alpha, layer=layer, path=missing)
    except FileNotFoundError:
        raised = True
    check("missing fit artifact raises (no silent CAA fallback)", raised)

    print(f"[selftest-actadd] OVERALL: {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--fit", action="store_true", help="build single-pair dir (NEEDS GPU)")
    ap.add_argument("--run", action="store_true", help="run ActAdd condition (NEEDS GPU)")
    ap.add_argument("--pair-index", type=int, default=0)
    ap.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    ap.add_argument("--layer", type=int, default=DEFAULT_LAYER)
    ap.add_argument("--items", default=None,
                    help="BBQ items jsonl (default: sweep400; e.g. a position-balance rotation file)")
    ap.add_argument("--out_dir", default=RESULTS_DIR)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--baseline_black_rate", type=float, default=None)
    args = ap.parse_args()

    if args.selftest:
        sys.exit(0 if _selftest() else 1)
    if args.fit:
        fit(pair_index=args.pair_index); return
    if args.run:
        run(out_dir=args.out_dir, alpha=args.alpha, layer=args.layer,
            items_path=args.items, limit=args.limit,
            baseline_black_rate=args.baseline_black_rate, tag=args.tag)
        return
    ap.error("nothing to do: pass --selftest, --fit (GPU) or --run (GPU)")


if __name__ == "__main__":
    main()
