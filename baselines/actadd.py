#!/usr/bin/env python
"""ActAdd baseline (Turner et al., 2023, "Activation Addition: Steering
Language Models Without Optimization") at its native granularity.

A steering vector is built from a single contrast prompt pair (positive minus
negative) at one layer, scaled by a coefficient and added to the residual
stream during generation:
    r_actadd[k] = h_target[k] - h_other[k],   h <- h + alpha * unit(r[layer])
This is the n=1, single-layer, purely additive special case of the
diff-in-means / CAA family (CAA uses the n=400 mean over all layers).  The
additive edit is the AcT InterventionHook output edit
(act/hooks/intervention_hook.py).

Native granularity: the residual stream at one block (blocks[L] OUTPUT[0],
H=4096).  The target answer is the positive prompt and the non-target answer
the negative prompt, so positive alpha injects toward the target (same sign
convention as steering/arrows.pt).  Position: all tokens, bidirectional.

Defaults: layer 14, alpha 8.0, pair index 0.

Usage:
    python baselines/actadd.py --selftest                    # offline, no GPU
    python baselines/actadd.py --fit [--pair-index N]        # GPU
    python baselines/actadd.py --run [--alpha A --layer L]   # GPU
"""
import argparse
import os
import sys

import torch

# Harness import (same convention as steering/pid_steer.py).
ROOT = os.environ.get("DLM_BIAS_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "eval"))
sys.path.insert(0, os.path.join(ROOT, "steering"))
import pid_steer  # noqa: E402  (unit_rows)

import common  # noqa: E402
import calib   # noqa: E402
from common import (  # noqa: E402
    CACHE_DIR,
    N_LAYERS,
    H_MODEL,
    add_vec_hook,
    attach,
    block_paths,
    run_baseline,
)

DEFAULT_LAYER = 14
DEFAULT_ALPHA = 8.0
DIR_PATH = os.path.join(CACHE_DIR, "actadd_dir.pt")
RESULTS_DIR = os.path.join(ROOT, "results", "actadd")


def dir_path_for(target):
    """cache/actadd_dir.pt for black; cache/actadd_dir_<target>.pt otherwise."""
    return os.path.join(CACHE_DIR, f"actadd_dir{common.target_suffix(target)}.pt")


# --------------------------------------------------------------------------- #
# FIT: build the single-pair direction r_actadd (32,4096).  NEEDS A GPU.       #
# --------------------------------------------------------------------------- #
def fit(pair_index=0, model=None, tok=None, save=True, out_path=None,
        target="black"):
    """Build the ActAdd single-pair direction and cache it.  NEEDS A GPU.

    Picks ONE target-vs-other contrast pair (the held-out item at `pair_index`
    from calib.heldout_items(target=...)), runs a clean
    forward on the fully-materialized prompt+target-answer and
    prompt+other-answer (reusing the SAME masked-mean answer-span pooling as
    build_arrows.all_layer_hidden, via calib.collect_activations on a single
    item), and computes

        r_actadd[k] = h_target[k] - h_other[k]    for every block k (0..31).

    This is build_arrows' per-item difference for ONE item instead of the
    n=400 mean -- i.e. ActAdd, not CAA.

    Caches {"r": (32,4096), ...} to cache/actadd_dir[_<target>].pt.
    """
    out_path = out_path or dir_path_for(target)
    if model is None or tok is None:
        model, tok = common.load_model()
    items = calib.heldout_items(tok=tok, target=target)
    assert len(items) > 0, "no held-out contrast items available"
    assert 0 <= pair_index < len(items), (
        f"pair_index {pair_index} out of range [0,{len(items)})")
    item = items[pair_index]

    # Reuse the exact calib collection pipeline (build_arrows masked-mean pool,
    # block granularity) on this ONE item -> acts (2,32,4096), labels [1,0].
    blob = calib.collect_activations(
        "block", model=model, tok=tok, items=[item], save=False, target=target)
    acts = blob["acts"].to(torch.float32)      # (2, 32, 4096)
    labels = blob["labels"]                    # tensor([1, 0])
    h_target = acts[(labels == 1).nonzero(as_tuple=True)[0][0]]  # (32,4096)
    h_other = acts[(labels == 0).nonzero(as_tuple=True)[0][0]]   # (32,4096)
    r = h_target - h_other                     # (32,4096)  target-minus-other

    out = {
        "r": r,
        "n_layers": N_LAYERS,
        "n_items": 1,
        "pair_index": pair_index,
        "method": "actadd_single_pair",
        "granularity": "block_residual",
        "other_idx": item["other_idx"],
        "other_answer_text": item["other_answer_text"],
        "source": calib.SOURCE_BY_TARGET[target],
        "per_layer_raw_norm": [float(r[k].norm()) for k in range(N_LAYERS)],
    }
    if target == "black":   # black metadata key names
        out["black_idx"] = item["target_idx"]
        out["black_answer_text"] = item["target_answer_text"]
    else:
        out["target"] = target
        out["target_idx"] = item["target_idx"]
        out["target_answer_text"] = item["target_answer_text"]
    if save:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        torch.save(out, out_path)
        print(f"[actadd:fit] SAVED single-pair dir {tuple(r.shape)} "
              f"(pair_index={pair_index}, target={target}) -> {out_path}", flush=True)
    return out


def load_direction(path=DIR_PATH):
    """Load the cached single-pair r_actadd (32,4096), raw (un-normalized)."""
    blob = torch.load(path, map_location="cpu")
    return blob["r"].to(torch.float32)


# --------------------------------------------------------------------------- #
# Direction -> injection vector: vec = alpha * unit(r_actadd[layer]).          #
# --------------------------------------------------------------------------- #
def build_injection(alpha=DEFAULT_ALPHA, layer=DEFAULT_LAYER, r=None,
                    path=DIR_PATH):
    """Return the additive injection vector at `layer`:  alpha * unit(r[layer]).

    r: optional pre-loaded (32,4096) single-pair direction; if None we load the
    fitted cache/actadd_dir.pt (the ActAdd n=1 artifact).

    FALLBACK NOTE: if the fit artifact is absent one COULD substitute the raw
    steering/arrows.pt row (directions.load_arrows(raw=True)[layer]) -- but that
    is the n=400 dataset mean, i.e. CAA, NOT ActAdd (n=1).  We deliberately do
    NOT auto-substitute: the whole point of this baseline is n=1 vs n=400.  If
    the cache is missing, --fit must be run first (GPU).

    Per-layer unit-normalization uses pid_steer.unit_rows, the
    same normalizer every other method in this suite uses, so ||vec|| == alpha.
    """
    if r is None:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"ActAdd direction not fitted: {path} missing. Run "
                f"`python baselines/actadd.py --fit` (GPU) first. Do NOT "
                f"silently substitute arrows.pt -- that is CAA (n=400), not "
                f"ActAdd (n=1).")
        r = load_direction(path)
    r = torch.as_tensor(r, dtype=torch.float32)
    assert r.shape == (N_LAYERS, H_MODEL), (
        f"expected r shape ({N_LAYERS},{H_MODEL}), got {tuple(r.shape)}")
    assert 0 <= layer < N_LAYERS, f"layer {layer} out of range [0,{N_LAYERS})"
    u = pid_steer.unit_rows(r)          # (32,4096), each row unit-norm
    return alpha * u[layer]             # (4096,)


# --------------------------------------------------------------------------- #
# Attach: additive residual hook at a SINGLE block layer.                      #
# --------------------------------------------------------------------------- #
def make_attach_fn(alpha=DEFAULT_ALPHA, layer=DEFAULT_LAYER, r=None,
                   path=DIR_PATH):
    """Return attach_fn(model)->handles that adds alpha*unit(r[layer]) to the
    residual stream at block `layer` (all positions, bidirectional)."""
    vec = build_injection(alpha=alpha, layer=layer, r=r, path=path)

    def attach_fn(model):
        return attach(model, block_paths([layer]), add_vec_hook(vec), pre=False)

    return attach_fn


def run(out_dir=RESULTS_DIR, alpha=DEFAULT_ALPHA, layer=DEFAULT_LAYER,
        limit=0, baseline_black_rate=None, model=None, tok=None, tag=None,
        target="black", **kw):
    """Run the ActAdd BBQ-injection condition end-to-end.  NEEDS A GPU.

    Writes cond_<tag>.json / cond_<tag>_samples.jsonl under out_dir via the
    shared common.run_baseline driver (identical scoring & file format to every
    other baseline).  Default tag encodes alpha (actadd_a<alpha>) so sweeps at
    different alphas don't overwrite each other.  target selects the fitted
    single-pair direction (cache/actadd_dir[_<target>].pt) AND the eval
    classification."""
    tag = tag or f"actadd_a{alpha:g}"
    attach_fn = make_attach_fn(alpha=alpha, layer=layer, path=dir_path_for(target))
    return run_baseline(
        attach_fn, out_dir=out_dir, tag=tag, limit=limit,
        baseline_black_rate=baseline_black_rate, model=model, tok=tok,
        target=target,
        config_extra={"method": "actadd_single_pair", "alpha": alpha,
                      "layer": layer, "n_items_fit": 1, "target": target,
                      "granularity": "block_residual"},
        **kw)


# --------------------------------------------------------------------------- #
# Offline self-test: single-pair diff + injection math on synthetic tensors.   #
# --------------------------------------------------------------------------- #
def _selftest():
    torch.manual_seed(0)
    ok = True

    def check(name, cond):
        nonlocal ok
        ok &= bool(cond)
        print(f"[selftest-actadd] {name:52s} : {'PASS' if cond else 'FAIL'}")

    # Synthetic single-pair pooled activations (h_black, h_other) at 32 layers.
    h_black = torch.randn(N_LAYERS, H_MODEL)
    h_other = torch.randn(N_LAYERS, H_MODEL)
    r = h_black - h_other                       # the ActAdd single-pair direction

    check("single-pair r == h_black - h_other",
          torch.allclose(r, h_black - h_other))
    check("r shape (32,4096)", tuple(r.shape) == (N_LAYERS, H_MODEL))

    # Injection vector at default layer: alpha * unit(r[layer]).
    alpha, layer = 8.0, DEFAULT_LAYER
    vec = build_injection(alpha=alpha, layer=layer, r=r)
    check("vec shape (4096,)", tuple(vec.shape) == (H_MODEL,))
    check("||vec|| == alpha (unit-normalized row scaled by alpha)",
          torch.allclose(vec.norm(), torch.tensor(alpha), atol=1e-4))
    # direction parallel to r[layer]
    cos = torch.dot(vec, r[layer]) / (vec.norm() * r[layer].norm())
    check("vec is parallel to r[layer] (cos==1)",
          torch.allclose(cos, torch.tensor(1.0), atol=1e-4))

    # sign: positive alpha points toward Black (h_black), i.e. dot with
    # (h_black-h_other)[layer] is positive.
    check("positive alpha points toward Black (r-dir)",
          float(torch.dot(vec, r[layer])) > 0)
    vec_neg = build_injection(alpha=-alpha, layer=layer, r=r)
    check("negative alpha flips direction",
          float(torch.dot(vec_neg, r[layer])) < 0)

    # alpha=0 -> zero injection (clean pass-through).
    check("alpha=0 -> zero vector",
          torch.allclose(build_injection(alpha=0.0, layer=layer, r=r),
                         torch.zeros(H_MODEL)))

    # different layer selects a different row.
    vec_l0 = build_injection(alpha=alpha, layer=0, r=r)
    cos0 = torch.dot(vec_l0, r[0]) / (vec_l0.norm() * r[0].norm())
    check("layer arg selects that row", torch.allclose(cos0, torch.tensor(1.0),
                                                       atol=1e-4))

    # attach_fn builds an additive hook that does h <- h + vec on a block tuple.
    hook = add_vec_hook(vec)
    B, S = 2, 5
    h = torch.randn(B, S, H_MODEL)
    tup = (h.clone(), "cache")
    out = hook(None, None, tup)
    check("additive hook: h <- h + vec (tuple preserved)",
          isinstance(out, tuple) and out[1] == "cache"
          and torch.allclose(out[0], h + vec))

    # missing-artifact guard raises (does NOT silently fall back to arrows.pt).
    missing = os.path.join(CACHE_DIR, "definitely_absent_actadd_dir.pt")
    raised = False
    try:
        build_injection(alpha=alpha, layer=layer, path=missing)
    except FileNotFoundError:
        raised = True
    check("missing fit artifact raises (no silent CAA fallback)", raised)

    # target-parameterized fit-artifact paths.
    check("dir_path_for('black') == DIR_PATH (regression)",
          dir_path_for("black") == DIR_PATH)
    check("dir_path_for gender -> cache/actadd_dir_<t>.pt",
          dir_path_for("woman").endswith("cache/actadd_dir_woman.pt")
          and dir_path_for("man").endswith("cache/actadd_dir_man.pt"))

    print(f"[selftest-actadd] OVERALL: {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true",
                    help="offline single-pair/injection math check (no GPU)")
    ap.add_argument("--fit", action="store_true",
                    help="build the single-pair direction (NEEDS GPU + model)")
    ap.add_argument("--run", action="store_true",
                    help="run the ActAdd BBQ condition (NEEDS GPU + model)")
    ap.add_argument("--pair-index", type=int, default=0,
                    help="which held-out contrast pair to use for the fit")
    ap.add_argument("--alpha", type=float, default=DEFAULT_ALPHA,
                    help="injection strength (positive -> toward the target)")
    ap.add_argument("--layer", type=int, default=DEFAULT_LAYER,
                    help="block layer to add the vector at (native: residual)")
    ap.add_argument("--target", choices=common.SUPPORTED_TARGETS, default="black",
                    help="steering target (black = default; woman/man = "
                         "gender heldout fit + classification)")
    ap.add_argument("--items", default=None,
                    help="BBQ items jsonl (e.g. a position-balance rotation "
                         "file); default = the target's own _sweep400 file")
    ap.add_argument("--out-dir", default=RESULTS_DIR)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--baseline-black-rate", type=float, default=None)
    args = ap.parse_args()

    if args.selftest:
        sys.exit(0 if _selftest() else 1)
    if args.fit:
        fit(pair_index=args.pair_index, target=args.target)
        return
    if args.run:
        run(out_dir=args.out_dir, alpha=args.alpha, layer=args.layer,
            limit=args.limit, baseline_black_rate=args.baseline_black_rate,
            items_path=args.items, target=args.target)
        return
    ap.error("nothing to do: pass --selftest (offline), --fit (GPU) or --run (GPU)")


if __name__ == "__main__":
    main()
