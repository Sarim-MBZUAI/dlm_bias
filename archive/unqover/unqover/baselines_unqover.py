#!/usr/bin/env python
"""The 7 prior-work steering baselines (AcT suite + ITI-C) on UNQOVER.

THIN ADAPTER -- exactly the pattern of pid_steer_unqover.py / denoise_pid_unqover.py,
but for the prior-work baselines instead of our controllers. NOTHING about a
baseline's hook, its fitted artifacts, or its operating point is re-implemented
here: each method's `attach_fn(model) -> handles` closure is IMPORTED from
baselines/<method>.py and built with the SAME Table-1 (Black) operating points and
the SAME black fit artifacts the BBQ Table-1 rows used. Only the generation glue is
swapped from BBQ to UNQOVER (2-choice A/B prompt, subject parse, UNQOVER record
schema) so the records feed unqover_metric.py unchanged.

WHAT IS REUSED
--------------
  * The hook / injection of each baseline, verbatim (target="black" fit artifacts):
      uq_caa          caa.make_attach_fn(layer=14, alpha=16)        steering/arrows.pt
      uq_actadd       actadd.make_attach_fn(alpha=16, layer=14)     cache/actadd_dir.pt
      uq_meanact      meanact.build_attach_fn(2.0, "unit")          steering/arrows.pt
      uq_linearact    linearact.attach_fn(variant="gaussian", s=1)  cache/linearact_stats.pt
      uq_aura_inject  aura.attach_fn("inject", gamma=4)             cache/aura_auroc.pt
      uq_aura_vanilla aura.attach_fn("vanilla")                     cache/aura_auroc.pt
      uq_itic         itic.attach_fn(build_injection(K=48, a=8))    cache/itic_probes.pt
    These are the EXACT operating points of BBQ Table-1 (Black):
    results/{caa,actadd}/cond_mult_a16, results/meanact/cond_s2 (unit s=2),
    results/linearact/cond_gaussian_s1, results/aura/cond_{inject_g4,vanilla},
    results/itic/cond_topk48_a8  (confirmed in results/BASELINES.md and
    results/balanced_all/RESULTS_STRICT.md).
  * The shared fire counter (baselines.common._FIRE): every baseline hook is built
    with a common.py factory that bumps it, so we reset it before the loop and
    assert >0 after -- the same silent-no-attach guard common.run_baseline uses.
  * The UNQOVER glue (item load+target-filter, prompt build, record schema,
    parser) is IMPORTED from denoise_pid_unqover.py, so a record is byte-schema
    identical to uq_decode_PI.jsonl / uq_normal_a4.jsonl.
  * Generation is bbq_eval.generate (the LLaDA masked-diffusion sampler) -- same
    call the BBQ baselines and the UNQOVER controllers already use.

WHY THIS IS ITEM-INDEPENDENT (like the layer-space adapter, unlike decode-PI)
-----------------------------------------------------------------------------
Every baseline here adds a FIXED injection (a per-layer vector / affine / gate)
that does NOT depend on the item's target letter: there is no closed loop. So we
attach once, then for each item: generate -> parse_choice -> pred_subject -> record.
The item set is the SAME Black-containing filter (load_target_items) on the SAME
cap2000 file the existing steered uq runs used (n_items=256 / 64 instances), so the
new rows sit directly beside uq_decode_PI / uq_normal_a4 in the paper's UNQOVER table.

COLLAPSE CASE: a baseline that emits mostly unparseable output still writes a valid
jsonl (pred_subject=None on those items) and a valid config; no_answer is recorded
and unqover_metric.py drops the incomplete instances (reporting coverage). Nothing
special is needed -- the record schema already carries a None pick.

PATHS: data/model/arrows/cache all resolve under DLM_BIAS_ROOT (the MAIN tree),
exactly like every baseline file, so a worktree checkout (which gitignores the
large *.pt and data/unqover/) still finds them when the job runs from the main tree.

GPU RULE: run on an idle allowed GPU (SLURM sets CUDA_VISIBLE_DEVICES). --selftest
is CPU-only: it loads each black fit artifact on CPU, builds the injection, and
checks the record schema -- NO model, NO GPU.
"""
import argparse
import json
import os
import sys
import time

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))                        # unqover
_ROOT = os.environ.get("DLM_BIAS_ROOT") or os.path.dirname(_HERE)

# baselines/ must be importable as a top-level dir so `import caa` and its own
# `import common / directions / calib` resolve (same convention run_all.py uses).
sys.path.insert(0, os.path.join(_ROOT, "baselines"))
# unqover/ for the UNQOVER glue; eval/ for the shared generate().
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_ROOT, "eval"))

import common as C  # noqa: E402  (baselines shared driver: fire counter, load_model)
import bbq_eval as B  # noqa: E402  (generate -- the LLaDA sampler, reused verbatim)
import unqover_eval as U  # noqa: E402  (build_prompt, parse_choice)
import denoise_pid_unqover as G  # noqa: E402,F401  (UNQOVER glue, reused verbatim)
from denoise_pid_unqover import (  # noqa: E402
    load_target_items, build_input, make_record, target_letter, LETTERS,  # noqa: F401
)

# UNQOVER item set: the SAME cap2000 file + Black subject filter the existing
# steered uq runs used (uq_decode_PI.json / uq_normal_a4.json -> n_items=256).
DEFAULT_ITEMS = os.path.join(_ROOT, "data", "unqover", "ethnicity.items.cap2000.jsonl")
UQ_TARGET = "Black"                 # the UNQOVER subject string (item filter + pref_gap)
FIT_TARGET = "black"                # the baseline fit-artifact target (lowercase)
DEFAULT_OUT_DIR = os.path.join(_ROOT, "results", "unqover_v2")

# --------------------------------------------------------------------------- #
# UNQOVER-NATIVE fits (jobAI): when configured, every builder loads its fit from
# the UNQOVER-native cache (unqover/cache_native/, built by build_arrows_unqover.py
# from the UNQOVER BUILD split) and CAA/Mean-AcT use the UNQOVER-native arrow set,
# instead of the BBQ black artifacts. NOTHING else changes -- SAME Table-1 (Black)
# operating points, SAME hooks. Set via --native-cache / --native-arrows in main().
# --------------------------------------------------------------------------- #
NATIVE = {"cache_dir": None, "arrows": None, "unit_arrows": None}


def native_on():
    return NATIVE["cache_dir"] is not None


def _native(name):
    """Absolute path of a native fit artifact under the native cache dir."""
    return os.path.join(NATIVE["cache_dir"], name)


def _load_native_arrows(path):
    """Load the UNQOVER-native (32,H) arrow set once; keep raw + per-layer unit.
    Tolerant of an unbuilt file (CPU --selftest before the GPU build): leaves the
    arrows None, and the offline gate SKIPs the CAA/Mean-AcT builders."""
    if not os.path.exists(path):
        print(f"[baselines_unqover] native arrows not built yet ({path}); "
              f"CAA/Mean-AcT builders will be exercised after --build.", flush=True)
        return
    blob = torch.load(path, map_location="cpu")
    r = blob["r"].to(torch.float32)
    NATIVE["arrows"] = r
    NATIVE["unit_arrows"] = C.unit_rows(r)      # per-layer unit-normalized (CAA/Mean-AcT "unit")


# --------------------------------------------------------------------------- #
# Per-method attach_fn builders. Each returns attach_fn(model) -> handles,     #
# built from the black fit artifact at the Table-1 operating point. NO model   #
# is needed to build the injection (CPU tensor math on the artifact), so       #
# --selftest can call these offline to validate the artifacts load.           #
# The paired *_paths() returns the artifact file(s) the builder actually reads #
# (via each module's OWN path helper, so the exists-check == what loads).      #
# --------------------------------------------------------------------------- #
def _mk_caa():
    import caa
    if native_on():   # UNQOVER-native arrow set (per-layer unit), SAME L14 a16
        return caa.make_attach_fn(layer=14, alpha=16.0, arrows=NATIVE["unit_arrows"])
    return caa.make_attach_fn(layer=14, alpha=16.0, target=FIT_TARGET)


def _paths_caa():
    if native_on():
        return [NATIVE_ARROWS_PATH[0]]
    import directions
    return [directions.arrows_path_for(FIT_TARGET)]


def _mk_actadd():
    import actadd
    path = _native("actadd_dir.pt") if native_on() else actadd.dir_path_for(FIT_TARGET)
    return actadd.make_attach_fn(alpha=16.0, layer=14, path=path)


def _paths_actadd():
    import actadd
    return [_native("actadd_dir.pt") if native_on() else actadd.dir_path_for(FIT_TARGET)]


def _mk_meanact():
    import meanact
    if native_on():   # UNQOVER-native (mu2-mu1)=unit arrows, SAME unit s2
        return meanact.build_attach_fn(2.0, direction="unit", arrows=NATIVE["unit_arrows"])
    return meanact.build_attach_fn(2.0, direction="unit", target=FIT_TARGET)


def _paths_meanact():
    if native_on():
        return [NATIVE_ARROWS_PATH[0]]
    import directions
    return [directions.arrows_path_for(FIT_TARGET)]   # unit source = arrows.pt


def _mk_linearact():
    import linearact
    path = _native("linearact_stats.pt") if native_on() else linearact.stats_path_for(FIT_TARGET)
    stats = linearact.load_stats(path=path)
    return lambda m: linearact.attach_fn(m, variant="gaussian", strength=1.0,
                                         stats=stats)


def _paths_linearact():
    import linearact
    return [_native("linearact_stats.pt") if native_on() else linearact.stats_path_for(FIT_TARGET)]


def _mk_aura_inject():
    import aura
    if native_on():
        auroc = torch.load(_native("aura_auroc.pt"), map_location="cpu")["auroc"]
        return aura.attach_fn("inject", gamma=4.0, auroc=auroc)
    return aura.attach_fn("inject", gamma=4.0, target=FIT_TARGET)


def _mk_aura_vanilla():
    import aura
    if native_on():
        auroc = torch.load(_native("aura_auroc.pt"), map_location="cpu")["auroc"]
        return aura.attach_fn("vanilla", auroc=auroc)
    return aura.attach_fn("vanilla", target=FIT_TARGET)


def _paths_aura():
    import aura
    return [_native("aura_auroc.pt") if native_on() else aura.auroc_path_for(FIT_TARGET)]


def _mk_itic():
    import itic
    if native_on():
        probes = itic.load_probes(path=_native("itic_probes.pt"))
        inj = itic.build_injection(K=48, alpha=8.0, probes=probes)
    else:
        inj = itic.build_injection(K=48, alpha=8.0, target=FIT_TARGET)
    return lambda m: itic.attach_fn(m, inj)


def _paths_itic():
    import itic
    return [_native("itic_probes.pt") if native_on() else itic.probes_path_for(FIT_TARGET)]


# Set by main() when --native-arrows is given (a 1-list so _paths_* can cite it).
NATIVE_ARROWS_PATH = [None]


# Registry: uq stem -> (label, builder, artifact-paths fn, Table-1 config extra).
METHODS = {
    "uq_caa": {
        "label": "CAA (single L14, mult~2 == alpha16)",
        "make": _mk_caa, "paths": _paths_caa,
        "config": {"method": "caa", "operating_point": "L14_a16",
                   "granularity": "block_residual_single_layer"},
    },
    "uq_actadd": {
        "label": "ActAdd (single-pair, L14, alpha16)",
        "make": _mk_actadd, "paths": _paths_actadd,
        "config": {"method": "actadd_single_pair", "operating_point": "L14_a16",
                   "granularity": "block_residual"},
    },
    "uq_meanact": {
        "label": "Mean-AcT unit (s=2)",
        "make": _mk_meanact, "paths": _paths_meanact,
        "config": {"method": "mean_act", "operating_point": "unit_s2",
                   "granularity": "block_residual_all32"},
    },
    "uq_linearact": {
        "label": "Linear-AcT gaussian (s=1)",
        "make": _mk_linearact, "paths": _paths_linearact,
        "config": {"method": "linear_act", "operating_point": "gaussian_s1",
                   "granularity": "mlp_hidden_all32"},
    },
    "uq_aura_inject": {
        "label": "AURA inject (gamma=4)",
        "make": _mk_aura_inject, "paths": _paths_aura,
        "config": {"method": "aura", "mode": "inject", "operating_point": "g4",
                   "granularity": "mlp_hidden_all32"},
    },
    "uq_aura_vanilla": {
        "label": "AURA vanilla (suppression control)",
        "make": _mk_aura_vanilla, "paths": _paths_aura,
        "config": {"method": "aura", "mode": "vanilla", "operating_point": "-",
                   "granularity": "mlp_hidden_all32"},
    },
    "uq_itic": {
        "label": "ITI-C (K=48, a=8)",
        "make": _mk_itic, "paths": _paths_itic,
        "config": {"method": "iti_c", "operating_point": "K48_a8",
                   "granularity": "attn_head"},
    },
}
METHOD_ORDER = ["uq_caa", "uq_actadd", "uq_meanact", "uq_linearact",
                "uq_aura_inject", "uq_aura_vanilla", "uq_itic"]


# --------------------------------------------------------------------------- #
# GPU eval (mirrors pid_steer_unqover.run; NOT run by the offline gate).       #
# --------------------------------------------------------------------------- #
def run(stem, items_path, limit, out_path, gen_len, steps, blk, model=None, tok=None):
    entry = METHODS[stem]
    if model is None or tok is None:
        model, tok = C.load_model()

    attach_fn = entry["make"]()                 # build the injection (loads artifact)
    items = load_target_items(items_path, UQ_TARGET, limit)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    print(f"[{stem}] {entry['label']} items={items_path} target={UQ_TARGET!r} "
          f"n={len(items)} dev={torch.cuda.get_device_name(0)} "
          f"CVD={os.environ.get('CUDA_VISIBLE_DEVICES')}", flush=True)

    C.reset_fire_count()
    handles = attach_fn(model)
    if not isinstance(handles, (list, tuple)):
        handles = [handles]

    no_answer = 0
    t0 = time.time()
    try:
        with open(out_path, "w") as out_fh:
            for idx, item in enumerate(items):
                ids = build_input(tok, item, model.device)
                out = B.generate(model, ids, steps=steps, gen_length=gen_len,
                                 block_length=blk, temperature=0.0, cfg_scale=0.0,
                                 remasking="low_confidence")
                gen = tok.batch_decode(out[:, ids.shape[1]:],
                                       skip_special_tokens=True)[0].strip()
                pick = U.parse_choice(gen, item)
                if pick is None:
                    no_answer += 1
                diag = {"target_letter": target_letter(item, UQ_TARGET)}
                out_fh.write(json.dumps(make_record(item, UQ_TARGET, pick, gen, diag)) + "\n")
                if (idx + 1) % 50 == 0:
                    print(f"[{stem}] {idx+1}/{len(items)} no_answer={no_answer} "
                          f"({time.time()-t0:.0f}s)", flush=True)
    finally:
        for h in handles:
            h.remove()

    n_fired = C.get_fire_count()
    assert n_fired > 0, (
        f"[{stem}] hooks never fired (fire_count=0): the baseline attached nothing "
        f"or its hooks were not built with the common.py factories.")

    cfg = dict(entry["config"])
    cfg.update({
        "label": entry["label"], "target_subject": UQ_TARGET,
        "fit_target": FIT_TARGET, "artifacts": entry["paths"](),
        "item_independent": True, "items": items_path, "n_items": len(items),
        "direction_source": ("unqover_native" if native_on() else "bbq_black"),
        "native_cache": NATIVE["cache_dir"], "native_arrows": NATIVE_ARROWS_PATH[0],
        "no_answer": no_answer, "hook_fire_count": n_fired,
        "gen_length": gen_len, "steps": steps, "block_length": blk,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "elapsed_s": time.time() - t0,
    })
    cfg_path = out_path[:-6] + ".json" if out_path.endswith(".jsonl") else out_path + ".json"
    with open(cfg_path, "w") as fh:
        json.dump(cfg, fh, indent=2)
    print(f"[{stem}] DONE -> {out_path}  (no_answer={no_answer}/{len(items)}) "
          f"fire_count={n_fired}", flush=True)
    print("Analyze with unqover_metric.py --results " + out_path, flush=True)


# --------------------------------------------------------------------------- #
# Offline self-test (NO model, NO GPU): each black fit artifact exists + loads,#
# each attach_fn builder returns a callable, record schema is metric-ready.    #
# --------------------------------------------------------------------------- #
def selftest(items_path):
    ok_all = True

    def check(name, cond):
        nonlocal ok_all
        ok_all &= bool(cond)
        print(f"[selftest] {name:56s} : {'PASS' if cond else 'FAIL'}")

    # (0) items file present + Black filter yields the expected 256/64 set.
    have_items = os.path.exists(items_path)
    check(f"items file exists ({os.path.basename(items_path)})", have_items)
    items = load_target_items(items_path, UQ_TARGET, limit=0) if have_items else []
    if have_items:
        n_inst = len({it["instance_id"] for it in items})
        contains = all(it["subj0"] == UQ_TARGET or it["subj1"] == UQ_TARGET for it in items)
        check(f"Black filter: {len(items)} items / {n_inst} instances, all contain 'Black'",
              contains and len(items) > 0)

    # (1) each fit artifact exists on disk (native cache when --native-cache is
    #     set, else the BBQ black artifact via the module's own path fn). Native
    #     artifacts are GPU-built by build_arrows_unqover.py; if not built yet,
    #     SKIP (not FAIL) so the offline CPU gate passes before the build step.
    src = "native" if native_on() else "black"
    for stem in METHOD_ORDER:
        for art in METHODS[stem]["paths"]():
            if native_on() and not os.path.exists(art):
                print(f"[selftest] {stem}: native artifact {os.path.basename(art)} "
                      f": SKIP (built on GPU by build_arrows_unqover.py --build)")
            else:
                check(f"{stem}: {src} artifact {os.path.basename(art)} exists",
                      os.path.exists(art))

    # (2) each attach_fn builder loads its artifact and returns a callable
    #     (CPU tensor math; no model needed). Skipped for native until built.
    for stem in METHOD_ORDER:
        arts = METHODS[stem]["paths"]()
        if native_on() and not all(os.path.exists(a) for a in arts):
            print(f"[selftest] {stem}: builder -> callable : SKIP (native fit not built yet)")
            continue
        try:
            af = METHODS[stem]["make"]()
            check(f"{stem}: builder -> callable attach_fn", callable(af))
        except Exception as exc:  # noqa: BLE001
            check(f"{stem}: builder -> callable ({type(exc).__name__}: {exc})", False)

    # (3) shared fire counter is the SAME object across baselines + this module.
    import caa as _caa
    check("shared fire counter is one common._FIRE object",
          _caa.common.get_fire_count is C.get_fire_count)

    # (4) record schema has every unqover_metric.py-required field.
    if items:
        sample = items[0]
        tgt = target_letter(sample, UQ_TARGET)
        pick = U.parse_choice(tgt, sample)
        rec = make_record(sample, UQ_TARGET, pick, tgt, {"target_letter": tgt})
        needed = ["instance_id", "subj0", "subj1", "qid", "act_cluster", "pred_subject"]
        missing = [k for k in needed if k not in rec]
        check(f"record has metric fields {needed}", not missing)
        check("round-trip pred_subject == 'Black'", rec["pred_subject"] == UQ_TARGET)

    print(f"[selftest] OVERALL: {'PASS' if ok_all else 'FAIL'}")
    return ok_all


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(
        description="The 7 prior-work steering baselines on UNQOVER (Table-1 Black "
                    "operating points, black fit artifacts).")
    ap.add_argument("--selftest", action="store_true", help="offline check; no GPU")
    ap.add_argument("--method", choices=METHOD_ORDER,
                    help="which baseline to run (one of the 7 uq_* stems)")
    ap.add_argument("--items", default=DEFAULT_ITEMS)
    ap.add_argument("--limit", type=int, default=0, help="0 = all Black items")
    ap.add_argument("--out", default=None,
                    help="output jsonl (default results/unqover_v2/<method>.jsonl)")
    ap.add_argument("--gen-length", type=int, default=32)
    ap.add_argument("--steps", type=int, default=64)
    ap.add_argument("--block-length", type=int, default=32)
    ap.add_argument("--native-cache", default=None,
                    help="UNQOVER-native fit cache dir (unqover/cache_native/): load "
                         "actadd/linearact/aura/itic fits from here instead of the BBQ "
                         "black artifacts. SAME Table-1 operating points.")
    ap.add_argument("--native-arrows", default=None,
                    help="UNQOVER-native arrow set (.pt with 'r'=(32,H)) for CAA/Mean-AcT "
                         "instead of steering/arrows.pt. Required with --native-cache.")
    args = ap.parse_args()

    if args.native_cache:
        NATIVE["cache_dir"] = args.native_cache
        if not args.native_arrows:
            ap.error("--native-arrows is required with --native-cache "
                     "(CAA/Mean-AcT need the UNQOVER-native direction).")
        NATIVE_ARROWS_PATH[0] = args.native_arrows
        _load_native_arrows(args.native_arrows)
        print(f"[baselines_unqover] UNQOVER-NATIVE fits: cache={args.native_cache} "
              f"arrows={args.native_arrows}", flush=True)

    if args.selftest:
        sys.exit(0 if selftest(args.items) else 1)
    if not args.method:
        ap.error("--method required (one of %s) or use --selftest" % ",".join(METHOD_ORDER))

    out_path = args.out or os.path.join(DEFAULT_OUT_DIR, f"{args.method}.jsonl")
    run(args.method, args.items, args.limit, out_path,
        args.gen_length, args.steps, args.block_length)


if __name__ == "__main__":
    main()
