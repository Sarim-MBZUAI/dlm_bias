#!/usr/bin/env python
"""dream/baselines/run_all.py -- CLI orchestrator for the 6 Dream steering baselines.

Port of baselines/run_all.py.  Ties the six Dream method modules behind one CLI so
a single command fits (where needed) and evaluates any subset against the
Dream-v0-Instruct-7B masked-diffusion BBQ bias-INJECTION harness (common_dream).

The six methods (all steer toward "Black"; positive strength/alpha = stronger):
    caa        Contrastive Activation Addition -- diff-in-means residual add (no fit).
    meanact    Mean-AcT -- per-neuron mean-shift (needs calib fit for direction=fitted).
    actadd     ActAdd -- single-pair residual direction add (needs its --fit artifact).
    linearact  Linear-AcT -- per-neuron 1-D OT affine on MLP-hidden (needs calib fit).
    aura       AURA -- per-neuron AUROC gate on MLP-hidden (needs calib fit).
    itic       ITI-c -- per-HEAD contrastive attn intervention (needs calib fit).

Contract (each method exposes run(), _selftest(); fit() where needed; optional
module.NEEDS_FIT).  Signatures differ: caa/actadd/itic take `alpha`,
meanact/linearact `strength`, aura `gamma` (+ mode="inject").  Dispatch is
adapter-driven and inspect.signature-filtered, so a method never receives a knob
it didn't ask for.  Under --run/--fit the model loads ONCE (common_dream) and is
shared across every method whose run()/fit() accepts model=/tok=.

Eval output ALWAYS lands under <worktree>/results/dream/<method>/ (never the main
tree), regardless of each method's own default out_dir.

CLI:
    python dream/baselines/run_all.py --selftest                 # offline contract check
    CUDA_VISIBLE_DEVICES=4 python dream/baselines/run_all.py --run --method all
    CUDA_VISIBLE_DEVICES=4 python dream/baselines/run_all.py --fit --run --method meanact,linearact,aura,itic
"""
import argparse
import importlib
import inspect
import os
import sys
import traceback

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))   # common_dream

# Eval output in the worktree tree of THIS file: <worktree>/results/dream/<method>.
_TREE = os.path.dirname(os.path.dirname(_HERE))   # .../<tree>/dream/baselines -> <tree>
RESULTS_ROOT = os.path.join(_TREE, "results", "dream")

METHOD_ORDER = ["caa", "meanact", "actadd", "linearact", "aura", "itic"]
METHODS = {
    "caa": {"module": "caa", "needs_fit": False,
            "strength_param": "alpha", "run_defaults": {},
            "ref": "Contrastive Activation Addition (diff-in-means residual add)"},
    "meanact": {"module": "meanact", "needs_fit": True,
                "strength_param": "strength", "run_defaults": {},
                "ref": "Mean-AcT: per-neuron mean-shift (transport.py:259)"},
    "actadd": {"module": "actadd", "needs_fit": False,
               "strength_param": "alpha", "run_defaults": {},
               "ref": "ActAdd: single-pair residual direction add"},
    "linearact": {"module": "linearact", "needs_fit": True,
                  "strength_param": "strength", "run_defaults": {},
                  "ref": "Linear-AcT: per-neuron empirical 1-D OT affine (archs.py)"},
    "aura": {"module": "aura", "needs_fit": True,
             "strength_param": "gamma", "run_defaults": {"mode": "inject"},
             "ref": "AURA: per-neuron AUROC gate, headline mode=inject (aura_hook.py:65-67)"},
    "itic": {"module": "itic", "needs_fit": True,
             "strength_param": "alpha", "run_defaults": {},
             "ref": "ITI-c: per-head contrastive attn intervention (where=attn_head)"},
}


def results_dir(method):
    return os.path.join(RESULTS_ROOT, method)


def parse_methods(spec):
    if spec == "all":
        return list(METHOD_ORDER)
    out = []
    for name in spec.split(","):
        name = name.strip()
        if not name:
            continue
        if name not in METHODS:
            raise SystemExit(
                f"unknown method '{name}'; choose from all,{','.join(METHOD_ORDER)}")
        if name not in out:
            out.append(name)
    return out


def _module_needs_fit(method, mod=None):
    if mod is not None and hasattr(mod, "NEEDS_FIT"):
        return bool(mod.NEEDS_FIT)
    return METHODS[method]["needs_fit"]


def _declared(fn):
    params = inspect.signature(fn).parameters
    named = {n for n, p in params.items()
             if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)}
    var_kw = any(p.kind == p.VAR_KEYWORD for p in params.values())
    return named, var_kw


def _unsatisfied_required(fn, candidates):
    params = inspect.signature(fn).parameters
    return [n for n, p in params.items()
            if p.default is inspect._empty
            and p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
            and n not in candidates]


def _call_filtered(fn, candidates):
    named, _ = _declared(fn)
    return fn(**{k: v for k, v in candidates.items() if k in named})


def _run_candidates(method, out_dir, strength_val, limit, model, tok):
    entry = METHODS[method]
    cand = {"out_dir": out_dir, "limit": limit, "model": model, "tok": tok}
    cand.update(entry.get("run_defaults", {}))
    sp = entry.get("strength_param")
    if sp and strength_val is not None:
        cand[sp] = strength_val
    return cand


def dispatch(methods, do_fit, do_run, strength_val, limit, model, tok):
    summary = []
    for method in methods:
        modname = METHODS[method]["module"]
        mod = importlib.import_module(modname)
        out_dir = results_dir(method)
        os.makedirs(out_dir, exist_ok=True)

        fit_ran = False
        if do_fit and _module_needs_fit(method, mod):
            if hasattr(mod, "fit") and callable(mod.fit):
                print(f"[run_all] FIT  {method} ({modname}.fit) ...", flush=True)
                _call_filtered(mod.fit, {"model": model, "tok": tok})
                fit_ran = True
            else:
                print(f"[run_all] FIT  {method}: SKIP (needs a fit but exposes no fit())",
                      flush=True)

        result = None
        if do_run:
            if not (hasattr(mod, "run") and callable(mod.run)):
                raise SystemExit(f"[run_all] {method}: module '{modname}' exposes no run()")
            cand = _run_candidates(method, out_dir, strength_val, limit, model, tok)
            missing = _unsatisfied_required(mod.run, cand)
            if missing:
                print(f"[run_all] RUN  {method}: SKIP -- run() needs {missing}", flush=True)
            else:
                sp = METHODS[method].get("strength_param")
                print(f"[run_all] RUN  {method} ({modname}.run) "
                      f"{sp}={cand.get(sp) if sp else None} limit={limit} -> {out_dir}",
                      flush=True)
                result = _call_filtered(mod.run, cand)

        summary.append({"method": method, "module": modname, "out_dir": out_dir,
                        "fit_ran": fit_ran, "result": result})
    return summary


def _selftest():
    ok = True
    present = 0
    rows = []

    def check(name, cond):
        nonlocal ok
        ok &= bool(cond)
        print(f"[selftest-run_all] {name:52s} : {'PASS' if cond else 'FAIL'}")

    check("registry order matches keys",
          set(METHOD_ORDER) == set(METHODS) and len(METHOD_ORDER) == len(METHODS))
    check("parse_methods('all') == METHOD_ORDER", parse_methods("all") == METHOD_ORDER)
    check("parse_methods dedups + orders",
          parse_methods("aura,caa,aura") == ["aura", "caa"])
    try:
        parse_methods("nope"); bad = False
    except SystemExit:
        bad = True
    check("parse_methods rejects unknown", bad)

    for method in METHOD_ORDER:
        modname = METHODS[method]["module"]
        try:
            mod = importlib.import_module(modname)
        except Exception as e:  # noqa: BLE001
            rows.append((method, modname, "PENDING", "-", "-", results_dir(method)))
            print(f"[selftest-run_all] import {modname:12s} : PENDING ({type(e).__name__}: {e})")
            continue
        present += 1
        has_run = hasattr(mod, "run") and callable(mod.run)
        has_fit = hasattr(mod, "fit") and callable(mod.fit)
        has_selftest = any(callable(getattr(mod, a, None)) for a in ("_selftest", "selftest"))
        needs_fit = _module_needs_fit(method, mod)
        check(f"{method}: exposes run()", has_run)
        check(f"{method}: exposes _selftest()", has_selftest)
        if needs_fit:
            check(f"{method}: exposes fit() (needs_fit)", has_fit)
        if has_run:
            cand = _run_candidates(method, results_dir(method), strength_val=1.0,
                                   limit=0, model=object(), tok=object())
            missing = _unsatisfied_required(mod.run, cand)
            check(f"{method}: adapter satisfies run() required args", not missing)
            if missing:
                print(f"[selftest-run_all]   -> unsatisfied: {missing}")
        rows.append((method, modname, "OK", "Y" if has_fit else "n",
                     "Y" if needs_fit else "n", results_dir(method)))

    _print_write_map(rows)
    print(f"[selftest-run_all] OVERALL: {'PASS' if ok else 'FAIL'} "
          f"({present}/{len(METHOD_ORDER)} method modules present)")
    return ok


def _print_write_map(rows):
    print("\n  method     module      status   fit  needs_fit  writes")
    print("  " + "-" * 82)
    for method, modname, status, hasfit, needsfit, out_dir in rows:
        rel = os.path.relpath(out_dir, _TREE)
        print(f"  {method:<10} {modname:<11} {status:<8} {hasfit:<4} {needsfit:<10} {rel}/")
    print()


def _print_dispatch_summary(summary):
    print("\n[run_all] SUMMARY")
    print("  method     module      fit_ran  writes")
    print("  " + "-" * 64)
    for s in summary:
        rel = os.path.relpath(s["out_dir"], _TREE)
        print(f"  {s['method']:<10} {s['module']:<11} "
              f"{'yes' if s['fit_ran'] else 'no ':<8} {rel}/")
    print()


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--method", default="all",
                    help="'all' or comma-separated: " + ",".join(METHOD_ORDER))
    ap.add_argument("--fit", action="store_true", help="build fitted artifacts first; NEEDS GPU.")
    ap.add_argument("--run", action="store_true", help="evaluate steered condition(s); NEEDS GPU.")
    ap.add_argument("--strength", type=float, default=None,
                    help="generic strength; mapped per-method (strength/alpha/gamma).")
    ap.add_argument("--alpha", type=float, default=None, help="alias for --strength.")
    ap.add_argument("--limit", type=int, default=0, help="cap BBQ items (0 = all sweep400).")
    ap.add_argument("--selftest", action="store_true",
                    help="offline: import the 6, assert contract, print write map (no GPU).")
    args = ap.parse_args()

    if args.selftest:
        sys.exit(0 if _selftest() else 1)

    if not (args.fit or args.run):
        ap.error("nothing to do: pass --selftest (offline), and/or --fit / --run (GPU).")

    methods = parse_methods(args.method)
    strength_val = args.alpha if args.alpha is not None else args.strength
    print(f"[run_all] methods={methods} fit={args.fit} run={args.run} "
          f"strength={strength_val} limit={args.limit}", flush=True)

    model = tok = None
    if args.fit or args.run:
        import common_dream  # noqa: E402
        print("[run_all] loading Dream-v0-Instruct-7B once (shared)...", flush=True)
        model, tok = common_dream.load_model_tok()

    try:
        summary = dispatch(methods, do_fit=args.fit, do_run=args.run,
                           strength_val=strength_val, limit=args.limit,
                           model=model, tok=tok)
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        sys.exit(1)

    _print_dispatch_summary(summary)
    print(f"[run_all] DONE ({len(summary)} method(s)).", flush=True)


if __name__ == "__main__":
    main()
