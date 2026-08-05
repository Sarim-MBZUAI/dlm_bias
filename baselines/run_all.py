#!/usr/bin/env python
"""baselines/run_all.py -- CLI orchestrator for the 6 AcT steering baselines.

Ties the six method modules together behind one argparse CLI so a single command
fits (where needed) and evaluates any subset of the baselines against our
LLaDA-8B-Instruct masked-diffusion BBQ bias-INJECTION harness.

The six methods (all steer toward the "Black" answer; positive strength/alpha =
stronger injection; "Black" is the OT DESTINATION when a map is fitted):

    caa        Contrastive Activation Addition -- diff-in-means residual add
               (directions.load_arrows; no calibration fit needed).
    meanact    Mean-AcT -- per-neuron GAUSSIAN 1-D OT (mean+std) transport
               (act/hooks/transport.py:261; needs a calib fit).
    actadd     ActAdd -- contrastive-prompt residual direction add
               (needs its single-pair fit; run() requires cache/actadd_dir.pt).
    linearact  Linear-AcT -- per-neuron EMPIRICAL 1-D OT, closed-form LS affine
               (act/optimal_transport/archs.py; needs a calib fit).
    aura       AURA -- per-neuron AUROC dampening gate
               (act/hooks/aura_hook.py:65-67; needs a calib fit).
    itic       ITI-c -- per-HEAD contrastive intervention on attn activations
               (needs a calib fit at where=attn_head).

Contract this orchestrator relies on (the method files are authored in parallel;
each exposes exactly this, per the shared spec):

    module.run(...)      -> evaluate one steered condition, writing the
                            pid_steer-format result files under results/<method>/.
    module.fit(...)      -> build fitted artifacts (GPU); present only on the
                            methods that need calibration (meanact/linearact/
                            aura/itic) or a fitted direction (actadd).  caa has
                            no fit.
    module._selftest()   -> offline math check (no GPU).
    module.NEEDS_FIT     -> optional bool; if present it overrides this file's
                            registry guess for whether fit() is required.

Signatures differ between method files: caa/actadd/itic take the injection
strength as `alpha`, meanact/linearact as `strength`, aura as `gamma` (and aura
also needs mode="inject"), etc.
Dispatch is therefore adapter-driven -- each registry entry names the run()
parameter the generic --strength/--alpha maps to (`strength_param`) plus any
fixed run kwargs (`run_defaults`, e.g. aura's mode="inject").  Every candidate
kwarg is then filtered to the names run()/fit() actually declare
(inspect.signature), so a method never receives a knob it didn't ask for.

CRITICAL: each method's own run() defaults out_dir to the MAIN tree
(<repo root>/results/<m>).  This orchestrator ALWAYS
passes out_dir=<worktree>/results/<m> so evaluation output lands in the worktree,
never in the main tree.

Under --run the model is loaded ONCE (common.load_model) and shared across every
method whose run()/fit() accepts model=/tok=, so a multi-method sweep pays the
load cost once.

CLI:
    python run_all.py --selftest                 # offline: import the 6, assert
                                                 #   the contract, print write map.
    python run_all.py --run --method all         # NEEDS GPU (evaluate all 6)
    python run_all.py --fit --run --method meanact,linearact,aura,itic  # NEEDS GPU
    python run_all.py --fit --method all         # NEEDS GPU (build artifacts only)

--run and --fit BOTH need a GPU + the model and are NOT exercised in --selftest.
"""
import argparse
import importlib
import inspect
import os
import sys
import traceback

# Make sibling modules (common.py, directions.py, calib.py, the 6 methods)
# importable as top-level modules regardless of the invoking cwd -- the same
# top-level import style the method files use (e.g. calib.py: `from common import`).
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

RESULTS_ROOT = os.path.join(_HERE, "results")

# --------------------------------------------------------------------------- #
# Method registry.  Deterministic order; `needs_fit` is this file's default    #
# expectation (a method may override it by defining module.NEEDS_FIT).         #
# --------------------------------------------------------------------------- #
#   strength_param : run() kwarg that the generic --strength/--alpha maps to
#                    (None -> this method takes no scalar strength knob).
#   run_defaults   : fixed run() kwargs the orchestrator always supplies.
# Verified against the present method files' real run() signatures; guessed
# (and flagged) for the still-PENDING meanact/linearact (both AcT scale knobs,
# expected `alpha`) -- a landed module may override needs_fit via NEEDS_FIT.
METHOD_ORDER = ["caa", "meanact", "actadd", "linearact", "aura", "itic"]
METHODS = {
    "caa": {
        "module": "caa", "needs_fit": False,
        "strength_param": "alpha", "run_defaults": {},
        "ref": "Contrastive Activation Addition (diff-in-means residual add)",
    },
    "meanact": {
        # needs_fit stays True (fit builds the mean-diff artifact), but the
        # default run (direction="unit") does not consume that fitted artifact.
        # strength default matches meanact's own CLI default (run() requires one).
        "module": "meanact", "needs_fit": True,
        "strength_param": "strength", "run_defaults": {"strength": 2.0},
        "ref": "Mean-AcT: per-neuron Gaussian 1-D OT (transport.py:261)",
    },
    "actadd": {
        # run() REQUIRES cache/actadd_dir.pt (build_injection raises if absent).
        "module": "actadd", "needs_fit": True,
        "strength_param": "alpha", "run_defaults": {},
        "ref": "ActAdd: contrastive-prompt residual direction add",
    },
    "linearact": {
        "module": "linearact", "needs_fit": True,
        "strength_param": "strength", "run_defaults": {},
        "ref": "Linear-AcT: per-neuron empirical 1-D OT affine (archs.py)",
    },
    "aura": {
        "module": "aura", "needs_fit": True,
        "strength_param": "gamma", "run_defaults": {"mode": "inject"},
        "ref": "AURA: per-neuron AUROC gate, headline mode=inject (aura_hook.py:65-67)",
    },
    "itic": {
        "module": "itic", "needs_fit": True,
        "strength_param": "alpha", "run_defaults": {},
        "ref": "ITI-c: per-head contrastive attn intervention (where=attn_head)",
    },
}


def results_dir(method):
    """results/<method>/ -- where this method's cond_*.json files land."""
    return os.path.join(RESULTS_ROOT, method)


def parse_methods(spec):
    """'all' or comma-separated names -> ordered, de-duplicated list of methods."""
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
    """module.NEEDS_FIT (if defined) overrides the registry default."""
    if mod is not None and hasattr(mod, "NEEDS_FIT"):
        return bool(mod.NEEDS_FIT)
    return METHODS[method]["needs_fit"]


def _declared(fn):
    """Named (non-var) parameters fn declares, and whether it takes **kwargs."""
    params = inspect.signature(fn).parameters
    named = {n for n, p in params.items()
             if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)}
    var_kw = any(p.kind == p.VAR_KEYWORD for p in params.values())
    return named, var_kw


def _unsatisfied_required(fn, candidates):
    """Required params (no default, not var) that `candidates` cannot supply."""
    params = inspect.signature(fn).parameters
    return [n for n, p in params.items()
            if p.default is inspect._empty
            and p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
            and n not in candidates]


def _call_filtered(fn, candidates):
    """Call fn with only the (curated) candidate kwargs it declares.

    `candidates` is already method-specific and intentional (no stray flags), so
    we pass the intersection with fn's named params; if fn also takes **kwargs we
    still only forward curated keys -- never anything fn didn't ask for."""
    named, _var_kw = _declared(fn)
    kwargs = {k: v for k, v in candidates.items() if k in named}
    return fn(**kwargs)


def _run_candidates(method, out_dir, strength_val, limit, model, tok):
    """Build the curated run() kwargs for one method (adapter-driven)."""
    entry = METHODS[method]
    cand = {"out_dir": out_dir, "limit": limit, "model": model, "tok": tok}
    cand.update(entry.get("run_defaults", {}))
    sp = entry.get("strength_param")
    if sp and strength_val is not None:
        cand[sp] = strength_val            # map generic strength -> alpha/gamma/...
    return cand


# --------------------------------------------------------------------------- #
# Dispatch (NEEDS GPU).  fit first if requested + applicable, then run.        #
# --------------------------------------------------------------------------- #
def _ensure_block_calib(model, tok):
    """meanact.fit() takes no args and bare-torch.loads cache/calib_block.pt;
    build that calib here (reusing the shared model) if it is missing."""
    import calib  # noqa: E402  (deferred: imports torch)
    path = os.path.join(calib.CACHE_DIR, "calib_block.pt")
    if not os.path.exists(path):
        print(f"[run_all] building missing block calib -> {path}", flush=True)
        calib.collect_activations("block", model=model, tok=tok, save=True)


def dispatch(methods, do_fit, do_run, strength_val, limit, model, tok):
    summary = []
    for method in methods:
        modname = METHODS[method]["module"]
        out_dir = results_dir(method)
        status, fit_ran, result = "OK", False, None
        try:
            mod = importlib.import_module(modname)
            os.makedirs(out_dir, exist_ok=True)

            if do_fit and _module_needs_fit(method, mod):
                if hasattr(mod, "fit") and callable(mod.fit):
                    if method == "meanact":
                        _ensure_block_calib(model, tok)
                    print(f"[run_all] FIT  {method} ({modname}.fit) ...", flush=True)
                    _call_filtered(mod.fit, {"model": model, "tok": tok})
                    fit_ran = True
                else:
                    print(f"[run_all] FIT  {method}: SKIP (needs a fit but exposes "
                          f"no fit()); run() may load a pre-fit artifact.", flush=True)

            if do_run:
                if not (hasattr(mod, "run") and callable(mod.run)):
                    raise RuntimeError(f"module '{modname}' exposes no run()")
                cand = _run_candidates(method, out_dir, strength_val, limit, model, tok)
                missing = _unsatisfied_required(mod.run, cand)
                if missing:
                    status = "SKIP"
                    print(f"[run_all] RUN  {method}: SKIP -- run() needs argument(s) "
                          f"{missing} the orchestrator cannot supply generically; "
                          f"add them to METHODS['{method}']['run_defaults'].", flush=True)
                else:
                    sp = METHODS[method].get("strength_param")
                    shown = cand.get(sp) if sp else None
                    print(f"[run_all] RUN  {method} ({modname}.run) "
                          f"{sp}={shown} limit={limit} -> {out_dir}", flush=True)
                    result = _call_filtered(mod.run, cand)
        except Exception as exc:  # noqa: BLE001 -- one method must not abort the rest
            status = "FAIL"
            print(f"[run_all] FAIL {method}: {exc}", flush=True)
            traceback.print_exc()

        summary.append({"method": method, "module": modname, "out_dir": out_dir,
                        "fit_ran": fit_ran, "result": result, "status": status})
    return summary


# --------------------------------------------------------------------------- #
# Offline self-test: import the 6, assert the contract, print the write map.   #
# Missing siblings are reported PENDING (written in parallel) -- not a failure; #
# a PRESENT module that violates the contract IS a failure.                    #
# --------------------------------------------------------------------------- #
def _selftest():
    ok = True
    present = 0
    rows = []

    def check(name, cond):
        nonlocal ok
        ok &= bool(cond)
        print(f"[selftest-run_all] {name:52s} : {'PASS' if cond else 'FAIL'}")

    # The orchestrator's own registry must be internally consistent.
    check("registry order matches keys",
          set(METHOD_ORDER) == set(METHODS) and len(METHOD_ORDER) == len(METHODS))
    check("parse_methods('all') == METHOD_ORDER", parse_methods("all") == METHOD_ORDER)
    check("parse_methods dedups + orders",
          parse_methods("aura,caa,aura") == ["aura", "caa"])
    try:
        parse_methods("nope")
        bad = False
    except SystemExit:
        bad = True
    check("parse_methods rejects unknown", bad)
    # Facts fixed by review: actadd's run() REQUIRES its fit artifact, and
    # meanact's run() requires a strength (supplied via run_defaults).
    check("actadd registered needs_fit=True", METHODS["actadd"]["needs_fit"] is True)
    check("meanact run_defaults supplies strength=2.0",
          METHODS["meanact"]["run_defaults"].get("strength") == 2.0)

    for method in METHOD_ORDER:
        modname = METHODS[method]["module"]
        try:
            mod = importlib.import_module(modname)
        except Exception as e:  # noqa: BLE001
            rows.append((method, modname, "PENDING", "-", "-", results_dir(method)))
            print(f"[selftest-run_all] import {modname:12s} : PENDING "
                  f"(not yet written -- {type(e).__name__})")
            continue
        present += 1
        has_run = hasattr(mod, "run") and callable(mod.run)
        has_fit = hasattr(mod, "fit") and callable(mod.fit)
        has_selftest = any(
            callable(getattr(mod, a, None)) for a in ("_selftest", "selftest"))
        needs_fit = _module_needs_fit(method, mod)
        # Contract: every present method exposes run() and an offline selftest.
        check(f"{method}: exposes run()", has_run)
        check(f"{method}: exposes _selftest()", has_selftest)
        # fit() required only where applicable.
        if needs_fit:
            check(f"{method}: exposes fit() (needs_fit)", has_fit)
        # Adapter completeness: the curated run() kwargs must satisfy every
        # REQUIRED run() arg (offline; catches e.g. a missing mode=).
        if has_run:
            cand = _run_candidates(method, results_dir(method),
                                   strength_val=1.0, limit=0,
                                   model=object(), tok=object())
            missing = _unsatisfied_required(mod.run, cand)
            check(f"{method}: adapter satisfies run() required args",
                  not missing)
            if missing:
                print(f"[selftest-run_all]   -> unsatisfied: {missing} "
                      f"(add to METHODS['{method}']['run_defaults'])")
            # Same check with --strength omitted (regression: meanact must not
            # SKIP just because no strength was passed).
            cand_none = _run_candidates(method, results_dir(method),
                                        strength_val=None, limit=0,
                                        model=object(), tok=object())
            check(f"{method}: run() satisfied with --strength omitted",
                  not _unsatisfied_required(mod.run, cand_none))
        rows.append((method, modname, "OK", "Y" if has_fit else "n",
                     "Y" if needs_fit else "n", results_dir(method)))

    _print_write_map(rows)
    if present == 0:
        print("[selftest-run_all] NOTE: all 6 method modules are still PENDING "
              "(authored in parallel); contract assertions run once they land.")
    print(f"[selftest-run_all] OVERALL: {'PASS' if ok else 'FAIL'} "
          f"({present}/{len(METHOD_ORDER)} method modules present)")
    return ok


def _print_write_map(rows):
    print("\n  method     module      status   fit  needs_fit  writes")
    print("  " + "-" * 82)
    for method, modname, status, hasfit, needsfit, out_dir in rows:
        rel = os.path.relpath(out_dir, _HERE)
        print(f"  {method:<10} {modname:<11} {status:<8} {hasfit:<4} "
              f"{needsfit:<10} {rel}/")
    print()


def _print_dispatch_summary(summary):
    print("\n[run_all] SUMMARY")
    print("  method     module      status  fit_ran  writes")
    print("  " + "-" * 64)
    for s in summary:
        rel = os.path.relpath(s["out_dir"], _HERE)
        print(f"  {s['method']:<10} {s['module']:<11} {s['status']:<7} "
              f"{'yes' if s['fit_ran'] else 'no ':<8} {rel}/")
    print()


# --------------------------------------------------------------------------- #
# CLI.                                                                         #
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--method", default="all",
                    help="'all' or comma-separated: " + ",".join(METHOD_ORDER))
    ap.add_argument("--fit", action="store_true",
                    help="build fitted artifacts first (only for methods that "
                         "need a fit); NEEDS GPU.")
    ap.add_argument("--run", action="store_true",
                    help="evaluate the steered condition(s); NEEDS GPU.")
    ap.add_argument("--strength", type=float, default=None,
                    help="generic injection strength; mapped per-method to its own "
                         "knob (strength for meanact/linearact, alpha for "
                         "caa/actadd/itic, gamma for aura).  Omit to use each "
                         "method's own default; NOTE meanact's run() REQUIRES a "
                         "strength -- run_defaults supplies 2.0 when omitted, and "
                         "an explicit --strength overrides it.")
    ap.add_argument("--alpha", type=float, default=None,
                    help="alias for --strength (takes precedence if both given).")
    ap.add_argument("--limit", type=int, default=0,
                    help="cap BBQ items (0 = all of _sweep400.jsonl); passthrough.")
    ap.add_argument("--selftest", action="store_true",
                    help="offline: import the 6 methods, assert the contract, "
                         "print the write map (no GPU).")
    args = ap.parse_args()

    if args.selftest:
        sys.exit(0 if _selftest() else 1)

    if not (args.fit or args.run):
        ap.error("nothing to do: pass --selftest (offline), and/or --fit / --run "
                 "(both NEED a GPU + the model).")

    methods = parse_methods(args.method)
    strength_val = args.alpha if args.alpha is not None else args.strength
    print(f"[run_all] methods={methods} fit={args.fit} run={args.run} "
          f"strength={strength_val} limit={args.limit}", flush=True)

    # Share one model load across every method under --run (each run()/fit() that
    # accepts model=/tok= gets it; those that don't will lazily load their own).
    model = tok = None
    if args.fit or args.run:
        import common  # noqa: E402  (deferred: importing loads torch/harness only)
        print("[run_all] loading LLaDA-8B-Instruct once (shared)...", flush=True)
        model, tok = common.load_model()

    summary = dispatch(methods, do_fit=args.fit, do_run=args.run,
                       strength_val=strength_val, limit=args.limit,
                       model=model, tok=tok)

    _print_dispatch_summary(summary)
    failed = [s["method"] for s in summary if s["status"] == "FAIL"]
    print(f"[run_all] DONE ({len(summary)} method(s), {len(failed)} failed).",
          flush=True)
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
