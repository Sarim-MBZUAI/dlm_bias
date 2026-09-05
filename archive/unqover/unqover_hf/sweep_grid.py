#!/usr/bin/env python
"""The PRE-REGISTERED BUILD-split sweep grid (unqover_hf/PREREG.md section 3).

Single source of truth: the SLURM array driver reads its command lines from here
and `--selftest` re-checks the per-method config counts against the committed
table, so the grid that runs is the grid that was pre-registered.

  python -m unqover_hf.sweep_grid --selftest
  python -m unqover_hf.sweep_grid --list          # one shell arg-string per line
  python -m unqover_hf.sweep_grid --table         # the PREREG section-3 table
"""
import argparse
import sys

N_SWEEP = 120          # instances per config (PREREG section 1.1); 480 generations
SWEEP_SEED = 42

# method -> list of knob dicts.  PREREG section 3, verbatim.
GRID = {
    "base":         [{}],
    "ours_PI":      [{"amax": a} for a in (2, 4, 6, 9, 12)],
    "ours_PID":     [{"amax": a} for a in (2, 4, 6, 9, 12)],
    "normal":       [{"alpha": a} for a in (2, 4, 6, 9, 12, 16)],
    "caa":          [{"alpha": a} for a in (4, 8, 16, 32, 64, 128)],
    "actadd":       [{"alpha": a, "pair_index": p}
                     for a in (8, 32) for p in (0, 1, 2)],
    "aura_vanilla": [{"gamma": g} for g in (0.25, 0.5, 1, 1.5, 2)],
    "aura_inject":  [{"gamma": g} for g in (1, 2, 4, 8, 16)],
    "itic":         [{"topk": k, "alpha": a}
                     for k in (16, 48, 96) for a in (15, 45)],
    "linearact":    [{"variant": v, "strength": s}
                     for v in ("gaussian", "empirical") for s in (0.5, 1, 2)],
    "meanact_raw":  [{"strength": s} for s in (0.25, 0.5, 1, 2, 4)],
    "meanact_unit": [{"strength": s} for s in (2, 4, 6, 8, 10)],
}

# The counts committed in PREREG section 3.  --selftest asserts GRID matches.
PREREG_COUNTS = {
    "base": 1, "ours_PI": 5, "ours_PID": 5, "normal": 6, "caa": 6, "actadd": 6,
    "aura_vanilla": 5, "aura_inject": 5, "itic": 6, "linearact": 6,
    "meanact_raw": 5, "meanact_unit": 5,
}

ORDER = ["base", "ours_PI", "ours_PID", "normal", "caa", "actadd",
         "aura_vanilla", "aura_inject", "itic", "linearact", "meanact_raw",
         "meanact_unit"]

FLAG = {"amax": "--amax", "alpha": "--alpha", "gamma": "--gamma",
        "topk": "--topk", "strength": "--strength", "variant": "--variant",
        "pair_index": "--pair-index"}


def configs():
    """[(method, knobs), ...] in the committed order."""
    return [(m, k) for m in ORDER for k in GRID[m]]


def arg_strings(n_instances=N_SWEEP, seed=SWEEP_SEED):
    """One `run_race.py` argument string per config, in the committed order."""
    out = []
    for m, k in configs():
        parts = ["--method", m, "--n-instances", str(n_instances),
                 "--seed", str(seed)]
        for key in sorted(k):
            parts += [FLAG[key], "%g" % k[key] if isinstance(k[key], float)
                      else str(k[key])]
        out.append(" ".join(parts))
    return out


def selftest():
    ok = True

    def check(name, cond, extra=""):
        nonlocal ok
        ok &= bool(cond)
        print("[selftest] %-56s %s%s" % (name, "PASS" if cond else "FAIL",
                                         (" " + extra) if extra else ""))

    got = {m: len(GRID[m]) for m in GRID}
    check("per-method config counts match PREREG section 3",
          got == PREREG_COUNTS, str({m: (got[m], PREREG_COUNTS[m])
                                     for m in got if got[m] != PREREG_COUNTS[m]}))
    check("61 configs in total", len(configs()) == 61, "(%d)" % len(configs()))
    check("every method in ORDER has a grid and vice versa",
          set(ORDER) == set(GRID) == set(PREREG_COUNTS))
    check("every method except base is swept over >= 5 configs",
          all(len(GRID[m]) >= 5 for m in GRID if m != "base"))
    check("no method is swept over more than 6 configs",
          all(len(GRID[m]) <= 6 for m in GRID))
    check("ours (PI + PID) gets 5 configs each, not more than any baseline",
          len(GRID["ours_PI"]) == len(GRID["ours_PID"]) == 5
          and max(len(GRID[m]) for m in GRID if not m.startswith("ours")) >= 5)
    check("the open-loop normal ladder reaches past ours' amax ladder",
          max(c["alpha"] for c in GRID["normal"])
          > max(c["amax"] for c in GRID["ours_PI"]))
    check("CAA is swept well past alpha=16 (the prior audit's inert cap)",
          max(c["alpha"] for c in GRID["caa"]) >= 64)
    check("ActAdd sweeps the pair index, not just alpha",
          len({c["pair_index"] for c in GRID["actadd"]}) == 3)
    check("MeanAct runs BOTH raw and unit", "meanact_raw" in GRID
          and "meanact_unit" in GRID)
    check("AurA runs BOTH vanilla and inject, both swept over gamma",
          len(GRID["aura_vanilla"]) >= 5 and len(GRID["aura_inject"]) >= 5)
    check("AurA vanilla's grid contains the published gate (gamma == 1)",
          any(c["gamma"] == 1 for c in GRID["aura_vanilla"]))
    check("ITI-C sweeps top-K AND alpha",
          len({c["topk"] for c in GRID["itic"]}) == 3
          and len({c["alpha"] for c in GRID["itic"]}) == 2)
    check("LinearAct sweeps variant AND strength",
          len({c["variant"] for c in GRID["linearact"]}) == 2
          and len({c["strength"] for c in GRID["linearact"]}) == 3)

    a = arg_strings()
    check("one arg string per config", len(a) == len(configs()))
    check("every arg string pins the pre-registered subset size and seed",
          all(("--n-instances %d" % N_SWEEP) in s and ("--seed %d" % SWEEP_SEED) in s
              for s in a))
    check("arg strings are unique", len(set(a)) == len(a))

    sys.path.insert(0, __file__.rsplit("/", 2)[0])
    from unqover_hf.run_race import METHOD_KNOBS
    check("every grid method is a run_race method",
          set(GRID) <= set(METHOD_KNOBS),
          str(sorted(set(GRID) - set(METHOD_KNOBS))))
    check("every knob a method declares is actually swept",
          all(set(METHOD_KNOBS[m]) == set().union(*[set(k) for k in GRID[m]])
              if GRID[m] and GRID[m][0] else METHOD_KNOBS[m] == ()
              for m in GRID),
          str({m: (METHOD_KNOBS[m], sorted(set().union(*[set(k) for k in GRID[m]])))
               for m in GRID
               if (set().union(*[set(k) for k in GRID[m]]) != set(METHOD_KNOBS[m]))}))

    print("[selftest] sweep_grid OVERALL: %s" % ("PASS" if ok else "FAIL"))
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--table", action="store_true")
    ap.add_argument("--n-instances", type=int, default=N_SWEEP)
    args = ap.parse_args()
    if args.selftest:
        sys.exit(0 if selftest() else 1)
    if args.list:
        for s in arg_strings(args.n_instances):
            print(s)
        return
    if args.table:
        print("%-14s %-4s %s" % ("method", "n", "configs"))
        for m in ORDER:
            print("%-14s %-4d %s" % (m, len(GRID[m]),
                                     "; ".join(str(k) for k in GRID[m])))
        print("total configs: %d  (%d instances / %d generations each)"
              % (len(configs()), args.n_instances, 4 * args.n_instances))
        return
    ap.error("pass --selftest, --list or --table")


if __name__ == "__main__":
    main()
