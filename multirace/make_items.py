#!/usr/bin/env python
"""multirace/make_items.py -- per-target item sets for multi-race steering.

For each target in {white, asian, latino, arab} (black is the existing
reference, untouched):

  1. SELECT from data/bbq_cache/Race_ethnicity.jsonl: AMBIGUOUS rows with
     exactly ONE option carrying a TARGET_TAGS[target] tag and an "unknown"
     option present (same structure as the Black setup).
  2. EXCLUDE the Black experiment's contaminating keys -- (a) the seed-42
     n=1000 eval sample's Race_ethnicity keys, (b) _sweep400.jsonl keys --
     reusing steering/build_arrows.py's eval_race_keys / sweep400_keys
     verbatim (loaded by absolute path). Rows can qualify for two targets at
     once (e.g. Black-vs-White), hence the exclusion.
  3. SPLIT reproducibly (seed 42 shuffle): 400 EVAL items ->
     data/bbq_items/_sweep400_<target>.jsonl, then up to 400 HELDOUT
     direction items (disjoint from the eval 400) recorded BY KEY in
     multirace/items_manifest.json -- build_arrows.py reads keys from the
     manifest, never recomputes.

Asserts per target: >=800 usable rows (warns with the exact count when the
heldout side is under 400 -- expected for white, ~811 total) and
eval INTERSECT heldout == EMPTY.

CPU-only, offline (BBQ cache already on disk).
Run:            python multirace/make_items.py
Offline check:  python multirace/make_items.py --selftest
"""
import argparse
import importlib.util
import json
import os
import random
import sys

ROOT = os.environ.get("DLM_BIAS_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from targets import TARGET_TAGS, NEW_TARGETS, target_idx_of, unk_idx_of  # noqa: E402


# Reuse the LLaDA arrow builder's exclusion helpers VERBATIM. Loaded by absolute
# path (same pattern as dream/build_arrows.py) because multirace/build_arrows.py
# shares its basename.
def load_llada_builder():
    path = os.path.join(ROOT, "steering", "build_arrows.py")
    spec = importlib.util.spec_from_file_location("llada_build_arrows", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ITEMS_DIR = os.path.join(ROOT, "data", "bbq_items")
MANIFEST = os.path.join(HERE, "items_manifest.json")
SEED = 42
N_EVAL = 400
N_HELDOUT_CAP = 400
MIN_USABLE = 800


def row_key(r):
    return (int(r.get("example_id", -1)), str(r.get("question_index", "")))


def select_target_rows(full_rows, target, exclude_keys):
    """Ambiguous, exactly-one-target-option, unknown present, key not excluded."""
    picked = []
    for r in full_rows:
        if r.get("context_condition") != "ambig":
            continue
        tags = TARGET_TAGS[target]
        hits = [k for k in range(3)
                if (p := r["answer_info"].get(f"ans{k}"))
                and str(p[-1]).strip().lower() in tags]
        if len(hits) != 1:
            continue
        if unk_idx_of(r) is None:
            continue
        if row_key(r) in exclude_keys:
            continue
        picked.append(r)
    return picked


def split_items(rows, seed=SEED):
    """Seed-`seed` shuffle -> (eval_rows[:400], heldout_rows[:400])."""
    order = list(rows)
    random.Random(seed).shuffle(order)
    ev = order[:N_EVAL]
    held = order[N_EVAL:N_EVAL + N_HELDOUT_CAP]
    return ev, held


def main():
    lb = load_llada_builder()
    full = lb.load_full_race()
    seed_keys = lb.eval_race_keys()
    sw_keys = lb.sweep400_keys()
    exclude = seed_keys | sw_keys
    print(f"[items] full={len(full)} seed42_keys={len(seed_keys)} "
          f"sweep400_keys={len(sw_keys)} excluded={len(exclude)}", flush=True)

    manifest = {
        "seed": SEED,
        "n_eval": N_EVAL,
        "n_heldout_cap": N_HELDOUT_CAP,
        "cache": os.path.join(ROOT, "data", "bbq_cache", "Race_ethnicity.jsonl"),
        "exclusions": {"black_seed42_race_keys": len(seed_keys),
                       "black_sweep400_keys": len(sw_keys)},
        "targets": {},
    }

    for target in NEW_TARGETS:
        usable = select_target_rows(full, target, exclude)
        n = len(usable)
        assert n >= MIN_USABLE, f"[{target}] only {n} usable rows (< {MIN_USABLE})"
        ev, held = split_items(usable)
        if len(held) < N_HELDOUT_CAP:
            print(f"[items] WARN {target}: only {n} usable rows -> "
                  f"{len(ev)} eval + {len(held)} heldout (< {N_HELDOUT_CAP})", flush=True)
        ev_keys = {row_key(r) for r in ev}
        held_keys = {row_key(r) for r in held}
        assert ev_keys.isdisjoint(held_keys), f"[{target}] eval/heldout overlap"
        assert ev_keys.isdisjoint(exclude) and held_keys.isdisjoint(exclude), \
            f"[{target}] contamination with Black-experiment keys"

        out = os.path.join(ITEMS_DIR, f"_sweep400_{target}.jsonl")
        with open(out, "w") as f:
            for r in ev:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        manifest["targets"][target] = {
            "tags": sorted(TARGET_TAGS[target]),
            "n_usable_after_exclusion": n,
            "n_eval": len(ev),
            "n_heldout": len(held),
            "eval_file": out,
            "heldout_keys": sorted([list(k) for k in held_keys]),
        }
        print(f"[items] {target:6s} usable={n:4d} eval={len(ev)} "
              f"heldout={len(held)} -> {out}", flush=True)

    with open(MANIFEST, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[items] manifest -> {MANIFEST}", flush=True)


# --------------------------------------------------------------------------- #
# Offline self-test (no data files).
# --------------------------------------------------------------------------- #
def _mk_row(eid, qi, cc, tags):
    return {
        "example_id": eid, "question_index": qi, "context_condition": cc,
        "context": "c", "question": "q",
        "ans0": "a0", "ans1": "a1", "ans2": "a2",
        "answer_info": {f"ans{k}": [f"w{k}", tags[k]] for k in range(3)},
    }


def _selftest():
    ok = True

    def check(name, cond):
        nonlocal ok
        ok &= bool(cond)
        print(f"[selftest-items] {name:56s} : {'PASS' if cond else 'FAIL'}")

    full = [
        _mk_row(1, "q1", "ambig",    ["Asian", "Black", "unknown"]),    # keep (asian)
        _mk_row(2, "q2", "ambig",    ["latino", "F-Asian", "unknown"]), # keep (asian)
        _mk_row(3, "q3", "disambig", ["asian", "white", "unknown"]),    # drop: disambig
        _mk_row(4, "q4", "ambig",    ["asian", "m-asian", "unknown"]),  # drop: 2 asian
        _mk_row(5, "q5", "ambig",    ["black", "white", "unknown"]),    # drop: 0 asian
        _mk_row(6, "q6", "ambig",    ["asian", "black", "white"]),      # drop: no unknown
        _mk_row(7, "q7", "ambig",    ["asian", "black", "unknown"]),    # drop: excluded
    ]
    exclude = {(7, "q7")}
    picked = select_target_rows(full, "asian", exclude)
    keys = {row_key(r) for r in picked}
    check("selects exactly the 2 valid asian rows", keys == {(1, "q1"), (2, "q2")})
    check("excluded key dropped", (7, "q7") not in keys)
    # Same pool, other target: row 5 is exactly-one-black AND exactly-one-white.
    check("two-target row usable for white",
          {row_key(r) for r in select_target_rows(full, "white", set())} >= {(5, "q5")})

    # Split: seeded, disjoint, capped, deterministic.
    pool = [_mk_row(100 + i, "q", "ambig", ["arab", "black", "unknown"]) for i in range(950)]
    ev, held = split_items(pool)
    ke, kh = {row_key(r) for r in ev}, {row_key(r) for r in held}
    check("eval size == 400", len(ev) == 400)
    check("heldout capped at 400", len(held) == 400)
    check("eval INTERSECT heldout == EMPTY", ke.isdisjoint(kh))
    ev2, held2 = split_items(pool)
    check("split deterministic under seed 42",
          [row_key(r) for r in ev2] == [row_key(r) for r in ev]
          and [row_key(r) for r in held2] == [row_key(r) for r in held])
    # White-like pool: 811 usable -> 400 eval + 411 remaining, heldout capped at 400.
    ev3, held3 = split_items(pool[:811])
    check("pool 811 -> 400 eval + 400 heldout (411 remaining, capped)",
          len(ev3) == 400 and len(held3) == 400)
    ev4, held4 = split_items(pool[:750])
    check("pool 750 -> 400 eval + 350 heldout, disjoint",
          len(ev4) == 400 and len(held4) == 350
          and {row_key(r) for r in ev4}.isdisjoint({row_key(r) for r in held4}))

    print(f"[selftest-items] OVERALL: {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    sys.exit(0 if _selftest() else 1) if args.selftest else main()
