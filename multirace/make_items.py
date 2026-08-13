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

GENDER (E3, --category Gender_identity): woman and man are OPPOSITE POLES on
the SAME Gender_identity rows -- their usable pools are (near-)identical, so
per-target eval/heldout splits drawn independently would overlap almost
completely. Instead the SHARED pool (rows usable for BOTH targets) gets ONE
seed-42 shuffle and is cut into FOUR mutually disjoint 400-row sets:

    [0:400)     eval(woman)    -> data/bbq_items/_sweep400_woman.jsonl
    [400:800)   heldout(woman) -> keys in multirace/items_manifest_gender.json
    [800:1200)  eval(man)      -> data/bbq_items/_sweep400_man.jsonl
    [1200:1600) heldout(man)   -> keys in multirace/items_manifest_gender.json

(2,396 shared usable rows >= 1,600.) The race experiment's exclusion keys do
NOT apply (different BBQ category); manifest schema matches
items_manifest.json plus "cross_target_disjoint": true. STRICT gender tag
sets exclude the trans_/nontrans_ compounds (see targets.py).

CPU-only, offline (BBQ cache already on disk).
Run:            python multirace/make_items.py                  # race, unchanged
                python multirace/make_items.py --category Gender_identity
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

from targets import (TARGET_TAGS, NEW_TARGETS, GENDER_TARGETS,  # noqa: E402
                     TARGET_CATEGORY, target_idx_of, unk_idx_of)


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
MANIFEST_GENDER = os.path.join(HERE, "items_manifest_gender.json")
GENDER_CACHE = os.path.join(ROOT, "data", "bbq_cache", "Gender_identity.jsonl")
SEED = 42
N_EVAL = 400
N_HELDOUT_CAP = 400
MIN_USABLE = 800
N_GENDER_SET = 400          # each of the 4 disjoint gender sets
MIN_GENDER_POOL = 4 * N_GENDER_SET


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


def load_gender_cache():
    """data/bbq_cache/Gender_identity.jsonl (category stamped, like load_full_race)."""
    rows = []
    with open(GENDER_CACHE) as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                r["category"] = "Gender_identity"
                rows.append(r)
    return rows


def split_gender_pool(shared_rows, seed=SEED, n=N_GENDER_SET):
    """ONE seed-`seed` shuffle of the SHARED woman/man pool -> 4 mutually
    disjoint sets: eval(woman), heldout(woman), eval(man), heldout(man)."""
    order = list(shared_rows)
    random.Random(seed).shuffle(order)
    return order[:n], order[n:2 * n], order[2 * n:3 * n], order[3 * n:4 * n]


def main_gender():
    full = load_gender_cache()
    pools = {t: select_target_rows(full, t, set()) for t in GENDER_TARGETS}
    keysets = {t: {row_key(r) for r in pools[t]} for t in GENDER_TARGETS}
    shared_keys = keysets["woman"] & keysets["man"]
    # cache-order shared pool (deterministic input to the seeded shuffle)
    shared = [r for r in pools["woman"] if row_key(r) in shared_keys]
    print(f"[items-gender] full={len(full)} usable_woman={len(pools['woman'])} "
          f"usable_man={len(pools['man'])} shared={len(shared)}", flush=True)
    assert len(shared) >= MIN_GENDER_POOL, \
        f"shared gender pool {len(shared)} < {MIN_GENDER_POOL}"

    ev_w, held_w, ev_m, held_m = split_gender_pool(shared)
    sets = {("woman", "eval"): ev_w, ("woman", "heldout"): held_w,
            ("man", "eval"): ev_m, ("man", "heldout"): held_m}
    keys = {name: {row_key(r) for r in rows} for name, rows in sets.items()}
    names = list(sets)
    for name in names:
        assert len(sets[name]) == N_GENDER_SET and len(keys[name]) == N_GENDER_SET, \
            f"{name}: expected {N_GENDER_SET} unique rows, got {len(keys[name])}"
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            assert keys[a].isdisjoint(keys[b]), f"overlap between {a} and {b}"

    manifest = {
        "seed": SEED,
        "n_eval": N_GENDER_SET,
        "n_heldout_cap": N_GENDER_SET,
        "cache": os.path.join("data", "bbq_cache", "Gender_identity.jsonl"),  # ROOT-relative (portable)
        "category": "Gender_identity",
        "cross_target_disjoint": True,  # 4 sets cut from ONE shuffle of the shared pool
        "n_shared_pool": len(shared),
        "exclusions": {},  # race-experiment keys don't apply (different category)
        "targets": {},
    }
    for target, ev, held in (("woman", ev_w, held_w), ("man", ev_m, held_m)):
        out = os.path.join(ITEMS_DIR, f"_sweep400_{target}.jsonl")
        with open(out, "w") as f:
            for r in ev:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        manifest["targets"][target] = {
            "tags": sorted(TARGET_TAGS[target]),
            "n_usable_after_exclusion": len(shared),
            "n_eval": len(ev),
            "n_heldout": len(held),
            "eval_file": os.path.relpath(out, ROOT),  # ROOT-relative (portable)
            "heldout_keys": sorted([list(k) for k in {row_key(r) for r in held}]),
        }
        print(f"[items-gender] {target:6s} eval={len(ev)} heldout={len(held)} -> {out}",
              flush=True)

    with open(MANIFEST_GENDER, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[items-gender] manifest -> {MANIFEST_GENDER}", flush=True)


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
        "cache": os.path.join("data", "bbq_cache", "Race_ethnicity.jsonl"),  # ROOT-relative (portable)
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
            "eval_file": os.path.relpath(out, ROOT),  # ROOT-relative (portable)
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

    # --- E3 gender: strict selection + 4-way disjoint split --------------- #
    gfull = [
        _mk_row(10, "q1", "ambig",    ["F", "M", "unknown"]),         # keep both
        _mk_row(11, "q2", "ambig",    ["man", "Woman", "unknown"]),   # keep both
        _mk_row(12, "q3", "ambig",    ["boy", "girl", "unknown"]),    # keep both
        _mk_row(13, "q4", "ambig",    ["trans_f", "nontrans_f", "unknown"]),  # strict: drop
        _mk_row(14, "q5", "disambig", ["f", "m", "unknown"]),         # drop: disambig
        _mk_row(15, "q6", "ambig",    ["f", "m", "black"]),           # drop: no unknown
        _mk_row(16, "q7", "ambig",    ["f", "girl", "unknown"]),      # drop: 2 woman
    ]
    kw = {row_key(r) for r in select_target_rows(gfull, "woman", set())}
    km = {row_key(r) for r in select_target_rows(gfull, "man", set())}
    check("gender strict selection: woman == man == 3 shared rows",
          kw == km == {(10, "q1"), (11, "q2"), (12, "q3")})
    check("trans_f/nontrans_f row NOT selected (strict)", (13, "q4") not in kw)
    check("row 16 unusable for both (two woman options, zero man)",
          (16, "q7") not in kw and (16, "q7") not in km)

    gpool = [_mk_row(1000 + i, "q", "ambig", ["f", "m", "unknown"]) for i in range(2396)]
    ev_w, held_w, ev_m, held_m = split_gender_pool(gpool)
    gsets = {"eval_w": ev_w, "held_w": held_w, "eval_m": ev_m, "held_m": held_m}
    gkeys = {k: {row_key(r) for r in v} for k, v in gsets.items()}
    check("gender split: 4 x 400", all(len(v) == 400 for v in gsets.values()))
    gnames = list(gkeys)
    check("gender split: pairwise disjoint (all 6 pairs)",
          all(gkeys[a].isdisjoint(gkeys[b])
              for i, a in enumerate(gnames) for b in gnames[i + 1:]))
    ev_w2, held_w2, ev_m2, held_m2 = split_gender_pool(gpool)
    check("gender split deterministic under seed 42",
          [row_key(r) for r in ev_w2] == [row_key(r) for r in ev_w]
          and [row_key(r) for r in held_m2] == [row_key(r) for r in held_m])

    print(f"[selftest-items] OVERALL: {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--category", default="Race_ethnicity",
                    choices=["Race_ethnicity", "Gender_identity"],
                    help="Race_ethnicity (default, byte-identical to the "
                         "original behavior) or Gender_identity (E3)")
    args = ap.parse_args()
    if args.selftest:
        sys.exit(0 if _selftest() else 1)
    main_gender() if args.category == "Gender_identity" else main()
