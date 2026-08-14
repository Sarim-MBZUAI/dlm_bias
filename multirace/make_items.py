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

SES / AGE (E9, --category SES / --category Age): the SAME pair-axis situation
as gender -- every ambiguous row of the category carries BOTH poles plus an
unknown option, so the two poles' pools are identical and the split is the
SAME gender-style 4-way mutually disjoint cut (ONE seed-42 shuffle):
    SES  (3,432 shared rows >= 1,600):  lowses / highses
        -> data/bbq_items/_sweep400_{lowses,highses}.jsonl
           + multirace/items_manifest_ses.json
    Age  (1,840 shared rows >= 1,600, tight but fits):  old / young
        -> data/bbq_items/_sweep400_{old,young}.jsonl
           + multirace/items_manifest_age.json
No prior experiment touches these caches, so exclusions == {} (like gender).
The 'young' target tag-matches BBQ's 'nonOld' tag (see targets.py). The
gender path is UNCHANGED (byte-identical items + manifest): the shared
pair-axis code below is main_gender generalized by category, same logic.

INTERSECTIONAL TARGET (E6, --target-set intersectional): fblack (Black women,
the single compound tag f-black) lives on the SAME Race_ethnicity cache as
the Black experiment, so the SAME exclusions apply (seed-42 eval keys UNION
_sweep400.jsonl keys). Only 712 usable rows survive them (952 raw - 240
excluded), so the split is a DOCUMENTED DEVIATION from the 400/400 standard:
eval 400 + heldout 312 (= ALL remaining rows), recorded as "deviation" in
multirace/items_manifest_fblack.json. The manifest also records the
composition of the heldout NEGATIVE (contrast) options: the policy ADMITS
same-race other-gender negatives (m-black), but empirically BBQ
Race_ethnicity pairs people of the SAME gender, so ALL 312 realized
negatives are other-race WOMEN and 0 are m-black -- the built direction is
therefore a GENDER-CONDITIONED RACE direction (f-black vs other-race women),
NOT a full intersectional contrast (see multirace/build_arrows.py).
Gender_identity is a DIFFERENT category/cache:
(example_id, question_index) keys are only meaningful within a category, so
no key collision with the gender manifests is possible and no gender
exclusion is needed (it would be dead code).

CPU-only, offline (BBQ cache already on disk).
Run:            python multirace/make_items.py                  # race, unchanged
                python multirace/make_items.py --category Gender_identity
                python multirace/make_items.py --target-set intersectional  # E6
                python multirace/make_items.py --category SES   # E9
                python multirace/make_items.py --category Age   # E9
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
                     INTERSECTIONAL_TARGETS, SES_TARGETS, AGE_TARGETS,
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
MANIFEST_FBLACK = os.path.join(HERE, "items_manifest_fblack.json")
MANIFEST_SES = os.path.join(HERE, "items_manifest_ses.json")
MANIFEST_AGE = os.path.join(HERE, "items_manifest_age.json")
GENDER_CACHE = os.path.join(ROOT, "data", "bbq_cache", "Gender_identity.jsonl")
SEED = 42
N_EVAL = 400
N_HELDOUT_CAP = 400
MIN_USABLE = 800
N_GENDER_SET = 400          # each of the 4 disjoint pair-axis sets
MIN_GENDER_POOL = 4 * N_GENDER_SET

# Pair-axis registry (E3 gender, E9 SES/Age): categories whose two poles live
# on the SAME rows, requiring the 4-way mutually disjoint split. The log tag
# keeps gender's original "[items-gender]" prefix byte-identical.
PAIR_AXES = {
    "Gender_identity": {"targets": GENDER_TARGETS, "manifest": MANIFEST_GENDER,
                        "log": "items-gender"},
    "SES":             {"targets": SES_TARGETS, "manifest": MANIFEST_SES,
                        "log": "items-ses"},
    "Age":             {"targets": AGE_TARGETS, "manifest": MANIFEST_AGE,
                        "log": "items-age"},
}
# E6 fblack: DOCUMENTED DEVIATION -- 712 usable after exclusions, so heldout
# is ALL 312 remaining rows, not the standard 400 (asserted exactly below).
N_FBLACK_HELDOUT = 312


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


def load_category_cache(category):
    """data/bbq_cache/<category>.jsonl (category stamped, like load_full_race)."""
    path = os.path.join(ROOT, "data", "bbq_cache", f"{category}.jsonl")
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                r["category"] = category
                rows.append(r)
    return rows


def load_gender_cache():
    """Back-compat wrapper (build_arrows.py imports it)."""
    return load_category_cache("Gender_identity")


def split_gender_pool(shared_rows, seed=SEED, n=N_GENDER_SET):
    """ONE seed-`seed` shuffle of the SHARED pair-axis pool -> 4 mutually
    disjoint sets: eval(pole0), heldout(pole0), eval(pole1), heldout(pole1)
    (gender: woman then man; SES: lowses then highses; Age: old then young)."""
    order = list(shared_rows)
    random.Random(seed).shuffle(order)
    return order[:n], order[n:2 * n], order[2 * n:3 * n], order[3 * n:4 * n]


def main_pair_axis(category):
    """Shared pair-axis item builder (E3 gender = the original main_gender,
    generalized by category; E9 SES/Age). Same logic, same manifest schema --
    the gender output stays byte-identical."""
    ax = PAIR_AXES[category]
    t0, t1 = ax["targets"]
    log, manifest_path = ax["log"], ax["manifest"]
    full = load_category_cache(category)
    pools = {t: select_target_rows(full, t, set()) for t in (t0, t1)}
    keysets = {t: {row_key(r) for r in pools[t]} for t in (t0, t1)}
    shared_keys = keysets[t0] & keysets[t1]
    # cache-order shared pool (deterministic input to the seeded shuffle)
    shared = [r for r in pools[t0] if row_key(r) in shared_keys]
    print(f"[{log}] full={len(full)} usable_{t0}={len(pools[t0])} "
          f"usable_{t1}={len(pools[t1])} shared={len(shared)}", flush=True)
    assert len(shared) >= MIN_GENDER_POOL, \
        f"shared {category} pool {len(shared)} < {MIN_GENDER_POOL}"

    ev_0, held_0, ev_1, held_1 = split_gender_pool(shared)
    sets = {(t0, "eval"): ev_0, (t0, "heldout"): held_0,
            (t1, "eval"): ev_1, (t1, "heldout"): held_1}
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
        "cache": os.path.join("data", "bbq_cache", f"{category}.jsonl"),  # ROOT-relative (portable)
        "category": category,
        "cross_target_disjoint": True,  # 4 sets cut from ONE shuffle of the shared pool
        "n_shared_pool": len(shared),
        "exclusions": {},  # no other experiment shares this category's cache
        "targets": {},
    }
    for target, ev, held in ((t0, ev_0, held_0), (t1, ev_1, held_1)):
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
        print(f"[{log}] {target:6s} eval={len(ev)} heldout={len(held)} -> {out}",
              flush=True)

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[{log}] manifest -> {manifest_path}", flush=True)


def main_gender():
    """Back-compat entry point: the E3 gender build, now via main_pair_axis."""
    main_pair_axis("Gender_identity")


def negative_tag_of(row, target):
    """Lowercase tag of the CONTRAST (negative) option build_arrows.py will
    pick for this row: the FIRST option that is neither the target nor the
    unknown option (identical rule to build_arrows.heldout_from_manifest)."""
    tidx = target_idx_of(row, target)
    unk = unk_idx_of(row)
    oidx = next(k for k in range(3) if k != tidx and k != unk)
    return str(row["answer_info"][f"ans{oidx}"][-1]).strip().lower()


def negative_composition(rows, target, contrast_tag="m-black"):
    """Tally the negative-option tags over `rows`. Returns
    {"by_tag": {...}, contrast_tag (as key): n, "other": n} -- recorded in the
    manifest so the paper can state how often the fblack direction contrasts
    against the same-race other gender (m-black) vs another race
    (empirically: never vs always -- the direction is gender-conditioned
    race, f-black vs other-race women)."""
    by_tag = {}
    for r in rows:
        tag = negative_tag_of(r, target)
        by_tag[tag] = by_tag.get(tag, 0) + 1
    n_contrast = by_tag.get(contrast_tag, 0)
    return {"by_tag": dict(sorted(by_tag.items())),
            contrast_tag: n_contrast,
            "other": len(rows) - n_contrast}


def main_fblack():
    """E6: fblack items. Same category/cache as the Black experiment ->
    the SAME exclusions apply (unlike gender: Gender_identity is a different
    cache, keys are per-category, no collision possible, no exclusion added)."""
    lb = load_llada_builder()
    full = lb.load_full_race()
    seed_keys = lb.eval_race_keys()
    sw_keys = lb.sweep400_keys()
    exclude = seed_keys | sw_keys
    target = "fblack"
    raw = select_target_rows(full, target, set())
    usable = select_target_rows(full, target, exclude)
    print(f"[items-fblack] full={len(full)} raw_usable={len(raw)} "
          f"seed42_keys={len(seed_keys)} sweep400_keys={len(sw_keys)} "
          f"excluded_union={len(exclude)} usable={len(usable)}", flush=True)

    ev, held = split_items(usable)
    ev_keys = {row_key(r) for r in ev}
    held_keys = {row_key(r) for r in held}
    # DOCUMENTED DEVIATION: exactly 400 eval + 312 heldout (= ALL remaining).
    assert len(ev) == len(ev_keys) == N_EVAL, f"eval {len(ev)} != {N_EVAL}"
    assert len(held) == len(held_keys) == N_FBLACK_HELDOUT, \
        f"heldout {len(held)} != {N_FBLACK_HELDOUT} -- cache/exclusions drifted"
    assert len(ev) + len(held) == len(usable), "heldout must be ALL remaining rows"
    assert ev_keys.isdisjoint(held_keys), "eval/heldout overlap"
    assert ev_keys.isdisjoint(exclude) and held_keys.isdisjoint(exclude), \
        "contamination with Black-experiment keys"

    neg = negative_composition(held, target)
    print(f"[items-fblack] heldout negatives: m-black={neg['m-black']} "
          f"other={neg['other']} by_tag={neg['by_tag']}", flush=True)
    if neg["m-black"] == 0:
        direction_note = (
            "All %d realized heldout negatives are other-race women "
            "(0 m-black): the built direction is a GENDER-CONDITIONED RACE "
            "direction (f-black vs other-race women), NOT a full "
            "intersectional contrast. BBQ Race_ethnicity pairs people of the "
            "same gender, so same-race other-gender negatives never occur "
            "even though the policy admits them." % neg["other"])
    else:
        direction_note = ("Heldout negatives mix m-black (%d) and other-race "
                          "(%d) options." % (neg["m-black"], neg["other"]))

    out = os.path.join(ITEMS_DIR, f"_sweep400_{target}.jsonl")
    with open(out, "w") as f:
        for r in ev:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    manifest = {
        "seed": SEED,
        "n_eval": N_EVAL,
        "n_heldout": N_FBLACK_HELDOUT,
        "deviation": ("heldout is 312 = ALL usable rows remaining after the "
                      "400-item eval draw, NOT the standard 400: only 712 "
                      "f-black rows survive the Black-experiment exclusions "
                      "(952 raw - 240 excluded)."),
        "cache": os.path.join("data", "bbq_cache", "Race_ethnicity.jsonl"),  # ROOT-relative (portable)
        "category": "Race_ethnicity",
        "exclusions": {"black_seed42_race_keys": len(seed_keys),
                       "black_sweep400_keys": len(sw_keys),
                       "union": len(exclude),
                       "raw_usable_before_exclusion": len(raw),
                       "raw_rows_excluded": len(raw) - len(usable)},
        "targets": {
            target: {
                "tags": sorted(TARGET_TAGS[target]),
                "n_usable_after_exclusion": len(usable),
                "n_eval": len(ev),
                "n_heldout": len(held),
                "eval_file": os.path.relpath(out, ROOT),  # ROOT-relative (portable)
                # Negative = FIRST non-fblack non-unknown option; the policy
                # ADMITS m-black (Black men), but BBQ pairs same-gender
                # people, so realized m-black is 0 -- see direction_note and
                # build_arrows.py module doc.
                "heldout_negative_composition": neg,
                "direction_note": direction_note,
                "heldout_keys": sorted([list(k) for k in held_keys]),
            }
        },
    }
    with open(MANIFEST_FBLACK, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[items-fblack] eval={len(ev)} -> {out}", flush=True)
    print(f"[items-fblack] manifest -> {MANIFEST_FBLACK}", flush=True)


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

    # --- E9 SES/Age: pair-axis selection + 4-way disjoint split ----------- #
    check("PAIR_AXES registers gender + SES + Age",
          set(PAIR_AXES) == {"Gender_identity", "SES", "Age"}
          and PAIR_AXES["SES"]["targets"] == ("lowses", "highses")
          and PAIR_AXES["Age"]["targets"] == ("old", "young"))
    sfull = [
        _mk_row(30, "q1", "ambig",    ["lowSES", "highSES", "unknown"]),  # keep both
        _mk_row(31, "q2", "ambig",    ["highSES", "unknown", "lowSES"]),  # keep both
        _mk_row(32, "q3", "disambig", ["lowSES", "highSES", "unknown"]),  # drop: disambig
        _mk_row(33, "q4", "ambig",    ["lowSES", "highSES", "black"]),    # drop: no unknown
        _mk_row(34, "q5", "ambig",    ["lowSES", "lowSES", "unknown"]),   # drop: 2 lowses
    ]
    kl = {row_key(r) for r in select_target_rows(sfull, "lowses", set())}
    kh = {row_key(r) for r in select_target_rows(sfull, "highses", set())}
    check("SES selection: lowses == highses == 2 shared rows",
          kl == kh == {(30, "q1"), (31, "q2")})
    afull = [
        _mk_row(40, "q1", "ambig",    ["old", "nonOld", "unknown"]),   # keep both
        _mk_row(41, "q2", "ambig",    ["nonOld", "unknown", "old"]),   # keep both
        _mk_row(42, "q3", "ambig",    ["nonOld", "nonOld", "unknown"]),  # drop: 2 young, 0 old
        _mk_row(43, "q4", "disambig", ["old", "nonOld", "unknown"]),   # drop: disambig
    ]
    ko = {row_key(r) for r in select_target_rows(afull, "old", set())}
    ky = {row_key(r) for r in select_target_rows(afull, "young", set())}
    check("Age selection: old == young == 2 shared rows (whole-tag nonold)",
          ko == ky == {(40, "q1"), (41, "q2")})
    check("Age: double-nonOld row unusable for both",
          (42, "q3") not in ko and (42, "q3") not in ky)
    # Age-sized pool (1,840 = the real shared count, tight but >= 1,600):
    # 4-way split leaves 240 rows unused, all four sets disjoint.
    apool = [_mk_row(3000 + i, "q", "ambig", ["old", "nonOld", "unknown"])
             for i in range(1840)]
    ev_o, held_o, ev_y, held_y = split_gender_pool(apool)
    asets = {"eval_o": ev_o, "held_o": held_o, "eval_y": ev_y, "held_y": held_y}
    akeys = {k: {row_key(r) for r in v} for k, v in asets.items()}
    check("Age-sized pool 1840: 4 x 400 (240 unused)",
          all(len(v) == 400 for v in asets.values()))
    anames = list(akeys)
    check("Age-sized split: pairwise disjoint (all 6 pairs)",
          all(akeys[a].isdisjoint(akeys[b])
              for i, a in enumerate(anames) for b in anames[i + 1:]))
    ev_o2, _, _, held_y2 = split_gender_pool(apool)
    check("pair-axis split deterministic under seed 42 (Age-sized)",
          [row_key(r) for r in ev_o2] == [row_key(r) for r in ev_o]
          and [row_key(r) for r in held_y2] == [row_key(r) for r in held_y])

    # --- E6 fblack: selection, negative composition, 400/312 split -------- #
    ffull = [
        _mk_row(20, "q1", "ambig",    ["F-Black", "M-White", "unknown"]),  # keep
        _mk_row(21, "q2", "ambig",    ["m-black", "f-black", "unknown"]),  # keep (neg = m-black!)
        _mk_row(22, "q3", "ambig",    ["Black", "white", "unknown"]),      # drop: no f-black
        _mk_row(23, "q4", "ambig",    ["f-black", "f-black", "unknown"]),  # drop: 2 f-black
        _mk_row(24, "q5", "disambig", ["f-black", "white", "unknown"]),    # drop: disambig
        _mk_row(25, "q6", "ambig",    ["f-black", "white", "unknown"]),    # drop: excluded
    ]
    fk = {row_key(r) for r in select_target_rows(ffull, "fblack", {(25, "q6")})}
    check("fblack selection: exactly the 2 valid rows",
          fk == {(20, "q1"), (21, "q2")})
    check("fblack: plain-black / m-black-only rows not selected",
          (22, "q3") not in fk)
    # negative composition: row 20 contrasts m-white, row 21 contrasts m-black.
    fneg = negative_composition([ffull[0], ffull[1]], "fblack")
    check("negative composition: m-black=1 other=1, tags tallied",
          fneg["m-black"] == 1 and fneg["other"] == 1
          and fneg["by_tag"] == {"m-black": 1, "m-white": 1})
    # 712-row pool (the real fblack count) -> 400 eval + 312 heldout = ALL rest.
    fpool = [_mk_row(2000 + i, "q", "ambig", ["f-black", "white", "unknown"])
             for i in range(712)]
    fev, fheld = split_items(fpool)
    check("fblack pool 712 -> 400 eval + 312 heldout (all remaining), disjoint",
          len(fev) == 400 and len(fheld) == N_FBLACK_HELDOUT == 312
          and len(fev) + len(fheld) == 712
          and {row_key(r) for r in fev}.isdisjoint({row_key(r) for r in fheld}))

    print(f"[selftest-items] OVERALL: {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--category", default="Race_ethnicity",
                    choices=["Race_ethnicity", "Gender_identity", "SES", "Age"],
                    help="Race_ethnicity (default, byte-identical to the "
                         "original behavior), Gender_identity (E3), or the E9 "
                         "pair axes SES / Age")
    ap.add_argument("--target-set", default="race",
                    choices=["race", "intersectional"],
                    help="Race_ethnicity only: 'race' = the four round-2 "
                         "targets (default, byte-identical) or "
                         "'intersectional' = E6 fblack (400 eval / 312 heldout)")
    args = ap.parse_args()
    if args.selftest:
        sys.exit(0 if _selftest() else 1)
    if args.category in PAIR_AXES:
        if args.target_set != "race":
            ap.error("--target-set intersectional requires --category Race_ethnicity")
        main_pair_axis(args.category)
    elif args.target_set == "intersectional":
        main_fblack()
    else:
        main()
