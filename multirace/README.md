# multirace/ — multi-race steering targets for LLaDA-8B-Instruct

Generalizes the repo's Black-referent steering (`steering/build_arrows.py`,
`steering/denoise_pid.py`) from ONE target to five: per-race item sets and
per-race diff-in-means arrows for **white, asian, latino, arab**. **black**
stays the existing reference (`steering/arrows.pt`, `_sweep400.jsonl`) and is
NOT rebuilt here.

## Targets

Tags are BBQ `answer_info` group tags, matched WHOLE and lowercase (compound
tags like `f-black` are literal tags, not split). Counts from
`data/bbq_cache/Race_ethnicity.jsonl` (6880 rows, 3440 ambiguous), after
requiring ambiguous + exactly-one-target-option + unknown-option and excluding
the Black experiment's keys:

| target | tags | usable | eval | heldout |
|--------|------|-------:|-----:|--------:|
| white  | white, caucasian, european, f-white, m-white | 810 | 400 | 400 |
| asian  | asian, f-asian, m-asian                      | 934 | 400 | 400 |
| latino | latino, hispanic, f-latino, m-latino         | 1217 | 400 | 400 |
| arab   | arab, middle eastern, f-arab, m-arab         | 1040 | 400 | 400 |
| black  | (existing: black, african american, african, f-black, m-black) | — reference, untouched — | | |

`european` is required to get white past 800 usable rows (810 total: 400 eval
+ 410 remaining, heldout capped at 400).

## Layout

- `targets.py` — `TARGET_TAGS` registry, `target_idx_of(row, target)`
  (mirrors `pid_steer.black_idx_of`), `unk_idx_of` re-export.
- `make_items.py` — selects + splits (seed 42): 400 eval items ->
  `data/bbq_items/_sweep400_<target>.jsonl`, up to 400 heldout direction items
  recorded BY KEY in `items_manifest.json`.
- `items_manifest.json` — single source of truth for the heldout keys
  (committed; `build_arrows.py` reads it, never recomputes).
- `build_arrows.py --target T` — answer-text-anchored r(k) over the heldout
  items, all 32 blocks, saves RAW `(32, 4096)` -> `arrows_<target>.pt`
  (+ per-layer norms; gitignored) and `direction_examples_<target>.jsonl`
  (gitignored contrast-pair dump).

## Run

```bash
# CPU, offline
python multirace/targets.py --selftest
python multirace/make_items.py --selftest
python multirace/make_items.py            # writes item files + manifest
python multirace/build_arrows.py --selftest

# one GPU per target
CUDA_VISIBLE_DEVICES=3 python multirace/build_arrows.py --target asian
```

Downstream eval reuses the existing runner:
`steering/pid_steer.py --items data/bbq_items/_sweep400_<target>.jsonl
--arrows multirace/arrows_<target>.pt ...` (note: its counters are labeled
black/nonblack; interpretation is target/non-target).

## Contamination design

Excluded from EVERY target's pool (via `steering/build_arrows.py`'s
`eval_race_keys` / `sweep400_keys`, loaded by absolute path):

1. the Black experiment's seed-42 n=1000 eval sample's Race_ethnicity keys (125);
2. the Black `_sweep400.jsonl` keys (400).

Why: BBQ rows can qualify for two targets at once (e.g. a Black-vs-White row
is in both pools), so without this exclusion a white/latino/arab arrow could
be trained on rows the Black experiment evaluates on. Per target, eval and
heldout are additionally asserted disjoint, and `build_arrows.py --selftest`
re-verifies all three disjointness properties against the real manifest.

Heldout sets of DIFFERENT targets may overlap each other (a Latino-vs-Arab
row can train both arrows) — accepted, since cross-target evals always pair a
target's arrows with that same target's disjoint eval file.

## Honest deviations

- `steering/build_arrows.py` selects heldout items inline at build time;
  here selection lives in `make_items.py` and the manifest, so arrows and
  eval sets are fixed before any GPU run.
- Its pooling/hook code is nested inside `main()` and not importable, so
  `build_arrows.py` mirrors it line-for-line (via `bbq_eval` helpers) instead
  of importing it; only `load_full_race` and the exclusion helpers are
  imported (importlib-by-path, the `dream/build_arrows.py` pattern).
- Manifest `heldout_keys` are stored sorted, not in shuffle order (order is
  irrelevant to the mean over items).
- Scout estimate said ~811 usable white rows; the actual count is 810 (the
  seed-42 exclusion removes one more row than the sweep400-only estimate).
