# race_steering — race-targeted activation-steering experiment

Separate from the Ghostwriter input-space baseline in `baseline/`. This folder
holds the **race-specific** (Black-targeted) steering experiment on LLaDA-8B.

## Question
The generic `race_color` steering direction behaved as a *non-directional*
abstention-suppressor. Does building the direction on the **Black subset** of
CrowS-Pairs make it a *directional* "pick Black" lever on BBQ?

## Steering dataset (REAL benchmark data only)
Built by `race_steering/build_black_data.py` (CrowS subset + StereoSet extract →
`data/black_pairs.csv` + `data/black_pairs.json`). No synthetic/templated pairs.

- **Total: 792 pairs** (exact-dedup applied) — per source:
  - `crows`      : **261** — CrowS-Pairs `race_color`, Black-referent subset
    (terms `black, african american, african-american`).
  - `stereoset`  : **531** — StereoSet dev.json `race` domain, Sub-Saharan-African /
    Black targets only (intrasentence + intersentence, stereotype vs anti-stereotype).
- **StereoSet target subset used** (10 of 36 race targets): `African, Ethiopian,
  Ethiopia, Somalia, Ghanaian, Cameroon, Cape Verde, Eritrean, Eriteria, Sierra Leon`.
  DELIBERATELY excludes North-African/Arab (Morocco, Arab), Middle-Eastern, Asian,
  European, and Latin-American nationalities — not Black/African referents.
- StereoSet `dev.json` is cached under `race_steering/.stereoset_cache/` (gitignored).

## Pipeline
1. **Direction** — `bias_steering/build_direction.py --source json
   --pairs race_steering/data/black_pairs.json --layer 14
   --out race_steering/race_black.pt` builds the L14 block direction from all
   **792 pairs** (split-half cosine **0.9084**, norm_ratio 0.18, raw_norm 3.07,
   avg_embed_norm 92.87 — still coherent; slightly below the old 261-pair 0.96
   because StereoSet's varied phrasings add directional variance). `.pt` is
   force-added to git here despite the `*.pt` ignore rule.

   > ⚠️ **STALE RESULTS:** the committed `results/bbq_L14_race_black_a{8,16,32}.json`
   > and `figs/black_steering.png` were produced from the OLD **261-pair** vector.
   > They are now STALE pending a re-run of the BBQ sweep + analysis against this
   > larger **792-pair** `race_black.pt`.
2. **Sweep** — `eval/bbq_eval.py --layer 14 --alpha {8,16,32}
   --direction-path .../race_black.pt` → `results/bbq_L14_race_black_a*.json`.
3. **Analysis** — `python race_steering/black_analysis.py` computes Black-specific
   metrics on Black-referent ambiguous BBQ items (black_pick vs non-black_pick vs
   abstention), compares to the generic `race_color` runs, and writes
   `figs/black_steering.png`.

## Result (honest)
**Weakly directional at best, and underpowered (n=37 Black-referent ambiguous items).**
At the usable point α=16 (acc_disambig 0.90): black_pick 0.16→0.43 but non-black_pick
also 0.08→0.30 — directional gap only ~+0.05, dominated by abstention collapse
(0.76→0.27). α=32 is degenerate (acc_disambig 0.28, 57% no-answer). Targeting the
direction to the Black subset did **not** convert it into a clean directional bias
vector; it remains largely a non-directional abstention suppressor, like `race_color`.

Next steps to get a conclusive answer: score on far more Black-referent items
(n=37 is the bottleneck); sweep finer at low α (4/8/12); or build item-specific
evidence/directions that name each item's target group.
