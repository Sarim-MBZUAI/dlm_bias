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
2. **Sweep** — `eval/bbq_eval.py --layer 14 --alpha {4,8,12,16,24,32}
   --direction-path .../race_black.pt` → `results/bbq_L14_race_black_a*.json`.
   All six race_black runs are regenerated against the **792-pair** vector.
   The generic `race_color` comparator only exists at α∈{8,16,32}.
3. **Analysis** — `python race_steering/black_analysis.py` computes Black-specific
   metrics on Black-referent ambiguous BBQ items (black_pick vs non-black_pick vs
   abstention), compares to the generic `race_color` runs, and writes
   `figs/black_steering.png`.

## Result (honest) — 792-pair vector, α∈{4,8,12,16,24,32}
**Still not meaningfully directional, and now competence breaks EARLIER.**
Rates on Black-referent AMBIGUOUS items (n=37; clean a=0: black 0.162 / non-black
0.081 / abstention 0.757 / acc_disambig 0.970):

| α | black_pick | non-black_pick | abstention | acc_disambig | **d_gap** (Δblack − Δnon-black) |
|---|-----------|----------------|-----------|--------------|-------------------------------|
| 4  | 0.216 | 0.108 | 0.676 | 0.974 | **+0.027** |
| 8  | 0.324 | 0.189 | 0.486 | 0.949 | **+0.054** |
| 12 | 0.270 | 0.351 | 0.378 | 0.380 | **−0.162** |
| 16 | 0.270 | 0.432 | 0.297 | 0.373 | **−0.243** |
| 24 | 0.297 | 0.324 | 0.378 | 0.293 | **−0.108** |
| 32 | 0.378 | 0.297 | 0.324 | 0.321 |  +0.000 |

**Directionality verdict.** The peak directional gap is only **+0.054 at α=8** — the
same weak magnitude as the old 261-pair vector (≈+0.08 at a8, +0.05 at a16). More/
broader data did **not** make the vector more directional. Worse: at every α where
competence survives (α≤8) the gap stays ≤+0.05, and once α≥12 the gap goes **negative**
(non-Black pick actually outruns Black pick, −0.16 at α12, −0.24 at α16). The picks
that appear come almost entirely from abstention collapse (0.757→0.49 by α8), not a
directional "pick Black" push — the same non-directional behaviour as `race_color`.

**Competence cliff.** acc_disambig falls off a cliff between **α=8 (0.949) and α=12
(0.380)** — earlier and sharper than the whole-benchmark runs suggested (those showed
~0.90 at a16 → ~0.29 at a24). So the only competent operating region is α≤8, where the
directional effect is negligible.

**Bottom line.** The 792-pair vector is still dominated by non-directional abstention
collapse, not a clean directional bias lever; the extra data mainly moved the
competence cliff to lower α. **Caveat:** n=37 Black-referent ambiguous items is small,
so these gaps are noisy.

Next steps to get a conclusive answer: score on far more Black-referent items
(n=37 is the bottleneck); or build item-specific evidence/directions that name each
item's target group.
