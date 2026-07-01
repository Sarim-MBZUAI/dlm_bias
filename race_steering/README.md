# race_steering — race-targeted activation-steering experiment

Separate from the Ghostwriter input-space baseline in `baseline/`. This folder
holds the **race-specific** (Black-targeted) steering experiment on LLaDA-8B.

## Question
The generic `race_color` steering direction behaved as a *non-directional*
abstention-suppressor. Does building the direction on the **Black subset** of
CrowS-Pairs make it a *directional* "pick Black" lever on BBQ?

## Pipeline
1. **Direction** — `bias_steering/build_direction.py --source crows --categories race_color
   --layers 14 --subset-terms "black,african american,african-american"` builds
   `bias_steering/directions/L14/race_black.pt` (261 Black-referent pairs,
   split-half cosine 0.96 — coherent). `.pt` is gitignored.
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
