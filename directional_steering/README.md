# directional_steering — ITEM-ANCHORED (CAA-style) "prefer Black" direction

Red-team bias-injection research on LLaDA-8B-Instruct (masked-diffusion LM),
block **L14**. Separate from `baseline/` (Ghostwriter input-space) and
`race_steering/` (group mean-diff). Nothing here touches those folders.

## The "can steering AIM?" test

`race_steering/` built a Black-targeted direction from a **group mean-difference**
over CrowS/StereoSet Black-vs-anti sentence pairs. On BBQ it was **non-directional**:
it collapsed abstention onto *both* named options roughly equally instead of aiming
at the Black option (peak directional gap only **+0.054 @ alpha=8**, and negative
once competence broke). That left an open question: *can activation steering aim at
a specific answer option at all, or is the target group simply not linearly
encoded at L14?*

This experiment builds a **decision-anchored** contrast on the **answer token
itself** (CAA-style) rather than on group-descriptive sentences — the most
direct possible "prefer option X" signal — and measures whether *that* is coherent.

## Contrast (locked): Black option vs the OTHER named non-Black person

```
                     BBQ Race_ethnicity, AMBIGUOUS, Black-referent
                     (exactly ONE option is Black-tagged)
                                     │
        build_prompt(row) ── apply_chat_template ──> chat_prompt
                                     │
        ┌────────────────────────────┴────────────────────────────┐
   chat_prompt + black_letter                        chat_prompt + other_letter
   (assistant answers the Black option)              (assistant answers the
                                                      OTHER named, non-Unknown,
                                                      non-Black option)
        │  forward pass (NO sampling)                        │  forward pass
        ▼                                                    ▼
   block-L14 hidden @ answer-letter token           block-L14 hidden @ answer-letter token
        h_black                                             h_other
        └──────────────── pair_diff = h_black − h_other ────┘

   direction = mean_over_items(pair_diff)      (float32, (4096,))
```

Black tag set (case-insensitive), identical to `race_steering/black_analysis.py`:
`{black, african american, f-black, m-black, african}`.

## Held-out / disjoint rationale (NO contamination)

The direction must be evaluated on the seed-42 eval sample, so it may not be
*built* from it. Selection:

1. read the **FULL** `eval/.bbq_cache/Race_ethnicity.jsonl` (6880 items);
2. reproduce the eval keys via `bbq_eval.load_bbq("nyu-mll/BBQ (jsonl)", 42, 1000, None)`
   and collect the `(example_id, question_index)` of its Race_ethnicity items
   (**125** keys);
3. **HELD-OUT** = ambiguous, Black-referent (exactly one Black option), and **NOT**
   in those eval keys.

Held-out available: **1963** items → **capped to 400** (`CAP=400`, reported by the
builder). Disjoint from the eval sample, so the next stage can score on the
seed-42 sample with no train/test leakage.

## Coherence (n=400 anchored pairs)

| metric              | anchored (this) | race_black group mean-diff |
|---------------------|-----------------|----------------------------|
| split-half cosine   | **0.274**       | 0.908                      |
| norm_ratio          | **0.069**       | 0.18                       |
| raw_norm            | **1.966**       | 3.07                       |
| mean_diff_norm      | 28.52           | —                          |
| avg activation norm | 88.66           | 92.87 (embedding)          |

Letter positions are balanced (black_letter A/B/C ≈ 127/138/135), so the raw
"which letter token sits last" component largely averages out across items — which
is *why* the mean direction's norm (1.97) is tiny next to the per-pair diff norm
(~28.5). What survives that cancellation is only weakly consistent: **split-half
cosine 0.27**, far below race_black's **0.91**.

**Read: this anchored direction is NOT coherent enough to expect clean steering.**
The per-pair L14 answer-token diff is dominated by the concrete A-vs-B-vs-C token
identity (item-specific), and the residual "prefer the Black option" semantic
component is small and low-agreement. It is worth running the BBQ sweep as the
falsification test (that is the next stage), but on coherence grounds alone the
anchored contrast at L14 does **not** look like a cleaner AIMing lever than the
group direction — if anything it is *less* coherent. This is consistent with the
prior finding that the Black referent is not strongly linearly encoded at L14.

## Diffusion-LM answer-token capture

LLaDA is **masked-diffusion, not autoregressive**, so "the next-token position"
is not the natural capture site. We do **not** sample here. We build the FULLY
MATERIALIZED sequence `chat_prompt + letter` (no `<mask>` tokens present) and run
a single forward pass; the block-L14 output at the letter position is LLaDA's
**bidirectional contextual representation of that concrete answer letter given the
whole prompt** — the "committed to answer X" activation a CAA contrast needs. The
letter tokenizes to a **single trailing token** (`A/B/C` → ids `32/33/34`), so the
answer token is unambiguously at position **−1** (the last real token).

## Files

- `build_anchored.py` — reproducible builder (held-out selection → forward passes →
  direction + coherence → `.pt` + audit jsonl). Run:
  `CUDA_VISIBLE_DEVICES=0 python directional_steering/build_anchored.py`.
- `race_black_anchored.pt` — the direction + metadata (`layer=14`,
  `method="anchored_caa"`, `contrast="black_vs_other_named"`, `n_pairs=400`,
  coherence metrics). Force-added past the `*.pt` gitignore rule.
- `data/anchored_items.jsonl` — one row per held-out item
  (`example_id, question_index, prompt, black_letter, other_letter, black_group_tag`).
  Force-added past the `*.jsonl` ignore.

The BBQ steering eval is **NOT** run here — that is the next stage.
