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

This experiment builds a **decision-anchored** contrast (CAA-style) on the model's
own answer — rather than on group-descriptive sentences — and measures whether
*that* is coherent. Two variants were built:

- **letter-anchored** (`race_black_anchored.pt`, `method="anchored_caa"`) —
  captures the single last token = the bare `A/B/C` letter. **FAILED**
  (split-half cosine **0.27**): the per-pair L14 diff is dominated by raw
  letter-token identity, not by "prefer the Black person". Kept for the record.
- **answer-text-anchored** (`race_black_anchored_text.pt`,
  `method="anchored_caa_text"`) — appends the actual OPTION SURFACE TEXT (e.g.
  "The Black man" vs "The white man") and captures the **masked-mean over the
  answer-text token span**. This removes the letter-identity confound and is
  **highly coherent** (split-half cosine **0.98**).

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
   chat_prompt + black_answer_text                   chat_prompt + other_answer_text
   ("The Black man")                                 ("The white man")
        │  forward pass (NO sampling)                        │  forward pass
        ▼                                                    ▼
   masked-mean L14 over answer-text span            masked-mean L14 over answer-text span
        h_black                                             h_other
        └──────────────── pair_diff = h_black − h_other ────┘

   direction = mean_over_items(pair_diff)      (float32, (4096,))
```

The **answer-text** variant (final) appends the option's surface string and
averages L14 over that span; the **letter** variant (failed) appended a bare
`A/B/C` letter and read the last token only.

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

## Coherence (n=400 anchored pairs, same held-out items for both variants)

| metric              | **answer-text** (final) | letter (failed) | race_black group mean-diff |
|---------------------|-------------------------|-----------------|----------------------------|
| split-half cosine   | **0.983**               | 0.274           | 0.908                      |
| norm_ratio          | **0.419**               | 0.069           | 0.18                       |
| raw_norm            | **8.535**               | 1.966           | 3.07                       |
| mean_diff_norm      | 20.36                   | 28.52           | —                          |
| avg activation norm | 89.15                   | 88.66           | 92.87 (embedding)          |

**Letter variant (failed).** Letter positions are balanced (black_letter A/B/C ≈
127/138/135), so the raw "which letter token sits last" component largely averages
out — which is *why* the mean direction's norm (1.97) is tiny next to the per-pair
diff norm (~28.5). What survives is only weakly consistent: **split-half cosine
0.27**, far below race_black's 0.91. The per-pair L14 diff was dominated by the
concrete A-vs-B-vs-C token identity, not the semantic "prefer the Black option"
component.

**Answer-text variant (final).** Anchoring on the OPTION SURFACE TEXT and averaging
L14 over the answer-text span removes the letter-identity confound. Coherence jumps
to **split-half cosine 0.98** and **norm_ratio 0.42** — *more* coherent than the
race_black group mean-diff (0.91 / 0.18) and dramatically above the letter variant.
The pair diffs now agree strongly on a single "prefer the named Black person over
the other named person" direction at L14.

**Read: the answer-text-anchored direction IS coherent enough to steer with.**
Split-half 0.98 says the "prefer Black option" signal is a stable, well-defined
L14 direction — the linear-encoding worry that killed the letter variant does not
apply once we anchor on the option text. Whether it *behaves* directionally on BBQ
(pushes picks onto the Black option, not onto both named options) is the next-stage
sweep; but on coherence grounds this is the first anchored vector that looks like a
genuine AIMing lever.

## Diffusion-LM answer capture

LLaDA is **masked-diffusion, not autoregressive**, so "the next-token position" is
not the natural capture site. We do **not** sample here. We build the FULLY
MATERIALIZED sequence `chat_prompt + answer` (no `<mask>` tokens present) and run a
single forward pass; the block-L14 output over the answer tokens is LLaDA's
**bidirectional contextual representation of that concrete option given the whole
prompt** — the "committed to this option" activation a CAA contrast needs.

- **answer-text (final):** capture the **masked-mean of L14 over the answer-text
  token span**. The span boundary = `len(tok(chat_prompt))`; the answer text is
  everything after it.
- **letter (failed):** the bare letter tokenizes to a single trailing token
  (`A/B/C` → ids `32/33/34`), captured at position **−1**. This is what let raw
  letter-token identity dominate the diff.

## Files

- `build_anchored.py` — reproducible builder (held-out selection → forward passes →
  direction + coherence → `.pt` + audit jsonl). Now builds the **answer-text**
  variant. Run:
  `CUDA_VISIBLE_DEVICES=0 python directional_steering/build_anchored.py`.
- `race_black_anchored_text.pt` — **final** direction + metadata (`layer=14`,
  `method="anchored_caa_text"`, `contrast="black_vs_other_named_answer_text"`,
  `n_pairs=400`, coherence metrics). Force-added past `*.pt`.
- `race_black_anchored.pt` — **superseded** letter-anchored direction
  (`method="anchored_caa"`, split-half 0.27). Kept to document the letter-method
  failure. Force-added past `*.pt`.
- `data/anchored_items.jsonl` — one row per held-out item (`example_id,
  question_index, prompt, black_letter, other_letter, black_answer_text,
  other_answer_text, black_group_tag`). Force-added past the `*.jsonl` ignore.

The BBQ steering eval is **NOT** run here — that is the next stage.
