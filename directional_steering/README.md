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

- `anchored_analysis.py` — reproducible directional analysis (adapted from
  `race_steering/black_analysis.py`). Reads only saved JSON/JSONL; no GPU. Builds
  the Black-referent map (seed-42 BBQ, `BLACK_TAGS`), computes per-alpha
  Black-referent AMBIGUOUS metrics (**n=37**), directionality gaps, flips, and
  `acc_disambig`, and prints the anchored-vs-group-mean-diff peak-gap comparison.
  Run: `python directional_steering/anchored_analysis.py`.
- `results/bbq_L14_anchored_a{1,2,3,4,6,8}.json` (+ `_samples.jsonl`) — the
  answer-text-anchored steering sweep on BBQ. Force-added past `*.jsonl` ignore.
- `figs/anchored_steering.png` — Black-pick vs non-Black-pick vs abstention vs
  alpha, with the `acc_disambig` competence line.

## RESULT — the anchored vector IS directional (payoff analysis)

Black-referent AMBIGUOUS items, **n=37** (seed-42 sample; small — treat as
indicative, not a population estimate).

| alpha | black_pick | nonblk_pick | abstention | acc_disambig | d_gap (Δblack−Δnonblk) |
|------:|-----------:|------------:|-----------:|-------------:|-----------------------:|
| 0     | 0.162      | 0.081       | 0.757      | 0.970        | —        |
| 1     | 0.189      | 0.081       | 0.730      | 0.969        | +0.027   |
| 2     | 0.216      | 0.054       | 0.730      | 0.969        | +0.081   |
| 3     | 0.270      | 0.081       | 0.649      | 0.967        | +0.108   |
| 4     | 0.243      | 0.108       | 0.649      | 0.969        | +0.054   |
| 6     | 0.351      | 0.162       | 0.486      | 0.959        | +0.108   |
| **8** | **0.568**  | **0.351**   | **0.081**  | **0.874**    | **+0.135** |

**Peak directional gap: +0.135 @ alpha=8** — and, crucially, at alpha=8
competence is still **acc_disambig 0.874** (no competence cliff anywhere in the
sweep; min 0.874). black_pick outruns non-Black-pick at **every** alpha; the gap
is monotone-ish and positive throughout the competent range.

**Anchored vs group mean-diff (race_black):**

| direction                    | peak d_gap | @alpha | acc_disambig@peak |
|------------------------------|-----------:|-------:|------------------:|
| **item-anchored (answer-text)** | **+0.135** | **8** | **0.874** (competent) |
| group mean-diff race_black   | +0.054     | 8      | 0.949             |

The group mean-diff vector peaked at only **+0.054** and then went **negative**
(gap −0.16 to −0.24 at alpha 12–16) once it started pushing harder — its extra
picks went onto the *non-Black* option, and only when competence had already
cratered (acc_disambig 0.29–0.38). The anchored vector's peak gap is **2.5×
larger AND achieved while still competent**.

**Flips at the best alpha (a=8), of the 28 items clean answered Unknown:**
**14 → Black** vs **11 → non-Black** (3 stayed Unknown). Black-directed flips
outnumber non-Black at the same alpha — the abstention it collapses lands
disproportionately on the Black option.

### VERDICT

**Yes — the coherent (split-half 0.98) item-anchored answer-text vector finally
produced DIRECTIONAL Black bias.** black_pick rises materially faster than
nonblack_pick across the entire sweep, the peak directional gap (**+0.135**) is
~2.5× the group mean-diff's peak (**+0.054**), and — unlike race_black — it is
reached **without a competence cliff** (acc_disambig 0.874 at alpha=8, vs
race_black needing alpha≥12 where the gap flipped negative and competence
collapsed). It still collapses abstention (0.757→0.081), but this time the
collapse is *aimed*: it lands on the Black option preferentially, not on both
named options equally. This is the first vector in the project to steer toward a
specific answer rather than merely suppress "Unknown."

**Caveat (n=37).** The denominator is 37 Black-referent AMBIGUOUS items, so
individual rates move in ~0.027 steps and the exact gap is noisy. The signal is
consistent (monotone, positive at every alpha, corroborated by the flip counts),
but the magnitudes are indicative, not precise population estimates.

The BBQ steering eval itself is run upstream (GPU); this folder only analyses the
saved outputs.

---

## diagnostics/ — control-formulation probes (closed-loop vs open-loop)

Two GPU diagnostics that test whether a **closed-loop controller** over the
LLaDA denoising trajectory is justified, versus plain open-loop addition. Both
run on the L14 anchored answer-text direction, 16 Black-referent ambiguous items,
default sampler (gen=32/steps=64/block=32, temp=0). Each writes a `.py` + `.json`
+ `.png`.

### `fig3_denoise_error.*` — is the residual-projection axis an integrating plant?
Transplants PID-Steering's (arXiv:2510.04309) Fig. 3 steady-state-error test onto
the **denoising-step axis**. Measures the incoming L14 projection `<h, v̂>` per
step under clean / open-loop α8 / clamp(P) c*=60 / cmom(PI) c*=60.

**Finding — closed-loop over the residual projection is NOT justified.** The
incoming projection reverts to its natural value (~−3, far below c*=60) every
step regardless of the previous step's correction: the plant has **no cross-step
memory** at the measurement point. Clamp fully rejects the disturbance *within*
each step, so an integral term has nothing to accumulate — **clamp ≡ cmom to
within 0.036 over all 64 steps** (confirms the "cmom is a unity-gain leaky
integrator" note). Because the measured state barely varies, the clamp correction
`(c*−a)` is a near-constant ~62 every step, so **clamp degenerates into a fixed
open-loop push** (≈62/‖v‖ ≈ 7.3) — which is why open-loop α8 and clamp land in
the same ballpark and why closed-loop did not beat open-loop at n=1600.

### `pB_ptarget_accum.*` — is the DECODE the right controlled variable?
Reformulation "B": close the loop on a decode-space signal that actually
accumulates. Controlled variable `p_target(t)` = model probability on the target
letter token at the answer position, per denoising step. Validated against ground
truth (committed token == parsed letter 16/16; final p_target 0.98–0.99 for
target-outcome items vs 0.004–0.03 otherwise).

**Finding — qualified PASS; the decode IS the right variable.** Unlike the L14
projection, `p_target` is non-memoryless and **strongly controllable**: open-loop
steering holds it **+0.31 above clean from the first forward pass and sustains it**
across the pre-commit window (steps 0–31), flipping committed outcomes **2/16 →
12/16**. But the accumulation is modest (~+0.15 gradual drift) plus a **one-shot
commitment snap** at step ~31, after which (temp 0) the token is frozen and steps
32–63 are inert. So a **proportional controller acting on `p_target` over steps
0–31 is well-posed and diffusion-native**, but an integral/PID term is **not**
supported here (no persistent step-to-step disturbance to reject). Bonus: the
effect is front-loaded at t0 and the second half of the schedule is dead compute
— an act-early / early-commit efficiency angle.

**Takeaway for the formulation:** drop the P-vs-PI(vs-PID) controller framing (it
reduces to open-loop on the projection axis, and PID buys nothing on the decode
axis); lead with (1) the aim-vs-disinhibit construction result and (2) a
proportional decode-space controller with an early-action/efficiency story.
