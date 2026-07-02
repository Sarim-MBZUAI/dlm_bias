# Research Plan — "Aim vs. Disinhibit"

Living plan doc. Companion to `handoff.md` (state), `literature.md` (landscape),
`dlm_bias_report.pdf` (results). This file = the **question we are answering** and
the **experiments that answer it**.

## Research question

> **Aim vs. disinhibit: what makes an activation-steering direction _inject_
> directional demographic bias rather than merely _erode_ abstention — and does
> it hold in a frozen diffusion LM?**

**Locked framing:** MECHANISM. The paper's claim is not "we can attack a DLM"
(breadth) nor "the effect is real, not a sampler artifact" (rigor) — it is:

> In a frozen masked-diffusion LM, whether a training-free residual-stream
> intervention **injects directional demographic bias** or **merely erodes
> abstention** is determined not by data volume or target subset but by **HOW the
> direction is constructed** — a *decision-anchored* answer-text contrast **aims**;
> a *group/concept* mean-diff **disinhibits**.

## Why this is novel (from `literature.md`)

The space is **unoccupied**. Every DLM safety/fairness paper does
jailbreak / toxicity / style / tabular fairness, and all **mitigate**. None does
demographic-bias **injection** on **BBQ** via **activation steering**, and none
reports a **directional** result. LLaDA's own paper never studies social bias.
The huge CAA/RepE steering literature is autoregressive; nobody has separated
*aiming* from *disinhibition* as a property of the direction's construction, in a
diffusion LM where the hook fires on **every denoising step**.

```
        DIRECTIONAL GAP  (does it AIM at a group?)
              ▲
   +0.14 ┤                          ● item-anchored answer-text CAA  ← the win
         │                        ╱
         │                      ╱   (aiming lever)
   +0.05 ┤        ● group mean-diff (race_black, 792 pairs)
         │      ╱
     0   ┼────●──────●───────────────────────────►  ABSTENTION DROP
         │  clean  Ghostwriter          (does it just DISINHIBIT?)
              (input-space baseline: pure disinhibition, no aim)
```

Prior "bias steering" all lives on the x-axis. The claim: **only
decision-anchored construction moves you up the y-axis**, provable causally on a
frozen model.

## Sub-questions

| | Sub-question | Status |
|---|---|---|
| **RQ1 — Existence** | Can a training-free residual intervention make a frozen DLM *prefer* a group's answer, not just abandon "Unknown"? | shown at n=37; needs E1 to be solid |
| **RQ2 — Mechanism** | *Which property* causes aiming vs disinhibition? Hypothesis: decision-anchoring (answer-text) aims; group mean-diff disinhibits. | E2 (the heart) |
| RQ3 — Generality | Across groups / DLMs / open-gen? | **deferred** to later papers |
| RQ4 — Rigor | Content injection vs decoding/sampler artifact? | E7-lite here; full pack deferred |

## Spine — three experiments, one story

```
 E1 ── measurement fix ──►  E2 ── construction ablation ──►  E3 ── the map + metric
 (make RQ1 solid)           (the mechanism claim)            (the headline figure)
```

| | What it does | Holds fixed | Varies | Produces | Success criterion |
|---|---|---|---|---|---|
| **E1** | Rescore the answer-text vector (+ group mean-diff for contrast) on the **FULL** Race_ethnicity Black-referent ambiguous set, not just the 37 in seed-42 random-1000 | model, L14, α-grid, sampler | denominator (37 → hundreds) | tight CIs on the +0.135 gap | gap stays **positive & significant** at large n |
| **E2** | **THE HEART.** Vary *only* direction construction | held-out items, L14, α-grid, sampler, capture logic | {group mean-diff · letter-anchored · **answer-text-anchored** · FairPCA-subspace} | ablation table: coherence × aim × disinhibit × competence, per construction | answer-text anchoring is the **only** one high on the AIM axis |
| **E3** | Decompose into 2 axes; define **aiming ratio** = Δgap / Δabstention | — | every (construction × α) + Ghostwriter overlay | the 2-axis map (headline fig) + a scalar that ranks constructions | constructions **separate cleanly** on the y-axis |

## Minimal support (report, not spine)

- **E4-lite** — one more category (gender *or* religion), item-anchored → proves
  it is not a race-only artifact. 🟠
- **E7-lite** — two confound controls: matched-sampler baseline + answer-position
  randomization. Preempts the "it's just a decoding / proximity artifact"
  reviewer. **Not optional** for a mechanism paper. 🟠
- **E9** — CCS (Concept Coherence Score) + the `acc_disambig` competence cliff →
  semantic preservation. 🟡

## Dependencies / risks

```
 n=37   ──is the gate──►  E1 must land before E2/E3 are trustworthy
 confound (proximity / answer-position)  ──►  E7-lite is not optional
```

## Benchmarks

| Benchmark | Role | Have it? |
|---|---|---|
| **BBQ** (11 cats; `s_AMB`/`s_DIS` + attack metrics) | Primary injection surface | cached |
| **Full Race_ethnicity Black-referent set** | E1 measurement fix | derivable from 6880-item cache |
| **CrowS-Pairs / StereoSet** | Direction sources; group mean-diff arm of E2 | used |
| **CCS** | E9 semantic preservation metric | metric only |
| WinoBias / Winogender, BOLD / HolisticBias / RealToxicityPrompts | generality / open-gen | **deferred** |

## Diffusion LMs

| Model | Layers / H | Role | Have it? |
|---|---|---|---|
| **LLaDA-8B-Instruct** | 32 / 4096, mid=**L14** | Primary (all spine experiments) | yes |
| Dream-v0-Instruct-7B | 28 / 3584, mid≈**L12–14** | cross-model — **deferred** | yes |
| Dream-v0-Base-7B | 28 / 3584 | base-vs-instruct — **deferred** | yes |

## Deferred (out of this paper's scope, kept live)

Cross-model (Dream), open-generation transfer (BOLD/HolisticBias), denoising-
trajectory localization + SAE (E8) → the *generality* and *rigor* follow-ups.

## Status

Plan locked 2026-07-02 (mechanism framing). No spine experiment run yet.
Next action = **E1** (measurement fix), the gate for E2/E3.
