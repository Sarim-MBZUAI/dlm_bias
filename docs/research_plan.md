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

## Two phases

- **Phase 1 — mechanism spine (E1→E2→E3), LLaDA-8B + BBQ only.** Proves the
  aim-vs-disinhibit claim once, cleanly. Does NOT wait on Phase 2.
- **Phase 2 — replication matrix (3 benchmarks × 3 DLMs).** Shows the construction
  property is not an LLaDA/BBQ artifact. This is the strong form of contribution #2.

## Benchmarks

The aim-vs-disinhibit metric needs **an abstention option + two named groups**.
Benchmarks are chosen by which axis they give:

| Benchmark | Abstain | Directional | Role | Have it? |
|---|:--:|:--:|---|---|
| **BBQ** (11 cats; `s_AMB`/`s_DIS` + attack metrics) | ✓ | ✓ | Core — both axes; all of Phase 1 | cached |
| Full Race_ethnicity Black-referent set | ✓ | ✓ | E1 measurement fix | derivable from 6880-item cache |
| **UNQOVER** | ✗ | ✓ | Phase 2 — directional replication under a *different* construction; positional-confound controls double as rigor | download |
| **BBG** (Bias Benchmark for Generation) *(or Open-BBQ if scoring too heavy)* | ✓-ish | ✓ | Phase 2 — free-generation test (MCQ understates bias); most diffusion-native | download |
| CrowS-Pairs / StereoSet | — | — | Direction *sources*; group mean-diff arm of E2 | used |
| CALM · OpenBiasBench | — | — | Considered, **cut** — CALM overlaps E7-lite; OpenBiasBench (unseen categories) orthogonal to the mechanism claim | — |
| CCS | — | — | E9 semantic-preservation metric | metric only |

## Diffusion LMs

Replicate the construction property across **3 DLMs** (all local):

| Model | Layers / H | Mid layer | Role | Have it? |
|---|---|---|---|---|
| **LLaDA-8B-Instruct** | 32 / 4096 | **L14** | Primary — Phase 1 + Phase 2 | yes |
| **Dream-v0-Instruct-7B** | 28 / 3584 | ≈**L12–13** | Phase 2 — different DLM family (AR-adapted) | yes |
| **Dream-v0-Base-7B** *(provisional)* | 28 / 3584 | ≈**L12–13** | Phase 2 — free "alignment vs injectability" ablation; caveat: base is weak at following MCQ/gen format | yes |

Each new model needs **layer + α recalibration** (different depth and hidden-state
norm → the α grid does not transfer). Recalibrate α so `α·raw_norm` is a matched
fraction of the per-model activation norm.

```
              BBQ        UNQOVER      BBG/Open-BBQ
LLaDA-Inst     ●            ●              ●
Dream-Inst     ●            ●              ●
Dream-Base     ●            ●              ○  (base gen format uncertain)
```

## Deferred (out of scope, kept live)

WinoBias/Winogender, BOLD/HolisticBias/RealToxicityPrompts; denoising-trajectory
localization + SAE (E8); LLaDA-1.5 / LLaDA-MoE / Mercury (need download).

## Full experiment list (execution order)

**Phase 1 — mechanism (LLaDA + BBQ):**
- **E1** 🔴 gate — measurement fix (full Black-referent set minus the 400 build items)
- **E2** 🔴 core — construction ablation (group mean-diff · letter · answer-text · FairPCA)
- **E3** 🟠 — aim-vs-disinhibit map + aiming-ratio metric
- **E7-lite** 🟠 — matched-sampler baseline + answer-position randomization (rigor)
- **E4-lite** 🟠 — one more category (gender or religion), winning construction
- **E9** 🟡 — CCS + competence, folded into E2/E3

**Phase 2 — replication matrix (3 benchmarks × 3 DLMs):**
- **E5** — port winning construction to Dream-Instruct (recal L≈13, α) on BBQ
- **E6** — UNQOVER on all 3 models (directional replication + positional rigor)
- **E8** — BBG/Open-BBQ generation eval (aim survives free generation)
- **E10** — Dream-Base alignment-vs-injectability ablation *(provisional)*

## Status

Plan locked 2026-07-02 (mechanism framing; Phase 2 replication matrix added).
No experiment run yet. Next action = **E1**, the gate for everything.
Provisional Phase-2 picks (BBG gen benchmark, Dream-Base 3rd model) are
correctable — confirm before Phase 2.
