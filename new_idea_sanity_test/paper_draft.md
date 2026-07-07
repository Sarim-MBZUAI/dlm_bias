# Steer and Hold: Closed-Loop Activation Steering for Diffusion Language Models

*Working draft. Results on LLaDA-8B-Instruct; bias results on BBQ.*

## Abstract

Activation steering controls a language model at inference by adding a behavior
direction to its hidden states. In **diffusion language models (DLMs)**, which
generate by iteratively denoising a masked sequence over many steps, the same
token positions are revisited and re-contextualized at *every* denoising step,
so a steering signal is not applied once but repeatedly, against a moving
representation. We show this fundamentally changes how steering should be done.
First, **open-loop steering is fragile**: a fixed additive push, or any schedule
that *reduces* the push as the sequence forms (fading it out, ramping it in, or
gating it by a confidence signal), loses the target behavior, because the
denoiser erodes an un-maintained intervention. Second, we cast steering as
**closed-loop control over the denoising trajectory**: at every step we *measure*
the current attribute level and *correct it to a setpoint*. A proportional
controller (**clamp**) holds the behavior; adding an integral term (**PI**) can
strengthen it. On LLaDA-8B, closed-loop steering beats fixed additive steering on
two tasks. On sentiment control it reaches higher attribute strength at equal
fluency (PI 0.955 > P 0.925 > additive 0.874). More importantly, on **demographic
bias injection (BBQ)** it is both **more directional and less destructive**:
using an item-anchored "prefer the Black option" direction, closed-loop control
raises the directional bias gap from **+0.135** (best open-loop) to **+0.216**
while *improving* task competence (disambiguated accuracy 0.874 → 0.91). Closed-
loop steering thus injects a targeted demographic bias more precisely, and with
less collateral damage, than the standard additive method.

## 1. Introduction

Inference-time activation steering adds a precomputed direction to a model's
residual stream to control behavior without retraining. It is well studied for
autoregressive LLMs and has recently been shown to work for DLMs. All existing
DLM steering, however, applies a **static, open-loop** intervention: a fixed
direction, at fixed strength, at every denoising step. We argue that the
iterative, bidirectional nature of DLM generation makes this the wrong control
structure, and that steering should instead be **closed-loop** — measured and
corrected to a target at each step. We demonstrate the failure of open-loop
schedules, introduce closed-loop (P and PI) steering for DLMs, and show it
improves both attribute control and — our headline application — the precision
of demographic-bias injection on BBQ.

## 2. Method

### 2.1 Notation

| symbol | meaning |
|---|---|
| `T`, `t` | number of reverse-denoising steps; step index (`t = T,…,1`) |
| `x_t` | the partially-unmasked token sequence at step `t` |
| `i` | token position index |
| `L` | the transformer block we intervene at (here `L = 14`) |
| `D` | hidden size (`D = 4096` for LLaDA-8B) |
| `h^{(L)}_{t,i} ∈ R^D` | residual-stream activation at block `L`, step `t`, position `i` |
| `v ∈ R^D`, `v̂ = v/‖v‖` | the steering direction and its unit vector |
| `a_{t,i} = ⟨h^{(L)}_{t,i}, v̂⟩` | **attribute level**: scalar projection of the activation onto the direction (how much of the behavior is present) |
| `α` | open-loop steering strength (coefficient) |
| `c*` | closed-loop **setpoint**: the attribute level we want to hold |
| `e_{t,i} = c* − a_{t,i}` | the control **error** (deficit from the setpoint) |
| `β ∈ [0,1)` | integral (momentum) coefficient for the PI controller |
| `u_{t,i}` | integrated (EMA) error used by the PI controller |

A DLM generates by starting from an all-masked target region and, over `T`
steps, predicting and progressively unmasking tokens; the full network is re-run
at every step, so `h^{(L)}_{t,i}` for a given position `i` is recomputed and
re-contextualized at each `t`. We attach a forward hook at block `L` that fires
at every step and modifies `h^{(L)}_{t,i}` for the generated positions.

### 2.2 The steering direction `v`

`v` is a contrastive mean-difference direction:
```
v  =  mean over positive examples of h^{(L)}  −  mean over negative examples of h^{(L)},     v̂ = v / ‖v‖ .
```
For **sentiment**, positive/negative = attribute-labeled sentences. For **bias**,
`v` is the *item-anchored answer-text* direction: on held-out BBQ Black-referent
items, `h^{(L)}` when the assistant answers the **Black** option minus when it
answers the **other named** option (split-half cosine 0.98).

### 2.3 The three interventions (applied at every generated position, every step)

**(a) Open-loop / additive (the standard method, incl. CAA):**
```
h^{(L)}_{t,i}  ←  h^{(L)}_{t,i} + α · v̂ .
```
A fixed push. `α` is a hyperparameter; the controller never observes the result.

**(b) Closed-loop proportional — `clamp` (P):** measure the current level and add
exactly the deficit needed to reach the setpoint `c*`:
```
a_{t,i} = ⟨h^{(L)}_{t,i}, v̂⟩ ,     e_{t,i} = c* − a_{t,i} ,
h^{(L)}_{t,i}  ←  h^{(L)}_{t,i} + e_{t,i} · v̂ .
```
This *holds* `a_{t,i}` at `c*`. If the denoiser pulls the activation back between
steps, `e_{t,i}` grows and the correction automatically increases — self-correcting.

**(c) Closed-loop PI — `cmom`:** add an integral (memory) term over denoising
steps, implemented as an exponential moving average of the error:
```
u_{t,i}  =  β · u_{t+1,i} + (1 − β) · e_{t,i}      (reset at the first step of each sequence),
h^{(L)}_{t,i}  ←  h^{(L)}_{t,i} + u_{t,i} · v̂ .
```
`β = 0` recovers clamp (P); `β > 0` accumulates the persistent deficit the model
keeps reverting, i.e. a PI controller. The integral resets per sequence, so it
integrates across the `T` denoising steps of one generation, not across examples.

### 2.4 Setpoint calibration

`c*` sets the held attribute level. We choose it to match the projection the
best open-loop setting reaches: e.g. for the bias direction the natural
projection is `a ≈ −3` and open-loop `α=8` drives it to `≈ +65`, so we set
`c* = 60` — a matched-strength comparison rather than a free parameter advantage.

## 3. Results

### 3.1 Sentiment control (LLaDA-8B, n=24 prompts; P(negative) / mean length)

| method | control | P(neg) | len |
|---|---|---:|---:|
| none | — | 0.05 | 53 |
| additive (CAA) | open-loop | 0.874 | 51 |
| clamp | P | 0.925 | 50 |
| **cmom** | **PI** | **0.955** | 51 |
| additive + momentum (no feedback) | — | 0.875 | 50 |
| open-loop schedules (decay / warm-up / early-commit / confidence-gate) | open-loop | 0.17–0.50 | — |

Ordering **PI > P > open-loop** holds; open-loop schedules that reduce the push
all lose the attribute.

### 3.2 Demographic bias injection (BBQ, LLaDA-8B; Black-referent ambiguous items, n=37)

Direction: item-anchored "prefer the Black option". `d_gap = Δblack − Δnonblack`
(directionality vs clean); `acc_disambig` = task competence.

| method | control | black | nonblk | abstain | **d_gap** | acc_disambig |
|---|---|---:|---:|---:|---:|---:|
| clean | — | 0.162 | 0.081 | 0.757 | 0.000 | 0.970 |
| additive α=4 | open-loop | 0.243 | 0.108 | 0.649 | +0.054 | 0.969 |
| additive α=8 | open-loop | 0.568 | 0.351 | 0.081 | +0.135 | 0.874 |
| **clamp** | **P** | 0.541 | 0.243 | 0.216 | **+0.216** | **0.907** |
| **cmom** | **PI** | 0.541 | 0.243 | 0.216 | **+0.216** | **0.913** |

Closed-loop control wins on **both** axes: a larger directional gap (**+0.216 vs
+0.135**) *and* better competence (**0.91 vs 0.874**), while not over-collapsing
abstention. The freed picks concentrate on the target group (0.541 vs 0.243)
rather than splitting evenly as under open-loop α=8 (0.568 vs 0.351). On this
task P and PI are equivalent (the integral term does not add), so the win comes
from the closed-loop hold itself.

### 3.3 Why open-loop fails (supporting analysis)

The steering direction is **noise-level dependent**: its split-half coherence
rises from ≈0.28 at a mostly-masked step to ≈0.996 at a nearly-clean step, and
`cos(v̂_clean, v̂_masked) = 0.42`. The behavior is only linearly encoded once the
sequence has partly decoded, and the denoiser re-contextualizes every position at
every step — so an un-maintained push is eroded. Closed-loop control succeeds
because it **re-asserts the setpoint at every step** against this erosion.

## 4. Related work

Activation steering for LLMs — ActAdd, CAA (Panickssery et al. 2024), ITI (Li et
al. 2023) — adds a fixed contrastive direction to the residual stream. Adaptive
variants scale or gate the intervention by input semantics or hidden-state
deviation (SADI, ACT, DAC, FASB, CAST). Our controllers relate to **activation
clamping** (Templeton et al. 2024; AxBench, Wu et al. 2025, arXiv:2501.17148) and
**affine concept editing** (ACE, arXiv:2411.09003), and to **closed-loop / PID
steering** for autoregressive models (arXiv:2506.18831, arXiv:2510.04309). For
DLMs, prior steering is static/open-loop: residual-stream additive
(arXiv:2512.24143), reference-alignment (ILRR, arXiv:2601.21647), step-scheduled
(arXiv:2605.10971), and logit-space (DLM-SWAI, arXiv:2605.29626). Prophet
(arXiv:2508.19982) shows DLM decoding exposes a confidence/convergence signal.

## 5. Contributions

1. **Closed-loop activation steering for diffusion LMs** — P (clamp) and PI
   (cmom) controllers that measure the attribute level and correct it to a
   setpoint at every denoising step.
2. **Open-loop steering is fragile in DLMs, closed-loop holds** — we show fixed
   or reduced-over-time pushes lose the behavior because iterative denoising
   erodes them, and give the mechanism (noise-dependent, re-contextualized
   directions).
3. **More precise bias injection** — closed-loop control injects a targeted
   demographic bias on BBQ with a larger directional gap (+0.216 vs +0.135) and
   *higher* task competence (0.91 vs 0.874) than standard additive steering.
4. A control-theoretic view of DLM steering (open-loop → P → PI), borne out on
   sentiment (PI > P > open-loop).

## 6. Limitations

Preliminary scope: one model (LLaDA-8B-Instruct), one layer (`L=14`); sentiment
at n=24 prompts, bias at n=37 Black-referent ambiguous items (magnitudes are
indicative, directional sign is reliable); a single setpoint `c*` per task; fixed
`β`. The integral term (PI) helps on sentiment but not on the bias task. Firming
up requires the full Black-referent set, a `c*` sweep, and a second attribute/model.
