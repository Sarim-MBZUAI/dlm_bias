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

## Figure 1 — Open-loop vs closed-loop steering (in simple terms)

A diffusion LM rewrites the **same** token positions over `T` denoising steps, and
the steering hook fires at **every** step:

```
        denoising:   step T  ───────────────────►  step 0
                   [ mostly masked ]            [ finished text ]
                          ↑↑↑ at every step we can nudge the hidden state h
```

**OPEN-LOOP (fixed additive — the standard method).** Push by a constant `α·v̂`,
never look at the result:

```
        ┌──────────────────────────────────┐
   h ──►│   h̃ = h + α·v̂    (α fixed, blind)  │──►  model  ──►  next-step h
        └──────────────────────────────────┘
                     ✗  no feedback wire
   the denoiser pulls h̃ back toward default each step → an un-maintained
   push is ERODED → the behavior reverts.
```

**CLOSED-LOOP (clamp = P / cmom = PI — ours).** Measure how much behavior is
present, then add exactly enough to hit a target `c*` — every step:

```
        ┌───────────────────────────────────────────────┐
   h ──►│  measure:  a = ⟨h, v̂⟩                          │
        │  error:    e = c* − a                          │
        │  correct:  h̃ = h + e·v̂   (PI: + integral of e) │──► model ──► next h
        └───────────────▲───────────────────────────────┘                │
                        │                                                 │
                        └──────── feedback: re-measure a next step ◄───────┘
   SELF-CORRECTING: if the model reverts (a drops), e grows → it pushes
   harder automatically → the behavior is HELD at c*.
```

**One-line difference:** open-loop *shoves with a fixed force and hopes it sticks*
(it doesn't — denoising erodes it); closed-loop *reads the current level and holds
it at the target every step*. Like **pressing a spring** (open-loop — springs back)
vs a **thermostat** (closed-loop — holds the setpoint).

**Control-theory view** (borne out empirically: PI > P > open-loop):
```
   open-loop :  c* ─►[ gain ]───────────────► plant ─► out        (no sensor)

   clamp (P) :  c* ─►(⊕)─►[ e·v̂ ]───────────► plant ─► out ─┐
                     ▲                                       │
                     └─────────── sensor a = ⟨h,v̂⟩ ──────────┘

   cmom (PI) :  same loop, plus an INTEGRAL (memory) of e across denoising steps
```

## 2. Method

### 2.1 Setup: where and when we intervene

A masked-diffusion LM generates a target region of tokens by starting from an
all-`[MASK]` region and, over `T` reverse-denoising steps `t = T, …, 1`, running a
**full forward pass** at each step and progressively unmasking tokens. Crucially,
a token position is **not** written once: the whole network is re-run every step,
so its internal activation is **recomputed and re-contextualized at every `t`**
until it is finalized. We intervene with a single **forward hook on transformer
block `L`** (here `L = 14`) that fires on every step and edits the residual-stream
activation at the generated positions before it flows into block `L+1`.

### 2.2 Notation

| symbol | meaning |
|---|---|
| `T`, `t` | number of denoising steps; step index (`t = T,…,1`) |
| `i` | token-position index; generated positions are the ones we steer |
| `L` | block we hook (`L = 14`) |
| `D` | hidden size, `D = 4096` (LLaDA-8B) |
| `h ≡ h^{(L)}_{t,i} ∈ R^D` | residual-stream activation vector at block `L`, step `t`, position `i` |
| `v ∈ R^D` ; `v̂ = v/‖v‖` | steering direction (raw) and its **unit** vector, `‖v̂‖ = 1` |
| `a ≡ a_{t,i} ∈ R` | **attribute level** = projection of `h` onto `v̂` (§2.4) |
| `c* ∈ R` | **setpoint**: the attribute level we want to hold (a scalar) |
| `e ≡ e_{t,i} = c* − a` | control **error** (how far below the setpoint we are) |
| `α ∈ R` | open-loop steering coefficient |
| `β ∈ [0,1)` | integral coefficient (momentum) for the PI controller |
| `u ≡ u_{t,i}` | integrated (EMA) error used by PI |

### 2.3 The steering direction `v̂`

`v` is a contrastive mean-difference of block-`L` activations between examples that
exhibit the behavior and examples that do not, then normalized to unit length:
```
v  =  E_{x ∈ D+}[ h^{(L)}(x) ]  −  E_{x ∈ D-}[ h^{(L)}(x) ] ,        v̂ = v / ‖v‖ .
```
- **Sentiment:** `D+`/`D-` are positive/negative sentences.
- **Bias (headline):** the *item-anchored answer-text* direction — on held-out BBQ
  Black-referent items, `h^{(L)}` averaged over the answer-text span when the
  assistant answers the **Black** option, minus the same for the **other named**
  option (split-half cosine 0.98).

`v̂` points **toward** the behavior; steering in the `+v̂` direction increases it.

### 2.4 The attribute level `a = ⟨h, v̂⟩` (definition and computation)

`a` is the **scalar (dot-product) projection of the activation `h` onto the unit
behavior direction `v̂`** — a single number saying *how much of the behavior is
present in this activation*:
```
a  =  ⟨h, v̂⟩  =  Σ_{d=1}^{D}  h_d · v̂_d  =  ‖h‖ · cos θ ,
```
where `h_d, v̂_d` are the `d`-th coordinates and `θ` is the angle between `h` and
`v̂`. Because `v̂` is unit, `a` is the **signed length of the component of `h` along
`v̂`**, in the same units as the activations: `a > 0` = behavior present, `a < 0` =
opposite, `|a|` = how strongly.

**How it is computed (per position, per step).** At denoising step `t`, the hook
receives the block-`L` output for the whole sequence, shape `(positions × D)`. For
each generated position `i` we take one dot product:
```
a_{t,i} = Σ_d  h^{(L)}_{t,i,d} · v̂_d          # in code: a = (h * v̂).sum(dim=-1)  -> one scalar per position
```
This is `O(D)` per position — a few thousand multiply-adds, negligible next to the
forward pass.

**Key property (why this enables control).** Adding `e·v̂` to `h` moves the
projection by exactly `e`, because `v̂` is unit:
```
a' = ⟨h + e·v̂, v̂⟩ = ⟨h,v̂⟩ + e·⟨v̂,v̂⟩ = a + e·‖v̂‖² = a + e .
```
So choosing `e = c* − a` gives `a' = c*` — **one correction sets the attribute
level exactly to the setpoint** (until the model recomputes `h` at the next step,
which is why we re-apply it every step). This identity is what makes "hold `a` at
`c*`" well-defined, and is why the direction is normalized.

### 2.5 The three interventions (applied at every generated position, every step)

**(a) Open-loop / additive** (standard steering, incl. CAA) — a fixed, blind push:
```
h  ←  h + α·v̂ .
```
`α` is set once; the intervention never observes `a`. The projection is nudged by
`+α` each step, but the denoiser pulls it back, and the push is not adjusted.

**(b) Closed-loop proportional — `clamp` (P)** — measure, then correct to the
setpoint:
```
a = ⟨h, v̂⟩ ,     e = c* − a ,     h  ←  h + e·v̂         ⇒   projection becomes c* .
```
Self-correcting: if the model reverts (`a` drops), `e` grows, so the applied
correction automatically grows — the level is **held** at `c*`.

**(c) Closed-loop PI — `cmom`** — add an integral (memory) term over denoising
steps, as an exponential moving average of the error (reset at each new sequence):
```
u_t = β·u_{t+1} + (1−β)·e_t ,        h  ←  h + u·v̂ .
```
`β = 0` recovers clamp (P); `β > 0` accumulates the *persistent* deficit the model
keeps reverting (the integral term of a PI controller). The EMA resets per
sequence, so it integrates across the `T` steps of one generation, not across
examples.

### 2.6 Setpoint calibration

`c*` sets the held level. We pick it to **match the projection the best open-loop
run reaches**, so the comparison is at equal strength rather than a free parameter
advantage. Concretely (bias direction): the natural projection is `a ≈ −3`, and
open-loop `α=8` moves it to `≈ −3 + 8·‖v‖ = +65`; we therefore set `c* = 60`.

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

### 3.3 Single-layer vs all-layer (is L14 an unfair choice?)

We repeat the comparison applying the same direction at **all 32 blocks** (matched
total strength; per-layer setpoint = that layer's natural projection + offset).

| method | steering site | d_gap | acc_disambig |
|---|---|---:|---:|
| open-loop α=8 | L14 only | +0.135 | 0.874 |
| clamp (P) | L14 only | **+0.216** | 0.907 |
| cmom (PI) | L14 only | **+0.216** | 0.913 |
| additive | all 32 layers | +0.027 | 0.969 |
| clamp (P) | all 32 layers | +0.054 | 0.974 |
| cmom (PI) | all 32 layers | +0.081 | 0.972 |

Two points. (i) **Single-layer L14 is not a disadvantage — it is the stronger
site.** All-layer steering is much weaker (best gap +0.081 vs +0.216), because the
L14-built direction is only a genuine feature at L14; applying it at the other 31
blocks dilutes the aim (and increasing all-layer strength to compensate destroys
generation). (ii) **The closed-loop ordering is robust across regimes**: all-layer
gives cmom (+0.081) > clamp (+0.054) > additive (+0.027) — and here PI beats P,
so the integral term helps in the harder all-layer setting.

### 3.4 Why open-loop fails (supporting analysis)

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
**affine concept editing** (ACE, arXiv:2411.09003). Most directly, **PID Steering**
(Nguyen et al., ICLR 2026, arXiv:2510.04309; also arXiv:2506.18831) provides the
control-theoretic foundation we build on: it shows standard steering *is* a
proportional (P) controller and introduces the full PID controller — but for
**autoregressive LLMs**, with the loop running over **transformer layers (depth)**
inside one forward pass. Our loop differs in kind: it runs over the **denoising
trajectory (time)** of a diffusion LM, re-correcting the *same* positions across
steps before they commit, and we show it is *necessary* there because open-loop
steering is eroded by re-contextualization — a failure mode absent in the
layer-depth setting. For DLMs, prior steering is static/open-loop: residual-stream additive
(arXiv:2512.24143), reference-alignment (ILRR, arXiv:2601.21647), step-scheduled
(arXiv:2605.10971), and logit-space (DLM-SWAI, arXiv:2605.29626). Prophet
(arXiv:2508.19982) shows DLM decoding exposes a confidence/convergence signal.

## 5. Contributions

Control-theoretic steering (steering = P; extend to PID) was introduced for
autoregressive LLMs over the *layer* axis (Nguyen et al., 2026). We do not claim
that framing; our contributions are what the **diffusion** architecture changes:

1. **Closed-loop steering over the *denoising trajectory*** — P (clamp) and PI
   (cmom) controllers that re-measure and re-correct the *same positions* across
   denoising steps before they commit. This is a distinct control loop from the
   AR/layer-depth formulation: a real temporal feedback loop that only exists
   because DLMs revisit positions, amplified by bidirectional attention.
2. **Open-loop steering is fragile in DLMs, closed-loop holds** — a
   *DLM-specific* failure mode: fixed or reduced-over-time pushes lose the
   behavior because iterative denoising re-contextualizes and erodes them (we give
   the mechanism: directions are noise-dependent and only coherent at low noise).
   This does not arise in the layer-depth setting.
3. **More precise bias injection (headline application)** — closed-loop control
   injects a targeted demographic bias on BBQ with a larger directional gap
   (+0.216 vs +0.135) and *higher* task competence (0.91 vs 0.874) than standard
   additive steering; the result is robust across single-layer and all-layer.
4. Empirically the control-theoretic ordering **PI ≥ P > open-loop** holds in the
   DLM setting (sentiment and all-layer bias), consistent with (and extending to
   the denoising-time axis) the account of Nguyen et al. (2026).

## 6. Limitations

Preliminary scope: one model (LLaDA-8B-Instruct), one layer (`L=14`); sentiment
at n=24 prompts, bias at n=37 Black-referent ambiguous items (magnitudes are
indicative, directional sign is reliable); a single setpoint `c*` per task; fixed
`β`. The integral term (PI) helps on sentiment but not on the bias task. Firming
up requires the full Black-referent set, a `c*` sweep, and a second attribute/model.
