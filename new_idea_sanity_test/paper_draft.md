# Steer and Hold: Closed-Loop Activation Steering for Diffusion Language Models

*Working draft. Results on LLaDA-8B-Instruct; bias results on BBQ.*

## Abstract

Activation steering controls a language model at inference by adding a behavior
direction to its hidden states. A masked-**diffusion language model (DLM)**
recomputes every token position at every denoising step, so a steering push is
not applied once to a fixed representation — it lands on a representation the
denoiser keeps rewriting. We show this changes what steering has to be. First,
**open-loop steering is fragile**: a fixed additive push, or any schedule that
*reduces* the push as the sequence forms (fading it out, ramping it in, or
gating it by confidence), loses the target behavior, because the denoiser
erodes an un-maintained intervention. Second, we cast steering as **closed-loop
control over the denoising trajectory**: treating the frozen denoiser as a plant
whose per-step re-contextualization is a disturbance, at every step we *measure*
the current attribute level `a = ⟨h, v̂⟩` and re-apply the error `e = c* − a`,
holding the level at a setpoint `c*` like a thermostat. This is a **proportional
(P) controller** (**clamp**); a variant (**cmom**) smooths the correction across
steps with an exponentially-weighted average of the error. On LLaDA-8B,
closed-loop steering beats fixed additive steering on two tasks. On sentiment
control it reaches higher attribute strength at equal length (cmom 0.955 > clamp
0.925 > additive 0.874). More importantly, on **demographic bias injection
(BBQ)** it is both **more directional and less destructive**: using an
item-anchored "prefer the Black option" direction, closed-loop control raises
the directional bias gap from **+0.135** (best open-loop) to **+0.216** while
degrading task competence less (disambiguated accuracy 0.874 → 0.91). Closed-loop
steering thus injects a targeted demographic bias more precisely, and with less
collateral damage, than the standard additive method. Feedback control of
activations was concurrently proposed for autoregressive LLMs over network depth
(Nguyen et al., 2026); both instantiate the classical control primitive on
different loops — theirs across layers, ours across the denoising trajectory, a
loop that exists only because a DLM revisits the same positions.

## 1. Introduction

Inference-time activation steering adds a precomputed direction to a model's
residual stream to control behavior without retraining. It is well studied for
autoregressive LLMs and has recently been shown to work for DLMs — but all
existing DLM steering is **static and open-loop**: a fixed direction, at fixed
strength, at every denoising step.

We argue this is the wrong control structure for a diffusion LM. An
autoregressive model writes each token once; a masked-diffusion LM re-runs the
full network at every one of its `T` denoising steps, recomputing and
re-contextualizing every position until it commits. A fixed push therefore
lands on a representation the denoiser immediately rewrites: the push is
eroded and the behavior drifts back. The fix is **feedback**. Borrowing the
language of classical control, we treat the frozen denoiser as a *plant* and the
attribute we want as a *setpoint*: at every step we *measure* how much of the
behavior is present — the projection `a = ⟨h, v̂⟩` of the hidden state onto the
unit behavior direction — and *re-apply* exactly the error `e = c* − a` needed to
restore a setpoint `c*`. Open-loop steering shoves with a fixed force and hopes
it sticks (feedforward); closed-loop steering reads the current level and holds
it — a pressed spring vs a thermostat.

Casting activation steering as control is not unique to us. Concurrently,
Nguyen et al. (2026) show that standard steering *is* a proportional controller
and extend it to a full P/I/D controller — but for **autoregressive LLMs**,
running the loop over transformer **layers (depth)** inside a single forward
pass. Proportional and integral control are classical, decades-old primitives
from control theory; both their work and ours instantiate them, independently,
on different loops. Our loop is the axis a diffusion LM uniquely exposes: the
**denoising trajectory (time)**, re-correcting the *same* positions across steps
before they commit. We show feedback is not just beneficial but *necessary*
here, because open-loop steering is eroded by re-contextualization — a failure
mode that the layer-depth setting does not have.

We demonstrate the failure of open-loop schedules, formalize closed-loop
proportional steering (and an error-smoothing variant) for DLMs, and show it
improves both attribute control and — our headline application — the precision
of demographic-bias injection on BBQ.

## Figure 1 — Open-loop vs closed-loop steering

A diffusion LM rewrites the **same** token positions over `T` denoising steps,
and the steering hook fires at **every** step:

```
        denoising:   step T  ───────────────────►  step 0
                   [ mostly masked ]            [ finished text ]
                          ↑↑↑ at every step we can nudge the hidden state h
```

**OPEN-LOOP (fixed additive — the standard method).** Push by a constant `α·v`,
never look at the result (pure feedforward):

```
        ┌──────────────────────────────────┐
   h ──►│   h̃ = h + α·v    (α fixed, blind)  │──►  model  ──►  next-step h
        └──────────────────────────────────┘
                     ✗  no feedback wire
   the denoiser pulls h̃ back toward default each step → an un-maintained
   push is ERODED → the behavior reverts.
```

**CLOSED-LOOP (clamp = P / cmom = error-smoothed P — ours).** Measure how much
behavior is present, then add exactly enough to hit the setpoint `c*` — every
step:

```
        ┌───────────────────────────────────────────────┐
   h ──►│  measure:  a = ⟨h, v̂⟩                          │
        │  error:    e = c* − a                          │
        │  correct:  h̃ = h + e·v̂   (cmom: smooth e first) │──► model ──► next h
        └───────────────▲───────────────────────────────┘                │
                        │                                                 │
                        └──────── feedback: re-measure a next step ◄───────┘
   SELF-CORRECTING: if the model reverts (a drops), e grows → it pushes
   harder automatically → the behavior is HELD at c*.
```

In control terms, clamp is a proportional (P) controller and cmom applies an
exponentially-weighted moving average of the error across denoising steps.
Empirically the ordering **cmom ≥ clamp > open-loop** holds (§3).

## 2. Method

### 2.1 Setup: where and when we intervene

A masked-diffusion LM generates a target region of tokens by starting from an
all-`[MASK]` region and, over `T` reverse-denoising steps `t = T, …, 1`, running
a **full forward pass** at each step and progressively unmasking tokens.
Crucially, a position is **not** written once: its activation is recomputed and
re-contextualized at every step until it is finalized. We intervene with a
single **forward hook on transformer block `L`** (here `L = 14`) that fires on
every step and edits the residual-stream activation before it flows into block
`L+1`.

**Which positions.** The two experimental settings differ here, and we are
explicit about it. In the **bias/BBQ** setting the hook edits **all positions**
of the block's hidden state, prompt tokens included — the setpoint is imposed on
context and generated tokens alike. In the **sentiment** setting the hook is
restricted to the generated positions only. All BBQ numbers in this paper were
produced with all-position steering; a generated-only ablation for BBQ is future
work (E7-lite).

### 2.2 Notation

| symbol | meaning |
|---|---|
| `T`, `t` | number of denoising steps; step index (`t = T,…,1`) |
| `i` | token-position index |
| `L` | block we hook (`L = 14`) |
| `D` | hidden size, `D = 4096` (LLaDA-8B) |
| `h ≡ h^{(L)}_{t,i} ∈ R^D` | residual-stream activation at block `L`, step `t`, position `i` |
| `v ∈ R^D` ; `v̂ = v/‖v‖` | steering direction (raw) and its **unit** vector, `‖v̂‖ = 1` |
| `a ≡ a_{t,i} ∈ R` | **attribute level** = projection of `h` onto `v̂` (§2.4) |
| `c* ∈ R` | **setpoint**: the attribute level we want to hold |
| `e ≡ e_{t,i} = c* − a` | control **error** |
| `K_p` | proportional gain (we use `K_p = 1`) |
| `α ∈ R` | open-loop steering coefficient (multiplies the **raw** `v`) |
| `β ∈ [0,1)` | smoothing coefficient of the cmom variant |
| `u ≡ u_t` | error-smoothed correction used by cmom |

**Vector convention (stated once).** Open-loop additive steering multiplies the
**raw** direction `v` (norm ≈ 8.54 for the bias direction): `h ← h + α·v`. The
closed-loop controllers operate on the **unit** direction `v̂`, both to *measure*
`a = ⟨h, v̂⟩` and to *apply* the correction along `v̂`. Whenever we compare a
setpoint `c*` to an open-loop `α`, we account for this `‖v‖` factor explicitly
(§2.6).

### 2.3 The steering direction

`v` is a contrastive mean-difference of block-`L` activations between examples
that exhibit the behavior and examples that do not, normalized to unit length:
```
v  =  E_{x ∈ D+}[ h^{(L)}(x) ]  −  E_{x ∈ D-}[ h^{(L)}(x) ] ,        v̂ = v / ‖v‖ .
```
- **Sentiment:** `D+`/`D-` are positive/negative sentences.
- **Bias (headline):** the *item-anchored answer-text* direction — on held-out
  BBQ Black-referent items, `h^{(L)}` averaged over the answer-text span when
  the assistant answers the **Black** option, minus the same for the **other
  named** option (split-half cosine 0.98). A *group* mean-difference direction
  (CrowS-Pairs contrast pairs) is the baseline construction (§4, §6); how the
  direction is built is itself a variable we study (E2).

`v̂` points **toward** the behavior; steering in the `+v̂` direction increases it.

### 2.4 The attribute level `a = ⟨h, v̂⟩`

`a` is the scalar projection of the activation onto the unit behavior
direction — one number saying *how much of the behavior this activation
carries* (`a > 0` present, `a < 0` opposite, `|a|` strength). In the hook it is
a single dot product per steered position per step,
`a = (h * v̂).sum(dim=-1)`, `O(D)` — negligible next to the forward pass.

**Key identity (why proportional control is exact).** Because `‖v̂‖ = 1`, adding
`e·v̂` moves the projection by exactly `e`:
```
a' = ⟨h + e·v̂, v̂⟩ = a + e·‖v̂‖² = a + e .
```
Choosing `e = c* − a` therefore sets the projection **exactly to `c*`** — until
the model recomputes `h` at the next step, which is why the correction is
re-applied every step. This identity is what makes "hold `a` at `c*`"
well-defined, and is why the direction is normalized.

### 2.5 The three interventions (applied at every steered position, every step)

**(a) Open-loop / additive** (standard steering, incl. CAA) — a fixed, blind,
feedforward push on the raw direction:
```
h  ←  h + α·v .
```
`α` is set once; the intervention never observes `a`. The projection is nudged
each step, the denoiser pulls it back, and the push is never adjusted.

**(b) Closed-loop proportional — `clamp` (P)** — measure, then correct to the
setpoint:
```
a = ⟨h, v̂⟩ ,     e = c* − a ,     u = K_p·e ,     h  ←  h + u·v̂
                                                   ⇒   projection becomes c* (K_p = 1) .
```
Self-correcting: if the model reverts (`a` drops), `e` grows, so the applied
correction grows — the level is **held** at `c*`.

**(c) cmom — proportional control with an error-smoothed correction** — instead
of applying the raw per-step error, apply an exponentially-weighted moving
average (EMA) of it over the denoising steps of one generation, reset at each
new sequence:
```
u_t = β·u_{t−1} + (1−β)·e_t ,        h  ←  h + u_t·v̂ .
```
`β = 0` recovers clamp (P). **What cmom is, precisely.** The EMA is a first-order
low-pass filter / *leaky integrator* with **unity steady-state gain**: its fixed
point is `u* = e*`, identical to the proportional correction. It is therefore
**not** a classical integral term — a true integral controller has a unit-gain
*pole*, drives the steady-state error to zero, and can overshoot the setpoint;
the EMA does none of these. Its only effect is to **smooth the correction's
transient** across steps (damping step-to-step swings in `e`), not to change
where the loop settles.

**A prediction, not a coincidence.** Because cmom and clamp share the same fixed
point, they must agree once the trajectory settles. This *predicts* the
observation (§3.2) that clamp and cmom are identical on the single-layer bias
task: there is no steady-state room for the smoothing to matter. Any measured
cmom-vs-clamp difference must come from the transient (early denoising steps),
which is where the harder all-layer setting shows a small gap (§3.3). Whether a
*true* integral controller — which would eliminate residual steady-state error —
helps is an open question we leave to a matched-strength P-vs-PI ablation (§6).

### 2.6 Setpoint calibration

`c*` sets the held level. We pick it to **match the projection the best
open-loop run reaches**, so the comparison is at equal strength rather than a
free-parameter advantage. Concretely (bias direction): the natural projection
is `a ≈ −3`, and open-loop `α=8` (on the raw `v`, `‖v‖ ≈ 8.54`) moves it to
`≈ −3 + 8·‖v‖ ≈ +65`; we therefore set `c* = 60`.

## 3. Results

### 3.1 Sentiment control (LLaDA-8B, n=24 prompts; P(negative) / mean length)

| method | control | P(neg) | len |
|---|---|---:|---:|
| none | — | 0.05 | 53 |
| additive (CAA) | open-loop | 0.874 | 51 |
| clamp | P | 0.925 | 50 |
| **cmom** | **P + EMA** | **0.955** | 51 |
| additive + momentum (no feedback) | — | 0.875 | 50 |
| open-loop schedules (decay / warm-up / early-commit / confidence-gate) | open-loop | 0.17–0.50 | — |

What to see: the ordering **cmom > clamp > open-loop** at equal length, and — the
fragility result — every open-loop schedule that reduces the push loses the
attribute (0.17–0.50 vs 0.874 for the constant push). Momentum without feedback
(row 5) adds nothing: the gain comes from *measuring* `a`, not from smoothing.
(Length is only a crude fluency proxy; a proper perplexity/distinct-n check is
in §6.)

### 3.2 Demographic bias injection (BBQ, LLaDA-8B; Black-referent ambiguous items, n=37)

Direction: item-anchored "prefer the Black option". `d_gap = Δblack − Δnonblack`
(directionality vs clean); `acc_disambig` = task competence.

| method | control | site | black | nonblk | abstain | d_gap | acc_disambig |
|--------|---------|------|------:|-------:|--------:|------:|-------------:|
| clean | — | — | 0.162 | 0.081 | 0.757 | +0.000 | 0.970 |
| additive α=4 | open-loop | L14 | 0.243 | 0.108 | 0.649 | +0.054 | 0.969 |
| additive α=8 | open-loop | L14 | 0.568 | 0.351 | 0.081 | +0.135 | 0.874 |
| **clamp** | **P** | **L14** | 0.541 | 0.243 | 0.216 | **+0.216** | 0.907 |
| **cmom** | **P + EMA** | **L14** | 0.541 | 0.243 | 0.216 | **+0.216** | 0.913 |
| additive | open-loop | all 32 | 0.216 | 0.108 | 0.676 | +0.027 | 0.969 |
| clamp | P | all 32 | 0.189 | 0.054 | 0.757 | +0.054 | 0.974 |
| cmom | P + EMA | all 32 | 0.216 | 0.054 | 0.730 | +0.081 | 0.972 |

What to see: closed-loop wins on **both** axes — a larger directional gap
(**+0.216 vs +0.135**) *and* higher competence (**0.91 vs 0.874**) — without
over-collapsing abstention (0.216 kept, vs 0.081 under α=8). And the
probability mass freed from abstention *concentrates on the target group*
(0.541 vs 0.243) instead of splitting nearly evenly as under open-loop α=8
(0.568 vs 0.351): closed-loop aims, open-loop mostly disinhibits. On this task
clamp and cmom are identical — exactly as §2.5 predicts (same fixed point, no
steady-state room for the smoothing), so the win comes from the closed-loop
*hold* itself, not from the EMA. **These magnitudes are on n=37 items and are
indicative, not significant; the directional sign is the reliable part.** Firming
this up is E1 (§6).

### 3.3 Single-layer vs all-layer (is L14 an unfair choice?)

The `all 32` rows apply the same direction at every block (matched total
strength; per-layer setpoint = that layer's natural projection + offset). Two
points. (i) **Single-layer L14 is not a disadvantage — it is the stronger
site.** All-layer steering is much weaker (best gap +0.081 vs +0.216), because
the L14-built direction is only a genuine feature at L14; applying it at the
other 31 blocks dilutes the aim (and increasing all-layer strength to compensate
destroys generation). (ii) **The closed-loop ordering is preserved across
regimes**: all-layer gives cmom (+0.081) ≥ clamp (+0.054) > additive (+0.027).
The small cmom-over-clamp gap here (unlike the single-layer tie) is consistent
with §2.5: the smoothing can only act in the transient, and the harder
all-layer setting is where a transient difference would show. On n=37 this gap
is one to two items and should not be over-read.

### 3.4 Why open-loop fails (supporting analysis)

The steering direction is **noise-level dependent**: rebuilding `v̂` at
different mask ratios, its split-half coherence is ≈0.28 at a mostly-masked
step (mask ratio 0.9) but ≈0.996 at a nearly-clean step, and
`cos(v̂_clean, v̂_masked) = 0.42`. So the behavior is only *linearly encoded*
once the sequence has partly decoded: early in denoising there is no stable
feature for a push to bind to. This *motivates* the disturbance picture — an
un-maintained push has little to hold onto early and is re-contextualized as
positions are recomputed — but it does not yet *prove* that an applied push is
eroded step-by-step. The direct test is a per-step projection-trajectory plot
(open-loop `a_t` decays while closed-loop holds), which is planned, not yet run
(E4, §6); the coherence evidence above is currently on the sentiment direction.

## 4. Related work

**Activation steering (AR LLMs).** Steering adds a fixed contrastive direction
to the residual stream — ActAdd (Turner et al.), CAA (Panickssery/Rimsky et al.
2024), ITI (Li et al. 2023); adaptive variants scale or gate the intervention by
input semantics or hidden-state deviation. All are static and open-loop.

**Control-theoretic steering (concurrent).** Nguyen et al., "Activation Steering
with a Feedback Controller" (arXiv:2510.04309), cast steering as a proportional
controller and add P/I/D terms, running the loop over transformer **layers
(depth)** in **autoregressive** LLMs within one forward pass. This is concurrent
work on the same classical control primitive. We do not build on it; we
instantiate proportional control on a different and DLM-specific loop — the
**denoising trajectory (time)** — where feedback is *necessary* rather than
merely helpful, because open-loop steering is eroded by re-contextualization, a
failure mode absent in the layer-depth setting.

**Inference-time steering of DLMs (the near neighbors).** All are static /
open-loop:
- **Shnaidman et al.** (arXiv:2512.24143) — the **closest method**: a contrastive
  residual-stream direction applied as a global intervention throughout reverse
  diffusion on masked DLMs. Target is **safety refusal**, not demographic bias,
  and it does not measure or hold a setpoint.
- **ILRR** (Avrahami & Nachmani, arXiv:2601.21647) — activation/latent-space
  steering that aligns generated activations toward a **single reference
  sequence** each step; **sentiment**, not bias; a reference-transfer signal, not
  a fixed contrastive direction or a setpoint.
- **DLM-SWAI** (An & Han, arXiv:2605.29626) — **logit-space** token-score
  steering during denoising (a different channel from our residual-stream hook);
  style/safety, not bias.
- **Steering Without Breaking** (Zhou, Roy & Gangadharaiah, arXiv:2605.10971) —
  the **nearest work on the trajectory axis**: SAE-derived residual contrastive
  steering applied on an **adaptive per-denoising-step schedule** (attributes
  "commit" at different steps). It varies the intervention across steps by a
  **fixed schedule**; we vary it by **feedback from a measured setpoint**. Its
  results also largely pre-empt trajectory-*timing* as a contribution for
  non-bias attributes, which is why our headline is bias injection and the
  control loop, and E4 is designed to separate schedule from feedback.
- **Adaptive Steering & Remasking** (Lee & Han, arXiv:2605.13043) — a training-free
  **defense** that projects a safety direction *out* of the residual stream in
  **late layers** and remasks; we **add** a direction to **inject**, at a
  mid-block.

**Bias framing and benchmark.** In the taxonomy of Gallegos et al.
(arXiv:2309.00770), our method is an **intra-processing** intervention (a
training-free change to a fixed model) **inverted to injection** rather than
mitigation, measured at the **generated-text** level. We evaluate on **BBQ**
(Parrish et al., arXiv:2110.08193). Input-space prompt-injection attacks such as
**Ghostwriter** (Yang et al., arXiv:2606.06244) operate on a different channel
(input vs activation) and a different threat model, so we do not treat them as
steering baselines and do not compare against them head-to-head.

**DLM decoding confounds.** Masked-diffusion decoding can move answer
distributions on its own, independent of our injection: proximity bias and
initial-trajectory anchoring (Kim et al., arXiv:2604.10567), mask-block
distraction (Piskorz et al., arXiv:2511.21338), and few-step sampler
incorrectness even under an oracle denoiser (Tang et al., arXiv:2602.19619).
These motivate the confound controls in E7-lite (matched decoding schedule and
answer-position randomization).

## 5. Contributions

Feedback control of activations was proposed concurrently for autoregressive
LLMs over the *layer* axis (Nguyen et al., 2026); proportional/integral control
itself is classical. We do not claim the control framing. Our contributions are
what the **diffusion** architecture changes:

1. **Closed-loop steering over the *denoising trajectory*** — a proportional
   controller (clamp) and an error-smoothed variant (cmom) that re-measure and
   re-correct the *same positions* across denoising steps before they commit.
   This is a distinct control loop from the AR/layer-depth formulation: a real
   temporal feedback loop that only exists because DLMs revisit positions,
   amplified by bidirectional attention.
2. **Open-loop steering is fragile in DLMs, closed-loop holds** — a
   *DLM-specific* failure mode: fixed or reduced-over-time pushes lose the
   behavior. We give a supporting mechanism (directions are noise-dependent and
   only coherent at low noise) and specify the direct per-step test (E4).
3. **More precise bias injection (headline application)** — closed-loop control
   injects a targeted demographic bias on BBQ with a larger directional gap
   (+0.216 vs +0.135) and *less* competence loss (0.91 vs 0.874) than standard
   additive steering; the effect is present at single-layer and all-layer,
   though the n=37 magnitudes are indicative pending E1.
4. **A control-theoretic account of the P-vs-smoothing result** — clamp and cmom
   share a fixed point (the EMA has unity steady-state gain), which *predicts*
   their observed single-layer equivalence and localizes any difference to the
   transient. This companion analysis pairs with the construction finding — an
   item-anchored answer-text direction **aims** (d_gap +0.135) while a group
   mean-difference direction only **disinhibits** (peak +0.05), with split-half
   coherence 0.98 vs 0.27 — which the closed-loop method is layered on top of.

## 6. Planned Experiments, Baselines, Benchmarks & Analysis

The results above are directional and preliminary (n=37 bias items, one model,
one layer, one setpoint). This section lays out the evaluation that would firm
them up. It follows standard control- and steering-evaluation practice —
matched-strength comparisons, gain sweeps, per-step trajectory analysis — in the
bias-injection setting. Two upstream data/scoring bugs were fixed before this
plan: (a) CrowS-Pairs group directions had ~14.5% of pairs (the `antistereo`
rows) oriented backwards; corrected, so the **group mean-difference baseline can
be rebuilt cleanly**; (b) BBQ target-group matching failed for several
categories (canonical group names vs `answer_info` tags), so target groups now
resolve across all categories. The confirmatory large-n runs below are **planned,
not yet run**.

### 6.1 Baselines

Grouped by intervention channel.

| group | baseline | what it is | our contrast |
|---|---|---|---|
| **Open-loop / static (feedforward)** | Constant additive, CAA-style | fixed `α·v` every step — our open-loop | the feedforward baseline; closed-loop holds what this erodes |
| | ActAdd | activation addition at a chosen layer | static, no feedback |
| | Group mean-difference direction | CrowS-Pairs contrast pairs (`race_color`, CrowS-only; a 792-pair CrowS+StereoSet `race_black` variant also exists) | tests aim-vs-disinhibit: expected non-directional |
| **DLM-specific prior steering** | Shnaidman (arXiv:2512.24143) | contrastive direction, global residual intervention over denoising | closest primitive; static, no per-step feedback |
| | ILRR (arXiv:2601.21647) | aligns activations to one reference sequence each step | per-step but reference-transfer, not setpoint-hold |
| | Steering Without Breaking (arXiv:2605.10971) | schedule-varied residual contrastive steering | **nearest "vary across denoising steps" work — schedule vs our feedback; head-to-head in E4** |
| | DLM-SWAI (arXiv:2605.29626) | logit-space token-score steering | different channel (logit vs residual) |
| **Control-theoretic (concurrent)** | Feedback-controller steering (arXiv:2510.04309) | P/I/D loop over **layers (depth)**, AR LLMs | same primitive, different loop; ours is over **denoising time** |

All baselines are **steering methods** (activation- or logit-space), so the
comparison holds the channel and threat model fixed and isolates the control
structure. Input-space prompt-injection attacks (e.g. Ghostwriter) are a
different channel and are out of scope as steering baselines (§4).

### 6.2 Benchmarks & datasets

We evaluate the injected bias on QA benchmarks; the primary comparison is on BBQ.

| benchmark | measures / structure | abstain? | directional? | role (phase) | status |
|---|---|:--:|:--:|---|---|
| **BBQ** (Parrish et al., arXiv:2110.08193) | 11 categories, ambiguous vs disambiguated MCQ; official `s_DIS`/`s_AMB` + our attack metrics (abstention-collapse, target-rate, flip-rate, directional gap) | ✓ | ✓ | **core** — all of Phase 1 | cached / in use |
| **Full Race_ethnicity Black-referent ambiguous set** | subset derived from the 6880-item BBQ cache, disjoint from the 400 anchored-build items | ✓ | ✓ | E1 measurement-fix — turn n=37 into large-n with CIs | derivable from cache |
| **UNQOVER** | directional stereotype probe, no abstention option | ✗ | ✓ | Phase 2 — directional replication + positional-confound rigor | to download |
| **BBG / Open-BBQ** | free-generation bias (MCQ understates bias; most diffusion-native) | ~ | ✓ | Phase 2 — free-generation test | to download |

**Direction sources (not evaluation benchmarks).** **CrowS-Pairs** and
**StereoSet** are used only to *build* the group mean-difference steering vectors;
we do not evaluate on them. The item-anchored answer-text direction is built from
held-out BBQ items (§2.3).

Benchmark selection criterion: the aim-vs-disinhibit metric needs **an abstention
option plus two named groups**, which is why BBQ is the core; UNQOVER adds
directionality without abstention and BBG adds free generation for Phase-2
breadth.

### 6.3 Experiments — mechanism spine (Phase 1: LLaDA-8B + BBQ)

Numbering follows the project research plan. None of these are run yet.

| # | tests | design | success criterion |
|---|---|---|---|
| **E1** (gate) | significance | rescore on the **full** Race_ethnicity Black-referent ambiguous set (derivable from the 6880-item cache, disjoint from the 400 anchored-build items), multiple seeds, bootstrap + McNemar CIs on `d_gap` | the +0.135/+0.216 gaps stay positive & significant at large n |
| **E2** (heart) | construction → aim vs disinhibit | vary *only* the direction construction: group mean-diff · letter-anchored · **answer-text-anchored** · FairPCA-subspace | answer-text anchoring is the only construction high on the aim axis |
| **E3** | the map + metric | 2-axis (directional gap vs abstention drop) map; define **aiming ratio** = Δgap/Δabstention, over every (construction × strength) | constructions separate cleanly on the aim axis |
| **E7-lite** | confound controls | matched-sampler baseline + answer-position randomization (motivated by Kim / Piskorz / Tang); also a generated-only vs all-position steering ablation for BBQ | directional gap survives fixed decoding schedule and position randomization |
| **E4-lite** | 2nd category | winning construction on gender *or* religion, item-anchored | aim is not race-only |
| **E9** | semantic preservation | Concept Coherence Score + the `acc_disambig` competence cliff | injection changes the answer without wrecking semantics |

### 6.4 Experiments — replication matrix (Phase 2: 3 benchmarks × 3 DLMs)

| # | design |
|---|---|
| **E5** | port the winning construction to **Dream-v0-Instruct-7B** on BBQ (recalibrate L≈13 and strength) |
| **E6** | **UNQOVER** on all three models (directional replication + positional-confound rigor) |
| **E8** | **BBG / Open-BBQ** free-generation eval (MCQ understates bias; not present, marked to download) |
| **E10** | Dream-v0-Base-7B alignment-vs-injectability ablation (provisional) |

### 6.5 Experiments — closed-loop ("Steer and Hold") method analyses

Distinct from the aim-vs-disinhibit spine; these probe the controller itself.

- **P vs true-PI at matched effective strength.** cmom (EMA) shares clamp's
  fixed point, so the headline cannot separate them. Add a **true integral**
  controller (unit-gain pole) and compare P vs PI at *equal mean applied push*,
  to answer whether integral action — eliminating residual steady-state error —
  helps or overshoots. Resolves the open question of §2.5.
- **Setpoint `c*` sweep.** Sensitivity of `d_gap` and competence to `c*`; the
  current single `c*=60` is calibrated to open-loop α=8, and a reviewer will
  expect the curve.
- **Per-step projection-trajectory (E4).** Log `a_t` across the `T` steps under
  open-loop vs closed-loop: open-loop `a_t` should decay after each push while
  closed-loop holds at `c*`. Turns the erosion mechanism from asserted (§3.4) to
  demonstrated, and — because Steering Without Breaking varies interventions by a
  fixed *schedule* — isolates **feedback** from scheduling.
- **Gain / `β` robustness.** Sensitivity to `K_p` and the EMA `β`. A true
  integral (not the EMA) is the variant that could overshoot; we check for it.

### 6.6 Metrics

| metric | definition | role |
|---|---|---|
| `d_gap` | `Δblack − Δnonblack` vs clean | primary directionality |
| `s_DIS` | official BBQ disambiguated score `2·(n_biased/n_non-unknown) − 1` (matches our implementation) | comparability to BBQ literature |
| `s_AMB` | as computed here, `(1 − acc_ambig)·s_DIS`; **caveat:** this reuses the disambiguated leaning and must be validated against the official BBQ reference before we call it official | ambiguous-context bias |
| abstain / target / nontarget pick-rates | on a **matched population** (state which: n=37 Black-referent vs all-ambiguous) | aim-vs-disinhibit decomposition |
| `acc_disambig` | disambiguated accuracy on the same population | task competence |
| fluency / perplexity, distinct-n | generation quality on the generated span | collateral-damage check (beyond mean length) |
| split-half coherence | cosine of `v̂` across data halves | direction quality / aim (E2) |

### 6.7 Ethics & dual use

This is a bias-**injection** paper. The motivation is **red-teaming and
auditing**: quantifying how precisely a frozen DLM's demographic behavior can be
pushed at inference exposes an attack surface that deployers and auditors need to
know exists, and gives a measurable target for defenses (e.g. the project-out
defenses in the DLM-safety literature). We acknowledge the **dual-use** risk
directly — the same controller that audits a model can bias it — and therefore
release **no injected model artifacts, no tuned attack directions, and no
ready-to-run injection prompts**; only the method, metrics, and analysis needed
to reproduce the measurements and build mitigations.

## 7. Limitations

Preliminary scope: one model (LLaDA-8B-Instruct), one layer (`L=14`); sentiment
at n=24 prompts, bias at n=37 Black-referent ambiguous items (magnitudes are
indicative, directional sign is reliable); a single setpoint `c*` per task; a
single smoothing coefficient `β`. cmom's EMA is a leaky integrator with unity
steady-state gain, not a classical integral term, so it cannot eliminate
steady-state error — it ties clamp on the single-layer bias task and helps only
slightly in the transient-dominated all-layer setting; whether a *true* integral
controller helps is untested. All BBQ numbers use all-position steering; a
generated-only ablation is future work. Firming up requires the full
Black-referent set (E1), a `c*` sweep, the P-vs-true-PI ablation, and a second
attribute/model.
