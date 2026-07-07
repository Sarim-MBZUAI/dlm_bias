# Steer and Hold: Closed-Loop Activation Steering for Diffusion Language Models

*Working draft — results are preliminary (LLaDA-8B, sentiment, n=24 prompts).*

## Abstract

Activation steering controls a language model at inference by adding a fixed
direction to its hidden states. In **diffusion language models (DLMs)**, which
generate by iteratively denoising a masked sequence over many steps, the same
positions are revisited and re-contextualized at every step — so a steering
signal is not applied once but repeatedly, against a moving representation. We
show that this changes what steering *should* be. First, **open-loop schedules
fail**: fading the push as the sequence "converges" (decay), ramping it late
(warm-up), or gating it by a Prophet-style confidence signal all lose the target
attribute, because the attribute is a *sustained force* that the denoising
process erodes — remove the push and the model reverts. Second, we recast
steering as **closed-loop control** over the denoising trajectory: rather than
add a fixed vector, we **measure the current attribute level and correct it to a
setpoint every step**. A proportional controller (*clamp*: hold the projection
`⟨h,v̂⟩` at a target) already beats fixed addition, and adding an integral term
(*clamp+momentum*, a PI controller) beats it further. On LLaDA-8B sentiment
steering, PI (0.955) > proportional (0.925) > fixed addition / CAA (0.874) at
equal fluency, while all open-loop schedules land at 0.17–0.50. To our knowledge
this is the first **closed-loop / clamping** activation-steering method applied
inside a DLM denoising loop. We do **not** claim clamping as a new primitive —
it is adapted from SAE feature-clamping; the contribution is the control-theoretic
framing for DLMs and the empirical open-loop-fails / closed-loop-holds result.

## 1. Setup

DLM (LLaDA-8B-Instruct) generates by unmasking over `T` denoising steps. We
intervene on the **residual stream at block L14** via a forward hook that fires
on every step, on the generated positions. A steering direction `v̂` is the unit
mean-difference of L14 activations between contrastive (attribute-positive vs
-negative) examples. Attribute strength is scored by an external classifier
(distilbert-sst2, `P(negative)`); fluency is tracked by mean output length.

## 2. Method

Let `h` be a generated position's residual at L14, `a = ⟨h, v̂⟩` the current
attribute level, `λ₀` a strength, `c*` a target level.

| name | rule | control view |
|---|---|---|
| additive (CAA) | `h ← h + λ₀ v̂` | open-loop |
| **clamp** | `h ← h + (c* − a) v̂` | **proportional (P)** — correct the deficit each step |
| **clamp+momentum** | `v_t = βv_{t-1} + (1−β)(c*−a);  h ← h + v_t v̂` | **PI** — integral memory of the persistent deficit |

The controllers are **adaptive on the target itself** (`a` is measured every
step), not on a proxy for "doneness." This matters because the DLM re-derives
`h` at each denoising step, so a proportional/integral controller re-asserts the
setpoint against the model's reversion.

**What did not work (open-loop / proxy-signal):** fading the push as tokens
stabilize (*decay*), ramping it late (*warm-up*), early-committing once confident
(*conv*), or gating by per-position confidence (*clampconf*) — all lose the
attribute, because reducing the push anywhere lets the denoiser revert it. A
norm-preserving/rotation operator and SADI-style dim-masking also underperformed
fixed addition. Momentum on *plain* addition (no feedback) merely ties addition —
it is the **feedback + integral** combination that wins.

**Supporting finding (steering emerges across denoising):** the attribute
direction is *noise-level-dependent* — its split-half coherence rises from ≈0.28
at a mostly-masked step to ≈0.996 at a nearly-clean step, and `cos(v̂_clean,
v̂_masked)=0.42`. The attribute is only linearly encoded once the sequence has
partly decoded, which is why the coherent (low-noise) direction, applied every
step under feedback, is what works.

## 3. Preliminary results (LLaDA-8B, sentiment→negative, n=24, L14, 64 steps)

| condition | mechanism | P(neg) | len |
|---|---|---:|---:|
| none | — | 0.048 | 53 |
| additive (CAA) | open-loop push | 0.874 | 51 |
| clamp | P control | 0.925 | 50 |
| **clamp+momentum** | **PI control** | **0.955** | 51 |
| additive+momentum | momentum, no feedback | 0.875 | 50 |
| decay / warm-up / conv / clampconf | open-loop / proxy-gated | 0.17–0.50 | — |

Ordering **PI > P > open-loop** matches control theory. Caveats: single attribute,
n=24, modest PI-vs-P margin; a P-vs-PI ablation (is it the integral term or just
effective strength?) and a second attribute (toxicity) are needed to confirm.

## 4. Related work (and what we borrow)

- **Clamping (borrowed, not novel):** SAE feature-clamping — pin a feature to a
  target value — Templeton et al., *Scaling Monosemanticity* (2024); benchmarked
  in **AxBench** (Wu et al., 2025, [arXiv:2501.17148](https://arxiv.org/abs/2501.17148)).
  We adapt it from an SAE feature to a raw contrastive direction.
- **Affine steering:** *Affine Concept Editing / ACE* (Marshall, Scherlis,
  Belrose, 2024, [arXiv:2411.09003](https://arxiv.org/abs/2411.09003)) — steering
  as an affine map with a baseline offset; clamp-to-setpoint is a special case.
- **Closed-loop / PID steering (autoregressive):** *Adaptive Activation Steering
  via Closed-Loop PID* ([arXiv:2506.18831](https://arxiv.org/abs/2506.18831));
  feedback-controller steering ([arXiv:2510.04309](https://arxiv.org/abs/2510.04309)).
  We bring P/PI control into the DLM denoising loop.
- **DLM steering (all static / open-loop):** Activation Steering for MDLMs
  ([arXiv:2512.24143](https://arxiv.org/abs/2512.24143)); ILRR
  ([arXiv:2601.21647](https://arxiv.org/abs/2601.21647)); Steering Without Breaking
  ([arXiv:2605.10971](https://arxiv.org/abs/2605.10971)); DLM-SWAI (logit-space,
  [arXiv:2605.29626](https://arxiv.org/abs/2605.29626)).
- **Confidence signal:** Prophet ([arXiv:2508.19982](https://arxiv.org/abs/2508.19982))
  — DLM answers converge early; we tested its gap as a steering gate and found it
  *fails* as a gate (proxy for doneness ≠ attribute presence).

## 5. Contributions

1. **First closed-loop / clamping activation steering inside a DLM denoising
   loop** (clamping and PID steering exist only for SAE features / AR models).
2. **Open-loop-fails / closed-loop-holds** finding for DLMs, with a mechanism:
   steering is a sustained force eroded by iterative re-contextualization.
3. A **control-theoretic ordering** (PI > P > open-loop) borne out empirically.
4. Supporting analysis: the steering direction **emerges across the denoising
   trajectory** (coherence 0.28→0.996).

## 6. Limitations

Preliminary: one model (LLaDA-8B), one attribute (sentiment), n=24, single layer
(L14), fixed `β`, hand-set `c*` (natural target was mis-scaled → no-op; required
calibration). No efficiency claim (all conditions run full steps; "fewer steps"
is Prophet's territory). The PI margin is modest and needs firming (more prompts,
2nd attribute, P-vs-PI ablation).
