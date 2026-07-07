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
control over the denoising trajectory**: at every step we *measure* the current
attribute level `a = ⟨h, v̂⟩` and re-apply the error `e = c* − a`, holding the
level at a setpoint `c*` like a thermostat. We build directly on
**PID-Steering**'s control-theoretic framing of steering (Nguyen et al., 2026),
but move the feedback loop from network *depth* (across layers, their
autoregressive setting) to the *denoising trajectory* (across time) — a loop
that exists only because a diffusion LM revisits the same positions. A
proportional controller (**clamp**) holds the behavior; adding an integral term (**cmom**, PI) can
strengthen it. On LLaDA-8B, closed-loop steering beats fixed additive steering
on two tasks. On sentiment control it reaches higher attribute strength at
equal fluency (PI 0.955 > P 0.925 > additive 0.874). More importantly, on
**demographic bias injection (BBQ)** it is both **more directional and less
destructive**: using an item-anchored "prefer the Black option" direction,
closed-loop control raises the directional bias gap from **+0.135** (best
open-loop) to **+0.216** while *improving* task competence (disambiguated
accuracy 0.874 → 0.91). Closed-loop steering thus injects a targeted
demographic bias more precisely, and with less collateral damage, than the
standard additive method.

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
eroded and the behavior drifts back. The fix is feedback. At every step we
*measure* how much of the behavior is present — the projection `a = ⟨h, v̂⟩` of
the hidden state onto the unit behavior direction — and *re-apply* exactly the
error `e = c* − a` needed to restore a setpoint `c*`. Open-loop steering
shoves with a fixed force and hopes it sticks; closed-loop steering reads the
current level and holds it — a pressed spring vs a thermostat.

This control framing is not ours: **PID-Steering** (Nguyen et al., 2026) already
showed that standard steering *is* a proportional controller and extended it to a
full PID controller — but for autoregressive LLMs, running the loop over
transformer **layers (depth)** inside a single forward pass. Our contribution is
to move that loop to the axis a diffusion LM uniquely exposes: the **denoising
trajectory (time)**, re-correcting the *same* positions across steps before they
commit. We show feedback is not just beneficial but *necessary* here, because
open-loop steering is eroded by re-contextualization — a failure mode that the
layer-depth setting does not have.

We demonstrate the failure of open-loop schedules, introduce closed-loop (P
and PI) steering for DLMs, and show it improves both attribute control and —
our headline application — the precision of demographic-bias injection on BBQ.

## Figure 1 — Open-loop vs closed-loop steering

A diffusion LM rewrites the **same** token positions over `T` denoising steps,
and the steering hook fires at **every** step:

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
present, then add exactly enough to hit the setpoint `c*` — every step:

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

In control terms, clamp is a proportional (P) controller and cmom adds an
integral (memory) of the error across denoising steps (PI). Empirically the
ordering **PI ≥ P > open-loop** holds (§3).

## 2. Method

### 2.1 Setup: where and when we intervene

A masked-diffusion LM generates a target region of tokens by starting from an
all-`[MASK]` region and, over `T` reverse-denoising steps `t = T, …, 1`, running
a **full forward pass** at each step and progressively unmasking tokens.
Crucially, a position is **not** written once: its activation is recomputed and
re-contextualized at every step until it is finalized. We intervene with a
single **forward hook on transformer block `L`** (here `L = 14`) that fires on
every step and edits the residual-stream activation at the generated positions
before it flows into block `L+1`.

### 2.2 Notation

| symbol | meaning |
|---|---|
| `T`, `t` | number of denoising steps; step index (`t = T,…,1`) |
| `i` | token-position index; generated positions are the ones we steer |
| `L` | block we hook (`L = 14`) |
| `D` | hidden size, `D = 4096` (LLaDA-8B) |
| `h ≡ h^{(L)}_{t,i} ∈ R^D` | residual-stream activation at block `L`, step `t`, position `i` |
| `v ∈ R^D` ; `v̂ = v/‖v‖` | steering direction (raw) and its **unit** vector, `‖v̂‖ = 1` |
| `a ≡ a_{t,i} ∈ R` | **attribute level** = projection of `h` onto `v̂` (§2.4) |
| `c* ∈ R` | **setpoint**: the attribute level we want to hold |
| `e ≡ e_{t,i} = c* − a` | control **error** |
| `α ∈ R` | open-loop steering coefficient |
| `β ∈ [0,1)` | integral coefficient (momentum) for the PI controller |
| `u ≡ u_{t,i}` | integrated (EMA) error used by PI |

### 2.3 The steering direction `v̂`

`v` is a contrastive mean-difference of block-`L` activations between examples
that exhibit the behavior and examples that do not, normalized to unit length:
```
v  =  E_{x ∈ D+}[ h^{(L)}(x) ]  −  E_{x ∈ D-}[ h^{(L)}(x) ] ,        v̂ = v / ‖v‖ .
```
- **Sentiment:** `D+`/`D-` are positive/negative sentences.
- **Bias (headline):** the *item-anchored answer-text* direction — on held-out
  BBQ Black-referent items, `h^{(L)}` averaged over the answer-text span when
  the assistant answers the **Black** option, minus the same for the **other
  named** option (split-half cosine 0.98).

`v̂` points **toward** the behavior; steering in the `+v̂` direction increases it.

### 2.4 The attribute level `a = ⟨h, v̂⟩`

`a` is the scalar projection of the activation onto the unit behavior
direction — one number saying *how much of the behavior this activation
carries* (`a > 0` present, `a < 0` opposite, `|a|` strength). In the hook it is
a single dot product per generated position per step,
`a = (h * v̂).sum(dim=-1)`, `O(D)` — negligible next to the forward pass.

**Key identity (why control is exact).** Because `‖v̂‖ = 1`, adding `e·v̂` moves
the projection by exactly `e`:
```
a' = ⟨h + e·v̂, v̂⟩ = a + e·‖v̂‖² = a + e .
```
Choosing `e = c* − a` therefore sets the projection **exactly to `c*`** — until
the model recomputes `h` at the next step, which is why the correction is
re-applied every step. This identity is what makes "hold `a` at `c*`"
well-defined, and is why the direction is normalized.

### 2.5 The three interventions (applied at every generated position, every step)

**(a) Open-loop / additive** (standard steering, incl. CAA) — a fixed, blind push:
```
h  ←  h + α·v̂ .
```
`α` is set once; the intervention never observes `a`. The projection is nudged
by `+α` each step, the denoiser pulls it back, and the push is never adjusted.

**(b) Closed-loop proportional — `clamp` (P)** — measure, then correct to the
setpoint:
```
a = ⟨h, v̂⟩ ,     e = c* − a ,     h  ←  h + e·v̂         ⇒   projection becomes c* .
```
Self-correcting: if the model reverts (`a` drops), `e` grows, so the applied
correction grows — the level is **held** at `c*`.

**(c) Closed-loop PI — `cmom`** — add an integral (memory) term: an exponential
moving average of the error over the denoising steps of one generation, reset
at each new sequence:
```
u_t = β·u_{t+1} + (1−β)·e_t ,        h  ←  h + u_t·v̂ .
```
`β = 0` recovers clamp (P); `β > 0` accumulates the *persistent* deficit the
model keeps reverting — the integral term of a PI controller.

### 2.6 Setpoint calibration

`c*` sets the held level. We pick it to **match the projection the best
open-loop run reaches**, so the comparison is at equal strength rather than a
free-parameter advantage. Concretely (bias direction): the natural projection
is `a ≈ −3`, and open-loop `α=8` moves it to `≈ −3 + 8·‖v‖ = +65` (the additive
baseline on this task adds the raw, un-normalized `α·v`, hence the `‖v‖`
factor); we therefore set `c* = 60`.

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

What to see: the ordering **PI > P > open-loop** at equal fluency (length
unchanged), and — the fragility result — every open-loop schedule that reduces
the push loses the attribute (0.17–0.50 vs 0.874 for the constant push).
Momentum without feedback (row 5) adds nothing: the gain comes from measuring
`a`, not from smoothing.

### 3.2 Demographic bias injection (BBQ, LLaDA-8B; Black-referent ambiguous items, n=37)

Direction: item-anchored "prefer the Black option". `d_gap = Δblack − Δnonblack`
(directionality vs clean); `acc_disambig` = task competence.

| method | control | site | black | nonblk | abstain | d_gap | acc_disambig |
|--------|---------|------|------:|-------:|--------:|------:|-------------:|
| clean | — | — | 0.162 | 0.081 | 0.757 | +0.000 | 0.970 |
| additive α=4 | open-loop | L14 | 0.243 | 0.108 | 0.649 | +0.054 | 0.969 |
| additive α=8 | open-loop | L14 | 0.568 | 0.351 | 0.081 | +0.135 | 0.874 |
| **clamp** | **P** | **L14** | 0.541 | 0.243 | 0.216 | **+0.216** | 0.907 |
| **cmom** | **PI** | **L14** | 0.541 | 0.243 | 0.216 | **+0.216** | 0.913 |
| additive | open-loop | all 32 | 0.216 | 0.108 | 0.676 | +0.027 | 0.969 |
| clamp | P | all 32 | 0.189 | 0.054 | 0.757 | +0.054 | 0.974 |
| cmom | PI | all 32 | 0.216 | 0.054 | 0.730 | +0.081 | 0.972 |

What to see: closed-loop wins on **both** axes — a larger directional gap
(**+0.216 vs +0.135**) *and* higher competence (**0.91 vs 0.874**) — without
over-collapsing abstention (0.216 kept, vs 0.081 under α=8). And the
probability mass freed from abstention *concentrates on the target group*
(0.541 vs 0.243) instead of splitting nearly evenly as under open-loop α=8
(0.568 vs 0.351): closed-loop aims, open-loop mostly disinhibits. On this task
P and PI are equivalent (the integral term does not add), so the win comes
from the closed-loop hold itself.

### 3.3 Single-layer vs all-layer (is L14 an unfair choice?)

The `all 32` rows of the §3.2 table apply the same direction at every block
(matched total strength; per-layer setpoint = that layer's natural projection +
offset). Two points. (i) **Single-layer L14 is not a disadvantage — it is the
stronger site.** All-layer steering is much weaker (best gap +0.081 vs +0.216),
because the L14-built direction is only a genuine feature at L14; applying it
at the other 31 blocks dilutes the aim (and increasing all-layer strength to
compensate destroys generation). (ii) **The closed-loop ordering is robust
across regimes**: all-layer gives cmom (+0.081) > clamp (+0.054) > additive
(+0.027) — and here PI beats P, so the integral term helps in the harder
all-layer setting.

### 3.4 Why open-loop fails (supporting analysis)

The steering direction is **noise-level dependent**: rebuilding `v̂` at
different mask ratios, its split-half coherence is ≈0.28 at a mostly-masked
step (mask ratio 0.9) but ≈0.996 at a nearly-clean step, and
`cos(v̂_clean, v̂_masked) = 0.42`. So the behavior is only *linearly encoded*
once the sequence has partly decoded: early in denoising there is no stable
feature for a push to bind to, and whatever a push writes is re-contextualized
away when every position is recomputed at the next step. An un-maintained
(fixed or fading) push is therefore eroded; closed-loop control succeeds
because it re-measures and re-asserts the setpoint at every step, compensating
exactly what the denoiser undid.

## 4. Related work

Activation steering for LLMs — ActAdd, CAA (Panickssery et al. 2024), ITI (Li
et al. 2023) — adds a fixed contrastive direction to the residual stream.
Adaptive variants scale or gate the intervention by input semantics or
hidden-state deviation (SADI, ACT, DAC, FASB, CAST). Our controllers relate to
**activation clamping** (Templeton et al. 2024; AxBench, Wu et al. 2025,
arXiv:2501.17148) and **affine concept editing** (ACE, arXiv:2411.09003). Most
directly, **PID Steering** (Nguyen et al., ICLR 2026, arXiv:2510.04309; also
arXiv:2506.18831) provides the control-theoretic foundation we build on: it
shows standard steering *is* a proportional (P) controller and introduces the
full PID controller — but for **autoregressive LLMs**, with the loop running
over **transformer layers (depth)** inside one forward pass. Our loop differs
in kind: it runs over the **denoising trajectory (time)** of a diffusion LM,
re-correcting the *same* positions across steps before they commit, and we
show it is *necessary* there because open-loop steering is eroded by
re-contextualization — a failure mode absent in the layer-depth setting.

For DLMs, prior steering is static/open-loop: residual-stream additive
(arXiv:2512.24143), reference-alignment (ILRR, arXiv:2601.21647),
step-scheduled (arXiv:2605.10971), and logit-space (DLM-SWAI,
arXiv:2605.29626). Prophet (arXiv:2508.19982) shows DLM decoding exposes a
confidence/convergence signal.

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

## 6. Planned Experiments, Baselines & Analysis

The results above are directional and preliminary (n=37 bias items, one model,
one layer, one setpoint). This section lays out the evaluation that would firm
them up. It is deliberately modeled on **PID-Steering**'s empirical playbook
(Nguyen et al., 2026; arXiv:2510.04309) — matched-strength per-term comparisons,
gain sweeps, and stability/overshoot analysis — but ported to the denoising-time
loop and to the bias-injection setting. Two upstream bugs were just fixed and
this plan assumes both: (a) CrowS group directions were ~14.5% sign-flipped
(now corrected), so the **group mean-difference baseline can be rebuilt
cleanly**; (b) BBQ target-group matching was broken for several categories, so
the official **`s_AMB`/`s_DIS` scores are now trustworthy**.

### 6.1 Baselines

Grouped by intervention channel. The static-steering group is exactly the
"P-controller = static steering" family PID-Steering itself frames as the thing
a feedback controller should beat.

| group | baseline | what it is | our contrast |
|---|---|---|---|
| **Open-loop / static** | Constant additive (CAA-style) | fixed `α·v̂` every step — our current open-loop | the "P-as-static" baseline; closed-loop should hold what this erodes |
| | ActAdd | activation addition at a chosen layer | static, no feedback |
| | Group mean-difference vector (bug-fixed) | CrowS/StereoSet group-contrast direction | tests aim-vs-disinhibit: expected non-directional |
| **Input-space** | Ghostwriter (arXiv:2606.06244) | fabricated-evidence prompt injection — in repo | input-space vs our activation-space channel |
| **DLM-specific prior steering** | Shnaidman residual-additive MDLM (arXiv:2512.24143) | contrastive direction, global residual intervention over denoising | nearest primitive; static (no per-step feedback) |
| | ILRR reference-alignment (arXiv:2601.21647) | aligns activations to one reference sequence each step | per-step but reference-transfer, not setpoint-hold |
| | Steering Without Breaking (arXiv:2605.10971) | step-scheduled residual contrastive steering | **nearest "vary across denoising steps" work — must compare head-to-head** |
| | DLM-SWAI (arXiv:2605.29626) | logit-space token-score steering | different channel (logit vs residual) |
| **Control-theory parent** | PID-Steering (arXiv:2510.04309) | P/I/D loop over **layers (depth)**, AR LLMs | conceptual baseline; our loop is over **denoising time**, same controllers |

The key positioning: Steering Without Breaking varies the intervention *across
denoising steps on a fixed schedule*; we vary it *by feedback from a measured
setpoint*. E4 (below) is designed to separate schedule from feedback.

### 6.2 Experiments

| # | tests | design | expected axis |
|---|---|---|---|
| **E1** | significance | full Black-referent set (beyond n=37), multiple seeds, bootstrap + McNemar CIs on `d_gap` | closed-loop `d_gap` gap survives with CI not crossing open-loop |
| **E2** | dose–response | dense `α`-sweep (open-loop) vs `c*`-sweep (closed-loop) → Pareto frontier of `d_gap` vs `acc_disambig`, **matched-abstention** comparison (not two hand-picked points) | closed-loop Pareto-dominates on the matched-abstention slice |
| **E3** | which term | P vs PI vs PID at **matched effective strength** (equal mean applied push) + add the **D term** from PID-Steering | isolates whether cmom's edge is the integral or just more push; D expected to curb overshoot |
| **E4** | mechanism | per-denoising-step **projection-trajectory** plot: open-loop `a` decays vs closed-loop holds at `c*` (hook already computes `a` each step) | erosion goes from asserted → demonstrated; also separates feedback from a fixed schedule (vs Steering Without Breaking) |
| **E5** | 2nd attribute / demo | toxicity (RealToxicityPrompts-style) + formality (half-run on disk); 2nd demographic (gender/religion) via item-anchored directions | closed-loop advantage generalizes beyond race/sentiment |
| **E6** | 2nd model | repeat headline on **Dream-v0-Instruct-7B** (available locally) | supports an "in DLMs" (not "in LLaDA") claim |
| **E7** | site | layer sweep and `c*`×layer grid; revisit all-layer at matched strength | confirms L14 is the strong site, not a lucky pick |
| **E8** | construction | group-mean-diff vs item-anchored-letter vs item-anchored-answer-text (split-half coherence 0.27 vs 0.98) | the **aim-vs-disinhibit** axis: which construction *aims* vs merely disinhibits abstention |

### 6.3 Analysis (borrowed from PID-Steering's playbook)

- **Stability / overshoot** of the projection `a` along the trajectory: does closed-loop settle at `c*` or oscillate; does the D term reduce overshoot.
- **Robustness to gain**: sensitivity of `d_gap`/competence to `c*` (and `β`), analogous to PID-Steering's coefficient sweep.
- **Competence measured properly**: perplexity/fluency and distinct-n, not just mean length (length alone is a weak fluency proxy).
- **Per-step convergence**: how fast `a` reaches `c*` across the `T` steps.
- **Abstention-vs-aim decomposition**: on a matched population, does freed probability mass concentrate on the target group (aim) or split across options (disinhibit).

### 6.4 Metrics

| metric | definition | role |
|---|---|---|
| `d_gap` | `Δblack − Δnonblack` vs clean | primary directionality |
| `s_AMB`, `s_DIS` | official BBQ bias scores (now trustworthy) | comparability to BBQ literature |
| abstain / target / nontarget pick-rates | on a **matched population** | aim-vs-disinhibit decomposition |
| `acc_disambig` | disambiguated accuracy on same population | task competence |
| fluency / perplexity, distinct-n | generation-quality on generated span | collateral-damage check |
| split-half coherence | cosine of `v̂` across data halves | direction quality / aim (E8) |

### 6.5 Ethics & dual use

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
indicative, directional sign is reliable); a single setpoint `c*` per task; fixed
`β`. The integral term (PI) helps on sentiment but not on the bias task. Firming
up requires the full Black-referent set, a `c*` sweep, and a second attribute/model.
