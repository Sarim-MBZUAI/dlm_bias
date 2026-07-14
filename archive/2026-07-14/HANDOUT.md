# Bias Injection in a Frozen Diffusion LM — Session Handout

**Model:** LLaDA-8B-Instruct (masked-diffusion LM, 32 blocks, H=4096), weights frozen.
**Setting:** training-free activation steering to *inject* demographic bias on BBQ (red-team / mechanism study).

## One-line thesis (survives all experiments)
> **What injects *directional* bias is HOW you build the steering direction and WHERE you inject it — not how cleverly you control the push.** Construction + site are the mechanism; actuation is just "add it, don't overdo it."

---

## The algorithm that works

```
 A) BUILD THE ARROW  (offline, once)
    at block L14, for ~400 held-out BBQ items:
      h_black  = activation while "looking at" the Black answer
      h_other  = activation while "looking at" the other answer
    arrow v = average over items of (h_black - h_other)     # answer-text anchored CAA
    → one fixed vector, split-half cosine 0.98 (highly coherent = it "aims")

 B) USE THE ARROW  (at answer time)
    at EVERY denoising step, at block L14:
      hidden += alpha * v          # alpha ≈ 8, raw vector
    → model stops abstaining and preferentially picks the Black option
```

Fixed arrow, constant push, no feedback. That's it.

---

## What works vs what's dead (all evidenced this session)

| | Approach | Result |
|---|---|---|
| ✅ | **answer-text anchored CAA @ L14, open-loop α≈8** | **AIMS** — d_gap **+0.15** (n=400) |
| ✅ | dose-response has a clean band 0→α≈8 | monotonic; peak target-pick 0.69 |
| ❌ | group mean-diff direction | only DISINHIBITS (peak +0.05) |
| ❌ | all-layer / embedding-layer injection | too weak / non-directional |
| ❌ | α > 8 | REVERSES — competence cliff, nontarget overtakes |
| ❌ | closed-loop on the residual projection (clamp/cmom) | no cross-step memory → clamp ≡ cmom |
| ❌ | closed-loop on the decode (PI / deadline / MPC / remask) | no headroom; PI *degrades*; "wins" are static timing, not feedback |
| ❌ | **PID-Steering's layer-depth method (2510.04309)** | d_gap **+0.02** vs our +0.15; I/D add nothing; disinhibits |

---

## Key numbers

**Clean baseline (n=400 Black-referent ambiguous):** black 0.12 · nonblack 0.10 · abstain 0.78 — essentially unbiased.

**L14 open-loop α=8 (the working attack):** black **0.515** · nonblack 0.343 · abstain 0.14 · **d_gap +0.152**. (At n=1600 the directional gap is +0.089; both positive and aimed.)

**Layer-depth PID (the paper's method), n=400:**

| condition | black | nonblack | abstain | d_gap |
|---|---|---|---|---|
| L14 open-loop α=8 (ours) | 0.515 | 0.343 | 0.14 | **+0.152** |
| layerwise P | 0.188 | 0.142 | 0.67 | +0.025 |
| layerwise PI | 0.210 | 0.170 | 0.62 | +0.020 |
| layerwise PID | 0.205 | 0.168 | 0.63 | +0.017 |

---

## Two closed-loop dead-ends (with the *reason*, not just the result)

1. **Denoising-step axis:** the residual-stream projection has **no memory across steps** (each step recomputes it from the token sequence), so feedback has nothing to integrate — clamp ≡ cmom to <0.04 over 64 steps, and every controller reduces to (or underperforms) open-loop. The one "efficiency win" (match open-loop at half the dose) is a **static schedule** — steer steps 0–31, off after — needing no sensing.
2. **Layer-depth axis (PID-Steering):** the residual stream *does* integrate over depth, so the method is well-posed — but on our bias task the per-layer arrows only **disinhibit**, the integral/derivative terms add nothing, and it aims ~7× weaker than one strong L14 push.

---

## Paper framing (what to claim)

A **mechanism / red-team paper**, not a control-theory paper:
1. **Construction** — answer-text anchored CAA *aims*; group mean-diff only *disinhibits* (aim-vs-disinhibit).
2. **Site** — L14 aims; all-layer / embedding don't.
3. **Characterized open-loop attack** — clean dose-response with a hard competence ceiling at α≈8.
4. **Negative result on control** — feedback doesn't help on *either* the denoising-step axis (no memory) or the layer-depth axis (no headroom / only disinhibits). Cleanly differentiates from PID-Steering (2510.04309), which we tested directly.

## Next step
**E2 construction ablation at scale** (group mean-diff / letter-anchored / answer-text / FairPCA, head-to-head on aim-vs-disinhibit, n=1600 + CIs) — the result that *makes* the paper.

---
*Artifacts & figures: `directional_steering/diagnostics/` (fig3, pB_ptarget_accum, pB_alpha_response, pB_controllers, pid_layerwise). Local handout — not committed. Dual-use: contains method + aggregate metrics only, no direction vectors/prompts.*
