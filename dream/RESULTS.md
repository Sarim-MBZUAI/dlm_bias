# Dream-v0-Instruct-7B — decode-space PID + baselines (BBQ-400)

Second-model check of the project's contribution (**steering in the denoising
space** — feedback PID over the diffusion step) plus the faithful baselines, run
on **Dream-v0-Instruct-7B** (a masked-diffusion LM, 28 layers × 3584, native
`diffusion_generate`, `alg="entropy"`). Same 400 Black-referent ambiguous BBQ
items (`data/bbq_items/_sweep400.jsonl`), same harness/metric as LLaDA.

`d_gap = ΔBlack − Δnon-Black` vs the clean Dream base (black 0.225 / non-Black
0.195); higher = aims at the Black person. `unparse` = fraction of outputs that
did not yield a parseable A/B/C letter — the **coherence** gauge.

## Headline

1. **Our decode-space PID transfers to a second diffusion LM — but the actuation
   magnitude is model-specific.** At LLaDA's setting (`amax=6`) it drives Dream to
   **97.5 % unparseable** (incoherent). Swept down, it recovers full coherence and
   a real Black-ward aim, **peaking at `amax≈1.0` with d_gap +0.055** — the
   strongest aim among all *coherent* methods on Dream.
2. **Dream is far more perturbation-sensitive than LLaDA.** Every all-28-block /
   high-gain configuration tuned on LLaDA collapses here (normal α4, Mean-AcT s2,
   ITI-C, decode `amax=6`). Single-layer methods (CAA, ActAdd) stay coherent.
3. **Clean dose-response with a clear optimum** (below): aim rises 0.25→1.0 then
   *inverts* by 1.5 (over-driving broadly disinhibits — non-Black rises faster
   than Black), then collapses to garbage by 6.

## Our method — decode-PI dose-response (all coherent except amax=6)

| amax | black | non-Black | abstain | unparse | d_gap |
|---:|---:|---:|---:|---:|:--:|
| 0.25 | 0.212 | 0.190 | 0.598 | 0.000 | −0.008 |
| 0.5  | 0.220 | 0.163 | 0.618 | 0.000 | +0.027 |
| **1.0** | **0.258** | **0.172** | 0.570 | 0.000 | **+0.055** |
| 1.5  | 0.367 | 0.385 | 0.247 | 0.000 | −0.048 |
| 6 (LLaDA) | 0.025 | 0.000 | 0.000 | **0.975** | — (collapsed) |

## Ours vs baselines — coherent methods, ranked by aim

| method | who | family | black | non-Black | d_gap | unparse |
|---|---|---|---:|---:|:--:|---:|
| **decode-PI amax1.0** | **OURS** | denoise-step feedback | 0.258 | 0.172 | **+0.055** | 0.000 |
| CAA (L14, mult 2) | baseline | single-layer vector | 0.253 | 0.175 | +0.048 | 0.000 |
| ActAdd (L14, α8) | baseline | single-pair vector | 0.240 | 0.170 | +0.040 | 0.000 |
| AURA inject (γ4) | baseline | neuron gating | 0.390 | 0.325 | +0.035 | 0.000 |
| decode-PI amax0.5 | OURS | denoise-step feedback | 0.220 | 0.163 | +0.027 | 0.000 |
| CAA (L14, mult 1) | baseline | single-layer vector | 0.235 | 0.193 | +0.012 | 0.000 |
| base (clean) | — | — | 0.225 | 0.195 | +0.000 | 0.000 |
| Linear-AcT gaussian (s2) | baseline | per-neuron OT | 0.347 | 0.340 | −0.023 | 0.000 |
| AURA vanilla (ctrl) | baseline | neuron gating | 0.228 | 0.235 | −0.037 | 0.000 |

- **decode-PI amax1.0 leads every coherent baseline** — same qualitative result as
  LLaDA (feedback over the denoising step is the strongest aimer), at a ~6× smaller
  actuation magnitude. Margins are modest at n=400 (no CIs yet — see caveats).
- **AURA inject** moves the Black rate most (0.225→0.390) but lifts non-Black almost
  as much (disinhibition), so its *aim* is only +0.035.
- **Linear-AcT** raises both rates together (broad disinhibition), near-zero aim.
- **AURA vanilla** (suppression control) is correctly slightly negative.

## Collapsed at LLaDA strengths (excluded from the ranking — incoherent)

| method | strength | unparse | note |
|---|---|---:|---|
| decode-PI / PID | amax 6 | 0.975 | our method at LLaDA's amax — too hot for Dream |
| normal vector | α4, all 28 blocks | 0.960 | open-loop all-block over-steer |
| Mean-AcT (raw) | s2, all 28 blocks | 1.000 | total collapse |
| ITI-C | top-48, α16 | 1.000 | per-head shift over-steer |

decode-**P** at amax6 stayed parseable but drove strongly toward non-Black
(d_gap −0.382) — an anomaly of the un-tuned P-only gain at huge amax; not a
meaningful operating point.

## Method / strength notes

- Direction: `dream/arrows.pt` — 28-layer answer-anchored diff-in-means, built from
  the **same** contamination-safe held-out BBQ contrast set as LLaDA.
- decode-space PID: `vhat = unit(r[14])` added as `alpha(t)·vhat` to all 28 blocks,
  `alpha(t)` closed-loop over the 64 denoising steps (Kp=3, Ki=0.1, setpoint 0.9),
  via Dream's native `generation_logits_hook_func`. Only `amax` was swept.
- Baseline strengths chosen to match the LLaDA `results/BASELINES.md` faithful picks
  (CAA mult 2, ActAdd α8, AURA γ4, Linear-AcT gaussian s2, ITI-C top-48/α16).
- Reproduce: `dream/run_dream_gpu3_sweep.sh` (ours) + `dream/run_dream_gpu4.sh`
  (baselines), env `sarim_awm` (transformers 4.46.2), GPUs 3 & 4.

## Caveats

- Single run, n=400, temperature 0, **no confidence intervals** — margins between
  the top coherent methods (+0.055 vs +0.048 vs +0.040) are not yet statistically
  separated. Bootstrap CIs + the position-balanced arbiter are the obvious next step
  (as on LLaDA).
- Dream's base **abstains** far more than LLaDA (0.58 "unknown"), so absolute pick
  rates are lower; d_gap (the aim) is the comparable quantity.
- Only decode-**PI** was dose-swept; decode-PID at amax1.0 + a coherent low-α
  layer-space PID / normal point are the immediate follow-ups.
