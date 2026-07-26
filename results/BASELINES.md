# Baselines vs. decode-space PID (ours) — BBQ-400 (bias injection toward "Black")

Layer schemes verified against the source papers (arXiv:2312.06681 CAA, 2308.10248 ActAdd):
**CAA and ActAdd are SINGLE-layer** methods (swept, applied at one mid layer; CAA lands on L13–15 for
Llama, so our L14 is in-band), raw diff-in-means vector, small tuned multiplier (~2; larger breaks
fluency). They are NOT all-layer, and they are NOT the `normal` vector. **Ours is decode-space PID**
(PID feedback over the denoising step).

`d_gap = ΔBlack − Δnon-Black` vs `base` (0.120 / 0.100); higher = aims at Black.
95% CI = 2000-sample bootstrap over the 400 items (base fixed).

| method | who | family / axis | Black | non-Black | d_gap | 95% CI |
|---|---|---|---:|---:|---:|:--:|
| **decode-space PI** | **OURS** | denoise-step feedback | 0.302 | 0.147 | **+0.135** | [+0.068,+0.200] |
| **decode-space PID** | **OURS** | denoise-step feedback | 0.287 | 0.165 | +0.102 | [+0.038,+0.165] |
| Linear-AcT (gaussian) | baseline | per-neuron transport (MLP) | 0.280 | 0.200 | +0.060 | [−0.007,+0.128] |
| normal vector α4 | baseline | diff-in-means at ALL 32 blocks | 0.370 | 0.295 | +0.055 | [−0.022,+0.130] |
| AURA inject γ4 | baseline | neuron gating (MLP) | 0.245 | 0.175 | +0.050 | [−0.010,+0.115] |
| Linear-AcT (empirical) | baseline | per-neuron transport (MLP) | 0.263 | 0.200 | +0.043 | [−0.022,+0.107] |
| layer-space PI | baseline (prior work) | PID over layers | 0.242 | 0.182 | +0.040 | [−0.020,+0.105] |
| layer-space PID | baseline (prior work) | PID over layers | 0.240 | 0.188 | +0.033 | — |
| Mean-AcT | baseline | diff-in-means at all blocks (raw) | 0.212 | 0.170 | +0.022 | [−0.038,+0.082] |
| ActAdd (single L14, mult 2) | baseline | single-layer vector | 0.122 | 0.095 | +0.008 | — |
| CAA (single L14, mult 2) | baseline | single-layer vector | 0.133 | 0.102 | +0.010 | — |
| ITI-C (top-48) | baseline | attn heads | 0.117 | 0.100 | −0.003 | [−0.050,+0.045] |
| AURA vanilla (control) | baseline | neuron gating | 0.100 | 0.113 | −0.032 | [−0.077,+0.010] |
| base (clean) | — | — | 0.120 | 0.100 | +0.000 | — |

## Reading
- **decode-space PI (+0.135) leads every baseline** by a clear margin. Feedback over the denoising step
  is the strongest aimer here.
- **Faithful single-layer CAA / ActAdd are near-zero** (+0.010 / +0.008) at the paper's ~2× multiplier —
  a single-layer edit is just weak on this task. (Cranking CAA to multiplier ~7.5 reaches +0.148, but
  that is ~4× past the fluency cap the CAA paper explicitly imposes; it is not a valid operating point.)
- **The strongest baselines are the all-layer / per-neuron ones:** Linear-AcT gaussian (+0.060), the
  `normal` all-layers vector (+0.055), AURA inject (+0.050) — all still below decode-space PI.
- **AURA vanilla** is a passing negative control (−0.032). **ITI-C never aims.** Mean-AcT (raw) saturates;
  Linear-AcT gaussian degenerates at strength ≥ 2. Parseable-only d_gap ≈ full d_gap (no unparse artifact).

## Method taxonomy (so labels are honest)
- **Single-layer steering vector:** CAA (dataset diff-in-means, 1 swept layer), ActAdd (single-pair diff,
  1 swept layer). Both raw vector × small multiplier, per their papers.
- **All-layer steering vector (the project's own baseline):** `normal` = one diff-in-means direction added
  at all 32 blocks. This is NOT a specific paper's method — do not call it CAA.
- **Genuinely distinct methods:** Mean-AcT / Linear-AcT (activation transport, all layers), AURA (neuron
  gating, MLP), ITI-C (per-head shift), layer-space PID (Nguyen et al., PID over layers).

## Remaining rigor (not fairness — statistics)
- n=400 → CI half-width ≈ 0.07, so +0.135 vs +0.060 is numerically clear but not yet *statistically*
  separated. Larger n resolves it.
- **Position-balanced arbiter** is the rigorous version; ours already runs there (decode-PI +0.200
  balanced), the baselines still need to go through `eval/balanced/`.

## Code / fixes
- Layer schemes confirmed from the papers (both single-layer). The earlier "CAA +0.148 beats decode" and
  "CAA = normal / all-layers" claims were WRONG and are removed. Faithful CAA/ActAdd runs live in
  `results/{caa,actadd}/cond_mult_a{16,32}` (α16 ≈ multiplier 2 given ‖r[14]‖≈8.5).
- Linear-AcT σ-mask bug fixed (low-variance neurons → identity). AURA/Linear-AcT hook MLP-hidden;
  ITI-C head-level (from scratch). Fits reuse the contamination-safe held-out BBQ-Race set.

## Reproduce
```
CUDA_VISIBLE_DEVICES=N python baselines/caa.py    --run --alpha 16 --layer 14   # CAA ~mult 2
CUDA_VISIBLE_DEVICES=N python baselines/actadd.py --run --alpha 16 --layer 14   # ActAdd ~mult 2
CUDA_VISIBLE_DEVICES=N python baselines/<m>.py    --run [--strength|--variant|--mode|--topk ..]
```
