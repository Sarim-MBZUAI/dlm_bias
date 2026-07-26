# Baselines vs. decode-space PID (ours) — BBQ-400 (bias injection toward "Black")

**Fair, all-layers comparison.** Every method steers all 32 blocks (as CAA/ActAdd/AcT do in their
papers). **Ours is decode-space PID** (PID feedback over the diffusion denoising step). Baselines:

- **CAA** (Rimsky et al.) = add the diff-in-means steering vector across the layers = **our `normal`
  vector**. So `normal α4` **is** the CAA result — no separate run needed.
- **ActAdd** (Turner et al.) = the same diff-in-means vector built from a single (n=1) pair — a noisier
  CAA; not run separately (dominated by CAA).
- **Mean-AcT / Linear-AcT** (Rodriguez et al.), **AURA** (Suau et al.), **ITI-C** (Li et al.) — the
  genuinely distinct methods (transport maps / neuron gating / head shift).
- **layer-space PI/PID** = the ported prior-work PID (Nguyen et al.), i.e. PID over layer depth.

`d_gap = ΔBlack − Δnon-Black` vs `base` (0.120 / 0.100); higher = aims at Black.
95% CI = 2000-sample bootstrap over the 400 items (base fixed).

| method | who | family / axis | Black | non-Black | d_gap | 95% CI |
|---|---|---|---:|---:|---:|:--:|
| **decode-space PI** | **OURS** | denoise-step feedback | 0.302 | 0.147 | **+0.135** | [+0.068,+0.200] |
| **decode-space PID** | **OURS** | denoise-step feedback | 0.287 | 0.165 | **+0.102** | [+0.038,+0.165] |
| Linear-AcT (gaussian) | baseline | per-neuron transport (MLP) | 0.280 | 0.200 | +0.060 | [−0.007,+0.128] |
| CAA (= normal vector α4) | baseline | diff-in-means, all layers | 0.370 | 0.295 | +0.055 | [−0.022,+0.130] |
| AURA inject γ4 | baseline | neuron gating (MLP) | 0.245 | 0.175 | +0.050 | [−0.010,+0.115] |
| Linear-AcT (empirical) | baseline | per-neuron transport (MLP) | 0.263 | 0.200 | +0.043 | [−0.022,+0.107] |
| layer-space PI | baseline (prior work) | PID over layers | 0.242 | 0.182 | +0.040 | [−0.020,+0.105] |
| layer-space PID | baseline (prior work) | PID over layers | 0.240 | 0.188 | +0.033 | — |
| Mean-AcT | baseline | diff-in-means (raw), all layers | 0.212 | 0.170 | +0.022 | [−0.038,+0.082] |
| ITI-C (top-48) | baseline | head shift | 0.117 | 0.100 | −0.003 | [−0.050,+0.045] |
| AURA vanilla (control) | baseline | neuron gating | 0.100 | 0.113 | −0.032 | [−0.077,+0.010] |
| base (clean) | — | — | 0.120 | 0.100 | +0.000 | — |

## Reading
- **decode-space PI (+0.135) leads every baseline.** Best genuine baseline is Linear-AcT gaussian
  (+0.060); CAA (= the normal vector) is +0.055. Feedback over the denoising step is the strongest aimer.
- **AURA vanilla** is a passing negative control (−0.032, correctly suppresses); **inject** flips it to +0.050.
- **ITI-C never aims** (best −0.003). **Mean-AcT (raw) saturates** into disinhibition; **Linear-AcT
  gaussian degenerates at strength ≥ 2** (s3 → 100% unparse), so s1 (+0.060) is its ceiling.
- **Coherence check:** parseable-only d_gap ≈ full d_gap everywhere, so decode-space PI's edge is not an
  unparse artifact.

## Remaining rigor (not fairness problems — just tighter stats)
- **n=400 → wide CIs (±≈0.07).** +0.135 (ours) vs +0.060 (Linear-AcT) is numerically clear but not yet
  *statistically* separated. Larger n would resolve it.
- **Position-balanced arbiter** is the rigorous version (removes letter-position bias). Ours already runs
  there (decode-PI +0.200 balanced); the baselines still need to go through `eval/balanced/`.

## Code / fixes
- Fair all-layers CAA = `normal` (no separate run). Bogus single-layer `caa/`+`actadd/` runs removed.
- Linear-AcT σ-mask bug fixed (low-variance neurons → identity, per AcT `transport.py`).
- AURA/Linear-AcT hook the MLP-hidden units; ITI-C is head-level (built from scratch). Fits reuse the
  contamination-safe held-out BBQ-Race set; artifacts in `baselines/cache/` (gitignored).

## Reproduce
```
CUDA_VISIBLE_DEVICES=N python baselines/<m>.py --fit    # meanact/linearact/aura/itic (GPU)
CUDA_VISIBLE_DEVICES=N python baselines/<m>.py --run [--strength|--variant|--mode|--topk ..]
# CAA = normal vector: python steering/pid_steer.py --mode normal --alpha 4  (results/normal/)
```
