# Baselines vs. decode-space PID (ours) — BBQ-400 (bias injection toward "Black")

**Ours is decode-space PID** (PID feedback over the diffusion denoising step). Everything else is a
**baseline**: layer-space PI/PID (ported prior work, Nguyen et al.), normal vector (open-loop), and the
six external methods in `baselines/` (CAA, ActAdd, Mean-AcT, Linear-AcT, AURA, ITI-C).

`d_gap = ΔBlack − Δnon-Black` vs `base` (0.120 / 0.100). **Raw / position-confounded** — the
position-balanced arbiter (`eval/balanced/`) is the tiebreaker and is still TODO for the baselines.
95% CI = 2000-sample bootstrap over the 400 items (base held fixed). `d_gap|p` = parseable-only.

## Best operating point per method (after a full dose-response sweep)

| method | who | axis | Black | non-Black | unparse | d_gap | 95% CI | d_gap\|p |
|---|---|---|---:|---:|---:|---:|:--:|---:|
| **CAA α64** | baseline | single layer L14 | 0.468 | 0.300 | 0.00 | **+0.148** | [+0.065,+0.228] | +0.148 |
| **decode-space PI** | **OURS** | denoising step | 0.302 | 0.147 | 0.03 | **+0.135** | [+0.068,+0.200] | +0.139 |
| **ActAdd α64** | baseline | single layer L14 | 0.450 | 0.305 | 0.00 | **+0.125** | [+0.038,+0.210] | +0.125 |
| **decode-space PID** | **OURS** | denoising step | 0.287 | 0.165 | 0.02 | **+0.102** | [+0.038,+0.165] | +0.105 |
| Linear-AcT gaussian s1 | baseline | MLP-hidden | 0.280 | 0.200 | 0.00 | +0.060 | [−0.007,+0.128] | +0.060 |
| normal vector α4 | baseline | open-loop | 0.370 | 0.295 | 0.01 | +0.055 | [−0.022,+0.130] | +0.056 |
| AURA inject γ4 | baseline | MLP-hidden | 0.245 | 0.175 | 0.00 | +0.050 | [−0.010,+0.115] | +0.050 |
| Linear-AcT empirical s1 | baseline | MLP-hidden | 0.263 | 0.200 | 0.00 | +0.043 | [−0.022,+0.107] | +0.043 |
| layer-space PI | baseline | layer depth | 0.242 | 0.182 | 0.00 | +0.040 | [−0.020,+0.105] | +0.040 |
| Mean-AcT (unit s2) | baseline | all 32 blocks | 0.212 | 0.170 | 0.00 | +0.022 | [−0.038,+0.082] | +0.023 |
| ITI-C (best, top48 α8) | baseline | attn heads | 0.117 | 0.100 | 0.00 | −0.003 | [−0.050,+0.045] | −0.003 |
| AURA vanilla (control) | baseline | MLP-hidden | 0.100 | 0.113 | 0.00 | −0.032 | [−0.077,+0.010] | −0.032 |
| base (clean) | — | — | 0.120 | 0.100 | 0.00 | +0.000 | — | — |

## Honest reading (this supersedes the earlier "beats every baseline" claim)
1. **decode-space PI does NOT beat every baseline once baselines get a dose-response.** Single-layer
   **CAA α64 (+0.148)** numerically tops the table and **ActAdd α64 (+0.125)** is close; all three CIs
   overlap decode-PI's [+0.068,+0.200] → **statistically tied, not beaten.**
2. **At n=400 almost nothing is separated** — CI half-width ≈ 0.07. Only CAA-α64, ActAdd-α48/64, and
   decode-PI/PID exclude zero; Linear-AcT / normal / AURA / layer-PI all include 0.
3. **Parseable-only ≈ full d_gap** — decode-PI's edge is not an unparse/coherence artifact.
4. **Faithful Mean-AcT (raw) saturates** (raw_s{1,4,8} all ≈ 0.355/0.338, d_gap ≈ 0; s2 negative) — the
   unit-normalized s2 (+0.022) is its practical best. **Linear-AcT gaussian degenerates at s≥2**
   (s3 → 100% unparse); s1 (+0.060) is its ceiling. **ITI-C never aims** (best −0.003).

## The defensible claim
Not "beats every baseline." The clean result is the **matched-actuation ablation**: at equal total
actuation (α≈4 × 64 steps ≈ 256, same v̂, same all-32-block actuator) **decode-space PI (+0.135)
vs open-loop normal-α4 (+0.055)** — feedback over the denoising step beats open-loop. Even this needs
the position-balanced arbiter + paired significance to stand, since CAA/ActAdd at very large single-layer
α reach the same raw d_gap and the numbers are position-confounded.

## Caveats / TODO
- **Position-balanced arbiter (top priority):** CAA/ActAdd α64 are huge single-layer edits (Black 0.47);
  their raw +0.148/+0.125 may be position-driven. Run best-of-each through `eval/balanced/` (3 rotations).
- Bootstrap CIs hold base fixed; a paired (per-item, vs-base) test would tighten the ours-vs-baseline call.
- Fits reuse the contamination-safe held-out BBQ-Race set. Linear-AcT σ-mask bug fixed (identity for
  low-variance neurons); AURA/Linear-AcT hooked at MLP-hidden; ITI-C head-level (from scratch).

## Reproduce
```
CUDA_VISIBLE_DEVICES=N python baselines/<m>.py --fit
CUDA_VISIBLE_DEVICES=N python baselines/<m>.py --run --alpha .. [--layer|--topk|--variant|--mode|--direction ..]
```
