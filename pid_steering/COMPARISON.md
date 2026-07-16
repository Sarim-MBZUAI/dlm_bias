# PID-Steering on LLaDA-8B-Instruct — grand comparison (BBQ-400, Black-referent ambiguous)

> ⚠️ **This raw table is position-confounded** (the model has a letter preference and
> collapses onto "A" under strong steering, which inflates "black-pick" whenever the Black
> option sits at A). For the **rigorous, position-balanced result** — which is the one to
> trust — see [`balanced/RESULTS.md`](balanced/RESULTS.md). Short version there: only
> decode-space PI shows a genuine Black-vs-non-Black preference; the others mostly disinhibit.

All conditions steer toward the Black option; injection at all 32 layers, every denoising step, temp 0, n=400.
`d_gap = (ΔBlack − Δnon-Black)` vs base; positive = *aimed* toward Black (not just disinhibiting abstention).

**Seed: 42** — governs BBQ sampling and the held-out / eval split (`_sweep400.jsonl`; direction-build
items disjoint from the seed-42 n=1000 sample). Generation is **deterministic** (temperature 0, greedy),
so there is no separate sampling seed and runs are exactly reproducible.

Raw pick-rates (fractions of n=400); `d_gap` positive = aims at Black, ≈0 = only disinhibits. Higher is
better, but these numbers are position-confounded — read the balanced table before trusting magnitudes.

| method | control axis | Black | non-Black | abstain | unparse | d_gap |
|---|---|---:|---:|---:|---:|---:|
| base (clean) | — | 0.120 | 0.100 | 0.780 | 0.000 | +0.000 |
| normal vector, all-layer α=2 | open-loop | 0.177 | 0.138 | 0.685 | 0.000 | +0.020 |
| normal vector, all-layer α=4 | open-loop | 0.370 | 0.295 | 0.328 | 0.007 | +0.055 |
| layer-space P (Kp=1) | layer depth | 0.188 | 0.142 | 0.670 | 0.000 | +0.025 |
| layer-space PI | layer depth | 0.242 | 0.182 | 0.575 | 0.000 | +0.040 |
| layer-space PID | layer depth | 0.240 | 0.188 | 0.573 | 0.000 | +0.033 |
| decode-space P (Kp=3) | denoising step | 0.180 | 0.128 | 0.690 | 0.003 | +0.033 |
| **decode-space PI** | **denoising step** | **0.302** | **0.147** | **0.522** | **0.028** | **+0.135** |
| decode-space PID | denoising step | 0.287 | 0.165 | 0.525 | 0.022 | +0.102 |

## Read-out
- **Decode-space PI is the strongest aimer (d_gap +0.135)** — Black-pick 0.30 vs non-Black 0.15. Controlling P(Black letter) over denoising steps is *target-aware*, so it aims rather than disinhibits.
- **Open-loop normal vector disinhibits**: at α=4 it lifts Black to 0.37 but non-Black to 0.30 (d_gap only +0.055).
- **Layer-space PID (the paper's axis) is modest** (+0.033–0.040) and no better than a plain vector.
- **Integral helps on both axes; Derivative does not** (PI ≥ PID everywhere).
- Caveat: decode-space PI/PID carry a mild coherence cost (~2–3% unparseable at mean α≈4); layer-space & normal-α2 stay at 0%. n=400 (single-run, no CIs).
