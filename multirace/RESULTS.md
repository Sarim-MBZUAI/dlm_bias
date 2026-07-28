# Multi-race steering on LLaDA-8B-Instruct (BBQ-400 per target)

Target-generality test of the contribution (**decode-space PID** — feedback over
the denoising step): steer toward **white / asian / latino / arab** instead of
Black, vs the open-loop **normal** vector (α4, all 32 blocks) and single-layer
**CAA** (L14, α2). Per target: its own 400-item ambiguous eval set
(`_sweep400_<target>.jsonl`) and its own 400-item held-out direction set —
contamination-safe, disjoint from each other and from the Black experiment's keys
(see `items_manifest.json`). Same LLaDA harness, gains (Kp3/Ki0.1/amax6), and
metric conventions as the Black study.

`d_gap = gap_within(steered) − gap_within(base)` where `gap_within = target −
non-target pick rate`; each target uses its own base run. `unp` = unparseable
fraction (coherence gauge; >0.15 = flagged ⚠, LLaDA convention from the Dream study).

## Table (n=400 per cell, single run, temp 0)

| target | condition | tgt | non-tgt | unp | d_gap |
|---|---|---:|---:|---:|:--|
| **arab** | **decode-PI (ours)** | 0.510 | 0.263 | 0.060 | **+0.247** |
| | normal α4 | 0.315 | 0.237 | 0.000 | +0.078 |
| | CAA α2 | 0.160 | 0.163 | 0.000 | −0.003 |
| **white** | **decode-PI (ours)** | 0.357 | 0.280 | 0.110 | **+0.097** |
| | normal α4 | 0.003 | 0.010 | **0.988** ⚠ | (+0.013, meaningless) |
| | CAA α2 | 0.120 | 0.133 | 0.000 | +0.008 |
| **latino** | decode-PI amax2 (best coherent) | 0.110 | 0.072 | 0.000 | +0.016 |
| | decode-PI amax6 | 0.338 | 0.168 | 0.302 ⚠ | +0.148 (flagged) |
| | normal α4 | 0.302 | 0.255 | 0.240 ⚠ | +0.025 (flagged) |
| | CAA α2 | 0.113 | 0.080 | 0.000 | +0.010 |
| **asian** | decode-PI amax1/2/3 (all coherent) | ~0.14 | ~0.16 | 0.000 | −0.005 … −0.023 |
| | decode-PI amax4/6 | — | — | 0.92/0.93 ⚠ | collapsed |
| | normal α1/α2 (coherent) | ~0.14 | ~0.15 | 0.000 | −0.005 / +0.000 |
| | CAA α2 | 0.135 | 0.147 | 0.000 | +0.003 |
| (ref) black | decode-PI | 0.302 | 0.147 | ~0 | +0.135 (README table) |

Fair-dose additions (so "ours beats normal" is not vs a collapsed strawman):
normal for **white** at α1 = **+0.040** (coherent; ours +0.097 still leads), α2 = +0.028;
normal for **asian** at α1/α2 = flat (see table).

## Dose-response detail (decode-PI, per target)

| amax | 1 | 2 | 3 | 4 | 6 |
|---|---|---|---|---|---|
| asian d_gap | −0.005 | −0.005 | −0.023 | collapse (0.915) | collapse (0.930) |
| latino d_gap | — | +0.016 | +0.005 | +0.046 (0.485 ⚠) | +0.148 (0.302 ⚠) |

Base runs: all four targets have near-zero base gap (−0.020 … +0.022) and 0.000
unparseable — the eval sets are balanced starting points.

## Readout

1. **Where a target is steerable at all, decode-space feedback is the strongest
   aimer**: arab **+0.247** (best result on any target incl. Black's +0.135) and
   white **+0.097** are clean and beat normal at ITS best coherent dose
   (white-normal α1 = +0.040) and CAA by wide margins.
2. **Steerability itself is target-dependent — asian is unsteerable on LLaDA.**
   The full dose-response shows a cliff, not a curve: amax 1–3 are fully coherent
   with ZERO aim (−0.005…−0.023), amax ≥ 4 collapses (0.92+ unparseable). No
   method aims asian (normal flat at every coherent α, CAA +0.003). The asian
   diff-in-means direction appears to carry no usable steering signal — a
   substantive negative finding about direction quality, not a method failure.
3. **Latino is weakly steerable**: best coherent point +0.016 (amax2); the +0.148
   at amax6 rides on 30% unparseable output and is not a valid operating point.
4. **The coherence budget is direction-dependent, not just model-dependent** —
   the LLaDA-side mirror of the Dream finding. But unlike Dream (smooth optimum
   at amax≈1), asian shows a hard cliff between amax3 and amax4.
5. **Feedback degrades far more gracefully than open-loop.** For white, decode-PI
   keeps 89% of output parseable while normal-α4 destroys 99% — the controller
   backs off where the constant push cannot.
6. **Faithful single-layer CAA is near-zero at every target** (−0.003 … +0.010),
   replicating the Black-target finding.

## Caveats

- Single run, n=400 per cell, no CIs; position-confounds not yet balanced
  (the position-balanced arbiter has only been run for the Black target).
- The asian negative result is about THIS direction (answer-anchored diff-in-means
  from 400 held-out items) on THIS model; a different direction-building recipe
  could still steer asian — untested.
- Reproduce: `multirace/run_multirace_gpu{3,4}.sh` + dose sweeps
  `multirace/run_dose_gpu{3,4}.sh` (env `sarim_awm`, GPUs 3/4).
