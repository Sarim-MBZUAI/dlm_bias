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
| **latino** | **decode-PI (ours)** | 0.338 | 0.168 | **0.302** ⚠ | **+0.148** (flagged) |
| | normal α4 | 0.302 | 0.255 | 0.240 ⚠ | +0.025 (flagged) |
| | CAA α2 | 0.113 | 0.080 | 0.000 | +0.010 |
| **asian** | decode-PI (ours) | 0.068 | 0.000 | **0.930** ⚠ | (+0.082, meaningless) |
| | normal α4 | 0.003 | 0.000 | **0.998** ⚠ | (+0.017, meaningless) |
| | CAA α2 | 0.135 | 0.147 | 0.000 | +0.003 |
| (ref) black | decode-PI | 0.302 | 0.147 | ~0 | +0.135 (README table) |

Base runs: all four targets have near-zero base gap (−0.020 … +0.022) and 0.000
unparseable — the eval sets are balanced starting points.

## Readout

1. **Decode-space feedback aims at every target where output stays coherent.**
   Arab **+0.247** (its best result on any target, including Black's +0.135) and
   white **+0.097** are clean (unp ≤ 0.11) and beat normal/CAA by wide margins.
2. **The coherence budget is direction-dependent, not just model-dependent.**
   The same α settings that are coherent for Black/arab collapse for asian
   (decode 0.93, normal 0.998 unparseable) and strain for latino (0.24–0.30).
   This is the LLaDA-side mirror of the Dream finding: actuation magnitude must
   be tuned per direction, not just per model. The asian arrows are the obvious
   next dose-response sweep (lower amax), exactly as done for Dream.
3. **Feedback degrades far more gracefully than open-loop.** For white, decode-PI
   keeps 89% of output parseable while normal-α4 destroys 99% — the controller
   backs off (alpha follows the error) where the constant push cannot.
4. **Faithful single-layer CAA is near-zero at every target** (−0.003 … +0.010),
   replicating the Black-target finding that single-layer steering is weak on LLaDA.

## Caveats

- Single run, n=400 per cell, no CIs; position-confounds not yet balanced
  (the position-balanced arbiter has only been run for the Black target).
- Latino decode-PI (+0.148) exceeds the 0.15 unparseable threshold (0.302) —
  treat as provisional until a lower-amax point is run.
- Asian rows are collapsed and excluded from any ranking; a dose-response sweep
  (amax < 6) is required before claiming anything about the asian target.
- Reproduce: `multirace/run_multirace_gpu{3,4}.sh` (env `sarim_awm`, GPUs 3/4).
