# Position-balanced arbiter — ALL conditions (BBQ, 3-rotation, 1200 evals each)

Closes the biggest rigor hole: the letter-position confound was previously
controlled only for a few Black-target conditions. Here every headline condition —
all six faithful baselines, decode-space PID, and the arab/white generality
targets — is run through the same 3-rotation protocol (each option visits A/B/C
exactly once; an always-"A" oracle scores exactly 0). `BAL gap = target −
non-target`, pooled over 1200 evals; letter-jam-immune by construction.

Baseline operating points are pinned to the **exact configs that produced the
published `results/BASELINES.md` raw rates** (CAA α16, ActAdd α16, Mean-AcT unit
s2, Linear-AcT gaussian s1, AURA inject γ4 / vanilla, ITI-C top48 α8).

## Black target — ours vs every baseline

| method | BAL gap | (raw d_gap) |
|---|---:|---:|
| **decode-space PID (OURS)** | **+0.158** | +0.102 |
| Mean-AcT (unit s2) | +0.036 | +0.022 |
| ActAdd (α16) | +0.034 | +0.008 |
| CAA (α16) | +0.033 | +0.010 |
| Linear-AcT (gaussian s1) | +0.026 | +0.060 |
| ITI-C (top48 α8) | +0.015 | −0.003 |
| AURA vanilla (ctrl) | +0.008 | −0.032 |
| AURA inject (γ4) | +0.007 | +0.050 |

**The lead widens under balancing** — 4.4× the best baseline (raw was ~2×).
Several baselines' raw numbers were partly position-jamming: AURA-inject collapses
+0.050→+0.007, Linear-AcT +0.060→+0.026, once the confound is removed. Decode-PID
barely moves (+0.102→+0.158; the existing `results/balanced/RESULTS.md` decode-**PI**
run is +0.200). Feedback over the denoising step is the strongest *and* the most
position-robust aimer.

## Generality targets (balanced)

| target | ours (decode-PI) | normal | base |
|---|---:|---:|---:|
| **arab** | **+0.250** | +0.134 | −0.005 |
| **white** | +0.051 | +0.021 | −0.016 |

Ours leads at both. Arab is unchanged from raw (+0.247→+0.250) — not an artifact.
White drops (+0.097 raw → +0.051 balanced): roughly half its raw lead was position
confound.

## The honest catch: per-position decomposition

The pooled BAL gap is position-immune *on average*, but decomposing it by the
letter the target sits on shows the effect is **A-concentrated**:

| condition | @A | @B | @C |
|---|---:|---:|---:|
| Black decode-PID | +0.467 | −0.057 | +0.063 |
| arab decode-PI | +0.818 | −0.258 | +0.190 |
| white decode-PI | +0.893 | −0.415 | −0.325 |

- **Black and arab are genuine aims** that happen to favor A: both stay **positive
  at C** (+0.063, +0.190), so the signal is not purely positional. Discounting A,
  black ≈ +0.00…+0.06 and arab ≈ +0.19 remain — arab is robust, black is modest.
- **White is essentially a position-A phenomenon** — negative at both B and C. Its
  +0.051 pooled gap should NOT be read as a clean target aim; the controller is
  largely learning "emit A" for white, not "prefer the White option regardless of
  slot." Reported here rather than buried.

This mirrors and extends the caveat in `results/balanced/RESULTS.md` (decode-PI's
Black gap concentrating at A, true effect ∈ [+0.05, +0.20]). The direction of the
finding: **the method's ranking survives position-balancing, but the magnitude of
the "aim" is entangled with a learned position-A bias that varies by target** —
strong for arab, real-but-modest for black, largely artifactual for white.

## Caveats / next

- Single run per rotation (1200 pooled), still **no CIs** — bootstrap over the
  1200 items is the next step (Tier-1 item 2).
- Per-position asymmetry motivates a *setpoint-on-position* control or an
  A/B/C-symmetric direction as a mitigation — untested.
- Reproduce: `balanced_all/run_balanced_gpu2.sh` + `run_balanced_gpu4.sh`
  (`BAL_GPU=<id>` to retarget), aggregate with `balanced_all/aggregate.py`.
