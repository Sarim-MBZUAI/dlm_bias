# Pre-registration: UNQOVER decode-PI tuning (jobAJ)

Written BEFORE any EVAL-split number is inspected. The sweep is run and the
winner is selected on the BUILD split (105 instances / 420 target items) ONLY;
the 157-instance EVAL split is held out and is read only for the final chosen
config + its effort-matched control. Direction `arrows_unqover_black.pt` was fit
on the BUILD split (jobAI); tuning hyperparameters on the same BUILD split adds
no eval contamination (BUILD ∩ EVAL instances = ∅, verified).

## Diagnosis being fixed (verified on the existing EVAL decode-PI run)
Default PI (Kp=3, Ki=0.1, Kd=0, s*=0.9, amax=6, vhat L14) realizes on EVAL:
mean alpha = 2.71, sat_frac = 0.11 (NOT ceiling-limited), and is BIMODAL:
51% of items reach p_target>=0.9 (then the controller backs off) while 48% never
move (p_final<0.3) with alpha stuck at mid-range and NOT escalating to amax. So
the integral term is too weak (and Kd=0) to escalate on persistent error on the
hard items. Tuning aims to raise the integral escalation (Ki, optionally Kd)
without wrecking coherence on the easy items or over-saturating.

## Grid (8 configs; C0 = default reference)
| tag | cond | Kp | Ki  | Kd | amax | rationale                                            |
|-----|------|----|-----|----|------|-----------------------------------------------------|
| C0  | PI   | 3  | 0.1 | 0  | 6    | default / reference (reproduces current PI on BUILD)|
| C1  | PI   | 3  | 0.3 | 0  | 6    | 3x integral: escalate on persistent error (main fix)|
| C2  | PI   | 3  | 0.5 | 0  | 6    | 5x integral: stronger escalation                    |
| C3  | PI   | 5  | 0.3 | 0  | 6    | faster proportional + integral                      |
| C4  | PI   | 5  | 0.5 | 0  | 6    | strongest PI corner                                 |
| C5  | PID  | 3  | 0.3 | 1  | 6    | add derivative to damp overshoot while Ki escalates |
| C6  | PI   | 3  | 0.5 | 0  | 8    | headroom: only helps if Ki=0.5 saturates at amax=6  |
| C7  | PID  | 5  | 0.5 | 1  | 8    | aggressive PID + headroom (upper corner)            |

Setpoint fixed at 0.9 and vhat layer fixed at 14 throughout (unchanged from the
paper's operating point). All configs use the BUILD-fit native arrows.

## Metrics computed per config on BUILD (via unqover_metric.compute / target_gap)
- `cov`  : coverage (fraction of instances with all 4 records parseable)
- `f90`  : fraction of items with p_target_final >= 0.9  (control-objective hit-rate)
- `mean_alpha`, `sat_frac` : realized actuation and saturation
- `gdeb` : pref_gap_debiased toward Black (paper's fully-debiased bias)
- `graw` : pref_gap_raw toward Black (BBQ-like)

## Selection rule (deterministic; implemented in unqover/select_tuned_pi.py)
Let REF = C0, f0 = f90(C0), d0 = gdeb(C0).
1. GATE coherence: keep configs with `cov >= 0.95`.
2. GATE anti-saturation: among those, ELIGIBLE = configs with `gdeb >= d0 - 0.01`
   (the extra actuation must not collapse the debiased bias signal — a saturated
   controller that always picks the target both ways drives gdeb toward 0, which
   is exactly the failure mode of the over-strength open loop).
3. PRIMARY objective: if any ELIGIBLE config has `f90 > f0` (tuning genuinely
   reduces the stuck-item under-actuation), pick the ELIGIBLE config with the
   HIGHEST `f90`.
4. FALLBACK: if nothing improves `f90` without collapsing gdeb, keep the config
   with the HIGHEST `gdeb` among coverage-passing configs (i.e. do not change the
   default unless something strictly helps).
5. Tie-breaks (values within 0.01 on the primary key): higher `gdeb` -> lower
   `mean_alpha` (efficiency) -> lexical tag (= grid order). Fully reproducible.

Rationale: the closed loop's distinctive claim vs the open loop is that it
*reaches the p=0.9 setpoint adaptively*. `f90` is the direct control-objective
hit-rate and the exact quantity the diagnosis shows is broken, so it is the
primary. The gdeb anti-saturation guard prevents "fixing" f90 by over-steering
into the same signal-collapse the fixed-alpha open loop suffers.

## Effort-matched open-loop control (the fair comparator)
`pid_steer_unqover.py --mode normal --alpha A` accepts FRACTIONAL alpha, so we
match the EXACT realized mean actuation (no integer rounding needed):
- `uq_normal_effort`       : alpha = realized mean alpha of the DEFAULT PI on
  EVAL = 2.7066 (the honest comparator to the existing uq_decode_PI.jsonl;
  the prior uq_normal_a4 at alpha=4 was an unfair over-strength control).
- `uq_normal_effort_tuned` : alpha = realized mean alpha of the CHOSEN tuned
  controller, recomputed from its EVAL run at run time.

## Outputs (NOTHING is overwritten)
New EVAL stems in results/unqover_native/: `uq_decode_PI_tuned.{jsonl,json}`,
`uq_normal_effort.{jsonl,json}`, `uq_normal_effort_tuned.{jsonl,json}`, plus
metric_*.txt. The existing `uq_decode_PI.jsonl` / `uq_normal_a4.jsonl` are kept
so default-vs-tuned can be compared honestly. Sweep artifacts live under
results/unqover_native/sweep/.

## Overfitting caveat
Tuning hyperparameters per-benchmark could be read as overfitting. Mitigations:
tuning uses the BUILD split only (disjoint from EVAL); the grid is small (8) and
motivated by the mechanistic diagnosis (weak integral) rather than eval score;
the selection rule is pre-registered here and coded before any EVAL number is
seen; and both the DEFAULT (untuned) and the TUNED configs are reported on EVAL
side by side, so the reader sees the untuned result too.
