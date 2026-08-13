# E1: Trajectory-level analysis of the decode-space PI/PID controller

Produced 2026-08-13 by `analysis/trajectory/traj_analysis.py` (branch
`e1/trajectory-analysis`) from the committed per-item trajectories of every
decode-PI/PID run (`pblack_traj`/`ptarget_traj` = P(target letter at the
answer slot) at each of the 64 denoising steps; `alpha_traj` = the injection
strength applied at each step). 19 runs, 22,800 item-trajectories, CPU-only.

Regenerate everything (tables, CSVs, figures) with:

```bash
python analysis/trajectory/traj_analysis.py        # needs numpy + matplotlib
```

## Methods / definitions

**Outcomes** use the repo's authoritative STRICT rule (replicated verbatim
from `balanced_all/strict_pool.py`): valid iff an A/B/C letter (either case)
starts `model_output`, followed by whitespace/punctuation/end; valid letters
classify via `target_idx`/`black_idx` and `unk_idx` into
target / abstain / comparator; else invalid.

**Lock-in step** `t_lock(th)` = first step t with p(s) >= th for *all*
s >= t (the last never-undone upward crossing). Primary th = 0.5; robustness
th in {0.3, 0.5, 0.7, 0.9}. Undefined when p_final < th.

**Freeze step** `t_freeze(eps)` = 1 + last step with |dp| >= eps (primary
eps = 0.02; robustness 0.01/0.05). Threshold-free commitment proxy: once
LLaDA transfers the answer-slot token, the measured distribution pins. On
black decode-PI the two definitions agree (median |t_lock - t_freeze| = 2,
r = 0.69), and t_freeze is defined for failures too. Failure freeze medians
are 31/32/31 at eps = 0.02/0.01/0.05 -- insensitive to eps.

**First saturation** `t_pin` = first step with alpha(t) >= alpha_max
(6.0 LLaDA, 1.0 Dream), computed from `alpha_traj` directly (upper pin only;
the logged `sat_frac` also counts lower-clamp raw < 0 events).

**Conventions.** LLaDA logs alpha(t) applied at step t, aligned with p(t).
Dream logs the *command issued after* observing p(t); we re-align
(applied[t] = cmd[t-1], applied[0] = 0). The round-1 black base predates
trajectory logging; the three seed bases (3,600 unsteered trajectories) are
the no-steering reference. All items in every pool are ambiguous-context BBQ
(the balanced-400 pools contain no disambig items), and answer positions are
rotation-balanced (3x400 per run).

**Sanity check**: strict outcome rates recomputed here reproduce
`results/balanced_all/RESULTS_STRICT.md` / `results/ROUND2_STRICT.md`
exactly (e.g. black PI gap +0.167, PID +0.160, arab +0.223, asian 93.8%
invalid, white 96.1% invalid).

## RQ1 -- When does the biased answer lock in?

Black decode-PI (round 1, pooled 3 rotations, n = 1200; fig_lockin,
fig_p_mean):

- 326/1200 items lock at th = 0.5 (= 89% of the 365 strict-target picks;
  the rest are mostly lowercase-letter successes the sensor cannot see,
  below).
- **The distribution is bimodal**: 22% lock at t <= 8 (median 0-2: p is
  above th from the unsteered probe onward -- *prior-driven*), 64% lock
  inside the token-commit wave t in [24, 40], 8% later. Median t_lock = 30
  of 64.
- **Lock-in is a discrete commit event, not a drift**: p jumps by a median
  0.44 (IQR 0.22-0.54) at the lock step, i.e. the answer-slot token is
  transferred and the distribution snaps.
- **Robustness**: per-rotation medians 30/30/30; seeds 1-3 medians 31/30/31
  (early fractions 0.17-0.22); PID median 29; th = 0.3/0.5/0.7/0.9 keeps
  n_locked at 332/326/308/291 with the same wave-centered ECDF shape
  (fig_lockin, right); the median shifts mechanically with the threshold
  (26.5/30/32/32) but stays inside the commit wave.
- **Steering does not move the commit time -- it biases what gets
  committed.** Unsteered base runs lock at the same wave (seed-base medians
  24-28) but on only ~10% of items; steering multiplies the lock count by
  ~2.5x (2.3-2.7x across the three paired seeds) at unchanged timing. The mechanism is a modest pre-commit lift: mean
  p(t < 24) rises from 0.130-0.142 (base) to 0.170-0.182 (steered) across
  the three seeds, which the irreversible commit then amplifies to ~0.99
  or ~0.
- Position note: *early* (prior) locks are depleted of position-A items
  (21-25% target@A) while commit-wave locks are enriched (45-49%) -- the
  known gap@A position effect lives in the commit wave, not in the prior.
- The steps=32 run shifts the wave in *fractional* terms: median lock 27/32
  (84% of decode) vs 30/64 (47%). Commit timing is a property of the
  transfer schedule (~1 token/step at s32 vs ~0.5 at s64), not of the
  controller.

## RQ2 -- Steerable vs unsteerable targets

Mean trajectories, decode-PI at amax 6 on all targets (fig_p_mean right,
fig_alpha_targets):

| target | mean p: t0 -> peak -> final | mean alpha final | ever-pin | strict target / invalid |
|---|---|---|---|---|
| black  | 0.12 -> 0.32 (t~29) -> 0.27 | 4.7 | 0.72 | 0.304 / 0.078 |
| latino | 0.12 -> 0.26 (t~22) -> 0.22 | 4.9 | 0.78 | 0.225 / 0.596 |
| arab   | 0.15 -> 0.18 (t~2) -> 0.08  | 5.5 | 0.92 | 0.483 / 0.097 |
| asian  | 0.175 -> 0.175 (t=0) -> 0.06 | 5.7 | 0.94 | 0.059 / 0.938 |
| white  | 0.076 -> 0.076 (t=0) -> 0.04 | 5.8 | 0.96 | 0.039 / 0.961 |

- **On asian/white the mean p_target(t) curve never exceeds its t = 0
  probe value** (its peak is at t = 0) -- apart from sub-0.02 wiggles it
  declines steadily to 0.06/0.04 while alpha ramps to the ceiling and
  stays pinned for the rest of the decode. There is no transient success
  followed by collapse; the loop simply gets no reward at any alpha, and
  integral windup does the rest. This is the
  trajectory-level picture behind ROUND2_STRICT section 5's "hard targets
  drive the controller hotter".
- **The few asian/white "successes" are prior hits, not steering wins**:
  all are sensor-visible locks with median t_lock = 3 (asian) / 0 (white),
  54%/70% locked by t = 8, and median p0 = 0.53/0.62 -- the model already
  favored the target letter before any steering acted. Contrast black:
  median p0 of successes 0.14, median lock 30.
- **Sensor case-blindness (implementation finding).** The controller's
  sensor measures P("A")+P(" A") -- *uppercase only* -- at generated
  position 0 (`steering/denoise_pid.py::letter_token_ids` /
  `p_black_from_logits`), while the STRICT parser accepts lowercase
  letters. On arab, **480/580 (82.8%) of strict-target successes end with
  p_final ~ 0** (median 0.0001, max 0.0007) because the model emitted a
  lowercase letter ('c', 'a') that the sensor cannot see. The lowercase
  emission is itself a steering side effect specific to the arab direction:
  no base run ever emits a lowercase answer (0/6000 valid base answers
  across all five targets), 89% of *all* arab valid answers are lowercase
  vs 6% for black and ~0% for latino/asian/white, and the lowercase token
  commits early at moderate alpha (freeze median 7, alpha at freeze median
  3.1, i.e. half the ceiling) -- so the windup (first pin median 37,
  freeze < pin on 100% of these items) follows the invisible win rather
  than causing it. The controller winds up to the ceiling on wins and
  losses alike (ever-pin 0.92). Black is mildly affected (7.9-9.7%
  lowercase successes); asian/latino/white not at all (0%). Consequence: arab's "hot" telemetry
  (mean alpha 4.56, sat 0.38) reflects a blind sensor, not a hard target --
  arab is in fact the *most* steerable direction (strict gap +0.223).
  Actuation effort separates steerable from unsteerable directions only
  when the sensor can see success.

## RQ3 -- Commitment vs saturation ordering

fig_commit_vs_sat; `summary_ordering.csv`. Black decode-PI failures
(comparator+abstain+invalid, n = 835): 100% eventually pin at alpha_max,
but **t_freeze precedes t_pin on 93% of items, by a median 12 steps**
(freeze median 31, pin median 42). Success items almost never saturate at
all: ever-pin 8% (PI) / 10% (PID) / 7-9% (seeds) vs 100% for failures --
the success/failure dichotomy in saturation is near-total. Replication:
PID 84% freeze-first (median +13), seeds 91-92% (+12), latino 100% (+14).

A caveat on how much of the ordering is baked in: with Kp = 3, Ki = 0.1
and setpoint 0.9 the applied alpha is bounded by 2.7 + 0.09(t+1), so the
integrator *cannot* reach the 6.0 ceiling before step ~36 (observed
minimum pin on PI runs: exactly 36). The ordering statistic is therefore
partly mechanical -- saturation arrives too late, by construction, to
cause any commit inside the t in [24, 33] wave. It is not fully forced
(27.5% of black-PI failures freeze at t >= 36, where either order was
possible, and freeze-first still holds at 93%), but the causal claim
rests on the two independent pieces of evidence below more than on the
93% itself.

**Saturation is a symptom of an already-lost item (windup on an unfixable
error), not a cause of failure.** Two further pieces of evidence:

1. The steps=32 run **never saturates on a single item** (the integral has
   half the steps to wind; mean alpha 3.27) yet reproduces the 64-step
   outcome rates (target 0.294 vs 0.304; gap +0.151 vs +0.167). Removing
   saturation entirely does not change steering success.
2. On arab (blind sensor) and white (flat p), freeze trivially precedes pin
   (100%), because commitment/collapse happens in the first steps while the
   ramp needs ~35 steps to reach 6.0 from alpha(0) ~ 2.4 at Ki = 0.1.

**Dream is the stated exception and is excluded from this claim**: at
amax = 1.0 the proportional command alone (Kp*e ~ 2.4) exceeds the ceiling
from step 0, so the controller pins at median step 1 and stays (pin-frac
0.95) -- saturation there precedes everything and carries no information;
the run behaves like a weak open-loop vector (as ROUND2_STRICT already
concluded from sat_frac 0.97).

## RQ4 -- Success vs failure trajectories (black decode-PI)

fig_success_failure. Mean trajectories by STRICT class:

- **target (n=365)**: p starts elevated (0.26 vs 0.04-0.11 for the other
  classes), climbs under moderate alpha, snaps to ~0.86 at the commit wave
  and holds. The controller *backs off* after lock (mean alpha falls from
  2.9 to ~1.9 as the error vanishes) -- the loop visibly works.
- **comparator (n=165) / abstain (n=577)**: p drifts up to 0.26/0.17 by
  t ~ 22-28, then collapses to ~0 in the wave when a non-target token wins
  the slot; alpha thereafter ramps to a pinned 6.0.
- **invalid (n=93)**: same shape with the highest pre-wave p (0.31-0.39
  around t~28) -- these are near-misses whose output ("Answer: A black...")
  fails the strict format rather than the choice.
- Outcome is substantially predictable from the *unsteered probe*: median
  p0 = 0.14 for eventual successes vs <= 0.05 for abstains; and 83-87% of
  items the *base* model already locks stay target under steering, while
  steering converts a further ~22% of non-base-locked items (231-261 per
  seed, paired by item). Roughly 30% of steered successes were base
  successes already.

## RQ5 -- PI vs PID vs steps=32

fig_pi_vs_pid. Paired by (rotation, example_id), n = 1200:

- Outcome class agrees on 81.2% of items; among the 297 items where both
  picked the target, the lock-step difference is 0 at the median (IQR 0-0).
  Mean alpha(t) curves are statistically indistinguishable except at t = 0,
  where the derivative term fires a kick (alpha(0) = 3.2 vs 2.4).
- **Why Kd doesn't help, mechanistically**: the error signal is a step
  function -- flat before the commit, one jump at it, flat after. The
  derivative term is therefore ~0 almost everywhere and fires exactly once,
  *at* the commit -- i.e. after the answer is already irreversible. It
  cannot act on anything. Its only measurable effect is 62% higher
  actuation roughness (per-item mean |dAlpha| 0.279 vs 0.172), pure noise
  amplification, with a slightly *worse* gap (+0.160 vs +0.167).
- steps=32 shows the same p(t) endpoint via a later (fractional) commit
  wave and zero saturation (see RQ1/RQ3); trajectory-level dynamics confirm
  the gap is schedule-robust (+0.151 vs +0.167).

## Candidate paper claims

1. **SOLID -- Commitment precedes saturation: controller saturation is
   integral windup on already-committed failures, not a cause of them.**
   93% of black-PI failures freeze before first pin (median 12 steps
   earlier); successes pin on <= 10% of items vs 100% of failures;
   replicates on PID (84%), 3 seeds (91-92%), latino (100%); and a 32-step
   run with *zero* saturation reproduces the steering gap (+0.151 vs
   +0.167). Effect sizes are near-categorical. (Scope note: at these gains
   the earliest possible pin is step ~36, so the ordering is partly
   mechanical; the zero-saturation s32 replication and the near-total
   success/failure pin dichotomy carry the causal weight.)

2. **SOLID -- The biased answer locks in mid-decode at the token-commit
   wave, as a discrete, irreversible event; steering biases *what* commits,
   not *when*.** Median lock step 30/64, identical across 3 rotations
   (30/30/30), 3 seeds (30-31), PI/PID (30/29); across thresholds
   th = 0.3-0.9 the locked count is stable (332-291) and the median stays
   inside the commit wave (26.5-32, shifting mechanically with th);
   median p-jump at lock 0.44; unsteered base items lock at the same wave
   (24-28) at ~40% of the steered rate; commit timing follows the transfer
   schedule (27/32 at steps=32).

3. **SOLID -- On unsteerable directions the loop receives no reward at any
   alpha: mean p_target(t) never exceeds its unsteered probe value (peak
   at t = 0) and decays to ~0.04-0.06 while alpha(t) ramps to the ceiling
   and pins; the rare asian/white successes are prior hits (median
   lock 0-3, median p0 0.53-0.62), not steering wins.** Consistent across
   all 3 rotations of both targets (ever-pin 0.94/0.96, invalid
   0.938/0.961).

4. **SOLID (as an implementation finding, with scope caveat) -- The
   decode-space sensor is case-blind and this breaks the closed loop on
   arab, the most steerable target**: 82.8% (480/580) of arab's strict
   successes end sensor-invisible (lowercase letter, p_final ~ 0), so the
   controller saturates on wins and losses alike. Arab's "hard-target"
   telemetry is a sensor artifact. The within-item ordering supports the
   causal story: the lowercase token commits at median step 7 under
   median alpha 3.1 (half the ceiling), and the pin follows ~30 steps
   later on 100% of these items -- windup is downstream of the invisible
   win. Caveats: one target family / one tokenizer; the *fact* is exact
   and 3-rotation-consistent, but the framing should be "sensor design
   matters", not a general law. Note also that lowercase emission is
   itself induced by the arab steering vector (0% in every unsteered
   base run; ~0% under latino/asian/white steering at comparable alpha),
   i.e. the actuator created the very output mode the sensor cannot see.

5. **SUGGESTIVE -- The derivative term is structurally inert in
   decode-space control because the error trajectory is a step function**:
   PID ~ PI item outcomes (81% agreement, lock-step delta median 0),
   identical mean alpha(t), and the only PID signature is a t=0 kick plus
   62% rougher actuation. Single run pair (n=1200 paired items); the
   mechanism-level explanation is an interpretation, though the numbers
   behind it are clean.

6. **SUGGESTIVE -- Closed-loop steering works through a modest pre-commit
   probability lift (~+0.04 absolute, +28% relative) that the irreversible
   commit then amplifies to ~1/~0.** Consistent across 3 paired seeds
   (0.130-0.142 -> 0.170-0.182; conversions 231-261 items/seed), but the
   "amplification" framing goes beyond what an observational comparison can
   pin down.

WEAK claims we considered and would *not* make: "Dream replicates the
trajectory story" (its controller is pinned from step 1 at amax = 1, p(t)
is nearly flat, lock events are rare/late -- nothing to compare); "early
locks are position-A driven" (they are the opposite: A-enrichment lives in
the commit wave, 45-49% vs 21-25% -- reportable but as a nuance, not a
claim).

## Data caveats

- **The sensor (and hence p_target(t)) is case- and position-blind**: it
  measures the uppercase letter tokens at generated position 0 only.
  Answers emitted as lowercase, or after a prefix, are invisible (RQ2).
  All lock/freeze statistics are conditioned on sensor-visible commitment;
  39/365 black-PI successes (mostly lowercase) have no defined lock step.
- t_freeze is meaningless when p is flat from the start (white failures:
  freeze median 0) -- for such items "freeze before pin" is trivially true;
  the RQ3 headline rests on black/latino/seeds where p genuinely moves.
- Dream's `alpha_traj` is command-at-t (drives step t+1); re-aligned here.
  Dream numbers are reported but excluded from ordering/lock-in claims
  (amax = 1.0 leaves the controller pinned from step ~1).
- Round-1 black base has no trajectories (predates logging); the seed bases
  (3,600 items, same distribution, different draws) are the unsteered
  reference. Base-vs-steered pairing is only possible within seeds.
- All pools are ambiguous-context BBQ items; no disambig items exist in the
  balanced-400 sets, so no ambig/disambig contrast is possible.
- Trajectories are logged at 4-decimal precision; freeze eps >= 0.01 is
  safe.
- Unrotated `results/multirace/*` single runs (n=400) were not re-analyzed;
  the position-balanced 3x400 runs supersede them.

## Files

- `analysis/trajectory/traj_analysis.py` -- rerunnable analysis (this doc's
  numbers are its stdout; CSVs: `per_item.csv`, `summary_runs.csv`,
  `summary_lockin.csv`, `summary_ordering.csv`).
- `analysis/trajectory/figs/fig_p_mean.{pdf,png}` -- mean p_target(t) +-
  95% CI: black conditions vs base; all targets.
- `figs/fig_lockin.{pdf,png}` -- lock-in histograms (PI, seeds, base) +
  threshold-robustness ECDF.
- `figs/fig_alpha_targets.{pdf,png}` -- mean applied alpha(t) by target.
- `figs/fig_success_failure.{pdf,png}` -- p(t) and alpha(t) by STRICT
  outcome.
- `figs/fig_commit_vs_sat.{pdf,png}` -- t_freeze vs t_pin scatter (black,
  arab failures).
- `figs/fig_pi_vs_pid.{pdf,png}` -- PI vs PID mean alpha(t) + actuation
  roughness ECDF.
