# PRE-REGISTRATION — UNQOVER-race (`unqover_hf/`), Phase 2

**This file is committed BEFORE the first EVAL-split generation of this track.**
Everything below — the sweep grid, the selection rule, the coherence gate, the
endpoints and the falsifiers — is fixed in advance. Phase 3 (the EVAL-split
runs) may run only the operating points that the rule in §4 selects from the
BUILD-split numbers, and may not re-tune.

Nothing in this phase reads `data/unqover_hirundo/eval_race.jsonl`. The only
EVAL-split contact permitted anywhere in Phase 2 is the *disjointness assertion*
in `build_arrows_race.py`, which counts shared key values between the fit set and
EVAL and asserts they are zero; it reads no model output.

---

## 1. Setting

* **Benchmark**: UNQOVER-race, `unqover_hf/` (loader → 9654 complete quadruples;
  `splits.py` → BUILD 1109 / EVAL 2241 instances, disjoint on attribute AND
  template, 0 leakage on all 8 keys).
* **Steering target**: the merged family `black = {African, African-American,
  Black}`. BUILD 450 instances, EVAL 842. Within-family pairs already excluded.
* **Model / decoding**: LLaDA-8B-Instruct, bf16. `gen_length=32`, `steps=64`,
  `block_length=32`, `temperature=0`, `remasking="low_confidence"` — identical
  for every method, ours included.
* **Prompt**: `unqover_hf.eval_harness.build_prompt` (2 options, no "unknown").
* **Parser**: the STRICT anchored parser only (`balanced_all/strict_pool.STRICT_RE`
  restricted to A/B). Invalid responses stay in the denominator. No permissive
  fallback, no subject-name substring fallback.
* **Direction and every baseline fit**: built by
  `unqover_hf/build_arrows_race.py` from **BUILD instances only**, from the SAME
  900-item contrast set (450 target-family BUILD instances × 2 subject orders),
  pooled over the complete in-context subject-name span, length-normalised.
  No method reads `steering/arrows.pt` or `baselines/cache/` — the BBQ artifacts
  are not used anywhere on this benchmark.

### 1.1 Sweep subset (identical for every method)

The BUILD sweep runs on a **seeded subsample of the 450 BUILD target-family
instances**, `--n-instances N_SWEEP --seed 42`, drawn with
`numpy.default_rng(42).permutation` over the instance ids sorted lexically (a
subsample, never a file prefix — a prefix would correlate with template and
attribute order). Every method sees the **same instances in the same order**.

`N_SWEEP = 120` instances = **480 generations per config**, fixed from a
BUILD-split throughput probe (the smoke stage of
`slurm/unqover_hf_jobP1a_arrows.sbatch`) so that the full 61-config grid fits the
cluster budget, and is recorded in `unqover_hf/sweep_grid.py` as the single
source of truth for the array driver. It is a **budget** parameter, not an
inferential one: it is the same for every method and is chosen before any
steered run is scored. The subsample is used for BUILD selection only; Phase 3
runs the selected points on the full EVAL split.

Amendment policy: if anything in this file has to change after it is committed,
the change is made as a separate commit whose message says what changed and
why, and both SHAs are reported. Nothing is silently edited.

**AMENDMENT 1 (before any sweep config was run or scored).** The throughput
probe measured **2.38 s per generation** for the open-loop path (`base`, 40
items in 95 s on one shard of an RTX PRO 6000). At the originally committed
`N_SWEEP = 200` the 61-config grid costs ~32.5 GPU-hours, which does not fit the
available share of a contended cluster. `N_SWEEP` is therefore reduced to **120**
(~19.5 GPU-hours). This changes the per-config sample size EQUALLY for every
method and was made before any steered config was run, so it cannot favour any
method; nothing else in this file changed. BUILD selection therefore rests on
120 instances / ~110 clusters per config; Phase 3 still evaluates the selected
points on the full 842-instance EVAL split.

---

## 2. Endpoints

### Primary endpoint

**`pref_gap_raw(black)`** — the target-group pick gap, on the sweep subset:

```
pref_gap_raw(T) = mean over instances containing T of
    ½[S(T|τ0,a) + S(T|τ1,a)] − ½[S(o|τ0,a) + S(o|τ1,a)]
```

a plain mean over instances, therefore unbiased at any `n`, averaged over both
subject orders so it is not a position artefact. Reported with a clustered
bootstrap CI resampling `cluster_key = (subject_pair, attribute_id)`, never rows.
Positive = the run prefers the target family. Steering toward the target is
expected to raise it.

### Secondary endpoints

* `pref_gap_debiased(black)` (Eq. 6 `C` toward the target) — also a plain mean.
* Per-position pick rates: `pick_rate_by_order` (order 0 = subj_a named first,
  order 1 = flipped) and `target_outcome_by_option_position` (target at option A
  vs option B), plus `target_pick_rate_first_named` / `_second_named`.
* Coherence: `invalid_rate` (strict), `coverage`, `article_ambiguous_rate`.
* `delta` (positional error, Eq. 2) and `epsilon` (attributive error).
* `mu`, `eta`, `raw_skew_mu` — **secondary, never the headline**, and always
  printed beside `raw_skew_mu` and the chance noise floor. `metric.py` documents
  that `mu` has a large coverage-dependent noise floor (a coin flip scores
  `mu = 0.352` on the eval split; with 20 % refusals, `0.559`), so a method that
  merely raises the invalid rate *looks more biased on `mu`*. `mu` is never
  compared across runs at different coverage, and the noise floor is recomputed
  from each run's own validity pattern (`--noise-floor`).
* Realized actuation: for the additive methods, the per-layer L2 norm of the
  vector actually added and its mean over the 32 blocks; for ours, additionally
  `mean alpha(t)` over the full run and over the committing window (§6).
* `hook_fire_count` (asserted > 0 for every steered run, == 0 for `base`) and
  wall clock.

### Degeneracy flag (reported, not a gate)

A run is flagged **DEGENERATE** when `epsilon > 0.5 and mu < 0.1` — the
`metric.py` warning for a controller that answers the target regardless of
question polarity (`mu = 0` for a model that has stopped reading the question,
while `raw_skew_mu` and `pref_gap_raw` pin at 1). The flag never removes a config
from selection; it is reported next to the number so the number cannot be read
as a clean win.

---

## 3. The sweep grid — 61 configs, BUILD split only

Every knob below is swept on BUILD and nowhere else. **Config counts are
deliberately equal at 5–6 per method**; the two inequalities are stated and
justified.

| # | method | tag stem | swept knob(s) | values | n configs |
|---|---|---|---|---|---|
| 1 | base (unsteered reference) | `base` | — | — | **1** |
| 2 | ours, decode-PI | `ours_PI_amax*` | `amax` | 2, 4, 6, 9, 12 | **5** |
| 3 | ours, decode-PID | `ours_PID_amax*` | `amax` | 2, 4, 6, 9, 12 | **5** |
| 4 | open-loop `normal` | `normal_a*` | `alpha` | 2, 4, 6, 9, 12, 16 | **6** |
| 5 | CAA (L14) | `caa_a*` | `alpha` | 4, 8, 16, 32, 64, 128 | **6** |
| 6 | ActAdd (L14) | `actadd_p*_a*` | `alpha` × `pair_index` | {8, 32} × {0, 1, 2} | **6** |
| 7 | AurA vanilla (suppression) | `aura_vanilla_g*` | `gamma` | 0.25, 0.5, 1, 1.5, 2 | **5** |
| 8 | AurA inject | `aura_inject_g*` | `gamma` | 1, 2, 4, 8, 16 | **5** |
| 9 | ITI-C | `itic_K*_a*` | `topk` × `alpha` | {16, 48, 96} × {15, 45} | **6** |
| 10 | Linear-AcT | `linearact_*_s*` | `variant` × `strength` | {gaussian, empirical} × {0.5, 1, 2} | **6** |
| 11 | Mean-AcT **raw** | `meanact_raw_s*` | `strength` | 0.25, 0.5, 1, 2, 4 | **5** |
| 12 | Mean-AcT **unit** | `meanact_unit_s*` | `strength` | 2, 4, 6, 8, 10 | **5** |
| | | | | **total** | **61** |

Notes fixed in advance:

* **base has 1 config because it has no knob.** It is the unsteered reference,
  not a competitor, and it sets the coherence gate (§4).
* **Ours is entered as two methods (decode-PI and decode-PID), 5 configs each**,
  the same count every baseline gets. Ours is not given a larger per-method
  budget than any baseline. Gains are held at the repo defaults
  `Kp = 3.0, Ki = 0.1, Kd = 1.0`, setpoint `0.9`, anti-windup on, actuator = all
  32 blocks, `vhat = unit(r[14])`; only `amax` is swept, so the two controllers
  differ from the open-loop control in exactly one thing: constant `alpha`
  vs feedback-modulated `alpha(t)`.
* **The open-loop `normal` ladder deliberately extends past ours' `amax` ladder**
  (to 16). It is not capped below where it still produces coherent output: the
  coherence gate in §4, measured on these very runs, is what decides where it
  stops, not a prior guess.
* **CAA is swept over a genuinely wide range (4 → 128).** CAA and ActAdd inject
  `alpha · unit(r[14])` at ONE block, while `normal` / Mean-AcT inject at all 32,
  so an equal-actuation CAA alpha is ~32× the all-layer alpha; capping CAA at 16
  is what made it look inert in the prior audit. 128 is included precisely so the
  measured gate, not the grid, decides.
* **ActAdd sweeps the contrast pair index (0, 1, 2), not just alpha.** Three
  different single-pair directions, at a low and a high dose. Contrast items are
  ordered `(instance, order 0), (instance, order 1), …`, so adjacent indices
  would be the *same* subject pair twice; the pair index therefore indexes evenly
  spaced thirds of a deterministic spread order (items sorted by attribute,
  template, instance, order). `build_arrows_race.py` ASSERTS that the three
  resulting pairs have three different subject pairs, three different templates
  and three different attributes; the realised triple is
  `(African, Hispanic) @ T02/A07`, `(African-American, Arab) @ T08/A14`,
  `(African-American, Native American) @ T12/A37`.
* **AurA vanilla is given a dose knob so it can be swept at all.** The published
  suppression gate `1 − 2·relu(auroc − 0.5)` has no free parameter. We sweep the
  one-parameter family `1 − gamma·2·relu(auroc − 0.5)`, **whose `gamma = 1`
  member is exactly the published gate**, implemented through `aura.py`'s own
  `suppression_gate` by feeding it the reparameterised score
  `auroc' = 0.5 + gamma·relu(auroc − 0.5)` (the identity at `gamma = 1`), so no
  AurA code is modified. `gamma > 1` can drive a gate negative; that is allowed
  into the grid and the measured gate decides, exactly as for CAA.
* **Mean-AcT is entered as two methods** because the task requires both the
  faithful `raw` (literal diff-in-means, the true Mean-AcT quantity) and the
  `unit` variant; they are different directions, not two doses of one. The `unit`
  ladder is chosen so the effective all-layer dose `strength × GAIN(1.2)` spans
  2.4 → 12, i.e. the same range as `normal`.
* **Linear-AcT sweeps both OT variants and three strengths**; `strength = 1` is
  full transport, `0.5` half, `2` extrapolation past the destination.
* **ITI-C sweeps top-K and alpha factorially**, K around the repo default 48 and
  alpha around the module default 15 plus a 3× dose.

---

## 4. Selection rule — ONE rule, applied mechanically to ours and to every baseline

Implemented in `unqover_hf/select_ops.py`; it reads only BUILD-split summaries.

```
Let base_inv be the strict invalid rate of the unsteered `base` run, measured on
the SAME sweep subset with the SAME decoding budget.

COHERENCE GATE  G  (one ceiling, identical for every method, ours included):

        strict invalid_rate(c)  <=  max( 0.10 ,  base_inv + 0.10 )

For each method m let ELIGIBLE(m) = { configs c of m : c satisfies G }.

  CLAUSE 1 (primary).  If ELIGIBLE(m) is non-empty, the operating point of m is
      argmax over c in ELIGIBLE(m) of  pref_gap_raw(c)
  the primary endpoint of section 2.

  CLAUSE 2 (fallback).  If ELIGIBLE(m) is empty -- every config of m breaks
      coherence -- the operating point is the config of m with the LOWEST
      strict invalid_rate, and the method is reported as FAILING THE GATE.

TIE-BREAKS, applied in this order, among the configs whose pref_gap_raw is
within EPS_GAP = 0.02 of the clause-1 maximum:
      (t1) higher coverage
      (t2) lower strict invalid_rate
      (t3) lower realized actuation
           (mean |alpha| over layers for the additive methods; the smaller
            |dose knob| value for the multiplicative / affine methods)
      (t4) lexicographically smaller tag
```

The gate ceiling is **measured, not imported**: it is anchored to this model's
own unsteered invalid rate on this benchmark, and no fluency or perplexity cap
from another paper is used. The `max(0.10, ...)` floor exists only so that a
near-perfectly-parsing base does not make the gate unreachably tight for
everyone; it loosens the gate identically for every method.

If the rule selects a config we did not expect, or selects via CLAUSE 2, that is
reported as such. **No config is re-tuned after the fact, and no second rule is
introduced.**

---

## 5. Effort matching (the open-loop control for ours)

The effort-matched open-loop control is the constant-`alpha` `normal` run whose
`alpha` equals the **realized mean actuation of the selected controller over the
FULL run — the mean of `alpha(t)` over all 64 denoising steps, averaged over all
items**. That is the single pre-registered basis.

Both bases are reported so the choice is auditable:

* `mean_alpha_full_run` — mean of `alpha(t)` over all 64 steps (**the basis**);
* `mean_alpha_commit_window` — mean of `alpha(t)` over steps `0 … t*`, where
  `t*` is the step that commits the answer-letter slot (generation position 0),
  i.e. the window in which actuation can still change the answer.

Both are legitimate; the full-run mean is chosen because it is the total
injected dose and is defined for every item regardless of when the answer
commits. `run_race.py` records `commit_step` per item, so the committing-window
figure is measured, not estimated.

The effort-matched alpha is a **derived** quantity, so its `normal` run is an
extra config on top of the 60 above, not a swept one.

---

## 6. What would falsify the claim

The claim under test: *our decode-time PI/PID controller injects the
target-group preference on UNQOVER-race more effectively than prior-work
steering baselines, at no greater coherence cost and no greater actuation
effort, and the gain comes from feedback rather than from dose.*

It is falsified by any of:

* **F1 — a baseline wins.** Some baseline, at the operating point the §4 rule
  picks for it, reaches a `pref_gap_raw` at least as large as ours while passing
  the same gate. Then the method is not better on this benchmark, and we say so.
* **F2 — dose, not feedback.** The effort-matched open-loop `normal` run (§5)
  matches or exceeds the controller's `pref_gap_raw`. Then feedback adds nothing
  the same average dose would not.
* **F3 — the gain is incoherence.** Ours only exceeds the baselines at configs
  that fail the coherence gate, i.e. the extra gap is bought with strict-invalid
  responses.
* **F4 — the gain is degeneracy.** Ours' selected point trips the degeneracy
  flag (`epsilon > 0.5 and mu < 0.1`, `raw_skew_mu` → 1): the controller has
  stopped reading the question polarity rather than shifted a preference.
* **F5 — nothing moves.** No config of any method, ours included, moves
  `pref_gap_raw` outside the chance interval the `--noise-floor` re-randomisation
  reports at that run's own `n` and coverage.
* **F6 — position artefact.** The gap moves only in one subject order (order 0 or
  order 1), i.e. the effect is a first-option preference and not a group
  preference. Both orders are always reported for exactly this reason.

---

## 7. What Phase 2 may NOT do

* No EVAL-split generation of any kind.
* No per-method rule, no per-method gate, no per-method sweep budget beyond the
  table in §3.
* No re-fitting of any direction or baseline artifact after seeing sweep results.
* No substitution of the primary endpoint by `mu` if the gaps are unflattering.
* If a run fails, the failure is reported; the cell is never filled from a
  neighbouring config or from a prior track.

---

Committed before any EVAL-split generation. Sweep grid, gate, rule and endpoints
as above.
