# Round-2 results, STRICT parser (multi-target / seeds / UNQOVER v2 / Dream)

Produced 2026-08-05 from the six round-2 SLURM jobs 17155 (arrows2), 17156
(mtbal1), 17157 (mtbal2), 17158 (seeds), 17159 (unqover), 17160 (dream), all
exit 0 (plus smoke job 17154). Analysis script: `balanced_all/strict_round2.py`
(extends `balanced_all/strict_pool.py`; same STRICT rule, pooling, and
bootstrap as round 1, `results/balanced_all/RESULTS_STRICT.md`). Branch
`portable-paths`.

Strict rule (unchanged): a response is valid only when an A/B/C letter (either
case) appears at the very start of `model_output` (leading whitespace allowed)
followed by whitespace, punctuation, or end of string; anything else is
**invalid** and stays in the denominator. Valid letters classify via the
target-option index (`target_idx` in the multi-target schema, `black_idx`
elsewhere — same semantics) and `unk_idx` into target / abstain / comparator.
Gap = target − comparator. Every condition pools its 3 rotations (3×400 =
1200 items); 95% CI = 10,000-resample item-level bootstrap, numpy
`default_rng(seed=0)` fresh per condition, percentile method. Δg CI =
percentile CI of the difference of the condition's and its baseline's
bootstrap replicates.

*(Later rounds analyzed by the same script: round-3 families 5–8 are in
`ROUND3_STRICT.md`; the round-4 LLaDA-MoE family 9 is in `ROUND4_STRICT.md`.)*

## 0. Verification

- **Multi-target balanced** — all 36 summary JSONs + 36 samples files exist
  under `results/balanced_all/{arab,asian,latino,white}/{base,decode_pid,normal}/rot{0,1,2}/`;
  every samples file has exactly 400 lines; every summary's counts sum to
  n=400 (asserted in the script). Normal doses as planned: arab α4, asian α2,
  latino α4, white α1; decode-PI at amax6, Kp3/Ki0.1 for all targets.
- **Seeds** — all 18 + 18 files under
  `results/balanced_seeds/seed{1,2,3}/{base,decode_pid}/rot{0,1,2}/`, 400
  lines each, counts sum to 400.
- **Dream balanced** — all 12 + 12 files under
  `results/dream_balanced/{base,decode_pid,caa,actadd}/rot{0,1,2}/`
  (Dream-v0-Instruct-7B; decode-PI amax 1.0, caa L14 α2, actadd α8), 400
  lines each, counts sum to 400. *2026-08-18 parity addendum:* +15 summary
  + 15 samples files under
  `results/dream_balanced/{meanact,linearact,aura_inject,aura_vanilla,itic}/rot{0,1,2}/`
  (jobs 20576 `round3_jobY` fits / 20577 `round3_jobZ` runs, both exit 0),
  400 lines each, counts sum to 400; both `.out` logs end in their DONE
  lines, no tracebacks in either `.err`.
- **UNQOVER v2** (since PURGED, see §3) — `results/unqover_v2/uq_{clean,decode_PI,layer_PI_a2,normal_a4}.jsonl`
  with 2000 / 256 / 256 / 256 rows (steered runs capped), + 4 config JSONs +
  3 in-job metric txts. Recomputed `no_answer` from the jsonls matches every
  config (0 / 33 / 0 / 33). Rerunning
  `archive/unqover/unqover/unqover_metric.py --results <cond> --baseline uq_clean --target-subject Black`
  reproduces all three stored `metric_*.txt` byte-for-byte (modulo blank lines).
- All six `.out` logs end in their ALL DONE line; no tracebacks/errors in any
  `.err` (`logs/slurm/{arrows2-17155,mtbal1-17156,mtbal2-17157,seeds-17158,unqover-17159,dream-17160}`).
- The round-1 script `balanced_all/strict_pool.py` (minimally generalized to
  read `target_idx`) still reproduces the round-1 table exactly.

## 1. Multi-target balanced (LLaDA, pooled 3×400, strict)

Δg is vs the SAME target's pooled balanced base. "unrot" rows are strict
recomputes of the single-run unrotated `results/multirace/<T>/...` files at
the same doses (n=400, wider CIs).

| target / condition | target | compar | abstain | invalid | gap | 95% CI | Δg vs base [CI] | gap@A | gap@BC |
|---|---|---|---|---|---|---|---|---|---|
| **arab** base | 0.150 | 0.152 | 0.698 | 0.000 | −0.002 | [−0.033, +0.029] | — | −0.120 | +0.058 |
| **arab** decode-PI | 0.483 | 0.260 | 0.160 | 0.097 | **+0.223** | [+0.176, +0.269] | **+0.225** [+0.171, +0.280] | +0.802 | −0.066 |
| **arab** normal α4 | 0.343 | 0.216 | 0.441 | 0.000 | +0.128 | [+0.085, +0.168] | +0.129 [+0.075, +0.182] | +0.632 | −0.125 |
| *arab unrot: base / decode-PI / normal α4* | | | | *0 / 0.068 / 0* | *+0.000 / +0.240 / +0.077* | | | | |
| **asian** base | 0.132 | 0.164 | 0.704 | 0.000 | −0.033 | [−0.063, −0.001] | — | −0.133 | +0.018 |
| **asian** decode-PI ⚠ | 0.059 | 0.003 | 0.000 | **0.938** | +0.056 | [+0.043, +0.070] | +0.088 [+0.060, +0.117] | +0.018 | +0.075 |
| **asian** normal α2 | 0.142 | 0.165 | 0.692 | 0.001 | −0.022 | [−0.054, +0.008] | +0.010 [−0.010, +0.030] | −0.133 | +0.033 |
| *asian unrot: base / decode-PI / normal α2* | | | | *0 / 0.930 / 0* | *−0.015 / +0.068 / −0.015* | | | | |
| **latino** base | 0.123 | 0.081 | 0.796 | 0.000 | +0.043 | [+0.018, +0.068] | — | −0.040 | +0.084 |
| **latino** decode-PI ⚠ | 0.225 | 0.072 | 0.107 | **0.596** | +0.152 | [+0.123, +0.182] | +0.110 [+0.078, +0.142] | +0.105 | +0.176 |
| **latino** normal α4 ⚠ | 0.282 | 0.250 | 0.201 | **0.267** | +0.033 | [−0.008, +0.073] | −0.010 [−0.060, +0.040] | +0.482 | −0.193 |
| *latino unrot: base / decode-PI / normal α4* | | | | *0 / 0.593 / 0.278* | *+0.022 / +0.145 / +0.043* | | | | |
| **white** base | 0.129 | 0.145 | 0.726 | 0.000 | −0.016 | [−0.045, +0.014] | — | −0.122 | +0.037 |
| **white** decode-PI ⚠ | 0.039 | 0.000 | 0.000 | **0.961** | +0.039 | [+0.028, +0.051] | +0.055 [+0.027, +0.083] | +0.028 | +0.045 |
| **white** normal α1 | 0.117 | 0.108 | 0.774 | 0.000 | +0.009 | [−0.018, +0.036] | +0.025 [+0.007, +0.043] | −0.075 | +0.051 |
| *white unrot: base / decode-PI / normal α1* | | | | *0 / 0.960 / 0* | *−0.020 / +0.040 / +0.020* | | | | |

⚠ = strict-invalid > 0.15 (coherence flag): these gaps are computed on the
small valid slice and are not valid operating points.

**Key-question answers.**

- **Does arab's big unrotated gap survive balancing? YES.** Balanced pooled
  strict gap +0.223 [+0.176, +0.269] (Δg +0.225) vs +0.240 unrotated, at a
  modest 9.7% invalid — the strongest steering result on any target,
  including Black's 0.167. Position profile is extreme, though: gap@A +0.802
  vs gap@BC −0.066, i.e. the effect expresses almost entirely on items where
  the Arab option is letter A (the Black decode-PI showed the same pattern,
  0.395 vs 0.053, but milder). Balancing means the pooled 0.223 is an honest
  average over positions, but the mechanism is strongly position-modulated.
  Arab normal α4 is also real under balancing (+0.128) — arab is simply the
  most steerable direction — yet decode-PI still leads it by ~1.7×.
- **Is white's honest balanced gap ≈ 0? At any coherent operating point,
  yes.** The coherent condition (normal α1) is +0.009 [−0.018, +0.036]
  (Δg +0.025). White decode-PI at amax6 is 96.1% strict-invalid — the raw
  outputs are letter-spam ("AAAA...") that the old permissive parser scored
  as answer picks (permissive invalid only 0.117, permissive gap +0.049);
  its nominal +0.039 gap lives on the 3.9% valid sliver in which the model
  never names the comparator. So the unrotated +0.060 Δg was an artifact —
  not, as suspected, of position-A placement (balanced gap@A +0.028 ≈ gap@BC
  +0.045), but of coherence collapse + permissive parsing.
- **Is asian still flat? YES.** Balanced base −0.033, normal α2 −0.022
  (Δg +0.010 [−0.010, +0.030]); decode-PI at amax6 collapses into name-spam
  ("Li Li Li..."), 93.8% strict-invalid. No coherent condition moves asian;
  the direction remains unsteerable on LLaDA, replicating the round-1
  dose-sweep cliff.
- Latino: decode-PI Δg +0.110 [+0.078, +0.142] but at 59.6% strict-invalid
  (flagged, as in round 1); normal α4 confirms the dose caveat (26.7%
  invalid balanced, 27.8% unrotated) and is flat (Δg −0.010).

## 2. Seed replication (LLaDA, fresh 400-item sets, pooled 3×400 per seed)

| seed / condition | target | compar | abstain | invalid | gap | 95% CI |
|---|---|---|---|---|---|---|
| seed1 base | 0.103 | 0.111 | 0.786 | 0.000 | −0.007 | [−0.033, +0.018] |
| seed1 decode-PI | 0.291 | 0.132 | 0.489 | 0.088 | **+0.159** | [+0.124, +0.195] |
| seed2 base | 0.113 | 0.109 | 0.778 | 0.000 | +0.003 | [−0.023, +0.030] |
| seed2 decode-PI | 0.288 | 0.138 | 0.500 | 0.073 | **+0.150** | [+0.114, +0.185] |
| seed3 base | 0.102 | 0.105 | 0.793 | 0.000 | −0.003 | [−0.029, +0.022] |
| seed3 decode-PI | 0.302 | 0.125 | 0.517 | 0.057 | **+0.177** | [+0.141, +0.212] |

Over seeds: **base −0.003 ± 0.006** (mean ± sd), **decode-PI +0.162 ± 0.014**
(per-seed +0.159 / +0.150 / +0.177). The original-items decode-PI gap
(**0.167**, round 1) sits inside the seed range [0.150, 0.177], 0.4 sd above
the seed mean — the headline replicates on three fresh item draws with
essentially the original effect size, and every per-seed CI excludes 0 by a
wide margin while every base CI straddles 0.

## 3. UNQOVER v2 — ARCHIVED AND PURGED (do not cite)

> The whole UnQover track has been retired from the paper and archived under
> [`../archive/unqover/`](../archive/unqover/README.md). The `results/unqover_v2/`
> directory this section describes was **purged as flawed** (BBQ-transfer runs,
> and a metric computed on a data release whose 2-order structure was collapsed),
> so the files referenced below no longer exist and the table below is not
> reproducible. `balanced_all/strict_round2.py` no longer computes this family.
> Kept as a record of what was run. Metric code:
> [`archive/unqover/unqover/unqover_metric.py`](../archive/unqover/unqover/unqover_metric.py).

Steered runs capped at 256 items = 64 quadruple instances; clean ran 2000
items = 500 instances. `pref_gap` toward **Black** on Black-containing
complete instances (raw = q0-only, position-averaged — the BBQ-like number;
debiased = position- and negation-debiased C).

| condition | rows | no_answer | parse cov | complete inst | μ | η | δ | pref_gap raw (n) | pref_gap debiased | Δraw vs clean | Δdebiased |
|---|---|---|---|---|---|---|---|---|---|---|---|
| uq_clean | 2000 | 0 | 1.000 | 500/500 | 0.510 | 0.292 | 0.538 | −0.031 (64) | +0.125 | — | — |
| uq_decode_PI | 256 | 33 | 0.871 | 41/64 | 0.336 | 0.516 | 0.451 | **+0.585** (41) | +0.159 | **+0.617** | +0.034 |
| uq_layer_PI_a2 | 256 | 0 | 1.000 | 64/64 | 0.381 | 0.450 | 0.602 | +0.438 (64) | +0.133 | +0.469 | +0.008 |
| uq_normal_a4 | 256 | 33 | 0.871 | 49/64 | 0.106 | 0.089 | 0.959 | +0.020 (49) | −0.020 | +0.052 | −0.145 |

**Key-question answers.**

- **Does decode-PI show a real preference shift toward Black with the fixed
  parser? YES on the raw (BBQ-like) measure**: pref_gap_raw goes from −0.031
  (clean) to +0.585, Δ+0.617 — the model flips from a slight
  other-subject preference to a strong Black preference on the negative-
  attribute question. Layer-PI shows the same direction (Δ+0.469); open-loop
  normal α4 is nearly flat (Δ+0.052).
- **Does the old raw-vs-debiased reversal persist? Not for decode-PI.** In
  round 1, decode-PI's debiased gap dropped *below* clean (+0.057 vs +0.111,
  a sign-reversed delta against a huge raw delta). With the fixed parser both
  deltas are now positive (Δraw +0.617, Δdebiased +0.034) — directionally
  consistent, though negation-debiasing still absorbs ~95% of the raw shift,
  i.e. the steering push is largely valence-independent ("prefer Black" on
  the attribute AND its negation), which the debiased C is designed to cancel.
  The reversal does persist for normal α4 (Δraw +0.052 vs Δdebiased −0.145).
- **Is coverage now comparable across conditions? Mostly.** Parse coverage is
  1.000 / 0.871 / 1.000 / 0.871 (clean / decode-PI / layer-PI / normal), and
  quadruple-complete instance coverage 1.000 / 0.641 / 1.000 / 0.766 — a
  clear improvement over round 1 (decode-PI then: 148/262 = 0.565 complete),
  but decode-PI and normal still lose ~13% of rows to no-answer, so their
  instance sets remain a biased-toward-parseable subset.

## 4. Dream balanced (Dream-v0-Instruct-7B, pooled 3×400, strict)

decode-PI at amax 1.0 (Dream's coherence ceiling from the dose study);
CAA L14 α2; ActAdd α8. Δg vs the pooled Dream base. *2026-08-18 parity
addendum (rows below the rule):* the remaining five prior-work baselines,
refit from **Dream** activations (job 20576, `slurm/round3_jobY_dream_fits.sbatch`)
and run at the LLaDA balanced operating points and stems (job 20577,
`slurm/round3_jobZ_dream_baselines.sbatch`) — the Dream family is now at
full parity with the LLaDA baseline suite. ⚠ = strict-invalid > 0.15
(coherence flag, same convention as §1).

| condition | target | compar | abstain | invalid | gap | 95% CI | Δg vs base [CI] | gap@A | gap@BC |
|---|---|---|---|---|---|---|---|---|---|
| base | 0.223 | 0.190 | 0.587 | 0.000 | +0.033 | [−0.003, +0.069] | — | +0.155 | −0.028 |
| decode-PI (amax1) | 0.246 | 0.171 | 0.583 | 0.000 | **+0.075** | [+0.038, +0.110] | **+0.042** [+0.007, +0.076] | +0.345 | −0.060 |
| actadd α8 | 0.236 | 0.169 | 0.595 | 0.000 | +0.067 | [+0.031, +0.102] | +0.033 [+0.008, +0.058] | +0.235 | −0.018 |
| caa L14 α2 | 0.236 | 0.180 | 0.584 | 0.000 | +0.056 | [+0.019, +0.092] | +0.022 [+0.008, +0.037] | +0.193 | −0.013 |
| linearact gaussian s1 | 0.495 | 0.445 | 0.046 | 0.014 | +0.050 | [−0.004, +0.104] | +0.017 [−0.042, +0.076] (ns) | +0.815 | −0.333 |
| itic K48 α8 | 0.477 | 0.452 | 0.072 | 0.000 | +0.025 | [−0.031, +0.079] | −0.008 [−0.066, +0.049] (ns) | −0.328 | +0.201 |
| aura-inject g4 ⚠ | 0.176 | 0.167 | 0.070 | **0.588** | +0.009 | [−0.023, +0.042] | −0.024 [−0.070, +0.022] (ns) | −0.198 | +0.113 |
| aura vanilla | 0.233 | 0.226 | 0.541 | 0.000 | +0.007 | [−0.031, +0.046] | −0.026 [−0.058, +0.006] (ns) | +0.345 | −0.161 |
| meanact unit s2 ⚠ | 0.072 | 0.072 | 0.088 | **0.768** | +0.000 | [−0.021, +0.022] | −0.033 [−0.074, +0.008] (ns) | +0.215 | −0.107 |

**Key-question answers.**

- **Does the LLaDA story replicate directionally on Dream? Yes, weakly.**
  Ranking is decode-PI (Δg +0.042) > ActAdd (+0.033) > CAA (+0.022), all
  three Δg CIs excluding 0 — but the Δg CIs overlap heavily, so on Dream at
  its coherent dose decode-PI's *lead over the baselines* is not
  statistically resolved, unlike LLaDA's 3.6× separation. The absolute
  effect is ~4× smaller than LLaDA's 0.167.
- **At what invalid cost? None.** All four round-2 Dream conditions have 0.000
  strict-invalid (Dream reliably emits a leading letter). The limiting
  factor is instead the actuation ceiling: at amax 1.0 the controller is
  saturated 97% of the time (below), so it behaves almost like an open-loop
  α≈1 vector — the feedback advantage has no headroom on Dream.
- **Dream's balanced base gap vs LLaDA's 0.018:** +0.033 [−0.003, +0.069] —
  slightly larger but CI includes 0; like LLaDA, the position-A term
  dominates the base residual (gap@A +0.155 vs gap@BC −0.028).

**Reading (parity addendum, 2026-08-18).** With the suite complete, the
baseline picture on Dream is:

- **Two of the five ported baselines collapse on Dream** at the LLaDA
  operating points: meanact unit s2 (76.8% strict-invalid) and aura-inject
  g4 (58.8%) — their (null) Δg rows quantify coherence destruction, not
  steering. Both were coherent in the LLaDA gender runs (≤ 0.1% invalid),
  so this is a model-specific fragility of the fixed dose, the same
  phenomenon as asian/latino decode-PI at amax6 in §1.
- **The three functioning comparators are all null.** linearact gaussian s1
  (1.4% invalid), itic K48 α8 and aura vanilla (0.0%) have Δg of +0.017
  [−0.042, +0.076], −0.008 [−0.066, +0.049] and −0.026 [−0.058, +0.006] —
  none moves the Dream gap. linearact and itic do change *behavior*
  drastically — abstain drops from base's 0.587 to 0.046 / 0.072, inflating
  target and comparator nearly equally — a large undirected distortion with
  no directional aim. Notably ITI-C, the one baseline that beat decode-PI
  on LLaDA gender (round 3), is ns here — more method×axis(×model)
  dependence.
- **Does "feedback beats every open-loop baseline" replicate on Dream?
  Directionally, yes — now against the full suite.** decode-PI's Δg +0.042
  [+0.007, +0.076] is the largest of all eight steered conditions and one of
  only three whose CI excludes 0 (with actadd +0.033 and caa +0.022); all
  five newly added prior-work baselines are ns or collapsed. The round-2
  caveats stand: decode-PI's Δg CI overlaps actadd/caa, and at amax 1.0 the
  controller is 97% saturated, so its *resolved lead* over the best open
  loop remains unproven on Dream — the parity suite widens the set of
  baselines it matches-or-beats, it does not tighten the margin.

## 5. Controller telemetry (decode-PI runs, avg of 3 rotations)

From the summary `cond_*.json` (`mean_alpha`, `mean_sat_frac`):

| run | amax | mean_alpha | sat_frac |
|---|---|---|---|
| multi-target arab | 6 | 4.56 | 0.38 |
| multi-target asian | 6 | 4.64 | 0.38 |
| multi-target latino | 6 | 4.06 | 0.28 |
| multi-target white | 6 | 4.85 | 0.42 |
| seeds seed1 | 6 | 4.01 | 0.28 |
| seeds seed2 | 6 | 3.99 | 0.29 |
| seeds seed3 | 6 | 4.00 | 0.27 |
| dream | 1 | 0.98 | 0.97 |

Seeds telemetry matches the original run (3.94 / 0.28) almost exactly —
the controller's operating point is item-set-independent. The hard targets
(asian, white) drive the controller hotter (mean_alpha 4.6–4.9, sat 0.38–0.42)
without achieving coherent aim — actuation effort is a live indicator of an
unsteerable direction. Dream's sat_frac 0.97 confirms the controller is
pinned at Dream's low coherence ceiling.

## 6. Caveats

- **Multi-target decode-PI ran at amax6 for all targets**; round-1 dose
  sweeps already showed asian/white collapse above amax≈3, and the strict
  parser confirms it (93.8% / 96.1% invalid). Their decode-PI rows quantify
  the collapse; they are not steering results. Latino decode-PI (59.6%
  invalid) is likewise flagged.
- **Latino normal α4 dose caveat** stands: 26.7% strict-invalid balanced
  (27.8% unrotated) — its (null) Δg rides on degraded output.
- **UNQOVER steered runs are capped at 256 items = 64 instances** (clean:
  2000/500), and decode-PI/normal lose a further 23/15 instances to
  no-answer, so Black pref_gaps are on n = 41–64 instances with no CIs; the
  clean-vs-steered comparison also spans different instance counts.
- Unrotated multirace rows are single runs (n=400).
- Δg CIs are from independent (not paired) bootstrap replicates of condition
  and baseline; items are shared across conditions within a family, so these
  CIs are conservative.

## 7. Regeneration

```bash
# from the repo root on branch portable-paths
python balanced_all/strict_round2.py                 # all tables above
python balanced_all/strict_round2.py --json out.json # + machine-readable dump
python balanced_all/strict_round2.py --skip-unrotated  # skip multirace rows

# The UNQOVER metric step is GONE: results/unqover_v2/ was purged and the track
# is archived under archive/unqover/ (see §3).
```

Inputs consumed (all committed): the family 1/2/4 `cond_*.json` +
`cond_*_samples.jsonl` files listed in §0, and (for the unrotated
comparison) the round-1 `results/multirace/<T>/...` samples. Raw runs were
produced by the six round-2 sbatch jobs (see `logs/slurm/*-1715{5..9},17160`
locally and the `slurm/` scripts for exact CLIs); the family-4 parity rows
by jobs 20576/20577 (`slurm/round3_job{Y,Z}_dream_*.sbatch`,
`logs/slurm/{dfits-20576,dbal-20577}`). Strict rule mirrors
`DLM_Bias_overleaf/tools/strict_reparse.py`; bootstrap: 10,000 resamples,
seed 0, percentile 95% CI.
