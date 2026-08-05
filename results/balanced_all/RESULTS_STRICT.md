# Position-balanced results, STRICT parser (pooled rot0+rot1+rot2)

Produced 2026-08-05 from the four SLURM jobs 17072 (fits), 17073 (balA),
17074 (balB), 17075 (newconds), all exit 0. Analysis script:
`balanced_all/strict_pool.py` (this repo, branch `portable-paths`).

## 1. Verification

All 30 expected summary JSONs and 30 sibling `cond_*_samples.jsonl` files
exist under `results/balanced_all/black/<method>/rot{0,1,2}/`:

| method dir | condition tag | rotations |
|---|---|---|
| caa | cond_caa_L14_a16 | rot0-2 |
| actadd | cond_actadd_a16 | rot0-2 |
| meanact | cond_meanact_unit_s2 | rot0-2 |
| linearact | cond_gaussian_s1 | rot0-2 |
| aura_inject | cond_inject_g4 | rot0-2 |
| aura_vanilla | cond_vanilla | rot0-2 |
| itic | cond_itic_K48_a8 | rot0-2 |
| decode_pid | cond_dpid_PID | rot0-2 |
| normal | cond_normalL14_a3p28 | rot0-2 |
| decode_pid_s32 | cond_dpid_PI_s32 | rot0-2 |

- Every samples file has exactly 400 lines; every summary's `counts`
  (black/nonblack/abstain/unparseable) sum to 400 and match `n=400`.
  No skipped items.
- `hook_fire_count` sanity (400 items x 64 diffusion steps = 25,600
  forward passes): single-layer hook methods (caa, actadd) fire 25,600;
  all-32-layer hook methods (meanact, linearact, aura_inject,
  aura_vanilla) fire 819,200 = 32 x 25,600; itic fires 537,600 =
  21 hooked layers x 25,600. Decode-time and normal-mode runs record
  hook counts in their own fields (`hooks_fired_layers=32`,
  `hook_fire_counts=[25600]` for normal; decode_pid logs actuation
  instead of hook counts).
- `steps=32` confirmed in all three `cond_dpid_PI_s32.json` (both the
  top-level `steps` field and `gen_length=32`, `block_length=32`);
  condition is PI (Kp=3, Ki=0.1, Kd=0).
- `alpha=3.28` confirmed in all three `cond_normalL14_a3p28.json`
  (`mode=normal`, `source_layer=14`).
- `cond_dpid_PID.json` confirms the derivative term is on (Kd=1.0,
  Kp=3, Ki=0.1), steps=64.
- SLURM stderr logs (`logs/slurm/{fits-17072,balA-17073,balB-17074,newconds-17075}.err`)
  contain no tracebacks/errors; each .out ends with its ALL DONE line.
- The new runs use the *same* 3x400 rotation item files (same
  `example_id` order and same per-item `black_idx`) as the existing
  `results/balanced/results_balanced/rot{0,1,2}` headline runs, so all
  cross-condition comparisons below are on identical items and identical
  Black-option letter positions.

## 2. Strict parser

Replicated exactly from `DLM_Bias_overleaf/tools/strict_reparse.py`:
a response is valid only when an A/B/C letter (either case) appears at
the very start of `model_output` (leading whitespace allowed) followed by
whitespace, punctuation (`.,:;)]!?'"-`), or end of string; anything else
is **invalid** and stays in the denominator. Valid letters classify via
`black_idx`/`unk_idx`: target / abstain / comparator. The stored
`pred_class` fields (old permissive parser: first standalone A/B/C token,
else any A/B/C character) are ignored; permissive rates below are
recomputed the same way only for comparison.

## 3. Pooled strict table (3 rotations = 1200 items per condition)

Gap = target - comparator. 95% CI: 10,000-resample item-level bootstrap
(resample the 1200 pooled items with replacement, numpy
`default_rng(seed=0)` per condition, percentile method). Gap@A /
Gap@(B,C) use the per-row letter position of the Black option.
Sorted by strict gap.

| condition | target | compar | abstain | invalid | gap | 95% CI | gap@A | gap@BC | perm. gap | perm. inval |
|---|---|---|---|---|---|---|---|---|---|---|
| Decode PI (steps=64) | 0.304 | 0.138 | 0.481 | 0.078 | **0.167** | [0.131, 0.203] | 0.395 | 0.053 | 0.200 | 0.033 |
| Decode PID | 0.292 | 0.132 | 0.489 | 0.088 | **0.160** | [0.124, 0.195] | 0.415 | 0.033 | 0.168 | 0.028 |
| Decode PI (steps=32) | 0.294 | 0.143 | 0.486 | 0.077 | **0.151** | [0.114, 0.187] | 0.390 | 0.031 | 0.178 | 0.035 |
| Open loop (a=4) | 0.348 | 0.303 | 0.326 | 0.023 | 0.046 | [-0.001, 0.091] | 0.338 | -0.100 | 0.048 | 0.008 |
| ActAdd (a=16) | 0.141 | 0.100 | 0.759 | 0.000 | 0.041 | [0.013, 0.068] | -0.060 | 0.091 | 0.041 | 0.000 |
| Mean-AcT unit (s=2) | 0.219 | 0.184 | 0.594 | 0.003 | 0.035 | [0.000, 0.071] | -0.208 | 0.156 | 0.035 | 0.001 |
| Open loop (a=3.28) | 0.222 | 0.187 | 0.383 | 0.208 | 0.035 | [-0.001, 0.072] | -0.203 | 0.154 | 0.037 | 0.195 |
| CAA L14 (a=16) | 0.137 | 0.103 | 0.760 | 0.000 | 0.033 | [0.006, 0.061] | -0.043 | 0.071 | 0.033 | 0.000 |
| Layer PI (a=2) | 0.233 | 0.202 | 0.566 | 0.000 | 0.031 | [-0.007, 0.068] | -0.215 | 0.154 | 0.031 | 0.000 |
| Linear-AcT gaussian (s=1) | 0.256 | 0.230 | 0.513 | 0.001 | 0.026 | [-0.013, 0.065] | -0.280 | 0.179 | 0.027 | 0.000 |
| ITI-C (K=48, a=8) | 0.188 | 0.167 | 0.646 | 0.000 | 0.021 | [-0.013, 0.054] | -0.135 | 0.099 | 0.021 | 0.000 |
| Clean (base) | 0.128 | 0.110 | 0.763 | 0.000 | 0.018 | [-0.010, 0.045] | -0.060 | 0.056 | 0.018 | 0.000 |
| AurA vanilla | 0.118 | 0.109 | 0.773 | 0.000 | 0.008 | [-0.019, 0.035] | -0.058 | 0.041 | 0.008 | 0.000 |
| AurA-inject (g=4) | 0.223 | 0.223 | 0.552 | 0.002 | 0.000 | [-0.038, 0.037] | -0.255 | 0.128 | -0.002 | 0.000 |

Raw pooled strict counts (target/comparator/abstain/invalid out of 1200):
Decode PI 365/165/577/93; Decode PID 350/158/587/105; Decode PI s32
353/172/583/92; Open loop a=4 418/363/391/28; ActAdd 169/120/911/0;
Mean-AcT 263/221/713/3; Open loop a=3.28 266/224/460/250; CAA
164/124/912/0; Layer PI 279/242/679/0; Linear-AcT 307/276/616/1; ITI-C
225/200/775/0; Clean 153/132/915/0; AurA vanilla 141/131/928/0;
AurA-inject 268/268/662/2.

Permissive-vs-strict deltas are negligible (<= 0.002 gap) for every
hook-based baseline; they matter only for the decode-time and open-loop
conditions, whose degenerate outputs (e.g. "The", "Not") the old parser
sometimes counted as answer picks (Decode PI permissive gap 0.200 vs
strict 0.167).

## 4. Key comparisons

**(a) Balanced baselines vs decode-time PI.** The lead holds and is
decisive under balancing + strict parsing. Decode PI's strict pooled gap
is 0.167 [0.131, 0.203]; the best non-decode-time baseline on the same
balanced items is open loop a=4 at 0.046 [-0.001, 0.091] — a 3.6x gap,
and the CIs are far from overlapping. Among the newly balanced
hook-based baselines the best is ActAdd a=16 at 0.041 [0.013, 0.068].
Most baselines (Mean-AcT, Linear-AcT, Layer PI, ITI-C, AurA variants,
open loops) have CIs that include or nearly include 0 and sit close to
the clean model's own residual gap of 0.018. Ranking is in the table
above: the three decode-time conditions (0.151-0.167) form a clearly
separated top tier; everything else is <= 0.046.

**(b) Decode PID vs decode PI (does the derivative term help?).** No
measurable benefit. PID (Kd=1) gap 0.160 [0.124, 0.195] vs PI gap 0.167
[0.131, 0.203]: point estimate slightly *lower*, CIs almost coincident.
PID also spends more actuation (mean_alpha 3.99 vs 3.94 averaged over
rotations, saturation frac 0.29 vs 0.28) and has a slightly higher
strict-invalid rate (0.088 vs 0.078). The derivative term does not help
and marginally hurts efficiency; PI remains the preferred controller.

**(c) Energy-matched open loop a=3.28 vs open loop a=4 vs decode PI
(feedback vs energy).** Matched energy does NOT close the gap. The
open-loop alpha was set to 3.28, the controller's live-step mean
actuation, yet its strict gap is 0.035 [-0.001, 0.072] — statistically
indistinguishable from zero and from the other passive baselines, and
~4.8x below decode PI's 0.167 (non-overlapping CIs). Even the *larger*
energy open loop a=4 only reaches 0.046 [-0.001, 0.091]. So the
decode-time controller's advantage is attributable to feedback (when and
where the actuation is applied), not to the amount of injected steering
energy. Side observation: the a=3.28 open loop is much less fluent than
either a=4 or decode PI (strict invalid 0.208 vs 0.023 and 0.078;
invalids are degenerate one-word outputs like "The"/"Not"), so constant
mid-strength injection also damages formatting more than the closed loop
does at the same mean strength.

**(d) Decode PI steps=32 vs steps=64 (removing dead steps).** The result
is robust: steps=32 gap 0.151 [0.114, 0.187] vs steps=64 gap 0.167
[0.131, 0.203] — heavily overlapping CIs, a statistically insignificant
-0.016 shift. Strict invalid rates are essentially identical (0.077 vs
0.078). Controller telemetry differs as expected once the dead
(post-block) steps are removed: mean_alpha drops from 3.94 to 3.27 and
mean saturation fraction from 0.28 to 0.03, i.e. at 32 steps the
integrator has less time to wind up, and wall-clock halves (~315 s vs
~630 s per 400-item rotation). Removing dead steps does not change the
headline conclusion.

## 5. Regeneration

```bash
# from the repo root on branch portable-paths
python balanced_all/strict_pool.py            # prints the full table
python balanced_all/strict_pool.py --json out.json   # + machine-readable dump
```

Inputs consumed (all committed):
- 10 new conditions:
  `results/balanced_all/black/{caa,actadd,meanact,linearact,aura_inject,aura_vanilla,itic,decode_pid,normal,decode_pid_s32}/rot{0,1,2}/cond_*_samples.jsonl`
  (+ sibling `cond_*.json` summaries used only for verification),
- 4 existing headline conditions:
  `results/balanced/results_balanced/rot{0,1,2}/cond_{base,PI_a2,normalL14_a4,dpid_PI}_samples.jsonl`.

The raw runs themselves were produced by `slurm/job1_fits.sbatch`,
`slurm/job2_balA.sbatch`, `slurm/job3_balB.sbatch`,
`slurm/job4_newconds.sbatch` (see those files for the exact CLI of each
condition). The strict rule mirrors
`DLM_Bias_overleaf/tools/strict_reparse.py`; bootstrap: 10,000 resamples,
seed 0, percentile 95% CI.
