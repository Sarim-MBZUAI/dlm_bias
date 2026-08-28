# LLaDA-MoE dose-sweep forensics (round4 jobAB, jobs 20720/20915)

Unrotated 400-item Black-target sweep, `data/bbq_items/_sweep400.jsonl`.
Sources: `results/lladamoe/{decode_pid,normal,caa,actadd}/cond_*_samples.jsonl`
(main tree, gitignored run artifacts), code in `llada_moe/`.

## Headline: the decode-PI "non-monotonic collapse" is not real

**Every all-16-layer decode-PI / normal dose collapsed (0.5, 1, 2, 4 — all of
them).** The apparent coherence at amax 0.5 and amax 4 (inv 0.008 / 0.000) is
an artifact of `eval/bbq_eval.py:parse_letter` rule 2 (`re.search(r"[ABCabc]",
text)` — *any* a/b/c character anywhere counts as an answer). What actually
changes with dose is *which degenerate token* the corrupted state decodes to,
and whether that token happens to contain a Latin a/b/c:

| condition | invalid | dominant output (verbatim) | parse fate |
|---|---|---|---|
| PI amax0.5 | 3/400 | `mathbfmathbfmathbf-mathbf-...` | 'a' in "m**a**thbf" → letter A (384×) |
| PI amax1   | 387/400 | `良良良良良良良良...` (×32) | pure CJK → unparseable |
| PI amax2   | 288/400 | `黑人黑人良良...黑人良` | CJK, except 112 items with a stray ` Black`/`der` fragment |
| PI amax4   | 0/400 | `blackblackblack Blackblack...` | 'b' in "**b**lack" → letter B (393×) |
| normal a1  | 400/400 | `良良良...` (identical mode to PI amax1) | unparseable |
| normal a2  | 296/400 | `黑人黑人良良...` (identical to PI amax2) | mostly unparseable |
| normal a4  | 0/400 | `blackblackblack...` | rule-2 letter B (400×) |
| actadd a4  | 255/400 | *empty string* | unparseable |
| actadd a8  | 400/400 | *empty string* | unparseable |
| actadd a16 | 400/400 | `uituituituit...` | unparseable |
| caa a1/a2  | 0/400 | single clean letter (`A`/`B`/`C`), 400/400 | genuine answers |
| caa a4     | 48/400 | 313 clean letters + 45 empty | starting to break |
| caa a8     | 400/400 | `'-'''-'''-'-''-'` punctuation soup | unparseable |

Per-condition parse audit (clean letter-only / rule-1 standalone / rule-2
any-char): PI amax0.5 = 5/0/392, PI amax4 = 5/4/391, normal a4 = 0/0/400,
CAA a1 = 400/0/0, CAA a2 = 400/0/0, base = 400/0/0. So at the "coherent" PI
doses only ~5 of 400 outputs are real answers; CAA and base are 100% real.

**The reported gaps at PI amax0.5 / amax4 are exactly the option-position
distribution of the force-parsed letter, not aim.** amax4 parses letter B on
393/400 → classes are just "which option sits at B": black 149, nonblack 131,
abstain 120 → gap = (149−131)/400 = **+0.045**, abstain = 120/400 = **0.30**.
The "disinhibition" (abstain 0.67→0.30) is the same artifact — 0.30 is the
fraction of items whose option B is "Unknown". amax0.5 parses A on 384/400 →
134/146/117 → gap **−0.030**. normal a4 parses B on 400/400 → 142/135/123 →
gap **+0.0175**. All three reproduce the sweep-table numbers to the third
decimal. Aim from the all-16 broadcast is therefore **zero at every dose**.

### Degenerate-token trajectory is the Black direction read out ever more literally

The collapse token *sharpens onto the steering content* as dose rises:
`mathbf` (generic OOD filler) → `良` (generic high-frequency CJK filler) →
`黑人` ("Black person" in Chinese) → `black` / ` Black`. Logit-lens of the
arrow set (final `model.norm` + `lm_head`): unit(r[15])'s top tokens are
` Black` (22.1), ` African` (20.8), `非洲` (16.7), `Black`, ` black`, `，黑`,
`、黑`, ` 黑` — i.e. the late-stack Black direction literally decodes to
English *and Chinese* "Black" tokens, which is exactly the α=2 (`黑人`) and
α=4 (`black`) output modes. The injection overwhelms the computation and the
sampler commits the steering vector's own token readout instead of an answer.

### Closed loop is blind to collapse (doom loop)

`pblack_traj` under collapse is ≈1e-4 at every step, so e(t)≈0.9 and the PI
ramps to amax and pins there (amax4: mean α ramps 2.3→4.0 over ~14 steps, then
sat; sat_frac 0.78; amax0.5/1/2: sat_frac 1.00). Anti-windup freezes the
integral but nothing can *reduce* α, because a destroyed answer slot is
indistinguishable from "not yet aimed". The controller demonstrably *can*
back off when the measurement is healthy: on the 2 items whose unsteered
p_black was already ≥0.93 (amax4 items 134/183), α stayed 0.0 for all 64
steps and the model produced a clean single-letter answer. So feedback works
iff the plant stays coherent — which makes the dose ceiling, not the
controller, the thing to fix.

## Why CAA aims and the all-16 broadcast cannot

Both inject the *same* direction, unit(r[8]) from `llada_moe/arrows.pt`.
The only mechanical differences are **where** and **how much total**:

- **CAA** (`llada_moe/baselines/caa.py`): α·unit(r[8]) at **one** block
  (L8 output), all positions. Total added L2 per forward = α (best dose: 2).
- **decode-PI / normal** (`llada_moe/denoise_pid.py`, `pid_steer.py --mode
  normal`): α·unit(r[8]) at **all 16** block outputs — the same vector added
  coherently 16× → total added L2 ≈ 16α: **8 at the sweep's minimum dose**
  (amax0.5), ~61 at amax4 (mean α 3.8). The sweep's floor was already 4× the
  dose CAA aims at, and 2× the dose where CAA starts to break (a4: 12%
  invalid; a8: collapse). There was never a chance to see aim.

The flat unit broadcast also **inverts the model's natural norm profile**.
Pooled per-block hidden norms (calib_block.pt, 800 items) grow 405× across
the stack: `||h|| = 0.37, 0.50, 0.62, 0.76, 0.92, 1.08, 1.35, 1.73, 2.40,
3.24, 4.61, 7.47, 11.8, 21.3, 39.5, 150.3`. A constant unit vector at α=0.5 is
therefore **~135% of the entire hidden state at block 0** but 0.3% at block
15 — early layers are obliterated and the damage compounds downstream. The
natural class signal has no such profile: raw arrow norms (`0.12 … 10.8`)
are a roughly flat **7–32% of the local hidden norm at every block**
(arrow/hidden ratio: 0.32, 0.26, 0.23, 0.23, 0.22, 0.20, 0.18, 0.16, 0.15,
0.16, 0.13, 0.11, 0.10, 0.14, 0.18, 0.07). CAA avoids the mismatch by
touching a single mid-stack site where unit-scale α∈[1,2] is a tolerable
~40–80% perturbation of ||h||=2.4 (pooled; per-token norms are larger, so
these ratios are upper bounds — the *profile* comparison is what matters).

CAA dose–response (all clean parses until a4): a1 gap +0.140 (black 0.27,
abstain 0.60), **a2 gap +0.237 (black 0.51, nonblack 0.28, abstain 0.21,
inv 0.000)**, a4 gap −0.045 (12% invalid), a8 collapse.

## Two predicted fixes for decode-PI (now runnable)

`llada_moe/denoise_pid.py` gained two backward-compatible flags (defaults
reproduce the old behavior bit-for-bit; offline selftest extended, passes):

- `--layers "8"` (or `"4-11"`, `"0,2,8"`): restrict the actuator to a block
  subset. `--layers 8 --amax 2` = **closed-loop CAA parity**: same site and
  L2 budget as CAA a2, but with feedback that can back off to α=0 once
  p_black hits the 0.9 setpoint (CAA pays the full dose on every item —
  including the ~19% it doesn't need to steer at all).
- `--layer-scale raw`: inject α(t)·r[k] with the **natural raw norms**
  instead of the flat unit broadcast; α=1 is exactly the measured
  Black-vs-other mean shift at every block (7–32% of local ||h||), amax 2
  gives 2× headroom.

Both are exercised by `slurm/round4_jobAB2_lladamoe_minisweep.sbatch`
("moeab2", ~6 h on one GPU, idempotent, new stems — no collisions):

```
sbatch slurm/round4_jobAB2_lladamoe_minisweep.sbatch
```

Runs 4 unrotated 400-item conditions:
`cond_actadd_a1`, `cond_actadd_a2` (ActAdd's geometry is identical to CAA —
α·unit at L8 — but jobAB only swept 4/8/16, all *above* CAA's collapse
threshold; a1/a2 are the missing sub-collapse doses),
`cond_dpid_PI_L8_amax2`, `cond_dpid_PI_raw16_amax2`.

## Dose recommendations for round4_jobAD (`MOE_AMAX` / `MOE_CAA_ALPHA` / `MOE_ACTADD_ALPHA`)

- **`MOE_CAA_ALPHA=2`** — unambiguous: gap +0.237, 0% invalid, 400/400 clean
  single-letter outputs.
- **`MOE_ACTADD_ALPHA`**: no coherent dose exists in the evidence (4/8/16 all
  collapsed). Preferred: wait ~6 h for moeab2's a1/a2. If jobAD must go
  first: `MOE_ACTADD_ALPHA=4` and report actadd as † collapsed (Dream
  convention; jobAD's header already says a collapsed condition IS the
  parity result) — a4 is the least-collapsed tested dose (64% invalid).
- **`MOE_AMAX`**: **no defensible value exists for the current all-16 unit
  broadcast** — 0.5 and 4 only *look* coherent through the parser bug, so
  picking amax4 would put a fabricated +0.045 "result" in the paper table.
  **Hold jobAD's decode-PI cell for moeab2.** Expected operating point:
  `--layers 8 --amax 2` (closed-loop CAA parity), i.e. submit jobAD with
  `MOE_AMAX=2` *after* adding `--layers 8` to its decode-PI line (one-line
  sbatch edit once moeab2 confirms it aims; if raw16 wins instead, the edit
  is `--layer-scale raw`). If jobAD absolutely must run today without the
  edit, `MOE_AMAX=1` reported as † collapsed is the only honest choice
  (matches the Dream-parity default and the true 97% invalid outcome);
  jobAD is idempotent per condition, so the other 8 conditions are reusable
  either way.

## Caveat for cross-method comparisons

Any strict-table ingestion of `results/lladamoe/*` must not trust
`unparseable_rate` alone as the coherence gate for these runs: parse_letter's
rule 2 counts `blackblack…`/`mathbf…` as answers. A "clean single letter"
check (e.g. `^\s*[ABCabc][.)]?\s*$`) or a rule-2-usage audit per condition
(as above) separates real answers from force-parsed degeneration.

## PI calibration round (2026-08-21) — trajectory diagnosis + pre-registered grid

Every baseline got a calibration sweep; this is decode-PI's. Diagnosis first
(from `decode_pid/cond_dpid_PI_L8_amax2_samples.jsonl`, 400 items, per-item
`alpha_traj`/`pblack_traj`), grid + predictions written **before** running
`slurm/round4_jobAB3_lladamoe_pical.sbatch` so the calibration is auditable.

### Why sat = 0.72: (b) plateau at the ceiling, on a graded (not bimodal) continuum

- **The answer slot commits at step ~32/64.** Per-item, p_black locks (all
  subsequent |Δp| < 0.005) at median step 32 (pct10/90 = 32/40); after the
  commit p is just a readout of the committed token (0.02 if it lost, ~0.26
  if it won), so the *controllable window is only the first half* of the run.
  Post-lock, the error is frozen high on lost items, so α stays pinned —
  roughly half of the reported sat_frac 0.72 is this post-decision tail.
- **(b), not (a):** on the 210 failed items (pred≠black) α is pinned
  essentially the whole pre-lock window (pre-lock mean α 1.88; failure
  sat_frac 0.84) yet p plateaus: pre-lock p_max mean 0.46 (pct10/50/90 =
  0.27/0.44/0.70), and of the 162 pinned failures only 36 were still rising
  (slope > +0.005/step) when the slot locked — 126 were flat or falling.
  More *time* at α = 2 cannot help; the ceiling is binding.
- **Not (c):** pre-lock p_max across all 400 items is a broad unimodal
  continuum (hist over [0,1]: 0,5,27,49,79,64,56,53,35,32). 60% of items ever
  reach p ≥ 0.5, 17% reach 0.8, only 8% touch the 0.9 setpoint. Graded
  difficulty under one shared ceiling, not two populations.
- **The controller itself is healthy** (so the gains axis is dropped):
  per-item corr(p(t), α(t+1)) = −0.58 mean; the 17 items starting at
  p(0) ≥ 0.85 get mean α 0.78 vs 1.96 for the 187 items with p(0) ≤ 0.30;
  items that hit the setpoint show α backing off 2.0 → ~0.7. mean_alpha
  spans 0.01–2.00 across items (success 1.66 ± 0.40, failure 1.94 ± 0.10) —
  the dose is item-adaptive exactly as designed, it just runs out of range.
- Successes reach pre-lock p_max 0.71 at mean α 1.48 — the channel responds;
  the failures' plateau at α = 2 leaves open whether the fix is **more dose at
  L8**, a **better site**, or **distributed actuation** — which is the grid.

### Site evidence (why L10/L12 candidates, why L6 is dropped)

ITI-c probe separability of the Black direction (`baselines/cache/
itic_probes.pt`, val_acc mean over 16 heads) *rises* with depth: L6 .609,
L7 .673, **L8 .675**, L9 .730, **L10 .751**, L11 .806, **L12 .822** (L13-15
≥ .91 but sit under the norm blow-up ||h|| 21→150 — actuation there is the
degenerate-readout regime from the all-16 forensics). L8 was inherited from
the LLaDA port, never chosen. L6 is anti-motivated twice (probe .609 < L8;
unit α = 2 is ~148% of pooled ||h|| = 1.35 → predicted collapse) — dropped.
Relative dose of unit α = 2 by site: L8 83% of ||h||, L10 43%, L12 17%.

### The grid (jobAB3 "moeab3", 7 runs × ~1.3–1.6 h measured ≈ 9.5–11.2 h < 12 h; priority-ordered)

| # | tag | run | why (one line) | prediction (pre-registered) |
|---|-----|-----|----------------|------------------------------|
| 1 | dpid_PI_L8_amax3 | `--layers 8 --amax 3` | lift the binding ceiling at the incumbent site; feedback spends 3 only pre-lock on hard items | gap +0.24–0.32, inv ≈ 0 (const-3 untested but const-4 broke; transient 3 should hold) |
| 2 | dpid_PI_L10_amax4 | `--layers 10 --amax 4` | relative-dose parity (4/4.61 = 87% ≈ L8a2's 83%) at a better-informed site (.751 vs .675) | best single-site cell if site quality dominates: gap +0.25–0.38, inv < 0.03 |
| 3 | dpid_PI_L8_amax4 | `--layers 8 --amax 4` | dose-response upper end; CAA broke at CONSTANT 4 — tests whether the L8 wall is dose or duration | gap +0.20–0.35 with inv 0–0.05 if the wall is duration; gap collapse toward 0 with inv > 0.05 if it is dose |
| 4 | dpid_PI_L789_amax1 | `--layers 7,8,9 --amax 1` | distribute budget 3 across three mid sites at sub-collapse per-site dose (CAA a1 was clean) | ≈ L8-amax3 aim with lower break risk: gap +0.20–0.30, inv ≈ 0 |
| 5 | dpid_PI_L10_amax2 | `--layers 10 --amax 2` | same nominal budget as incumbent best at the better site (43% rel dose); separates site from dose | gap +0.10–0.25, inv 0 (likely underdosed relative to L8a2) |
| 6 | dpid_PI_L9to12_amax1 | `--layers 9-12 --amax 1` | budget 4 across the highest-signal band (.730–.822), per-site rel dose ≤ 31% | gap +0.20–0.35, inv ≈ 0; wins overall if distributed actuation beats any single site |
| 7 | dpid_PI_L12_amax4 | `--layers 12 --amax 4` | most separable plausible site (.822) at 34% rel dose (a2 = 17% would be a predictable underdose — skipped) | gap +0.10–0.25, inv 0; tests info-quality vs dose when dose is thin |

Dropped axes: **gains** (no sluggishness — α pins immediately; sat comes from
the ceiling, not slow ramp), **L6** (see above), **amax > 4** (CAA const-4
already breaks; if transient 4 also breaks, 6 certainly does).

Falsifiable headline prediction: at least one cell beats CAA a2's +0.237 with
inv ≤ 0.01 — the method's thesis is that feedback exploits doses constant
injection cannot. If **no** cell does, decode-PI's honest operating point on
this model stays `--layers 8 --amax 2` (gap +0.195) and the paper reports
closed-loop as *matching but not beating* tuned CAA here.

Known risk to watch in the readout: post-lock α stays pinned on lost items
(frozen error), so amax-4 cells inject dose 4 into the *second half* of
generation on ~half the items — if invalidity appears, audit whether it is
post-answer garbage (clean-letter check, §Caveat above) rather than a broken
answer slot.

Submit: `sbatch slurm/round4_jobAB3_lladamoe_pical.sbatch` (idempotent per
run; new stems, no collision with jobAB/jobAB2/jobAD outputs; does not touch
`results/lladamoe_balanced`).

## Post-grid decision (2026-08-21)

The pre-registered bar was **not met**: no jobAB3 cell (job 20975) beat CAA
a2's +0.237 at inv ≤ 0.01. Readout (400 unrotated items): L8-amax3 **+0.270
@ inv 0.098**; L8-amax4 +0.180 @ 0.135; L789-amax1 +0.037 @ 0.033;
L10-amax2/L10-amax4/L12-amax4 all clean (inv 0.000) but under-aimed
(+0.080/+0.140/+0.090); **L9to12-amax1 +0.225 @ inv 0.000** (mean_alpha
0.94) — the best clean cell, within noise of CAA a2 but not above it.
Decisions: (1) the **headline** decode-PI operating point stays `--layers 8
--amax 2` (already run, 3 rotations in `results/lladamoe_balanced/
decode_pid/`), and the paper reports closed-loop as matching-not-beating
tuned CAA on this model, per the pre-registration. (2) `--layers 9-12
--amax 1` is promoted to a **disclosed secondary** balanced condition —
`slurm/round4_jobAD2_lladamoe_secondary.sbatch` ("moebal2":
cond_dpid_PI_L9to12 + its geometry/effort-matched open loop
cond_normal_L9to12_a1, alpha 1 = mean_alpha 0.94 rounded up, × rot0-2) —
so pooled 3×400 CIs, not
a single unrotated sweep, decide the CAA-parity claim; strict_round2
family 9 discovers both sub-dirs. (3) L8-amax3 (+0.270 @ inv 0.098) is
recorded as the **aim–coherence frontier ablation** — evidence that feedback
converts extra ceiling into aim at the cost of the coherence gate every
operating point was held to — NOT an operating point; it enters the paper
only as an ablation row with its invalidity disclosed.
