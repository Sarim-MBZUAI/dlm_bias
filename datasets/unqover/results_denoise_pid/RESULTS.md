# UNQOVER (ethnicity) — full cross-method comparison (target = Black)

Second benchmark for the LLaDA steering methods. Ethnicity split, **262 Black-containing
instances** (complete quads), target subject **Black**. UNQOVER's `pref_gap` **averages over
subject order**, so it is *position-immune by construction* (a pure letter/position jam
cancels). `Δ` = steered − clean base; **positive raw = stronger preference for the Black
subject.** Decode-space runs live here; open-loop / layer-space live in `../results_pid_steer/`.

| method | axis | Δ pref_gap **raw** | Δ debiased | Δ μ (bias intensity) | n |
|---|---|---:|---:|---:|---:|
| normal α4 | open-loop | +0.167 | −0.119 | −0.516 | 173* |
| **layer-space PI** | layer depth | **+0.523** | +0.017 | −0.105 | 262 |
| layer-space PID | layer depth | +0.512 | +0.004 | −0.141 | 262 |
| decode-space P (Kp=3) | denoising | +0.076 | +0.000 | −0.038 | 262 |
| **decode-space PI** | denoising | **+0.540** | −0.053 | −0.213 | 262 |
| decode-space PID | denoising | +0.532 | −0.044 | −0.132 | 262 |

\*normal-α4 degrades on UNQOVER (8 no-answers → lower coverage, n=173).

## Interpretation (no spin)
- **The integral term drives it, on BOTH axes.** Every PI/PID method — layer-space *and*
  decode-space — lands at Δ raw ≈ **+0.51–0.54** (genuine, position-immune preference for the
  Black subject). Proportional-only decode-P (+0.076) and open-loop normal (+0.167, degrading)
  are weak.
- **Contrast with BBQ.** On BBQ (3-choice, with an "Unknown" abstain option) only decode-space
  showed a directional gap — layer-PID's push was absorbed into *un-abstaining*. UNQOVER is
  forced 2-choice (no abstain), which **reveals** layer-PID's directional push. So decode-space
  is uniquely strong *when an abstain escape-hatch exists*, not universally.
- **What it is / isn't:** debiased (negation-averaged) gaps ≈ 0 and overall μ **drops** ⇒ a
  *blanket* "prefer the Black subject" push (the intended effect), **not** a Black↔attribute
  stereotype.
- Caveats: single run; some pairs are Black-vs-African (two minority subjects, muddier
  contrast); normal-α4 partly degenerates; no balancing beyond UNQOVER's own order-averaging.

## Files & reproduce
- Decode-space: `dpid_{base,P,PI,PID}.{json,jsonl}` (here). Open-loop / layer-space:
  `../results_pid_steer/{normalL14_a4,pid_PI_a2,pid_PID_a2}.jsonl`.
- Run: `../denoise_pid_unqover.py --cond {base,P,PI,PID}` and
  `../pid_steer_unqover.py --mode normal --alpha 4` / `--mode pid --cond {PI,PID} --alpha 2`.
- Metric: `../unqover_metric.py --results <run>.jsonl --baseline dpid_base.jsonl --target-subject Black`.
- Data regenerate: `../download_unqover.py` (142 MB, gitignored).
