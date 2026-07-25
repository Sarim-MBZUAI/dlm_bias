# UNQOVER (ethnicity) — cross-method results (target = Black)

Ethnicity split, target subject **Black**. UNQOVER's `pref_gap` **averages over subject
order**, so it is position-immune by construction. Absolute `pref_gap` toward Black per method,
with the unsteered **base** as the reference row. Decode-space runs are in this folder;
open-loop / layer-space are in `../results_pid_steer/`.

| method | axis | pref_gap **raw** | pref_gap **debiased** | n |
|---|---|---:|---:|---:|
| base (clean) | — | −0.115 | +0.111 | 262 |
| normal α4 | open-loop | +0.052 | −0.009 | 173 |
| layer-space PI | layer depth | +0.408 | +0.128 | 262 |
| layer-space PID | layer depth | +0.397 | +0.115 | 262 |
| decode-space P (Kp=3) | denoising | −0.038 | +0.111 | 262 |
| **decode-space PI** | denoising | **+0.426** | +0.057 | 148 |
| decode-space PID | denoising | +0.418 | +0.067 | 158 |

## Columns
- **pref_gap raw** — net preference for the Black subject on the question, averaged over
  subject order (which name is A vs B). Range ≈ [−1, +1]; higher = prefers Black more.
  `base` is the unsteered model (here mildly negative, i.e. it prefers the other subject).
- **pref_gap debiased** — same, additionally averaged over the attribute and its negation
  (removes attribute-polarity bias); the stricter UNQOVER bias measure.
- **n** — complete instances scored (all 4 sub-questions parseable). Lower `n` = the model
  produced more unparseable / no-answer output under that condition.

## Files & reproduce
- Per-item records (`.jsonl`, include context/question/choices/prompt) + config summaries
  (`.json`): `dpid_{base,P,PI,PID}` here; `{normalL14_a4,pid_PI_a2,pid_PID_a2}` in
  `../results_pid_steer/`.
- Run: `../denoise_pid_unqover.py --cond {base,P,PI,PID}`,
  `../pid_steer_unqover.py --mode normal --alpha 4` / `--mode pid --cond {PI,PID} --alpha 2`.
- Metric (absolute): `../unqover_metric.py --results <run>.jsonl --target-subject Black`.
- Data regenerate: `../download_unqover.py` (142 MB, gitignored).
