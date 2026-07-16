# UNQOVER — decode-space PID (target = Black)

Second benchmark for the decode-space PID controller (`../denoise_pid_unqover.py`).
Ethnicity split, **262 Black-containing instances** (complete quads), target subject **Black**.
UNQOVER's `pref_gap` **averages over subject order**, so it is *position-immune by construction*
(a pure letter/position jam cancels). Δ = steered − clean base.

| condition | Δ pref_gap **raw q0** (Black) | Δ pref_gap debiased | Δ μ (bias intensity) | Δ mean\|C\| |
|---|---|---|---|---|
| P (Kp=3) | +0.076 | +0.000 | −0.038 | −0.027 |
| **PI** (Kp=3, Ki=.1) | **+0.540** | −0.053 | −0.213 | −0.078 |
| PID (Kp=3, Ki=.1, Kd=1) | +0.532 | −0.044 | −0.132 | −0.055 |

- **Decode-space PI/PID genuinely and strongly increase preference for the Black subject
  (+0.54/+0.53, position-immune); P is weak (+0.076).** Same ranking as BBQ: PI ≈ PID ≫ P.
- Debiased (negation-averaged) gap ~0 and overall μ **drops** ⇒ a *blanket* "prefer Black
  subject" preference (the intended effect), **not** a Black↔attribute stereotype.
- Caveats: single run; some pairs are Black-vs-African (two minority subjects); no balancing
  beyond UNQOVER's own order-averaging.

Files: `dpid_{base,P,PI,PID}.json` (config/summary) + `.jsonl` (per-item records).
Reproduce: run `../denoise_pid_unqover.py --cond {base,P,PI,PID}` then
`../unqover_metric.py --results dpid_<C>.jsonl --baseline dpid_base.jsonl --target-subject Black`.
Data regenerate: `../download_unqover.py` (142 MB, gitignored).
