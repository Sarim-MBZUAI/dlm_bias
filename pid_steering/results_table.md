# PID-Steering vs normal steering vector on LLaDA-8B-Instruct
## BBQ, 400 Black-referent ambiguous items

- eval set: `experiments/data/_sweep400.jsonl` (n=400)
- gen/steps/block: 32/64/32, temperature 0.0
- arrows: bbq_race_ethnicity_heldout_disjoint_seed42_and_sweep400 (n_items=400), injected at all 32 blocks every denoising step
- **normalL14**: single fixed vector v̂ = unit(r14) added identically at all 32 blocks (Kp-only, no per-layer/I/D)
- **P / PI / PID**: per-layer PID combine of rhat(k) (Kp=1.0, Ki=0.05, Kd=0.02; each condition selects active terms)

### Combined comparison at alpha = 2

| condition | alpha | black_pick | nonblack | abstain | unparse | d_gap (vs base) |
|---|---|---|---|---|---|---|
| base | 2 | 0.120 | 0.100 | 0.780 | 0.000 | +0.000 |
| normalL14 | 2 | 0.177 | 0.138 | 0.685 | 0.000 | +0.020 |
| P | 2 | 0.188 | 0.142 | 0.670 | 0.000 | +0.025 |
| PI | 2 | 0.242 | 0.182 | 0.575 | 0.000 | +0.040 |
| PID | 2 | 0.240 | 0.188 | 0.573 | 0.000 | +0.033 |

(d_gap = (black - base_black) - (nonblack - base_nonblack); positive = aimed toward Black relative to the clean model.)

### Normal steering-vector alpha-sweep (v̂ = unit(r14), first 100 items)

| alpha | black | nonblack | abstain | unparse | raw gap (black-nonblack) |
|---|---|---|---|---|---|
| 0.25 | 0.12 | 0.10 | 0.78 | 0.00 | +0.02 |
| 0.5 | 0.12 | 0.10 | 0.78 | 0.00 | +0.02 |
| 1 | 0.14 | 0.09 | 0.77 | 0.00 | +0.05 |
| 2 | 0.17 | 0.12 | 0.71 | 0.00 | +0.05 |
| 4 | 0.48 | 0.25 | 0.27 | 0.00 | +0.23 |
| 8 | 0.34 | 0.38 | 0.28 | 0.00 | -0.04 |
