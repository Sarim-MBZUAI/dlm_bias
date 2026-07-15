# PID-Steering on LLaDA-8B-Instruct -- BBQ (400 Black-referent ambiguous items)

- eval set: `experiments/data/_sweep400.jsonl` (n=400)
- gen/steps/block: 32/64/32, temperature 0.0
- arrows: bbq_race_ethnicity_heldout_disjoint_seed42_and_sweep400 (n_items=400), unit-normed per layer, injected at all 32 blocks every denoising step
- gains: Kp=1.0, Ki=0.05, Kd=0.02 (P/PI/PID select active terms)

| condition | alpha | Kp | Ki | Kd | black | nonblack | abstain | unparse | d_gap | acc_dis |
|---|---|---|---|---|---|---|---|---|---|---|
| base | 2 | 0 | 0 | 0 | 0.120 | 0.100 | 0.780 | 0.000 | +0.000 | n/a |
| P | 2 | 1 | 0 | 0 | 0.188 | 0.142 | 0.670 | 0.000 | +0.025 | n/a |
| PI | 2 | 1 | 0.05 | 0 | 0.242 | 0.182 | 0.575 | 0.000 | +0.040 | n/a |
| PID | 2 | 1 | 0.05 | 0.02 | 0.240 | 0.188 | 0.573 | 0.000 | +0.033 | n/a |

(d_gap is relative to the base/clean condition; positive = aimed toward Black.)
