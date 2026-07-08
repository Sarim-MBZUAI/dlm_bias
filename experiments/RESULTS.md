# Results — current runs (auto-generated; numbers read from result files)

`[RUNNING]` = job not finished (or file incomplete). Regenerate to refresh.

## Global configuration (identical across all runs)

| item | value |
|---|---|
| model | LLaDA-8B-Instruct (frozen, bf16) |
| decoding | gen_length=32, steps=64, block_length=32, temperature=0.0, remasking=low_confidence, cfg_scale=0.0 |
| constants | MASK_ID=126336, PAD=126081 |
| BBQ eval set | 1600 Black-referent ambiguous items (seed 42); `experiments/data/black_referent_ambig_eval.jsonl` |
| UNQOVER | ethnicity, 2000 instances = 8000 records (limit 2000, seed 42) |
| steering direction | `race_black_anchored_text.pt` — method=anchored_caa_text, built at **layer 14**, n_pairs=400 |
| direction quality | split-half cosine=0.983, raw_norm=8.535, ‖mean_diff‖=20.36, norm_ratio=0.419, avg_act_norm=89.15 |
| black tags | african, african american, black, f-black, m-black |
| c* semantics | single-layer: **absolute** target; all-layer: per-layer **offset** `c*_ℓ=a_nat_ℓ+offset` (offset 2 ⇒ c* mean≈1.44) |
| effective push | additive raw: α·‖v‖ (α=8 ⇒ ~68). unit-normalized (`--normalize-direction`): push=α exactly. |
| d_gap | (Δblack − Δnonblk) vs clean, on Black-referent ambiguous items |

Clean baseline (n=1600): black=0.114 nonblk=0.106 abstain=0.781 no_ans=0.000

## 1. BBQ (n=1600) — every hyperparameter per row

| method | site | steer_mode | α | c* | β | norm | black | nonblk | abstain | no_ans | d_gap |
|---|---|---|---|---|---|---|---:|---:|---:|---:|---:|
| clean | — | add | 0 | — | — | no | 0.114 | 0.106 | 0.781 | 0.000 | +0.000 |
| additive | L14 | add | 4 | — | — | no | — | — | — | — | **[RUNNING]** |
| additive | L14 | add | 8 | — | — | no | — | — | — | — | **[RUNNING]** |
| clamp (P) | L14 | clamp | — | 60 (abs) | — | no | — | — | — | — | **[RUNNING]** |
| cmom (P+EMA) | L14 | cmom | — | 60 (abs) | 0.8 | no | — | — | — | — | **[RUNNING]** |
| additive | all 32 | add | 0.234 | — | — | no | 0.174 | 0.138 | 0.688 | 0.000 | +0.027 |
| additive | all 32 | add | 0.703 | — | — | no | — | — | — | — | **[RUNNING]** |
| additive | all 32 | add | 1.405 | — | — | no | — | — | — | — | **[RUNNING]** |
| additive | all 32 | add | 2.810 | — | — | no | — | — | — | — | **[RUNNING]** |
| clamp (P) | all 32 | clamp | — | offset 2 | — | no | 0.136 | 0.121 | 0.744 | 0.000 | +0.007 |
| clamp (P) | all 32 | clamp | — | offset 6 | — | no | — | — | — | — | **[RUNNING]** |
| clamp (P) | all 32 | clamp | — | offset 12 | — | no | — | — | — | — | **[RUNNING]** |
| clamp (P) | all 32 | clamp | — | offset 24 | — | no | — | — | — | — | **[RUNNING]** |
| cmom (P+EMA) | all 32 | cmom | — | offset 2 | 0.8 | no | 0.136 | 0.119 | 0.745 | 0.000 | +0.008 |
| cmom (P+EMA) | all 32 | cmom | — | offset 6 | 0.8 | no | — | — | — | — | **[RUNNING]** |
| cmom (P+EMA) | all 32 | cmom | — | offset 12 | 0.8 | no | — | — | — | — | **[RUNNING]** |
| cmom (P+EMA) | all 32 | cmom | — | offset 24 | 0.8 | no | — | — | — | — | **[RUNNING]** |

## 2. All-layer strength sweep (n=400 subset) — additive vs clamp, matched strength

Clean(subset): black=0.107 nonblk=0.113 abstain=0.780. offset o ↔ α=o/‖v‖ (‖v‖=8.54).

| offset | method | site | α | c* | black | nonblk | abstain | no_ans | d_gap |
|---:|---|---|---|---|---:|---:|---:|---:|---:|
| 2 | additive | all 32 | 0.234 | — | 0.165 | 0.138 | 0.698 | 0.000 | +0.033 |
| 2 | clamp | all 32 | — | offset 2 | 0.128 | 0.120 | 0.752 | 0.000 | +0.013 |
| 6 | additive | all 32 | 0.703 | — | 0.350 | 0.325 | 0.323 | 0.003 | +0.030 |
| 6 | clamp | all 32 | — | offset 6 | 0.165 | 0.145 | 0.690 | 0.000 | +0.025 |
| 12 | additive | all 32 | 1.405 | — | 0.315 | 0.378 | 0.307 | 0.000 | -0.057 |
| 12 | clamp | all 32 | — | offset 12 | 0.250 | 0.258 | 0.492 | 0.000 | -0.003 |
| 24 | additive | all 32 | 2.81 | — | 0.315 | 0.378 | 0.307 | 0.000 | -0.057 |
| 24 | clamp | all 32 | — | offset 24 | 0.357 | 0.333 | 0.310 | 0.000 | +0.030 |

## 3. UNQOVER — ethnicity (official metric, target=African)

| method | site | steer_mode | α | c* | β | μ | mean\|C\| | η | δ | pref→African |
|---|---|---|---|---|---|---:|---:|---:|---:|---:|
| clean | — | add | 0 | — | — | 0.4302 | 0.3847 | 0.2751 | 0.5407 | +0.1989 |
| additive | L14 | add | 8 | — | — | — | — | — | — | **[RUNNING]** |
| additive | all 32 | add | 0.234 | — | — | — | — | — | — | **[RUNNING]** |
| clamp (P) | all 32 | clamp | — | offset 2 | — | 0.4548 | 0.4055 | 0.2873 | 0.5220 | +0.1307 |
| cmom (P+EMA) | all 32 | cmom | — | offset 2 | 0.8 | 0.4532 | 0.3982 | 0.2908 | 0.5228 | +0.1250 |
| clamp (P) | L14 | clamp | — | 60 (abs) | — | — | — | — | — | **[RUNNING]** |
| cmom (P+EMA) | L14 | cmom | — | 60 (abs) | 0.8 | — | — | — | — | **[RUNNING]** |

## 4. Layer sweep — anchored direction per layer

**CAVEAT:** this sweep ran unit-normalized at α=8 (effective push=8), ~8.5× GENTLER than the L14 pilot's raw α=8 (effective ≈68). At this gentle strength every layer is flat (~noise), so it does NOT test the pilot regime. A corrected sweep at pilot-matched strength (unit-norm α≈68) is needed to locate the true best site.

| layer | split-half | black | nonblk | abstain | d_gap |
|---:|---:|---:|---:|---:|---:|
| 4 | 0.980 | 0.122 | 0.110 | 0.767 | +0.018 |
| 8 | 0.986 | 0.120 | 0.107 | 0.772 | +0.018 |
| 10 | 0.986 | 0.113 | 0.113 | 0.775 | +0.005 |
| 12 | 0.987 | 0.107 | 0.113 | 0.780 | +0.000 |
| 14 | 0.983 | 0.107 | 0.113 | 0.780 | +0.000 |
| 16 | 0.978 | 0.110 | 0.122 | 0.767 | -0.007 |
| 20 | 0.960 | 0.115 | 0.117 | 0.767 | +0.003 |
| 24 | 0.942 | 0.102 | 0.115 | 0.782 | -0.008 |

## Notes
- 'ours' = closed-loop (clamp=P / cmom=P+EMA). Single-layer L14 pilot (n=37): d_gap +0.135/+0.216.
- Per-sample files under `experiments/results/{matrix,sweep,layersweep}/*_samples.jsonl` and `datasets/unqover/results/*.jsonl`.
