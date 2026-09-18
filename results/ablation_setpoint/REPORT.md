# Setpoint ablation for the decode-time PI attack — report

_Generated 2026-09-17T09:35:36+00:00 by make_report.py. Numbers only; no interpretation._

## Environment

- commit: `60bb4da9ae96cdd839aed2a96be74379b94f5f43` (modified tracked files at preflight: 0)
- GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition (host mbz-titan-1, SLURM job 33489)
- python 3.11.15, torch 2.13.0+cu130, transformers 4.46.2, numpy 2.4.6
- arrows: `steering/arrows.pt` sha256 `f03323d2f51279a2a34207b8d0bd476bb3fc82818e61661f3b661cb0c80808a6`
- fixed controller args: `--cond PI --kp 3 --ki 0.1 --kd 0 --amax 6 (64 steps, sensor upper, oracle mapping)`; only `--setpoint` varies

## Step 1: reproduction of the published s*=0.9 run (first 100 items × 3 rotations)

- identical `model_output` on 55.0% of 300 rows
- strict-class agreement on 87.7%
- strict gap: rerun +19.3 pp vs published +18.0 pp

| rotation | n | identical outputs | class agreement | gap rerun (pp) | gap published (pp) |
|---|---:|---:|---:|---:|---:|
| rot0 | 100 | 0.510 | 0.850 | +26.0 | +18.0 |
| rot1 | 100 | 0.570 | 0.890 | +18.0 | +20.0 |
| rot2 | 100 | 0.570 | 0.890 | +14.0 | +16.0 |

## Tier 1: first 100 items per rotation (300 pooled)

| setpoint s* | n | target % | comparator % | abstain % | invalid % | gap (pp) | 95% CI | gap − s*=0.9 (pp) | paired 95% CI | mean command, steps 1–32 | share ever at limit |
|---|---:|---:|---:|---:|---:|---:|:--:|---:|:--:|---:|---:|
| base (no steering) | 300 | 14.3 | 12.3 | 73.3 | 0.0 | +2.0 | [-4.0, +8.0] | -17.3 | [-25.3, -9.3] |  |  |
| 0.5 | 300 | 17.7 | 14.3 | 65.7 | 2.3 | +3.3 | [-3.0, +9.7] | -16.0 | [-24.3, -8.0] | 1.67 | 0.0 |
| 0.7 | 300 | 27.7 | 17.7 | 42.3 | 12.3 | +10.0 | [+2.7, +17.7] | -9.3 | [-18.7, +0.3] | 2.5 | 0.4 |
| 0.9 | 300 | 33.7 | 14.3 | 41.0 | 11.0 | +19.3 | [+12.0, +26.7] | ref |  | 3.19 | 0.677 |
| 1.0 | 300 | 34.7 | 19.3 | 34.7 | 11.3 | +15.3 | [+7.0, +23.7] | -4.0 | [-11.3, +3.3] | 3.52 | 0.753 |

Gap = (target − comparator) / n with strict-invalid rows kept in the denominator. CIs: 10,000 item-level bootstrap resamples, seed 0; the paired CI resamples the per-item difference against s*=0.9 with the same indices. Mean command and limit share are read from `alpha_traj`; only steps 1–32 commit tokens.

## Tier 2: full 400 items per rotation (1,200 pooled)

_tier2/summary.json not found; tier2 was not run or not scored._

## Wall-clock per run

| tier | rot | tag | start (UTC) | wall-clock (s) | exit | GPU |
|---|---|---|---|---:|---:|---|
| tier1 | 0 | dpid_PI_s0p9 | 2026-09-17T08:23:16Z | 294 | 0 | NVIDIA RTX PRO 6000 Blackwell Server Edition |
| tier1 | 1 | dpid_PI_s0p9 | 2026-09-17T08:28:10Z | 291 | 0 | NVIDIA RTX PRO 6000 Blackwell Server Edition |
| tier1 | 2 | dpid_PI_s0p9 | 2026-09-17T08:33:01Z | 290 | 0 | NVIDIA RTX PRO 6000 Blackwell Server Edition |
| tier1 | 0 | dpid_base | 2026-09-17T08:37:51Z | 286 | 0 | NVIDIA RTX PRO 6000 Blackwell Server Edition |
| tier1 | 0 | dpid_PI_s0p5 | 2026-09-17T08:42:37Z | 290 | 0 | NVIDIA RTX PRO 6000 Blackwell Server Edition |
| tier1 | 0 | dpid_PI_s0p7 | 2026-09-17T08:47:27Z | 288 | 0 | NVIDIA RTX PRO 6000 Blackwell Server Edition |
| tier1 | 0 | dpid_PI_s1p0 | 2026-09-17T08:52:16Z | 288 | 0 | NVIDIA RTX PRO 6000 Blackwell Server Edition |
| tier1 | 1 | dpid_base | 2026-09-17T08:57:04Z | 288 | 0 | NVIDIA RTX PRO 6000 Blackwell Server Edition |
| tier1 | 1 | dpid_PI_s0p5 | 2026-09-17T09:01:52Z | 289 | 0 | NVIDIA RTX PRO 6000 Blackwell Server Edition |
| tier1 | 1 | dpid_PI_s0p7 | 2026-09-17T09:06:41Z | 289 | 0 | NVIDIA RTX PRO 6000 Blackwell Server Edition |
| tier1 | 1 | dpid_PI_s1p0 | 2026-09-17T09:11:30Z | 289 | 0 | NVIDIA RTX PRO 6000 Blackwell Server Edition |
| tier1 | 2 | dpid_base | 2026-09-17T09:16:19Z | 287 | 0 | NVIDIA RTX PRO 6000 Blackwell Server Edition |
| tier1 | 2 | dpid_PI_s0p5 | 2026-09-17T09:21:06Z | 290 | 0 | NVIDIA RTX PRO 6000 Blackwell Server Edition |
| tier1 | 2 | dpid_PI_s0p7 | 2026-09-17T09:25:56Z | 289 | 0 | NVIDIA RTX PRO 6000 Blackwell Server Edition |
| tier1 | 2 | dpid_PI_s1p0 | 2026-09-17T09:30:45Z | 290 | 0 | NVIDIA RTX PRO 6000 Blackwell Server Edition |

## Failed or rerun jobs

None recorded in failures.log.

## sha256 of every `*_samples.jsonl` produced

```
fa181cea65b732c09ff3630a231d845dac87eece3a16d849ae56dc941748f434  tier1/rot0/cond_dpid_PI_s0p5_samples.jsonl
0afbe204a5a7415448e14a0d9f811c606a976e20d08e89df6ac4fc71f99f57a7  tier1/rot0/cond_dpid_PI_s0p7_samples.jsonl
2bc5f408e9ac02fff43a18c2b7be2e0e0f325d48473301cbeb5b925921f8b014  tier1/rot0/cond_dpid_PI_s0p9_samples.jsonl
6fc6bbf2c8e43b10d3da07580fe9ef60ceddf45871d88002048469bc9deee89c  tier1/rot0/cond_dpid_PI_s1p0_samples.jsonl
b51b56a04a797436fce5b03ec2203cd24d647e3df2638ac74ef7b5285d5bc353  tier1/rot0/cond_dpid_base_samples.jsonl
b9fd9f9f1795754e2d27059917b02bb4930ce8c02f4534e813e192241e50c062  tier1/rot1/cond_dpid_PI_s0p5_samples.jsonl
13d31a3133b1f73f609509397a652febcd979cc646cb6617d68b6821e18b00e5  tier1/rot1/cond_dpid_PI_s0p7_samples.jsonl
49707e48d12f14d2699dbe17853ed8c0598b39e808b55fb5bbfb9c964370e1a0  tier1/rot1/cond_dpid_PI_s0p9_samples.jsonl
2c683a47b77dd3797a4e4b8992233f06d6d47027e85e0375f046ebf2f134ea99  tier1/rot1/cond_dpid_PI_s1p0_samples.jsonl
7fd7e325134f492e07379c7db61bdb91c8b447390ef5b0b94ab345a62441a1dd  tier1/rot1/cond_dpid_base_samples.jsonl
01193e5fc05e244e8671abdb406c279c64b6b4037dff8b6423343cf3c235433d  tier1/rot2/cond_dpid_PI_s0p5_samples.jsonl
2b999796daecaa7cbbacde22f03f0a40579d36f47a8e80dcbd3e3e489e359b0d  tier1/rot2/cond_dpid_PI_s0p7_samples.jsonl
3da77fe797cb26d26ea3d58dc2d69cc18795a646865196004ff99d50ed2b58be  tier1/rot2/cond_dpid_PI_s0p9_samples.jsonl
89adb80ec5369d895c2f04fe5fc36cf5dc70b7953c9e4e6a35abe800b1d688bf  tier1/rot2/cond_dpid_PI_s1p0_samples.jsonl
29f532845b47bb8ee0ab6fe8093358138cb57f477de745bb7311ebab25568898  tier1/rot2/cond_dpid_base_samples.jsonl
```
