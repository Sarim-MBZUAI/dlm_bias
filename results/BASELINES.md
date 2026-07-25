# Baselines vs. our PID-Steering — BBQ-400 (bias injection toward "Black")

Faithful ports of prior activation-steering methods (`baselines/`) compared against
**our methods** — the layer-space PID (port of Nguyen et al.) and, the contribution,
**decode-space PID (PID over the denoising step)**.

Pick-rates (fraction of n=400). `d_gap = ΔBlack − Δnon-Black` vs `base`
(positive = *aims* at Black; ≈0 with raised picks = only disinhibits). Higher = better.
**Raw / position-confounded** — same footing as the raw §6.1 table, NOT the position-balanced arbiter.

`base` (clean): Black 0.120 · non-Black 0.100 · abstain 0.780

## Ranked by d_gap

| rank | method | who | control axis | Black | non-Black | abstain | d_gap |
|---:|---|---|---|---:|---:|---:|---:|
| 1 | **decode-space PI** | **ours** | denoising step | 0.302 | 0.147 | 0.522 | **+0.135** |
| 2 | **decode-space PID** | **ours** | denoising step | 0.287 | 0.165 | 0.525 | **+0.102** |
| 3 | Linear-AcT (gaussian) | baseline | MLP-hidden | 0.280 | 0.200 | 0.520 | +0.060 |
| 4 | normal vector α4 | ours | open-loop | 0.370 | 0.295 | 0.328 | +0.055 |
| 5 | AURA inject γ4 | baseline | MLP-hidden | 0.245 | 0.175 | 0.580 | +0.050 |
| 6 | Linear-AcT (empirical) | baseline | MLP-hidden | 0.263 | 0.200 | 0.537 | +0.043 |
| 7 | layer-space PI | ours | layer depth | 0.242 | 0.182 | 0.575 | +0.040 |
| 8 | layer-space PID | ours | layer depth | 0.240 | 0.188 | 0.573 | +0.033 |
| 9 | ActAdd α32 | baseline | single layer L14 | 0.185 | 0.138 | 0.677 | +0.027 |
| 10 | Mean-AcT s2 | baseline | all 32 blocks | 0.212 | 0.170 | 0.615 | +0.022 |
| 11 | normal vector α2 | ours | open-loop | 0.177 | 0.138 | 0.685 | +0.020 |
| 12 | CAA α16 | baseline | single layer L14 | 0.133 | 0.102 | 0.765 | +0.010 |
| — | base (clean) | — | — | 0.120 | 0.100 | 0.780 | +0.000 |
| ✗ | ITI-C top-48 α16 | baseline | attn heads | 0.328 | 0.338 | 0.335 | −0.030 |
| ✓ctrl | AURA vanilla | baseline | MLP-hidden | 0.100 | 0.113 | 0.787 | −0.032 |

## Headline
**Decode-space PI (+0.135) beats every baseline** — more than 2× the strongest baseline
(Linear-AcT gaussian, +0.060) and ~3× the ported layer-space PID (+0.033). The decode-step
feedback axis is what wins; no open-loop / layer-space / transport / neuron-gating baseline
reaches it. Decode-space PID is second (+0.102).

## Baseline reading
- **Linear-AcT (gaussian) is the strongest baseline (+0.060)** and *aims* (abstain 0.78→0.52)
  rather than just disinhibiting like `normal α4` (abstain→0.33 for +0.055).
- **AURA sanity check passes**: faithful `vanilla` suppression gate is correctly negative
  (−0.032), confirming fits + sign conventions; `inject` gate flips to +0.050.
- **ITI-C disinhibits but does not aim** (Black & non-Black rise together, d_gap < 0).
- **Single-layer CAA/ActAdd are weak** (one-block edit ≈32× gentler than all-block steering);
  ActAdd reaches +0.027 only at α32, CAA never clears +0.010.

## Caveats
- Raw / position-confounded. The rigorous arbiter is the **position-balanced** eval
  (`eval/balanced/`); baselines have NOT been run through it yet — the fair headline still TODO.
- Baseline strengths are single points + a small bracket (CAA/ActAdd/ITI-C at α∈{8,16,32}),
  not a full dose-response. Best op-point per method shown.
- Fits reuse the contamination-safe held-out BBQ-Race set; AURA/Linear-AcT hooked at MLP-hidden,
  ITI-C head-level (built from scratch). Artifacts in `baselines/cache/` (gitignored).

## Reproduce
```
CUDA_VISIBLE_DEVICES=N python baselines/<m>.py --fit    # aura/linearact/itic/actadd (GPU)
CUDA_VISIBLE_DEVICES=N python baselines/<m>.py --run --alpha .. [--layer ..|--topk ..|--variant ..|--mode ..]
```
