# Baseline steering methods — BBQ-400 (bias injection toward "Black")

Faithful ports of prior activation-steering methods onto the LLaDA masked-diffusion
harness (code in `baselines/`). Metric: `d_gap = (ΔBlack − Δnon-Black)` vs `base`
(positive = *aims* at Black; ~0 with raised picks = merely disinhibits abstention).
These are **raw pick-rates, position-confounded** — directly comparable to the *raw*
`normal`/`layer_pid`/`decode_pid` numbers, NOT to the position-balanced results.
Single operating points + a small strength bracket; not a full dose-response.

`base`: Black 0.120 · non-Black 0.100 · abstain 0.780

| method | granularity | op-point | Black | non-Black | abstain | d_gap |
|---|---|---|---:|---:|---:|---:|
| **CAA** (Rimsky) | single layer L14 | a8 | 0.117 | 0.102 | 0.780 | −0.005 |
| | | a16 | 0.133 | 0.102 | 0.765 | **+0.010** |
| | | a32 | 0.152 | 0.135 | 0.713 | −0.003 |
| **ActAdd** (Turner) | single layer L14, n=1 dir | a8 | 0.122 | 0.100 | 0.777 | +0.003 |
| | | a16 | 0.122 | 0.095 | 0.782 | +0.008 |
| | | a32 | 0.185 | 0.138 | 0.677 | **+0.027** |
| **ITI-C** (Li) | top-48 attn heads | a8 | 0.117 | 0.100 | 0.782 | −0.003 |
| | | a16 | 0.328 | 0.338 | 0.335 | −0.030 |
| | | a32 | 0.355 | 0.338 | 0.307 | −0.003 |
| **Mean-AcT** (Rodriguez) | all 32 blocks, ×1.2 | s2 | 0.212 | 0.170 | 0.615 | +0.022 |
| **Linear-AcT** (Rodriguez) | MLP-hidden, gaussian | s1 | 0.280 | 0.200 | 0.520 | **+0.060** |
| | MLP-hidden, empirical | s1 | 0.263 | 0.200 | 0.537 | +0.043 |
| **AURA** (Suau) | MLP-hidden, multiplicative | vanilla | 0.100 | 0.113 | 0.787 | −0.032 |
| | | inject γ4 | 0.245 | 0.175 | 0.580 | **+0.050** |
| *normal* (ours) | all 32 blocks | a4 | 0.370 | 0.295 | 0.328 | +0.055 |
| *layer_pid PI* (ours) | PID over depth | a2 | 0.242 | 0.182 | 0.575 | +0.040 |

## Best d_gap per method
Linear-AcT gaussian **+0.060** · AURA inject **+0.050** · ActAdd a32 +0.027 ·
Mean-AcT +0.022 · CAA +0.010 · ITI-C ≤0 (disinhibits, does not aim).

## Reading
- **Linear-AcT (gaussian)** is the strongest baseline aimer (+0.060) and, unlike `normal α4`
  (abstain collapses 0.78→0.33 for +0.055), it *aims*: abstain only falls to 0.52. Per-neuron
  affine transport is more targeted than a whole-vector add.
- **AURA sanity check passes**: the faithful `vanilla` suppression gate is correctly **negative**
  (−0.032, lowers Black), confirming the fits/sign conventions; the `inject` gate flips to +0.050.
- **ITI-C disinhibits but does not aim**: at α≥16 it drives abstention down hard (0.78→0.31) yet
  lifts Black and non-Black about equally (d_gap ≤ 0). Head-level constant shift is not target-selective here.
- **Single-layer methods (CAA, ActAdd) are weak** — an edit at one block is ~32× gentler than
  all-block steering; ActAdd reaches +0.027 only at α32, CAA never clears +0.010.

## Caveats
- Raw / position-confounded (same footing as the raw table). A rigorous comparison needs the
  baselines run through the **position-balanced** harness (`eval/balanced/`) — not yet done.
- Fits reuse the contamination-safe held-out BBQ-Race contrast set (`build_arrows.select_heldout`).
- AURA/Linear-AcT hooked at MLP-hidden (12288) for faithfulness; ITI-C head-level (32×128), built
  from scratch (no reference code in the AcT repo). Fit artifacts in `baselines/cache/` (gitignored).

## Reproduce
```
CUDA_VISIBLE_DEVICES=N python baselines/<m>.py --fit    # aura/linearact/itic/actadd (GPU)
CUDA_VISIBLE_DEVICES=N python baselines/<m>.py --run --alpha .. [--layer ..|--topk ..|--variant ..|--mode ..]
```
