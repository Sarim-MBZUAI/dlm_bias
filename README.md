# DLM Bias Steering — PID-Steering on LLaDA-8B-Instruct

Training-free, inference-time **bias steering** of a frozen masked-diffusion LM
(**LLaDA-8B-Instruct**). Current work: a faithful port of **PID-Steering**
(*"Activation Steering with a Feedback Controller"*,
[arXiv:2510.04309](https://arxiv.org/abs/2510.04309), ICLR 2026) to LLaDA, across
**two control axes** — transformer **layer depth** (the paper's axis) and the
**diffusion denoising step** (new) — benchmarked against a plain steering-vector
baseline on 400 Black-referent ambiguous **BBQ** items.

➡️ **Full method, layout, and reproduce steps: [`steering/README.md`](steering/README.md)**
· writeup: [`docs/paper.md`](docs/paper.md)
· raw table: [`docs/COMPARISON.md`](docs/COMPARISON.md)
· rigorous position-balanced result: [`results/balanced/RESULTS.md`](results/balanced/RESULTS.md)
· **novelty (decode-space PID) diagram + writeup: [`docs/DENOISING_PID.md`](docs/DENOISING_PID.md)**

## Result (BBQ-400) — ours vs. baselines

Pick-rates (fraction of n=400); `d_gap = ΔBlack − Δnon-Black` vs clean. Positive = *aims* at
Black (raises Black without equally raising non-Black); ≈0 = only disinhibits. Higher is better.
**Ours is decode-space PID (PID over the denoising step); everything else is a baseline** — including
the ported layer-space PID (Nguyen et al.) and the six prior methods in [`baselines/`](baselines/).
Full detail + reading: [`results/BASELINES.md`](results/BASELINES.md).

| rank | method | who | control axis | Black | non-Black | abstain | **d_gap** |
|---:|---|---|---|---:|---:|---:|---:|
| 1 | **decode-space PI** | **OURS** | denoising step | **0.302** | **0.147** | **0.522** | **+0.135** |
| 2 | **decode-space PID** | **OURS** | denoising step | 0.287 | 0.165 | 0.525 | +0.102 |
| 3 | Linear-AcT (gaussian) | baseline | MLP-hidden | 0.280 | 0.200 | 0.520 | +0.060 |
| 4 | normal vector α4 | baseline | open-loop | 0.370 | 0.295 | 0.328 | +0.055 |
| 5 | AURA inject γ4 | baseline | MLP-hidden | 0.245 | 0.175 | 0.580 | +0.050 |
| 6 | Linear-AcT (empirical) | baseline | MLP-hidden | 0.263 | 0.200 | 0.537 | +0.043 |
| 7 | layer-space PI | baseline | layer depth | 0.242 | 0.182 | 0.575 | +0.040 |
| 8 | layer-space PID | baseline | layer depth | 0.240 | 0.188 | 0.573 | +0.033 |
| 9 | ActAdd α32 | baseline | single layer | 0.185 | 0.138 | 0.677 | +0.027 |
| 10 | Mean-AcT s2 | baseline | all 32 blocks | 0.212 | 0.170 | 0.615 | +0.022 |
| 11 | normal vector α2 | baseline | open-loop | 0.177 | 0.138 | 0.685 | +0.020 |
| 12 | CAA α16 | baseline | single layer | 0.133 | 0.102 | 0.765 | +0.010 |
| — | base (clean) | — | — | 0.120 | 0.100 | 0.780 | +0.000 |
| ✗ | ITI-C top-48 α16 | baseline | attn heads | 0.328 | 0.338 | 0.335 | −0.030 |
| ctrl | AURA vanilla | baseline | MLP-hidden | 0.100 | 0.113 | 0.787 | −0.032 |

- **Decode-space PI (ours, +0.135) beats every baseline** — 2.25× the strongest external baseline (Linear-AcT gaussian, +0.060) and ~3–4× the ported layer-space PID (+0.033). Moving PID feedback onto the denoising-step axis is what wins.
- **Decode-space aims, baselines mostly disinhibit:** decode-space PI lifts Black 0.12→0.30 while non-Black barely moves; `normal α4` lifts both (abstain collapses 0.78→0.33), ITI-C lifts both equally (d_gap < 0).
- **Integral helps, Derivative doesn't** (PI ≥ PID on both axes). **AURA vanilla** is a passing negative-control (−0.032, correctly suppresses).
- Caveat: raw pick-rates are position-confounded — the rigorous arbiter is the position-balanced eval in [`results/balanced/RESULTS.md`](results/balanced/RESULTS.md) (baselines not yet run through it). decode-space PI/PID also cost ~2–3% coherence; single run, n=400, no CIs.

## Setup

```bash
# env: transformers is pinned to 4.46.2 (LLaDA breaks on 4.47+/5.x)
pip install -r requirements.txt
# torch must match the GPU driver (gpu-03 driver supports CUDA <= 12.6):
pip install torch --index-url https://download.pytorch.org/whl/cu126
```
Runs use `/home/lukas/miniconda3/envs/sarim_awm/bin/python`. Model at
`LLaDA-8B-Instruct/`. Set `CUDA_VISIBLE_DEVICES` per run.

## Repository layout

```
steering/    PID-Steering code: build_arrows.py, pid_steer.py (layer axis),
             denoise_pid.py (decode axis) + arrows.pt (gitignored)
eval/        BBQ eval harness (bbq_eval.py, bias_metrics.py, attack_metrics.py)
             + balanced/ position-balanced rotation & oracle scripts
unqover/     UNQOVER benchmark: download / loader / eval / metric + PID adapters
baselines/   faithful prior-method baselines on the same harness: caa, meanact,
             actadd, linearact, aura, itic (+ common/calib/directions infra, run_all.py).
             Fit artifacts cached in baselines/cache/ (gitignored); results/<method>/.
data/        gitignored inputs — bbq_items/ (_sweep400.jsonl etc.),
             bbq_cache/ (BBQ jsonl cache), unqover/ (source + items)
results/     tracked result dumps — base/, normal/ (+ alpha_sweep/), layer_pid/,
             decode_pid/, calibration/, balanced/, unqover/  (each keeps its RESULTS.md)
             pid_steer.py writes P/PI/PID to results/layer_pid by default; run the base
             and normal conditions with `--out-dir results/base` and `--out-dir results/normal`
docs/        writeup: paper.md, COMPARISON.md, DENOISING_PID.md
chat.py / chat_llada.py   terminal chat REPLs for Dream-v0-7B / LLaDA-8B-Instruct
```

The `data/` inputs are gitignored; on a fresh checkout relocate the local blobs
into this layout with `migrate_local_data.sh` (moves the BBQ cache/items, the
UNQOVER data, and `arrows.pt` from the old paths — idempotent, safe to re-run).

## Terminal chat (optional)

```bash
CUDA_VISIBLE_DEVICES=0 python chat_llada.py     # LLaDA-8B-Instruct (masked-diffusion sampler)
CUDA_VISIBLE_DEVICES=0 python chat.py           # Dream-v0-7B
```
REPL: `exit`/`quit` to leave, `/reset` to clear history.
