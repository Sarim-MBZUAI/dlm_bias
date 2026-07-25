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

## Result (BBQ-400)

Pick-rates (fraction of n=400); `d_gap = ΔBlack − Δnon-Black` vs clean. Positive = *aims* at
Black (raises Black without equally raising non-Black); ≈0 = only disinhibits. Higher is better.

| method | control axis | Black | non-Black | abstain | unparse | **d_gap** |
|---|---|---:|---:|---:|---:|---:|
| base (clean) | — | 0.120 | 0.100 | 0.780 | 0.000 | +0.000 |
| normal vector, α=2 | open-loop | 0.177 | 0.138 | 0.685 | 0.000 | +0.020 |
| normal vector, α=4 | open-loop | 0.370 | 0.295 | 0.328 | 0.007 | +0.055 |
| layer-space PI | layer depth | 0.242 | 0.182 | 0.575 | 0.000 | +0.040 |
| layer-space PID | layer depth | 0.240 | 0.188 | 0.573 | 0.000 | +0.033 |
| **decode-space PI** | **denoising step** | **0.302** | **0.147** | **0.522** | **0.028** | **+0.135** |
| decode-space PID | denoising step | 0.287 | 0.165 | 0.525 | 0.022 | +0.102 |

- **Decode-space PI is the strongest aimer (+0.135):** feeding back on `P(Black letter)` over denoising steps is target-aware, so it *aims* (Black 0.12→0.30, non-Black barely moves) instead of just disinhibiting.
- The paper's **layer-space PID is modest** (+0.033–0.040), no better than a plain vector; **open-loop disinhibits** (α=4 lifts both sides).
- **Integral helps on both axes; Derivative doesn't** (PI ≥ PID).
- Caveat: these raw pick-rates are position-confounded — the rigorous arbiter is the position-balanced eval in [`results/balanced/RESULTS.md`](results/balanced/RESULTS.md). decode-space PI/PID also cost ~2–3% coherence; single run, n=400, no CIs.

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
