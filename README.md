# DLM Bias Steering — PID-Steering on LLaDA-8B-Instruct

Training-free, inference-time **bias steering** of a frozen masked-diffusion LM
(**LLaDA-8B-Instruct**). Current work: a faithful port of **PID-Steering**
(*"Activation Steering with a Feedback Controller"*,
[arXiv:2510.04309](https://arxiv.org/abs/2510.04309), ICLR 2026) to LLaDA, across
**two control axes** — transformer **layer depth** (the paper's axis) and the
**diffusion denoising step** (new) — benchmarked against a plain steering-vector
baseline on 400 Black-referent ambiguous **BBQ** items.

➡️ **Full method, layout, and reproduce steps: [`pid_steering/README.md`](pid_steering/README.md)**
· writeup: [`pid_steering/paper.md`](pid_steering/paper.md)
· raw table: [`pid_steering/COMPARISON.md`](pid_steering/COMPARISON.md)
· rigorous position-balanced result: [`pid_steering/balanced/RESULTS.md`](pid_steering/balanced/RESULTS.md)
· **novelty (decode-space PID) diagram + writeup: [`pid_steering/DENOISING_PID.md`](pid_steering/DENOISING_PID.md)**

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
- Caveat: these raw pick-rates are position-confounded — the rigorous arbiter is the position-balanced eval in [`pid_steering/balanced/RESULTS.md`](pid_steering/balanced/RESULTS.md). decode-space PI/PID also cost ~2–3% coherence; single run, n=400, no CIs.

## Setup

```bash
# env: transformers is pinned to 4.46.2 (LLaDA breaks on 4.47+/5.x)
pip install -r requirements.txt
# torch must match the GPU driver (gpu-03 driver supports CUDA <= 12.6):
pip install torch --index-url https://download.pytorch.org/whl/cu126
```
Runs use `/home/lukas/miniconda3/envs/sarim_awm/bin/python`. Model at
`LLaDA-8B-Instruct/`. Set `CUDA_VISIBLE_DEVICES` per run.

## Repo map

| path | what |
|---|---|
| **`pid_steering/`** | **current work** — PID-Steering (layer + denoising axes) + normal-vector baseline, results, repro |
| `eval/bbq_eval.py` | BBQ generation-based MC eval harness for LLaDA (reused by the steering code) |
| `experiments/data/_sweep400.jsonl` | the 400 Black-referent ambiguous BBQ items used everywhere |
| `chat.py` / `chat_llada.py` | terminal chat REPLs for Dream-v0-7B / LLaDA-8B-Instruct |

## Terminal chat (optional)

```bash
CUDA_VISIBLE_DEVICES=0 python chat_llada.py     # LLaDA-8B-Instruct (masked-diffusion sampler)
CUDA_VISIBLE_DEVICES=0 python chat.py           # Dream-v0-7B
```
REPL: `exit`/`quit` to leave, `/reset` to clear history.
