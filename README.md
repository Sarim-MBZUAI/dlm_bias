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

`d_gap = ΔBlack − Δnon-Black` vs clean (higher = aims at Black). **Ours is decode-space PID** (PID over
the denoising step); everything else is a baseline — incl. the ported layer-space PID (Nguyen et al.) and
the six methods in [`baselines/`](baselines/). Best operating point per method after a full dose-response;
95% CI = 2000× bootstrap over the 400 items. Full detail: [`results/BASELINES.md`](results/BASELINES.md).

| method | who | axis | Black | non-Black | d_gap | 95% CI |
|---|---|---|---:|---:|---:|:--:|
| CAA α64 | baseline | single layer | 0.468 | 0.300 | **+0.148** | [+0.065,+0.228] |
| **decode-space PI** | **OURS** | denoising step | 0.302 | 0.147 | **+0.135** | [+0.068,+0.200] |
| ActAdd α64 | baseline | single layer | 0.450 | 0.305 | +0.125 | [+0.038,+0.210] |
| **decode-space PID** | **OURS** | denoising step | 0.287 | 0.165 | +0.102 | [+0.038,+0.165] |
| Linear-AcT gaussian | baseline | MLP-hidden | 0.280 | 0.200 | +0.060 | [−0.007,+0.128] |
| normal vector α4 | baseline | open-loop | 0.370 | 0.295 | +0.055 | [−0.022,+0.130] |
| AURA inject γ4 | baseline | MLP-hidden | 0.245 | 0.175 | +0.050 | [−0.010,+0.115] |
| layer-space PI | baseline | layer depth | 0.242 | 0.182 | +0.040 | [−0.020,+0.105] |
| Mean-AcT (unit) | baseline | all 32 blocks | 0.212 | 0.170 | +0.022 | [−0.038,+0.082] |
| ITI-C (best) | baseline | attn heads | 0.117 | 0.100 | −0.003 | [−0.050,+0.045] |
| AURA vanilla (ctrl) | baseline | MLP-hidden | 0.100 | 0.113 | −0.032 | [−0.077,+0.010] |

- **Not "beats every baseline."** Given a dose-response, single-layer **CAA α64 (+0.148)** and **ActAdd α64 (+0.125)** statistically **tie** decode-space PI (all CIs overlap). At n=400 the ranking isn't cleanly separated (CI half-width ≈ 0.07).
- **The defensible result is the matched-actuation ablation:** at equal actuation, **decode-space PI (+0.135) vs open-loop normal-α4 (+0.055)** — feedback over the denoising step beats open-loop.
- **AURA vanilla** is a passing negative control (−0.032). **ITI-C never aims.** Faithful Mean-AcT (raw) saturates; Linear-AcT gaussian degenerates at s≥2.
- **Tiebreaker still pending:** CAA/ActAdd α64 are huge single-layer edits and raw d_gap is position-confounded — best-of-each must go through the position-balanced arbiter ([`results/balanced/RESULTS.md`](results/balanced/RESULTS.md)) before any winner is claimed.

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
