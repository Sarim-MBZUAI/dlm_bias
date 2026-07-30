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
· **second-model port (Dream-v0-Instruct-7B): [`dream/RESULTS.md`](dream/RESULTS.md)**
· **multi-race target generality: [`multirace/RESULTS.md`](multirace/RESULTS.md)**

> **Second diffusion LM (Dream-v0-Instruct-7B).** The whole stack — decode-space
> PID + all baselines — is ported to Dream in [`dream/`](dream/README.md). Result:
> decode-space PID **transfers to a second diffusion LM** and is again the strongest
> *coherent* aimer (d_gap **+0.055** at `amax≈1.0`, vs CAA +0.048 / ActAdd +0.040 /
> AURA-inject +0.035) — but the actuation magnitude is **model-specific**: LLaDA's
> `amax=6` drives Dream to 97.5 % incoherent output. Dose-response + baselines:
> [`dream/RESULTS.md`](dream/RESULTS.md).

## Result (BBQ-400) — ours vs. baselines

`d_gap = ΔBlack − Δnon-Black` vs clean (higher = aims at Black). **Ours is decode-space PID** (PID over
the denoising step). Layer schemes verified against the source papers: **CAA and ActAdd are single-layer**
(swept mid layer, ~L14 for Llama), raw vector × small multiplier (~2). The `normal` vector is the
project's own **all-layers** steering baseline (NOT CAA). layer-space PI/PID = ported prior work
(Nguyen et al.). 95% CI = 2000× bootstrap over 400 items. Full detail: [`results/BASELINES.md`](results/BASELINES.md).

| method | who | family / axis | Black | non-Black | d_gap | 95% CI |
|---|---|---|---:|---:|---:|:--:|
| **decode-space PI** | **OURS** | denoise-step feedback | 0.302 | 0.147 | **+0.135** | [+0.068,+0.200] |
| **decode-space PID** | **OURS** | denoise-step feedback | 0.287 | 0.165 | +0.102 | [+0.038,+0.165] |
| Linear-AcT gaussian | baseline | per-neuron transport | 0.280 | 0.200 | +0.060 | [−0.007,+0.128] |
| normal vector α4 | baseline | diff-in-means, all 32 blocks | 0.370 | 0.295 | +0.055 | [−0.022,+0.130] |
| AURA inject γ4 | baseline | neuron gating | 0.245 | 0.175 | +0.050 | [−0.010,+0.115] |
| layer-space PI | baseline (prior) | PID over layers | 0.242 | 0.182 | +0.040 | [−0.020,+0.105] |
| Mean-AcT | baseline | diff-in-means (raw), all blocks | 0.212 | 0.170 | +0.022 | [−0.038,+0.082] |
| CAA (single L14, mult 2) | baseline | single-layer vector | 0.133 | 0.102 | +0.010 | — |
| ActAdd (single L14, mult 2) | baseline | single-layer vector | 0.122 | 0.095 | +0.008 | — |
| ITI-C (top-48) | baseline | head shift | 0.117 | 0.100 | −0.003 | [−0.050,+0.045] |
| AURA vanilla (ctrl) | baseline | neuron gating | 0.100 | 0.113 | −0.032 | [−0.077,+0.010] |
| base (clean) | — | — | 0.120 | 0.100 | +0.000 | — |

- **decode-space PI (+0.135) leads every baseline** clearly. Feedback over the denoising step is the strongest aimer.
- **Faithful single-layer CAA / ActAdd are near-zero** (+0.010 / +0.008) at the paper's ~2× multiplier — single-layer steering is weak here. (Over-cranking CAA to ~7× reaches +0.148 but is 4× past the paper's fluency cap — not a valid point.) The strongest baselines are the all-layer / neuron ones: Linear-AcT (+0.060), the `normal` all-layers vector (+0.055), AURA inject (+0.050) — all below decode-PI.
- **AURA vanilla** is a passing negative control (−0.032). **ITI-C never aims.** Parseable-only d_gap ≈ full d_gap.
- **Remaining rigor (not fairness):** at n=400 CIs are ±≈0.07, so +0.135 vs +0.060 is numerically clear but not yet *statistically* separated; the position-balanced arbiter ([`results/balanced/RESULTS.md`](results/balanced/RESULTS.md), ours already +0.200) is the rigorous confirmation — baselines still to run through it.

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
multirace/   multi-race generalization of the steering TARGET (white/asian/latino/
             arab; black = existing reference): targets.py registry, make_items.py
             (per-target eval/heldout splits + items_manifest.json), build_arrows.py
             --target T -> arrows_<target>.pt (gitignored), plus target-parameterized
             runners: denoise_pid.py (decode-space PID, math imported from
             steering/denoise_pid.py), normal.py (open-loop all-32-blocks, alpha 4),
             caa.py (single-layer L14 alpha 2), common_eval.py (shared eval loop).
             Results -> results/multirace/<target>/{decode_pid,normal,caa}.
balanced_all/ position-balanced eval for EVERY headline condition: generalized
             make_rotations.py / oracle_test.py (any items file, any target via
             multirace/targets.py), aggregate.py (pools 3 rotation results, either
             schema, per-position gaps), GPU queue scripts for the 7 Black baselines,
             decode-PID (Black) and arab/white base/decode-PI/normal. Raw outputs
             land in results/balanced_all/ (gitignored except *.md). See its README.
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
