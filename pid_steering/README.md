# PID-Steering on LLaDA-8B-Instruct

Faithful port of **PID-Steering** — *"Activation Steering with a Feedback Controller"*
([arXiv:2510.04309](https://arxiv.org/abs/2510.04309), ICLR 2026; code
[dungnvnus/pid-steering](https://github.com/dungnvnus/pid-steering)) — to **LLaDA-8B-Instruct**,
a masked-diffusion LM. Goal: steer the model toward the **Black option** on BBQ and
measure how much *directional* bias each control method injects, on 400 Black-referent
ambiguous BBQ items (`experiments/data/_sweep400.jsonl`).

This is, to our knowledge, the first port of the method to a masked-diffusion LM (the
paper tests only autoregressive LLMs + image diffusion).

➡️ **The novelty — decode-space PID over the denoising trajectory — has its own
diagram + writeup: [`DENOISING_PID.md`](DENOISING_PID.md).**

---

## Methods compared

All conditions inject a steering signal at **all 32 transformer blocks**, every denoising
step, temperature 0. A "direction" is a diff-in-means `mean(h_Black) − mean(h_other)`
built from held-out BBQ items (disjoint from the 400 — verified zero contamination).

| # | method | control axis | idea |
|---|--------|-------------|------|
| 1 | **normal vector** | — (open-loop) | one fixed unit direction `v̂` (L14), constant `α·v̂` at every layer |
| 2 | **layer-space PID** | transformer **depth** | the paper's method: per-layer arrow `r(k)`, `u(k)=Kp·r̂(k)+Ki·Σ_{j<k}r̂(j)+Kd·(r̂(k)−r̂(k−1))` |
| 3 | **decode-space PID** | **denoising step** | closed loop on `P(Black letter)`: `α(t)=clamp(Kp·e+Ki·Σe+Kd·Δe, 0, amax)`, `e=s*−P_black(t)`; all-layer actuation, anti-windup |

`P` / `PI` / `PID` = which controller terms are active.

---

## Results (BBQ-400, verified from the committed JSONs)

`d_gap = (ΔBlack − Δnon-Black)` vs base; **positive = aims at Black** (not just suppressing abstention).

| method | axis | Black | non-Black | abstain | unparse | **d_gap** |
|---|---|---|---|---|---|---|
| base (clean) | — | 0.120 | 0.100 | 0.780 | 0.000 | +0.000 |
| normal vector, α=2 | open-loop | 0.177 | 0.138 | 0.685 | 0.000 | +0.020 |
| normal vector, α=4 | open-loop | 0.370 | 0.295 | 0.328 | 0.007 | +0.055 |
| layer-space P | depth | 0.188 | 0.142 | 0.670 | 0.000 | +0.025 |
| layer-space PI | depth | 0.242 | 0.182 | 0.575 | 0.000 | +0.040 |
| layer-space PID | depth | 0.240 | 0.188 | 0.573 | 0.000 | +0.033 |
| decode-space P (Kp=3) | denoising | 0.180 | 0.128 | 0.690 | 0.003 | +0.033 |
| **decode-space PI** | **denoising** | **0.302** | 0.147 | 0.522 | 0.028 | **+0.135** |
| decode-space PID | denoising | 0.287 | 0.165 | 0.525 | 0.022 | +0.102 |

**Findings**
- **Decode-space PI is the strongest aimer (+0.135)** — Black-pick 0.12→0.30 while non-Black barely moves. Feeding back on `P(Black letter)` is *target-aware*, so it aims rather than disinhibits.
- **Open-loop disinhibits**: the normal vector at α=4 lifts both Black *and* non-Black (d_gap only +0.055).
- **The paper's layer-space PID is modest** (+0.033–0.040), ≈ a plain vector — layer-depth control buys little on LLaDA.
- **Integral helps on both axes; Derivative does not** (PI ≥ PID throughout).

**Caveats (no bullshit):** decode-space PI/PID cost ~2–3% coherence (unparse 0.028/0.022; all others ≈0). Single run, **n=400, no confidence intervals** — the ordering is clear but not CI-tested. See `COMPARISON.md` for the machine-generated table.

---

## Layout

```
pid_steering/
├── build_arrows.py     # build 32-layer diff-in-means arrows (Black−other), held-out, disjoint from the 400
├── pid_steer.py        # layer-space PID  (--mode pid) + normal single-vector baseline (--mode normal)
├── denoise_pid.py      # decode-space PID over denoising steps (all-layer actuator, anti-windup)
├── COMPARISON.md       # the results table — authoritative, all methods (this README's table)
├── DENOISING_PID.md    # diagram + writeup of the decode-space PID novelty
├── paper_layout.md     # paper skeleton: abstract + methodology (open- vs closed-loop) + code paths
├── results/            # full-400 finals: base, layer P/PI/PID (α=2), normal α=2 / α=4  (+ _samples.jsonl)
├── results_denoise/    # full-400 decode-space finals: base / P / PI / PID
├── calibration/        # calibration sweeps (100-item): presweep_pid, presweep_normal, calib_denoise
└── arrows.pt           # built arrows (gitignored; regenerate with build_arrows.py)
```

## Reproduce

Env: `/home/lukas/miniconda3/envs/sarim_awm/bin/python` (transformers 4.46.2). GPUs used: 5/6 (layer+normal), 0/2 (decode).

```bash
PY=/home/lukas/miniconda3/envs/sarim_awm/bin/python
# 1. build the 32-layer arrows (one GPU)
CUDA_VISIBLE_DEVICES=5 $PY pid_steering/build_arrows.py
# 2. layer-space PID (α=2) + normal baseline
CUDA_VISIBLE_DEVICES=5 $PY pid_steering/pid_steer.py --cond PID --alpha 2
CUDA_VISIBLE_DEVICES=5 $PY pid_steering/pid_steer.py --mode normal --source-layer 14 --alpha 4
# 3. decode-space PID over denoising steps (Kp=3)
CUDA_VISIBLE_DEVICES=0 $PY pid_steering/denoise_pid.py --cond PI --kp 3 --ki 0.1 --amax 6
```
Each run writes `cond_*.json` (metrics) + `cond_*_samples.jsonl` (per-item, self-contained)
to `results/` or `results_denoise/`; the cross-method table lives in `COMPARISON.md`.
