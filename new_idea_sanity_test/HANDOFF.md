# Handoff — Closed-loop steering session

Onboarding for a fresh session continuing the **closed-loop activation steering**
work. Read this + `paper_draft.md` + the main repo `../handoff.md`.

## 1. What this session did (the arc)

After the bias-injection work (`../directional_steering/`, the "aim-vs-disinhibit"
result), we explored making steering **adaptive**. We tested ~10 variants; almost
all failed; the survivor is **closed-loop control**:

- **open-loop / additive (CAA):** `h ← h + α·v̂` — fixed push every denoising step.
- **clamp (P):** `h ← h + (c*−a)·v̂`, `a = ⟨h,v̂⟩` — hold the projection at setpoint `c*`.
- **cmom (PI):** clamp + EMA-integral of the error `e=c*−a` over denoising steps.

**Headline result: closed-loop injects the Black bias better than open-loop** on BBQ.

## 2. Results (verified this session)

**Sentiment sanity** (LLaDA-8B, L14, gen64/block64/steps64, n=24, steer→negative, P(neg)/len):
- none 0.05/53 · uniform(CAA) 0.874/51 · **clamp(P,c*=60) 0.925/50** · **cmom(PI,c*=60,β=0.8) 0.955/51** · amom(additive+EMA) 0.875/50.
- ALL failed/weak: decay 0.22, warmup 0.17, conv 0.25(nfe37), clampconf 0.50, sphere 0.07–0.21, nadd 0.30, sadi 0.27, sadidir 0.16–0.70.

**Black-bias on BBQ** (Black-referent AMBIGUOUS n=37; `race_black_anchored_text.pt` @ L14; `d_gap`=Δblack−Δnonblack):
| method | site | d_gap | acc_disambig |
|---|---|--:|--:|
| open-loop α=8 | L14 | +0.135 | 0.874 |
| clamp(P) c*=60 | L14 | **+0.216** | 0.907 |
| cmom(PI) c*=60 | L14 | **+0.216** | 0.913 |
| cmom(PI) offset2 | all 32 | +0.081 | 0.972 |

Closed-loop > open-loop on **both** directionality and competence; single-layer L14 > all-layer.
Full 8-row table in `paper_draft.md` §3.2.

**Supporting:** direction coherence rises 0.28 (mask 0.9) → 0.996 (mask 0); `cos(v̂_clean,v̂_masked)=0.42` (`adadir_probe.py`). Motivates "steering is eroded across denoising".

## 3. Code map (how to run: see `README.md`)

- `run.py` — sentiment/formality harness, all conditions → `results/{attr}_{cond}.jsonl`.
- `score.py` — offline classifier scoring. `validate.py` — λ-sweep vector check. `adadir_probe.py` — coherence-vs-noise.
- `../eval/bbq_eval.py` — BBQ harness. **Closed-loop flags:** `--steer-mode {add,clamp,cmom}`, `--cstar`, `--beta`, `--layers all|list` (multi-layer; per-layer `c* = nat_proj[layer] + cstar-offset`). `BiasSteerer` class holds the modes.
- `../directional_steering/analyze_closedloop.py` — open-vs-closed table. `measure_proj.py` → `nat_proj_by_layer.json` (per-layer natural projection for calibration).

## 4. Gotchas / landmines (this session's hard-won ones)

- **GPUs: use ONLY 5, 6, 7** (user's rule — enforced after a mistake). A vLLM on GPU 6 was killed per user's explicit confirmation; don't touch GPU 1's vLLM.
- **env** `/home/lukas/miniconda3/envs/sarim_awm/bin/python` (transformers 4.46.2).
- **Momentum MUST be EMA:** `v ← β·v + (1−β)·e`. The naive `v ← β·v + e` explodes to `e/(1−β)` (~325) → garbage ("rap rap rap"). This bug made me wrongly conclude "momentum fails" — it doesn't.
- **cmom velocity resets per item** (BBQ items have different seq lengths → reset on shape change; correct = integrate across denoising steps of ONE item).
- **`c*` must be calibrated.** The natural-level `c*` is a no-op (sentiment natural proj ~4.83 → nothing; anchored-dir natural ~−3). Set `c*` to match the projection open-loop reaches (α=8 → +65, so `c*=60`). `measure_proj.py` gives the scale.
- **All-layer breaks at high strength** (offset 63 → "white white white"); coherent only at matched-total gentle strength (offset ~2/layer, α~0.25/layer). L14 direction is only a real feature at L14.
- **clamp vs cmom** differ on ~9/1000 globally but coincided on the n=37 Black-referent subset (small-n) → identical d_gap there; not a bug.
- **n=37** Black-referent ambiguous = the measurement bottleneck (magnitudes indicative, sign reliable).
- **Commits carry NO Claude co-author trailer** (removed repo-wide this session). Push via `bash push.sh` then resync `origin/main` (see `../handoff.md` §2).

## 5. Novelty positioning (important — read before writing claims)

Control-theoretic / PID steering was **already introduced for AR LLMs** by
**Nguyen et al., "PID Steering", ICLR 2026, [arXiv:2510.04309](https://arxiv.org/abs/2510.04309)** —
they cast steering as a P controller and add PID, with the loop over **layers (depth)**.
**Do NOT claim the control-theoretic framing.** Our defensible novelty:
1. the loop is over the **denoising trajectory (time)** — re-correcting the *same* positions across steps before commit (AR has no such loop);
2. the **DLM-specific erosion finding** (open-loop reverts under re-contextualization — absent in the layer-depth setting);
3. the **bias-injection application** (directional demographic bias on BBQ).
Also cite: activation clamping (Templeton 2024; AxBench 2501.17148), ACE affine (2411.09003).

## 6. Open threads / next steps

1. **Firm up the headline (highest value):** re-score clamp/cmom on the **full**
   Black-referent set (beyond n=37) + a **`c*` sweep** + **P-vs-PI ablation**
   (is cmom's edge the integral term or just effective strength?) + a **2nd
   attribute** (toxicity). This turns "n=37 indicative" into solid.
2. Other layers / a `c*` × layer grid.
3. The underlying **bias-injection / aim-vs-disinhibit** result (earlier sessions,
   `../directional_steering/`, `../docs/research_plan.md`) is the stronger, more
   unoccupied contribution — closed-loop is a *method* on top of it.

## 7. Other repo changes this session (not closed-loop)

- **Ghostwriter** collapsed to a **single fabricated-evidence injection** (removed the invented none/mild/strong/repeated ladder), re-ran (`baseline/`).
- **Literature:** added ILRR (2601.21647), "Steering Without Breaking" (2605.10971); located Shnaidman "Activation Steering for MDLMs" (2512.24143); added `../docs/baselines.md` (5 papers); hyperlinked all arXiv refs.
- `../docs/research_plan.md` (Aim-vs-Disinhibit framing).
- Removed the Claude co-author trailer from all 26 commits (history rewrite + force-push).
