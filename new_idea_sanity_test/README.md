# new_idea_sanity_test — Closed-loop activation steering for DLMs

Code + results for the closed-loop steering study. Writeup: **`paper_draft.md`**.

Idea: standard steering adds a fixed vector every denoising step (**open-loop**);
we instead **measure** the attribute level `a = ⟨h, v̂⟩` at block L14 and **correct
it to a setpoint `c*`** every step — **clamp** (proportional, P) and **cmom**
(clamp + integral/EMA, PI). See `paper_draft.md` §2 for the full method.

## Where each piece of code lives

Two tasks, so the code spans two folders (this one for the sentiment sanity check;
`eval/` + `directional_steering/` for the BBQ bias result):

| file | what it does |
|---|---|
| **`run.py`** | Sentiment/formality harness on LLaDA (single-block gen). Builds the contrastive direction at L14, then generates under any of the conditions below and saves `results/{attr}_{cond}.jsonl` (+ `nfe`). |
| **`score.py`** | Scores the generations with an offline classifier (`sentiment`→distilbert-sst2 `P(neg)`; `formality`→roberta-formality-ranker) and prints the attribute × nfe × length table. |
| **`validate.py`** | λ-sweep sanity: confirms the steering vector actually moves the attribute (with a fluency check) before trusting anything. |
| **`adadir_probe.py`** | Diagnostic: does the steering *direction* change across noise levels? (builds `v̂` at mask-ratios 0…0.9, reports coherence + pairwise cosine). |
| `results/` | generated `*.jsonl` per condition + `{attr}_direction.pt`. |
| **`paper_draft.md`** | title / abstract / method / results / related work. |

**BBQ bias (the headline result) — code in sibling folders:**

| file | what it does |
|---|---|
| **`../eval/bbq_eval.py`** | BBQ harness. Closed-loop added via `--steer-mode {add,clamp,cmom}`, `--cstar` (setpoint), `--beta` (PI momentum), `--layers` (multi-layer; per-layer `c*` = layer natural proj + offset). |
| **`../directional_steering/measure_proj.py`** | Calibration: natural projection `⟨h,v̂⟩` per layer → `nat_proj_by_layer.json` (used to set per-layer `c*`). |
| **`../directional_steering/analyze_closedloop.py`** | Prints the open-loop vs closed-loop Black-referent table (black/nonblk/d_gap/acc_disambig). |
| `../directional_steering/race_black_anchored_text.pt` | the item-anchored "prefer the Black option" direction (L14). |

## `run.py` conditions

`--conditions` (comma list): `none`, `uniform` (open-loop additive), `clamp` (P),
`cmom` (PI), and the negative controls `decay` / `warmup` / `conv` / `clampconf`
(open-loop schedules — all fail), `amom` (additive+momentum), `sphere` (rotation),
`nadd` (norm-preserving), `sadi`/`sadidir` (SADI port). Key flags: `--sign -1`
(steer toward negative), `--lam-frac` (open-loop α = frac·act_norm),
`--clamp-target` (`c*`), `--beta`, `--slerp-t`, `--sadi-topk/-delta`.

## Reproduce

Env: `/home/lukas/miniconda3/envs/sarim_awm/bin/python` (transformers 4.46.2). GPU: `CUDA_VISIBLE_DEVICES=<n>`.

**Sentiment (Table 3.1):**
```bash
# open-loop, clamp(P), cmom(PI) at the working strength
CUDA_VISIBLE_DEVICES=5 python new_idea_sanity_test/run.py --attr sentiment \
  --gen-length 64 --block-length 64 --steps 64 --sign -1 --n-prompts 24 \
  --conditions none,uniform,clamp,cmom --lam-frac 0.6 --clamp-target 60 --beta 0.8
python new_idea_sanity_test/score.py --attr sentiment
# (validate the vector first, optional) 
CUDA_VISIBLE_DEVICES=5 python new_idea_sanity_test/validate.py --sign -1 --fracs 0,0.3,0.6,0.9
```

**BBQ Black-bias (Table 3.2 / README table):**
```bash
D=directional_steering/race_black_anchored_text.pt
R=directional_steering/results
# single-layer L14 (open-loop a4/a8 already exist as bbq_L14_anchored_a{4,8}.json)
CUDA_VISIBLE_DEVICES=5 python eval/bbq_eval.py --layer 14 --direction-path $D \
  --steer-mode clamp --cstar 60           --out $R/bbq_L14_anchored_clampC60.json
CUDA_VISIBLE_DEVICES=7 python eval/bbq_eval.py --layer 14 --direction-path $D \
  --steer-mode cmom  --cstar 60 --beta 0.8 --out $R/bbq_L14_anchored_cmomC60.json
# all-layer (needs the per-layer calibration file first)
python directional_steering/measure_proj.py     # writes nat_proj_by_layer.json
CUDA_VISIBLE_DEVICES=5 python eval/bbq_eval.py --layer 14 --direction-path $D --layers all \
  --steer-mode clamp --cstar 2             --out $R/bbq_alllayer_clamp2.json
CUDA_VISIBLE_DEVICES=7 python eval/bbq_eval.py --layer 14 --direction-path $D --layers all \
  --steer-mode cmom  --cstar 2 --beta 0.8  --out $R/bbq_alllayer_cmom2.json
# print the open-vs-closed table
python directional_steering/analyze_closedloop.py
```

## Result (headline)

Closed-loop (clamp/cmom) at L14 beats open-loop additive on **both** directionality
and competence on BBQ Black-referent items (`d_gap` +0.216 vs +0.135; acc_disambig
0.91 vs 0.874). Single-layer L14 > all-layer. See `paper_draft.md` §3 and the repo
README for the full tables.

**Honest status:** LLaDA-8B only, one layer, sentiment n=24 / bias n=37 (magnitudes
indicative, directional sign reliable), single `c*`. Not yet firmed up (full
Black-referent set + `c*` sweep + 2nd attribute pending).

## Updates

- 2026-07-08 (latest): `paper_draft.md` — **restricted baselines to steering
  methods only.** Dropped **Ghostwriter** (input-space prompt-injection, Yang et
  al. 2606.06244) as a baseline — different channel and threat model, out of
  scope for a steering-method comparison; kept only as a one-line related-work /
  baselines-rationale note. Removed its baseline row, its result numbers
  (0.791→0.589, 3:1 split), and the E3 "Ghostwriter overlay". Aim-vs-disinhibit
  argument now stands on steering-only contrast (group mean-diff +0.05 vs
  item-anchored answer-text +0.135). No steering-method numbers changed.
- 2026-07-08 (later): `paper_draft.md` — **populated from VERIFIED docs after
  fact-checking every citation (16/16 real) and cross-checking claims vs
  code/results.** Reframed the control theory: clamp = proportional (P)
  controller; **cmom = EMA / leaky integrator with unity steady-state gain, NOT a
  classical PI/integral** (this *predicts* the observed clamp≡cmom on the bias
  task). **PID-Steering (arXiv:2510.04309) demoted from "we build on" to
  concurrent related work** — proportional/integral control is a classical
  primitive; they loop over layer-depth (AR), we over denoising-time (DLM).
  Rewrote related-work / baselines / benchmarks / experiments grounded in
  `docs/literature.md`, `docs/baselines.md`, `docs/research_plan.md` (real E1–E10
  spine + separate closed-loop analyses). Fixed: BBQ hook steers **all positions**
  (not generated-only) in the bias setting; open-loop uses the **raw** `v`,
  closed-loop the **unit** `v̂`; added an **s_AMB caveat** (reuses disambiguated
  leaning; validate vs official BBQ). No result magnitudes changed.
- 2026-07-08 (earlier, SUPERSEDED framing): `paper_draft.md` — first added the
  PID-Steering motivation (as "we build on their framing") and a Planned
  Experiments section; the "build on" framing was corrected later the same day
  (see entry above). No numbers changed.
- 2026-07-07: `paper_draft.md` — clarity/explanation rewrite (core idea moved up
  front, tighter math exposition, erosion mechanism spelled out, sharper table
  captions); no numbers, results, or claims changed.
