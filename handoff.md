# Handoff — DLM Bias-Injection Project

Onboarding for a fresh session. Read this + `docs/literature.md` + each folder's README.

## 1. What this project is
**Training-free, inference-time bias _injection_ into a diffusion LM, measured on BBQ.**
Model = **LLaDA-8B-Instruct** (masked/discrete diffusion LM, frozen weights). Our method
is **activation steering**: a PyTorch forward hook adds `α · direction` to the residual
stream at transformer **block L14** during the denoising forward passes. We measure the
effect on **BBQ** (Bias Benchmark for QA).

Repo: `github.com/Sarim-MBZUAI/dlm_bias` (private). This working tree IS on the deploy node
(`/home/lukas/...`), so "deploy" = merged code + regenerated PDF present here.

## 2. Environment & how to run (IMPORTANT)
- **Python env:** `/home/lukas/miniconda3/envs/sarim_awm/bin/python` — has `transformers==4.46.2`,
  which is REQUIRED. `shashmi_etihad` / any `transformers` 5.x **fails to load LLaDA** with
  `AttributeError: 'LLaDAModelLM' object has no attribute 'all_tied_weights_keys'`.
- **GPU:** `CUDA_VISIBLE_DEVICES=<n>`; ~17 GB/run (two fit on one 80 GB card). A 1000-item
  BBQ run ≈ 30 min solo. Check free GPUs with `nvidia-smi`.
- **Model path** is absolute in the scripts (`/home/lukas/users/shashmi/dlm_bias/LLaDA-8B-Instruct`),
  so scripts work from any worktree.
- **Push:** HTTPS remote, no interactive creds. Use `bash push.sh` (sources `GITHUB_TOKEN`
  from `.env`, pushes `HEAD:main`). NOTE: `push.sh` pushes via a token URL, which does NOT
  update the local `origin/main` tracking ref — after pushing, resync it or `git status`
  will falsely say "ahead N":
  ```
  set -a; . ./.env; set +a
  git update-ref refs/remotes/origin/main "$(git ls-remote "https://x-access-token:${GITHUB_TOKEN}@github.com/Sarim-MBZUAI/dlm_bias.git" refs/heads/main | awk '{print $1}')"
  ```

Run examples (from repo root, env = sarim_awm python):
```bash
# clean BBQ baseline
CUDA_VISIBLE_DEVICES=0 <py> eval/bbq_eval.py                       # -> eval/results/bbq_clean.json
# steer with a direction at L14
CUDA_VISIBLE_DEVICES=0 <py> eval/bbq_eval.py --layer 14 --alpha 8 \
    --direction-path <path>.pt --out <out>.json
# Ghostwriter baseline (input-space, no steering)
CUDA_VISIBLE_DEVICES=0 <py> eval/bbq_eval.py --attack ghostwriter --gw-strength strong
# build a steering direction
CUDA_VISIBLE_DEVICES=0 <py> bias_steering/build_direction.py --source crows --layers 14 \
    [--categories race_color] [--subset-terms "black,african american"]
# regenerate the report PDF
<py> docs/make_report.py                                            # -> docs/dlm_bias_report.pdf
```

## 3. Repo layout
```
eval/            BBQ harness. bbq_eval.py (generation-based MC + optional steering hook +
                 --attack ghostwriter), bias_metrics.py (attack_metrics/flips),
                 attack_metrics.py (backfill). results/: bbq_clean, bbq_L14_race_color_a{8,16,32}.
bias_steering/   build_direction.py (--source crows|json, --layer, --subset-terms/-name),
                 directions/L14/*.pt (GITIGNORED — rebuild from data).
baseline/        GHOSTWRITER = input-space baseline ONLY (Yang et al. 2606.06244 port; NOT
                 our method). ghostwriter.py, compare_baseline.py, backfill_evidence.py,
                 figs/, results/bbq_ghostwriter_{none,mild,strong,repeated}.
race_steering/   Black-targeted GROUP mean-diff experiment. build_black_data.py,
                 black_analysis.py, race_black.pt, data/black_pairs.{csv,json} (792 pairs:
                 261 CrowS + 531 StereoSet), results/bbq_L14_race_black_a{4,8,12,16,24,32}, figs/.
directional_steering/  ITEM-ANCHORED CAA experiment (the win). build_anchored.py,
                 anchored_analysis.py, race_black_anchored_text.pt (the good vector),
                 race_black_anchored.pt (failed letter version, kept for the record),
                 data/anchored_items.jsonl, results/bbq_L14_anchored_a{1,2,3,4,6,8}, figs/.
docs/            make_report.py -> dlm_bias_report.pdf ; literature.md ; figs/ ;
                 embedding_layer_steering.md.
push.sh          token push helper.  .env  holds GITHUB_TOKEN (gitignored).
```
`.pt`, `*.csv`, `eval/results/`, caches are gitignored — result/data/`.pt` files that ARE
tracked were **force-added** (`git add -f`). BBQ data cached in `eval/.bbq_cache/`.

## 4. Findings (the arc — this is the story)
BBQ metrics: official `s_AMB`/`s_DIS` **plus** absolute pick-rate "attack metrics"
(abstention_rate, target_rate, nontarget_rate, unknown→group flips).

1. **Clean LLaDA is near-unbiased on BBQ** (`s_AMB ≈ 0`), abstains ~0.79 on ambiguous items.
2. **Embedding-layer (wte) steering is too shallow** — re-contextualized downstream; no bias.
   → moved to mid-block **L14** (standard CAA/RepE site).
3. **Group mean-diff steering (CrowS/StereoSet) is a NON-DIRECTIONAL abstention-suppressor.**
   Raising α collapses "Unknown" but the freed picks split ~evenly across groups; `s_AMB`
   stays ~0 (target/non-target cancel). At high α it just breaks competence.
4. **Ghostwriter (input-space BASELINE) reproduces the same abstention-collapse** — so the
   phenomenon isn't specific to activation steering. It saturates early, no directional aim.
   (Baseline for contrast only — NOT our method.)
5. **Black-targeting + 3× data (792 pairs) did NOT add directionality** — peak directional
   gap +0.05, same as before; more data just lowered the competence cliff. (`race_steering/`)
6. **BREAKTHROUGH — item-anchored answer-text steering IS directional.** (`directional_steering/`)
   Build the direction from held-out BBQ items' L14 activations when the model answers
   "**The Black man**" vs "**The white man**" (answer-TEXT, NOT the letter A/B/C — the letter
   version failed at split-half 0.27; answer-text = 0.98). Result on Black-referent ambiguous
   items (n=37): black_pick 0.162→0.568, non-black 0.081→0.351, **directional gap +0.135 @ α=8
   (~2.5× the group vector), Black-pick > non-Black-pick at every α, no competence cliff**
   (acc_disambig 0.97→0.87). First vector that AIMS rather than just erodes abstention.

**Method lesson:** directionality comes from HOW the direction is built (decision-anchored,
answer-text contrast), not data volume or subset. Concept/group mean-diff = suppressor.

## 5. Gotchas / landmines
- **env**: must be `sarim_awm` (transformers 4.46.2); 5.x breaks LLaDA.
- **`DEFAULT_OUT` in `eval/bbq_eval.py` is an absolute main-tree path** → runs launched from a
  worktree write results into the MAIN tree unless you pass explicit `--out`. Pass `--out`.
- **`--direction-path`**: pass the absolute `.pt` path; bbq_eval sanity-checks `saved["layer"]==--layer`.
- **Stdout is block-buffered** to log files — per-item `[N/1000]` progress won't flush until
  the run ends. Don't infer progress from logs; use `nvidia-smi` + result-file existence.
- **`n=37`** Black-referent ambiguous items in the seed-42 sample is the MEASUREMENT bottleneck
  — magnitudes are noisy (~2.7 pts/item). Directional *sign* is consistent; magnitudes indicative.
- **Coherence gate**: always report split-half cosine before trusting a direction (≥~0.7 usable).
- Ghostwriter has **no alpha** (input-space); don't overlay its ordinal doses on the α axis.
- SemDLM+ in `literature.md` is **UNVERIFIED** (couldn't be found) — flagged, not described.

## 6. Open threads / next steps
1. **Measurement fix (highest value):** re-score the winning anchored vector on FAR more
   Black-referent items (beyond the 37 in the random-1000 — use the full Race_ethnicity
   Black-referent set) to turn "consistent but n=37" into a solid magnitude.
2. Finer/again α sweep on the anchored vector; try other layers.
3. Item-anchored directions for other groups / categories (generalize beyond Black).
4. Control the decoding confound (`literature.md` bucket 2): hold sampler/steps/positions
   identical across conditions; add answer-position randomization checks.
5. Possible: distillation-cascade (persist the ephemeral steering into weights) — see the
   6-paper related-work read in the session history.

## 7. Working style / workflow
- Orchestrator + fan-out subagents; each feature built in an isolated **git worktree**, then
  `--ff-only` merged to `main` and pushed via `push.sh`, worktree + branch cleaned up.
- Keep experiments in **separate top-level folders** (baseline / race_steering /
  directional_steering) — do not mix results.
- Report changes go through `docs/make_report.py` (regenerate the PDF). No PDF-render tool on
  the node (no poppler) — verify report content by capturing reportlab flowables.
- Persistent findings live in the session memory file (`.claude/.../memory/dlm-bias-project.md`).

## 8. Current state
`main` @ latest (all experiments merged & pushed). Uncommitted in the working tree (left
intentionally, unrelated to the above): `chat.py` (Dream Base→Instruct default path change)
and untracked `prompts.txt`.
