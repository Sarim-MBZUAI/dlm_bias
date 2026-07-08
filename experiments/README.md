# Experiments E1 & E2

Runnable harness for the two Phase-1 experiments in `docs/research_plan.md`.
Everything reuses `eval/bbq_eval.py` (generation + steering + scoring) and
`directional_steering/anchored_analysis.py` (Black-referent metrics); nothing
re-implements generation or scoring.

```
E1 ── measurement fix (the GATE) ─────►  E2 ── construction ablation (the HEART)
 turn the n=37 pilot into n=1600          which CONSTRUCTION aims vs disinhibits,
 with bootstrap + McNemar CIs             at matched effective strength
```

## What E1 and E2 are

- **E1 — the gate.** The published closed-loop headline (`d_gap +0.216` clamp/cmom
  vs `+0.135` open-loop) is on only **n=37** Black-referent ambiguous items from the
  seed-42 random-1000 sample. E1 rescores the same conditions on the **full**
  Black-referent ambiguous set (**n=1600**) and attaches a **bootstrap 95% CI** and
  an **exact McNemar test** to each `d_gap`. If the gap's CI excludes 0, the effect
  is real; if not, the pilot was noise.
- **E2 — the heart.** Holds strength/layer/eval-set fixed and varies **only the
  direction construction**: group mean-diff · letter-anchored · answer-text-anchored
  · FairPCA-subspace. Reports coherence × `d_gap` (aim) × Δabstention (disinhibit) ×
  competence × aiming-ratio. Expected: **answer-text is the only construction high on
  the aim axis**; group mean-diff / FairPCA collapse abstention with ~0 gap.

## Contamination rule

The answer-text direction was built from 400 held-out Black-referent ambiguous
Race_ethnicity items (`directional_steering/data/anchored_items.jsonl`). The E1/E2
eval set **excludes those 400** — `build_eval_set.py` asserts disjointness and also
asserts the seed-42 pilot n=37 is a **subset** of the eval set (so the large-n number
is a strict superset of the pilot). Verified counts: 2000 Black-referent ambiguous
total − 400 build = **1600 eval**.

## Matched-strength protocol (E2)

Open-loop `add` normally applies the **raw** direction (`alpha·v`), so constructions
with different `‖v‖` get different effective pushes — an unfair comparison. E2 passes
`--normalize-direction`, so every construction is unit-normalized and the effective
projection push equals `alpha` for all four. We sweep matched `alpha ∈ {8, 20, 40}`.
(`clamp`/`cmom` already use the unit vector, so this only changes `add`.)

## Run order

```bash
PY=/home/lukas/miniconda3/envs/sarim_awm/bin/python

# 0. build the eval set (CPU, seconds) — also runnable standalone
$PY experiments/build_eval_set.py

# 1. E1 (GPU 5/6/7; ~45-60 min per 1600-item run, 2 fit per 80 GB card)
bash experiments/run_e1.sh          # runs 5 conditions + prints the CI table

# 2. E2 (GPU 5/6/7; builds FairPCA dir first, then 4 constructions × 3 alphas)
bash experiments/run_e2.sh          # reuses e1_clean as baseline; prints ablation
```

Analysis is also runnable standalone once results exist:
`$PY experiments/e1_analyze.py` and `$PY experiments/e2_analyze.py --alpha 20`.

**GPU rule for this project: use ONLY `CUDA_VISIBLE_DEVICES` 5, 6, or 7.**

## Files

| file | role | GPU? |
|---|---|:--:|
| `build_eval_set.py` | build the n=1600 eval set; assert disjoint + pilot⊆eval | no |
| `build_fairpca.py`  | build the FairPCA-subspace direction (E2 construction #4); `--self-test` for offline PCA check | yes (build) / no (`--self-test`) |
| `e1_analyze.py`     | E1 table: rates, `d_gap`, bootstrap 95% CI, exact McNemar; offline n=37 validation | no |
| `e2_analyze.py`     | E2 construction-ablation table (aim vs disinhibit) | no |
| `run_e1.sh` / `run_e2.sh` | exact GPU run commands | yes |

## Offline validation performed (before any GPU run)

1. **Eval-set integrity** (`build_eval_set.py`): 1600 eval items, disjoint from the
   400 build items, seed-42 pilot n=37 ⊆ eval — **PASS**.
2. **Analysis correctness** (`e1_analyze.py --validate-only`): recomputing `d_gap` on
   the existing seed-42 n=37 files reproduces the published headline exactly —
   `n=37, black=0.568, non-black=0.351, d_gap=+0.135` — **PASS**. This proves the
   metric pipeline before the big runs.
3. **FairPCA math** (`build_fairpca.py --self-test`): recovers a known principal axis
   (cos 1.000) and stays an axis under anti-coherent pairs where the mean-diff
   collapses — **PASS**.

## Note on `bbq_eval.py` additions (backward-compatible)

- `--items <jsonl>`: evaluate a fixed item list instead of the random-N sampler
  (default off → identical to before).
- `--normalize-direction`: unit-normalize the direction (matched-strength; only
  affects `add` mode; default off → raw-vector behavior).
- **Bug fix:** the `cmom` EMA integrator is now reset per **item** (via `reset()` in
  the loop), not on tensor-shape change — the old guard leaked integral state across
  consecutive same-length items. Correctness-critical for E1's cmom condition.

## Comparison matrix (baselines @ own config vs OURS full-layer, across BBQ + UNQOVER)

The capstone table: every steering method run against ours, on both benchmarks.

```
                     BBQ (--items black_referent_ambig_eval.jsonl)   UNQOVER (unqover_eval.py)
clean                        ●                                              ●
CAA            (own cfg L/α)  ●                                              ●
ActAdd         (own cfg L/α)  ●                                              ●
group mean-diff (own cfg L/α) ●                                              ●
OURS clamp (P)  full-layer    ●                                              ●
OURS cmom  (PI) full-layer    ●                                              ●
```

- **Baselines** run at **their own** layer+α (read from `baselines/*/config.json`); **OURS** is full-layer closed-loop (`--layers all`, `clamp`/`cmom`, `c*=60`).
- **BBQ block** reuses `e1_analyze` (d_gap + bootstrap 95% CI + McNemar p, abstention, target/nontarget pick-rate, acc_disambig).
- **UNQOVER block** reuses `datasets/unqover/unqover_metric` (μ bias-intensity, mean|C|, signed `pref_gap` toward the target subject, and Δμ / Δpref_gap vs clean).

### Run order (end to end, GPUs 5/6/7 only)
```bash
PY=/home/lukas/miniconda3/envs/sarim_awm/bin/python
# 1. BBQ eval set (CPU) + baseline directions (GPU)
$PY experiments/build_eval_set.py
bash baselines/run_baselines.sh
# 2. UNQOVER data (CPU download + load)
$PY datasets/unqover/download_unqover.py --classes ethnicity
$PY datasets/unqover/unqover_loader.py \
   --source datasets/unqover/data/generated/ethnicity.source.json \
   --out datasets/unqover/data/ethnicity.items.jsonl --limit 2000 --seed 42
# 3. the matrix (GPU) + the table (CPU)
bash experiments/run_matrix.sh          # runs 6 methods × 2 benchmarks, then builds the table
# or just (re)build the table from existing results:
$PY experiments/build_comparison_table.py --target-subject African
```
`build_comparison_table.py` renders missing cells as "—" (partial matrix still tabulates) and has a `--selftest` that reproduces the +0.135 pilot from existing result files.
