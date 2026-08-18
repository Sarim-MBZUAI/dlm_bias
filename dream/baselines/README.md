# Dream-v0-Instruct-7B steering baselines (Lane C port)

Port of the LLaDA `baselines/` AcT/steering suite onto **Dream-v0-Instruct-7B**
(28 layers, hidden 3584, MLP 18944, 28 heads × 128, MASK_ID 151666).  Every method
drives the shared Dream harness `dream/common_dream.py` (`run_items`, the
tuple-contract hook factories, one model load, the LLaDA result schema).  The
per-neuron OT / AURA math is **imported verbatim** from the model-agnostic LLaDA
`baselines/directions.py` (`gaussian_ot`, `empirical_ot_fit`, `auroc_per_neuron`);
only `load_arrows` is overridden to read `dream/arrows.pt` (28×3584).

## Methods

| method | paper | hook site (Dream) | hook kind | layers | edit |
|---|---|---|---|---|---|
| `caa` | Contrastive Activation Addition | `model.layers[L]` output[0] (3584 residual) | forward hook | 1 (L=14) | `h += α·unit(r[L])` |
| `actadd` | ActAdd (Turner 2023) | `model.layers[L]` output[0] (3584) | forward hook | 1 (L=14) | `h += α·unit(r₁[L])`, r₁ = **n=1** pair |
| `meanact` | Mean-AcT (OnlyMeanHook) | `model.layers[k]` output[0] (3584) | forward hook | all 28 | `h += s·1.2·(μ₂−μ₁)_k` |
| `linearact` | Linear-AcT (1-D OT) | `model.layers[k].mlp.down_proj` **input** (18944) | forward **pre** | all 28 | `x ← β_eff·x + bias_eff` (gaussian/empirical) |
| `aura` | AurA (Suau 2024) | `model.layers[k].mlp.down_proj` **input** (18944) | forward **pre** | all 28 | `x ← x·gate_k` (mul) |
| `itic` | ITI-C (Li 2023) | `model.layers[k].self_attn.o_proj` **input** (28×128) | forward **pre** | top-K heads | `x_h += α·σ_h·θ_h` |

"Black" is the injection destination throughout; positive α/strength/γ pushes the
model toward the Black answer.  All edits are open-loop (fixed per step), applied to
**every token, every diffusion step**, bidirectional — Dream's native
`diffusion_generate` sampler is untouched.

### Granularity notes (Dream analogues of LLaDA `attn_out` / `ff_out`)
- **`down_proj` input = 18944-d gated MLP activation** (`down_proj(act(gate)*up)`):
  no module *outputs* it, so `linearact`/`aura` steer it as `down_proj`'s **input**
  via a forward-pre hook — the same tensor `calib(where='mlp_hidden')` captures.
- **`o_proj` input = 3584-d concat of the 28 QUERY heads.**  Dream is GQA (4 kv
  heads) but `o_proj`'s input is the full query-head concat, so ITI-C's per-head
  reshape is `(…,28,128)` over query heads (NOT kv heads).

## Files
- `calib.py` — collect labelled Black/other activations at `block` / `mlp_hidden` /
  `attn_head`, masked-mean over the answer span of one clean forward.  Reuses
  `dream/build_arrows.py`'s contamination-safe held-out selection.  → `cache/calib_<where>.pt`.
- `directions.py` — thin: re-exports the LLaDA pure primitives; `load_arrows` → `dream/arrows.pt`.
- `caa.py` `actadd.py` `meanact.py` `linearact.py` `aura.py` `itic.py` — the 6 methods.
- `run_all.py` — registry-driven orchestrator (shared model load, fit-then-run).

Each method file: `--selftest` (offline math, no GPU), `--fit` (GPU artifact build,
where needed), `--run` (GPU eval via `run_items`).

## Run
Offline (no GPU):
```
python dream/baselines/run_all.py --selftest      # asserts the 6-method contract
python dream/baselines/caa.py --selftest          # (…and each method)
```

### Per-method CLIs (canonical)
The shipped/verified flow drives each method's own CLI (fit → run) at its own
strength — this is what produced `dream/RESULTS.md`, scripted end-to-end in
**`dream/run_dream_gpu4.sh`** (the reference for the exact per-method picks).
GPU, pinned env (`requirements.txt` transformers==4.46.2):
```
# arrows prerequisite (caa/meanact):
CUDA_VISIBLE_DEVICES=N python dream/build_arrows.py
# one method, its own fit + strength:
CUDA_VISIBLE_DEVICES=N python dream/baselines/aura.py --fit
CUDA_VISIBLE_DEVICES=N python dream/baselines/aura.py --run --mode inject --gamma 4
# all six at the RESULTS.md picks:
bash dream/run_dream_gpu4.sh
```

### `run_all.py` (convenience only)
`run_all.py --fit --run --method all` batches all six under one model load, with
per-method error isolation (one failure doesn't abort the rest), `meanact` default
strength 2.0, and an automatic `actadd` fit before its run. Constraint: it applies
**one global `--strength`/`--alpha` to every method**, so it cannot reproduce the
per-method strengths behind `dream/RESULTS.md` — use the per-method CLIs (or
`run_dream_gpu4.sh`) for anything you intend to report.

Fitted artifacts land in `dream/baselines/cache/` (gitignored); eval output in
`results/dream/<method>/` (gitignored).

### Position-balanced parity runs (SLURM) — COMPLETE
The balanced Dream family (`results/dream_balanced/`, rotations
`results/balanced/_sweep400_rot{0,1,2}.jsonl`) got `caa` (L14 a2) and `actadd`
(a8) via `slurm/round2_jobF_dream.sbatch`. The remaining five prior-work
conditions run at the LLaDA balanced operating points and stems
(`cond_meanact_unit_s2`, `cond_gaussian_s1`, `cond_inject_g4`, `cond_vanilla`,
`cond_itic_K48_a8`) via:
```
sbatch slurm/round3_jobY_dream_fits.sbatch         # Dream calib + method fits
sbatch --dependency=afterok:<jobY_id> slurm/round3_jobZ_dream_baselines.sbatch
```
All fits are rebuilt from **Dream** activations (never reused from the LLaDA
caches); `balanced_all/strict_round2.py`'s Dream family
(`DREAM_BASELINE_CONDS`) picks the results up soft-missing.

**Status (2026-08-18): landed.** jobY = 20576 (dfits), jobZ = 20577 (dbal),
both exit 0; all 15 runs (5 conditions × rot0–2, 400 samples each) committed
under `results/dream_balanced/{meanact,linearact,aura_inject,aura_vanilla,itic}/`.
Full table + reading in `results/ROUND2_STRICT.md` §4 (parity addendum).
Headline: meanact-unit-s2 and aura-inject-g4 **collapse** on Dream (76.8% /
58.8% strict-invalid at the LLaDA doses); linearact / aura-vanilla / itic are
coherent but ns on Δg — decode-PI keeps the largest (and a significant) Δg
on Dream against the now-complete suite.

## Deviations from the LLaDA baselines (honest)
- **Sampler**: Dream's native `diffusion_generate` (entropy alg), not the copied
  LLaDA block-diffusion `generate`.  Baselines are open-loop edits, so the sampler
  is not touched — only hooks are attached.
- **Hook path fix**: `common_dream.BLOCKS_PATH` was `model.model.layers`; the real
  DreamModel tree is `model.layers` (DreamModel → `.model` (DreamBaseModel) →
  `.layers`).  Corrected here — the old value was latent because Lane A's only GPU
  smoke ran the *clean* baseline (`attach_fn=None`), which never resolves the path.
- **Env**: Dream's remote modeling code needs `transformers==4.46.2`
  (`ROPE_INIT_FUNCTIONS['default']`); it fails on transformers 5.x.
- **`meanact` default `direction='unit'`** (family convention, per-layer unit-normed
  arrows) drops the true native `(μ₂−μ₁)` magnitude; `--direction raw` / `fitted`
  recover full faithfulness, same as the LLaDA port.
- **`actadd`** refuses to substitute `arrows.pt` when its n=1 artifact is missing
  (that would be CAA), identical to the LLaDA guard.
- **`run_all` registry** mirrors the LLaDA one; `actadd` is listed `needs_fit=True`
  (its `run()` refuses to start without the single-pair artifact `cache/actadd_dir.pt`,
  so `run_all --fit` builds it automatically before the run).
- No `logits` shift is applied anywhere: all granularities read hidden states, and
  Dream's next-token shift applies only to logits.
