# llada_moe/ — full bias-steering port to LLaDA-MoE-7B-A1B-Instruct

Port of the complete pipeline (arrows, decode-PID ours, layer-PID/normal
open loop, and the 6-method prior-work baseline suite) to
**inclusionAI/LLaDA-MoE-7B-A1B-Instruct**, exactly parallel to the Dream port
(`dream/`).  Target: the same 9-row balanced Black table
(base, decode-PI ours, CAA, ActAdd, Mean-AcT, Linear-AcT, AurA-inject,
AurA-vanilla, ITI-C) on `results/balanced/_sweep400_rot{0,1,2}.jsonl`, scored
by `balanced_all/strict_round2.py` (family 9, soft-missing).

## Model facts (verified: local snapshot config + modeling code, CPU meta-load
## under the repo's pinned transformers 4.46.2)

| fact | value | note |
|---|---|---|
| params | 7.36B total, ~1B active | 64 experts, top-8 routing |
| layers | 16 `LLaDAMoEDecoderLayer` | `model.layers` (AutoModel → `LLaDAMoEModelLM`) |
| d_model | 2048 | block residual width |
| heads | 16 × 128 (no GQA) | `o_proj` input = 2048 concat heads |
| MoE MLP | 64 × `LLaDAMoEMLP` (1024-d experts) | `layers[k].mlp` = `LLaDAMoESparseMoeBlock`, no shared expert |
| mask id | 156895 (`<|mask|>`) | == the model README's `mask_id` |
| eos/pad | 156892 (`<|endoftext|>`) | |
| chat template | `<role>HUMAN</role>…<|role_end|><role>ASSISTANT</role>` | Instruct model; used everywhere |
| sampler | **external LLaDA loop** | NO `diffusion_generate`; the model README's `generate()` is line-for-line the LLaDA loop, so we reuse `eval/bbq_eval.generate` verbatim with `mask_id=156895` |
| logits | position-aligned | no Dream-style next-token shift anywhere |

Same block/step semantics as LLaDA-8B: `gen_length=32, steps=64,
block_length=32, temperature=0, cfg_scale=0, remasking=low_confidence`.

## Deviations from the Dream/LLaDA ports (all deliberate, all documented in
## the module docstrings)

* **MLP granularity (`mlp_hidden`) = the MoE block OUTPUT (2048-d)**, hooked
  with plain forward hooks on `model.layers[k].mlp` (bare-tensor output).
  There is NO dense gated hidden bank (Dream 18944-d / LLaDA 12288-d): tokens
  are routed across 64 experts of 1024-d each, so per-expert down_proj inputs
  see variable token subsets and per-neuron stats there are ill-posed.  The
  MoE block output is the densest per-neuron MLP bank every token passes
  through — and editing a module OUTPUT is exactly AcT's InterventionHook
  contract, so Linear-AcT/AurA here need no pre-hook workaround at all.
  Affects: `baselines/calib.py --where mlp_hidden`, `baselines/linearact.py`,
  `baselines/aura.py`.
* **Layer midpoint = 8** (of 16), the analogue of L14/28 (Dream) and L14/32
  (LLaDA): vhat source layer for decode-PID, `--source-layer` default for
  normal mode, `--layer` default for CAA/ActAdd.
* **Conservative decode-PID `ALPHA_MAX = 1.0`** (Dream-parity: LLaDA's amax 6
  was 97% garbage on Dream).  `round4_jobAB` sweeps `--amax {0.5,1,2,4}`; the
  balanced job takes the calibrated dose via `$MOE_AMAX`.
* **ITI-C top-K = 48 of 256 heads** (16×16) at alpha 8 — keeps the LLaDA
  published operating-point stem `cond_itic_K48_a8`; note 48/256 is a larger
  fraction of the head pool than LLaDA's 48/1024 or Dream's 48/784.

## Files

```
common_lladamoe.py    shared infra: facts, hook factories (tuple contract),
                      generate() via bbq_eval's LLaDA loop, run_items()
                      writing the shared cond_<tag>.json schema. --selftest/--smoke
build_arrows.py       16x2048 answer-anchored diff-in-means arrows ->
                      llada_moe/arrows.pt (contamination-safe held-out set,
                      selection reused VERBATIM from steering/build_arrows.py)
denoise_pid.py        decode-space closed loop (ours): external LLaDA sampler
                      with per-step PID on p_target; PID law imported VERBATIM
                      from steering/denoise_pid.py. --selftest/--smoke/--cond
pid_steer.py          open-loop layer-space PID + normal mode; build_u/GAINS
                      imported VERBATIM from steering/pid_steer.py
baselines/
  directions.py       gaussian_ot/empirical_ot_fit/auroc imported VERBATIM
                      from baselines/directions.py; load_arrows -> (16,2048)
  calib.py            labelled Black/other activation collection
                      (block / mlp_hidden / attn_head)
  caa.py actadd.py meanact.py linearact.py aura.py itic.py
                      the 6 prior methods at MoE-native granularities
  run_all.py          one-CLI orchestrator (fit/run any subset, shared model)
```

Every module has a `--selftest` CPU path (13/13 PASS offline); GPU entry
points are `--smoke` / `--fit` / `--run` / `--cond`.

## SLURM DAG (round 4)

```
sbatch slurm/round4_jobAA_lladamoe_smoke.sbatch     # debug partition gate
#   GATE: 20-item clean unparseable_rate must be low (~<0.2) before continuing
A=$(sbatch --parsable slurm/round4_jobAB_lladamoe_arrows.sbatch)
B=$(sbatch --parsable --dependency=afterok:$A slurm/round4_jobAC_lladamoe_fits.sbatch)
# read the jobAB dose sweep, then:
sbatch --dependency=afterok:$B \
  --export=ALL,MOE_AMAX=1.0,MOE_CAA_ALPHA=2,MOE_ACTADD_ALPHA=8 \
  slurm/round4_jobAD_lladamoe_balanced.sbatch
```

Outputs: dose sweep under `results/lladamoe/`, the 9×3 balanced table under
`results/lladamoe_balanced/<cond>/rot{r}/`, analyzed by
`python balanced_all/strict_round2.py` (family 9 activates automatically once
the directory exists; stems, incl. the submit-time doses, are discovered from
disk).

## Compatibility

The snapshot's remote code (`configuration_lladamoe.py` /
`modeling_lladamoe.py`) loads cleanly under the repo pin
**transformers==4.46.2** (dlm env): config, fast tokenizer, chat template and
a full meta-device model build were verified on CPU.  `AutoModel` maps to
`LLaDAMoEModelLM`; forward asserts `labels is None` (we never pass labels).
Model weights: symlink `LLaDA-MoE-7B-A1B-Instruct` at the repo root (here →
`/shared/home/sarim.hashmi/LLaDA-MoE-7B-A1B-Instruct`, ~14 GB).
