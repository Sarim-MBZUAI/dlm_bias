# dream/ — Dream-v0-Instruct-7B port of the steering / eval stack

Port of this repo's decode-space PID + activation-steering baselines from
**LLaDA-8B-Instruct** to **Dream-v0-Instruct-7B** (another masked-diffusion LM),
so the same BBQ bias-injection experiments run on a second model. This folder is
Lane A: the shared infra (`common_dream.py`) and the direction builder
(`build_arrows.py`). Lane B adds the two steering controllers (`denoise_pid.py`,
`pid_steer.py`), which import the shared infra here and the pure PID math from the
LLaDA `steering/` files.

## What's here

| file | role |
|------|------|
| `common_dream.py` | shared infra: constants, `load_model_tok`, module-path helpers (`block_path`/`o_proj_path`/`down_proj_path`), tuple-contract hook factories (`add_vec_hook`, `affine_pre_hook`, …) with a shared fire counter, a `generate()` BBQ wrapper over `model.diffusion_generate`, and the `run_items()` eval loop (writes `cond_<tag>.json` + `_samples.jsonl` in the LLaDA schema). |
| `build_arrows.py` | 28-layer "prefer Black option" diff-in-means arrows `r(k)`; reuses the LLaDA builder's contamination-safe held-out item selection; saves RAW `(28, 3584)` tensor → `arrows.pt` (gitignored) + `direction_examples.jsonl`. |
| `denoise_pid.py` | **(Lane B, the headline contribution)** decode-space PID: closes a feedback loop on `p_black(t)` over the 64 denoising steps and modulates one scalar `alpha(t)` broadcast as `alpha(t)·vhat` to all 28 blocks (vhat = unit `r[14]`). Implemented via Dream's **native** `generation_logits_hook_func`. Imports `PID`/`COND_MASK`/`p_black_from_logits`/… from `steering/denoise_pid.py` (math not reimplemented). |
| `pid_steer.py` | **(Lane B, prior-work comparison)** open-loop layer-space PID (Eq.18, `--mode pid`) + normal single-vector baseline (`--mode normal`); fixed per-block injection at all 28 blocks via `common_dream.run_items(attach_fn=…)`. Imports `build_u`/`unit_rows`/`GAINS` from `steering/pid_steer.py`. |
| `arrows.pt` | built directions (gitignored — large, rebuild locally). |
| `cache/` | default eval output dir (gitignored). |

## Model facts (Dream-v0-Instruct-7B)

- 28 layers, hidden 3584, MLP intermediate 18944, 28 heads × head_dim 128 (GQA, 4 KV heads), `MASK_ID = 151666`.
- Load with `AutoModel.from_pretrained(..., trust_remote_code=True, torch_dtype=bf16)` (**not** `AutoModelForCausalLM`).
- Block list: `model.layers` (`AutoModel` returns `DreamModel`; its `.model.layers` is the 28-`DreamDecoderLayer` `ModuleList`, so from the top object the path is `model.layers`); `.forward` returns a **tuple** `(hidden,)+…`. **Fix (Lane B):** `common_dream.BLOCKS_PATH` was `model.model.layers` (one `model.` too many) — never exercised because Lane A's `--smoke` used `attach_fn=None`; corrected to `model.layers` (verified via `named_modules` on GPU).
- **Environment:** Dream's remote code targets `transformers~=4.46`. Load + generate verified on `conda env sarim_awm` (transformers **4.46.2**). The default project shell (`shashmi_etihad`) now has transformers **5.9.0**, which is incompatible (`ROPE_INIT_FUNCTIONS['default']` removed, `GenerationConfig.validate` signature changed) — run all Dream GPU scripts with `/home/lukas/miniconda3/envs/sarim_awm/bin/python`.
- Attn out-proj input `layers[k].self_attn.o_proj` (3584-d concat heads) and MLP down-proj input `layers[k].mlp.down_proj` (18944-d gated act) are the Dream analogues of LLaDA's `attn_out` / `ff_out` pre-residual deltas.
- Sampler: `model.diffusion_generate(..., alg="entropy", alg_temp=0)`, full bidirectional attention; the returned sequence includes the prompt.

## Run commands

```bash
# offline self-tests (CPU, no model)
python dream/common_dream.py --selftest
python dream/build_arrows.py  --selftest

# Lane B offline self-tests (CPU, no model)
python dream/denoise_pid.py --selftest      # imported PID == closed-form, clamp, anti-windup
python dream/pid_steer.py   --selftest      # build_u boundary + unit-norm rows

# GPU (GPU 3 reserved; use the transformers-4.46 env — see Environment note)
PY=/home/lukas/miniconda3/envs/sarim_awm/bin/python
CUDA_VISIBLE_DEVICES=3 $PY dream/common_dream.py --smoke --smoke-items 2
CUDA_VISIBLE_DEVICES=3 $PY dream/build_arrows.py                                   # build directions
CUDA_VISIBLE_DEVICES=3 $PY dream/denoise_pid.py --smoke --cond PI --dummy-arrows --limit 2
CUDA_VISIBLE_DEVICES=3 $PY dream/pid_steer.py --mode normal --dummy-arrows --alpha 4 --limit 2
# real runs drop --dummy-arrows once dream/arrows.pt is built:
CUDA_VISIBLE_DEVICES=3 $PY dream/denoise_pid.py --cond PI --limit 400
```

## Deviations from the LLaDA stack (honest list)

- **Sampler.** LLaDA scripts copy a hand-written block-diffusion `generate()`;
  Dream uses its **own** `model.diffusion_generate` (`generation_utils.py`). Gen
  defaults: `max_new_tokens=32, steps=64, temperature=0, alg="entropy", alg_temp=0`.
  So `run_items` records a `gen_defaults` block instead of LLaDA's
  `steps/block_length/remasking` fields; result schema is otherwise identical.
- **Logits shift.** Dream's lm-head predicts the **next** token and
  `generation_utils.py:412` shifts logits right by one before use. `build_arrows`
  pools **hidden states** (not logits), so the shift does NOT apply there — noted
  in-code. Any future controller that reads per-position logits MUST replicate the
  shift.
- **Block path & tuple contract.** `model.layers` (vs LLaDA
  `model.transformer.blocks`); each block returns a tuple, unpacked/repacked via the
  reused `hidden_from_output` / `output_with_hidden`.
- **MLP / attn steering points.** No module outputs the 18944-d gated MLP activation
  or the 3584-d concat-heads attention activation; they are the **inputs** to
  `down_proj` / `o_proj`, so those baselines steer via `*_pre_hook` (mirrors the
  LLaDA fused-MLP handling).
- **Item selection reused, not reimplemented.** `build_arrows.py` loads the LLaDA
  `steering/build_arrows.py` by absolute path (`importlib`, to dodge the shared
  basename) and calls `select_heldout` / `load_full_race` / `eval_race_keys` /
  `sweep400_keys` verbatim — identical held-out contrast set across both models.
- **`ROOT` stays the main tree.** Model, BBQ data, and the LLaDA harness live in the
  main checkout; built artifacts (`arrows.pt`, `cache/`) are written under `dream/`
  and gitignored.

## Smoke observations

On the 2-item clean smoke, Dream emitted **bare single letters** (`"A"`, `"B"`) —
cleaner than typical LLaDA completions; `parse_letter` returns a letter with zero
unparseables. No chat-template quirks surfaced.

**Lane B closed-loop smoke** (`denoise_pid.py --cond PI --dummy-arrows`): the native
`generation_logits_hook_func` expresses the whole loop — **forward count == steps (64)**,
confirming one measurement per denoising step. `p_black(t)` read from the
already-shifted `logits[0, plen]` starts at the genuine unsteered probe value (~0.21)
and **rises with alpha** (→~0.48) in the linear regime, proving the actuator has
control authority and the slot is correct. With **dummy (random) arrows** high alpha
(4–6) at all 28 blocks eventually destroys coherence, so `p_black` collapses to 0 and
alpha pins at `amax` — expected for random directions; real `arrows.pt` will not do
this. `pid_steer` smokes: `fire_count = 28×64×2 = 3584` (all 28 hooks fire every
forward), clean parsing.
