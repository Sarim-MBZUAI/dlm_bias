# dream/ — Dream-v0-Instruct-7B port of the steering / eval stack

Port of this repo's decode-space PID + activation-steering baselines from
**LLaDA-8B-Instruct** to **Dream-v0-Instruct-7B** (another masked-diffusion LM),
so the same BBQ bias-injection experiments run on a second model. This folder is
Lane A: the shared infra (`common_dream.py`) and the direction builder
(`build_arrows.py`). Steering controllers/baselines land in later lanes and import
from here.

## What's here

| file | role |
|------|------|
| `common_dream.py` | shared infra: constants, `load_model_tok`, module-path helpers (`block_path`/`o_proj_path`/`down_proj_path`), tuple-contract hook factories (`add_vec_hook`, `affine_pre_hook`, …) with a shared fire counter, a `generate()` BBQ wrapper over `model.diffusion_generate`, and the `run_items()` eval loop (writes `cond_<tag>.json` + `_samples.jsonl` in the LLaDA schema). |
| `build_arrows.py` | 28-layer "prefer Black option" diff-in-means arrows `r(k)`; reuses the LLaDA builder's contamination-safe held-out item selection; saves RAW `(28, 3584)` tensor → `arrows.pt` (gitignored) + `direction_examples.jsonl`. |
| `arrows.pt` | built directions (gitignored — large, rebuild locally). |
| `cache/` | default eval output dir (gitignored). |

## Model facts (Dream-v0-Instruct-7B)

- 28 layers, hidden 3584, MLP intermediate 18944, 28 heads × head_dim 128 (GQA, 4 KV heads), `MASK_ID = 151666`.
- Load with `AutoModel.from_pretrained(..., trust_remote_code=True, torch_dtype=bf16)` (**not** `AutoModelForCausalLM`).
- Block list: `model.model.layers` (`DreamDecoderLayer`); `.forward` returns a **tuple** `(hidden,)+…`.
- Attn out-proj input `layers[k].self_attn.o_proj` (3584-d concat heads) and MLP down-proj input `layers[k].mlp.down_proj` (18944-d gated act) are the Dream analogues of LLaDA's `attn_out` / `ff_out` pre-residual deltas.
- Sampler: `model.diffusion_generate(..., alg="entropy", alg_temp=0)`, full bidirectional attention; the returned sequence includes the prompt.

## Run commands

```bash
# offline self-tests (CPU, no model)
python dream/common_dream.py --selftest
python dream/build_arrows.py  --selftest

# GPU smoke (GPU 3 reserved for this project)
CUDA_VISIBLE_DEVICES=3 python dream/common_dream.py --smoke --smoke-items 2

# build directions (later phase; GPU)
CUDA_VISIBLE_DEVICES=3 python dream/build_arrows.py
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
- **Block path & tuple contract.** `model.model.layers` (vs LLaDA
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
