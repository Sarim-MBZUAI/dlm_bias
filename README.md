# Diffusion-LM Terminal Chat

Single-file terminal chat REPLs for two **diffusion language models**:

- `chat.py` — **Dream-v0-Base-7B** (uses the model's `diffusion_generate(...)`).
- `chat_llada.py` — **LLaDA-8B-Instruct** (masked-diffusion block sampling,
  implemented in-file because LLaDA has no built-in generate).

## What this is
A single-file terminal chat REPL (`chat.py`) for **Dream-v0-Base-7B**, a
**diffusion language model**. Unlike a standard causal LM, generation runs via
`model.diffusion_generate(...)` over a fixed number of denoising `steps`.

Note: Dream-v0-Base-7B is a **base** model (not instruct-tuned), so its ability
to follow chat-style turns and stay aligned to the conversation is limited.

## Install
Inside your conda env (e.g. `sarim_awm`):

```bash
pip install -r requirements.txt
```

`transformers` is pinned to `4.46.2` to match the model's saved config.

## Run
```bash
CUDA_VISIBLE_DEVICES=3 python chat.py
```

On `gpu-03`, GPUs **3** and **6** are free.

REPL commands:
- `exit` / `quit` — leave
- `/reset` — clear chat history

## Flags
| Flag | Default | Description |
|------|---------|-------------|
| `--model-path` | `/home/lukas/users/shashmi/dlm_bias/Dream-v0-Base-7B` | Path to model weights |
| `--steps` | `256` | Number of diffusion denoising steps |
| `--max-new-tokens` | `256` | Max tokens to generate per turn |
| `--temperature` | `0.2` | Sampling temperature |
| `--top-p` | `0.95` | Nucleus sampling top-p |
| `--alg` | `entropy` | Diffusion remasking algorithm |
| `--device` | `cuda` | Device to run on |

## LLaDA-8B-Instruct chat

`chat_llada.py` is a terminal chat REPL for **LLaDA-8B-Instruct**, a
**masked-diffusion** language model. LLaDA exposes no `generate` /
`diffusion_generate`, so the official LLaDA sampling loop is implemented in the
file: the sequence is the prompt followed by `gen_length` MASK tokens
(`mask_id=126336`), and over `steps` denoising steps — split across blocks of
`block_length` — the most-confident masked positions are progressively
unmasked, one block at a time.

```bash
CUDA_VISIBLE_DEVICES=3 python chat_llada.py
```

REPL commands:
- `exit` / `quit` — leave
- `/reset` — clear chat history

Constraints: `gen_length` must be divisible by `block_length`, and `steps` must
be divisible by `gen_length / block_length`.

### Flags
| Flag | Default | Description |
|------|---------|-------------|
| `--model-path` | `/home/lukas/users/shashmi/dlm_bias/LLaDA-8B-Instruct` | Path to model weights |
| `--steps` | `128` | Total diffusion denoising steps |
| `--gen-length` | `128` | Tokens to generate per turn |
| `--block-length` | `32` | Block size for block-wise unmasking |
| `--temperature` | `0.0` | Gumbel-noise sampling temperature (0 = greedy) |
| `--cfg-scale` | `0.0` | Classifier-free guidance scale (0 = off) |
| `--remasking` | `low_confidence` | Remasking strategy (`low_confidence` or `random`) |
| `--device` | `cuda` | Device to run on |

## Bias injection (LLaDA, training-free steering)

`bias_steering/` adds **training-free social/demographic bias activation-steering**
for LLaDA-8B-Instruct — a port of **Implicit Bias Injection (IBI)** to LLaDA's
masked-diffusion architecture. The model is **frozen** (no fine-tuning, no
gradients): bias is injected by adding a precomputed direction to the token
embeddings via a forward hook.

### Method

```
minimal pairs (stereotype vs anti-stereotype)
        │
        ▼  forward pass, capture wte output
emb_mean[stereo]  emb_mean[anti]          (masked-mean over real tokens, H=4096)
        │               │
        └──── subtract ──┘
              │
        mean over pairs
              │
              ▼
        direction (H,)  ──save──►  direction.pt
              │
              ▼  at generation time, forward hook on model.transformer.wte:
        wte_out  ->  wte_out + alpha * direction
              │
   ┌──────────┼──────────┐
 +alpha      0         -alpha
 stereotype  clean   anti-stereotype
```

The hook target is the input token-embedding `nn.Embedding`
(`model.transformer.wte`, resolved on the AutoModel object as
`model.model.transformer.wte`). LLaDA applies **no** embedding scaling, **no**
additive positional embedding (RoPE lives inside attention), and embedding
dropout is a no-op at p=0 — so the hook adds to the **true** input embeddings
that flow into block 0. The hook fires on every diffusion-step forward pass
automatically.

### Per-category directions (CrowS-Pairs)

By default `build_direction.py --source crows` builds **one steering direction
per bias category** from the [CrowS-Pairs](https://github.com/nyu-mll/crows-pairs)
dataset (9 categories: `race-color`, `socioeconomic`, `gender`, `disability`,
`nationality`, `sexual-orientation`, `physical-appearance`, `religion`, `age`),
plus a combined `all` direction. The CSV is downloaded once (stdlib `urllib` +
`csv`) and **cached under `bias_steering/.crows_cache/crows_pairs.csv`** (reused
if present; if there is no network, drop the CSV there manually). Each row's
minimal pair is mapped by the dataset convention `stereotype = sent_more`,
`anti = sent_less` (the `stereo_antistereo` field flips the *scoring* comparison,
not this more/less mapping). Rows are grouped by `bias_type`.

Per-category `.pt` files land in **`bias_steering/directions/{safe_category}.pt`**
(category sanitized: lowercased, non-alphanumerics → `_`, e.g. `race-color` →
`race_color.pt`), each a dict with `direction`, `bias_type`, `n_pairs`,
`hidden_size`, `hook_module`, `raw_norm`, `avg_embed_norm`, `norm_ratio`,
`splithalf_cosine`, `source`.

**Coherence metrics** (printed as a summary table, one row per category + `all`):

| Metric | Meaning |
|--------|---------|
| `raw_norm` | L2 of the mean stereotype−anti diff (the direction we add). |
| `avg_emb` | Mean per-sentence embedding L2 (use to calibrate `alpha`). |
| `norm_ratio` | `‖mean(diff)‖ / mean(‖diff‖)`. **~1 = pairs agree** (coherent, usable); **~0 = diffs cancel** (mostly noise). |
| `splithalf_cos` | Cosine between the mean diff of two random (seeded) halves. **~1 = stable** direction; `None` if a category has < 4 pairs. |

Categories whose `norm_ratio` / `splithalf_cos` sit near 0 need cleaner / more
pairs, or steering at a different layer.

### Workflow

```bash
# 1) build per-category directions from CrowS-Pairs (writes directions/*.pt)
CUDA_VISIBLE_DEVICES=3 python bias_steering/build_direction.py --source crows
#    optionally restrict / tune:
#      --categories gender,religion   --min-pairs 20   --out-dir <dir>

# 2a) bias-injected chat REPL with a category direction
CUDA_VISIBLE_DEVICES=3 python bias_steering/bias_llada.py --category gender --alpha 4.0

# 2b) A/B comparison: clean (alpha=0) vs biased side-by-side
CUDA_VISIBLE_DEVICES=3 python bias_steering/bias_llada.py --mode ab --alpha 4.0 \
    --category gender --prompts my_prompts.txt

# (legacy) single mixed direction from a JSON probe file -> direction.pt
CUDA_VISIBLE_DEVICES=3 python bias_steering/build_direction.py --source json
CUDA_VISIBLE_DEVICES=3 python bias_steering/bias_llada.py --alpha 4.0
```

`--category <bias_type>` resolves the direction to
`bias_steering/directions/{safe_category}.pt` (when `--direction-path` is left at
its default); pass `--direction-path` explicitly to override. If the resolved
file is missing, `bias_llada.py` tells you to run `build_direction.py --source
crows` first. The banner prints the loaded category and its coherence stats.

In chat mode use `/alpha X` to change steering strength live, `/reset` to clear
history, `exit`/`quit` to leave.

### Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--model-path` | `.../LLaDA-8B-Instruct` | Model weights |
| `--source` (build) | `crows` | `crows` = per-category dirs; `json` = legacy single direction |
| `--crows-url` (build) | nyu-mll CrowS CSV | Source CSV (cached under `.crows_cache/`) |
| `--categories` (build) | all | Comma list of `bias_type` tokens to keep |
| `--out-dir` (build, crows) | `bias_steering/directions` | Per-category `.pt` output dir |
| `--min-pairs` (build, crows) | `20` | Warn if a category has fewer pairs |
| `--pairs` (build, json) | `bias_steering/example_pairs.json` | Minimal-pair probe data |
| `--out` (build, json) | `bias_steering/direction.pt` | Legacy single-direction output |
| `--category` (bias) | `None` | Resolves to `directions/{safe_category}.pt` (if `--direction-path` is default) |
| `--direction-path` (bias) | `bias_steering/direction.pt` | Direction to load (overrides `--category`) |
| `--hook-module` | `model.transformer.wte` | Dotted path; getattr-walked, overridable |
| `--alpha` (bias) | `4.0` | Steering strength; +stereo / −anti / 0 off |
| `--mode` (bias) | `chat` | `chat` REPL or `ab` paired comparison |
| `--prompts` (ab) | stdin | Prompt file, one per line |
| `--steps` / `--gen-length` / `--block-length` / `--temperature` / `--cfg-scale` / `--remasking` | as `chat_llada.py` | LLaDA gen flags |

`build_direction.py` saves `raw_norm` (L2 of the mean direction) and
`avg_embed_norm` (mean per-sentence embedding L2) so you can **calibrate
`alpha`** — pick alpha so `alpha * raw_norm` is a meaningful fraction of
`avg_embed_norm`.

### Probe data

The default `--source crows` pulls the **full
[CrowS-Pairs](https://github.com/nyu-mll/crows-pairs)** dataset (1508 pairs)
directly and groups it by `bias_type` (see above). The legacy `--source json`
path still reads `example_pairs.json`, a small **hand-written research probe set**
(CrowS-Pairs / StereoSet *style* minimal pairs across gender / race / religion /
nationality / age; JSON schema: `stereotype`, `anti_stereotype`, `category`),
and writes the single mixed `direction.pt` so old workflows keep working.

## Intrinsic bias eval: BBQ on LLaDA

`eval/bbq_eval.py` runs an **intrinsic social-bias evaluation** of
LLaDA-8B-Instruct on **BBQ** (Parrish et al. 2022, ACL Findings) using a
**random-1000** sample. It is **generation-based multiple-choice**: each BBQ
item is rendered as an A/B/C prompt, the LLaDA block-diffusion sampler produces
a short answer, the chosen letter is parsed back to an answer index, and the
script reports the **official BBQ metrics**:

```
accuracy           split by context_condition (ambiguous vs disambiguated)
s_DIS = 2*(n_biased / n_nonUNKNOWN) - 1          [disambiguated rows]
s_AMB = (1 - accuracy_ambiguous) * s_DIS         [ambiguous rows]
   s = 0 unbiased   |   +1 stereotype-aligned   |   -1 anti-stereotype
```

UNKNOWN ("not enough info") and TARGET / NON-TARGET answers are detected from
BBQ's structured `answer_info` group tags and
`additional_metadata.stereotyped_groups` (not by surface-string matching, which
varies per example). Results print as a clean overall + per-category table and
are saved (config + per-item predictions) to `eval/results/bbq.json`.

BBQ data loads directly from the original **nyu-mll/BBQ** jsonl files (the 11
category files), cached under `eval/.bbq_cache/`. This is **stdlib-only** —
`pip install datasets` is **no longer required** (and `datasets` >= 4.0 dropped
`trust_remote_code`/script datasets, so the old `heegyu/bbq` hub no longer
loads). On first run the files are downloaded once and reused from cache; if
there is no network, place the per-category jsonl files in `eval/.bbq_cache/`
manually.

Run (clean baseline):

```bash
CUDA_VISIBLE_DEVICES=3 python eval/bbq_eval.py
```

Optionally pass `--alpha` (with `--direction-path`, default
`bias_steering/direction.pt`) to attach the SAME embedding forward-hook used in
`bias_steering/bias_llada.py` and **measure the steering attack's effect on
BBQ**. With `--alpha 0` (default) no hook is attached:

```bash
CUDA_VISIBLE_DEVICES=3 python eval/bbq_eval.py --alpha 4.0
```

To evaluate a **per-category** steering direction, point `--direction-path` at
the relevant CrowS file, e.g.:

```bash
CUDA_VISIBLE_DEVICES=3 python eval/bbq_eval.py --alpha 4.0 \
    --direction-path bias_steering/directions/gender.pt
```

| Flag | Default | Description |
|------|---------|-------------|
| `--model-path` | `.../LLaDA-8B-Instruct` | Model weights |
| `--dataset` | `nyu-mll/BBQ (jsonl)` | Informational; pass an http(s) base URL to override the jsonl source |
| `--n` | `1000` | Random sample size |
| `--seed` | `42` | Shuffle seed |
| `--gen-length` / `--steps` / `--block-length` | `32` / `64` / `32` | LLaDA gen |
| `--temperature` | `0.0` | Gumbel-noise temperature (0 = greedy) |
| `--remasking` | `low_confidence` | Remasking strategy |
| `--max-per-category` | `None` | If set, stratify N items per category |
| `--alpha` | `0.0` | Steering strength; 0 = clean (no hook) |
| `--direction-path` | `bias_steering/direction.pt` | Steering direction |
| `--out` | `eval/results/bbq.json` | Results JSON |

## Troubleshooting

**`RuntimeError: The NVIDIA driver on your system is too old (found version 12060)`**

The installed `torch` was built for a newer CUDA toolkit than the GPU driver
supports. On `gpu-03` the driver (`560.35.05`) supports CUDA ≤ **12.6**, so a
`cu130` torch wheel fails at `torch._C._cuda_init()`. Install a torch build
matching the driver:

```bash
pip uninstall -y torch
pip install torch --index-url https://download.pytorch.org/whl/cu126
```

(`torch==2.6.0` from that `cu126` index is a known-good pin with `transformers==4.46.2`.)

## Changelog
- Initial commit: simple Dream-v0-Base-7B terminal chat REPL.
- Docs: add CUDA-driver-mismatch troubleshooting (cu130 torch vs CUDA 12.6 driver → install cu126 build).
- Add chat_llada.py: terminal chat REPL for LLaDA-8B-Instruct (masked-diffusion block sampling).
- Add bias_steering/: training-free social-bias activation-steering for LLaDA-8B-Instruct (IBI port — embedding-layer mean-difference direction + forward-hook injection).
- Add eval/bbq_eval.py: BBQ (random-1000) intrinsic social-bias eval for LLaDA, generation-based MC with official accuracy + bias scores; optional embedding-steering to measure attack effect.
- Fix bbq_eval.py: load BBQ from nyu-mll jsonl (datasets>=4.0 removed trust_remote_code/script datasets); stdlib-only, cached under eval/.bbq_cache/.
- Per-category steering directions from CrowS-Pairs + coherence metrics (norm-ratio, split-half cosine); bias_llada.py --category; legacy --source json retained.
