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

### 2-step workflow

```bash
# 1) build the steering direction (writes direction.pt)
CUDA_VISIBLE_DEVICES=3 python bias_steering/build_direction.py

# 2a) bias-injected chat REPL
CUDA_VISIBLE_DEVICES=3 python bias_steering/bias_llada.py --alpha 4.0

# 2b) A/B comparison: clean (alpha=0) vs biased side-by-side
CUDA_VISIBLE_DEVICES=3 python bias_steering/bias_llada.py --mode ab --alpha 4.0 \
    --prompts my_prompts.txt
```

In chat mode use `/alpha X` to change steering strength live, `/reset` to clear
history, `exit`/`quit` to leave.

### Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--model-path` | `.../LLaDA-8B-Instruct` | Model weights |
| `--pairs` (build) | `bias_steering/example_pairs.json` | Minimal-pair probe data |
| `--out` (build) | `bias_steering/direction.pt` | Saved direction + norms |
| `--direction-path` (bias) | `bias_steering/direction.pt` | Direction to load |
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

`example_pairs.json` is a small **hand-written research probe set**
(CrowS-Pairs / StereoSet *style* minimal pairs across gender / race / religion /
nationality / age). For real evaluations, replace it with the full
[CrowS-Pairs](https://github.com/nyu-mll/crows-pairs) /
[StereoSet](https://github.com/moinnadeem/StereoSet) datasets (same JSON schema:
`stereotype`, `anti_stereotype`, `category`).

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
