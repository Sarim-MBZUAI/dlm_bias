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
