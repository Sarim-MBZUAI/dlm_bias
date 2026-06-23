# Dream-v0-Base-7B Terminal Chat

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
