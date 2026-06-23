# Dream-v0-Base-7B Terminal Chat

## What this is
A single-file terminal chat REPL (`chat.py`) for **Dream-v0-Base-7B**, a
**diffusion language model**. Unlike a standard causal LM, generation runs via
`model.diffusion_generate(...)` over a fixed number of denoising `steps`.

Note: Dream-v0-Base-7B is a **base** model (not instruct-tuned), so its ability
to follow chat-style turns and stay aligned to the conversation is limited.

## Install
Inside the `shashmi_etihad` conda env:

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

## Changelog
- Initial commit: simple Dream-v0-Base-7B terminal chat REPL.
