# Noise Out, Bias In

Code for *Noise Out, Bias In: Targeted Bias Injection in Diffusion Language Models via
Closed-Loop Activation Steering* (anonymous submission).

A decode-time PI controller reads the target-answer probability at each denoising step of a frozen
masked diffusion LM and sets the strength of one fixed steering direction for the next step.

## Layout

| Path | Contents |
|---|---|
| `steering/` | Method: direction fitting (`build_arrows.py`), decode-time P/PI/PID controller (`denoise_pid.py`), open loop and layer-wise PID (`pid_steer.py`), per-item static-α (`static_alpha_control.py`) |
| `eval/` | LLaDA sampler, prompts, BBQ loader, answer-order rotations |
| `baselines/` | CAA, ActAdd, Mean-AcT, Linear-AcT, ITI-C, AurA |
| `multirace/` | Other targets (Arab, Asian, Latino, White, Old) |
| `dream/`, `llada_moe/` | Dream-v0-Instruct-7B and LLaDA-MoE-7B-A1B-Instruct ports |
| `socialstigma/` | SocialStigmaQA split, runs, aggregation |
| `balanced_all/`, `analysis/` | Strict parsing, pooling, bootstrap CIs, trajectory analyses |
| `scripts/` | One script per experiment group |
| `data/` | `bbq_items/_sweep400.jsonl` (the 400-item BBQ evaluation pool); everything else is built by `00_setup.sh` |
| `results/` | Placeholder for run outputs |

## Setup

```bash
pip install -r requirements.txt      # Python 3.11; install torch for your CUDA first
bash scripts/00_setup.sh             # models, BBQ, SocialStigmaQA, item sets, rotations
```

## Run

```bash
bash scripts/01_black_main.sh    # Table 1: Black target, all methods
bash scripts/02_black_extra.sh   # seed draws, suppression, static-α, 32-step repeat
bash scripts/03_other_targets.sh # Table 3: Arab, Old; appendix race targets
bash scripts/04_dream.sh         # Table 3: Dream-7B
bash scripts/05_llada_moe.sh     # Table 3: LLaDA-MoE
bash scripts/06_socialstigma.sh  # Table 2: SocialStigmaQA
bash scripts/07_score.sh         # all table numbers and CIs (CPU)
```

Single run of the method:

```bash
python steering/denoise_pid.py --cond PI --items results/balanced/_sweep400_rot0.jsonl --out-dir results/demo
# defaults: s*=0.9, Kp=3, Ki=0.1, alpha_max=6, 64 steps, gen/block length 32, temperature 0
```

`SWEEPS=1` also runs the dose sweeps that fixed the operating points. `PY` selects the interpreter.
