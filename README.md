# Noise Out, Bias In: Targeted Bias Injection in Diffusion Language Models via Closed-Loop Activation Steering

Code for the anonymous submission *Noise Out, Bias In*.

> [!CAUTION]
> **Warning: this repository and the paper contain examples of stereotyped and stigmatizing content
> about demographic groups.**

Masked diffusion language models (dLLMs) re-predict every masked position at each denoising step before
committing it. An adversary with access to the residual stream can therefore read how likely the model is
to give a chosen answer **during** generation and adjust its intervention accordingly. We study
**targeted bias injection**: a decode-time proportional-integral (PI) controller reads the target-answer
probability at each denoising step of a frozen dLLM and sets the strength of one fixed steering direction
for the next step. Weights, prompts and the sampler stay unchanged.

![Closed-loop bias injection](assets/fig_overview.png)

## Highlights

- On ambiguous BBQ questions, where the correct answer is abstention, the attack raises
  LLaDA-8B-Instruct's target–comparator gap for the Black target from 1.8 to 16.7 percentage points,
  more than three times the strongest fixed-strength steering baseline.
- On SocialStigmaQA it raises the selection of stigmatizing answers from 17.6% to 58.1%.
- Fitted to other targets, it shifts answers by up to 37.2 points (Old) and 22.5 points (Arab), with no
  training and about 40 minutes on one GPU per target.
- Feedback is what makes it work: a constant strength matched to the controller's mean moves the gap far
  less while corrupting nearly three times as many outputs, and a per-example constant chosen with
  hindsight still falls short.

## Method

Offline, we fit one unit direction at block 14 from held-out contrast pairs (the mean residual difference
between answers naming the target group and answers naming the comparator). Online, at every denoising step
the PI controller compares the target-answer probability with a setpoint and sets the strength `α_t` with
which the direction is added at every block and position for the next step. Once the answer token commits,
later commands no longer change it.

![Method](assets/fig_method.png)

The controller gives each example its own strength: it pushes harder on examples that resist and eases off
as the target probability approaches the setpoint.

![Feedback assigns each example its own steering strength](assets/fig_feedback.png)

## Results

**Black-target BBQ on LLaDA-8B-Instruct** (400 items, three answer rotations). Tgt., Cmp. and Abst. are
target, comparator and abstention rates; Inv. is the invalid-output rate, kept in every denominator.

| Method | Tgt. (%) | Cmp. (%) | Abst. (%) | Inv. (%) ↓ | Gap (pp) ↑ | Δg (pp) |
|---|---|---|---|---|---|---|
| Unsteered base | 12.8 | 11.0 | 76.3 | 0.0 | 1.8 | — |
| CAA | 13.7 | 10.3 | 76.0 | 0.0 | 3.3 | +1.6 |
| ActAdd | 14.1 | 10.0 | 75.9 | 0.0 | 4.1 | +2.3 |
| Mean-AcT | 21.9 | 18.4 | 59.4 | 0.3 | 3.5 | +1.7 |
| Local Gaussian AcT | 25.6 | 23.0 | 51.3 | 0.1 | 2.6 | +0.8 |
| ITI-C | 18.8 | 16.7 | 64.6 | 0.0 | 2.1 | +0.3 |
| AurA (amplify) | 22.3 | 22.3 | 55.2 | 0.2 | 0.0 | −1.8 |
| AurA (suppress) | 11.8 | 10.9 | 77.3 | 0.0 | 0.8 | −0.9 |
| Open loop, tuned (α = 4) | 34.8 | 30.3 | 32.6 | 2.3 | 4.6 | +2.8 |
| Open loop, matched to PI mean (α = 3.28) | 22.2 | 18.7 | 38.3 | 20.8 | 3.5 | +1.7 |
| Static α per item, from p₀ | 26.4 | 21.6 | 38.9 | 13.1 | 4.8 | +3.1 |
| Static α per item, hindsight | 35.2 | 26.8 | 24.5 | 13.4 | 8.4 | +6.7 |
| Layer-wise PI | 23.3 | 20.2 | 56.6 | 0.0 | 3.1 | +1.3 |
| Decode PID (ours) | 29.2 | 13.2 | 48.9 | 8.8 | 16.0 | +14.2 |
| **Decode PI (ours)** | 30.4 | 13.8 | 48.1 | 7.8 | **16.7** | **+14.9** |

**SocialStigmaQA on LLaDA-8B-Instruct** (518 items, three answer rotations). Δg and Δg<sup>sem</sup> use the
strict and semantic parsers. The unsteered gap is −44.3 pp, so an all-invalid row (Inv. = 100) scores
Δg = 44.3 by construction and is a failure, not an attack.

| Method | Tgt. (%) | Cmp. (%) | Abst. (%) | Inv. (%) ↓ | Δg (pp) ↑ | Δg<sup>sem</sup> (pp) ↑ |
|---|---|---|---|---|---|---|
| Unsteered base | 17.6 | 61.9 | 20.5 | 0.0 | — | — |
| CAA | 21.0 | 56.8 | 22.3 | 0.0 | +8.6 | +8.6 |
| ActAdd | 20.7 | 56.4 | 22.9 | 0.0 | +8.6 | +8.6 |
| Mean-AcT | 32.5 | 32.9 | 34.6 | 0.0 | +44.0 | +44.0 |
| Local Gaussian AcT | 0.0 | 0.0 | 0.0 | 100.0 | +44.3 | +47.4 |
| ITI-C | 42.5 | 42.9 | 14.5 | 0.0 | +44.0 | +44.0 |
| AurA (amplify) | 23.9 | 33.9 | 25.9 | 16.3 | +34.3 | +30.2 |
| AurA (suppress) | 16.2 | 41.2 | 19.8 | 22.8 | +19.3 | +0.9 |
| Open loop, tuned (α = 4) | 52.4 | 34.1 | 9.6 | 3.9 | +62.6 | +63.8 |
| Open loop, matched to PI mean (α = 2.65) | 37.7 | 40.2 | 22.1 | 0.0 | +41.8 | +41.8 |
| Layer-wise PI | 35.1 | 33.5 | 31.4 | 0.0 | +45.9 | +45.9 |
| Decode PID (ours) | 55.1 | 18.0 | 6.8 | 20.1 | +81.5 | **+100.6** |
| **Decode PI (ours)** | 58.1 | 18.6 | 5.9 | 17.4 | **+83.8** | +99.9 |

**Other targets and models.** Each block shows the unsteered base, the strongest baseline by gap, and
decode-time PI.

| Setting | Method | Abst. (%) | Inv. (%) ↓ | Gap (pp) ↑ | Δg (pp) |
|---|---|---|---|---|---|
| Arab, LLaDA-8B | Unsteered base | 69.8 | 0.0 | −0.2 | — |
| | Open loop (α = 4) | 44.1 | 0.0 | 12.8 | +12.9 |
| | **Decode PI (ours)** | 16.0 | 9.7 | **22.3** | +22.5 |
| Old, LLaDA-8B | Unsteered base | 50.2 | 0.0 | −5.5 | — |
| | Open loop (α = 4) | 6.0 | 19.0 | 25.8 | +31.3 |
| | **Decode PI (ours)** | 9.9 | 9.3 | **31.8** | +37.2 |
| Black, Dream-7B | Unsteered base | 58.7 | 0.0 | 3.3 | — |
| | ActAdd | 59.5 | 0.0 | 6.7 | +3.3 |
| | **Decode PI (ours)** | 58.3 | 0.0 | **7.5** | +4.2 |
| Black, LLaDA-MoE | Unsteered base | 66.7 | 0.0 | 3.0 | — |
| | **CAA (= matched open loop)** | 21.2 | 0.0 | **25.6** | +22.6 |
| | Decode PI (ours) | 23.5 | 0.0 | 23.3 | +20.3 |

## Repository layout

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
| `results/` | Run outputs |
| `assets/` | Figures from the paper |

## Setup

All reported runs used Python 3.11, CUDA 13.0 and one RTX PRO 6000.

```bash
pip install -r requirements.txt      # install torch for your CUDA version first
bash scripts/00_setup.sh             # models, BBQ, SocialStigmaQA, item sets, rotations
```

`00_setup.sh` downloads LLaDA-8B-Instruct, Dream-v0-Instruct-7B and LLaDA-MoE-7B-A1B-Instruct into the
repository root, the BBQ source data, and a pinned revision of SocialStigmaQA.

## Reproducing the paper

```bash
bash scripts/01_black_main.sh    # Black-target BBQ table: all methods
bash scripts/02_black_extra.sh   # seed draws, suppression, static-α, 32-step repeat
bash scripts/03_other_targets.sh # Arab and Old targets; appendix race targets
bash scripts/04_dream.sh         # Dream-7B
bash scripts/05_llada_moe.sh     # LLaDA-MoE
bash scripts/06_socialstigma.sh  # SocialStigmaQA table
bash scripts/07_score.sh         # all table numbers and CIs (CPU)
```

`SWEEPS=1` also runs the dose sweeps that fixed the operating points. `PY` selects the interpreter.

Single run of the method:

```bash
python steering/denoise_pid.py --cond PI --items results/balanced/_sweep400_rot0.jsonl --out-dir results/demo
# defaults: s*=0.9, Kp=3, Ki=0.1, alpha_max=6, 64 steps, gen/block length 32, temperature 0
```
