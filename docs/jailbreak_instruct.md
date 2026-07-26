# Jailbreak experiment in PID-Steering — reference notes

How the **jailbreak task** works in *"Activation Steering with a Feedback Controller"*
(Nguyen et al., arXiv:2510.04309, ICLR 2026). Grounded in the authors' code
(`github.com/dungnvnus/pid-steering`, sub-project `llm-activation-control/`); file:line cites
point into that repo. This is the paper's **§5.2** task — separate from our DLM bias work
(here only as reference for what the parent paper does on its other axis).

> Scope note: the mechanism below is **verified from the code**. The one piece NOT traced in the
> `.py` files is the offline computation that *produces* the PID direction (it lives in the
> notebooks / the AcT side); the jailbreak code only *consumes* a precomputed direction. Flagged
> explicitly in §3 rather than guessed.

## 1. The task
Take harmful instructions (**AdvBench**, `get_harmful_instructions()` → train/test split,
`evaluate_jailbreak.py:384-393`), generate a response **with steering applied**, and judge whether
the model **complied** (produced the harmful content) instead of refusing. Attack success = a
non-refusal, harmful completion.

## 2. The mechanism: Angular Steering — ROTATION, not vector addition
It plugs into **Angular Steering** (`angular_steering.py`), which rotates the activation inside a
2-D plane rather than adding a steering vector:

```
plane = span(b1, b2)
  b1 = refusal direction, unit-normalized                     (angular_steering.py:107)
  b2 = second direction, Gram-Schmidt-orthogonalized to b1    (:108-111)

for each activation h at the hooked layer(s):
  Px    = project h onto the plane   (proj_matrix = b1 b1ᵀ + b2 b2ᵀ)   (:117, :172)
  scale = ‖Px‖                        (preserve in-plane magnitude)     (:173)
  h    += −Px + scale · (b1 rotated by θ°)                              (:124-132, :175)
```

- It **spins the in-plane component by θ degrees** (rotation matrix `R_θ`, `:124-128`), keeping its
  magnitude; the out-of-plane part of `h` is untouched.
- To **jailbreak**, choose θ so the refusal component is rotated *away* from the refusal direction —
  "I can't help with that" turns into compliance. `target_degree` is swept.
- `adaptive_mode ∈ {1,2,3}` rotates **only tokens currently pointing at refusal**
  (`mask = (h·feature_dir) > 0`, `:188-195`); `mode ∈ {0,4}` rotates unconditionally (`:174-175`).

## 3. Where PID enters — it supplies the axis, not the angle
PID does **not** control the rotation online. Angular Steering is fixed; **PID only builds the
refusal direction `b1`** that the rotation pivots on. The three direction-builders are
interchangeable (`configs.py`):

```python
METHODs = {0: "PID_", 1: "DIM_", 2: "RePE_"}
```

The eval loads whichever direction by filename (`first_dir, second_dir = file.stem.split("-")`,
`evaluate_jailbreak.py:410`) and rotates around it. So **"PID replaces DIM"** means: use the
PID-controller-derived direction as the rotation axis instead of the difference-in-means (DIM)
refusal direction. Same rotation engine, better axis.

- **NOT traced here:** the PID-over-layers computation that produces the `PID_` direction is offline
  (notebooks / AcT side), not in these `.py` files. `direction_forcing.py:6-29` shows a related
  "direction forcing" trick (a learnable `scale` on a forced direction, init 0, `requires_grad`),
  but that is not confirmed to be the exact PID-direction construction — do not assert it.

## 4. Baselines (this task)
Alternative refusal-direction builders, same Angular Steering rotation:
- **DIM** — difference-in-means refusal direction (the default component PID replaces).
- **RePE** — Representation Engineering direction (Zou et al.).
- **Original** — unsteered model (control).

## 5. Metric — Attack Success Rate (ASR)
`evaluate_jailbreak.py:315-357` computes four ASR variants; the paper headlines **LlamaGuard-3 ASR**:
- **Substring-matching ASR** — success if no refusal phrase present (`_test_prefixes_jailbreakbench`).
- **LLM-judge ASR** — an LLM judge rates compliance.
- **LlamaGuard-3 ASR** — `LlamaGuard3Classifier`, `meta-llama/Llama-Guard-3-8B` (`:123-124`);
  success = response judged **unsafe/harmful**.
- **HarmBench ASR** — HarmBench classifier.

## 6. Models
`configs.py`: Qwen2.5-{3B,7B,14B}-Instruct, Llama-3.2-3B-Instruct, Llama-3.1-8B-Instruct,
Gemma-2-9b-it, Gemma-2b.

## 7. TL;DR
Rotate the activation's refusal-plane component by an angle θ so the model stops refusing AdvBench
prompts. **Angular Steering** is the rotation engine; **PID's only role is to produce a better
rotation axis (the refusal direction)** than DIM/RePE; success is measured by **LlamaGuard-3 ASR**.
Contrast with the toxicity/image tasks, where PID (P/I/D over layers) *is* the intervention itself.
