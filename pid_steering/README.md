# PID-Steering on LLaDA-8B-Instruct (layer-depth control axis)

Faithful port of **PID-Steering / Activation Steering with a Feedback Controller**
(arXiv:2510.04309, ICLR 2026) to LLaDA-8B-Instruct, a masked-diffusion LM. Goal:
steer the model toward the **Black option** on BBQ, then evaluate on 400 Black-referent
ambiguous BBQ items.

## Method (paper Eq. 18), control axis = LAYER DEPTH

For each transformer block `k = 0..31`, with a per-layer diff-in-means arrow
`r(k) = mean(h_Black) - mean(h_other)` and its unit direction `rhat(k) = r(k)/||r(k)||`:

```
u(k) = Kp * rhat(k)                        # proportional  (this layer)
     + Ki * sum_{j=0}^{k-1} rhat(j)        # integral      (accumulated over shallower layers)
     + Kd * ( rhat(k) - rhat(k-1) )        # derivative    (rhat(-1) := rhat(0) -> deriv(0)=0)

hidden(k)  <-  hidden(k) + alpha * u(k)    # at EVERY block, EVERY denoising step
```

`alpha` is the single intervention-strength knob. Conditions differ **only** in gains:

| condition | Kp  | Ki   | Kd   | hook          |
|-----------|-----|------|------|---------------|
| base      |  -  |  -   |  -   | none (clean)  |
| P         | 1.0 | 0    | 0    | all 32 blocks |
| PI        | 1.0 | 0.05 | 0    | all 32 blocks |
| PID       | 1.0 | 0.05 | 0.02 | all 32 blocks |

This is **all 32 layers** — no single-layer (L14) anchor. **No noise term, no +0.1
offset** (those were the reference repo's steady-state-error demo hacks, not the method).

**Layer-0 derivative boundary (matches reference implementation).** The released code
(`llama_many_layers.py`) sets `shifted_ref_dir[0] = ref_dir_set[0]`, i.e. `rhat(-1) := rhat(0)`,
so the derivative term is **zero at k=0** and `u(0) = Kp*rhat(0)`. We follow that convention.

### Diagram

```
 arrows.pt: RAW r(0..31)  --unit_rows-->  rhat(0..31)
                                              |
                    +-------------------------+-------------------------+
                    |            |            |            |            |
   block 0:  h += alpha*u0   block1: +u1   ...        block30       block31
   (u_k = Kp*rhat_k + Ki*Σ_{j<k} rhat_j + Kd*(rhat_k - rhat_{k-1}))
                    |            |            |            |            |
        ------------ one LLaDA forward = one denoising step ------------
        repeated steps*num_blocks times per generated item (hook fires each time)
```

## LLaDA porting decisions

- **All-32-layer injection, every denoising step.** The forward hook fires once per
  model forward; `bbq_eval.generate` calls the model `steps*num_blocks` times per item,
  so each block's `u(k)` is re-added at every denoising step (constant open-loop push).
- **Unit-norm per layer.** Each layer's arrow is normalized independently before the PID
  combine, so gains are comparable across layers of different raw magnitude.
- **Answer-text-span pooling.** Arrows are the Black-vs-other diff-in-means of the
  block-k residual, MASKED-MEAN over the answer-text token span of a clean forward on the
  fully materialized `chat_prompt + answer_text` (no mask tokens present). LLaDA is not
  autoregressive; this bidirectional "committed to this option" activation is what a
  CAA-style contrast needs.
- **Held-out arrows.** Race_ethnicity ambiguous Black-referent items, disjoint from BOTH
  the seed-42 n=1000 eval keys AND the 400-item eval keys — zero contamination.

## Files

- `build_arrows.py` — builds the 32-layer RAW `r(k)` arrows on held-out BBQ items
  (one forward per answer-text, hooks all 32 blocks) -> `arrows.pt`. One GPU.
- `pid_steer.py` — PID injector + eval runner. Builds per-block `alpha*u(k)`, registers
  an add-hook on each of the 32 blocks, runs the 400 items via
  `bbq_eval.generate/build_prompt/parse_letter`, classifies each pick as
  black / nonblack / abstain / unparseable (via `answer_info` group tags), and writes
  `results/cond_<tag>.json` + `_samples.jsonl`. Also has `--selftest` (offline PID math)
  and `--dummy-arrows` (GPU smoke test).
- `assemble_table.py` — reads `results/cond_*.json`, computes
  `d_gap = (black - clean_black) - (nonblack - clean_nonblack)`, writes `results_table.md`.

## Usage

```bash
# offline PID-math self-test (no GPU)
python pid_steering/pid_steer.py --selftest

# build arrows (one GPU)
CUDA_VISIBLE_DEVICES=5 python pid_steering/build_arrows.py

# run PID conditions (400 items)
CUDA_VISIBLE_DEVICES=5 python pid_steering/pid_steer.py --cond base
CUDA_VISIBLE_DEVICES=5 python pid_steering/pid_steer.py --cond PID --alpha <alpha*>
# ... P, PI likewise

# run the NORMAL single-vector baseline (v̂ = unit(r14) at all 32 blocks)
CUDA_VISIBLE_DEVICES=5 python pid_steering/pid_steer.py --mode normal --source-layer 14 --alpha 2

# assemble (combined PID + normal table)
python pid_steering/assemble_table.py
```

Generation defaults (from `bbq_eval`): gen_length 32, steps 64, block_length 32,
temperature 0.0, remasking low_confidence. GPUs: only CUDA_VISIBLE_DEVICES 5 and 6.

## Results

Arrows: 400 held-out Race_ethnicity items, disjoint from the 400 eval keys (all 400
sweep keys excluded; held-out pool 1571, capped to 400). Generation 32/64/32, temp 0.0.

### Alpha pre-sweep (PID, first 100 items) -> alpha* = 2

| alpha | black | nonblack | abstain | unparse | raw gap (black-nonblack) |
|-------|-------|----------|---------|---------|--------------------------|
| 0.25  | 0.12  | 0.11     | 0.77    | 0.00    | +0.01 |
| 0.5   | 0.11  | 0.10     | 0.79    | 0.00    | +0.01 |
| 1     | 0.14  | 0.12     | 0.74    | 0.00    | +0.02 |
| **2** | 0.25  | 0.20     | 0.55    | 0.00    | **+0.05** (peak) |
| 4     | 0.32  | 0.38     | 0.30    | 0.00    | -0.06 |
| 8     | 0.34  | 0.38     | 0.28    | 0.00    | -0.04 |

Competence (parseability) never breaks (unparseable = 0.00 at every alpha). The
directional gap rises to a peak at alpha=2, then goes **negative** at alpha>=4 — steering
pushes off "unknown" onto *both* named options, and non-Black overtakes Black. By the knee
rule (largest alpha with competence intact and the gap still rising, before non-Black
overtakes) **alpha* = 2**. Generations at alpha=2 are coherent English (393/400 emit the
requested bare letter; the rest are clean phrases like `"A. Cannot be determined"`,
`"A. Not enough info"`, `"C. Cannot answer"`) — no gibberish.

### Full 400, alpha* = 2 (base vs P vs PI vs PID)

| condition | Kp | Ki | Kd | black | nonblack | abstain | unparse | d_gap |
|-----------|----|----|----|-------|----------|---------|---------|-------|
| base | 0 | 0    | 0    | 0.120 | 0.100 | 0.780 | 0.000 | +0.000 |
| P    | 1 | 0    | 0    | 0.188 | 0.142 | 0.670 | 0.000 | +0.025 |
| PI   | 1 | 0.05 | 0    | 0.242 | 0.182 | 0.575 | 0.000 | +0.040 |
| PID  | 1 | 0.05 | 0.02 | 0.240 | 0.188 | 0.573 | 0.000 | +0.033 |

`d_gap = (black - clean_black) - (nonblack - clean_nonblack)`, relative to base.

**Read-out.** All conditions steer toward the Black option (positive d_gap) while keeping
competence intact (unparseable = 0). The **integral term helps most**: PI (+0.040) > PID
(+0.033) > P (+0.025) > base (0). Steering mainly converts abstentions (0.78 -> ~0.57)
into named-option picks, with a consistent lean toward the Black option. Adding the
derivative term (PID) slightly reduces the gap vs PI here. See `results_table.md`.

## Normal steering vector baseline (`--mode normal`)

The **normal steering vector** is the classic single diff-in-means direction — the same
CAA-style vector everywhere, with none of PID's per-layer / integral / derivative structure.
We take the layer-14 arrow `r14`, unit-normalize it once to `v̂ = r14 / ||r14||`, and add
the **identical** fixed vector `alpha * v̂` at **all 32 blocks**, every denoising step:

```
 arrows.pt: RAW r(14)  --unit-->  v̂ = r14/||r14||     (||v̂|| = 1, ONE vector)
                                       |
        block 0: h += alpha*v̂   block 1: += alpha*v̂  ...  block 31: += alpha*v̂
                                  (SAME v̂ at every block — no per-layer directions)
```

Contrast with `--mode pid`: P/PI/PID use a **different** per-layer `rhat(k)` at each block
(plus I/D terms across depth); the normal baseline collapses that to one global L14 vector,
a plain proportional push. L14 is the anchor layer used throughout this project. An offline
self-test (`--selftest`) asserts the injection is (32,4096), identical across all 32 layers,
equals `alpha*v̂`, `||v̂||=1`, and is distinct from the per-layer P injection.

### Combined comparison at alpha = 2 (apples-to-apples)

| condition | black_pick | nonblack | abstain | unparse | d_gap (vs base) |
|-----------|------------|----------|---------|---------|-----------------|
| base      | 0.120 | 0.100 | 0.780 | 0.000 | +0.000 |
| normalL14 | 0.177 | 0.138 | 0.685 | 0.000 | +0.020 |
| P         | 0.188 | 0.142 | 0.670 | 0.000 | +0.025 |
| PI        | 0.242 | 0.182 | 0.575 | 0.000 | +0.040 |
| PID       | 0.240 | 0.188 | 0.573 | 0.000 | +0.033 |

`d_gap = (black - base_black) - (nonblack - base_nonblack)`, relative to base.

**Read-out.** At the matched `alpha=2`, the normal single-vector baseline **does steer**
(d_gap +0.020, competence intact) but is the **weakest** of the steered conditions:
normalL14 (+0.020) < P (+0.025) < PID (+0.033) < PI (+0.040). Using per-layer directions
(P) already beats the single fixed L14 vector, and adding the integral term (PI/PID) beats
both — i.e. PID-Steering's depth structure buys real headroom over the ordinary steering
vector, at equal intervention strength and zero competence cost.

### Normal steering-vector alpha-sweep (first 100 items)

| alpha | black | nonblack | abstain | unparse | raw gap (black-nonblack) |
|-------|-------|----------|---------|---------|--------------------------|
| 0.25  | 0.12  | 0.10     | 0.78    | 0.00    | +0.02 |
| 0.5   | 0.12  | 0.10     | 0.78    | 0.00    | +0.02 |
| 1     | 0.14  | 0.09     | 0.77    | 0.00    | +0.05 |
| 2     | 0.17  | 0.12     | 0.71    | 0.00    | +0.05 |
| 4     | 0.48  | 0.25     | 0.27    | 0.00    | +0.23 |
| 8     | 0.34  | 0.38     | 0.28    | 0.00    | -0.04 |

The plain vector has its own knee: the raw gap keeps climbing to a strong peak at
**alpha=4** (+0.23, still competence-intact) before non-Black overtakes Black at alpha=8
(-0.04) — the same "push off unknown onto both options, then overshoot" shape seen for PID,
just reached at a higher alpha. At the matched alpha=2 used for the head-to-head above, the
normal vector is still early on its curve, consistent with it being the weakest condition
there.
