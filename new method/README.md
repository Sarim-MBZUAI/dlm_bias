# Automatic target selection and PI steering

This folder runs a complete inference path for the existing **Black-target LLaDA
direction**: identify which displayed answer denotes the direction's target,
freeze that A/B/C identity, then control its probability during diffusion
generation. It also runs the controls needed to measure whether feedback helps.
**The new experiment has not been run on a GPU.** CPU checks establish software
contracts; they do not establish mapping accuracy or improvement over baselines.

The default positive direction and setpoint reproduce the paper's target-pushing
experiment. This is an experiment in inducing a target preference, not a claim
that the default operation removes bias.

## Start here

Use the project's installed `dlm` environment and a GPU you are authorized to use. The
launcher works from any directory, including paths containing spaces. It does
not select a GPU, submit a job, download weights, or install dependencies.

```bash
export PYTHON=/path/to/dlm/bin/python
export DATA_ROOT=/path/to/dlm_bias
export MODEL=/path/to/local/LLaDA-8B-Instruct
export RUN_ROOT=/path/to/new_outputs

# CPU preparation and cost estimate; no model is loaded.
bash "$DATA_ROOT/new method/run.sh" --preset smoke --compare --plan-only \
  --data-root "$DATA_ROOT" --model "$MODEL" --out-root "$RUN_ROOT/smoke_plan"

# Later, inside your GPU allocation: two semantic items in three option orders.
bash "$DATA_ROOT/new method/run.sh" --preset smoke --compare \
  --data-root "$DATA_ROOT" --model "$MODEL" --out-root "$RUN_ROOT/smoke"

# Freeze the protocol before this 100-item, three-rotation comparison.
bash "$DATA_ROOT/new method/run.sh" --preset pilot --compare \
  --data-root "$DATA_ROOT" --model "$MODEL" --out-root "$RUN_ROOT/pilot"

# Run only the requested automatic method, without comparison arms.
bash "$DATA_ROOT/new method/run.sh" --preset pilot \
  --data-root "$DATA_ROOT" --model "$MODEL" --out-root "$RUN_ROOT/auto_pi"
```

`--preset full` uses all eligible semantic items; use a CPU plan first to inspect
the actual count. `--limit N` overrides a preset's item count. `--help` lists the
implemented options. Use a separate output directory for each run.

**Weights were excluded from the requested machine transfer.** Restore a complete
local LLaDA snapshot before an inference run; an absolute project symlink alone
does not provide its weights. The runner checks local assets and uses offline
loading. Model configuration, custom model code, tokenizer files, and all weight
shards are needed. The existing local `steering/arrows.pt` and benchmark data are
also required; a source-only Git checkout does not contain these ignored files.
Set `PYTHON` to the interpreter itself, not a shell command with extra arguments.
`--execution-mode auto` uses an existing SLURM allocation when present and
supports directly managed GPUs on other machines. On `mbz-titan-*`, an allocation
is always required, including when `--execution-mode direct` is requested.
Use `--execution-mode slurm` to explicitly require an allocation elsewhere, or
`--execution-mode direct` for a directly managed lab GPU. Preserve the
allocation's `CUDA_VISIBLE_DEVICES` setting; on a direct machine, select only
the GPU assigned to you. The launcher does not override device visibility.

Before loading, the runner checks free memory on its logical CUDA device 0.
The conservative default is 24,000 MiB. `--min-free-memory-mib` changes this
preflight threshold; it does not reduce the model's memory requirement.

For custom prompts, supply `--items /path/to/items.jsonl`. Each row needs only
`context`, `question`, `ans0`, `ans1`, and `ans2`; annotations are optional and
automatic inference does not read them. Custom input retains its original answer
order by default. Pass `--rotations 3` for a three-order diagnostic. Without
annotations, generation and selector outputs remain available; demographic
mapping accuracy and target-minus-comparator scores cannot be inferred from the
method's own chosen letter.

```json
{"context":"A Black person and a White person arrived.","question":"Who arrived first?","ans0":"The White person","ans1":"Cannot tell","ans2":"The Black person"}
```

## What the method does

The fitted direction is supervised: the earlier contrast set contains target and
comparator answer texts. Inference receives visible text and the fixed direction,
not benchmark group tags or the correct-answer label.

For each of the three displayed answers, one unsteered forward pass appends its
answer text to the chat-formatted prompt. The selector averages the appended
answer-token residuals at transformer **block index 14**, projects that vector
onto the unit fitted `r[14]`, and chooses the largest score. The uncertainty
answer participates equally. Exact token-prefix checks reject invalid answer
spans; ties are logged with a deterministic first-position tie break. There is no
annotation fallback and no retuning of the direction for a displayed letter.

The chosen letter stays fixed for the whole diffusion trajectory. PI observes
its answer-position probability and chooses one scalar alpha each step. The
actuator adds `alpha * unit(r[14])` to every position at all 32 transformer blocks.
The sensor's letter selects what to measure; it does not replace the semantic
direction with an A/B/C logit intervention. The inherited fixed settings are
`Kp=3`, `Ki=.1`, `Kd=0`, setpoint `.9`, alpha limits `[0,6]`, anti-windup,
64 denoising steps, 32 generated tokens, greedy sampling, and one 32-token block.

## Comparison and interpretation

`--compare` runs these arms with one loaded model and the same prompts, rotations,
unit direction, and generation settings:

| Arm | Behavior | Standalone model forwards per displayed prompt |
|---|---|---:|
| `clean` | Unsteered generation | 64 |
| `selector_only` | Output the selector's A/B/C directly; a readout diagnostic | 3 |
| `fixed_openloop` | Constant alpha 4 using the same direction at all 32 blocks | 64 |
| `auto_openloop` | Infer the target, measure its initial probability, hold the first PI command fixed | 68 |
| `auto_pi` | Infer the target, then update PI throughout generation | 68 |

`auto_openloop` holds `clip(3.1 * (.9 - p0), 0, 6)` for the complete trajectory.
It gives the static control the same inferred mapping and initial observation as
automatic PI, so their difference tests the value of continued feedback for this
declared static policy. Alpha 4 adds the published fixed operating point. Neither
comparison proves superiority over every possible static controller.

The direct selector readout is useful because target identification might already
solve the forced-choice task. Its result must remain separate from activation
steering: it generates no denoised answer and should not be described as a
compute-matched open-loop actuator. `--oracle-diagnostic` adds the explicitly
annotation-aware PI reference; it is not part of automatic inference.

The comparison shares each prompt's three candidate forwards across the arms
that use the selector: **261 actual forwards per displayed prompt**, versus 267
if the five methods were run independently. Logs must distinguish shared work
from the standalone cost attributed to each method. Candidate and denoising
forwards have different sequence lengths, so call counts are not elapsed time.
The existing sampler commits all 32 tokens in its first 32 steps; its final
32 forwards remain part of the historical configuration and measured cost.

## How long to allow

For **100 semantic items in three rotations**, provisionally allow **10–15 minutes
for automatic PI alone**, or **40–60 minutes for the five-arm comparison**,
including a single model load and analysis. These are planning estimates for a
similar high-memory GPU, not timings measured for the new method. A two-item,
three-rotation comparison is a short infrastructure check; model startup can
dominate it. Full-run time scales with the eligible item count printed by the
CPU plan. Queue waiting time is additional.

The estimates use historical PI records at roughly 1.53–2.00 seconds per output
and fixed-alpha records at 2.23–2.28 seconds per output. The faster seed runs
identify an NVIDIA RTX PRO 6000 Blackwell Server Edition in their SLURM log.
Those logs exclude model loading, and the three selector forwards have different
sequence lengths from generation. Hardware, precision, storage, and contention
can change the result substantially. Use the new run's measured item rate,
elapsed times, and tqdm ETA after warmup to revise the estimate.

Local timing evidence, when the ignored results/logs are available:
`results/balanced/results_balanced/rot{0,1,2}/cond_dpid_PI.json`,
the corresponding `cond_normalL14_a4.json` files,
`results/balanced_seeds/seed{1,2,3}/decode_pid/rot*/cond_dpid_PI.json`, and
`logs/slurm/seeds-17158.out`.

## Cohort and evaluation

Benchmark cohort preparation uses local data and annotations offline, before
generation, and records excluded known fitting/evaluation keys. Exclusion of
known keys is not proof that an item was never seen in any historical sweep.
Freeze the selected manifest and method settings before looking at outcomes.
Keep mapping errors, unknown selections, and invalid outputs in the primary
denominators. Evaluation scores the intended annotated target, not whichever
answer the automatic selector happened to choose.

Compare conditions by resampling underlying semantic items with all rotations
and arms together. Report target/comparator/unknown/invalid outcomes, mapping
accuracy, rotation consistency, strict unconditional gap, and paired intervals.
The implemented primary interval is automatic PI minus automatic open loop,
using 10,000 seeded bootstrap draws of semantic items. Other arm summaries are
descriptive; additional intervals or a blinded coherence audit require separate
analysis of the saved outputs.
A strict valid letter is not a coherence assessment. Successful CPU tests or an
accurate selector alone do not demonstrate that PI improves end-to-end steering.

Each output directory contains `plan.json` with input/code hashes, exclusions,
frozen settings and forward budgets; `cohort.jsonl` with offline evaluation
metadata; and `visible_inputs.jsonl` containing only inference text. A CPU plan
writes these files and marks `status.json` as planned. An inference run additionally
writes `generations.jsonl` before scoring, `samples.jsonl` with subsequent
evaluation, and `summary.json` with metrics, actual call counts and measured
loading/inference times. `status.json` distinguishes planned, running, complete,
and failed runs. A failure preserves partial files and requires a fresh output
directory; a partial run must not be interpreted as a completed comparison.

The cohort is Black-versus-other: the comparator need not be White. It includes
both negative and nonnegative question polarities. Do not replace the full
denominator with only correctly identified targets or one favorable subgroup.

## Direction fitting and reproducibility

The first experiment reuses the existing 400-contrast `steering/arrows.pt` and its
`steering/direction_examples.jsonl` provenance. The direction is not fitted again
on evaluation items. A different target or model needs its own validated fitted
direction and held-out experiment; this launcher is not an automatic Dream/MoE
or multi-demographic port.

The historical fitter is [`steering/build_arrows.py`](../steering/build_arrows.py).
It uses paired supervised target/comparator answer states and excludes the
declared evaluation keys. Refitting requires a separate GPU allocation, complete
local caches, and a separate output checkout because the legacy fitter writes
into its own `steering/` directory. Do not overwrite the frozen direction midway
through a comparison. No direction fitting runs implicitly from this launcher.

Run the CPU tests from the repository root:

```bash
"$PYTHON" -m unittest discover -s 'new method/tests' -p 'test_*.py' -v
```

The tests use tiny model/annotation fixtures and shell argument checks. They do
not load model weights or estimate model-level scientific outcomes.
