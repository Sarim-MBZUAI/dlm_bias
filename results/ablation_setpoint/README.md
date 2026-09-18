# Setpoint ablation for the decode-time PI attack

**Audience:** a coding agent (or a person) running experiments in the `dlm_bias`
repository on a GPU machine. Follow the steps in order. Do not skip the
preflight checks.

**Status:** planned, not run. Everything in this folder was written and
CPU-checked at commit `a95dc52`+; no GPU inference has been executed.

## Why this experiment exists

The paper's closed-loop attack drives the target-answer probability toward a
setpoint s\* = 0.9. That value is a hard-coded default
(`SETPOINT = 0.9`, `steering/denoise_pid.py:103`). The calibration sweep in
`results/calibration/` tuned the gains and the command limit but never varied
the setpoint. Reviewers will ask whether the result depends on this choice.

This ablation reruns the primary Black-target PI attack with different
setpoints and changes nothing else. The expected outcome is unknown; report
whatever the numbers show.

## Ground rules

- **Change only `--setpoint`.** Keep Kp=3, Ki=0.1, Kd=0 (PI condition),
  amax=6, 64 steps, the default sensor (`--sensor-case upper`), the default
  Black direction (`steering/arrows.pt`, block 14), and oracle target mapping
  (`--target-mapping oracle`, the default).
- **Do not edit any tracked source file**, and do not overwrite anything under
  existing `results/` subfolders. Write all outputs to
  `results/ablation_setpoint/` only. The scripts here already do this.
- **GPU request.** Titan rejects `--gres=gpu:1`; both sbatch files request `--gres=shard:3` (~28.8 GB) for the bf16 8B model. See `titan-shards`.
- **Run every condition on the same GPU model.** Output differs slightly across
  GPU types; an earlier repeat of this attack on a different GPU gave a
  target-comparator gap of 15.1 pp against 16.7 pp. `preflight.sh` records the
  GPU model in `env.json`; the Tier 2 sbatch refuses to start on a different one.
- **Do not tune or rerun based on results.** Run the full grid once, then score it.
- **If a run fails, fix the environment and rerun that run only.** Every run
  script skips conditions whose output already exists, so rerunning the same
  script after a fix redoes only the failed run. Failures are appended to
  `failures.log`; log what happened there.

## What is in this folder

| file | step | purpose |
|---|---|---|
| `common.sh` | – | shared helpers: root resolution, `run_one` (timed, skip-if-present, logs to `timings.tsv` / `failures.log`) |
| `preflight.sh` | 0 | records commit / GPU / versions / file hashes to `env.json`; checks git-ignored inputs; runs `--selftest` and a 2-item `--smoke` |
| `run_repro.sh` | 1 | reruns s\*=0.9 on the first 100 items × 3 rotations; calls `compare_repro.py` → `repro.json` |
| `run_tier1.sh` | 2 | base + s\* ∈ {0.5, 0.7, 1.0} on the first 100 items × 3 rotations; then `check_setpoints.py` and `score.py` |
| `run_tier2.sh` | 3 | s\* ∈ {0.5, 0.9, 1.0} on all 400 items × 3 rotations; then `check_setpoints.py` and `score.py` |
| `compare_repro.py` | 1 | row-by-row match (on `example_id`) of the rerun against `results/balanced/results_balanced/rot*/cond_dpid_PI_samples.jsonl` |
| `check_setpoints.py` | 2, 3 | asserts every `cond_*.json` records the setpoint its tag encodes and the fixed hyperparameters |
| `score.py` | 4 | the paper's strict parser, 3-rotation pooling, 10k bootstrap, paired difference vs s\*=0.9 → `tier*/summary.json` |
| `make_report.py` | 5 | assembles `REPORT.md` from `env.json`, `repro.json`, `tier*/summary.json`, `timings.tsv`, `failures.log`, sha256 of every samples file |
| `selftest.py` | – | CPU-only test of the four Python scripts on synthetic data; run it first |
| `slurm/ablation_setpoint_tier1.sbatch` | 0–2, 5 | one SLURM job for preflight + repro + Tier 1 + report |
| `slurm/ablation_setpoint_tier2.sbatch` | 3, 5 | one SLURM job for Tier 2 + report; checks the GPU model matches Tier 1 |

Output layout after a full run:

```
results/ablation_setpoint/
  env.json  repro.json  timings.tsv  failures.log  REPORT.md
  tier1/rot{0,1,2}/cond_{dpid_base,dpid_PI_s0p5,dpid_PI_s0p7,dpid_PI_s0p9,dpid_PI_s1p0}{.json,_samples.jsonl}
  tier1/summary.json
  tier2/rot{0,1,2}/cond_{dpid_PI_s0p5,dpid_PI_s0p9,dpid_PI_s1p0}{.json,_samples.jsonl}
  tier2/summary.json
```

Tag convention: `dpid_PI_s0p9` means PI at s\* = 0.9 (`.` → `p`). The
controller's `--tag` flag sets the output stem directly
(`steering/denoise_pid.py:776`, `:730-733`), so these names are exact.

## Quick path (SLURM)

```bash
python results/ablation_setpoint/selftest.py                       # CPU, seconds
sbatch results/ablation_setpoint/slurm/ablation_setpoint_tier1.sbatch
# after it prints "ABLATION SETPOINT TIER 1 DONE" and tier1/summary.json looks sane:
sbatch --nodelist=<same node type> results/ablation_setpoint/slurm/ablation_setpoint_tier2.sbatch
```

The manual path follows. The sbatch files run exactly these steps.

## Step 0: Preflight

Run from the repository root. The code resolves paths from `DLM_BIAS_ROOT` if
set, otherwise from the repository root. Set `PY` to the interpreter you want
(defaults to `python`).

```bash
bash results/ablation_setpoint/preflight.sh
```

This does, and records in `env.json`:

```bash
git rev-parse HEAD          # results were produced at a95dc52 or later
nvidia-smi --query-gpu=name --format=csv,noheader   # the GPU model
python -c "import torch, transformers; print(torch.__version__, transformers.__version__)"
# transformers must be 4.46.2 (requirements.txt); preflight exits otherwise
```

The following are git-ignored and must exist before anything runs. Preflight
checks each one and exits if any is missing.

| path (relative to `DLM_BIAS_ROOT`) | what it is |
|---|---|
| `LLaDA-8B-Instruct/` | model weights directory (a symlink in the shared checkout) |
| `steering/arrows.pt` | the Black-target steering directions (the controller uses block 14); override with `DLM_ARROWS_PATH` only if the file lives elsewhere. Expected sha256 `f03323d2…0808a6` (recorded in `env.json`) |
| `results/balanced/_sweep400_rot{0,1,2}.jsonl` | the three position-balanced item files (tracked; 400 rows each) |
| `results/balanced/results_balanced/rot{0,1,2}/cond_dpid_PI_samples.jsonl` | the published PI outputs used by the Step 1 comparison (tracked) |

If `arrows.pt` or the data are missing on a fresh clone, see
`migrate_local_data.sh` and ask the repository owner for the blobs. **Do not
rebuild `arrows.pt` with `build_arrows.py`**; a rebuilt direction would make the
ablation incomparable with the paper.

Preflight then runs the built-in checks; both must succeed before continuing:

```bash
python steering/denoise_pid.py --selftest
python steering/denoise_pid.py --smoke --smoke-items 2 --kp 3 --ki 0.1 --kd 0 --setpoint 0.9
```

## Step 1: Reproduction check (about 10 minutes)

Rerun the primary setting on the first 100 items of each rotation. These are
the same 100 items the original gain calibration used
(`load_items` takes `rows[:limit]`; `results/calibration/calib_denoise/*_n100`
starts at example_id 4536, as does `_sweep400_rot0.jsonl`).

```bash
bash results/ablation_setpoint/run_repro.sh
```

Equivalent to:

```bash
OUT=results/ablation_setpoint/tier1
for r in 0 1 2; do
  python steering/denoise_pid.py --cond PI --kp 3 --ki 0.1 --kd 0 --amax 6 \
    --setpoint 0.9 --limit 100 \
    --items results/balanced/_sweep400_rot${r}.jsonl \
    --out-dir $OUT/rot${r} --tag dpid_PI_s0p9
done
python results/ablation_setpoint/compare_repro.py --tier-dir $OUT \
    --ref-dir results/balanced/results_balanced --limit 100 \
    --out results/ablation_setpoint/repro.json
```

`compare_repro.py` matches `model_output` row by row against the first 100
rows of `results/balanced/results_balanced/rot${r}/cond_dpid_PI_samples.jsonl`
(item order is identical, verified: the `example_id` sequence of every
`_sweep400_rot{r}.jsonl` equals that of the published samples file; matching is
still done on `example_id`). It reports the fraction of identical outputs and
the strict-class agreement. A high but imperfect match is acceptable on a
different GPU type; report it either way and continue.

## Step 2: Tier 1 grid, first 100 items (about 40 minutes)

The original 100-item calibration runs took about 200 seconds each.

```bash
bash results/ablation_setpoint/run_tier1.sh
```

Equivalent to:

```bash
OUT=results/ablation_setpoint/tier1
for r in 0 1 2; do
  ITEMS=results/balanced/_sweep400_rot${r}.jsonl
  python steering/denoise_pid.py --cond base --limit 100 \
    --items $ITEMS --out-dir $OUT/rot${r} --tag dpid_base
  for s in 0.5 0.7 1.0; do
    tag=dpid_PI_s$(echo $s | tr . p)
    python steering/denoise_pid.py --cond PI --kp 3 --ki 0.1 --kd 0 --amax 6 \
      --setpoint $s --limit 100 \
      --items $ITEMS --out-dir $OUT/rot${r} --tag $tag
  done
done
python results/ablation_setpoint/check_setpoints.py $OUT
python results/ablation_setpoint/score.py $OUT
```

The s\* = 0.9 runs from Step 1 complete the grid (`run_tier1.sh` refuses to
score without them). Each run writes `cond_<tag>.json` and
`cond_<tag>_samples.jsonl`; `check_setpoints.py` confirms that every JSON
records the intended setpoint, `condition`, gains, `alpha_max`, `steps`, and
`n`.

Note on `--cond base`: the controller still runs with `steer_on = False`
(`denoise_pid.py:606`), so the base samples carry an all-zero `alpha_traj`.
`score.py` skips the command statistics for it.

## Step 3 (optional): Tier 2, full evaluation (about 2.5 hours)

Only if Tier 1 finishes cleanly. This covers all 1,200 position-balanced
prompts, which is what the paper reports, for the two extreme setpoints plus
the primary one rerun on this machine.

```bash
bash results/ablation_setpoint/run_tier2.sh
```

Equivalent to:

```bash
OUT=results/ablation_setpoint/tier2
for r in 0 1 2; do
  for s in 0.5 0.9 1.0; do
    tag=dpid_PI_s$(echo $s | tr . p)
    python steering/denoise_pid.py --cond PI --kp 3 --ki 0.1 --kd 0 --amax 6 \
      --setpoint $s \
      --items results/balanced/_sweep400_rot${r}.jsonl \
      --out-dir $OUT/rot${r} --tag $tag
  done
done
python results/ablation_setpoint/check_setpoints.py $OUT
python results/ablation_setpoint/score.py $OUT
```

## Step 4: Score

`score.py` uses the paper's strict parser (identical regex to
`balanced_all/strict_pool.py:391`), pools the three rotations, and keeps
invalid outputs in the denominator. Per condition it reports n, target /
comparator / abstain / invalid percentages, the gap in percentage points with a
10,000-resample item-level bootstrap CI (seed 0), the paired gap difference
against `dpid_PI_s0p9` with its CI (same resample indices), the mean command
over steps 1–32 (the only steps that commit tokens), and the share of
sequences whose command ever reaches the limit of 6.

```bash
python results/ablation_setpoint/score.py results/ablation_setpoint/tier1
python results/ablation_setpoint/score.py results/ablation_setpoint/tier2   # if run
```

Row order across conditions is asserted equal (`rot`, `example_id`) before the
paired difference is computed.

## Step 5: Report back

```bash
python results/ablation_setpoint/make_report.py
```

writes `results/ablation_setpoint/REPORT.md` containing:

- Commit hash, GPU model, torch and transformers versions, and wall-clock time
  per run (from `timings.tsv`).
- The Step 1 reproduction match rate against the published PI outputs.
- For each tier, a table with one row per condition: setpoint, target %,
  comparator %, abstention %, invalid %, gap (pp) with 95% CI, gap difference
  from s\* = 0.9 with 95% paired CI, mean command over steps 1–32, and share of
  sequences that ever reach the command limit.
- Any failed or rerun jobs and why (from `failures.log`; add prose there if
  the fix needs explaining).
- `sha256sum` of every `*_samples.jsonl` produced.

Report the numbers without interpretation or paper edits. **Do not commit
anything unless the repository owner asks.**

## Notes

- **Tier 1 is a sensitivity check, not a held-out test.** The first 100 items
  were also used to calibrate the gains, and they are part of the evaluation
  pool. Tier 2 matches what the paper reports.
- **What would count as an answer for reviewers.** Roughly flat gaps and
  invalid rates across s\* = 0.7 to 1.0 support the claim that the setpoint
  mainly needs to sit above the pre-commitment target probability, which the
  logged trajectories place well below 0.9. A clear drop at s\* = 0.5 would be
  consistent with that account, because a low setpoint lets the integral stop
  accumulating once an example leans toward the target. Either result is
  reportable.
- **Related data that already exists.** `results/calibration/calib_denoise/`
  contains P-only runs (`cond_P_kp{1,3,6}_n100`) on the same 100 items, which
  could support a P-versus-PI comparison without new compute.
- **Half of the 64 steps are no-ops.** With `gen_length = block_length = 32`,
  steps 33–64 commit no tokens (README, "steps=32" paragraph). This is the
  paper's protocol and is kept here for comparability; `score.py` therefore
  averages the command over steps 1–32 only.
- **Timing accounting.** `run_one` in `common.sh` records start time,
  wall-clock seconds, exit code and GPU name for every invocation, including
  reruns, so the report's per-run timings are complete without manual notes.
