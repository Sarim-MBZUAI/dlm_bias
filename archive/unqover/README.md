# UNQOVER — archived

This tree holds the whole UNQOVER line of work. It is **archived, not deleted**:
nothing here is part of the paper, nothing here is cited, and nothing outside
`archive/` depends on it any more. It is kept because it is a record of two
benchmarks that were built, run, and then retired for a reason worth
remembering.

Everything moved here keeps the directory structure it had at the repo root, so
paths inside the scripts still read the way they did when they ran:

```
archive/unqover/
  unqover/        the original UNQOVER track — download / loader / eval /
                  metric plus the four steering adapters
                  (build_arrows_unqover.py, denoise_pid_unqover.py,
                  pid_steer_unqover.py, baselines_unqover.py,
                  select_tuned_pi.py) and its README / TUNING_PREREG_jobAJ.md
  unqover_hf/     the replacement HF-based UNQOVER-race benchmark — loader.py,
                  splits.py, metric.py, eval_harness.py, build_arrows_race.py,
                  run_race.py, select_ops.py, sweep_grid.py, PREREG.md, README.md
  data/           tracked manifests plus gitignored source data and derived
                  BUILD/EVAL files when present in a local checkout
  results/        unqover/ (round-1 and round-2 runs), unqover_religion/
                  (the E9 religion baseline), unqover_native/, and
                  unqover_hf/build_sweep/ (summaries plus local raw generations)
  slurm/          the fourteen UNQOVER-only batch jobs plus _uqhf_prologue.sh
  logs/           local SLURM logs, gitignored
```

## The original track, and why its results are gone

The first UNQOVER track produced the numbers that used to appear as the
cross-benchmark row: a large raw shift in the Black preference gap under
decode-PI, most of which vanished under negation debiasing. Those runs lived in
`results/unqover_v2/` and they were **purged as flawed**. Two things were wrong
with them. They were BBQ-transfer runs — the direction and the fits came from
the BBQ pipeline and were carried across to UNQOVER items rather than built on
UNQOVER itself. And the metric was computed on a data release whose two-order
structure had been collapsed, so the order-averaging that makes UNQOVER's
`pref_gap` position-immune by construction was not actually averaging over two
orders.

That purge is why `balanced_all/strict_round2.py` used to die: its
`family_unqover` read `results/unqover_v2/`, the directory no longer existed,
and the resulting `FileNotFoundError` killed the whole analysis before it could
write its `--json` output. The family has been removed from the live analysis
path; there is no UNQOVER number left in the paper for it to back.

What survives in `results/unqover/` and `results/unqover_religion/` are the runs
that were *not* part of that purge, kept for the record.

## The replacement track, and why the whole line was dropped

`unqover_hf/` is the repaired benchmark, rebuilt from the HuggingFace release:
9654 complete quadruples, with a BUILD/EVAL split that is disjoint on both
attribute and template, so no question string and no context string is shared
across the split. The direction and every baseline fit were built on BUILD only.

On that benchmark the unsteered base scores a negation-debiased preference gap
of **+0.146** toward the Black family. Steering raises the target-group pick
rate substantially — from **0.550 to 0.725** on the negative-valence question
q0 — but it raises it *equally* on the negated question q1, **0.404 to 0.725**.
The debiased gap therefore falls to **0.000**.

This is structural rather than a tuning failure. The steering direction is a
group difference-in-means; it carries no information about the valence of the
question being asked. An additive constant injection can therefore only produce
a valence-invariant shift, and at high dose it compresses the q0−q1 difference
to zero. The effect is dose-monotone in both families:

* CAA: **+0.154 → +0.129 → 0.000** at alpha 4 / 16 / 64.
* ours: **+0.125** at `amax=2` → **0.000** at `amax=4`.

A method that moves the raw pick rate but cannot move the debiased gap is not
measuring what the paper claims to measure, and that is the reason the whole
UNQOVER line came out.

The per-configuration table behind those numbers is
`results/unqover_hf/build_sweep/BUILD_TABLE.md`. It and `SELECTED.json` are an
**interim snapshot over 39 configurations**, not a completed selection. The
61-configuration sweep was stopped during archival after 48 configurations had
finished. All 48 available summaries and raw generations are preserved locally;
the nine runs completed after the table snapshot are not represented in
`BUILD_TABLE.md` or `SELECTED.json`. Do not treat the recorded selection as the
pre-registered final operating-point selection. The four tasks still running at
retirement (`25732_{2,3,4}` and `25735_9`) were cancelled; no EVAL task was
submitted.

## What these numbers are and are not

Every number above is from the **BUILD split**, and specifically from the
pre-registered sweep subset of **n = 120 instances** (480 generations per
configuration). The confidence intervals at that size are wide — the base's
debiased gap of +0.146 has a 95% interval of roughly [+0.03, +0.26] — so the
individual per-configuration values should be read as the shape of a dose
response, not as resolved measurements.

The **EVAL split was never run**. It holds 842 complete Black-family instances
(2241 instances in total across families) and no generation was ever produced on
it. Pre-registration for that run is in `unqover_hf/PREREG.md`, written and
committed before any EVAL-split run precisely so it could be run honestly later.

## Local-only artifacts

Several large artifacts were gitignored while the track was live, and they stay
gitignored in the archive. In the checkout where the track was retired they were
moved under the paths below; a fresh clone will not contain them. They are
regenerable from the tracked manifests and code:

* `archive/unqover/data/unqover/` — the raw and generated UNQOVER release
  (~218 MB), rebuilt by `unqover/download_unqover.py` and
  `unqover/unqover_loader.py`.
* `archive/unqover/results/unqover_native/` and
  `archive/unqover/unqover/cache_native/` — the UNQOVER-native direction,
  baseline fits and run outputs (~567 MB together), rebuilt from the committed
  split manifest by `unqover/build_arrows_unqover.py` and the jobAI script.
* `archive/unqover/unqover_hf/arrows_race_black.pt`,
  `archive/unqover/unqover_hf/cache_race/` (~914 MB), and the 48
  `*_raw.jsonl` generations under
  `archive/unqover/results/unqover_hf/build_sweep/` — rebuilt by
  `unqover_hf/build_arrows_race.py` and `unqover_hf/run_race.py`.
* `archive/unqover/data/unqover_hirundo/*.jsonl` — the canonical and split
  JSONL files, rebuilt by `python -m unqover_hf.loader` then
  `python -m unqover_hf.splits`.

The complete local archive is about 1.8 GB; only the compact code, manifests,
summaries and documentation are committed to Git.

## Running any of it where it now sits

The code was moved without being edited, so every module still resolves its own
data and results relative to its parent directory — which is now
`archive/unqover/`, and which is exactly where this track's `data/` and
`results/` were moved to. Those paths therefore still work.

What does *not* still work in place is the shared repo code that these scripts
import as siblings: `steering/`, `baselines/`, `eval/` and `balanced_all/` live
at the real repo root, two levels above this tree.

* For `unqover/`: set `DLM_BIAS_ROOT` to the repo root, exactly as the batch
  jobs in `slurm/` already do. That is enough to make the imports resolve — but
  note it is a single knob: with it set, these modules resolve their *data and
  results* at the repo root too, not inside this archive. That is the behaviour
  they had when they actually ran, so the batch scripts in `slurm/` are still
  self-consistent; just do not expect them to read `archive/unqover/results/`.
* For `unqover_hf/`: there is no such override. Put the repo root's `steering`,
  `baselines` and `eval` directories on `PYTHONPATH` (and `balanced_all` for
  `eval_harness`), which leaves data and results resolving inside this archive,
  or move the tree back to the root as described below.

Concretely, the failure you get without this is
`ModuleNotFoundError: No module named 'denoise_pid'` (or `bbq_eval`) out of a
`--selftest`; with the path set, those selftests pass. Anything that still fails
after that is asking for one of the gitignored artifacts listed above.

## How to revive the track

Work from the replacement benchmark, not the original one. The original track's
metric and data handling are the thing that failed; `unqover_hf/` is the
repaired version and is the only sensible starting point.

1. Move `archive/unqover/unqover_hf/` back to the repo root as `unqover_hf/`,
   and `archive/unqover/data/unqover_hirundo/` back to `data/unqover_hirundo/`.
   Every module imports itself as `unqover_hf.<mod>` and resolves the repo root
   relative to its own file, so it expects to sit at the top level.
2. Re-fetch the source data: the hirundo parquets plus
   `data/unqover/generated/ethnicity.source.json` (via
   `archive/unqover/unqover/download_unqover.py`).
3. Rebuild the derived data and the split, in this order:
   `python -m unqover_hf.loader`, then `python -m unqover_hf.splits`. Check the
   result against the committed `split_manifest_race.json` and
   `loader_report_race.json` — they pin the exact split this archive's numbers
   were computed on.
4. Rebuild the direction and the baseline fits on BUILD only:
   `python -m unqover_hf.build_arrows_race --build` (GPU). This writes
   `unqover_hf/arrows_race_black.pt` and `unqover_hf/cache_race/`.
5. Re-run the BUILD-split sweep to confirm you reproduce
   `results/unqover_hf/build_sweep/`: `unqover_hf/sweep_grid.py --list` gives the
   pre-registered grid, `unqover_hf/run_race.py` runs one configuration, and
   `unqover_hf/select_ops.py --sweep-dir results/unqover_hf/build_sweep` applies
   the `PREREG.md` section-4 selection rule. The batch jobs for all of this are
   `archive/unqover/slurm/unqover_hf_jobP1a_arrows.sbatch`,
   `unqover_hf_jobP1b_smoke.sbatch`, `unqover_hf_jobP2_build_sweep.sbatch` and
   `unqover_hf_jobP3_effort.sbatch`. Note that `_uqhf_prologue.sh` hard-codes the
   worktree path it was written for; fix `WT` before running any of them.
6. Only then run the EVAL split, once, on the operating points that `SELECTED.json`
   already fixes. `PREREG.md` states what that run is allowed to look at.

The core runnable modules in both `unqover/` and `unqover_hf/` carry CPU-only
`--selftest` checks; run those first, as they identify missing imports and
artifacts before a GPU job does. The package-level test commands are listed in
`unqover_hf/README.md`; archived modules that import live shared code need the
`PYTHONPATH` setup described above.

Whoever picks this up should decide first whether the structural objection above
has an answer. A direction with no valence information cannot produce a
valence-dependent shift, and no amount of dose tuning changes that; reviving the
track means changing what is injected, not how much.
