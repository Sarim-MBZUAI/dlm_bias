# SocialStigmaQA-MC3

This directory implements the preregistered second-benchmark evaluation of the
LLaDA steering methods on SocialStigmaQA.  It uses the official greedy-decoding
(`yes_no`) data pinned at Hugging Face revision
`e53d65d53bc74b7079a2a57f97e35c73d6f9fba4`.

## Task adaptation

The original question text is unchanged.  Its final `Answer with yes/no/can't
tell.` instruction is replaced with explicit A/B/C options for `Yes`, `No`, and
`Can't tell`, followed by the repository's existing answer-letter instruction.
All three cyclic option rotations are evaluated.

The dataset's `biased answer` is the steering target, the opposite yes/no answer
is the comparator, and `Can't tell` is abstention.  `original` items have
`Can't tell` as gold; `positive` items have the comparator as gold.  The base
items have no gold label and are reported only as identity-free diagnostics.

## Frozen design

- Evaluation: seven race stigmas x 37 templates x (`original`, `positive`) =
  518 scored semantic items.
- Diagnostics: 37 no-stigma base prompts.
- Position balance: 555 semantic items x three rotations = 1,665 generations
  per condition.
- Calibration: 200 yes-biased + 200 no-biased `original` prompts sampled
  deterministically from non-race stigmas.  Race prompts never enter fitting.
- Polarity-specific artifacts prevent yes-target and no-target vectors from
  cancelling.
- Generation and operating points match the authoritative BBQ 14-condition
  suite.  The only benchmark-derived operating point is the label-free
  effort-matched open-loop alpha, computed from decode-PI's first 32 live steps.

Raw source data, derived JSONL, tensor caches, and raw generations are ignored.
The pinned split manifest and compact final result summaries are tracked.

## Local preparation

```bash
python -m socialstigma.prepare --check
python -m socialstigma.selftest
```

GPU fitting and evaluation are run through the `slurm/socialstigma_*.sbatch`
scripts.  Every runner checks both its summary and sample file before skipping,
so interrupted/requeued jobs are safe to resume.
