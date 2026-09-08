# Automatic target mapping pilot

This harness compares clean generation, annotated-target PI, automatic-target PI,
and fixed alpha 4 steering using the same unit block 14 direction at all 32 blocks.
It implements the [frozen experiment protocol](../../docs/AUTOMATIC_TARGET_EXPERIMENT.md).
**GPU inference has not been run.** CPU tests verify contracts, not mapping quality.

Run from the integrated code checkout with its Python environment. The data root
must contain the local `LLaDA-8B-Instruct` model/tokenizer, `steering/arrows.pt`,
`results/balanced/_sweep400_rot{0,1,2}.jsonl`, and the matching `eval/` sources.
Set `DLM_ARROWS_PATH` to use an explicitly chosen existing arrow file. The harness
never downloads model weights. A clean Git checkout alone does not contain all
these ignored assets.

```bash
# CPU only: persist the 100-item hash-selected cohort, hashes, and exact commands.
python -m experiments.target_mapping.run plan \
  --data-root /path/to/existing/dlm_bias \
  --out-root /path/to/new/pilot

# Later, inside an authorized GPU allocation: twelve sequential condition/rotation jobs.
python -m experiments.target_mapping.run run /path/to/new/pilot/plan.json

# CPU analysis after all jobs complete; missing/mismatched items are errors.
python -m experiments.target_mapping.analyze /path/to/new/pilot/plan.json \
  --out /path/to/new/pilot/analysis.json

# Focused CPU tests, no model downloads or GPU jobs.
python -m unittest discover -s tests -p 'test_target*.py' -v
```

`plan --limit 2` creates a separate infrastructure smoke cohort; the scientific
pilot defaults to 100. `--secondary-alpha3p28` adds the historical secondary dose.
Do not change the primary selector, PI gains, or alpha 4 based on pilot outcomes.
The prior first 100 rows were used for gain calibration; the fixed hash sample
avoids selecting that prefix wholesale but remains a historically exposed pilot.

Planning writes `plan.json` and three input files, and does not load a model.
Execution verifies input, direction, and source-code hashes. It also requires the
normal runner's data-root sampler to match the code checkout's sampler. Existing
run directories are not overwritten or silently resumed; after a failed run,
diagnose the failure and create a fresh plan/output directory. Selector failures
stop the run; no fallback to annotations or deletion of failed items is allowed.

The analysis verifies complete condition/item/rotation pairing and the actual
prompt/options before scoring. It reports strict target/comparator/abstention/
invalid counts; paired 10,000-resample semantic-item gap intervals; mapping
accuracy, unknown selections, ties, rotational consistency and all-rotation
correctness; item-clustered mapping/invalid intervals; and logged selector/total
forward counts. Mapping mistakes remain in all primary denominators. PI alpha
traces supply live 32/full 64 scalar-effort proxies; fixed-alpha values are marked
analytic. Available runner elapsed times exclude model loading. Peak GPU memory,
token-length-normalized compute, and the protocol's blinded coherence audit need
separate collection; strict validity alone is not coherence.

The automatic mode currently applies to the primary LLaDA runner. This pilot
does not silently port it to Dream, MoE, or multi-target wrappers.

Repository deployment documentation names historical local environments and
SLURM on `mbz-titan-3`. A `/home/lukas/...` interpreter path appears in the old
steering README, but no Lukas hostname, application service, or deployment
procedure is specified. These commands target an allocated research GPU; they
do not deploy a UI or select a remote host.
