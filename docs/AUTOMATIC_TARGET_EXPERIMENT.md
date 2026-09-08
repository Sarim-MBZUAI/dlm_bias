# Automatic target-option selection: first experiment

Protocol prepared September 8, 2026 against code `60a0186` and manuscript `9a2e4dc`. This is an experiment design, not a result. It proposes an **annotation-free selector at inference**; benchmark annotations remain legitimate inputs to offline cohort construction and subsequent evaluation. No manuscript claim should change until the experiment is measured. The selector and pilot tooling are now implemented; GPU inference remains unrun. See [portable commands and output limits](../experiments/target_mapping/README.md).

## Question and primary method

Can the existing fitted Black-versus-comparator semantic direction identify which displayed answer option denotes its target, then support PI steering without looking up that option in `answer_info`?

The primary method keeps the original actuator: one fixed unit direction from block 14, broadcast to all 32 LLaDA blocks with the current PI scalar command. It replaces only the per-item target-letter lookup. It does not select a new actuator direction or inject a letter-specific logit bias.

For each displayed candidate `j` in A, B, C:

1. Construct the same chat-formatted question and three answer options used for generation, using only `context`, `question`, `ans0`, `ans1`, and `ans2` (plus the visible `prompt_override` for compatible adapters).
2. Append candidate **answer text**, stripped as in the existing direction builder. Run the frozen model without steering on this fully materialized sequence.
3. Capture block-14 residual states and average over all tokens in the appended answer span, obtaining `h_j`.
4. Score `s_j = dot(vhat, h_j)`, where `vhat = r14 / norm(r14)` is the existing fixed direction. Select the largest score and freeze that A/B/C choice before masked generation begins.
5. Run the unchanged PI control law using the selected letter's probability. The actuator remains `hidden += alpha_t * vhat`.

This matches how the original direction was constructed: it averages differences between pooled target and comparator **answer-text** states, not letter states. [Arrow extraction and averaging](../steering/build_arrows.py).

Use raw projection for this first experiment. Subtracting the same candidate-mean representation from every `h_j` cannot change their ranking. Dividing each candidate state by its own norm would change the ranking and therefore defines a different selector; it is not a harmless implementation detail. No cosine, layer, threshold, or prompt variant is selected by examining pilot outcomes.

This selector is plausible, not guaranteed. A difference-in-means intervention is not automatically a reliable three-way classifier. The abstention answer was excluded from the original positive-versus-comparator fit and may project strongly on the direction. That possible failure is part of the experiment.

## Inference contract

- Score **all three options**, including whichever text happens to express uncertainty. Do not identify or remove it with an annotated unknown index, a gold label, or a special unknown-answer heuristic.
- The selector/control inputs contain only the visible prompt/options, fixed model/tokenizer, and fixed direction. Do not pass `answer_info`, `label`, `target_idx`, `unknown_idx`, positive/negative tags, or stereotype metadata. Keep opaque IDs outside the mathematical selector for joining outputs later.
- Automatic generation receives the sanitized row. Annotation-based filtering belongs to the offline cohort-preparation step; do not condition runtime execution on whether a group-tag lookup succeeds.
- Tokenize the exact prompt and prompt-plus-answer. Assert that the prompt token IDs form the unchanged prefix and that a nonempty answer-token span exists. The historical builder's last-token fallback is unsuitable here. Nonfinite scores, a zero/invalid direction, missing hooks, and invalid spans are explicit failures, never triggers for an oracle fallback or silent sample deletion.
- Candidate passes run with steering disabled and model evaluation mode enabled. Remove their capture hook and restore/reset controller state before generation. Freeze the chosen letter for the complete denoising trajectory; do not chase whichever option currently has the highest probability.
- Use a declared deterministic A/B/C tie break, and log ties. A tie-driven A preference can hurt rotation consistency and must remain visible.
- The oracle-PI reference is deliberately separate and explicitly annotation-aware. Its label access is not permitted to leak into the automatic condition.

CPU contract checks should establish that removing or scrambling annotation fields leaves automatic selection inputs and outputs unchanged, every candidate is scored, direction/steering hooks are correctly separated, and rotated candidates are mapped by their displayed positions. These checks cannot establish model-level mapping accuracy.

## Smallest informative pilot

Use **100 underlying ambiguous Black-referent BBQ items, each in all three cyclic option rotations**: 300 generated prompts per condition. Draw the 100 from the existing 400-item primary cohort using a fixed hash order over canonical string-valued `(category, example_id, question_index)` keys with salt `target-mapping-pilot-20260908`. Hash the salt followed by the key's compact JSON serialization with SHA256. Take the first 100 hashes, then preserve that same semantic subset in every rotation and condition. Save the selected keys, selection rule, source hashes, code revision, and direction hash before obtaining outputs.

Avoid the first 100 rows as the default: the manuscript explicitly identifies those as its earlier gain-calibration set. Hash selection is a reproducible sampling choice, not a claim to erase historical exposure. The primary cohort is disjoint from the direction-building set but has already been used for the paper's tuning/reporting. This is therefore a **historically exposed feasibility pilot held out from direction fitting**, not a pristine confirmation set. (Manuscript appendix, Operating-Point Selection: the first100 primary-cohort items were the gain-calibration subset.)

No new selector calibration is required for the primary raw-projection rule. Keep the existing direction and published PI gains/setpoint/cap fixed. The original 400 direction-building items are training data, not mapping-evaluation examples. Runtime annotations are forbidden even though those original training contrasts were supervised.

Run these four conditions on the same device, model snapshot, precision, prompts, and decoding settings:

| Condition | Target selection | Intervention | Purpose |
|---|---|---|---|
| Clean | None needed for generation | No activation offset | Establish the pilot's own baseline |
| Oracle PI | Existing annotated mapping | Existing PI, `(3, .1, 0)`, setpoint .9, cap 6 | Quantify loss when replacing the lookup |
| Automatic PI | Frozen direction-projection selector | Identical PI and actuator | Test the requested method |
| Fixed open loop | No mapping needed | Same unit `r14`, all 32 blocks, constant alpha 4 | Strong published same-direction static comparator |

Use the current 64-step, 32-token, single-block, greedy configuration for direct continuity with the paper. The final 32 steps make no token transfers; retain this fact in cost/effort reporting. Regenerate the reference conditions with the new pilot rather than treating old outputs from other hardware as a perfectly paired control.

Alpha 4 is the primary fixed control because it is the stronger published Black fixed operating point. Alpha 3.28 may be an explicitly secondary historical comparison; do not describe it as an exact energy match. Do not tune the fixed dose or PI gains on these 100 outcomes.

The first implementation need not add more methods. A useful next control is **automatic mapping plus one initial target-probability probe, followed by frozen alpha**. A fully specified untuned version freezes PI's first command, `clip(3.1 * (.9 - p0), 0, 6)`, for the whole trajectory. It tests whether continuing updates help beyond that initial decision, but is only one static policy. A stronger calibrated `p0 -> alpha` rule needs separate development data and a declared search budget. Neither variant has been run here.

## Costs and intervention matching

The selector costs three unsteered candidate-answer forwards per rotated prompt: **900 candidate forwards for this pilot**. It changes the inference query budget and must be counted. Cache scores only across conditions sharing the exact prompt, options, model/tokenizer, direction, and selector version; do not reuse one rotation's scores for another rotation.

Expected unbatched forward counts for the current implementations are automatic PI `3 + 1 + 64 = 68`, oracle PI `1 + 64 = 65`, and the existing fixed open-loop sampler 64. A clean condition through `denoise_pid` also makes 65 because its initial probe remains present. Verify actual counters in the new harness rather than assuming every clean/static entrypoint is identical. Candidate forwards have different sequence lengths from denoising forwards. The current analysis includes available generation-loop elapsed time; token lengths and peak memory require separate collection during a future run. Batching candidates reduces launch count but does not make their computational work disappear.

If the optional frozen-alpha control is added, it must use the same cached automatic mapping and initial target probability as automatic PI. Report both standalone method cost and amortized cache reuse. Do not charge unused dummy selector calls to ordinary baselines and then claim intrinsic compute parity.

Shared direction/sites/cap do not imply equal realized intervention. Report `sum(abs(alpha))`, `sum(alpha^2)`, and mean command over the first 32 committing steps, alongside full-run totals. Call the squared-command quantity a control-effort proxy; it is not the complete energy of the model. The primary comparison is a fixed operating-point comparison, not an exact resource-matched causal identification.

## Outcomes and analysis fixed before outputs

**Mapping is evaluated separately from steering.** After inference, join annotations in the scorer and report:

- Target-selection accuracy over all 300 rotated prompts, with failures retained in coverage accounting.
- Target/comparator/unknown selection counts and accuracy by actual target position A/B/C.
- Semantic rotation consistency: fraction of the 100 items for which all three rotations select the same underlying answer option after undoing the permutations. Also report the fraction correct in all three rotations. Consistent selection of the wrong option is not accuracy.
- All three scores, top-two margin, selected index/text, tie/failure status, model/direction hashes, and selector-forward count. Any confidence-versus-accuracy analysis is diagnostic; it must not retroactively introduce a threshold.

**End-to-end primary outcomes** are the strict unconditional target-minus-comparator gap, its change from the pilot clean condition, and invalid-answer rate. Report all target/comparator/abstention/invalid counts. The important paired differences are automatic PI minus clean, automatic PI minus oracle PI, and automatic PI minus fixed alpha 4. Do not redefine success as the option the selector chose: success remains the intended demographic target defined by the offline evaluator.

Resample the **100 underlying semantic items**, keeping each item's three rotations and every compared condition together. Use 10,000 seeded paired bootstrap replicates and percentile intervals. Do not resample the 300 rotations as if independent. Mapping metrics should use the same item-clustered treatment. Confirm exact shared item/rotation keys before any paired subtraction.

Conditional results for correctly versus incorrectly mapped items, valid-only gaps, and target-position strata are secondary diagnostics. The primary denominator includes mapping errors and invalid outputs. A successful correctly-mapped subset does not rescue a failing end-to-end method.

Strict parser validity is not the same as coherence. Inspect a preselected hash-random 50 outputs per condition, blinded to condition labels, for a single interpretable answer versus relisting, contradiction, or gibberish. Report this small audit separately with its sample denominator; additionally inspect all parser disagreements descriptively if an existing semantic parser is used. Do not invent a new parser after seeing which condition benefits and make it the primary endpoint.

## Falsifiable decisions and confirmation

Before running, use these **engineering progression thresholds**, not claims of statistical proof:

1. **Contract gate:** no annotation access in the automatic inference path; all three options scored; no oracle fallback; no silently dropped selector failures. A violation stops interpretation until fixed.
2. **Mapping gate:** at least 90% target mapping accuracy and at least 90% semantic consistency across rotations. Report item-clustered intervals regardless of whether the point estimates pass. Failure means the proposed projection is not yet a sufficiently reliable automatic target identifier; do not hide unknown selections or tune a variant on the same pilot and relabel it confirmatory.
3. **Steering progression gate:** automatic PI has positive clean-relative gap, loses at most .05 absolute gap to oracle PI in the point estimate, and increases strict invalidity by at most .05 versus oracle PI. These declared tolerances decide whether the automatic method is promising enough for a larger evaluation. They do not establish equivalence or noninferiority.

The stronger scientific question—whether ongoing feedback improves over the fixed comparator—requires the paired automatic-PI-minus-open-loop interval. If it spans zero, report the result as inconclusive; if it favors the fixed control, report that. A useful automatic mapper can succeed even when feedback does not outperform a static intervention. No superiority claim is justified merely by passing the engineering gates.

If the pilot motivates a change to scoring, layers, prompts, thresholds, or gains, it becomes development data. For subsequent confirmation, audit the 1,600-item pool outside direction fitting and exclude **all known** prior evaluation, sweep, and new-pilot keys. Reserve a fresh fixed manifest before testing further variants. If at least 150 genuinely unused semantic items remain, a concrete allocation is 50 development and 100 untouched confirmation items, each with all rotations; otherwise report the available count and limitation instead of silently reusing exposed items. Existing seeded draws overlap the primary evaluation and are not automatically fresh holdouts. [Pool construction](../eval/balanced/make_superset.py).

This experiment removes the need for runtime benchmark annotations in target identification. It does not automatically establish broad baseline fairness, superiority over letter-conditioned static directions or direct logit interventions, generalization to other targets/models, or novelty. Those remain separate questions. The first result to establish is simpler: whether the existing semantic direction can reliably identify its own target option and preserve useful end-to-end steering when the annotated lookup is removed.
