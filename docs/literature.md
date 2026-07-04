# Literature Review — Social Bias in Diffusion Language Models

Reading list for this project, organized into five buckets. For each paper we
record **the model(s) used**, **the benchmark(s)/datasets used**, its **method/focus**,
and **how it differs from us**. Papers with future (2026) dates that could not be
confirmed are marked **UNVERIFIED** rather than described from assumption.

## What "us" means (the anchor for every comparison)
**Our method (the contribution):** training-free, inference-time bias _injection_ into
**LLaDA-8B-Instruct** (a masked/discrete **diffusion** language model, frozen weights) via
**activation steering** — a forward hook adds `α · direction` to the residual stream at
transformer **block L14**. Directions are built two ways: (a) group mean-difference of
contrast pairs (CrowS-Pairs / StereoSet), and (b) **item-anchored CAA** (answer-text
contrast: "The Black man" vs "The white man"). Activation steering is the whole of our method.

**Baseline we compare against (NOT our method):** **Ghostwriter** — an **input-space**
fabricated-evidence prompt-injection attack, reimplemented from Yang et al.
(arXiv:2606.06244) purely as a **comparison point** (input-space vs. our activation-space
channel). We do not propose it, and it is not part of our contribution — it exists only to
contrast against, and we found it is a non-directional abstention-suppressor like the group
mean-diff steering.

We evaluate on **BBQ** (Bias Benchmark for QA) with the official `s_AMB`/`s_DIS` scores
**plus** absolute pick-rate "attack metrics": abstention collapse, target-rate,
unknown→group flips, and the black-vs-non-black **directional gap**.

**Headline internal findings:** group mean-diff steering is a *non-directional*
abstention-suppressor (erodes "Unknown", doesn't aim); **item-anchored answer-text**
steering is the first *directional* lever (aims at a specific option, competence-preserved).

## Headline comparison — the papers closest to us
| Paper | Model | Benchmark / data | Channel & aim | How it differs from us |
|---|---|---|---|---|
| **Adaptive Steering & Remasking** (Lee & Han 2026) | **LLaDA** (diffusion) | jailbreak-safety sets | Inference-time residual-stream **direction**, **project-out @ L31**, **defend** | Closest method — but *removes* a safety direction to defend; we *add* α·direction @ L14 to **inject**; safety not demographic bias |
| **TrustLDM** (Mo et al. 2026) | LLaDA / multiple LDMs | UCI-Adult / EOD (fairness), safety, privacy | **Benchmark**, evaluation | Closest on scope (LDM + fairness) but tabular EOD, **not BBQ**; measures, doesn't inject |
| **DiffuGuard** (Li et al. 2025) | diffusion LLMs | jailbreak-safety | Training-free **defense** (stochastic remasking) | Defense vs our attack; safety not social bias; identifies remasking "harmful bias" |
| **DLM-SWAI** (An & Han 2026) | DLM | style/safety control | Token-level attribute scores during denoising; **logit-space** steering | Logit-space (found > activation for style) vs our **activation-space**; not bias |
| **ILRR** (Avrahami & Nachmani 2026) | **LLaDA**, MDLM | sentiment / attribute transfer | Inference-time **activation/latent** steering; align generated activations to a **single reference sequence** each denoising step | **Same channel as us** (activation, not logit) — but reference-transfer of one example's activations, not a fixed contrastive direction; differs on *how the signal is built* (our axis); sentiment not BBQ bias |
| **Steering Without Breaking** (Zhou, Roy & Gangadharaiah 2026, AWS) | **LLaDA, Dream, MDLM** (124M–8B) | sentiment / topic / style (7 tasks) | **Residual-stream SAE contrastive** (target vs non-target) steering + **adaptive per-denoising-step schedule** (attributes commit on distinct schedules) | **Closest to our method** — residual contrastive steering ≈ our mean-diff AND the "when-in-denoising to intervene" axis = our deferred **E8**; but style/sentiment, never BBQ/bias, never aim-vs-disinhibit |
| **Diffusion Guided LM** (Lovelace et al. 2024) | continuous diffusion LM | RealToxicityPrompts, Jigsaw Unintended Bias | Plug-and-play **guidance**, toxicity **mitigation** | Mitigation vs injection; toxicity not BBQ stereotype QA |
| **DiffLens / DiffusionBias** (Shi et al. CVPR 2025) | Stable Diffusion (**image**) | demographic attrs | SAE **mechanistic** activation editing | Image not text; debias not inject — but the direct interpretability analogue of our L14 steering |
| **LLaDA** (Nie et al. 2025) | **LLaDA-8B (our model)** | MMLU/GSM8K/… (capability only) | Foundation model | The model we attack; **no bias eval** — frames our gap |
| **BBQ** (Parrish et al. 2022) | UnifiedQA etc. (AR) | — (it *is* the benchmark) | 11-category ambiguous/disambiguated QA; `s_AMB`/`s_DIS` | Our evaluation; we add attack-metrics on top; originally on AR models, not diffusion |

---
## 1. Direct DLM fairness / safety / trust

*Our project, for reference:* training-free, inference-time **bias INJECTION (attack)** into **LLaDA-8B** (masked diffusion LM) via (1) **L14 activation steering** — a forward hook adds `alpha * direction` to the residual stream —. Separately, we compare against a **Ghostwriter** input-space fabricated-evidence prompt-injection baseline (Yang et al., 2606.06244) — a comparison point, NOT our method. Evaluated on **BBQ** with abstention-collapse / target-rate / directional-gap metrics. Frozen weights, no training.

### TrustLDM: Benchmarking Trustworthiness in Language Diffusion Models (Mo et al., 2026)
- **Model(s):** Four open LDMs — **LLaDA**, **LLaDA-1.5** (RL-enhanced), **LLaDA-MoE** (sparse MoE), **Dream** (AR-adapted); plus closed-source **Mercury Edit 2**. Spans six decoding orders and six categories of static "post contexts."
- **Benchmark(s)/datasets:** Custom TrustLDM suite across three axes — **Safety:** TrustLDM-Adv (AdvBench, 50 harmful Qs) + TrustLDM-JBB (JailbreakBench); **Privacy:** TrustLDM-PRI (500 ex); **Fairness:** TrustLDM-Fair = **UCI Adult** (200 gender-balanced instances, income prediction). **Does NOT use BBQ or StereoSet.** Fairness metric = **Equalized Odds Difference (EOD)** (max accuracy gap male vs. female).
- **Method / focus:** First trustworthiness benchmark for LDMs. Finds LDMs are trustworthy on bare user prompts but degrade sharply when malicious post-contexts are appended to the masked response; proposes TrustLDM-Auto to auto-discover vulnerable decoding configs.
- **vs. ours:** Closest neighbor by scope (LDM trust incl. LLaDA + fairness). But: measurement benchmark, not an attack method; fairness = tabular UCI-Adult/EOD, **not BBQ demographic QA bias**; its "attack" surface is malicious **input-space post-contexts** (like the Ghostwriter baseline (comparison point), not our activation channel); no activation/residual-stream steering. We inject bias via L14 hooks and measure BBQ abstention-collapse — orthogonal channel and metric.
- **URL / status:** VERIFIED — https://arxiv.org/abs/2606.00023 (arXiv 2606.00023).

### DiffuGuard: How Intrinsic Safety Is Lost and Found in Diffusion Large Language Models (Li et al., 2025 / ICLR 2026)
- **Model(s):** Four dLLMs (LLaDA-family / Dream class dLLMs; evaluated across four models).
- **Benchmark(s)/datasets:** Jailbreak/safety — six jailbreak attack methods; reports **Attack Success Rate (ASR)** reduced 47.9% -> 14.7%; utility preserved. No BBQ / demographic-bias benchmark.
- **Method / focus:** Analyzes dLLM jailbreak vulnerability along intra-step vs. inter-step dynamics; identifies "Denoising-path Dependence" (early-token safety dictates final output) and a harmful bias in greedy remasking. **Training-free DEFENSE:** Stochastic Annealing Remasking + Block-level Audit and Repair.
- **vs. ours:** Same family (training-free, dLLM, remasking dynamics) but opposite intent — **defense against jailbreaks**, not bias injection. "Harmful bias" here = toxicity/jailbreak, **not demographic/social bias**; no BBQ, no activation-steering (works via remasking + internal-rep audit). We are an attack on BBQ social bias via residual-stream hooks.
- **URL / status:** VERIFIED — https://arxiv.org/abs/2509.24296 ; OpenReview https://openreview.net/forum?id=zBPzxhso8M .

### Adaptive Steering and Remasking for Safe Generation in Diffusion Language Models (Lee & Han, 2026)
- **Model(s):** **LLaDA (7B)** and **Dream (8B)** — both masked diffusion.
- **Benchmark(s)/datasets:** JailbreakBench (100), AdvBench (520), HarmBench, StrongReject; utility on TruthfulQA, MATH-500, MMLU. No BBQ / demographic bias.
- **Method / focus:** Training-free, inference-time **DEFENSE**. Learns a contrastive "safety direction" (SGD) and, during early denoising steps, **projects it out of the residual-stream hidden states at the final transformer layer (layer 31)** (`h - beta*<h,v>v`), then remasks suspicious tokens. Cuts jailbreak success to 0.64%.
- **vs. ours:** **Most methodologically similar channel** — a latent/residual-stream direction applied via inference-time intervention on LLaDA, no training. Key differences: it **subtracts/projects-out** a direction to *remove* harm at **layer 31** on the final step region; **we ADD `alpha*direction` at L14 to INJECT bias**. Defense vs. attack; jailbreak-safety vs. **BBQ social bias**; plus it adds a remasking stage we do not use.
- **URL / status:** VERIFIED — https://arxiv.org/abs/2605.13043 (arXiv 2605.13043).

### DLM-SWAI: Steering Diffusion Language Models Before They Unmask (An & Han, 2026)
- **Model(s):** **LLaDA-8B-Instruct** and **Dream-v0-Instruct-7B**.
- **Benchmark(s)/datasets:** OneStopEnglish (OSE, readability), WikiPol (politeness), RealTox (toxicity). No BBQ / demographic bias.
- **Method / focus:** Training-free, inference-time controllable generation. "Statistical Writing style Aligned Inference": adds a **pre-computed vocabulary-level steering bias to the logits at every masked position each denoising step** — i.e., **token-distribution / logit steering, explicitly NOT residual-stream activation steering** (they tested activation steering as a baseline and found it *less* effective). No auxiliary model, no hidden-state access.
- **vs. ours:** Same goal family (training-free inference-time steering of LLaDA-8B) but the **opposite steering channel from ours**: logit-space bias vs. our residual-stream L14 activation hook. Useful as a contrast point — it argues logit steering beats activation steering for *style*, whereas we use activation steering for *bias injection*. Attribute = style/politeness/toxicity, not BBQ demographic bias; beneficial control, not attack.
- **URL / status:** VERIFIED — https://arxiv.org/abs/2605.29626 (arXiv 2605.29626).

### ILRR: Iterative Latent Representation Refinement — Inference-Time Steering for Masked DLMs (Avrahami & Nachmani, 2026)
- **Model(s):** **LLaDA** and **MDLM** (masked / discrete diffusion). Same DLM family as us.
- **Benchmark(s)/datasets:** attribute steering, primarily **sentiment**; generation-quality metrics (fluency). No BBQ, no demographic bias.
- **Method / focus:** learning-free, inference-time. Steers using a **single reference sequence**: at each denoising step it runs **one extra parallel forward pass** and dynamically **aligns the generated sequence's internal activations toward the reference's activations** ("latent-representation refinement"), with a **tunable steering scale**. *Spatially Modulated Steering* regulates guidance intensity across positions so a short reference can steer longer text. This is **activation/latent-space steering — the SAME channel as us**, unlike DLM-SWAI's logit-space.
- **vs. ours:** The **closest paper on steering channel** — both are training-free activation/latent-space steering of a masked DLM with a tunable scale (their scale ≈ our α). It differs on **HOW THE STEERING SIGNAL IS CONSTRUCTED**, which is *exactly our contribution axis*: ILRR **transfers one reference example's activations by online per-step alignment**; we **add a FIXED, precomputed CONTRASTIVE direction** (CrowS/StereoSet mean-diff, or the item-anchored answer-text CAA) via a forward hook — **no reference, no per-step realignment**. Attribute = sentiment/semantic transfer, never demographic bias, BBQ, target/non-target, or abstention, so it never studies the **aim-vs-disinhibit** question. Beneficial control/transfer, not injection. **Key implication:** ILRR (activation-space) + DLM-SWAI (logit-space) together show that *inference-time steering of DLMs is NOT unoccupied territory* — our novelty rests on the **demographic-bias / BBQ setting** and the **"how the direction is built → aim vs disinhibit" finding**, not on "we steer a DLM at inference" or "we use activation space."
- **URL / status:** VERIFIED — https://arxiv.org/abs/2601.21647 (arXiv 2601.21647).

### Steering Without Breaking: Mechanistically Informed Interventions for Discrete DLMs (Zhou, Roy & Gangadharaiah, 2026; AWS AI Labs)
- **Model(s):** four DLMs, **124M–8B** — **LLaDA, Dream, MDLM**.
- **Benchmark(s)/datasets:** 7 steering tasks over **sentiment / topic / style** attributes (incl. simultaneous 3-attribute control). **No BBQ, no demographic bias, no abstention.**
- **Method / focus:** trains **sparse autoencoders (SAEs)** on DLM **residual-stream** activations to find **WHEN each attribute "commits"** during denoising (topic within the first ~2% of steps, sentiment gradually over ~20%). Shows a **uniform** per-step intervention wastes steering capacity and degrades quality (worse when steering multiple attributes jointly), then proposes an **adaptive scheduler** that concentrates **residual-stream contrastive** (target-corpus vs non-target-corpus) intervention only on the steps where the attribute is actively forming. Cost/quality trade-off has a closed-form characterization via a dispersion statistic of the commitment distribution. Reaches up to 93% steering strength (+15pts over the strongest baseline) while preserving quality.
- **vs. ours:** **The closest paper to our METHOD, on two axes at once.** (1) **Channel:** residual-stream *contrastive* activation steering (target vs non-target) ≈ our group mean-diff construction. (2) **Denoising trajectory:** its "when does an attribute commit / intervene adaptively across denoising steps" analysis **is exactly our deferred E8** (denoising-trajectory localization). What stays ours: **demographic bias on BBQ** (they steer sentiment/topic/style, never a *decision with an abstention option*), and the **aim-vs-disinhibit** distinction (their metric is attribute accuracy, not "aims at option X vs collapses Unknown"). **Implication for the paper:** E8 as a novelty is now **substantially pre-empted** for non-bias attributes — if we pursue trajectory-timing we must cite this heavily and differentiate on the bias/decision setting; our safe, unoccupied ground is the **construction → aim-vs-disinhibit** finding, *not* the trajectory-timing idea and *not* "residual contrastive steering of a DLM."
- **URL / status:** VERIFIED — https://arxiv.org/abs/2605.10971 (arXiv 2605.10971).

### Diffusion Guided Language Modeling (Lovelace et al., ACL Findings 2024)
- **Model(s):** A guided **continuous latent diffusion** model that produces a latent proposal to steer a frozen **auto-regressive** LM (GPT2-class) decoder — a diffusion+AR hybrid, not a masked-diffusion LM.
- **Benchmark(s)/datasets:** Controllable-attribute generation — sentiment and **toxicity** control (classifier-guided attributes); fluency/quality metrics. No BBQ.
- **Method / focus:** Trains a diffusion model with plug-in attribute classifier guidance to generate a soft latent prompt/proposal that conditions AR decoding, giving controllable attributes (sentiment, low-toxicity) without gradient cascade errors.
- **vs. ours:** Diffusion is used only as a *guidance/proposal* module for an **AR** decoder (not a standalone masked DLM like LLaDA); requires **training** a diffusion guidance model (not training-free at inference in our sense); controls sentiment/toxicity, not BBQ social bias; beneficial control. Contrast: latent-space guidance of AR gen vs. our residual-stream hook on a native DLM.
- **URL / status:** VERIFIED — https://aclanthology.org/2024.findings-acl.887/ ; arXiv https://arxiv.org/abs/2408.04220 .

### DiffuDetox: A Mixed Diffusion Model for Text Detoxification (Floto et al., ACL Findings 2023)
- **Model(s):** A text (embedding-space) **diffusion model** trained from scratch — mixed conditional + unconditional diffusion. Not an LLaDA-style masked LLM; not LLM-scale.
- **Benchmark(s)/datasets:** Text detoxification (parallel toxic->neutral rewriting, ParaDetox-style task); human-eval of detox quality/fluency. No BBQ.
- **Method / focus:** Conditional model takes toxic text and emits diverse detoxified rewrites; unconditional model (trained to recover input) injects fluent text to preserve fluency. Achieves human-level detoxification.
- **vs. ours:** A **trained** task-specific diffusion detoxifier (opposite direction: removes toxicity), a rewriting task on a small custom benchmark — no inference-time steering, no LLaDA, no residual-stream intervention, no BBQ, no bias injection. Oldest/most peripheral to our setup; relevant only as early "diffusion for text safety" prior work.
- **URL / status:** VERIFIED — https://aclanthology.org/2023.findings-acl.478/ ; arXiv https://arxiv.org/abs/2306.08505 .
## 2. Technical DLM bias (locality / proximity / sampling / exposure / optimization)

*Scope note.* This bucket covers the word "bias" as it is used inside the diffusion-LM (DLM) methods literature — **locality/proximity bias in denoising, mask-induced distraction, sampler-induced distributional error, training/optimization bias, and quality–diversity trade-offs.** None of these are demographic/social bias. Their relevance to our project is as **confounds**: mechanisms in LLaDA-8B's masked-diffusion decoding that can *mimic, inflate, or suppress* apparent social bias on BBQ independently of what our L14 steering vector actually injects. Our pipeline is training-free and inference-time (frozen weights + forward-hook activation steering + a Ghostwriter prompt-injection baseline), so the decoding-time effects below sit *directly on the causal path* between our intervention and the BBQ abstention/target/directional-gap metrics.

---

### Semantic DLM+ / SemDLM+: Improving Diffusion Language Models through Bias-Variance Trade-off in Transition Kernel Design (Jiang et al., 2026)

- **Model(s):** UNVERIFIED — could not confirm.
- **Benchmark(s)/datasets:** UNVERIFIED — could not confirm.
- **Method / focus:** UNVERIFIED. Could not locate this paper under this title/author/year through multiple query variants (exact title, "SemDLM+"/"Semantic DLM+", Jiang + "bias-variance" + "transition kernel", arXiv 2606 range). From the title alone the intended sense of "bias" is the *statistical* bias–variance decomposition applied to the design of the discrete forward transition kernel Q (e.g., masked vs. uniform/absorbing kernels) — an optimization/estimation bias, not demographic bias. Adjacent confirmed work in this space (D3PM structured transition matrices; sampler-correctness work at 2602.19619 which explicitly discusses transition-kernel choice) exists, but this specific paper does not surface.
- **vs. ours:** *If real*, transition-kernel bias–variance design governs how noise is injected/removed during LLaDA's forward/reverse process; a mis-specified or high-bias kernel could systematically distort which tokens get committed early, which is exactly the kind of decoding-side artifact that could show up as a spurious BBQ answer skew. But because we cannot verify the paper, treat this connection as speculative.
- **URL / status:** **UNVERIFIED — could not confirm.** Not found on arXiv, Google Scholar, or via title/author search (as of 2026-07-02). Recommend the user re-check the exact title/author; possible it is unpublished, renamed, or misattributed.

---

### Early Decisions Matter: Proximity Bias and Initial Trajectory Shaping in Non-Autoregressive Diffusion Language Models (Kim et al., 2026)

- **Model(s):** Diffusion LLMs (dLLMs) in the non-autoregressive masked-diffusion regime generally; the paper frames the phenomenon architecture-agnostically (confidence-based parallel decoders of the LLaDA/Dream family) rather than naming a single model in the abstract.
- **Benchmark(s)/datasets:** Evaluated on "various reasoning and planning tasks"; specific dataset names/metrics not enumerated in the abstract. Method contribution is a lightweight planner + end-of-sequence (EOS) temperature annealing.
- **Method / focus:** "Proximity bias" = the denoising order concentrates on **spatially adjacent tokens** — once a token is unmasked, nearby positions acquire high confidence and get denoised in close temporal proximity, causing local **spatial error propagation**. Under deterministic confidence-based decoding this collapses into an effectively *reverse-autoregressive* process; combined with premature EOS selection, an early EOS anchors the whole trajectory and starves content generation. The generation path is "critically contingent on the initial unmasking position."
- **vs. ours:** **This is the single most dangerous confound for our BBQ measurements.** BBQ items are multiple-choice; the model must commit to an answer token/letter and often an "unknown"/abstention option. If the *first* high-confidence unmask lands on (say) the answer-letter region or the EOS, proximity bias means the surrounding answer tokens are effectively locked in by decoding order, **not** by the content of the residual-stream direction we added at L14. Two concrete risks: (1) our measured abstention rate and target-answer rate could shift simply because steering perturbs *which position* gets unmasked first, changing the trajectory rather than the model's "belief"; (2) our directional-gap metric (Black-vs-other) could be partly an artifact of position-dependent proximity dynamics interacting with where the demographic token sits in the prompt/answer. Mitigation to note: hold decoding schedule/EOS-temperature fixed across steered vs. unsteered runs, and check that directional gaps survive when the answer-letter position is randomized.
- **URL / status:** Working — https://arxiv.org/abs/2604.10567 (HTML: https://arxiv.org/html/2604.10567). Submitted Apr 2026.

---

### Understanding and Accelerating the Training of Masked Diffusion Language Models (Hong et al., 2026)

- **Model(s):** Masked Diffusion Models (MDMs); trained/evaluated at LM1B scale (not LLaDA-8B). Authors from KAIST / Sony AI / Univ. of Tokyo.
- **Benchmark(s)/datasets:** One Billion Word Benchmark (LM1B); metrics = validation negative log-likelihood (NLL), generative perplexity, zero-shot perplexity, downstream task performance.
- **Method / focus:** "Locality bias" here is a **property of language + a training-dynamics/optimization bias**: the predictive information for a token is concentrated in nearby positions, so MDMs learn slowly (they must learn long-range structure the AR objective gets for free). Remedy: **bell-shaped time sampling** during training → same NLL up to ~4× faster.
- **vs. ours:** Indirect for us (we freeze weights, do no training), but it establishes the *mechanistic origin* of the locality bias that the Kim and Piskorz papers exploit at inference. The takeaway for our confound story: LLaDA-8B's learned denoiser inherently over-weights local context, so a demographic token placed *near* the answer slot in a BBQ template may exert outsized influence relative to one placed far away — a template-position sensitivity we should control for even though it stems from training, not our steering.
- **URL / status:** Working — https://arxiv.org/abs/2605.13026 (HTML: https://arxiv.org/html/2605.13026). Submitted May 2026.

---

### Masks Can Be Distracting: On Context Comprehension in Diffusion Language Models (Piskorz et al., 2025)

- **Model(s):** Masked Diffusion Language Models (MDLMs) vs. autoregressive LMs; specific checkpoints not named in the abstract but the study targets the MDLM family (LLaDA/MDLM-style).
- **Benchmark(s)/datasets:** Context-comprehension / position-of-relevant-information probes (à la "needle-in-a-haystack" style position sensitivity); specific dataset names not given in abstract. Contribution: a **mask-agnostic loss** making predictions invariant to the number of appended mask tokens (fine-tuning-time fix).
- **Method / focus:** Two limitations: (1) **locality bias** — despite bidirectional attention, MDLM performance is highly sensitive to *where* the relevant information sits, favoring local over distant context; (2) **mask distraction** — appending the large block of `[MASK]` tokens required for generation acts as a *distractor* that degrades comprehension of the actual context.
- **vs. ours:** **Second-most important confound, and it is specific to how we run BBQ on LLaDA.** To generate an answer, LLaDA appends a block of mask tokens; Piskorz shows those masks measurably degrade how well the model reads the prompt. For BBQ this means: (a) the number/length of the generation mask block is an experimental knob that can change abstention/target rates independent of any social content — if steered vs. baseline runs differ in effective mask block usage, the comparison is contaminated; (b) the model's uptake of the *demographic mention* in the context depends on that mention's position relative to the mask block, so the same social cue can read as "more biased" or "less biased" purely from prompt geometry. Directly relevant to whether the Ghostwriter prompt-injection baseline and the L14 activation steering are being read *equally* under the mask-distraction regime. Mitigation: fix mask-block length and demographic-token position across all conditions; report position-robustness checks.
- **URL / status:** Working — https://arxiv.org/abs/2511.21338 (HTML: https://arxiv.org/html/2511.21338v1; OpenReview: https://openreview.net/forum?id=CdJwNTisx1). Submitted Nov 2025.

---

### Is Your Diffusion Sampler Actually Correct? A Sampler-Centric Evaluation of Discrete Diffusion Language Models (Tang et al., 2026; ICML 2026)

- **Model(s):** Discrete diffusion LMs in general; the *evaluation* uses a synthetic **Hidden Markov Model (HMM) oracle** — the learned denoiser is replaced by an exact HMM posterior from a ground-truth Markov chain, so sampler error can be isolated. (Not a specific LLaDA/Dream checkpoint; a controlled oracle setting.)
- **Benchmark(s)/datasets:** Controlled ground-truth Markov-chain setting; metrics scrutinized = NLL, generative perplexity (GenPPL), MAUVE. Code at https://github.com/LuhanTang/dllm_sampler.
- **Method / focus:** "Bias" here = **sampler-induced distributional error**, distinct from denoiser approximation error. Standard metrics conflate the two. Key results: few-step discrete-diffusion samplers are **not distributionally correct even under a perfect (oracle) denoiser**; transition-level mismatch vanishes only as #steps → sequence length; and improvements in NLL/GenPPL/MAUVE **do not imply** correct sampling.
- **vs. ours:** **The deepest methodological confound.** It says: even if LLaDA-8B's weights encoded some "true" answer distribution over a BBQ item, the *few-step sampler we actually run* draws from a distorted distribution — and that distortion is systematic, not just noise. So an observed directional gap or target-answer preference could be a **sampler artifact** rather than a property of the model or of our L14 injection. Two implications: (1) our absolute BBQ bias numbers are sampler-conditioned and only meaningful *relative to a matched-sampler baseline* (which our design has — same sampler for steered vs. unsteered); (2) if we ever change number of denoising steps between conditions, or compare Ghostwriter (which changes the prompt, hence trajectory) against activation steering, sampler-induced error can differ between conditions and masquerade as a difference in injected bias. Also a caution that our metrics (which are effectively answer-distribution reads) can look "better/worse" without the underlying sampling being correct.
- **URL / status:** Working — https://arxiv.org/abs/2602.19619 (HTML: https://arxiv.org/html/2602.19619; project: https://luhantang.github.io/dllm_sampler/; code: https://github.com/LuhanTang/dllm_sampler). ICML 2026.

---

### Understanding the Quality-Diversity Trade-off in Diffusion Language Models (Buzzard, 2025)

- **Model(s):** Continuous-embedding-space diffusion LMs for sequence-to-sequence tasks (not masked-discrete LLaDA specifically; embedding-space DLMs).
- **Benchmark(s)/datasets:** Seq2seq generation; metrics = BLEU-4, ROUGE-L, BERTScore (reports +5.9 BLEU-4, +4.8 ROUGE-L, +2.2 BERTScore over baseline).
- **Method / focus:** DLMs "lack a natural means to control the quality–diversity trade-off" that AR models get from the **temperature** hyperparameter. Proposes **classifier-free guidance** and **stochastic clamping** as cheap inference-time knobs to move along the quality–diversity frontier. "Bias" sense here = the sampling/exposure bias in how diffusion decoding trades sharpness (quality) against entropy (diversity).
- **vs. ours:** Relevant because our BBQ metrics are **entropy-sensitive**. Abstention ("unknown") vs. committing to a target answer is exactly a quality–diversity/sharpness question: a sharper (low-diversity) decode will commit and can look *more* biased (higher target rate, lower abstention); a more diverse decode hedges toward "unknown" and looks *less* biased. Since LLaDA has no clean temperature and our activation steering *is itself* a guidance-like push on the residual stream, our L14 injection may be partly acting as a quality–diversity/guidance-strength lever — inflating decisiveness — rather than injecting demographic content per se. Confound to rule out: verify that increasing alpha doesn't merely sharpen the answer distribution uniformly (which would raise target rate for *all* groups and could spuriously move the directional gap); compare against a content-neutral guidance-strength control.
- **URL / status:** Working — https://arxiv.org/abs/2503.10683 (HTML: https://arxiv.org/html/2503.10683v1). Submitted Mar 2025.

---

**Cross-cutting confound summary for our BBQ social-bias readout.** All five verified papers point to one theme: in masked-diffusion decoding, *where* and *when* a token is committed (proximity bias, initial-trajectory anchoring, mask-block distraction), and *how* the few-step sampler distorts the answer distribution (sampler-correctness, quality–diversity sharpness), can all move BBQ abstention/target/directional-gap numbers **without any change in the model's underlying social disposition or in what our L14 vector injects.** The safeguard our design must enforce is *strict matching of decoding schedule, step count, mask-block length, and demographic-token position across steered vs. unsteered vs. Ghostwriter conditions*, plus position/answer-letter randomization checks, so that a measured directional bias is attributable to the intervention and not to DLM decoding artifacts.
## 3. Core DLM foundations & surveys

These works establish the diffusion-language-model (DLM) paradigm our project builds *on*. They are background/foundations, **not** baselines we compare against. Critically, none of them study demographic or social bias (e.g., BBQ), which frames the gap our work addresses: training-free, inference-time bias injection into a frozen masked-diffusion LM.

---

### Large Language Diffusion Models (LLaDA) — Nie et al., 2025  [THIS IS OUR MODEL]

- **Model(s):** LLaDA — a **masked (discrete) diffusion** language model at **8B scale**, released as two open-sourced variants: **LLaDA-8B-Base** and **LLaDA-8B-Instruct** (open-sourced Feb 14, 2025). Architecture is a **Transformer** (bidirectional, full attention) used to predict masked tokens — *not* autoregressive, *not* continuous-Gaussian diffusion. It is trained from scratch and is competitive with LLaMA3-8B. Our project uses **LLaDA-8B-Instruct with frozen weights**.
- **Training / how it works:** Standard **pre-training + supervised fine-tuning (SFT)** paradigm, but with a diffusion objective instead of next-token prediction. **Forward process:** tokens are progressively **masked** as the noise level t goes 0→1 (at t=1 the sequence is fully masked). **Reverse process:** a Transformer is trained to **predict the masked tokens**, optimizing a **likelihood lower bound**. This forward-masking / reverse-mask-prediction structure is exactly the mechanism our L14 activation-steering intervention operates within.
- **Generation / sampling:** Iterative denoising from full masking (t=1) to fully unmasked (t=0). At each step the model **predicts all masked positions in parallel**, then applies **flexible remasking** (e.g., low-confidence remasking; semi-autoregressive block decoding) to decide which predictions to keep vs. re-mask for the next step. Generation is bidirectional and parallel rather than left-to-right.
- **Benchmark(s)/datasets:** General knowledge and reasoning — **MMLU, CMMLU, ARC-C, PIQA, GSM8K (math), HumanEval (code)**, plus in-context learning and instruction-following case studies (e.g., multi-turn dialogue). Notably it **addresses the reversal curse**, surpassing GPT-4o on a reversal poem-completion task. **No bias/fairness benchmark (no BBQ, no demographic/social-bias evaluation).**
- **vs. ours:** This is **exactly our model** (LLaDA-8B-Instruct), and confirms the two mechanistic facts our method relies on: the **forward masking process** and the **reverse masked-token prediction** on a frozen Transformer. The LLaDA paper evaluates only capability benchmarks and does **not** study demographic/social bias — establishing the gap we fill by probing and steering bias at inference time on frozen weights.
- **URL / status:** https://arxiv.org/abs/2502.09992 — **VERIFIED** (arXiv:2502.09992). Code: https://github.com/ML-GSAI/LLaDA ; demo: https://ml-gsai.github.io/LLaDA-demo/

---

### A Survey on Diffusion Language Models — 2025

- **Model(s):** Survey (not a single model). Covers the DLM landscape — discrete/masked and continuous diffusion LMs — positioning them as an alternative to the dominant autoregressive (AR) paradigm. Authors: Tianyi Li, Mingda Chen, Bowei Guo, Zhiqiang Shen (VILA-Lab). Submitted Aug 14, 2025 (v3 revised Jun 2026).
- **Method / focus:** Surveys DLM foundations, architectures, training objectives, and inference/decoding. Emphasizes DLM advantages: **parallel token generation via iterative denoising**, reduced inference latency (several-fold speed-ups), **bidirectional context**, and **fine-grained controllability** over generation.
- **Benchmark(s)/datasets:** Aggregates capability/efficiency results reported across the field (comparability to AR models, speed-ups); it is a survey, so no new evaluation. Focus is throughput and general NLP performance — **not** bias/fairness. **No BBQ / social-bias coverage.**
- **vs. ours:** Provides the umbrella framing for why DLMs matter and why controllability is interesting — but treats "control" as content/attribute control, **not** demographic-bias analysis. Confirms bias in DLMs is an under-studied gap.
- **URL / status:** https://arxiv.org/abs/2508.10875 — **VERIFIED** (arXiv:2508.10875). Repo: https://github.com/VILA-Lab/Awesome-DLMs

---

### Discrete Diffusion in Large Language and Multimodal Models: A Survey — Yu, Li & Wang, 2025

- **Model(s):** Survey focused specifically on **discrete diffusion** dLLMs and dMLLMs (multimodal). Authors: Runpeng Yu, Qi Li, Xinchao Wang (NUS). Submitted Jun 16, 2025 (multiple revisions).
- **Method / focus:** Systematizes the **discrete/masked-diffusion** family (the same family as LLaDA). Highlights the **multi-token parallel decoding** paradigm with **full attention** and **denoising-based generation**, enabling parallel generation, **fine-grained output controllability**, and dynamic response-aware perception — with up to ~10x faster inference than AR.
- **Benchmark(s)/datasets:** Survey; catalogs capability and efficiency results across dLLM/dMLLM works. **No dedicated bias/fairness evaluation; no BBQ.**
- **vs. ours:** Most directly describes the model class we intervene on (masked/discrete diffusion). Its emphasis on **controllability** motivates our angle, but it does not treat demographic/social bias — reinforcing our gap.
- **URL / status:** https://arxiv.org/abs/2506.13759 — **VERIFIED** (arXiv:2506.13759). Repo: https://github.com/LiQiiiii/DLLM-Survey

---

### Diffusion-LM Improves Controllable Text Generation — Li et al., NeurIPS 2022

- **Model(s):** **Diffusion-LM** — a **non-autoregressive, continuous (Gaussian) diffusion** language model. Distinct from LLaDA's discrete/masked diffusion: it denoises a sequence of **continuous Gaussian vectors** into word vectors, producing continuous intermediate latents. Authors: Xiang Lisa Li et al.
- **Method / focus:** The continuous, hierarchical latents enable **simple gradient-based, plug-and-play controllable generation** — steering outputs toward target attributes without retraining the base LM. This is the conceptual ancestor of "controllable generation via latent manipulation," analogous in spirit (though not mechanism) to our activation-steering intervention.
- **Benchmark(s)/datasets:** Six fine-grained **controllable-generation** control tasks (semantic content, POS, syntax tree, syntax spans, length, etc.), evaluated on datasets such as **E2E** and **ROCStories**. Control targets are *linguistic/attribute* controls — **not** demographic or social-bias controls; **no BBQ.**
- **vs. ours:** Foundational precedent that diffusion LMs support **controllability via latent/gradient manipulation**. But (a) it is continuous, not masked-diffusion, and (b) its "control" is stylistic/structural, never demographic bias. We adapt the controllability idea to **discrete masked-diffusion activation steering** targeting social bias.
- **URL / status:** https://arxiv.org/abs/2205.14217 — **VERIFIED** (arXiv:2205.14217; NeurIPS 2022, proceedings: https://proceedings.neurips.cc/paper_files/paper/2022/hash/1be5bc25d50895ee656b8c2d9eb89d6a-Abstract-Conference.html)

---

### DiffusionBERT: Improving Generative Masked Language Models with Diffusion Models — He et al., ACL 2023

- **Model(s):** **DiffusionBERT** — a generative **masked LM based on discrete diffusion**, built on **BERT**. It is the closest precursor to LLaDA's masked-diffusion formulation (discrete, mask-based), though at BERT scale rather than 8B. Authors: Zhengfu He, Tianxiang Sun, Qiong Tang, Kuanning Wang, Xuanjing Huang, Xipeng Qiu.
- **Method / focus:** Introduces a **token-informed noise schedule** for the forward (masking) diffusion process — noise added per step depends on each token's information — and studies designs for injecting the **time step** into BERT. Directly bridges masked LMs and discrete diffusion.
- **Benchmark(s)/datasets:** **Unconditional text generation** (perplexity, BLEU; improving over D3PM and Diffusion-LM) and conditional generation (comparable quality, more diverse than baselines). Metrics are generation quality/diversity — **no bias/fairness evaluation, no BBQ.**
- **vs. ours:** Establishes the **discrete masked-diffusion** lineage that LLaDA scales up and that our intervention targets. Confirms the foundational line focused on generation quality and never on demographic/social bias.
- **URL / status:** https://aclanthology.org/2023.acl-long.248/ — **VERIFIED** (ACL 2023, pp. 4521–4534). arXiv preprint: https://arxiv.org/abs/2211.15029 . Code: https://github.com/Hzfinfdu/Diffusion-BERT

---

**Gap summary:** LLaDA (our model) and every survey/foundational DLM paper above evaluate on capability, efficiency, or linguistic-controllability benchmarks — **none evaluate demographic/social bias (no BBQ, no fairness benchmark)**. This foundational literature therefore establishes the paradigm and the controllability intuition we exploit, while leaving inference-time bias behavior in frozen masked-diffusion LMs unstudied.
## 4. Adjacent diffusion fairness (text-to-image; method transfer)

These works all target **diffusion models for images**, not language. We include them because our method (adding `alpha*direction` to the residual stream of a masked-diffusion *language* model, LLaDA-8B, at inference) is conceptually a cousin of their inference-time / embedding-space interventions. The modality gap is large (pixels vs. tokens; U-Net/DiT denoisers vs. a masked-token transformer), and their *goal is opposite to ours* — they **debias / equalize** demographic outputs, whereas we **inject** a controllable directional bias and measure it on BBQ (abstention / target / directional-gap). The value here is **method transfer**: several of them manipulate an internal direction or a guidance signal in a training-free way, which is exactly our operating regime.

### FairGen: Controlling Sensitive Attributes for Fair Generations in Diffusion Models via Adaptive Latent Guidance — Kang et al., EMNLP 2025
- **Model(s):** Text-to-image diffusion. Evaluated primarily on **Stable Diffusion 2** (reports ~68.5% gender-bias reduction on SD2); an IMAGE model, not a language model.
- **Benchmark(s)/datasets:** Introduces **HBE (Holistic Bias Evaluation)** benchmark (diverse domains, complex prompts) plus the **Stable Bias** dataset; measures demographic distribution over sensitive attributes (e.g., gender) against a target fair distribution.
- **Method / focus:** **Adaptive latent guidance** applied *during inference*: a latent-guidance module dynamically nudges the diffusion process to enforce an attribute, and a **memory module** tracks running generation statistics so the guidance steers the output distribution toward a targeted (fair) attribute distribution.
- **vs. ours:** Modality gap = image vs. language. Transferable idea: **inference-time latent guidance with a running memory/statistics controller** — instead of a fixed alpha, adapt the steering strength per-generation to hit a *target distribution*. For us this maps to an adaptive-alpha controller that watches the running BBQ target/directional-gap and adjusts L14 steering strength. Key differences: they *debias* (drive toward parity), we *inject* (drive toward a chosen group); their guidance acts on the diffusion latent of a U-Net, ours edits the residual stream of a transformer LM; they still need a trained/attribute-aware guidance module, we are pure training-free residual addition.
- **URL / status:** VERIFIED — https://arxiv.org/abs/2503.01872 ; ACL Anthology https://aclanthology.org/2025.emnlp-main.1287/ ; OpenReview https://openreview.net/forum?id=PgC5UqKDye

### FairImagen: Post-Processing for Bias Mitigation in Text-to-Image Models — Fu et al., NeurIPS 2025
- **Model(s):** Text-to-image diffusion (**Stable Diffusion** family); model-agnostic post-hoc method operating on prompt/text embeddings. IMAGE model.
- **Benchmark(s)/datasets:** Evaluated across **gender, race, and intersectional** settings; reports fairness (demographic balance) vs. image-quality / prompt-fidelity trade-off.
- **Method / focus:** **FairPCA embedding debiasing** — projects CLIP/text embeddings into a subspace that removes group-specific information via **Fair PCA** while preserving semantics, adds **empirical noise injection** to trade off fairness vs. fidelity, and a **unified cross-demographic projection** to debias several attributes at once. No retraining / no weight changes.
- **vs. ours:** Modality gap = image vs. language, and it edits the *text-encoder embedding* upstream rather than the denoiser's internal activations. Transferable idea: **linear-subspace editing of an embedding to remove/steer a demographic direction** — the FairPCA "find the group subspace, then project" recipe is a principled way to *construct* our steering direction (we currently build L14 directions from CrowS/StereoSet pairs; a Fair-PCA-style group subspace is an alternative/complementary estimator). Key differences: they **project out** (subtract) the group subspace to equalize; we **add** a scaled direction to amplify a chosen group; they edit input-space embeddings (closer to the Ghostwriter baseline than to L14 residual steering); they are post-processing but distribution-equalizing, we are attribute-injecting.
- **URL / status:** VERIFIED — https://arxiv.org/abs/2510.21363 ; NeurIPS 2025 poster https://neurips.cc/virtual/2025/poster/115484 ; code https://github.com/fuzihaofzh/FairImagen

### Stay Fair! Ensuring Group Fairness in Diffusion Models Across Guidance Scales — Kim et al., 2026
- **Model(s):** Text-to-image diffusion using both **classifier guidance** and **classifier-free guidance (CFG)**; IMAGE model. (Kim, Kim, Chae, Mo — POSTECH / Amazon.)
- **Benchmark(s)/datasets:** Group-fairness (demographic parity across groups) evaluated **as a function of the guidance scale**; quality preserved across scales.
- **Method / focus:** **Guidance-scale bias analysis + StayFair.** Decomposes total bias into **model bias** + **guidance bias**, shows guidance bias grows *monotonically with the guidance scale* (dominating the high-guidance regime users prefer). StayFair modifies only the guidance step — equalizing the classifier's output distributions across groups (classifier guidance) or shifting the CFG **null/unconditional embedding by a prompt-dependent offset** — so it is orthogonal to and stackable on any debiased model.
- **vs. ours:** Modality gap = image vs. language, but this is the **most directly analogous knob to our alpha**: their "guidance scale" is a scalar strength on a directional signal, exactly like our steering coefficient alpha on the L14 direction. Their key finding — **bias scales monotonically with guidance strength** — predicts and mirrors our dose-response alpha sweeps ({4,8,12,16,24,32}); their "guidance bias vs. model bias" decomposition is a clean framing for separating LLaDA's *baseline* BBQ bias from the *injected* bias we add. The **null-embedding offset** trick is a transferable, training-free way to bias generation. Difference: they add the offset to *cancel* bias and keep it constant across scales; we deliberately *scale up* the injected direction and study the resulting directional gap.
- **URL / status:** VERIFIED — https://arxiv.org/pdf/2605.28036 (arXiv:2605.28036)

### DiffLens: Dissecting and Mitigating Diffusion Bias via Mechanistic Interpretability — Shi et al., CVPR 2025
- **Model(s):** Text-to-image diffusion models (Stable Diffusion family); IMAGE model.
- **Benchmark(s)/datasets:** Social-bias evaluation over **gender, race, age**; measures demographic distribution / bias level with fine-grained controllability, alongside image-quality preservation.
- **Method / focus:** **Mechanistic interpretability.** Trains a **sparse autoencoder (SAE)** to disentangle the diffusion model's hidden neurons into a sparse semantic feature space, **identifies specific "bias features"** (neurons/directions responsible for gender/age/etc.), then **directly manipulates those features** to neutralize or granularly control bias.
- **vs. ours:** Modality gap = image vs. language, but methodologically this is the **closest to activation steering**: it locates *internal feature directions* and edits activations to move a demographic attribute — precisely what our L14 residual-stream intervention does. Transferable idea: **use an SAE / mechanistic feature-localization to find a cleaner, more monosemantic bias direction** in LLaDA rather than a contrastive-pair mean-difference vector, and to *localize which layer/features* to steer (they show granular per-feature control; we picked L14 empirically). Key differences: they identify features to *turn down* bias; we would use the same localization to *turn up / redirect* it; U-Net neurons vs. LM residual stream; SAE training is an extra (offline) step whereas our current pipeline is a single training-free direction.
- **URL / status:** VERIFIED — https://arxiv.org/abs/2503.20483 ; CVPR 2025 paper https://openaccess.thecvf.com/content/CVPR2025/papers/Shi_Dissecting_and_Mitigating_Diffusion_Bias_via_Mechanistic_Interpretability_CVPR_2025_paper.pdf ; project https://foundation-model-research.github.io/difflens/ ; code https://github.com/foundation-model-research/DiffLens

### Embedding Arithmetic: A Lightweight, Tuning-Free Framework for (Post-hoc) Bias Mitigation in Text-to-Image Models — 2026
- **Model(s):** Text-to-image diffusion — evaluated on **FLUX.1-Dev** and **Stable Diffusion 3.5-Large**; IMAGE models. (Note: verified title reads "…for Post-hoc Bias Mitigation…"; authors Sambandham & Schön, TH Ingolstadt.)
- **Benchmark(s)/datasets:** Social-bias / diversity over demographic attributes; introduces a **Concept Coherence Score (CCS)** to measure semantic preservation beyond standard diversity metrics.
- **Method / focus:** **Embedding arithmetic** — an *inference-time*, training-free correction applied directly in the **conditional embedding space** (no weight/prompt/dataset changes), i.e. add/subtract concept vectors to shift demographics while preserving prompt semantics. Finds the conditional embedding space is an **entangled manifold**, not a clean grid of disentangled concepts (which complicates naive vector arithmetic).
- **vs. ours:** Modality gap = image vs. language, and it edits the *conditional/text embedding* (upstream), closer to the **Ghostwriter baseline** (comparison point, not our method) than to L14 residual steering. Transferable idea: this is the **purest analogue of our core operation** — "add a scaled concept direction at inference, no training." Two directly useful takeaways: (1) their **entangled-manifold** finding warns that a single linear bias direction may not be disentangled from semantics — motivating our coherence checks and the directional-gap metric; (2) **CCS** is a ready-made metric for verifying that our L14 injection changes the demographic answer *without* degrading task semantics/coherence. Key differences: they subtract to *debias*, we scale to *inject*; embedding-space (their method / the Ghostwriter baseline) vs. mid-network residual-stream (our L14); image vs. language.
- **URL / status:** VERIFIED — https://arxiv.org/pdf/2604.18167 (arXiv:2604.18167) ; code https://github.com/cvims/EMBEDDING-ARITHMETIC
## 5. General LLM bias surveys & our benchmark (BBQ)

This section anchors the theoretical framing and evaluation of our work. We use the
Gallegos et al. survey taxonomy to locate our contribution (an *inference-time
intra-processing bias INJECTION*, not a mitigation), reference SAGED as the state of
the art in holistic bias *benchmarking pipelines*, and characterize BBQ, the benchmark
on which all of our attack metrics are computed.

---

### Bias and Fairness in Large Language Models: A Survey — Gallegos et al. (2024)

- **Model(s):** Survey; no single model. Covers the LLM landscape broadly (BERT/RoBERTa-style
  encoders through GPT-family autoregressive LLMs). It formalizes bias/fairness generically
  and is architecture-agnostic — notably it predates and does not cover masked-diffusion LMs
  such as LLaDA, which is exactly the gap our work probes.

- **Benchmark(s)/datasets — the taxonomy that anchors our eval section:**
  The survey organizes the field into **three taxonomies**:

  1. **Bias-evaluation METRICS, by the level at which they operate on the model:**
     - **Embedding-based** — geometry of static/contextual embeddings (e.g., WEAT/SEAT).
     - **Probability-based** — token/sequence likelihoods over masked or pseudo-log-likelihood
       scored pairs (e.g., CrowS-Pairs, StereoSet).
     - **Generated-text-based** — properties of the model's *actual output* (e.g., distributional
       bias, toxicity, and QA-answer bias such as BBQ).
  2. **Bias-evaluation DATASETS, by data structure:**
     - **Counterfactual inputs** — paired sentences with a perturbed social group
       (CrowS-Pairs, StereoSet, WinoBias, Winogender).
     - **Prompts** — phrases that condition open generation (RealToxicityPrompts, BOLD,
       HolisticBias); **BBQ** sits here as a QA/generated-text benchmark.
  3. **Bias-MITIGATION techniques, by intervention stage:**
     **pre-processing** (data/prompt edits) → **in-training** (loss/objective, retraining) →
     **intra-processing** (inference-time changes to a *fixed* model: decoding, weight/activation
     modification, no retraining) → **post-processing** (rewriting/filtering outputs).

  Key datasets it consolidates include **BBQ, CrowS-Pairs, StereoSet, RealToxicityPrompts,
  WinoBias, Winogender, BOLD, HolisticBias**.

- **Method / focus:** A comprehensive, mathematically unified survey of how bias in LLMs is
  *defined, measured, and mitigated*, with a unified notation across metrics and a mapping of
  each dataset to the harms and social groups it targets.

- **vs. ours:** The survey is the map; we are a point on it that its authors did not anticipate.
  - **Stage:** Our L14 activation steering is an **intra-processing intervention** — inference-time,
    training-free, on a frozen LLaDA-8B — but applied *in reverse*: we **INJECT** bias rather than
    mitigate it. The survey's intra-processing bucket is entirely about *reduction*; we invert it
    into a red-team/attack setting.
  - **Baseline:** The Ghostwriter baseline (a comparison point, not our method) is a **pre-processing** manipulation
    (prompt/input edits), giving us one probe per stage-adjacent axis (input vs. internal-activation).
  - **Metric level:** All our measurements are **generated-text-level** (BBQ answer choices),
    the survey's third metric level — not embedding or probability level.
  - **Beyond the standard metrics:** The survey catalogs *mitigation-oriented* fairness metrics.
    We add **absolute "attack" metrics** — abstention-collapse rate, target pick-rate, answer
    flip-rate, and a directional-gap — that quantify *how far a group's selection rate can be
    pushed*, which the survey's bias-*score* framing does not cover.

- **URL / status:** arXiv:2309.00770 — https://arxiv.org/abs/2309.00770 (VERIFIED).
  Published version: *Computational Linguistics* 50(3):1097–1179, 2024 —
  https://aclanthology.org/2024.cl-3.8/ (VERIFIED).

---

### SAGED: A Holistic Bias-Benchmarking Pipeline for Language Models with Customisable Fairness Calibration — Guan et al. (COLING 2025)

- **Model(s):** Pipeline/tool, not tied to one model; demonstrated on contemporary generative LLMs
  (e.g., GPT-family and Mistral-class instruction models). Model-agnostic by design.

- **Benchmark(s)/datasets:** Does not ship a fixed dataset; instead it *builds* benchmarks. Five
  pipeline stages: (1) **scraping** source material → (2) **assembling** benchmarks →
  (3) **generating** model responses → (4) **extracting** numeric features → (5) **diagnosing**
  with disparity metrics. Metrics include **max-disparity (impact ratio)** and **bias-concentration
  (Max Z-scores)**; it adds **counterfactual branching** and **baseline calibration** to remove
  metric-tool bias and contextual prompt bias.

- **Method / focus:** The first *holistic* benchmarking pipeline — it targets the limitations of
  fixed benchmarks (limited scope, data contamination, no fairness baseline) by making benchmark
  construction customisable and calibrating against a fairness baseline before reporting disparity.

- **vs. ours:** SAGED is the modern answer to "how should bias be *measured* rigorously"; we adopt
  its *spirit* (calibrated, disparity-style reporting) but not its pipeline. Our evaluation is
  intentionally anchored to the **fixed, hand-built BBQ** benchmark so results are comparable to the
  large body of BBQ literature. Crucially, SAGED (like the survey) frames everything as *diagnosis*;
  our work is a *causal manipulation* — we intervene on internal activations and *report the induced
  shift* in BBQ answer distributions, using attack metrics rather than SAGED's disparity diagnostics.
  SAGED's baseline-calibration idea does, however, motivate our reporting of a directional-gap
  (steered vs. baseline) rather than raw rates alone.

- **URL / status:** arXiv:2409.11149 — https://arxiv.org/abs/2409.11149 (VERIFIED).
  COLING 2025 main conf — https://aclanthology.org/2025.coling-main.202/ (VERIFIED).
  Code: https://github.com/holistic-ai/SAGED-Bias (VERIFIED).

---

### BBQ: A Hand-Built Bias Benchmark for Question Answering — Parrish et al. (ACL Findings 2022) — OUR BENCHMARK

- **Model(s):** Originally evaluated on **UnifiedQA** (T5-based QA model), plus RoBERTa- and
  DeBERTaV3-family models fine-tuned for multiple-choice QA. Since adopted as a standard LLM bias
  eval (GPT-3/3.5/4, Llama, etc.). We run it on **LLaDA-8B**, a masked-diffusion LM — a class not
  present in the original study.

- **Benchmark(s)/dataset — its own structure (this IS our evaluation):**
  - **Scale:** 58,492 hand-written examples over **11 categories**: the **9 social dimensions** —
    Age, Disability status, Gender identity, Nationality, Physical appearance, Race/ethnicity,
    Religion, Socio-economic status (SES), Sexual orientation — **plus 2 intersectional** categories:
    **Race × SES** and **Race × Gender**. ≥25 templates per category.
  - **Two context conditions per item:**
    - **AMBIGUOUS** — under-informative context; the correct answer is always **UNKNOWN**. Measures
      how strongly the model falls back on stereotype when no evidence is given.
    - **DISAMBIGUATED** — context supplies the answer. Measures whether bias overrides the
      evidence-supported correct answer.
  - **Three answer options per question:** the **target** (the stereotyped group for that item),
    the **non-target** (the other group), and **UNKNOWN** (e.g., "Not enough information"). Each
    question comes in a **negative** and a **non-negative** polarity form.
  - **Bias scores:**
    - **s_DIS** (disambiguated) = 2·(n_biased_answers / n_non-UNKNOWN answers) − 1, ranging in
      [−1, +1]; +1 = answers always align with the social bias, −1 = always against.
    - **s_AMB** (ambiguous) = **s_DIS × (1 − accuracy)** — scales the disambiguated bias by the
      error rate, so a model that correctly answers UNKNOWN in ambiguous contexts shows ~0 bias.
    Positive scores indicate answers reinforce the attested stereotype. The original paper finds
    bias is strongest in **ambiguous** contexts, where models fail to pick UNKNOWN and default to
    stereotypes.

- **Method / focus:** A hand-built multiple-choice QA benchmark that quantifies whether a model's
  answers reflect attested U.S.-context social stereotypes, separately for information-poor
  (ambiguous) and information-rich (disambiguated) settings.

- **vs. ours:** BBQ is **exactly our evaluation harness** — we compute the official **s_AMB / s_DIS**
  on LLaDA-8B under our L14 injection and Ghostwriter baseline. On top of the official scores we add
  **attack-oriented metrics**: **abstention-collapse** (drop in UNKNOWN pick-rate under steering),
  **target pick-rate** (absolute rate at which the stereotyped group is selected), **answer
  flip-rate** (items whose choice changes under injection), and a **directional-gap**
  (target-vs-non-target rate separation induced by the steering vector). These convert BBQ from a
  static *diagnostic* into a *dose-response attack surface*, letting us report how far bias can be
  actively driven at inference time rather than merely observed.

- **URL / status:** arXiv:2110.08193 — https://arxiv.org/abs/2110.08193 (VERIFIED).
  ACL Findings 2022 — https://aclanthology.org/2022.findings-acl.165/ (VERIFIED).

---

**Note on category count:** The abstract phrasing "nine social dimensions" refers to the base social
categories; the released dataset ships **11** by adding the two intersectional categories
(Race × SES, Race × Gender). We report on the 11-category release.
---

## Research gap (what this project occupies)
**Honest scoping (updated).** Inference-time steering of DLMs is *not* empty ground:
**ILRR** (activation/latent-space, reference-transfer), **DLM-SWAI** (logit-space, token
scores), and **Steering Without Breaking** (Zhou et al.; residual-stream SAE contrastive +
adaptive per-step schedule) already steer masked DLMs at inference, training-free, with a
tunable scale. So the gap is **not** "steering a DLM at inference," **not** "activation-space
steering of a DLM," **not** "residual-stream contrastive steering," and — because Zhou et al.
already analyse *when attributes commit across denoising* — **not** the denoising-trajectory
timing idea (our deferred **E8** is largely pre-empted for non-bias attributes). What remains
unoccupied is narrower and is where our contribution must sit. (**Shnaidman et al.** —
"Activation Steering for Masked Diffusion Language Models," arXiv:2512.24143, ReALM-GEN
@ ICLR 2026 — is now **located and VERIFIED**: it is the closest method to us (contrastive
direction + residual-stream global intervention on MDLMs), case study = safety refusal, not
demographic bias. See `baselines.md` §1.)

1. **DLM fairness/safety + steering work** (bucket 1) covers jailbreak-safety,
   toxicity/style, sentiment transfer, and tabular fairness (TrustLDM's EOD) — **none uses
   BBQ**, **none targets demographic stereotyping**, and **none treats it as an inference-time
   _attack_**. The existing DLM steering methods (ILRR, DLM-SWAI) steer *style/sentiment*
   and never ask whether a direction **aims at a specific answer option vs merely suppresses
   abstention** — the aim-vs-disinhibit question is ours alone.
2. **Technical DLM "bias"** (bucket 2) is about decoding/training dynamics
   (locality, proximity, sampler, exposure bias), *not* demographic fairness — but it is a
   **confound we must control** (see below).
3. **Foundational/survey DLM papers** (bucket 3), including **LLaDA itself**, evaluate only
   capability benchmarks and never study social bias.
4. **Diffusion fairness methods** (bucket 4) are **text-to-image** and **debias** rather
   than inject; useful only for *method transfer*.
5. In the **general LLM bias taxonomy** (bucket 5, Gallegos et al.), our work is an
   **intra-processing intervention that is _inverted_** — injection, not mitigation —
   measured at the generated-text level; the Ghostwriter baseline is **pre-processing**.

**So the contribution is** (framed against ILRR / DLM-SWAI, which already do inference-time
DLM steering): the first study of *demographic-bias steering* in a **diffusion** LM
(LLaDA-8B) on **BBQ**, and — the real payload — the finding that **whether a steering
direction _aims_ (drives a specific group's answer) or merely _disinhibits_ (collapses the
"Unknown" abstention) is determined by _how the direction is constructed_** (item-anchored
answer-text CAA aims; group/concept mean-diff only disinhibits), with an input-space
Ghostwriter baseline that isolates disinhibition. Novelty rests on the **bias/BBQ setting +
the construction→aim-vs-disinhibit mechanism**, not on "steering a DLM" or "activation space."

## Confound to control (from bucket 2)
Proximity bias + initial-trajectory anchoring (Kim et al.) and sampler-induced error
(Tang et al.) mean BBQ abstention/target/directional-gap numbers can shift from **decoding
order and few-step sampling alone** — not the model's disposition or our injection. **Design
requirement:** hold decoding schedule, step count, block length, and demographic-token
position **identical** across clean / steered / Ghostwriter conditions, with
answer-position randomization checks.

## Suggested review structure (for the paper)
1. **DLM foundations** — Diffusion-LM, DiffusionBERT, LLaDA, DLM surveys (bucket 3).
2. **Why DLM bias may differ from AR-LLM bias** — any-order generation, bidirectional
   context, iterative denoising, remasking, decoding-order effects (buckets 2–3).
3. **Direct trust/fairness evidence** — TrustLDM as the central current paper (bucket 1).
4. **DLM-specific technical biases** — locality/proximity/guidance/sampling/exposure
   (bucket 2), framed as measurement confounds.
5. **Mitigation methods & transfer** — steering, remasking, latent guidance, mechanistic
   editing; lessons from text-to-image fairness (buckets 1, 4).
6. **Gap** — no comprehensive demographic/social-bias study of LLaDA-style DLMs on
   stereotype/counterfactual benchmarks; this project addresses it.

---
*Compiled by parallel literature-survey agents; 24 of 25 listed papers verified with
working links, 1 (SemDLM+) unverifiable and flagged. Per-paper details in the buckets above.*
