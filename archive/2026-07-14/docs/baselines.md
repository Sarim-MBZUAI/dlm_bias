# Baselines & closest related methods

Short, factual summaries of the methods we position against. All facts below are
from the papers' own text (read directly). "Relation to us" = one line on how each
differs from our work (residual-stream activation steering that injects **demographic**
bias into LLaDA-8B on **BBQ**, with the finding that *how the direction is built*
decides whether it **aims** at a group vs merely **disinhibits** abstention).

Our one implemented baseline is **Ghostwriter** (paper 3), reimplemented in `baseline/`.
The rest are method neighbors, not run in our repo.

---

## 1. Activation Steering for Masked Diffusion Language Models
Shnaidman, Feiglin, Yaari, Mentel, LeVi, Lapid (Deepkeep / Technion). ReALM-GEN
workshop, ICLR 2026. [arXiv:2512.24143](https://arxiv.org/abs/2512.24143).

Extracts a single low-dimensional direction from **contrastive prompt sets** with one
prompt-only forward pass, then applies a **global intervention on residual-stream
activations** throughout reverse diffusion — no optimization, no change to sampling.
Case study is **safety refusal** (not demographic bias): refusal in several MDLMs sits
on an ~1-D activation subspace, and the direction beats prompt- and optimization-based
baselines. Diffusion-specific findings: directions are also extractable from
pre-instruction tokens (unlike AR causal attention); max leverage is at **early
denoising steps and mid-to-late layers**; directions transfer EN↔ZH but not to AR.

**Relation to us:** the **closest method** — same primitive (contrastive direction +
residual-stream global intervention on an MDLM, one prompt-only pass). Differs on
target (safety refusal, not demographic bias/BBQ) and never asks the aim-vs-disinhibit
question. Its early-step / mid-late-layer localization overlaps our deferred E8.

## 2. DLM-SWAI: Steering Diffusion Language Models Before They Unmask
An & Han (Yonsei). [arXiv:2605.29626](https://arxiv.org/abs/2605.29626).

Training-free inference-time steering that adds **pre-computed token-level style scores
to the logits** at each masked position during denoising — i.e. **logit-space**
steering, explicitly *not* residual-stream (they tested activation steering as a
baseline and found it weaker for style). Models: LLaDA-8B-Instruct, Dream-v0-Instruct-7B.
Tasks: writing level (OSE), politeness (WikiPol), toxicity (RealTox).

**Relation to us:** opposite channel (logit-space vs our residual-stream activation
hook); style/safety, not demographic bias; beneficial control, not injection.

## 3. Steering LLM Viewpoints through Fabricated Evidence Injection ("Ghostwriter")
Yang, Liu, Huang, Li, Zhang, Weng, Song (HKUST / USTC / Liverpool / Guangzhou).
[arXiv:2606.06244](https://arxiv.org/abs/2606.06244). **This is the source of our `baseline/` implementation.**

A two-phase **input-space** attack: (1) repackage a misleading statement with a
fabricated, credibility-laden rationale; (2) instruct the LLM to incorporate that
viewpoint when answering. Pure prompt transform — no weights/activations touched.
Evaluated on **BBQ, ToxiGen, and a custom dataset**, against commercial LLMs (incl.
classifier-guarded frontier models, which reduce but do not eliminate the attack).

**Relation to us:** our contrast baseline for input-space vs activation-space injection.
We reimplement it (hand-crafted per-BBQ-category fabricated evidence, single injection)
and port it to LLaDA-8B. On BBQ it collapses abstention (0.791→0.589) but the freed
picks split ~evenly, so it **suppresses abstention rather than aiming** — the same
finding we make for group-contrast steering.

## 4. ILRR: Iterative Latent Representation Refinement
Avrahami & Nachmani (Tel Aviv U / Ben Gurion U). [arXiv:2601.21647](https://arxiv.org/abs/2601.21647).

Learning-free inference-time steering that uses a **single reference sequence**: at each
denoising step it runs one extra parallel forward pass and **aligns the generated
sequence's internal activations toward the reference's activations** (a tunable steering
scale). *Spatially Modulated Steering* lets a short reference steer longer text. Models:
LLaDA, MDLM. Attribute: mainly **sentiment**.

**Relation to us:** same **channel** (activation/latent space), but the signal is a
reference example's activations realigned online — not a fixed precomputed contrastive
direction. Sentiment, not demographic bias/BBQ; no aim-vs-disinhibit.

## 5. Steering Without Breaking: Mechanistically Informed Interventions for Discrete DLMs
Zhou, Roy, Gangadharaiah (AWS AI Labs). [arXiv:2605.10971](https://arxiv.org/abs/2605.10971).

Trains **sparse autoencoders** on residual-stream activations of four DLMs (124M–8B:
LLaDA, Dream, MDLM) to find **when each attribute "commits"** during denoising (topic
~2%, sentiment ~20%). Shows uniform per-step intervention degrades quality, then proposes
an **adaptive scheduler** applying **residual-stream contrastive** (target vs non-target
corpus) steering only on the steps where the attribute is forming. Tasks: sentiment /
topic / style (up to 3 jointly).

**Relation to us:** closest on two axes — residual-stream contrastive steering (≈ our
group mean-diff) and the "when in denoising to intervene" analysis (our deferred E8).
Style/sentiment, never BBQ/demographic bias, never aim-vs-disinhibit.
