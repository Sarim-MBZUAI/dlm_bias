# Steering baselines

Faithful ports of published **activation-steering** methods to LLaDA-8B-Instruct,
each mirroring its OFFICIAL repo and carrying its **own** config (its own layer +
coefficient — we do NOT force a common layer). Each baseline is a **direction
builder** that emits a `.pt` in this repo's schema (`{direction, layer, raw_norm,
splithalf_cosine, hook_module, hidden_size, …}`, see `bias_steering/build_direction.py`)
and plugs straight into our harness:

```
eval/bbq_eval.py --items <eval set> --direction-path <pt> --layer <L> --alpha <coeff>
```

| baseline | official repo (URL @ commit) | method one-liner | its config (layer, coeff) | LLaDA mapping |
|---|---|---|---|---|
| **CAA** (`caa/`) | [nrimsky/CAA](https://github.com/nrimsky/CAA) @ `5dabbbd` — arXiv:2312.06681 | mean over A/B pairs of (answer-letter activation for matching − non-matching option) | **layer 13** (paper: Llama-2-7B best of 32), **alpha 4** add | contrast built on the SAME held-out Black-referent ambiguous BBQ items as ours; activation captured at the answer-letter token of `chat_prompt+<letter>` (materialized, no masks) |
| **ActAdd** (`actadd/`) | [montemac/activation_additions](https://github.com/montemac/activation_additions) @ `cc3178c` — arXiv:2308.10248 | single contrast prompt pair: act(+) − act(−) at a layer | **layer 6** (GPT-2-XL 6/48 → early-mid), **alpha 5** add | single stereotype−anti pair; per-position diff reduced to one `(H,)` mean over aligned tokens |
| **group mean-diff** (`group_meandiff/`) | [nyu-mll/crows-pairs](https://github.com/nyu-mll/crows-pairs) + [StereoSet](https://github.com/moinnadeem/StereoSet) | mean over group minimal-pairs of (stereotype emb − anti-stereotype emb) | **layer 14**, **alpha 4** add | REUSES `bias_steering/build_direction.py --source crows` (not reimplemented) |

## Each baseline uses its OWN layer/coeff — OURS does not
The three baselines above are **single-layer, open-loop `add`** at their own layer.
**Our method is different**: full-layer **closed-loop** steering (`clamp` = proportional,
`cmom` = leaky-integral EMA) applied at **every** block (`--layers all`), see
`directional_steering/`, `eval/bbq_eval.py --steer-mode {clamp,cmom} --layers all`.
The baselines are the point of comparison, not our configuration.

## Config provenance — what is from the paper vs an adaptation (flag for user)
- **CAA layer 13** — from the paper (Llama-2-7B best layer); reused on LLaDA (also 32 blocks) by direct index → mild adaptation.
- **CAA alpha 4** — **adaptation.** CAA's canonical multiplier (±1/±2) acts on a per-layer-normalized vector; our harness injects `alpha*raw_direction`. 4 matches this repo's E1 regime. (`normalize_direction=true` + `alpha=multiplier*avg_act_norm` mirrors CAA exactly.)
- **ActAdd layer 6** — **adaptation.** GPT-2-XL uses 6/48; LLaDA has 32 blocks, so this is an early-mid block, not a LLaDA paper value.
- **ActAdd alpha 5** — ActAdd's canonical love/hate coeff (~5), applied to the raw single-pair diff; scale adaptation for LLaDA.
- **group mean-diff layer 14 / alpha 4** — this repo's own race-steering regime.

Every layer/coeff lives in each folder's `config.json` and is trivially changed.

## Run everything
```bash
bash baselines/run_baselines.sh      # builds all 3 directions + evals on the E1 set
```
GPUs: ONLY 5/6/7. Env: `/home/lukas/miniconda3/envs/sarim_awm/bin/python`.
Eval item set: `experiments/data/black_referent_ambig_eval.jsonl` (the E1 set, so
baselines and ours are scored on identical items). Results land in `baselines/results/`.
