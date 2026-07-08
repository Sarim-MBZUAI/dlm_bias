# CAA — Contrastive Activation Addition (steering baseline)

**Official repo:** https://github.com/nrimsky/CAA @ `5dabbbd9a0bca5f25e174501e959de378806aa48`
**Paper:** Panickssery, Gabrieli, Schulz, Tong, Hubinger, Turner — *Steering Llama 2 via Contrastive Activation Addition*, ACL 2024 ([arXiv:2312.06681](https://arxiv.org/abs/2312.06681)).

## The official method (faithfully mirrored)
```
for each contrastive A/B item:
    pos = activation at the ANSWER-LETTER token for the matching-behavior option
    neg = activation at the ANSWER-LETTER token for the not-matching option
steering_vector = mean_over_pairs(pos - neg)          # (generate_vectors.py)
                = (all_pos_layer - all_neg_layer).mean(dim=0)
apply: hidden += multiplier * steering_vector          # all post-prompt positions
```
- Official token position: `activations[0, -2, :]` (the answer-letter token).
- Best layer for **Llama-2-7B-chat = layer 13** (of 32); multipliers swept in `{-2,-1,+1,+2}`.

## How it maps to LLaDA-8B (masked-diffusion LM)
LLaDA is **not** autoregressive, so there is no "next-token" answer slot. Adaptations (all in `config.json`):

| CAA (official) | Here (LLaDA-8B) |
|---|---|
| behavior A/B datasets | **same held-out Black-referent ambiguous BBQ items our method builds on** (`directional_steering/build_anchored.py:select_heldout`, disjoint from the eval set) — CAA and ours use identical data |
| pos/neg = matching / non-matching letter | pos = the **Black option letter**, neg = the **other named (non-Black, non-Unknown) letter** |
| activation `[0,-2,:]` (answer-letter token) | materialize `chat_prompt + <letter>` (no mask tokens) and capture the block-L output at the **answer-letter token** (last token) — LLaDA's bidirectional analogue |
| layer 13 of 32 | **block 13** by direct index (LLaDA-8B is also 32 blocks) |
| multiplier ~1–2 on per-layer-normalized vectors | `alpha=4` on the **raw** mean-diff vector in add mode — see below |

## Config (this baseline's OWN layer + coeff)
- **layer = 13** — from the CAA paper (Llama-2-7B best layer). *Adaptation:* applied to a different model family (LLaDA-8B), kept by direct index.
- **alpha = 4**, add mode, raw vector — **ADAPTATION, flag for the user.** CAA's canonical multiplier (`±1/±2`) acts on a *per-layer-normalized* vector; our harness injects `alpha*raw_direction`, so alpha ≠ CAA multiplier. `alpha=4` matches this repo's answer-position steering regime (E1 ran the analogous anchored direction at alpha 4/8). To mirror CAA's normalized form exactly, set `normalize_direction=true` and `alpha = multiplier * avg_act_norm` (`avg_act_norm` is stored in `caa_direction.pt`).

## Build + run
```bash
# build (GPU; USER runs). Writes caa_direction.pt in the repo .pt schema.
python baselines/caa/build_caa.py --layer 13 --device cuda
# CPU-only schema sanity (no model):
python baselines/caa/build_caa.py --self-test
# eval on the E1 item set:
python eval/bbq_eval.py --items experiments/data/black_referent_ambig_eval.jsonl \
    --direction-path baselines/caa/caa_direction.pt --layer 13 --alpha 4 \
    --out baselines/results/bbq_caa.json
```
