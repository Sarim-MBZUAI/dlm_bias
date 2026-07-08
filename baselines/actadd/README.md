# ActAdd — Activation Addition (steering baseline)

**Official repo:** https://github.com/montemac/activation_additions @ `cc3178cb813b640cd9644cf656d43a51e28869bd`
**Paper:** Turner, Thiergart, Leech, Udell, Vazquez, Mini, MacDiarmid — *Steering GPT-2-XL by adding an activation vector* ([arXiv:2308.10248](https://arxiv.org/abs/2308.10248)).

## The official method (faithfully mirrored)
```
direction = act(+prompt) - act(-prompt)     # ONE contrast prompt pair, a SINGLE diff
apply: hidden += coeff * direction           # at act_name's layer/position
```
- Canonical GPT-2-XL example: `ActivationAddition(prompt="Love", coeff=5, act_name="blocks.6.hook_resid_pre")` minus the same for `"Hate"`.
- The two prompts are padded to **equal token length**; the difference is taken position-by-position.
- Layer = `blocks.6.hook_resid_pre` (**layer 6 of 48**); coeff examples `2.5`, `5`, `15`.

## How it maps to LLaDA-8B (masked-diffusion LM)
| ActAdd (official) | Here (LLaDA-8B) |
|---|---|
| hand-picked `+`/`-` prompt pair (love/hate) | single **stereotype vs anti-stereotype** pair (default `"Black people are dangerous criminals…"` − `"White people are dangerous criminals…"`), overridable via `--pos/--neg` |
| per-position difference added at aligned positions | reduced to **one `(H,)` vector = mean of the equal-length per-position difference** (our hook adds a single vector at all positions) |
| `blocks.6.hook_resid_pre` (layer 6/48) | **block 6** of LLaDA's 32 (early-mid) — *adaptation* |
| coeff ~5 | `alpha=5` on the raw difference vector — *adaptation for LLaDA scale* |

## Config (this baseline's OWN layer + coeff)
- **layer = 6** — **ADAPTATION, flag for the user.** GPT-2-XL's canonical example is layer 6/48 (~front third); LLaDA has 32 blocks, so this is an early-mid block, not a paper value for LLaDA. Tune via `config.json`.
- **alpha = 5**, add mode, raw vector — ActAdd's canonical love/hate coefficient (`~5`); applied to the raw single-pair diff. **Adaptation for LLaDA scale, flag for the user.**

## Build + run
```bash
# build (GPU; USER runs). Writes actadd_direction.pt in the repo .pt schema.
python baselines/actadd/build_actadd.py --layer 6 --device cuda
# CPU-only schema sanity (no model):
python baselines/actadd/build_actadd.py --self-test
# eval on the E1 item set:
python eval/bbq_eval.py --items experiments/data/black_referent_ambig_eval.jsonl \
    --direction-path baselines/actadd/actadd_direction.pt --layer 6 --alpha 5 \
    --out baselines/results/bbq_actadd.json
```
