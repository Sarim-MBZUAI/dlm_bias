# Group mean-difference (steering baseline)

**Implementation:** REUSES the existing `bias_steering/build_direction.py --source crows`
(**not** reimplemented). This folder is only a thin wrapper + config.

**Sources:**
- CrowS-Pairs — https://github.com/nyu-mll/crows-pairs (`data/crows_pairs_anonymized.csv`, the exact URL `build_direction.py` downloads). Nangia et al., EMNLP 2020.
- StereoSet — https://github.com/moinnadeem/StereoSet (Nadeem et al., ACL 2021) — comparable group-contrast minimal pairs; `build_direction.py` is source-agnostic over stereotype/anti-stereotype pairs.

## Method (as implemented in build_direction.py)
```
direction = mean_over_group_pairs( emb(stereotype) - emb(anti_stereotype) )
```
Per-sentence masked-mean of the block-L (or `emb`) activation, differenced over each
group minimal pair, averaged over the category. Orientation follows the official
CrowS `metric.py` (antistereo rows are swapped). See `build_direction.py:load_crows`
and `run_crows`.

## Config (this baseline's OWN layer + coeff)
- **layer = 14** — this repo's directional-steering regime (`directional_steering/`, `race_steering/` built/evaluated race directions at block L14).
- **alpha = 4**, add mode — low end of this repo's race sweep `{4,8,12,…}`, matched to CAA/ActAdd/ours for comparison.
- **category = `race-color`** — the CrowS bias_type; sanitized to `race_color.pt`.

These are this baseline's choices (not our method's closed-loop full-layer setup).

## Build + run
```bash
# build (GPU; USER runs) — wraps build_direction.py, writes
# bias_steering/directions/L14/race_color.pt
bash baselines/group_meandiff/build.sh          # optional arg: GPU id (5/6/7), default 5
# eval on the E1 item set:
python eval/bbq_eval.py --items experiments/data/black_referent_ambig_eval.jsonl \
    --direction-path bias_steering/directions/L14/race_color.pt --layer 14 --alpha 4 \
    --out baselines/results/bbq_group_meandiff.json
```
> There is no separate `.pt` schema sanity here: `build_direction.py` already writes
> the canonical schema this repo consumes.
