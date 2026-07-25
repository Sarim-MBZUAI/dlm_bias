# Position-balanced evaluation (the rigorous arbiter)

The raw BBQ pick-rates are confounded by (1) the model's letter-position preference,
(2) which letter the **Black** option sits on, and (3) the **Unknown/correct** option
sitting at "C" most often (152/400). Under strong steering the model also **collapses
onto the letter "A"**, which fakes a high "black-pick" whenever Black happens to be "A".

This eval removes all of that: each of the 400 items is evaluated under **3 cyclic
rotations** of its options, so the Black option (and Unknown, and the non-Black person)
sits at A, B, C **exactly equally**. 1200 evals per condition.

## Harness is proven correct (oracle_test.py, offline)

Four offline checks establish that the harness is balanced and that the metric is a real
letter-jam null — no model is involved, so these are exact:

| oracle / check | expected if correct | result |
|---|---|---|
| option balance (Black at A/B/C) | 400 / 400 / 400 | **400 / 400 / 400** |
| option balance (Unknown at A/B/C) | 400 / 400 / 400 | **400 / 400 / 400** |
| pick-Black oracle → black rate | 1.000 | **1.000** |
| pick-Unknown oracle → abstain rate | 1.000 | **1.000** |
| always-"A" oracle → black rate | 0.333 (pure letter-jammer = chance) | **0.3333** |
| rotation 0 vs original items | identical | **rot0 == original** |

So **0.333 is the letter-jam null**, and the letter-immune signal is **black − nonblack**
(a jam raises both equally → the gap cancels it; a pure jammer has gap = 0).

## Results (1200 evals/condition; base=clean, layer-PI Kp1/Ki.05 α2, decode-PI Kp3/Ki.1, normal-α4)

`gap = black − nonblack`, both position-balanced; **positive = genuinely prefers the Black
person**, a pure letter-jammer scores gap ≈ 0. The last column splits the gap by the letter
Black sits on (A/B/C); a real preference stays positive across positions, a letter-jam does not.

| condition | black | nonblack | **gap (b−nb)** | gap @A / @B / @C |
|---|---:|---:|---:|---|
| base | 0.128 | 0.110 | +0.018 | −0.06 / −0.02 / +0.13 |
| layer-space PI | 0.233 | 0.202 | +0.031 | −0.22 / +0.06 / +0.25 |
| **decode-space PI** | **0.341** | **0.141** | **+0.200** | +0.49 / −0.06 / +0.17 |
| normal α4 (open-loop) | 0.356 | 0.308 | +0.047 | +0.34 / +0.14 / −0.34 |

## Interpretation (no spin)
- **Decode-space PI is the only method that genuinely picks the Black *person* over the
  other person** (0.341 vs 0.141, gap +0.200, letter-position-immune). Layer-space PI
  (+0.031) and normal-α4 (+0.047) pick both people **≈ equally** — they mostly suppress
  "Unknown" (disinhibit) rather than *choose*.
- The **letter-jam is real** (decode-PI emits "A" 535/1200; normal-α4 618/1200) but it
  **cannot explain** the black-vs-nonblack asymmetry — a jam makes those equal.
- **Honest caveat on magnitude:** decode-PI's gap is concentrated where Black sits at "A"
  (+0.49); at "B" it is slightly negative. If you discount position A entirely (where
  correct steering and residual A-jamming coincide), its genuine edge shrinks to ~**+0.05**,
  comparable to the others. So decode-PI *aims the most*, but the residual A-jam leaves the
  magnitude between **+0.05 (strict)** and **+0.20 (pooled)**. Not a clean knockout.

## Limitations
- 400 items × 3 rotations (rotations are not independent); **single run, no confidence
  intervals**. Seed 42; temperature-0 (deterministic).
- Supersedes the raw `../COMPARISON.md` ranking, which is position-confounded.

## Reproduce
```bash
PY=/home/lukas/miniconda3/envs/sarim_awm/bin/python
$PY eval/balanced/make_rotations.py                  # -> _sweep400_rot0/1/2.jsonl
$PY eval/balanced/oracle_test.py                     # correctness proof (offline)
# per condition, per rotation r in 0/1/2 (GPUs 4-7):
CUDA_VISIBLE_DEVICES=6 $PY steering/denoise_pid.py --cond PI \
    --items results/balanced/_sweep400_rot${r}.jsonl \
    --out-dir results/balanced/results_balanced/rot${r} --tag dpid_PI
```
