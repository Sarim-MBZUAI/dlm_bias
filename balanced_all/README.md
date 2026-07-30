# balanced_all/ — position-balanced eval for EVERY headline condition

`eval/balanced/` proved the position-balanced harness (3 cyclic rotations, every
option visits A/B/C exactly once; pooled 1200 evals; `gap = target − nontarget`
is letter-jam-immune — an always-"A" oracle scores exactly 1/3) but only ran it
for the Black `_sweep400.jsonl` steering conditions. This folder generalizes the
tooling and queues the missing conditions through it:

* **LLaDA Black baselines** (7 conditions, never position-balanced before)
* **decode-PID (Black)**
* **multirace arab / white** (base, decode-PI, normal at the fair coherent dose)

## Tools

| file | what |
|---|---|
| `make_rotations.py` | `--items X.jsonl --out-dir D [--stem S]` → `S_rot{0,1,2}.jsonl`. Reuses `rotate_row` from `eval/balanced/make_rotations.py` by import (single source of truth). |
| `oracle_test.py` | offline proof for any rotation set: `--items-glob 'D/S_rot*.jsonl' --target T --orig X.jsonl`. Balance (target & Unknown at A/B/C = n/3 each), pick-target → 1.0, pick-unknown → abstain 1.0, always-"A" → exactly 1/3, rot0 == original. Uses `multirace/targets.py`. Must PASS before any GPU run. |
| `aggregate.py` | `--cond-glob 'D/cond_X_rot… /cond_X.json' [--label L]` pools one condition's 3 rotation JSONs (either counts schema: `black/…` or `target/…`), prints pooled rates over 1200, balanced gap, per-position gap @A/@B/@C (recomputed from the `_samples.jsonl` files), and a markdown row. `--selftest` offline. |
| `run_balanced_gpu2.sh` | Black baselines queue (CUDA 2). |
| `run_balanced_gpu4.sh` | decode-PID Black + arab/white queue (CUDA 4; generates + oracle-tests the arab/white rotations first). |

## Run matrix (all 400 items × 3 rotations = 1200 evals per condition)

| GPU | target | condition | exact flags (verified against results/BASELINES.md + each CLI) | out dir (`results/balanced_all/…`) | result stem |
|---|---|---|---|---|---|
| 2 | black | CAA | `baselines/caa.py --run --alpha 16 --layer 14` (~paper mult 2) | `black/caa/rot{r}` | `cond_caa_L14_a16` |
| 2 | black | ActAdd | `baselines/actadd.py --run --alpha 16 --layer 14` (~paper mult 2) | `black/actadd/rot{r}` | `cond_actadd_a16` |
| 2 | black | Mean-AcT | `baselines/meanact.py --run --direction unit --strength 2` | `black/meanact/rot{r}` | `cond_meanact_unit_s2` |
| 2 | black | Linear-AcT | `baselines/linearact.py --run --variant gaussian --strength 1` | `black/linearact/rot{r}` | `cond_gaussian_s1` |
| 2 | black | AURA inject | `baselines/aura.py --run --mode inject --gamma 4` | `black/aura_inject/rot{r}` | `cond_inject_g4` |
| 2 | black | AURA vanilla | `baselines/aura.py --run --mode vanilla` | `black/aura_vanilla/rot{r}` | `cond_vanilla` |
| 2 | black | ITI-C | `baselines/itic.py --run --topk 48 --alpha 8` | `black/itic/rot{r}` | `cond_itic_K48_a8` |
| 4 | black | decode-PID | `steering/denoise_pid.py --cond PID --tag dpid_PID` | `black/decode_pid/rot{r}` | `cond_dpid_PID` |
| 4 | arab/white | base | `multirace/denoise_pid.py --target T --cond base` | `T/base/rot{r}` | `cond_dpid_T_base` |
| 4 | arab/white | decode-PI | `multirace/denoise_pid.py --target T --cond PI` | `T/decode_pid/rot{r}` | `cond_dpid_T_PI` |
| 4 | arab | normal α4 | `multirace/normal.py --target arab --alpha 4` | `arab/normal/rot{r}` | `cond_normal_arab_a4` |
| 4 | white | normal α1 | `multirace/normal.py --target white --alpha 1` (α4 = 98.8% unparseable) | `white/normal/rot{r}` | `cond_normal_white_a1` |

Black rotations already exist (`results/balanced/_sweep400_rot{0,1,2}.jsonl`) and
are REUSED. Arab/white rotations are generated (deterministically) into
`results/balanced_all/rotations/` and oracle-tested by the gpu4 script before any
GPU work. Fit caches in `baselines/cache/` are reused — no refits.

Two operating points in the published table were mislabeled in the task notes and
were pinned to the configs that actually produced the published rates:
Mean-AcT row (0.212/0.170) = `cond_s2.json` → `--direction unit --strength 2`
(the `raw_s*` runs do not match the table despite the "(raw)" label);
Linear-AcT gaussian row (0.280/0.200) = `gaussian_s1` → `--strength 1`;
ITI-C row (0.117/0.100) = `topk48_a8` → `--alpha 8`.

## Aggregate

```bash
PY=/home/lukas/miniconda3/envs/sarim_awm/bin/python
$PY balanced_all/aggregate.py \
    --cond-glob 'results/balanced_all/black/caa/rot*/cond_caa_L14_a16.json' \
    --label 'CAA (L14, a16 ~ mult 2)'
# one call per condition; paste the printed markdown rows into a RESULTS table.
```

Raw result JSONs under `results/balanced_all/` stay local (gitignored); only
summary markdown gets committed.
