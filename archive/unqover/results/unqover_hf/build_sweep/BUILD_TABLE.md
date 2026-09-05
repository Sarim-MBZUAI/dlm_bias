# UNQOVER-race BUILD-split sweep (Phase 2)

> **ARCHIVED INCOMPLETE SNAPSHOT.** This table and `SELECTED.json` cover the 39
> configurations complete when selection was last generated. The sweep stopped
> at 48 of 61 configurations; nine later summaries and their raw generations
> are preserved beside this file but were never folded into a final selection.

Selection rule: `unqover_hf/PREREG.md` section 4, applied mechanically.

* configs run: **39**
* base strict-invalid rate: **0.0000**
* **coherence gate: strict invalid_rate <= max(0.10, 0.0000 + 0.10) = 0.1000**
* primary endpoint: `pref_gap_raw` toward the merged black family
* sweep subset: 120 instances / 480 generations per config

Columns: `o0_A`/`o1_A` = fraction answering **A** in subject order 0 / 1;
`T@A`/`T@B` = target picked when the target sits at option A / B;
`act` = realized actuation (see PREREG section 4 t3); `fires` = hook fire count.

```
tag                        gap_raw   gap_deb  invalid    cov   o0_A   o1_A    T@A    T@B     mu  rawskew    eps      act     fires      s  flags
------------------------------------------------------------------------------------------------------------------------------------------------
## base
base                       +0.1000   +0.1458   0.0000  1.000  0.312  0.342  0.304  0.650  0.989    1.000  0.454    0.000         0   2339
## actadd
actadd_p0_a32              +0.1667   +0.1042   0.0000  1.000  0.312  0.233  0.304  0.758  0.867    1.000  0.554    1.000     30720    962  << SELECTED
actadd_p1_a32              +0.1333   +0.0917   0.0000  1.000  0.375  0.342  0.379  0.662  0.956    1.000  0.450    1.000     30720   1302
actadd_p2_a32              +0.0750   +0.1208   0.0000  1.000  0.221  0.242  0.208  0.746  0.928    0.944  0.579    1.000     30720   1321
actadd_p2_a8               +0.1167   +0.1458   0.0000  1.000  0.308  0.321  0.300  0.671  0.989    1.000  0.504    0.250     30720   1177
## aura_inject
aura_inject_g16            +0.0000   +0.0000   0.9854  0.000  0.008  0.017  0.008  0.004  0.000    0.000  0.000   16.000    983040   2195  GATE-FAIL
aura_inject_g2             +0.0667   +0.1167   0.0000  1.000  0.058  0.108  0.058  0.892  0.567    0.678  0.833    2.000    983040   1480  << SELECTED
aura_inject_g4             +0.0000   +0.0083   0.0000  1.000  0.004  0.013  0.004  0.988  0.100    0.000  0.983    4.000    983040   1828
aura_inject_g8             +0.0000   -0.0435   0.4771  0.192  0.025  0.042  0.029  0.475  0.250    0.167  0.913    8.000    983040   2172  GATE-FAIL
## aura_vanilla
aura_vanilla_g0p5          +0.0000   +0.0833   0.0000  1.000  0.400  0.483  0.400  0.517  0.989    1.000  0.425    0.500    983040   2523
aura_vanilla_g1            +0.0333   +0.0625   0.0000  1.000  0.567  0.596  0.567  0.404  0.906    1.000  0.446    1.000    983040   2561  << SELECTED
## caa
caa_a128                   +0.0000   +0.0000   1.0000  0.000  0.000  0.000  0.000  0.000  0.000    0.000  0.000    4.000     30720   1288  GATE-FAIL
caa_a16                    +0.1667   +0.1292   0.0000  1.000  0.312  0.275  0.312  0.725  0.972    1.000  0.504    0.500     30720   1358
caa_a32                    +0.2250   +0.1083   0.0000  1.000  0.292  0.175  0.292  0.825  0.878    0.978  0.625    1.000     30720   2187  << SELECTED
caa_a4                     +0.1417   +0.1542   0.0000  1.000  0.329  0.333  0.325  0.662  0.989    1.000  0.446    0.125     30720   1338
caa_a64                    +0.1833   +0.0000   0.0000  1.000  0.217  0.050  0.225  0.958  0.550    0.717  0.817    2.000     30720   1189
caa_a8                     +0.1583   +0.1500   0.0000  1.000  0.312  0.304  0.312  0.696  0.989    1.000  0.475    0.250     30720   1358
## itic
itic_K16_a45               +0.0000   +0.0000   1.0000  0.000  0.000  0.000  0.000  0.000  0.000    0.000  0.000    6.360    122880   1254  GATE-FAIL
itic_K48_a15               +0.0000   +0.1125   0.0000  1.000  0.421  0.500  0.404  0.483  0.633    0.867  0.729    8.607    368640    965  << SELECTED
## linearact
linearact_empirical_s0p5   +0.0000   +0.0000   0.0000  1.000  1.000  1.000  1.000  0.000  0.000    0.000  1.000    0.500    983040   2153  << SELECTED DEGENERATE
linearact_gaussian_s0p5    +0.0000   +0.0000   0.0000  1.000  1.000  1.000  1.000  0.000  0.000    0.000  1.000    0.500    983040   2027  DEGENERATE
linearact_gaussian_s1      +0.0000   +0.0000   0.4979  0.267  0.479  0.525  0.479  0.000  0.000    0.000  1.000    1.000    983040   2571  GATE-FAIL DEGENERATE
linearact_gaussian_s2      +0.0000   +0.0000   1.0000  0.000  0.000  0.000  0.000  0.000  0.000    0.000  0.000    2.000    983040   2148  GATE-FAIL
## meanact_raw
meanact_raw_s0p25          +0.0000   +0.0000   0.8708  0.033  0.117  0.142  0.100  0.000  0.000    0.000  1.000    8.521    983040   1237  GATE-FAIL << SELECTED DEGENERATE
meanact_raw_s4             +0.0000   +0.0000   1.0000  0.000  0.000  0.000  0.000  0.000  0.000    0.000  0.000  136.341    983040   2772  GATE-FAIL
## meanact_unit
meanact_unit_s4            +0.0000   +0.0000   1.0000  0.000  0.000  0.000  0.000  0.000  0.000    0.000  0.000    4.800    983040   1563  GATE-FAIL << SELECTED
meanact_unit_s6            +0.0000   +0.0000   1.0000  0.000  0.000  0.000  0.000  0.000  0.000    0.000  0.000    7.200    983040   1568  GATE-FAIL
## normal
normal_a12                 +0.0000   +0.0000   1.0000  0.000  0.000  0.000  0.000  0.000  0.000    0.000  0.000   12.000    983040   2531  GATE-FAIL
normal_a16                 +0.0000   +0.0000   1.0000  0.000  0.000  0.000  0.000  0.000  0.000    0.000  0.000   16.000    983040   3223  GATE-FAIL
normal_a2                  +0.2750   +0.1250   0.0000  1.000  0.358  0.225  0.367  0.783  0.789    0.967  0.617    2.000    983040   1836  << SELECTED
normal_a4                  +0.0784   -0.0588   0.3688  0.425  0.508  0.554  0.537  0.138  0.383    0.500  0.784    4.000    983040   1811  GATE-FAIL
## ours_PI
ours_PI_amax12             +0.4828   +0.0172   0.0083  0.967  0.450  0.037  0.475  0.988  0.533    1.000  0.802    2.325    998400    799  << SELECTED
ours_PI_amax2              +0.2750   +0.1250   0.0000  1.000  0.358  0.225  0.367  0.783  0.789    0.967  0.617    1.506    998400   1165
ours_PI_amax4              +0.4569   +0.0043   0.0083  0.967  0.438  0.037  0.463  0.988  0.533    1.000  0.806    2.040    998400    802
## ours_PID
ours_PID_amax12            +0.4298   +0.0000   0.0125  0.950  0.425  0.046  0.446  0.971  0.533    1.000  0.781    2.383    998400   2304
ours_PID_amax2             +0.2750   +0.1250   0.0000  1.000  0.358  0.225  0.367  0.783  0.789    0.967  0.617    1.499    998400   1732
ours_PID_amax4             +0.4248   +0.0088   0.0146  0.942  0.408  0.046  0.429  0.971  0.533    1.000  0.770    2.071    998400   1714
ours_PID_amax6             +0.4298   +0.0000   0.0125  0.950  0.425  0.046  0.446  0.971  0.533    1.000  0.781    2.334    998400   1684  << SELECTED
ours_PID_amax9             +0.4298   +0.0000   0.0125  0.950  0.425  0.046  0.446  0.971  0.533    1.000  0.781    2.383    998400   1309
```

## Selected operating points

| method | selected tag | knobs | gap_raw | gap_deb | invalid | cov | clause |
|---|---|---|---|---|---|---|---|
| actadd | `actadd_p0_a32` | {'alpha': 32.0, 'pair_index': 0} | +0.1667 | +0.1042 | 0.0000 | 1.000 | CLAUSE 1: max pref_gap_raw among the 4/4 configs passing the gate (invalid <= 0.1000) |
| aura_inject | `aura_inject_g2` | {'gamma': 2.0} | +0.0667 | +0.1167 | 0.0000 | 1.000 | CLAUSE 1: max pref_gap_raw among the 2/4 configs passing the gate (invalid <= 0.1000) |
| aura_vanilla | `aura_vanilla_g1` | {'gamma': 1.0} | +0.0333 | +0.0625 | 0.0000 | 1.000 | CLAUSE 1: max pref_gap_raw among the 2/2 configs passing the gate (invalid <= 0.1000) |
| caa | `caa_a32` | {'alpha': 32.0} | +0.2250 | +0.1083 | 0.0000 | 1.000 | CLAUSE 1: max pref_gap_raw among the 5/6 configs passing the gate (invalid <= 0.1000) |
| itic | `itic_K48_a15` | {'topk': 48, 'alpha': 15.0} | +0.0000 | +0.1125 | 0.0000 | 1.000 | CLAUSE 1: max pref_gap_raw among the 1/2 configs passing the gate (invalid <= 0.1000) |
| linearact | `linearact_empirical_s0p5` | {'variant': 'empirical', 'strength': 0.5} | +0.0000 | +0.0000 | 0.0000 | 1.000 | CLAUSE 1: max pref_gap_raw among the 2/4 configs passing the gate (invalid <= 0.1000); 2 configs within EPS_GAP=0.02 of the max, broken by coverage -> invalid -> actuation -> tag |
| meanact_raw | `meanact_raw_s0p25` | {'strength': 0.25} | +0.0000 | +0.0000 | 0.8708 | 0.033 | CLAUSE 2 (FALLBACK): all 2 configs FAIL the coherence gate (invalid <= 0.1000); picked the lowest invalid_rate |
| meanact_unit | `meanact_unit_s4` | {'strength': 4.0} | +0.0000 | +0.0000 | 1.0000 | 0.000 | CLAUSE 2 (FALLBACK): all 2 configs FAIL the coherence gate (invalid <= 0.1000); picked the lowest invalid_rate |
| normal | `normal_a2` | {'alpha': 2.0} | +0.2750 | +0.1250 | 0.0000 | 1.000 | CLAUSE 1: max pref_gap_raw among the 1/4 configs passing the gate (invalid <= 0.1000) |
| ours_PI | `ours_PI_amax12` | {'amax': 12.0} | +0.4828 | +0.0172 | 0.0083 | 0.967 | CLAUSE 1: max pref_gap_raw among the 3/3 configs passing the gate (invalid <= 0.1000) |
| ours_PID | `ours_PID_amax6` | {'amax': 6.0} | +0.4298 | +0.0000 | 0.0125 | 0.950 | CLAUSE 1: max pref_gap_raw among the 5/5 configs passing the gate (invalid <= 0.1000); 4 configs within EPS_GAP=0.02 of the max, broken by coverage -> invalid -> actuation -> tag |

## mu vs its own chance floor (PREREG section 2: mu is SECONDARY)

| method | tag | mu | chance mu (p95, this run's n and coverage) | above floor? | raw_skew_mu | epsilon |
|---|---|---|---|---|---|---|
| base | `base` | 0.9889 | 0.8417 | yes | 1.0000 | 0.4542 |
| actadd | `actadd_p0_a32` | 0.8667 | 0.8417 | yes | 1.0000 | 0.5542 |
| aura_inject | `aura_inject_g2` | 0.5667 | 0.8417 | **no** | 0.6778 | 0.8333 |
| aura_vanilla | `aura_vanilla_g1` | 0.9056 | 0.8417 | yes | 1.0000 | 0.4458 |
| caa | `caa_a32` | 0.8778 | 0.8417 | yes | 0.9778 | 0.6250 |
| itic | `itic_K48_a15` | 0.6333 | 0.8417 | **no** | 0.8667 | 0.7292 |
| linearact | `linearact_empirical_s0p5` | 0.0000 | 0.8417 | **no** | 0.0000 | 1.0000 |
| meanact_raw | `meanact_raw_s0p25` | 0.0000 | 0.7500 | **no** | 0.0000 | 1.0000 |
| meanact_unit | `meanact_unit_s4` | 0.0000 | 0.0000 | **no** | 0.0000 | 0.0000 |
| normal | `normal_a2` | 0.7889 | 0.8417 | **no** | 0.9667 | 0.6167 |
| ours_PI | `ours_PI_amax12` | 0.5333 | 0.8333 | **no** | 1.0000 | 0.8017 |
| ours_PID | `ours_PID_amax6` | 0.5333 | 0.8556 | **no** | 1.0000 | 0.7807 |

## Effort matching (PREREG section 5)

* **ours_PI** (`ours_PI_amax12`): effort-matched open-loop alpha = **2.3250** (basis: mean of alpha(t) over ALL 64 denoising steps, averaged over items); committing-window basis = 2.1028 (mean commit step 30.9 of 64); saturated fraction 0.001.
* **ours_PID** (`ours_PID_amax6`): effort-matched open-loop alpha = **2.3337** (basis: mean of alpha(t) over ALL 64 denoising steps, averaged over items); committing-window basis = 2.1164 (mean commit step 30.9 of 64); saturated fraction 0.076.
