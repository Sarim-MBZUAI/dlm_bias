"""UNQOVER-race second-benchmark package (hirundo HF selection, completed from
the official allenai/unqover generation).  See README.md.

Modules
    loader.py        parquets + ethnicity.source.json -> canonical quadruples
                     (recovers the missing subject-order twins, applies the
                     attribute allowlist and the within-family exclusion)
    splits.py        BUILD/EVAL product split, disjoint on attribute x template
    metric.py        UNQOVER Eqs. 2/5-9, the raw_skew degeneracy detector,
                     clustered bootstrap and the chance noise floor
    eval_harness.py  prompt builder + strict scoring -> per-item jsonl + summary

Every module has a selftest:  python -m unqover_hf.<mod> --selftest
Nothing here loads or runs a model.
"""
